import ast
import gc
import json
import re
import time
import os
from dataclasses import dataclass
import dataclasses
import torch
from transformers import AutoTokenizer
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from openai import OpenAI, OpenAIError
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

@dataclass
class GenerateResult:
    output: str
    extra: Any

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class EvalsetPayload:
    item: Any
    prompt: Union[str, List[ChatMessage]]
    id: str

@dataclass
class InferenceResult:
    id: str
    result: str
    raw_result: Any
    item: EvalsetPayload

class MTBenchTask():
    def __init__(
        self,
        task_name: str,
        result_path: str,
        judge_result_path: str,
        is_pair=False,
        baseline_result_path=None,
        language: Union[Literal['th'], Literal['en']] = 'en',
        judge_num_workers=8,
    ) -> None:
        super().__init__(task_name, result_path)
        self.language = language
        self.judge_result_path = judge_result_path
        self.openai_client = OpenAI()
        self.is_pair = is_pair
        self.TIE_DELTA = 0.1
        self.NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]
        self.judge_prompts = self._load_judge_prompts()
        self.judge_num_workers = judge_num_workers
        self.temperature_config = {
            "writing": 0.7,
            "roleplay": 0.7,
            "extraction": 0.0,
            "math": 0.0,
            "coding": 0.0,
            "reasoning": 0.0,
            "stem": 0.1,
            "humanities": 0.1,
            "arena-hard-200": 0.0,
        }
        if self.is_pair:
            self.baseline_result_path = baseline_result_path
            assert baseline_result_path is not None

    def _get_task_name(self):
        return self.task_name + '-' + self.language

    def get_sampling_params(self, model_name: str) -> Dict[str, Any]:
        return {"temperature": 0.1, "max_tokens": 1024}

    def _get_system(self):
        if self.language == 'th':
            return [ChatMessage(role='system', content='You are helpful assistant who\'re always answer in Thai.')]
        else:
            return []

    def _get_conversations(self, turns, response):
        results = []
        current_turn = 0
        while True:
            if current_turn >= len(response):
                break
            results.append(ChatMessage(role="user", content=turns[current_turn]))
            results.append(
                ChatMessage(role="assistant", content=response[current_turn])
            )
            current_turn += 1
        if current_turn < len(turns):
            results.append(ChatMessage(role="user", content=turns[current_turn]))
        return self._get_system() + results

    def load_dataset(self, model_name: str) -> List[EvalsetPayload]:
        if self.task_name == "mt_bench" and self.language == 'en':
            with open(f"{root_path}/data/mtbench/mt_bench.jsonl", "r") as f:
                data = [json.loads(l) for l in f.readlines()]
            with open(f"{root_path}/data/mtbench/mt_bench_ref_gpt4.jsonl", "r") as f:
                ref_data = [json.loads(l) for l in f.readlines()]
                ref_data_dict = {}
                for row in ref_data:
                    ref_data_dict[row["question_id"]] = row["choices"][0]["turns"]
            self.judge_model = 'gpt-4'

        elif self.task_name == "mt_bench" and self.language == 'th':
            with open(f"{root_path}/data/mtbench/mt_bench_thai.jsonl", "r") as f:
                data = [json.loads(l) for l in f.readlines()]
            with open(f"{root_path}/data/mtbench/mt_bench_thai_ref_gpt4_1106_v2.jsonl", "r") as f:
                ref_data = [json.loads(l) for l in f.readlines()]
                ref_data_dict = {}
                for row in ref_data:
                    ref_data_dict[row["question_id"]] = row["choices"][0]["turns"]
            self.judge_model = 'gpt-4-turbo-2024-04-09'

        else:
            raise NotImplementedError()

        res = []
        for row in data:
            current_turn = 0
            turns = row["turns"]
            category = row["category"]
            question_id = row["question_id"]
            reference = (
                ref_data_dict[row["question_id"]]
                if category in self.NEED_REF_CATS
                else None
            )
            r = EvalsetPayload(
                {
                    "turns": turns,
                    "turn": current_turn,
                    "category": category,
                    "reference": reference,
                    "question_id": question_id,
                    "responses": [],
                },
                prompt=self._get_conversations(turns, []),
                id=question_id,
            )
            res.append(r)
        return res

    def next_turn_data_creation(
        self, results: List[InferenceResult]
    ) -> Tuple[Optional[List[EvalsetPayload]], bool]:
        res = []
        first_result = results[0]
        current_turn = first_result.item.item["turn"] + 1
        total_turns = len(first_result.item.item["turns"])
        if current_turn >= total_turns:
            return None, True

        for item in results:
            payload = item.item
            resp = item.result
            data = payload.item
            turns = data["turns"]
            assert current_turn == data["turn"] + 1
            category = data["category"]
            question_id = data["question_id"]
            responses = data["responses"]
            responses.append(resp)
            r = EvalsetPayload(
                {
                    "turns": turns,
                    "turn": current_turn,
                    "category": category,
                    "reference": data["reference"],
                    "question_id": question_id,
                    "responses": responses,
                },
                prompt=self._get_conversations(turns, responses),
                id=question_id,
            )
            res.append(r)
        return res, False

    def _load_judge_prompts(self):
        current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        prompts = {}
        with open(f"{current_dir}/mt_bench_data/judge_prompt.jsonl") as fin:
            for line in fin:
                line = json.loads(line)
                prompts[line["name"]] = line
        return prompts

    def _run_judge_pair(
        self,
        question: List[str],
        answer_a: List[str],
        answer_b: List[str],
        model_a: str,
        model_b: str,
        category: str,
        ref_answer: Optional[List[str]],
        multi_turn=False,
    ):
        prompt_template_key = (
            "pair-math-v1" if category in self.NEED_REF_CATS else "pair-v2"
        )
        if multi_turn:
            prompt_template_key += "-multi-turn"
        prompt_template = self.judge_prompts[prompt_template_key]

        kwargs = {}
        if ref_answer is not None:
            kwargs["ref_answer_1"] = ref_answer[0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer[1]

        if multi_turn:
            user_prompt = prompt_template["prompt_template"].format(
                question_1=question[0],
                question_2=question[1],
                answer_a_1=answer_a[0],
                answer_b_1=answer_b[0],
                answer_a_2=answer_a[1],
                answer_b_2=answer_b[1],
                **kwargs,
            )
        else:
            user_prompt = prompt_template["prompt_template"].format(
                question=question[0],
                answer_a=answer_a[0],
                answer_b=answer_b[0],
                **kwargs,
            )

        winner = "error"
        if "gpt-4" in self.judge_model or "gpt-3.5" in self.judge_model:
            conv = [
                {"role": "system", "content": prompt_template["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ]
            temperature = self.temperature_config[category]
            judgment = self._call_openai(
                self.judge_model, conv, temperature=temperature, max_tokens=2048
            )
        else:
            raise NotImplementedError()

        if prompt_template["output_format"] == "[[A]]":
            if "[[A]]" in judgment:
                winner = "A"
            elif "[[B]]" in judgment:
                winner = "B"
            elif "[[C]]" in judgment:
                winner = "tie"
            else:
                winner = "error"
        elif prompt_template["output_format"] == "[[rating_a,rating_b]]":
            match = re.search(two_score_pattern, judgment)
            if not match:
                match = re.search(two_score_pattern_backup, judgment)
            if match:
                scores = [ast.literal_eval(s.strip()) for s in match.groups()]
                if abs(scores[0] - scores[1]) <= self.TIE_DELTA:
                    winner = "tie"
                elif scores[0] > scores[1]:
                    winner = "A"
                else:
                    winner = "B"
            else:
                winner = "error"
        else:
            raise ValueError(
                f"invalid output format: {prompt_template['output_format']}"
            )
        return {
            "winner": winner,
            "user_prompt": user_prompt,
            "judgment": judgment,
            "model_a": model_a,
            "model_b": model_b,
        }

    def _run_judge_single(
        self,
        question: List[str],
        answer: List[str],
        category: str,
        ref_answer: Optional[List[str]],
        multi_turn=True,
    ):
        prompt_template_key = (
            "single-math-v1" if category in self.NEED_REF_CATS else "single-v1"
        )
        if multi_turn:
            prompt_template_key += "-multi-turn"
        prompt_template = self.judge_prompts[prompt_template_key]

        kwargs = {}
        if ref_answer is not None:
            kwargs["ref_answer_1"] = ref_answer[0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer[1]
        if multi_turn:
            user_prompt = prompt_template["prompt_template"].format(
                question_1=question[0],
                question_2=question[1],
                answer_1=answer[0],
                answer_2=answer[1],
                **kwargs,
            )
        else:
            user_prompt = prompt_template["prompt_template"].format(
                question=question[0],
                answer=answer[0],
                **kwargs,
            )

        rating = -1

        if "gpt-4" in self.judge_model or "gpt-3.5" in self.judge_model:
            conv = [
                {"role": "system", "content": prompt_template["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ]
            temperature = self.temperature_config[category]
            judgment = self._call_openai(
                self.judge_model, conv, temperature=temperature, max_tokens=2048
            )
        else:
            raise NotImplementedError()

        if prompt_template["output_format"] == "[[rating]]":
            match = re.search(one_score_pattern, judgment)
            if not match:
                match = re.search(one_score_pattern_backup, judgment)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = -1
        else:
            raise ValueError(
                f"invalid output format: {prompt_template['output_format']}"
            )

        return {"rating": rating, "user_prompt": user_prompt, "judgment": judgment}

    def _get_baseline_result(self):
        with open(self.baseline_result_path) as f:
            data = [json.loads(l) for l in f.readlines()]
            baseline_result = {}
            for row in data:
                question_id = row["question_id"]
                for i, turn in enumerate(row["choices"][0]["turns"]):
                    baseline_result[f"{question_id}_{i}"] = turn
        return baseline_result

    def _call_openai(self, model, conv, temperature, max_tokens):
        output = "$ERROR$"
        for _ in range(16):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=conv,
                    n=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content
                break
            except OpenAIError as e:
                print(type(e), e)
                time.sleep(5)
        return output

    def calculate_result(
        self, results: List[InferenceResult], model_name: str = None
    ) -> Dict[str, Any]:
        judge_inputs = []
        for item in results:
            responses = item.item.item["responses"]
            turns = item.item.item["turns"]
            category = item.item.item["category"]
            reference = item.item.item["reference"]
            if reference is not None:
                assert len(turns) == len(reference)
            responses.append(item.result)
            assert len(turns) == len(responses)

            result = {
                "question_id": item.item.id,
                "reference": reference,
                "turns": turns,
                "category": category,
                "responses": responses,
            }
            judge_inputs.append(result)

        def _judge_fn(item):
            jrs = []
            question_id = item["question_id"]
            reference = item["reference"]
            category = item["category"]
            turns = item["turns"]
            responses = item["responses"]
            for i in range(len(turns)):
                if i == 0:
                    multi_turn = False
                else:
                    multi_turn = True

                if self.is_pair:
                    baseline = self._get_baseline_result()
                    answer_key = f"{question_id}_{i}"
                    if answer_key not in baseline.keys():
                        print(f"answer in ref not found -> {answer_key}; skip")
                        continue
                    g1_result = self._run_judge_pair(
                        question=turns[: i + 1],
                        answer_a=responses[: i + 1],
                        answer_b=baseline[f"{question_id}_{i}"],
                        model_a=model_name,
                        model_b="baseline",
                        category=category,
                        ref_answer=(
                            reference if reference is None else reference[: i + 1]
                        ),
                        multi_turn=multi_turn,
                    )
                    g2_result = self._run_judge_pair(
                        question=turns[: i + 1],
                        answer_b=responses[: i + 1],
                        answer_a=baseline[f"{question_id}_{i}"],
                        model_a="baseline",
                        model_b=model_name,
                        category=category,
                        ref_answer=(
                            reference if reference is None else reference[: i + 1]
                        ),
                        multi_turn=multi_turn,
                    )
                    jrs.append(
                        {
                            "question_id": question_id,
                            "turn": i,
                            "g1_model_name": model_name,
                            "g2_model_name": "baseline",
                            "g1_result": g1_result,
                            "g2_result": g2_result,
                            "item": item,
                        }
                    )
                else:
                    judge_result = self._run_judge_single(
                        question=turns[: i + 1],
                        answer=responses[: i + 1],
                        category=category,
                        ref_answer=(
                            reference if reference is None else reference[: i + 1]
                        ),
                        multi_turn=multi_turn,
                    )
                    jrs.append(
                        {
                            "question_id": question_id,
                            "turn": i,
                            "result": judge_result,
                            "item": item,
                        }
                    )
            return jrs

        judge_results = thread_map(
            _judge_fn, judge_inputs, max_workers=self.judge_num_workers
        )

        return_results = []
        for res in judge_results:
            return_results.extend(res)

        extra_returns = {}
        ratings = []
        for res in return_results:
            rating = res["result"]["rating"]
            ratings.append(rating)
        extra_returns = {"avg_rating": sum(ratings) / len(ratings)}
        os.makedirs(os.path.dirname(self.judge_result_path), exist_ok=True)
        with open(self.judge_result_path, "w") as w:
            json.dump(return_results, w, ensure_ascii=False)
        return {**extra_returns}

class VllmRunnerEvaluator():

    def __init__(
        self, model_name: str, model_name_or_path: str, stop_tokens=[], vllm_kwargs={}
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.stop_tokens = stop_tokens
        default_vllm_kwargs = {"max_model_len": 8192}
        default_vllm_kwargs.update(vllm_kwargs)
        self.llm = LLM(model=model_name_or_path, **default_vllm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def _get_instruct_prompt_for_model(self, prompts: Union[str, List[ChatMessage]]):
        if isinstance(prompts, str):
            # if it is string then do "completion" not "chat-completion"
            # prompts = [ChatMessage(role="user", content=prompts)]
            return prompts
        prompts = [dataclasses.asdict(p) for p in prompts]
        full_prompt = self.tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )
        return full_prompt

    def _generate_function(
        self, prompts: Union[str, List[ChatMessage]], sampling_params: Dict[str, Any]
    ) -> List[GenerateResult]:
        sampling_kwargs = {}
        if len(self.stop_tokens) != 0:
            stop_token_ids = [self.tokenizer.encode(t) for t in self.stop_tokens]
            stop_token_ids = list(filter(lambda x: len(x) == 1, stop_token_ids))
            sampling_kwargs["stop_token_ids"] = stop_token_ids
            sampling_kwargs["stop"] = self.stop_tokens
        print("model_sampling_kwargs", sampling_kwargs)
        results = self.llm.generate(
            prompts=[self._get_instruct_prompt_for_model(p) for p in prompts],
            use_tqdm=True,
            sampling_params=SamplingParams(**sampling_params, **sampling_kwargs),
        )
        return [
            GenerateResult(
                res.outputs[0].text,
                extra={
                    "finish_reason": res.outputs[0].finish_reason,
                    "prompt": res.prompt,
                },
            )
            for res in results
        ]
    
    def _filter_result(self, result):
        if result is None:
            return {}
        return {
            "task_name": result["task_name"],
            "model_name": result["model_name"],
            "result": result["result"],
        }

    def inference_and_eval(self, tasks):
        results = []
        for task in tqdm(tasks):
            result = task.pipeline(self._generate_function, self.model_name)
            results.append(self._filter_result(result))
            print("task ->", results[-1])
        print("all task completed")
        return results

    def close(self):
        destroy_model_parallel()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        time.sleep(10)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    handler = VllmRunnerEvaluator(
        model_name=model_name,
        model_name_or_path=model_name_or_path,
        vllm_kwargs={
            **vllm_kwargs,
        },
        stop_tokens=stop_tokens,
    )
    handler.inference_and_eval([MTBenchTask('mt_eval', f"{output_path}/{model_name}/mt-bench-th.json",f"{output_path}/{model_name}/mt-bench-judge-th.json", language='th')])