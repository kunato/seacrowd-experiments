from dotenv import load_dotenv
load_dotenv()
import ast
import json
import re
import time
import os
import dataclasses
from dataclasses import dataclass, field
import torch
from typing import Any, Dict, List, Optional
from openai import OpenAI, OpenAIError
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from model_utils import load_model_and_tokenizer


two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class LLMJudgePayload:
    turns: List[str]
    category: str
    reference: Optional[str]
    question_id: str
    responses: List[str]
    is_done: bool = field(default=False)
    generation_kwargs: Dict[str, Any] = field(default_factory=lambda: {})

class LLMJudgeEvalHandler:
    def __init__(
        self,
        model_name: str,
        data_path: str,
        judge_num_workers=8,
    ) -> None:
        self.data_path = data_path
        self.judge_num_workers = judge_num_workers
        self.judge_model = 'gpt-4o'
        self.openai_client = OpenAI()
        self.judge_prompts = self._load_judge_prompts()
        model, tokenizer = load_model_and_tokenizer(model_name, compile=True)
        self.tokenizer = tokenizer
        self.model = model
        

    
    def _load_judge_prompts(self):
        current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        prompts = {}
        with open(f"{current_dir}/mt_bench_data/judge_prompt.jsonl") as fin:
            for line in fin:
                line = json.loads(line)
                prompts[line["name"]] = line
        return prompts

    def _get_conversations(self, turns, responses) -> List[ChatMessage]:
        results = []
        current_turn = 0
        while True:
            if current_turn >= len(responses):
                break
            results.append(ChatMessage(role="user", content=turns[current_turn]))
            results.append(
                ChatMessage(role="assistant", content=responses[current_turn])
            )
            current_turn += 1
        if current_turn < len(turns):
            results.append(ChatMessage(role="user", content=turns[current_turn]))
        return results

    def load_dataset(self) -> List[LLMJudgePayload]:
        res = []
        with open(self.data_path) as f:
            data = json.load(f)
        for row in data:
            turns = row["turns"]
            category = row["category"]
            question_id = row["question_id"]
            reference = row['reference']
            r = LLMJudgePayload(turns, category=category, reference=reference, question_id=question_id, responses=[])
            res.append(r)
        return res

    def is_everything_finish(self, payload: List[LLMJudgePayload]):
        return all(map(lambda x: x.is_done, payload))


    def _get_terminator(self):
        eos_tokens = ["<|eot_id|>", "<|im_start|>", "<|im_end|>"]
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        for t in eos_tokens:
            tok = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tok, int):
                terminators.append(tok)
        return terminators

    @torch.inference_mode()
    def generate(self, payload: List[LLMJudgePayload], bs=4) -> List[LLMJudgePayload]:
        prompts = []
        for row in payload:
            conv = self._get_conversations(row.turns, row.responses)
            prompts.append(self.tokenizer.apply_chat_template([dataclasses.asdict(p) for p in conv], add_generation_prompt=True, tokenize=False))

        results = []
        assert len(prompts) == len(payload)
        for i in tqdm(range(0, len(prompts), bs)):
            batchs = []
            for j in range(bs):
                if i + j >= len(prompts):
                    break
                batchs.append(prompts[i + j])
            
            inputs = self.tokenizer(batchs, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(**inputs, do_sample=True, **payload[i].generation_kwargs, eos_token_id=self._get_terminator(), max_new_tokens=512)
            preds = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.extend(preds)

        assert len(results) == len(payload)
        for i, res in enumerate(results):
            payload[i].responses.append(res)
            if len(payload[i].responses) == len(payload[i].turns):
                payload[i].is_done = True
        return payload

    def _run_judge_single(
        self,
        payload: LLMJudgePayload,
        turn: int
    ):
        multi_turn = turn > 0
        prompt_template_key = (
            "single-math-v1" if payload.reference is not None else "single-v1"
        )
        if multi_turn:
            prompt_template_key += "-multi-turn"
        prompt_template = self.judge_prompts[prompt_template_key]

        kwargs = {}
        if payload.reference is not None:
            kwargs["ref_answer_1"] = payload.reference[0]
            if multi_turn:
                kwargs["ref_answer_2"] = payload.reference[1]
        
        if multi_turn:
            user_prompt = prompt_template["prompt_template"].format(
                question_1=payload.turns[0],
                question_2=payload.turns[1],
                answer_1=payload.responses[0],
                answer_2=payload.responses[1],
                **kwargs,
            )
        else:
            user_prompt = prompt_template["prompt_template"].format(
                question=payload.turns[0],
                answer=payload.responses[0],
                **kwargs,
            )

        rating = -1
        if "gpt-4" in self.judge_model or "gpt-3.5" in self.judge_model:
            conv = [
                {"role": "system", "content": prompt_template["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ]
            temperature = 0.0 # Hard-coded temp
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
        self, payload: List[LLMJudgePayload]
    ) -> Dict[str, Any]:
        judge_inputs = []
        for p in payload:
            for i in range(len(p.turns)):
                judge_inputs.append((p, i))

        def _judge_fn(item):
            p, i = item
            r = self._run_judge_single(p, i)
            return {'result': r, 'question_id': p.question_id, 'turn': i, "payload": dataclasses.asdict(p)}

        judge_results = thread_map(
            _judge_fn, judge_inputs, max_workers=self.judge_num_workers
        )

        extra_returns = {}
        ratings = []
        for res in judge_results:
            rating = res["result"]["rating"]
            ratings.append(rating)
        extra_returns = {"avg_rating": sum(ratings) / len(ratings)}
        return {**extra_returns, 'judge_results': judge_results}
    
    def pipeline(self) -> Any:
        payload = self.load_dataset()
        while not self.is_everything_finish(payload):
            payload = self.generate(payload)    
        return self.calculate_result(payload)
        

if __name__ == '__main__':
    handler = LLMJudgeEvalHandler('meta-llama/Meta-Llama-3-8B-Instruct', '/workspace3/seacrowd-experiments/temp/mt_bench_thai_mock.json')
    results = handler.pipeline()
    print('avg_rating: ', results['avg_rating'])
    