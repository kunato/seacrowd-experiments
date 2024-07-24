#!/bin/bash
echo Eval on $MODEL_NAME
python evaluation/main_nlu_prompt_batch.py tha $MODEL_NAME 4
python evaluation/main_nlg_prompt_batch.py tha $MODEL_NAME 0 4
python evaluation/main_llm_judge_batch.py $MODEL_NAME --data evaluation/mt_bench_data/mt_bench_thai_example.json