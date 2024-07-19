#!/bin/bash
echo Eval on $MODEL_NAME
python main_nlu_prompt_batch.py eng $MODEL_NAME 4
python main_nlg_prompt_batch.py eng $MODEL_NAME 0 4