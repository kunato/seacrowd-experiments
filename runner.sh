#!/bin/bash
echo Eval on $MODEL_NAME
python evaluation/main_nlu_prompt_batch.py tha $MODEL_NAME 4
python evaluation/main_nlg_prompt_batch.py tha $MODEL_NAME 0 4