import pandas as pd
from collections import defaultdict
from huggingface_hub import HfApi
import os
import json

HF_TOKEN = os.environ.get("TOKEN")
RESULTS_REPO = os.environ.get('RESULTS_REPO')

def upload_file(result_path: str, task_type: str, model_name: str):
    API = HfApi(token=HF_TOKEN)
    API.upload_file(
        path_or_fileobj=result_path,
        path_in_repo=f'{task_type}/{model_name}/results.json',
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {result_path} to result queue",
    )

def process_nlu_result(model_name: str, outpath: str):
    task_type = 'NLU'
    df = pd.read_csv(f'evaluation/metrics_nlu/nlu_results_eng_{model_name}.csv')
    results = defaultdict(list)
    dataset_key = 'dataset'
    metrics_key = ['accuracy']
    for i, row in df.iterrows():
        for k in metrics_key:
            results[row[dataset_key]].append({k: row[k]}) 
    results_final = {}
    for k in results.keys():
        d = results[k][0]
        results_final[k] = d
    data = {
        'config': {
            "model_name": model_name,
        },
        "results": results_final
    }
    result_path = f'{outpath}/{task_type}/{model_name}/results.json'
    print(os.path.dirname(result_path))
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as w:
        json.dump(data, w, ensure_ascii=False)
    upload_file(result_path, task_type, model_name)


def process_nlg_result(model_name: str, outpath: str):
    task_type = 'NLG'
    df = pd.read_csv(f'evaluation/metrics_nlg/nlg_results_eng_0_{model_name}.csv')
    results = defaultdict(dict)
    dataset_key = 'dataset'
    dataset_to_metrics = {
        'lr_sum_tha_seacrowd_t2t': ['ROUGE1', 'ROUGE2', 'ROUGEL'],
        'xl_sum_tha_seacrowd_t2t': ['ROUGE1', 'ROUGE2', 'ROUGEL'],
        'flores200_eng_Latn_tha_Thai_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'ntrex_128_eng-US_tha_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'flores200_tha_Thai_eng_Latn_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'ntrex_128_tha_eng-US_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'mkqa_tha_seacrowd_qa': []
    }

    for i, row in df.iterrows():
        metrics_key = dataset_to_metrics[row[dataset_key]]
        for k in metrics_key:
            results[row[dataset_key]][k] = row[k]
    data = {
        'config': {
            "model_name": model_name,
        },
        "results": results
    }
    result_path = f'{outpath}/{task_type}/{model_name}/results.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as w:
        json.dump(data, w, ensure_ascii=False)
    
    upload_file(result_path, task_type, model_name)



if __name__ == '__main__':
    process_nlu_result('Meta-Llama-3-8B-Instruct', 'results')
    process_nlg_result('Meta-Llama-3-8B-Instruct', 'results')