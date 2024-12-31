import os
import json
import math
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import convert_to_llama_prompt, read_json


def main(args):
    
    # load model and tokenizer
    llm = LLM(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left"
        )


    def make_forking_path(input_prompt, total_all_s, answer):
        base_prompt = input_prompt
        all_S = total_all_s

        temp_all_S = []
        total_index = []
        total_probs = []
        for t in tqdm(range(len(all_S))):
            
            if len(all_S[t]) == 1:
                continue 
            
            sampling_params = SamplingParams(
                temperature=1.1,
                top_p=0.9,
                max_tokens=500 - t,
                logprobs=1
            )
            batch_prompts = [base_prompt + s for s in all_S[t]] * 30
            outputs = llm.generate(batch_prompts, sampling_params)
            
            token_ids = [output.outputs[0].token_ids for output in outputs]
            temp_outputs_scores = [output.outputs[0].logprobs for output in outputs]
            probs = []
            for output_s, t_ids in zip(temp_outputs_scores, token_ids):
                prob = 0.0
                for s, t_ in zip(output_s, t_ids):
                    prob += s[t_].logprob
                probs.append(math.exp(prob))    
            decoded_outputs = [output.outputs[0].text for output in outputs]
            
            temp_all_S.append(decoded_outputs)
            total_index.append(t)
            total_probs.append(probs)
        
        return {'all_path': temp_all_S,
                't_index': total_index,
                'answer': answer, 
                'prob': total_probs}


    def make_S(base_path_list, replace_token_list):
        total_all_S = []
        for b_s, e_s in zip(base_path_list, replace_token_list):
            all_S = []
            for t in range(len(e_s)):
                temp_S = []
                if len(e_s[t]) > 1:
                    ws = e_s[t]
                    for w in ws:
                        temp_b_s = b_s.copy()
                        temp_b_s[t] = w
                        temp_S.append(tokenizer.decode(temp_b_s[:t+1]))
                else:
                    temp_S.append(tokenizer.decode(b_s[:t+1]))
                all_S.append(temp_S)
            total_all_S.append(all_S)
        return total_all_S
    
    
    # load original datasets
    data_paths = os.listdir(args.data_path)
    datas = [pd.read_csv(os.path.join(args.data_path, p)) for p in data_paths]

    processed_datas = []
    for data in datas:
        new_data = data.copy()
        new_data['input'] = convert_to_llama_prompt(data['input'])
        processed_datas.append(new_data)
        
    # load base path datasets
    all_data_path = [os.path.join(args.base_path_data, p) \
        for p in os.listdir(args.base_path_data)]
    
    encoded_datas = []
    for p in all_data_path:
        encoded_datas.append(read_json(p))
        
    # generate forking paths
    for data_name, data, e_data in zip(data_paths, processed_datas, encoded_datas):
        
        input_data = data['input']
        answer_data = data['ground_truth']
        data_name = data_name.split('.csv')[0]

        print(f"Start sampling: {data_name}.")
        total_all_S = make_S(e_data['base_path'], e_data['replace_token'])

        for idx in range(len(total_all_S)):
            
            # too short sentence does not have to be resampled
            if len(total_all_S[idx][-1][0]) < 500:
                continue
            
            output = make_forking_path(input_data[idx], total_all_S[idx], answer_data[idx])
            with open(f'./forking_path/{data_name}_sampling_{idx}.json', 'w', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False, indent=4)

        print("Saved.")


def parser():
    parser = argparse.ArgumentParser(description='do resampling')
    parser.add_argument('--model_name',
                        type=str,
                        default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data/original_data')
    parser.add_argument('--base_path_data',
                        type=str,
                        default='./data/base_path_data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    main(args)