import os
import json
import torch
import argparse
import pandas as pd  
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import convert_to_llama_prompt

def parser():
    parser = argparse.ArgumentParser(description='do resampling')
    parser.add_argument('--model_name',
                        type=str,
                        default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data/original_data')
    args = parser.parse_args()
    return args


def main(args):
    
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto")
    
    # load original datasets
    data_paths = os.listdir(args.data_path)
    datas = [pd.read_csv(os.path.join(args.data_path, p)) for p in data_paths]

    processed_datas = []
    for data in datas:
        new_data = data.copy()
        new_data['input'] = convert_to_llama_prompt(data['input'])
        processed_datas.append(new_data)
        
        
    def make_base_path(data_name, input_prompts):
        
        base_paths = []
        all_ws = []; all_ws_scores = []
        
        for prompt in tqdm(input_prompts):
            input_ids = tokenizer(prompt,
                                return_tensors="pt").to(model.device) 
            input_length = input_ids.input_ids.size(1)
            
            outputs = model.generate(**input_ids,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    max_new_tokens=500,
                                    do_sample=False,
                                    num_beams=1)
            
            base_path = outputs.sequences[0][input_length:].detach().cpu().numpy().tolist()
           
            ws =[]
            ws_scores = []
            for i in range(len(outputs.scores)):
                x_t = outputs.scores[i].squeeze(0).softmax(dim=0)
                values, indices = torch.topk(x_t, 10, dim=-1)
                mask = values >= 0.05 
                filtered_values = values[mask]
                filtered_indices = indices[mask]
                ws.append(filtered_indices.detach().cpu().numpy().tolist())
                ws_scores.append(filtered_values.detach().cpu().numpy().tolist())
            
            base_paths.append(base_path)
            all_ws.append(ws)
            all_ws_scores.append(ws_scores)
        
        return {'data_name': data_name,
                'base_path':base_paths,
                'replace_token': all_ws,
                'replace_token_score': all_ws_scores}
    
    # start making base path
    for data_name, data in zip(data_paths, processed_datas):
        
        input_data = data['input']
        data_name = data_name.split('.csv')[0]
        
        print(f"Start sampling: {data_name}.")
        output = make_base_path(data_name, input_data)
        
        with open(f'./base_path/{data_name}_sampling.json', 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)
        
        print("Saved.")


if __name__ == '__main__':
    args = parser()
    main(args)