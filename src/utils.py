import json 


def convert_to_llama_prompt(data_list):
    new_data_list = []
    for entry in data_list:
        formatted_text = ""
        question, answer = entry.split("A:")
        answer = "A:" + answer

        formatted_text += "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        formatted_text += f"{question}\n"
        formatted_text += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        formatted_text += f"{answer}\n"
        formatted_text += "<|eot_id|>"

        new_data_list.append(formatted_text)
    return new_data_list


def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None