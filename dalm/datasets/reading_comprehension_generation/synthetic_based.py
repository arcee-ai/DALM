import os
import argparse
from transformers import pipeline, AutoTokenizer
import torch
import pickle
from typing import Optional

def gen_prompt(text):
    prompt = f"There are 4 types of reading comprehension tasks. The point of reading comprehension tasks is to be assigned a text and questions to prompt answers so as to test conceptual and procedural knowledge present in the text. The four types of reading comprehension tasks are : 1. complete-the-sentence Q&A TASK   2.true/false Q&A TASK (description: a sentence is posed and the user is asked to state the correctness of the statement)3. frame a sentence with domain specific keywords(these keywords are required to be present in the text) Q&A TASK  4. Normal questions and answer Task (description: longform Q&A to test procedural and conceptual knowledge). An example of all four tasks given an example text is as follows: \n EXAMPLE TEXT: The insights into the mechanisms of memory consolidation during the sleep processes in human and animal brain led to other biologically inspired approaches. While declarative memories are in the classical picture consolidated by hippocampo-neocortical dialog during NREM phase of sleep, some types of procedural memories were suggested not to rely on the hippocampus and involve REM phase of the sleep. This inspired models where internal representations (memories) created by previous learning are spontaneously replayed during sleep-like periods in the network itself (i.e. without help of secondary network performed by generative replay approaches mentioned above).\nQuestion: [type: true/false] Is the following sentence true?  all types of procedural memories rely on the hippocampus\nAnswer: False. The text clearly states there are some types of procedural memories not reliant on the hippocampus\n--------\nQuestion [type: complete-the-sentence] Complete the following sentence:  The insights into ____ in human and animal brain led to other _____ approaches\nAnswer: The insights into the mechanisms of memory consolidation during the sleep processes in human and animal brain led to other biologically inspired approaches\n------\nQuestion [type 3 domain-keywords] Make a sentence with the following keywords 'hippocampo-neocortical', 'declarative' and 'NREM'\nAnswer: declarative memories are in the classical picture consolidated by hippocampo-neocortical dialog during NREM phase of sleep\n-------\nQuestion [type: normal q&a] Some types of procedural memories were suggested not to rely on the hippocampus and involve REM phase of the sleep. What did this go on to inspire?\nAnswer This inspired models where internal representations (memories) created by previous learning are spontaneously replayed during sleep-like periods in the network itself [END OF EXAMPLE]\n\n Similar to the above, could you craft 4 different reading comprehension tasks (make sure your output is a list of question answer pairs and each question is labelled QUESTION and answer is labelled ANSWER and there is  one question and answer per task) based solely and completely focused on the following TEXT: {text}"
    return [
            {
                "role": "system",
                "content": "You are a helpful and meticulous instruction following question and answer making chatbot. Please refrain from acknowledgments, additions or niceties of any sort"
                ,
            },
            {"role": "user", "content": prompt}
    ]


def list_dir(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as file_contents:
            contents = file_contents.read()
        yield file, contents


def text_chunker(text, tokenizer, chunk_size):
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    for i in range(0, tokens.shape[1], chunk_size):
        chunk = tokens[:, i:i+chunk_size]
        chunk = tokenizer.decode(chunk[0], skip_special_tokens=True)
        yield chunk


# standalone
def files_chunker(input_directory, model, context_length, output_directory):
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokens = tokenizer.apply_chat_template(gen_prompt(), tokenize=False, add_generation_prompt=True)
    CONSTANT = len(tokenizer(tokens)["input_ids"])

    k = context_length - CONSTANT

    for filename, text in list_dir(input_directory):
        extension = filename.split('.')[-1]
        output_file_name = filename.split('.')[0]
        for index, chunk in enumerate(text_chunker(text, tokenizer, k)):
            output_file = f'{output_file_name}_{index}.{extension}'
            with open(os.path.join(output_directory, output_file), 'w') as o:
                o.write(chunk)


def generate_synthetic_data(model_pipeline, text, generation_params):
    prompt = gen_prompt(text)
    prompt = model_pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    outputs = model_pipeline(prompt, **generation_params)

    return outputs[0]["generated_text"]


def generate_synthetic_dataset(model_name, 
                               input_directory, 
                               state_file, 
                               generation_params= {}, 
                               chunk=False, 
                               context_length=2048):
    
    model_pipeline = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

    input_files = list_dir(input_directory)

    if chunk:
        tokenizer = model_pipeline.tokenizer
        tokens = tokenizer.apply_chat_template(gen_prompt(), tokenize=False, add_generation_prompt=True)
        CONSTANT = len(tokenizer(tokens)["input_ids"])
        k = context_length - CONSTANT

    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
    else:
        state = {"processed_files": []}
        pickle.dump(state, open(state_file, 'wb'))

    for file, text in input_files:
        if file in state["processed_files"]:
            continue
        else:
            if chunk:
                for chunk_ in text_chunker(text, tokenizer, k):
                    gen_text = generate_synthetic_data(model_pipeline, chunk_, generation_params)
                    yield gen_text
            else:
                gen_text = generate_synthetic_data(model_pipeline, text, generation_params)
                yield gen_text

            state["processed_files"].append(file)
            pickle.dump(state, open(state_file, 'wb'))