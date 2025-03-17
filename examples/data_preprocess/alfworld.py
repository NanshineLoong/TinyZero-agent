"""
Preprocess dataset for alfworld task
1. 读取数据集
2. 修改 prompt
3. 定义倒数第几步的答案
"""

import re
import json
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse

def make_prefix(template_type):
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = """User: Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "<think> your thoughts </think>\n<action> your next action </action>".

The available actions are:
1. go to {{recep}}
2. take {{obj}} from {{recep}}
3. put {{obj}} in/on {{recep}}
4. open {{recep}}
5. close {{recep}}
6. toggle {{obj}} {{recep}}
7. clean {{obj}} with {{recep}}
8. heat {{obj}} with {{recep}}
9. cool {{obj}} with {{recep}}
10. use {{obj}}
11. inventory
where {{obj}} and {{recep}} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:
<think> your thoughts </think>
<action> your next action </action>.
Assistant: OK
"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = """<|im_start|>user
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "<think> your thoughts </think>\n<action> your next action </action>".

The available actions are:
1. go to {{recep}}
2. take {{obj}} from {{recep}}
3. put {{obj}} in/on {{recep}}
4. open {{recep}}
5. close {{recep}}
6. toggle {{obj}} {{recep}}
7. clean {{obj}} with {{recep}}
8. heat {{obj}} with {{recep}}
9. cool {{obj}} with {{recep}}
10. use {{obj}}
11. inventory
where {{obj}} and {{recep}} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:
<think> your thoughts </think>
<action> your next action </action>.<|im_end|>
<|im_start|>assistant
OK<|im_end|>
"""
    return prefix

def format_response(response):
    if 'AVAILABLE ACTIONS' in response:
        response = response.split('AVAILABLE ACTIONS')[0].strip()
    if 'Observation: ' in response:
        response = response.split('Observation: ')[1].strip()
    if 'Thought: ' in response:
        response = response.split('Thought: ')[1].strip()
    
    if 'Action: ' in response:
        thought, action = response.split('Action: ')
        if thought:
            thought = f"<think> {thought} </think>\n"
        action = f"<action> {action} </action>"
        response = thought + action
    
    return response

def make_question(conv, template_type):
    assert len(conv) % 2 == 1, "The length of conversation should be odd"
    question = make_prefix(template_type)
    for i in range(0, len(conv)):
        if template_type == 'base':
            if i % 2 == 0:
                question += f"User: {format_response(conv[i]['value'])}\n"
            else:
                question += f"Assistant: {format_response(conv[i]['value'])}\n"
        elif template_type == 'qwen-instruct':
            if i % 2 == 0:
                question += f"<|im_start|>user\n{format_response(conv[i]['value'])}<|im_end|>\n"
            else:
                question += f"<|im_start|>assistant\n{format_response(conv[i]['value'])}<|im_end|>\n"
    if template_type == 'base':
        question += "Assistant: <think> "
    elif template_type == 'qwen-instruct':
        question += "<|im_start|>assistant\n<think> "
    return question

def make_solu(response):
    return response['value'].split('Action: ')[1].strip()

def load_agent_gym_dataset():
    dataset = load_dataset("json", data_files='./agentgym_alfworld_train.json', split='train')
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/alfworld')
    parser.add_argument('--steps', default='[1]')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'alfworld'

    raw_dataset = load_agent_gym_dataset()

    # raw_dataset = raw_dataset.select(range(10))

    seed(42)
    raw_dataset = raw_dataset.shuffle()
    train_dataset = raw_dataset.select(range(int(len(raw_dataset) * 0.9)))
    test_dataset = raw_dataset.select(range(int(len(raw_dataset) * 0.9), len(raw_dataset)))


    # TODO: filter some data that turns to Nothing Happen
    def make_map_fn(split, step):
        def process_fn(example):
            conv = example['conversations'][2:4]
            if len(conv) < step * 2: 
                return {}
            question = conv[:-(step * 2) + 1]
            question = make_question(conv[:-(step * 2) + 1], template_type=args.template_type)
            solution = make_solu(conv[-(step * 2) + 1])
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "agent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'step': step
                }
            }
            return data
        return process_fn
    steps = eval(args.steps)
    all_train_dataset = Dataset.from_dict({})
    all_test_dataset = Dataset.from_dict({})
    for step in steps:
        train_mapped = train_dataset.map(
            function=make_map_fn('train', step),
            remove_columns=train_dataset.column_names,
        ).filter(lambda x: len(x) > 0)
        test_mapped = test_dataset.map(
            make_map_fn('test', step),
            remove_columns=test_dataset.column_names,
        ).filter(lambda x: len(x) > 0)

        all_train_dataset = concatenate_datasets([all_train_dataset, train_mapped])
        all_test_dataset = concatenate_datasets([all_test_dataset, test_mapped])

    local_dir = args.local_dir

    # 保存为parquet格式
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 保存为 json 格式
    all_train_dataset.to_json(os.path.join(local_dir, 'train.json'))
    all_test_dataset.to_json(os.path.join(local_dir, 'test.json'))
