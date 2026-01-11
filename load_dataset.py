import os,json
import random
# from datasets import Dataset

def get_path():
    path = [
    "/mnt/ky2307909/going/data/GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Math_II_MCQs.json",
    "/mnt/ky2307909/going/data/GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Math_I_MCQs.json",
    "/mnt/ky2307909/going/data/GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_History_MCQs.json",
    "/mnt/ky2307909/going/data/GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Biology_MCQs.json",
    "/mnt/ky2307909/going/data/SuperGLUE/WiC/val.jsonl",
    "/mnt/ky2307909/going/data/triviaqa/triviaqa-train.jsonl",
    "/mnt/ky2307909/going/data/race/validation/middle.jsonl",
    "/mnt/ky2307909/going/data/race/validation/high.jsonl",
    "/mnt/ky2307909/going/data/gsm8k/train.jsonl"
    ]
    return path

def load_prefetch_random(dataset_path='GAOKAO', batch_size =1, input_num=50, idx=0, range=1):
    path_set = get_path()
    all_inputs = []
    for i,path in enumerate(path_set):
        if dataset_path in path:
            dataset_path = path
            one_inputs = load_GAOKAO_MCQs(path, batch_size,input_num)
            all_inputs.extend(one_inputs)
    random.shuffle(all_inputs)
    return all_inputs


def load_all(dataset_path, batch_size =1, input_num=100, idx=0, range=1, shuffle=True):
    path_set = get_path()
    for i,path in enumerate(path_set):
        if dataset_path.lower() in path.lower():
            dataset_path = path
            break
    all_inputs = []
    print('dataset_path:',dataset_path)
    if "2010-2022_Math_II_MCQs.json" in dataset_path:
        all_inputs = load_GAOKAO_MCQs(dataset_path,batch_size,input_num)
    elif "2010-2022_Math_I_MCQs.json" in dataset_path:
        all_inputs = load_GAOKAO_MCQs(dataset_path,batch_size,input_num)
    elif "2010-2022_History_MCQs.json" in dataset_path:
        all_inputs = load_GAOKAO_MCQs(dataset_path,batch_size,input_num)
    elif "2010-2022_Biology_MCQs.json" in dataset_path:
        all_inputs = load_GAOKAO_MCQs(dataset_path,batch_size,input_num)
    elif "SuperGLUE/WiC" in dataset_path:
        all_inputs = load_superglue_wic(dataset_path,batch_size,input_num)
    elif "triviaqa" in dataset_path:
        all_inputs = load_triviaqa(dataset_path,batch_size,input_num)
    elif "middle" in dataset_path: #race_middle
        all_inputs = load_race(dataset_path,batch_size,input_num)
    elif "high" in dataset_path: #race_high
        all_inputs = load_race(dataset_path,batch_size,input_num)
    elif "gsm8k" in dataset_path:
        all_inputs = load_gsm8k(dataset_path,batch_size,input_num)
    if shuffle:
        random.shuffle(all_inputs)
    return all_inputs



def load_superglue_wic(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template="Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre '{word}' in the above two sentenses the same?\nA. Yes\nB. No\nAnswer:"
    questions = []
    all_inputs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
            
                prompt = prompt_template.format(
                    word=data['word'],
                    sentence1=data['sentence1'].strip(),
                    sentence2=data['sentence2'].strip()
                )
                questions.append(prompt)

                line_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析错误跳过第{line_count}行: {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    
    # 确保每个批次格式正确
    return [batch for batch in all_inputs if len(batch) > 0]
    




def load_GAOKAO_MCQs(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    if "2010-2022_Math_II_MCQs.json" in dataset_path:
        subject = '数学'
    elif "2010-2022_Math_I_MCQs.json" in dataset_path:
        subject = '数学'
    elif "2010-2022_History_MCQs.json" in dataset_path:
        subject = '历史'
    elif "2010-2022_Biology_MCQs.json" in dataset_path:
        subject = '生物'
    prefix_template = f"请你做一道{subject}选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        question_list = data["example"]
    questions = [prefix_template + item['question'] for item in question_list]
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return all_inputs
    
def load_triviaqa(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    questions = []
    prompt_template = "Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}"
    all_inputs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
            
                prompt = prompt_template.format(
                    question=data['question'].strip(),
                )
                questions.append(prompt)

                line_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析错误跳过第{line_count}行: {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    
    # 确保每个批次格式正确
    return [batch for batch in all_inputs if len(batch) > 0]

def load_race(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template="Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    questions = []
    all_inputs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
            
                prompt = prompt_template.format(
                    article=data['article'],
                    question=data['question'].strip(),
                    A=data['options'][0].strip(),
                    B=data['options'][1].strip(),
                    C=data['options'][2].strip(),
                    D=data['options'][3].strip()
                )
                questions.append(prompt)

                line_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析错误跳过第{line_count}行: {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    
    # 确保每个批次格式正确
    return [batch for batch in all_inputs if len(batch) > 0]

def load_gsm8k(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion:{question}.Let's think step by step\nAnswer:"
    questions = []
    all_inputs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
            
                prompt = prompt_template.format(
                    question=data['question'],
                )
                questions.append(prompt)

                line_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析错误跳过第{line_count}行: {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    
    # 确保每个批次格式正确
    return [batch for batch in all_inputs if len(batch) > 0]

def load_gsm8k_simple(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除可能的空行
            if line.strip():
                entry = json.loads(line)
                questions.append(entry['question'])  
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return all_inputs


# def load_gsm_dataset(dataset_path, batch_size =1, input_num=100, idx=0, range=1):
#     """
#     "features": 
#         "question": {
#         "dtype": "string",
#         },
#         "answer": {
#         "dtype": "string",
#         }
#     """
#     if batch_size == 0:
#         raise ValueError(f"batch size {batch_size} is wrong")
#     from torch.utils.data import Dataset, DataLoader
#     from datasets import load_from_disk

#     dataset = load_from_disk(dataset_path)
#     dl = DataLoader(
#         dataset,          # 直接传入 Dataset 对象
#         batch_size=batch_size,    # 按需调整 batch_size
#         shuffle=False, 
#         collate_fn=lambda batch: [item["question"] for item in batch]
#     )

#     max_batches = input_num
#     all_inputs = []
#     for batch_idx, batch_data in enumerate(dl):
#         if batch_idx >= max_batches:
#             break
#         all_inputs.append(batch_data)

#     return all_inputs

# dataset_name = "tasksource/bigbench"
# names = datasets.get_dataset_config_names(dataset_name)

# remove empty entry in BIGBench dataset
# names.remove("simple_arithmetic_json_multiple_choice")
# names.remove("simple_arithmetic_multiple_targets_json")
# names.remove("cifar10_classification")

# pool = mp.Pool(mp.cpu_count())
# all_inputs = [None] * len(names)
# all_inputs = pool.map(partial(datasets.load_dataset, dataset_name), names)

# all_inputs = [
#     text for dataset in all_inputs for text in dataset["validation"]["inputs"]
# ]

import torch
def load_deepseekr1(dataset_path, batch_size =1, input_num=100, idx=0, range=1):
    '''
    "input": {
        "dtype": "string",
    },
    '''

    if batch_size == 0:
        raise ValueError(f"batch size {batch_size} is wrong")
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    dl = DataLoader(
        dataset,          # 直接传入 Dataset 对象
        batch_size=batch_size,    # 按需调整 batch_size
        shuffle=False, 
        collate_fn=lambda batch: [item["input"] for item in batch]
    )

    max_batches = input_num
    all_inputs = []
    for batch_idx, batch_data in enumerate(dl):
        if batch_idx >= max_batches:
            break
        all_inputs.append(batch_data)

    return all_inputs



def load_all_pt(dataset_path,batch_size = 1):
    print(dataset_path)
    prompts = torch.load(dataset_path)

    from torch.utils.data import Dataset
    class StringListDataset(Dataset):
        def __init__(self, string_list):
            self.string_list = string_list
        def __len__(self):
            return len(self.string_list)
        def __getitem__(self, idx):
            return self.string_list[idx]
    ds = StringListDataset(prompts)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

def load_all_arrow(dataset_path,batch_size = 2):
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_from_disk

    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    dl = DataLoader(
        dataset,          # 直接传入 Dataset 对象
        batch_size=batch_size,    # 按需调整 batch_size
        shuffle=False, 
        collate_fn=lambda batch: [item["question"] for item in batch]
    )


if __name__ == '__main__':
    # import debugpy
    # try:
    #     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    #     debugpy.listen(("localhost", 9501))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    # -------------------------------------------------------------
    # dataset_path1 = '/home/easyai/guoying/datasets/gsm8k_main/train'
    # dataset_path2 = '/home/easyai/guoying/datasets/Chinese-deepseekr1/train'
    # batch = 1
    # input_num = 5
    
    # # inputs_all = load_gsm_dataset(dataset_path1, batch, input_num)
    # inputs_all = load_deepseekr1(dataset_path2, batch, input_num=input_num)
    # for inputs in inputs_all:
    #     print(inputs)
    # -------------------------------------------------------------
    # dataset_path = 'Math_I_MCQs'
    dataset_path = 'wic'
    # dataset_path = 'gsm8k'
    all_inputs = load_all(dataset_path, batch_size=1, input_num=5)
    for i, inputs in enumerate(all_inputs):
        print('i=',i ,inputs)
