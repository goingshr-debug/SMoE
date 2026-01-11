# tokenizer_service.py
from flask import Flask, request, jsonify
import requests
import torch
from transformers import AutoTokenizer
import time

app = Flask(__name__)

# Tokenizer Service
model_name = "/home/easyai/guoying/ourwork/models/xversemoe/model_path"  # 请修改为实际路径
tokenizer = AutoTokenizer.from_pretrained(model_name)

import requests
import json
# 1. 准备请求数据
request_data = {
    "texts": [      # 这里放入要处理的文本列表
        "今天天气真好，适合",
        "人工智能的发展方向是",
        "量子计算机的原理"
    ]
}


def get_prompts():
    from load_dataset import load_all, load_prefetch_random
    import importlib.util
    module_path = "/home/easyai/guoying/baselines/load_dataset.py"
    spec = importlib.util.spec_from_file_location("load_dataset", module_path)
    load_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_dataset)
    # 使用导入的函数
    load_all = load_dataset.load_all

    dataset_path = 'History'
    batch_size = 3
    input_num = 1
    all_inputs = load_all(dataset_path, batch_size, input_num)
    return all_inputs

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收来自客户端的文本列表，并作为一个批次发送给模型服务进行推理。
    """

    texts = request.json.get('texts', [])
    # texts = get_prompts()
    print('batch_size:',len(texts))
    

    print('texts:',texts)
    
    if not isinstance(texts, list):
        return jsonify({"error": "Input must be a list of texts"}), 400
    
    # Step 1: Encode the text input into model-compatible inputs
    if len(texts) == 1:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt" )
    else:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Convert inputs to list of lists so they can be serialized
    inputs_list = {key: val.tolist() for key, val in inputs.items()}
    
    # Step 2: Send the inputs to the model service
    s = time.time()
    response = requests.post("http://localhost:5001/generate", json=inputs_list)  # Ensure the model service is running on this address and port
    t = time.time()
    print("inference time", t-s)
    print("Response status code:", response.status_code)
    print("Response content:", response.text)
    outputs_list = response.json()
    
    # Step 3: Decode the model outputs into human-readable text
    outputs_tensor = torch.tensor(outputs_list)
    results = tokenizer.batch_decode(outputs_tensor, skip_special_tokens=True)
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)