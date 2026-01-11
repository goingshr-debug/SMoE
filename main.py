import numpy
import sys
import torch
from transformers import AutoTokenizer,TextStreamer
from utils import build_model
import time
import psutil
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='qwenmoe')
parser.add_argument("--input_num", type=int, default=100)
parser.add_argument("--dataset_path", type=str, default='')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--output_len", type=int, default=32)
parser.add_argument("--GPU mem", type=int, default=24)
args = parser.parse_args()



class StopWatch(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1

        if self.decoding_iterations % 100 == 0:
            current_time = time.time()
            print(f"Prefilling time: {self.prefilling_time} seconds")
            print(f"Decoding time: {self.decoding_time} seconds")
            print(f"Decoding iterations: {self.decoding_iterations}")
            print(
                f"Decoding time per iteration: {(current_time-self.start_decoding) / self.decoding_iterations} seconds"
            )

        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
            current_time = time.time()
            print(f"Prefilling time: {self.prefilling_time} seconds")
            print(f"Decoding time: {self.decoding_time} seconds")
            print(f"Decoding iterations: {self.decoding_iterations}")
            print(
                f"Decoding time per iteration: {(current_time-self.start_decoding) / self.decoding_iterations} seconds"
            )

        return super().end()

# args.model_name == 'deepseekmoe'


if args.model_name == 'deepseekmoe':
    model_name = "/home/easyai/guoying/ourwork/models/deepseekmoe/model_path"
    model_type = "deepseekmoe"
    from models.deepseekmoe.model_path import modeling_deepseek
    from models.deepseekmoe import deepseek_module
elif args.model_name == 'xversemoe':
    model_name = "/home/easyai/guoying/ourwork/models/xversemoe/model_path"
    model_type = "xversemoe"
    import models.xversemoe.xverse_module
    from models.xversemoe import xverse_module
    from models.xversemoe.model_path import modeling_xverse
elif args.model_name == 'qwenmoe':
    model_name = "/mnt/ky2307909/going/smoe-xversemoe-4060/ourwork/models/qwenmoe/model_path"
    model_type = "qwenmoe"
    from models.qwenmoe import qwen_module
    from models.qwenmoe.model_path import modeling_qwen
else:
    assert False, 'invalid model'


if args.debug:
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass

# Tokenizer service (using conda Rmoe environment)
# Initialization
tokenizer = AutoTokenizer.from_pretrained(model_name)
# First function: Convert text to inputs and send inputs to model service
# texts = [
#     "Please introduce Nanjing University.",
#     # "The transformer architecture has revolutionized natural language processing by enabling models to process sequences in parallel, significantly improving training and inference speed.",
#     # "In machine learning, a neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates."
# ]

from load_dataset import load_all, load_prefetch_random
import importlib.util
module_path = "/mnt/ky2307909/going/smoe-xversemoe-4060/ourwork/load_dataset.py"
spec = importlib.util.spec_from_file_location("load_dataset", module_path)
load_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset)
# 使用导入的函数
load_all = load_dataset.load_all

dataset_path = args.dataset_path
all_inputs = load_all(dataset_path, args.batch_size, args.input_num)

# all_inputs = load_prefetch_random()






# Model service (using conda Nmoe environment)
# Initialization

device = torch.device("npu")


if "qwen" in model_name:
    cache_size = 386
elif "xverse" in model_name:
    cache_size = 300 # 392
elif "deepseek" in model_name:
    cache_size = 500 #800=>20GB  900=>21.5GB 1000=>23GB
    
#4090
#load stream需要空间，self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype，kv cache都需要
#xverse 700
#qwen 
# cache_size = 700
# 显存大小除以专家大小
k = 1
    # Build the model
model = build_model(model_path=model_name,
                    model_type=model_type,
                    device=device,
                    main_size=cache_size)

# Move the model to the specified device
model = model.to(device)



# Print memory usage after model initialization
logger.info(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
logger.debug(f"debug")

output_len = args.output_len

for i, input in enumerate(all_inputs):
    if args.model_name == 'deepseekmoe':
        modeling_deepseek.tokens = 0
        modeling_deepseek.decode_time = 0
        deepseek_module.tops = 0
        deepseek_module.lows = 0
        deepseek_module.top_loads = 0
        deepseek_module.low_loads = 0
        deepseek_module.prefetch_experts = []
        deepseek_module.prefetch_hit_sum = 0
        deepseek_module.load_sum = 0
        deepseek_module.prefetch_number = 0
        deepseek_module.all_loads = 0

    elif args.model_name == 'xversemoe':
        modeling_xverse.tokens = 0
        modeling_xverse.decode_time = 0
        xverse_module.tops = 0
        xverse_module.lows = 0
        xverse_module.top_loads = 0
        xverse_module.low_loads = 0
    elif args.model_name == 'qwenmoe':
        modeling_qwen.tokens = 0
        modeling_qwen.decode_time = 0
        qwen_module.tops = 0
        qwen_module.lows = 0
        qwen_module.top_loads = 0
        qwen_module.low_loads = 0

    # Convert inputs to the model's device
    texts = all_inputs[i]
    l = len(texts[0])
    print(texts)
    # for j in range(10,l,5):
    #     deepseek_module.least_batch=10000
    #     text = [texts[0][:j]]
    print('='*20, flush=True)
    print(f"input_id: {i}")
    print(f'text:{texts}',flush=True)
    print(f'batch:{args.batch_size}')
    if args.batch_size == 1:
        inputs = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=4096,
                return_tensors="pt",
            )
    else:
        tokenizer.padding_side = 'left'
        inputs = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=1024,
                return_tensors="pt",
            )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        # Perform parallel inference
    streamer = StopWatch(tokenizer)
    with torch.no_grad():
        start = time.time()
        
        # outputs = model.generate(**inputs, max_new_tokens=output_len, streamer=streamer)
        outputs = model.generate(**inputs, max_new_tokens=output_len)
        # outputs = model.generate(**inputs, max_new_tokens=1)
        end = time.time()

    logger.warning(f"Parallel inference time: {end - start:.2f} seconds")

    # Decode outputs
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.warning(f"results: %s", results)
    print('='*20, flush=True)
    

    # for i, result in enumerate(results):
    #     print(f"Result {i + 1}:")
    #     print(result)
    #     print('='*20)