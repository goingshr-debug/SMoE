import torch
from transformers import AutoModelForCausalLM
import os
from safetensors.torch import save_file
#expert的存储格式中，key的名字就是gate.weight...
model_name = "/mnt/ky2307909/qwen2moe"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cpu")

output_dir = "/mnt/ky2307909/going/smoe-xversemoe-4060/ourwork/models/qwenmoe/model_path"
os.makedirs(output_dir, exist_ok=True)

other_params = {}
expert_params = {}

for name, param in model.named_parameters():
    print(name)
    if "model.layers" in name and "mlp.experts" in name:
        layer_num = int(name.split('.')[2])
        expert_num = int(name.split('.')[5])
        
        expert_key = f"expert_{layer_num}_{expert_num}"
        if expert_key not in expert_params:
            expert_params[expert_key] = {}
        expert_params[expert_key][name] = param
    else:
        other_params[name] = param

for expert_key, params in expert_params.items():
    #todo:修改params里的key值，只保留类似key的格式"model.layers.10.mlp.experts.9.down_proj.weight"中，最后一个“.”的左右两个词
    new_params = {}
    for name, param in params.items():
        new_name = name.split('.')[6]+"."+name.split('.')[7]
        new_params[new_name] = param
    params = new_params
    save_file(params, os.path.join(output_dir, f"{expert_key}.safetensors"))

save_file(other_params, os.path.join(output_dir, "common_params.safetensors"))
index = {
    "metadata": {
        "total_size": 0  # 需要根据实际文件大小计算
    },
    "weight_map": {}
}

total_size = 0

for expert_key, params in expert_params.items():
    file_path = os.path.join(output_dir, f"{expert_key}.safetensors")
    file_size = os.path.getsize(file_path)
    total_size += file_size
    
    for name in params.keys():
        index["weight_map"][name] = f"{expert_key}.safetensors"

common_file_path = os.path.join(output_dir, "common_params.safetensors")
common_file_size = os.path.getsize(common_file_path)
total_size += common_file_size

for name in other_params.keys():
    index["weight_map"][name] = "common_params.safetensors"

index["metadata"]["total_size"] = total_size

with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
    import json
    json.dump(index, f, indent=2)

print("完成！专家参数已分开存储，index文件已生成。")