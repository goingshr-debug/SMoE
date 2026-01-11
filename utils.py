from models.deepseekmoe.deepseek_module import DeepseekMLP,DeepseekMoEwithCache
from models.qwenmoe.qwen_module import Qwen2MoeMLP,Qwen2MoeSparseMoeBlockwithCache
from models.xversemoe.xverse_module import XverseMLP,XverseMoEMLPwithCache
from typing import Tuple
from expertcachewithscorefullcpu import ExpertCache
import os
import json
from dataclasses import dataclass
import torch
from models.deepseekmoe.model_path.configuration_deepseek import DeepseekConfig
from models.qwenmoe.model_path.qwen2_config import Qwen2MoeConfig
from models.xversemoe.model_path.configuration_xverse import XverseConfig
from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange
from contextlib import contextmanager
from models.deepseekmoe.model_path.modeling_deepseek import DeepseekForCausalLM
from models.qwenmoe.model_path.modeling_qwen import Qwen2MoeForCausalLM
from models.xversemoe.model_path.modeling_xverse import XverseForCausalLM
import psutil
import gc,sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
current_supportmoe = ["deepseekmoe","qwenmoe","xversemoe"]
ExpertUID = Tuple[int,int]

@contextmanager
def with_default_dtype(dtype):
    _dtype_original = torch.get_default_dtype()

    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(_dtype_original)

def nested_flatten(t):
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t

def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)

def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)

def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int

def make_empty_expert(
    model_config,
    model_type: str
):
    # logger.debug(model_type)
    if model_type == "deepseekmoe":
        return DeepseekMLP(model_config)
    elif model_type == "qwenmoe":
        # logger.debug(111111111111)
        return Qwen2MoeMLP(model_config)
    elif model_type == "xversemoe":
        return XverseMLP(model_config)
    else:
        raise ValueError("This model is not supported")


def make_and_load_expert_wrapper(
    config,
    states_dir: str,
    expert_uid: tuple[int, int],
    model_type,
    device: torch.device,
):
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.up_proj.weight"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert(config,model_type)
    expert.load_state_dict(state_dict, strict=True)

    return ExpertWrapper(expert, model_type,device,tocpu=True)


def load_00_expert_state_dict(states_dir: str, model_type:str,device: torch.device):
    if model_type=="deepseekmoe":
        index_path = os.path.join(states_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            module_idx = f"model.layers.1.mlp.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]
        # logger.debug(f"Current CPU memory usage11111: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        a=load_file(os.path.join(states_dir, state_fpath), str(device))
        # logger.debug(f"Current CPU memory usage22222: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        return a
    elif model_type == "qwenmoe":
        index_path = os.path.join(states_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            module_idx = f"model.layers.0.mlp.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]
        a=load_file(os.path.join(states_dir, state_fpath), str(device))
        return a
    elif model_type == "xversemoe":
        index_path = os.path.join(states_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            module_idx = f"model.layers.0.mlp.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.gate_proj.weight"]
        a=load_file(os.path.join(states_dir, state_fpath), str(device))
        return a
    else:
        raise ValueError("This model is not supported")


class ExpertWrapper(nn.Module):
    def __init__(
            self,
            expert_module: DeepseekMLP,
            model_type:str,
            device: torch.device,
            tocpu:bool
    ):
        super().__init__()
        self.replace_layer_storage = None
        if model_type == "deepseekmoe":
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if model_type == "qwenmoe":
        #qwenmoe的一个专家构成和deepseekmoe一样。
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if model_type == "xversemoe":
        #qwenmoe的一个专家构成和deepseekmoe一样。
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if self.replace_layer_storage == None:
            raise ValueError("This model is not supported")
        expert_module, self.storage = self.replace_layer_storage(expert_module, device,tocpu)
        self.expert_module = lambda *args, **kwargs: expert_module(*args, **kwargs)
        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)

    @staticmethod
    def _add_storage_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.uint8)
        return state_dict

    def _load_storage_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
                                           unexpected_keys, error_msgs):
        self.storage.copy_(state_dict[prefix + 'storage'].storage().untyped())
        del state_dict[prefix + 'storage']

    def forward(self, *args, **kwargs):
        return self.expert_module(*args, **kwargs)


    def replace_layer_storage_deepseekmoe(self,
            layer: DeepseekMLP,
            device: torch.device,
            tocpu:bool
    ):
        # logger.debug("------")
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        states = [
           ( "gate_proj",  getattr(layer,"gate_proj").weight.data),
           ( "down_proj",  getattr(layer, "down_proj").weight.data),
           ( "up_proj",    getattr(layer, "up_proj").weight.data),
        ]

        storage_size = 0
        offsets = [0]

        for k,x in states:
            if not isinstance(x, torch.Tensor):
                continue
            storage_size += x.nbytes
            offsets.append(storage_size)
        if tocpu:
            pinned_tensor = torch.empty(storage_size, dtype=torch.uint8, device="cpu", pin_memory=True)
            storage = pinned_tensor.untyped_storage()
        else:
            storage = torch.UntypedStorage(storage_size, device=device)
        # logger.debug(f"Current CPU memory usage1: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        i = 0
        # logger.debug("------")
        newtensors = dict()
        for k,x in states:
            if not isinstance(x, torch.Tensor):
                continue
            # logger.debug(k)
            start = offsets[i]
            end = offsets[i + 1]
            if tocpu:
                a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device="cpu").view(x.shape)
            else:
                a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            a_view[...] = x
            assert a_view.data_ptr() == storage.data_ptr() + start
            i += 1
            newtensors[k]=a_view

        for k, newtensor in newtensors.items():
            patched = getattr(layer, k)
            patched.weight.data = newtensor
            assert patched.weight.data.data_ptr() == newtensor.data_ptr()
        return layer, storage
def _make_module_cuda(model_path,model_type,device,state_dict_00):
    config = None
    if model_type == "deepseekmoe":
        config = DeepseekConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    if model_type == "qwenmoe":
        config = Qwen2MoeConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    if model_type == "xversemoe":
        config = XverseConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    with torch.no_grad():
        expert = make_empty_expert(config, model_type)
        expert.load_state_dict(state_dict_00)
    expertwrapper = ExpertWrapper(expert, model_type,device=device,tocpu=False)
    return expertwrapper
def _make_module_cpu(model_path,model_type,device,state_dict_00):
    config = None
    if model_type == "deepseekmoe":
        config = DeepseekConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    if model_type == "qwenmoe":
        config = Qwen2MoeConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    if model_type == "xversemoe":
        config = XverseConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    expert = make_empty_expert(config, model_type)
    expert.load_state_dict(state_dict_00)
    expertwrapper = ExpertWrapper(expert, model_type,device=device,tocpu=True)
    return expertwrapper


def build_model(
    model_path: str,
    model_type: str,
    device: torch.device,
    main_size: int,
    # predictor1: Predictor,
    # predictor2: Predictor
):
    state_dict_00 = load_00_expert_state_dict(model_path,model_type, device)
    if model_type not in current_supportmoe:
        raise ValueError("This model is not supported")
    logger.info("GPU memory allocation for common params begins.")
    if model_type == "deepseekmoe":
        with device, with_default_dtype(torch.bfloat16):
            oldconfig = DeepseekConfig.from_pretrained(
                    model_path,
                    n_routed_experts=0,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = DeepseekForCausalLM(oldconfig)
        model_config = DeepseekConfig.from_pretrained(model_path,trust_remote_code=True)
    elif model_type == "qwenmoe":
        with device, with_default_dtype(torch.bfloat16):
            oldconfig = Qwen2MoeConfig.from_pretrained(
                    model_path,
                    num_experts=0,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = Qwen2MoeForCausalLM(oldconfig)
        model_config = Qwen2MoeConfig.from_pretrained(model_path,trust_remote_code=True)
    elif model_type == "xversemoe":
        with device, with_default_dtype(torch.bfloat16):
            oldconfig = XverseConfig.from_pretrained(
                    model_path,
                    num_experts=0,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = XverseForCausalLM(oldconfig)
        model_config = XverseConfig.from_pretrained(model_path,trust_remote_code=True)
    print("Init GPU memory cost", torch.npu.memory_summary())
    print("GPU memory allocation for common params ends.")
    
    offload_size_ = None
    if model_type == "deepseekmoe":
        offload_size_ = (model_config.num_hidden_layers-1)*model_config.n_routed_experts
    if model_type == "qwenmoe":
        offload_size_ = model_config.num_hidden_layers*model_config.num_experts
    if model_type == "xversemoe":
        offload_size_ = model_config.num_hidden_layers*model_config.num_experts
    if offload_size_ == None:
        raise ValueError("This model is not supported")
    
    expert_cache = ExpertCache(
        model_config,
        make_module_cuda=_make_module_cuda,
        make_module_cpu=_make_module_cpu,
        main_size=main_size,
        offload_size=offload_size_,
        window_size = model_config.window_size,
        state_dict_00=state_dict_00,
        model_type = model_type,
        model_path=model_path
    )
    if model_type == "deepseekmoe":
        for layer_idx in range(model_config.first_k_dense_replace, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.gate.weight = nn.Parameter(torch.empty((model_config.n_routed_experts, model_config.hidden_size),device = model_config.device,dtype=torch.bfloat16))
    if model_type == "qwenmoe":
        for layer_idx in range(0, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.gate = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False,dtype=torch.bfloat16,device=model_config.device)
    if model_type == "xversemoe":
        for layer_idx in range(0, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.router = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False,dtype=torch.bfloat16,device=model_config.device)
    state_index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    trunk_state_path = os.path.join(
        model_path,
        weight_map["model.embed_tokens.weight"],
    )
    logger.info("loading common params...")
    model.load_state_dict(load_file(trunk_state_path), strict=True)
    device = next(model.parameters()).device
    logger.info(f"Model is on device: {device}")
    logger.debug("Common params have loaded.")
    #需要把各层换成带cache的实现
    if model_type == "deepseekmoe":
        init_size = 0
        for layer_idx in trange(1,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.gate.weight
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm
                curr_layer.mlp = DeepseekMoEwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate.weight,
                    curr_layer.mlp.shared_experts,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = DeepseekMoEwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate.weight,
                    curr_layer.mlp.shared_experts,
                    None,None,None,None
                )
            
            for expert_idx in range(model_config.n_routed_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.npu.synchronize(device)
            torch.npu.empty_cache()
    if model_type == "qwenmoe":
        init_size=0
        for layer_idx in trange(0,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.gate
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm

                curr_layer.mlp = Qwen2MoeSparseMoeBlockwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate,
                    curr_layer.mlp.shared_expert,
                    curr_layer.mlp.shared_expert_gate,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = Qwen2MoeSparseMoeBlockwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate,
                    curr_layer.mlp.shared_expert,
                    curr_layer.mlp.shared_expert_gate,
                    None,None,None,None
                )
            for expert_idx in range(model_config.num_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device,
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.npu.synchronize(device)
            torch.npu.empty_cache()
    if model_type == "xversemoe":
        init_size=0
        for layer_idx in trange(0,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.router
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm

                curr_layer.mlp = XverseMoEMLPwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.router,
                    curr_layer.mlp.shared_experts,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = XverseMoEMLPwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.router,
                    curr_layer.mlp.shared_experts,
                    None,None,None,None
                )
            for expert_idx in range(model_config.num_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device,
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.npu.synchronize(device)
            torch.npu.empty_cache()
    logger.info(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
    allocated_memory = torch.cuda.memory_allocated(0)  # 检查设备编号为 0 的 GPU
    logger.info(f"Allocated GPU memory on device 0: {allocated_memory / (1024 ** 2):.2f} MB")

    return model