---
license: apache-2.0

inference: false

---

# XVERSE-MoE-A4.2B-Chat


## 模型介绍

**XVERSE-MoE-A4.2B-Chat**为 **XVERSE-MoE-A4.2B** 底座模型对齐后的版本。

**XVERSE-MoE-A4.2B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），使用混合专家模型（MoE，Mixture-of-experts）架构，模型的总参数规模为 258 亿，实际激活的参数量为 42 亿，本次开源的模型为底座模型 **XVERSE-MoE-A4.2B**，主要特点如下：

- **模型结构**：XVERSE-MoE-A4.2B 为 Decoder-only 的 Transformer 架构，将密集模型的 FFN 层扩展为专家层，不同于传统 MoE 中每个专家的大小与标准 FFN 相同（如Mixtral 8x7B ），使用了更细粒度的专家，每个专家是标准 FFN 大小的 1/4，并设置了共享专家（Shared Expert）和非共享专家（Non-shared Expert）两类，共享专家在计算时始终被激活，非共享专家通过 Router 选择性激活。
- **训练数据**：构建了 2.7 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果；模型使用 8K 长度的训练样本进行训练。
- **训练框架**：针对 MoE 模型中独有的专家路由和权重计算逻辑，进行了深入定制优化，开发出一套高效的融合算子，以提升计算效率。同时，为解决 MoE 模型显存占用和通信量大的挑战，设计了计算、通信和 CPU-Offload 的 Overlap 处理方式，从而提高整体吞吐量。

**XVERSE-MoE-A4.2B** 的模型大小、架构和学习率如下：

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    25.8B     |       4.2B       |    28    |  2560   |   32    | 1728 |          64          |        2         |   6   | 3.5e−4 |

## Model Introduction

**XVERSE-MoE-A4.2B-Chat** is the aligned version of model **XVERSE-MoE-A4.2B**.

**XVERSE-MoE-A4.2B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology which is using Mixture-of-experts (MoE) architecture. The total parameter scale of the model is 25.8 billion, with an actual number of activated parameters being 4.2 billion. The models released this time is the base model **XVERSE-MoE-A4.2B**. Its key features are as follows:

- **Model Structure**: XVERSE-MoE-A4.2B uses the mainstream Decoder-only Transformer network structure that extends the FFN layer of dense models to expert layers. Unlike traditional MoE model where each expert has the same size as standard FFN (such as Mixtral 8x7B), it uses more fine-grained experts, with each expert being 1/4 the size of a standard FFN. It includes shared experts and non-shared experts, where shared experts are always activated during computation, and non-shared experts are selectively activated through a Router.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 2.7 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages; The model is trained using training samples of length 8k.
- **Training Framework**: We conducted in-depth customized optimization for the unique expert routing and weight calculation logic in the MoE model, developed an efficient fusion operator to improve computational efficiency. At the same time, to address the challenges of high memory consumption and communication volume in the MoE model, we designed a processing method for overlapping computation, communication, and CPU-Offload to increase overall throughput.

The models sizes, architectures and learning rate of **XVERSE-MoE-A4.2B** are showed as follows:

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    25.8B     |       4.2B       |    28    |  2560   |   32    | 1728 |          64          |        2         |   6   | 3.5e−4 |


## 使用方法

### Transformers 加载方式

可通过以下代码加载 XVERSE-MoE-A4.2B-Chat 模型来进行推理：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-MoE-A4.2B-Chat")
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-MoE-A4.2B-Chat", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()
inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
inputs = inputs.cuda()
generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```


## Usage

### Loading with Transformers

The XVERSE-MoE-A4.2B-Chat model can be loaded for inference using the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-MoE-A4.2B-Chat")
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-MoE-A4.2B-Chat", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()
inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
inputs = inputs.cuda()
generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```


## 局限性与免责申明

XVERSE-MoE-A4.2B-Chat 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-MoE-A4.2B-Chat 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-MoE-A4.2B-Chat 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-MoE-A4.2B-Chat 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](https://github.com/xverse-ai/XVERSE-MoE-A4.2B/blob/main/LICENSE) 开源协议，使用 XVERSE-MoE-A4.2B-Chat 的模型权重则需要遵循[模型许可协议](https://github.com/xverse-ai/XVERSE-MoE-A4.2B/blob/main/MODEL_LICENSE.pdf)。

XVERSE-MoE-A4.2B-Chat 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-MoE-A4.2B-Chat may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-MoE-A4.2B-Chat, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-MoE-A4.2B-Chat model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-MoE-A4.2B-Chat model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](https://github.com/xverse-ai/XVERSE-MoE-A4.2B/blob/main/LICENSE) open-source license, while the use of the model weights of XVERSE-MoE-A4.2B-Chat needs to adhere to the [Model License Agreement](https://github.com/xverse-ai/XVERSE-MoE-A4.2B/blob/main/MODEL_LICENSE.pdf).

The XVERSE-MoE-A4.2B-Chat model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.