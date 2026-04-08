# SMoEaligned

SMoE inference acceleration framework with GPU/CPU expert caching for Mixture-of-Experts LLMs. SMoE leverages expert importance to guide decisions, substituting low-importance active experts with functionally similar ones already cached in GPU memory, thereby preserving accuracy. 

Supports three models: **DeepSeek-MoE**, **Qwen2-MoE**, **Xverse-MoE**.

---

## Requirements

```bash
conda activate SMoE
pip install -r requirements.txt
```
---

## Quick Start

### Run with `run.sh`

```bash
# Qwen2-MoE
MODEL_NAME=qwenmoe \
MODEL_PATH=/path/to/qwen2_moe \
bash run.sh

# DeepSeek-MoE
MODEL_NAME=deepseekmoe \
MODEL_PATH=/path/to/deepseekmoe \
bash run.sh

# Xverse-MoE
MODEL_NAME=xversemoe \
MODEL_PATH=/path/to/xversemoe \
bash run.sh
```

### Run with `main.py` directly

```bash
python main.py \
    --model_name   qwenmoe \
    --model_path   /path/to/qwen2_moe \
    --config_path  configs/qwen2moe_config.json \
    --dataset_path wic \
    --input_num    20 \
    --output_len   100 \
    --cpu_cores    16 \
    --GPU_mem      24
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `qwenmoe` | Model to run: `qwenmoe` \| `deepseekmoe` \| `xversemoe` |
| `--model_path` | *(empty)* | Path to model weights directory. If empty, uses the hardcoded default in `main.py` |
| `--config_path` | *(empty)* | Path to SMoE config JSON (see `configs/`). If empty, falls back to `config.json` inside model directory |
| `--dataset_path` | `wic` | Dataset name or path passed to `utils/load_dataset.py` |
| `--input_num` | `20` | Number of prompts to run |
| `--batch_size` | `1` | Batch size per forward pass |
| `--output_len` | `100` | Max new tokens to generate per prompt |
| `--GPU_mem` | `24` | GPU memory in GB, used to compute expert cache offload size |
| `--cpu_cores` | `3` | Number of CPU cores allocated to inference (n-1 for compute, 1 for loading/bg worker) |
| `--debug` | `False` | Enable debugpy remote debugger on port 9501 |

## SMoE Config JSON

Each model has a config JSON under `configs/`:

Key SMoE-specific fields:

| Field | Description |
|---|---|
| `replaceScoreRatio` | Ratio about experts replaced|
| `window_size` | SCore-eviction cache method window size (`null` = LRU) |
| `if_prefetch` | Enable background prefetch prediction |
| `if_usecpu` | Enable CPU fallback for cache-miss experts |
| `if_replace` | Enable expert cache replacement |

The expert cache size (number of experts kept on GPU) is set via `--GPU_mem` in `main.py` / `GPU_MEM` in `run.sh` and is computed automatically inside `build_model()`.

---

### `run.sh` environment variables

All `run.sh` arguments are controlled via environment variables (same names, uppercase):

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `qwenmoe` | Same as `--model_name` |
| `MODEL_PATH` | *(empty)* | Same as `--model_path` |
| `CONFIG_PATH` | *(empty)* | Same as `--config_path` |
| `DATASET_PATH` | `wic` | Same as `--dataset_path` |
| `INPUT_NUM` | `20` | Same as `--input_num` |
| `BATCH_SIZE` | `1` | Same as `--batch_size` |
| `OUTPUT_LEN` | `100` | Same as `--output_len` |
| `GPU_MEM` | `24` | Same as `--GPU_mem` |
| `CPU_CORES` | `16` | Same as `--cpu_cores` |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_DIR` | `./logs` | Directory for log files |
| `CONDA_ENV` | `Nmoe` | Conda environment to activate |

---

