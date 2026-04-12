# OpenCompass Evaluation for SMoE

This directory contains the OpenCompass-based benchmark evaluation integration for the SMoE project. It supports evaluating DeepSeek MoE, Xverse MoE, and Qwen MoE models on standard benchmarks.

## Supported Benchmarks

- GaokaoBench
- GSM8K
- RACE
- TriviaQA
- WiC

## Environment Setup

### 1. Create and activate the conda environment

```bash
conda create -n test_SMoE python=3.10
conda activate test_SMoE
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Install the Custom Model Adapter

Copy `mymodel.py` into the OpenCompass models directory so the framework can discover the custom MoE model classes:

```bash
OPENCOMPASS_DIR=$(python -c "import opencompass, os; print(os.path.dirname(opencompass.__file__))")
cp mymodel.py $OPENCOMPASS_DIR/models/mymodel.py
```

### Register the models in OpenCompass

Append the following lines to `$OPENCOMPASS_DIR/models/__init__.py`:

```python
from .mymodel import MydeepseekmoeModel
from .mymodel import MyxversemoeModel
from .mymodel import MyqwenmoeModel
```

You can do this with:

```bash
OPENCOMPASS_MODELS_INIT="$OPENCOMPASS_DIR/models/__init__.py"
echo "" >> $OPENCOMPASS_MODELS_INIT
echo "from .mymodel import MydeepseekmoeModel" >> $OPENCOMPASS_MODELS_INIT
echo "from .mymodel import MyxversemoeModel" >> $OPENCOMPASS_MODELS_INIT
echo "from .mymodel import MyqwenmoeModel" >> $OPENCOMPASS_MODELS_INIT
```

## Configuration

Set the following environment variables before running:

### `SMOE_ROOT` — SMoE project root (required)

`mymodel.py` delegates model loading to the SMoE project. Point it to your SMoE checkout:

```bash
export SMOE_ROOT=/path/to/SMoE
```

### `MODEL_BASE` — model weights directory (required)

Set this to the directory containing your model weight folders:

```bash
export MODEL_BASE=/path/to/your/models
```

The expected directory structure under `MODEL_BASE` is:

```
$MODEL_BASE/
├── qwenmoe/model_path/       # Qwen MoE model weights
├── deepseekmoe/model_path/   # DeepSeek MoE model weights (optional)
└── xversemoe/model_path/     # Xverse MoE model weights (optional)
```

By default (if `MODEL_BASE` is not set), it falls back to `../parameters` relative to this directory, which corresponds to the SMoE project's `parameters/` folder (already in `.gitignore`).

## Run Evaluation

```bash
conda activate test_SMoE
export SMOE_ROOT=/path/to/SMoE
export MODEL_BASE=/path/to/your/models
opencompass opencompasstest.py
```

To switch between models, edit `opencompasstest.py` and uncomment the desired model block (`MydeepseekmoeModel`, `MyxversemoeModel`, or `MyqwenmoeModel`).
