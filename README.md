# dynamic_lora

Self-contained copy of the continual stacked-LoRA experiment.

## Setup

Create and activate a Python environment from the repository root, then install
the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Install the PyTorch build that matches your CUDA environment if the default `torch`
wheel is not appropriate for your machine.

The default model is:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Set `HF_TOKEN` before running if your Hugging Face account needs access to the model:

```bash
export HF_TOKEN=...
```

This repo contains the experiment code, but it does not vendor model weights or
datasets. The first run downloads the configured Hugging Face model and task
datasets into the local Hugging Face cache, then writes generated experiment
outputs under `artifacts/`.

## Commands

There are two CLI entry points:

- `python3 -m dynamic_lora.continual_lora` for continual learning
- `python3 -m dynamic_lora.unlearn` for DPO unlearning of any learned task

Run commands from the repository root after `python3 -m pip install -e .`.

## Layout

This package contains:

- `continual_lora.py`: continual-learning CLI entry point
- `unlearn.py`: task-agnostic DPO unlearning CLI entry point
- `requirements.txt`: Python dependencies for this package
- `core/constants.py`: shared experiment defaults and artifact paths
- `core/adapters.py`: LoRA config, stacked-adapter state, penalties, and model builders
- `core/continual_training.py`: one-task continual training loop
- `core/data_pipeline.py`: task definitions, dataset sampling, and dataloaders
- `core/dpo.py`: DPO loss and classification pair construction helpers
- `core/eval_pipeline.py`: generation-based classification eval
- `core/eval_export.py`: text/JSON eval exports
- `core/io_utils.py`: JSON, memory, and LoRA A/B artifact helpers
- `core/retain_regularization.py`: retain-set projection regularizer
- `core/seed_utils.py`: deterministic seed setup
- `core/lora_app/`: local model/config/data/training helpers

## Continual Learning

Full run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m dynamic_lora.continual_lora \
  --train-samples-per-task 2000 \
  --eval-samples-per-task 200 \
  --epochs 10
```

Quick smoke test:

```bash
python3 -m dynamic_lora.continual_lora \
  --train-samples-per-task 1 \
  --eval-samples-per-task 1 \
  --epochs 1 \
  --output-dir artifacts/dynamic_lora/smoke_test
```

Default outputs go to:

```text
artifacts/dynamic_lora/ag_news_yelp_dbpedia
```

Sampled datasets are cached under:

```text
artifacts/dynamic_lora/datasets
```

## DPO Unlearning

Run DPO unlearning from the saved stacked adapter:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m dynamic_lora.unlearn \
  --unlearn-task dbpedia_14 \
  --unlearn-train-samples 2000 \
  --retain-samples-per-task 100 \
  --quick-eval-samples 200 \
  --epochs 5
```

Supported `--unlearn-task` values are:

```text
ag_news
yelp_review_full
dbpedia_14
```

By default this reads the stacked adapter from:

```text
artifacts/dynamic_lora/ag_news_yelp_dbpedia/final/stack
```

Create this adapter first by running the continual-learning command, or pass a
different adapter path with `--stacked-adapter-dir`.

and writes outputs to:

```text
artifacts/dynamic_lora/unlearn/<unlearn-task>
```

The unlearning objective is:

```text
DPO loss
+ orthogonal_penalty_weight * ||A_previous A_unlearn^T||_1
+ l2_penalty_weight * ||A_unlearn||_2 + ||B_unlearn||_2
+ retain_projection_penalty_weight * ||W_unlearn (I - Pi_unlearn) x_retain||
```

`x_retain` is sampled from tasks learned before `--unlearn-task`. For example, unlearning
`dbpedia_14` samples retain examples from `ag_news` and `yelp_review_full`.
