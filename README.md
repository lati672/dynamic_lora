# dynamic_lora

Self-contained copy of the continual stacked-LoRA experiment.

## Setup

Create and activate a Python environment from the repository root, then install
the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Install the PyTorch build that matches your CUDA environment if the default `torch`
wheel is not appropriate for your machine.

The default model is:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Log in with the Hugging Face CLI before running if your account needs access to the model:

```bash
hf auth login
```

This repo contains the experiment code, but it does not vendor model weights or
datasets. The first run downloads the configured Hugging Face model and task
datasets into the local Hugging Face cache, then writes model checkpoints under
`artifacts/` and eval result summaries under `outputs/`.

## Commands

There are five CLI entry points:

- `python3 -m dynamic_lora.continual_lora` for continual learning
- `python3 -m dynamic_lora.unlearn` for DPO unlearning of any learned task
- `python3 -m dynamic_lora.eval_lora` for evaluating a saved stacked LoRA adapter
- `python3 -m dynamic_lora.spectral_analysis` for singular-vector/intruder analysis
- `python3 -m dynamic_lora.load_hf_model` for loading an artifact from Hugging Face

Run commands from the repository root after `python3 -m pip install -e .`.

## Layout

- `continual_lora.py`, `continual_full_finetune.py`, `eval_lora.py`, `unlearn.py`: CLI entry points
- `core/`: experiment implementation
- `requirements.txt`, `pyproject.toml`: Python dependency and package metadata

## Continual Learning

Stacked LoRA run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m dynamic_lora.continual_lora
```

Full finetuning run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m dynamic_lora.continual_full_finetune
```

The stacked LoRA run defaults to 2000 train samples per task, 200 eval samples
per task, and 10 epochs. Full finetuning uses gentler continual-learning
defaults: 3 epochs, a `2e-5` learning rate, and no replay. This keeps the
full-model run directly comparable with stacked LoRA for alignment analysis.

Enable full-model replay as an optional stronger continual-learning baseline:

```bash
python3 -m dynamic_lora.continual_full_finetune \
  --replay-samples-per-previous-task 500
```

Quick smoke test:

```bash
python3 -m dynamic_lora.continual_lora \
  --train-samples-per-task 1 \
  --eval-samples-per-task 1 \
  --epochs 1 \
  --output-dir artifacts/dynamic_lora/smoke_test
```

Default model/checkpoint outputs go to:

```text
artifacts/dynamic_lora/ag_news_yelp_dbpedia
```

Full finetuning model/checkpoint outputs go to:

```text
artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune
```

Eval summaries from training go to:

```text
outputs/ag_news_yelp_dbpedia
outputs/ag_news_yelp_dbpedia_full_finetune
```

## Evaluation

`eval_lora` evaluates a saved stacked LoRA adapter with generation-based
classification accuracy on:

```text
ag_news
yelp_review_full
dbpedia_14
```

Run evaluation on the saved adapter:

```bash
python3 -m dynamic_lora.eval_lora \
  --adapter-dir artifacts/dynamic_lora/ag_news_yelp_dbpedia/final/stack
```

Common eval args:

```text
--adapter-dir
--output-dir
--eval-samples-per-task
--eval-seed
--max-new-tokens
--model-id
--max-length
```

By default this writes fresh eval outputs to:

```text
outputs/ag_news_yelp_dbpedia/eval
```

The current eval path does not compute perplexity, validation loss, or
logits-based classification.

## Load Models From Hugging Face

The artifact repo is `Kt672/artifacts`. Load the final continual stacked-LoRA
model, including its pretrained base model:

```bash
python3 -m dynamic_lora.load_hf_model \
  --mode lora \
  --prompt "Classify this news article: The team won the championship."
```

Load the final continual full-finetuned model:

```bash
python3 -m dynamic_lora.load_hf_model \
  --mode full \
  --prompt "Classify this news article: The team won the championship."
```

The reusable Python API is:

```python
from dynamic_lora.core.hf_model_loader import load_hf_continual_model

tokenizer, lora_model = load_hf_continual_model("lora")
tokenizer, full_model = load_hf_continual_model("full")
```

Use `subfolder=` to load an intermediate continual-learning checkpoint. For
LoRA, provide the checkpoint folder containing `stack/`, such as
`dynamic_lora/ag_news_yelp_dbpedia/task_1_yelp_review_full`.

## Spectral Analysis

Compare full-model and merged LoRA checkpoints against the pretrained base model:

```bash
python3 -m dynamic_lora.spectral_analysis \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --layers 0,8,15 \
  --modules q_proj,v_proj,up_proj,down_proj \
  --lora-checkpoint after_ag_news=artifects/ag_news_yelp_dbpedia/task_0_ag_news \
  --lora-checkpoint after_ag_news_yelp=artifects/ag_news_yelp_dbpedia/task_1_yelp_review_full \
  --lora-checkpoint after_all=artifects/ag_news_yelp_dbpedia/task_2_dbpedia_14 \
  --full-checkpoint after_ag_news=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_0_ag_news_full \
  --full-checkpoint after_ag_news_yelp=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_1_yelp_review_full_full \
  --full-checkpoint after_all=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_2_dbpedia_14_full
```

Checkpoint arguments are repeatable and their order defines continual-learning
stage order. A LoRA path can point directly to an adapter or to a stage directory
containing `stack/`. Outputs include per-matrix heatmaps, `intruder_counts.csv`,
and `intruder_summary.png`.

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

Unlearning eval summaries go to:

```text
outputs/unlearn/<unlearn-task>
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
