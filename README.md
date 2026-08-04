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

There are six CLI entry points:

- `python3 -m dynamic_lora.continual_lora` for continual learning
- `python3 -m dynamic_lora.unlearn` for DPO unlearning of any learned task
- `python3 -m dynamic_lora.eval_lora` for evaluating a saved stacked LoRA adapter
- `python3 -m dynamic_lora.spectral_analysis` for singular-vector/intruder analysis
- `python3 -m dynamic_lora.intruder_scale_eval` for scaling LoRA-update intruder singular vectors
- `python3 -m dynamic_lora.load_hf_model` for loading an artifact from Hugging Face

Run commands from the repository root after `python3 -m pip install -e .`.

## Layout

- `continual_lora.py`, `continual_full_finetune.py`, `eval_lora.py`,
  `intruder_scale_eval.py`, `unlearn.py`: CLI entry points
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

Evaluate the three continual full-model checkpoints after AG News, Yelp, and
DBPedia on all three datasets:

```bash
python3 -m dynamic_lora.eval_lora \
  --mode full \
  --all-checkpoints \
  --model-dir artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune
```

This writes the 3-by-3 accuracy matrix to:

```text
outputs/ag_news_yelp_dbpedia_full_finetune/eval_all_checkpoints/accuracy_matrix.csv
```

Use `--mode lora --all-checkpoints` with the stacked-LoRA run directory to
produce the same matrix for LoRA checkpoints.

The current eval path does not compute perplexity, validation loss, or
logits-based classification.

## Load Models From Hugging Face

The artifact repo is `Kt672/dynamic_lora`. Download all continual stacked-LoRA
checkpoints into their expected local paths:

```bash
hf download Kt672/dynamic_lora \
  --include "artifacts/dynamic_lora/ag_news_yelp_dbpedia/**" \
  --local-dir .
```

Download all continual full-finetuned checkpoints:

```bash
hf download Kt672/dynamic_lora \
  --include "artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/**" \
  --local-dir .
```

Load the final continual stacked-LoRA
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
`artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_1_yelp_review_full`.

## Spectral Analysis

From the `dynamic_lora` repository directory, compare full-model and merged
LoRA checkpoints against the pretrained base model:

```bash
cd /workspace/dynamic_lora

python3 spectral_analysis.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --layers 0,8,15 \
  --top-k 20 \
  --modules q_proj,v_proj \
  --lora-checkpoint after_ag_news=artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_0_ag_news \
  --lora-checkpoint after_ag_news_yelp=artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_1_yelp_review_full \
  --lora-checkpoint after_all=artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_2_dbpedia_14 \
  --full-checkpoint after_ag_news=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_0_ag_news_full \
  --full-checkpoint after_ag_news_yelp=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_1_yelp_review_full_full \
  --full-checkpoint after_all=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_2_dbpedia_14_full
```

Run the after-AG-News LoRA and full-model analyses separately so none of their
summary files or heatmaps overwrite each other:

```bash
python3 -m dynamic_lora.spectral_analysis \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --lora-checkpoint lora_after_ag=artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_0_ag_news \
  --layers 0,8,15 \
  --output-dir ./figures/spectral_analysis/lora_after_ag

python3 -m dynamic_lora.spectral_analysis \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --full-checkpoint full_after_ag=artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/task_0_ag_news_full \
  --layers 0,8,15 \
  --output-dir ./figures/spectral_analysis/full_after_ag
```

These commands compare each checkpoint independently against the same
pretrained base model and write their outputs under:

```text
figures/spectral_analysis/lora_after_ag/
figures/spectral_analysis/full_after_ag/
```

Checkpoint arguments are repeatable and their order defines continual-learning
stage order. A LoRA path can point directly to an adapter or to a stage directory
containing `stack/`. Spectral analysis defaults to the trained `q_proj` and
`v_proj` modules; use `--modules` to override them. Outputs include per-matrix heatmaps, `intruder_counts.csv`,
`intruder_summary.png`, and `matching_vector_cosine_mean_over_layers.csv`.
Each checkpoint also gets a `matching_vectors_mean_over_layers.png` heatmap,
which compares each of the top matching singular-vector pairs and averages
their absolute cosine similarities across the selected layers. Per-module
`MODULE_mean_over_layers.png` heatmaps contain the full top-20 by top-20
pairwise cosine-similarity matrix averaged across the selected layers. All
heatmaps use the size selected by `--top-k`, which defaults to 20. By default,
the outputs are written under
`figures/spectral_analysis/` relative to the repository directory. Use
`--output-dir PATH` to choose a different location.

## Intruder Scaling Evaluation

Analyze the LoRA update after DBPedia across every layer and its trained
`q_proj,v_proj` modules. Intruder singular values are
multiplied by `0.5` and `0.1`; each modified update is reconstructed and added
back to the pretrained base weights before evaluation on all three datasets.
The original merged LoRA model is also evaluated first as the `pre_eval`
baseline:

```bash
python3 -m dynamic_lora.intruder_scale_eval \
  --adapter-dir artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_2_dbpedia_14 \
  --modules q_proj,v_proj \
  --top-k 20 \
  --threshold 0.5 \
  --scales 0.5,0.1 \
  --output-dir outputs/intruder_scale_eval/lora_after_dbpedia
```

The reconstruction for each updated matrix is:

```text
DeltaW = U Sigma V^T
DeltaW_scaled = U Sigma_scaled V^T
W_scaled = W0 + DeltaW_scaled
```

The downloaded LoRA checkpoint targets only `q_proj` and `v_proj`, so untrained
modules are excluded from this analysis.

Outputs include:

```text
outputs/intruder_scale_eval/lora_after_dbpedia/module_summary.csv
outputs/intruder_scale_eval/lora_after_dbpedia/intruder_vectors.csv
outputs/intruder_scale_eval/lora_after_dbpedia/accuracy_by_scale.csv
outputs/intruder_scale_eval/lora_after_dbpedia/results.json
```

`accuracy_by_scale.csv` contains one row each for `pre_eval`, `scale_0.5`, and
`scale_0.1`, with their AG News, Yelp, and DBPedia accuracies.

Add `--save-models` to also save each reconstructed full model.

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

# Intruder experiment continual-learning task sequence

The isolated `intruder_experiment` module reproduces the small-scale task
sequence MNLI → QQP → SST-2 → SIQA → WinoGrande → FEVER with a shared encoder,
one classification head per task, full-model continual fine-tuning, a single shared LoRA trained with cross-entropy, and additive stacked LoRA
with orthogonal loss. Samples and their source indices are cached under
`sampled_data/`. Malformed or unlabelled source rows are skipped with warnings.

```bash
python run_intruder_experiment.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir outputs/intruder_experiment \
  --train_samples_per_task 1000 --eval_samples_per_task 500 \
  --task_sequence mnli qqp sst2 siqa winogrande fever \
  --methods full single_lora stacked_lora \
  --rank=16 --lora_alpha 32 --lora_dropout 0.05 \
  --orthogonal_penalty_weight 0.1 --orthogonal_penalty_type effective_update \
  --orthogonal_schedule linear --adapter_eval_mode learned_gates \
  --epochs 10 --batch_size 8 --learning_rate 2e-5 --seed 42
```

For `single_lora`, one shared adapter is updated throughout the task sequence
using cross-entropy only.
For stacked LoRA, the integrated default uses rank 16, a linearly ramped weight
of 0.1, and squared Frobenius-cosine orthogonality between effective `BA`
updates. Learned task gates mix the adapters available when each task is trained;
the gate, task head, and previous adapters are then frozen, preventing later tasks
from changing that task’s inference path.

Run SVD analysis after training (CPU float32 SVD is used even when training used
CUDA):

```bash
python analyze_intruder_experiment.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --checkpoints_dir outputs/intruder_experiment \
  --methods full single_lora stacked_lora \
  --layers 0 8 15 \
  --modules q_proj v_proj up_proj down_proj \
  --top_k 50 --epsilon 0.5 --adapter_eval_mode all

python plot_intruder_results.py \
  --output_dir outputs/intruder_experiment
```

Run the analysis again with `--adapter_eval_mode task_specific` to compare
modes. Each mode has its own cache, and `intruder_counts.csv` combines all
completed modes for plotting. `--overwrite` recomputes only the selected mode.
