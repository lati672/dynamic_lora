# Experiment configuration

This directory combines results from two intentional runs over the same cached
task samples. The full-weight family was rerun with its gentler historical
optimization settings; the two LoRA families retain the tuned settings.

## Shared settings

- Base model: `Kt672/Dolma_pretain`
- Tasks: MNLI, QQP, SST-2, SIQA, WinoGrande, FEVER
- Training samples per task: 8,000
- Evaluation samples per task: 1,000
- Batch size: 8
- Maximum sequence length: 256
- Seed: 42
- Weight decay: 0.01
- Warmup ratio: 0.06

## Full-weight

- Epochs: 2
- Learning rate: `2e-5`

## Single LoRA

- Epochs: 4
- Learning rate: `5e-5`
- Rank: 32
- Alpha: 32
- Dropout: 0.05

## Stacked LoRA

- Same epoch, learning-rate, rank, alpha, and dropout settings as single LoRA
- Every evaluation task uses all adapters accumulated through the checkpoint
- Every active adapter has fixed weight `g=1`; no gate is learned
- Orthogonal penalty: effective-update squared Frobenius cosine
- Orthogonal weight: 0.03 with a linear schedule
- Target modules: `q_proj`, `v_proj`, `up_proj`, `down_proj`
