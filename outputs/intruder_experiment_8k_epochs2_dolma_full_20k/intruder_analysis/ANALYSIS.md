# Intruder Analysis: Dolma 20k Backbone, 8k Examples per Task

## Setup

- Backbone: `artifacts/dolma_full_20k/final`
- Sequence: MNLI → QQP → SST-2 → SIQA → WinoGrande → FEVER
- Training: 8,000 examples per task, 2 epochs
- Methods: full fine-tuning and additive stacked LoRA (all adapters active)
- SVD: layers 0, 8, and 15; q/v/up/down projections; top 150 vectors
- Intruder threshold: maximum base-vector similarity below 0.5

## Main result

No formal intruder vectors were detected for either method in any of the 144
method/stage/layer/module combinations. The thresholded count is therefore zero,
but continuous singular-vector alignment still separates the methods:

| Method | Mean maximum similarity |
|---|---:|
| Full fine-tuning | 0.98540 |
| Stacked LoRA | 0.99670 |

Stacked LoRA remains closer to the Dolma-adapted starting subspaces.

## Evolution over tasks

| Stage | Full FT | Stacked LoRA |
|---|---:|---:|
| MNLI | 0.99472 | 0.99862 |
| QQP | 0.99203 | 0.99792 |
| SST-2 | 0.98746 | 0.99713 |
| SIQA | 0.98333 | 0.99569 |
| WinoGrande | 0.97953 | 0.99554 |
| FEVER | 0.97531 | 0.99530 |

Full fine-tuning accumulates monotonic subspace drift. Stacked LoRA also drifts across the sequence, but by a much smaller amount.

## Module and layer sensitivity

The value projection is the dominant source of drift:

| Method | q_proj | v_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|
| Full FT | 0.99941 | 0.96813 | 0.98851 | 0.98552 |
| Stacked LoRA | 0.99960 | 0.99250 | 0.99627 | 0.99843 |

Layers 0 and 8 move more than layer 15. Full-model mean similarities are
0.97143, 0.98675, and 0.99801 for layers 0, 8, and 15 respectively. The lowest
individual alignment is full fine-tuning after FEVER at layer 0 `v_proj`
(0.90519).

## Relation to accuracy

Full fine-tuning shows more geometric drift and more average forgetting
(7.78 percentage points), while stacked LoRA has higher alignment and lower
forgetting (2.48 points). This is consistent with parameter isolation preserving
the backbone subspace. It is an association, not proof that singular-vector
drift causes forgetting.

## Interpretation limitation

The 0.5 threshold is too permissive for this experiment: all top-150 tuned
vectors retain a base-vector match above it. Intruder counts alone consequently
hide meaningful differences. Mean/max-similarity trajectories should be
reported, or a stricter threshold should be declared in advance for a separate
sensitivity analysis.
