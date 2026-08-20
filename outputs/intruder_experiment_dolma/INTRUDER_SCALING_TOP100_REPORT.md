# Top-100 full-weight intruder-scaling report

## Objective

This follow-up experiment tests whether suppressing singular components classified
as intruders improves retention in the full-weight continual-learning model. It
post-processes each saved full-weight checkpoint independently; it does not
retrain the encoder or classification heads.

## Intervention

For each checkpoint, the experiment processes all 16 transformer layers and the
`q_proj`, `v_proj`, `up_proj`, and `down_proj` matrices. It computes a
reduced SVD and considers only the top 100 singular triplets from both the
Dolma-pretrained reference and the tuned matrix.

A tuned left singular vector `u_i` is an intruder when

```text
max_j |u_base_j^T u_i| < epsilon
```

where both `i` and `j` are restricted to the top 100 vectors and
`epsilon = 0.8`. Each detected tuned component is changed using

```text
W_scaled = W_tuned + (lambda - 1) * sum_i u_i sigma_i v_i^T
```

Two values from the configured discrete set were evaluated:

- `lambda = 0.5`: halve every detected intruder component.
- `lambda = 0`: remove every detected intruder component.

The same task-aware classification heads and sampled evaluation examples as the
main experiment are used. No scaled model checkpoints were saved.

## Aggregate continual-learning results

| Model | Final average accuracy | Average forgetting |
|---|---:|---:|
| Original full-weight | **72.10%** | **6.88 points** |
| Top-100, epsilon 0.8, lambda 0.5 | 71.70% | 7.17 points |
| Top-100, epsilon 0.8, lambda 0 | 71.28% | 7.43 points |

Average forgetting is the mean, over the first five tasks, of the task's
accuracy immediately after acquisition minus its accuracy after the final FEVER
stage.

## Final accuracy after FEVER

| Task | Original full-weight | Lambda 0.5 | Lambda 0 |
|---|---:|---:|---:|
| MNLI | **62.40%** | 61.40% | 60.60% |
| QQP | 82.20% | **82.50%** | 82.20% |
| SST-2 | **89.79%** | 89.22% | 88.88% |
| SIQA | **68.80%** | 68.60% | 68.00% |
| WinoGrande | **58.70%** | 58.10% | 58.30% |
| FEVER | **70.70%** | 70.40% | 69.70% |

## Accuracy at task acquisition

| Task | Original full-weight | Lambda 0.5 | Lambda 0 |
|---|---:|---:|---:|
| MNLI | 84.10% | **84.20%** | 83.90% |
| QQP | **86.10%** | 85.90% | **86.10%** |
| SST-2 | **94.50%** | 94.38% | 93.92% |
| SIQA | **71.40%** | 71.00% | 70.50% |
| WinoGrande | 60.20% | 60.20% | **60.70%** |
| FEVER | **70.70%** | 70.40% | 69.70% |

## Per-task forgetting

The final FEVER task is excluded because there is no later training stage after
its acquisition.

| Task | Original full-weight | Lambda 0.5 | Lambda 0 |
|---|---:|---:|---:|
| MNLI | **21.70** | 22.80 | 23.30 |
| QQP | 3.90 | **3.40** | 3.90 |
| SST-2 | **4.70** | 5.16 | 5.05 |
| SIQA | 2.60 | **2.40** | 2.50 |
| WinoGrande | **1.50** | 2.10 | 2.40 |
| Mean | **6.88** | 7.17 | 7.43 |

Values are percentage points. Lambda 0.5 improves QQP and SIQA retention
slightly, but those gains do not offset worse MNLI, SST-2, and WinoGrande
retention.

## Intruder counts

The detected intruder identities and counts are the same for both lambda values;
lambda controls only the amount of scaling.

### Events by checkpoint stage

| Stage | Intruder events |
|---|---:|
| MNLI | 14 |
| QQP | 22 |
| SST-2 | 53 |
| SIQA | 77 |
| WinoGrande | 77 |
| FEVER | 109 |
| **Total** | **352** |

### Events by projection type

| Module | Intruder events | Share |
|---|---:|---:|
| `q_proj` | 0 | 0.0% |
| `v_proj` | 251 | 71.3% |
| `up_proj` | 77 | 21.9% |
| `down_proj` | 24 | 6.8% |
| **Total** | **352** | **100.0%** |

Most top-100 intruders occur in `v_proj`. Intruder events also become more
frequent later in the continual-learning sequence.

## Interpretation

Neither intervention improves the original full-weight result. Halving detected
components is less harmful than removing them, but lambda 0.5 still lowers final
average accuracy by 0.40 points and increases average forgetting by 0.29 points.
Lambda 0 lowers final average accuracy by 0.82 points and increases forgetting
by 0.55 points.

This result argues against treating every threshold-defined intruder as purely
harmful. Some detected components appear to contribute useful task information,
or their removal perturbs representations used by the task heads. The monotonic
increase in intruder events across later checkpoints can coexist with forgetting
without establishing that the events cause forgetting.

Restricting the intervention to the top 100 is much less destructive than the
earlier all-vector intervention, which produced 64.45% final average accuracy
and 13.77 points of forgetting at lambda 0.5. The top-100 experiment therefore
shows that intervention scope matters substantially, although the restricted
version still does not outperform the unmodified model.

## Limitations

- Results use one model, one task order, and one random seed.
- Intruder classification depends on the top-100 cutoff and epsilon 0.8.
- The intervention is applied independently after each checkpoint rather than
  during continual training.
- Only four projection types are modified.
- The experiment measures association and post-hoc intervention effects, not a
  general causal theory of catastrophic forgetting.

## Outputs

### Lambda 0.5

- `full_finetune_intruder_scaled_top100_eps0.8_lambda0.5/results.csv`
- `full_finetune_intruder_scaled_top100_eps0.8_lambda0.5/intruder_scaling.csv`
- `full_finetune_intruder_scaled_top100_eps0.8_lambda0.5/config.json`

### Lambda 0

- `full_finetune_intruder_scaled_top100_eps0.8_lambda0/results.csv`
- `full_finetune_intruder_scaled_top100_eps0.8_lambda0/intruder_scaling.csv`
- `full_finetune_intruder_scaled_top100_eps0.8_lambda0/config.json`
