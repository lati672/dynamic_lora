# 10-Epoch Continual-Learning Experiment Analysis

## Summary

The 10-epoch run substantially improves stacked LoRA, but it does not improve full fine-tuning overall. More epochs also increase forgetting and show signs of overfitting.

The experiment used the task sequence MNLI → QQP → SST-2 → SIQA → WinoGrande → FEVER, with 1,000 training samples and 500 evaluation samples per task. Stacked LoRA was evaluated with all accumulated adapters active.

## Final 10-Epoch Comparison

| Task | Full FT | Stacked LoRA | Winner |
|---|---:|---:|---|
| MNLI | **73.6%** | 67.6% | Full +6.0 pp |
| QQP | **79.2%** | 73.8% | Full +5.4 pp |
| SST-2 | 90.0% | **90.6%** | LoRA +0.6 pp |
| SIQA | 51.6% | **60.0%** | LoRA +8.4 pp |
| WinoGrande | 53.4% | **53.8%** | LoRA +0.4 pp |
| FEVER | **64.4%** | 62.0% | Full +2.4 pp |
| **Average** | **68.70%** | **67.97%** | Full +0.73 pp |

Full fine-tuning still leads, but the gap narrowed from 4.17 percentage points at three epochs to only 0.73 points at ten epochs.

## Effect of Increasing From 3 to 10 Epochs

| Task | Full FT change | Stacked LoRA change |
|---|---:|---:|
| MNLI | +7.0 pp | +5.2 pp |
| QQP | +3.0 pp | -4.0 pp |
| SST-2 | -3.6 pp | -2.0 pp |
| SIQA | -8.6 pp | **+13.8 pp** |
| WinoGrande | -1.0 pp | +2.6 pp |
| FEVER | +0.2 pp | +2.0 pp |
| **Average** | **-0.5 pp** | **+2.93 pp** |

The most important change is SIQA: extra training raises stacked LoRA from 46.2% to 60.0%, while full fine-tuning falls from 60.2% to 51.6%.

## Task Acquisition

Accuracy immediately after each task was trained:

| Task | Full FT | Stacked LoRA |
|---|---:|---:|
| MNLI | 80.4% | 76.6% |
| QQP | 81.6% | 79.2% |
| SST-2 | 91.2% | 91.8% |
| SIQA | 64.4% | 59.6% |
| WinoGrande | 52.6% | 48.8% |
| FEVER | 64.4% | 62.0% |
| **Average** | **72.43%** | **69.67%** |

LoRA's task-acquisition deficit shrank from 6.6 points at three epochs to 2.77 points at ten epochs. This suggests that three epochs were insufficient for the lower-capacity adapters, particularly on MNLI and SIQA.

## Forgetting

Peak-to-final accuracy reduction over the first five tasks:

| Task | Full FT | Stacked LoRA |
|---|---:|---:|
| MNLI | 6.8 pp | 9.0 pp |
| QQP | 2.4 pp | 5.4 pp |
| SST-2 | 1.6 pp | 1.2 pp |
| SIQA | **12.8 pp** | 0.0 pp |
| WinoGrande | 0.0 pp | 0.0 pp |
| **Average** | 4.72 pp | **3.12 pp** |

Stacked LoRA still forgets less overall, but its MNLI and QQP retention worsened with more epochs. Full fine-tuning's dominant problem is SIQA: it drops from 64.4% immediately after training to 51.6% after FEVER.

## Evidence of Overfitting

Several final evaluation losses increased substantially even though training losses became very low:

| Method and task | 3 epochs | 10 epochs |
|---|---:|---:|
| Full FT, WinoGrande | 0.771 | 1.420 |
| Full FT, SIQA | 0.979 | 1.204 |
| Full FT, SST-2 | 0.184 | 0.311 |
| Stacked LoRA, WinoGrande | 0.699 | 0.807 |
| Stacked LoRA, FEVER | 0.952 | 1.139 |

This suggests that ten epochs are excessive for some tasks when training on only 1,000 examples per task.

## Conclusions

- Ten epochs benefit stacked LoRA, improving its final average accuracy by 2.93 percentage points.
- Ten epochs do not benefit full fine-tuning overall; its final average falls by 0.5 points.
- Full fine-tuning remains marginally stronger in final average accuracy, but stacked LoRA nearly closes the gap while retaining lower average forgetting.
- A single epoch count is probably not optimal across all tasks.
- Validation-based early stopping or task-specific epoch counts could retain LoRA's SIQA gains without increasing MNLI/QQP interference and evaluation loss.
- A useful next comparison would use five or six epochs, or validation-based early stopping.

## Source Files

- `full_finetune/results.csv`: 10-epoch full fine-tuning results
- `stacked_lora/results.csv`: 10-epoch stacked-LoRA results
- `config.json`: experiment configuration
- `../intruder_experiment/full_finetune/results.csv`: original 3-epoch full fine-tuning results
- `../intruder_experiment/stacked_lora/results.csv`: original 3-epoch stacked-LoRA results
