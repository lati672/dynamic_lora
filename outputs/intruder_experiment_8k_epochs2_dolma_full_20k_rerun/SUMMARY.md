# Continual-Learning Experiment Summary

## Experiment setup

This experiment compares three continual-learning methods initialized from the
same Llama-3.2-1B backbone after full-parameter adaptation on 20,000 Dolma
documents:

- full-weight fine-tuning;
- one shared LoRA adapter updated throughout the sequence; and
- stacked LoRA with a separate adapter per task and every available adapter
  fixed at weight 1.

The fixed task order is MNLI, QQP, SST-2, SIQA, WinoGrande, and FEVER. Each
method uses the same sampled data: 8,000 training examples per task and up to
1,000 evaluation examples per task (SST-2 contributes 872 evaluation examples).
Training runs for two epochs per task with batch size 8 and learning rate
`2e-5`. Both LoRA methods use rank 16, alpha 32, and dropout 0.05. Stacked LoRA uses effective-update orthogonality with a linearly
scheduled weight of 0.1 and additive evaluation of all adapters learned so far.

## Aggregate results

| Method | Mean accuracy when each task was learned | Final average accuracy | Average forgetting | Worst forgetting |
|---|---:|---:|---:|
| Full-weight | **77.92%** | 72.44% | 5.48 pp | 17.50 pp |
| Single LoRA | 73.59% | 70.56% | 3.08 pp | 12.70 pp |
| Stacked LoRA | 73.42% | **71.45%** | **2.04 pp** | **6.20 pp** |

Accuracy when learned is the score recorded immediately after training a task.
Forgetting is the highest score observed for a task minus its score after the
final FEVER stage.

## Final accuracy after FEVER

| Task | Full-weight | Single LoRA | Stacked LoRA |
|---|---:|---:|---:|
| MNLI | 67.10% | 67.80% | **74.20%** |
| QQP | 81.40% | **82.00%** | 80.70% |
| SST-2 | 89.33% | 91.86% | **92.78%** |
| SIQA | **67.80%** | 63.10% | 63.70% |
| WinoGrande | **59.00%** | 51.70% | 52.10% |
| FEVER | **70.00%** | 66.90% | 65.20% |
| **Average** | **72.44%** | 70.56% | 71.45% |

## Findings

Full-weight tuning has the greatest plasticity. It achieves the highest average
accuracy immediately after tasks are learned and leads on the final three tasks,
SIQA, WinoGrande, and FEVER. That advantage comes with the most forgetting.
MNLI falls from 84.60% after its own training stage to 67.10% after FEVER, a
17.50-point reduction.

Stacked LoRA with every available adapter fixed at weight 1 has the lowest
forgetting: 2.04 points on average and 6.20 points at worst. It retains earlier
tasks better than full tuning and the shared single adapter, but summing later
adapters still changes the inference path and causes measurable interference.
Its final average is 71.45%, between full tuning and single LoRA.

Single LoRA is intermediate in forgetting but weakest in final average. Updating
one shared low-rank adapter across the entire sequence still overwrites earlier
knowledge: MNLI loses 12.70 points. In this run it does not match full tuning on
new-task learning or stacked LoRA on retention.

The split between the first and last three tasks illustrates the
stability-plasticity tradeoff. Freezing earlier stacked adapters protects old
tasks, while unrestricted full tuning adapts more strongly to later tasks at the
cost of changing representations needed by earlier ones.

## Limitations

These results come from one seed, one task order, and one sample per task.
Repeated seeds and alternate or reversed task orders are needed to determine
whether the rankings are statistically robust. Stacked LoRA also retains a separate task-specific head and adapter,
so its retention should be interpreted in light of that growing task-specific capacity
and an increasing number of summed adapters.

## Artifacts

Per-stage accuracy and loss are stored in:

- `full_finetune/results.csv`
- `single_lora/results.csv`
- `../intruder_experiment_8k_epochs2_dolma_full_20k_all_adapters/stacked_lora/results.csv`

Each method directory also contains a checkpoint after every task stage.
