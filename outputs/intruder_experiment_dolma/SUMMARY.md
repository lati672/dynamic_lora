# Intruder continual-learning experiment summary

## Objective

This experiment compares three continual-learning strategies initialized from
the Dolma-pretrained `Kt672/Dolma_pretain` model:

1. Full-weight continual fine-tuning
2. One shared continually updated LoRA
3. Additive stacked LoRA with one frozen adapter per task

The task sequence is:

`MNLI → QQP → SST-2 → SIQA → WinoGrande → FEVER`

All methods use separate task-aware classification heads. The caller supplies
the task identity during training and evaluation, which selects the appropriate
head. For stacked LoRA, every task uses all adapters accumulated through the
current checkpoint.

## Data

Each task uses 8,000 reproducibly sampled training examples and up to 1,000
evaluation examples with seed 42. SST-2 has only 872 usable validation examples.
The cached source indices in `sampled_data/` ensure that all three methods use
the same examples.

The tasks cover natural-language inference (MNLI), duplicate-question detection
(QQP), sentiment classification (SST-2), social commonsense reasoning (SIQA),
commonsense/coreference reasoning (WinoGrande), and factual verification
(FEVER).

## Training configuration

The saved directory combines a gentler full-weight rerun with the tuned LoRA
runs:

| Method | Epochs | Learning rate | Rank | Alpha |
|---|---:|---:|---:|---:|
| Full-weight | 2 | 2e-5 | — | — |
| Single LoRA | 4 | 5e-5 | 32 | 32 |
| Stacked LoRA | 4 | 5e-5 | 32 | 32 |

All runs use batch size 8, maximum length 256, weight decay 0.01, and warmup
ratio 0.06.

For stacked LoRA, adapters target `q_proj`, `v_proj`, `up_proj`, and
`down_proj`. Each task adds a new adapter while the backbone, previous
adapters, and previous heads remain frozen. Every adapter accumulated through
the current stage has fixed weight `g=1`; no gate is learned. New effective updates
`BA` receive squared Frobenius-cosine orthogonal regularization with a linearly
increasing weight up to 0.03.

## Continual-learning results

### Evaluation protocol correction

The main stacked-LoRA results in `stacked_lora/results.csv` use the corrected
all-adapters protocol. At checkpoint stage `t`, every task being evaluated uses
all adapters learned through stage `t`, each with fixed weight `g=1`; only the
task-aware classification head changes with the evaluation task. Consequently,
later adapters can affect earlier tasks even though earlier adapter parameters
are frozen.

The older task-prefix evaluation is retained only for reference in
`stacked_lora/results_task_prefix.csv`; it is not used for any result reported
in this summary because it excluded later adapters when scoring earlier tasks.

| Method | Final average accuracy | Average forgetting |
|---|---:|---:|
| Full-weight | 72.10% | 6.88 points |
| Single LoRA | 73.29% | 5.81 points |
| Stacked LoRA | **74.36%** | **4.06 points** |

Average forgetting is computed over the first five tasks (all tasks except the
final FEVER task). For each task, forgetting is its accuracy immediately after
it was learned minus its accuracy after the final training stage; the reported
metric is the mean of those five differences. For corrected stacked LoRA, the
task-level differences are 8.00, 1.90, 2.41, 3.10, and 4.90 percentage points,
whose mean is 4.06 points.

### Final accuracy after FEVER

| Task | Full-weight | Single LoRA | Stacked LoRA |
|---|---:|---:|---:|
| MNLI | 62.4% | 69.1% | **76.4%** |
| QQP | 82.2% | 82.7% | **84.2%** |
| SST-2 | 89.8% | 90.1% | **92.0%** |
| SIQA | 68.8% | 67.4% | **69.3%** |
| WinoGrande | 58.7% | **60.6%** | 55.0% |
| FEVER | **70.7%** | 69.8% | 69.3% |

Full-weight tuning learns individual tasks strongly but changes the shared
encoder and forgets earlier tasks. Single LoRA reduces this interference, but
its shared adapter still changes the representation supplied to every old head.
Stacked LoRA freezes old adapter parameters but evaluates old tasks with later
adapters included. It therefore has nonzero interference, although it still has
the best final average and the lowest average forgetting.

## Retention of pretrained Dolma performance

The final FEVER checkpoint from each method was also evaluated on the same
1,000 held-out Dolma documents. The evaluation skips the 20,000 documents used
to train `Kt672/Dolma_pretain`, uses seed 42, truncates sequences to 1,024
tokens, and scores 379,606 next-token predictions. Classification heads are
ignored; each encoder is evaluated as a causal language model using its tied
token-embedding output projection. Lower negative log-likelihood (NLL) and
perplexity are better.

| Model | Mean NLL | Perplexity | Perplexity change vs. pretrained |
|---|---:|---:|---:|
| Pretrained Dolma | **2.981630** | **19.7199** | — |
| Full-weight after FEVER | 3.605847 | 36.8129 | +86.68% |
| Single LoRA after FEVER | 3.072011 | 21.5853 | +9.46% |
| Stacked LoRA after FEVER | 3.093423 | 22.0524 | +11.83% |

The percentage change is
`(checkpoint perplexity / pretrained perplexity - 1) * 100`; a smaller increase
means better retention. Full-weight continual tuning loses substantially more
of the pretrained language-model capability. Single LoRA retains it best among
the continual-learning methods, with stacked LoRA close behind. The stacked
checkpoint was evaluated with all six accumulated adapters active.

## Intruder analysis

The analysis compares the top 50 left singular vectors of tuned weights with
the top 50 vectors from the Dolma base. It covers layers 0, 8, and 15 and the
four targeted projection types. A tuned vector is an intruder when its maximum
absolute cosine similarity with the selected pretrained vectors is below the
chosen epsilon threshold.

At epsilon 0.5, no intruders were detected for any method.

At epsilon 0.8:

| Method | Aggregate intruder events |
|---|---:|
| Full-weight | 17 |
| Single LoRA | 9 |
| Stacked LoRA | **6** |

### Events by continual-learning stage at epsilon 0.8

| Stage | Full-weight | Single LoRA | Stacked LoRA |
|---|---:|---:|---:|
| MNLI | 2 | 2 | 2 |
| QQP | 2 | 0 | 2 |
| SST-2 | 2 | 6 | 0 |
| SIQA | 2 | 1 | 0 |
| WinoGrande | 2 | 0 | 0 |
| FEVER | 7 | 0 | 2 |

Intruders do not monotonically accumulate. Single and stacked LoRA show
threshold crossings that appear and later disappear. Full-weight tuning remains
at two events through the first five stages and rises to seven after FEVER.

Of the 32 total events, 29 occur in `v_proj`. No events occur in `q_proj` or
`up_proj`. Layer 15 has no intruders for any method; full-weight events are
concentrated in layer 8, while LoRA events are concentrated in layer 0.

## Main conclusion

All-adapters stacked LoRA is the strongest method in this experiment. It combines
the highest final average accuracy, the lowest forgetting, and the fewest
intruder events. Previous adapters remain frozen, but new adapters can still
change earlier-task predictions because all accumulated adapters are used.

The result should be interpreted as task-aware continual learning. At inference,
the system must know which task head to use; the accumulated adapter stack is shared. The
intruder counts are also threshold-dependent diagnostics rather than proof that
spectral intruders cause forgetting.

## Code and reproducibility status

The repository code uses all accumulated adapters by default for stacked-LoRA
evaluation. `run_intruder_experiment.py` applies this behavior during continual
training, and `reevaluate_intruder_experiment.py` applies it when evaluating
saved checkpoints. For each checkpoint, the reevaluator activates the complete
adapter list stored in that checkpoint's metadata before scoring every task.

The stacked checkpoints were reevaluated with:

`./.venv/bin/python reevaluate_intruder_experiment.py --experiment-dir outputs/intruder_experiment_dolma --method stacked_lora --adapter-eval-mode all --output-name results.csv`

The regenerated `results.csv` produces 4.06 points of average forgetting, and
the continual-learning plots were rebuilt from that corrected file.

## Output guide

- `full_finetune/results.csv`: full-weight accuracy history
- `single_lora/results.csv`: shared-LoRA accuracy history
- `stacked_lora/results.csv`: stacked-LoRA accuracy history
- `stacked_lora/results_task_prefix.csv`: preserved legacy prefix-routing evaluation
- `stacked_lora/evaluation_config.json`: corrected all-adapters evaluation settings
- `sampled_data/`: reproducible source indices
- `EXPERIMENT_CONFIG.md`: exact method-specific configuration
- `intruder_analysis/`: epsilon-0.8 analysis and heatmaps
- `intruder_analysis_eps0.5/`: preserved epsilon-0.5 analysis
- `INTRUDER_SCALING_TOP100_REPORT.md`: top-100 epsilon-0.8 lambda intervention report
- `dolma_pretrained_comparison/REPORT.md`: held-out Dolma comparison and interpretation
- `dolma_pretrained_comparison/results.csv`: raw NLL and perplexity results
- `dolma_pretrained_comparison/config.json`: exact held-out evaluation configuration
