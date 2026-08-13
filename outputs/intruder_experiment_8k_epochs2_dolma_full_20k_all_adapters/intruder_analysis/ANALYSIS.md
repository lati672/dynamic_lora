# Top-150 Intruder Analysis (epsilon = 0.8)

## Setup

This analysis compares continual-learning checkpoints against the Dolma-adapted
Llama-3.2-1B backbone. It covers full-weight fine-tuning, one continually
updated shared LoRA, and stacked LoRA with a separate adapter per task and every
available adapter active at weight 1.

The task sequence is MNLI, QQP, SST-2, SIQA, WinoGrande, and FEVER. For layers
0, 8, and 15 and modules `q_proj`, `v_proj`, `up_proj`, and `down_proj`, the
analysis computes left singular vectors of the pretrained and effective tuned
weights. Each of the top 150 tuned vectors is an intruder when its maximum
absolute cosine similarity to the pretrained reference subspace is below 0.8.

Full-weight and single-LoRA measurements come from this archived directory. The
updated stacked-LoRA measurements come from
`../intruder_experiment_8k_epochs2_dolma_full_20k_all_adapters/intruder_analysis`
and replace the earlier learned-gate stacked run.

## Aggregate intruder events

| Method | q_proj | v_proj | up_proj | down_proj | Total |
|---|---:|---:|---:|---:|---:|
| Full-weight | 0 | 100 | 15 | 23 | 138 |
| Single LoRA | 0 | 4 | 0 | 0 | 4 |
| Stacked LoRA | 0 | 4 | 0 | 4 | 8 |
| **All methods** | **0** | **108** | **15** | **27** | **150** |

`v_proj` accounts for 108 of 150 aggregate intruder events (72.0%). Full-weight
tuning contributes most of the events: 138 total, compared with 4 for single
LoRA and 8 for stacked LoRA. No `q_proj` intruders occur in any method.

Counts are aggregate checkpoint events rather than unique singular directions
tracked through time. A direction can be counted at one stage and realign above
the threshold at a later stage.

## Distribution by layer

| Layer | Aggregate intruder events |
|---:|---:|
| 0 | 112 |
| 8 | 38 |
| 15 | 0 |

Spectral misalignment is concentrated in the early and middle layers. None of
the three methods produces an intruder in layer 15 under this configuration.

## Stacked-LoRA trajectory

| Stage | Intruders | Rate across 1,800 vectors | Mean maximum similarity |
|---|---:|---:|---:|
| MNLI | 0 | 0.000% | 0.998601 |
| QQP | 0 | 0.000% | 0.997713 |
| SST-2 | 2 | 0.111% | 0.996853 |
| SIQA | 4 | 0.222% | 0.995594 |
| WinoGrande | 2 | 0.111% | 0.995517 |
| FEVER | 0 | 0.000% | 0.994669 |

The stacked run has eight aggregate intruder events: four in `v_proj` and four
in `down_proj`. The events are temporary. Two appear after SST-2, four after
SIQA, and two after WinoGrande, while the final FEVER checkpoint has none.
Despite zero threshold crossings at FEVER, its mean maximum similarity is the
lowest in the sequence, showing gradual subspace movement that remains mostly
above 0.8.

## Final checkpoint

At FEVER, full tuning has 36 intruders: 21 in layer-0 `v_proj`, 6 in layer-8
`v_proj`, 5 in layer-0 `up_proj`, and 4 in layer-0 `down_proj`. Both LoRA
methods have zero final intruders. This does not mean their effective weights
are unchanged; their mean similarities remain below 1, but no tested vector
falls below the chosen threshold.

## Interpretation limits

These results are based on one seed, one task order, three sampled layers, four
module types, and a fixed threshold. Counts depend on `top_k` and epsilon and
should not be interpreted as a threshold-independent property of the models.
The analysis also excludes `k_proj` and `o_proj` and measures left singular
vectors only. Repeated seeds, alternate task orders, principal-angle metrics,
and update-norm normalization would be needed for stronger conclusions.
