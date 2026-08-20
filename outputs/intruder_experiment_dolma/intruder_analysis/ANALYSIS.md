# Intruder analysis

## Scope

This analysis compares the continual-learning checkpoints in
`outputs/intruder_experiment_dolma` against the Dolma-pretrained
`Kt672/Dolma_pretain` encoder.

The comparison covers full-weight tuning, one shared continually updated LoRA,
and all-adapters stacked LoRA. Checkpoints were evaluated after MNLI, QQP, SST-2,
SIQA, WinoGrande, and FEVER.

## Method

For layers 0, 8, and 15, the analysis extracts the weight matrices for
`q_proj`, `v_proj`, `up_proj`, and `down_proj`. It computes float32 SVDs
and compares each of the top 50 tuned left singular vectors with the top 50
vectors from the pretrained base.

A tuned vector is counted as an intruder when its largest absolute cosine
similarity with every selected pretrained vector is below 0.8. Counts below are
aggregate events across stages, layers, and matrices; they are not necessarily
distinct semantic features or unique vectors.

## Aggregate results

| Method | Intruder events | Share of 3,600 tested vector positions | Final average accuracy | Average forgetting |
|---|---:|---:|---:|---:|
| Full-weight | 17 | 0.47% | 72.10% | 6.88 points |
| Single LoRA | 9 | 0.25% | 73.29% | 5.81 points |
| Stacked LoRA | 6 | 0.17% | 74.36% | 4.06 points |

Stacked LoRA has the fewest intruder events, the highest final average accuracy,
and the lowest measured forgetting. Full-weight tuning has the most intruders and the
most forgetting. This is an association in this experiment, not evidence that
intruders directly cause forgetting.

## Evolution across tasks

| Stage | Full-weight | Single LoRA | Stacked LoRA |
|---|---:|---:|---:|
| MNLI | 2 | 2 | 2 |
| QQP | 2 | 0 | 2 |
| SST-2 | 2 | 6 | 0 |
| SIQA | 2 | 1 | 0 |
| WinoGrande | 2 | 0 | 0 |
| FEVER | 7 | 0 | 2 |

Full-weight tuning accumulates its largest departure at the final FEVER stage.
Single LoRA shows a concentrated spike after SST-2 rather than a monotonic
increase. Stacked LoRA has no detected events during SST-2, SIQA, or
WinoGrande, with two events returning after FEVER.

## Location of intruders

| Method | q_proj | v_proj | up_proj | down_proj |
|---|---:|---:|---:|---:|
| Full-weight | 0 | 16 | 0 | 1 |
| Single LoRA | 0 | 7 | 0 | 2 |
| Stacked LoRA | 0 | 6 | 0 | 0 |

The `v_proj` matrices account for 29 of the 32 aggregate events. No
`q_proj` or `up_proj` intruders occur at this threshold.

| Method | Layer 0 | Layer 8 | Layer 15 |
|---|---:|---:|---:|
| Full-weight | 5 | 12 | 0 |
| Single LoRA | 7 | 2 | 0 |
| Stacked LoRA | 6 | 0 | 0 |

Layer 15 has no intruders for any method. Full-weight events are concentrated in
layer 8, whereas both LoRA methods are concentrated in layer 0.

## Similarity trend

Mean maximum similarity remains high because most of the top-50 vectors remain
closely aligned with the base even when a small tail crosses the 0.8 threshold.
At the final FEVER stage, mean similarities are 0.9909 for full-weight, 0.9955
for single LoRA, and 0.9948 for stacked LoRA. Full-weight tuning therefore shows
the largest average spectral drift at the final stage.

## Interpretation

The results support three observations:

1. Full-parameter continual tuning changes the pretrained singular subspaces
   more broadly than either LoRA method.
2. All-adapters stacked LoRA combines the lowest intruder count with the lowest
   accuracy forgetting. Old adapters are frozen, but later adapters still affect
   earlier tasks because the accumulated stack is shared.
3. The lower intruder count does not explain every accuracy difference.
   Single LoRA has fewer intruders than full tuning but still forgets because
   its one shared adapter changes the representation used by every old head.

## Limitations

The conclusion depends on the selected layers, modules, top-50 truncation, and
0.8 threshold. An intruder is defined relative only to the selected pretrained
top-50 subspace. Counts should therefore be treated as a thresholded diagnostic,
not a universal property of a model or proof of a causal forgetting mechanism.

The earlier 0.5 analysis is preserved in
`../intruder_analysis_eps0.5`; it detected no intruders for any method.
Detailed 0.8 counts are in `intruder_counts_fixed_gates.csv`, and per-vector
similarities are in `vector_similarities_fixed_gates.csv`.
