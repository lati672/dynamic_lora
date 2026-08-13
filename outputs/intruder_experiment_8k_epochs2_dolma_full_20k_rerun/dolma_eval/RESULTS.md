# Held-Out Dolma Evaluation

All models were evaluated on the same 1,000 held-out Dolma documents (379,606
predicted tokens). The held-out set was constructed by reproducing the original
seed-42 shuffled stream, skipping the first 20,000 usable documents used for
Dolma fine-tuning, and selecting the next 1,000 documents.

Lower perplexity is better.

## Perplexity

The Dolma-finetuned model before continual learning has a perplexity of
**19.7200** (mean NLL: **2.981633**).

| Model | After MNLI | After QQP | After SST-2 | After SIQA | After WinoGrande | After FEVER |
|---|---:|---:|---:|---:|---:|---:|
| Full-weight | 21.0589 | 22.2581 | 22.7155 | 26.0315 | 36.6199 | 37.1686 |
| Single LoRA | 19.8462 | 19.8768 | 19.8794 | 20.0613 | 20.0719 | 20.2241 |
| Stacked LoRA | 19.8454 | 19.9418 | 20.0047 | 20.1642 | 20.2077 | 20.4041 |

## Summary

Full-weight continual learning shows substantial forgetting on held-out Dolma,
with perplexity increasing from the 19.7200 baseline to 37.1686 after FEVER.
Both LoRA methods preserve language-model performance much more effectively.
After FEVER, single LoRA reaches 20.2241 perplexity and stacked LoRA reaches
20.4041, with single LoRA slightly better on this metric in the later stages.

The machine-readable results are available in `results.csv` and `results.json`.
