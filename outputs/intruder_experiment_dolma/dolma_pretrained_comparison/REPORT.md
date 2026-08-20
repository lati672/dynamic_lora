# Dolma pretrained-model comparison

This evaluation measures held-out Dolma next-token negative log-likelihood
and perplexity. Lower values are better. The three continual models are
their final checkpoints after FEVER; stacked LoRA activates all six
accumulated adapters.

| Model | Mean NLL | Perplexity | PPL change vs. pretrained |
|---|---:|---:|---:|
| Pretrained Dolma | 2.981630 | 19.7199 | +0.00% |
| Full-weight after FEVER | 3.605847 | 36.8129 | +86.68% |
| Single LoRA after FEVER | 3.072011 | 21.5853 | +9.46% |
| Stacked LoRA after FEVER | 3.093423 | 22.0524 | +11.83% |

The evaluation uses the same held-out token cache for every model.
Documents: 1000; maximum length: 1024; seed: 42.

Classification heads are ignored. Each encoder is evaluated as a
causal language model with the model's tied token-embedding output
projection, matching the pretrained Llama configuration.
