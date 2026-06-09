import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.hf_model_loader import (  # noqa: E402
    DEFAULT_HF_FULL_SUBFOLDER,
    DEFAULT_HF_LORA_SUBFOLDER,
    DEFAULT_HF_REPO_ID,
    load_hf_continual_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a LoRA or full-finetuned model from the artifact HF repo.")
    parser.add_argument("--mode", choices=("lora", "full"), required=True)
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument(
        "--subfolder",
        default=None,
        help=(
            "Checkpoint folder in the repo. Defaults to the final checkpoint for the selected mode. "
            "For LoRA, pass the parent folder containing stack/."
        ),
    )
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights into the base model.")
    parser.add_argument("--prompt", default=None, help="Optional prompt used to verify the loaded model.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_subfolder = DEFAULT_HF_LORA_SUBFOLDER if args.mode == "lora" else DEFAULT_HF_FULL_SUBFOLDER
    subfolder = args.subfolder or default_subfolder
    print(f"[model:start] mode={args.mode} repo={args.repo_id} subfolder={subfolder}", flush=True)
    tokenizer, model = load_hf_continual_model(
        mode=args.mode,
        repo_id=args.repo_id,
        subfolder=subfolder,
        token=os.environ.get("HF_TOKEN"),
        merge_lora=args.merge_lora,
    )
    print(f"[model:done] class={type(model).__name__} device={next(model.parameters()).device}", flush=True)

    if args.prompt is None:
        return

    inputs = tokenizer(args.prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
