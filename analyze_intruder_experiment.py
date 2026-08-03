#!/usr/bin/env python3
"""SVD analysis for checkpoints produced by the intruder experiment."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel

REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from dynamic_lora.intruder_experiment.modeling import (
    AdditiveLoRALinear,
    ContinualClassifier,
    matches_target,
)

LAYER_RE = re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)")


def arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model", "--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoints_dir", "--checkpoints-dir", type=Path,
                        default=Path("outputs/intruder_experiment"))
    parser.add_argument("--methods", nargs="+", choices=("full", "stacked_lora"), default=["full", "stacked_lora"])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 8, 15])
    parser.add_argument("--modules", nargs="+",
                        default=["q_proj", "v_proj", "up_proj", "down_proj"])
    parser.add_argument("--top_k", "--top-k", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--adapter_eval_mode", "--adapter-eval-mode", choices=("all", "task_specific"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def matrices(model, layers, suffixes, active=None):
    found = {}
    for path, module in model.named_modules():
        match = LAYER_RE.search(path)
        if match and int(match.group(1)) in layers and any(matches_target(path, suffix) for suffix in suffixes):
            if isinstance(module, AdditiveLoRALinear):
                weight = module.effective_weight(active)
            elif isinstance(module, torch.nn.Linear):
                weight = module.weight
            else:
                continue
            suffix = next(name for name in suffixes if matches_target(path, name))
            found[(int(match.group(1)), suffix)] = weight.detach().cpu().float()
    return found


def heatmap(similarity, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(similarity.numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set(title=title, xlabel="tuned singular vector", ylabel="pretrained singular vector")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = arguments()
    output = args.checkpoints_dir / "intruder_analysis"
    csv_path = output / "intruder_counts.csv"
    mode_path = output / f"intruder_counts_{args.adapter_eval_mode}.csv"
    if mode_path.exists() and not args.overwrite:
        print(f"[skip] {mode_path} exists; pass --overwrite to recompute")
        return
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        base = AutoModel.from_pretrained(args.base_model)
        base_matrices = matrices(base, args.layers, args.modules)
        base_u = {key: torch.linalg.svd(weight, full_matrices=False).U for key, weight in base_matrices.items()}
        del base
        rows = []
        roots = {"full": args.checkpoints_dir / "full_finetune",
                 "stacked_lora": args.checkpoints_dir / "stacked_lora"}
        for method in args.methods:
            for checkpoint in sorted(roots[method].glob(f"{method if method != 'full' else 'full'}_after_*")):
                model, metadata = ContinualClassifier.load_checkpoint(checkpoint)
                active = None
                if method == "stacked_lora":
                    active = metadata["adapters"] if args.adapter_eval_mode == "all" else [metadata["stage"]]
                tuned = matrices(model.encoder, args.layers, args.modules, active)
                missing = set(base_u) - set(tuned)
                if missing:
                    raise ValueError(f"{checkpoint} is missing matrices: {sorted(missing)}")
                for (layer, module), weight in tuned.items():
                    tuned_u = torch.linalg.svd(weight, full_matrices=False).U
                    k = min(args.top_k, tuned_u.shape[1])
                    similarity = (base_u[(layer, module)].T @ tuned_u[:, :k]).abs()
                    maxima = similarity.max(dim=0).values
                    rows.append({
                        "method": method, "stage": metadata["stage"],
                        "adapter_eval_mode": args.adapter_eval_mode if method == "stacked_lora" else "n/a",
                        "layer": layer, "module": module, "top_k": k, "epsilon": args.epsilon,
                        "num_intruders": int((maxima < args.epsilon).sum()), "mean_max_similarity": maxima.mean().item(),
                    })
                    heatmap(similarity, output / "heatmaps" / method / metadata["stage"] /
                            f"layer_{layer}_{module.replace('.', '_')}_{args.adapter_eval_mode}.png",
                            f"{method} after {metadata['stage']}: layer {layer} {module}")
                del model
    fields = ["method", "stage", "adapter_eval_mode", "layer", "module", "top_k", "epsilon",
              "num_intruders", "mean_max_similarity"]
    with mode_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in sorted(output.glob("intruder_counts_*.csv")):
            with path.open(newline="") as source:
                writer.writerows(csv.DictReader(source))
    print(f"[done] wrote {mode_path} and combined {csv_path}")


if __name__ == "__main__":
    main()
