"""Compare singular-vector subspaces of base, full-FT, and LoRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class CheckpointSpec:
    model_type: str
    name: str
    path: Path
    stage: int


@dataclass
class Spectrum:
    module_path: str
    left_vectors: torch.Tensor
    singular_values: torch.Tensor


def parse_name_path(value: str) -> tuple[str, Path]:
    """Parse a CLI checkpoint value in NAME=PATH format."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, received {value!r}. Example: after_ag_news=/path/to/checkpoint"
        )
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"Both NAME and PATH are required in {value!r}")
    return name.strip(), Path(path).expanduser()


def parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one comma-separated value is required")
    return values


def parse_layer_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layer indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("Layer indices must contain non-negative integers")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect intruder singular-vector dimensions in continual full-FT and LoRA checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 -m dynamic_lora.spectral_analysis \\\n"
            "    --model-id meta-llama/Llama-3.2-1B-Instruct \\\n"
            "    --lora-checkpoint after_ag_news=artifacts/run/task_0_ag_news \\\n"
            "    --full-checkpoint after_ag_news=artifacts/full/task_0_ag_news_full"
        ),
    )
    parser.add_argument("--model-id", required=True, help="Pretrained base model ID or local path.")
    parser.add_argument(
        "--full-checkpoint",
        action="append",
        default=[],
        type=parse_name_path,
        metavar="NAME=PATH",
        help="Full-model checkpoint. Repeat in continual-learning stage order.",
    )
    parser.add_argument(
        "--lora-checkpoint",
        action="append",
        default=[],
        type=parse_name_path,
        metavar="NAME=PATH",
        help="PEFT adapter checkpoint. Repeat in continual-learning stage order.",
    )
    parser.add_argument("--modules", type=parse_csv_strings, default=("q_proj", "v_proj", "up_proj", "down_proj"))
    parser.add_argument("--layers", type=parse_layer_indices, default=(0,), help="Comma-separated layer indices.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=50,
        help="Number of leading base and tuned singular vectors shown in each heatmap.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/spectral_analysis"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--load-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model loading dtype. SVD is always computed in float32 on CPU.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if not args.full_checkpoint and not args.lora_checkpoint:
        parser.error("Provide at least one --full-checkpoint or --lora-checkpoint")
    if args.top_k <= 0 or args.heatmap_size <= 0:
        parser.error("--top-k and --heatmap-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def model_load_kwargs(args: argparse.Namespace, device: torch.device) -> dict:
    return {
        "torch_dtype": resolve_dtype(args.load_dtype, device),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "token": os.environ.get("HF_TOKEN"),
    }


def load_model(
    model_type: str,
    model_id: str,
    checkpoint_path: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """Load a base/full model or load and merge a PEFT adapter into the base model."""
    kwargs = model_load_kwargs(args, device)
    if model_type in {"base", "full"}:
        source = model_id if model_type == "base" else str(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    else:
        adapter_path = resolve_adapter_path(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        try:
            model = model.merge_and_unload(safe_merge=True)
        except TypeError:
            model = model.merge_and_unload()
    model.eval()
    return model.to(device)


def resolve_adapter_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("A LoRA checkpoint path is required")
    if (path / "adapter_config.json").is_file():
        return path
    stack_path = path / "stack"
    if (stack_path / "adapter_config.json").is_file():
        return stack_path
    candidates = sorted(path.glob("*/adapter_config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    raise FileNotFoundError(
        f"Could not find adapter_config.json in {path}, {stack_path}, or a unique direct child"
    )


def selected_weight_modules(
    model: nn.Module,
    layer_indices: Iterable[int],
    module_names: Iterable[str],
) -> dict[tuple[int, str], tuple[str, nn.Module]]:
    wanted_layers = set(layer_indices)
    wanted_modules = set(module_names)
    selected: dict[tuple[int, str], tuple[str, nn.Module]] = {}
    for path, module in model.named_modules():
        short_name = path.rsplit(".", 1)[-1]
        if short_name not in wanted_modules:
            continue
        match = LAYER_PATTERN.search(path)
        if match is None:
            continue
        layer_index = int(match.group(1))
        weight = getattr(module, "weight", None)
        if layer_index in wanted_layers and isinstance(weight, torch.Tensor) and weight.ndim == 2:
            key = (layer_index, short_name)
            if key in selected:
                raise ValueError(f"Multiple modules matched layer={layer_index}, module={short_name}")
            selected[key] = (path, module)
    return selected


def compute_spectrum(module_path: str, module: nn.Module) -> Spectrum:
    # CPU float32 SVD is broadly supported and avoids retaining large GPU workspaces.
    weight = module.weight.detach().to(device="cpu", dtype=torch.float32)
    left_vectors, singular_values, _ = torch.linalg.svd(weight, full_matrices=False)
    return Spectrum(module_path=module_path, left_vectors=left_vectors, singular_values=singular_values)


def build_base_spectra(
    model: nn.Module,
    layers: tuple[int, ...],
    modules: tuple[str, ...],
) -> dict[tuple[int, str], Spectrum]:
    selected = selected_weight_modules(model, layers, modules)
    expected = {(layer, module) for layer in layers for module in modules}
    missing = sorted(expected - set(selected))
    if missing:
        raise ValueError(f"Base model is missing requested matrices: {missing}")
    spectra = {}
    for key in sorted(selected):
        path, module = selected[key]
        print(f"[svd:base] layer={key[0]} module={key[1]} path={path}", flush=True)
        spectra[key] = compute_spectrum(path, module)
    return spectra


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save_heatmap(
    similarity: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(similarity.numpy(), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xlabel("Singular vectors in W_tuned")
    axis.set_ylabel("Singular vectors in W_base")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, label="Absolute cosine similarity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_checkpoint(
    spec: CheckpointSpec,
    model: nn.Module,
    base_spectra: dict[tuple[int, str], Spectrum],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    selected = selected_weight_modules(model, args.layers, args.modules)
    missing = sorted(set(base_spectra) - set(selected))
    if missing:
        raise ValueError(f"Checkpoint {spec.name!r} is missing requested matrices: {missing}")

    rows = []
    heatmap_dir = args.output_dir / "heatmaps" / spec.model_type / safe_filename(spec.name)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for (layer_index, module_name), base in sorted(base_spectra.items()):
        tuned_path, tuned_module = selected[(layer_index, module_name)]
        print(
            f"[svd:tuned] type={spec.model_type} checkpoint={spec.name} "
            f"layer={layer_index} module={module_name} path={tuned_path}",
            flush=True,
        )
        tuned = compute_spectrum(tuned_path, tuned_module)
        if base.left_vectors.shape[0] != tuned.left_vectors.shape[0]:
            raise ValueError(
                f"Left singular-vector dimensions differ for {spec.name}/{tuned_path}: "
                f"base={base.left_vectors.shape}, tuned={tuned.left_vectors.shape}"
            )

        heatmap_base_count = min(args.heatmap_size, base.left_vectors.shape[1])
        heatmap_tuned_count = min(args.heatmap_size, tuned.left_vectors.shape[1])
        similarity = torch.abs(
            base.left_vectors[:, :heatmap_base_count].T
            @ tuned.left_vectors[:, :heatmap_tuned_count]
        )
        title = (
            f"{spec.model_type} | {spec.name} | layer {layer_index} | {module_name}\n"
            f"{base.module_path} vs {tuned.module_path}"
        )
        save_heatmap(
            similarity,
            heatmap_dir / f"layer_{layer_index}_{safe_filename(module_name)}.png",
            title,
        )

        effective_top_k = min(args.top_k, tuned.left_vectors.shape[1])
        max_similarities = torch.abs(
            tuned.left_vectors[:, :effective_top_k].T @ base.left_vectors
        ).amax(dim=1)
        num_intruders = int((max_similarities < args.threshold).sum().item())
        rows.append(
            {
                "model_type": spec.model_type,
                "checkpoint_name": spec.name,
                "layer_name": f"layer_{layer_index}",
                "module_name": module_name,
                "top_k": effective_top_k,
                "threshold": args.threshold,
                "num_intruders": num_intruders,
            }
        )
        print(
            f"[intruders] type={spec.model_type} checkpoint={spec.name} layer={layer_index} "
            f"module={module_name} count={num_intruders}/{effective_top_k}",
            flush=True,
        )
        del tuned, similarity, max_similarities
        gc.collect()
    return rows


def save_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "model_type",
        "checkpoint_name",
        "layer_name",
        "module_name",
        "top_k",
        "threshold",
        "num_intruders",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(
    rows: list[dict[str, object]],
    specs: list[CheckpointSpec],
    output_path: Path,
) -> None:
    # Aggregate across selected matrices so each model type has one line over its CL stages.
    totals = {}
    for row in rows:
        key = (row["model_type"], row["checkpoint_name"])
        totals[key] = totals.get(key, 0) + int(row["num_intruders"])

    figure, axis = plt.subplots(figsize=(9, 5))
    for model_type in ("full", "lora"):
        type_specs = [spec for spec in specs if spec.model_type == model_type]
        if not type_specs:
            continue
        x = [spec.stage for spec in type_specs]
        y = [totals[(model_type, spec.name)] for spec in type_specs]
        axis.plot(x, y, marker="o", linewidth=2, label=model_type)
        for stage, count, spec in zip(x, y, type_specs):
            axis.annotate(
                spec.name,
                (stage, count),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xlabel("Continual learning stage")
    axis.set_ylabel("Number of intruders (sum across selected matrices)")
    max_stages = max(
        (len([spec for spec in specs if spec.model_type == kind]) for kind in ("full", "lora")),
        default=1,
    )
    axis.set_xticks(range(1, max_stages + 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    specs = [
        CheckpointSpec("full", name, path, stage)
        for stage, (name, path) in enumerate(args.full_checkpoint, start=1)
    ] + [
        CheckpointSpec("lora", name, path, stage)
        for stage, (name, path) in enumerate(args.lora_checkpoint, start=1)
    ]
    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {spec.path}")

    print(f"[load:base] model={args.model_id} device={device}", flush=True)
    with torch.no_grad():
        base_model = load_model("base", args.model_id, None, args, device)
        base_spectra = build_base_spectra(base_model, args.layers, args.modules)
        del base_model
        release_memory()

        rows = []
        for spec in specs:
            print(f"[load:{spec.model_type}] checkpoint={spec.name} path={spec.path}", flush=True)
            tuned_model = load_model(spec.model_type, args.model_id, spec.path, args, device)
            rows.extend(analyze_checkpoint(spec, tuned_model, base_spectra, args))
            del tuned_model
            release_memory()

    save_results_csv(rows, args.output_dir / "intruder_counts.csv")
    save_summary_plot(rows, specs, args.output_dir / "intruder_summary.png")
    print(f"[done] outputs={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
