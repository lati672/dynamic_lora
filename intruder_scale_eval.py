"""Scale LoRA-update intruder singular vectors and evaluate the reconstructed models."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.constants import DEFAULT_EVAL_SAMPLES, DEFAULT_TASKS, STACK_ADAPTER_NAME  # noqa: E402
from dynamic_lora.core.data_pipeline import build_question, label_text, task_spec  # noqa: E402
from dynamic_lora.core.eval_pipeline import evaluate_task_sequence  # noqa: E402
from dynamic_lora.core.io_utils import release_memory, save_json  # noqa: E402
from dynamic_lora.core.lora_app.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    MAX_LENGTH,
    MODEL_ID,
    TrainingConfig,
)
from dynamic_lora.core.lora_app.modeling import load_base_model  # noqa: E402
from dynamic_lora.eval_lora import load_eval_datasets, set_active_full_model  # noqa: E402


LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")
DEFAULT_ADAPTER_DIR = Path("artifacts/dynamic_lora/ag_news_yelp_dbpedia/task_2_dbpedia_14")
DEFAULT_OUTPUT_DIR = Path("outputs/intruder_scale_eval/lora_after_dbpedia")


@dataclass
class ScalingPlan:
    layer_index: int
    module_name: str
    singular_values: torch.Tensor
    left_vectors: torch.Tensor
    right_vectors_h: torch.Tensor
    intruder_mask: torch.Tensor
    max_base_cosines: torch.Tensor


def parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one comma-separated value is required")
    return values


def parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numeric scaling values") from exc
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("Scaling values must be non-negative")
    return values


def parse_layer_indices(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative layer indices") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Layer indices must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect intruder singular vectors in a LoRA update, scale their singular values, "
            "reconstruct W0 + scaled DeltaW, and evaluate each reconstructed model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--modules",
        type=parse_csv_strings,
        default=("q_proj", "v_proj"),
    )
    parser.add_argument("--layers", type=parse_layer_indices, default=None, help="Defaults to all model layers.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scales", type=parse_csv_floats, default=(0.5, 0.1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-samples-per-task", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save each reconstructed full model under OUTPUT_DIR/models/scale_SCALE.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def resolve_adapter_path(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    stack_path = path / STACK_ADAPTER_NAME
    if (stack_path / "adapter_config.json").is_file():
        return stack_path
    raise FileNotFoundError(f"Could not find adapter_config.json in {path} or {stack_path}")


def build_eval_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_id=args.model_id,
        output_dir=str(args.output_dir),
        dataset_id="continual_classification",
        dataset_subset="+".join(DEFAULT_TASKS),
        dataset_split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )


def compact_lora_svd(module, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an exact compact SVD of DeltaW = scaling * B @ A."""
    a = module.lora_A[adapter_name].weight.detach().to(device="cpu", dtype=torch.float32)
    b = module.lora_B[adapter_name].weight.detach().to(device="cpu", dtype=torch.float32)
    b = b * float(module.scaling[adapter_name])
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
    small_left, singular_values, small_right_h = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    left_vectors = q_b @ small_left
    right_vectors_h = small_right_h @ q_a.T
    return left_vectors, singular_values, right_vectors_h


def lora_layers(
    model: PeftModel,
    modules: tuple[str, ...],
    layer_indices: tuple[int, ...] | None = None,
) -> dict[tuple[int, str], object]:
    selected = {}
    wanted_layers = set(layer_indices) if layer_indices is not None else None
    for path, module in model.named_modules():
        module_name = path.rsplit(".", 1)[-1]
        match = LAYER_PATTERN.search(path)
        if (
            module_name not in modules
            or match is None
            or not hasattr(module, "lora_A")
            or STACK_ADAPTER_NAME not in module.lora_A
        ):
            continue
        key = (int(match.group(1)), module_name)
        if wanted_layers is not None and key[0] not in wanted_layers:
            continue
        if key in selected:
            raise ValueError(f"Multiple LoRA modules matched layer={key[0]}, module={key[1]}")
        selected[key] = module
    return selected


def discover_layer_indices(
    model: PeftModel,
    modules: tuple[str, ...],
    layer_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    indices = set()
    wanted_layers = set(layer_indices) if layer_indices is not None else None
    for path, module in model.named_modules():
        if path.rsplit(".", 1)[-1] not in modules:
            continue
        match = LAYER_PATTERN.search(path)
        weight = getattr(module, "weight", None)
        if match is not None and isinstance(weight, torch.Tensor) and weight.ndim == 2:
            layer_index = int(match.group(1))
            if wanted_layers is None or layer_index in wanted_layers:
                indices.add(layer_index)
    return tuple(sorted(indices))


def build_scaling_plans(
    model: PeftModel,
    modules: tuple[str, ...],
    top_k: int,
    threshold: float,
    layer_indices: tuple[int, ...] | None = None,
) -> tuple[dict[tuple[int, str], ScalingPlan], list[dict[str, object]]]:
    selected = lora_layers(model, modules, layer_indices)
    plans = {}
    summaries = []

    with torch.no_grad():
        for key in sorted(selected):
            layer_index, module_name = key
            module = selected[key]
            print(f"[svd:start] layer={layer_index} module={module_name}", flush=True)
            left_vectors, singular_values, right_vectors_h = compact_lora_svd(module, STACK_ADAPTER_NAME)
            tolerance = float(singular_values.max()) * 1e-6 if singular_values.numel() else 0.0
            active_count = int((singular_values > tolerance).sum().item())
            analyzed_count = min(top_k, active_count)
            left_vectors = left_vectors[:, :analyzed_count]
            singular_values = singular_values[:analyzed_count]
            right_vectors_h = right_vectors_h[:analyzed_count]

            base_weight = module.base_layer.weight.detach().to(device="cpu", dtype=torch.float32)
            base_left_vectors = torch.linalg.svd(base_weight, full_matrices=False).U
            max_base_cosines = torch.abs(left_vectors.T @ base_left_vectors).amax(dim=1)
            intruder_mask = max_base_cosines < threshold
            plans[key] = ScalingPlan(
                layer_index=layer_index,
                module_name=module_name,
                singular_values=singular_values,
                left_vectors=left_vectors,
                right_vectors_h=right_vectors_h,
                intruder_mask=intruder_mask,
                max_base_cosines=max_base_cosines,
            )
            summaries.append(
                {
                    "layer_index": layer_index,
                    "module_name": module_name,
                    "has_lora_update": True,
                    "active_singular_values": active_count,
                    "analyzed_singular_values": analyzed_count,
                    "threshold": threshold,
                    "num_intruders": int(intruder_mask.sum().item()),
                }
            )
            print(
                f"[svd:done] layer={layer_index} module={module_name} "
                f"intruders={int(intruder_mask.sum().item())}/{analyzed_count}",
                flush=True,
            )
            del base_weight, base_left_vectors

    return plans, sorted(summaries, key=lambda row: (row["layer_index"], row["module_name"]))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_intruder_vectors(path: Path, plans: dict[tuple[int, str], ScalingPlan]) -> None:
    rows = []
    for plan in plans.values():
        for index, (singular_value, max_cosine, is_intruder) in enumerate(
            zip(plan.singular_values, plan.max_base_cosines, plan.intruder_mask),
            start=1,
        ):
            rows.append(
                {
                    "layer_index": plan.layer_index,
                    "module_name": plan.module_name,
                    "vector_index": index,
                    "singular_value": float(singular_value),
                    "max_base_cosine": float(max_cosine),
                    "is_intruder": bool(is_intruder),
                }
            )
    write_csv(
        path,
        rows,
        ["layer_index", "module_name", "vector_index", "singular_value", "max_base_cosine", "is_intruder"],
    )


def merged_weight_modules(model, modules: tuple[str, ...]) -> dict[tuple[int, str], object]:
    selected = {}
    for path, module in model.named_modules():
        module_name = path.rsplit(".", 1)[-1]
        match = LAYER_PATTERN.search(path)
        weight = getattr(module, "weight", None)
        if module_name in modules and match is not None and isinstance(weight, torch.Tensor) and weight.ndim == 2:
            selected[(int(match.group(1)), module_name)] = module
    return selected


def apply_intruder_scale(model, plans: dict[tuple[int, str], ScalingPlan], scale: float) -> None:
    selected = merged_weight_modules(model, tuple(sorted({key[1] for key in plans})))
    with torch.no_grad():
        for key, plan in plans.items():
            module = selected[key]
            if not plan.intruder_mask.any():
                continue
            left = plan.left_vectors[:, plan.intruder_mask]
            singular_values = plan.singular_values[plan.intruder_mask]
            right_h = plan.right_vectors_h[plan.intruder_mask]
            correction = (left * ((scale - 1.0) * singular_values)) @ right_h
            module.weight.add_(correction.to(device=module.weight.device, dtype=module.weight.dtype))


def load_merged_model(config: TrainingConfig, adapter_path: Path, token: str | None):
    tokenizer, base_model = load_base_model(config, token)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        adapter_name=STACK_ADAPTER_NAME,
    )
    try:
        model = peft_model.merge_and_unload(safe_merge=True, adapter_names=[STACK_ADAPTER_NAME])
    except TypeError:
        model = peft_model.merge_and_unload()
    model.eval()
    return tokenizer, model


def scale_name(scale: float) -> str:
    return f"{scale:g}".replace("-", "minus_").replace(".", "_")


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_path(args.adapter_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_eval_config(args)
    token = os.environ.get("HF_TOKEN")

    print(f"[plan:start] adapter={adapter_path}", flush=True)
    tokenizer, base_model = load_base_model(config, token)
    planning_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        adapter_name=STACK_ADAPTER_NAME,
    )
    plans, module_summaries = build_scaling_plans(
        planning_model,
        args.modules,
        args.top_k,
        args.threshold,
        args.layers,
    )
    if not plans:
        raise SystemExit(f"No LoRA updates found for requested modules: {','.join(args.modules)}")
    write_csv(
        args.output_dir / "module_summary.csv",
        module_summaries,
        [
            "layer_index",
            "module_name",
            "has_lora_update",
            "active_singular_values",
            "analyzed_singular_values",
            "threshold",
            "num_intruders",
        ],
    )
    write_intruder_vectors(args.output_dir / "intruder_vectors.csv", plans)
    del planning_model, base_model, tokenizer
    release_memory()
    print(f"[plan:done] updated_matrices={len(plans)}", flush=True)

    eval_datasets = load_eval_datasets(args.eval_samples_per_task, args.eval_seed)
    evaluation_results = []
    accuracy_rows = []
    variants: list[tuple[str, float | None]] = [("pre_eval", None)]
    variants.extend((f"scale_{scale:g}", scale) for scale in args.scales)
    for variant, scale in variants:
        print(f"[eval:start] variant={variant}", flush=True)
        tokenizer, model = load_merged_model(config, adapter_path, token)
        if scale is not None:
            apply_intruder_scale(model, plans, scale)
        eval_results = evaluate_task_sequence(
            tokenizer=tokenizer,
            model=model,
            config=config,
            task_names=DEFAULT_TASKS,
            eval_datasets=eval_datasets,
            max_new_tokens=args.max_new_tokens,
            stack_adapter_name="full_model",
            build_question=build_question,
            label_text=label_text,
            task_spec=task_spec,
            set_active_adapters=set_active_full_model,
        )
        evaluation_result = {
            "variant": variant,
            "scale": scale,
            "task_eval_results": eval_results,
        }
        evaluation_results.append(evaluation_result)
        accuracy_rows.append(
            {
                "variant": variant,
                "scale": scale,
                **{row["task_name"]: row["accuracy"] for row in eval_results},
            }
        )
        save_json(args.output_dir / f"eval_results_{variant.replace('.', '_')}.json", evaluation_result)
        if args.save_models and scale is not None:
            model_dir = args.output_dir / "models" / f"scale_{scale_name(scale)}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        del model, tokenizer
        release_memory()
        print(f"[eval:done] variant={variant}", flush=True)

    write_csv(
        args.output_dir / "accuracy_by_scale.csv",
        accuracy_rows,
        ["variant", "scale", *DEFAULT_TASKS],
    )
    save_json(
        args.output_dir / "results.json",
        {
            "experiment": "lora_intruder_singular_value_scaling",
            "model_id": args.model_id,
            "adapter_dir": str(adapter_path),
            "modules": list(args.modules),
            "layers": list(args.layers) if args.layers is not None else "all",
            "top_k": args.top_k,
            "threshold": args.threshold,
            "scales": list(args.scales),
            "eval_samples_per_task": args.eval_samples_per_task,
            "eval_seed": args.eval_seed,
            "evaluation_results": evaluation_results,
        },
    )
    print(f"[done] outputs={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
