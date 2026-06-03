import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.adapters import set_active_adapters  # noqa: E402
from dynamic_lora.core.constants import (
    DEFAULT_CONTINUAL_FULL_OUTPUT_DIR,
    DEFAULT_EVAL_OUTPUT_ROOT,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_LORA_EVAL_SAMPLES,
    DEFAULT_STACKED_ADAPTER_DIR,
    DEFAULT_TASKS,
    STACK_ADAPTER_NAME,
)  # noqa: E402
from dynamic_lora.core.data_pipeline import build_question, label_text, load_or_create_subset, task_spec  # noqa: E402
from dynamic_lora.core.eval_pipeline import evaluate_task_sequence  # noqa: E402
from dynamic_lora.core.io_utils import save_json  # noqa: E402
from dynamic_lora.core.lora_app.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    MAX_LENGTH,
    MODEL_ID,
    TrainingConfig,
)  # noqa: E402
from dynamic_lora.core.lora_app.modeling import load_base_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved stacked LoRA adapter or full-finetuned model on the continual tasks."
    )
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-dir", default=DEFAULT_STACKED_ADAPTER_DIR)
    parser.add_argument("--model-dir", default=f"{DEFAULT_CONTINUAL_FULL_OUTPUT_DIR}/final")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-samples-per-task", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def default_lora_output_dir(adapter_dir: Path) -> Path:
    if adapter_dir.name == STACK_ADAPTER_NAME and adapter_dir.parent.name == "final":
        return Path(DEFAULT_EVAL_OUTPUT_ROOT) / adapter_dir.parent.parent.name / "eval"
    return Path(DEFAULT_EVAL_OUTPUT_ROOT) / adapter_dir.name / "eval"


def default_full_output_dir(model_dir: Path) -> Path:
    if model_dir.name == "final":
        return Path(DEFAULT_EVAL_OUTPUT_ROOT) / model_dir.parent.name / "eval"
    return Path(DEFAULT_EVAL_OUTPUT_ROOT) / model_dir.name / "eval"


def load_eval_datasets(eval_count: int, eval_seed: int) -> dict[str, object]:
    eval_datasets = {}
    for task_name in DEFAULT_TASKS:
        spec = task_spec(task_name)
        eval_datasets[task_name] = load_or_create_subset(
            task_name=task_name,
            split_name=spec["eval_split"],
            sample_count=eval_count,
            seed=eval_seed,
        )
        print(
            f"[data] task={task_name} eval_rows={len(eval_datasets[task_name])} eval_seed={eval_seed}",
            flush=True,
        )
    return eval_datasets


def write_eval_results_txt(path: Path, eval_results: list[dict], header: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"[{header}]\n")
        for row in eval_results:
            file.write(
                f"task={row['task_name']} num_examples={row['num_examples']} "
                f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
            )


def set_active_full_model(*args, **kwargs) -> None:
    return None


def load_full_model(model_dir: Path, token: str | None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        token=token,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    return tokenizer, model


def main() -> None:
    args = parse_args()
    eval_samples_per_task = args.eval_samples_per_task
    if eval_samples_per_task is None:
        eval_samples_per_task = DEFAULT_LORA_EVAL_SAMPLES if args.mode == "lora" else DEFAULT_EVAL_SAMPLES

    if args.mode == "lora":
        source_dir = Path(args.adapter_dir)
        if not source_dir.exists():
            raise SystemExit(f"Adapter directory not found: {source_dir}")
        output_dir = Path(args.output_dir) if args.output_dir is not None else default_lora_output_dir(source_dir)
        model_id = args.model_id
    else:
        source_dir = Path(args.model_dir)
        if not source_dir.exists():
            raise SystemExit(f"Model directory not found: {source_dir}")
        output_dir = Path(args.output_dir) if args.output_dir is not None else default_full_output_dir(source_dir)
        model_id = str(source_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(
        model_id=model_id,
        output_dir=str(output_dir),
        dataset_id="continual_classification",
        dataset_subset="+".join(DEFAULT_TASKS),
        dataset_split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    token = os.environ.get("HF_TOKEN")
    if args.mode == "lora":
        print(f"[model:start] mode=lora model={args.model_id} adapter={source_dir}", flush=True)
        tokenizer, base_model = load_base_model(config, token)
        model = PeftModel.from_pretrained(base_model, str(source_dir), adapter_name=STACK_ADAPTER_NAME)
        stack_adapter_name = STACK_ADAPTER_NAME
        set_active = set_active_adapters
        print("[model:done] loaded base model and adapter", flush=True)
    else:
        print(f"[model:start] mode=full model_dir={source_dir}", flush=True)
        tokenizer, model = load_full_model(source_dir, token)
        stack_adapter_name = "full_model"
        set_active = set_active_full_model
        print("[model:done] loaded full-finetuned model", flush=True)

    eval_datasets = load_eval_datasets(eval_samples_per_task, args.eval_seed)
    print(f"[eval:start] tasks={','.join(DEFAULT_TASKS)}", flush=True)
    eval_results = evaluate_task_sequence(
        tokenizer=tokenizer,
        model=model,
        config=config,
        task_names=DEFAULT_TASKS,
        eval_datasets=eval_datasets,
        max_new_tokens=args.max_new_tokens,
        stack_adapter_name=stack_adapter_name,
        build_question=build_question,
        label_text=label_text,
        task_spec=task_spec,
        set_active_adapters=set_active,
    )

    summary = {
        "experiment": f"{args.mode}_afterwards_eval",
        "mode": args.mode,
        "model_id": model_id,
        "source_dir": str(source_dir),
        "tasks": list(DEFAULT_TASKS),
        "eval_samples_per_task": eval_samples_per_task,
        "eval_seed": args.eval_seed,
        "max_new_tokens": args.max_new_tokens,
        "task_eval_results": eval_results,
    }
    save_json(output_dir / "eval_results.json", summary)
    write_eval_results_txt(output_dir / "eval_results.txt", eval_results, f"{args.mode}_afterwards_eval")
    print(f"[done] outputs -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
