import argparse
import os
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.adapters import (  # noqa: E402
    adapter_rank,
    build_model_from_stacked_state,
    concatenate_adapter_states,
    extract_adapter_state,
    l2_penalty,
    lora_config,
    orthogonal_penalty_first_task_slices,
    parse_target_modules,
    set_active_adapters,
    set_only_adapter_trainable,
)
from dynamic_lora.core.constants import (  # noqa: E402
    DEFAULT_LEARNING_RATE,
    DEFAULT_ORTHOGONAL_PENALTY_WEIGHT,
    DEFAULT_STACKED_ADAPTER_DIR,
    DEFAULT_TASKS,
    DEFAULT_UNLEARN_OUTPUT_DIR,
    STACK_ADAPTER_NAME,
)
from dynamic_lora.core.dpo import DPO, build_classification_dpo_inputs  # noqa: E402
from dynamic_lora.core.data_pipeline import (  # noqa: E402
    build_dataloader,
    build_question,
    label_text,
    load_task_datasets,
    task_spec,
)
from dynamic_lora.core.eval_pipeline import evaluate_task, generate_label_with_raw  # noqa: E402
from dynamic_lora.core.io_utils import release_memory, save_json, save_lora_ab  # noqa: E402
from dynamic_lora.core.retain_regularization import (  # noqa: E402
    collect_retain_batches,
    previous_tasks_for_unlearning,
    retain_projection_penalty,
)
from dynamic_lora.core.seed_utils import set_global_seed  # noqa: E402
from dynamic_lora.core.lora_app.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    MAX_LENGTH,
    MODEL_ID,
    TARGET_MODULES,
    TrainingConfig,
)
from dynamic_lora.core.lora_app.modeling import load_base_model  # noqa: E402


DEFAULT_QUICK_EVAL_SAMPLES = 200
DEFAULT_UNLEARN_TRAIN_SAMPLES = 2000
DEFAULT_RETAIN_SAMPLES_PER_TASK = 100
DEFAULT_EPOCHS = 5
DEFAULT_BETA = 1.0
DEFAULT_OUTPUT_DIR = DEFAULT_UNLEARN_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start from a saved stacked LoRA adapter, run a quick evaluation, "
            "apply task DPO unlearning with a fresh LoRA adapter plus orthogonal loss, "
            "and then evaluate again."
        )
    )
    parser.set_defaults(enable_orthogonal_penalty=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stacked-adapter-dir", default=DEFAULT_STACKED_ADAPTER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unlearn-task", choices=DEFAULT_TASKS, default="dbpedia_14")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(TARGET_MODULES))
    parser.add_argument("--dpo-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--unlearn-train-samples", type=int, default=DEFAULT_UNLEARN_TRAIN_SAMPLES)
    parser.add_argument("--retain-samples-per-task", type=int, default=DEFAULT_RETAIN_SAMPLES_PER_TASK)
    parser.add_argument("--quick-eval-samples", type=int, default=DEFAULT_QUICK_EVAL_SAMPLES)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--enable-orthogonal-penalty",
        action="store_true",
        help="Enable the L1 orthogonality penalty ||A_old @ A_new.T||_1.",
    )
    parser.add_argument(
        "--disable-orthogonal-penalty",
        action="store_false",
        dest="enable_orthogonal_penalty",
        help="Disable the orthogonality penalty.",
    )
    parser.add_argument("--orthogonal-penalty-weight", type=float, default=DEFAULT_ORTHOGONAL_PENALTY_WEIGHT)
    parser.add_argument(
        "--orthogonal-previous-task-count",
        type=int,
        default=None,
        help=(
            "Apply orthogonal loss only to the first N rank slices in the stacked adapter. "
            "Defaults to the unlearn task index in the learned task sequence."
        ),
    )
    parser.add_argument(
        "--orthogonal-task-rank",
        type=int,
        default=8,
        help="Rank of each previous task slice used by --orthogonal-previous-task-count.",
    )
    parser.add_argument("--l2-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--retain-projection-penalty-weight",
        type=float,
        default=1.0,
        help="Weight for ||W_unlearn (I - Pi_unlearn) x_retain|| on previous-task retain examples.",
    )
    parser.add_argument(
        "--retain-projection-ridge",
        type=float,
        default=1e-6,
        help="Ridge added before inverting A^T A in the retain projection penalty.",
    )
    return parser.parse_args()


def unlearn_adapter_name(task_name: str) -> str:
    return f"unlearn_{task_name}_dpo"


def default_output_dir(base_output_dir: Path, task_name: str) -> Path:
    if str(base_output_dir) != DEFAULT_OUTPUT_DIR:
        return base_output_dir
    return base_output_dir / task_name


def default_orthogonal_previous_task_count(task_name: str) -> int:
    return DEFAULT_TASKS.index(task_name)


def load_stacked_model(
    config: TrainingConfig,
    token: str | None,
    stacked_adapter_dir: Path,
):
    tokenizer, base_model = load_base_model(config, token)
    model = PeftModel.from_pretrained(base_model, str(stacked_adapter_dir), adapter_name=STACK_ADAPTER_NAME)
    set_active_adapters(model, STACK_ADAPTER_NAME)
    return tokenizer, model


def quick_evaluate(
    tokenizer,
    model: PeftModel,
    config: TrainingConfig,
    quick_eval_datasets: dict[str, object],
    max_new_tokens: int,
) -> list[dict]:
    results = []
    for task_name in DEFAULT_TASKS:
        results.append(
            evaluate_task(
                tokenizer=tokenizer,
                model=model,
                config=config,
                task_name=task_name,
                eval_dataset=quick_eval_datasets[task_name],
                max_new_tokens=max_new_tokens,
                stack_adapter_name=STACK_ADAPTER_NAME,
                build_question=build_question,
                label_text=label_text,
                task_spec=task_spec,
                set_active_adapters=set_active_adapters,
            )
        )
    return results


def task_eval_result_from_quick_eval(quick_eval_results: list[dict], task_name: str) -> dict:
    for row in quick_eval_results:
        if row["task_name"] == task_name:
            result = dict(row)
            result["split"] = "eval"
            return result
    raise ValueError(f"Missing {task_name} result in quick eval results.")


def evaluate_unlearn_epoch(
    tokenizer,
    model: PeftModel,
    config: TrainingConfig,
    task_name: str,
    eval_dataset,
    max_new_tokens: int,
    train_adapter_name: str,
) -> dict:
    active_adapters = [STACK_ADAPTER_NAME, train_adapter_name]
    eval_result = evaluate_unlearn_split(
        tokenizer=tokenizer,
        model=model,
        config=config,
        task_name=task_name,
        eval_dataset=eval_dataset,
        max_new_tokens=max_new_tokens,
        active_adapters=active_adapters,
        split_name="eval",
    )
    return eval_result


def evaluate_unlearn_split(
    tokenizer,
    model: PeftModel,
    config: TrainingConfig,
    task_name: str,
    eval_dataset,
    max_new_tokens: int,
    active_adapters: str | list[str],
    split_name: str,
) -> dict:
    set_active_adapters(model, active_adapters, inference_mode=True)
    model.eval()
    rows = []
    correct = 0
    text_field = task_spec(task_name)["text_field"]
    for index, example in enumerate(eval_dataset):
        expected = label_text(task_name, int(example["label"]))
        prediction, raw_decoded = generate_label_with_raw(
            tokenizer=tokenizer,
            model=model,
            config=config,
            question_text=build_question(task_name, example[text_field]),
            max_new_tokens=max_new_tokens,
            task_name=task_name,
        )
        is_correct = prediction == expected
        correct += int(is_correct)
        rows.append(
            {
                "index": index,
                "expected": expected,
                "prediction": prediction,
                "raw_decoded": raw_decoded,
                "correct": is_correct,
            }
        )
    accuracy = correct / len(rows) if rows else 0.0
    print(
        f"[split-eval] task={task_name} split={split_name} "
        f"num_examples={len(rows)} mean_accuracy={accuracy:.4f}"
    )
    return {
        "task_name": task_name,
        "split": split_name,
        "num_examples": len(rows),
        "accuracy": accuracy,
        "correct": correct,
        "rows": rows,
    }


def precompute_unlearn_dpo_batches(
    ref_model: PeftModel,
    tokenizer,
    dataloader,
    config: TrainingConfig,
    task_name: str,
) -> tuple[list[dict[str, dict[str, torch.Tensor]]], int]:
    prepared_batches: list[dict[str, dict[str, torch.Tensor]]] = []
    skipped_batches = 0
    candidate_labels = tuple(task_spec(task_name)["labels"])
    for batch in dataloader:
        dpo_inputs = build_classification_dpo_inputs(
            model=ref_model,
            tokenizer=tokenizer,
            batch=batch,
            candidate_labels=candidate_labels,
            max_length=config.max_length,
            assistant_end_tag=config.asst_end_tag if config.apply_chat_template else None,
        )
        if dpo_inputs is None:
            skipped_batches += 1
            continue
        prepared_batches.append(dpo_inputs)
    return prepared_batches, skipped_batches


def train_task_dpo_unlearning(
    model: PeftModel,
    tokenizer,
    prepared_batches: list[dict[str, dict[str, torch.Tensor]]],
    retain_batches: list,
    precompute_skipped_batches: int,
    config: TrainingConfig,
    task_name: str,
    train_adapter_name: str,
    beta: float,
    orthogonal_penalty_enabled: bool,
    orthogonal_penalty_weight: float,
    orthogonal_previous_task_count: int,
    orthogonal_task_rank: int,
    l2_penalty_weight: float,
    retain_projection_penalty_weight: float,
    retain_projection_ridge: float,
    eval_dataset,
    max_new_tokens: int,
    shuffle_seed: int,
    ref_model: PeftModel,
) -> list[dict[str, float | int]]:
    active_adapters = [STACK_ADAPTER_NAME, train_adapter_name]
    set_active_adapters(model, active_adapters)
    set_only_adapter_trainable(model, train_adapter_name)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    device = next(model.parameters()).device
    accumulation_steps = config.gradient_accumulation_steps
    steps_per_epoch = max(1, (len(prepared_batches) + accumulation_steps - 1) // accumulation_steps)
    warmup_steps = max(0, int(config.warmup_epochs * steps_per_epoch))
    optimizer_step_count = 0
    epoch_losses: list[dict[str, float | int]] = []
    dpo_trainer = DPO(model=model, ref_model=ref_model, beta=beta)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.train()

    for epoch in range(config.epochs):
        set_active_adapters(model, active_adapters, inference_mode=False)
        set_only_adapter_trainable(model, train_adapter_name)
        total_loss = 0.0
        total_dpo_loss = 0.0
        total_orthogonal_penalty = 0.0
        total_l2_penalty = 0.0
        total_retain_projection_penalty = 0.0
        processed_batches = 0
        skipped_batches = precompute_skipped_batches
        optimizer.zero_grad()
        shuffled_batch_indices = list(range(len(prepared_batches)))
        random.Random(shuffle_seed + epoch).shuffle(shuffled_batch_indices)
        for step_index, batch_index in enumerate(shuffled_batch_indices, start=1):
            dpo_inputs = prepared_batches[batch_index]
            dpo_inputs = {
                "forget": {
                    split_name: {
                        key: value.to(device)
                        for key, value in split_batch.items()
                    }
                    for split_name, split_batch in dpo_inputs["forget"].items()
                }
            }
            dpo_loss, _ = dpo_trainer.compute_loss(model=model, inputs=dpo_inputs, return_outputs=True)
            use_orthogonal_penalty = orthogonal_penalty_enabled and orthogonal_previous_task_count > 0
            orth_penalty = (
                orthogonal_penalty_first_task_slices(
                    model=model,
                    old_adapter_name=STACK_ADAPTER_NAME,
                    new_adapter_name=train_adapter_name,
                    task_rank=orthogonal_task_rank,
                    task_count=orthogonal_previous_task_count,
                )
                if use_orthogonal_penalty
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            l2_reg = (
                l2_penalty(model, train_adapter_name)
                if l2_penalty_weight > 0
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            retain_reg = (
                retain_projection_penalty(
                    model=model,
                    adapter_name=train_adapter_name,
                    retain_batch=retain_batches[(step_index - 1) % len(retain_batches)],
                    ridge=retain_projection_ridge,
                )
                if retain_projection_penalty_weight > 0 and retain_batches
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            loss = (
                dpo_loss
                + orthogonal_penalty_weight * orth_penalty
                + l2_penalty_weight * l2_reg
                + retain_projection_penalty_weight * retain_reg
            )
            (loss / accumulation_steps).backward()

            if step_index % accumulation_steps == 0 or step_index == len(prepared_batches):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            processed_batches += 1
            total_loss += float(loss.detach().cpu())
            total_dpo_loss += float(dpo_loss.detach().cpu())
            total_orthogonal_penalty += float(orth_penalty.detach().cpu())
            total_l2_penalty += float(l2_reg.detach().cpu())
            total_retain_projection_penalty += float(retain_reg.detach().cpu())

        denominator = processed_batches if processed_batches > 0 else 1
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / denominator,
            "dpo_loss": total_dpo_loss / denominator,
            "orthogonal_penalty": total_orthogonal_penalty / denominator,
            "l2_penalty": total_l2_penalty / denominator,
            "retain_projection_penalty": total_retain_projection_penalty / denominator,
            "processed_batches": processed_batches,
            "skipped_batches": skipped_batches,
            "optimizer_steps": optimizer_step_count,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        eval_result = evaluate_unlearn_epoch(
            tokenizer=tokenizer,
            model=model,
            config=config,
            task_name=task_name,
            eval_dataset=eval_dataset,
            max_new_tokens=max_new_tokens,
            train_adapter_name=train_adapter_name,
        )
        row["eval_accuracy"] = eval_result["accuracy"]
        row["eval_correct"] = eval_result["correct"]
        row["eval_num_examples"] = eval_result["num_examples"]
        row["eval_rows"] = eval_result["rows"]
        epoch_losses.append(row)
        print(
            f"adapter={train_adapter_name} epoch={row['epoch']} "
            f"loss={row['loss']:.4f} dpo_loss={row['dpo_loss']:.4f} "
            f"orthogonal_penalty={row['orthogonal_penalty']:.6f} "
            f"l2_penalty={row['l2_penalty']:.6f} "
            f"retain_projection_penalty={row['retain_projection_penalty']:.6f} "
            f"eval_accuracy={row['eval_accuracy']:.4f} "
            f"processed_batches={row['processed_batches']} skipped_batches={row['skipped_batches']} "
            f"lr={row['learning_rate']:.6g} optimizer_steps={optimizer_step_count}"
        )
        set_active_adapters(model, active_adapters, inference_mode=False)
        set_only_adapter_trainable(model, train_adapter_name)
        model.train()

    return epoch_losses


def main() -> None:
    args = parse_args()
    set_global_seed(args.train_seed)
    output_dir = default_output_dir(Path(args.output_dir), args.unlearn_task)
    if not args.enable_orthogonal_penalty and "no_orthogonal" not in output_dir.name:
        output_dir = output_dir.parent / f"{output_dir.name}_no_orthogonal"
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked_adapter_dir = Path(args.stacked_adapter_dir)
    if not stacked_adapter_dir.exists():
        raise SystemExit(f"Stacked adapter directory not found: {stacked_adapter_dir}")

    target_modules = parse_target_modules(args.target_modules)
    orthogonal_previous_task_count = (
        args.orthogonal_previous_task_count
        if args.orthogonal_previous_task_count is not None
        else default_orthogonal_previous_task_count(args.unlearn_task)
    )
    if orthogonal_previous_task_count < 0:
        raise SystemExit("--orthogonal-previous-task-count must be non-negative.")
    token = os.environ.get("HF_TOKEN")

    config = TrainingConfig(
        model_id=args.model_id,
        output_dir=str(output_dir),
        dataset_id="continual_classification",
        dataset_subset="+".join(DEFAULT_TASKS),
        dataset_split="train",
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    quick_eval_datasets = {}
    for task_name in DEFAULT_TASKS:
        _, eval_dataset = load_task_datasets(
            task_name,
            train_count=1,
            eval_count=args.quick_eval_samples,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
        )
        quick_eval_datasets[task_name] = eval_dataset

    unlearn_train_dataset, unlearn_eval_dataset = load_task_datasets(
        args.unlearn_task,
        train_count=args.unlearn_train_samples,
        eval_count=args.quick_eval_samples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
    )
    retain_task_names = previous_tasks_for_unlearning(DEFAULT_TASKS, args.unlearn_task)
    retain_train_datasets = {}
    for task_name in retain_task_names:
        retain_train_dataset, _ = load_task_datasets(
            task_name,
            train_count=args.retain_samples_per_task,
            eval_count=1,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
        )
        retain_train_datasets[task_name] = retain_train_dataset

    tokenizer, base_model = load_stacked_model(
        config=config,
        token=token,
        stacked_adapter_dir=stacked_adapter_dir,
    )
    pre_eval_results = quick_evaluate(
        tokenizer=tokenizer,
        model=base_model,
        config=config,
        quick_eval_datasets=quick_eval_datasets,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(output_dir / "pre_eval_results.json", pre_eval_results)
    pre_unlearn_eval_results = task_eval_result_from_quick_eval(pre_eval_results, args.unlearn_task)
    save_json(output_dir / "pre_unlearn_task_eval_results.json", pre_unlearn_eval_results)

    set_global_seed(args.train_seed)
    train_adapter_name = unlearn_adapter_name(args.unlearn_task)
    base_model.add_adapter(train_adapter_name, lora_config(args, target_modules))
    unlearn_dataloader = build_dataloader(
        tokenizer,
        config,
        args.unlearn_task,
        unlearn_train_dataset,
        shuffle_seed=args.train_seed,
    )
    retain_dataloaders = [
        build_dataloader(
            tokenizer,
            config,
            task_name,
            retain_train_datasets[task_name],
            shuffle_seed=args.train_seed + index,
        )
        for index, task_name in enumerate(retain_task_names)
    ]
    retain_batches = collect_retain_batches(retain_dataloaders)
    ref_model_for_pairs = DPO(model=base_model, beta=args.dpo_beta).ref_model
    prepared_dpo_batches, precompute_skipped_batches = precompute_unlearn_dpo_batches(
        ref_model=ref_model_for_pairs,
        tokenizer=tokenizer,
        dataloader=unlearn_dataloader,
        config=config,
        task_name=args.unlearn_task,
    )
    if not prepared_dpo_batches:
        raise SystemExit("All DPO batches were skipped during precomputation; no training pairs were generated.")
    epoch_losses = train_task_dpo_unlearning(
        model=base_model,
        tokenizer=tokenizer,
        prepared_batches=prepared_dpo_batches,
        retain_batches=retain_batches,
        precompute_skipped_batches=precompute_skipped_batches,
        config=config,
        task_name=args.unlearn_task,
        train_adapter_name=train_adapter_name,
        beta=args.dpo_beta,
        orthogonal_penalty_enabled=args.enable_orthogonal_penalty,
        orthogonal_penalty_weight=args.orthogonal_penalty_weight,
        orthogonal_previous_task_count=orthogonal_previous_task_count,
        orthogonal_task_rank=args.orthogonal_task_rank,
        l2_penalty_weight=args.l2_penalty_weight,
        retain_projection_penalty_weight=args.retain_projection_penalty_weight,
        retain_projection_ridge=args.retain_projection_ridge,
        eval_dataset=unlearn_eval_dataset,
        max_new_tokens=args.max_new_tokens,
        shuffle_seed=args.train_seed,
        ref_model=ref_model_for_pairs,
    )
    save_json(output_dir / "epoch_losses.json", epoch_losses)
    save_lora_ab(base_model, output_dir)

    stacked_state = extract_adapter_state(base_model, STACK_ADAPTER_NAME)
    unlearn_state = extract_adapter_state(base_model, train_adapter_name)
    merged_state = concatenate_adapter_states(stacked_state, unlearn_state)
    final_rank = adapter_rank(merged_state)

    del base_model
    release_memory()

    tokenizer, final_model = build_model_from_stacked_state(
        config=config,
        token=token,
        args=args,
        target_modules=target_modules,
        stacked_state=merged_state,
    )
    post_eval_results = quick_evaluate(
        tokenizer=tokenizer,
        model=final_model,
        config=config,
        quick_eval_datasets=quick_eval_datasets,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(output_dir / "post_eval_results.json", post_eval_results)
    post_unlearn_train_eval_results = evaluate_unlearn_split(
        tokenizer=tokenizer,
        model=final_model,
        config=config,
        task_name=args.unlearn_task,
        eval_dataset=unlearn_train_dataset,
        max_new_tokens=args.max_new_tokens,
        active_adapters=STACK_ADAPTER_NAME,
        split_name="train",
    )
    save_json(output_dir / "post_unlearn_task_train_eval_results.json", post_unlearn_train_eval_results)
    post_unlearn_eval_results = task_eval_result_from_quick_eval(post_eval_results, args.unlearn_task)
    save_json(output_dir / "post_unlearn_task_eval_results.json", post_unlearn_eval_results)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_lora_ab(final_model, final_dir)

    summary = {
        "experiment": "task_dpo_unlearn_from_stacked_lora",
        "model_id": args.model_id,
        "stacked_adapter_dir": str(stacked_adapter_dir),
        "tasks_evaluated": list(DEFAULT_TASKS),
        "unlearning_task": args.unlearn_task,
        "retain_tasks": list(retain_task_names),
        "quick_eval_samples": args.quick_eval_samples,
        "unlearn_train_samples": args.unlearn_train_samples,
        "retain_samples_per_task": args.retain_samples_per_task,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "dpo_beta": args.dpo_beta,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": list(target_modules),
        "orthogonal_penalty_enabled": args.enable_orthogonal_penalty,
        "orthogonal_penalty_weight": args.orthogonal_penalty_weight,
        "orthogonal_previous_task_count": orthogonal_previous_task_count,
        "orthogonal_task_rank": args.orthogonal_task_rank,
        "l2_penalty_weight": args.l2_penalty_weight,
        "retain_projection_penalty": "||W_unlearn (I - Pi_unlearn) x_retain||",
        "retain_projection_penalty_weight": args.retain_projection_penalty_weight,
        "retain_projection_ridge": args.retain_projection_ridge,
        "initial_stack_rank": final_rank - args.lora_rank,
        "final_stacked_rank": final_rank,
        "unlearn_adapter_name": train_adapter_name,
        "pre_eval_results": pre_eval_results,
        "pre_unlearn_task_eval_results": pre_unlearn_eval_results,
        "post_eval_results": post_eval_results,
        "post_unlearn_task_train_eval_results": post_unlearn_train_eval_results,
        "post_unlearn_task_eval_results": post_unlearn_eval_results,
    }
    save_json(output_dir / "experiment_metadata.json", summary)

    with (output_dir / "eval_summary.txt").open("w", encoding="utf-8") as file:
        file.write("[pre_unlearning]\n")
        for row in pre_eval_results:
            file.write(
                f"task={row['task_name']} num_examples={row['num_examples']} "
                f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
            )
        file.write("[pre_unlearning_task_eval]\n")
        file.write(
            f"task={pre_unlearn_eval_results['task_name']} split={pre_unlearn_eval_results['split']} "
            f"num_examples={pre_unlearn_eval_results['num_examples']} "
            f"accuracy={pre_unlearn_eval_results['accuracy']:.6f} "
            f"correct={pre_unlearn_eval_results['correct']}\n"
        )
        file.write("[post_unlearning]\n")
        for row in post_eval_results:
            file.write(
                f"task={row['task_name']} num_examples={row['num_examples']} "
                f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
            )
        file.write("[post_unlearning_task_train]\n")
        file.write(
            f"task={post_unlearn_train_eval_results['task_name']} split={post_unlearn_train_eval_results['split']} "
            f"num_examples={post_unlearn_train_eval_results['num_examples']} "
            f"accuracy={post_unlearn_train_eval_results['accuracy']:.6f} "
            f"correct={post_unlearn_train_eval_results['correct']}\n"
        )
        file.write("[post_unlearning_task_eval]\n")
        file.write(
            f"task={post_unlearn_eval_results['task_name']} split={post_unlearn_eval_results['split']} "
            f"num_examples={post_unlearn_eval_results['num_examples']} "
            f"accuracy={post_unlearn_eval_results['accuracy']:.6f} "
            f"correct={post_unlearn_eval_results['correct']}\n"
        )

    del final_model
    release_memory()
    print(f"[done] outputs -> {output_dir}")


if __name__ == "__main__":
    main()
