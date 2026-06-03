import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.adapters import (  # noqa: E402
    adapter_rank,
    build_model_for_task,
    build_model_from_stacked_state,
    concatenate_adapter_states,
    extract_adapter_state,
    parse_target_modules,
    set_active_adapters,
)
from dynamic_lora.core.constants import (  # noqa: E402
    DEFAULT_CONTINUAL_OUTPUT_DIR,
    DEFAULT_L2_PENALTY_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_CONTINUAL_EPOCHS,
    DEFAULT_LORA_EVAL_SAMPLES,
    DEFAULT_LORA_TRAIN_SAMPLES,
    DEFAULT_ORTHOGONAL_PENALTY_WEIGHT,
    DEFAULT_TASKS,
    STACK_ADAPTER_NAME,
)
from dynamic_lora.core.continual_training import train_one_task  # noqa: E402
from dynamic_lora.core.data_pipeline import (  # noqa: E402
    build_dataloader,
    build_question,
    label_text,
    load_task_datasets,
    task_spec,
)
from dynamic_lora.core.eval_export import (  # noqa: E402
    build_task_eval_summary,
    eval_output_dir_for_checkpoint,
    eval_output_dir_for_run,
    write_learned_task_eval_results_txt,
    write_task_eval_results_txt,
)
from dynamic_lora.core.eval_pipeline import evaluate_task_sequence  # noqa: E402
from dynamic_lora.core.io_utils import release_memory, save_json, save_lora_ab  # noqa: E402
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
from dynamic_lora.core.seed_utils import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a continual-learning classification experiment with stacked LoRA "
            "on AG News, Yelp Review Full, and DBPedia 14."
        )
    )
    parser.set_defaults(enable_orthogonal_penalty=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_CONTINUAL_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_LORA_CONTINUAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
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
    parser.add_argument("--l2-penalty-weight", type=float, default=DEFAULT_L2_PENALTY_WEIGHT)
    parser.add_argument("--target-modules", default=",".join(TARGET_MODULES))
    parser.add_argument("--train-samples-per-task", type=int, default=DEFAULT_LORA_TRAIN_SAMPLES)
    parser.add_argument("--eval-samples-per-task", type=int, default=DEFAULT_LORA_EVAL_SAMPLES)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
        help="Print training progress every N batches. Set to 0 to disable batch logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.train_seed)
    target_modules = parse_target_modules(args.target_modules)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = eval_output_dir_for_run(output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    print(
        f"[run:start] model={args.model_id} tasks={','.join(DEFAULT_TASKS)} "
        f"train_samples_per_task={args.train_samples_per_task} "
        f"eval_samples_per_task={args.eval_samples_per_task} epochs={args.epochs} "
        f"output_dir={output_dir}",
        flush=True,
    )

    train_datasets = {}
    eval_datasets = {}
    for task_name in DEFAULT_TASKS:
        print(f"[data:start] task={task_name}", flush=True)
        train_dataset, eval_dataset = load_task_datasets(
            task_name,
            train_count=args.train_samples_per_task,
            eval_count=args.eval_samples_per_task,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
        )
        train_datasets[task_name] = train_dataset
        eval_datasets[task_name] = eval_dataset
        print(
            f"[data] task={task_name} train_rows={len(train_dataset)} eval_rows={len(eval_dataset)} "
            f"train_seed={args.train_seed} eval_seed={args.eval_seed}",
            flush=True,
        )

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

    stacked_state: dict[str, torch.Tensor] | None = None
    all_losses = {}
    learned_task_eval_results = []

    for task_index, task_name in enumerate(DEFAULT_TASKS):
        task_seed = args.train_seed + task_index
        set_global_seed(task_seed)
        train_adapter_name = f"task_{task_index}_{task_name}"
        print(
            f"[task:start] index={task_index + 1}/{len(DEFAULT_TASKS)} name={task_name} "
            f"adapter={train_adapter_name} seed={task_seed}",
            flush=True,
        )
        print(f"[model:start] adapter={train_adapter_name} loading base model/adapters", flush=True)
        tokenizer, model, old_adapter_name, active_adapters = build_model_for_task(
            config=config,
            token=token,
            args=args,
            target_modules=target_modules,
            stacked_state=stacked_state,
            train_adapter_name=train_adapter_name,
        )
        print(f"[model:done] adapter={train_adapter_name}", flush=True)

        task_dir = output_dir / train_adapter_name
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dataloader:start] task={task_name}", flush=True)
        dataloader = build_dataloader(
            tokenizer,
            config,
            task_name,
            train_datasets[task_name],
            shuffle_seed=task_seed,
        )
        print(f"[dataloader:done] task={task_name} batches={len(dataloader)}", flush=True)
        print(
            f"[task] index={task_index + 1}/{len(DEFAULT_TASKS)} name={task_name} adapter={train_adapter_name} "
            f"active_adapters={active_adapters} stack_rank={adapter_rank(stacked_state) if stacked_state is not None else 0}",
            flush=True,
        )

        epoch_losses = train_one_task(
            model=model,
            dataloader=dataloader,
            config=config,
            train_adapter_name=train_adapter_name,
            active_adapters=active_adapters,
            old_adapter_name=old_adapter_name,
            orthogonal_penalty_enabled=args.enable_orthogonal_penalty,
            orthogonal_penalty_weight=args.orthogonal_penalty_weight,
            l2_penalty_weight=args.l2_penalty_weight,
            log_every_steps=args.log_every_steps,
        )
        all_losses[task_name] = epoch_losses
        save_json(task_dir / "epoch_losses.json", epoch_losses)
        print(f"[save] path={task_dir / 'epoch_losses.json'}", flush=True)

        print(f"[adapter:start] adapter={train_adapter_name} extracting and stacking LoRA state", flush=True)
        new_state = extract_adapter_state(model, train_adapter_name)
        stacked_state = concatenate_adapter_states(stacked_state, new_state)
        del model
        release_memory()
        print(
            f"[adapter:done] adapter={train_adapter_name} stacked_rank={adapter_rank(stacked_state)}",
            flush=True,
        )

        print(f"[eval:model:start] after_task={task_name} building stacked snapshot", flush=True)
        tokenizer, snapshot_model = build_model_from_stacked_state(
            config=config,
            token=token,
            args=args,
            target_modules=target_modules,
            stacked_state=stacked_state,
        )
        learned_tasks = DEFAULT_TASKS[: task_index + 1]
        print(
            f"[eval:start] after_task={task_name} tasks={','.join(learned_tasks)}",
            flush=True,
        )
        stage_eval_results = evaluate_task_sequence(
            tokenizer=tokenizer,
            model=snapshot_model,
            config=config,
            task_names=learned_tasks,
            eval_datasets=eval_datasets,
            max_new_tokens=args.max_new_tokens,
            stack_adapter_name=STACK_ADAPTER_NAME,
            build_question=build_question,
            label_text=label_text,
            task_spec=task_spec,
            set_active_adapters=set_active_adapters,
        )
        learned_task_eval_results.append(
            {
                "after_task": task_name,
                "after_task_index": task_index,
                "tasks_evaluated": list(learned_tasks),
                "task_eval_results": stage_eval_results,
            }
        )
        task_eval_output_dir = eval_output_dir_for_checkpoint(output_dir, train_adapter_name)
        task_eval_output_dir.mkdir(parents=True, exist_ok=True)
        save_json(task_eval_output_dir / "learned_task_eval_results.json", stage_eval_results)
        print(f"[save] path={task_eval_output_dir / 'learned_task_eval_results.json'}", flush=True)
        print(f"[save:start] adapter={train_adapter_name} path={task_dir}", flush=True)
        snapshot_model.save_pretrained(task_dir)
        tokenizer.save_pretrained(task_dir)
        save_lora_ab(snapshot_model, task_dir)
        print(f"[save:done] adapter={train_adapter_name} path={task_dir}", flush=True)
        del snapshot_model
        release_memory()
        print(f"[task:done] index={task_index + 1}/{len(DEFAULT_TASKS)} name={task_name}", flush=True)

    if stacked_state is None:
        raise SystemExit("No tasks were trained, so there is no stacked adapter to evaluate.")

    print("[final:model:start] building final stacked model", flush=True)
    tokenizer, final_model = build_model_from_stacked_state(
        config=config,
        token=token,
        args=args,
        target_modules=target_modules,
        stacked_state=stacked_state,
    )
    print(f"[final:eval:start] tasks={','.join(DEFAULT_TASKS)}", flush=True)
    eval_results = evaluate_task_sequence(
        tokenizer=tokenizer,
        model=final_model,
        config=config,
        task_names=DEFAULT_TASKS,
        eval_datasets=eval_datasets,
        max_new_tokens=args.max_new_tokens,
        stack_adapter_name=STACK_ADAPTER_NAME,
        build_question=build_question,
        label_text=label_text,
        task_spec=task_spec,
        set_active_adapters=set_active_adapters,
    )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"[final:save:start] path={final_dir}", flush=True)
    final_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_lora_ab(final_model, final_dir)
    print(f"[final:save:done] path={final_dir}", flush=True)

    result = {
        "experiment": "ag_news_yelp_dbpedia_stacked_lora_cl",
        "model_id": args.model_id,
        "tasks": list(DEFAULT_TASKS),
        "train_samples_per_task": args.train_samples_per_task,
        "eval_samples_per_task": args.eval_samples_per_task,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "final_stacked_rank": adapter_rank(stacked_state),
        "target_modules": list(target_modules),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "orthogonal_penalty_enabled": args.enable_orthogonal_penalty,
        "orthogonal_penalty": "||A_old @ A_new.T||_1",
        "orthogonal_penalty_weight": args.orthogonal_penalty_weight,
        "l2_penalty": "||A_new||_2 + ||B_new||_2",
        "l2_penalty_weight": args.l2_penalty_weight,
        "adapter_composition": "stacked_concatenation",
        "adapter_composition_formula": (
            "W + (alpha/r) * B_old A_old + (alpha/r) * B_new A_new during training, "
            "then save by concatenation"
        ),
        "evaluation_metric": "accuracy",
        "learned_task_eval_results": learned_task_eval_results,
        "task_eval_results": eval_results,
    }
    task_eval_summary = build_task_eval_summary(
        learned_task_eval_results=learned_task_eval_results,
        final_task_eval_results=eval_results,
    )
    save_json(output_dir / "experiment_metadata.json", result)
    save_json(output_dir / "all_epoch_losses.json", all_losses)
    save_json(eval_output_dir / "learned_task_eval_results.json", learned_task_eval_results)
    save_json(eval_output_dir / "task_eval_results.json", task_eval_summary)
    write_learned_task_eval_results_txt(eval_output_dir / "learned_task_eval_results.txt", learned_task_eval_results)
    write_task_eval_results_txt(eval_output_dir / "task_eval_results.txt", learned_task_eval_results, eval_results)
    print(f"[metadata:done] path={output_dir}", flush=True)
    print(f"[eval:outputs] path={eval_output_dir}", flush=True)

    del final_model
    release_memory()
    print(f"[done] outputs -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
