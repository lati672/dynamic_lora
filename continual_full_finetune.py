import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_lora.core.constants import (  # noqa: E402
    DEFAULT_CONTINUAL_EPOCHS,
    DEFAULT_CONTINUAL_FULL_OUTPUT_DIR,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_SAMPLES,
)
from dynamic_lora.core.continual_full_training import train_full_one_task  # noqa: E402
from dynamic_lora.core.data_pipeline import (  # noqa: E402
    build_dataloader,
    build_question,
    label_text,
    load_task_datasets,
    task_spec,
)
from dynamic_lora.core.eval_export import (  # noqa: E402
    build_task_eval_summary,
    write_learned_task_eval_results_txt,
    write_task_eval_results_txt,
)
from dynamic_lora.core.eval_pipeline import evaluate_task_sequence  # noqa: E402
from dynamic_lora.core.io_utils import release_memory, save_json  # noqa: E402
from dynamic_lora.core.lora_app.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    FULL_LEARNING_RATE,
    MAX_LENGTH,
    MODEL_ID,
    TrainingConfig,
)
from dynamic_lora.core.lora_app.modeling import load_base_model  # noqa: E402
from dynamic_lora.core.seed_utils import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a continual-learning classification experiment with full finetuning "
            "on AG News, Yelp Review Full, and DBPedia 14."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_CONTINUAL_FULL_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONTINUAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=FULL_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--train-samples-per-task", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--eval-samples-per-task", type=int, default=DEFAULT_EVAL_SAMPLES)
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


def set_active_full_model(*args, **kwargs) -> None:
    return None


def main() -> None:
    args = parse_args()
    set_global_seed(args.train_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    print(
        f"[run:start] experiment=continual_full_finetune model={args.model_id} "
        f"tasks={','.join(DEFAULT_TASKS)} train_samples_per_task={args.train_samples_per_task} "
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

    print("[model:start] loading base model for full finetuning", flush=True)
    tokenizer, model = load_base_model(config, token)
    print("[model:done] base model loaded", flush=True)

    all_losses = {}
    learned_task_eval_results = []

    for task_index, task_name in enumerate(DEFAULT_TASKS):
        task_seed = args.train_seed + task_index
        set_global_seed(task_seed)
        checkpoint_name = f"task_{task_index}_{task_name}_full"
        task_dir = output_dir / checkpoint_name
        task_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[task:start] index={task_index + 1}/{len(DEFAULT_TASKS)} name={task_name} "
            f"checkpoint={checkpoint_name} seed={task_seed}",
            flush=True,
        )

        print(f"[dataloader:start] task={task_name}", flush=True)
        dataloader = build_dataloader(
            tokenizer,
            config,
            task_name,
            train_datasets[task_name],
            shuffle_seed=task_seed,
        )
        print(f"[dataloader:done] task={task_name} batches={len(dataloader)}", flush=True)

        epoch_losses = train_full_one_task(
            model=model,
            dataloader=dataloader,
            config=config,
            task_name=task_name,
            log_every_steps=args.log_every_steps,
        )
        all_losses[task_name] = epoch_losses
        save_json(task_dir / "epoch_losses.json", epoch_losses)
        print(f"[save] path={task_dir / 'epoch_losses.json'}", flush=True)

        learned_tasks = DEFAULT_TASKS[: task_index + 1]
        print(
            f"[eval:start] after_task={task_name} tasks={','.join(learned_tasks)}",
            flush=True,
        )
        stage_eval_results = evaluate_task_sequence(
            tokenizer=tokenizer,
            model=model,
            config=config,
            task_names=learned_tasks,
            eval_datasets=eval_datasets,
            max_new_tokens=args.max_new_tokens,
            stack_adapter_name="full_model",
            build_question=build_question,
            label_text=label_text,
            task_spec=task_spec,
            set_active_adapters=set_active_full_model,
        )
        learned_task_eval_results.append(
            {
                "after_task": task_name,
                "after_task_index": task_index,
                "tasks_evaluated": list(learned_tasks),
                "task_eval_results": stage_eval_results,
            }
        )
        save_json(task_dir / "learned_task_eval_results.json", stage_eval_results)
        print(f"[save] path={task_dir / 'learned_task_eval_results.json'}", flush=True)

        print(f"[save:start] checkpoint={checkpoint_name} path={task_dir}", flush=True)
        model.save_pretrained(task_dir)
        tokenizer.save_pretrained(task_dir)
        print(f"[save:done] checkpoint={checkpoint_name} path={task_dir}", flush=True)
        release_memory()
        print(f"[task:done] index={task_index + 1}/{len(DEFAULT_TASKS)} name={task_name}", flush=True)

    print(f"[final:eval:start] tasks={','.join(DEFAULT_TASKS)}", flush=True)
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

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"[final:save:start] path={final_dir}", flush=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[final:save:done] path={final_dir}", flush=True)

    result = {
        "experiment": "ag_news_yelp_dbpedia_full_finetune_cl",
        "model_id": args.model_id,
        "tasks": list(DEFAULT_TASKS),
        "train_samples_per_task": args.train_samples_per_task,
        "eval_samples_per_task": args.eval_samples_per_task,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "training_mode": "full_finetuning",
        "checkpoint_strategy": "save_full_model_after_each_task",
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
    save_json(output_dir / "learned_task_eval_results.json", learned_task_eval_results)
    save_json(output_dir / "task_eval_results.json", task_eval_summary)
    write_learned_task_eval_results_txt(output_dir / "learned_task_eval_results.txt", learned_task_eval_results)
    write_task_eval_results_txt(output_dir / "task_eval_results.txt", learned_task_eval_results, eval_results)
    print(f"[metadata:done] path={output_dir}", flush=True)

    del model
    release_memory()
    print(f"[done] outputs -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
