from pathlib import Path


def build_task_eval_summary(
    learned_task_eval_results: list[dict],
    final_task_eval_results: list[dict],
) -> dict:
    return {
        "learned_task_eval_results": learned_task_eval_results,
        "final_task_eval_results": final_task_eval_results,
    }


def write_learned_task_eval_results_txt(path: Path, learned_task_eval_results: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for stage in learned_task_eval_results:
            file.write(
                f"[after_task] name={stage['after_task']} index={stage['after_task_index']} "
                f"tasks_evaluated={','.join(stage['tasks_evaluated'])}\n"
            )
            for row in stage["task_eval_results"]:
                file.write(
                    f"task={row['task_name']} num_examples={row['num_examples']} "
                    f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
                )


def write_task_eval_results_txt(
    path: Path,
    learned_task_eval_results: list[dict],
    final_task_eval_results: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for stage in learned_task_eval_results:
            file.write(
                f"[after_task] name={stage['after_task']} index={stage['after_task_index']} "
                f"tasks_evaluated={','.join(stage['tasks_evaluated'])}\n"
            )
            for row in stage["task_eval_results"]:
                file.write(
                    f"task={row['task_name']} num_examples={row['num_examples']} "
                    f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
                )
        file.write("[continual_classification_eval]\n")
        for row in final_task_eval_results:
            file.write(
                f"task={row['task_name']} num_examples={row['num_examples']} "
                f"accuracy={row['accuracy']:.6f} correct={row['correct']}\n"
            )
