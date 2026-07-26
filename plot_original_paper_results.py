#!/usr/bin/env python3
"""Create accuracy, forgetting, intruder accumulation, and association plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("outputs/original_paper_tasks"))
    args = parser.parse_args()
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    results = []
    for folder in ("full_finetune", "stacked_lora"):
        path = args.output_dir / folder / "results.csv"
        if path.exists():
            results.extend(read_csv(path))
    if not results:
        raise SystemExit("No results.csv files found; run the experiment first.")
    stages = list(dict.fromkeys(row["stage"] for row in results))
    methods = list(dict.fromkeys(row["method"] for row in results))

    for task in dict.fromkeys(row["eval_task"] for row in results):
        fig, axis = plt.subplots(figsize=(8, 5))
        for method in methods:
            rows = [row for row in results if row["method"] == method and row["eval_task"] == task]
            axis.plot([stages.index(row["stage"]) for row in rows], [float(row["accuracy"]) for row in rows],
                      marker="o", label=method)
        axis.set(xticks=range(len(stages)), xticklabels=stages, xlabel="training stage",
                 ylabel="accuracy", title=f"Continual accuracy: {task}")
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
        save(fig, plots / f"accuracy_{task}.png")

    forgetting = defaultdict(dict)
    fig, axis = plt.subplots(figsize=(8, 5))
    for method in methods:
        histories = defaultdict(list)
        values = []
        for stage in stages:
            stage_rows = [row for row in results if row["method"] == method and row["stage"] == stage]
            current = []
            for row in stage_rows:
                score = float(row["accuracy"])
                histories[row["eval_task"]].append(score)
                current.append(max(histories[row["eval_task"]]) - score)
            if current:
                forgetting[method][stage] = sum(current) / len(current)
                values.append(forgetting[method][stage])
        axis.plot(range(len(values)), values, marker="o", label=method)
    axis.set(xticks=range(len(stages)), xticklabels=stages, xlabel="training stage",
             ylabel="average forgetting", title="Average forgetting")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    save(fig, plots / "average_forgetting.png")

    intruder_path = args.output_dir / "intruder_analysis" / "intruder_counts.csv"
    if intruder_path.exists():
        intruders = read_csv(intruder_path)
        grouped = defaultdict(list)
        for row in intruders:
            grouped[(row["method"], row["adapter_eval_mode"], row["stage"])].append(float(row["num_intruders"]))
        means = {key: sum(values) / len(values) for key, values in grouped.items()}
        fig, axis = plt.subplots(figsize=(8, 5))
        series = list(dict.fromkeys((key[0], key[1]) for key in means))
        for method, mode in series:
            values = [means.get((method, mode, stage), float("nan")) for stage in stages]
            label = method if mode == "n/a" else f"{method} ({mode})"
            axis.plot(range(len(stages)), values, marker="o", label=label)
        axis.set(xticks=range(len(stages)), xticklabels=stages, xlabel="training stage",
                 ylabel="mean intruder count", title="Intruder accumulation")
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
        save(fig, plots / "intruder_accumulation.png")

        fig, axis = plt.subplots(figsize=(7, 5))
        for method, mode in series:
            points = [(means[(method, mode, stage)], forgetting.get(method, {}).get(stage))
                      for stage in stages if (method, mode, stage) in means and stage in forgetting.get(method, {})]
            if points:
                axis.scatter([point[0] for point in points], [point[1] for point in points],
                             label=method if mode == "n/a" else f"{method} ({mode})")
        axis.set(xlabel="mean intruder count", ylabel="average forgetting",
                 title="Intruder count vs forgetting")
        axis.legend()
        save(fig, plots / "intruders_vs_forgetting.png")
    else:
        print(f"[warning] {intruder_path} not found; skipping intruder plots")
    print(f"[done] plots written to {plots}")


if __name__ == "__main__":
    main()
