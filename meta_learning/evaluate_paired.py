"""Compare Reptile and its LODO initialization on identical test episodes."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def bootstrap_interval(values, rng, repeats):
    """Percentile interval for an episode mean, conditional on these episodes."""
    values = np.asarray(values, dtype=float)
    draws = rng.randint(len(values), size=(repeats, len(values)))
    means = values[draws].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--reptile-checkpoint", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("support_size", "query_size", "inner_steps", "episodes", "bootstrap_repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.inner_lr <= 0:
        parser.error("--inner-lr must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == -1 else str(args.gpu)
    device = "cpu" if args.gpu == -1 else "cuda:0"
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "model"))
    from model import GNNPolicy
    from meta_learning.common import TOP_K, adapt, evaluate, load_batch
    from meta_learning.task_sampler import TaskEpisodeSampler

    torch.manual_seed(args.seed)
    policies = {}
    for label, checkpoint in (
        ("baseline", args.baseline_checkpoint),
        ("reptile", args.reptile_checkpoint),
    ):
        policy = GNNPolicy().to(device)
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        policy.eval()
        policies[label] = policy

    metrics = ("loss",) + tuple(f"acc@{k}" for k in TOP_K)
    rows = []
    per_task = {}
    for task_index, task in enumerate(args.tasks):
        sampler = TaskEpisodeSampler(
            args.data_root, [task], args.split, args.support_size,
            args.query_size, args.seed + task_index,
        )
        task_rows = []
        for episode_index in range(1, args.episodes + 1):
            episode = sampler.sample_task(task)
            support = load_batch(episode["support_files"], device)
            query = load_batch(episode["query_files"], device)
            row = {"task": task, "episode": episode_index}
            for label, policy in policies.items():
                zero_shot = evaluate(policy, query)
                adapted = adapt(policy, support, args.inner_lr, args.inner_steps)
                few_shot = evaluate(policy, query, adapted)
                for metric in metrics:
                    row[f"{label}_zero_shot_{metric}"] = zero_shot[metric]
                    row[f"{label}_adapted_{metric}"] = few_shot[metric]
            for metric in metrics:
                row[f"delta_adapted_{metric}"] = (
                    row[f"reptile_adapted_{metric}"]
                    - row[f"baseline_adapted_{metric}"]
                )
            task_rows.append(row)
            rows.append(row)
        per_task[task] = task_rows
        print(f"{task}: {len(task_rows)} paired episodes completed", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.RandomState(args.seed + 10000)
    summary = []
    for task, task_rows in per_task.items():
        result = {"task": task, "episodes": len(task_rows)}
        for metric in ("loss", "acc@1"):
            values = [row[f"delta_adapted_{metric}"] for row in task_rows]
            interval = bootstrap_interval(values, rng, args.bootstrap_repeats)
            result[f"delta_{metric}_mean"] = float(np.mean(values))
            result[f"delta_{metric}_ci_low"] = float(interval[0])
            result[f"delta_{metric}_ci_high"] = float(interval[1])
        summary.append(result)

    macro = {"task": "MACRO", "episodes": args.episodes * len(args.tasks)}
    for metric in ("loss", "acc@1"):
        task_values = [
            np.asarray([row[f"delta_adapted_{metric}"] for row in per_task[task]])
            for task in args.tasks
        ]
        bootstrapped = []
        for values in task_values:
            draws = rng.randint(len(values), size=(args.bootstrap_repeats, len(values)))
            bootstrapped.append(values[draws].mean(axis=1))
        macro_draws = np.stack(bootstrapped).mean(axis=0)
        interval = np.percentile(macro_draws, [2.5, 97.5])
        macro[f"delta_{metric}_mean"] = float(np.mean([values.mean() for values in task_values]))
        macro[f"delta_{metric}_ci_low"] = float(interval[0])
        macro[f"delta_{metric}_ci_high"] = float(interval[1])
    summary.append(macro)

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['task']}: Reptile - LODO adapted acc@1 "
            f"{row['delta_acc@1_mean']:+.4f} "
            f"[{row['delta_acc@1_ci_low']:+.4f}, "
            f"{row['delta_acc@1_ci_high']:+.4f}]",
            flush=True,
        )
    print(f"episode results: {args.output}")
    print(f"summary: {summary_path}")
    print("Intervals resample episodes within each task; overlapping source instances can make them optimistic.")


if __name__ == "__main__":
    main()
