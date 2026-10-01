"""Compute uniform-random top-k baselines for graph branching samples.

For a sample with ``n`` candidates, a uniformly random ranking contains the
expert action in its first ``k`` positions with probability ``min(k, n) / n``.
The reported baseline averages this exact expectation over all checked samples;
it does not use Monte Carlo random predictions.

Examples
--------
Check every meta-task test split::

    python 06_random_baseline.py

Check only the four LODO target domains::

    python 06_random_baseline.py \
        --task cauctions/50_250 --task cauctions/75_375 \
        --task indset/300_4 --task indset/500_6 \
        --task setcover/250r_500c_0.05d \
        --task setcover/500r_1000c_0.02d \
        --task facilities/50_50_5 --task facilities/100_100_3
"""

import argparse
import gzip
import pickle
import random
from pathlib import Path

import numpy as np


DEFAULT_TOP_K = (1, 3, 5, 10)


def discover_tasks(root):
    return sorted(
        f"{domain_dir.name}/{task_dir.name}"
        for domain_dir in root.iterdir()
        if domain_dir.is_dir()
        for task_dir in domain_dir.iterdir()
        if task_dir.is_dir()
    )


def inspect_task(root, task, split, top_k, max_samples, rng):
    sample_dir = root / task / split
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample split does not exist: {sample_dir}")

    files = sorted(sample_dir.glob("sample_*.pkl"))
    total_files = len(files)
    if not files:
        raise ValueError(f"no sample_*.pkl files found in {sample_dir}")
    if max_samples and len(files) > max_samples:
        files = rng.sample(files, max_samples)

    candidate_counts = []
    corrupt_files = 0
    invalid_actions = 0
    for filename in files:
        try:
            with gzip.open(filename, "rb") as file:
                sample = pickle.load(file)
            _, action, action_set, _ = sample["data"]
            action_set = np.asarray(action_set, dtype=np.int64)
            if action_set.ndim != 1 or action_set.size == 0:
                raise ValueError("candidate set must be a non-empty vector")
            if np.count_nonzero(action_set == int(action)) != 1:
                invalid_actions += 1
                continue
            candidate_counts.append(action_set.size)
        except Exception as error:
            corrupt_files += 1
            print(f"WARNING: could not read {filename}: {error}")

    if not candidate_counts:
        raise ValueError(f"no valid samples found in {sample_dir}")

    counts = np.asarray(candidate_counts, dtype=np.int64)
    random_acc = {
        k: float(np.mean(np.minimum(k, counts) / counts)) for k in top_k
    }
    return {
        "task": task,
        "split": split,
        "total_files": total_files,
        "checked": len(files),
        "valid": len(counts),
        "corrupt": corrupt_files,
        "invalid_actions": invalid_actions,
        "candidate_min": int(counts.min()),
        "candidate_mean": float(counts.mean()),
        "candidate_median": float(np.median(counts)),
        "candidate_max": int(counts.max()),
        "random_acc": random_acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute exact expected top-k accuracy of a uniform random ranking."
    )
    parser.add_argument("--root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task relative to --root, for example indset/500_6; repeat as needed.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Randomly inspect at most this many samples per task; 0 uses all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"sample root does not exist: {args.root}")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if not args.top_k or any(k < 1 for k in args.top_k):
        parser.error("--top-k values must be positive")
    top_k = tuple(sorted(set(args.top_k)))

    tasks = args.task or discover_tasks(args.root)
    if not tasks:
        parser.error(f"no tasks found below {args.root}")

    rng = random.Random(args.seed)
    results = [
        inspect_task(
            args.root, task, args.split, top_k, args.max_samples, rng
        )
        for task in tasks
    ]

    metric_headers = " ".join(f"random@{k:>2}" for k in top_k)
    print(f"root: {args.root}")
    print(f"split: {args.split}")
    print(
        f"{'task':38} {'valid':>6} {'candidates min/mean/median/max':>32} "
        f"{metric_headers}"
    )
    for result in results:
        candidate_summary = (
            f"{result['candidate_min']}/"
            f"{result['candidate_mean']:.2f}/"
            f"{result['candidate_median']:.1f}/"
            f"{result['candidate_max']}"
        )
        metrics = " ".join(
            f"{result['random_acc'][k]:9.6f}" for k in top_k
        )
        print(
            f"{result['task']:38} {result['valid']:6d} "
            f"{candidate_summary:>32} {metrics}"
        )
        if result["corrupt"] or result["invalid_actions"]:
            print(
                f"  WARNING: corrupt={result['corrupt']}, "
                f"invalid_actions={result['invalid_actions']}"
            )


if __name__ == "__main__":
    main()
