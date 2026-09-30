"""Audit every generated meta-learning sample task.

Run from the learn2branch-ecole repository root:

    python check_meta_samples.py
    python check_meta_samples.py --max-files-per-split 300

The default checks every ``.pkl`` below
``data/samples_meta/<domain>/<task>/<split>``.
"""

import argparse
import gzip
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "valid", "test")


def mean_or_zero(total, count):
    return total / count if count else 0.0


def inspect_split(sample_dir, max_files, rng):
    files = sorted(sample_dir.glob("*.pkl"))
    if max_files and len(files) > max_files:
        checked = rng.sample(files, max_files)
    else:
        checked = files

    result = {
        "total_files": len(files),
        "checked_files": len(checked),
        "nonfinite_samples": 0,
        "nonfinite_candidates": 0,
        "total_candidates": 0,
        "wrong_actions": 0,
        "corrupt_files": 0,
        "constraints": 0,
        "variables": 0,
        "edges": 0,
        "instance_counts": Counter(),
    }

    for filename in checked:
        try:
            with gzip.open(filename, "rb") as file:
                sample = pickle.load(file)

            node_observation, action, action_set, scores = sample["data"]
            row_features, edge_data, variable_features = node_observation
            edge_indices, edge_values = edge_data

            row_features = np.asarray(row_features)
            edge_indices = np.asarray(edge_indices)
            edge_values = np.asarray(edge_values)
            variable_features = np.asarray(variable_features)
            action_set = np.asarray(action_set, dtype=np.int64)
            scores = np.asarray(scores)
            action = int(action)

            if edge_indices.ndim != 2 or edge_indices.shape[0] != 2:
                raise ValueError(f"invalid edge_indices shape {edge_indices.shape}")
            if len(edge_values) != edge_indices.shape[1]:
                raise ValueError("edge_values and edge_indices have different sizes")
            if action_set.size == 0:
                raise ValueError("empty action_set")
            if action_set.min() < 0 or action_set.max() >= len(scores):
                raise ValueError("action_set contains an out-of-range variable index")

            candidate_scores = scores[action_set]
            finite_candidates = np.isfinite(candidate_scores)
            all_finite = (
                np.isfinite(row_features).all()
                and np.isfinite(edge_values).all()
                and np.isfinite(variable_features).all()
                and finite_candidates.all()
            )

            result["total_candidates"] += len(action_set)
            result["nonfinite_candidates"] += int((~finite_candidates).sum())
            result["nonfinite_samples"] += int(not all_finite)
            result["constraints"] += row_features.shape[0]
            result["variables"] += variable_features.shape[0]
            result["edges"] += edge_indices.shape[1]
            result["instance_counts"][sample.get("instance", "<missing>")] += 1

            action_positions = np.flatnonzero(action_set == action)
            if len(action_positions) != 1 or not finite_candidates.all():
                result["wrong_actions"] += 1
            else:
                chosen_score = candidate_scores[action_positions[0]]
                best_score = candidate_scores.max()
                if not np.isclose(chosen_score, best_score, rtol=1e-10, atol=1e-12):
                    result["wrong_actions"] += 1

        except Exception as error:
            result["corrupt_files"] += 1
            print(f"    ERROR {filename}: {error}")

    return result


def print_result(split, result):
    checked = result["checked_files"]
    instance_counts = list(result["instance_counts"].values())
    nonfinite_ratio = mean_or_zero(
        result["nonfinite_candidates"], result["total_candidates"]
    )

    print(f"  {split}:")
    print(f"    total files: {result['total_files']}")
    print(f"    checked files: {checked}")
    print(f"    samples containing NaN/Inf: {result['nonfinite_samples']}")
    print(f"    nonfinite candidates: {result['nonfinite_candidates']}")
    print(f"    nonfinite candidate ratio: {nonfinite_ratio:.8f}")
    print(f"    wrong expert actions: {result['wrong_actions']}")
    print(f"    corrupt files: {result['corrupt_files']}")
    print(f"    average candidates: {mean_or_zero(result['total_candidates'], checked):.2f}")
    print(f"    average constraints: {mean_or_zero(result['constraints'], checked):.2f}")
    print(f"    average variables: {mean_or_zero(result['variables'], checked):.2f}")
    print(f"    average edges: {mean_or_zero(result['edges'], checked):.2f}")
    print(f"    unique source instances: {len(instance_counts)}")
    if instance_counts:
        print(
            "    samples per source instance (min/mean/max): "
            f"{min(instance_counts)}/"
            f"{np.mean(instance_counts):.2f}/"
            f"{max(instance_counts)}"
        )


def discover_tasks(root):
    return sorted(
        task_dir
        for domain_dir in root.iterdir()
        if domain_dir.is_dir()
        for task_dir in domain_dir.iterdir()
        if task_dir.is_dir()
    )


def main():
    parser = argparse.ArgumentParser(description="Audit all meta-learning sample tasks.")
    parser.add_argument("--root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=0,
        help="Randomly check at most this many files per split; 0 checks all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_files_per_split < 0:
        parser.error("--max-files-per-split must be non-negative")
    if not args.root.is_dir():
        parser.error(f"sample root does not exist: {args.root}")

    tasks = discover_tasks(args.root)
    if not tasks:
        parser.error(f"no <domain>/<task> directories found below {args.root}")

    rng = random.Random(args.seed)
    failures = 0
    print(f"root: {args.root}")
    print(f"tasks found: {len(tasks)}")
    print(
        "check mode: "
        + ("all files" if args.max_files_per_split == 0 else
           f"at most {args.max_files_per_split} files per split")
    )

    for task_dir in tasks:
        task_name = task_dir.relative_to(args.root)
        print(f"\n{task_name}:")
        for split in SPLITS:
            split_dir = task_dir / split
            if not split_dir.is_dir():
                print(f"  {split}: MISSING DIRECTORY")
                failures += 1
                continue

            result = inspect_split(split_dir, args.max_files_per_split, rng)
            print_result(split, result)
            failures += (
                result["total_files"] == 0
                or result["nonfinite_samples"] > 0
                or result["wrong_actions"] > 0
                or result["corrupt_files"] > 0
            )

    print(f"\nAUDIT RESULT: {'PASS' if failures == 0 else 'FAIL'}")
    if failures:
        print(f"problematic split checks: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
