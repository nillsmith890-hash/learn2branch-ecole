"""Evaluate zero-shot and few-shot performance of a Reptile initialization."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    rows = []
    for task_index, task in enumerate(args.tasks):
        sampler = TaskEpisodeSampler(
            args.data_root, [task], args.split, args.support_size,
            args.query_size, args.seed + task_index
        )
        zero_shot, adapted_results = [], []
        for _ in range(args.episodes):
            episode = sampler.sample_task(task)
            support = load_batch(episode["support_files"], device)
            query = load_batch(episode["query_files"], device)
            zero_shot.append(evaluate(policy, query))
            params = adapt(policy, support, args.inner_lr, args.inner_steps)
            adapted_results.append(evaluate(policy, query, params))
        row = {"task": task}
        for prefix, results in (("zero_shot", zero_shot), ("adapted", adapted_results)):
            for metric in ("loss",) + tuple(f"acc@{k}" for k in TOP_K):
                values = [result[metric] for result in results]
                row[f"{prefix}_{metric}_mean"] = float(np.mean(values))
                row[f"{prefix}_{metric}_std"] = float(np.std(values))
        rows.append(row)
        print(f"{task}: zero-shot acc@1 {row['zero_shot_acc@1_mean']:.3f}; "
              f"adapted acc@1 {row['adapted_acc@1_mean']:.3f}")

    output = args.output or args.checkpoint.with_name(f"eval_{args.split}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"results written to: {output}")


if __name__ == "__main__":
    main()

