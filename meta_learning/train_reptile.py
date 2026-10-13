"""Train a Reptile initialization across parameterized MILP tasks."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_TASKS = [
    "cauctions/50_250", "cauctions/75_375",
    "indset/300_4", "indset/500_6",
    "setcover/250r_500c_0.05d", "setcover/500r_1000c_0.02d",
    "facilities/50_50_5", "facilities/100_100_3",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--data-root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA id; -1 uses CPU")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--meta-step-size", type=float, default=0.1)
    parser.add_argument("--valid-every", type=int, default=20)
    parser.add_argument("--valid-episodes", type=int, default=8)
    parser.add_argument(
        "--valid-episodes-per-task", type=int,
        help="use a fixed, balanced validation bank with this many episodes per task",
    )
    parser.add_argument("--pretrain-samples-per-task", type=int, default=50)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    args.tasks = args.tasks or DEFAULT_TASKS
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    for name in ("iterations", "meta_batch_size", "support_size", "query_size",
                 "inner_steps", "valid_every", "valid_episodes",
                 "pretrain_samples_per_task"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.valid_episodes_per_task is not None and args.valid_episodes_per_task <= 0:
        parser.error("--valid-episodes-per-task must be positive")
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == -1 else str(args.gpu)
    device = "cpu" if args.gpu == -1 else "cuda:0"

    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "model"))
    from model import GNNPolicy
    from utilities import log
    from meta_learning.common import adapt, calibrate_prenorm, evaluate, load_batch, mean_metrics
    from meta_learning.task_sampler import TaskEpisodeSampler

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = root / "model" / "meta_learning" / args.run_name / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / "train_log.txt"
    if logfile.exists():
        logfile.unlink()
    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["initial_checkpoint"] = str(args.initial_checkpoint) if args.initial_checkpoint else None
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2))

    train_sampler = TaskEpisodeSampler(
        args.data_root, args.tasks, "train", args.support_size, args.query_size, args.seed
    )
    valid_sampler = TaskEpisodeSampler(
        args.data_root, args.tasks, "valid", args.support_size, args.query_size, args.seed + 1
    )
    fixed_valid_episodes = None
    valid_metrics_file = output_dir / "valid_metrics.csv"
    if args.valid_episodes_per_task is not None:
        fixed_valid_episodes = {
            task: [valid_sampler.sample_task(task) for _ in range(args.valid_episodes_per_task)]
            for task in args.tasks
        }
        (output_dir / "validation_episodes.json").write_text(
            json.dumps(fixed_valid_episodes, indent=2)
        )
        with valid_metrics_file.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("iteration", "task", "loss", "acc@1"))
    policy = GNNPolicy().to(device)
    if args.initial_checkpoint:
        policy.load_state_dict(torch.load(args.initial_checkpoint, map_location=device))
        log(f"loaded initialization: {args.initial_checkpoint}", logfile)
    else:
        rng = np.random.RandomState(args.seed)
        pretrain_files = []
        for task in args.tasks:
            all_files = [
                path for paths in train_sampler.files_by_instance[task].values()
                for path in paths
            ]
            count = min(args.pretrain_samples_per_task, len(all_files))
            pretrain_files.extend(rng.choice(all_files, count, replace=False).tolist())
        layers = calibrate_prenorm(policy, pretrain_files, device)
        log(f"calibrated {layers} PreNorm layers on {len(pretrain_files)} samples", logfile)

    log(f"device: {device} (requested gpu {args.gpu})", logfile)
    log(f"tasks: {args.tasks}", logfile)
    if fixed_valid_episodes is not None:
        log(
            f"fixed validation: {args.valid_episodes_per_task} episodes per task; "
            "checkpoint selection uses the macro-average across tasks",
            logfile,
        )
    best_loss = float("inf")
    checkpoint = output_dir / "best_params.pkl"

    for iteration in range(1, args.iterations + 1):
        episodes = train_sampler.sample_meta_batch(args.meta_batch_size)
        adapted_sets = []
        train_results = []
        for episode in episodes:
            support = load_batch(episode["support_files"], device)
            adapted = adapt(policy, support, args.inner_lr, args.inner_steps)
            adapted_sets.append(adapted)
            query = load_batch(episode["query_files"], device)
            train_results.append(evaluate(policy, query, adapted))

        fraction = args.meta_step_size
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                target = torch.stack([adapted[name].detach() for adapted in adapted_sets]).mean(0)
                parameter.add_(fraction * (target - parameter))

        train_mean = mean_metrics(train_results)
        log(f"ITER {iteration} TRAIN query loss: {train_mean['loss']:.6f} "
            f"acc@1: {train_mean['acc@1']:.3f}", logfile)

        if iteration % args.valid_every == 0:
            if fixed_valid_episodes is None:
                episodes_by_task = {"pooled": [
                    valid_sampler.sample_task() for _ in range(args.valid_episodes)
                ]}
            else:
                episodes_by_task = fixed_valid_episodes
            task_means = {}
            for task, episodes in episodes_by_task.items():
                results = []
                for episode in episodes:
                    support = load_batch(episode["support_files"], device)
                    adapted = adapt(policy, support, args.inner_lr, args.inner_steps)
                    query = load_batch(episode["query_files"], device)
                    results.append(evaluate(policy, query, adapted))
                task_means[task] = mean_metrics(results)
            valid_mean = {
                metric: float(np.mean([result[metric] for result in task_means.values()]))
                for metric in next(iter(task_means.values()))
            }
            if fixed_valid_episodes is not None:
                with valid_metrics_file.open("a", newline="") as stream:
                    writer = csv.writer(stream)
                    for task, result in task_means.items():
                        writer.writerow((iteration, task, result["loss"], result["acc@1"]))
                    writer.writerow((iteration, "MACRO", valid_mean["loss"], valid_mean["acc@1"]))
            log(f"ITER {iteration} VALID query loss: {valid_mean['loss']:.6f} "
                f"acc@1: {valid_mean['acc@1']:.3f}", logfile)
            if valid_mean["loss"] < best_loss:
                best_loss = valid_mean["loss"]
                torch.save(policy.state_dict(), checkpoint)
                log("  best meta-initialization so far", logfile)

    log(f"done; best validation loss: {best_loss:.6f}", logfile)


if __name__ == "__main__":
    main()

