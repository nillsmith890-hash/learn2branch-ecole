"""Diagnose cross-domain PreNorm mismatch without updating model weights.

The script evaluates a trained checkpoint in two conditions:

1. source statistics: use the PreNorm buffers stored in the checkpoint;
2. target recalibration: recompute only the seven PreNorm buffers from
   unlabeled samples in the target task's training split.

Evaluation always uses the target task's separate test split. Recalibration
loads the graph features but never uses expert actions or scores.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare source-domain and target-recalibrated PreNorm statistics."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        required=True,
        help="Target task relative to data root; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=pathlib.Path("data/samples_meta")
    )
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 300],
        help="Number of target train samples per task used for recalibration.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional JSON output path. Parent directories are created.",
    )
    args = parser.parse_args()

    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    if any(size <= 0 for size in args.calibration_sizes):
        parser.error("--calibration-sizes must contain positive integers")
    if args.batch_size <= 0 or args.pretrain_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def list_samples(data_root, task, split):
    directory = data_root / task / split
    files = sorted(str(path) for path in directory.glob("sample_*.pkl"))
    if not files:
        raise FileNotFoundError(f"no sample_*.pkl files found in {directory}")
    return files


def recalibrate_prenorm(policy, data_loader, device):
    """Re-estimate all PreNorm buffers while leaving learned weights untouched."""
    policy.pre_train_init()
    n_layers = 0
    while True:
        for batch in data_loader:
            batch = batch.to(device)
            if not policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            ):
                break
        if policy.pre_train_next() is None:
            break
        n_layers += 1
    return n_layers


def evaluate(policy, data_loader, device, torch, functional, pad_tensor, top_k):
    policy.eval()
    total_loss = 0.0
    total_kacc = np.zeros(len(top_k), dtype=np.float64)
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            loss = functional.cross_entropy(
                logits, batch.candidate_choices, reduction="mean"
            )

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_best = true_scores.max(dim=-1, keepdim=True).values
            batch_kacc = []
            for k in top_k:
                effective_k = min(k, logits.shape[-1])
                predicted = logits.topk(effective_k, dim=-1).indices
                predicted_scores = true_scores.gather(-1, predicted)
                accuracy = (predicted_scores == true_best).any(dim=-1).float().mean()
                batch_kacc.append(accuracy.item())

            n_graphs = batch.num_graphs
            total_loss += loss.item() * n_graphs
            total_kacc += np.asarray(batch_kacc) * n_graphs
            total_samples += n_graphs

    if total_samples == 0:
        raise RuntimeError("evaluation loader produced no samples")
    return total_loss / total_samples, total_kacc / total_samples, total_samples


def result_dict(loss, kacc, n_samples, top_k):
    result = {"samples": int(n_samples), "loss": float(loss)}
    result.update({f"acc@{k}": float(value) for k, value in zip(top_k, kacc)})
    return result


def print_result(prefix, result, top_k):
    metrics = " ".join(f"acc@{k}: {result[f'acc@{k}']:.3f}" for k in top_k)
    print(
        f"{prefix} samples: {result['samples']} loss: {result['loss']:.3f} {metrics}",
        flush=True,
    )


def main():
    args = parse_args()

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda:0"

    import torch
    import torch.nn.functional as functional
    import torch_geometric

    from utilities import GraphDataset, pad_tensor

    sys.path.insert(0, os.path.abspath("model"))
    from model import GNNPolicy

    top_k = [1, 3, 5, 10]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    train_files = {
        task: list_samples(args.data_root, task, "train") for task in args.tasks
    }
    test_files = {
        task: list_samples(args.data_root, task, "test") for task in args.tasks
    }
    test_loaders = {
        task: torch_geometric.loader.DataLoader(
            GraphDataset(files),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.gpu != -1,
        )
        for task, files in test_files.items()
    }

    checkpoint_state = torch.load(args.checkpoint, map_location=device)

    def fresh_policy():
        policy = GNNPolicy().to(device)
        policy.load_state_dict(checkpoint_state)
        return policy

    output = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "tasks": args.tasks,
        "seed": args.seed,
        "device": device,
        "source_prenorm": {},
        "target_recalibration": {},
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device} (requested gpu {args.gpu})")
    print(f"target tasks: {args.tasks}")
    print("\n=== SOURCE CHECKPOINT PRENORM ===")
    policy = fresh_policy()
    for task, loader in test_loaders.items():
        result = result_dict(
            *evaluate(policy, loader, device, torch, functional, pad_tensor, top_k),
            top_k,
        )
        output["source_prenorm"][task] = result
        print_result(task, result, top_k)
    del policy

    for size in sorted(set(args.calibration_sizes)):
        calibration_files = []
        selected_per_task = {}
        for task, files in train_files.items():
            count = min(size, len(files))
            selected = rng.choice(files, size=count, replace=False).tolist()
            calibration_files.extend(selected)
            selected_per_task[task] = count

        calibration_loader = torch_geometric.loader.DataLoader(
            GraphDataset(calibration_files),
            batch_size=args.pretrain_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.gpu != -1,
        )
        policy = fresh_policy()
        started = time.time()
        n_layers = recalibrate_prenorm(policy, calibration_loader, device)
        elapsed = time.time() - started

        setting = {
            "samples_per_task": selected_per_task,
            "total_calibration_samples": len(calibration_files),
            "prenorm_layers": n_layers,
            "recalibration_seconds": elapsed,
            "test": {},
        }
        print(f"\n=== TARGET PRENORM: {size} TRAIN SAMPLES PER TASK ===")
        print(
            f"recalibrated layers: {n_layers}; total calibration samples: "
            f"{len(calibration_files)}; time: {elapsed:.1f}s"
        )
        for task, loader in test_loaders.items():
            result = result_dict(
                *evaluate(policy, loader, device, torch, functional, pad_tensor, top_k),
                top_k,
            )
            setting["test"][task] = result
            print_result(task, result, top_k)
        output["target_recalibration"][str(size)] = setting
        del policy

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(output, file, indent=2)
        print(f"\nresults written to: {args.output}")


if __name__ == "__main__":
    main()
