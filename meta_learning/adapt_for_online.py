"""Make paired few-shot checkpoints for an online SCIP comparison."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

def select_support(data_root, task, size, seed):
    """Draw without replacement, spreading samples across source instances."""
    directory = data_root / task / "valid"
    files = sorted(directory.glob("sample_*.pkl"))
    if len(files) < size:
        raise ValueError(f"{directory} has {len(files)} samples, need {size}")

    groups = defaultdict(list)
    for path in files:
        with gzip.open(path, "rb") as stream:
            instance = pickle.load(stream).get("instance")
        if instance is None:
            raise ValueError(f"sample has no source instance: {path}")
        source = Path(instance)
        if (source.parent.name, source.parent.parent.name, source.parent.parent.parent.name) != (
            "valid", task.split("/")[1], task.split("/")[0]
        ):
            raise ValueError(f"sample is not from task {task!r} valid split: {path}: {instance}")
        groups[str(source)].append(path)

    rng = random.Random(seed)
    for paths in groups.values():
        rng.shuffle(paths)
    selected = []
    names = list(groups)
    while len(selected) < size:
        rng.shuffle(names)
        for name in names:
            if groups[name]:
                selected.append((groups[name].pop(), name))
                if len(selected) == size:
                    break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--reptile-checkpoint", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/samples_meta"))
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID, or -1 for CPU (default)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.support_size <= 0 or args.inner_steps <= 0 or args.inner_lr <= 0:
        parser.error("support size, inner steps, and inner learning rate must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    if any(len(task.split("/")) != 2 for task in args.tasks):
        parser.error("each --task must be domain/configuration")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    for checkpoint in (args.baseline_checkpoint, args.reptile_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")

    # Select all support sets before importing the GPU stack or writing outputs.
    support_by_task = {
        task: select_support(args.data_root, task, args.support_size, args.seed + index)
        for index, task in enumerate(args.tasks)
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == -1 else str(args.gpu)
    device = "cpu" if args.gpu == -1 else "cuda:0"
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "model"))
    from model import GNNPolicy
    from meta_learning.common import adapt, load_batch

    torch.manual_seed(args.seed)
    adapted_states = {}
    for task, support in support_by_task.items():
        batch = load_batch([str(path) for path, _ in support], device)
        for label, checkpoint in (
            ("lodo", args.baseline_checkpoint),
            ("reptile", args.reptile_checkpoint),
        ):
            policy = GNNPolicy().to(device)
            policy.load_state_dict(torch.load(checkpoint, map_location=device))
            policy.eval()
            adapted_params = adapt(policy, batch, args.inner_lr, args.inner_steps)
            state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            for name, value in adapted_params.items():
                state[name] = value.detach().cpu().clone()
            adapted_states[(task, label)] = state
            print(f"adapted {label} on {task}: {len(support)} shared support samples", flush=True)

    args.output_dir.mkdir(parents=True)
    manifest = {
        "protocol": "valid-split support; same support and gradient settings for both checkpoints",
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "reptile_checkpoint": str(args.reptile_checkpoint),
        "support_size": args.support_size,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "seed": args.seed,
        "tasks": {},
    }
    for task, support in support_by_task.items():
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True)
        manifest["tasks"][task] = {
            "support": [
                {
                    "sample": str(path),
                    "source_instance": instance,
                    "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path, instance in support
            ],
            "adapted_checkpoints": {},
        }
        for label in ("lodo", "reptile"):
            output = task_dir / f"{label}_adapted.pkl"
            torch.save(adapted_states[(task, label)], output)
            manifest["tasks"][task]["adapted_checkpoints"][label] = str(output)
            print(output, flush=True)
    manifest_path = args.output_dir / "support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"support manifest: {manifest_path}")


if __name__ == "__main__":
    main()
