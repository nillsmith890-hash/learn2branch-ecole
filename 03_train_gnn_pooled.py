"""Train one task-balanced GNN branching policy on multiple MILP tasks."""

import argparse
import json
import os
import pathlib
import sys

import numpy as np


DEFAULT_TRAIN_TASKS = [
    "cauctions/50_250",
    "indset/300_4",
    "setcover/250r_500c_0.05d",
    "facilities/50_50_5",
]

DEFAULT_TEST_TASKS = [
    "cauctions/75_375",
    "indset/500_6",
    "setcover/500r_1000c_0.02d",
    "facilities/100_100_3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a task-balanced pooled Learn2Branch GNN."
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="CUDA GPU id; -1 uses CPU.")
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--data-root", type=pathlib.Path,
                        default=pathlib.Path("data/samples_meta"))
    parser.add_argument("--train-task", action="append", dest="train_tasks",
                        help="Task path relative to data root; may be repeated.")
    parser.add_argument("--test-task", action="append", dest="test_tasks",
                        help="Held-out task path; may be repeated.")
    parser.add_argument("--samples-per-task", type=int, default=1000)
    parser.add_argument("--pretrain-samples-per-task", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pretrain-batch-size", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    args = parser.parse_args()

    args.train_tasks = args.train_tasks or DEFAULT_TRAIN_TASKS
    args.test_tasks = args.test_tasks or DEFAULT_TEST_TASKS
    if len(set(args.train_tasks)) != len(args.train_tasks):
        parser.error("--train-task contains duplicates")
    if set(args.train_tasks) & set(args.test_tasks):
        parser.error("train and held-out task lists must be disjoint")
    for name in ("samples_per_task", "pretrain_samples_per_task", "batch_size",
                 "pretrain_batch_size", "valid_batch_size", "max_epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def list_samples(data_root, task, split):
    directory = data_root / task / split
    files = sorted(str(path) for path in directory.glob("sample_*.pkl"))
    if not files:
        raise FileNotFoundError(f"no samples found in {directory}")
    return files


def pretrain(policy, data_loader, device):
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


def process(policy, data_loader, device, pad_tensor, torch, F,
            top_k, entropy_bonus=0.0, optimizer=None):
    total_loss = 0.0
    total_entropy = 0.0
    total_kacc = np.zeros(len(top_k), dtype=np.float64)
    total_samples = 0

    policy.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy = F.cross_entropy(
                logits, batch.candidate_choices, reduction="mean"
            )
            entropy = (
                -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            ).sum(-1).mean()
            loss = cross_entropy - entropy_bonus * entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
            total_loss += cross_entropy.item() * n_graphs
            total_entropy += entropy.item() * n_graphs
            total_kacc += np.asarray(batch_kacc) * n_graphs
            total_samples += n_graphs

    if total_samples == 0:
        raise RuntimeError("data loader produced no samples")
    return (
        total_loss / total_samples,
        total_kacc / total_samples,
        total_entropy / total_samples,
    )


def metric_text(loss, kacc, top_k):
    return f"loss: {loss:.3f}" + "".join(
        f" acc@{k}: {accuracy:.3f}" for k, accuracy in zip(top_k, kacc)
    )


def evaluate_tasks(policy, loaders, **process_kwargs):
    results = {}
    for task, loader in loaders.items():
        loss, kacc, entropy = process(policy, loader, optimizer=None, **process_kwargs)
        results[task] = (loss, kacc, entropy)
    return results


def main():
    args = parse_args()

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda:0"

    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import (
        GraphDataset,
        Scheduler,
        TaskBalancedGraphDataset,
        TaskBalancedSampler,
        log,
        pad_tensor,
    )
    sys.path.insert(0, os.path.abspath("model"))
    from model import GNNPolicy

    top_k = [1, 3, 5, 10]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    running_dir = pathlib.Path("model") / "pooled" / args.run_name / str(args.seed)
    running_dir.mkdir(parents=True, exist_ok=True)
    logfile = running_dir / "train_log.txt"
    if logfile.exists():
        logfile.unlink()

    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    with open(running_dir / "train_config.json", "w") as file:
        json.dump(config, file, indent=2)

    log(f"device: {device} (requested gpu {args.gpu})", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"train tasks: {args.train_tasks}", logfile)
    log(f"held-out tasks: {args.test_tasks}", logfile)
    log(f"samples per task per epoch: {args.samples_per_task}", logfile)
    log(f"batch size: {args.batch_size}", logfile)
    log(f"learning rate: {args.lr}", logfile)

    train_task_files = {
        task: list_samples(args.data_root, task, "train")
        for task in args.train_tasks
    }
    train_dataset = TaskBalancedGraphDataset(train_task_files)
    train_sampler = TaskBalancedSampler(
        train_dataset.task_indices,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
    )
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    pretrain_files = []
    pretrain_rng = np.random.RandomState(args.seed)
    for task, files in train_task_files.items():
        count = min(args.pretrain_samples_per_task, len(files))
        chosen = pretrain_rng.choice(files, size=count, replace=False)
        pretrain_files.extend(chosen.tolist())
    pretrain_loader = torch_geometric.loader.DataLoader(
        GraphDataset(pretrain_files),
        batch_size=args.pretrain_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    valid_loaders = {
        task: torch_geometric.loader.DataLoader(
            GraphDataset(list_samples(args.data_root, task, "valid")),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for task in args.train_tasks
    }
    heldout_loaders = {
        task: torch_geometric.loader.DataLoader(
            GraphDataset(list_samples(args.data_root, task, "test")),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for task in args.test_tasks
    }

    policy = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = Scheduler(
        optimizer, mode="min", patience=10, factor=0.2, verbose=True
    )
    process_kwargs = {
        "device": device,
        "pad_tensor": pad_tensor,
        "torch": torch,
        "F": F,
        "top_k": top_k,
        "entropy_bonus": args.entropy_bonus,
    }

    checkpoint = running_dir / "train_params.pkl"
    for epoch in range(args.max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n_layers = pretrain(policy, pretrain_loader, device)
            log(f"PRETRAINED {n_layers} LAYERS", logfile)
        else:
            train_sampler.set_epoch(epoch)
            train_loss, train_kacc, _ = process(
                policy, train_loader, optimizer=optimizer, **process_kwargs
            )
            log("POOLED TRAIN " + metric_text(train_loss, train_kacc, top_k), logfile)

        valid_results = evaluate_tasks(policy, valid_loaders, **process_kwargs)
        for task, (loss, kacc, _) in valid_results.items():
            log(f"VALID {task} " + metric_text(loss, kacc, top_k), logfile)

        # Equal task weighting: a large/easy task cannot dominate checkpoint selection.
        mean_valid_loss = float(np.mean([result[0] for result in valid_results.values()]))
        log(f"VALID MACRO LOSS: {mean_valid_loss:.6f}", logfile)
        scheduler.step(mean_valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), checkpoint)
            log("  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log("  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log("  20 epochs without improvement, early stopping", logfile)
            break

    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    log("LOADED BEST CHECKPOINT", logfile)
    final_valid = evaluate_tasks(policy, valid_loaders, **process_kwargs)
    for task, (loss, kacc, _) in final_valid.items():
        log(f"BEST VALID {task} " + metric_text(loss, kacc, top_k), logfile)

    # Held-out configurations are evaluated only after model selection.
    heldout_results = evaluate_tasks(policy, heldout_loaders, **process_kwargs)
    for task, (loss, kacc, _) in heldout_results.items():
        log(f"HELDOUT TEST {task} " + metric_text(loss, kacc, top_k), logfile)


if __name__ == "__main__":
    main()
