"""Run layer-group and individual-layer PreNorm recalibration ablations."""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np


INPUT_LAYERS = ["cons_embedding.0", "edge_embedding.0", "var_embedding.0"]
CONVOLUTION_LAYERS = [
    "conv_v_to_c.feature_module_final.0",
    "conv_v_to_c.post_conv_module.0",
    "conv_c_to_v.feature_module_final.0",
    "conv_c_to_v.post_conv_module.0",
]
ALL_LAYERS = INPUT_LAYERS + CONVOLUTION_LAYERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify which PreNorm layers help or harm target adaptation."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=pathlib.Path("data/samples_meta")
    )
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    if args.calibration_size <= 0:
        parser.error("--calibration-size must be positive")
    if args.batch_size <= 0 or args.pretrain_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def list_samples(root, task, split):
    directory = root / task / split
    files = sorted(str(path) for path in directory.glob("sample_*.pkl"))
    if not files:
        raise FileNotFoundError(f"no sample_*.pkl files found in {directory}")
    return files


def prenorm_modules(policy, layer_type):
    return {
        name: module
        for name, module in policy.named_modules()
        if isinstance(module, layer_type)
    }


def recalibrate(policy, selected_names, loader, device, layer_type):
    modules = prenorm_modules(policy, layer_type)
    missing = set(selected_names) - set(modules)
    if missing:
        raise RuntimeError(f"unknown PreNorm layer names: {sorted(missing)}")

    for name in selected_names:
        modules[name].start_updates()

    completed = []
    while True:
        for batch in loader:
            batch = batch.to(device)
            if not policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            ):
                break
        stopped = policy.pre_train_next()
        if stopped is None:
            break
        completed.append(next(name for name, module in modules.items() if module is stopped))
    return completed


def evaluate(policy, loader, device, torch, functional, pad_tensor, top_k):
    policy.eval()
    loss_sum = 0.0
    kacc_sum = np.zeros(len(top_k), dtype=np.float64)
    n_samples = 0
    abs_logit_sum = 0.0
    n_logits = 0
    max_abs_logit = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            variable_logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            candidate_logits = variable_logits[batch.candidates]
            abs_logit_sum += candidate_logits.abs().sum().item()
            n_logits += candidate_logits.numel()
            max_abs_logit = max(max_abs_logit, candidate_logits.abs().max().item())

            logits = pad_tensor(candidate_logits, batch.nb_candidates)
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
                batch_kacc.append(
                    (predicted_scores == true_best).any(dim=-1).float().mean().item()
                )

            count = batch.num_graphs
            loss_sum += loss.item() * count
            kacc_sum += np.asarray(batch_kacc) * count
            n_samples += count

    result = {
        "samples": n_samples,
        "loss": loss_sum / n_samples,
        "mean_abs_logit": abs_logit_sum / n_logits,
        "max_abs_logit": max_abs_logit,
    }
    result.update({f"acc@{k}": float(value) for k, value in zip(top_k, kacc_sum / n_samples)})
    return result


def print_result(task, result, top_k):
    accuracy = " ".join(f"acc@{k}: {result[f'acc@{k}']:.3f}" for k in top_k)
    print(
        f"{task} samples: {result['samples']} loss: {result['loss']:.3f} "
        f"{accuracy} mean|logit|: {result['mean_abs_logit']:.3f} "
        f"max|logit|: {result['max_abs_logit']:.3f}",
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
    from model import GNNPolicy, PreNormLayer

    top_k = [1, 3, 5, 10]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    train_files = {task: list_samples(args.data_root, task, "train") for task in args.tasks}
    test_files = {task: list_samples(args.data_root, task, "test") for task in args.tasks}

    selected_files = []
    selected_counts = {}
    for task, files in train_files.items():
        count = min(args.calibration_size, len(files))
        selected_files.extend(rng.choice(files, size=count, replace=False).tolist())
        selected_counts[task] = count

    calibration_loader = torch_geometric.loader.DataLoader(
        GraphDataset(selected_files),
        batch_size=args.pretrain_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.gpu != -1,
    )
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

    settings = {
        "source": [],
        "input_only": INPUT_LAYERS,
        "convolution_only": CONVOLUTION_LAYERS,
    }
    settings.update({f"only:{name}": [name] for name in ALL_LAYERS})
    settings["all"] = ALL_LAYERS

    probe = fresh_policy()
    actual_names = list(prenorm_modules(probe, PreNormLayer))
    del probe
    if actual_names != ALL_LAYERS:
        raise RuntimeError(
            "model PreNorm layout changed; expected "
            f"{ALL_LAYERS}, found {actual_names}"
        )

    output = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "tasks": args.tasks,
        "seed": args.seed,
        "device": device,
        "calibration_samples_per_task": selected_counts,
        "layer_names": ALL_LAYERS,
        "settings": {},
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device} (requested gpu {args.gpu})")
    print(f"target tasks: {args.tasks}")
    print(f"calibration samples per task: {selected_counts}")
    print(f"PreNorm layers: {ALL_LAYERS}")

    for setting_name, selected_layers in settings.items():
        policy = fresh_policy()
        started = time.time()
        completed = []
        if selected_layers:
            completed = recalibrate(
                policy, selected_layers, calibration_loader, device, PreNormLayer
            )
        elapsed = time.time() - started

        print(f"\n=== SETTING: {setting_name} ===")
        print(f"recalibrated layers: {completed}; time: {elapsed:.1f}s")
        setting_result = {
            "selected_layers": selected_layers,
            "completed_layers": completed,
            "recalibration_seconds": elapsed,
            "test": {},
        }
        for task, loader in test_loaders.items():
            result = evaluate(
                policy, loader, device, torch, functional, pad_tensor, top_k
            )
            setting_result["test"][task] = result
            print_result(task, result, top_k)
        output["settings"][setting_name] = setting_result
        del policy

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(output, file, indent=2)
        print(f"\nresults written to: {args.output}")


if __name__ == "__main__":
    main()
