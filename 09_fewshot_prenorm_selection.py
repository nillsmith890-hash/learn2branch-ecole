"""Few-shot selection of target-domain PreNorm recalibration settings.

Candidate settings are selected using labeled support samples from each target
task's train split. Only the selected setting is evaluated on the independent
test/query split. GNN weights are never updated.
"""

import argparse
import collections
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


def candidate_settings():
    settings = collections.OrderedDict()
    settings["source"] = []
    settings["input_only"] = INPUT_LAYERS
    settings["convolution_only"] = CONVOLUTION_LAYERS
    for name in ALL_LAYERS:
        settings[f"only:{name}"] = [name]
    settings["all"] = ALL_LAYERS
    return settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a PreNorm adaptation using target support labels."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=pathlib.Path("data/samples_meta")
    )
    parser.add_argument("--support-size", type=int, default=100,
                        help="Support samples drawn per target task.")
    parser.add_argument("--support-seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--selection-metric", choices=["loss", "acc@1"],
                        default="loss")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    if len(set(args.support_seeds)) != len(args.support_seeds):
        parser.error("--support-seeds contains duplicates")
    if args.support_size <= 0:
        parser.error("--support-size must be positive")
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

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            variable_logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            logits = pad_tensor(variable_logits[batch.candidates], batch.nb_candidates)
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

    result = {"samples": int(n_samples), "loss": float(loss_sum / n_samples)}
    result.update({f"acc@{k}": float(value) for k, value in zip(top_k, kacc_sum / n_samples)})
    return result


def macro_results(per_task, top_k):
    keys = ["loss"] + [f"acc@{k}" for k in top_k]
    return {
        key: float(np.mean([result[key] for result in per_task.values()]))
        for key in keys
    }


def evaluate_tasks(policy, loaders, **kwargs):
    return {
        task: evaluate(policy, loader, **kwargs)
        for task, loader in loaders.items()
    }


def print_metrics(prefix, result, top_k):
    accuracy = " ".join(f"acc@{k}: {result[f'acc@{k}']:.3f}" for k in top_k)
    print(f"{prefix} loss: {result['loss']:.3f} {accuracy}", flush=True)


def summarize_runs(runs, tasks, top_k):
    summary = {"selection_counts": dict(collections.Counter(
        run["selected_setting"] for run in runs
    )), "query": {}}
    for task in tasks + ["macro"]:
        summary["query"][task] = {}
        keys = ["loss"] + [f"acc@{k}" for k in top_k]
        for key in keys:
            values = np.asarray([
                run["query_macro"][key] if task == "macro"
                else run["query"][task][key]
                for run in runs
            ], dtype=np.float64)
            summary["query"][task][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
    return summary


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
    settings = candidate_settings()
    train_files = {task: list_samples(args.data_root, task, "train") for task in args.tasks}
    test_files = {task: list_samples(args.data_root, task, "test") for task in args.tasks}
    for task, files in train_files.items():
        if args.support_size > len(files):
            raise ValueError(
                f"support size {args.support_size} exceeds {len(files)} files for {task}"
            )

    query_loaders = {
        task: torch_geometric.loader.DataLoader(
            GraphDataset(files), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.gpu != -1,
        )
        for task, files in test_files.items()
    }
    checkpoint_state = torch.load(args.checkpoint, map_location=device)

    def fresh_policy():
        policy = GNNPolicy().to(device)
        policy.load_state_dict(checkpoint_state)
        return policy

    probe = fresh_policy()
    actual_names = list(prenorm_modules(probe, PreNormLayer))
    del probe
    if actual_names != ALL_LAYERS:
        raise RuntimeError(
            f"model PreNorm layout changed; expected {ALL_LAYERS}, found {actual_names}"
        )

    common_eval_kwargs = {
        "device": device,
        "torch": torch,
        "functional": functional,
        "pad_tensor": pad_tensor,
        "top_k": top_k,
    }
    output = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "tasks": args.tasks,
        "device": device,
        "support_size_per_task": args.support_size,
        "support_seeds": args.support_seeds,
        "selection_metric": args.selection_metric,
        "candidate_settings": settings,
        "runs": [],
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device} (requested gpu {args.gpu})")
    print(f"target tasks: {args.tasks}")
    print(f"support size per task: {args.support_size}")
    print(f"support seeds: {args.support_seeds}")
    print(f"selection metric: {args.selection_metric}")

    # Query baseline is computed once and is not used for model selection.
    source_policy = fresh_policy()
    source_query = evaluate_tasks(source_policy, query_loaders, **common_eval_kwargs)
    source_query_macro = macro_results(source_query, top_k)
    del source_policy
    print("\n=== FIXED SOURCE QUERY BASELINE (NOT USED FOR SELECTION) ===")
    print_metrics("macro", source_query_macro, top_k)

    for support_seed in args.support_seeds:
        rng = np.random.RandomState(support_seed)
        torch.manual_seed(support_seed)
        support_files = {
            task: rng.choice(files, size=args.support_size, replace=False).tolist()
            for task, files in train_files.items()
        }
        support_loaders = {
            task: torch_geometric.loader.DataLoader(
                GraphDataset(files), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=args.gpu != -1,
            )
            for task, files in support_files.items()
        }
        calibration_files = [
            path for task in args.tasks for path in support_files[task]
        ]
        calibration_loader = torch_geometric.loader.DataLoader(
            GraphDataset(calibration_files),
            batch_size=args.pretrain_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.gpu != -1,
        )

        print(f"\n=== SUPPORT SEED {support_seed}: CANDIDATE SELECTION ===")
        candidates = {}
        for setting_name, selected_layers in settings.items():
            policy = fresh_policy()
            started = time.time()
            completed = []
            if selected_layers:
                completed = recalibrate(
                    policy, selected_layers, calibration_loader, device, PreNormLayer
                )
            support_result = evaluate_tasks(
                policy, support_loaders, **common_eval_kwargs
            )
            support_macro = macro_results(support_result, top_k)
            candidates[setting_name] = {
                "selected_layers": selected_layers,
                "completed_layers": completed,
                "seconds": time.time() - started,
                "support": support_result,
                "support_macro": support_macro,
            }
            print_metrics(setting_name, support_macro, top_k)
            del policy

        if args.selection_metric == "loss":
            selected_setting = min(
                candidates, key=lambda name: candidates[name]["support_macro"]["loss"]
            )
        else:
            selected_setting = max(
                candidates, key=lambda name: candidates[name]["support_macro"]["acc@1"]
            )

        selected_layers = settings[selected_setting]
        selected_policy = fresh_policy()
        if selected_layers:
            recalibrate(
                selected_policy, selected_layers, calibration_loader, device, PreNormLayer
            )
        query_result = evaluate_tasks(
            selected_policy, query_loaders, **common_eval_kwargs
        )
        query_macro = macro_results(query_result, top_k)
        del selected_policy

        print(f"SELECTED: {selected_setting}")
        print_metrics("QUERY macro", query_macro, top_k)
        output["runs"].append({
            "support_seed": support_seed,
            "support_files": support_files,
            "candidates": candidates,
            "selected_setting": selected_setting,
            "query": query_result,
            "query_macro": query_macro,
        })

    output["source_query"] = source_query
    output["source_query_macro"] = source_query_macro
    output["summary"] = summarize_runs(output["runs"], args.tasks, top_k)

    print("\n=== FINAL SUMMARY ===")
    print(f"selection counts: {output['summary']['selection_counts']}")
    macro_summary = output["summary"]["query"]["macro"]
    print("query macro mean +/- population std:")
    for key, stats in macro_summary.items():
        print(f"  {key}: {stats['mean']:.3f} +/- {stats['std']:.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(output, file, indent=2)
        print(f"results written to: {args.output}")


if __name__ == "__main__":
    main()
