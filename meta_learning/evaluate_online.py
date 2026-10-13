"""Run paired SCIP solves with adapted GNNs on fresh MILP instances."""

import argparse
import csv
import os
import re
import statistics
import sys
import time
from pathlib import Path

def parse_checkpoints(items):
    checkpoints = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError("--checkpoint must be TASK:LABEL=PATH")
        task_label, path = item.split("=", 1)
        task, label = task_label.rsplit(":", 1)
        if len(task.split("/")) != 2:
            raise ValueError(f"invalid checkpoint task: {task!r}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label):
            raise ValueError(f"invalid checkpoint label: {label!r}")
        if (task, label) in checkpoints or label == "relpscost":
            raise ValueError(f"duplicate or reserved checkpoint label: {label!r}")
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints[(task, label)] = checkpoint
    return checkpoints


def numeric_instance_key(path):
    match = re.fullmatch(r"instance_(\d+)\.lp", path.name)
    if match is None:
        raise ValueError(f"unexpected LP filename: {path}")
    return int(match.group(1))


def collect_instances(root, tasks, limit):
    instances = []
    for task in tasks:
        directory = root / task / "test"
        paths = sorted(directory.glob("instance_*.lp"), key=numeric_instance_key)
        if len(paths) < limit:
            raise ValueError(f"{directory} has {len(paths)} LP files, need {limit}")
        instances.extend((task, path) for path in paths[:limit])
    return instances


def solve_gnn(ecole, torch, model, path, device, seed, scip_parameters):
    import numpy as np

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=scip_parameters,
    )
    env.seed(seed)
    start = time.perf_counter()
    observation, action_set, _, done, _ = env.reset(str(path))
    while not done:
        features = (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32))
                .view(-1, 1).to(device),
            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
        )
        with torch.no_grad():
            logits = model(*features)
            # Ecole can return uint64 candidates, which PyTorch cannot ingest
            # directly as an index tensor. Variable indices fit in int64.
            candidate_indices = torch.from_numpy(
                action_set.astype(np.int64, copy=False)
            ).to(device)
            choice = int(torch.argmax(logits[candidate_indices]).item())
        observation, action_set, _, done, _ = env.step(int(action_set[choice]))
    return env, time.perf_counter() - start


def solve_internal(ecole, path, seed, scip_parameters):
    env = ecole.environment.Configuring(
        scip_params={**scip_parameters, "branching/relpscost/priority": 9999999},
    )
    env.seed(seed)
    start = time.perf_counter()
    env.reset(str(path))
    env.step({})
    return env, time.perf_counter() - start


def summarize_pairs(rows, tasks):
    """Summarize matched model solves without treating timeouts as solves."""
    by_pair = {}
    for row in rows:
        if row["policy"] in ("lodo", "reptile"):
            key = (row["task"], row["instance"], row["seed"])
            by_pair.setdefault(key, {})[row["policy"]] = row
    summaries = []
    for task in tasks:
        pairs = [policies for (name, _, _), policies in by_pair.items() if name == task]
        both_solved = [
            pair for pair in pairs
            if pair["lodo"]["solved"] and pair["reptile"]["solved"]
        ]
        both_unsolved = [
            pair for pair in pairs
            if not pair["lodo"]["solved"] and not pair["reptile"]["solved"]
        ]
        summaries.append({
            "task": task,
            "pairs": len(pairs),
            "lodo_solved": sum(pair["lodo"]["solved"] for pair in pairs),
            "reptile_solved": sum(pair["reptile"]["solved"] for pair in pairs),
            "both_solved": len(both_solved),
            "reptile_faster_both_solved": sum(
                pair["reptile"]["walltime"] < pair["lodo"]["walltime"]
                for pair in both_solved
            ),
            "median_reptile_minus_lodo_walltime_both_solved": (
                statistics.median(
                    pair["reptile"]["walltime"] - pair["lodo"]["walltime"]
                    for pair in both_solved
                ) if both_solved else ""
            ),
            "median_reptile_minus_lodo_nodes_both_solved": (
                statistics.median(
                    pair["reptile"]["nnodes"] - pair["lodo"]["nnodes"]
                    for pair in both_solved
                ) if both_solved else ""
            ),
            "both_unsolved": len(both_unsolved),
            "reptile_smaller_gap_both_unsolved": sum(
                pair["reptile"]["gap"] < pair["lodo"]["gap"]
                for pair in both_unsolved
            ),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="TASK:LABEL=PATH; repeat per task/model")
    parser.add_argument("--instances-root", type=Path, default=Path("data/instances_online"))
    parser.add_argument("--limit-instances", type=int, default=3)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--include-relpscost", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID, or -1 for CPU (default)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit_instances <= 0 or args.time_limit <= 0:
        parser.error("--limit-instances and --time-limit must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--task contains duplicates")
    if any(len(task.split("/")) != 2 for task in args.tasks):
        parser.error("each --task must be domain/configuration")
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    for output in (args.output, summary_path):
        if output.exists():
            parser.error(f"refusing to overwrite existing output: {output}")
    try:
        checkpoints = parse_checkpoints(args.checkpoint)
        instances = collect_instances(args.instances_root, args.tasks, args.limit_instances)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if set(task for task, _ in checkpoints) != set(args.tasks):
        parser.error("checkpoint tasks must exactly match --task values")
    for task in args.tasks:
        if {label for checkpoint_task, label in checkpoints if checkpoint_task == task} != {"lodo", "reptile"}:
            parser.error(f"{task} requires exactly lodo and reptile checkpoints")
    seeds = args.seeds if args.seeds is not None else [0]
    if len(set(seeds)) != len(seeds):
        parser.error("--seed contains duplicates")

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == -1 else str(args.gpu)
    device = "cpu" if args.gpu == -1 else "cuda:0"
    import ecole
    import torch
    try:
        import pyscipopt
    except ModuleNotFoundError as exc:
        if exc.name != "pyscipopt":
            raise
        parser.error(
            "PySCIPOpt is required for SCIP result metrics. Install it in the "
            "same environment as Ecole (for a conda-forge environment: "
            "conda install -c conda-forge pyscipopt)."
        )
    if not hasattr(pyscipopt, "Model"):
        parser.error("a real PySCIPOpt installation is required; pyscipopt_stub is insufficient")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "model"))
    from model import GNNPolicy

    models = {}
    for key, checkpoint in checkpoints.items():
        model = GNNPolicy().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        models[key] = model

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": args.time_limit,
        "timing/clocktype": 1,
        "branching/vanillafullstrong/idempotent": True,
    }
    fields = [
        "task", "instance", "policy", "seed", "status", "solved", "nnodes",
        "nlps", "scip_time", "walltime", "gap", "primal_bound", "dual_bound",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task, path in instances:
            for seed in seeds:
                # Alternate execution order to reduce a systematic first-run bias.
                labels = ["lodo", "reptile"]
                if numeric_instance_key(path) % 2 == 0:
                    labels.reverse()
                if args.include_relpscost:
                    labels.append("relpscost")
                for label in labels:
                    if label == "relpscost":
                        env, walltime = solve_internal(ecole, path, seed, scip_parameters)
                    else:
                        env, walltime = solve_gnn(
                            ecole, torch, models[(task, label)], path, device, seed,
                            scip_parameters,
                        )
                    scip = env.model.as_pyscipopt()
                    status = str(scip.getStatus())
                    row = {
                        "task": task,
                        "instance": str(path),
                        "policy": label,
                        "seed": seed,
                        "status": status,
                        "solved": int(status == "optimal"),
                        "nnodes": scip.getNNodes(),
                        "nlps": scip.getNLPs(),
                        "scip_time": scip.getSolvingTime(),
                        "walltime": walltime,
                        "gap": scip.getGap(),
                        "primal_bound": scip.getPrimalbound(),
                        "dual_bound": scip.getDualbound(),
                    }
                    writer.writerow(row)
                    rows.append(row)
                    stream.flush()
                    print(
                        f"{task} {path.name} seed={seed} {label}: "
                        f"{status}, {row['nnodes']} nodes, "
                        f"{row['scip_time']:.2f}s SCIP, {walltime:.2f}s wall",
                        flush=True,
                    )
    summaries = summarize_pairs(rows, args.tasks)
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(
            f"{summary['task']}: solved LODO {summary['lodo_solved']}/"
            f"{summary['pairs']}, Reptile {summary['reptile_solved']}/"
            f"{summary['pairs']}; Reptile faster on "
            f"{summary['reptile_faster_both_solved']}/"
            f"{summary['both_solved']} jointly solved pairs",
            flush=True,
        )
    print(f"results: {args.output}")
    print(f"paired summary: {summary_path}")


if __name__ == "__main__":
    main()
