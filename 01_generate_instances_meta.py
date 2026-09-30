"""Generate a small, configurable MILP task for meta-learning experiments.

The original ``01_generate_instances.py`` keeps reproducing the data layout
from the Learn2Branch experiments.  This companion script deliberately writes
to ``data/instances_meta`` so pilot tasks cannot overwrite that data.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_instance_generators():
    source = Path(__file__).with_name("01_generate_instances.py")
    spec = importlib.util.spec_from_file_location("l2b_instance_generators", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_number(value):
    return f"{value:g}"


def task_name(args):
    if args.problem == "cauctions":
        return f"{args.n_items}_{args.n_bids}"
    if args.problem == "setcover":
        return f"{args.n_rows}r_{args.n_cols}c_{format_number(args.density)}d"
    if args.problem == "indset":
        return f"{args.n_nodes}_{args.affinity}"
    if args.problem == "facilities":
        return f"{args.n_customers}_{args.n_facilities}_{format_number(args.ratio)}"
    raise ValueError(args.problem)


def validate_args(args):
    for split, count in vars(args).items():
        if split.endswith("_instances") and count < 1:
            raise ValueError(f"--{split.replace('_', '-')} must be positive")

    if args.problem == "cauctions":
        if args.n_items < 2 or args.n_bids < 1:
            raise ValueError("cauctions requires at least 2 items and 1 bid")
    elif args.problem == "setcover":
        if args.n_rows < 1 or args.n_cols < 1 or not 0 < args.density <= 1:
            raise ValueError("setcover dimensions must be positive and density in (0, 1]")
        nnz = int(args.n_rows * args.n_cols * args.density)
        if nnz < args.n_rows or nnz < 2 * args.n_cols:
            raise ValueError("density is too small to cover every row and every column twice")
    elif args.problem == "indset":
        if not 1 <= args.affinity < args.n_nodes:
            raise ValueError("indset requires 1 <= affinity < n-nodes")
    elif args.problem == "facilities":
        if args.n_customers < 1 or args.n_facilities < 1 or args.ratio <= 0:
            raise ValueError("facility dimensions and ratio must be positive")


def generate_one(generators, args, rng, filename):
    if args.problem == "cauctions":
        generators.generate_cauctions(
            rng,
            str(filename),
            n_items=args.n_items,
            n_bids=args.n_bids,
            add_item_prob=args.add_item_prob,
        )
    elif args.problem == "setcover":
        generators.generate_setcover(
            nrows=args.n_rows,
            ncols=args.n_cols,
            density=args.density,
            filename=str(filename),
            rng=rng,
            max_coef=args.max_coef,
        )
    elif args.problem == "indset":
        graph = generators.Graph.barabasi_albert(args.n_nodes, args.affinity, rng)
        generators.generate_indset(graph, str(filename))
    elif args.problem == "facilities":
        # The upstream implementation currently reads its module-level ``rng``.
        # Set it as well as passing it explicitly, preserving reproducibility.
        generators.rng = rng
        generators.generate_capacited_facility_location(
            rng,
            str(filename),
            n_customers=args.n_customers,
            n_facilities=args.n_facilities,
            ratio=args.ratio,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate one configurable meta-learning MILP task."
    )
    parser.add_argument("problem", choices=["setcover", "cauctions", "facilities", "indset"])
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--output-root", default="data/instances_meta")
    parser.add_argument("--train-instances", type=int, default=100)
    parser.add_argument("--valid-instances", type=int, default=30)
    parser.add_argument("--test-instances", type=int, default=30)

    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--n-bids", type=int, default=500)
    parser.add_argument("--add-item-prob", type=float, default=0.7)

    parser.add_argument("--n-rows", type=int, default=500)
    parser.add_argument("--n-cols", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--max-coef", type=int, default=100)

    parser.add_argument("--n-nodes", type=int, default=500)
    parser.add_argument("--affinity", type=int, default=4)

    parser.add_argument("--n-customers", type=int, default=100)
    parser.add_argument("--n-facilities", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=5)
    args = parser.parse_args()
    validate_args(args)

    name = task_name(args)
    task_dir = Path(args.output_root) / args.problem / name
    split_sizes = {
        "train": args.train_instances,
        "valid": args.valid_instances,
        "test": args.test_instances,
    }

    existing = [task_dir / split for split in split_sizes if (task_dir / split).exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing task directories: {paths}"
        )

    generators = load_instance_generators()
    rng = np.random.RandomState(args.seed)
    print(f"task: {args.problem}/{name}")
    print(f"seed: {args.seed}")

    for split, count in split_sizes.items():
        split_dir = task_dir / split
        split_dir.mkdir(parents=True)
        print(f"{split}: generating {count} instances in {split_dir}")
        for index in range(1, count + 1):
            filename = split_dir / f"instance_{index}.lp"
            generate_one(generators, args, rng, filename)
            print(f"  {index} / {count}: {filename}")

    print(f"done: {task_dir}")


if __name__ == "__main__":
    main()
