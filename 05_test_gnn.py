import argparse
import os
import pathlib

import numpy as np


def evaluate(policy, data_loader, device, pad_tensor, torch, functional, top_k):
    mean_loss = 0.0
    mean_kacc = np.zeros(len(top_k), dtype=np.float64)
    n_samples = 0

    policy.eval()
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

            true_scores = pad_tensor(
                batch.candidate_scores, batch.nb_candidates
            )
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            batch_kacc = []
            for k in top_k:
                # A graph with fewer than k candidates necessarily has its
                # best-scoring candidate in its top-k set.  For the remaining
                # graphs, ignore padded entries and apply the usual metric.
                effective_k = min(k, logits.size(-1))
                pred_top_k = logits.topk(effective_k, dim=-1).indices
                pred_scores = true_scores.gather(-1, pred_top_k)
                correct = (pred_scores == true_bestscore).any(dim=-1)
                correct = correct | (batch.nb_candidates <= k)
                batch_kacc.append(correct.float().mean().item())

            batch_size = batch.num_graphs
            mean_loss += loss.item() * batch_size
            mean_kacc += np.asarray(batch_kacc) * batch_size
            n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError("No test samples were found")

    return mean_loss / n_samples, mean_kacc / n_samples, n_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN on offline test samples."
    )
    parser.add_argument(
        "problem",
        choices=["setcover", "cauctions", "facilities", "indset", "mknapsack"],
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    problem_folders = {
        "setcover": "setcover/500r_1000c_0.05d",
        "cauctions": "cauctions/100_500",
        "facilities": "facilities/100_100_5",
        "indset": "indset/500_4",
        "mknapsack": "mknapsack/100_6",
    }

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda:0"

    import torch
    import torch.nn.functional as functional
    import torch_geometric

    from model.model import GNNPolicy
    from utilities import GraphDataset, pad_tensor

    test_dir = pathlib.Path("data/samples") / problem_folders[args.problem] / "test"
    test_files = sorted(test_dir.glob("sample_*.pkl"))
    checkpoint = pathlib.Path("model") / args.problem / str(args.seed) / "train_params.pkl"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_files:
        raise FileNotFoundError(f"No sample_*.pkl files found in: {test_dir}")

    print(f"problem: {args.problem}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint}")
    print(f"test samples: {len(test_files)}")

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))

    test_data = GraphDataset([str(path) for path in test_files])
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.gpu != -1,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    test_loader = torch_geometric.loader.DataLoader(test_data, **loader_kwargs)
    top_k = [1, 3, 5, 10]
    loss, kacc, n_samples = evaluate(
        policy,
        test_loader,
        device,
        pad_tensor,
        torch,
        functional,
        top_k,
    )

    metrics = " ".join(
        f"acc@{k}: {accuracy:.3f}" for k, accuracy in zip(top_k, kacc)
    )
    print(f"TEST SAMPLES: {n_samples}")
    print(f"TEST LOSS: {loss:.3f} {metrics}")
