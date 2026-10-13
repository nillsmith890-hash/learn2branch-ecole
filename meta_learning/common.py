"""Shared model and metric helpers for the Reptile pilot."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from utilities import GraphDataset, pad_tensor


TOP_K = (1, 3, 5, 10)


def load_batch(files, device):
    dataset = GraphDataset(files)
    graphs = [dataset[index] for index in range(len(dataset))]
    return torch_geometric.data.Batch.from_data_list(graphs).to(device)


def loss_for_batch(policy, batch, params=None):
    inputs = (
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
    )
    logits = policy(*inputs) if params is None else functional_call(policy, params, inputs)
    logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
    return F.cross_entropy(logits, batch.candidate_choices, reduction="mean")


def adapt(policy, support_batch, learning_rate, steps, max_grad_norm=10.0):
    params = OrderedDict(
        (name, value.detach().clone().requires_grad_(True))
        for name, value in policy.named_parameters()
    )
    for _ in range(steps):
        loss = loss_for_batch(policy, support_batch, params)
        gradients = torch.autograd.grad(loss, params.values())
        if max_grad_norm is not None:
            norm = torch.sqrt(sum((gradient ** 2).sum() for gradient in gradients))
            coefficient = torch.clamp(max_grad_norm / (norm + 1e-6), max=1.0)
            gradients = [gradient * coefficient for gradient in gradients]
        params = OrderedDict(
            (name, (value - learning_rate * gradient).detach().requires_grad_(True))
            for (name, value), gradient in zip(params.items(), gradients)
        )
    return params


def evaluate(policy, batch, params=None):
    inputs = (
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
    )
    with torch.no_grad():
        logits = policy(*inputs) if params is None else functional_call(policy, params, inputs)
        logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
        loss = F.cross_entropy(logits, batch.candidate_choices, reduction="mean")
        true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
        best_score = true_scores.max(dim=-1, keepdim=True).values
        accuracies = []
        for k in TOP_K:
            effective_k = min(k, logits.shape[-1])
            predictions = logits.topk(effective_k, dim=-1).indices
            predicted_scores = true_scores.gather(-1, predictions)
            accuracies.append(
                (predicted_scores == best_score).any(dim=-1).float().mean().item()
            )
    return {"loss": loss.item(), **{
        f"acc@{k}": value for k, value in zip(TOP_K, accuracies)
    }}


def calibrate_prenorm(policy, files, device, batch_size=16):
    loader = torch_geometric.loader.DataLoader(
        GraphDataset(files), batch_size=batch_size, shuffle=False
    )
    policy.pre_train_init()
    layers = 0
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
        if policy.pre_train_next() is None:
            break
        layers += 1
    return layers


def mean_metrics(results):
    return {
        key: float(np.mean([result[key] for result in results]))
        for key in results[0]
    }

