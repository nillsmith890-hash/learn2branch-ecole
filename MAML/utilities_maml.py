from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch_geometric

try:
    from torch.func import functional_call
except ImportError:  # torch < 2.0
    from torch.nn.utils.stateless import functional_call

from utilities import pad_tensor, GraphDataset


def load_task_batch(files, device):
    """Loads and collates a list of sample files into one task batch (support or query)."""
    dataset = GraphDataset(files)
    batch = torch_geometric.data.Batch.from_data_list([dataset[i] for i in range(len(dataset))])
    return batch.to(device)


def compute_loss(policy, batch, params=None):
    """
    Cross-entropy loss of `policy` on `batch`. If `params` is given (an
    OrderedDict of tensors matching policy.named_parameters()), runs a
    *functional* forward pass against those weights instead of the module's
    own parameters -- this is what lets the MAML inner loop adapt a task-local
    copy of the weights without ever touching `policy` itself.
    """
    args = (batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
    if params is None:
        logits = policy(*args)
    else:
        logits = functional_call(policy, params, args)

    logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
    return F.cross_entropy(logits, batch.candidate_choices, reduction='mean')


def inner_loop_adapt(policy, support_batch, inner_lr, inner_steps, first_order=True):
    """
    Adapts a *copy* of policy's parameters via inner_steps plain-SGD steps on
    the same support batch: theta -> theta'_i. Returns the adapted weights as
    an OrderedDict; policy.parameters() themselves are never modified, so
    tasks don't need to snapshot/restore theta around each other (unlike the
    TF1.x version, which had to because it updated the live model in place).

    first_order=True (FOMAML): each step detaches theta'_i from the graph, so
    it only tracks gradients within that one step. Cheap, and what the
    original TF1.x code did (it had no easy way to do otherwise).
    first_order=False: keeps the full computation graph back to policy's
    real parameters across all inner_steps, so a later .backward() on the
    query loss differentiates through the inner loop itself (true MAML,
    second-order). Needs more memory/compute, but PyTorch makes it easy where
    TF1.x eager did not -- hence the flag.
    """
    params = OrderedDict(policy.named_parameters())
    for _ in range(inner_steps):
        loss = compute_loss(policy, support_batch, params=params)
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

        if first_order:
            params = OrderedDict(
                (name, (p - inner_lr * g).detach().requires_grad_(True))
                for (name, p), g in zip(params.items(), grads)
            )
        else:
            params = OrderedDict(
                (name, p - inner_lr * g)
                for (name, p), g in zip(params.items(), grads)
            )

    return params
