import os
import sys
import argparse

import numpy as np


def meta_train_step(policy, outer_optimizer, tasks, inner_lr, inner_steps, device, first_order=True):
    query_losses = []

    if first_order:
        # FOMAML: accumulate the gradient w.r.t. each task's adapted (and
        # detached) weights theta'_i, then apply that sum directly to
        # policy's real parameters theta. This approximates the true
        # meta-gradient by assuming d theta'_i / d theta ~= I.
        accumulated_grads = None
        for problem, support_files, query_files in tasks:
            support_batch = load_task_batch(support_files, device)
            adapted_params = inner_loop_adapt(policy, support_batch, inner_lr, inner_steps, first_order=True)

            query_batch = load_task_batch(query_files, device)
            q_loss = compute_loss(policy, query_batch, params=adapted_params)
            grads = torch.autograd.grad(q_loss, adapted_params.values())
            query_losses.append(q_loss.item())

            if accumulated_grads is None:
                accumulated_grads = [g.detach().clone() for g in grads]
            else:
                accumulated_grads = [ag + g.detach() for ag, g in zip(accumulated_grads, grads)]

        outer_optimizer.zero_grad()
        for p, g in zip(policy.parameters(), accumulated_grads):
            p.grad = g / len(tasks)
        outer_optimizer.step()

    else:
        # True (second-order) MAML: each task's adapted weights keep their
        # graph back to policy's real parameters (inner_loop_adapt was called
        # with first_order=False), so summing the query losses and calling
        # .backward() once differentiates through the inner loop for every
        # task and accumulates the exact meta-gradient on policy.parameters().
        meta_loss = 0.0
        for problem, support_files, query_files in tasks:
            support_batch = load_task_batch(support_files, device)
            adapted_params = inner_loop_adapt(policy, support_batch, inner_lr, inner_steps, first_order=False)

            query_batch = load_task_batch(query_files, device)
            q_loss = compute_loss(policy, query_batch, params=adapted_params)
            query_losses.append(q_loss.item())
            meta_loss = meta_loss + q_loss

        meta_loss = meta_loss / len(tasks)
        outer_optimizer.zero_grad()
        meta_loss.backward()
        outer_optimizer.step()

    return float(np.mean(query_losses))


def pretrain(policy, pretrain_files, device, batch_size=128):
    """
    Calibrates every PreNormLayer's shift/scale via one pass of online
    mean/variance stats collection (BaseModel.pre_train_init/pre_train/
    pre_train_next), exactly like 03_train_gnn.py does before real training.

    Skipping this leaves every PreNormLayer at its __init__ default (shift=0,
    scale=1, i.e. identity), so raw, unnormalized MILP features go straight
    into the GCN's Linear+ReLU stack. Combined with repeated inner-loop SGD
    steps that's enough to blow activations up to inf/nan within a handful
    of meta-iterations on real (larger-magnitude) data.
    """
    pretrain_data = GraphDataset(pretrain_files)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, batch_size, shuffle=False)

    policy.pre_train_init()
    n_layers = 0
    while True:
        for batch in pretrain_loader:
            batch = batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                break
        if policy.pre_train_next() is None:
            break
        n_layers += 1
    return n_layers


def meta_valid(policy, tasks, inner_lr, inner_steps, device):
    query_losses = []

    for problem, support_files, query_files in tasks:
        support_batch = load_task_batch(support_files, device)
        # first_order=True here is just the cheaper inner-loop mode (no
        # graph kept past each step); meta_valid never calls .backward(), so
        # it wouldn't matter for correctness whether meta-training is
        # running in first- or second-order mode.
        adapted_params = inner_loop_adapt(policy, support_batch, inner_lr, inner_steps, first_order=True)

        query_batch = load_task_batch(query_files, device)
        with torch.no_grad():
            q_loss = compute_loss(policy, query_batch, params=adapted_params)
        query_losses.append(q_loss.item())

    return float(np.mean(query_losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problems',
        nargs='+',
        default=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
        help='MILP problem domains to treat as MAML tasks.',
    )
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='CUDA GPU id (-1 for CPU).')
    parser.add_argument('--meta_iterations', type=int, default=1000)
    parser.add_argument('--meta_batch_size', type=int, default=4, help='number of tasks per meta-update')
    parser.add_argument('--k_support', type=int, default=32)
    parser.add_argument('--k_query', type=int, default=32)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--valid_every', type=int, default=20)
    parser.add_argument('--data_root', type=str, default=None, help='defaults to data/samples')
    parser.add_argument('--first_order', dest='first_order', action='store_true', default=True,
                         help='use first-order MAML / FOMAML (default; cheaper, matches the old TF1.x behavior)')
    parser.add_argument('--second_order', dest='first_order', action='store_false',
                         help='use true second-order MAML (differentiates through the inner loop; more memory/compute)')
    args = parser.parse_args()

    running_dir = f"MAML/trained_models/{'_'.join(args.problems)}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)
    logfile = os.path.join(running_dir, 'log.txt')

    ### PYTORCH SETUP ###
    # CUDA_VISIBLE_DEVICES must be set before the first `import torch`
    # anywhere in the process (utilities.py imports torch at module scope
    # too), so all torch-dependent imports are deferred to here, matching
    # 03_train_gnn.py's convention.
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = 'cuda:0'

    import torch
    import torch_geometric

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    L2B_DIR = os.path.dirname(CURRENT_DIR)
    sys.path.insert(0, CURRENT_DIR)
    sys.path.insert(0, L2B_DIR)

    from utilities import log, valid_seed, GraphDataset
    from task_sampler import TaskSampler
    from utilities_maml import load_task_batch, compute_loss, inner_loop_adapt

    sys.path.insert(0, os.path.join(L2B_DIR, 'model'))
    from model import GNNPolicy
    del sys.path[0]

    args.seed = valid_seed(args.seed)

    for k, v in vars(args).items():
        log(f"{k}: {v}", logfile)
    log(f"device: {device}", logfile)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root or os.path.join(L2B_DIR, 'data', 'samples')
    train_sampler = TaskSampler(data_root, args.problems, 'train', args.k_support, args.k_query, seed=args.seed)
    valid_sampler = TaskSampler(data_root, args.problems, 'valid', args.k_support, args.k_query, seed=args.seed + 1)

    policy = GNNPolicy().to(device)

    pretrain_files = []
    for problem in args.problems:
        pretrain_files += [f for i, f in enumerate(train_sampler.task_files[problem]) if i % 10 == 0]
    n_layers = pretrain(policy, pretrain_files, device)
    log(f"pretrained {n_layers} PreNormLayers on {len(pretrain_files)} samples", logfile)

    outer_optimizer = torch.optim.Adam(policy.parameters(), lr=args.outer_lr)

    best_valid_loss = np.inf
    for it in range(args.meta_iterations):
        tasks = train_sampler.sample_meta_batch(args.meta_batch_size)
        meta_loss = meta_train_step(
            policy, outer_optimizer, tasks, args.inner_lr, args.inner_steps, device,
            first_order=args.first_order)
        log(f"[iter {it}] meta-train query loss: {meta_loss:.4f}", logfile)

        if (it + 1) % args.valid_every == 0:
            valid_tasks = valid_sampler.sample_meta_batch(args.meta_batch_size)
            valid_loss = meta_valid(policy, valid_tasks, args.inner_lr, args.inner_steps, device)
            log(f"[iter {it}] meta-valid query loss (post-adapt): {valid_loss:.4f}", logfile)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(policy.state_dict(), os.path.join(running_dir, 'best_params.pkl'))
                log("  best meta-model so far", logfile)

    log("done.", logfile)
