import os
import sys
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates a train_maml.py checkpoint: for each problem, repeatedly '
                     'draws a fresh (support, query) task from the given split, adapts on '
                     'support (same inner loop as training), and scores on query.')
    parser.add_argument(
        '--problems',
        nargs='+',
        default=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
        help='MILP problem domains to evaluate on (each evaluated independently).',
    )
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='CUDA GPU id (-1 for CPU).')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help="path to a best_params.pkl from train_maml.py; "
                              "defaults to MAML/trained_models/<problems>/<seed>/best_params.pkl")
    parser.add_argument('--k_support', type=int, default=32)
    parser.add_argument('--k_query', type=int, default=32)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--n_eval_tasks', type=int, default=100,
                         help='number of independent (support, query) adaptation episodes per problem')
    parser.add_argument('--data_root', type=str, default=None, help='defaults to data/samples')
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"MAML/trained_models/{'_'.join(args.problems)}/{args.seed}/best_params.pkl"

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = 'cuda:0'

    import torch

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    L2B_DIR = os.path.dirname(CURRENT_DIR)
    sys.path.insert(0, CURRENT_DIR)
    sys.path.insert(0, L2B_DIR)

    from task_sampler import TaskSampler
    from utilities_maml import load_task_batch, inner_loop_adapt, evaluate_batch

    sys.path.insert(0, os.path.join(L2B_DIR, 'model'))
    from model import GNNPolicy
    del sys.path[0]

    print(f"checkpoint: {checkpoint}")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root or os.path.join(L2B_DIR, 'data', 'samples')

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))

    top_k = (1, 3, 5, 10)
    results = []

    for problem in args.problems:
        sampler = TaskSampler(data_root, [problem], args.split, args.k_support, args.k_query, seed=args.seed)

        losses, kaccs = [], []
        zs_losses, zs_kaccs = [], []  # zero-shot: same theta, no inner-loop adaptation at all
        for i in range(args.n_eval_tasks):
            _, support_files, query_files = sampler.sample_task(problem)

            support_batch = load_task_batch(support_files, device)
            adapted_params = inner_loop_adapt(policy, support_batch, args.inner_lr, args.inner_steps, first_order=True)

            query_batch = load_task_batch(query_files, device)
            loss, kacc = evaluate_batch(policy, query_batch, params=adapted_params, top_k=top_k)
            losses.append(loss)
            kaccs.append(kacc)

            zs_loss, zs_kacc = evaluate_batch(policy, query_batch, params=None, top_k=top_k)
            zs_losses.append(zs_loss)
            zs_kaccs.append(zs_kacc)

            if (i + 1) % 10 == 0:
                print(f"  {problem}: {i + 1}/{args.n_eval_tasks} tasks done")

        loss_mean, loss_std = float(np.mean(losses)), float(np.std(losses))
        kacc_mean = np.mean(kaccs, axis=0)
        zs_loss_mean = float(np.mean(zs_losses))
        zs_kacc_mean = np.mean(zs_kaccs, axis=0)

        results.append((problem, loss_mean, loss_std, kacc_mean, zs_loss_mean, zs_kacc_mean))

        acc_str = ' '.join(f'acc@{k}: {a:.3f}' for k, a in zip(top_k, kacc_mean))
        zs_acc_str = ' '.join(f'acc@{k}: {a:.3f}' for k, a in zip(top_k, zs_kacc_mean))
        print(f"{problem} [{args.split}, n={args.n_eval_tasks}] "
              f"few-shot (adapted)  loss: {loss_mean:.4f} +/- {loss_std:.4f}  {acc_str}")
        print(f"{problem} [{args.split}, n={args.n_eval_tasks}] "
              f"zero-shot (no adapt) loss: {zs_loss_mean:.4f}  {zs_acc_str}")

    results_path = os.path.join(os.path.dirname(checkpoint), f'eval_{args.split}.csv')
    with open(results_path, 'w') as f:
        f.write('problem,split,n_eval_tasks,k_support,k_query,inner_steps,inner_lr,'
                f"loss_mean,loss_std,{','.join(f'acc@{k}' for k in top_k)},"
                f"zeroshot_loss_mean,{','.join(f'zeroshot_acc@{k}' for k in top_k)}\n")
        for problem, loss_mean, loss_std, kacc_mean, zs_loss_mean, zs_kacc_mean in results:
            acc_cols = ','.join(f'{a:.6f}' for a in kacc_mean)
            zs_acc_cols = ','.join(f'{a:.6f}' for a in zs_kacc_mean)
            f.write(f"{problem},{args.split},{args.n_eval_tasks},{args.k_support},{args.k_query},"
                    f"{args.inner_steps},{args.inner_lr},{loss_mean:.6f},{loss_std:.6f},{acc_cols},"
                    f"{zs_loss_mean:.6f},{zs_acc_cols}\n")
    print(f"wrote {results_path}")
