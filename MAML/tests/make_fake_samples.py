import gzip
import os
import pickle

import numpy as np

# must match GNNPolicy's expected dims (learn2branch-ecole/model/model.py)
CONS_NFEATS = 5
EDGE_NFEATS = 1
VAR_NFEATS = 19

# same layout as 03_train_gnn.py's problem_folders / MAML/task_sampler.py's PROBLEM_FOLDERS
PROBLEM_FOLDERS = {
    'setcover': 'setcover/500r_1000c_0.05d',
    'cauctions': 'cauctions/100_500',
    'facilities': 'facilities/100_100_5',
    'indset': 'indset/500_4',
    'mknapsack': 'mknapsack/100_6',
}


def make_one_sample(rng, n_cons=6, n_vars=10, n_cands=4):
    """
    Mimics the (node_observation, action, action_set, scores) tuple that
    02_generate_dataset.py pickles per sample, with random values, so
    utilities.GraphDataset can load it without a real ecole/SCIP episode.
    node_observation matches ecole.observation.NodeBipartite's layout:
    (row_features, (edge_indices, edge_values), variable_features).
    """
    row_features = rng.randn(n_cons, CONS_NFEATS).astype(np.float32)

    n_edges = n_cons * 3
    edge_row = rng.randint(0, n_cons, size=n_edges)
    edge_col = rng.randint(0, n_vars, size=n_edges)
    edge_indices = np.vstack([edge_row, edge_col]).astype(np.int64)
    edge_values = rng.randn(n_edges).astype(np.float32)

    variable_features = rng.randn(n_vars, VAR_NFEATS).astype(np.float32)

    node_observation = (row_features, (edge_indices, edge_values), variable_features)

    action_set = rng.choice(n_vars, size=n_cands, replace=False).astype(np.int64)
    scores = rng.rand(n_vars).astype(np.float32)  # indexable by global variable index, like the real scores array
    action = action_set[np.argmax(scores[action_set])]

    return {
        'episode': 0,
        'instance': 'fake.lp',
        'seed': int(rng.randint(2**31 - 1)),
        'data': [node_observation, action, action_set, scores],
    }


def write_samples(out_dir, n_samples, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(n_samples):
        sample = make_one_sample(rng)
        with gzip.open(os.path.join(out_dir, f'sample_{i}.pkl'), 'wb') as f:
            pickle.dump(sample, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--problems', nargs='+', default=['setcover', 'cauctions'])
    parser.add_argument('--n_train', type=int, default=10)
    parser.add_argument('--n_valid', type=int, default=10)
    args = parser.parse_args()

    for i, problem in enumerate(args.problems):
        folder = os.path.join(args.data_root, PROBLEM_FOLDERS[problem])
        write_samples(os.path.join(folder, 'train'), args.n_train, seed=10 * i)
        write_samples(os.path.join(folder, 'valid'), args.n_valid, seed=10 * i + 1)
        print(f'{problem}: wrote {args.n_train} train + {args.n_valid} valid fake samples to {folder}')
