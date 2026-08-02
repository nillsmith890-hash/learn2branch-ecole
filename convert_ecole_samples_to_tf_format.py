"""
Converts learn2branch-ecole's sample_*.pkl format (raw Ecole NodeBipartite
tuples) into learn2branch's (TF1.x) sample_*.pkl format (dict-wrapped
features), so the *already-extracted* Ecole features can be loaded and
trained on by the original repo's utilities_tf.load_batch_gcnn + baseline
GCNN model.

Why: learn2branch-ecole's re-trained GCNN (both plain and MAML) plateaus at
~10% top-1 accuracy on setcover, vs. 65.5% reported in the original paper
(Table 1) using learn2branch's own pipeline. This conversion isolates where
that gap comes from without needing to reinstall PySCIPOpt + a patched SCIP
and regenerate instances from scratch:
  - old model + converted Ecole features still ~10%  -> the bug is in
    Ecole's feature extraction itself (the numbers written into the
    sample_*.pkl files).
  - old model + converted Ecole features ~65%        -> the bug is in
    learn2branch-ecole's own model/training code, not the data.

This only repackages already-recorded feature values into the old dict
"wire format" (utilities_tf.load_batch_gcnn reads c['values'], e['indices'],
e['values'], v['values'] -- see that function for the exact contract). It
cannot retroactively recompute what learn2branch's own PySCIPOpt-based
extractor would have produced from the live SCIP solve, since that state no
longer exists once Ecole has already recorded its own features.

Usage:
    python convert_ecole_samples_to_tf_format.py \\
        learn2branch-ecole/data/samples/setcover/500r_1000c_0.05d \\
        learn2branch/data/samples/setcover/500r_1000c_0.05d
"""
import argparse
import gzip
import os
import pickle

import numpy as np


def convert_sample(ecole_sample):
    node_observation, action, action_set, scores = ecole_sample['data']
    row_features, (edge_indices, edge_values), variable_features = node_observation

    action_set = np.asarray(action_set)
    scores = np.asarray(scores)

    constraint_features = {
        'names': [f'c{i}' for i in range(row_features.shape[0])],
        'values': np.asarray(row_features, dtype=np.float32),
    }
    edge_features = {
        'names': ['coef_normalized'],
        'indices': np.asarray(edge_indices, dtype=np.int32),
        'values': np.asarray(edge_values, dtype=np.float32).reshape(-1, 1),
    }
    variable_features = {
        'names': [f'v{i}' for i in range(variable_features.shape[0])],
        'values': np.asarray(variable_features, dtype=np.float32),
    }
    state = (constraint_features, edge_features, variable_features)
    state_khalil = {}  # unused by the GCNN loader; only the TREES/SVMRANK/LMART baselines need it

    # old format stores scores already restricted to (and ordered like) the
    # candidates, unlike Ecole's `scores`, which is indexable by any global
    # variable id -- see utilities_tf.load_batch_gcnn's cand_scoress handling.
    cand_scores = scores[action_set].astype(np.float32)

    return {
        'episode': ecole_sample.get('episode'),
        'instance': ecole_sample.get('instance'),
        'seed': ecole_sample.get('seed'),
        'node_number': -1,  # not read by load_batch_gcnn; Ecole doesn't record an equivalent
        'node_depth': -1,
        'data': [state, state_khalil, int(action), action_set.astype(np.int32), cand_scores],
    }


def convert_file(src_path, dst_path):
    with gzip.open(src_path, 'rb') as f:
        sample = pickle.load(f)
    converted = convert_sample(sample)
    with gzip.open(dst_path, 'wb') as f:
        pickle.dump(converted, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src_dir', help='learn2branch-ecole sample_*.pkl directory (recursed, e.g. .../setcover/500r_1000c_0.05d)')
    parser.add_argument('dst_dir', help='output directory; mirrors src_dir structure (train/valid/test subfolders included)')
    parser.add_argument('--limit_per_dir', type=int, default=None,
                         help='convert at most this many sample_*.pkl per leaf directory (e.g. per train/valid/test), '
                              'for a quick diagnostic run without converting an entire 100k-sample split')
    args = parser.parse_args()

    n = 0
    for root, _, files in os.walk(args.src_dir):
        sample_files = sorted(f for f in files if f.startswith('sample_') and f.endswith('.pkl'))
        if args.limit_per_dir is not None:
            sample_files = sample_files[:args.limit_per_dir]
        for fname in sample_files:
            src_path = os.path.join(root, fname)
            rel = os.path.relpath(src_path, args.src_dir)
            dst_path = os.path.join(args.dst_dir, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            convert_file(src_path, dst_path)
            n += 1
            if n % 1000 == 0:
                print(f'{n} files converted...')

    print(f'done: {n} files converted from {args.src_dir} to {args.dst_dir}')
