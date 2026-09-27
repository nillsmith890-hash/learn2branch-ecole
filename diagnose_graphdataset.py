"""
Sanity-checks utilities.GraphDataset.get() against a hand-rolled re-parse of
the same raw sample_*.pkl files, on real data. Run from learn2branch-ecole/:

    python diagnose_graphdataset.py data/samples/setcover/500r_1000c_0.05d/train sample_0.pkl sample_1.pkl sample_2.pkl

If no filenames are given, picks the first 3 files in that directory.
"""
import sys
import gzip
import pickle
import glob
import numpy as np

sys.path.insert(0, '.')
from utilities import GraphDataset


def raw_reparse(path):
    with gzip.open(path, 'rb') as f:
        sample = pickle.load(f)
    node_observation, action, action_set, scores = sample['data']
    row_features, (edge_indices, edge_values), variable_features = node_observation
    action_set = np.asarray(action_set)
    scores = np.asarray(scores)
    choice = int(np.where(action_set == action)[0][0])
    cand_scores = scores[action_set]
    return {
        'n_cons': row_features.shape[0],
        'n_vars': variable_features.shape[0],
        'n_edges': edge_indices.shape[1],
        'cons_feat_range': (float(row_features.min()), float(row_features.max())),
        'var_feat_range': (float(variable_features.min()), float(variable_features.max())),
        'edge_feat_range': (float(edge_values.min()), float(edge_values.max())),
        'n_candidates': len(action_set),
        'action': int(action),
        'choice_idx': choice,
        'cand_scores_range': (float(cand_scores.min()), float(cand_scores.max())),
        'best_score': float(cand_scores.max()),
        'chosen_score': float(cand_scores[choice]),
    }


def via_graphdataset(path):
    ds = GraphDataset([path])
    g = ds[0]
    cand_scores = g.candidate_scores.numpy()
    return {
        'n_cons': g.constraint_features.shape[0],
        'n_vars': g.variable_features.shape[0],
        'n_edges': g.edge_index.shape[1],
        'cons_feat_range': (float(g.constraint_features.min()), float(g.constraint_features.max())),
        'var_feat_range': (float(g.variable_features.min()), float(g.variable_features.max())),
        'edge_feat_range': (float(g.edge_attr.min()), float(g.edge_attr.max())),
        'n_candidates': int(g.nb_candidates),
        'choice_idx': int(g.candidate_choices),
        'cand_scores_range': (float(cand_scores.min()), float(cand_scores.max())),
        'best_score': float(cand_scores.max()),
        'chosen_score': float(cand_scores[int(g.candidate_choices)]),
    }


if __name__ == '__main__':
    if len(sys.argv) > 2:
        directory, files = sys.argv[1], sys.argv[2:]
        paths = [f"{directory}/{f}" for f in files]
    elif len(sys.argv) == 2:
        paths = sorted(glob.glob(f"{sys.argv[1]}/sample_*.pkl"))[:3]
    else:
        print("usage: python diagnose_graphdataset.py <dir> [file1 file2 ...]")
        sys.exit(1)

    n_mismatch = 0
    for path in paths:
        a = raw_reparse(path)
        b = via_graphdataset(path)
        print(f"\n=== {path} ===")
        keys = list(a.keys())
        for k in keys:
            av, bv = a[k], b[k]
            same = 'OK' if av == bv else '!! MISMATCH'
            print(f"  {k:20s} raw={av!s:30s} graphdataset={bv!s:30s} {same}")
            if av != bv:
                n_mismatch += 1
        # a key sanity check independent of both paths: is the "chosen" score
        # actually the best score among candidates? if not, something upstream
        # (in 02_generate_dataset.py's own scores/action_set) is already off.
        gap = a['best_score'] - a['chosen_score']
        print(f"  chosen candidate is best-scoring: {'YES' if gap < 1e-6 else f'NO (gap={gap:.4f})'}")

    print(f"\n{n_mismatch} field mismatches across {len(paths)} files")
