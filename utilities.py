import gzip
import pickle
import datetime
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch_geometric

def valid_seed(seed):
    """Check whether seed is a valid seed number."""
    seed = int(seed)
    if seed < 0 or seed > 2**31 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**31 - 1 inclusive")
    return seed


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class TaskBalancedGraphDataset(GraphDataset):
    """A concatenated graph dataset that retains per-task index ranges."""

    def __init__(self, task_files):
        if not task_files:
            raise ValueError("task_files must contain at least one task")

        self.task_files = OrderedDict()
        self.task_indices = OrderedDict()
        sample_files = []
        offset = 0

        for task_name, files in task_files.items():
            files = list(files)
            if not files:
                raise ValueError(f"task {task_name!r} contains no sample files")
            self.task_files[task_name] = files
            indices = np.arange(offset, offset + len(files), dtype=np.int64)
            self.task_indices[task_name] = indices
            sample_files.extend(files)
            offset += len(files)

        super().__init__(sample_files)


class TaskBalancedSampler(torch.utils.data.Sampler):
    """Sample an equal number of graph states from every task each epoch."""

    def __init__(self, task_indices, samples_per_task, seed=0):
        if samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if not task_indices:
            raise ValueError("task_indices must contain at least one task")

        self.task_indices = OrderedDict(
            (name, np.asarray(indices, dtype=np.int64))
            for name, indices in task_indices.items()
        )
        if any(len(indices) == 0 for indices in self.task_indices.values()):
            raise ValueError("every task must contain at least one index")

        self.samples_per_task = int(samples_per_task)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        sampled = []
        for indices in self.task_indices.values():
            task_sample = rng.choice(
                indices,
                size=self.samples_per_task,
                replace=self.samples_per_task > len(indices),
            )
            sampled.append(task_sample)

        balanced_indices = np.concatenate(sampled)
        rng.shuffle(balanced_indices)
        return iter(balanced_indices.tolist())

    def __len__(self):
        return len(self.task_indices) * self.samples_per_task


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
