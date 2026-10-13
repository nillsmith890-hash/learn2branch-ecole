"""Episode sampling for parameterized MILP tasks.

A task is a ``domain/parameter_configuration`` directory.  Support and query
samples are drawn from disjoint source MILP instances, not merely from
different pickle files produced by the same branch-and-bound trajectory.
"""

import gzip
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class TaskEpisodeSampler:
    def __init__(self, data_root, tasks, split, support_size, query_size, seed=0):
        if support_size <= 0 or query_size <= 0:
            raise ValueError("support_size and query_size must be positive")
        if not tasks:
            raise ValueError("at least one task is required")

        self.data_root = Path(data_root)
        self.tasks = list(tasks)
        self.split = split
        self.support_size = support_size
        self.query_size = query_size
        self.rng = np.random.RandomState(seed)
        self.files_by_instance = {
            task: self._index_task(task) for task in self.tasks
        }

    def _index_task(self, task):
        directory = self.data_root / task / self.split
        files = sorted(directory.glob("sample_*.pkl"))
        if not files:
            raise FileNotFoundError(f"no samples found in {directory}")

        groups = defaultdict(list)
        for path in files:
            with gzip.open(path, "rb") as stream:
                sample = pickle.load(stream)
            instance = sample.get("instance")
            if instance is None:
                raise ValueError(f"sample has no source instance: {path}")
            groups[str(instance)].append(str(path))

        if len(groups) < 2:
            raise ValueError(
                f"task {task!r} has only {len(groups)} source instance(s); "
                "support/query separation requires at least two"
            )
        return dict(groups)

    def _draw_files(self, groups, target_size):
        instance_names = list(groups)
        self.rng.shuffle(instance_names)
        selected = []
        used_instances = []
        for name in instance_names:
            candidates = list(groups[name])
            self.rng.shuffle(candidates)
            take = min(target_size - len(selected), len(candidates))
            if take:
                selected.extend(candidates[:take])
                used_instances.append(name)
            if len(selected) == target_size:
                break
        return selected, used_instances

    def sample_task(self, task=None):
        if task is None:
            task = self.tasks[self.rng.randint(len(self.tasks))]
        if task not in self.files_by_instance:
            raise KeyError(f"unknown task: {task}")

        groups = self.files_by_instance[task]
        support, support_instances = self._draw_files(groups, self.support_size)
        remaining = {
            name: files for name, files in groups.items()
            if name not in set(support_instances)
        }
        query, query_instances = self._draw_files(remaining, self.query_size)

        if len(support) < self.support_size or len(query) < self.query_size:
            raise ValueError(
                f"task {task!r} cannot provide {self.support_size} support and "
                f"{self.query_size} query samples from disjoint instances"
            )
        return {
            "task": task,
            "support_files": support,
            "query_files": query,
            "support_instances": support_instances,
            "query_instances": query_instances,
        }

    def sample_meta_batch(self, size):
        if size <= 0:
            raise ValueError("meta-batch size must be positive")
        replace = size > len(self.tasks)
        chosen = self.rng.choice(self.tasks, size=size, replace=replace)
        return [self.sample_task(str(task)) for task in chosen]

