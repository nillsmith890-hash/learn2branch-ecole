import os
import glob
import gzip
import argparse
import pickle
import queue
import re
import shutil
import threading
import numpy as np
import ecole
from collections import namedtuple


class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)


def send_orders(orders_queue, instances, seed, query_expert_prob, time_limit, out_dir, stop_flag):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : queue.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while not stop_flag.is_set():
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, time_limit, out_dir])
        episode += 1


def make_samples(in_queue, out_queue, stop_flag, max_samples_per_episode):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    max_samples_per_episode : int
        Maximum number of samples collected from one branch-and-bound episode.
        A value of 0 disables the limit.
    """
    sample_counter = 0
    invalid_expert_counter = 0
    while not stop_flag.is_set():
        try:
            episode, instance, seed, query_expert_prob, time_limit, out_dir = in_queue.get(timeout=1)
        except queue.Empty:
            continue

        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                 "node_observation": ecole.observation.NodeBipartite() }
        # StrongBranchingScores is defined on LP branching candidates.  Using
        # pseudo candidates here can add candidates without a valid strong-
        # branching score (NaN), which in turn makes np.argmax select an
        # invalid expert action.
        env = ecole.environment.Branching(observation_function=observation_function,
                                          scip_params=scip_parameters, pseudo_candidates=False)

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        env.seed(seed)
        observation, action_set, _, done, _ = env.reset(instance)
        episode_sample_counter = 0
        while not done:
            scores, scores_are_expert = observation["scores"]
            node_observation = observation["node_observation"]
            node_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices,
                                 node_observation.edge_features.values),
                                node_observation.variable_features)

            action_set = np.asarray(action_set, dtype=np.int64)
            candidate_scores = np.asarray(scores)[action_set]
            finite_mask = np.isfinite(candidate_scores)

            # Never pass NaN/Inf to argmax.  A partially scored expert state is
            # unsuitable as a supervised sample because the true best action
            # among all candidates is unknown.  We can still advance the
            # environment using the best action among the finite scores.
            if finite_mask.any():
                finite_actions = action_set[finite_mask]
                finite_scores = candidate_scores[finite_mask]
                action = finite_actions[finite_scores.argmax()]
            else:
                # This should be rare with LP candidates, but keep the episode
                # moving without recording a bogus label.
                action = action_set[0]

            valid_expert_sample = scores_are_expert and finite_mask.all()
            if scores_are_expert and not valid_expert_sample:
                invalid_expert_counter += 1

            if valid_expert_sample and not stop_flag.is_set():
                data = [node_observation, action, action_set, scores]
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                })
                sample_counter += 1
                episode_sample_counter += 1

                # Prevent a single large search tree from dominating a split.
                # Ending the episode here also avoids spending a long time on
                # samples that the main process will ultimately discard.
                if (max_samples_per_episode > 0
                        and episode_sample_counter >= max_samples_per_episode):
                    break

            try:
                observation, action_set, _, done, _ = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")

        print(f"[w {threading.current_thread().name}] episode {episode} done, "
              f"{episode_sample_counter} episode samples "
              f"({sample_counter} worker total), "
              f"{invalid_expert_counter} invalid expert states skipped\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    query_expert_prob, time_limit, max_samples_per_episode=0,
                    start_index=0, target_total=None):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.
    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    max_samples_per_episode : int
        Maximum samples contributed by one episode.  A value of 0 disables
        the limit.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  time_limit, tmp_samples_dir, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag,
                      max_samples_per_episode),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(
                        sample['filename'],
                        f'{out_dir}/sample_{start_index+i+1}.pkl',
                    )
                    in_buffer -= 1
                    i += 1
                    written = start_index + i
                    target = target_total if target_total is not None else written
                    print(f"[m {threading.current_thread().name}] {written} / {target} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # # stop all workers
    workers_stop_flag.set()
    for p in workers:
        p.join()

    print(f"Done collecting samples for {out_dir}")
    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


def count_existing_samples(out_dir):
    """Count a contiguous sample_1.pkl, ..., sample_N.pkl sequence."""
    indices = []
    for path in glob.glob(os.path.join(out_dir, 'sample_*.pkl')):
        match = re.fullmatch(r'sample_(\d+)\.pkl', os.path.basename(path))
        if match:
            indices.append(int(match.group(1)))

    indices.sort()
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        raise RuntimeError(
            f"cannot resume {out_dir}: sample numbering must be contiguous "
            "from sample_1.pkl"
        )
    return len(indices)


def collect_split(instances, out_dir, rng, target_size, args, time_limit):
    """Generate a fresh split or append to its requested total size."""
    existing = count_existing_samples(out_dir) if args.resume else 0
    if args.resume and existing >= target_size:
        print(f"Skipping {out_dir}: already has {existing} samples "
              f"(target {target_size}).")
        return

    collect_samples(
        instances,
        out_dir,
        rng,
        target_size - existing,
        args.njobs,
        query_expert_prob=args.node_record_prob,
        time_limit=time_limit,
        max_samples_per_episode=args.max_samples_per_episode,
        start_index=existing,
        target_total=target_size,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--instances-dir',
        help='Custom task directory containing train/, valid/, and test/ LP files.',
    )
    parser.add_argument(
        '--output-dir',
        help='Custom sample output directory (required with --instances-dir).',
    )
    parser.add_argument('--train-size', type=int, default=100000)
    parser.add_argument('--valid-size', type=int, default=20000)
    parser.add_argument('--test-size', type=int, default=20000)
    parser.add_argument('--node-record-prob', type=float, default=0.05)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument(
        '--max-samples-per-episode',
        type=int,
        default=0,
        help='Maximum samples from one B&B episode; 0 disables the limit.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Append without overwriting until each split reaches its requested '
            'target size. Existing sample numbering must be contiguous.'
        ),
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    node_record_prob = args.node_record_prob
    time_limit = 3600 if args.time_limit is None else args.time_limit

    if min(train_size, valid_size, test_size) < 1:
        parser.error('sample sizes must be positive')
    if not 0 < node_record_prob <= 1:
        parser.error('--node-record-prob must be in (0, 1]')
    if time_limit <= 0:
        parser.error('--time-limit must be positive')
    if args.max_samples_per_episode < 0:
        parser.error('--max-samples-per-episode must be non-negative')
    if bool(args.instances_dir) != bool(args.output_dir):
        parser.error('--instances-dir and --output-dir must be used together')

    if args.instances_dir:
        instances_train = glob.glob(os.path.join(args.instances_dir, 'train', '*.lp'))
        instances_valid = glob.glob(os.path.join(args.instances_dir, 'valid', '*.lp'))
        instances_test = glob.glob(os.path.join(args.instances_dir, 'test', '*.lp'))
        out_dir = args.output_dir

    elif args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
        out_dir = 'data/samples/setcover/500r_1000c_0.05d'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
        out_dir = 'data/samples/cauctions/100_500'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
        instances_test = glob.glob('data/instances/indset/test_500_4/*.lp')
        out_dir = 'data/samples/indset/500_4'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
        out_dir = 'data/samples/facilities/100_100_5'
        if args.time_limit is None:
            time_limit = 600

    elif args.problem == 'mknapsack':
        instances_train = glob.glob('data/instances/mknapsack/train_100_6/*.lp')
        instances_valid = glob.glob('data/instances/mknapsack/valid_100_6/*.lp')
        instances_test = glob.glob('data/instances/mknapsack/test_100_6/*.lp')
        out_dir = 'data/samples/mknapsack/100_6'
        time_limit = 60

    else:
        raise NotImplementedError

    instances_train = sorted(instances_train)
    instances_valid = sorted(instances_valid)
    instances_test = sorted(instances_test)
    for split, instances in (
        ('train', instances_train),
        ('valid', instances_valid),
        ('test', instances_test),
    ):
        if not instances:
            parser.error(f'no .lp files found for the {split} split')

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_split(
        instances_train, out_dir + '/train', rng, train_size, args, time_limit
    )

    rng = np.random.RandomState(args.seed + 1)
    collect_split(
        instances_valid, out_dir + '/valid', rng, valid_size, args, time_limit
    )

    rng = np.random.RandomState(args.seed + 2)
    collect_split(
        instances_test, out_dir + '/test', rng, test_size, args, time_limit
    )
