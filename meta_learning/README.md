# Reptile meta-learning pilot

Each meta-learning task is one `domain/parameter_configuration`. Every episode
uses labeled support and query samples from disjoint source MILP instances.

## Recommended first pilot: held-out combinatorial auctions

Run from the repository root on the training machine. This initializes Reptile
from the existing 5k LODO checkpoint so that the comparison isolates the effect
of meta-training rather than ordinary pooled pretraining.

```bash
mkdir -p model/meta_learning/reptile_lodo_cauctions_5k/0 && \
python meta_learning/train_reptile.py \
  --run-name reptile_lodo_cauctions_5k \
  --task indset/300_4 \
  --task indset/500_6 \
  --task setcover/250r_500c_0.05d \
  --task setcover/500r_1000c_0.02d \
  --task facilities/50_50_5 \
  --task facilities/100_100_3 \
  --initial-checkpoint model/pooled/lodo_cauctions_5k/0/train_params.pkl \
  --support-size 32 \
  --query-size 64 \
  --inner-steps 5 \
  --inner-lr 0.01 \
  --meta-step-size 0.1 \
  --iterations 1000 \
  --meta-batch-size 4 \
  --valid-every 20 \
  --valid-episodes 8 \
  --seed 0 \
  --gpu 0 \
  2>&1 | tee model/meta_learning/reptile_lodo_cauctions_5k/0/console.txt
```

Evaluate the learned initialization on the two unseen configurations:

```bash
mkdir -p model/meta_learning/reptile_lodo_cauctions_5k/0 && \
python meta_learning/evaluate_reptile.py \
  --checkpoint model/meta_learning/reptile_lodo_cauctions_5k/0/best_params.pkl \
  --task cauctions/50_250 \
  --task cauctions/75_375 \
  --support-size 32 \
  --query-size 100 \
  --inner-steps 5 \
  --inner-lr 0.01 \
  --episodes 20 \
  --seed 0 \
  --gpu 0 \
  --output model/meta_learning/reptile_lodo_cauctions_5k/0/eval_test.csv \
  2>&1 | tee model/meta_learning/reptile_lodo_cauctions_5k/0/eval_test.txt
```

For the pooled-initialization few-shot baseline, run the same evaluator with
`model/pooled/lodo_cauctions_5k/0/train_params.pkl` as the checkpoint. The
support/query seeds and sizes must remain identical.

## Interpretation

Compare three values on exactly the same query episodes:

1. zero-shot LODO accuracy;
2. LODO checkpoint after ordinary few-shot gradient adaptation;
3. Reptile initialization after the same adaptation.

Reptile is useful only if the third result consistently improves on the second.

## Paired checkpoint comparison

The two summary CSVs above do not preserve episode-level differences. For a
direct comparison on exactly the same support/query episodes, run:

```bash
python meta_learning/evaluate_paired.py \
  --baseline-checkpoint model/pooled/lodo_cauctions_5k/0/train_params.pkl \
  --reptile-checkpoint model/meta_learning/reptile_lodo_cauctions_5k/0/best_params.pkl \
  --task cauctions/50_250 \
  --task cauctions/75_375 \
  --support-size 32 \
  --query-size 100 \
  --inner-steps 5 \
  --inner-lr 0.01 \
  --episodes 100 \
  --seed 0 \
  --gpu 0 \
  --output model/meta_learning/reptile_lodo_cauctions_5k/0/eval_paired.csv
```

The script saves one row per episode and a separate `eval_paired_summary.csv`.
Reported differences are Reptile minus LODO; positive acc@1 is better. The
bootstrap intervals resample episodes within each task and can be optimistic
when different episodes reuse source MILP instances. This is a diagnostic,
not a new held-out dataset or a basis for tuning on the test split.

## Fixed-validation follow-up

The first 1000-iteration pilot selected its checkpoint using only 8 newly
sampled validation episodes per check. For a more stable source-task comparison,
rerun from the same LODO checkpoint with 4 fixed validation episodes for each
of the 6 source tasks. Use a new run name to preserve the first pilot:

```bash
python meta_learning/train_reptile.py \
  --run-name reptile_lodo_cauctions_5k_fixedval \
  --task indset/300_4 \
  --task indset/500_6 \
  --task setcover/250r_500c_0.05d \
  --task setcover/500r_1000c_0.02d \
  --task facilities/50_50_5 \
  --task facilities/100_100_3 \
  --initial-checkpoint model/pooled/lodo_cauctions_5k/0/train_params.pkl \
  --support-size 32 \
  --query-size 64 \
  --inner-steps 5 \
  --inner-lr 0.01 \
  --meta-step-size 0.1 \
  --iterations 1000 \
  --meta-batch-size 4 \
  --valid-every 20 \
  --valid-episodes-per-task 4 \
  --seed 0 \
  --gpu 0
```

This saves `validation_episodes.json` and per-task `valid_metrics.csv` alongside
the checkpoint. The fixed bank is drawn only from source-task `valid` splits;
the held-out cauctions test split is not used for checkpoint selection. The
older `--valid-episodes` option remains available for reproducing the original
random-validation pilot, but should not be used for this follow-up.

## Online SCIP pilot on new auction instances

This is a *new* test: do not solve the LPs that produced the offline support
samples. Run the following commands from the repository root after syncing the
new scripts to the training machine. The support set comes only from each
held-out task's `valid` split; both checkpoints receive exactly the same 32
samples and five gradient steps. The fixed-validation Reptile checkpoint is
used, not the earlier random-validation pilot.
The commands below use CPU inference (`--gpu -1`) so they can run on a
CPU-oriented server. Keep all compared policies on the same server under
similar load; solver wall times from different machines are not comparable.
The online evaluator calls Ecole's `as_pyscipopt()` to read SCIP statistics,
so the CPU server must also have the real `pyscipopt` installed in the same
environment as Ecole. The repository's `pyscipopt_stub` is not sufficient.

First, generate three new test LPs per auction configuration. Zero-sized
train/valid splits are allowed by `01_generate_instances_meta.py` for this
purpose. Its overwrite guard will reject an existing test directory; use a
new output root if rerunning.

```bash
python 01_generate_instances_meta.py cauctions \
  --n-items 50 --n-bids 250 --seed 20261013 \
  --train-instances 0 --valid-instances 0 --test-instances 3 \
  --output-root data/instances_online_pilot

python 01_generate_instances_meta.py cauctions \
  --n-items 75 --n-bids 375 --seed 20261014 \
  --train-instances 0 --valid-instances 0 --test-instances 3 \
  --output-root data/instances_online_pilot
```

Then create task-specific adapted weights and a support-file manifest:

```bash
python meta_learning/adapt_for_online.py \
  --baseline-checkpoint model/pooled/lodo_cauctions_5k/0/train_params.pkl \
  --reptile-checkpoint model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/best_params.pkl \
  --task cauctions/50_250 --task cauctions/75_375 \
  --data-root data/samples_meta \
  --support-size 32 --inner-steps 5 --inner-lr 0.01 \
  --seed 0 --gpu -1 \
  --output-dir model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot
```

Finally, solve each new LP with LODO-adapted, Reptile-adapted, and SCIP's
`relpscost` brancher under the same solver parameters and seed. The 60-second
time limit keeps this first check small. Results are flushed to CSV after each
solve, so completed rows survive an interruption.

```bash
python meta_learning/evaluate_online.py \
  --task cauctions/50_250 --task cauctions/75_375 \
  --checkpoint cauctions/50_250:lodo=model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/cauctions/50_250/lodo_adapted.pkl \
  --checkpoint cauctions/50_250:reptile=model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/cauctions/50_250/reptile_adapted.pkl \
  --checkpoint cauctions/75_375:lodo=model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/cauctions/75_375/lodo_adapted.pkl \
  --checkpoint cauctions/75_375:reptile=model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/cauctions/75_375/reptile_adapted.pkl \
  --instances-root data/instances_online_pilot \
  --limit-instances 3 --seed 0 --time-limit 60 \
  --include-relpscost --gpu -1 \
  --output model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/solve_results.csv
```

Compare matched rows on `task`, `instance`, and `seed`. First check solved
status and gap, then nodes and time; do not treat an unsolved run's time limit
as a normal solve time. If the pilot works, run a separate, larger confirmatory
set with fresh instances and seeds. Do not tune Reptile using these held-out
auction outcomes.

The evaluator also writes a compact `solve_results_summary.csv`. To share the
pilot result without pasting the full per-solve CSV:

```bash
cat model/meta_learning/reptile_lodo_cauctions_5k_fixedval/0/online_pilot/solve_results_summary.csv
```

