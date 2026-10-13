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

