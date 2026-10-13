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

