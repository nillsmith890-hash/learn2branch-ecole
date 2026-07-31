#!/usr/bin/env bash
# Quick sanity check for train_maml.py: runs a few meta-iterations (both the
# first-order and second-order code paths) against synthetic data, so it
# catches shape/wiring bugs in minutes without needing real
# learn2branch-ecole data or a real SCIP/ecole install.
#
# Unlike the old TF1.x MAML/, this training path never imports ecole or
# pyscipopt (those are only used by 01_generate_instances.py /
# 02_generate_dataset.py to produce the sample_*.pkl files) -- utilities.py,
# model/model.py and MAML/*.py only need torch + torch_geometric. One-time
# env setup:
#   conda create -n maml_ecole_smoke python=3.9
#   conda run -n maml_ecole_smoke conda install pytorch cpuonly -c pytorch
#   conda run -n maml_ecole_smoke conda install pyg -c pyg -c conda-forge
#
# Usage:
#   MAML/tests/smoke_test.sh
#   MAML_SMOKE_PYTHON=/path/to/python MAML/tests/smoke_test.sh   # override interpreter

set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAML_DIR="$(dirname "$TEST_DIR")"
PYTHON="${MAML_SMOKE_PYTHON:-$(conda info --base 2>/dev/null)/envs/maml_ecole_smoke/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "error: no python at $PYTHON (see setup instructions at the top of this script)" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

"$PYTHON" "$TEST_DIR/make_fake_samples.py" \
    --data_root "$WORK_DIR/data/samples" \
    --problems setcover cauctions \
    --n_train 10 --n_valid 10

cd "$WORK_DIR"

COMMON_ARGS=(
    --problems setcover cauctions
    --data_root "$WORK_DIR/data/samples"
    --k_support 3 --k_query 3
    --meta_batch_size 2
    --inner_steps 2
    --meta_iterations 3
    --valid_every 1
    -g -1
)

echo "=== first-order (FOMAML) ==="
"$PYTHON" "$MAML_DIR/train_maml.py" "${COMMON_ARGS[@]}" --first_order -s 0

echo "=== second-order (true MAML) ==="
"$PYTHON" "$MAML_DIR/train_maml.py" "${COMMON_ARGS[@]}" --second_order -s 1

echo "smoke test OK"
