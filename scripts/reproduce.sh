#!/bin/bash
# Reproduce the paper's results end-to-end (CPU, a few minutes).
#
# Trains the paired-only contrastive baseline and SUE on the public weakly
# paired data (data/handwritten/), writes their encoder directories under
# artifacts/, and scores them on the development pairs (scoring/scores.json).
#
# Assumes the container from environment/ and runs from /home/user.

set -e

cd /home/user

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf artifacts
mkdir -p artifacts
bash scripts/baseline.sh artifacts
bash scripts/method.sh artifacts
