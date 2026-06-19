#!/bin/bash
# Visible development-slice evaluator for the handwritten retrieval gym.
#
# Mirrors scripts/evaluate.sh: pass an artifact directory containing method
# .npz outputs, and this writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

ARTIFACT_DIR="${1:-artifacts}"
python -m eval.train.evaluate --artifact-dir "$ARTIFACT_DIR" --scores scoring/scores_train.json
