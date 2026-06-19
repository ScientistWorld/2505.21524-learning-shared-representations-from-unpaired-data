#!/bin/bash
# Held-out evaluator for the handwritten retrieval gym.
#
# Mirrors scripts/evaluate.sh: pass an artifact directory containing method
# .npz outputs, and this writes scoring/scores_test.json. When
# scoring/scores_train.json is present, the test slice scores the same
# experiment/method entries that appeared in the train slice.

set -e

cd /home/user

mkdir -p scoring

ARTIFACT_DIR="${1:-artifacts}"
python -m eval.test.evaluate \
  --artifact-dir "$ARTIFACT_DIR" \
  --train-scores scoring/scores_train.json \
  --scores scoring/scores_test.json
