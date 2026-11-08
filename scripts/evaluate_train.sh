#!/bin/bash
# Development evaluator for the handwritten retrieval gym.
#
# Scores every method directory (with encoder.json) under the artifact
# directory on the 400 public development pairs in eval/train/data/ and writes
# scoring/scores_train.json.  Same protocol as the hidden test.
#
# Usage: bash scripts/evaluate_train.sh [ARTIFACT_DIR]   (default: artifacts)

set -e

cd /home/user

mkdir -p scoring

ARTIFACT_DIR="${1:-artifacts}"
python -m eval.train.evaluate --artifact-dir "$ARTIFACT_DIR" --scores scoring/scores_train.json
