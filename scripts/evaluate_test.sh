#!/bin/bash
# Hidden held-out evaluator for the handwritten retrieval gym.
#
# Scores every method directory (with encoder.json) under the artifact
# directory on the 400 held-out pairs owned by eval/test/ and writes
# scoring/scores_test.json.  Runs with no arguments.
#
# Usage: bash scripts/evaluate_test.sh [ARTIFACT_DIR]   (default: artifacts)

set -e

cd /home/user

mkdir -p scoring

ARTIFACT_DIR="${1:-artifacts}"
python -m eval.test.evaluate --artifact-dir "$ARTIFACT_DIR" --scores scoring/scores_test.json
