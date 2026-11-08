#!/bin/bash
# Standard evaluation: scores every method directory under the artifact
# directory on the public development pairs and writes scoring/scores.json.
#
# A method directory <artifact-dir>/<name>/ must contain encoder.json with a
# "command" list; the evaluator appends
#   --view {kar|pix} --input <rows.npy> --output <embeddings.npy>
# and replaces "{model_dir}" with a private copy of the directory.  See
# briefing/evaluation.md for the full contract.  Uses only code in eval/.
#
# Usage: bash scripts/evaluate.sh [ARTIFACT_DIR]   (default: artifacts)

set -e

cd /home/user

mkdir -p scoring

ARTIFACT_DIR="${1:-artifacts}"
python -m eval.train.evaluate --artifact-dir "$ARTIFACT_DIR" --scores scoring/scores.json
