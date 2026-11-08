#!/bin/bash
# Run the paper's paired-only contrastive baseline.
#
# Trains two MLP encoders on the 100 known pairs and writes an encoder
# directory <OUTPUT_DIR>/contrastive/.  The evaluators report the frozen copy
# of this baseline that they own (the name "contrastive" is reserved); this
# script is how that frozen copy was produced.
#
# Usage: bash scripts/baseline.sh [OUTPUT_DIR]   (default: artifacts)

set -e

cd /home/user

OUTPUT_DIR="${1:-artifacts}"
python -m baseline.handwritten_baselines train --output-dir "$OUTPUT_DIR/contrastive"
bash scripts/evaluate.sh "$OUTPUT_DIR"
