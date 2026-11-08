#!/bin/bash
# Run the paper's proposed method (SUE) and score it on the development pairs.
#
# Trains on the public weakly paired data and writes an encoder directory
# <OUTPUT_DIR>/sue/ (encoder.json + model files).
#
# Usage: bash scripts/method.sh [OUTPUT_DIR]   (default: artifacts)

set -e

cd /home/user

OUTPUT_DIR="${1:-artifacts}"
python -m method.sue train --output-dir "$OUTPUT_DIR/sue"
bash scripts/evaluate.sh "$OUTPUT_DIR"
