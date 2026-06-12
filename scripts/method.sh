#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

OUTPUT_DIR="${1:-artifacts}"
python -m method.sue --output-dir "$OUTPUT_DIR"
bash scripts/evaluate.sh "$OUTPUT_DIR"
