#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

# Data must already be present before submission because compute nodes have no
# internet. Run `bash scripts/download.sh` on the login node before submitting.
bash scripts/reproduce.sh
python validate.py --compare
