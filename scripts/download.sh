#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.006 GB
# Estimated disk usage after extraction: 0.006 GB
#
# This reproduction only needs the public UCI Multiple Features archive.
# The script writes to workspace-local data/downloads/ for portability.

set -e

cd /home/user

mkdir -p data/downloads

if [ ! -e data/downloads/multiple_features.zip ]; then
  curl --fail --location --retry 3 https://archive.ics.uci.edu/static/public/72/multiple+features.zip \
    -o data/downloads/multiple_features.zip
fi
