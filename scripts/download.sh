#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# Nothing has to be fetched.  The only source is the public UCI Multiple
# Features archive (https://archive.ics.uci.edu/static/public/72/multiple+features.zip,
# CC BY 4.0, sha256 898a50a7637f1ed5a8cee2493aaa0e7f4d52795c412f59b0969ec7c3046ee7bd).
# The rows this gym uses are committed verbatim in git:
#   data/handwritten/   public weakly paired training data (100 pairs + 990 unpaired rows per view)
#   eval/train/data/    public development pairs used by scripts/evaluate_train.sh
#   eval/test/data/     held-out pairs used by the hidden test
# The full row-aligned archive is deliberately NOT placed in the workspace: it
# would expose the held-out correspondences.  The evaluator directory that owns
# the held-out split also holds the builder that re-derives every committed
# file from the archive (it fetches the archive itself into a temporary
# directory).  Python packages are installed by environment/setup.sh.
#
# This script is idempotent and works from an empty data/downloads/.

set -e

cd /home/user

mkdir -p data/downloads
for f in data/handwritten/paired_kar.txt data/handwritten/paired_pix.txt \
         data/handwritten/unpaired_kar.txt data/handwritten/unpaired_pix.txt; do
  if [ ! -s "$f" ]; then
    echo "missing $f: restore it from git (git checkout -- $f)" >&2
    exit 1
  fi
done
echo "All data is committed in the repository; nothing to download."
