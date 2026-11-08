#!/bin/bash
# Job script: reproduce the reference end to end, score it on the development
# and hidden test pairs, validate, then run the hardening audit.
#
# Runs inside the gym container with the workspace at /home/user, no internet.

set -e

cd /home/user

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

bash scripts/download.sh
bash scripts/reproduce.sh

# Freeze the paired-only baseline into the evaluators (no-op once frozen).
python -m eval.train.evaluate --freeze-baseline artifacts/contrastive
python -m eval.test.evaluate --freeze-baseline artifacts/contrastive

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare --post-design || echo "VALIDATION FAILED (see above)"

# Hardening audit: variants, seed noise and shortcuts through the hidden test.
python -m redteam.hardening.audit || echo "AUDIT FAILED (see above)"

python validate.py --compare --post-design
