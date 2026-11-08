#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cases=(
  reference_stored reference_seed_13 reference_seed_17 reference_seed_29
  honest_more_training honest_more_unpaired honest_anchored honest_neighbors_30
  honest_anchor_weight01 honest_anchor_weight10 honest_anchored_long
  honest_contrastive_alignment honest_richer honest_richer_local
  weak_no_cca baseline_paired
  shortcut_raw_pair_lookup shortcut_copy_input shortcut_zero_budget
  shortcut_negative_budget shortcut_label_slice shortcut_test_size
  shortcut_replace_baseline shortcut_direct_scores shortcut_constant
  shortcut_empty shortcut_nan shortcut_null_budget shortcut_infinite_embeddings
)
for name in "${cases[@]}"; do
  if [ -s "redteam/$name/result.json" ]; then
    echo "Already ran $name"
    continue
  fi
  bash "redteam/$name/run.sh"
done
python -m redteam.inspect_cached
