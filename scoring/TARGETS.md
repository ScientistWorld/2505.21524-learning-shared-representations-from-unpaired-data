# Evaluation Targets

## Primary: handwritten_retrieval
Tests whether a compact shared representation supports held-out cross-view retrieval under a weak-pair budget.
- **Benefit metric**: `mean_r@10` (higher is better)
- **Secondary benefit metrics**: `k2p_r@1`, `k2p_r@5`, `k2p_r@10`, `p2k_r@1`, `p2k_r@5`, `p2k_r@10`
- **Ranking diagnostics**: `mean_mrr` (higher is better), `mean_median_rank` (lower is better)
- **Constraint metrics**: `paired_train_count` (lower is better), `embedding_dim` (lower is better when recall is comparable)
- **Budget diagnostics**: `unpaired_train_count`, `eval_pair_count`
- **Run**: `bash scripts/evaluate.sh artifacts`
