# Evaluation Targets

## Primary: handwritten_retrieval
Tests whether a compact shared representation supports held-out cross-view retrieval under a weak-pair budget.
- **Benefit metric**: `mean_r@10` (higher is better)
- **Secondary benefit metrics**: `k2p_r@1`, `k2p_r@5`, `k2p_r@10`, `p2k_r@1`, `p2k_r@5`, `p2k_r@10`
- **Constraint metric**: `embedding_dim` (lower is better when recall is comparable)
- **Run**: `bash scripts/evaluate.sh artifacts`
