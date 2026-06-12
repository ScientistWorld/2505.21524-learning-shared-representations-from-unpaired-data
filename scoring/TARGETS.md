# Evaluation Targets

## Primary: handwritten_retrieval
Tests whether a shared representation supports held-out cross-view retrieval under a weak-pair budget.
- **Metric**: `mean_r@10` (higher is better)
- **Run**: `bash scripts/evaluate.sh artifacts`

## Constraint: compact_shared_space
Tracks the dimensionality of the shared embedding used for retrieval.
- **Metric**: `embedding_dim` (lower is better when recall is comparable)
- **Run**: `bash scripts/evaluate.sh artifacts`
