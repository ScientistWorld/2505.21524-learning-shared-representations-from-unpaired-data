# Evaluation Targets

## Primary: handwritten_retrieval
Tests whether a compact shared representation supports held-out cross-view retrieval under a weak-pair budget.
- **Benefit metric**: `mean_r@10` (higher is better)
- **Secondary benefit metrics**: `k2p_r@1`, `k2p_r@5`, `k2p_r@10`, `p2k_r@1`, `p2k_r@5`, `p2k_r@10` (higher is better)
- **Ranking-quality metrics**: `mean_mrr` (higher is better), `mean_median_rank` (lower is better)
- **Constraints (enforced as gates, not scored)**:
  - supervision budget: the public training data contains exactly 100 known pairs and no labels; the development and test pairs are evaluator-owned;
  - operating point: encoders must output exactly 8 dimensions;
  - test size: 400 held-out pairs chosen and ordered by the evaluator; every pair is a query in both directions;
  - per-view independence: each view is encoded in a separate call with only that view's rows;
  - validity: crashes, wrong shapes or non-finite outputs score as failed (every true match ranked last).
- **Reference comparison**: the paired-only `contrastive` baseline is frozen inside the evaluator and reported alongside every run.
- **Run**: `bash scripts/evaluate_train.sh` (development pairs, also `scripts/evaluate.sh`); the hidden test runs `bash scripts/evaluate_test.sh` with no arguments.
