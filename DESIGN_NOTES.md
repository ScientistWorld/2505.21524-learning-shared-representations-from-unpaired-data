# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The test script calls `eval.test.evaluate`, writes `scoring/scores_test.json`, and does not invoke the train evaluator.
- Checked `eval/train/` and `eval/test/` for cross-imports. Neither slice imports from the other.
- Checked that the split evaluators read method-produced `.npz` artifacts instead of regenerating the method or importing `method/`.
- Checked for hardcoded method defaults. The evaluators enumerate `.npz` artifacts from the supplied artifact directory.
- Checked forbidden paper-agreement metric names in the train/test evaluator surface and `scoring/reference.json`; none were present.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran `bash -n` on both train/test shell scripts.

## Changed

- Added bidirectional rank-quality metrics to the train/test evaluators: `mean_mrr` and `mean_median_rank`.
- Added the same metrics to the standard retrieval evaluator so the visible schema remains computable by the existing artifact evaluator.
- Added `_slice` markers to train/test score payloads to make accidental identical train/test outputs easier to diagnose.
- Added required `description` fields for every metric in `scoring/reference.json`.
- Added `mean_mrr` and `mean_median_rank` entries to `scoring/reference.json` with coefficients that reward true-pair rank quality beyond recall@10.

## Runtime Concerns For Next Iteration

- I did not execute the Python evaluators or produce train/test numbers in this pass. The next runner should verify that the new rank metrics serialize as expected.
- The train/test split is a stratified half/complement split over method artifacts. This is good for disjointness, but the hidden test remains a slice of the same fixed artifact rows; future changes that improve only unsliced rows could still overfit if the held-out split logic leaked.
- Existing reference result entries were left untouched, so historical methods do not include `mean_mrr` or `mean_median_rank` values until scores are regenerated.
