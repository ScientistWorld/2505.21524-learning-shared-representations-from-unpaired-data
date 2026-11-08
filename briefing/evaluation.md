# Evaluation

## What is measured

Cross-view instance retrieval between the two views of held-out handwritten digits, in an 8-dimensional shared space. `k2p` queries with the Karhunen-Loeve view (`kar`) and retrieves among the pixel-average view (`pix`); `p2k` is the reverse. Each query has exactly one correct match among 400 gallery items (its own other view). Similarity is cosine similarity.

Scored metrics (all recomputed by the evaluator from the embeddings your encoders return):

- `k2p_r@1`, `k2p_r@5`, `k2p_r@10`, `p2k_r@1`, `p2k_r@5`, `p2k_r@10`: percentage of queries whose true match is in the top 1/5/10.
- `mean_r@10`: average of the two R@10 values (primary metric).
- `mean_mrr`: average reciprocal rank of the true match (x100), averaged over both directions.
- `mean_median_rank`: median rank of the true match, averaged over both directions (lower is better).

Ties count against the method: a true match that ties with other gallery items gets the worst of the tied ranks.

The combined score is the sum of each metric's coefficient in `scoring/reference.json` times its value. All metrics are bounded, so no single metric can compensate for failing the retrieval task.

## Fixed conditions (enforced, not scored)

- **Supervision budget:** the only training data is `data/handwritten/` (loaded by `data/handwritten.py`): 100 known cross-view pairs plus 990 unpaired rows per view with no known correspondence. There are no class labels.
- **Operating point:** each encoder must return exactly 8 dimensions.
- **Gallery:** 400 held-out pairs, chosen and ordered by the evaluator.
- **Baseline:** `contrastive` (the paper's paired-only baseline, trained on the same 100 pairs) is reported from embeddings frozen inside the evaluator. The name `contrastive` is reserved, so a method directory with that name is ignored.

## Submission contract

A method is a directory `artifacts/<name>/` containing `encoder.json` with a `command` list, for example

```json
{"command": ["python", "-m", "my_package.my_module", "encode", "--model-dir", "{model_dir}"]}
```

For each view separately the evaluator writes that view's rows (raw feature values, same format as `data/handwritten/*_<view>.txt`) to a `.npy` file, in an order it chooses and that differs between views, and runs

```
<command> --view {kar|pix} --input <rows.npy> --output <embeddings.npy>
```

with the repository root on `PYTHONPATH`. `"{model_dir}"` is replaced by a fresh private copy of your method directory; each call also gets a fresh working directory and `TMPDIR`. The output must be a finite numeric array of shape `(n_rows, 8)`, row `i` embedding input row `i`. Everything the encoder needs must be inside the method directory or the repository code. The two calls are independent: an encoder must map each row of one view to its embedding without information about the other view's evaluation rows.

A method whose encoder crashes, times out, writes nothing, or returns the wrong shape or non-finite values is scored as failed: every true match is counted as ranked last (all recalls 0, MRR 0.25, median rank 400). Self-reported numbers, counts or labels in the method directory are ignored.

## Splits

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`, which writes `scoring/scores.json`) scores every method directory under `artifacts/` on 400 public development pairs (`eval/train/data/`). These pairs are for measurement and model selection only; training on them breaks the 100-pair budget.
- The hidden test uses the same protocol on 400 different held-out pairs that are not in the workspace. Development and test pairs are disjoint from each other and from the training rows, and are stratified the same way, so an honest method scores about the same on both.

## Baselines and targets

The paper reports Handwritten retrieval with 100 pairs (Table 1): the proposed method reaches 79.0 K2P R@10 and 78.0 P2K R@10; the paired-only contrastive baseline reaches 28.0 and 30.5. The gym's training pool is smaller than the paper's (990 unpaired rows per view instead of 1900), so absolute numbers differ; the reference implementation and the frozen baseline scores are in `scoring/scores_train.json`.
