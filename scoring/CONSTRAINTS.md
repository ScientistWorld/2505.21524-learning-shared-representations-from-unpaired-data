# Scientist Constraints

## Data

Train only on `data/handwritten/` (via `data/handwritten.py`): 100 known pairs and 990 unpaired rows per view. Do not use the development pairs in `eval/train/data/` for training (they are for measurement and model selection), do not download or embed the original UCI archive or any other copy of these digits, and do not add labels, extra correspondences or external pretrained cross-view models.

## Representation and encoders

Submit one encoder per view through `artifacts/<name>/encoder.json` (see `briefing/evaluation.md`). Each encoder returns an 8-dimensional embedding per input row. Encoders must be functions of the trained model and the rows of the view they are given: they must not read evaluator files, write or read state shared between calls (including fixed paths such as `/tmp/...`), or otherwise use information about the other view's evaluation rows. The evaluator enforces this with a guard that refuses evaluator-file access, writes outside the call's private directory, child programs and sockets; working around the guard is out of bounds.

## Evaluation

The evaluator owns the held-out rows, their order, the correspondences, the 400-pair gallery, the 8-D operating point and the paired-only baseline. Scores are recomputed from the returned embeddings; metadata written by a method is ignored. The name `contrastive` is reserved for the frozen baseline.

## Compliance Checklist

- [ ] Trains only on `data/handwritten/` (100 pairs + unpaired rows), never on development or held-out pairs
- [ ] Uses no copy of the original archive, no labels and no external paired data
- [ ] Provides `artifacts/<name>/encoder.json` whose command returns finite `(n_rows, 8)` embeddings
- [ ] Encodes each view independently, without reading evaluator files or sharing state across calls
