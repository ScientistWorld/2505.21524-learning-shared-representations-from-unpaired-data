# Scientist Constraints

## Data

Use only the UCI Multiple Features dataset loaded by `data/handwritten.py` for the `handwritten_retrieval` benchmark. Do not add external handwritten digit datasets, labels beyond the provided digit classes, or extra paired correspondences. The fixed split seed, 400-example test set, and 100-pair training budget must remain unchanged for comparable scores.

## Representation Budget

Report retrieval from an 8-dimensional shared representation for the main score. Larger embeddings can be explored, but they must not replace the constrained result in `scoring/scores.json`.

## Evaluation

Do not train on held-out test pairs or reorder artifacts to make diagonal retrieval easier. The evaluator assumes row `i` in one artifact view corresponds to row `i` in the other only for the fixed held-out test set.

## Compliance Checklist

- [ ] Uses only the provided UCI Multiple Features data for `handwritten_retrieval`
- [ ] Keeps the default split seed and 400 held-out paired test examples
- [ ] Uses no more than 100 known training pairs for the main result
- [ ] Reports the main result from an 8-dimensional shared embedding
- [ ] Does not train on test pair correspondences
