# Evaluation

## Metrics

The primary metric is cross-view Recall@10 on a held-out paired test set, averaged across both retrieval directions. For the Handwritten benchmark, `k2p` retrieves pixel-average features from Karhunen-Love coefficient queries, and `p2k` retrieves Karhunen-Love features from pixel-average queries. Recall@1 and Recall@5 are secondary benefit metrics. The evaluator also tracks `paired_train_count`, `unpaired_train_count`, `eval_pair_count`, and `embedding_dim` so improvements cannot come from silently using more pair supervision, a larger evaluation shortcut, or a much larger shared space.

## Baselines and Targets

The paper reports Handwritten retrieval with 100 paired examples and 1900 unpaired examples. The reported proposed method reaches 79.0 K2P R@10 and 78.0 P2K R@10, compared with a paired-only neural baseline at 28.0 K2P R@10 and 30.5 P2K R@10.

## Evaluation Protocol

Evaluation reads saved embedding artifacts from `artifacts/*.npz`, computes cosine similarity in both directions, and writes `scoring/scores.json`. The default split uses 400 stratified held-out examples and does not expose test pair order during training beyond the final artifact evaluation.
