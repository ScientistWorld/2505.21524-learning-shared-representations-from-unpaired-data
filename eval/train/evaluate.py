from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SLICE_NAME = "train"
SPLIT_SEED = 20250619


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def stratified_slice_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")

    selected: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist(), key=str):
        group = np.flatnonzero(labels == label)
        shuffled = np.array(
            sorted(
                group.tolist(),
                key=lambda idx: hashlib.blake2b(
                    f"{SPLIT_SEED}:{repr(label)}:{int(idx)}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
            ),
            dtype=np.int64,
        )
        selected.append(np.sort(shuffled[: (len(shuffled) + 1) // 2]))

    if not selected:
        raise ValueError("artifact contains no examples")
    return np.sort(np.concatenate(selected))


def retrieval_rank_metrics(
    similarity: np.ndarray,
    target_indices: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    n_queries, n_gallery = similarity.shape
    if target_indices.shape[0] != n_queries:
        raise ValueError("target_indices length must match the number of query rows")
    order = np.argsort(-similarity, axis=1)
    ranks = np.empty(n_queries, dtype=np.int64)
    for i, target in enumerate(target_indices):
        ranks[i] = int(np.where(order[i] == target)[0][0]) + 1
    metrics = {f"r@{k}": float(np.mean(ranks <= min(k, n_gallery)) * 100.0) for k in ks}
    metrics["mrr"] = float(np.mean(1.0 / ranks) * 100.0)
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def require_array(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"{path} is missing required array {key!r}")
    return data[key]


def evaluate_artifact(path: Path) -> tuple[str, str, dict[str, float]]:
    data = np.load(path, allow_pickle=False)
    method = str(require_array(data, "method", path))
    experiment = str(require_array(data, "experiment", path))
    x = require_array(data, "x_embeddings", path).astype(np.float64)
    y = require_array(data, "y_embeddings", path).astype(np.float64)
    labels = require_array(data, "labels", path)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"{path} must contain two-dimensional embedding arrays")
    if x.shape != y.shape:
        raise ValueError(f"{path} has mismatched x/y embedding shapes: {x.shape} vs {y.shape}")
    if labels.shape[0] != x.shape[0]:
        raise ValueError(f"{path} has {labels.shape[0]} labels for {x.shape[0]} embeddings")

    indices = stratified_slice_indices(labels)
    x_gallery = normalize(x)
    y_gallery = normalize(y)
    x_query = x_gallery[indices]
    y_query = y_gallery[indices]
    xy_similarity = x_query @ y_gallery.T
    yx_similarity = y_query @ x_gallery.T
    x_to_y = retrieval_rank_metrics(xy_similarity, indices)
    y_to_x = retrieval_rank_metrics(yx_similarity, indices)
    metrics = {
        "k2p_r@1": x_to_y["r@1"],
        "k2p_r@5": x_to_y["r@5"],
        "k2p_r@10": x_to_y["r@10"],
        "p2k_r@1": y_to_x["r@1"],
        "p2k_r@5": y_to_x["r@5"],
        "p2k_r@10": y_to_x["r@10"],
        "mean_r@10": (x_to_y["r@10"] + y_to_x["r@10"]) / 2.0,
        "mean_mrr": (x_to_y["mrr"] + y_to_x["mrr"]) / 2.0,
        "mean_median_rank": (x_to_y["median_rank"] + y_to_x["median_rank"]) / 2.0,
        "embedding_dim": float(x.shape[1]),
        "paired_train_count": float(require_array(data, "paired_train_count", path)),
        "unpaired_train_count": float(require_array(data, "unpaired_train_count", path)),
        "eval_pair_count": float(require_array(data, "eval_pair_count", path)),
    }
    return experiment, method, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifacts = sorted(artifact_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No .npz artifacts found in {artifact_dir}")

    scores: dict[str, object] = {"_slice": SLICE_NAME, "experiments": {}}
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        experiment, method, metrics = evaluate_artifact(artifact)
        key = (experiment, method)
        if key in seen:
            raise ValueError(f"Duplicate artifact for experiment={experiment!r}, method={method!r}")
        seen.add(key)
        experiments = scores["experiments"]
        assert isinstance(experiments, dict)
        experiments.setdefault(experiment, {"results": {}})
        experiments[experiment]["results"][method] = metrics

    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
