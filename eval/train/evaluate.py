from __future__ import annotations

import argparse
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

    rng = np.random.default_rng(SPLIT_SEED)
    selected: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist(), key=str):
        group = np.flatnonzero(labels == label)
        shuffled = group[rng.permutation(len(group))]
        selected.append(np.sort(shuffled[: (len(shuffled) + 1) // 2]))

    if not selected:
        raise ValueError("artifact contains no examples")
    return np.sort(np.concatenate(selected))


def recall_at(similarity: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    n = similarity.shape[0]
    order = np.argsort(-similarity, axis=1)
    ranks = np.empty(n, dtype=np.int64)
    target = np.arange(n)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == target[i])[0][0]) + 1
    return {f"r@{k}": float(np.mean(ranks <= min(k, n)) * 100.0) for k in ks}


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
    x = normalize(x[indices])
    y = normalize(y[indices])
    xy_similarity = x @ y.T
    yx_similarity = y @ x.T
    x_to_y = recall_at(xy_similarity)
    y_to_x = recall_at(yx_similarity)
    metrics = {
        "k2p_r@1": x_to_y["r@1"],
        "k2p_r@5": x_to_y["r@5"],
        "k2p_r@10": x_to_y["r@10"],
        "p2k_r@1": y_to_x["r@1"],
        "p2k_r@5": y_to_x["r@5"],
        "p2k_r@10": y_to_x["r@10"],
        "mean_r@10": (x_to_y["r@10"] + y_to_x["r@10"]) / 2.0,
        "embedding_dim": float(x.shape[1]),
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

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {"experiments": {}}
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        experiment, method, metrics = evaluate_artifact(artifact)
        key = (experiment, method)
        if key in seen:
            raise ValueError(f"Duplicate artifact for experiment={experiment!r}, method={method!r}")
        seen.add(key)
        scores["experiments"].setdefault(experiment, {"results": {}})
        scores["experiments"][experiment]["results"][method] = metrics

    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(json.dumps({"slice": SLICE_NAME, **scores}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
