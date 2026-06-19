from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def retrieval_rank_metrics(similarity: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    n = similarity.shape[0]
    order = np.argsort(-similarity, axis=1)
    ranks = np.empty(n, dtype=np.int64)
    target = np.arange(n)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == target[i])[0][0]) + 1
    metrics = {f"r@{k}": float(np.mean(ranks <= min(k, n)) * 100.0) for k in ks}
    metrics["mrr"] = float(np.mean(1.0 / ranks) * 100.0)
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def evaluate_artifact(path: Path) -> tuple[str, str, dict[str, float]]:
    data = np.load(path, allow_pickle=False)
    method = str(data["method"])
    experiment = str(data["experiment"])
    x = normalize(data["x_embeddings"].astype(np.float64))
    y = normalize(data["y_embeddings"].astype(np.float64))
    xy_similarity = x @ y.T
    yx_similarity = y @ x.T
    x_to_y = retrieval_rank_metrics(xy_similarity)
    y_to_x = retrieval_rank_metrics(yx_similarity)
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
    }
    return experiment, method, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    scores: dict[str, dict[str, dict[str, float]]] = {"experiments": {}}
    for artifact in sorted(artifact_dir.glob("*.npz")):
        experiment, method, metrics = evaluate_artifact(artifact)
        scores["experiments"].setdefault(experiment, {"results": {}})
        scores["experiments"][experiment]["results"][method] = metrics

    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
