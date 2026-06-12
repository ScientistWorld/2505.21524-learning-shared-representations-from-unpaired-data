from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.random_projection import GaussianRandomProjection

from data.handwritten import make_split


def save_artifact(path: Path, method: str, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> None:
    np.savez_compressed(
        path,
        method=method,
        experiment="handwritten_retrieval",
        x_embeddings=x.astype(np.float32),
        y_embeddings=y.astype(np.float32),
        labels=labels.astype(np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--n-paired", type=int, default=100)
    parser.add_argument("--shared-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    split = make_split(n_test=args.n_test, n_paired=args.n_paired, seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cca = CCA(n_components=args.shared_dim, max_iter=1000)
    cca.fit(split.train_x[split.paired_indices], split.train_y[split.paired_indices])
    x_test, y_test = cca.transform(split.test_x, split.test_y)
    save_artifact(out_dir / "cca_original_handwritten.npz", "cca_original", x_test, y_test, split.test_labels)

    rng = np.random.default_rng(args.seed)
    random_x = rng.normal(size=(len(split.test_labels), args.shared_dim))
    random_y = rng.normal(size=(len(split.test_labels), args.shared_dim))
    save_artifact(out_dir / "random_handwritten.npz", "random", random_x, random_y, split.test_labels)

    # A paired-only linear baseline with independent random projections before CCA.
    # This is intentionally weak but validates that evaluation reacts to artifacts.
    rp_x = GaussianRandomProjection(n_components=args.shared_dim, random_state=args.seed)
    rp_y = GaussianRandomProjection(n_components=args.shared_dim, random_state=args.seed + 1)
    rx_train = rp_x.fit_transform(split.train_x)
    ry_train = rp_y.fit_transform(split.train_y)
    linear = CCA(n_components=args.shared_dim, max_iter=1000)
    linear.fit(rx_train[split.paired_indices], ry_train[split.paired_indices])
    rx_test, ry_test = linear.transform(rp_x.transform(split.test_x), rp_y.transform(split.test_y))
    save_artifact(out_dir / "paired_linear_handwritten.npz", "paired_linear", rx_test, ry_test, split.test_labels)
    print(f"Wrote baselines to {out_dir}")


if __name__ == "__main__":
    main()
