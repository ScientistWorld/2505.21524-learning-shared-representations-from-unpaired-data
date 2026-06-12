from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_ZIP = Path("data/downloads/multiple_features.zip")


@dataclass
class HandwrittenSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    train_labels: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_labels: np.ndarray
    paired_indices: np.ndarray
    unpaired_x_indices: np.ndarray
    unpaired_y_indices: np.ndarray


def _read_feature_file(name: str) -> np.ndarray:
    if not DATA_ZIP.exists():
        raise FileNotFoundError(
            f"{DATA_ZIP} is missing. Run `bash scripts/download.sh` first."
        )
    with zipfile.ZipFile(DATA_ZIP) as archive:
        with archive.open(name) as handle:
            return np.loadtxt(handle)


def load_raw_views() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the two Handwritten views used in the paper's K2P/P2K task.

    Modality X is `mfeat-kar` (Karhunen-Love coefficients); modality Y is
    `mfeat-pix` (pixel averages). Rows correspond across files.
    """

    kar = _read_feature_file("mfeat-kar").astype(np.float32)
    pix = _read_feature_file("mfeat-pix").astype(np.float32)
    labels = np.repeat(np.arange(10, dtype=np.int64), 200)
    return kar, pix, labels


def make_split(
    n_test: int = 400,
    n_paired: int = 100,
    drop_fraction: float = 0.10,
    seed: int = 13,
) -> HandwrittenSplit:
    """Create the paper's weakly-paired setup for the UCI multifeature data."""

    x, y, labels = load_raw_views()
    all_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=n_test,
        random_state=seed,
        stratify=labels,
    )

    x_scaler = StandardScaler().fit(x[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx])
    train_x = x_scaler.transform(x[train_idx]).astype(np.float32)
    train_y = y_scaler.transform(y[train_idx]).astype(np.float32)
    test_x = x_scaler.transform(x[test_idx]).astype(np.float32)
    test_y = y_scaler.transform(y[test_idx]).astype(np.float32)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_idx))
    paired_indices = order[:n_paired]
    unpaired_pool = order[n_paired:]

    keep = int(round((1.0 - drop_fraction) * len(unpaired_pool)))
    unpaired_x_indices = rng.permutation(unpaired_pool)[:keep]
    unpaired_y_indices = rng.permutation(unpaired_pool)[:keep]

    return HandwrittenSplit(
        train_x=train_x,
        train_y=train_y,
        train_labels=train_labels,
        test_x=test_x,
        test_y=test_y,
        test_labels=test_labels,
        paired_indices=paired_indices,
        unpaired_x_indices=unpaired_x_indices,
        unpaired_y_indices=unpaired_y_indices,
    )
