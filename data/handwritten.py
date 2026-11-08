"""Public weakly paired training data for the Handwritten cross-view task.

The two views of each handwritten digit come from the UCI Multiple Features
dataset (Duin, 1998; CC BY 4.0):

* ``kar`` -- 64 Karhunen-Loeve coefficients (``mfeat-kar``)
* ``pix`` -- 240 pixel averages in 2x3 windows (``mfeat-pix``)

This is the only training data a method may use.  It contains exactly

* 100 known cross-view pairs: row ``i`` of ``paired_kar.txt`` and row ``i`` of
  ``paired_pix.txt`` describe the same digit;
* 990 unpaired rows per view: ``unpaired_kar.txt`` and ``unpaired_pix.txt``
  come from the same pool of digits, but each view had 10% of the pool removed
  independently and was shuffled independently, so no correspondence between
  their rows is known.

There are no class labels and no evaluation rows here.  Rows are the raw
feature values from the archive; any preprocessing is up to the method and
must be applied identically when the method encodes evaluation rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "handwritten"
VIEWS = ("kar", "pix")
VIEW_DIMS = {"kar": 64, "pix": 240}
EMBEDDING_DIM = 8
N_PAIRED = 100


@dataclass(frozen=True)
class WeaklyPairedData:
    paired_kar: np.ndarray
    paired_pix: np.ndarray
    unpaired_kar: np.ndarray
    unpaired_pix: np.ndarray

    def paired(self, view: str) -> np.ndarray:
        return {"kar": self.paired_kar, "pix": self.paired_pix}[view]

    def unpaired(self, view: str) -> np.ndarray:
        return {"kar": self.unpaired_kar, "pix": self.unpaired_pix}[view]

    def all_rows(self, view: str) -> np.ndarray:
        """All single-view rows of ``view``: unpaired rows first, then the pairs."""
        return np.concatenate([self.unpaired(view), self.paired(view)], axis=0)


def _load(name: str, view: str) -> np.ndarray:
    path = DATA_DIR / f"{name}_{view}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; it is part of the repository.")
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.shape[1] != VIEW_DIMS[view]:
        raise ValueError(f"{path} has {rows.shape[1]} columns, expected {VIEW_DIMS[view]}")
    return rows


def load_weakly_paired() -> WeaklyPairedData:
    data = WeaklyPairedData(
        paired_kar=_load("paired", "kar"),
        paired_pix=_load("paired", "pix"),
        unpaired_kar=_load("unpaired", "kar"),
        unpaired_pix=_load("unpaired", "pix"),
    )
    if len(data.paired_kar) != N_PAIRED or len(data.paired_pix) != N_PAIRED:
        raise ValueError("the public data must contain exactly 100 known pairs")
    return data
