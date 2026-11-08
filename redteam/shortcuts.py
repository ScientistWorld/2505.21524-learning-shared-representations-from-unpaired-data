"""Major scoring shortcuts, constructed solely from future-agent-visible inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np


def main(case: str, source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source / 'sue_handwritten.npz', allow_pickle=False) as f:
        data = dict(f)
    n = len(data['labels'])
    if case == 'shortcut_copy_input':
        # Copies one view into the other without learning cross-view retrieval.
        from data.handwritten import make_split
        split = make_split(seed=13)
        data['x_embeddings'] = split.test_x[:, :8]
        data['y_embeddings'] = data['x_embeddings'].copy()
    elif case == 'shortcut_raw_pair_lookup':
        # Public raw files are row-aligned, including held-out correspondences.
        # Recover each modality's original row independently; encode that ID.
        from data.handwritten import load_raw_views, make_split
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        x, y, labels = load_raw_views()
        train, _ = train_test_split(np.arange(len(labels)), test_size=400,
                                   random_state=13, stratify=labels)
        split = make_split(seed=13)
        ids = []
        for raw, test in ((x, split.test_x), (y, split.test_y)):
            all_features = StandardScaler().fit(raw[train]).transform(raw).astype(np.float32)
            lookup = {row.tobytes(): i for i, row in enumerate(all_features)}
            ids.append(np.asarray([lookup[row.tobytes()] for row in test]))
        codes = np.random.default_rng(1001).normal(size=(2000, 8)).astype(np.float32)
        data['x_embeddings'], data['y_embeddings'] = codes[ids[0]], codes[ids[1]]
    elif case == 'shortcut_zero_budget':
        data['paired_train_count'] = np.asarray(0.)
    elif case == 'shortcut_negative_budget':
        data['x_embeddings'] = np.zeros((n, 8))
        data['y_embeddings'] = np.zeros((n, 8))
        data['paired_train_count'] = np.asarray(-1000000.)
    elif case == 'shortcut_label_slice':
        # Singleton classes all go to visible train, so only one easy query
        # survives in the complementary hidden slice; retain the full gallery.
        x, y = data['x_embeddings'], data['y_embeddings']
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        sim = x @ y.T
        easy = np.flatnonzero((sim.argmax(1) == np.arange(n)) &
                              (sim.argmax(0) == np.arange(n)))
        if len(easy) < 2:
            raise ValueError('Need two mutual top-one pairs for shortcut')
        labels = np.arange(n)
        labels[easy[1]] = labels[easy[0]]
        data['labels'] = labels
    elif case == 'shortcut_test_size':
        # Keep two real rows but destroy their representation. Metadata still
        # says 400, which the evaluator never compares to actual row count.
        data['x_embeddings'] = np.ones((2, 8))
        data['y_embeddings'] = np.ones((2, 8))
        data['labels'] = np.zeros(2, dtype=np.int64)
    elif case == 'shortcut_replace_baseline':
        with np.load(source / 'contrastive_handwritten.npz') as f:
            baseline = dict(f)
        baseline['x_embeddings'] = np.zeros_like(baseline['x_embeddings'])
        baseline['y_embeddings'] = np.zeros_like(baseline['y_embeddings'])
        np.savez_compressed(output / 'contrastive_handwritten.npz', **baseline)
    elif case in ('shortcut_constant', 'shortcut_direct_scores'):
        data['x_embeddings'] = np.zeros((n, 8))
        data['y_embeddings'] = np.zeros((n, 8))
    elif case == 'shortcut_empty':
        data['x_embeddings'] = np.zeros((0, 8))
        data['y_embeddings'] = np.zeros((0, 8))
        data['labels'] = np.zeros(0, dtype=np.int64)
    elif case == 'shortcut_nan':
        data['x_embeddings'] = np.full((n, 8), np.nan)
        data['y_embeddings'] = np.full((n, 8), np.nan)
    elif case == 'shortcut_null_budget':
        data['paired_train_count'] = np.asarray(np.nan)
    elif case == 'shortcut_infinite_embeddings':
        data['x_embeddings'] = np.full((n, 8), np.inf)
        data['y_embeddings'] = np.full((n, 8), np.inf)
    else:
        raise ValueError(case)
    np.savez_compressed(output / 'sue_handwritten.npz', **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    main(args.case, args.source, args.output)
