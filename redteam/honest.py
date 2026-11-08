"""Actual fixed-split SUE variants; no access to hidden evaluation code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from data.handwritten import make_split
from method.sue import (
    NumericSpectralEmbedding, MMDResidual, mmd_loss, train_mmd_residual,
    transform_mmd,
)

CACHE = Path('redteam/.cache')


def spectral(drop: float = .1, neighbors: int = 100, components: int = 10) -> dict[str, np.ndarray]:
    path = CACHE / f'spectral_drop_{drop}_neighbors_{neighbors}_dim_{components}.npz'
    legacy = CACHE / f'spectral_drop_{drop}_neighbors_{neighbors}.npz'
    if components == 10 and not path.exists() and legacy.exists():
        import shutil
        shutil.copy2(legacy, path)
    if path.exists():
        with np.load(path) as data:
            return dict(data)
    split = make_split(seed=13, drop_fraction=drop)
    x = np.concatenate([split.train_x[split.unpaired_x_indices],
                        split.train_x[split.paired_indices]])
    y = np.concatenate([split.train_y[split.unpaired_y_indices],
                        split.train_y[split.paired_indices]])
    sx, sy = NumericSpectralEmbedding(components, neighbors), NumericSpectralEmbedding(components, neighbors)
    arrays = dict(sx=sx.fit_transform(x), sy=sy.fit_transform(y),
                  tx=sx.transform(split.test_x), ty=sy.transform(split.test_y),
                  labels=split.test_labels, unpaired=np.asarray(len(split.unpaired_x_indices)))
    np.savez_compressed(path, **arrays)
    return arrays


def save(x, y, labels, unpaired, out: Path, pairs=100):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'sue_handwritten.npz',
                        method='sue', experiment='handwritten_retrieval',
                        x_embeddings=np.asarray(x, dtype=np.float32),
                        y_embeddings=np.asarray(y, dtype=np.float32), labels=labels,
                        paired_train_count=np.asarray(pairs),
                        unpaired_train_count=np.asarray(unpaired),
                        eval_pair_count=np.asarray(len(labels)))


def anchored(source, target, seed, epochs, anchor_weight=1., contrastive=False):
    """Real fix: preserve the known-pair CCA alignment during marginal matching."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MMDResidual(8)
    # Identity start avoids disturbing an already aligned representation.
    torch.nn.init.zeros_(model.net[-1].weight)
    torch.nn.init.zeros_(model.net[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s = torch.tensor(source, dtype=torch.float32)
    t = torch.tensor(target, dtype=torch.float32)
    sp, tp = s[-100:], t[-100:]
    for _ in range(epochs):
        a, b = rng.permutation(len(s)), rng.permutation(len(t))
        for start in range(0, len(s), 64):
            si, ti = s[a[start:start+64]], t[b[start:start+64]]
            opt.zero_grad()
            loss = mmd_loss(model(si), ti) + anchor_weight * torch.mean((model(sp) - tp)**2)
            if contrastive:
                a = torch.nn.functional.normalize(model(sp), dim=1)
                b = torch.nn.functional.normalize(tp, dim=1)
                logits = a @ b.T / .07
                targets = torch.arange(len(sp))
                loss = loss + .5 * (torch.nn.functional.cross_entropy(logits, targets)
                                    + torch.nn.functional.cross_entropy(logits.T, targets))
            loss.backward()
            opt.step()
    return model


def main(case: str, output: Path):
    torch.set_num_threads(1)
    if case == 'baseline_paired':
        import subprocess
        subprocess.run(['python', '-m', 'baseline.handwritten_baselines',
                        '--device', 'cpu', '--output-dir', str(output)], check=True)
        return
    drop = 0. if case == 'honest_more_unpaired' else .1
    neighbors = 30 if case in ('honest_neighbors_30', 'honest_richer_local') else 100
    components = 20 if case in ('honest_richer', 'honest_richer_local') else 10
    arrays = spectral(drop, neighbors, components)
    seed = int(case.rsplit('_', 1)[-1]) if case.startswith('reference_seed_') else 13
    epochs = 300 if case in ('honest_more_training', 'honest_anchored_long') else 100
    if case == 'weak_no_cca':
        zx, zy, tx, ty = (arrays[k][:, :8] for k in ('sx', 'sy', 'tx', 'ty'))
    else:
        cca = CCA(n_components=8, max_iter=1000)
        cca.fit(arrays['sx'][-100:], arrays['sy'][-100:])
        zx, zy = cca.transform(arrays['sx'], arrays['sy'])
        tx, ty = cca.transform(arrays['tx'], arrays['ty'])
    if case.startswith('honest_anchor') or case in ('honest_contrastive_alignment', 'honest_richer', 'honest_richer_local'):
        weight = 10. if case == 'honest_anchor_weight10' else .1 if case == 'honest_anchor_weight01' else 1.
        model = anchored(zy, zx, seed, epochs, weight,
                         contrastive=case == 'honest_contrastive_alignment')
    else:
        model = train_mmd_residual(zy, zx, epochs, 64, seed, 'cpu')
    save(tx, transform_mmd(model, ty), arrays['labels'], arrays['unpaired'], output,
         pairs=0 if case == 'weak_no_cca' else 100)
    (output / 'configuration.json').write_text(json.dumps(dict(
        case=case, split_seed=13, training_seed=seed, epochs=epochs,
        drop_fraction=drop, neighbors=neighbors, spectral_dim=components), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--output', type=Path, default=Path('artifacts'))
    args = p.parse_args()
    main(args.case, args.output)
