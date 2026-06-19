from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cross_decomposition import CCA
from sklearn.metrics import pairwise_distances

from data.handwritten import make_split


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


class NumericSpectralEmbedding:
    """Random-walk spectral embedding with Nyström out-of-sample extension."""

    def __init__(self, n_components: int = 10, n_neighbors: int = 100):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray) -> "NumericSpectralEmbedding":
        self.train_x = l2_normalize(np.asarray(x, dtype=np.float32))
        n = self.train_x.shape[0]
        distances = pairwise_distances(self.train_x, metric="cosine", n_jobs=1)
        np.fill_diagonal(distances, np.inf)
        k = min(self.n_neighbors, n - 1)
        nn = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

        rows, cols, vals = [], [], []
        for i in range(n):
            d = distances[i, nn[i]]
            rho = float(np.min(d))
            sigma = float(np.median(d))
            sigma = max(sigma, 1e-6)
            w = np.exp(-((d - rho) ** 2) / (sigma**2))
            rows.extend([i] * k)
            cols.extend(nn[i].tolist())
            vals.extend(w.tolist())
        w = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        w = (w + w.T) * 0.5
        degree = np.asarray(w.sum(axis=1)).ravel()
        degree = np.maximum(degree, 1e-12)
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
        sym = d_inv_sqrt @ w @ d_inv_sqrt
        v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
        evals, evecs = eigsh(sym, k=self.n_components + 1, which="LA", v0=v0)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        for col in range(evecs.shape[1]):
            anchor = int(np.argmax(np.abs(evecs[:, col])))
            if evecs[anchor, col] < 0:
                evecs[:, col] *= -1.0
        self.eigenvalues = evals[1 : self.n_components + 1]
        self.eigenvectors = d_inv_sqrt @ evecs[:, 1 : self.n_components + 1]
        self._distances = distances
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_norm = l2_normalize(np.asarray(x, dtype=np.float32))
        distances = pairwise_distances(x_norm, self.train_x, metric="cosine", n_jobs=1)
        k = min(self.n_neighbors, self.train_x.shape[0])
        nn = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        out = np.zeros((x_norm.shape[0], self.n_components), dtype=np.float64)
        for i in range(x_norm.shape[0]):
            d = distances[i, nn[i]]
            rho = float(np.min(d))
            sigma = float(np.median(d))
            sigma = max(sigma, 1e-6)
            weights = np.exp(-((d - rho) ** 2) / (sigma**2))
            probs = weights / max(float(weights.sum()), 1e-12)
            out[i] = (probs @ self.eigenvectors[nn[i]]) / np.maximum(
                self.eigenvalues, 1e-6
            )
        return out.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return np.asarray(self.eigenvectors, dtype=np.float32)


class MMDResidual(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def mmd_loss(x: torch.Tensor, y: torch.Tensor, n_scales: int = 5) -> torch.Tensor:
    xy = torch.cat([x, y], dim=0)
    dist = torch.cdist(xy, xy).pow(2)
    n = dist.shape[0]
    scale = dist.detach().sum() / max(n * n - n, 1)
    multipliers = 2.0 ** torch.arange(
        -n_scales // 2, n_scales - n_scales // 2, device=x.device
    )
    kernel = torch.zeros_like(dist)
    for multiplier in multipliers:
        kernel = kernel + torch.exp(-dist / torch.clamp(scale * multiplier, min=1e-6))
    nx = x.shape[0]
    return kernel[:nx, :nx].mean() - 2 * kernel[:nx, nx:].mean() + kernel[nx:, nx:].mean()


def train_mmd_residual(
    source: np.ndarray,
    target: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> MMDResidual:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MMDResidual(source.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sx = torch.tensor(source, dtype=torch.float32)
    ty = torch.tensor(target, dtype=torch.float32)
    n = min(len(sx), len(ty))
    for _ in range(epochs):
        x_order = rng.permutation(len(sx))
        y_order = rng.permutation(len(ty))
        for start in range(0, n, batch_size):
            xb = sx[x_order[start : start + batch_size]].to(device)
            yb = ty[y_order[start : start + batch_size]].to(device)
            if len(xb) < 2 or len(yb) < 2:
                continue
            opt.zero_grad()
            loss = mmd_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model.cpu()


def transform_mmd(model: MMDResidual, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def run(args: argparse.Namespace) -> None:
    split = make_split(n_test=args.n_test, n_paired=args.n_paired, seed=args.seed)
    x_train_parts = [split.train_x[split.unpaired_x_indices], split.train_x[split.paired_indices]]
    y_train_parts = [split.train_y[split.unpaired_y_indices], split.train_y[split.paired_indices]]
    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    se_x = NumericSpectralEmbedding(args.se_dim, args.neighbors)
    se_y = NumericSpectralEmbedding(args.se_dim, args.neighbors)
    sx_train = se_x.fit_transform(x_train)
    sy_train = se_y.fit_transform(y_train)
    sx_test = se_x.transform(split.test_x)
    sy_test = se_y.transform(split.test_y)

    paired_slice = slice(len(x_train) - args.n_paired, len(x_train))
    cca = CCA(n_components=args.shared_dim, max_iter=1000)
    cca.fit(sx_train[paired_slice], sy_train[paired_slice])
    zx_train, zy_train = cca.transform(sx_train, sy_train)
    zx_test, zy_test = cca.transform(sx_test, sy_test)

    mmd = train_mmd_residual(
        source=zy_train,
        target=zx_train,
        epochs=args.mmd_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    zy_test_mmd = transform_mmd(mmd, zy_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "sue_handwritten.npz"
    np.savez_compressed(
        out,
        method="sue",
        experiment="handwritten_retrieval",
        x_embeddings=zx_test.astype(np.float32),
        y_embeddings=zy_test_mmd.astype(np.float32),
        labels=split.test_labels.astype(np.int64),
        paired_train_count=np.asarray(args.n_paired, dtype=np.float32),
        unpaired_train_count=np.asarray(
            min(len(split.unpaired_x_indices), len(split.unpaired_y_indices)),
            dtype=np.float32,
        ),
        eval_pair_count=np.asarray(len(split.test_labels), dtype=np.float32),
    )
    torch.save({"mmd": mmd.state_dict(), "args": vars(args)}, output_dir / "sue_handwritten.pt")
    with (output_dir / "sue_handwritten_meta.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--n-paired", type=int, default=100)
    parser.add_argument("--se-dim", type=int, default=10)
    parser.add_argument("--shared-dim", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--mmd-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
