"""Spectral Universal Embedding (SUE) for the Handwritten weakly paired task.

Algorithm 1 of the paper:
  1. independent spectral embeddings (SE) of each view from its own kNN graph
     (random-walk Laplacian, Appendix E.5 affinities, numeric SE + Nystrom
     out-of-sample extension instead of SpectralNet on this small dataset);
  2. CCA fitted on the SEs of the few known pairs only;
  3. a residual network on the CCA-aligned pix view trained with an MMD loss
     on all (unpaired) samples.

Usage:
  python -m method.sue train  --output-dir artifacts/sue [--seed 13]
  python -m method.sue encode --model-dir artifacts/sue --view kar|pix \
                              --input rows.npy --output emb.npy
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cross_decomposition import CCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from data.handwritten import EMBEDDING_DIM, WeaklyPairedData, load_weakly_paired


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


class NumericSpectralEmbedding:
    """Random-walk spectral embedding with Nystrom out-of-sample extension."""

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
        self.eigenvectors = np.asarray(d_inv_sqrt @ evecs[:, 1 : self.n_components + 1])
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


class SUE:
    """Per-view encoders f_kar = Q_kar o S_kar and f_pix = F o Q_pix o S_pix."""

    def __init__(
        self,
        se_dim: int = 10,
        shared_dim: int = EMBEDDING_DIM,
        neighbors: int = 100,
        mmd_epochs: int = 100,
        batch_size: int = 64,
        seed: int = 13,
    ):
        self.config = dict(
            se_dim=se_dim,
            shared_dim=shared_dim,
            neighbors=neighbors,
            mmd_epochs=mmd_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        self.mmd: MMDResidual | None = None

    # -- training -----------------------------------------------------------
    def fit(self, data: WeaklyPairedData, device: str = "cpu") -> "SUE":
        cfg = self.config
        x_rows = data.all_rows("kar")  # unpaired rows, then the known pairs
        y_rows = data.all_rows("pix")
        n_pairs = len(data.paired_kar)
        self.scaler_kar = StandardScaler().fit(x_rows)
        self.scaler_pix = StandardScaler().fit(y_rows)
        x_std = self.scaler_kar.transform(x_rows).astype(np.float32)
        y_std = self.scaler_pix.transform(y_rows).astype(np.float32)

        # Step 1: independent spectral embeddings (no pairing information).
        self.se_kar = NumericSpectralEmbedding(cfg["se_dim"], cfg["neighbors"])
        self.se_pix = NumericSpectralEmbedding(cfg["se_dim"], cfg["neighbors"])
        sx = self.se_kar.fit_transform(x_std)
        sy = self.se_pix.fit_transform(y_std)

        # Step 2: CCA on the known pairs only (they are the last n_pairs rows).
        paired = slice(len(x_rows) - n_pairs, len(x_rows))
        self.cca = CCA(n_components=cfg["shared_dim"], max_iter=1000)
        self.cca.fit(sx[paired], sy[paired])
        zx = self._cca_kar(sx)
        zy = self._cca_pix(sy)

        # Step 3: residual MMD alignment of the pix view onto the kar view,
        # using all samples without correspondences.
        self.mmd = self._fit_residual(zy, zx, device)
        return self

    def _fit_residual(self, source: np.ndarray, target: np.ndarray, device: str) -> MMDResidual:
        cfg = self.config
        return train_mmd_residual(
            source=source,
            target=target,
            epochs=cfg["mmd_epochs"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
            device=device,
        )

    def _cca_kar(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(self.cca.transform(s), dtype=np.float32)

    def _cca_pix(self, s: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(s), self.cca.n_features_in_), dtype=np.float64)
        _, zy = self.cca.transform(dummy, s)
        return np.asarray(zy, dtype=np.float32)

    # -- inference ----------------------------------------------------------
    def encode(self, view: str, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        if view == "kar":
            s = self.se_kar.transform(self.scaler_kar.transform(rows).astype(np.float32))
            return self._cca_kar(s)
        if view == "pix":
            s = self.se_pix.transform(self.scaler_pix.transform(rows).astype(np.float32))
            z = self._cca_pix(s)
            return transform_mmd(self.mmd, z) if self.mmd is not None else z
        raise ValueError(f"unknown view {view!r}")

    # -- persistence --------------------------------------------------------
    def save(self, model_dir: Path, method_name: str, module: str) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        mmd_state = None if self.mmd is None else self.mmd.state_dict()
        mmd = self.mmd
        self.mmd = None
        with (model_dir / "model.pkl").open("wb") as f:
            pickle.dump(self, f)
        self.mmd = mmd
        torch.save({"mmd": mmd_state}, model_dir / "residual.pt")
        write_encoder_spec(model_dir, method_name, module, self.config)

    @classmethod
    def load(cls, model_dir: Path) -> "SUE":
        with (model_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        state = torch.load(model_dir / "residual.pt", map_location="cpu")["mmd"]
        if state is not None:
            model.mmd = MMDResidual(model.config["shared_dim"])
            model.mmd.load_state_dict(state)
            model.mmd.eval()
        return model


def write_encoder_spec(model_dir: Path, method_name: str, module: str, config: dict) -> None:
    spec = {
        "method": method_name,
        "experiment": "handwritten_retrieval",
        "command": ["python", "-m", module, "encode", "--model-dir", "{model_dir}"],
        "config": config,
    }
    (model_dir / "encoder.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")


def encode_cli(args: argparse.Namespace, loader) -> None:
    rows = np.load(args.input, allow_pickle=False)
    model = loader(Path(args.model_dir))
    emb = np.asarray(model.encode(args.view, rows), dtype=np.float32)
    np.save(args.output, emb)


def add_encode_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--view", required=True, choices=["kar", "pix"])
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)


def main() -> None:
    # Use the importable module (not __main__) so pickled models load anywhere.
    from method import sue as lib

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output-dir", default="artifacts/sue")
    train.add_argument("--se-dim", type=int, default=10)
    train.add_argument("--neighbors", type=int, default=100)
    train.add_argument("--mmd-epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_encode_args(commands.add_parser("encode"))
    args = parser.parse_args()

    if args.command == "train":
        model = lib.SUE(
            se_dim=args.se_dim,
            neighbors=args.neighbors,
            mmd_epochs=args.mmd_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        ).fit(lib.load_weakly_paired(), device=args.device)
        out = Path(args.output_dir)
        model.save(out, method_name=out.name, module="method.sue")
        print(f"Wrote SUE encoder to {out}")
    else:
        lib.encode_cli(args, lib.SUE.load)


if __name__ == "__main__":
    main()
