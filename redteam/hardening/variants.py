"""Genuine method variants for the hardening audit (not part of the gym runtime).

Better variants (from the red-team report, re-implemented for the encoder
interface):
  anchored      SUE + identity-initialised residual with a known-pair anchor term
  richer        anchored, 20 spectral coordinates (still an 8-D output)
  richer_local  richer with 30 graph neighbours
Weaker variants (the paper's own ablations, Table 2 / Table 9):
  no_mmd        SE + CCA, no residual alignment
  no_cca        SE + MMD, no CCA (first 8 spectral coordinates)
  no_se         CCA + MMD on standardized raw features, no spectral embedding

Usage:
  python -m redteam.hardening.variants train --variant richer --output-dir DIR
  python -m redteam.hardening.variants encode --model-dir DIR --view kar --input X --output Y
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from method import sue as lib

ANCHORED = {"anchored": (10, 100), "richer": (20, 100), "richer_local": (20, 30)}
ABLATIONS = ("no_mmd", "no_cca", "no_se")


def anchored_residual(source: np.ndarray, target: np.ndarray, n_pairs: int, seed: int,
                      epochs: int, weight: float = 1.0, batch: int = 64) -> lib.MMDResidual:
    """MMD residual that starts at the identity and keeps the known pairs aligned."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = lib.MMDResidual(source.shape[1])
    torch.nn.init.zeros_(model.net[-1].weight)
    torch.nn.init.zeros_(model.net[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s = torch.tensor(source, dtype=torch.float32)
    t = torch.tensor(target, dtype=torch.float32)
    sp, tp = s[-n_pairs:], t[-n_pairs:]
    for _ in range(epochs):
        a, b = rng.permutation(len(s)), rng.permutation(len(t))
        for start in range(0, min(len(s), len(t)), batch):
            si, ti = s[a[start:start + batch]], t[b[start:start + batch]]
            if len(si) < 2 or len(ti) < 2:
                continue
            opt.zero_grad()
            loss = lib.mmd_loss(model(si), ti) + weight * torch.mean((model(sp) - tp) ** 2)
            loss.backward()
            opt.step()
    return model.eval()


class Variant(lib.SUE):
    def __init__(self, variant: str, seed: int = 13):
        se_dim, neighbors = ANCHORED.get(variant, (10, 100))
        super().__init__(se_dim=se_dim, neighbors=neighbors, seed=seed)
        self.config["variant"] = variant
        self.n_pairs = 100

    def fit(self, data, device: str = "cpu") -> "Variant":
        variant = self.config["variant"]
        self.n_pairs = len(data.paired_kar)
        if variant in ANCHORED or variant == "no_mmd":
            return super().fit(data, device)
        cfg = self.config
        x_rows, y_rows = data.all_rows("kar"), data.all_rows("pix")
        self.scaler_kar = StandardScaler().fit(x_rows)
        self.scaler_pix = StandardScaler().fit(y_rows)
        x_std = self.scaler_kar.transform(x_rows).astype(np.float32)
        y_std = self.scaler_pix.transform(y_rows).astype(np.float32)
        if variant == "no_cca":
            self.se_kar = lib.NumericSpectralEmbedding(cfg["se_dim"], cfg["neighbors"])
            self.se_pix = lib.NumericSpectralEmbedding(cfg["se_dim"], cfg["neighbors"])
            zx = self.se_kar.fit_transform(x_std)[:, : cfg["shared_dim"]]
            zy = self.se_pix.fit_transform(y_std)[:, : cfg["shared_dim"]]
        elif variant == "no_se":
            paired = slice(len(x_rows) - self.n_pairs, len(x_rows))
            self.cca = CCA(n_components=cfg["shared_dim"], max_iter=1000)
            self.cca.fit(x_std[paired], y_std[paired])
            zx, zy = self._cca_kar(x_std), self._cca_pix(y_std)
        else:
            raise ValueError(variant)
        self.mmd = lib.train_mmd_residual(zy, zx, cfg["mmd_epochs"], cfg["batch_size"], cfg["seed"], device)
        return self

    def _fit_residual(self, source, target, device):
        variant = self.config["variant"]
        if variant == "no_mmd":
            return None
        if variant in ANCHORED:
            return anchored_residual(source, target, self.n_pairs, self.config["seed"],
                                     self.config["mmd_epochs"])
        return super()._fit_residual(source, target, device)

    def encode(self, view: str, rows: np.ndarray) -> np.ndarray:
        variant = self.config["variant"]
        if variant in ANCHORED or variant == "no_mmd":
            return super().encode(view, rows)
        rows = np.asarray(rows, dtype=np.float64)
        dim = self.config["shared_dim"]
        if variant == "no_cca":
            if view == "kar":
                return self.se_kar.transform(self.scaler_kar.transform(rows).astype(np.float32))[:, :dim]
            z = self.se_pix.transform(self.scaler_pix.transform(rows).astype(np.float32))[:, :dim]
        else:  # no_se
            if view == "kar":
                return self._cca_kar(self.scaler_kar.transform(rows).astype(np.float32))
            z = self._cca_pix(self.scaler_pix.transform(rows).astype(np.float32))
        if view != "pix":
            raise ValueError(view)
        return lib.transform_mmd(self.mmd, z)


def main() -> None:
    from redteam.hardening import variants as me  # pickle under an importable name

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--variant", required=True, choices=sorted(ANCHORED) + list(ABLATIONS))
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seed", type=int, default=13)
    lib.add_encode_args(commands.add_parser("encode"))
    args = parser.parse_args()
    if args.command == "train":
        model = me.Variant(args.variant, seed=args.seed).fit(lib.load_weakly_paired())
        out = Path(args.output_dir)
        model.save(out, method_name=out.name, module="redteam.hardening.variants")
        print(f"Wrote {args.variant} encoder to {out}")
    else:
        lib.encode_cli(args, lib.SUE.load)


if __name__ == "__main__":
    main()
