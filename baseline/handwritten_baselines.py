"""Paired-only contrastive baseline (the paper's "Contrastive" column).

Two MLP encoders trained with a symmetric InfoNCE loss on the 100 known pairs
only.  Per-view feature standardization uses that view's public rows; no
cross-view information beyond the known pairs is used.

Usage:
  python -m baseline.handwritten_baselines train  --output-dir artifacts/contrastive
  python -m baseline.handwritten_baselines encode --model-dir artifacts/contrastive \
      --view kar|pix --input rows.npy --output emb.npy
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data.handwritten import EMBEDDING_DIM, VIEW_DIMS, load_weakly_paired


class ContrastiveMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return torch.nn.functional.normalize(z, dim=1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    data = load_weakly_paired()
    scalers = {view: StandardScaler().fit(data.all_rows(view)) for view in ("kar", "pix")}
    x_pair = torch.tensor(scalers["kar"].transform(data.paired_kar), dtype=torch.float32)
    y_pair = torch.tensor(scalers["pix"].transform(data.paired_pix), dtype=torch.float32)

    x_model = ContrastiveMLP(VIEW_DIMS["kar"], EMBEDDING_DIM).to(args.device)
    y_model = ContrastiveMLP(VIEW_DIMS["pix"], EMBEDDING_DIM).to(args.device)
    opt = torch.optim.AdamW(list(x_model.parameters()) + list(y_model.parameters()), lr=1e-3)
    loader = DataLoader(TensorDataset(x_pair, y_pair), batch_size=min(64, len(x_pair)), shuffle=True)

    for _ in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = x_model(xb) @ y_model(yb).T / 0.07
            labels = torch.arange(len(xb), device=args.device)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            ) * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scalers.pkl").open("wb") as f:
        pickle.dump(scalers, f)
    torch.save(
        {"kar": x_model.cpu().state_dict(), "pix": y_model.cpu().state_dict()},
        out / "encoders.pt",
    )
    spec = {
        "method": out.name,
        "experiment": "handwritten_retrieval",
        "command": [
            "python", "-m", "baseline.handwritten_baselines", "encode", "--model-dir", "{model_dir}",
        ],
        "config": {"epochs": args.epochs, "seed": args.seed, "temperature": 0.07},
    }
    (out / "encoder.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(f"Wrote contrastive baseline encoder to {out}")


def encode(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    with (model_dir / "scalers.pkl").open("rb") as f:
        scalers = pickle.load(f)
    states = torch.load(model_dir / "encoders.pt", map_location="cpu")
    model = ContrastiveMLP(VIEW_DIMS[args.view], EMBEDDING_DIM)
    model.load_state_dict(states[args.view])
    model.eval()
    rows = np.load(args.input, allow_pickle=False)
    with torch.no_grad():
        emb = model(torch.tensor(scalers[args.view].transform(rows), dtype=torch.float32)).numpy()
    np.save(args.output, emb.astype(np.float32))


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    tr.add_argument("--output-dir", default="artifacts/contrastive")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--seed", type=int, default=13)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    enc = commands.add_parser("encode")
    enc.add_argument("--model-dir", required=True)
    enc.add_argument("--view", required=True, choices=["kar", "pix"])
    enc.add_argument("--input", required=True)
    enc.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else encode(args)


if __name__ == "__main__":
    main()
