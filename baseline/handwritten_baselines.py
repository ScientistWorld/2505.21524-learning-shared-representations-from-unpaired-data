from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.handwritten import make_split


class ContrastiveMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 8):
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    split = make_split(n_test=args.n_test, n_paired=args.n_paired, seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_model = ContrastiveMLP(split.train_x.shape[1], args.shared_dim).to(args.device)
    y_model = ContrastiveMLP(split.train_y.shape[1], args.shared_dim).to(args.device)
    opt = torch.optim.AdamW(list(x_model.parameters()) + list(y_model.parameters()), lr=1e-3)
    x_pair = torch.tensor(split.train_x[split.paired_indices], dtype=torch.float32)
    y_pair = torch.tensor(split.train_y[split.paired_indices], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_pair, y_pair), batch_size=min(64, args.n_paired), shuffle=True)

    for _ in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            zx = x_model(xb)
            zy = y_model(yb)
            logits = zx @ zy.T / 0.07
            labels = torch.arange(len(xb), device=args.device)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            ) * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()

    x_model.eval()
    y_model.eval()
    with torch.no_grad():
        x_test = x_model(torch.tensor(split.test_x, dtype=torch.float32).to(args.device)).cpu().numpy()
        y_test = y_model(torch.tensor(split.test_y, dtype=torch.float32).to(args.device)).cpu().numpy()
    save_artifact(out_dir / "contrastive_handwritten.npz", "contrastive", x_test, y_test, split.test_labels)
    print(f"Wrote contrastive baseline to {out_dir}")


if __name__ == "__main__":
    main()
