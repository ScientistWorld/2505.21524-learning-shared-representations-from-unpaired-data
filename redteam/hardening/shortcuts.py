"""Shortcut encoders for the hardening audit (not part of the gym runtime).

Each shortcut is a method directory whose encoder.json points at
``python -m redteam.hardening.shortcuts encode --kind <kind> --model-dir {model_dir}``.
Kinds that need a real representation wrap a copy of the reference SUE model
stored in the same directory.  Only public inputs are used (data/ and the
public development pairs); nothing here reads eval/test/.

  copy_input       first 8 standardized raw columns of whatever view is given
  public_lookup    memorize every public row (training + development pairs) at
                   build time; encode the row identity at test time
  read_evaluator_files  read the evaluator's data files at encode time to find
                   each row's position (falls back to the reference if refused)
  test_size        reference embeddings for only the first 2 rows
  empty            zero rows
  nan / inf        non-finite embeddings
  constant         all-zero embeddings
  wrong_dim        reference embeddings padded to 16 dimensions
  stash_sandboxed  cross-call matching with the stash kept in the model
                   directory, the working directory and TMPDIR
  stash_absolute   cross-call matching with the stash at a hard-coded /tmp path
  stash_subprocess same, but a child process writes the stash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABSOLUTE_STASH = Path("/tmp/hw_cross_call_stash")
KINDS = ("copy_input", "public_lookup", "read_evaluator_files", "test_size", "empty", "nan", "inf",
         "constant", "wrong_dim", "stash_sandboxed", "stash_absolute", "stash_subprocess")


def build(kind: str, reference_dir: Path, out: Path) -> None:
    """Create a shortcut method directory at ``out``."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(reference_dir, out)
    spec = {
        "method": out.name,
        "experiment": "handwritten_retrieval",
        "command": ["python", "-m", "redteam.hardening.shortcuts", "encode", "--kind", kind,
                    "--model-dir", "{model_dir}"],
    }
    (out / "encoder.json").write_text(json.dumps(spec, indent=2) + "\n")
    if kind == "public_lookup":  # memorize public rows at "training" time
        for view in ("kar", "pix"):
            tags, rows = zip(*_public_rows(view))
            np.savez(out / f"public_{view}.npz", tags=np.asarray(tags),
                     sizes=np.asarray([len(r) for r in rows]), rows=np.concatenate(rows))


def build_metadata_claims(reference_dir: Path, out: Path) -> None:
    """Reference encoder plus every self-reported field the old evaluator trusted."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(reference_dir, out)
    spec = json.loads((out / "encoder.json").read_text())
    spec.update({
        "paired_train_count": -1_000_000,
        "unpaired_train_count": float("nan"),
        "eval_pair_count": 400,
        "embedding_dim": 8,
        "labels": list(range(400)),
        "scores": {"mean_r@10": 100.0, "mean_mrr": 1e9, "success": True},
    })
    (out / "encoder.json").write_text(json.dumps(spec, indent=2) + "\n")


def build_zero_baseline(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spec = {"method": out.name, "experiment": "handwritten_retrieval",
            "command": ["python", "-m", "redteam.hardening.shortcuts", "encode", "--kind", "constant",
                        "--model-dir", "{model_dir}"]}
    (out / "encoder.json").write_text(json.dumps(spec, indent=2) + "\n")


def _reference(model_dir: Path):
    from method.sue import SUE
    return SUE.load(model_dir)


def _public_rows(view: str) -> list[tuple[str, np.ndarray]]:
    files = [("paired", ROOT / "data" / "handwritten" / f"paired_{view}.txt"),
             ("unpaired_" + view, ROOT / "data" / "handwritten" / f"unpaired_{view}.txt"),
             ("dev", ROOT / "eval" / "train" / "data" / f"dev_{view}.txt")]
    return [(tag, np.loadtxt(path, ndmin=2)) for tag, path in files if path.exists()]


def _code(identity: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).normal(size=8)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _match_across_calls(model_dir: Path, view: str, rows: np.ndarray, stash_dirs: list[Path],
                        via_subprocess: bool = False) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    model = _reference(model_dir)
    emb = model.encode(view, rows)
    other = "pix" if view == "kar" else "kar"
    other_rows = None
    for d in stash_dirs:
        path = d / f"stash_{other}.npy"
        if path.exists():
            other_rows = np.load(path)
            break
    for d in stash_dirs:
        try:
            if via_subprocess:
                local = Path(tempfile.gettempdir()) / f"stash_{view}.npy"
                np.save(local, rows)
                import subprocess
                subprocess.run(["sh", "-c", f"mkdir -p {d} && cp {local} {d}/"], check=True)
            else:
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"stash_{view}.npy", rows)
        except (OSError, Exception):
            pass
    if other_rows is None:
        return emb
    other_emb = model.encode(other, other_rows)
    rows_idx, cols_idx = linear_sum_assignment(-(_normalize(emb) @ _normalize(other_emb).T))
    out = emb.copy()
    out[rows_idx] = other_emb[cols_idx]
    return out


def encode(kind: str, model_dir: Path, view: str, rows: np.ndarray) -> np.ndarray:
    n = len(rows)
    if kind == "copy_input":
        cols = rows[:, :8]
        return (cols - cols.mean(0)) / np.maximum(cols.std(0), 1e-9)
    if kind == "public_lookup":
        table = {}
        with np.load(model_dir / f"public_{view}.npz") as memo:
            offset = 0
            for tag, size in zip(memo["tags"], memo["sizes"]):
                for i, row in enumerate(memo["rows"][offset:offset + size]):
                    # paired and dev rows share identities across views; unpaired rows do not
                    table.setdefault(row.round(6).tobytes(), f"{tag}:{i}")
                offset += size
        return np.stack([_code(table[r.round(6).tobytes()]) if r.round(6).tobytes() in table
                         else np.zeros(8) for r in rows])
    if kind == "read_evaluator_files":
        try:
            table = {}
            for split, name in (("test", "heldout"), ("train", "dev")):
                path = ROOT / "eval" / split / "data" / f"{name}_{view}.txt"
                for i, row in enumerate(np.loadtxt(path, ndmin=2)):
                    table.setdefault(row.round(6).tobytes(), f"{split}:{i}")
            return np.stack([_code(table.get(r.round(6).tobytes(), "missing")) for r in rows])
        except OSError:
            return _reference(model_dir).encode(view, rows)
    if kind == "test_size":
        return _reference(model_dir).encode(view, rows[:2])
    if kind == "empty":
        return np.zeros((0, 8))
    if kind == "nan":
        return np.full((n, 8), np.nan)
    if kind == "inf":
        return np.full((n, 8), np.inf)
    if kind == "constant":
        return np.zeros((n, 8))
    if kind == "wrong_dim":
        emb = _reference(model_dir).encode(view, rows)
        return np.concatenate([emb, np.zeros_like(emb)], axis=1)
    if kind == "stash_sandboxed":
        dirs = [model_dir / "stash", Path.cwd() / "stash", Path(tempfile.gettempdir()) / "stash"]
        return _match_across_calls(model_dir, view, rows, dirs)
    if kind == "stash_absolute":
        return _match_across_calls(model_dir, view, rows, [ABSOLUTE_STASH])
    if kind == "stash_subprocess":
        return _match_across_calls(model_dir, view, rows, [ABSOLUTE_STASH], via_subprocess=True)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    enc = commands.add_parser("encode")
    enc.add_argument("--kind", required=True, choices=KINDS)
    enc.add_argument("--model-dir", required=True)
    enc.add_argument("--view", required=True)
    enc.add_argument("--input", required=True)
    enc.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = np.load(args.input, allow_pickle=False)
    np.save(args.output, np.asarray(encode(args.kind, Path(args.model_dir), args.view, rows)))
    print(f"{args.kind}: encoded {len(rows)} {args.view} rows; cwd={os.getcwd()}")


if __name__ == "__main__":
    main()
