"""Hidden held-out evaluator for Handwritten cross-view retrieval.

A method is a directory ``<artifact-dir>/<name>/`` holding ``encoder.json``
with a ``command`` list (``"{model_dir}"`` is replaced by a private copy of the
directory).  For each view the evaluator writes only that view's held-out rows,
in an evaluator-chosen order that differs between views, to a ``.npy`` file
and runs

    <command> --view {kar|pix} --input <rows.npy> --output <embeddings.npy>

in a fresh working directory with a fresh TMPDIR and a fresh copy of the
method directory, so nothing written during one call is visible to the other.

Everything scored is recomputed here from the returned embeddings.  The held-out
rows, their correspondences, the gallery size (400), the 8-dimensional operating
point and the paired-only baseline (frozen embeddings, reported as
``contrastive``) are owned by this directory.  A method whose encoder fails,
times out, or returns anything other than a finite ``(n_rows, 8)`` array is
scored as if every true match were ranked last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SLICE_NAME = "test"
EXPERIMENT = "handwritten_retrieval"
BASELINE_NAME = "contrastive"
EMBEDDING_DIM = 8
VIEWS = ("kar", "pix")
VIEW_DIMS = {"kar": 64, "pix": 240}
N_PAIRS = 400
ORDER_KEYS = {"kar": "cead863146ea4c9c", "pix": "f2f2a0a4968098ff"}
ENCODER_TIMEOUT_SECONDS = 1800
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"


class EncoderError(Exception):
    pass


def load_heldout() -> dict[str, np.ndarray]:
    rows = {}
    for view in VIEWS:
        data = np.loadtxt(DATA_DIR / f"heldout_{view}.txt", dtype=np.float64, ndmin=2)
        if data.shape != (N_PAIRS, VIEW_DIMS[view]):
            raise RuntimeError(f"held-out {view} rows have shape {data.shape}")
        rows[view] = data
    return rows


def evaluator_order(n: int, key: str) -> np.ndarray:
    digest = lambda i: hashlib.blake2b(f"{key}:{i}".encode(), digest_size=8).digest()
    return np.asarray(sorted(range(n), key=digest), dtype=np.int64)


def check_embeddings(emb: object, n_rows: int) -> np.ndarray:
    if not isinstance(emb, np.ndarray) or emb.dtype.kind not in "fiu":
        raise EncoderError("output is not a numeric array")
    if emb.shape != (n_rows, EMBEDDING_DIM):
        raise EncoderError(f"output shape {emb.shape}, required {(n_rows, EMBEDDING_DIM)}")
    emb = emb.astype(np.float64)
    if not np.all(np.isfinite(emb)):
        raise EncoderError("output contains non-finite values")
    return emb


def load_command(method_dir: Path) -> list[str]:
    try:
        spec = json.loads((method_dir / "encoder.json").read_text())
    except (OSError, ValueError) as exc:
        raise EncoderError(f"unreadable encoder.json: {exc}") from exc
    command = spec.get("command") if isinstance(spec, dict) else None
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise EncoderError("encoder.json must contain a non-empty 'command' list of strings")
    return command


def run_encoder(method_dir: Path, command: list[str], view: str, rows: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="hw-encode-") as tmp_name:
        tmp = Path(tmp_name)
        model_copy = tmp / "model"
        shutil.copytree(method_dir, model_copy)
        work = tmp / "work"
        scratch = tmp / "scratch"
        work.mkdir()
        scratch.mkdir()
        rows_path = tmp / "rows.npy"
        out_path = tmp / "embeddings.npy"
        np.save(rows_path, np.ascontiguousarray(rows, dtype=np.float64))
        argv = [
            sys.executable if i == 0 and part in ("python", "python3")
            else part.replace("{model_dir}", str(model_copy))
            for i, part in enumerate(command)
        ]
        argv += ["--view", view, "--input", str(rows_path), "--output", str(out_path)]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(scratch)
        try:
            proc = subprocess.run(
                argv, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", timeout=ENCODER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise EncoderError(f"{view} encoder timed out") from exc
        except OSError as exc:
            raise EncoderError(f"{view} encoder could not start: {exc}") from exc
        if proc.returncode != 0:
            tail = " | ".join(proc.stdout.strip().splitlines()[-3:])
            raise EncoderError(f"{view} encoder exited with {proc.returncode}: {tail}")
        if not out_path.is_file():
            raise EncoderError(f"{view} encoder wrote no output")
        try:
            emb = np.load(out_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - any unreadable output is a failure
            raise EncoderError(f"{view} output unreadable: {exc}") from exc
    return check_embeddings(emb, len(rows))


def encode_heldout(method_dir: Path, command: list[str], rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    embeddings = {}
    for view in VIEWS:
        order = evaluator_order(len(rows[view]), ORDER_KEYS[view])
        emb = run_encoder(method_dir, command, view, rows[view][order])
        canonical = np.empty_like(emb)
        canonical[order] = emb
        embeddings[view] = canonical
    return embeddings


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def true_match_ranks(similarity: np.ndarray) -> np.ndarray:
    """Rank of the true match (diagonal) per query row; ties count against it."""
    true = np.diag(similarity)[:, None]
    return (similarity >= true).sum(axis=1).astype(np.int64)


def summarize(k2p: np.ndarray, p2k: np.ndarray) -> dict[str, float]:
    def stats(ranks: np.ndarray) -> dict[str, float]:
        out = {f"r@{k}": float(np.mean(ranks <= k) * 100.0) for k in (1, 5, 10)}
        out["mrr"] = float(np.mean(1.0 / ranks) * 100.0)
        out["median_rank"] = float(np.median(ranks))
        return out

    a, b = stats(k2p), stats(p2k)
    return {
        "k2p_r@1": a["r@1"],
        "k2p_r@5": a["r@5"],
        "k2p_r@10": a["r@10"],
        "p2k_r@1": b["r@1"],
        "p2k_r@5": b["r@5"],
        "p2k_r@10": b["r@10"],
        "mean_r@10": (a["r@10"] + b["r@10"]) / 2.0,
        "mean_mrr": (a["mrr"] + b["mrr"]) / 2.0,
        "mean_median_rank": (a["median_rank"] + b["median_rank"]) / 2.0,
    }


def score_embeddings(emb: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, list[int]]]:
    x = normalize(check_embeddings(emb["kar"], N_PAIRS))
    y = normalize(check_embeddings(emb["pix"], N_PAIRS))
    similarity = x @ y.T
    k2p = true_match_ranks(similarity)
    p2k = true_match_ranks(similarity.T)
    return summarize(k2p, p2k), {"k2p": k2p.tolist(), "p2k": p2k.tolist()}


def failed_score() -> tuple[dict[str, float], dict[str, list[int]]]:
    worst = np.full(N_PAIRS, N_PAIRS, dtype=np.int64)
    return summarize(worst, worst), {"k2p": worst.tolist(), "p2k": worst.tolist()}


def frozen_baseline_paths() -> dict[str, Path]:
    return {view: DATA_DIR / f"frozen_{BASELINE_NAME}_{view}.txt" for view in VIEWS}


def freeze_baseline(method_dir: Path, rows: dict[str, np.ndarray], force: bool) -> None:
    paths = frozen_baseline_paths()
    if all(p.exists() for p in paths.values()) and not force:
        print(f"[{SLICE_NAME}] frozen baseline already present; leaving it unchanged")
        return
    emb = encode_heldout(method_dir, load_command(method_dir), rows)
    for view, path in paths.items():
        np.savetxt(path, emb[view], fmt="%.9e")
    print(f"[{SLICE_NAME}] froze {method_dir} as the '{BASELINE_NAME}' baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--ranks-out", default=None, help="optional per-query ranks (analysis only)")
    parser.add_argument("--freeze-baseline", default=None, metavar="METHOD_DIR")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_heldout()
    if args.freeze_baseline:
        freeze_baseline(Path(args.freeze_baseline).resolve(), rows, args.force)
        return

    results: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, list[int]]] = {}
    status: dict[str, str] = {}

    paths = frozen_baseline_paths()
    frozen = all(p.exists() for p in paths.values())
    if frozen:
        emb = {v: np.loadtxt(p, dtype=np.float64, ndmin=2) for v, p in paths.items()}
        results[BASELINE_NAME], ranks[BASELINE_NAME] = score_embeddings(emb)
        status[BASELINE_NAME] = "frozen evaluator-owned paired-only baseline"

    artifact_dir = Path(args.artifact_dir).resolve()
    method_dirs = []
    if artifact_dir.is_dir():
        method_dirs = sorted(p for p in artifact_dir.iterdir() if p.is_dir() and (p / "encoder.json").is_file())
        for legacy in sorted(artifact_dir.glob("*.npz")):
            print(f"[{SLICE_NAME}] ignoring {legacy.name}: precomputed embeddings are not accepted")
    if not method_dirs:
        print(f"[{SLICE_NAME}] no method directories with encoder.json under {artifact_dir}")

    for method_dir in method_dirs:
        name = method_dir.name
        if name == BASELINE_NAME and frozen:
            print(f"[{SLICE_NAME}] '{name}' is reserved for the frozen baseline; skipping {method_dir}")
            continue
        try:
            emb = encode_heldout(method_dir, load_command(method_dir), rows)
            results[name], ranks[name] = score_embeddings(emb)
            status[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - any failure scores as failed
            results[name], ranks[name] = failed_score()
            status[name] = f"failed: {exc}"
        print(f"[{SLICE_NAME}] {name}: {status[name]} | mean_r@10={results[name]['mean_r@10']:.2f}")

    scores = {"_slice": SLICE_NAME, "_status": status, "experiments": {EXPERIMENT: {"results": results}}}
    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    if args.ranks_out:
        Path(args.ranks_out).write_text(json.dumps(ranks) + "\n")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
