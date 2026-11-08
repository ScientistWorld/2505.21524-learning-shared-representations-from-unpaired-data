"""Hardening audit: genuine variants, seed noise and shortcuts through the hidden test.

Run after scripts/reproduce.sh and the reference evaluation (scripts/run.sh does
this).  Writes redteam/hardening/results/*.json; large model copies live in the
ignored redteam/hardening/.work/.

Variants are scored together (every method sees the same 400 held-out pairs)
with the hidden evaluator code; shortcut cases are each installed alone in
artifacts/ and scored with `bash scripts/evaluate_test.sh` with no arguments,
after which the reference artifacts and score files are restored.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "redteam" / "hardening"
WORK = HERE / ".work"
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
EXPERIMENT = "handwritten_retrieval"
SCORE_FILES = ("scores.json", "scores_train.json", "scores_test.json")

SEEDS = (17, 29)
VARIANTS = ("anchored", "richer", "richer_local", "no_mmd", "no_cca", "no_se")
SHORTCUTS = (
    "copy_input", "public_lookup", "metadata_claims", "test_size", "empty", "nan", "inf",
    "constant", "wrong_dim", "direct_scores", "replace_baseline", "stash_sandboxed",
    "stash_absolute",
)


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def coefficients() -> dict[str, float]:
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())
    metrics = ref["experiments"][EXPERIMENT]["metrics"]
    return {name: float(spec["coefficient"]) for name, spec in metrics.items() if spec["coefficient"]}


def weighted(metrics: dict[str, float], coef: dict[str, float]) -> float:
    return sum(c * float(metrics[m]) for m, c in coef.items())


def summarize(k2p: np.ndarray, p2k: np.ndarray) -> dict[str, float]:
    def stats(r):
        out = {f"r@{k}": float(np.mean(r <= k) * 100.0) for k in (1, 5, 10)}
        out["mrr"] = float(np.mean(1.0 / r) * 100.0)
        out["median_rank"] = float(np.median(r))
        return out

    a, b = stats(k2p), stats(p2k)
    return {"k2p_r@1": a["r@1"], "k2p_r@5": a["r@5"], "k2p_r@10": a["r@10"],
            "p2k_r@1": b["r@1"], "p2k_r@5": b["r@5"], "p2k_r@10": b["r@10"],
            "mean_r@10": (a["r@10"] + b["r@10"]) / 2, "mean_mrr": (a["mrr"] + b["mrr"]) / 2,
            "mean_median_rank": (a["median_rank"] + b["median_rank"]) / 2}


def bootstrap(ranks: dict, coef: dict[str, float], reference: str = "sue", reps: int = 5000) -> dict:
    """Paired bootstrap over held-out pairs of the weighted-score difference to the reference."""
    rng = np.random.default_rng(0)
    arrays = {m: (np.asarray(r["k2p"]), np.asarray(r["p2k"])) for m, r in ranks.items()}
    n = len(arrays[reference][0])
    draws = {m: [] for m in arrays}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        ref = weighted(summarize(arrays[reference][0][idx], arrays[reference][1][idx]), coef)
        for m, (k2p, p2k) in arrays.items():
            draws[m].append(weighted(summarize(k2p[idx], p2k[idx]), coef) - ref)
    out = {}
    for m, values in draws.items():
        v = np.asarray(values)
        out[m] = {"ci95_low": float(np.percentile(v, 2.5)), "ci95_high": float(np.percentile(v, 97.5)),
                  "prob_above_reference": float(np.mean(v > 0)), "sd": float(v.std(ddof=1))}
    return out


def variants(coef: dict[str, float]) -> dict:
    combined = WORK / "combined"
    shutil.rmtree(combined, ignore_errors=True)
    combined.mkdir(parents=True)
    shutil.copytree(ROOT / "artifacts" / "sue", combined / "sue")
    trained = {}
    for seed in SEEDS:
        name = f"sue_seed{seed}"
        trained[name] = run([sys.executable, "-m", "method.sue", "train", "--seed", str(seed),
                             "--output-dir", str(combined / name)], LOGS / f"train_{name}.log")
    for name in VARIANTS:
        trained[name] = run([sys.executable, "-m", "redteam.hardening.variants", "train", "--variant", name,
                             "--output-dir", str(combined / name)], LOGS / f"train_{name}.log")
    out = {"train_returncodes": trained}
    for split, module in (("test", "eval.test.evaluate"), ("train", "eval.train.evaluate")):
        scores_path = RESULTS / f"variants_{split}.json"
        ranks_path = WORK / f"variants_{split}_ranks.json"
        rc = run([sys.executable, "-m", module, "--artifact-dir", str(combined), "--scores", str(scores_path),
                  "--ranks-out", str(ranks_path)], LOGS / f"variants_{split}.log")
        scores = json.loads(scores_path.read_text())
        results = scores["experiments"][EXPERIMENT]["results"]
        ranks = json.loads(ranks_path.read_text())
        ref_score = weighted(results["sue"], coef)
        seed_scores = [weighted(results[m], coef) for m in ("sue", *[f"sue_seed{s}" for s in SEEDS])
                       if m in results]
        out[split] = {
            "returncode": rc,
            "status": scores.get("_status", {}),
            "weighted_score": {m: weighted(v, coef) for m, v in results.items()},
            "delta_vs_reference": {m: weighted(v, coef) - ref_score for m, v in results.items()},
            "mean_r@10": {m: v["mean_r@10"] for m, v in results.items()},
            "mean_mrr": {m: v["mean_mrr"] for m, v in results.items()},
            "seed_spread": float(max(seed_scores) - min(seed_scores)),
            "seed_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else None,
            "bootstrap": bootstrap(ranks, coef),
        }
    return out


def install_case(case: str, reference: Path, artifacts: Path) -> None:
    from redteam.hardening import shortcuts

    if case == "metadata_claims":
        shortcuts.build_metadata_claims(reference, artifacts / "sue")
    elif case == "direct_scores":
        shutil.copytree(reference, artifacts / "sue")
    elif case == "replace_baseline":
        shutil.copytree(reference, artifacts / "sue")
        shortcuts.build_zero_baseline(artifacts / "contrastive")
    else:
        shortcuts.build(case, reference, artifacts / "sue")


def shortcut_cases(coef: dict[str, float], reference_metrics: dict[str, float]) -> dict:
    from redteam.hardening import shortcuts

    artifacts = ROOT / "artifacts"
    backup = WORK / "backup"
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(artifacts, backup / "artifacts")
    for name in SCORE_FILES:
        shutil.copy2(ROOT / "scoring" / name, backup / name)
    reference = backup / "artifacts" / "sue"
    ref_score = weighted(reference_metrics, coef)
    out = {}
    for case in SHORTCUTS:
        folder = RESULTS / "shortcuts" / case
        folder.mkdir(parents=True, exist_ok=True)
        record = {"case": case, "test_command": "bash scripts/evaluate_test.sh"}
        try:
            shutil.rmtree(artifacts, ignore_errors=True)
            artifacts.mkdir()
            install_case(case, reference, artifacts)
            shutil.rmtree(shortcuts.ABSOLUTE_STASH, ignore_errors=True)
            record["train_returncode"] = run(["bash", "scripts/evaluate_train.sh"], folder / "train.log")
            shutil.copy2(ROOT / "scoring" / "scores_train.json", folder / "scores_train.json")
            shutil.rmtree(shortcuts.ABSOLUTE_STASH, ignore_errors=True)
            if case == "direct_scores":
                claim = {"experiments": {EXPERIMENT: {"results": {"sue": {
                    "mean_r@10": 1e9, "mean_mrr": 1e9, "k2p_r@1": 1e9, "success": True}}}}}
                (ROOT / "scoring" / "scores_test.json").write_text(json.dumps(claim))
                (folder / "claimed_scores.json").write_text(json.dumps(claim, indent=2))
            record["test_returncode"] = run(["bash", "scripts/evaluate_test.sh"], folder / "test.log")
            shutil.copy2(ROOT / "scoring" / "scores_test.json", folder / "scores_test.json")
            for split in ("train", "test"):
                scores = json.loads((folder / f"scores_{split}.json").read_text())
                results = scores["experiments"][EXPERIMENT]["results"]
                record[split] = {
                    "status": scores.get("_status", {}),
                    "candidate": results.get("sue"),
                    "baseline_reported": results.get("contrastive"),
                }
            record["test_delta_vs_reference"] = weighted(record["test"]["candidate"], coef) - ref_score
            record["test_baseline_delta_vs_reference"] = (
                weighted(record["test"]["baseline_reported"], coef) - ref_score)
        except Exception as exc:  # noqa: BLE001 - keep auditing the other cases
            record["error"] = repr(exc)
        finally:
            shutil.rmtree(artifacts, ignore_errors=True)
            shutil.copytree(backup / "artifacts", artifacts)
            for name in SCORE_FILES:
                shutil.copy2(backup / name, ROOT / "scoring" / name)
            shutil.rmtree(shortcuts.ABSOLUTE_STASH, ignore_errors=True)
        (folder / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"case": case, "delta": record.get("test_delta_vs_reference"),
                          "error": record.get("error")}), flush=True)
        out[case] = record
    return out


def main() -> None:
    start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    coef = coefficients()
    reference_scores = json.loads((ROOT / "scoring" / "scores_test.json").read_text())
    reference_metrics = reference_scores["experiments"][EXPERIMENT]["results"]["sue"]
    summary = {"coefficients": coef, "reference_test_metrics": reference_metrics,
               "reference_weighted_score": weighted(reference_metrics, coef)}
    summary["variants"] = variants(coef)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["shortcuts"] = shortcut_cases(coef, reference_metrics)
    summary["elapsed_seconds"] = time.time() - start
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    v = summary["variants"]["test"]
    print("\nweighted-score delta vs reference (hidden test):")
    for m, d in sorted(v["delta_vs_reference"].items(), key=lambda kv: -kv[1]):
        b = v["bootstrap"][m]
        print(f"  {m:16s} {d:+9.3f}  R@10={v['mean_r@10'][m]:6.2f}  CI95=[{b['ci95_low']:+.2f}, {b['ci95_high']:+.2f}]")
    print(f"  seed spread {v['seed_spread']:.3f}")
    for case, rec in summary["shortcuts"].items():
        print(f"  shortcut {case:18s} {rec.get('test_delta_vs_reference')}")


if __name__ == "__main__":
    main()
