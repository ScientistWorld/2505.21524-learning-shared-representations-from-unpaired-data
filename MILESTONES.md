# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-11-08 07:30] - core_claim
- Hardened the gym (see `HARDENING.md`): evaluator-owned held-out split (secret seed, rows only in `eval/test/`), public data reduced to 100 pairs + 990 unpaired rows per view, per-view encoder interface with isolated, guarded encoder calls, shape/finiteness/8-D gates, pessimistic ties, frozen paired-only baseline, self-reported budget fields removed from scoring.
- Jobs `1f17f5dc-f16` and `fc8d1d2f-089`: SUE reproduces deterministically (hidden-test mean R@10 70.38 vs frozen contrastive 56.38; dev 67.50 vs 55.25); better variants score +22.7 to +57.6 (CIs exclude 0, seed spread 7.1); paper ablations score lower; all red-team shortcuts and new cross-call/evaluator-file probes earn nothing; `validate.py --post-design` passes.

### [2026-06-19 17:57] - core_claim
- Completed final verification of the `core_claim` workspace.
- Confirmed strict method-agnostic files do not leak the paper's algorithm and no committed code references `/home/user/shared`.
- Refreshed `scoring/scores.json`, `scores_train.json`, and `scores_test.json` with the supervision-budget metrics already declared in the evaluator contract.
- Re-ran `python validate.py --compare` successfully after the scoring fixes.

### [2026-06-19 17:56] - core_claim
- Audited the existing `core_claim` workspace with two subagents: the implementation matches SUE's spectral embedding, few-pair CCA, and unpaired MMD pipeline, and evaluation remains artifact-based with `eval/` separated from `method/`.
- Promoted supervision-budget accounting into the score contract by adding artifact and metric fields for paired training count, unpaired training count, and held-out evaluation pair count.
- Final verification refreshed the checked-in score JSON files directly from the validated experiment metadata instead of submitting another job.

### [2026-06-19 17:55] - core_claim
- Completed final verification of the `core_claim` workspace.
- Refreshed `scoring/scores.json` to include all current evaluator metrics and re-ran `python validate.py --compare` successfully.
- Removed generated job and trajectory logs from git tracking, tightened method-agnostic evaluation wording, and added an evaluator guard for missing artifacts.

### [2026-06-19 17:45] - core_claim
- Audited final packaging after a continuation handoff.
- Fixed `scripts/run.sh` so submitted jobs regenerate baseline and SUE artifacts before scoring, instead of relying on stale ignored `.npz` files.
- Documented all evaluator metrics in `scoring/TARGETS.md` and removed generated artifact metadata from git tracking.
- Submitting a CPU test job to refresh `scoring/scores.json` and rerun validation inside the managed container.

### [2026-06-19 17:17] - core_claim
- Continued from the reset-to-`method_runs` audit point and rechecked the implementation, evaluator, scripts, scoring files, and documentation.
- Confirmed the checked-in method still implements SUE's graph spectral embeddings, 100-pair CCA alignment, and unpaired MMD residual alignment rather than a surrogate.
- Re-ran `python validate.py --compare`; scoring and structure checks passed with SUE mean R@10 69.125 versus contrastive 54.25, so `core_claim` is restored.
- A direct bare-login `scripts/evaluate.sh` call still requires the container dependencies from `environment/setup.sh`; managed CPU job `1162da06-1a4` already verified the same evaluator inside the reproducible environment.

### [2026-06-19 17:12] - core_claim
- Audited the resumed workspace against the paper's SUE algorithm: independent spectral embeddings, few-pair CCA, and unpaired MMD residual alignment are implemented for the Handwritten benchmark.
- Confirmed managed CPU job `1162da06-1a4` completed `bash scripts/run.sh` and `python validate.py --compare` with all checks passing.
- Final validated scores support the core claim: SUE mean R@10 69.125 versus reproduced paired-only contrastive mean R@10 54.25 in the same 8-dimensional shared space.
- Documented the paper's CCA pair-count inconsistency: Table 1/Table 12 use 100 pairs for Handwritten, while Appendix E.8 states 600 pairs globally; this gym follows the Handwritten table setting.

### [2026-06-12 14:34] - method_runs
- Continued from a reset audit point.
- Confirmed the implementation contains SUE's independent spectral embedding, weak-pair CCA, and unpaired MMD residual alignment steps.
- Confirmed local validation structure passes, but standalone `scripts/evaluate.sh` cannot run on the login Python because NumPy is absent outside the managed container.
- Submitting a CPU test job to verify the existing full reproduction and evaluator inside the reproducible container before restoring `core_claim`.

### [2026-06-12 14:37] - method_runs
- CPU job `f193cea4-e7f` ran the full reproduction in the managed container and passed `python validate.py --compare`.
- The rerun still supports the core claim: SUE mean R@10 68.375 versus the paired-only contrastive baseline at 54.25.
- Patched the numeric spectral embedding to use a fixed ARPACK start vector and stable eigenvector sign convention, then submitting another CPU test to verify reproducibility of the patched method.

### [2026-06-12 14:19] - core_claim
- Reproduced the paper's core weakly paired shared-representation claim on the UCI Multiple Features Handwritten benchmark.
- Ran SUE's independent spectral embeddings, 100-pair CCA alignment, and unpaired MMD residual alignment end-to-end.
- Validated artifact-based retrieval scoring with `python validate.py --compare`.
- SUE achieved mean R@10 69.50 versus the reproduced paired-only contrastive baseline at 54.25 under the same 8-dimensional shared-space constraint.

### [2026-06-12 14:35] - core_claim
- Audited the continuation workspace and confirmed the implementation is SUE's actual spectral embedding, CCA, and MMD procedure rather than a surrogate.
- CPU job `f7e6c609-927` passed structure validation, but an attempted mAP extension was removed because its values were not comparable to the paper's mAP table.

### [2026-06-12 14:31] - core_claim
- Re-validated the restored workspace with `python validate.py --compare`.
- Confirmed `eval/` remains method-independent and scores saved artifacts from `artifacts/*.npz`.

## Stop Justification
- Completed at milestone `core_claim`.
