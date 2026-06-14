# Reproduction Milestones

**Current: method_runs**

## Progress Log

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
