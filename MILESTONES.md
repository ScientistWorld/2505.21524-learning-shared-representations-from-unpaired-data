# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
