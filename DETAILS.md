# Overview

- **Paper ID:** 2505.21524
- **Title:** Learning Shared Representations from Unpaired Data
- **Domain:** Multimodal representation learning
- **TL;DR:** SUE learns shared cross-view embeddings from mostly unpaired data by combining independent spectral embeddings, few-pair CCA alignment, and unpaired MMD refinement.

## Short Summary

The paper studies weakly paired multimodal learning, where only a tiny number of paired samples are available but many unpaired samples exist in each modality. Its proposed SUE pipeline recovers modality-agnostic structure through spectral embeddings of unimodal neighborhood graphs, then aligns those spaces with CCA and an MMD residual network. The headline result is substantially better retrieval than paired-only contrastive training under the same pair budget.

## Key Results

- On MSCOCO with 100 pairs, SUE improves average retrieval over contrastive learning by more than 250%.
- On UCI Handwritten with 100 pairs, SUE reports 79.0 K2P R@10 and 78.0 P2K R@10 versus contrastive at 28.0 and 30.5.
- Ablations show that alignment without the spectral embedding step is near-trivial, supporting the spectral component as central to the claim.

---

# Problem Definition

## Research Question

Can two different data views that describe the same underlying item be mapped into a shared representation when only a very small subset of cross-view pairs is available and most training samples are unpaired or misaligned?

## Setting

Two tabular views of handwritten digits (UCI Multiple Features): 64 Karhunen-Loeve coefficients (`kar`) and 240 pixel averages (`pix`). Training data: 100 known pairs and 990 rows per view whose cross-view correspondence is unknown (each view independently thinned and shuffled). No class labels. The output is a pair of encoders, one per view, that map single rows into a common 8-dimensional space.

## Why It Matters

Paired multimodal data is often expensive or unavailable. A good solution would let researchers use abundant single-view data while requiring only a small number of known correspondences.

## Success Criteria

A successful system supports cross-view retrieval in both directions on held-out paired examples it never saw, outperforms methods trained only on the small paired subset, and stays within the fixed budget: 100 known pairs, 8 output dimensions, each view encoded independently.

---

# Paper's Method

## Key Contribution

The paper introduces Spectral Universal Embedding (SUE), a weakly paired representation learning pipeline that uses mostly unpaired unimodal samples to build a shared cross-modal embedding.

## Approach

SUE first computes a spectral embedding for each modality independently from that modality's neighborhood graph. The paper uses graph affinities based on local cosine neighborhoods and the leading non-trivial eigenvectors of a random-walk diffusion operator, with SpectralNet as the scalable parametric implementation. A small paired subset is then used for Canonical Correlation Analysis to resolve sign, rotation, and linear alignment ambiguities between the two spectral spaces. Finally, a residual neural network is trained with a Maximum Mean Discrepancy loss on unpaired samples to reduce distributional mismatch without requiring additional pairs.

## Main Claims

The primary claim is that shared representations can be learned almost exclusively from unpaired data plus a very small number of pairs, yielding much stronger retrieval than paired-only contrastive learning under the same pair budget. Secondary claims include the importance of the spectral embedding step, improved performance as unpaired data increases, and usefulness for generation, arithmetic, zero-shot classification, and cross-domain classification.

---

# Evaluation

## What is measured

Cross-view instance retrieval between the two views of held-out handwritten digits, in an 8-dimensional shared space. `k2p` queries with the Karhunen-Loeve view (`kar`) and retrieves among the pixel-average view (`pix`); `p2k` is the reverse. Each query has exactly one correct match among 400 gallery items (its own other view). Similarity is cosine similarity.

Scored metrics (all recomputed by the evaluator from the embeddings your encoders return):

- `k2p_r@1`, `k2p_r@5`, `k2p_r@10`, `p2k_r@1`, `p2k_r@5`, `p2k_r@10`: percentage of queries whose true match is in the top 1/5/10.
- `mean_r@10`: average of the two R@10 values (primary metric).
- `mean_mrr`: average reciprocal rank of the true match (x100), averaged over both directions.
- `mean_median_rank`: median rank of the true match, averaged over both directions (lower is better).

Ties count against the method: a true match that ties with other gallery items gets the worst of the tied ranks.

The combined score is the sum of each metric's coefficient in `scoring/reference.json` times its value. All metrics are bounded, so no single metric can compensate for failing the retrieval task.

## Fixed conditions (enforced, not scored)

- **Supervision budget:** the only training data is `data/handwritten/` (loaded by `data/handwritten.py`): 100 known cross-view pairs plus 990 unpaired rows per view with no known correspondence. There are no class labels.
- **Operating point:** each encoder must return exactly 8 dimensions.
- **Gallery:** 400 held-out pairs, chosen and ordered by the evaluator.
- **Baseline:** `contrastive` (the paper's paired-only baseline, trained on the same 100 pairs) is reported from embeddings frozen inside the evaluator. The name `contrastive` is reserved, so a method directory with that name is ignored.

## Submission contract

A method is a directory `artifacts/<name>/` containing `encoder.json` with a `command` list, for example

```json
{"command": ["python", "-m", "my_package.my_module", "encode", "--model-dir", "{model_dir}"]}
```

For each view separately the evaluator writes that view's rows (raw feature values, same format as `data/handwritten/*_<view>.txt`) to a `.npy` file, in an order it chooses and that differs between views, and runs

```
<command> --view {kar|pix} --input <rows.npy> --output <embeddings.npy>
```

with the repository root on `PYTHONPATH`. The command must be a Python invocation, `["python", "-m", "<module>", ...]` or `["python", "<script>.py", ...]`. `"{model_dir}"` is replaced by a fresh private copy of your method directory; each call also gets a fresh working directory, `HOME` and `TMPDIR`. The output must be a finite numeric array of shape `(n_rows, 8)`, row `i` embedding input row `i`. Everything the encoder needs must be inside the method directory or the repository code. The two calls are independent: an encoder must map each row of one view to its embedding without information about the other view's evaluation rows.

Encoders run under a guard that refuses (with `PermissionError`) any access to the evaluator directory `eval/`, any file write outside the call's private directory, starting other programs, and opening sockets. Reading repository code and `data/` is allowed; run single-process (threads are fine).

A method whose encoder crashes, times out, writes nothing, or returns the wrong shape or non-finite values is scored as failed: every true match is counted as ranked last (all recalls 0, MRR 0.25, median rank 400). Self-reported numbers, counts or labels in the method directory are ignored.

## Splits

- `bash scripts/evaluate_train.sh` (and `scripts/evaluate.sh`, which writes `scoring/scores.json`) scores every method directory under `artifacts/` on 400 public development pairs (`eval/train/data/`). These pairs are for measurement and model selection only; training on them breaks the 100-pair budget.
- The hidden test uses the same protocol on 400 different held-out pairs that are not in the workspace. Development and test pairs are disjoint from each other and from the training rows, and are stratified the same way, so an honest method scores about the same on both.

## Baselines and targets

The paper reports Handwritten retrieval with 100 pairs (Table 1): the proposed method reaches 79.0 K2P R@10 and 78.0 P2K R@10; the paired-only contrastive baseline reaches 28.0 and 30.5. The gym's training pool is smaller than the paper's (990 unpaired rows per view instead of 1900), so absolute numbers differ; the reference implementation and the frozen baseline scores are in `scoring/scores_train.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 14:15] - none
- Read the paper and appendix details for SUE.
- Implemented the Handwritten reproduction pipeline and prepared the first validation job.

### [2026-06-12 14:24] - method_runs
- CPU job `07052327-9a9` executed the SUE Handwritten pipeline end to end and wrote retrieval artifacts.
- The job failed validation only because debug baseline method names were not present in `reference.json`.

### [2026-06-12 14:20] - core_claim
- CPU job `1711044e-d35` completed successfully and passed `python validate.py --compare`.
- Reproduced the core SUE weakly paired retrieval claim on the Handwritten benchmark: SUE mean R@10 69.50 versus contrastive baseline 54.25, both using 8-dimensional embeddings.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 14:19] - core_claim
- Reproduced the paper's core weakly paired shared-representation claim on the UCI Multiple Features Handwritten benchmark.
- Ran SUE's independent spectral embeddings, 100-pair CCA alignment, and unpaired MMD residual alignment end-to-end.
- Validated artifact-based retrieval scoring with `python validate.py --compare`.
- SUE achieved mean R@10 69.50 versus the reproduced paired-only contrastive baseline at 54.25 under the same 8-dimensional shared-space constraint.

### [2026-06-12 14:35] - core_claim
- Audited the continuation workspace and confirmed the implementation is SUE's actual spectral embedding, CCA, and MMD procedure rather than a surrogate.
- CPU job `f7e6c609-927` passed structure validation, but an attempted mAP extension was removed because its values were not comparable to the paper's mAP table.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 31m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 31m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 28m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
