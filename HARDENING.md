# Hardening report

**Bottom line.** The hidden test now rewards better research. Genuinely better variants score higher, and weaker ones lower, by more than both seed noise and test-sampling noise. Every shortcut in `REDTEAM.md` is closed, as is a stronger cross-call shortcut found during this audit. The paper's method still reproduces the paper's claim in the paper's direction.

Two jobs produced the evidence: `1f17f5dc-f16` (new protocol) and `fc8d1d2f-089` (the same plus the encoder guard).
- The reference reproduced bit-identically in both jobs.
- The variant encoders trained in job 1 were rescored, not retrained, in job 2.
- Numbers below are from job 2 (`redteam/hardening/results/summary.json`).
- Job 1's audit, before the guard existed, is kept in `summary_job1_before_guard.json`.
- Rebuild with `python -m redteam.hardening.audit` after `scripts/reproduce.sh`; `scripts/run.sh` does both.

All deltas are ΔS, the sum over scored metrics of the coefficient in `scoring/reference.json` times (method − reference).
- They are measured on the hidden test: 400 held-out pairs, every pair a query in both directions, and the same pairs for every method.
- CI95 is a paired bootstrap over the 400 held-out pairs (5000 resamples).

## What was wrong and what changed

| Problem in the red-team report | Fix |
|---|---|
| The public data interface exposed held-out correspondences. The full row-aligned UCI archive sat in `data/downloads/`, and `data/handwritten.py` produced aligned test rows. | The archive is removed from the workspace and `download.sh` no longer fetches it. The 2000 rows are re-split with a seed that exists only in `eval/test/build_split.py`. Public data (`data/handwritten/`) is exactly 100 known pairs plus 990 unpaired rows per view (10% dropped and shuffled independently per view, App. E.7), with no labels. There are 400 public development pairs in `eval/train/data/`, and 400 held-out pairs (with the split manifest and builder) only in `eval/test/`. |
| The evaluator scored arbitrary aligned embedding files, which allowed copied views, fabricated row IDs, label-defined slices and two-row "test sets". | Methods now submit per-view encoders (`artifacts/<name>/encoder.json`, a `python -m …` or `python script.py` command). The evaluator owns the rows, the gallery size (400), query selection (all 400, both directions) and the correspondences. It encodes each view in a separate subprocess, on evaluator-ordered rows that differ per view. Every call gets a fresh copy of the method directory, working directory, HOME and TMPDIR. |
| Self-reported `paired_train_count`, `embedding_dim`, `eval_pair_count` and `unpaired_train_count` were read from the artifact. The pair count had a −1 coefficient and was unbounded, so −1e6 earned about 1e6 points. | These fields are no longer scored or read. The 100-pair budget is enforced by the data interface, and the 8-D operating point by the evaluator: wrong shape or dimension scores as failed. All scored metrics are bounded retrieval quantities, so a method that fails retrieval cannot win through another metric. The paper's settings are kept verbatim under `paper_setting` in `reference.json`. Its reported R@k numbers are unchanged. |
| NaN or infinite embeddings were accepted, an empty output crashed the evaluator, and ties were broken by index order. | Non-finite, wrong-shape, crashing or timed-out encoders are scored as failed: every true match counts as ranked last. Ties count against the method. One bad method no longer crashes the run. |
| The baseline artifact was replaceable. | The paired-only baseline is frozen inside each evaluator (`eval/*/data/frozen_contrastive_*.txt`) and reported as `contrastive`; that name is reserved. |
| The method's `--seed` also changed the data split. | The split is fixed by the data files. `--seed` now changes only training randomness. |
| Found during this audit: an encoder that stashes one view's rows at a fixed `/tmp` path and solves a matching problem in the other view's call paid +27.96. | Each encoder call runs under a Python audit hook (`GUARD_BOOTSTRAP` in both evaluators). It refuses access to `eval/`, writes outside the call's private directory, starting other programs, and sockets. Every reference, baseline and variant encoder runs unchanged under it. |

The method itself (`method/sue.py`) is unchanged algorithmically: numeric spectral embedding with Nyström extension, CCA on the 100 known pairs, then an MMD residual on unpaired samples. It was restructured into `train` and `encode` entry points; per-view feature standardization now uses that view's public rows. The baseline was restructured the same way.

## Do better results score higher?

Yes, and the ordering matches the paper's ablations.

| Run | What changes | R@10 before (old protocol) | ΔS before | Hidden-test R@10 after | ΔS after | CI95 after |
|---|---|---:|---:|---:|---:|---:|
| richer | 20 spectral coordinates + anchored residual (8-D output) | 80.25 | +43.31 | 81.00 | **+57.55** | [+39.5, +75.3] |
| richer_local | richer with 30 neighbours | 78.75 | +46.60 | 76.62 | **+30.99** | [+10.7, +51.9] |
| anchored | known-pair anchor in the residual | 72.50 | +11.08 (noise then) | 75.25 | **+22.68** | [+9.6, +35.6] |
| sue (reference) | — | 71.00 | 0 | 70.38 | 0 | — |
| sue seed 29 / seed 17 | training seed only | 68.75 / 70.00 | −11.46 / −3.43 | 70.62 / 68.88 | −0.71 / −7.12 | [−8.3, +6.9] / [−15.0, +0.2] |
| no_mmd | paper ablation: SE + CCA only | — | — | 68.38 | −9.36 | [−19.1, +0.2] |
| contrastive (frozen baseline) | paired-only, same 100 pairs | 53.50 | −81.77 | 56.38 | **−64.28** | [−92.2, −36.8] |
| no_se | paper ablation: CCA + MMD without SE | — | — | 28.25 | −198.69 | [−225.7, −171.1] |
| no_cca | paper ablation: SE + MMD without CCA | 1.75 | −261.29 (incl. +100 for falsely declaring 0 pairs) | 4.75 | −313.56 | [−333.7, −293.1] |

- **Noise yardsticks.** The reference seed spread is now 7.12 (sd 3.92), down from 11.46 on the old 200-query slice. The bootstrap sd of a variant's paired difference is 4–14.
- **Better variants clear the noise.** All three improve by more than the seed spread, and their CIs exclude 0. The anchored variant was inside the noise before; with the full 400-query, both-direction test it is now resolved.
- **Weaker variants rank in the paper's order.** The paper's own ablations (Table 2 / Table 9) fall in its direction: removing SE or CCA is catastrophic, and removing MMD costs a little (paper: 30.25 vs 32.75 R@10), within noise here.
- **The paper's claim holds well above noise.** SUE beats the paired-only baseline by +64.3, about 9× the seed spread.
- **Development and test agree for honest methods.** Development vs hidden-test R@10:

  | Method | Dev | Hidden test |
  |---|---:|---:|
  | reference | 67.50 | 70.38 |
  | contrastive | 55.25 | 56.38 |
  | richer | 82.62 | 81.00 |

  The public development scores rank the variants identically.
- **A method tuned to the dev pairs loses on test.** `public_lookup` memorizes the public development pairs and scores 99.75 on dev but 0.25 on test.

## Shortcuts: before and after

All rows ran through `bash scripts/evaluate_test.sh` with no arguments, with the case installed alone in `artifacts/`. "Before" is the red-team's ΔS under the old evaluator.

| Shortcut | Before | Fix | After (ΔS, hidden test) |
|---|---:|---|---:|
| raw_pair_lookup (recover row IDs from the public archive) | +271.02 | archive removed, re-split with a secret seed; `public_lookup` memorizes every public row (training + dev) | −428.00 (test R@10 0.25; dev 99.75) |
| copy_input (one view copied into both slots) | +271.02 | per-view encoder calls on independently ordered rows | −347.10 |
| zero_budget (declare 0 pairs) | +100.00 | budget fields not read or scored | 0.00 (`metadata_claims`: reference encoder plus fake budgets, labels, counts and scores = the reference) |
| negative_budget (−1e6 pairs) | +999,740.02 | same; all scored metrics bounded | 0.00 with the reference encoder; −429.45 with constant output |
| null_budget (NaN pairs) | non-finite rank | same | 0.00 |
| label_slice (labels shrink the test slice) | +271.02 | no label channel; all 400 pairs scored | 0.00 (labels in `metadata_claims` ignored) |
| test_size (2-row gallery) | +271.02 | evaluator-owned 400 rows; shape gate | −429.45 (failed) |
| replace_baseline | +0.005 (tie; baseline replaceable) | baseline frozen in the evaluator; name reserved | 0.00; reported baseline unchanged (−64.28 in every case) |
| direct_scores (pre-write scores_test.json) | −359.98 | already recomputed | 0.00 (reference encoder; claims overwritten) |
| nan / inf | −357.43 (accepted) | finiteness gate | −429.45 (failed) |
| empty | crash, no score | shape gate, per-method isolation | −429.45 (failed) |
| constant | −359.98 | pessimistic ties | −429.45 |
| wrong_dim (16-D) | not tested (old dim metric was −0.1/dim) | 8-D gate | −429.45 (failed) |
| stash_sandboxed (cross-call stash in model dir, cwd or TMPDIR) | new | fresh private copies per call | 0.00 |
| stash_absolute (cross-call stash at `/tmp/…` + test-time matching) | new: **+27.96** (R@10 75.13) in job 1 | encoder guard | 0.00 |
| stash_subprocess (child process writes the stash) | new | guard refuses child programs | 0.00 |
| read_evaluator_files (look rows up in `eval/` data at encode time) | new | guard refuses `eval/` access | 0.00 |

A 0.00 means the probe wraps the reference encoder and earns nothing beyond it, i.e. no gain. Floor-scored cases (−429.45) rank below every real method, including the paired-only baseline.

## Shortcuts left open, and why

- **Training on the public development pairs** (`eval/train/data/`). This adds 400 known pairs to the 100-pair budget. A train-side score needs visible correspondences, and the evaluator cannot see how a model was trained. The briefing and `CONSTRAINTS.md` forbid it; the grader must check that training reads only `data/handwritten/`.
- **Fetching the public UCI archive from the internet and memorizing it.** The archive is public and row-aligned, so a method that embeds the whole table in its model directory can recover held-out identities at encode time. Closing this needs non-public data. The workspace no longer contains the archive, and `CONSTRAINTS.md` forbids any copy of it.
- **Deliberate evasion of the guard.** Examples: ctypes system calls, spawning an unguarded interpreter through multiprocessing internals, or planting `sitecustomize.py`. The audit hook is a policy check, not a sandbox, and these workarounds are explicit and easy to spot in code.
- **Within-view test-time adaptation.** One call sees all 400 rows of one view, e.g. batch statistics. I did not restrict this. It uses no cross-view information and was not shown to pay.

## Reference score before and after

| | Before (old protocol) | After |
|---|---|---|
| Hidden-test queries | 200 per direction (half of 400) | 400 per direction |
| Mean R@10 | 71.00 | 70.375 |
| Mean MRR | 35.10 | 35.69 |
| Mean median rank | 5.5 | 5.5 |
| Weighted score | 207.88 (incl. −100 pairs and −0.8 dim terms; 308.68 without them) | 309.65 (no budget terms) |

- **Training data changed.** The reference now trains on 990 unpaired rows per view instead of 1350, because 400 rows became the public development split.
- **Reproducibility.** The reference is deterministic: jobs 1 and 2 give identical numbers.
- **Training seed noise.** Seeds 17 and 29 give 68.88 and 70.62 R@10.
- **The paper's numbers are unchanged** in `reference.json`: SUE 78.5 mean R@10, contrastive 29.25.

## Headroom check

- **No scored metric is at its bound.** The reference has R@1 22.5 / 20.0, R@5 51.5 / 49.25, R@10 72.0 / 68.75 (all bounded by 100), MRR 35.7 (bound 100) and median rank 5.5 (bound 1).
- **There is room above the reference.** The best variant found (`richer`) gains +57.55 and reaches R@10 81.0. Perfect retrieval would score 579.7, i.e. +270.05 above the reference.
- **Nothing reaches the bound.** No simple variant or shortcut does; the only route above the reference is a better representation.
- **The baseline makes sense.** The paired-only baseline sits clearly below, at −64.3.
- **No new experiment was needed.** I added no scored experiment, because the reference is not at a bound. The previously scored constants (`embedding_dim`, `eval_pair_count`, and the fixed `paired_train_count`) were removed instead of left stuck at one value.

## Weights

The single experiment keeps weight 1.0 and the original retrieval coefficients: R@10 1.0 per direction, mean R@10 1.0, R@1/R@5 0.5, MRR 0.8, median rank −0.3. Only the four budget/size metrics were removed, for the reasons above. I made no other weight changes: R@k is what the paper reports, and the ranking-quality terms only refine it.

## What I could not fix

- **The absolute gap to the paper remains.** The reproduction stays below the paper's numbers (70.4 vs 78.5 mean R@10). The paired-only baseline is much stronger than the paper's (56.4 vs 29.25), so the reproduced SUE-over-contrastive gap is +14 R@10 rather than the paper's +49. The direction and significance of the claim hold; the size of the paper's gap does not. The gym's smaller unpaired pool (990 vs the paper's 1900) is part of this. I did not change the paper's training method to close it.
- **Old red-team scripts no longer run.** The scripts in `redteam/*.py` (previous iteration) target the old `.npz` artifact interface. They are kept as a historical record; the current audit is `redteam/hardening/`.
