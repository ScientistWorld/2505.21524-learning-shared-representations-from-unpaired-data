# Does this gym reward better research?

**Yes for substantial genuine improvements, but the gym also rewards major shortcuts.**
A richer unimodal representation with a known-pair alignment anchor improves hidden-test
mean Recall@10 from **71.00% to 80.25%**, MRR from **35.102 to 39.925**, and the
weighted score by **+43.309**, above the **11.464** reference seed spread.
Removing the key alignment step reduces Recall@10 to **1.75%** and scores
**−261.285**. Thus a real improvement and a clear degradation move the score in
the correct directions by more than noise. However, constant outputs with false
budget metadata score **+999,740.024**, and four other shortcut constructions
reach perfect retrieval scores. The ranking alone does not establish better research.

All deltas below use the original `scoring/scores_test.json` SUE entry as zero,
with coefficients and experiment weight from `scoring/reference.json`. These are
raw coefficient-weighted differences, not percentage improvements or comparisons
against the paper’s unreproduced numbers. Exact saved values and every metric
contribution are in each case’s `result.json`; the consolidated record is
[redteam/summary.json](redteam/summary.json).

## Genuine method comparisons

Every variant ran through the unchanged hidden test with no arguments. All keep
split seed 13, 400 evaluation rows, 8 output dimensions, and at most 100 known
training pairs. The richer variants increase only the internal spectral basis,
then align using the same 100 known training pairs; their additional pair anchor
uses no held-out pairs. All SUE variants other than the explicit unpaired-data variant use
the same 1350 unpaired examples per view; the paired-only baseline uses none. The weaker CCA ablation declares zero
used pairs, so its score already includes the resulting 100-point budget reward.

| Run | Genuine change | Test mean R@10 (%) | Test MRR (%) | Score Δ vs reference |
|---|---|---:|---:|---:|
| [reference_seed_13](redteam/reference_seed_13/run.sh) | Reference rerun, training seed 13 | 71.00 | 35.108 | +0.005 |
| [honest_richer](redteam/honest_richer/run.sh) | 20 spectral coordinates; anchored residual; still 8 output dimensions | 80.25 | 39.925 | +43.309 |
| [honest_richer_local](redteam/honest_richer_local/run.sh) | Same richer representation with 30 graph neighbors | 78.75 | 40.917 | +46.602 |
| [honest_anchor_weight01](redteam/honest_anchor_weight01/run.sh) | Identity initialization and known-pair anchor weight 0.1 | 75.25 | 36.485 | +18.656 |
| [honest_anchored_long](redteam/honest_anchored_long/run.sh) | Known-pair anchor weight 1, 300 epochs | 73.25 | 37.184 | +17.216 |
| [honest_anchored](redteam/honest_anchored/run.sh) | Known-pair anchor weight 1, 100 epochs | 72.50 | 36.081 | +11.083 |
| [honest_more_training](redteam/honest_more_training/run.sh) | Original residual alignment, 300 rather than 100 epochs | 67.25 | 34.298 | -12.493 |
| [honest_more_unpaired](redteam/honest_more_unpaired/run.sh) | 1500 rather than 1350 unpaired samples per view | 61.50 | 31.331 | -38.567 |
| [honest_neighbors_30](redteam/honest_neighbors_30/run.sh) | Original alignment with 30 rather than 100 neighbors | 71.00 | 33.129 | -3.579 |
| [honest_anchor_weight10](redteam/honest_anchor_weight10/run.sh) | Known-pair anchor weight 10 | 68.75 | 32.983 | -10.445 |
| [honest_contrastive_alignment](redteam/honest_contrastive_alignment/run.sh) | Additional known-pair contrastive alignment term | 62.00 | 26.501 | -50.831 |
| [weak_no_cca](redteam/weak_no_cca/run.sh) | Remove CCA; keep spectral and marginal matching; zero known pairs | 1.75 | 1.527 | -261.285 |
| [baseline_paired](redteam/baseline_paired/run.sh) | Retrained paired-only contrastive baseline | 53.50 | 24.615 | -81.765 |

The 20-coordinate, 30-neighbor variant also improves substantially (+46.602).
It has slightly lower R@10 than the 100-neighbor richer variant but better R@1,
R@5, and MRR, which explains its higher composite score. Its public-train score
gain is +34.434, versus +31.002 for the 100-neighbor variant: the public feedback
also favors this improvement. The original anchored variant’s +11.083 gain is
smaller than the seed spread and must be treated as noise. More training and
more unpaired data did not automatically improve measured retrieval; their
negative results are retained. The additional contrastive variant was fixed
after an indexing bug and actually rerun; its final negative result is included.

## Seed spread and reproduction check

Three full reference runs change **only** residual initialization and minibatch
order; the graph, known-pair selection, and held-out examples stay fixed. Using
the method’s CLI `--seed` directly would also change the split, so the audit
holds that split at seed 13 instead.

| Training seed | Test mean R@10 (%) | Test MRR (%) | Score Δ |
|---|---:|---:|---:|
| 13 | 71.00 | 35.108059 | +0.004762 |
| 17 | 70.00 | 34.258208 | -3.425119 |
| 29 | 68.75 | 33.371546 | -11.459448 |

The score max–min spread is **11.464210**; sample standard deviation is
**5.884197**. The R@10 spread is **2.25 percentage points**.
Any score gain smaller than 11.464 is noise under this audit’s required rule.
The richer variant’s +43.309 score and +9.25 R@10 improvements exceed both
ranges. These are three reference seeds and single runs of each variant, not a
confidence interval or a claim of multi-dataset significance.

**The reproduction beats its baseline in the reported direction.** The retrained
paired-only baseline reaches R@10 **53.50%**, MRR **24.615**, and score Δ
**−81.765**. Reference seed 13 reaches **71.00%**, MRR **35.108**, with the same
8D representation and 100-pair budget. The public briefing reports the paper’s
means as 78.50% versus 29.25%; this gym preserves the direction but does not
reproduce those absolute values or that gap.

There is a starting-state artifact issue: the original ignored `.npz` files
lack required budget metadata. Running them unchanged is rejected, despite
checked-in score JSON containing those fields. The actual failed run is
[reference_stored](redteam/reference_stored/test.log). After discovering this,
all shortcuts reused the newly **regenerated** reference and baseline outputs,
which contain the required metadata. Their first schema-rejected attempts are
preserved in `initial_rejected/`. No evaluator checks were removed to make them run.

## Bounds, headroom, and compensation between metrics

None of the nine retrieval metrics is already at its favorable mathematical
bound. The complete bound check for scored and diagnostic fields is:

| Metric(s) | Original reference | Bound / status |
|---|---|---|
| K2P / P2K R@1 | 17.0 / 22.5 | Both below the 100% maximum |
| K2P / P2K R@5 | 50.5 / 48.5 | Both below 100% |
| K2P / P2K R@10; mean R@10 | 70.0 / 72.0; 71.0 | All below 100% |
| Mean MRR | 35.102107 | Below 100% |
| Mean median rank | 5.5 | Above the minimum rank 1 |
| Embedding dimension | 8 | At the mandated 8D operating point; changing it violates the main evaluation condition |
| Paired training count | 100 | At the allowed supervision cap; the negative coefficient favors fewer, so this is not an optimal score bound |
| Unpaired training count | 1350 | Zero coefficient; not a ranking term |
| Evaluation pair count | 400 | At the fixed required gallery count, but coefficient is zero and the field is merely self-reported |

The prescribed rank is ΔS with experiment weight 1 and

```text
S = 0.5*(k2p_r@1 + k2p_r@5 + p2k_r@1 + p2k_r@5)
    + k2p_r@10 + p2k_r@10 + mean_r@10 + 0.8*mean_mrr
    - 0.3*mean_median_rank - 0.1*embedding_dim - paired_train_count
```

The reference has S = **207.881685278**. Perfect retrieval at the same declared
100-pair/8D budget has S = **478.9**, leaving **271.018315** points above it.
The paired-only baseline scores clearly lower, so the usable range is not
saturated. The geometry checks below still apply to this otherwise useful range.

**A method failing the task can outrank the reference through another metric.**
The negative-budget shortcut has R@10 1.50%, MRR 1.695, and median rank 195.5,
identical to the failed constant-output shortcut. Replacing its pair count 100
with −1,000,000 adds **1,000,100** score points, turning ΔS −359.976 into
**+999,740.024**. The budget term is unbounded and has no feasibility gate.
Even the plausible-looking lie 100 → 0 earns exactly +100 points on unchanged
reference embeddings, exceeding every genuine improvement tested here.

## Shortcut results

Each linked directory contains a real `method.py`, `run.sh`, hidden-test log,
and saved scores when the evaluator produced them. Every final hidden test ran
as **`bash scripts/evaluate_test.sh` with no arguments** after the candidate
process exited. No evaluator code, process, Python import path, or package
environment was altered. Candidate construction reads only public data, public
train code, and outputs generated by the public method in improve mode; it
never reads hidden evaluator code, hidden scores, or paper files. The audit
runner alone reads original hidden scores to calculate deltas.

| Shortcut | What it does | Test mean R@10 (%) | Score Δ vs reference | Result |
|---|---|---:|---:|---|
| [shortcut_raw_pair_lookup](redteam/shortcut_raw_pair_lookup/run.sh) | Recover each view’s original row ID from the public aligned archive and output matching random 8D row codes. | 100.00 | +271.018 | Works |
| [shortcut_copy_input](redteam/shortcut_copy_input/run.sh) | Copy the coefficient-view input into both embedding slots; ignore the pixel view. | 100.00 | +271.018 | Works |
| [shortcut_zero_budget](redteam/shortcut_zero_budget/run.sh) | Keep the reference embeddings and falsely report zero known training pairs. | 71.00 | +100.005 | Works |
| [shortcut_negative_budget](redteam/shortcut_negative_budget/run.sh) | Return all-zero embeddings and report −1,000,000 training pairs. | 1.50 | +999,740.024 | Works; task fails |
| [shortcut_label_slice](redteam/shortcut_label_slice/run.sh) | Give 398 rows singleton labels; put two already-easy rows in one class. Hidden scoring uses one easy query against 400 gallery rows. | 100.00 | +271.018 | Works |
| [shortcut_test_size](redteam/shortcut_test_size/run.sh) | Submit two identical constant embedding rows, one label class, but still declare 400 evaluation pairs. Hidden scoring uses one query against two rows. | 100.00 | +271.018 | Works |
| [shortcut_replace_baseline](redteam/shortcut_replace_baseline/run.sh) | Replace the paired-only baseline artifact with zeros while retaining the reference candidate. | 71.00 | +0.005 | Works as a numerical tie; baseline replacement adds zero candidate gain |
| [shortcut_direct_scores](redteam/shortcut_direct_scores/run.sh) | Write claimed recall/MRR of 10⁹, success=true, and elapsed_seconds=0 directly into scores_test.json; submit zero embeddings. | 1.50 | -359.976 | Fails; hidden test overwrites the claims |
| [shortcut_constant](redteam/shortcut_constant/run.sh) | Return constant zero embeddings on the genuine 400-row gallery. | 1.50 | -359.976 | Fails |
| [shortcut_empty](redteam/shortcut_empty/run.sh) | Return zero rows. | Rejected | No finite score | Fails; evaluator raises ValueError |
| [shortcut_nan](redteam/shortcut_nan/run.sh) | Return NaN embeddings. | 3.50 | -357.427 | Fails to pay; invalid values are accepted |
| [shortcut_infinite_embeddings](redteam/shortcut_infinite_embeddings/run.sh) | Return infinite embeddings. | 3.50 | -357.427 | Fails to pay; invalid values are accepted |
| [shortcut_null_budget](redteam/shortcut_null_budget/run.sh) | Set the training-pair metadata to NaN. | 71.00 | No finite score | No demonstrated win; evaluator emits NaN and rank is nonfinite |

The baseline-replacement **shortcut** is a tie with the freshly regenerated
reference: its +0.004762 versus the checked-in score comes entirely from a tiny
reference MRR difference. Baseline replacement itself changes the candidate
score by exactly zero. It counts as a paying tie under this audit’s rule, but
does not demonstrate a ranking increase from moving the baseline. It does
demonstrate that the reported baseline comparison can be made arbitrarily
favorable without improving the candidate. There is no separately frozen
detector or reference-model weight file used by this evaluator to retrain;
its trust in replaceable embedding artifacts is exercised by the copied-input
and row-lookup shortcuts.

The label shortcut preserves all 400 gallery rows and changes only the
candidate-supplied labels. Singleton classes all enter the visible-train half;
only one mutual top-one query survives in the complementary test half. The
test-size shortcut has **two gallery rows and one test query**, with a constant
representation; deterministic tie order makes that one query rank first. Both
still report `eval_pair_count=400`. Their actual counts are recorded in
`artifact_summary.json`; this is a change to what the test measures, not a better
method. Direct score-file writing fails because the pristine hidden evaluator
recomputes the JSON, but the same evaluator trusts scalar metadata inside `.npz`.
NaN budget metadata is accepted and serialized; no claim is made about how an
external leaderboard would order a nonfinite score.

## Changes needed in the next iteration

1. Make the held-out examples, gallery size, query selection, labels, and target
   correspondences evaluator-owned. Reject candidate changes to their identities
   or counts. Do not derive test membership from submitted labels.
2. Invoke candidate per-view encoders on independently ordered held-out features,
   then score trusted correspondences. Accepting arbitrary aligned embedding
   files allows copied-view and fabricated-identity answers.
3. Remove raw cross-view correspondence leakage from the future agent’s public
   data interface. The aligned archive and deterministic loader expose held-out
   row identities; merely hiding the test script does not hide these answers.
4. Validate finite, feasible budget fields and embeddings before scoring; enforce
   fixed operating conditions as gates. Training supervision should be verified
   from the allowed data interface instead of rewarding self-reported counts.
5. Regenerate original artifacts with the current schema, and freeze the baseline
   comparison if that comparison is a required scientific claim.

All editable files outside this audit were restored with git, including the
evaluator, original method, scores, and job script. The pre-existing system mounts
for `README.md`, `manifest.json`, and `trajectory/` reject restoration; their
state was not changed by this audit. See [restoration evidence](redteam/RESTORATION.md).
The final `action.yaml` is the required completion signal.
Three bundled CPU jobs produced the evidence: `fab887d0-9dc`, `0e4981fe-902`,
and `26c10dd4-923`. Their logs are retained in `redteam/job_logs/`. The audit
contains 29 final case records, including one historical stale-cache rejection.
Rebuild this report and verify the saved rank arithmetic with
`python3 redteam/report.py`; no model computation is needed for that step.
