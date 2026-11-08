# First job: fab887d0-9dc

Completed successfully as a CPU job. Accessible log path in this workspace:
`redteam/job_logs/fab887d0-9dc_{stdout,stderr}.log` (the host absolute path in the completion
message is outside this mount).

Fixed-split reference seeds 13/17/29 have score deltas +0.004761905,
-3.425118885, -11.459448160 versus the original hidden reference. Their range
is 11.464210065. Mean R@10 is 71.0, 70.0, 68.75, a range of 2.25 points.
The known-pair-anchored genuine variant gains 11.082793583, slightly less than
the score range, and gains only 1.5 points of R@10. No robust improvement has
yet been established. Removing CCA gives -261.284978825 and R@10 1.75;
the paired-only baseline gives -81.765007308 and R@10 53.5.

The original ignored reference artifacts are stale: their files lack required
budget metadata, although checked-in scores have those metrics. Consequently,
all shortcuts reusing those particular files failed schema validation before
the property of interest could be tested. These first failed runs remain in
each shortcut's `initial_rejected/`. Candidate code is fixed to reuse the
**newly generated**, fully public reference_seed_13 and baseline_paired outputs
instead. This does not change evaluator code or fill in fabricated metadata.

The second bundled job adds real variants with different anchoring strengths,
longer anchored training, instance-discriminative alignment, and a richer
unimodal spectral representation. It reruns all shortcuts with valid inputs.
