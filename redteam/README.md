# Scoring audit

Run the bundled CPU audit with `bash redteam/run_all.sh` in the gym container.
Each case also has its own `run.sh`. Each invocation installs candidate
artifacts into the default `artifacts/`, runs the public train script, and
then launches **`bash scripts/evaluate_test.sh` with no arguments** in a
separate process. No evaluator or environment patch is used. Original artifacts
and score files are restored after each invocation, including failed cases.

`honest.py` implements genuine variants using the existing method components.
`shortcuts.py` uses only the public data, public train evaluator, and candidate
artifacts. It never imports or reads the hidden evaluator or hidden scores.
`runner.py` alone reads the original hidden scores to calculate audit deltas;
those numbers are never supplied to a candidate.

The fixed data split is seed 13 throughout. Reference seeds 13, 17, and 29
change only residual-network initialization and minibatch order, avoiding a
confounded change in test examples. All training uses CPU with one BLAS thread.
The original default artifact is retained for comparisons as `reference_stored`.

Per-case `scores_test.json`, `scores_train.json`, `result.json`, and logs are
saved here. Large generated embedding/checkpoint caches are intentionally
ignored. `run_all.sh` saves completed cases immediately and resumes them.
