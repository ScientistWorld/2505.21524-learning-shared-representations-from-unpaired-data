# Restoration and final checks

Starting commit: `0d1294352912c5ab64ee0c61458621d24d2df21e`.

After all three jobs, the audit ran:

```sh
git restore --source=0d1294352912c5ab64ee0c61458621d24d2df21e --staged --worktree -- . ':(exclude)redteam' ':(exclude)REDTEAM.md'
```

This restored every editable original file, including `scripts/run.sh`, all
score JSON files, and the original `action.yaml`. Automatically recorded job
files outside `redteam/` were removed; copies of all stdout/stderr evidence
remain in `redteam/job_logs/`. The final `action.yaml` is then set to the
mandatory done signal.

The restoration command reported errors for system mounts that existed before
the audit: `README.md` and `manifest.json` are read-only devtmpfs `/null` mounts;
`trajectory/` is a read-only mount of the system's empty trajectory directory.
They cannot be unlinked or written. The initial state is recorded in
`start.json`. These mounted working-tree differences are not audit changes;
the git index for them matches the starting commit. No unmount or mount change
was attempted.

Final checks confirmed:

- All editable original files match the starting commit. This includes the
  pristine hidden evaluator, train evaluator, all scripts, method, baseline,
  data, scoring contract, environment, briefing, and original milestone.
- The index outside `redteam/` and `REDTEAM.md` matches the starting commit.
- The only remaining working-tree differences outside the audit, before the
  final done signal, are the six pre-existing mounted paths listed above.
- All 29 cases have `run.sh` and `method.py`; every successful finite score has
  its saved hidden-test JSON. All 13 shortcut cases are represented.
- Every saved finite rank was recomputed with all original metric coefficients
  and experiment weights. All Python files compile and all case shell scripts
  pass syntax checks. No extra model computation was run on the login node.

Machine-readable evidence is in [verification.json](verification.json).
