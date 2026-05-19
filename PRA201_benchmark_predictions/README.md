# PRA201 benchmark predictions

These files contain the fold-level predictions used to compute the PRA201 benchmark metrics and post-hoc hard-case sensitivity analysis.

Columns:
- `y_true`: experimental binding free energy label (Delta G)
- `y_pred`: predicted binding free energy

The post-hoc hard-case analysis ranks samples within each held-out fold by `abs(y_pred - y_true)` and removes the highest-error fraction before recomputing PCC.
