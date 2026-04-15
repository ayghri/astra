# autoresearch — ADMM pruning sweep

Autonomous hyperparameter search for `admm.py`. Goal: minimize layer-0
reconstruction loss under a fixed 2:4 structured sparsity constraint.

## Setup

1. **Read in-scope files** for context:
   - `admm.py` — the reference implementation. **Do not modify this file.**
     Every experiment is a new copy named `admm_v{N}.py`.
   - `../admm_vs_gpt.md`, `../plan.md` — background on the algorithm and why
     it exists. Skim, don't obsess.
2. **Verify inputs exist**: `/scratch/alpine/aygh9582/checkpoints/layer_0_W.cpt` and
   `/scratch/alpine/aygh9582/checkpoints/layer_0_X.cpt`. If missing, stop and tell the user.
3. **Python**: every run uses `/projects/aygh9582/envs/astra/bin/python` from the mamba env. Not `python`,
   not `uv run`. The `sparsekit` and `torch` builds live in that venv.
4. **Initialize `results.tsv`** with just the header row (see Logging). The
   baseline row is written after run 1.
5. **No git branch**: versioned files serve the same purpose as commits here.
   Do not commit `admm_v*.py` or `results.tsv` — leave them untracked.

## The search space

Fixed (do not change):
- `W_PATH`, `X_PATH` — same checkpoints for every run.
- The sparsity pattern — `BlockSpec(Z, (1,1), ...)` + `ScopeSpec(..., (1,4))`
  with `nnz=2` at mask extraction. This IS the 2:4 constraint we are
  pruning under. Changing group shape changes the problem.
- `compute_H` and the loss formulation `((dW @ H) * dW).sum() * N`.
- FP16/FP32 mixed-precision strategy in `mm16`. Never introduce TF32.

Fair game:
- `PSI_BETA` (EMA on lamb, currently 0.995) — controls how fast the
  threshold schedule adapts.
- `NUM_ITER` (2000) — total ADMM iterations. Raising helps until it
  plateaus; raising too far just wastes time.
- `K_VAL_WEIGHT` (1999.0) — weighting inside `g_spec.kth_mid`.
- `PERCDAMP` (1e-4) — Hessian diagonal damping.
- `MAX_PSI` (0.0002) — clamp on `psi_pre` before `kth_mid`.
- `log_step` — logging cadence; purely cosmetic, leave alone unless you
  have a reason.
- Algorithmic variants already sketched in the commented-out blocks:
  - `E_val = V.abs() * sqrt(rho)` as conditioner instead of `rho_cond`.
  - `psi_pre = mm16(W, H_offdiag_h) - C_target` (gradient-based) instead
    of `(W+U) * rho_diag` (proximal-based).
  - Initialize `Z` from zeros instead of `W0.clone()`.
  - Use the mask derived from `E_val` or `z_clone` instead of current Z.
  - Turn on `skip_ols=False` for a final OLS refine on the chosen mask.
- Anything else that doesn't violate the fixed list above (new update
  rules, line searches, restarts, warm starts, etc.).

**Constraint**: the `nnz=2` mask in the final `g_spec.get_masks` call must
stay. If your variant produces a mask that isn't 2:4, it is invalid and
should be logged as `crash`.

## Experiment file layout

Each experiment is one file: `admm_v{N}.py`, where `N` increments
monotonically starting at 1. Copy `admm.py`, edit the copy, run the copy.

```bash
cp admm.py admm_v1.py
# edit admm_v1.py
/projects/aygh9582/envs/astra/bin/python admm_v1.py > run_v1.log 2>&1
```

Do not reuse `N`. Do not rewrite an existing `admm_v{N}.py`. Each run gets
a new file so the full history is reconstructible from the filesystem.

Keep the `if __name__ == "__main__": main()` block at the bottom.

## Metric

Two loss numbers are printed during a run:

1. Intermediate losses inside the ADMM loop (every `log_step` iters),
   prefixed `Loss:`. These track progress but are NOT the reported metric.
2. The **final** line after the loop exits (and optional OLS refinement):

   ```
   Loss: 123456.78
   ```

The **last** `Loss:` value in the log is the metric. Lower is better.

Extract it with:

```bash
grep "^Loss:" run_v{N}.log | tail -n 1
```

Also capture wall time from the `The whole thing took Xs` line.

## Logging results

Append one row per run to `results.tsv` (tab-separated). Columns:

```
version	loss	time_s	status	psi_beta	num_iter	k_val_weight	percdamp	max_psi	kappa	skip_ols	description
```

- `version` — the integer `N` from `admm_v{N}.py`.
- `loss` — final loss, `0.0` on crash.
- `time_s` — from the "The whole thing took" line, `0.0` on crash.
- `status` — `keep`, `discard`, or `crash`.
  - `keep` if this run is the new best (strictly lower loss than the
    current best kept run).
  - `discard` if it ran cleanly but didn't improve.
  - `crash` if it failed or produced a non-2:4 mask.
- The numeric columns after `status` record the hyperparameters used
  **in this run**, even for values you did not change. This way the TSV
  is self-contained — the agent can later reconstruct the best config
  without re-reading source files.
- `description` — short text, no tabs, explaining what this run tried.

Example:

```
version	loss	time_s	status	psi_beta	num_iter	k_val_weight	percdamp	max_psi	kappa	skip_ols	description
1	1523.42	48.1	keep	0.995	2000	1999.0	1e-4	2e-4	2	True	baseline
2	1489.17	47.8	keep	0.99	2000	1999.0	1e-4	2e-4	2	True	faster EMA
3	1495.00	95.3	discard	0.99	4000	1999.0	1e-4	2e-4	2	True	more iters
4	0.0	0.0	crash	0.99	2000	1999.0	1e-4	2e-4	2	True	E_val conditioner: NaN at iter 60
```

## The loop

Each experiment takes ~30–90s of GPU time depending on `NUM_ITER` and
whether OLS refinement runs. If a run exceeds 10 minutes, kill it and log
`crash`.

LOOP FOREVER:

1. Pick the next unused `N`.
2. `cp admm.py admm_v{N}.py` (always start from the original baseline,
   not from the previous best — we are sweeping, not chaining edits).
   Exception: if you deliberately want to build on a previous variant,
   copy from `admm_v{M}.py` instead and say so in the description.
3. Edit `admm_v{N}.py`: change hyperparameters or the algorithmic variant
   you want to test.
4. Run: `/projects/aygh9582/envs/astra/bin/python admm_v{N}.py > run_v{N}.log 2>&1`.
5. Extract final loss and time:
   `grep -E "^Loss:|^The whole thing took" run_v{N}.log | tail -n 3`.
6. If no `Loss:` line, tail 50 lines of the log and diagnose. Obvious
   fixes (typo, import) — fix and retry **in the same file**. Otherwise
   log as `crash` and move on.
7. Append row to `results.tsv`. If this run beats the current best, mark
   `keep`; otherwise `discard`.
8. Keep going.

**NEVER STOP** the loop on your own. The user expects unattended
operation. Out of ideas? Re-read `admm.py` for knobs you haven't touched,
read `admm_vs_gpt.md` for angles, try combinations that individually
helped, read the commented-out code blocks in `main()` — several
unexplored variants are sitting right there.

## Reporting

When the user asks for a report (or when explicitly told to stop), do
**not** dump the whole TSV. Produce:

1. The best `admm_v{N}.py` by loss, with its full hyperparameter row.
2. The top 5 runs sorted ascending by loss.
3. A 3–5 bullet summary of what mattered: which knobs moved the needle,
   which did nothing, which broke things. Ground every claim in specific
   row numbers from the TSV.
4. One paragraph of recommended next directions if the user wanted to
   continue the sweep.

Do not speculate beyond what the TSV supports. If two runs tie on loss,
prefer the one with lower `time_s`, then the simpler change.
