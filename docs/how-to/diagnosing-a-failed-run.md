# Diagnose a failed or incomplete run

**Goal:** find out what actually went wrong, in an order that narrows quickly,
without re-running anything.

**Prerequisites:** a completed or partially-completed analysis directory.

This is the page for *"it said it ran and produced nothing"* and *"the job died
at eleven hours."* Start from the symptom you actually have. If no entry in the
symptom index matches, or you do not yet know enough to pick one, work through
the numbered steps instead.

## Symptom index

| Symptom | What it usually means |
|---|---|
| [It ran but produced nothing](#it-ran-but-produced-nothing) | A phase never started — find which flag is missing |
| [The job died at walltime](#the-job-died-at-walltime) | Expected; it resumes from its last checkpoint |
| [The report is wrong but the results are fine](#the-report-is-wrong-but-the-results-are-fine) | The simulations are sound; the stages after them need rebuilding |
| [A run I expected to be a no-op re-executed everything](#a-run-i-expected-to-be-a-no-op-re-executed-everything) | **Not a failure** — a re-run trigger fired |
| [Results look wrong and I do not know if it is me](#results-look-wrong-and-i-do-not-know-if-it-is-me) | Possibly a known bug rather than your configuration |

!!! warning "A green exit is not evidence of success"
    Two things about this toolkit make that specifically true, and both are
    deliberate rather than defects:

    - **Completion is detected from solver log markers, not exit codes.** A
      solver can exit `0` having failed partway.
    - **In `batch_job` dispatch, a successful return means the tmux session
      exited** — not that the workflow inside it succeeded.

    So start from artifacts, never from a return code.

## 1. Ask what the toolkit thinks happened

```python
status = analysis.get_workflow_status()
```

This inspects logs and outputs to report per-phase completion. It is the fastest
way to find out *which phase* to look at, which is the only thing you need from
step 1.

If you prefer the filesystem, the same information is in the status flags:

```bash
ls analysis_dir/_status/
```

The leading letter encodes phase order, so a sorted listing reads as progress:

| Prefix | Phase |
|---|---|
| `a_setup_*` | System setup — DEM, Manning's, compilation |
| `b_prepare_*` | Scenario preparation — SWMM `.inp` generation, boundary conditions. Emitted only when preparation runs as its own rule; on a run where it does not, the ladder reads `a_` then `c_` with no gap. |
| `c_run_*` | Simulation |
| `d_process_*` | Per-scenario output processing |
| `e_consolidate_sa-*` | Per-sub-analysis consolidation |
| `f_consolidate_master_*` | Master consolidation |

**The last prefix present is the phase that completed; the failure is in the next
one.** Each flag has a `.flag.json` sidecar naming the rule, model type,
sub-analysis and event it belongs to.

## 2. Read the log for the phase that did not complete

Per-simulation logs are written at the analysis level, one per
(event, model type):

```bash
ls analysis_dir/logs/sims/
# model_{model_type}_evt{N}.log                      — regular analysis
# model_{model_type}_{analysis_id}_evt{N}.log        — sensitivity sub-analysis
```

Look for the completion marker. TRITON writes `Simulation ends`; the runner
writes `simulation completed successfully`. **Absence of the marker on a job that
`sacct` reports COMPLETED is the signature of a silent early exit** — that
combination means the process was reaped or returned early, so the elapsed time
the scheduler reports is not the time the solver ran.

!!! danger "These logs are truncated on every attempt"
    The per-simulation log is opened in `"w"` mode on **every** exec, so it holds
    only the most recent attempt. If retries fired, earlier attempts are gone
    from here.
    For per-attempt history read the scheduler logs instead:

    ```bash
    ls analysis_dir/.snakemake/slurm_logs/
    ```

    That directory also holds the runner's own stderr, including the `Command:`
    line showing exactly what was executed — which is where to confirm a
    container wrap or an MPI launcher did what you expected.

## 3. Match the symptom

### It ran but produced nothing

If no `c_run_*` flag exists in `analysis_dir/_status/`, the simulation never
started, and the cause is upstream — setup, compilation, or dispatch. If
`c_run_*` exists but `d_process_*` does not, the simulation ran and processing
failed; the raw outputs are still on disk under `sims/{event_id}/out_*/`, so
nothing is lost and `reprocess` can retry.

### The job died at walltime

A killed simulation resumes from its most recent
checkpoint on the next attempt — it does not restart from zero. Raise the
simulation retry count rather than the walltime if checkpoints are frequent
enough. Note that `perf_*` columns on a resumed run are cumulative across
allocations, so they will exceed what the scheduler reports for the final
allocation; that is correct.

### The report is wrong but the results are fine

Do not re-run. Use
`reprocess`, which re-runs processing, consolidation and rendering against
existing simulation outputs.

### A run I expected to be a no-op re-executed everything

A re-run trigger fired; nothing failed — see
[When and why re-runs happen](../explanation/rerun-faq.md).

### Results look wrong and I do not know if it is me

Run
`hhemt check-invalidating-fixes` — it reports whether a known invalidating bug
matches this analysis, i.e. whether its outputs are suspect for a reason that has
already been identified and fixed. `hhemt recompute-plan` then shows what would
need re-running.

## 4. Read the exit code, if you have one

The CLI uses structured exit codes — `2` config, `3` workflow/compilation, `4`
simulation, `5` processing. See the [CLI reference](../reference/cli.md#exit-codes).

**A `10` is the catch-all**, and what to do about one is on that same page.

## 5. Check the report's own validation section

A rendered report carries an **Errors and Warnings** section built from the same
post-completion validation the toolkit runs internally — system-level checks,
aggregate per-scenario checks, granular per-scenario failures, and
resource-utilisation mismatches. On an analysis that completed far enough to
render, read that before reading logs: it is the same information, already
triaged.

**Verifiable end state:** you can name the phase that failed, the artifact that
shows it, and whether the raw outputs needed to retry still exist.

## Before you re-run

Re-running a whole analysis to fix a downstream problem is the most common
expensive mistake here. In increasing order of cost:

1. `render_report()` — the report is wrong, the data is fine.
2. `reprocess()` — processing or consolidation is wrong, the simulations are fine.
3. `run()` with force-rerun scoped to what actually needs it.
4. A full re-run.

See [Forcing and suppressing re-runs](forcing-reruns.md) for how to scope 3.

## See also

- [Operating on an analysis while jobs are in flight](in-flight-operations.md) — monitoring a run that has not failed yet.
- [When and why re-runs happen](../explanation/rerun-faq.md)
- [CLI reference](../reference/cli.md)
