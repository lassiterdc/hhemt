# Diagnose a failed or incomplete analysis

## Goal

Name the phase that failed, the artifact that shows the failure, and whether the raw outputs needed to retry still exist. The steps narrow the search and rerun no step.

## Prerequisites

- A completed or partially completed analysis directory.

## Key references

- [Operating on an analysis while jobs are in flight](in-flight-operations.md) describes the operations that leave running jobs in place.
- [Rerun triggers](../reference/rerun-triggers.md) explains the rerun triggers.
- The [CLI reference](../reference/cli.md) lists the commands and exit codes.

## Symptom index

Start from the symptom you have. When no entry matches or the information to pick one is missing, work through the numbered steps instead.

| Symptom | Usual meaning |
|---|---|
| [The analysis produced no output](#the-analysis-produced-no-output) | A phase never started, so find the missing flag |
| [Setup says a model is enabled but not compiled](#setup-says-a-model-is-enabled-but-not-compiled) | Native mode needs a one-time compile, and the machine has no build |
| [The job died at walltime](#the-job-died-at-walltime) | Expected, and the job resumes from the last checkpoint |
| [The report is wrong but the results are fine](#the-report-is-wrong-but-the-results-are-fine) | The simulations are sound, and the stages after them need rebuilding |
| [A resumed analysis that I expected to be a no-op re-executed everything](#a-resumed-analysis-that-i-expected-to-be-a-no-op-re-executed-everything) | A rerun trigger fired and no failure occurred |
| [Results look wrong and the cause is unknown](#results-look-wrong-and-the-cause-is-unknown) | Possibly a known bug and no configuration error |

!!! warning "A green exit is not evidence of success"
    - **The toolkit detects completion from solver log markers and ignores exit codes.** A solver can exit `0` after failing partway.
    - **In `batch_job` dispatch, a successful return means the tmux session exited.** The return carries no information about the workflow inside the session.

    Start from artifacts and never from a return code.

## 1. Ask what the toolkit thinks happened

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --status
```

The command inspects logs and outputs and reports completion per phase. The report is the fastest way to find the phase to examine, and step 1 needs no other output.

The status flags hold the same information on the filesystem.

```bash
ls analysis_dir/_status/
```

The table lists the phases in execution order. The leading letter encodes the order with one exception. The flag `f_consolidate_scenario_evt-*` runs before `e_consolidate_*` and sorts after that rung. Read progress down the table and never down a sorted listing.

| Prefix | Phase |
|---|---|
| `a_setup_*` | System setup of the DEM and Manning's roughness, plus a check that every enabled solver is compiled |
| `b_prepare_*` | Scenario preparation by SWMM `.inp` generation and boundary conditions. Emitted only when preparation runs as a separate rule. Otherwise the ladder reads `a_` then `c_` with no gap. |
| `c_run_*` | Simulation, one flag per model type and event |
| `d_process_*` | Per-scenario output processing, one flag per model type and event |
| `f_consolidate_scenario_evt-*` | Per-scenario fan-in once every enabled model has processed the event. When `remove_after_processing` elects scenario-scoped classes, the phase reclaims their inputs. Regular analyses only. The flag runs before `e_consolidate_*` despite the letter. |
| `e_consolidate_*` | Analysis-level consolidation, with `e_consolidate_complete.flag` on a regular analysis and `e_consolidate_member-*` per member of an experiment |
| `f_consolidate_experiment_*` | Experiment consolidation |

**The last rung present, reading down the table, is the phase that completed, and the failure is in the next rung down.** Each flag has a `.flag.json` sidecar that names the rule, model type, member, and event of the flag.

A cold-cache failure shows as a missing `a_setup_*` flag, with the model that is not compiled named in `logs/setup.log`, or in `logs/setup_target_{N}.log` on an experiment. In native mode the setup rule checks that every enabled solver is compiled, and a container ships the solver inside the image. [Compile the solver](compiling-the-solver.md#compilation-as-a-separate-step) describes the compile.

## 2. Read the log for the phase that did not complete

The analysis level holds one log per event and model type.

```bash
ls analysis_dir/logs/sims/
# model_{model_type}_evt{N}.log                      (regular analysis)
# model_{model_type}_{analysis_id}_evt{N}.log        (sensitivity member)
```

Look for the completion marker in the model log. TRITON writes `Simulation ends`, and standalone SWMM writes `EPA SWMM completed` with no `There are errors.` on the same line. The runner writes the line `Simulation completed successfully` to the rule log beside the model log, named `{model_type}_evt-{event_id}.log`, or `simulation_member_{member_id}_evt_{event_id}.log` on an experiment. **A job that `sacct` reports as COMPLETED with the marker absent signals a silent early exit.** The combination means the process was reaped or returned early, so the elapsed time that the scheduler reports differs from the time the solver ran.

!!! danger "The per-simulation logs are truncated on every attempt"
    The per-simulation log opens in `"w"` mode on **every** execution, so the log holds only the most recent attempt, and retries erase the earlier attempts. The scheduler logs hold the history per attempt.

    ```bash
    ls analysis_dir/.snakemake/slurm_logs/
    ```

    The directory also holds the stderr of the runner, including the `Command:` line that shows the executed command. Use the line to confirm that a container wrap or an MPI launcher behaved as expected.

## 3. Match the symptom

### The analysis produced no output

When no `c_run_*` flag exists in `analysis_dir/_status/`, the simulation never started. The cause is upstream in setup, compilation, or dispatch. When a `c_run_*` flag exists and no `d_process_*` flag does, the simulation completed. Processing failed. The raw outputs stay on disk under `sims/{event_id}/out_*/`, so no output is lost and `reprocess` can retry.

### Setup says a model is enabled but not compiled

No `a_setup_*` flag exists, and `logs/setup.log`, or `logs/setup_target_{N}.log` on an experiment, ends with a line of this form.

```text
TRITON-SWMM is enabled but not compiled and --compile-triton-swmm not specified
```

The TRITON-only and SWMM checks write the same line with their own names and flags. The flag in the line belongs to the standalone compile command. `hhemt run` and `analysis.run()` both reject the flag. Compile once, following [Compile the solver](compiling-the-solver.md), and start the analysis again. Setup work that completed stays.

### The job died at walltime

Raise the simulation retry count in place of the walltime when checkpoints are frequent enough. [Scale and hardware](../explanation/limitations.md#scale-and-hardware) states that a killed simulation resumes from the latest checkpoint.

??? note "`perf_*` exceeding the scheduler's elapsed time on a resumed analysis"
    [A caveat on performance columns](../reference/output-data-model.md#a-caveat-on-performance-columns) explains the excess and why the excess is correct.

### The report is wrong but the results are fine

Avoid a rerun. Use `reprocess` to repeat processing, consolidation, and rendering against the existing simulation outputs.

### A resumed analysis that I expected to be a no-op re-executed everything

A rerun trigger fired and no failure occurred. [Rerun triggers](../reference/rerun-triggers.md) explains the triggers.

### Results look wrong and the cause is unknown

Run `hhemt check-invalidating-fixes` and then `hhemt recompute-plan` to list the work that needs re-running. The [CLI reference](../reference/cli.md#checking-correctness) describes both commands.

## 4. Read the exit code

The CLI uses one structured exit code per failure class. The [CLI reference](../reference/cli.md#exit-codes) lists the codes and the remedy for an unmapped code.

## 5. Check the validation section of the report

A rendered report carries an **Errors and Warnings** section, and [The interactive analysis report](../reference/example-report.md#sidebar-sections) lists the contents. On an analysis that completed far enough to render, read the section before the logs, because the section holds the same information, already triaged.

## Before you rerun

Re-running a whole analysis to fix a downstream problem is the most common expensive mistake. The options run in increasing order of cost.

1. `hhemt reprocess --start-with render` when the report is wrong and the data is fine.
2. `hhemt reprocess --start-with consolidate` or `process` when processing or consolidation is wrong and the simulations are fine.
3. `hhemt run` with `--override-force-rerun` scoped to the work that needs a rerun.
4. A full rerun.

Scope option 3 as [Forcing and suppressing reruns](forcing-reruns.md) describes.
