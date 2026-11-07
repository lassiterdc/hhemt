# Forcing and suppressing re-runs

## Goal

Override the default resume behavior by forcing a re-run, adding work to a running sweep, or regenerating processed results without simulating again.

## Prerequisites

- An analysis started with `hhemt run`.

## Key references

- [Rerun triggers](../reference/rerun-triggers.md) explains the default resume behavior.
- The [CLI reference](../reference/cli.md) covers the options of each command below.
- [Running an experiment definition](running-an-experiment-definition.md#run-modes-and-the-wipe-guard) describes the wipe guard.

## Force a full re-run of specific scenarios

Pass `--override-force-rerun` to `hhemt run`.

??? note "Flag and log records cleared by a forced re-run"
    The invalidation has a flag layer and a log layer. A forced re-run deletes the `_status/*.flag` markers and clears the per-scenario processing-log records, because both layers track completion. [Rerun triggers](../reference/rerun-triggers.md) gives the reason.

```bash
# Re-run everything, regardless of the recorded completion state
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --override-force-rerun all

# Re-run only specific scenarios (by event index)
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --override-force-rerun '{"event_iloc": [0, 2]}'

# For a sensitivity analysis, target specific members
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --override-force-rerun '{"sa_id": ["member_3", "member_7"]}'

# Force from a later stage (re-render the plots and the report, no re-simulation)
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --override-force-rerun '{"subject": "all", "stage": "render"}'

# Re-process and re-consolidate two members, leaving their simulations alone
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --override-force-rerun '{"subject": {"sa_id": ["member_3", "member_7"]}, "stage": "process"}'
```

`--override-force-rerun` carries a subject axis and a stage axis. The subject names the scenarios to force, and the stage names how far back to force them.

The short forms set the subject and leave the stage at `"simulate"`. The short forms are `all`, `none`, or a JSON dict with one key, either `"event_iloc"` for a non-sensitivity analysis or `"sa_id"` for a sensitivity analysis, mapping to a non-empty list of identifiers. The full form is a dict that carries `subject` and `stage`, where `subject` takes any short form, as in `{"subject": "all", "stage": "render"}`.

`stage` is a **floor**. The named stage and everything downstream re-run. The default `"simulate"` keeps the historical meaning of the field. The other values are `"process"`, `"consolidate"`, and `"render"`. The `"render"` floor accepts no subject scope, and value validation rejects a scoped floor, because `plots/` also holds cross-analysis aggregate figures with no subject token, and a scoped re-render would leave the aggregates stale. The floor works by deleting figures so that the plot rules fire again, so the floor removes everything under `plots/` except `plots/eda/`. The `plots/eda/` figures come from `hhemt eda`, and no Snakemake rule regenerates them.

Either form overrides the `force_rerun` field of the config for a single invocation and leaves the config unchanged. `hhemt run-experiment` and `hhemt reprocess` carry the same flag and take the same values written as JSON.

The analysis config holds the full form as follows.

```yaml
force_rerun:
  subject: all
  stage: process
```

## Add scenarios or events to a running sweep

Resume in the default mode. Under the graceful-rerun semantics of the toolkit, `hhemt run` picks up the added scenarios and sweeps only the additions. Completed scenarios stay as they are, and the command waits on jobs that a previous submission left queued, without resubmitting them.

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml
```

Avoid `--from-scratch` for adding work. The flag is the full-tree delete of `hhemt run`, and the guard refuses whenever the analysis directory holds work. Only a directory that the guard sees as empty is wiped and rebuilt from the beginning.

`hhemt run` offers no flag that overrides the guard. The flag for a wipe on purpose is `--override-wipe-nonempty` on `hhemt run-experiment --mode fresh`, and the command prints the cost before the delete. To remove an analysis, use `hhemt delete`. The command carries a confirmation and in-flight guards. [Run modes and the wipe guard](running-an-experiment-definition.md#run-modes-and-the-wipe-guard) describes the wipe flag and the delete, and gives the command-line form of the guard.

## See re-processed results without re-simulating

To regenerate the processed outputs, the consolidation, or the report, use `hhemt reprocess`. `--start-with` controls how far back the re-processing begins.

```bash
# On a non-sensitivity analysis these recipes re-render the regenerable figures
# under plots/ and SPARE plots/eda/, which nothing regenerates. A dry run spares
# every figure, including the regenerable ones, and deletes only the report shell.

# Re-render the report and the plots against the existing consolidated store.
# An existing store is left as it is (see --regenerate-existing below).
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --start-with consolidate

# Re-render the report only (fastest; leaves the plots in place, plots/eda/ included)
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --start-with render

# Rebuild the consolidated store, then re-render
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --start-with consolidate --regenerate-existing

# Rebuild per-scenario processed outputs from the raw outputs, then everything downstream
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --start-with process --regenerate-existing
```

`--regenerate-existing` decides whether an existing consolidated store or existing per-scenario processed outputs are rebuilt, and the flag defaults to off. At the default, `hhemt reprocess` deletes the rendered report and renders a new report against the outputs already on disk, and an existing consolidated store and existing per-scenario processed outputs stay in place. The `consolidate` and `process` stages therefore leave the store unchanged on an already-consolidated analysis, even when `--start-with` names them. An analysis with complete simulations that was never consolidated is consolidated.

Under `--start-with render`, `hhemt reprocess` leaves `plots/` untouched. Under `--start-with consolidate` or `process`, the command re-renders the plots as well. On a non-sensitivity analysis, the plot re-render deletes the regenerable figures under `plots/` and spares `plots/eda/`, and no rule regenerates `plots/eda/`. A dry run deletes no figure and deletes only the report shell, because the shell is the mtime trigger that the preview depends on. Pass `--regenerate-existing` to delete and rebuild the store and the processed outputs. Rebuilding the consolidated store also re-emits `ro-crate-metadata.json`, and [Publishing and fetching](publishing.md) requires that file.

`hhemt reprocess` never re-runs simulations. The command operates on the raw outputs on disk and runs Snakemake with `--nolock`. Queued or running simulation jobs leave the command unblocked. The command refuses in the cases below.

- A sentinel under `_status/_orchestrator` names an analysis or reprocess driver that may be alive. The [CLI reference](../reference/cli.md#running-an-analysis) describes the driver check that `hhemt run` makes. The refusal names the sentinel. Wait for the driver to finish or for the age cap of the driver to elapse, or resume from the host that wrote the sentinel so that the probe can answer. Delete the named sentinel file only after you confirm independently that the driver is gone.
- `--start-with process --regenerate-existing` meets an `_status/_submitted` sentinel. Deleting a processed output that a live worker may be rewriting is unsafe.
