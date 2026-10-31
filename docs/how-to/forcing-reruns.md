# Forcing and suppressing re-runs

Task recipes for taking control of the workflow when the default resume
behavior is not what you want. For the *why* behind the default behavior, see
[When and why re-runs happen](../explanation/rerun-faq.md).

Every command below takes a system and an analysis config; the
[CLI reference](../reference/cli.md) lists every verb and option.

## Force a full re-run of specific scenarios

Pass `--override-force-rerun` to `hhemt run`:

??? note "Why a forced re-run clears two things, not one"
    The invalidation is two-layer: it deletes the `_status/*.flag` markers *and*
    clears the per-scenario processing-log records, because completion is tracked
    by both ([Surprising cases](../explanation/rerun-faq.md#surprising-cases)).

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

`--override-force-rerun` carries two orthogonal axes: *which* scenarios to force
(the subject) and *how far back* to force them (the stage).

The short forms set the subject and leave the stage at `"simulate"`: `all`,
`none`, or a JSON dict with exactly one key, `"event_iloc"` (non-sensitivity only)
or `"sa_id"` (sensitivity only), mapping to a non-empty list of identifiers. The
two-axis form is a dict carrying `subject` and `stage`, where `subject` takes any
of the short forms: `{"subject": "all", "stage": "render"}`.

`stage` is a **floor**: that stage and everything downstream re-run. Its values
are `"simulate"` (the default, and the historical meaning of this field),
`"process"`, `"consolidate"` and `"render"`. Two things to know about the
`"render"` floor. It cannot be scoped to a subject, and a scoped one is rejected
when the value is validated, because `plots/` also holds cross-analysis aggregate
figures that carry no subject token and a scoped re-render would leave those
stale. It also works by deleting the figures so the plot rules re-fire, which
means it removes everything under `plots/` except `plots/eda/`: those come from
`hhemt eda`, which no Snakemake rule regenerates.

Either form overrides the config's `force_rerun` field for a single invocation
without mutating the config. `hhemt run-experiment` and `hhemt reprocess` carry
the same flag, taking the same values written as JSON.

Written in the analysis config, the two-axis form is:

```yaml
force_rerun:
  subject: all
  stage: process
```

## Add scenarios or events to a running sweep

Re-run in the default resume mode. Under the toolkit's graceful-rerun semantics,
`hhemt run` picks up the newly added scenarios and resume-sweeps only the
additions. Completed scenarios are not re-run, and any still-queued jobs from a
previous submission are waited on rather than re-submitted:

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml
```

Do **not** reach for `--from-scratch` to add work. It is the run path's full-tree
delete, and it is guarded: `hhemt run --from-scratch` raises
`ConfigurationError` and deletes nothing whenever the analysis directory holds
completed simulations (`_status/c_run_*.flag`), a consolidated root store,
in-flight submission sentinels (`_status/_submitted`, `_status/_queued`), or an
orchestrator sentinel (`_status/_orchestrator`). The refusal names what it found.
Only a directory the guard sees as empty is wiped and rebuilt from the beginning.

`hhemt run` offers no flag that overrides that guard. The deliberate-wipe flag is
`--override-wipe-nonempty`, on `hhemt run-experiment --mode fresh`; the cost is
printed before the delete. To remove an analysis rather than re-run it, use
`hhemt delete`, which carries its own confirmation and in-flight guards. Both are
described under
[Run modes and the wipe guard](running-an-experiment-definition.md#run-modes-and-the-wipe-guard),
which also gives the command-line form of the same guard.

## See re-processed results without re-simulating

When you have completed simulations and only want to regenerate the processed
outputs, consolidation, or report (for example after fixing a renderer), use
`hhemt reprocess`. `--start-with` controls how far back the re-processing begins:

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

`--regenerate-existing` decides whether an existing consolidated store or existing
per-scenario processed outputs are rebuilt. It defaults to off. At that default
`hhemt reprocess` deletes the rendered report and re-renders it against whatever is
already on disk: an existing consolidated store and existing per-scenario processed
outputs are left in place. On an already-consolidated analysis the `consolidate`
and `process` stages therefore leave the store as it is even when `--start-with`
names them. An analysis whose simulations are all complete but that has never been
consolidated is consolidated. `hhemt reprocess` under `--start-with render` never
touches `plots/`, but under `--start-with consolidate` or `process` it re-renders
the plots too. On a non-sensitivity analysis that plot re-render works by deleting
the regenerable figures under `plots/` and sparing `plots/eda/`, which nothing
regenerates. On a dry run no figure is deleted at all. Only the report shell is,
because it is the mtime trigger the preview depends on. Pass
`--regenerate-existing` to delete and rebuild the store and the processed outputs.
Rebuilding the consolidated store is also what re-emits `ro-crate-metadata.json`,
which [Publishing and fetching](publishing.md) requires.

`hhemt reprocess` never re-runs simulations: it operates on the raw outputs
already on disk, and it runs Snakemake with `--nolock`, so queued or running
simulation jobs do not block it. It does refuse in two cases. The first is while an
`_status/_orchestrator` sentinel names a run or reprocess driver that is alive or
that this host cannot prove dead. The refusal names the sentinel. The way out is
to wait for that driver to finish or for its age cap to elapse, or to re-run from
the sentinel's origin host so the probe can answer. Only once you have
independently confirmed the driver is gone should you delete the named sentinel
file instead. The second is `--start-with process --regenerate-existing` while any
`_status/_submitted` sentinel is present, because deleting a processed output a
live worker may be re-writing is not safe.
