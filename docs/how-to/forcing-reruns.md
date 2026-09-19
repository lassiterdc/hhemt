# Forcing and suppressing re-runs

Task recipes for taking control of the workflow when the default resume
behavior is not what you want. For the *why* behind the default behavior, see
[When and why re-runs happen](../explanation/rerun-faq.md).

## Force a full re-run of specific scenarios

Pass `override_force_rerun` to `analysis.run()`:

??? note "Why a forced re-run clears two things, not one"
    The invalidation is two-layer: it deletes the `_status/*.flag` markers *and*
    clears the per-scenario processing-log records. Both are needed, because a
    surviving completion record would otherwise cause the step to be skipped even
    with its flag gone.

```python
# Re-run everything, regardless of the recorded completion state:
analysis.run(override_force_rerun="all")

# Re-run only specific scenarios (by event index):
analysis.run(override_force_rerun={"event_iloc": [0, 2]})

# For a sensitivity analysis, target specific members:
analysis.run(override_force_rerun={"sa_id": ["member_3", "member_7"]})

# Force from a later stage (re-render the plots and the report, no re-simulation):
analysis.run(override_force_rerun={"subject": "all", "stage": "render"})

# Re-process and re-consolidate two members, leaving their simulations alone:
analysis.run(
    override_force_rerun={"subject": {"sa_id": ["member_3", "member_7"]}, "stage": "process"}
)
```

`override_force_rerun` carries two orthogonal axes: *which* scenarios to force
(the subject) and *how far back* to force them (the stage).

The short forms set the subject and leave the stage at `"simulate"`: `"all"`,
`"none"`, or a dict with exactly one key, `"event_iloc"` (non-sensitivity only)
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
`analysis.eda()`, which no Snakemake rule regenerates.

Either form overrides the config's `force_rerun` field for a single invocation
without mutating the config. The same knob is available on the command line as
`hhemt run-experiment --override-force-rerun` (and on `hhemt run` and
`hhemt reprocess`), taking the same values written as JSON.

## Add scenarios or events to a running sweep

Re-run with `from_scratch=False` (the default). Under the toolkit's
graceful-rerun semantics, `analysis.run()` picks up the newly added scenarios
and resume-sweeps only the additions. Completed scenarios are not re-run, and
any still-queued jobs from a previous submission are waited on rather than
re-submitted:

```python
analysis.run(from_scratch=False)
```

Do **not** reach for `from_scratch=True` to add work. `from_scratch=True` is the
run path's full-tree delete, and it is guarded: `analysis.run(from_scratch=True)`
raises `ConfigurationError` and deletes nothing whenever the analysis directory
holds completed simulations (`_status/c_run_*.flag`), a consolidated root store,
in-flight submission sentinels (`_status/_submitted`, `_status/_queued`), or an
orchestrator sentinel (`_status/_orchestrator`). The refusal names what it found.
Only a directory the guard sees as empty is wiped and rebuilt from the beginning.

To wipe deliberately anyway, pass `override_wipe_nonempty=True`; the cost is
printed before the delete. To remove an analysis rather than re-run it, use
`hhemt delete`, which carries its own confirmation and in-flight guards. The
command-line form of the same guard is described under
[Run modes and the wipe guard](running-an-experiment-bundle.md#run-modes-and-the-wipe-guard).

## See re-processed results without re-simulating

When you have completed simulations and only want to regenerate the processed
outputs, consolidation, or report (for example after fixing a renderer), use
`analysis.reprocess()`. The `start_with` argument controls how far back the
re-processing begins:

```python
# On a non-sensitivity analysis the recipes below re-render the regenerable figures
# under plots/ and SPARE plots/eda/, which nothing regenerates. A dry run spares
# every figure, including the regenerable ones, and deletes only the report shell.

# Re-render the report and the plots against the existing consolidated store.
# An existing store is left as it is (see regenerate_existing below):
analysis.reprocess(start_with="consolidate")

# Re-render the report only (fastest; leaves the plots in place, plots/eda/ included):
analysis.reprocess(start_with="render")

# Rebuild the consolidated store, then re-render:
analysis.reprocess(start_with="consolidate", regenerate_existing=True)

# Rebuild per-scenario processed outputs from the raw outputs, then everything
# downstream:
analysis.reprocess(start_with="process", regenerate_existing=True)
```

`regenerate_existing` decides whether an existing consolidated store or existing
per-scenario processed outputs are rebuilt. It defaults to `False`. At that
default `reprocess()` deletes the rendered report and re-renders it against
whatever is already on disk: an existing consolidated store and existing
per-scenario processed outputs are left in place. On an already-consolidated
analysis the `consolidate` and `process` stages therefore leave the store as it
is even when `start_with` names them. An analysis whose simulations are all
complete but that has never been consolidated is consolidated. `reprocess()`
under `start_with="render"` never touches `plots/`, but under
`start_with="consolidate"` or `"process"` it re-renders the plots too. On a
non-sensitivity analysis that plot re-render works by deleting the regenerable
figures under `plots/` and sparing `plots/eda/`, which nothing regenerates. On a
dry run no figure is deleted at all. Only the report shell is, because it is the
mtime trigger the preview depends on. Pass `regenerate_existing=True` to
delete and rebuild the store and the processed outputs. Rebuilding the
consolidated store is also what re-emits `ro-crate-metadata.json`, which
[Publishing and fetching](publishing.md) requires.

`reprocess()` never re-runs simulations: it operates on the raw outputs already
on disk, and it runs Snakemake with `--nolock`, so queued or running simulation
jobs do not block it. It does refuse in two cases. The first is while an
`_status/_orchestrator` sentinel names a run or reprocess driver that is alive
or that this host cannot prove dead. The refusal names the sentinel. The way
out is to wait for that driver to finish or for its age cap to elapse, or to
re-run from the sentinel's origin host so the probe can answer. Only once you
have independently confirmed the driver is gone should you delete the named
sentinel file instead. The second is
`start_with="process", regenerate_existing=True` while any `_status/_submitted`
sentinel is present, because deleting a processed output a live worker may be
re-writing is not safe.
