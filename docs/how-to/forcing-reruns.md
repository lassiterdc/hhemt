# Forcing and suppressing re-runs

Task recipes for taking control of the workflow when the default resume
behavior is not what you want. For the *why* behind the default behavior, see
[When and why re-runs happen](../explanation/rerun-faq.md).

## Force a full re-run of specific scenarios

Pass `override_force_rerun` to `analysis.run()`. This performs a two-layer
invalidation — it deletes the `_status/*.flag` markers *and* clears the
per-scenario processing-log records — so the targeted steps genuinely
re-execute rather than being skipped by a surviving completion record:

```python
# Re-run everything from scratch of the completion state:
analysis.run(override_force_rerun="all")

# Re-run only specific scenarios (by event index):
analysis.run(override_force_rerun={"event_iloc": [0, 2]})

# For a sensitivity analysis, target specific sub-analyses:
analysis.run(override_force_rerun={"sa_id": ["sa_3", "sa_7"]})
```

`override_force_rerun` accepts `"all"`, `"none"`, or a dict keyed by
`"event_iloc"` / `"sa_id"`. It overrides the config's `force_rerun` field for a
single invocation without mutating the config.

## Add scenarios or events to a running sweep

Re-run with `from_scratch=False` (the default). Under the toolkit's
graceful-rerun semantics, `analysis.run()` picks up the newly added scenarios
and resume-sweeps only the additions — completed scenarios are not re-run, and
any still-queued jobs from a previous submission are waited on rather than
re-submitted:

```python
analysis.run(from_scratch=False)
```

Do **not** reach for `from_scratch=True` to add work — that wipes the analysis
directory and rebuilds everything from the beginning.

## See re-processed results without re-simulating

When you have completed simulations and only want to regenerate the processed
outputs, consolidation, or report — for example after fixing a renderer or to
inspect partial results — use `analysis.reprocess()`. The `start_with`
argument controls how far back the re-processing begins:

```python
# Re-consolidate + re-render the report from existing per-scenario outputs:
analysis.reprocess(start_with="consolidate")

# Re-render the report only (fastest; re-uses the consolidated datatree):
analysis.reprocess(start_with="render")

# Re-run per-scenario processing too (raw outputs -> zarr/nc):
analysis.reprocess(start_with="process")
```

`reprocess()` never re-runs simulations — it operates on the raw outputs
already on disk, so it is safe to run while queued or running simulation
workers exist.
