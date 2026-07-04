# When and why re-runs happen

hhemt drives its workflow with [Snakemake](https://snakemake.readthedocs.io/),
so a step re-runs when one of three things changes: an input file's
modification time (mtime), the *set* of declared inputs for a rule, or the
rule's code. Most of the time this is exactly what you want — edit a config,
re-run, and only the affected steps rebuild. This page explains the cases that
surprise people. To *deliberately* force or suppress a re-run, see
[Forcing and suppressing re-runs](../how-to/forcing-reruns.md).

## The three trigger classes

A rule re-runs when Snakemake detects any of:

1. **Input mtime** — a file the rule declares as input is newer than the
   output it produced. Editing a DEM, a weather file, or a SWMM template
   re-runs the steps downstream of it.
2. **Input-set change** — the *list* of declared inputs changed, even if no
   file content did. When a toolkit release adds a new declared input to a
   rule (for example, a per-scenario fingerprint file), Snakemake sees the
   changed input set and re-runs that rule once.
3. **Rule code change** — the `shell:`/`run:` body of a rule changed between
   toolkit versions.

The toolkit is designed so that steady-state re-invocations are cheap: once a
phase is complete, its status flag and per-model log record the success, and a
re-run of `analysis.run()` resumes rather than rebuilds.

## Surprising cases

??? question "I edited the sensitivity XLSX — what re-runs?"
    The `sensitivity_analysis_definition.csv` is re-derived from the XLSX on
    **every** `analysis.run()`, and only the touched `sa_id` chains re-run.
    Editing the CSV directly is a no-op — it is silently overwritten before
    Snakemake plans, so the run reports "resuming, N/N complete" even though
    you expected a new row. Always edit the XLSX, not the derived CSV.

??? question "I removed a row (sa_id) from the XLSX — why did it abort?"
    Removing an `sa_id` leaves an orphan `subanalyses/sa_*/` directory on disk.
    The toolkit detects the orphan and `analysis.run()` **aborts** with a
    `ConfigurationError` rather than silently deleting data. Re-invoke with
    `cleanup_orphans=True` (or run `hhemt cleanup-orphans --apply --force`) to
    delete the orphan artifacts. Deletion is opt-in because it is
    irrecoverable.

??? question "Everything re-ran once after I upgraded the toolkit — is that a bug?"
    No — this is the expected one-shot cascade. The first `analysis.run()`
    after a release that changes a rule's declared input set, a rule's code,
    or a plot-ID grammar invalidates Snakemake's metadata and re-fires the
    affected rules exactly once. Steady state resumes on the second
    invocation, producing no observable artifact change. A single full rebuild
    right after an upgrade is normal, not a defect.

??? question "I re-ran and nothing happened, but I expected a rebuild."
    Completion is tracked by `_status/*.flag` markers plus per-model log
    success records, not by file presence alone. If a flag survives, the step
    is considered done and is skipped. To force a specific step to rebuild
    regardless of its flag, use the override knobs described in
    [Forcing and suppressing re-runs](../how-to/forcing-reruns.md).
