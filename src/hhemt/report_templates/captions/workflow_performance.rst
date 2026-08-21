Workflow performance for **{{ snakemake.config.analysis_id }}** — how the workflow RAN,
as distinct from what it produced (which the Metadata page covers).

- **Run timeline** — projected from the per-rule ``_status/*.flag.json`` sidecars, which
  are appended one file at a time as each workflow rule finishes. It therefore covers
  setup through consolidation for the WHOLE experiment and does not shrink when this
  report is regenerated.
- **SLURM efficiency** — the union of every ``slurm_efficiency_report_*.csv`` written
  across all submissions for this analysis, joined to each rule's purpose. It is
  finalized at workflow teardown, AFTER this report is rendered, so it is expected to be
  absent on the run that produces this page; re-render after the run completes.

**Sources:**

{{ snakemake.params.source_paths_rst }}
