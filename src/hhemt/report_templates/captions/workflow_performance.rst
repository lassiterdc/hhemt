Workflow Performance for {{ snakemake.config.analysis_id }}: how the workflow ran. The
Metadata page covers what it produced.

- **Run timeline**: the workflow rules completed so far, from setup through consolidation
  and across every workflow run, read from the ``_status/*.flag.json`` records. Records
  that name a member no longer in the analysis are left out, and the section states how
  many. A record that names no member is shown with an empty Member cell. When no record
  exists, the section says so.
- **SLURM Efficiency**: one row per job, joined to the purpose of its rule. Jobs cancelled
  before they started are counted separately from jobs that ran. When no job is listed,
  the section says why.

**Sources:**

{{ snakemake.params.source_paths_rst }}
