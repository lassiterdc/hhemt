Validation report for **{{ snakemake.config.analysis_id }}** — runs the same checks as the pytest aggregate ``assert_analysis_workflow_completed_successfully`` and surfaces the structured failures organized by (1) system-level checks, (2) aggregate per-scenario checks, (3) granular per-scenario stage failures, and (4) resource-utilization mismatches (actual vs intended MPI / OMP / GPUs / GPU backend per scenario).

**Sources:**

{{ snakemake.params.source_paths_rst }}
