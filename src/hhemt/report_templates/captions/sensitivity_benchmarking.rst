Benchmarking-mode sensitivity plot for independent variable **{{ snakemake.wildcards.independent_var }}**.

The dependent variable (default ``performance.Total`` from the per-scenario performance summary, slowest-rank wallclock (max across MPI ranks of per-rank cumulative deltas; per stipulation ``wallclock reduction uses max over rank``), restart-safe) is aggregated per sub-analysis using the ``aggregation`` mode declared in ``report_config.yaml`` under ``sensitivity``. SWMM-only sub-analyses route to ``swmm_full_rpt_file`` Total elapsed time (parsed via ``swmm_output_parser.parse_total_elapsed``).

**Sources:**

{{ snakemake.params.source_paths_rst }}
