Benchmarking-mode sensitivity plot for independent variable **{{ snakemake.wildcards.independent_var }}**.

The dependent variable (default ``performance.Total`` from the per-scenario performance summary, slowest-rank wallclock (max across MPI ranks of per-rank cumulative deltas; per stipulation ``wallclock reduction uses max over rank``), restart-safe) is aggregated per member using the ``aggregation`` mode declared in ``report_config.yaml`` under ``sensitivity``. SWMM-only members route to ``swmm_full_rpt_file`` Total elapsed time (parsed via ``swmm_output_parser.parse_total_elapsed``).

**Sources:**

**Axis ranges are auto-scaled independently per report.** Each panel's y-axis is fitted to this master's own data, so an equal bar height or curve position in the coupled (TRITON-SWMM) and uncoupled (TRITON) reports does NOT denote an equal value — the uncoupled arm is faster, and its axis is scaled accordingly. Read the axis tick labels before comparing across reports; the paired small-multiple in the combined report is the surface where the two arms share a page and a direct comparison is intended.

{{ snakemake.params.source_paths_rst }}
