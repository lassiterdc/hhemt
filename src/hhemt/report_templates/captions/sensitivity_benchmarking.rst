Benchmarking-mode sensitivity plot for independent variable **{{ snakemake.wildcards.independent_var }}**.

The dependent variable (default ``performance.Total`` from the per-scenario performance summary, slowest-rank wallclock (max across MPI ranks of per-rank cumulative deltas; per stipulation ``wallclock reduction uses max over rank``), restart-safe) is aggregated per member using the ``aggregation`` mode declared in ``cfg_analysis.yaml`` under ``report.sensitivity``. SWMM-only members route to ``swmm_full_rpt_file`` Total elapsed time (parsed via ``swmm_output_parser.parse_total_elapsed``).

**Axis ranges are auto-scaled independently per report.** Each panel's y-axis is fitted to this master's own data, so an equal bar height or curve position in the coupled (TRITON-SWMM) and uncoupled (TRITON) reports does NOT denote an equal value — the uncoupled arm is faster, and its axis is scaled accordingly. Read the axis tick labels before comparing across reports; the paired small-multiple in the combined report is the surface where the two arms share a page and a direct comparison is intended.

**A master that swept several DEM resolutions is drawn as one BLOCK of panels per resolution.** The four metric panels (wall-clock, compute cost, speedup, efficiency) repeat once per resolution, ordered finest to coarsest, and each block's subplot titles name its resolution. Every block carries the same hardware columns and the same four metrics, so a panel is identified by BOTH its metric and its block: wall-clock at 0.35 m and wall-clock at 3.5 m are different panels and are not directly comparable — each block's y-axis is fitted to its own data, and each block's speedup and efficiency are measured against that block's own single-device run. The hardware columns come from the whole master, not from one block, so a column that has no runs at a given resolution is still drawn there as an EMPTY panel — it is not dropped and its absence is not a zero — while a series (a group) that never ran at that resolution is absent from that block's legend.

**Sources:**

{{ snakemake.params.source_paths_rst }}
