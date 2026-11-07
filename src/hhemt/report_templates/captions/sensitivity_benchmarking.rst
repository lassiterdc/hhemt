Time against ``{{ snakemake.wildcards.independent_var }}``, as set by ``report.sensitivity.dependent_var``.

The top panel plots that time, and its y-axis title names the measure. For a TRITON or TRITON-SWMM member the value is the wall-clock time of the slowest MPI rank, summed over restarts, for the phase the variable names; the default ``performance.Total`` is the whole run. With the ``plotly`` static backend, the default, the ``aggregation`` statistic under ``report.sensitivity`` in ``cfg_analysis.yaml`` (``mean`` by default; ``median``, ``min`` or ``max``) collapses the replicates of each configuration before the best configuration at each device count is chosen; the ``matplotlib`` backend takes the minimum over the individual runs.

Each report scales its y-axes to its own data, so compare values by the axis tick labels and not by position.

**Sources:**

{{ snakemake.params.source_paths_rst }}
