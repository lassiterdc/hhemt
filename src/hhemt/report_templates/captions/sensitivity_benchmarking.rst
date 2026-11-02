Time against ``{{ snakemake.wildcards.independent_var }}``, as set by ``report.sensitivity.dependent_var``.

The top panel plots that time, and its y-axis title names the measure. For a TRITON or TRITON-SWMM member the value is the wall-clock time of the slowest MPI rank, summed over restarts, for the phase the variable names; the default ``performance.Total`` is the whole run.

Each report scales its y-axes to its own data, so compare values by the axis tick labels and not by position.

**Sources:**

{{ snakemake.params.source_paths_rst }}
