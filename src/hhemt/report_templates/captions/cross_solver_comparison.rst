Comparison of solver versions across the combined experiments: did the solver change move the measurement? There is one grouped panel per model (TRITON or TRITON-SWMM). Within a panel the x axis is the compute configuration, each bar series is one solver, and the y-axis title names the metric and its unit. The legend names a solver ``{sha8}_{YYYYMMDD}``, the first 8 characters of its pinned ref and the committer date of the compiled commit, or the ref prefix alone when the date was not captured. Series run oldest first by commit date, and a solver without a date follows every dated one, ordered by its ref. An ``x`` marker above the bars means the solver has no model run at that compute configuration. The compute configuration stays on the x axis.

**Sources:**

{{ snakemake.params.source_paths_rst }}
