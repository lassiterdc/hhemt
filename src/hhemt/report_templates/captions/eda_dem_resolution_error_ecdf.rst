DEM-resolution EDA: depth-error ECDF.

One curve per coarser resolution rung. The x axis is the absolute depth error |coarse - fine| (m); the y axis is the cumulative fraction of union-wet cells, the cells where either run is at or above tau (0.03 m, the nuisance-depth floor). A curve that rises fast and far to the left disagrees little with the finest run.

The p95 is drawn as a legend line at y = 0.95; each curve's crossing there is that resolution's 95th-percentile depth error.

This figure shows error magnitude only; the over- versus under-estimation direction is in the signed difference maps.

A table below the curves reports each coarser run's flooded-area change versus the finest run (cells at or above tau).

The finest run is the reference, not truth: its own error is unquantified, so every error shown here is a lower bound on the true accuracy loss.

**Sources:**

{{ snakemake.params.source_paths_rst }}
