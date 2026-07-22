DEM-resolution EDA: depth-error ECDF.

One curve per coarser resolution rung. **x = absolute depth error** |coarse - fine| (m);
**y = cumulative fraction of union-wet cells** (cells where EITHER run is at or above
tau = 0.03 m, the model's nuisance-depth floor). A curve that rises fast and far to the
LEFT disagrees little with the finest run.

The p95 is drawn as a legend line at y = 0.95; each curve's crossing there is that
resolution's 95th-percentile depth error.

This figure shows error **magnitude** only. The over- versus under-estimation
**direction** is the signed difference maps, not this curve.

A companion table below the curves reports each coarser run's flooded-**area** change
versus the finest run (cells at or above tau) -- the extent story, distinct from the
depth-error magnitude the curves show.

This figure is not optional garnish: it is what discharges the prohibition on
presenting a percentile bound without its distribution and its denominator. The p95 is a
point ON this curve, and a p95 computed over a handful of wet cells is noise -- the ECDF
is what lets the reader see which case they are looking at.

The finest run is a **reference**, not truth: its own error is unquantified, so every
error shown here is a lower bound on the true accuracy loss.

**Sources:**

{{ snakemake.params.source_paths_rst }}
