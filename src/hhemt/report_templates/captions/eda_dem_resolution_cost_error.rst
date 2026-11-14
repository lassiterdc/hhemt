DEM-resolution EDA: cost versus error.

The x axis is compute cost, wall clock times the number of devices, on a log scale; the y axis is the 95th-percentile depth error against the finest run. Each resolution rung is one point, labeled with its cell size, and the points are connected in resolution order.

The finest run is the reference, not truth: its own error is unquantified, so every error shown here is a lower bound on the true accuracy loss.

.. seealso:: The depth-error ECDF shows the full error distribution behind each
   point's single summary value.

**Sources:**

{{ snakemake.params.source_paths_rst }}
