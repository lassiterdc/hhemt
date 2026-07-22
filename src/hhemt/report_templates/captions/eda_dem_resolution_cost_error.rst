DEM-resolution EDA: cost versus error.

The headline tradeoff for the DEM-resolution sweep. **x = compute-hours**
(``wallclock_hr x n_devices``); **y = the signed depth-magnitude headline** versus the
finest run. ONE point per resolution rung, each directly labelled with its cell size,
points connected in resolution order.

Compute-hours rather than wall-clock is the resource-fair currency: at a fixed compute
configuration across a DEM sweep the two are identical up to a scale factor, but
compute-hours stays correct if a future sweep varies devices per resolution, where
wall-clock would silently misrank.

This is a plain cost-versus-error tradeoff and is deliberately **not** called a Pareto
frontier. With cost and error both monotone in a single parameter (resolution), every
point is trivially non-dominated, so a frontier would have nothing to filter out -- the
term would do no work and would overstate the method.

Plotting error against cost rather than against resolution keeps the convergence shape
(resolution -> cost is monotone, so the two are reparameterizations of each other) and
adds the affordability read.

The finest run is a **reference**, not truth: its own error is unquantified, so every
error shown here is a lower bound on the true accuracy loss.

.. seealso:: The depth-error ECDF, which shows the full error distribution behind each
   point's single summary value.

**Sources:**

{{ snakemake.params.source_paths_rst }}
