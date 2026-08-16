Three-panel flood summary for scenario **{{ snakemake.wildcards.event_id }}**.

**Peak flood depth** is computed by the TRITON solver as the per-cell maximum H value across all simulation timesteps. **Water surface elevation** is that depth added to the underlying DEM, on a colorbar whose range is computed once across every event so the scale is comparable between scenarios. Color scales and extent are configured in ``report_config.yaml`` under ``per_sim.peak_flood_depth``.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

See the **System Information** sidebar section for the underlying DEM and boundary geometry.

**Sources:**

{{ snakemake.params.source_paths_rst }}
