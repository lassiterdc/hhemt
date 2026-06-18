Peak flood depth raster for scenario **{{ snakemake.wildcards.event_id }}**.

The raster is computed by the TRITON solver as the per-cell maximum H value across all simulation timesteps. Color scale and extent are configured in ``report_config.yaml`` under ``per_sim.peak_flood_depth``.

See the **System Information** sidebar section for the underlying DEM and boundary geometry.

**Sources:**

{{ snakemake.params.source_paths_rst }}
