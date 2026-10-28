Three-panel flood summary for scenario **{{ snakemake.wildcards.event_id }}**.

**Peak flood depth** is the per-cell maximum of TRITON's water depth across all output timesteps, taken by the toolkit's post-processing. **Water surface elevation** is that depth added to the underlying DEM, on a colorbar whose range is computed once across every event so the scale is comparable between scenarios; cells the DEM marks as building or wall are left out of that range. Colormaps, the depth colorbar floor and the quantile clips are configured in ``cfg_analysis.yaml`` under ``report.per_sim.map``.

Both map panels are **clipped to the watershed** polygon supplied as ``system_config.watershed_gis_polygon``, so every depth and water-surface-elevation cell shown lies inside it and cells outside it are excluded from both the display and the colorbar range. The watershed boundary is drawn on each map and is labelled **watershed extent (bbox)** in the legend; the label names the supplied geometry, which may be a bounding box rather than a hydrologic divide.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

See the **System Information** sidebar section for the underlying DEM and boundary geometry.

**Sources:**

{{ snakemake.params.source_paths_rst }}
