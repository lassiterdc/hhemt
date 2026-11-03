Peak flood depth, water surface elevation and event hydrology for member **{{ snakemake.wildcards.member_id }}**, event **{{ snakemake.wildcards.event_id }}**.

**Peak flood depth** is TRITON's per-cell running maximum of depth (the solver's MH), updated at every solver step, so it can sit above the largest depth printed at any output timestep (a peak between prints) and, in coupled runs at solver commits before b3820a44, below it at a manhole cell. **Water surface elevation** is that depth added to the underlying DEM, on a colorbar whose range is computed once across every event so the scale is comparable between events; cells the DEM marks as building or wall are left out of that range. Colormaps, the depth colorbar floor and the quantile clips are configured in ``cfg_analysis.yaml`` under ``report.per_sim.map``.

Both map panels are clipped to the watershed polygon supplied as ``system_config.watershed_gis_polygon``, so every cell shown lies inside it and cells outside it are excluded from both the display and the colorbar range. The polygon boundary is drawn on each map and labeled **watershed polygon** in the legend.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

**Sources:**

{{ snakemake.params.source_paths_rst }}
