Results for event **{{ snakemake.wildcards.event_id }}**: every enabled model's panels on one page.

**Peak flood depth** is the per-cell maximum water depth across all simulation timesteps, computed by the TRITON solver. **Water surface elevation** is that depth added to the underlying DEM. Both map panels are clipped to the watershed, so cells outside it are excluded from both the display and the color range.

**Max / full flow** is the ratio of maximum flow to full-pipe flow capacity per conduit. **Peak flow** is the peak flow magnitude per conduit. The watershed boundary is an overlay only: conduit values are not masked to it, so a conduit outside the boundary is still drawn with its true flow.

Color scales are shared across the models on this page: the depth, water surface elevation and peak flow ranges are each computed once over every model and every event in the analysis, so panels are directly comparable.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

Colormaps and bounds are configured in ``cfg_analysis.yaml`` under ``report.per_sim.peak_flood_depth`` and ``report.per_sim.conduit_flow``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
