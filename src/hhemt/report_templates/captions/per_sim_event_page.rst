Results for event **{{ snakemake.wildcards.event_id }}**: every enabled model's panels on one page.

**Peak flood depth** is TRITON's per-cell running maximum of depth (the solver's MH), updated at every solver step, so the value can sit above the largest depth printed at any output timestep (a peak between prints) and, in coupled runs at solver commits before b3820a44, below the printed depth at a manhole cell. **Water surface elevation** is that depth added to the underlying DEM. Both map panels are clipped to the watershed, so cells outside the watershed are excluded from both the display and the color range.

**Max / full flow** is the ratio of maximum flow to full-pipe flow capacity per conduit. **Peak flow** is the peak flow magnitude per conduit. The watershed boundary is an overlay only: conduit values are not masked by the boundary, so a conduit outside the boundary is still drawn with its true flow.

Color scales are shared across the models on this page: the depth, water surface elevation and peak flow ranges are each computed once over every model and every event in the analysis, so panels are directly comparable.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

Colormaps and bounds are configured in ``cfg_analysis.yaml`` under ``report.per_sim.peak_flood_depth`` and ``report.per_sim.conduit_flow``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
