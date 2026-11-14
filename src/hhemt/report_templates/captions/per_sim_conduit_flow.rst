Conduit flow for member **{{ snakemake.wildcards.member_id }}**, event **{{ snakemake.wildcards.event_id }}**.

**Max / full flow** is the ratio of maximum flow to full-pipe flow capacity per conduit. **Peak flow** is the peak flow magnitude per conduit. Colormaps and bounds are configured in ``cfg_analysis.yaml`` under ``report.per_sim.conduit_flow``.

The watershed boundary is drawn on each map and labeled **watershed polygon** in the legend. The watershed boundary is an overlay only: conduit values are not masked by the boundary, so a conduit outside the boundary is still drawn with its true flow.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

**Sources:**

{{ snakemake.params.source_paths_rst }}
