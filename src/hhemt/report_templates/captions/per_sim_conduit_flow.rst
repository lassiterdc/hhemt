SWMM conduit flow panel for scenario **{{ snakemake.wildcards.event_id }}**.

**Max / full flow**: ratio of maximum flow to full-pipe flow capacity per conduit. **Peak flow**: peak flow magnitude per conduit. Colormaps and bounds are configured in ``report_config.yaml`` under ``per_sim.conduit_flow``.

The watershed boundary is drawn on each map and is labelled **watershed extent (bbox)** in the legend. Here it is an **overlay only** — unlike the peak-flood-depth maps, the conduit values are not masked to it, so a conduit lying outside the boundary is still drawn and still carries its true flow.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one.

**Sources:**

{{ snakemake.params.source_paths_rst }}
