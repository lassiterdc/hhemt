Simulation results for scenario **{{ snakemake.wildcards.event_id }}** — one scrollable page carrying every model's panels for this event.

Scroll the page rather than clicking between figures: each enabled model gets its own header, in the order **TRITON**, **TRITON-SWMM**, **SWMM**, and each header is followed by that model's applicable panels. That order puts the two depth maps adjacent and the two conduit figures adjacent, so like panels can be compared without scrolling past an unlike one.

**Which sections appear is derived from the models this analysis ran, not fixed.** The two panel families cover different model sets because the models produce different things: peak flood depth needs a 2D depth field, which only the TRITON-bearing arms produce, while conduit flow needs a pipe network, which only the SWMM-bearing arms produce. So depth covers **TRITON-SWMM** and **TRITON**, conduit flow covers **TRITON-SWMM** and **SWMM**, and a three-model analysis yields four sections. A model that ran but has no applicable panel family is absent from this page by construction rather than dropped silently.

**Peak flood depth** is the per-cell maximum H value across all simulation timesteps, computed by the TRITON solver. **Water surface elevation** is that depth added to the underlying DEM. Both map panels are **clipped to the watershed** — the drainage area north of the sea wall — so the coastal storm-tide extreme outside it is excluded from both the display and the colorbar range.

**Max / full flow** is the ratio of maximum flow to full-pipe flow capacity per conduit; **peak flow** is the peak flow magnitude per conduit. Here the watershed boundary is an **overlay only** — unlike the depth maps, conduit values are not masked to it, so a conduit lying outside the boundary is still drawn and still carries its true flow.

**Colour scales are shared across the models on this page, not computed per panel.** The depth, water-surface-elevation, and peak-flow ranges are each computed once over the union of every model arm rendered here and every event in the analysis. Two panels stacked in one scroll are therefore directly comparable, and a difference in colour is a difference in the water — not a difference in the scale beneath it.

**Flood Drivers** carries the event hydrology: the rainfall time series in the upper sub-panel and the boundary-condition water level in the lower one. It is the same forcing for every model on this page, which is what makes the arms comparable.

Colormaps and bounds are configured in ``report_config.yaml`` under ``per_sim.peak_flood_depth`` and ``per_sim.conduit_flow``. See the **System Information** sidebar section for the underlying DEM and boundary geometry.

**Sources:**

{{ snakemake.params.source_paths_rst }}
