SWMM conduit flow panel for scenario **{{ snakemake.wildcards.event_id }}**.

Left panel: ratio of maximum flow to full-pipe flow capacity per conduit. Right panel: peak flow magnitude per conduit. Colormaps and bounds are configured in ``report_config.yaml`` under ``per_sim.conduit_flow``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
