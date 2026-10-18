System overview for **{{ snakemake.config['analysis_id'] }}**: the modeled system's inputs before any results.

The **TRITON DEM** panel shows the processed elevation raster with its color range confined to the modeled area; wall and outside-watershed fill cells take the configured over-color, and the storm-tide boundary-condition line is drawn when the analysis has one. The **Manning's n** panel shows the processed roughness raster on the DEM's grid. When the analysis includes a SWMM model, a **Hydrology** panel (subcatchment polygons, drainage lines to their outlet nodes, outlet markers, from the hydrology ``.inp``) and a **Hydraulics** panel (junctions, outfalls, conduits with slope labels, node rim/invert labels, from the hydraulics ``.inp``) are added below. Maps are drawn in the CRS set by ``system_config.crs.horizontal_epsg``, unless the deprecated ``report.system_map.target_epsg`` override is set.

**Sources:**

{{ snakemake.params.source_paths_rst }}
