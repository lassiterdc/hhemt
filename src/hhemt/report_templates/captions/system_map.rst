Model inputs for **{{ snakemake.config['analysis_id'] }}**.

The **TRITON DEM** panel shows the processed elevation raster with its color range confined to the modeled area; wall and outside-watershed fill cells take the configured over-color. The **Manning's n** panel shows the processed roughness raster on the DEM's grid. Maps are drawn in the CRS set by ``system_config.crs.horizontal_epsg``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
