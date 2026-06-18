System map for **{{ snakemake.config['analysis_id'] }}**.

Overlays the watershed polygon (red), TRITON DEM extent (blue dashed), boundary-condition shapefile location (orange markers), and SWMM nodes (black points) and links (gray lines) extracted from a representative scenario via swmmio. The map is rendered in the system CRS configured by ``system_config.crs.horizontal_epsg``.

**Sources:**

{{ snakemake.params.source_paths_rst }}
