Raw byte-for-byte identity over time, across compute configurations.

Compares, timestep by timestep, the raw TRITON output rasters (``out_triton/bin`` water depth and peak flood depth grids) of this sweep's members. Members are selected only by whether they kept their raw output rasters. Each member is compared only within its hardware family, because a GPU raster is never bit-identical to a CPU raster. The families are **CPU**, referenced to the serial-CPU configuration, and **GPU**, referenced to a GPU device chosen deterministically (partitions sort alphabetically, then member id); the starred row is that reference. Identical cells are green; differing cells shade from white to vermillion by the largest absolute difference. The verdict states whether every compared cell is identical and, when it is not, the number of differing cells.

**Sources:**

{{ snakemake.params.source_paths_rst }}
