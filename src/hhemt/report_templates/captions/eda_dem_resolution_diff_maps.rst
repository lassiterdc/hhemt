DEM-resolution EDA: signed depth-difference maps.

Signed difference in peak flood depth, each coarser run minus the finest reference run, with the coarser result restated on the finest grid. Panel A is the finest reference in absolute depth; each remaining panel pair is one coarser rung's signed-difference row and percent-difference row. Each panel is labeled with its own cell size.

Red means the coarser run under-estimates peak flood depth relative to the reference; blue means it over-estimates.

Percent differences are restricted to reference cells at or above tau (0.03 m, the nuisance-depth floor). Cells where the reference is dry and the coarse run is wet, or the reverse, have no defined percentage and are shown in the categorical fill with a count of newly wet cells.

The finest run is the reference, not truth: its own error is unquantified, so every disagreement shown here is a lower bound on the true accuracy loss.

Peak flood depth is measured above the local bed. Restating a coarse cell's depth on the fine cells it contains implies a non-flat water surface and can flood fine cells whose bed sits above the coarse block's mean plus its depth. This is a property of the coarse run's own coarseness, the phenomenon under study, not a rendering error.

.. seealso:: The resolution x coupling-junction table. Coarsening also shrinks the
   coupled SWMM inflow network, so the resolution axis co-varies with coupling density.

**Sources:**

{{ snakemake.params.source_paths_rst }}
