DEM-resolution EDA: resolution x coupling junctions.

One row per DEM resolution rung. Columns: **DEM resolution (m)** | **Cells** | **Coupling junctions** | peak depth and peak flow | the maximum over-estimate and under-estimate of depth and flow versus the finest run.

Coupling junctions counts the SWMM junctions coupled to the TRITON surface, identified by their ``[INFLOWS]`` entries: where TRITON writes surface water into SWMM, and where the junction surcharges back to the surface when its head exceeds the cell.

Peak is the absolute maximum per resolution. The signed over/under-estimate columns are coarse minus fine (depth regridded onto the finest grid; flow on the shared conduit index), so the finest rung is the reference and its delta is zero by construction.

The finest run is the reference, not truth: its own error is unquantified, so every delta shown here is a lower bound on the true accuracy loss.

.. seealso:: The signed difference maps show where in space the depth
   disagreements summarized here occur.

**Sources:**

{{ snakemake.params.source_paths_rst }}
