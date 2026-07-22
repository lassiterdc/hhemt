DEM-resolution EDA: resolution x coupling junctions.

One row per DEM resolution rung. Columns: **DEM resolution (m)** | **Cells** |
**Coupling junctions** | **Peak depth / flow** | **max over/under-estimate depth / flow
versus the finest run**.

``Coupling junctions`` is a universal, model-agnostic count of the SWMM junctions
coupled to the TRITON surface, identified by their ``[INFLOWS]`` entries -- where TRITON
writes surface water into SWMM, and where the junction surcharges back to the surface
when its head exceeds the cell. It makes no per-model node-name assumptions, so it
applies to any coupled model.

Each DEM grid cell holds at most one coupling junction, so **coarsening the grid can
reduce the count**: this is the mechanism by which the resolution axis co-varies with
coupling density. The count is flat on fixtures where every junction retains its own
cell even at the coarsest rung.

Peak is the absolute maximum per resolution. The signed over/under-estimate columns are
``coarse - fine`` (depth regridded onto the finest grid; flow on the shared conduit
index), so the finest rung is the reference and its delta is zero by construction.

The finest run is a **reference**, not truth: its own error is unquantified, so every
delta shown here is a lower bound on the true accuracy loss.

.. seealso:: The signed difference maps, which show where in space the depth
   disagreements summarized here occur.

**Sources:**

{{ snakemake.params.source_paths_rst }}
