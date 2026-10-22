# Coupled inflow nodes

TRITON-SWMM exchanges water between the 2D surface and the SWMM pipe network
only at the nodes listed in the hydraulics model's `[INFLOWS]` section. Scenario
preparation decides that list: it keeps the inflow lines of the nodes that can be
coupled and deletes the rest. This page explains which nodes are kept, why, and
what each error raised along the way means.

Every rule here assumes SWMM's dynamic wave routing, which is how the toolkit's
simulations are run. Under dynamic wave SWMM solves a head at every node except
an outfall, and a pipe can carry flow in either direction, so a coupled manhole
both drains surface water and surcharges back onto the surface.

## What is coupled

**Every `[INFLOWS]` junction is coupled by default**, one exchange point per
junction, so the number of exchange points does not depend on the DEM resolution:
a 10 m and a 1 m model of the same network couple the same nodes. Coupling more
than one node in a DEM cell needs a TRITON-SWMM build that supports it; an older
build stops at its own check that each DEM cell holds at most one coupled node.

Which node types can be coupled:

| Node type | Coupled | Why |
|---|---|---|
| Junction, whatever its links | yes | An ordinary manhole, or an inlet box draining through an orifice, a weir or a pump. SWMM solves its head like any junction's, and a junction with only incoming pipes still drains by reverse flow. |
| Outfall | no | SWMM sets an outfall's water level as a boundary condition, so the manhole exchange TRITON-SWMM uses does not apply. Coupling an outfall whose receiving water is in the surface model would need a solver change this build does not have. |
| Storage node | no | A solver limitation, not a judgement about the node: TRITON-SWMM does not yet couple storage nodes correctly. For a pond whose basin is also in the DEM, couple the junction of its outlet structure instead. |
| Divider | no | A solver limitation: TRITON-SWMM does not yet couple dividers correctly, although under dynamic wave a divider acts as a junction. |

A node of a type that is not coupled keeps its place in the pipe network; only
its `[INFLOWS]` line is removed, and one warning per scenario names every such
node and its type.

## One node per DEM cell (optional)

Setting `one_coupled_inflow_node_per_dem_cell: true` in the analysis config
restores the earlier limit of one coupled junction per DEM cell. In each cell
that holds more than one inflow junction, the toolkit keeps one:

1. **Largest pipe first.** Each junction is ranked by the largest flow area among
   its own outgoing conduits, orifices and weirs. A junction with no such outgoing
   link is ranked by its largest incoming one, the pipe it drains through by
   reverse flow, and a junction with neither ranks last. Pumps and outlets carry
   no cross-section and are never used for ranking.
2. **Ties go to where the water leaves the cell.** When two or more junctions
   share the largest key, the cell's junction whose outgoing link leaves the cell
   is kept, then the one with the lowest invert, then the first. That choice is
   made among all of the cell's junctions, so a tie can hand the cell to its exit
   junction even when its own pipe is smaller.

Flow areas are SWMM's own full areas for each cross-section shape, computed by
`calc_area` in `src/hhemt/scenario.py`. **Additional pipe shapes can be added
there.** Size-coded horizontal ellipses and arches, whose first dimension is a
standard-size code rather than a length, are treated as shapes that cannot be
sized.

The pipe-size key does not always pick the manhole that would overflow first. On
the toolkit's reference model (`test_data/TRITON_SWMM_test_model`, 32 cells with
more than one node; no other model was measured), it kept a manhole other than
the lowest-rimmed one in 8 of the 19 cells it decided without a tie. Within one
cell every manhole sits under one ground elevation, so a rim difference there is a
difference in recorded manhole depths. A manhole that is not coupled and fills to
its rim keeps its excess inside SWMM rather than spilling it onto the surface.

With the option on, every prepared scenario records each reduced cell in
`_status/inflow_node_selection.json`: the cell, every inflow node with its type,
the node kept, and why each other node was dropped. The report's
**Inflow node reduction** row in the Errors and Warnings section lists those
cells. A scenario prepared without the option carries no record and is reported
as not measured.

## Errors and what to do

| Error | When it is raised | What to do |
|---|---|---|
| Undefined inflow node | Before any job is submitted, when an inflow line of a template the run uses (the coupled model's `SWMM_hydraulics`, or the standalone model's `SWMM_full`) names a node that no `[JUNCTIONS]`, `[OUTFALLS]`, `[STORAGE]` or `[DIVIDERS]` line defines. SWMM would stop on that line with `ERROR 209`; its report labels the section `[INFLOW]`. Names are compared as SWMM compares them, ignoring the case of ASCII letters only. | Define the node, or correct or remove the inflow line. A name that is still a template placeholder is skipped with a warning. |
| No node of a cell can be coupled | At scenario preparation, when every inflow node of a DEM cell is an outfall, a storage node or a divider. The message gives each node's reason. | Move the inflow to a junction, or remove the inflow line. |
| Cross-section that cannot be sized | At scenario preparation, only with the one-node-per-cell option, when ranking a cell needs the flow area of a shape `calc_area` does not handle. The message names the link, its shape and its node. | Add the shape's full-area formula to `calc_area` in `src/hhemt/scenario.py`. |
| Node without coordinates | At scenario preparation, when a node has no `[COORDINATES]` line, so its DEM cell is unknown. | Add the node's coordinates. |
| Hydraulics or standalone model not in `CMS` | At scenario preparation, when the hydraulics model, or the standalone model when it is enabled, declares any flow unit but `CMS`, or declares no `FLOW_UNITS` at all, which SWMM reads as `CFS`. The same refusal is raised when a coupled report or a standalone output in another unit is read, for example by a reprocess of an existing analysis. | Set `FLOW_UNITS CMS` in the template. |
| US flow units in the hydrology model | At scenario preparation, when the hydrology model declares `CFS`, `GPM` or `MGD`, or declares no `FLOW_UNITS` at all. | Convert the model to SI flow units: `CMS`, `LPS` or `MLD`. |

## Rims and flow units

TRITON-SWMM takes a coupled node's rim, its surface connection height, from the
depth SWMM itself uses for that node, in the solver builds that read it, so the two
models agree on every coupled rim. Earlier builds read the depth recorded in
`[JUNCTIONS]`, which SWMM raises to the crown of the node's highest connecting
pipe, so a node recorded below that crown had a lower rim in TRITON-SWMM than in
SWMM.

Setting `swmm_node_rim_from_dem: true` in the analysis configuration takes each
coupled node's rim from the DEM instead: the elevation of the node's grid cell minus
the node's invert. It is off by default. Only solver builds that read the option
honour it, so scenario preparation stops with a named error when the build the run
uses is not known to read it. In native mode the TRITON clone must include the
commit that added the option; in container mode the image's recorded TRITON commit
must match `TRITONSWMM_branch_key` and be one of the commits known to read it.

The surface hydrographs are converted to cubic metres per second from the
hydrology model's own flow unit before TRITON reads them, so an `LPS` or `MLD`
model's runoff reaches the surface in the right units. The per-node inflow record
kept beside the hydrographs is converted the same way and names the unit it was
converted from.

The hydraulics model and the standalone model must use `CMS`. Their flows are read
from SWMM's report and output files as cubic metres per second with no conversion,
and SWMM writes those files in the model's own unit, so a model in `LPS` would be
read 1000 times too large and one in `MLD` 86.4 times. A `FLOW_UNITS` value is read
the way SWMM reads it: the last line that carries a value wins, and the unit word is
matched by its first letters.
