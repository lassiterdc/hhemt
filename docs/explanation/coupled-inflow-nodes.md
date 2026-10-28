# Coupled inflow nodes

TRITON-SWMM exchanges water between the 2D surface and the SWMM pipe network
only at the nodes it is told to couple. The toolkit always tells it explicitly:
scenario preparation writes a coupling-node list, `coupling_nodes.txt` beside the
scenario's `hydraulics.inp`, and the coupled TRITON cfg names that file with one
`swmm_coupling_nodes="{path}"` line. This page explains how the list is chosen,
why, and what each error raised along the way means.

The hydraulics model's `[INFLOWS]` section carries inflows only. Preparation leaves
it exactly as the template filled it: no line is added, removed or rewritten, so
every inflow the model declares, such as a sewer's baseflow, reaches SWMM whether
or not its node is coupled.

Every rule here assumes SWMM's dynamic wave routing, which is how the toolkit's
simulations are run. Under dynamic wave SWMM solves a head at every node except
an outfall, and a pipe can carry flow in either direction, so a coupled manhole
both drains surface water and surcharges back onto the surface.

## Declaring the coupling nodes

The system config's `swmm_coupling_mode` chooses the candidates the list is
selected from:

| `swmm_coupling_mode` | Candidates |
|---|---|
| `all_junctions` (the default) | Every `[JUNCTIONS]` node of `SWMM_hydraulics` that has a `[COORDINATES]` row. |
| `list` | The nodes named in `swmm_coupling_nodes_file`. Each must be a junction with a `[COORDINATES]` row. |
| `legacy` (an explicit opt-in) | Every node of `SWMM_hydraulics` that carries an `[INFLOWS]` line with constituent `FLOW`, read by SWMM's own rule. |

`swmm_coupling_nodes_file` is required with `list` and refused with any other mode,
because it would be ignored. The file is one node name per line, with surrounding
whitespace stripped, blank lines and lines starting `;` skipped, and no header.

**The legacy candidates**, an explicit opt-in with `swmm_coupling_mode: legacy`,
follow SWMM's reading of an inflow line: a line names a FLOW
inflow when it has at least three items and its constituent is not a name in
`[POLLUTANTS]` and begins, ignoring the case of ASCII letters, with `FLOW`. So a
node whose only inflow line is a pollutant is not a candidate, because SWMM gives it
no water inflow. Under `list` or `all_junctions` the toolkit reads `[INFLOWS]` for
nothing but the undefined-node check below, so no inflow line adds, removes or
reorders a candidate.

The coupled model alone reads the list. A system whose coupled model is off
(`toggle_tritonswmm_model: false`) selects no coupling nodes and writes no list,
and the coupling checks made before submission skip it. The undefined-inflow-node
check below still reads its standalone model's `SWMM_full` when that model is on.
Of the declaration, only the config-load rule on `swmm_coupling_nodes_file` above
still applies.

## What is coupled

**Every candidate junction is coupled by default**, one exchange point per
junction, so the number of exchange points does not depend on the DEM resolution:
a 10 m and a 1 m model of the same network couple the same nodes. Coupling more
than one node in a DEM cell needs a TRITON-SWMM build that supports it; an older
build stops at its own check that each DEM cell holds at most one coupled node.

The list is written in SWMM node order, the order the model defines its nodes in.

Which node types can be coupled:

| Node type | Coupled | Why |
|---|---|---|
| Junction, whatever its links | yes | An ordinary manhole, or an inlet box draining through an orifice, a weir or a pump. SWMM solves its head like any junction's, and a junction with only incoming pipes still drains by reverse flow. |
| Outfall | no | SWMM sets an outfall's water level as a boundary condition, so the manhole exchange TRITON-SWMM uses does not apply. Coupling an outfall whose receiving water is in the surface model would need a solver change this build does not have. |
| Storage node | no | A solver limitation, not a judgement about the node: TRITON-SWMM does not yet couple storage nodes correctly. For a pond whose basin is also in the DEM, couple the junction of its outlet structure instead. |
| Divider | no | A solver limitation: TRITON-SWMM does not yet couple dividers correctly, although under dynamic wave a divider acts as a junction. |

A candidate of a type that is not coupled keeps its place in the pipe network and
its inflow lines; it is only left off the coupling list, and one warning per
scenario names every such node and its type.

## One node per DEM cell (optional, legacy only)

Setting `one_coupled_inflow_node_per_dem_cell: true` in the analysis config
restores the earlier limit of one coupled junction per DEM cell. It applies only
to `swmm_coupling_mode: legacy`: under `list` or `all_junctions` the declaration is
the coupled set, so the two settings together are refused before any job is
submitted. In each cell that holds more than one candidate junction, the toolkit
keeps one:

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

A junction the ranking does not keep is left off the coupling list; its inflow
lines stay in the model.

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
`_status/inflow_node_selection.json`: the cell, every candidate with its type,
the node kept, and why each other node was left off the list. The report's
**Inflow node reduction** row in the Errors and Warnings section lists those
cells. A scenario prepared without the option, or for a system whose coupled
model is off, carries no record and is reported as not measured.

## Errors and what to do

| Error | When it is raised | What to do |
|---|---|---|
| Undefined inflow node | Before any job is submitted, when an inflow line of a template the run uses (the coupled model's `SWMM_hydraulics`, or the standalone model's `SWMM_full`) names a node that no `[JUNCTIONS]`, `[OUTFALLS]`, `[STORAGE]` or `[DIVIDERS]` line defines. SWMM would stop on that line with `ERROR 209`; its report labels the section `[INFLOW]`. Names are compared as SWMM compares them, ignoring the case of ASCII letters only. | Define the node, or correct or remove the inflow line. A name that is still a template placeholder is skipped with a warning. |
| Coupling-node file missing or unused | When the system config loads: `swmm_coupling_mode: list` with no `swmm_coupling_nodes_file`, or a `swmm_coupling_nodes_file` with any other mode. | Give the file with `list`, or remove it. |
| Coupling-node file refused | Before any job is submitted, when the file names a node no node section defines, names one node twice (ignoring the case of ASCII letters), names a node with no `[COORDINATES]` row, names a node that is not a junction, or names no node. Every failing case is reported. | Correct the file. An outfall, storage node or divider cannot be coupled; list a junction instead. For a surface-only run, turn off `toggle_tritonswmm_model` and run the TRITON-only model. |
| Template already sets `swmm_coupling_nodes` | Before any job is submitted, when the TRITON cfg template has an active `swmm_coupling_nodes` line. A line starting `#` is a comment and is not refused. | Remove the line; the toolkit writes it for every coupled scenario. Declare coupling nodes with `swmm_coupling_mode`. |
| One node per cell with a declared mode | Before any job is submitted, when `one_coupled_inflow_node_per_dem_cell` is on and `swmm_coupling_mode` is `list` or `all_junctions`. | Turn the option off, or set `swmm_coupling_mode: legacy`. |
| No coupling node declared | Before any job is submitted, when a coupled system's candidates are empty: under `all_junctions`, no junction has a `[COORDINATES]` row; under `legacy`, no inflow line with constituent `FLOW` names a node. | Declare coupling nodes, or for a surface-only run turn off `toggle_tritonswmm_model` and run the TRITON-only model (`toggle_triton_model`). |
| Coupling node outside the TRITON grid | Before any job is submitted, when a candidate junction lies outside the grid TRITON will be given, judged by TRITON's own placement rule against the processed DEM's header, or against the DEM setup will write when there is none, it fails its integrity check, or the run overwrites it. TRITON-SWMM would stop on each such node at startup. | List only the nodes inside the domain in a coupling-node file (`swmm_coupling_mode: list`, with `swmm_coupling_nodes_file`). |
| TRITON build that would couple other nodes | Before any job is submitted, and again at scenario preparation against the filled model, when the run's TRITON build does not read `swmm_coupling_nodes`. Such a build ignores the coupling list and couples the nodes named in the `[INFLOWS]` section, reading each once per line, so a node with two inflow lines is read twice and the build stops at its same-cell check. It is refused when those names, counted, differ from the coupling list; when its `[INFLOWS]` read is not one the toolkit recognizes; or when `one_coupled_inflow_node_per_dem_cell` is on. The message names the nodes on each side. | Move `TRITONSWMM_branch_key` to a TRITON commit that reads `swmm_coupling_nodes` and rebuild, or make the inflow lines and the coupling list agree. |
| No node of a cell can be coupled | At scenario preparation, when every candidate in a DEM cell is an outfall, a storage node or a divider. The message gives each node's reason. | Move the inflow to a junction, or remove the inflow line. |
| Cross-section that cannot be sized | At scenario preparation, only with the one-node-per-cell option, when ranking a cell needs the flow area of a shape `calc_area` does not handle. The message names the link, its shape and its node. | Add the shape's full-area formula to `calc_area` in `src/hhemt/scenario.py`. |
| Node without coordinates | At scenario preparation, when a coupling candidate has no `[COORDINATES]` line, so its DEM cell is unknown; and, when the hydrology model is used, when any node of the hydrology model has none. | Add the node's coordinates. |
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
uses is not known to read it. In native mode the TRITON clone's
`src/config_utils.h` must read the option, or the clone must include the commit that
added it; in container mode the image's recorded TRITON commit must match
`TRITONSWMM_branch_key` and be one of the commits known to read it.

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
