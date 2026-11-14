# Coupled inflow nodes

TRITON-SWMM exchanges water between the 2D surface and the SWMM pipe network only at the nodes named for coupling. The toolkit always names the nodes explicitly. Scenario preparation writes a coupling-node list, `coupling_nodes.txt`, beside the `hydraulics.inp` of the scenario. The coupled TRITON cfg names the file with a single `swmm_coupling_nodes="{path}"` line.

The `[INFLOWS]` section of the hydraulics model carries inflows only. Preparation leaves the section with every line unchanged from the filled template. Every inflow that the model declares reaches SWMM. The baseflow of a sewer is an example. Coupling of the node makes no difference.

Every rule here assumes SWMM dynamic wave routing, the routing of all simulations in the toolkit. Under dynamic wave SWMM solves a head at every node except an outfall, and a pipe can carry flow in either direction. A coupled manhole therefore drains surface water and surcharges back onto the surface.

## Declaring the coupling nodes

The `swmm_coupling_mode` field of the system config chooses the candidates that the list is selected from.

| `swmm_coupling_mode` | Candidates |
|---|---|
| `all_junctions`, the default | Every `[JUNCTIONS]` node of `SWMM_hydraulics` with a `[COORDINATES]` row. |
| `list` | The nodes named in `swmm_coupling_nodes_file`. Each node must be a junction with a `[COORDINATES]` row. |
| `legacy`, an explicit opt-in | Every node of `SWMM_hydraulics` with an `[INFLOWS]` line with constituent `FLOW`, as SWMM reads the line. |

`swmm_coupling_nodes_file` is required with `list` and refused with any other mode, because the toolkit would ignore the file. The file holds a node name per line. Surrounding whitespace is stripped. Blank lines and lines starting with a semicolon are skipped. The file has no header.

A system config that names coupling nodes sets both fields together.

```yaml
swmm_coupling_mode: list
swmm_coupling_nodes_file: swmm_coupling_nodes.txt
```

**The candidates of the opt-in mode** follow the SWMM reading of an inflow line. A line names a FLOW inflow when the line holds at least a node, a constituent, and a time series, and the constituent is no name in `[POLLUTANTS]` and begins with `FLOW`, ignoring the case of ASCII letters. A node with only a pollutant inflow line is therefore no candidate, because SWMM gives the node no water inflow. Under `list` or `all_junctions` the toolkit reads `[INFLOWS]` for the undefined-node check and for no other purpose, so no inflow line adds, removes, or reorders a candidate.

Only the coupled model reads the list. A system with the coupled model off, `toggle_tritonswmm_model: false`, selects no coupling nodes and writes no list. The coupling checks made before submission skip that system. The undefined-inflow-node check still reads the standalone `SWMM_full` model when that model is on. Of the declaration, only the config-load rule on `swmm_coupling_nodes_file` above still applies.

## Coupled node types

**Every candidate junction is coupled by default.** Each junction is one exchange point, so the number of exchange points is independent of the DEM resolution, and a 10 m model and a 1 m model of the same network couple the same nodes. A DEM cell can hold several coupled nodes.

The list is written in SWMM node order, the order that the model defines the nodes in.

| Node type | Coupled | Reason |
|---|---|---|
| Junction, with any links | yes | An ordinary manhole, or an inlet box that drains through an orifice, a weir, or a pump. SWMM solves the head like the head of any junction, and a junction with only incoming pipes still drains by reverse flow. |
| Outfall | no | SWMM sets the water level of an outfall as a boundary condition, so the manhole exchange of TRITON-SWMM is inapplicable. Coupling an outfall with receiving water in the surface model would need a solver change that this build lacks. |
| Storage node | no | A solver limitation and no judgement about the node. TRITON-SWMM couples no storage node. For a pond with the basin in the DEM, couple the junction of the outlet structure. |
| Divider | no | A solver limitation. TRITON-SWMM couples no divider, although under dynamic wave a divider acts as a junction. |

A candidate of an uncoupled type keeps the place in the pipe network and the inflow lines. The candidate is left off the coupling list. One warning per scenario names every such node and the type.

## One node per DEM cell with the opt-in mode

Setting `one_coupled_inflow_node_per_dem_cell: true` in the analysis config limits each DEM cell to one coupled junction. The option applies only to the opt-in mode. Under `list` or `all_junctions` the declaration is the coupled set, so the combination is refused before any job is submitted. In each cell with several candidate junctions the toolkit keeps one junction by the following rules.

1. **Largest pipe first.** The toolkit ranks each junction by the largest flow area among the outgoing conduits, orifices, and weirs of the junction. A junction without such an outgoing link ranks by the largest incoming link, the pipe that the junction drains through by reverse flow. A junction with neither ranks last. Pumps and outlets carry no cross-section and never serve for ranking.
2. **Ties go to the exit of the cell.** When junctions share the largest key, the toolkit keeps the junction of the cell whose outgoing link leaves the cell, then the junction with the lowest invert, then the first junction. The choice covers all junctions of the cell, so a tie can hand the cell to the exit junction even when the pipe of the exit junction is smaller.

A junction that the ranking drops is left off the coupling list, and the inflow lines of the junction stay in the model.

The flow areas are the full areas that SWMM uses for each cross-section shape, computed by `calc_area` in `src/hhemt/scenario.py`. **Additional pipe shapes can be added there.** Size-coded horizontal ellipses and arches have a first dimension that is a standard-size code and no length, and the toolkit computes no area for the shapes.

The pipe-size key sometimes misses the manhole that would overflow first. On the reference model of the toolkit, `test_data/TRITON_SWMM_test_model`, and on no other measured model, the key kept a manhole other than the lowest-rimmed manhole in some of the cells that the key decided without a tie. Within one cell every manhole sits under one ground elevation, so a rim difference there is a difference in recorded manhole depths. An uncoupled manhole that fills to the rim keeps the excess inside SWMM and spills no water onto the surface.

With the option on, every prepared scenario records each reduced cell in `_status/inflow_node_selection.json`. The record holds the cell, every candidate with the type, the node kept, and the reason for each node left off the list. The **Inflow node reduction** row in the Errors and Warnings section of the report lists the cells. A scenario prepared without the option has no record. The same holds for a system with the coupled model off. Both appear as not measured.

## Rims and flow units

TRITON-SWMM takes the rim of a coupled node from the depth that SWMM uses for the node. The rim is the surface connection height. SWMM and TRITON-SWMM therefore agree on every coupled rim. SWMM raises the depth recorded in `[JUNCTIONS]` to the crown of the highest connecting pipe of the node.

Setting `swmm_node_rim_from_dem: true` in the analysis configuration takes the rim of each coupled node from the DEM. The rim is the elevation of the grid cell of the node minus the invert of the node. The option is off by default.

The conversion of the surface hydrographs to cubic meters per second starts from the flow unit of the hydrology model and finishes before TRITON reads the hydrographs. The runoff of an `LPS` or `MLD` model therefore reaches the surface in the right units. The per-node inflow record beside the hydrographs is converted the same way and names the unit that the conversion started from. The hydrology model declares `CMS`, `LPS`, or `MLD`. Scenario preparation refuses `CFS`, `GPM`, and `MGD`, and counts a model with no `FLOW_UNITS` line as `CFS`.

The hydraulics model and the standalone model must use `CMS`. The toolkit reads their flows from the SWMM report and output files as cubic meters per second without conversion. SWMM writes the files in the unit of the model, so a model in `LPS` would be read 1000 times too large and a model in `MLD` 86.4 times too large. The toolkit reads a `FLOW_UNITS` value the way SWMM reads the value. The last line that carries a value wins, and the first letters match the unit word.
