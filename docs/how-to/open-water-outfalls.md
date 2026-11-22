# Represent outfalls to open water

## Goal

List the outfalls of the SWMM model that drain to open water, where the water can push back into the sewer, and choose how the coupled model represents them.

## Prerequisites

- A working [installation](installation.md).
- A system with `toggle_tritonswmm_model` on and a [runnable SWMM model](using-a-runnable-swmm-model.md) as `SWMM_model`.
- A native TRITON build that reads the cfg keys of the representation that you choose, as step 3 names. A container run is refused, because no container image carries the keys.

## Key references

- [Configuration schema](../reference/config-schema.md) lists `swmm_open_water_outfalls` and `open_water_outfall_representation`.
- [Coupled inflow nodes](../reference/coupled-inflow-nodes.md) covers the selection of coupling nodes and the rim of a converted outfall.
- [Set manhole values per coupling node](per-node-manhole-values.md) describes the manhole table that the `crown_rim` representation writes rows into.

## 1. Decide whether an outfall belongs in the list

List an outfall that drains to open water, where the water level can rise above the pipe and push water back into the sewer. A tidal outfall is the usual case.

A listed outfall carries a decision, like the [outfall decisions](using-a-runnable-swmm-model.md#3-decide-each-outfall) of the runnable-model guide. A listed outfall needs no other decision, and the check refuses a name that is also in `swmm_outfalls_to_couple` or `swmm_outfalls_to_keep`. An outfall outside the list follows the existing rules, unchanged.

The list is empty by default. With an empty list the toolkit treats no outfall differently, and every file that the toolkit writes matches the file written before the field existed. A non-empty list changes hydraulic results.

## 2. List the outfalls and choose a representation

Both fields are analysis config fields.

```yaml
swmm_open_water_outfalls: [E143250, E147007]
open_water_outfall_representation: driven_stage
```

Each name is a row of the `[OUTFALLS]` section of `SWMM_model`, and the case of the name makes no difference. One representation applies to every listed outfall, so each simulation treats the listed outfalls alike. `open_water_outfall_representation` defaults to `driven_stage`, and the toolkit reads the field only when the list is non-empty.

| Representation | Effect on the model |
|---|---|
| `driven_stage` | The outfall stays a SWMM outfall. The toolkit writes the outfall into the hydraulics model as a fixed outfall with the stage at the invert, and TRITON sets the stage before every SWMM step. |
| `crown_rim` | The toolkit converts the outfall to a coupled junction. SWMM raises the junction to the crown of the outlet conduit, and the toolkit writes the manhole diameter and loss coefficient of the junction. |

### The driven-stage representation

TRITON sets the stage of a listed outfall to the water-surface elevation of the TRITON cell that holds the node, or to the invert of the outfall when the cell is dry. SWMM then computes the flow of the outfall conduit as for any fixed outfall. Water that the stage pushes into the sewer is taken from the cell. Water that the sewer discharges at the outfall leaves the modeled domain and reaches no cell.

A driven outfall is not a coupling node, because the outfall has no exchange with the surface other than the stage. Scenario preparation refuses a driven outfall that is among the coupling nodes written for the scenario. The check also refuses a listed outfall with no `[COORDINATES]` row or a position outside the TRITON grid, because TRITON reads the cell that holds the node.

An external-boundary record of type `water_level` can still name a listed outfall. The record then drives the boundary line of TRITON as for any outfall, and the record leaves the outfall unconverted.

### The crown-rim representation

The `crown_rim` representation is the comparison representation. The toolkit converts each listed outfall to a junction with a depth of 0. SWMM lifts that depth to the crown of the conduits that meet at the node, so the rim of the junction is the crown of the outlet conduit, the lowest rim that SWMM allows, and never the invert.

The toolkit sizes the manhole of the junction from the one conduit whose outlet node is the outfall.

| Quantity | Value |
|---|---|
| Diameter | The diameter of the circle with the same area as the total flow area of the conduit, `d = sqrt(4 A / pi)`. The area counts every barrel of the conduit. |
| Loss coefficient | `manhole_loss_coefficient` of the analysis config. |

The twin rectangular barrels of the 2 Norfolk outfalls show the rule. Both conduits have zero offsets.

| Outfall | Outlet conduit | Cross-section | Total area | Diameter | Crown above the invert |
|---|---|---|---|---|---|
| `E143250` | `1444` | `RECT_CLOSED`, 1.524 m by 1.676 m, 2 barrels | 5.108 m² | 2.550 m | 1.524 m |
| `E147007` | `1445` | `RECT_CLOSED`, 1.676 m by 1.676 m, 2 barrels | 5.618 m² | 2.675 m | 1.676 m |

Each listed outfall needs 1 conduit that ends at the outfall, and the cross-section of that conduit needs a flow area above 0. The check refuses an outfall with no such conduit, an outfall with more than 1, and a cross-section whose flow area is missing or not above 0. A `manhole_table` row for a listed outfall is refused, because the toolkit writes the row.

A `crown_rim` outfall must be a coupling node. With `swmm_coupling_mode: all_junctions` the converted outfall is a candidate. With `list`, name the outfall in `swmm_coupling_nodes_file`. With `legacy` the check refuses the outfall, because that mode couples only the nodes that carry a `FLOW` inflow.

When `swmm_node_rim_from_dem` is on and `dem_rim_resolution` is `processed`, TRITON takes the rim of every coupled node from the DEM. Preparation then lists the `crown_rim` outfalls in an exemption file, so TRITON keeps the crown-raised depth at the listed nodes only. The DEM depth check skips a `crown_rim` outfall, because the toolkit takes no depth from the DEM for the outfall. With `dem_rim_resolution: full` the toolkit writes no TRITON rim key, and preparation writes no exemption file.

## 3. Pin a TRITON build that reads the keys

Each representation adds config keys to the TRITON cfg. A TRITON build that ignores a key runs the model without the change, with no error from TRITON. Scenario preparation therefore reads `src/config_utils.h` of the native clone and refuses a build that lacks the key.

| Representation | Key the build must read |
|---|---|
| `driven_stage` | `swmm_driven_outfalls` |
| `crown_rim` | `swmm_manhole_table`, and `swmm_dem_rim_exempt_nodes` when `swmm_node_rim_from_dem` is on with the processed DEM |

Set `TRITONSWMM_branch_key` in the system config to a commit that reads the keys.

## 4. Check the config

Run `analysis.validate()` ahead of the first submission. The check collects every refusal and names each offender. The check refuses the cases in the list.

- A listed name that matches no `[OUTFALLS]` row in `SWMM_model`.
- A name that appears more than once in the list.
- A name that is also in `swmm_outfalls_to_couple` or `swmm_outfalls_to_keep`.
- A name that has a row in `manhole_table`.
- Under `crown_rim`, an outfall that lacks a single sizable outlet conduit.
- Under `driven_stage`, an outfall with no `[COORDINATES]` row or outside the TRITON grid.

Scenario preparation applies the same rules again against the exact coupling nodes of the scenario.

## 5. Find the files that preparation writes

Scenario preparation writes the files in the table beside `coupling_nodes.txt`, in the folder of the `hydraulics.inp` of the scenario. With an empty list the toolkit writes no open-water file and adds no line to the TRITON cfg.

| File | Condition | Content |
|---|---|---|
| `driven_outfalls.txt` | `driven_stage` | The listed names, 1 per line, in the order of the `[OUTFALLS]` section. The cfg gains 1 `swmm_driven_outfalls` line with the path of the file. |
| `manhole_table.txt` | `crown_rim`, or any `manhole_table` | 1 line per coupling node as `name diameter loss`, with the sized diameter and loss at each listed outfall. The cfg gains 1 `swmm_manhole_table` line. |
| `dem_rim_exempt_nodes.txt` | `crown_rim` with the rim route on and the processed DEM | The listed names, 1 per line. The cfg gains 1 `swmm_dem_rim_exempt_nodes` line. |
| `crown_rim_outfalls.json` | `crown_rim` | The sizing of each listed outfall, with the outlet conduit, the barrels, the cross-section, the area, the diameter, the crown above the invert, and the loss coefficient. |

The files record the open-water inputs of each simulation.

## 6. Compare the representations

A non-empty list changes hydraulic results, because TRITON then treats the listed outfalls differently from the model. To compare the 2 representations, run 2 analyses that differ only in `open_water_outfall_representation`.
