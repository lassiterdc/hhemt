# Set manhole values per coupling node

## Goal

Give individual coupling nodes their own manhole diameter and loss coefficient with a table, and leave every other node at the constants of the analysis config.

## Prerequisites

- A working [installation](installation.md).
- A coupled system with `toggle_tritonswmm_model` on, built from a [runnable SWMM model](using-a-runnable-swmm-model.md).
- A native TRITON build that reads the config key `swmm_manhole_table`. Scenario preparation refuses a build that ignores the key, and refuses container mode, because no image reads the key.

## Key references

- [Coupled inflow nodes](../reference/coupled-inflow-nodes.md), the choice of coupling nodes.
- [Configuration schema](../reference/config-schema.md) lists `manhole_diameter`, `manhole_loss_coefficient`, `manhole_table`, and `manhole_table_fill_with_constants`.

## 1. Write the table

The table is a CSV file. The first row is the header as shown below. Each later row names a coupling node.

```text
node_id,manhole_diameter_m,manhole_loss_coefficient
J1,0.6,0.35
J2,2.4,0.2
```

| Column | Meaning |
|---|---|
| `node_id` | The name of a coupling node in the SWMM model. Matching ignores case. |
| `manhole_diameter_m` | The manhole diameter in meters. The value is finite and above zero. |
| `manhole_loss_coefficient` | The dimensionless loss coefficient. The value is finite and not negative. |

Every row carries both values. A row with an empty cell is refused, and the refusal names the row and the column. A node name appears in 1 row only.

## 2. Point the analysis config at the table

```yaml
manhole_table: /path/to/manhole_table.csv
```

A node with a row takes both values of the row. The solver ignores `manhole_diameter` and `manhole_loss_coefficient` for that node. A node without a row follows step 3.

## 3. Cover every coupling node, or fill the rest

By default the table matches the coupling nodes 1 to 1. A coupling node with no row refuses the run. The run-entry check applies the rule before the toolkit queues a job, except for a node whose name holds a placeholder and for the legacy mode that couples 1 node per DEM cell. The check leaves both cases to scenario preparation. Scenario preparation applies the rule to the exact set of coupling nodes. The error lists up to 10 nodes without a row and then counts the rest.

```text
manhole_table /path/to/manhole_table.csv is refused:
    - it gives a diameter and a loss coefficient for 18 of the 20 coupling nodes of model.inp; 2 coupling node(s) have no row: J7, J12. Every coupling node needs a row. To give the nodes without a row the constant manhole_diameter (1.2 m) and the constant manhole_loss_coefficient (0.1), set analysis_config.manhole_table_fill_with_constants: true
```

Resolve the error in 1 of 2 ways.

- Add a row for each node that the error names.
- Give the remaining nodes the constants by setting the fill toggle.

```yaml
manhole_table_fill_with_constants: true
```

The toggle defaults to `false`, and the config check refuses the toggle when `manhole_table` is unset. The toggle never excuses a misspelt name. A row that names no coupling node is refused whatever the toggle says, because a wrong name would leave the intended node at the constants without a message.

## 4. Check the file the solver reads

Scenario preparation writes `manhole_table.txt` beside `coupling_nodes.txt`. The file holds 1 line per coupling node, in the order of `coupling_nodes.txt`, as `name diameter loss`. A node filled from the constants carries both constants on the line of the node, so the solver always reads a complete table. The generated TRITON cfg gains one `swmm_manhole_table` line that holds the path of the file. With no table and no `crown_rim` open-water outfall, the toolkit writes no such line and the cfg is the cfg of a run without the feature. A `crown_rim` list of [open-water outfalls](open-water-outfalls.md) makes the toolkit write the file and the line even with no table, and each listed outfall takes the diameter and the loss that the toolkit sizes from the outlet conduit of the outfall.

The coupled node store of the consolidated outputs then carries `manhole_diameter` and `manhole_loss_coefficient` on `node_id`, with a missing value at every node that is not a coupling node.

## 5. Read the results with the change in mind

A table whose values differ from the constants changes the hydraulic results. The manhole diameter and the loss coefficient enter every exchange between the surface and the sewer at the node. A table whose rows all equal both constants gives the solver the same values as no table.
