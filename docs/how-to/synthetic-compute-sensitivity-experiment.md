# Running a synthetic compute-sensitivity experiment

## Goal

Sweep a small, generated synthetic TRITON-SWMM model across compute configurations, meaning MPI-rank counts, run modes, and the choice of GPU or CPU partition, and produce a report whose figures compare the results across the configurations. The sweep verifies that the physics is invariant to the compute configuration and quantifies where the invariance fails. The experiment needs no HPC to set up and characterizes how the compute choices of a cluster affect the model outputs.

## Prerequisites

- A working [installation](installation.md).
- An `hpc_system_config` for your cluster. The experiment resolves the GPU hardware and backend of each matrix row from the `PartitionSpec` of the chosen partition, so the cluster profile must describe your partitions. Anonymized examples are in [`test_data/norfolk_coastal_flooding/`](https://github.com/lassiterdc/hhemt/tree/main/test_data/norfolk_coastal_flooding) on the repository, and [Set up an HPC system profile](hpc-profile-setup.md) describes the file.
- A `synthetic_experiment_config` YAML. The file parameterizes the synthetic model with the grid dimensions, the grid resolution, the conduit counts, the subcatchment counts, and the event forcing. The file also parameterizes the experiment matrix with the compute configs and the clean-versus-resume choice. The MPI-rank sweep axis `rank_sweep` defaults to `{2,4,8}`. The file references the `hpc_system_config` and the partition selectors. The partition expresses the cross-hardware axis with an a6000 row and an a100 row. Each row takes the GPU model from the partition, and [Choose a partition](hpc-profile-setup.md#choose-a-partition) covers the choice.

## Key references

- The [CLI reference](../reference/cli.md#development) documents `hhemt synth-experiment`.
- [Reporting sets](../reference/reporting-sets.md) describes the reporting sets that this page selects.
- [Run the exploratory analysis](running-eda.md) covers the exploratory figures.
- [Combining experiments](combining-experiments.md) covers the clean-versus-resume comparison.

## Declaring the model arms and an optional resume schedule

A compute-sensitivity experiment executes as **one sensitivity primary per model arm**. Declare the arms in the matrix block of the experiment config as `model_arms: [tritonswmm, triton]`. Each arm executes the identical compute-config matrix and differs only in the system model toggles. `hhemt combine` joins the primaries for reporting, and the cross-arm figures live on the combined report and never on the report of either primary.

??? note "One primary per arm"
    The sensitivity Snakefile generator accepts one enabled model type per primary, so an experiment with several arms is several primaries by construction and never by preference.

To interrupt the simulations for a resume comparison, set `resume_interruption_schedule` on the analysis config to a strictly increasing tuple of hotstart-checkpoint indices. The values are neither percentages nor seconds. The simulation writes one checkpoint per reporting interval, so the 3.5 m / 24 h synthetic grid with a 10-minute reporting interval holds 144 checkpoints, and the 25% / 50% / 75% markers are `[36, 72, 108]`. The indices produce resumes at the same points in every compute config. The identity across configs makes cross-config resume comparisons meaningful. Convert the fractions to indices once, in the committed experiment config, because the driver performs no conversion. Leave the field unset for a clean sweep. The driver accepts no `deterministic_kill_after_n_checkpoints` field.

On the 3.5 m / 24 h grid, the analysis config holds the schedule as follows.

```yaml
resume_interruption_schedule: [36, 72, 108]
```

## Set up the experiment

Run `hhemt synth-experiment`, as the [CLI reference](../reference/cli.md#development) describes.

```bash
# Load-smoke: validate config + build matrix in memory, write nothing.
hhemt synth-experiment --config synth_experiment.yaml --dry-run

# Full setup: validate + build matrix + write the matrix CSV + generate the model.
hhemt synth-experiment --config synth_experiment.yaml \
    --hpc-system-config hpc_system_config_uva.yaml \
    --dest-dir runs/synth_cc/
```

The cross-field validators of the config reject any requested `(n_mpi_procs, n_gpus, n_nodes, partition)` tuple that exceeds the resolved `PartitionSpec` caps before submission.

!!! note "Running the full ensemble"
    `hhemt synth-experiment` prepares the experiment **inputs**, meaning the validated config, the matrix CSV, and the generated model. The companion driver `scripts/experiments/synth_compute_config.py` in a repository checkout runs the full clean and resume ensemble. The driver runs the matrix and consolidates the outputs into a sensitivity primary.

## Read the report

After the ensemble has run and consolidated, produce the exploratory figures and select the compute-sensitivity reporting set, so that the figures render alongside the benchmarking figures in `analysis_report.html`.

```bash
hhemt eda --system-config system.yaml --analysis-config analysis.yaml    # emit plots/eda/ figures
hhemt reprocess --system-config system.yaml --analysis-config analysis.yaml --start-with render
```

Both commands take the configs of the sensitivity primary. The second command renders the active reporting set against the existing plots.

Select the `compute_sensitivity` reporting set with `report.reporting_set: compute_sensitivity` on the analysis config, as [Reporting sets](../reference/reporting-sets.md) describes. The rendered report then carries the compute-config EDA figures under **Key Results**.

[Run the exploratory analysis](running-eda.md) covers the figures that exist, the figures that render, the files that each figure writes, and the reading of a verdict file. The experiment adds a default figure selection of `config_diff_maps`, the config-diff comparison across compute configurations. The clean sweep and the resume sweep execute as separate single-arm primaries, so [Combining experiments](combining-experiments.md) covers the clean-versus-resume comparison.

## Running a DEM-resolution sweep instead

The same synthetic machinery produces a **DEM-resolution sweep**, the same model across DEM cell sizes in place of compute configs. The figures compare peak flood depth from a finer grid against a coarser grid.

Set `dem_resolution_ladder` in the `synthetic_experiment_config` YAML, with the finest cell size first. The `dem` subcommand of `scripts/experiments/synth_compute_config.py` writes the matrix, selects the `dem_resolution` reporting set and the DEM-resolution figure types, and builds the analysis. `hhemt synth-experiment` builds the same matrix, with a row for each rung and replicate and a `system.target_dem_resolution` overlay column.

| `member_id` | ... | `system.target_dem_resolution` |
|---------|-----|--------------------------------|
| `dem_3p5_r1` | ... | 3.5 |
| `dem_3p5_r2` | ... | 3.5 |
| `dem_7p0_r1` | ... | 7.0 |
| `dem_7p0_r2` | ... | 7.0 |
| `dem_14p0_r1` | ... | 14.0 |
| `dem_14p0_r2` | ... | 14.0 |

Each row overlays `target_dem_resolution` onto the base system config through `system_config.model_validate({**base, **overlay})`, so the members share everything except the DEM cell size. Choose a **constant-ratio** ladder with each coarser rung an integer multiple of the finest, such as 3.5, 7.0, and 14.0 m with successive doubling, so that each coarse grid is a clean aggregation of the finest. The finest grid is the **reference** and never the truth, because the error of the finest grid is unquantified. Then select the DEM-resolution reporting set and the figures.

- `report.reporting_set: dem_resolution` on the analysis config, as [Reporting sets](../reference/reporting-sets.md) describes.
- `eda.enabled_plots: [dem_resolution_cost_error, dem_resolution_diff_maps, dem_resolution_error_ecdf, dem_resolution_coupling_table]` on the analysis config.

!!! warning "One EDA family per experiment"
    The `compute_sensitivity` and `dem_resolution` families need different grids, so no single sweep serves both, as [Composing several sets](../reference/reporting-sets.md#composing-several-sets) describes.

To compare experiments, such as clean versus resume or different clusters, in one report, emit a bundle from each and combine the bundles, as [Combining experiments](combining-experiments.md) describes. `combine` accepts sensitivity-primary bundles.
