# Running a synthetic compute-sensitivity experiment

A **synthetic compute-sensitivity experiment** sweeps a small, fully-generated
synthetic TRITON-SWMM model across compute configurations (MPI-rank counts, run
modes, GPU vs CPU partitions) and produces a report whose figures compare the
results across those configurations, verifying that the physics is invariant to
the compute configuration and quantifying where it is not.

Use this when you want a standardized experiment you can set up without HPC, to
characterize how a cluster's compute choices affect (or do not affect) the model
outputs.

## Prerequisites

--8<-- "prerequisites-base.md"
- **An `hpc_system_config` for your cluster.** The experiment resolves each
  matrix row's GPU hardware/backend from the chosen partition's `PartitionSpec`,
  so the cluster profile must describe your partitions. Anonymized examples ship
  in-repo (`test_data/norfolk_coastal_flooding/hpc_system_config_{uva,frontier}.yaml`).
  See [Set up an HPC system profile](hpc-profile-setup.md).
- **A `synthetic_experiment_config` YAML.** It parameterizes the synthetic model
  (grid dims/resolution, conduit + subcatchment counts, event forcing), the
  experiment matrix (compute configs, the MPI-rank sweep axis `rank_sweep`
  defaulting to `{2,4,8}`, clean-vs-resume), and a reference to the
  `hpc_system_config` + partition selectors. The cross-hardware axis is expressed
  as the **partition** (an a6000 row + an a100 row), not a `gpu_hardware` column.

## Declaring the model arms (and an optional resume schedule)

A compute-sensitivity experiment runs as **one sensitivity master per model arm**.
Declare the arms in the experiment config's matrix block:
`model_arms: [tritonswmm, triton]`. Each arm runs the identical compute-config
matrix and differs only in its system model toggles. The two masters are joined for
reporting by `hhemt combine`; the cross-arm figures live on the combined report, not
on either master's own report.

??? note "Why one master per arm rather than one master with both"
    The sensitivity Snakefile generator accepts exactly one enabled model type per
    master, so a two-arm experiment is two masters by construction rather than by
    preference.

To interrupt the sims for a resume study, set `resume_interruption_schedule` on the
analysis config to a strictly increasing tuple of hotstart-checkpoint indices, not
percentages, not seconds. One checkpoint is written per reporting interval, so on
the 3.5 m / 24 h synthetic grid (10-minute reporting = 144 checkpoints) the
25% / 50% / 75% markers are `[36, 72, 108]`, which produces three resumes at the
same three points in every compute config. That identity across configs is what
makes cross-config resume comparisons meaningful. Convert the fractions to indices
once, in the committed experiment config; the runner performs no conversion. Leave
it unset for a clean sweep. The retired `deterministic_kill_after_n_checkpoints`
field is no longer accepted.

## Set up the experiment

Validate the config and build the partition-as-axis matrix (and, without
`--dry-run`, write the matrix CSV and generate the synthetic model):

```bash
# Load-smoke: validate config + build matrix in memory, write nothing.
hhemt synth-experiment --config synth_experiment.yaml --dry-run

# Full setup: validate + build matrix + write the matrix CSV + generate the model.
hhemt synth-experiment --config synth_experiment.yaml \
    --hpc-system-config hpc_system_config_uva.yaml \
    --dest-dir runs/synth_cc/
```

The config's cross-field validators reject any requested
`(n_mpi_procs, n_gpus, n_nodes, partition)` tuple that exceeds the resolved
`PartitionSpec` caps before submission.

!!! note "Running the full ensemble"
    `hhemt synth-experiment` currently prepares the experiment **inputs**
    (validated config + matrix CSV + generated model). Composing and running the
    full clean+resume ensemble from the framework is a tracked follow-up; today
    the ensemble is driven by the companion driver
    (`scripts/experiments/synth_compute_config.py`), which runs the matrix and
    consolidates the outputs into a sensitivity master.

## Read the report

After the ensemble has run and consolidated, produce the exploratory figures and
select the compute-sensitivity reporting set so they render alongside the
benchmarking figures in `analysis_report.html`:

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")   # the sensitivity master
tk.analysis.eda()                                           # emit plots/eda/ figures
tk.analysis.render_report()                                 # renders the active reporting set
```

Select the `compute-sensitivity` reporting set by setting
`report.reporting_set: compute-sensitivity` on the analysis config (see
[Reporting sets](../reference/reporting-sets.md)). The rendered report then
carries the compute-config EDA figures under **Key Results**, alongside the
benchmarking figures.

Which exploratory figures exist, which of them render, what each one writes, and
how to read a verdict file are covered once in
[Run the exploratory analysis](running-eda.md). Two things are specific to this
experiment. Its default figure selection is `config_diff_maps`, the config-diff
comparison across compute configurations. And because it runs as two separate
single-arm masters, a clean sweep and a resume sweep, the clean-versus-resume
comparison is produced at combine level by `hhemt combine` rather than inside
either master.

## Running a DEM-resolution sweep instead

The same synthetic machinery produces a **DEM-resolution sweep**: the same model
run across DEM cell sizes instead of across compute configs, whose figures
compare peak flood depth from a finer versus a coarser grid.

Set it up through the **sensitivity-CSV overlay**, not through
`hhemt synth-experiment`: that CLI always builds the compute-config matrix and
does **not** dispatch on a resolution axis. A DEM-resolution sweep is instead
expressed by giving the sensitivity CSV a `system.target_dem_resolution` overlay
column, one row per resolution rung:

| `member_id` | ... | `system.target_dem_resolution` |
|---------|-----|--------------------------------|
| `res_3p5` | ... | 3.5 |
| `res_7p0` | ... | 7.0 |
| `res_14p0` | ... | 14.0 |

Each row overlays `target_dem_resolution` onto the master system config
(`system_config.model_validate({**master, **overlay})`), so the members share
everything but the DEM cell size. Choose a **constant-ratio** ladder, each coarser
rung an integer multiple of the finest (e.g. 3.5 / 7.0 / 14.0 m, successive
doubling), so each coarse grid is a clean aggregation of the finest, which is the
**reference** (never "truth": its own error is unquantified). Then select the
DEM-resolution reporting set:

- `report.reporting_set: dem-resolution` on the analysis config (see
  [Reporting sets](../reference/reporting-sets.md)), and
- `eda_config.enabled_plots: [dem_resolution_cost_error, dem_resolution_diff_maps, dem_resolution_error_ecdf, dem_resolution_coupling_table]`.

!!! warning "The two EDA families are mutually exclusive per experiment"
    The `compute-sensitivity` family's `config_diff_maps` requires a **uniform**
    grid across members and raises a named `ProcessingError` on a
    mixed-resolution master; the `dem-resolution` family requires the varying grid.
    Pick one reporting set per experiment: do not mix `config_diff_maps` with the
    `dem_resolution_*` renderers.

To compare two experiments (e.g. clean vs resume, or two clusters) in one
report, emit a bundle from each and combine them. See
[Combining experiments](combining-experiments.md); `combine` accepts
sensitivity-master bundles.
