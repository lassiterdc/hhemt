# Running a synthetic compute-sensitivity experiment

A **synthetic compute-sensitivity experiment** sweeps a small, fully-generated
synthetic TRITON-SWMM model across compute configurations (MPI-rank counts, run
modes, GPU vs CPU partitions) and produces a report whose figures compare the
results across those configurations — verifying that the physics is invariant to
the compute configuration and quantifying where it is not.

Use this when you want an HPC-free-to-scaffold, standardized experiment to
characterize how a cluster's compute choices affect (or do not affect) the model
outputs.

## Prerequisites

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

## Scaffold the experiment

Validate the config and build the partition-as-axis matrix (and, without
`--dry-run`, write the matrix CSV and generate the synthetic model):

```bash
# Load-smoke: validate config + build matrix in memory, write nothing.
hhemt synth-experiment --config synth_experiment.yaml --dry-run

# Scaffold: validate + build matrix + write the matrix CSV + generate the model.
hhemt synth-experiment --config synth_experiment.yaml \
    --hpc-system-config hpc_system_config_uva.yaml \
    --dest-dir runs/synth_cc/
```

The config's cross-field validators reject any requested
`(n_mpi_procs, n_gpus, n_nodes, partition)` tuple that exceeds the resolved
`PartitionSpec` caps before submission.

!!! note "Running the full ensemble"
    `hhemt synth-experiment` currently scaffolds the experiment **inputs**
    (validated config + matrix CSV + generated model). Composing and running the
    full clean+resume ensemble from the framework is a tracked follow-up; today
    the ensemble is driven by the companion estate driver
    (`scripts/experiments/synth_compute_config.py`), which runs the matrix and
    consolidates the outputs into a sensitivity master.

## Read the report

After the ensemble has run and consolidated, produce the exploratory figures and
select the compute-sensitivity reporting set so they render as config-selectable
tabs in `analysis_report.html`:

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")   # the sensitivity master
tk.analysis.eda()                                           # emit plots/eda/ figures
tk.analysis.render_report()                                 # renders the active reporting set
```

Running `hhemt eda` (or `analysis.eda()`) on the completed sensitivity master now
also emits the compute-sensitivity EDA family by default:
`plots/eda/eda_rank_sensitivity.html` (within-family mpi rank-N vs rank-1 identity +
magnitude) and `plots/eda/eda_cross_hardware_magnitude.html` (the ADR-4
characterized-divergence panel: 1-GPU vs 1-rank serial CPU). A third member,
`eda_resume_sensitivity` (clean-vs-resume identity + magnitude, paired per
compute-config), is an OPT-IN figure: it renders only for a single master carrying
BOTH a clean and a resume arm, so the compute-sensitivity experiment — run as two
SEPARATE single-arm masters (a clean sweep and a resume sweep) — skips it and produces
the clean-vs-resume comparison at COMBINE level via `hhemt combine` (the
`cross_experiment_intercomparison` figure) instead; enable it explicitly via
`enabled_plots` for a future both-arms master. Each rendered member writes a backing
`eda/{plot_id}.zarr` provenance artifact and an `eda/{plot_id}.verdict.json` whose
verdict is merged into the report's Errors-and-Warnings section. A member whose
experiment shape does not supply the required pair (for example, a matrix with no
resume arm) skips silently and emits no figure.

Select the `compute-sensitivity` reporting set via
`report_config.reporting_set: compute-sensitivity` in your report config. The
rendered report then carries the compute-config EDA figures (config-diff maps plus
the compute-sensitivity family described above — rank and cross-hardware by default;
resume is opt-in) under **Key Results**, alongside the benchmarking figures.

To compare two experiments (e.g. clean vs resume, or two clusters) in one
report, emit a bundle from each and combine them — see
[Combining experiments](combining-experiments.md); `combine` accepts
sensitivity-master bundles.
