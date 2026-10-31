# Benchmarking study: which compute configuration fits your hardware

A **benchmarking study** is a sensitivity analysis whose sweep axis is the compute
configuration. One spreadsheet row becomes one [member](../reference/glossary.md#member):
the same model, run again, with a different number of MPI ranks, threads or GPUs. A single
report then compares the members on run time, compute cost and strong-scaling speedup
(`sensitivity_benchmarking`), so the study answers a question you cannot answer by reading
a spec sheet. On this machine, and for this model, which compute configuration should I
use?

Every other tutorial here gets one run finished. This one is the study that says how the
rest of your compute should be spent, and it is the reason the toolkit carries a
reporting set built for it.

## Prerequisites

--8<-- "prerequisites-base.md"

- The [Quickstart](quickstart.md) completed, so you have a working environment, a
  compiled solver, and the Norfolk case study on disk. Setup 1 starts from exactly that
  state.
- For setups 2, 3 and 4, the toolkit installed and the solver built **on the cluster you
  will submit from**, because those setups dispatch through SLURM and `hhemt run` has to be
  invoked where both exist. The Quickstart's `compile_TRITON_SWMM()` builds the CPU backend
  on the machine it ran on and reads no HPC profile, so a laptop Quickstart does not
  provision the cluster. Use the command-line build in
  [Compile the solver: On a cluster](../how-to/compiling-the-solver.md#on-a-cluster), which
  takes the profile and builds the GPU backend when the partition declares one. Run it once
  for each partition you will submit to.
- For setups 2, 3 and 4, an `hpc_system_config` for your cluster, describing the
  partitions you intend to sweep. [HPC-profile setup](../how-to/hpc-profile-setup.md)
  authors one, and the anonymized UVA and Frontier profiles under
  `test_data/norfolk_coastal_flooding/` are working starting points.

## The four setups

Each setup is reproducible on its own, and together they form a ladder: the first needs
nothing but the machine you already ran the Quickstart on, and each later one adds one
piece of hardware or one further comparison.

| Setup | Members | Where it runs | What it tells you |
|---|---|---|---|
| 1. Two members on one machine | 2 | Your own machine, `local` dispatch | That the sweep plumbing is right, and what one more thread buys |
| 2. CPU scaling on one node | 6 | One cluster node, `batch_job` | How the model scales in MPI ranks and OpenMP threads |
| 3. GPU count on one node | 4 | One GPU node, `batch_job` | What each additional GPU buys, measured against the CPU baseline |
| 4. The same study on a second machine | 5 | A second cluster, one allocation | Whether the answer holds on other hardware, and how far apart the two machines are |

The sheets below are the whole input. Write each one as an `.xlsx` file with the columns
shown; the rows are the members.

## How a study is put together

Three ingredients.

1. **A sweep spreadsheet.** Its `member_id` column names each member. Every other column
   names one config field to override for that row, in the `analysis.{field}` form the
   next section describes.
2. **Two fields on the analysis config.** `toggle_sensitivity_analysis: true` turns the
   analysis into a sweep, and `sensitivity_analysis` names the spreadsheet.
3. **The `report` block**, selecting the `benchmarking` reporting set, which is the bundle
   of renderers that reads a sweep axis and draws run time, compute cost and speedup. See
   [Reporting sets](../reference/reporting-sets.md) for what else can be selected.

### The sweep spreadsheet

A column is read in one of the forms below, and the loader refuses any column it cannot
read, naming both the offending columns and the forms it accepts. That refusal is
deliberate: a column the toolkit cannot interpret would otherwise be a silent no-op.

| Column form | Example | Meaning |
|---|---|---|
| `member_id` | `c_mpi4` | Required, unique, and safe as a filename fragment. Values must match `^[A-Za-z0-9_.]+$`. |
| `analysis.{field}` | `analysis.n_mpi_procs` | The canonical form for an analysis-config field: overrides that field for the member. |
| `system.{field}` | `system.target_dem_resolution` | Overlays that system-config field for the member. |
| `system_config_yaml` | `systems/uva_two_node.yaml` | A whole system config for this member, by path. Mutually exclusive with `system.*` columns on the same row. |
| `hpc.partition` or `hpc.setup_partition` | `hpc.partition` | Aliases that set the two partition selectors per row, so a sweep can cross partitions and the GPU hardware that comes with them, in a dispatch mode that gives each member its own job (see setup 4). |
| A bare analysis-config field name | `run_mode` | Deprecated: still read, but the loader warns on every such cell and names `analysis.{field}` as the replacement, so the form is not one to author a new sheet in. |

The sheets on this page are written in the `analysis.{field}` form, so a study copied from
them runs without those deprecation warnings.

A blank cell means the member inherits the value from the analysis config, so a sheet
usually names only the fields that actually differ between its rows. GPU hardware is not
an axis of its own: it derives from the partition a row names, and the sweep refuses an
`hpc.gpu_hardware` column for that reason.

!!! warning "Start from the sheets on this page, not from the shipped suites"
    The spreadsheets under `test_data/norfolk_coastal_flooding/` record the project's own
    published suites. They carry annotation columns, such as `device`,
    `total_MPI_ranks` and `selected`, that are notes to a human reader and that the sweep
    loader refuses. Copy the config-field columns out of them, or write a sheet from the
    tables here.

### Point the analysis config at it

The three ingredients are these lines:

```yaml
toggle_sensitivity_analysis: true
sensitivity_analysis: sweeps/benchmarking_local.xlsx
report:
  reporting_set: benchmarking
```

Everything else stays as it is in the Norfolk templates shipped in the repository:
`test_data/norfolk_coastal_flooding/template_system_config.yaml` and
`template_analysis_config.yaml`, the pair `NorfolkIreneExperiment.load()` builds the
Quickstart's case study from. Copy them to `my_system.yaml` and `bench_local.yaml`, and
fill the placeholders they carry: the `${DATA_DIR}` and `${PACKAGE_DIR}` roots,
`hhemt_sha` (the full 40-hex commit you are running), and `weather_events_to_simulate`,
which the template ships as a `PLACEHOLDER.csv` path. The dispatch fields are the only
per-setup difference, and each setup below names the ones it needs.

### Run it

Validate first. A dry run validates the configs and prints the plan without executing
anything, which is the cheapest place to catch a member that cannot run as its row asks.

```bash
hhemt run --system-config my_system.yaml --analysis-config bench_local.yaml --dry-run
hhemt run --system-config my_system.yaml --analysis-config bench_local.yaml
```

Cluster setups add your HPC profile:

```bash
hhemt run --system-config my_system.yaml --analysis-config bench_cpu.yaml \
    --hpc-system-config hpc_system_config_uva.yaml --dry-run
hhemt run --system-config my_system.yaml --analysis-config bench_cpu.yaml \
    --hpc-system-config hpc_system_config_uva.yaml
```

The run creates one analysis directory per member, under `members/member_{member_id}/`,
and one report for the study. Edit the spreadsheet, never the derived
`sensitivity_analysis_definition.csv`: the CSV is regenerated from the spreadsheet on
every run, so a CSV edit is silently overwritten before the workflow plans.

## Setup 1: two members on one machine

The smallest study that is still a study, and the only one that needs no scheduler. It
runs the model twice: once on one process, once on one process with two OpenMP threads.
Two members is enough to exercise the parts that break: the member directories, the
per-member config overlay, and a report that puts the two side by side on run time and
compute cost.

| `member_id` | `analysis.run_mode` | `analysis.n_omp_threads` |
|---|---|---|
| `m_serial` | `serial` | |
| `m_omp2` | `openmp` | `2` |

```yaml
analysis_id: benchmarking_local
toggle_sensitivity_analysis: true
sensitivity_analysis: sweeps/benchmarking_local.xlsx
multi_sim_run_method: local
local_cpu_cores_for_workflow: 4
report:
  reporting_set: benchmarking
```

`multi_sim_run_method: local` dispatches the members to a local thread pool, with no
SLURM. Size `local_cpu_cores_for_workflow` to the cores you are willing to commit; it
bounds the pool, not the members.

## Setup 2: CPU scaling on one cluster node

The classic strong-scaling question, asked on your own partition: as MPI ranks and
threads go up, how much faster does the model get? Six members fix the domain size and
vary only the parallelism.

| `member_id` | `analysis.run_mode` | `analysis.n_mpi_procs` | `analysis.n_omp_threads` |
|---|---|---|---|
| `c_serial` | `serial` | | |
| `c_omp2` | `openmp` | | `2` |
| `c_mpi2` | `mpi` | `2` | `1` |
| `c_mpi4` | `mpi` | `4` | `1` |
| `c_hyb2x2` | `hybrid` | `2` | `2` |
| `c_hyb4x2` | `hybrid` | `4` | `2` |

```yaml
analysis_id: benchmarking_cpu
toggle_sensitivity_analysis: true
sensitivity_analysis: sweeps/benchmarking_cpu.xlsx
multi_sim_run_method: batch_job
hpc_total_job_duration_min: 240
hpc_time_min_per_sim: 120
hpc_ensemble_partition: standard
hpc_setup_and_analysis_processing_partition: standard
report:
  reporting_set: benchmarking
```

Name your own partitions in the two selector fields. `hpc_ensemble_partition` is the
partition the simulation rules run on, and `hpc_setup_and_analysis_processing_partition`
carries the single-node setup, processing and reporting rules around them.

`serial` is the one-device member. Keep one in every study: the scaling panels normalize
each series against the quickest member at the smallest device count, so a study with
nothing at that end has no anchor to read a speedup from.

## Setup 3: GPU count on one GPU node

Same idea on the GPU partition: one CPU baseline plus one, two and four GPUs. The CPU row
matters twice here. It gives the GPU rows something to be compared against, and it shows
when a GPU is not worth its queue time for your model at your domain size.

| `member_id` | `analysis.run_mode` | `analysis.n_gpus` | `analysis.n_mpi_procs` | `hpc.partition` |
|---|---|---|---|---|
| `g_cpu1` | `serial` | | | `standard` |
| `g_gpu1` | `gpu` | `1` | `1` | `gpu-a6000` |
| `g_gpu2` | `gpu` | `2` | `2` | `gpu-a6000` |
| `g_gpu4` | `gpu` | `4` | `4` | `gpu-a6000` |

```yaml
analysis_id: benchmarking_gpu
toggle_sensitivity_analysis: true
sensitivity_analysis: sweeps/benchmarking_gpu.xlsx
multi_sim_run_method: batch_job
hpc_total_job_duration_min: 180
hpc_time_min_per_sim: 90
hpc_ensemble_partition: standard
hpc_setup_and_analysis_processing_partition: standard
report:
  reporting_set: benchmarking
```

The `hpc.partition` column is what makes this a mixed study: each row names the partition
it runs on, so the CPU row stays off the GPU partition. That matters on a cluster whose GPU
partition enforces a GPU minimum, because a CPU-only job submitted there is refused. The two
partition names here are the shipped UVA profile's; substitute the ones your own profile
declares, and on a cluster with a single partition use that one name on every row.

Two things to expect before you read the figures. The GPU members need the solver built
for the GPU partition, which is a different build from the CPU one, so a study that mixes
both partitions builds more than one target. [Compile the solver](../how-to/compiling-the-solver.md#on-a-cluster)
says what a GPU partition needs that a CPU build does not. And how GPU count scales
depends on the domain size, so a member that adds nothing at a small grid is a
measurement, not a bug; the study exists to find where that changes.

## Setup 4: the same study on a second machine

The last question is the portable one: given the same model, how do two machines compare?
Five members span both axes, and running the sheet under a second HPC profile answers it.

| `member_id` | `analysis.run_mode` | `analysis.n_omp_threads` | `analysis.n_mpi_procs` | `analysis.n_gpus` | `hpc.partition` |
|---|---|---|---|---|---|
| `x_serial` | `serial` | | | | `gpu-a6000` |
| `x_omp2` | `openmp` | `2` | | | `gpu-a6000` |
| `x_mpi2` | `mpi` | `1` | `2` | | `gpu-a6000` |
| `x_gpu1` | `gpu` | | `1` | `1` | `gpu-a6000` |
| `x_gpu2` | `gpu` | | `2` | `2` | `gpu-a6000` |

```yaml
analysis_id: benchmarking_second_machine
toggle_sensitivity_analysis: true
sensitivity_analysis: sweeps/benchmarking_second_machine.xlsx
multi_sim_run_method: 1_job_many_srun_tasks
hpc_total_nodes: 1
hpc_total_job_duration_min: 300
hpc_ensemble_partition: gpu-a6000
hpc_setup_and_analysis_processing_partition: standard
report:
  reporting_set: benchmarking
```

All five members run inside **one** allocation, and `hpc_ensemble_partition` is the
`#SBATCH --partition` of that allocation. So this setup cannot cross partitions: every
row's `hpc.partition` names the same partition, and it is the one the allocation sits on.
It is `gpu-a6000` here because the study includes GPU members, and an allocation on a
partition that grants no GPUs leaves them nothing to bind. When you run the sheet on the
second machine, three things change together: both selector fields, the cluster profile,
and the partition name in every row. On the shipped Frontier profile all three take
`batch`, because that cluster has one partition rather than a CPU one and a GPU one. If
you need members on different partitions inside one study, run it under setup 3's dispatch
instead: `batch_job` submits one job per member, which is the mode that can put rows on
different partitions.

Run the study twice: the same five members, each time under that machine's HPC profile,
with the partition names set for that machine and a different `analysis_id`. Then put the
two studies side by side. A per-machine report compares members within one machine; the
cross-machine comparison is produced by combining the two studies.

```bash
# once per machine; this is the Frontier run
hhemt bundle --system-config my_system.yaml --analysis-config bench_second.yaml \
    --hpc-system-config hpc_system_config_frontier.yaml
```

The bundle command writes a zip under the analysis directory's `render_bundle/`, one per
machine. The combine takes **unpacked bundle directories, not the zips**: `hhemt combine`
accepts directories only, so a `.zip` path is refused rather than unpacked for you. Unpack
each run's bundle, then combine the two directories:

```bash
unzip benchmarking_second_frontier/render_bundle/*.zip -d bundles/frontier/
unzip benchmarking_second_uva/render_bundle/*.zip -d bundles/uva/

hhemt combine bundles/frontier/ bundles/uva/ -o combined/
```

[Combining experiments](../how-to/combining-experiments.md) covers the arguments, the
default output path and the severity table. Read the combined report from the top: it
opens on the compatibility check between the two studies before it draws anything, because
the differences it reports about the two machines' compute configurations decide how far
the cross-machine figures can be read at all.

## Read the report

The run's last step renders `analysis_report.html` in the study's analysis directory, with
every figure inlined, so it opens with no server and no unzip step. Open it and read the
benchmarking sections from the top: the metadata and per-analysis summary say what ran,
and the benchmarking figures give run time, compute cost and scaling across the members.
[The interactive analysis report](../reference/example-report.md) documents the rest of
the sidebar.

Three readings are easy to get wrong. Compare members within one study and not across
studies, because the numbers are what this domain, this build and this partition produced.
Read the scaling panels knowing what they divide by: speedup is normalized against the
quickest member at the smallest device count, so a study with a missing or badly timed
member at that end has speedup panels that mislead rather than merely wobble. And keep one
GPU hardware family within a study, because hardware is an axis of the figures; comparing
two families is what setup 4's cross-machine combine is for.

## Next steps

- [Reporting sets](../reference/reporting-sets.md): what `benchmarking` bundles, and how
  to add the compute-sensitivity figures that check whether results are invariant to the
  compute configuration.
- [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md):
  the same study on a fully generated model, for when you want to characterize a cluster
  and have no case-study data staged yet.
- [Configuration schema](../reference/config-schema.md): every field in the blocks above,
  including the toggle-dependent required fields each dispatch mode adds.
- [Norfolk end-to-end](norfolk-end-to-end.md): the compute-configuration and dispatch axes
  explained as a ladder, which is the fastest way to read the fields these sheets vary.
