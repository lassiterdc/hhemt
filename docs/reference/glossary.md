# Glossary

The terms the rest of these pages use without stopping to explain. The acronyms
here also show their expansion as a tooltip when you hover over them on any page.
Where a term names a configuration field or a class, the entry says which, so you
can find the field in the [configuration schema](config-schema.md) or the class in
the [API reference](api.md).

## The objects a study is made of

### system

The modelled place and the solver build. It holds the inputs that
every analysis over that place shares, which are the DEM, the Manning's roughness
inputs, the choice of which models to run, and the compiled solver binaries. It
is described by the system config (`system_config`) and is not the computer you
run on; that is the [HPC system config](#hpc-system-config). One system can have
many analyses.

### analysis

A set of weather events simulated against one system, together
with everything produced from them: the simulations, the processed outputs, the
consolidated datasets and the report. It is described by the analysis config
(`analysis_config`). Every analysis is either an [event ensemble](#event-ensemble)
or a [sensitivity analysis](#sensitivity-analysis). By default each analysis lives in its own
directory, named by its `analysis_id`, under its system's directory; the analysis
config's `analysis_dir` places it elsewhere.

### experiment

This word has three senses, and the page you are reading tells
you which.

- An **experiment bundle** is a directory holding an `experiment.yaml`
  descriptor that points to a system config and an analysis config, and records
  the input datasets, the toolkit version it is pinned to and whether it runs in
  a container. Its schema is `ExperimentConfig`. See
  [Running an experiment bundle](../how-to/running-an-experiment-bundle.md).
- A **case-study experiment** is the object that `NorfolkIreneExperiment.load`
  and `TRITON_SWMM_experiment.from_case_study` return. It holds a system as
  `.system` and an analysis as `.analysis`, built from a [case study](#case-study).
- A **sensitivity analysis's experiment root** is the level above its members:
  the place where `experiment_datatree.zarr` is written, with one node per
  completed [member](#member).

### experiment bundle

The directory with an `experiment.yaml` descriptor that makes a study runnable by
someone else. It is the first sense of [experiment](#experiment).

### event ensemble

An analysis whose runs are weather events. Each event is
simulated against the same system, and the results are consolidated for analysis
across the whole population of events. It is the kind of analysis you get when
`toggle_sensitivity_analysis` is `false`.

### sensitivity analysis

An analysis whose runs are the rows of a parameter table
that you supply. Each row varies something about the model or the compute
configuration, and each is run as a [member](#member). It is the kind of analysis
you get when `toggle_sensitivity_analysis` is `true`. A **sweep** is the same
thing under another name; the pages use both words interchangeably.

### sweep

A synonym for [sensitivity analysis](#sensitivity-analysis).

### member

One row of a sensitivity analysis's parameter table. A member is run
as an analysis of its own, in its own directory (`members/member_{member_id}/`),
and carries its own scenarios and consolidated output.

### scenario

One weather event simulated within an analysis. Its files live under
`sims/{event_id}/`. An event ensemble has one scenario per event, and each member
of a sensitivity analysis has its own.

### benchmarking

A sensitivity analysis whose axis is the compute configuration.
It also names the [reporting set](#reporting-set) that selects the figures such a
study needs, which are run time, compute cost and speedup.

### case study

A worked example that ships with the repository under
`test_data/{case_name}/`, described by a `case.yaml` manifest (`CaseManifest`)
that names where its large input data are hosted. The Norfolk coastal flooding
case is the one the tutorial runs.

## Compute and execution

### compute configuration

How one simulation uses the machine. It is set by
`run_mode` together with `n_mpi_procs`, `n_omp_threads`, `n_gpus` and `n_nodes`.

### `run_mode`

The analysis config field that selects the parallelism one
simulation uses. The options are `serial`, `openmp`, `mpi`, `hybrid` (MPI ranks that
each run OpenMP threads) and `gpu`.

### `multi_sim_run_method`

The analysis config field that selects how the many
simulations of an analysis are dispatched. `local` runs them in a thread pool on
the machine you are on, with no SLURM. `batch_job` runs Snakemake in a tmux session
on the login node and submits one SLURM job per scenario. `1_job_many_srun_tasks`
runs all of them as concurrent `srun` tasks inside one SLURM allocation.

### native mode

The default value of `execution_environment`. Simulation and
processing run directly on the host, using a solver you have compiled yourself.

### container mode

The value `container` of `execution_environment`. The
simulation executable and the processing runners are wrapped in
`apptainer exec` on a [SIF](#sif) image, which hhemt resolves for you.

### HPC system config

The third configuration file, which describes a cluster
rather than a study (`hpc_system_config`). It carries the partitions and their
limits, the account to charge and the container settings. Local runs do not need
one.

### partition

A SLURM queue with its own limits. In the HPC system config each
partition is a `PartitionSpec` that records caps such as the longest run time and the
most memory a single job may request, and hhemt rejects a request that exceeds
them before submitting.

### allocation

In SLURM, the resources granted to a job. The word is also used for
the account a job is charged to; the HPC system config's `default_account` field
describes itself that way.

### driver

The process that orchestrates a run. It starts the workflow, submits
the jobs and writes a liveness record under `_status/_orchestrator/`. A blocking
local driver ends when the call returns. A detached driver, such as the tmux session
of `batch_job` or a single batch job, outlives it.

### sentinel

A small JSON file under an analysis's `_status/` directory that marks
work as in flight. `_status/_submitted/{rule_token}.json` records a submitted job,
and `_status/_orchestrator/{driver_id}.json` records a live [driver](#driver). hhemt
checks them so that it does not submit the same work twice and so that deleting or
reprocessing refuses to run over live work. In DEM preparation the word has another
meaning: a sentinel elevation is a placeholder value burned into a DEM, for example
for buildings or for cells outside the watershed.

### hotstart

Restarting a TRITON simulation from a checkpoint it wrote earlier,
instead of from time zero. A simulation that a wall-time limit killed can be
dispatched again and continue from its latest checkpoint.

### Snakemake

The open-source workflow manager hhemt uses to run the steps of an
analysis. It skips steps that have already completed.

### Apptainer

The container runtime used on HPC systems. It runs
[SIF](#sif) images.

## Models and inputs

### TRITON

The two-dimensional hydrodynamic flood model that hhemt compiles and
runs. It is an external solver, not part of hhemt.

### SWMM

The EPA Storm Water Management Model, the stormwater network model that
hhemt compiles and runs. It is an external solver.

### coupled (TRITON-SWMM)

The solver build in which TRITON and SWMM exchange data
during the run. It is enabled by `toggle_tritonswmm_model`, independently of the
TRITON-only and SWMM-only models.

### DEM

Digital elevation model, the raster of ground elevation that TRITON runs
on. The system config points to it as `DEM_fullres`, and it can be coarsened to the
target resolution.

### CRS

Coordinate reference system, which fixes how a raster's coordinates map to
places on the ground. The system config takes the horizontal CRS of the DEM as an
EPSG code.

### Manning's roughness

The coefficient for how much the ground slows flowing
water. It is either one constant value or a lookup from a land-use raster through a
table, as selected by `toggle_use_constant_mannings`.

### NetCDF

A self-describing file format for gridded data. It is the alternative to
Zarr for per-scenario outputs, selected with `target_processed_output_type: nc`.
The weather input is also a NetCDF file.

## Outputs and data formats

### consolidation

Combining the processed outputs of every scenario, and of every
member for a sensitivity analysis, into one hierarchical dataset per analysis. The
result is called the **consolidated** dataset or tree.

### zarr

A chunked array storage format that holds a dataset as a directory of
files. hhemt writes its processed and consolidated outputs as Zarr stores.

### DataTree

The `xarray.DataTree` structure, which holds a hierarchy of datasets
in one object. The consolidated outputs are DataTrees saved as Zarr.

### CF-1.13

Version 1.13 of the Climate and Forecast conventions, a standard for
naming and describing variables (`standard_name`, `long_name`, `units`,
`cell_methods`). The consolidated outputs carry these attributes.

### `system_datatree.zarr`

The per-system tier of the output data model. It is written only when you call
`system.process.consolidate_system_datatree()` yourself; no run writes it for you.
It holds one group for each analysis that was bound to the system when a call ran,
and each call appends.
See [Output data model](output-data-model.md).

### `experiment_datatree.zarr` and `analysis_datatree.zarr`

The consolidated
trees of a sensitivity analysis's experiment root and of a single analysis. The
[output data model](output-data-model.md) lists every tier.

### reporting set

A named selection of report renderers, together with the order
of the report's sidebar categories. Choosing one decides which figures and tables a
report contains. See [Reporting sets](reporting-sets.md).

### render bundle

A portable zip that `bundle_report_data()` writes, holding the
data behind a report's figures, the configs and the Snakefile, so that you can
re-render the report on another machine. It is not an experiment bundle.

### EDA

Exploratory data analysis. In hhemt it is the step that runs after an
analysis has consolidated and compares the analysis against itself, to tell whether
a result belongs to the model or to how it was computed. See
[Exploratory analysis](../explanation/exploratory-analysis.md).

## Publishing and reproducing

### reprex

A reproducible example, here a bundle that anyone can fetch by DOI and
run. hhemt's reprex check confirms that a bundle can be run on a target cluster
profile and suggests the amendments it needs. See
[Publishing and fetching](../how-to/publishing.md).

### RO-Crate

Research Object Crate, a packaging convention that describes a set
of files and how they relate in a `ro-crate-metadata.json` file. hhemt writes one
in each analysis directory, beside its consolidated tree, and at a sensitivity analysis's
experiment root; the per-system tree has none.

### PROV (PROV-O)

The W3C ontology for recording where data came from and how it
was produced. The provenance recorded in the RO-Crate uses it.

### JSON-LD

JSON for Linking Data, the serialization of the `ro-crate-metadata.json`
file.

### CodeMeta

A metadata vocabulary for describing software, used in the RO-Crate's
description of the software.

### DOI

Digital object identifier, a permanent link to a published item.

### PID

Persistent identifier, the general term for a DOI or any similar permanent
identifier. A case study's `case.yaml` accepts a `doi` or, where there is none, the
host's own record `pid`.

### DataCite

The metadata schema that DOIs for datasets are registered with. A
deposit must supply its mandatory fields, which are the identifier, creators,
title, publisher, publication year and resource type.

### FAIR

Findable, accessible, interoperable, reusable, the four properties the
[FAIR scope table](fair-scope-table.md) grades each published artifact on.

### OCI

Open Container Initiative, the standard for container image formats.

### SIF

The image file format that Apptainer runs. hhemt addresses an image by its
identity, so you never name an image file in a config.

## Computing and platform terms

### HPC

High-performance computing, meaning a shared cluster of many nodes run by a
scheduler.

### SLURM

The job scheduler on the clusters hhemt targets.

### MPI

Message Passing Interface, the standard that lets the processes of one
simulation communicate across nodes.

### OpenMP

A standard for running threads that share memory within one node.

### CUDA, HIP and ROCm

GPU programming platforms. CUDA is NVIDIA's. HIP and
ROCm are AMD's. GPU code compiles against one of them, chosen in configuration.

### QOS

Quality of service, the SLURM setting that attaches priority and limits to
a job.

### GPFS

The IBM parallel file system that some clusters use for shared storage.

### OLCF

The Oak Ridge Leadership Computing Facility, which runs the Frontier
system.

### WSL

Windows Subsystem for Linux, the Linux environment inside Windows, on which
hhemt can run.

### CI

Continuous integration, the automated builds and tests that run on every
change to the repository.
