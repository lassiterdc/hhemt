# Glossary

Acronyms here show their expansion as a tooltip on any page. The [configuration schema](config-schema.md) lists configuration fields, and the [API reference](api.md) lists classes.

## Nomenclature

A matrix is a set of experiments. An experiment is one or more analyses. An analysis is one or more simulations. Every page uses the terms with the meanings in the entries.

### matrix

A set of [experiments](#experiment). The experiments of a matrix differ in the compute configuration, the DEM resolution, the models, or another parameter. The `synth-experiment` command builds the matrix of a synthetic experiment.

### experiment

One or more [analyses](#analysis). The name covers the following objects.

- An **experiment definition** is a directory holding an `experiment.yaml` descriptor. The descriptor points to the system config and the analysis config of the experiment. The descriptor records the input datasets, the toolkit version of the pin, and whether the experiment runs in a container. The schema is `ExperimentConfig`. See [Running an experiment definition](../how-to/running-an-experiment-definition.md).
- A **case-study experiment** is the object that `NorfolkIreneExperiment.load` and `TRITON_SWMM_experiment.from_case_study` return. The object holds a system as `.system` and an analysis as `.analysis`, built from a [case study](#case-study).
- The **experiment root** of a sensitivity analysis is the level above the [members](#member). [The per-experiment tier](output-data-model.md#the-per-experiment-tier) lists the contents of the root.

### analysis

One or more [simulations](#simulation) of weather events against one [system](#system), together with everything produced from them. The products are the processed outputs, the consolidated datasets, and the report. The analysis config, `analysis_config`, describes an analysis. Every analysis is an [event ensemble](#event-ensemble) or a [sensitivity analysis](#sensitivity-analysis). Each analysis has a directory under the directory of the system, named by `analysis_id`. The `analysis_dir` field of the analysis config places the directory elsewhere.

### simulation

One execution of the solver for one [scenario](#scenario) under one [compute configuration](#compute-configuration). A simulation writes raw outputs. Processing turns the raw outputs into the processed outputs of the analysis.

## Objects of an analysis

### system

The modelled place and the solver build. A system holds the inputs that every analysis over that place shares. The inputs are the DEM, the Manning's roughness inputs, the choice of models, and the compiled solver binaries. The system config, `system_config`, describes a system. A system is not the computer that executes the simulations, and the [HPC system config](#hpc-system-config) describes that computer. One system can have many analyses.

### experiment definition

The directory with an `experiment.yaml` descriptor that makes an experiment executable by anyone. The directory names the configs and inputs that the [experiment](#experiment) needs. A [bundle](#bundle) holds the outputs.

### event ensemble

An analysis whose simulations are weather events. Each event is simulated against the same system, and the consolidation combines the results across the whole population of events. An analysis is an event ensemble when `toggle_sensitivity_analysis` is `false`.

### sensitivity analysis

An analysis whose simulations follow the rows of a parameter table that you supply. Each row varies the model or the compute configuration and forms a [member](#member). An analysis is a sensitivity analysis when `toggle_sensitivity_analysis` is `true`. A **sweep** is a sensitivity analysis under another name, and the pages use both words.

### sweep

A synonym for [sensitivity analysis](#sensitivity-analysis).

### member

One row of the parameter table of a sensitivity analysis. A member is an analysis with a directory `members/member_{member_id}/`, scenarios, and consolidated output.

### scenario

One weather event within an analysis. The files are under `sims/{event_id}/`. An event ensemble has one scenario per event. Each member of a sensitivity analysis has scenarios.

### benchmarking

A sensitivity analysis whose axis is the compute configuration. The term also names the [reporting set](#reporting-set) that selects the figures for such an analysis. The figures show wall-clock time, compute cost, and speedup.

### case study

A worked example under `test_data/{case_name}/` in the repository. The `case.yaml` manifest describes the example and names the host of the large input data. The manifest schema is `CaseManifest`. The Norfolk coastal flooding case is the case study of the tutorials.

## Compute and execution

### compute configuration

The machine resources of one simulation. `run_mode` sets the configuration together with `n_mpi_procs`, `n_omp_threads`, `n_gpus` and `n_nodes`.

### `run_mode`

The analysis config field that selects the parallelism of one simulation. The options are `serial`, `openmp`, `mpi`, `hybrid` and `gpu`. The `hybrid` option combines MPI ranks that each use OpenMP threads.

### `multi_sim_run_method`

The analysis config field that selects the dispatch of the simulations of an analysis. The option `local` uses a thread pool on the current machine without SLURM. The option `batch_job` starts Snakemake in a tmux session on the login node and submits one SLURM job per scenario. The option `1_job_many_srun_tasks` executes all simulations as concurrent `srun` tasks inside one SLURM allocation.

### native mode

`execution_environment` has the default value native. Simulation and processing execute on the host with a solver that you compiled.

### container mode

`execution_environment` has the value `container` in container mode. The simulation executable and the processing runners start inside `apptainer exec` on a [SIF](#sif) image, and hhemt resolves the image.

### HPC system config

The configuration file that describes a cluster and no experiment, `hpc_system_config`. Only cluster-bound work needs the file. [HPC-profile setup](../how-to/hpc-profile-setup.md) describes the authoring.

### partition

A SLURM queue with limits. Each partition in the HPC system config is a `PartitionSpec` with caps such as the longest wall time and the most memory for a single job. hhemt rejects a request above a cap before submitting.

### allocation

In SLURM, the resources granted to a job. The word also names the account that a job is charged to, and the `default_account` field of the HPC system config uses that sense.

### driver

The process that orchestrates an experiment. The driver starts the workflow, submits the jobs, and writes a liveness record under `_status/_orchestrator/`. A blocking local driver ends when the call returns. A detached driver outlives the call. The tmux session of `batch_job` and a single batch job are detached drivers.

### sentinel

A small JSON file under the `_status/` directory of an analysis that marks work in flight. `_status/_submitted/{rule_token}.json` records a submitted job. `_status/_orchestrator/{driver_id}.json` records a live [driver](#driver). hhemt checks the files so that each piece of work is submitted once and so that deleting or reprocessing refuses to start over live work. In DEM preparation the word has another meaning. A sentinel elevation is a placeholder value burned into a DEM for buildings or for cells outside the watershed.

### hotstart

A restart of a TRITON simulation from an earlier checkpoint. [Scale and hardware](../explanation/limitations.md#scale-and-hardware) covers a simulation that a walltime limit stops.

### Snakemake

The open-source workflow manager that hhemt uses to execute the steps of an analysis. Snakemake skips completed steps.

### Apptainer

The container runtime on HPC systems. Apptainer starts [SIF](#sif) images.

## Models and inputs

### TRITON

The 2D hydrodynamic flood model that hhemt compiles and executes. TRITON is an external solver and no part of hhemt.

### SWMM

The EPA Storm Water Management Model for stormwater networks. hhemt compiles and executes SWMM, an external solver.

### coupled TRITON-SWMM

The solver build where TRITON and SWMM exchange data during the simulation. `toggle_tritonswmm_model` enables the coupled build independently of the TRITON-only and SWMM-only models.

### DEM

Digital elevation model, the raster of ground elevation that TRITON uses. The system config points to the DEM as `DEM_fullres`. hhemt coarsens the DEM to the target resolution.

### CRS

Coordinate reference system, the definition that maps raster coordinates to places on the ground. The system config takes the horizontal CRS of the DEM as an EPSG code.

### Manning's roughness

The coefficient for the friction of the ground on flowing water. The coefficient is a constant value or a lookup from a land-use raster through a table. `toggle_use_constant_mannings` selects the option.

### NetCDF

A self-describing file format for gridded data. The weather input is a NetCDF file. The [Output data model](output-data-model.md) names the per-scenario output option.

## Outputs and data formats

### consolidation

The combination of the processed outputs of every scenario into one hierarchical dataset per analysis. For a sensitivity analysis the combination also covers every member. The result is the **consolidated** dataset or tree.

### zarr

A chunked array storage format that holds a dataset as a directory of files. hhemt writes the processed and consolidated outputs as Zarr stores.

### DataTree

The `xarray.DataTree` structure for a hierarchy of datasets in one object. The consolidated outputs are DataTrees saved as Zarr.

### CF-1.13

Version 1.13 of the Climate and Forecast conventions, a standard for naming and describing variables with `standard_name`, `long_name`, `units` and `cell_methods`. The consolidated outputs carry the attributes.

### `system_datatree.zarr`

The per-system tier of the output data model. [The per-system tier](output-data-model.md#the-per-system-tier) gives the write timing and the contents of the store.

### `experiment_datatree.zarr` and `analysis_datatree.zarr`

The consolidated trees of the experiment root of a sensitivity analysis and of a single analysis. The [output data model](output-data-model.md) lists every tier.

### reporting set

A named selection of report renderers with an order for the sidebar categories of the report. The choice of a set decides the figures and tables of a report. See [Reporting sets](reporting-sets.md).

### render bundle

A portable zip that `bundle_report_data()` writes. The zip holds the data behind the figures of a report, the configs, and the Snakefile, so that the report can be re-rendered on another machine. A render bundle is no experiment definition.

### EDA

Exploratory data analysis. In hhemt, EDA is the step after consolidation that compares an analysis against itself. The comparison separates a result that belongs to the model from one that belongs to the compute configuration. See [Exploratory analysis](../tutorials/exploratory-analysis.md).

## Publishing and reproducing

### bundle

A reproducible and shareable directory for an experiment or an analysis. The directory holds inputs, outputs, and metadata, and can be published to an open data-sharing repository such as Zenodo or HydroShare. The [render bundle](#render-bundle) and the reprex bundle, described under [reprex](#reprex), are bundles. The directory with an `experiment.yaml` is an [experiment definition](#experiment-definition) and no bundle.

### reprex

A reproducible example. A reprex is a bundle that anyone can fetch by DOI and execute. See [Publishing and fetching](../how-to/publishing.md).

### RO-Crate

Research Object Crate, a packaging convention that describes a set of files and their relations in a `ro-crate-metadata.json` file. hhemt writes a crate in each analysis directory beside the consolidated tree and at the experiment root of a sensitivity analysis. The per-system tree has no crate.

### PROV-O

The W3C ontology for recording where data came from and how the data was produced. The provenance in the RO-Crate uses the ontology.

### JSON-LD

JSON for Linking Data, the serialization of the `ro-crate-metadata.json` file.

### CodeMeta

A metadata vocabulary for describing software. The RO-Crate uses the vocabulary for the description of the software.

### DOI

Digital object identifier, a permanent link to a published item.

### PID

Persistent identifier, the general term for a DOI or any similar permanent identifier. The `case.yaml` of a case study accepts a `doi` or, without one, the `pid` record of the host.

### DataCite

The metadata schema for the registration of dataset DOIs. A deposit supplies the identifier, the creators, the title, the publisher, the publication year, and the resource type.

### FAIR

Findable, accessible, interoperable, and reusable. The [FAIR scope table](fair-scope-table.md) grades each published artifact on the properties.

### OCI

Open Container Initiative, the standard for container image formats.

### SIF

The image file format that Apptainer starts. [Building container images](../how-to/building-container-images.md) covers image addressing.

## Computing and platform terms

### HPC

High-performance computing on a shared cluster of many nodes under a scheduler.

### SLURM

The job scheduler on the clusters that hhemt targets.

### MPI

Message Passing Interface, the standard for communication between the processes of one simulation across nodes.

### OpenMP

A standard for threads that share memory within one node.

### CUDA, HIP, and ROCm

GPU programming platforms. CUDA belongs to NVIDIA. HIP and ROCm belong to AMD. GPU code compiles against one platform, and the configuration chooses the platform.

### QOS

Quality of service, the SLURM setting that attaches priority and limits to a job.

### GPFS

The IBM parallel file system for the shared storage of some clusters.

### OLCF

The Oak Ridge Leadership Computing Facility, a high-performance computing facility at Oak Ridge National Laboratory.

### CI

Continuous integration. The repository runs automated builds and tests on every change.
