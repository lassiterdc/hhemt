# How-to guides

| Guide | Contents |
|---|---|
| [Installation](installation.md) | The conda and pip routes to a local or HPC install |
| [HPC-profile setup](hpc-profile-setup.md) | The `hpc_system_config` file that describes one cluster |
| [Config filling](config-filling.md) | The order for filling the system and analysis config YAMLs |
| [Compile the solver](compiling-the-solver.md) | The one-time TRITON-SWMM build for each machine and GPU target, and the cases that call for a rebuild |
| [Building container images](building-container-images.md) | The Apptainer images for container mode, built with `hhemt build-sifs` |
| [Running an experiment definition](running-an-experiment-definition.md) | The `hhemt run-experiment` command and the `experiment.yaml` definition that the command executes |
| [In-flight operations](in-flight-operations.md) | The commands that monitor, resume, and manage a running analysis |
| [Diagnosing a failed analysis](diagnosing-a-failed-run.md) | The symptom index, the status ladder, and the log reading for a failed or incomplete analysis |
| [Forcing and suppressing re-runs](forcing-reruns.md) | The options that force or suppress a rebuild |
| [Combining experiments](combining-experiments.md) | The merge of render bundles into one cross-experiment report |
| [Publishing and fetching](publishing.md) | The deposit of a dataset to Zenodo or HydroShare, and the fetch of case-study inputs by DOI or PID |
| [Reprex round-trip](reprex-roundtrip.md) | The emission of a reproducible-example bundle and the check that runs the bundle on another cluster |
| [DOI round-trip for operators](doi-roundtrip-e2e.md) | The DOI minting for a reprex bundle and the fetch and run on another machine |
| [Running a synthetic compute-sensitivity experiment](synthetic-compute-sensitivity-experiment.md) | A sweep of a generated synthetic experiment that measures compute sensitivity |
| [Run the exploratory analysis](running-eda.md) | The EDA notebook, the standalone figures, and the verdicts that the report reads |
