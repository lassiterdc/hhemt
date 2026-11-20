---
type: guide
title: Quickstart and task routing
description: Install the supported environment, enter through CLI or Toolkit, preview execution safely, and find the wiki page for each modeling and development task.
tags: [quickstart, navigation, installation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-701ea48ca27e88217baaad35
    resource: repo://docs/_snippets/install-commands.md
  - id: openwiki-source-2e06aac4578f4a397547ca94
    resource: repo://docs/how-to/installation.md
  - id: openwiki-source-830668a989b5de3f7924699f
    resource: repo://docs/quickstart.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-15bc8d0cbe6b4bd1d7c8f314
    resource: repo://src/hhemt/cli.py
  - id: openwiki-source-b0507628fb9723a5ed011ffa
    resource: repo://src/hhemt/toolkit.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Quickstart and task routing

hhemt orchestrates TRITON-SWMM flood ensembles on local workstations and SLURM systems. Snakemake drives execution; processed results consolidate into hierarchical stores used by reports and portable bundles. Start here for entrypoints, then follow the task map rather than reading every subsystem page.

## Install the supported stack

The package requires Python 3.11 or 3.12. The recommended production/HPC installation is the repository's pinned conda environment, followed by the two dependency-preserving pip steps:

```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install --no-deps "swmm-toolkit==0.15.5" "pyswmm==2.1.0" "swmmio==0.8.2"
pip install -e . --no-deps
```

Do not omit `--no-deps` within this route: pip dependency resolution can displace the conda numerical stack. `PYTHONNOUSERSITE=1` prevents user-site packages from shadowing the environment when that is a concern. The lightweight pip route is reduced-support; `[hpc]` supplies the SLURM executor/jobstep/storage plugins. Snakemake itself is a core dependency. See `docs/how-to/installation.md` for environment constraints and alternatives.

Installing Python tooling does not compile solver binaries. Native execution expects the appropriate solver build before a run; container execution uses its image's binary. See [Solver builds and containers](integrations/solver-builds-and-containers.md).

## Start with two configurations

An analysis minimally takes a system YAML and an analysis YAML. A separate HPC system profile supplies partition capacities, GPU targets, and account information for cluster use. `Toolkit.from_configs` validates both schemas, constructs linked system/analysis objects, and runs preflight by default. Configurations also need real input files and valid identity pins; examples below assume those are already filled, not that two empty YAML files suffice.

```bash
hhemt --help
hhemt run --system-config system.yaml --analysis-config analysis.yaml --dry-run
```

The installed `hhemt` console entrypoint is `hhemt.cli:app`. The run command validates the Snakemake DAG without executing it when `--dry-run` is selected. Preview is not a universal promise of zero writes: configuration materialization and status/report trigger handling depend on the entrypoint. In particular, reprocess dry runs can remove flag/report-shell triggers while sparing destructive figure/store deletion. Review [Reprocessing and retention](workflows/reprocessing-and-retention.md) before previewing rebuilds on valuable existing outputs.

## Use Python when inspecting state interactively

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
tk.system.compile_TRITON_SWMM()  # native coupled build, once per machine/target
preview = tk.run(mode="resume", dry_run=True)
result = tk.run(mode="resume")
status = tk.get_status()
```

Compile the solver matching your enabled model and hardware; the coupled compile call is not the build recipe for every model. `tk.run` is a supported high-level facade. For controls it does not expose, use `tk.analysis.run(...)`, whose fresh/resume control is `from_scratch`, not `mode`. The facade maps `mode="fresh"` to a full-tree fresh start and defaults to resume. Existing work makes a fresh wipe refuse without explicit authorization. Use scoped force-rerun rather than a wipe to redo selected work.

The repository's `docs/quickstart.md` also provides a Norfolk example through `NorfolkIreneExperiment.load(hhemt_sha=...)`. It requires your actual full toolkit commit and can fetch the public case-study data; this page has not fetched it. The example's serial/local configuration requires no HPC. Auto execution follows configuration, not merely the presence of a SLURM environment.

## Choose a task

| Task or question | Wiki entry |
|---|---|
| Understand object ownership and public layers | [Architecture and ownership](architecture/ownership.md) |
| Diagnose flags, logs, sentinels, and persisted completion | [Durable state](architecture/durable-state.md) |
| Fill system/analysis/HPC/report configs and understand validation | [Configuration and validation](concepts/configuration-and-validation.md) |
| Read DataTrees, summaries, time series, and dimensions | [Output data model](concepts/output-data-model.md) |
| Configure optional coupled exchange captures | [Coupling diagnostics](concepts/coupling-diagnostics.md) |
| Prepare weather, GIS, SWMM inputs and coupling boundaries | [Scenario inputs and SWMM](integrations/scenario-inputs-and-swmm.md) |
| Compile native solvers or build/resolve SIFs | [Solver builds and containers](integrations/solver-builds-and-containers.md) |
| Run or resume a normal ensemble | [Ensemble lifecycle](workflows/ensemble-lifecycle.md) |
| Reproduce a published or descriptor-based experiment | [Experiment reproduction](workflows/experiment-reproduction.md) |
| Build parameter/compute sweeps or synthetic cases | [Sensitivity and synthetic experiments](workflows/sensitivity-and-synthetic-experiments.md) |
| Select EDA figures and notebooks | [Exploratory analysis](workflows/exploratory-analysis.md) |
| Select reports, regenerate bundles, or combine experiments | [Reporting and portable bundles](workflows/reporting-and-portable-bundles.md) |
| Rebuild outputs or understand what deletion makes unrecoverable | [Reprocessing and retention](workflows/reprocessing-and-retention.md) |
| Configure SLURM resources, launch methods, and execution locus | [HPC execution](operations/hpc-execution.md) |
| Inspect failure or safely control in-flight work | [Diagnostics and in-flight control](operations/diagnostics-and-in-flight-control.md) |
| Upgrade existing output layouts | [Layout migrations](operations/layout-migrations.md) |
| Understand provenance, RO-Crates, transfers, and publishing | [Provenance and publication](integrations/provenance-and-publication.md) |
| Find supported development and verification recipes | [Testing and development](operations/testing-and-development.md) |

## Evidence boundary

Commands are usage recipes grounded in repository sources, not commands executed during wiki authoring. No installation, solver execution, remote download, rendering, or test run is asserted. See the development page for verification recipes and the snapshot's test-availability limits.
