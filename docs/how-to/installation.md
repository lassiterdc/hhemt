# Installation

## Goal

Install hhemt with the pinned dependencies into a conda environment on a workstation or a cluster.

## Prerequisites

--8<-- "platform-support.md"

Install `conda` from Miniforge or Miniconda first.

## Key references

- [Environment constraints](../explanation/limitations.md#environment-constraints) gives the reasoning for the recommended route.
- [Compile the solver](compiling-the-solver.md) follows the install.

## Install

### Option A, the recommended full conda environment

The repository ships an `environment.yaml` that pins every runtime dependency, including the Snakemake SLURM executor plugins that HPC `batch_job` orchestration requires. Use Option A for production HPC installs.

--8<-- "install-commands.md"

Both `--no-deps` flags are required.

??? note "Set PYTHONNOUSERSITE=1 in the environment build"
    `environment.yaml` documents the routes that build the environment. The commands are the direct-create route, and that route sets no `PYTHONNOUSERSITE`. The bare-environment route builds a bare environment first with `conda create --no-default-packages`, updates into that environment, and sets `export PYTHONNOUSERSITE=1` before the update.

    The variable stops Python from adding your `~/.local` packages to `sys.path`. Set the variable when you keep packages there and want certainty that the environment you built is the environment in use. The variable also matters on a cluster at run time, where a stray `~/.local` package can shadow the environment inside a batch job.

??? note "Reason for the --no-deps requirement"
    `--no-deps` stops `pip install -U` from touching the conda dependency graph. The flag guards against PyPI wheels displacing the conda `numpy` and `pandas`. The conda section of `environment.yaml` declares the real runtime dependencies of swmmio, and `pyproject.toml` pins `pyswmm` and `swmm-toolkit`, but a pin leaves open whether pip may replace the conda `numpy` and `pandas`.

    `scripts/check_env_lock_consistency.py` enforces this invariant in CI.

??? note "environment-lock.yaml is a snapshot and no lockfile"
    `environment-lock.yaml` is a `conda env export` snapshot, useful for inspecting the exact versions of a known-good environment. The file is a single-platform snapshot and no portable lockfile, and recreating an environment from the file still runs the `pip:` block of the file. When you use the file, apply the same `--no-deps` post-create steps. Generate a multi-platform `conda-lock.yml` for bit-level reproducibility across machines.

### Option B, the lightweight pip extras

!!! warning "Option B is a reduced-support path"
    `pyproject.toml` pins `swmm-toolkit` and `pyswmm` directly, so a pip-only environment passes the SWMM guard. [Environment constraints](../explanation/limitations.md#environment-constraints) lists the missing support of Option B and the use of the guard override. Use Option A for production HPC installs.

A conda environment that you control separately, such as a shared HPC environment or a CI job, takes the toolkit through pip extras that match the usage.

```bash
conda create -n hhemt python=3.11
conda activate hhemt
pip install -e '.[hpc]'
```

The `[hpc]` extra pulls `snakemake-executor-plugin-slurm` and `snakemake-executor-plugin-slurm-jobstep` for sensitivity `batch_job` analyses.

??? note "SLURM plugin call sites and the kaleido dependency"
    `pyproject.toml` declares the extra. The call sites are `SnakemakeWorkflowBuilder._run_snakemake_slurm_detached` and `._validate_batch_job_dry_run` in `src/hhemt/workflow.py`, and both pass `--executor slurm`.

    `kaleido`, required for Plotly to SVG figure export, is a **core** dependency and needs no extra.

Purely local use without HPC, such as local development with `multi_sim_run_method: serial`, needs neither extra.

```bash
pip install -e .
```
