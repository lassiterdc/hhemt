# Installation

## Install

### Option A (recommended — full conda env from yaml)

The repo ships an `environment.yaml` that pins every runtime dependency including the Snakemake SLURM executor plugins required for HPC `batch_job` orchestration. Use this for production HPC installs.

```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install -e .
```

For exactly-reproducible installs (e.g., debugging cross-machine drift), use `environment-lock.yaml` instead — it pins every transitive dependency.

### Option B (lightweight — pip extras only)

When you control the conda env separately (e.g., shared HPC env, CI), install the toolkit via pip extras matched to your usage:

```bash
conda create -n hhemt python=3.11
conda activate hhemt
pip install -e '.[hpc]'
```

The `[hpc]` extra pulls `snakemake-executor-plugin-slurm` + `snakemake-executor-plugin-slurm-jobstep` (required for sensitivity `batch_job` analyses; see `pyproject.toml` and `workflow.py:2326` for the call site). `kaleido` (required for Plotly→SVG figure export) is now a CORE dependency — no extra needed. The empty `viz-export` extra is retained as a no-op alias for one deprecation cycle.

For purely local non-HPC use (laptop development with `multi_sim_run_method: serial`), neither extra is required:

```bash
pip install -e .
```
