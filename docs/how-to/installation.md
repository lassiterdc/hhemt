# Installation

## Install

### Option A (recommended — full conda env from yaml)

The repo ships an `environment.yaml` that pins every runtime dependency including the Snakemake SLURM executor plugins required for HPC `batch_job` orchestration. Use this for production HPC installs.

```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install --no-deps "swmmio==0.8.5"
pip install -e . --no-deps
```

Both `--no-deps` flags are required, not optional. `swmmio 0.8.5` declares `pyswmm<2.0` and `numpy<2.0` in its metadata; a dependency-resolving install downgrades the conda-installed `pyswmm 2.0.1` to `1.5.1`, which breaks `prepare_scenario`'s SWMM-runoff step upstream of every render. hhemt uses only `swmmio.Model` (`.inp` parsing), so the cap is not load-bearing and `--no-deps` is safe; swmmio's real runtime dependencies are declared in `environment.yaml`'s conda section. `pip install -e . --no-deps` for the same reason (`pyproject.toml` leaves `pyswmm` unpinned). `scripts/check_env_lock_consistency.py` enforces this invariant in CI.

`environment-lock.yaml` is a `conda env export` snapshot, useful for *inspecting* the exact versions of a known-good env. It is **not** a portable lockfile: it is single-platform, and recreating an env from it still runs its `pip:` block. If you use it, apply the same two `--no-deps` post-create steps above. For genuine bit-level cross-machine reproducibility, generate a multi-platform `conda-lock.yml` instead.

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
