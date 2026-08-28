# Installation

## Before you start

--8<-- "platform-support.md"

You also need `conda` (Miniforge or Miniconda).

## Install

### Option A (recommended): full conda env from yaml

The repo ships an `environment.yaml` that pins every runtime dependency including the Snakemake SLURM executor plugins required for HPC `batch_job` orchestration. Use this for production HPC installs.

--8<-- "install-commands.md"

Both `--no-deps` flags are required, not optional.

??? note "When `PYTHONNOUSERSITE=1` is needed, and why it is not above"
    `environment.yaml` documents two ways to build the environment. The commands
    above are its *direct-create* route, which does not set `PYTHONNOUSERSITE`. Its
    other route builds a bare environment first
    (`conda create --no-default-packages`) and updates into it, and that one does set
    `export PYTHONNOUSERSITE=1` before the update.

    The variable stops Python from adding your `~/.local` packages to `sys.path`. Set
    it if you keep packages there and want to be certain the environment you just
    built is the one being used. It is also load-bearing at run time on a cluster,
    where a stray `~/.local` package can otherwise shadow the environment inside a
    batch job.

??? note "Why `--no-deps` is required"
    `--no-deps` prevents `pip install -U` from touching the conda graph at all. It
    now guards only against PyPI wheels displacing conda's `numpy`/`pandas`:
    swmmio's real runtime dependencies are declared in `environment.yaml`'s conda
    section, and `pyproject.toml` does pin `pyswmm` and `swmm-toolkit`, but pinning
    them says nothing about whether pip may replace conda's `numpy`/`pandas`.

    The original hazard was narrower and no longer applies at the pinned version.
    `swmmio 0.8.5` declared `pyswmm<2.0` and `numpy<2.0`, so a dependency-resolving
    install downgraded the conda-installed `pyswmm 2.0.1` to `1.5.1` and broke
    `prepare_scenario`'s SWMM-runoff step upstream of every render. At the pinned
    `0.8.2` that risk is gone: it declares no `pyswmm` requirement at all, and its
    numpy/pandas entries are floors rather than caps.

    `scripts/check_env_lock_consistency.py` enforces this invariant in CI.

??? note "What `environment-lock.yaml` is, and why it is not a lockfile"
    `environment-lock.yaml` is a `conda env export` snapshot, useful for
    *inspecting* the exact versions of a known-good env. It is **not** a portable
    lockfile: it is single-platform, and recreating an env from it still runs its
    `pip:` block. If you use it, apply the same two `--no-deps` post-create steps
    above. For genuine bit-level cross-machine reproducibility, generate a
    multi-platform `conda-lock.yml` instead.

### Option B (lightweight): pip extras only

> **Option B is a reduced-support path, and SWMM is not the reason.** `pyproject.toml`
> pins `swmm-toolkit` and `pyswmm` directly, so a pip-only environment resolves a
> stack that passes `prepare_scenario`'s runtime validation guard. What Option B
> does not give you is the rest of what `environment.yaml` pins: the HPC stack and
> the exact conda-resolved graph Option A reproduces. If the guard does refuse a
> stack you assembled yourself, it names the offending versions; the override is
> `HHEMT_ALLOW_UNVALIDATED_SWMM_STACK=1`, at your own risk. Use Option A for
> production HPC runs.

When you control the conda env separately (e.g., shared HPC env, CI), install the toolkit via pip extras matched to your usage:

```bash
conda create -n hhemt python=3.11
conda activate hhemt
pip install -e '.[hpc]'
```

The `[hpc]` extra pulls `snakemake-executor-plugin-slurm` and
`snakemake-executor-plugin-slurm-jobstep`, required for sensitivity `batch_job`
analyses.

??? note "Where the SLURM plugins are used, and what happened to `viz-export`"
    The extra is declared in `pyproject.toml`. The call sites are
    `SnakemakeWorkflowBuilder._run_snakemake_slurm_detached` and
    `._validate_batch_job_dry_run` in `src/hhemt/workflow.py`, both of which pass
    `--executor slurm`.

    `kaleido`, required for Plotly to SVG figure export, is now a **core**
    dependency with no extra needed. The empty `viz-export` extra is retained as a
    no-op alias for one deprecation cycle.

For purely local non-HPC use (laptop development with `multi_sim_run_method: serial`), neither extra is required:

```bash
pip install -e .
```
