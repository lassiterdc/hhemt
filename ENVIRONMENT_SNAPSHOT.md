# Environment Snapshot Documentation

This directory contains versioned snapshots of the TRITON-SWMM_toolkit environment to ensure reproducible installations across different machines and teams.

## Files Included

### 1. `environment-lock.yaml`
**Purpose:** Complete conda environment snapshot with all packages pinned to exact versions

**Contents:** 
- All conda packages (from conda-forge and bioconda channels)
- All pip packages installed in the environment
- Exact build strings for reproducible binary compatibility

**Best For:**
- Sharing with team members using conda
- Ensuring identical environments across different machines
- Long-term version tracking and reproducibility

### 2. `requirements-pinned.txt`
**Purpose:** Python pip requirements with all packages pinned to exact versions

**Contents:**
- All pip-installed packages with specific version numbers
- Can be used as a fallback or for pip-only installations

**Best For:**
- Quick pip installations
- CI/CD pipelines
- Docker containers with pip-based installations

## How to Use These Files

### Option 1: Recreate Environment from `environment.yaml` (RECOMMENDED)

Provision from the spec and apply the two `--no-deps` post-create steps. Do NOT create from the lock — see the warning below.

> [!WARNING]
> **Do NOT create an environment from `environment-lock.yaml`.** A `conda env export`
> lock records *observed state*, not a solvable spec, and conda still runs its `pip:`
> block on create — which re-triggers the very downgrade the pins were meant to prevent
> (pip satisfies `swmmio`'s `pyswmm<2.0` cap by uninstalling the conda `pyswmm 2.0.1`).
> `environment.yaml` is the provisioning path; the lock is an *inspection* snapshot.
> Enforced by `scripts/check_env_lock_consistency.py`.

#### Fresh Installation:
```bash
# Clean slate - remove old environment if it exists
conda deactivate
conda env remove -n hhemt

# Create environment from the SPEC (not the lock)
conda env create -f environment.yaml

# Activate the environment
conda activate hhemt

# Both post-create steps are REQUIRED (see docs/how-to/installation.md)
pip install --no-deps "swmmio==0.8.5"
pip install -e . --no-deps
```

#### Using on HPC systems (with module system):
```bash
# Load your conda/mamba module
module load miniforge  # or conda/mamba module on your system

# Create environment from the SPEC (not the lock)
conda env create -f environment.yaml

# Activate
conda activate hhemt

pip install --no-deps "swmmio==0.8.5"
pip install -e . --no-deps
```

### Option 2: Update Existing Environment

If you already have the environment and just want to update it:

```bash
# Activate your existing environment
conda activate hhemt

# Update to match the SPEC (not the lock — see the warning above)
conda env update -f environment.yaml --prune

# Re-apply the two --no-deps post-create steps
pip install --no-deps "swmmio==0.8.5"
pip install -e . --no-deps
```

### Option 3: Pip-Only Installation

If you prefer or need to use pip only:

```bash
# Create a Python 3.11 environment
conda create -n hhemt python=3.11

# Activate it
conda activate hhemt

# Install from requirements file
pip install -r requirements-pinned.txt
```

## Version Information

**Environment Created:** January 23, 2026
**Python Version:** 3.11.14
**Key Packages:**
- snakemake: 9.15.0
- numpy: 2.3.5
- scipy: 1.17.0
- matplotlib: 3.10.8
- xarray: 2025.12.0
- dask: 2026.1.1
- geopandas: 1.1.2
- netcdf4: 1.7.4
- And many more (see environment-lock.yaml for complete list)

## Troubleshooting

### Issue: "CondaError: Could not solve for environment"
This can occur if your system or channels have conflicts. Try:
```bash
conda env create -f environment.yaml --strict-channel-priority
```

### Issue: "Packages not available on my platform"
The lock file may have been created on Linux. If you're on macOS or Windows, you may need to:
1. Use the original `environment.yaml` instead
2. Let conda resolve the dependencies for your platform:
```bash
conda env create -f environment.yaml
```

### Issue: Channel not found
Ensure you have the correct channels configured:
```bash
conda config --add channels conda-forge
conda config --add channels bioconda
```

## Updating the Environment Snapshot

When you install new packages or update existing ones in your environment, regenerate the lock file:

```bash
# With conda
conda env export -n hhemt > environment-lock.yaml

# With pip
pip freeze > requirements-pinned.txt
```

> [!IMPORTANT]
> A raw `conda env export` re-poisons the lock with three artifacts that
> `scripts/check_env_lock_consistency.py` will reject (run it after every re-export):
>
> - `- hhemt==<version>` in the `pip:` block — the editable project install. It is
>   un-findable on PyPI and aborts `conda env create`. **Delete it.**
> - `- swmmio==0.8.5` in the `pip:` block — its `pyswmm<2.0` cap makes pip downgrade
>   the conda `pyswmm 2.0.1`. **Delete it** (swmmio is installed post-create with
>   `--no-deps`).
> - `prefix: /home/...` — leaks a machine-local path. **Delete it.**
>
> Also drop the `defaults` channel if the export adds it; this project is conda-forge only.

Then commit both files to version control to track environment changes over time.

## Best Practices

1. **Provision from the spec, not the lock:** use `environment.yaml` (plus the two `--no-deps` post-create steps) everywhere — local, HPC, CI, and Docker. `environment-lock.yaml` is an *inspection snapshot* of a known-good env, NOT a portable lockfile: it is single-platform, and creating from it still runs its `pip:` block. For genuine bit-level cross-machine reproducibility, generate a multi-platform `conda-lock.yml`.
2. **Track Versions:** Commit lock files to git to track when dependencies changed
3. **Test After Updates:** Test your code after updating the environment
4. **Document Changes:** When updating, add a note about what changed and why
5. **Keep Original Files:** Keep the original `environment.yaml` as a reference for flexible installations

## For CI/CD Pipelines

In your CI/CD configuration, provision from `environment.yaml` (never the lock) and apply the two `--no-deps` steps:

```yaml
# Example GitHub Actions
- name: Create conda environment
  uses: conda-incubator/setup-miniconda@v3
  with:
    environment-file: environment.yaml
    auto-activate-base: false

- name: Install swmmio + project (both --no-deps)
  run: |
    pip install --no-deps "swmmio==0.8.5"
    pip install -e . --no-deps
```

Or for Docker:

```dockerfile
RUN conda env create -f environment.yaml
RUN conda run -n hhemt pip install --no-deps "swmmio==0.8.5"
RUN conda run -n hhemt pip install -e . --no-deps
RUN echo "conda activate hhemt" >> ~/.bashrc
```

## Questions?

If you have issues recreating the environment, check:
1. Your conda/mamba version is up to date: `conda --version`
2. You have sufficient disk space (several GB may be needed)
3. Your network connection is stable (many packages to download)
4. Channels are properly configured: `conda config --show channels`

For more help, refer to the project documentation or open an issue on GitHub.
