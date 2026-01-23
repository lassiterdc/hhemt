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

### Option 1: Recreate Environment from Conda Lock File (RECOMMENDED)

This is the most reliable method as it captures all dependencies exactly as they were installed.

#### Fresh Installation:
```bash
# Clean slate - remove old environment if it exists
conda deactivate
conda env remove -n triton_swmm_toolkit

# Create environment from lock file
conda env create -f environment-lock.yaml

# Activate the environment
conda activate triton_swmm_toolkit
```

#### Using on HPC systems (with module system):
```bash
# Load your conda/mamba module
module load miniforge  # or conda/mamba module on your system

# Create environment from lock file
conda env create -f environment-lock.yaml

# Activate
conda activate triton_swmm_toolkit
```

### Option 2: Update Existing Environment

If you already have the environment and just want to update it:

```bash
# Activate your existing environment
conda activate triton_swmm_toolkit

# Update to match the lock file
conda env update -f environment-lock.yaml --prune
```

### Option 3: Pip-Only Installation

If you prefer or need to use pip only:

```bash
# Create a Python 3.11 environment
conda create -n triton_swmm_toolkit python=3.11

# Activate it
conda activate triton_swmm_toolkit

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
conda env create -f environment-lock.yaml --strict-channel-priority
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
conda env export -n triton_swmm_toolkit > environment-lock.yaml

# With pip
pip freeze > requirements-pinned.txt
```

Then commit both files to version control to track environment changes over time.

## Best Practices

1. **Use Lock Files in Production:** Always use `environment-lock.yaml` for reproducible deployments
2. **Track Versions:** Commit lock files to git to track when dependencies changed
3. **Test After Updates:** Test your code after updating the environment
4. **Document Changes:** When updating, add a note about what changed and why
5. **Keep Original Files:** Keep the original `environment.yaml` as a reference for flexible installations

## For CI/CD Pipelines

In your CI/CD configuration, use the lock file:

```yaml
# Example GitHub Actions
- name: Create conda environment
  uses: conda-incubator/setup-miniconda@v2
  with:
    environment-file: environment-lock.yaml
    auto-activate-base: false
```

Or for Docker:

```dockerfile
RUN conda env create -f environment-lock.yaml
RUN echo "conda activate triton_swmm_toolkit" >> ~/.bashrc
```

## Questions?

If you have issues recreating the environment, check:
1. Your conda/mamba version is up to date: `conda --version`
2. You have sufficient disk space (several GB may be needed)
3. Your network connection is stable (many packages to download)
4. Channels are properly configured: `conda config --show channels`

For more help, refer to the project documentation or open an issue on GitHub.
