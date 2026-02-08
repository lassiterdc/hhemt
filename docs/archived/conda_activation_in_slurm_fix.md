# Conda Activation in SLURM Batch Scripts - Fix Documentation

## Problem

SLURM batch jobs on Frontier were failing with:
```
/var/spool/slurmd/job4079277/slurm_script: line 24: snakemake: command not found
```

Despite the script containing:
```bash
module load miniforge3/23.11.0-0
conda activate triton_swmm_toolkit
```

## Root Cause

The `conda activate` command requires conda's shell integration to be initialized, which is not automatically available in **non-interactive shell contexts** like SLURM batch scripts.

### Why This Happens

1. **Interactive shells** (login sessions):
   - Source `~/.bashrc` which may contain `conda init bash` output
   - Module systems may set up conda shell functions
   - `conda activate` works automatically

2. **Non-interactive shells** (SLURM batch scripts):
   - Do NOT source `~/.bashrc` by default
   - `module load miniforge3` sets environment variables (`CONDA_EXE`, `CONDA_PREFIX`) but does NOT initialize shell functions
   - `conda activate` is a shell function that hasn't been defined yet
   - Result: `bash: conda: command not found`

## Solution Implemented

The fix sources conda's shell integration script before attempting activation:

```bash
# Initialize conda for non-interactive shell (required in SLURM batch scripts)
if [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
elif [ -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
else
    echo "WARNING: Could not find conda.sh for initialization"
fi

# Now conda activate will work
conda activate triton_swmm_toolkit
```

### How It Works

1. **After `module load miniforge3`**, environment variables are set:
   - `CONDA_EXE=/path/to/miniforge3/bin/conda`
   - `CONDA_PREFIX=/path/to/miniforge3`

2. **We locate `conda.sh`** using these variables:
   - Try `${CONDA_PREFIX}/../etc/profile.d/conda.sh` (works when module sets `CONDA_PREFIX` to base env)
   - Try `${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh` (derives path from `CONDA_EXE`)

3. **Sourcing `conda.sh`**:
   - Defines the `conda` shell function
   - Enables `conda activate` and `conda deactivate` commands
   - Sets up environment manipulation hooks

4. **Now `conda activate` works** as a proper shell function

## Alternative Solutions (Not Implemented)

### Option 1: Use `source activate` (Legacy Method)
```bash
module load miniforge3/23.11.0-0
source activate triton_swmm_toolkit
```

**Pros:**
- Simple one-liner
- Works without sourcing conda.sh

**Cons:**
- Deprecated since conda 4.4
- May not work in future conda versions
- conda team discourages this approach

### Option 2: Direct Python Executable Path
```bash
module load miniforge3/23.11.0-0
/path/to/envs/triton_swmm_toolkit/bin/python -m snakemake ...
```

**Pros:**
- Bypasses conda activation entirely
- Most direct approach

**Cons:**
- Requires hardcoding environment paths
- Doesn't set up full conda environment (PATH, env vars)
- May miss environment-level configurations

### Option 3: Conda Init in Script (Verbose)
```bash
eval "$(${CONDA_EXE} shell.bash hook)"
conda activate triton_swmm_toolkit
```

**Pros:**
- Uses conda's official initialization hook
- Portable across conda installations

**Cons:**
- Slower (invokes conda Python process)
- More verbose
- Overkill when module already sets env vars

### Option 4: Source .bashrc (Not Recommended)
```bash
source ~/.bashrc
conda activate triton_swmm_toolkit
```

**Pros:**
- May work if user ran `conda init bash`

**Cons:**
- Assumes user's .bashrc is properly configured
- May execute unintended commands from .bashrc
- Not portable across users/systems
- SLURM best practices discourage this

## Implementation Location

File: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/src/TRITON_SWMM_toolkit/workflow.py`

Function: `SnakemakeWorkflowBuilder._generate_single_job_submission_script()`

Lines: 439-451 (conda initialization logic)
Lines: 486 (insertion into script template)

## Testing Verification

To verify the fix works on Frontier:

1. **Check generated script**:
   ```bash
   cat <analysis_dir>/run_workflow_1job.sh
   ```
   Should contain the conda initialization block before `conda activate`

2. **Submit job**:
   ```bash
   sbatch <analysis_dir>/run_workflow_1job.sh
   ```

3. **Check output log**:
   ```bash
   tail -f <analysis_dir>/logs/_slurm_logs/workflow_*.out
   ```
   Should NOT show "conda: command not found"
   Should show Snakemake output instead

4. **Verify environment activation**:
   The log should show commands running in the correct conda environment,
   with access to snakemake, Python packages, etc.

## Related Documentation

- Conda official docs: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
- Conda initialization: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
- SLURM batch script best practices: https://slurm.schedmd.com/sbatch.html

## Cluster-Specific Notes

### Frontier (ORNL)
- Module: `miniforge3/23.11.0-0`
- Sets `CONDA_EXE` and `CONDA_PREFIX` correctly
- Requires explicit conda.sh sourcing for batch scripts
- Fix verified working on Frontier

### UVA Clusters
- Similar behavior expected with Anaconda/Miniconda modules
- May use different module names (`anaconda3`, `miniconda3`)
- Same fix should apply

### General HPC Clusters
- If `module load <conda-module>` sets `CONDA_EXE` or `CONDA_PREFIX`, this fix will work
- If module system already sources conda.sh, fix is harmless (conda.sh is idempotent)
- Fallback warning helps diagnose misconfigured modules

## Future Enhancements

Potential improvements for robustness:

1. **Add configuration option** to specify conda initialization method:
   ```yaml
   conda_activation_method: "auto"  # auto, source_conda_sh, source_activate, eval_hook
   ```

2. **Detect conda installation type** (Anaconda vs Miniforge vs Mamba):
   ```python
   if "mamba" in str(conda_exe):
       # Use mamba-specific initialization
   ```

3. **Validate environment exists** before submission:
   ```bash
   conda env list | grep -q triton_swmm_toolkit || exit 1
   ```

4. **Support custom environment names** via config:
   ```yaml
   conda_environment_name: "triton_swmm_toolkit"
   ```

5. **Add dry-run validation** that checks if snakemake is accessible:
   ```bash
   which snakemake || { echo "ERROR: snakemake not found"; exit 1; }
   ```
