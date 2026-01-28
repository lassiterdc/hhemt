# 1-Job Many `srun` Tasks: Goal, Findings, and Implementation Plan

## Goal

Enable Snakemake-based orchestration for HPC systems that require a single big
SLURM allocation, so that **`submit_workflow()` can be invoked from a shell and
submits one batch job** in which **each simulation is launched via `srun`**.
This corresponds to `multi_sim_run_method="1_job_many_srun_tasks"`.

## Findings

- `multi_sim_run_method` currently controls the execution strategy in
  `analysis._select_execution_strategy()`. The `SlurmExecutor` already enforces
  concurrency constraints using SLURM environment variables.
- `run_simulation.prepare_simulation_command()` already builds `srun` launch
  commands when running inside a SLURM job (`analysis.in_slurm`). Per direction,
  this behavior should **remain unchanged** and `srun` should be used for each
  simulation in big-job environments.
- `workflow.py` currently generates Snakemake profiles that submit **many**
  SLURM jobs (executor/cluster mode). This does **not** fit one-big-job systems
  where every rule must run inside a single allocation.
- `submit_workflow()` currently runs Snakemake directly (local or SLURM), but
  does not always submit a batch job. For big-job systems, it should **submit**
  a single job, and Snakemake should run *inside* that allocation.

## Proposed Implementation Plan

## Comprehensive Implementation Plan: 1-Job Many `srun` Tasks

Based on a thorough review of the documentation and codebase, here is a detailed
implementation plan.

---

### Current State Analysis

**Key Findings:**

1. **`workflow.py`**: The `SnakemakeWorkflowBuilder` currently generates two profile modes:
   - `local` - uses `cores` based on system capabilities
   - `slurm` - uses `executor: slurm` which submits **many SLURM jobs** (one per rule)

2. **`execution.py`**: The `SlurmExecutor` already correctly handles concurrent `srun` tasks
   within a single SLURM allocation, using
   `ResourceManager._get_slurm_resource_constraints()` to compute `max_concurrent`.

3. **`run_simulation.py`**: The `prepare_simulation_command()` already uses
   `using_srun = self._analysis.in_slurm`, which builds `srun` commands when inside a SLURM
   allocation. **This must remain unchanged.**

4. **`config.py`**: The `multi_sim_run_method` config already supports
   `"1_job_many_srun_tasks"` as a valid option.

5. **Gap Identified**: When `multi_sim_run_method == "1_job_many_srun_tasks"` and using
   Snakemake workflows, the current implementation submits many SLURM jobs instead of a
   single big job with `srun` tasks inside.

---

### Implementation Steps

#### **Step 1: Add `single_job` Snakemake Profile**

**File**: `src/TRITON_SWMM_toolkit/workflow.py`

Extend `generate_snakemake_config()` to support a new `single_job` mode:

```python
def generate_snakemake_config(self, mode: Literal["local", "slurm", "single_job"]) -> dict:
    # ...existing code...
    
    if mode == "single_job":
        # Single-job mode: behaves like local execution but respects SLURM allocation
        # Get max_concurrent from SLURM environment when inside allocation
        constraints = self.analysis._resource_manager._get_slurm_resource_constraints(verbose=False)
        max_concurrent = int(constraints["max_concurrent"])
        
        config.update({
            "cores": max_concurrent,
            "jobs": max_concurrent,
            "keep-going": True,  # Continue other sims if one fails
            "latency-wait": 30,
        })
```

**Key Points:**
- No `executor: slurm` - behaves like local execution
- `cores`/`jobs` capped to SLURM allocation limits
- Uses `_get_slurm_resource_constraints()` to compute concurrency

---

#### **Step 2: Add Batch Script Generation for 1-Job Mode**

**File**: `src/TRITON_SWMM_toolkit/workflow.py`

Add a new method `_generate_single_job_submission_script()`:

```python
def _generate_single_job_submission_script(self, snakefile_path: Path, **workflow_kwargs) -> Path:
    """Generate SLURM batch script that runs Snakemake with single_job profile."""
    
    script_content = f'''#!/bin/bash
#SBATCH --job-name=triton_workflow
#SBATCH --partition={self.cfg_analysis.hpc_ensemble_partition}
#SBATCH --account={self.cfg_analysis.hpc_account}
#SBATCH --nodes={self.cfg_analysis.n_nodes or 1}
#SBATCH --time={estimated_time}
#SBATCH --output=logs/workflow_%j.out
#SBATCH --error=logs/workflow_%j.err

# Snakemake with single_job profile (runs inside this allocation)
snakemake --profile {config_dir} --snakefile {snakefile_path}
'''
    script_path = self.analysis_paths.analysis_dir / "run_workflow_1job.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path
```

---

#### **Step 3: Modify `submit_workflow()` to Handle 1-Job Mode**

**File**: `src/TRITON_SWMM_toolkit/workflow.py`

Modify `SnakemakeWorkflowBuilder.submit_workflow()`:

```python
def submit_workflow(self, mode: Literal["local", "slurm", "auto"] = "auto", ...):
    # Detect if we should use 1-job mode
    multi_sim_method = self.cfg_analysis.multi_sim_run_method
    
    if multi_sim_method == "1_job_many_srun_tasks":
        # Always submit a batch job for 1-job mode
        return self._submit_single_job_workflow(...)
    
    # ...existing auto/local/slurm logic...
```

Add new method:

```python
def _submit_single_job_workflow(self, snakefile_path: Path, wait_for_completion: bool, verbose: bool, dry_run: bool, **kwargs) -> dict:
    """Submit workflow as a single SLURM batch job."""
    
    # Generate single_job profile
    config = self.generate_snakemake_config(mode="single_job")
    config_dir = self.write_snakemake_config(config, mode="single_job")
    
    # Generate submission script
    script_path = self._generate_single_job_submission_script(snakefile_path, config_dir, ...)
    
    if dry_run:
        return {"success": True, "mode": "single_job", "message": "Dry run - script generated", ...}
    
    # Submit with sbatch
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    job_id = self._parse_job_id(result.stdout)
    
    return {
        "success": result.returncode == 0,
        "mode": "single_job",
        "job_id": job_id,
        "script_path": script_path,
        ...
    }
```

---

#### **Step 4: Apply Same Changes to Sensitivity Analysis**

**File**: `src/TRITON_SWMM_toolkit/workflow.py`

Apply identical changes to `SensitivityAnalysisWorkflowBuilder.submit_workflow()`:
- Check for `multi_sim_run_method == "1_job_many_srun_tasks"`
- Generate `single_job` profile
- Submit via batch script with `sbatch`

---

#### **Step 5: Add Validation Tests**

**New File**: `tests/test_1job_many_srun.py`

```python
def test_single_job_profile_has_correct_cores_jobs():
    """Validate single_job profile respects SLURM allocation limits."""
    ...

def test_submit_workflow_generates_batch_script_for_1job_mode():
    """Validate batch script is generated for 1_job_many_srun_tasks."""
    ...

def test_submit_workflow_uses_sbatch_for_1job_mode():
    """Validate submit_workflow uses sbatch for single-job mode."""
    ...
```

---

### Summary of Files to Modify

| File | Changes |
|------|---------|
| `workflow.py` | Add `single_job` profile, `_generate_single_job_submission_script()`, `_submit_single_job_workflow()`, modify both `submit_workflow()` methods |
| `config.py` | No changes needed (already supports `1_job_many_srun_tasks`) |
| `run_simulation.py` | **No changes** - `using_srun` logic preserved |
| `tests/test_1job_many_srun.py` | New test file for validation |

---

### 1) Add a single-job Snakemake profile

- Add a new Snakemake profile mode (e.g., `single_job`) that does **not** use
  `executor: slurm` or `cluster` submission. It should behave like local
  execution but cap `cores`/`jobs` based on SLURM allocation variables.
- Compute `max_concurrent` as the **minimum** of:
  - `hpc_max_simultaneous_sims`
  - `total_sims` constrained by per-simulation resource requirements
    (based on the **most intensive** configuration when sensitivity analyses
    vary per-simulation settings).
- Use `ResourceManager._get_slurm_resource_constraints()` to compute
  resource limits used in the per-simulation calculation, then set:
  - `cores: max_concurrent`
  - `jobs: max_concurrent`

### 2) Ensure `submit_workflow()` *always* submits a batch job for
`1_job_many_srun_tasks`

- Add a `submit_workflow` pathway that writes a **single SLURM submission
  script** to the analysis directory (e.g., `run_workflow_1job.sh`) and submits
  it with `sbatch`.
- The script should invoke Snakemake with the new `single_job` profile.
- Add a new optional config parameter (required when
  `multi_sim_run_method == "1_job_many_srun_tasks"`) to specify the maximum
  SLURM job time for the one-job run.
- Keep `submit_workflow()` callable from a shell in all modes, but when
  `multi_sim_run_method == "1_job_many_srun_tasks"`, it should always submit a
  batch job (even if the caller is not in a SLURM allocation).
  If batch submission fails, **raise an error** (no fallback).

### 3) Keep `using_srun` logic unchanged

- Maintain `using_srun = self._analysis.in_slurm` in
  `run_simulation.prepare_simulation_command()` as requested.
- Ensure the one-job batch script runs Snakemake inside a SLURM allocation so
  all simulations launch via `srun`.

### 4) Sensitivity analysis parity

- Apply the same `single_job` Snakemake profile and batch submission logic to
  `SensitivityAnalysisWorkflowBuilder.submit_workflow()`.

### 5) Testing/Validation

- Add/update unit tests to validate:
  - the new Snakemake profile is created with the expected `cores/jobs`
  - the batch script is generated for `1_job_many_srun_tasks`
  - `submit_workflow()` uses `sbatch` for single-job mode
