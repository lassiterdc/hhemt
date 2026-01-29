# 1_job_many_srun_tasks Mode Redesign: Dynamic Concurrency

**Date**: 2026-01-29
**Status**: In Progress
**Goal**: Eliminate `hpc_max_simultaneous_sims` dependency for `1_job_many_srun_tasks` mode and enable dynamic concurrency based on SLURM allocation

## Problem Statement

The `_generate_single_job_submission_script()` function had a critical bug (undefined `max_concurrent` variable) that was intentionally introduced because the previous approach of calculating job time as `(n_sims / max_concurrent) × time_per_sim` was fragile.

**Root causes of fragility**:
- Simulation times vary unpredictably (I/O, convergence, network)
- Snakemake scheduling overhead isn't linear
- Resource contention creates cascading delays
- Required pre-estimating max_concurrent upfront

## Solution Architecture

### New Resource Model

**Before** (Fragile):
```
User Config:
  hpc_max_simultaneous_sims = 10
  hpc_time_min_per_sim = 30

Calculated:
  total_nodes = n_nodes_per_sim × max_concurrent
  total_time = (n_sims / max_concurrent) × time_per_sim × 1.2

SBATCH:
  --nodes={total_nodes}
  --ntasks={max_concurrent}  ← UNDEFINED!
  --cpus-per-task={cpus_per_sim}
```

**After** (Robust):
```
User Config:
  hpc_total_nodes = 2 (user specifies directly)
  hpc_total_job_duration_min = 60 (user specifies directly)
  hpc_gpus_per_node = 8 (cluster-specific)

SBATCH:
  --nodes={hpc_total_nodes}
  --exclusive
  --gres=gpu:{hpc_gpus_per_node} (if GPUs needed)
  --time={hpc_total_job_duration_min}
  # NO --ntasks, NO --cpus-per-task!

Inside Allocation:
  TOTAL_CPUS = SLURM_CPUS_ON_NODE × SLURM_JOB_NUM_NODES
  snakemake --cores $TOTAL_CPUS ...
```

### Key Enablers Already in Codebase

1. **`--overlap` in srun** (`run_simulation.py:237,276`): Allows multiple srun to share allocation
2. **`hpc_total_nodes` config**: User already specifies allocation size
3. **`_get_slurm_resource_constraints()`** (`resource_management.py:215`): Dynamically calculates concurrency from SLURM env vars

## Implementation Phases

### Phase 1: SBATCH Script Simplification
**File**: `src/TRITON_SWMM_toolkit/workflow.py`

**Changes**:
- Use `hpc_total_nodes` directly (not calculated from max_concurrent)
- Add `--exclusive` (hardcoded)
- Use `--gres=gpu:{hpc_gpus_per_node}` for GPU allocation
- Remove `--ntasks`, `--cpus-per-task`, `--mem=0`
- Calculate `TOTAL_CPUS` dynamically in bash from SLURM env vars
- Pass `--cores $TOTAL_CPUS` to snakemake command line

### Phase 2: Snakemake Profile Update
**File**: `src/TRITON_SWMM_toolkit/workflow.py`

**Changes**:
- Remove `cores` from single_job profile (passed via CLI)
- Add GPU resources: `total_gpus = hpc_total_nodes × hpc_gpus_per_node`
- Keep `keep-going` and `latency-wait` settings

### Phase 3: Remove Total Resource Calculations
**File**: `src/TRITON_SWMM_toolkit/resource_management.py`

**Changes**:
- Remove `hpc_max_simultaneous_sims` requirement from `_get_simulation_resource_requirements()`
- Remove `total_nodes`, `total_cpus`, `total_gpus`, `total_mem_mb` from return dict
- Find and update all callers (use grep to locate)
- Keep sensitivity analysis logic for finding max per-sim requirements

### Phase 4: Add GPU Config Field
**File**: `src/TRITON_SWMM_toolkit/config.py`

**Changes**:
- Add `hpc_gpus_per_node: Optional[int]` config field
- Update `hpc_max_simultaneous_sims` description to mark as deprecated for 1-job mode

## Benefits

1. **Simpler config**: User specifies nodes + duration, no max_concurrent guessing
2. **No fragile math**: No more `(n_sims / max_concurrent) × time_per_sim` calculations
3. **Dynamic concurrency**: Snakemake adapts to actual SLURM allocation
4. **Cleaner code**: Remove circular dependency on `hpc_max_simultaneous_sims`
5. **Better resource utilization**: Snakemake schedules based on available CPUs

## Testing Strategy

### Local Tests (Desktop)
- **File**: `tests/test_workflow_1job_sbatch_generation.py`
- Verify generated SBATCH script text:
  - Contains `--exclusive`, `--nodes=X`
  - Does NOT contain `--ntasks`, `--cpus-per-task`, `--mem=0`
  - Contains dynamic `TOTAL_CPUS` calculation
  - Contains `--gres=gpu:X` for GPU jobs

### Frontier Tests (Manual)
- **Files**: `tests/test_frontier_03_*.py` or `tests/test_frontier_04_*.py`
- End-to-end workflow execution on ORNL Frontier
- Manual verification of:
  - Job submission success
  - Dynamic CPU detection (64 CPUs/node × nodes)
  - Multiple concurrent simulations
  - No resource allocation errors

## Edge Cases Handled

1. **GPU Allocation**: Use `--gres=gpu:{hpc_gpus_per_node}` (per-node specification)
2. **CPU Detection**: Error if `SLURM_CPUS_ON_NODE` not set (psutil fallback possible in future)
3. **Sensitivity Analysis**: Still finds max per-sim requirements, no total calculations
4. **Backward Compatibility**: Keep `hpc_max_simultaneous_sims` field (deprecated for 1-job mode)

## Configuration Example

```python
additional_analysis_configs=dict(
    # NEW: Direct allocation specification
    hpc_total_nodes=2,
    hpc_total_job_duration_min=60,
    hpc_gpus_per_node=8,  # Frontier-specific

    # Existing per-simulation specs
    n_mpi_procs=1,
    n_omp_threads=1,
    n_gpus=1,
    n_nodes=1,

    # Mode setting
    multi_sim_run_method="1_job_many_srun_tasks",

    # DEPRECATED for 1-job mode (kept for backward compatibility)
    # hpc_max_simultaneous_sims=10,  # Not needed!
)
```

## References

- Original plan: `~/.claude/plans/peaceful-sparking-trinket.md`
- SLURM --gres documentation: `man sbatch` (--gres=gpu:X is per-node)
- Existing --overlap usage: `src/TRITON_SWMM_toolkit/run_simulation.py:237,276`
