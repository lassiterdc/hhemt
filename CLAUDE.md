# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRITON-SWMM Toolkit orchestrates coupled TRITON (2D hydrodynamic) and SWMM (stormwater management) simulations. It supports single simulations, multi-simulation ensembles, and sensitivity analysis studies across local machines and HPC clusters (UVA, Oak Ridge Frontier).

## Common Commands

```bash
# Development installation
uv sync                          # Using uv (recommended)
pip install -e .                 # Traditional pip

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_PC_01_singlesim.py

# Skip slow tests
pytest -m "not slow" tests/

# Lint and format
ruff check src/
ruff format src/
```

## Architecture

### Three-Layer Hierarchy

```
TRITONSWMM_system (system.py)
├─ Processes DEM, Manning's coefficients
├─ Compiles TRITON-SWMM executable
└─ Contains TRITONSWMM_analysis

TRITONSWMM_analysis (analysis.py)
├─ Orchestrates multi-simulation runs
├─ Manages TRITONSWMM_scenario instances
├─ Generates Snakemake workflows
└─ Selects execution strategy (Serial/Concurrent/SLURM)

TRITONSWMM_scenario (scenario.py)
├─ Single simulation for one weather event
├─ Creates SWMM models (hydrology/hydraulics)
└─ Runs TRITON-SWMM and processes outputs
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Pydantic-based configuration (system_config, analysis_config) |
| `workflow.py` | Dynamic Snakefile generation for parallel execution |
| `execution.py` | Execution strategies: SerialExecutor, LocalConcurrentExecutor, SlurmExecutor |
| `resource_management.py` | CPU/GPU/memory allocation for HPC |
| `sensitivity_analysis.py` | Parameter sweep orchestration with sub-analyses |
| `paths.py` | Dataclasses: SysPaths, AnalysisPaths, ScenarioPaths |
| `log.py` | JSON-persisted logging with LogField[T] pattern |

### Runner Scripts (Snakemake Entry Points)

Snakemake rules invoke these as subprocesses:

| Script | Purpose |
|--------|---------|
| `setup_workflow.py` | System inputs processing and TRITON compilation |
| `run_single_simulation.py` | Standalone simulation (maps to SLURM_ARRAY_TASK_ID) |
| `prepare_scenario_runner.py` | Scenario preparation in subprocess |
| `run_simulation_runner.py` | Simulation execution in subprocess |
| `process_timeseries_runner.py` | Output processing in subprocess |
| `consolidate_workflow.py` | Analysis-level output consolidation |

Exit codes: 0=success, 1=failure, 2=invalid arguments.

## Configuration System

Configuration flows: **YAML → Pydantic → Analysis/Scenario classes**

- `system_config`: DEM paths, TRITON compilation, Manning's coefficients
- `analysis_config`: Simulation parameters, HPC settings, weather data, execution mode

### Toggle-Based Validation

Many fields are conditionally required based on boolean toggles:
```python
toggle_use_constant_mannings: bool
constant_mannings: Optional[float]    # Required if toggle=True
landuse_lookup_file: Optional[Path]   # Required if toggle=False
```

### Critical Configuration Fields

| Field | Impact |
|-------|--------|
| `multi_sim_run_method` | Controls execution: `local`, `batch_job`, `1_job_many_srun_tasks` |
| `run_mode` | CPU/GPU config: `serial`, `openmp`, `mpi`, `gpu`, `hybrid` |
| `hpc_max_simultaneous_sims` | **Required** for `1_job_many_srun_tasks` mode (no default) |
| `hpc_sbatch_time_upper_limit_min` | Optional cap on SBATCH runtime |

## HPC & SLURM Integration

### SLURM Detection Logic

```python
in_slurm = ("SLURM_JOB_ID" in os.environ) or (multi_sim_run_method == "1_job_many_srun_tasks")
```

When `in_slurm=True`, simulations launch via `srun` (not direct execution).

### Execution Modes

| Mode | Behavior |
|------|----------|
| `local` | Serial or ThreadPoolExecutor on local machine |
| `batch_job` | SLURM job array (one job per simulation) |
| `1_job_many_srun_tasks` | Single SBATCH with multiple srun invocations (active development) |

### Key Environment Variables

- `SLURM_JOB_ID` - Indicates running in SLURM context
- `SLURM_ARRAY_TASK_ID` - Maps to event_iloc in array jobs
- Module loading via `additional_modules_needed_to_run_TRITON_SWMM_on_hpc`

### 1-Job-Many-srun-Tasks Mode (Current Focus)

Single SLURM allocation runs all simulations:
- Snakemake `cores` = total CPUs in allocation (not max_concurrent)
- Each simulation launches via `srun` inside the allocation
- GPU resources specified via Snakemake resource limits
- See `docs/one_job_many_srun_tasks_plan.md` for implementation details

## Workflow Phases

Snakemake workflows follow three phases:
1. **Setup**: System inputs, compilation
2. **Simulation**: Parallel scenario execution via wildcards
3. **Processing**: Output consolidation

## Testing

### Platform-Organized Tests

- `test_PC_*.py` - Local machine tests
- `test_UVA_*.py` - UVA HPC cluster tests
- `test_frontier_*.py` - Oak Ridge Frontier tests

Tests auto-skip based on platform detection.

### Test Fixtures

Fixtures use `GetTS_TestCases` from `examples.py`:
```python
@pytest.fixture
def norfolk_multi_sim_analysis():
    case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
    return case.system.analysis
```

Use `start_from_scratch=False` for cached fixtures (faster iteration).

### Platform Detection Utilities (`tests/utils_for_testing.py`)

```python
uses_slurm()      # True if SLURM_JOB_ID in environment
on_frontier()     # True if hostname contains "frontier"
on_UVA_HPC()      # True if hostname contains "virginia"
```

Assertion helpers: `assert_scenarios_setup()`, `assert_scenarios_run()`, `assert_timeseries_processed()`

## Code Style

- Line length: 120 characters
- Linter: ruff (E, W, F, I, B, UP rules)
- Python: ≥3.10, target 3.12+
- Configuration models inherit from `cfgBaseModel`
- Use `Literal` types for enumerated options

## Gotchas

1. **`hpc_max_simultaneous_sims` has no default** - Must be explicitly set for `1_job_many_srun_tasks` mode

2. **Sensitivity analysis GPU constraint** - Cannot mix GPU and non-GPU modes in single sensitivity analysis

3. **SLURM detection includes config check** - `in_slurm` is True when `multi_sim_run_method == "1_job_many_srun_tasks"` even without `SLURM_JOB_ID`

4. **Runner scripts use argparse** - Each has specific CLI flags; check docstrings for usage

## Specialized Agent Documentation

The `.claude/agents/` directory contains detailed guidance for specific subsystems:
- `pydantic-config-specialist.md` - Configuration validation patterns
- `snakemake-workflow.md` - Workflow generation and DAG structure
- `hpc-slurm-integration.md` - SLURM execution modes and cluster configs
- `output-processing.md` - SWMM/TRITON output parsing
- `sensitivity-analysis.md` - Parameter sweep orchestration
- `triton-test-suite.md` - Testing patterns and fixtures
- `triton-debugger.md` - Debugging workflow failures
