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

## Development Philosophy

### Backward Compatibility

**Backward compatibility is NOT a priority for this project.**

Rationale:
- Single developer codebase
- Better to have clean code than maintain deprecated APIs
- Refactoring should remove old patterns, not preserve them
- Git history provides access to old implementations if needed

When refactoring:
- ❌ Don't add deprecation warnings
- ❌ Don't keep old APIs "for compatibility"
- ❌ Don't create compatibility shims or aliases
- ✅ Do update all usage sites immediately
- ✅ Do delete obsolete code completely
- ✅ Do use git history if old patterns are needed later

Exception: Configuration file formats should maintain backward compatibility
where practical, since they may be versioned separately from code.

Example:
```python
# BAD - Keeping deprecated API
class OldAPI:
    @classmethod
    def old_method(cls):
        warnings.warn("Use new_method instead", DeprecationWarning)
        return cls.new_method()

# GOOD - Clean replacement
class NewAPI:
    @classmethod
    def improved_method(cls):
        # New implementation
        pass
# Old API completely removed, all call sites updated
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
| `prepare_scenario_runner.py` | Scenario preparation in subprocess |
| `run_simulation_runner.py` | Simulation execution in subprocess |
| `process_timeseries_runner.py` | Output processing in subprocess |
| `consolidate_workflow.py` | Analysis-level output consolidation |

Exit codes: 0=success, 1=failure, 2=invalid arguments.

### Runner Script Architecture

**IMPORTANT**: Runner scripts execute TRITON directly, not recursively.

Each runner script should use `prepare_simulation_command()` to get the actual TRITON-SWMM executable command:

```python
# ✅ CORRECT (in runner scripts):
simprep_result = run.prepare_simulation_command(pickup_where_leftoff=args.pickup_where_leftoff)
cmd, env, logfile, tstep = simprep_result
proc = subprocess.Popen(cmd, env={**os.environ, **env}, ...)  # Executes TRITON directly

# ❌ WRONG (causes recursive fork bomb):
launcher, finalize = run._create_subprocess_sim_run_launcher(...)  # Spawns another runner!
```

**Method Usage:**
- `prepare_simulation_command()`: Use in runner scripts to get TRITON executable command
- `_create_subprocess_sim_run_launcher()`: Use in analysis/executor classes for concurrent execution (spawns runner subprocess)

## Multi-Model Integration

The toolkit supports **concurrent execution** of three model types within a single analysis:

### Model Types

| Model Type | Description | Use Case |
|------------|-------------|----------|
| **TRITON-only** | 2D hydrodynamic (no SWMM coupling) | Pure surface water modeling, coastal flooding |
| **TRITON-SWMM** | Coupled 2D surface + 1D drainage | Urban flooding with stormwater infrastructure |
| **SWMM-only** | Standalone EPA SWMM | Stormwater network analysis without surface routing |

### Configuration Toggles

Enable model types via `system_config.yaml`:

```yaml
toggle_triton_model: true      # Enable TRITON-only
toggle_tritonswmm_model: true  # Enable TRITON-SWMM coupled
toggle_swmm_model: true        # Enable standalone SWMM
```

**Key behaviors:**
- Models run **concurrently** via separate Snakemake rules
- Each model has its own compilation, executable, and output directories
- Resource allocation: SWMM limited to 4 CPUs (no GPU), TRITON/TRITON-SWMM use configured resources

### Directory Structure (Per Scenario)

```
{scenario_dir}/
├── logs/                      # Centralized logs
│   ├── run_triton.log
│   ├── run_tritonswmm.log
│   └── run_swmm.log
├── out_triton/                # TRITON-only outputs
├── out_tritonswmm/            # Coupled model outputs
├── out_swmm/                  # SWMM-only outputs
├── TRITON.cfg                 # TRITON-only config (inp_filename commented)
├── TRITONSWMM.cfg             # Coupled model config
└── swmm_full.inp              # SWMM input
```

### Compilation

Setup workflow compiles enabled models:

```bash
python -m TRITON_SWMM_toolkit.setup_workflow \
    --system-config system.yaml \
    --analysis-config analysis.yaml \
    --compile-tritonswmm \    # Compile coupled model
    --compile-triton-only \   # Compile TRITON-only
    --compile-swmm            # Compile standalone SWMM
```

**Build directories:**
- TRITON-SWMM: `build_tritonswmm_cpu/`, `build_tritonswmm_gpu/`
- TRITON-only: `build_triton_cpu/`, `build_triton_gpu/` (CMake flag: `-DTRITON_ENABLE_SWMM=OFF`)
- SWMM: `swmm_build/` (standalone EPA SWMM executable)

### Workflow Rules

Snakemake generates **separate rules per model type**:

```python
rule run_triton:          # TRITON-only simulation
    threads: {cpus}
    resources: gpus={gpus}

rule run_tritonswmm:      # Coupled simulation
    threads: {cpus}
    resources: gpus={gpus}

rule run_swmm:            # SWMM-only simulation
    threads: 4            # CPU-only, limited threads
```

Processing rules similarly split: `process_triton`, `process_tritonswmm`, `process_swmm`

### Status Tracking

`analysis.df_status` includes `model_types_enabled` column showing which models are active:

```python
df_status["model_types_enabled"]  # e.g., "triton,tritonswmm,swmm"
```

For multi-model workflows, all enabled models run for each scenario, with outputs in separate directories.

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

Snakemake workflows follow five distinct phases with separate rules:

1. **Setup** (`rule setup`): System inputs processing and TRITON compilation
   - Runner: `setup_workflow.py`
   - Resources: 1 CPU, minimal memory

2. **Scenario Preparation** (`rule prepare_scenario`): SWMM model generation
   - Runner: `prepare_scenario_runner.py`
   - Resources: 1 CPU (lightweight file I/O)
   - Runs in parallel for all simulations

3. **Simulation Execution** (`rule run_simulation`): TRITON-SWMM runs
   - Runner: `run_simulation_runner.py`
   - Resources: Multi-CPU/GPU as configured (resource-intensive)
   - Depends on scenario preparation completion

4. **Output Processing** (`rule process_outputs`): Timeseries extraction and compression
   - Runner: `process_timeseries_runner.py`
   - Resources: 2 CPUs for parallel compression (I/O bound)
   - Depends on simulation completion

5. **Consolidation** (`rule consolidate`): Analysis-level output aggregation
   - Runner: `consolidate_workflow.py`
   - Resources: 1 CPU
   - Depends on all output processing completion

This separation allows:
- **Checkpoint recovery**: Restart from any phase if failure occurs
- **Resource optimization**: Each phase gets appropriate CPU/GPU/memory allocation
- **Clear dependencies**: Snakemake DAG explicitly shows workflow structure

## Conda Environment Architecture

The toolkit uses **two separate conda environments** for clean separation of concerns:

### Primary Environment (`environment.yaml`)
- **Purpose**: Orchestration layer (Snakemake, workflow management, development tools)
- **Contains**: Snakemake, Snakemake plugins (SLURM executor), Typer CLI, testing/linting tools
- **Location**: Root directory
- **When to update**: When adding orchestration features or new CLI commands

### Task Environment (`workflow/envs/triton_swmm.yaml`)
- **Purpose**: Simulation execution layer (individual runner scripts)
- **Contains**: SWMM, scientific Python stack (scipy, dask, zarr), geospatial tools (geopandas, cartopy), data I/O
- **Location**: `workflow/envs/`
- **When to update**: When runner scripts need new dependencies (e.g., new output formats, scientific libraries)

**Why this split?**
1. **Lightweight execution**: Snakemake doesn't need simulation dependencies; runner scripts don't need Snakemake
2. **HPC efficiency**: Smaller task environment = faster conda solve times on clusters
3. **Clean separation**: Orchestration (Snakemake rules) vs. execution (runner scripts) are isolated
4. **Caching**: Snakemake can cache and reuse the lighter task environment across jobs

**How it works:**
- Main environment (primary) is activated on login
- Snakemake reads Snakefile and generates rules
- Each rule's `conda:` directive specifies `workflow/envs/triton_swmm.yaml`
- Snakemake creates/caches the task environment and runs rules within it

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
- `swmm-model-generation.md` - SWMM .inp generation and hydrology/hydraulics split
- `triton-test-suite.md` - Testing patterns and fixtures
- `triton-debugger.md` - Debugging workflow failures

## Maintaining This Documentation

### When to Update CLAUDE.md

Update this file when:
- Adding new modules or major refactoring existing ones
- Changing build/test/lint commands or tooling
- Modifying the three-layer architecture (System/Analysis/Scenario)
- Adding new execution modes or HPC integration patterns
- Changing critical configuration fields or validation patterns
- Discovering new "gotchas" that developers should know upfront

### When to Update Agent Documentation

Update `.claude/agents/*.md` when:
- Modifying core patterns in agent's domain (e.g., changing toggle validation logic → update `pydantic-config-specialist.md`)
- Adding new runner scripts or changing subprocess invocation patterns → update `snakemake-workflow.md`
- Changing SLURM execution modes or resource management → update `hpc-slurm-integration.md`
- Modifying SWMM model generation patterns → update `swmm-model-generation.md`
- Adding new test utilities or platform detection helpers → update `triton-test-suite.md`

### Documentation Update Checklist

When making significant code changes:
- [ ] Does this change affect architecture described in CLAUDE.md?
- [ ] Does this change affect patterns documented in any agent file?
- [ ] Are there new "gotchas" or non-obvious behaviors to document?
- [ ] Do build/test commands still work as documented?
- [ ] Are there new critical configuration fields to highlight?
