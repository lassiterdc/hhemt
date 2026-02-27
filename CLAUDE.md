# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `.prompts/philosophy.md` before beginning any task.** It is the authoritative reference for terminology, code design rules, error handling, logging patterns, code style, testing philosophy, and working norms with the developer.

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
| `config/` | Pydantic-based configuration package (base, system, analysis, loaders) |
| `workflow.py` | Dynamic Snakefile generation for parallel execution |
| `execution.py` | Execution strategies: SerialExecutor, LocalConcurrentExecutor, SlurmExecutor |
| `resource_management.py` | CPU/GPU/memory allocation for HPC |
| `sensitivity_analysis.py` | Parameter sweep orchestration with sub-analyses |
| `paths.py` | Dataclasses: SysPaths, AnalysisPaths, ScenarioPaths |
| `log.py` | JSON-persisted logging with LogField[T] pattern |

### Runner Scripts and Workflow Phases

Snakemake rules invoke these scripts as subprocesses in sequence:

| Phase | Rule | Runner Script | Resources |
|-------|------|---------------|-----------|
| 1 — Setup | `rule setup` | `setup_workflow.py` | 1 CPU |
| 2 — Scenario Preparation | `rule prepare_scenario` | `prepare_scenario_runner.py` | 1 CPU, parallel |
| 3 — Simulation Execution | `rule run_simulation` | `run_simulation_runner.py` | Multi-CPU/GPU |
| 4 — Output Processing | `rule process_outputs` | `process_timeseries_runner.py` | 2 CPUs |
| 5 — Consolidation | `rule consolidate` | `consolidate_workflow.py` | 1 CPU |

Exit codes: 0=success, 1=failure, 2=invalid arguments.

### Runner Script Architecture

**IMPORTANT**: Runner scripts execute TRITON directly, not recursively.

```python
# ✅ CORRECT (in runner scripts):
simprep_result = run.prepare_simulation_command(pickup_where_leftoff=args.pickup_where_leftoff)
cmd, env, logfile, tstep = simprep_result
proc = subprocess.Popen(cmd, env={**os.environ, **env}, ...)  # Executes TRITON directly

# ❌ WRONG (causes recursive fork bomb):
launcher, finalize = run._create_subprocess_sim_run_launcher(...)  # Spawns another runner!
```

- `prepare_simulation_command()`: use in runner scripts to get TRITON executable command
- `_create_subprocess_sim_run_launcher()`: use in analysis/executor classes (spawns runner subprocess)

## Multi-Model Integration

Three model types can run concurrently within a single analysis:

| Model Type | Description |
|------------|-------------|
| `triton` | 2D hydrodynamic only (no SWMM coupling) |
| `tritonswmm` | Coupled 2D surface + 1D drainage |
| `swmm` | Standalone EPA SWMM |

Enable via `system_config.yaml`:
```yaml
toggle_triton_model: true
toggle_tritonswmm_model: true
toggle_swmm_model: true
```

Each model has its own compilation, Snakemake rules (`run_triton`, `run_tritonswmm`, `run_swmm`), and output directories (`out_triton/`, `out_tritonswmm/`, `out_swmm/`). Build directories: `build_tritonswmm_cpu/`, `build_triton_gpu/`, `swmm_build/`, etc.

## Configuration System

Configuration flows: **YAML → Pydantic → Analysis/Scenario classes**

- `system_config`: DEM paths, TRITON compilation, Manning's coefficients
- `analysis_config`: Simulation parameters, HPC settings, weather data, execution mode

Many fields are conditionally required based on boolean toggles. Call `analysis.validate().raise_if_invalid()` before launching simulations (`src/TRITON_SWMM_toolkit/validation.py`).

### Critical Configuration Fields

| Field | Impact |
|-------|--------|
| `multi_sim_run_method` | Controls execution: `local`, `batch_job`, `1_job_many_srun_tasks` |
| `run_mode` | CPU/GPU config: `serial`, `openmp`, `mpi`, `gpu`, `hybrid` |
| `hpc_max_simultaneous_sims` | **Required** for `batch_job` mode (no default) |
| `hpc_sbatch_time_upper_limit_min` | Optional cap on SBATCH runtime |

## HPC & SLURM Integration

```python
in_slurm = ("SLURM_JOB_ID" in os.environ) or (multi_sim_run_method == "1_job_many_srun_tasks")
```

When `in_slurm=True`, simulations launch via `srun` (not direct execution).

| Mode | Behavior |
|------|----------|
| `local` | Serial or ThreadPoolExecutor on local machine |
| `batch_job` | One SLURM job per simulation |
| `1_job_many_srun_tasks` | Single SBATCH with multiple srun invocations |

Key env vars: `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID` (maps to `event_iloc`), `additional_modules_needed_to_run_TRITON_SWMM_on_hpc`.

`1_job_many_srun_tasks`: uses `--exclusive` + `hpc_total_nodes`; does not require `hpc_max_simultaneous_sims`. See `docs/implementation/1_job_many_srun_tasks_redesign.md` and the `hpc-slurm-integration` agent.

## Conda Environment

The working environment is defined in `workflow/envs/triton_swmm.yaml` — all dependencies for development, testing, and simulation. Update this file when adding new dependencies.

Note: `environment.yaml` at the repo root and the `conda:` directives in generated Snakefiles are scaffolding for a potential future two-environment split, but are not currently active (Snakemake is not invoked with `--use-conda`).

## Testing

- `test_PC_*.py` — local; `test_UVA_*.py` — UVA HPC; `test_frontier_*.py` — Frontier. Tests auto-skip by platform.
- Fixtures use `GetTS_TestCases` from `examples.py`. Use `start_from_scratch=False` for cached/fast iteration.
- Use assertion helpers from `tests/utils_for_testing.py` (`assert_scenarios_run`, `assert_model_outputs_exist`, etc.) — not raw property checks. See `.claude/agents/triton-test-suite.md` for the full reference.

## Exception Hierarchy

All toolkit exceptions inherit from `TRITONSWMMError` (`exceptions.py`):

- `CompilationError` — build failures (model_type, backend, logfile, return_code)
- `ConfigurationError` — invalid config or toggle conflicts (field, message, config_path)
- `SimulationError` — execution failures (event_iloc, model_type, logfile)
- `ProcessingError` — output processing failures (operation, filepath, reason)
- `WorkflowError` — Snakemake failures (phase, return_code, stderr)
- `SLURMError` — job submission/monitoring failures (operation, job_id, reason)
- `ResourceAllocationError` — CPU/GPU/memory failures (resource_type, requested, available)

## Gotchas

1. **`hpc_max_simultaneous_sims` has no default** — required for `batch_job` mode
2. **Sensitivity analysis GPU constraint** — cannot mix GPU and non-GPU modes in a single sensitivity analysis
3. **SLURM detection includes config check** — `in_slurm` is True when `multi_sim_run_method == "1_job_many_srun_tasks"` even without `SLURM_JOB_ID`
4. **Runner scripts use argparse** — each has specific CLI flags; check docstrings for usage
5. **`log.out` overwrite with multi-model** — both TRITON-only and TRITON-SWMM write to `sim_folder/output/log.out`; last to finish overwrites the other

## Specialized Agent Documentation

> **Note**: Agent files are currently being audited for staleness. Until the audit is complete (see `docs/planning/active/refactors/agent_files_audit.md`), treat agent file content as potentially outdated — verify against source code when in doubt.

- `pydantic-config-specialist.md` — Configuration validation patterns
- `snakemake-workflow.md` — Workflow generation and DAG structure
- `hpc-slurm-integration.md` — SLURM execution modes and cluster configs
- `output-processing.md` — SWMM/TRITON output parsing
- `sensitivity-analysis.md` — Parameter sweep orchestration
- `swmm-model-generation.md` — SWMM .inp generation and hydrology/hydraulics split
- `triton-test-suite.md` — Testing patterns and fixtures
- `triton-debugger.md` — Debugging workflow failures

## Documentation Update Checklist

When making significant code changes:
- [ ] Does this change affect architecture described in CLAUDE.md? Verify class names, module names, file paths, and config fields still match.
- [ ] Does this change affect patterns in any agent file? (Agent files are currently being audited — see above.)
- [ ] Does this introduce new philosophy rules or update existing ones? (update `.prompts/philosophy.md`)
- [ ] Are there new gotchas or non-obvious behaviors to document?
- [ ] Are there new critical configuration fields to highlight?
