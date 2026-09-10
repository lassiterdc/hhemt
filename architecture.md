<!-- hhemt:repo-internal exempt=prose -->

# Architecture

Reference document for the TRITON-SWMM toolkit codebase. Tool-agnostic — load this alongside `CONTRIBUTING.md` at the start of any AI-assisted session.

---

## Project Overview

H&H Ensemble Modeling Toolkit orchestrates coupled TRITON (2D hydrodynamic) and SWMM (stormwater management) simulations. It supports single simulations, multi-simulation ensembles, and sensitivity analysis studies across local machines and HPC clusters (UVA, Oak Ridge Frontier).

---

## Three-Layer Hierarchy

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

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `config/` | Pydantic-based configuration package (base, system, analysis, loaders) |
| `workflow.py` | Dynamic Snakefile generation for parallel execution |
| `execution.py` | Execution strategies: SerialExecutor, LocalConcurrentExecutor, SlurmExecutor |
| `resource_management.py` | CPU/GPU/memory allocation for HPC |
| `sensitivity_analysis.py` | Parameter sweep orchestration with members |
| `paths.py` | Dataclasses: SysPaths, AnalysisPaths, ScenarioPaths |
| `log.py` | JSON-persisted logging with LogField[T] pattern |

---

## Workflow Phases

Snakemake rules invoke runner scripts as subprocesses in sequence:

| Phase | Rule | Runner Script | Resources |
|-------|------|---------------|-----------|
| 1 — Setup | `rule setup` | `setup_workflow.py` | 1 CPU |
| 2 — Scenario Preparation | `rule prepare_scenario` | `prepare_scenario_runner.py` | 1 CPU, parallel |
| 3 — Simulation Execution | `rule run_{model_type}` | `run_simulation_runner.py` | Multi-CPU/GPU |
| 4 — Output Processing | `rule process_{model_type}` | `process_timeseries_runner.py` | 2 CPUs |
| 5 — Consolidation | `rule consolidate` | `consolidate_workflow.py` | 1 CPU |

Exit codes: 0=success, 1=failure, 2=invalid arguments.

---

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

---

## Configuration System

Configuration flows: **YAML → Pydantic → Analysis/Scenario classes**

- `system_config`: DEM paths, TRITON compilation, Manning's coefficients
- `analysis_config`: Simulation parameters, HPC settings, weather data, execution mode

Many fields are conditionally required based on boolean toggles. Call `analysis.validate().raise_if_invalid()` before launching simulations (`src/hhemt/validation.py`).

### Critical Configuration Fields

| Field | Impact |
|-------|--------|
| `multi_sim_run_method` | Controls execution: `local`, `batch_job`, `1_job_many_srun_tasks` |
| `run_mode` | CPU/GPU config: `serial`, `openmp`, `mpi`, `gpu`, `hybrid` |
| `hpc_max_simultaneous_sims` | **Required** for `batch_job` mode (no default) |
| `hpc_time_min_per_sim` | Per-simulation time limit in minutes (required for `batch_job`) |
| `hpc_total_job_duration_min` | Total job duration cap in minutes (required for `batch_job`) |

---

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

`1_job_many_srun_tasks`: uses `--exclusive` + `hpc_total_nodes`; does not require `hpc_max_simultaneous_sims`. See `docs/implementation/1_job_many_srun_tasks_redesign.md`.

---

## Conda Environment

The working environment is defined in `workflow/envs/triton_swmm.yaml` — all dependencies for development, testing, and simulation. Update this file when adding new dependencies.

Note: `environment.yaml` at the repo root and the `conda:` directives in generated Snakefiles are scaffolding for a potential future two-environment split, but are not currently active (Snakemake is not invoked with `--use-conda`).

---

## Testing Quick Reference

- `test_PC_*.py` — local; `test_UVA_*.py` — UVA HPC; `test_frontier_*.py` — Frontier. Tests auto-skip by platform.
- Fixtures use `GetTS_TestCases` from `tests/fixtures/test_case_catalog.py`. Use `start_from_scratch=False` for cached/fast iteration.
- Use assertion helpers from `tests/utils_for_testing.py` (`assert_scenarios_run`, `assert_model_outputs_exist`, etc.) — not raw property checks.

---

## DOI kind convention

Two published surfaces carry a Zenodo DOI and they carry DIFFERENT ones on purpose.

- `CITATION.cff` carries the CONCEPT DOI (`...151`), which always resolves to the latest release.
- `docs/index.md`'s printed citation carries the VERSION DOI (`...152`), which pins v0.1.0 permanently.

The rule that makes both correct: a surface carries the most precise identifier that

1. EXISTS at the moment the surface is written, and
2. remains true of that surface's referent for as long as the surface is readable.

`CITATION.cff` fails (1). Zenodo mints a version DOI in response to a published release, so at the moment the file is written and the tag is cut, that release's version DOI does not yet exist. Measured: `git show v0.1.0:CITATION.cff` carries no DOI at all, and the commit that added one is not an ancestor of the tag — its own subject calls it a backfill.

Note which way the two conditions point, because it is not obvious. Condition (2) taken alone argues FOR the version DOI here: the file ships inside one specific release tarball, so a version DOI would stay true of it forever. Condition (1) wins because it is a hard constraint and (2) is a preference. **That is why this is stated as impossibility rather than as advice.** There is no counter-argument to a value that does not exist, whereas advice gets argued back by anyone weighting precision — and this project's own citation guidance weights precision, so advice would lose.

`docs/index.md` passes (1): it is rebuilt from the current branch after each release, so the version DOI exists by the time that page is written.

**This is NOT an equality invariant and nothing checks it.** The two values are expected to differ. A future editor who "aligns" them has removed the distinction rather than fixed an inconsistency.

Falsifier worth knowing: Zenodo supports reserving a DOI before publication. A reserve-then-tag release workflow would make the version DOI exist at write time, (1) would pass, and this convention would need revisiting. That is a change to how releases are made, not a Zenodo setting.

---

## Gotchas

1. **`hpc_max_simultaneous_sims` has no default** — required for `batch_job` mode
2. **Sensitivity analysis GPU constraint** — cannot mix GPU and non-GPU modes in a single sensitivity analysis
3. **SLURM detection includes config check** — `in_slurm` is True when `multi_sim_run_method == "1_job_many_srun_tasks"` even without `SLURM_JOB_ID`
4. **Runner scripts use argparse** — each has specific CLI flags; check docstrings for usage
5. **`log.out` overwrite with multi-model** — both TRITON-only and TRITON-SWMM write to `sim_folder/output/log.out`; last to finish overwrites the other
