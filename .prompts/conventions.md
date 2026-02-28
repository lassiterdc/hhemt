# Development Conventions

Consistent vocabulary and working principles for the TRITON-SWMM toolkit. Reference this document when planning implementations and during QA review. It is a living document — update it when new rules are established or existing ones are clarified.

---

## Part I: Universal Principles

*Portable to any solo Python project with minimal editing.*

### Raise questions rather than make assumptions
When you encounter uncertainty or discrepancies — especially when implementing a pre-written plan that may have stale components — err on the side of caution and ask the developer how to proceed.

### Plan, then implement
Follow a plan-then-implement strategy. If implementing a plan uncovers a need to change it or its success criteria — including deviations from the planned approach, scope changes, or new risks — raise the discrepancy before continuing rather than adapting silently.

### Let's do things right, even if it takes more effort
- Always be on the lookout for better ways of achieving development goals and raise these ideas
- Raise concerns when you suspect the developer is making design decisions that diverge from best practices
- Look for opportunities to make code more efficient (vectorize operations, avoid loops with pandas, etc.)

### Backward compatibility is NOT a priority

**Rationale**: Single developer codebase. Clean code matters more than preserved APIs. Git history is the safety net.

When refactoring:
- ❌ Don't add deprecation warnings
- ❌ Don't keep old APIs "for compatibility"
- ❌ Don't create compatibility shims or aliases
- ✅ Do update all usage sites immediately
- ✅ Do delete obsolete code completely

### Most function arguments should not have defaults

Default function arguments can lead to difficult-to-debug unexpected behavior. Avoid default values unless a default is almost always the correct choice (e.g., `verbose=True`). This is especially true for configuration fields that users populate — the user should make an intentional choice about every input.

### Avoid aliases

Do not create aliases for functions, classes, or variables. An alias is a second name for the same thing — it creates confusion about which name is authoritative and is a form of backward-compatibility shim. If something needs renaming, rename it and update all call sites.

### No cruft/all variables, imports, and function arguments must be used

Unused elements are a signal that implementation may be incomplete. Treat them as an investigation trigger, not just lint to suppress.

If you come across an unused variable, import, or function argument, investigate before removing:
1. Check whether the surrounding implementation is incomplete
2. Find planning documents that touched that function and determine whether implementation is planned
3. If still uncertain, raise the concern with the developer with hypotheses about why it exists
4. The only exception: elements included for a currently-planned implementation, marked with a comment referencing the planning document

Report your observations, hypotheses, and recommendations to the developer.

After investigation and with approval from the developer, remove unused code, dead branches, commented-out blocks, and stale imports. 

### Functions have docstrings, type hints, and type checking

Apply this standard to code you write or modify. For existing code in touched scripts, apply organically — accumulate adherence naturally as scripts are touched rather than doing a global retrofit pass.

### Fail-fast

Critical paths must raise exceptions; never silently return `False` or `None` on failure.

### Preserve context in exceptions

Exceptions should include file paths, return codes, and log locations for actionable debugging.

### Prefer log-based completion checks over file existence checks

A file may exist but be corrupt, incomplete, or from a previous failed run. File existence checks can mask errors when log checks are available.

- **Exception**: File existence is appropriate for verifying *input* files before reading them.

### Keep system-agnostic software

System-specific information belongs in user-defined configuration files. Avoid hardcoded paths or machine-specific constants in core code.

### Track project-agnostic utility candidates

When writing utility functions that could plausibly belong in a shared library (e.g., general-purpose file I/O helpers, generic array operations), note them in a dedicated tracking document. Do not extract them immediately — track them so they can be evaluated together.

---

## Part II: Project-Specific Rules

*TRITON-SWMM toolkit specifics. Replace or extend for a different project.*

### Terminology

Consistent vocabulary prevents AI/human miscommunication and catches conceptual errors early. The following terms have precise meanings in this codebase.

| Term | Meaning | Usage |
|------|---------|-------|
| `model_type` | Which simulation model is running: `"triton"` (2D hydrodynamic only), `"tritonswmm"` (coupled 2D + 1D drainage), `"swmm"` (standalone EPA SWMM). | Config fields, output directories, Snakemake rule names (`rule run_triton`, etc.) |
| `run_mode` | The computational execution strategy: `"serial"`, `"openmp"`, `"mpi"`, `"gpu"`, `"hybrid"`. Orthogonal to `model_type`. | `analysis_config.run_mode`; controls srun flags and resource allocation |
| `multi_sim_run_method` | How the ensemble of simulations is orchestrated: `"local"`, `"batch_job"`, `"1_job_many_srun_tasks"`. A separate axis from `run_mode`. | `analysis_config.multi_sim_run_method`; controls whether Snakemake submits SBATCH jobs or runs locally |
| `event_iloc` | The canonical flat integer index uniquely identifying a single simulated weather event within an analysis. Maps simulation results to meteorological inputs. | xarray dimension, CSV column, Snakemake wildcards, code variables |
| `in_slurm` | Derived boolean: `True` when `SLURM_JOB_ID` is in the environment OR when `multi_sim_run_method == "1_job_many_srun_tasks"`. Not just the env var check. | Used in `execution.py` to decide whether to launch via `srun` |

**Rule**: `model_type` and `run_mode` are frequently confused — they are independent configuration axes. A `tritonswmm` model can run with any `run_mode`; an `openmp` run_mode can apply to any `model_type`.

### Custom exception hierarchy

Use the hierarchy in `exceptions.py` (all inherit from `TRITONSWMMError`). Include full contextual attributes:
```python
raise CompilationError(
    model_type="tritonswmm",
    backend="cpu",
    logfile=compilation_log,
    return_code=proc.returncode,
)
```

Exception types:
- `CompilationError` — TRITON/SWMM build failures (model_type, backend, logfile, return_code)
- `ConfigurationError` — invalid config values or toggle conflicts (field, message, config_path)
- `SimulationError` — simulation execution failures (event_iloc, model_type, logfile)
- `ProcessingError` — output processing failures (operation, filepath, reason)
- `WorkflowError` — Snakemake workflow failures (phase, return_code, stderr)
- `SLURMError` — SLURM job submission/monitoring failures (operation, job_id, reason)
- `ResourceAllocationError` — CPU/GPU/memory allocation failures (resource_type, requested, available)

### Log-based completion checks — toolkit implementation

The toolkit implementation of the universal "prefer log-based checks" principle: `_already_written()` verifies a file was written *successfully*, not just that it exists. Prefer it over raw file existence checks.

### Logging patterns

- **User-facing progress**: `print(f"[NAMESPACE] Message", flush=True)` with verbose guards
  - Examples: `[CPU] Compiling...`, `[Snakemake] Job submitted`, `[SLURM] Running`
- **Library code**: `logger = logging.getLogger(__name__)`
- **Runner scripts**: Use module-level loggers; stdout is collected by Snakemake into logfiles

### Architecture patterns

**Use Pydantic models and user-defined YAMLs for inputs.**
Configuration flows: YAML → Pydantic (`cfgBaseModel` subclass) → Analysis/Scenario classes.

**Outputs generated by runner scripts.**
To accommodate Snakemake, outputs are generated by runner scripts that take command-line arguments. Snakemake rules invoke these scripts as subprocesses.

**Runner scripts execute TRITON directly — never recursively.**
Runner scripts must use `prepare_simulation_command()` to get the TRITON executable command, then launch it directly via `subprocess.Popen`. Never call `_create_subprocess_sim_run_launcher()` from a runner script — that method spawns another runner subprocess, causing a recursive fork bomb.

```python
# ✅ CORRECT (in runner scripts):
simprep_result = run.prepare_simulation_command(pickup_where_leftoff=args.pickup_where_leftoff)
cmd, env, logfile, tstep = simprep_result
proc = subprocess.Popen(cmd, env={**os.environ, **env}, ...)  # Executes TRITON directly

# ❌ WRONG (causes recursive fork bomb):
launcher, finalize = run._create_subprocess_sim_run_launcher(...)  # Spawns another runner!
```

- `prepare_simulation_command()` — use in runner scripts
- `_create_subprocess_sim_run_launcher()` — use only in analysis/executor classes

**Snakemake rule generation should use wildcards.**
`workflow.py` should use wildcards as much as possible to keep generated Snakefiles human-readable. Use rule-generating loops only when a cleaner canonical Snakemake approach isn't available.

**System-specific constants belong in `platform_configs.py`.**
Cluster-specific values (partition names, GPU hardware strings, NIC policy env vars) live in `platform_configs.py`, not hardcoded in core code.

**Track utility candidates in `docs/planning/utility_package_candidates.md`.**
Do not extract immediately — track for batch evaluation.

### Planning document lifecycle

Planning documents live in `docs/planning/` organized by type (`bugs/`, `features/`, `refactors/`). Each type directory has a `completed/` subdirectory; a `shelved/` subdirectory is created as needed.

- **Active work**: document lives directly in the type directory (e.g., `docs/planning/bugs/my_fix.md`)
- **Completed**: move to `completed/` within the same type directory (e.g., `docs/planning/bugs/completed/my_fix.md`)
- **Deprioritized/blocked**: move to `shelved/` within the same type directory (create if it doesn't exist)
- **No longer relevant**: delete

See `docs/planning/README.md` for the full structure.

### Code style

- **Python**: ≥3.10, target 3.12+
- **Formatter/linter**: `ruff format` and `ruff check` — run before submitting any code. Line length and all style rules are enforced by `pyproject.toml`; write code that will survive `ruff format` unchanged.
- **Configuration models** inherit from `cfgBaseModel`
- **Enumerated options** use `Literal` types
- **Type checker**: Pyright/Pylance — address squiggles organically as scripts are touched; do not leave new `# type: ignore` comments unless the issue is a known type checker limitation

### GIS data

Prefer **GeoJSON** over Shapefile format for new spatial inputs:
- GeoJSON is a single file (no `.prj`/`.dbf`/`.shx` sidecar files)
- GeoJSON is human-readable and version-control friendly

### Testing conventions

- **Platform-organized**: tests are split by execution environment (`test_PC_*.py`, `test_UVA_*.py`, `test_frontier_*.py`) and auto-skip based on platform detection
- **Fixtures from `examples.py`**: use `GetTS_TestCases` factory functions; prefer `start_from_scratch=False` for cached/fast iteration, `True` for clean runs
- **Standardized assertions**: use helpers from `tests/utils_for_testing.py` (e.g., `assert_scenarios_run`, `assert_model_outputs_exist`) rather than direct property checks — helpers provide consistent, actionable error messages
- **Log-based completion checks**: prefer log-based checks over file existence checks (see universal principle above; `_already_written()` is the toolkit implementation)
- **SLURM-specific tests require HPC**: do not attempt to mock SLURM behavior locally; coordinate HPC testing with the developer directly

**Smoke tests** — apply judgment to determine which tests, if any, are warranted for a given implementation. Run in the order listed; each test depends on lower-level functionality being correct.

| Test | Command | What it covers | Run when |
|------|---------|---------------|----------|
| PC_01 | `pytest tests/test_PC_01_singlesim.py -v` | Single simulation end-to-end: scenario setup, serial execution, output processing for all enabled model types | Touching scenario preparation, single-sim execution, output processing, or any code in the scenario→run→process pipeline |
| PC_02 | `pytest tests/test_PC_02_multisim.py -v` | Multi-simulation concurrent execution: parallel scenario prep, concurrent sim launch, timeseries processing | Touching concurrent execution, `LocalConcurrentExecutor`, or multi-sim orchestration in `analysis.py` |
| PC_04 | `pytest tests/test_PC_04_multisim_with_snakemake.py -v` | Snakemake workflow generation and local execution: Snakefile generation, rule structure, flags, full local Snakemake run | Touching `workflow.py`, Snakefile templates, rule generation, or any change that affects how Snakemake orchestrates local runs |
| PC_05 | `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v` | Sensitivity analysis workflow: sub-analysis Snakefile generation, master Snakefile generation, full sensitivity run via Snakemake | Touching `sensitivity_analysis.py`, sensitivity-specific workflow generation, or sub-analysis orchestration. **Do NOT impose artificial timeouts — legitimately takes 12-15 minutes** |

Not every implementation requires smoke tests. Pure documentation changes, planning doc updates, or changes isolated to HPC-only paths (SLURM submission, srun flags) do not need local smoke tests — coordinate HPC-specific validation with the developer directly.

### HPC debugging protocol

*The principle (don't guess at remote systems) is universal; the log paths below are toolkit-specific.*

You cannot execute commands on Frontier, UVA, or any HPC cluster. Do not guess at HPC runtime behavior or propose fixes based on speculation.

**Step 1: Exhaust available log files first.** Before requesting anything from the developer, check whether the answer is already in logs produced during the failed run. Key sources: master workflow log (`logs/tmux_session_YYYYMMDD_HHMMSS.log`), runner script logs (`logs/sims/{rule}_{event_iloc}.log` and model runtime logs `logs/sims/model_{model_type}_evt{event_iloc}.log`), Snakemake SLURM logs (`.snakemake/slurm_logs/`). Ask the developer to share relevant log files before requesting HPC commands. See `debugging_hpc_analysis.md` for the full log taxonomy.

**Step 2: Empirical testing when logs are insufficient.** When log files don't answer the question, coordinate tests with the developer:
1. Identify the specific unknowns.
2. Find or create the relevant planning document.
3. Add an `## Empirical HPC Testing` section with a subheader per test.
4. Each test entry must contain: a brief explanation of what is being tested and why, a copy-pastable bash code block, and an empty bash code block for the developer to fill in.
5. Ask the developer to run the tests and paste output back; interpret results against source code only.

Never propose a code change to fix an HPC issue that has not been empirically confirmed. If source code analysis alone is sufficient to identify the root cause with high confidence, state that explicitly before recommending a fix.

---

## Part III: AI Working Norms

*Claude Code conventions for this developer. Light editing needed to port to a different AI tool.*

### Never commit without explicit permission
All commits require prior approval from the developer.

### Never commit Jupyter notebooks
Notebooks in `tests/dev/` are developer testing scratchpads — they contain transient state (`start_from_scratch`, cell outputs, etc.) and should never be staged or committed. If a notebook appears in `git status`, exclude it from the commit.

### Always confirm before spawning subagents
Never invoke the Task tool to spawn a subagent or background agent without first confirming with the developer. Describe what you intend to delegate and why, and wait for explicit approval. The developer may prefer tighter back-and-forth in the current conversation.

### `#user:` comments in planning documents are blocking
In planning documents, all comments prefixed with `#user:` are developer feedback that must ALL be addressed before any implementation can take place. Remove each comment only after written confirmation from the developer that it has been sufficiently addressed. Implications for the entire planning document should be considered when addressing these comments.

### Keep documentation current

When making significant code changes:
- Does this change affect architecture described in `.prompts/architecture.md`? Verify class names, module names, file paths, and config fields still match.
- Does this change affect patterns documented in an active agent file? Update the agent.
- Does this introduce new conventions or update existing ones? Update `.prompts/conventions.md`.
- Are there new gotchas or non-obvious behaviors to document?
- Are there new critical configuration fields to highlight?

### Plan, then implement — toolkit workflow

Follow the plan-then-implement strategy outlined in `.prompts/implementation_plan.md`. Use `.prompts/proceed_with_implementation.md` for preflight checks before starting implementation. Use `.prompts/qaqc_and_commit.md` for post-implementation review.
