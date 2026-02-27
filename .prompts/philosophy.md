# Development Philosophy

Consistent vocabulary and working principles for the TRITON-SWMM toolkit. Reference this document when planning implementations and during QA review. It is a living document — update it when new rules are established or existing ones are clarified.

---

## Terminology

Consistent vocabulary prevents AI/human miscommunication and catches conceptual errors early. The following terms have precise meanings in this codebase.

| Term | Meaning | Usage |
|------|---------|-------|
| `model_type` | Which simulation model is running: `"triton"` (2D hydrodynamic only), `"tritonswmm"` (coupled 2D + 1D drainage), `"swmm"` (standalone EPA SWMM). | Config fields, output directories, Snakemake rule names (`rule run_triton`, etc.) |
| `run_mode` | The computational execution strategy: `"serial"`, `"openmp"`, `"mpi"`, `"gpu"`, `"hybrid"`. Orthogonal to `model_type`. | `analysis_config.run_mode`; controls srun flags and resource allocation |
| `multi_sim_run_method` | How the ensemble of simulations is orchestrated: `"local"`, `"batch_job"`, `"1_job_many_srun_tasks"`. A separate axis from `run_mode`. | `analysis_config.multi_sim_run_method`; controls whether Snakemake submits SBATCH jobs or runs locally |
| `event_iloc` | The canonical flat integer index uniquely identifying a single simulated weather event within an analysis. Maps simulation results to meteorological inputs. | xarray dimension, CSV column, Snakemake wildcards, code variables |
| `in_slurm` | Derived boolean: `True` when `SLURM_JOB_ID` is in the environment OR when `multi_sim_run_method == "1_job_many_srun_tasks"`. Not just the env var check. | Used in `execution.py` to decide whether to launch via `srun` |

**Rule**: `model_type` and `run_mode` are frequently confused — they are independent configuration axes. A `tritonswmm` model can run with any `run_mode`; an `openmp` run_mode can apply to any `model_type`.

---

## Working With the Developer

### Never commit without explicit permission
All commits require prior approval from the developer.

### Always confirm before spawning subagents
Never invoke the Task tool to spawn a subagent or background agent without first confirming with the developer. Describe what you intend to delegate and why, and wait for explicit approval. The developer may prefer tighter back-and-forth in the current conversation.

### Raise questions rather than make assumptions
When you encounter uncertainty or discrepancies — especially when implementing a pre-written plan that may have stale components — err on the side of caution and ask the developer how to proceed.

### Plan, then implement
Follow the plan-then-implement strategy outlined in `.prompts/implementation_plan.md`. If implementing a plan uncovers a need to change it or its success criteria — including deviations from the planned approach, scope changes, or new risks — raise the discrepancy before continuing rather than adapting silently.

### `#user:` comments in planning documents are blocking
In planning documents, all comments prefixed with `#user:` are developer feedback that must ALL be addressed before any implementation can take place. Remove each comment only after written confirmation from the developer that it has been sufficiently addressed. Implications for the entire planning document should be considered when addressing these comments.

### Let's do things right, even if it takes more effort
- Always be on the lookout for better ways of achieving development goals and raise these ideas
- Raise concerns when you suspect the developer is making design decisions that diverge from best practices
- Look for opportunities to make code more efficient (vectorize operations, avoid loops with pandas, etc.)

---

## Code Design Rules

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

### No cruft

Remove unused code, dead branches, commented-out blocks, and stale imports. If you come across an unused variable, import, or function argument, investigate before removing:
1. Check whether the surrounding implementation is incomplete
2. Find planning documents that touched that function and determine whether implementation is planned
3. If still uncertain, raise the concern with the developer with hypotheses about why it exists
4. The only exception: elements included for a currently-planned implementation, marked with a comment referencing the planning document

### All variables, imports, and function arguments must be used

Unused elements are a signal that implementation may be incomplete. Treat them as an investigation trigger, not just lint to suppress.

### Functions have docstrings, type hints, and type checking

Apply this standard to code you write or modify. For existing code in touched scripts, apply organically — the QA review step in `.prompts/qaqc_and_commit.md` prompts checking touched scripts against philosophy, so adherence accumulates naturally without a global retrofit pass.

---

## Error Handling

### Fail-fast
Critical paths must raise exceptions; never silently return `False` or `None` on failure.

### Preserve context
Exceptions should include file paths, return codes, and log locations for actionable debugging.

### Raise custom exceptions
Use the hierarchy in `exceptions.py` (all inherit from `TRITONSWMMError`). Include full contextual attributes:
```python
raise CompilationError(
    model_type="tritonswmm",
    backend="cpu",
    logfile=compilation_log,
    return_code=proc.returncode,
)
```

---

## Completion Status: Log-Based Checks over File Existence

Prefer log-based checks over file existence checks for determining whether processing completed successfully.

- `_already_written()` verifies a file was written *successfully*, not just that it exists
- A file may exist but be corrupt, incomplete, or from a previous failed run
- File existence checks can mask errors when log checks are available
- **Exception**: File existence is appropriate for verifying *input* files before reading them

---

## Logging Patterns

- **User-facing progress**: `print(f"[NAMESPACE] Message", flush=True)` with verbose guards
  - Examples: `[CPU] Compiling...`, `[Snakemake] Job submitted`, `[SLURM] Running`
- **Library code**: `logger = logging.getLogger(__name__)`
- **Runner scripts**: Use module-level loggers; stdout is collected by Snakemake into logfiles

---

## Architecture Patterns

### Use Pydantic models and user-defined YAMLs for inputs
Configuration flows: YAML → Pydantic (`cfgBaseModel` subclass) → Analysis/Scenario classes.

### Outputs generated by runner scripts
To accommodate Snakemake, outputs are generated by runner scripts that take command-line arguments. Snakemake rules invoke these scripts as subprocesses.

### Snakemake rule generation should use wildcards
`workflow.py` should use wildcards as much as possible to keep generated Snakefiles human-readable. Use rule-generating loops only when a cleaner canonical Snakemake approach isn't available.

### System-agnostic software
System-specific information belongs in user-defined YAML files. Avoid hardcoded paths or cluster-specific constants outside of `platform_configs.py`.

### Track project-agnostic utility candidates
When writing utility functions that could plausibly belong in a shared library (e.g., general-purpose file I/O helpers, generic array operations), note them in `docs/planning/utility_package_candidates.md`. Do not extract them immediately — just track them so they can be evaluated together.

---

## Code Style

- **Python**: ≥3.10, target 3.12+
- **Formatter/linter**: `ruff format` and `ruff check` — run these before submitting any code. Line length and all style rules are enforced by `pyproject.toml`; write code that will survive `ruff format` unchanged.
- **Configuration models** inherit from `cfgBaseModel`
- **Enumerated options** use `Literal` types

---

## GIS Data

The toolkit uses `geopandas` for spatial data loading (boundary lines, shapefiles). `gpd.read_file()` is format-agnostic, but prefer **GeoJSON** over Shapefile format for new inputs:
- GeoJSON is a single file (no `.prj`/`.dbf`/`.shx` sidecar files)
- GeoJSON is human-readable and version-control friendly
- When creating or recommending spatial input files, suggest GeoJSON

---

## Type Checking

Apply type hints to all code you write or modify (see "Functions have docstrings, type hints, and type checking" above). For type checking tooling:
- **Pyright/Pylance** is the recommended type checker for this project
- Address type squiggles organically as scripts are touched — do not retrofit the entire codebase at once
- When adding type hints to existing code, fix the squiggles in the scope of the functions you modified; do not leave new `# type: ignore` comments unless the issue is a known limitation of the type checker

---

## Testing Philosophy

- **Platform-organized**: tests are split by execution environment (`test_PC_*.py`, `test_UVA_*.py`, `test_frontier_*.py`) and auto-skip based on platform detection
- **Fixtures from `examples.py`**: use `GetTS_TestCases` factory functions; prefer `start_from_scratch=False` for cached/fast iteration, `True` for clean runs
- **Standardized assertions**: use helpers from `tests/utils_for_testing.py` (e.g., `assert_scenarios_run`, `assert_model_outputs_exist`) rather than direct property checks — helpers provide consistent, actionable error messages
- **Log-based completion checks**: prefer log-based checks over file existence checks to confirm processing success (see "Completion Status" above)
- **SLURM-specific tests require HPC**: do not attempt to mock SLURM behavior locally; coordinate HPC testing with the developer directly
