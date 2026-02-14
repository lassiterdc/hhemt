# Shared Orchestration Core Design

**Status**: Draft (Tier 3 Phase 2)
**Created**: 2026-02-09
**Purpose**: Consolidate run/setup/processing flow behind one orchestration layer

---

## Problem Statement

Currently, orchestration logic is duplicated across:
1. **CLI** (`cli.py`): Translates user-friendly flags to workflow parameters (lines 365-415)
2. **Notebooks/API**: Directly calls `analysis.submit_workflow()` with 15+ parameters
3. **Tests**: Use various patterns (serial execution, concurrent, Snakemake)

This creates:
- Code duplication (parameter translation logic repeated)
- Maintenance burden (changes require updating multiple call sites)
- Inconsistency risk (CLI vs API may behave differently)
- Poor UX (API users must understand 15+ low-level parameters)

## Current Architecture

```
User/CLI → Parameter Translation → analysis.submit_workflow() → _workflow_builder.submit_workflow()
                                   (15+ parameters)              (Snakemake generation)
```

The **parameter translation** step includes:
- Detecting if system inputs need processing (DEM/Manning's log checks)
- Translating `--from-scratch` → multiple overwrite flags
- Translating `--resume` → `pickup_where_leftoff`
- Determining execution mode (local vs SLURM)
- Setting compilation/preparation flags

## Proposed Architecture

```
User/CLI → High-Level Orchestration → Workflow Submission → Snakemake
           (simple intent-based API)   (parameter translation)  (execution)
```

### Design Principles

1. **Intent-based API**: Users specify *what* they want, not *how* to do it
2. **Smart defaults**: Orchestration layer infers correct parameters from state
3. **Single source of truth**: Parameter translation logic lives in ONE place
4. **CLI as thin adapter**: CLI just calls orchestration with user intent
5. **Testability**: Orchestration layer can be tested independently

## Proposed API

### High-Level Method: `analysis.run()`

```python
def run(
    self,
    # Execution intent (simplified)
    mode: Literal["fresh", "resume", "overwrite"] = "resume",

    # Scope control (what to run)
    phases: Optional[List[str]] = None,  # ["setup", "prepare", "simulate", "process", "consolidate"]
    events: Optional[List[int]] = None,  # Subset of events to run

    # Execution context
    execution_mode: Literal["auto", "local", "slurm"] = "auto",

    # Output control
    dry_run: bool = False,
    verbose: bool = True,
) -> WorkflowResult:
    """
    High-level orchestration method that handles full workflow lifecycle.

    This method replaces direct calls to submit_workflow() with a simpler,
    intent-based API.

    To determine which mode to use, check workflow status first:

        >>> status = analysis.get_workflow_status()
        >>> print(status.recommendation)
        >>> result = analysis.run(mode=status.recommended_mode)

    Parameters
    ----------
    mode : Literal["fresh", "resume", "overwrite"]
        Execution mode:
        - "fresh": Start from scratch, delete all artifacts
        - "resume": Continue from last checkpoint (default)
        - "overwrite": Recreate outputs even if logs show completion

    phases : Optional[List[str]]
        Which phases to run. If None, runs all phases.
        Phases: ["setup", "prepare", "simulate", "process", "consolidate"]

    events : Optional[List[int]]
        Subset of event_ilocs to process. If None, processes all events.

    execution_mode : Literal["auto", "local", "slurm"]
        Where to run: auto-detect, force local, or force SLURM

    dry_run : bool
        If True, validate workflow but don't execute

    verbose : bool
        If True, print progress messages

    Returns
    -------
    WorkflowResult
        Structured result object with:
        - success: bool
        - mode: str (local/slurm)
        - phases_completed: List[str]
        - job_id: Optional[str]
        - snakefile_path: Path
        - message: str
    """
```

### Internal Translation

The `run()` method would internally:

1. **Detect state** (check logs for what's completed)
2. **Translate mode**:
   - `"fresh"` → `from_scratch=True`, all overwrite flags
   - `"resume"` → `pickup_where_leftoff=True`, no overwrites
   - `"overwrite"` → All overwrite flags True, no from_scratch
   - For smart mode selection, see `workflow_status_reporting_plan.md` for --status flag
3. **Map phases to workflow flags**:
   - `"setup"` → `process_system_level_inputs=True`, `compile_TRITON_SWMM=True`
   - `"prepare"` → `prepare_scenarios=True`
   - `"simulate"` → (always enabled if scenarios prepared)
   - `"process"` → `process_timeseries=True`
   - `"consolidate"` → (handled by workflow's consolidate rule)
4. **Detect execution context**:
   - Check `SLURM_JOB_ID` env var
   - Check `analysis.cfg_analysis.multi_sim_run_method`
   - Override if `execution_mode != "auto"`
5. **Call underlying** `submit_workflow()` with translated parameters

### Result Object

```python
@dataclass
class WorkflowResult:
    """Structured result from workflow execution."""
    success: bool
    mode: str  # "local" or "slurm"
    execution_time: Optional[float]  # seconds
    phases_completed: List[str]
    events_processed: List[int]
    snakefile_path: Path
    job_id: Optional[str] = None  # SLURM only
    message: str = ""

    def __bool__(self) -> bool:
        """Allow truthiness check: if result: ..."""
        return self.success
```

## Migration Strategy

### Phase 1: Add new `run()` method alongside existing API
- Implement `analysis.run()` with translation logic
- Keep `submit_workflow()` for backward compatibility
- Add deprecation warning to `submit_workflow()` docstring

### Phase 2: Update CLI to use new API
- Refactor `cli.py` to call `analysis.run()`
- Remove parameter translation from CLI
- CLI becomes 20-30 lines instead of 60+

### Phase 3: Update notebooks/examples
- Replace `analysis.submit_workflow()` calls with `analysis.run()`
- Show simplified examples in documentation

### Phase 4 (future): Deprecate `submit_workflow()`
- After 1-2 versions, mark as deprecated
- Eventually remove or make internal-only

## Benefits

1. **Reduced duplication**: Parameter translation logic in ONE place
2. **Simpler CLI**: 60 lines → 30 lines, just calls `analysis.run()`
3. **Better API UX**: 4 intuitive parameters instead of 15 low-level ones
4. **Easier testing**: Can test orchestration logic independently
5. **Consistent behavior**: CLI and API guaranteed to behave identically
6. **Foundation for Phase 4**: Easy to wrap in high-level `Toolkit` API

## Open Questions

1. **Naming**: `run()` vs `execute()` vs `orchestrate()`?
   - **Decision**: `run()` (simplest, most intuitive)

2. **Mode names**: `"fresh"` vs `"from_scratch"`?
   - **Decision**: `"fresh"` (shorter, clearer intent)

3. **Phases granularity**: Should we support finer control?
   - **Decision**: Start with 5 phases, can add more later

4. **Backward compatibility**: Keep `submit_workflow()` forever?
   - **Decision**: Deprecate but keep for 2-3 versions

5. **Event selection**: Should this be in `run()` or separate method?
   - **Decision**: Include in `run()` for convenience

## Implementation Checklist

- [ ] Create `WorkflowResult` dataclass in new `orchestration.py` module
- [ ] Implement `analysis.run()` method with mode translation
- [ ] Add comprehensive docstring with examples
- [ ] Update CLI to use `analysis.run()`
- [ ] Add unit tests for mode translation logic
- [ ] Update at least one notebook to show new API
- [ ] Add deprecation notice to `submit_workflow()` docstring
- [ ] Update CLAUDE.md with new orchestration pattern

---

## Examples

### CLI Usage (After)

```python
# cli.py - simplified to ~30 lines
result = analysis.run(
    mode="fresh" if from_scratch else "resume",
    dry_run=dry_run,
    verbose=verbose,
)
if not result.success:
    raise typer.Exit(3)
```

### Notebook Usage (After)

```python
# Simple resume
result = analysis.run()

# Fresh start
result = analysis.run(mode="fresh")

# Process specific events only
result = analysis.run(events=[0, 1, 2])

# Dry-run with verbose output
result = analysis.run(dry_run=True, verbose=True)
```

### Test Usage (After)

```python
# Test orchestration logic without Snakemake
result = analysis.run(mode="fresh", dry_run=True)
assert result.success
assert "setup" in result.phases_completed
```

---

## References

- `src/TRITON_SWMM_toolkit/cli.py` (lines 365-415): Current CLI translation logic
- `src/TRITON_SWMM_toolkit/analysis.py` (line 1172): Existing `submit_workflow()` method
- `docs/planning/implementation_roadmap.md`: Phase 2 scope
