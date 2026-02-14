# Workflow Status Reporting Implementation Plan

**Status**: Approved (Tier 3 Phase 2 addition)
**Created**: 2026-02-09
**Purpose**: Provide users visibility into workflow progress to make informed decisions about execution modes

---

## Motivation

Currently, users must guess which mode to use (fresh/resume/overwrite) without knowing what's already been completed. This leads to:
- **Wasteful reruns**: Using `--from-scratch` when most work is done
- **Confusion**: Not knowing if previous run failed or succeeded
- **Trial and error**: Running workflow multiple times to figure out state

**Proposed Solution**: Add `--status` flag that shows completion state, letting users make informed decisions.

---

## User Experience

### CLI Usage

```bash
# Check status before deciding what to do
$ triton-swmm run --status --system-config sys.yaml --analysis-config analysis.yaml

Workflow Status Report
══════════════════════════════════════════════════════════════
Analysis: norfolk_multi_sim
Directory: /path/to/analysis

Phase Status:
  ✓ Setup (system inputs)
    ✓ DEM processed
    ✓ Manning's processed
    ✓ TRITON-SWMM compiled (CPU)
    ✗ TRITON-SWMM compiled (GPU) - not configured

  ✓ Scenario Preparation
    ✓ All 100 scenarios created

  ⚠ Simulation (90% complete)
    ✓ 90 of 100 simulations completed
    ✗ 10 simulations failed or incomplete:
      - Event 23 (event_id.23)
      - Event 47 (event_id.47)
      ... (see full list below)

  ✗ Output Processing (not started)
    ✗ TRITON timeseries: 0 of 100 processed
    ✗ SWMM timeseries: 0 of 100 processed

  ✗ Consolidation (not started)
    ✗ Analysis-level summaries not created

Recommendation:
  Use '--resume' to continue from checkpoint
  Failed simulations will be retried
══════════════════════════════════════════════════════════════

# Then user can make informed decision
$ triton-swmm run --resume --system-config sys.yaml --analysis-config analysis.yaml
```

### API Usage

```python
# Programmatic status check
status = analysis.get_workflow_status()

print(f"Simulations: {status.simulations_completed}/{status.total_simulations}")
print(f"Phase: {status.current_phase}")

if not status.all_complete:
    print(f"Recommendation: {status.recommendation}")
    result = analysis.run(mode=status.recommended_mode)
```

---

## Design

### Data Model

```python
@dataclass
class PhaseStatus:
    """Status of a single workflow phase."""
    name: str  # "setup", "prepare", "simulate", "process", "consolidate"
    complete: bool
    progress: float  # 0.0 to 1.0
    details: dict  # Phase-specific details
    failed_items: List[str]  # Items that failed (e.g., event ilocs)


@dataclass
class WorkflowStatus:
    """Complete workflow status report."""
    analysis_id: str
    analysis_dir: Path

    # Phase statuses
    setup: PhaseStatus
    preparation: PhaseStatus
    simulation: PhaseStatus
    processing: PhaseStatus
    consolidation: PhaseStatus

    # Overall metrics
    total_simulations: int
    simulations_completed: int
    simulations_failed: int
    simulations_pending: int

    # Recommendations
    current_phase: str  # Which phase is partially complete
    recommended_mode: str  # "fresh", "resume", or "overwrite"
    recommendation: str  # Human-readable explanation

    def __str__(self) -> str:
        """Generate formatted status report."""
        # Implementation below
```

### Status Detection Logic

The status checker will inspect:

1. **Setup Phase**:
   - `system.log.dem_processed`
   - `system.log.mannings_processed`
   - `system.log.compilation_tritonswmm_cpu_successful`
   - `system.log.compilation_tritonswmm_gpu_successful`

2. **Scenario Preparation**:
   - `analysis.all_scenarios_created` property
   - `analysis.scenarios_not_created` list

3. **Simulation**:
   - `analysis.all_sims_run` property
   - `analysis.scenarios_not_run` list
   - Check each event's model-specific logs

4. **Output Processing**:
   - `analysis.all_TRITON_timeseries_processed`
   - `analysis.all_SWMM_timeseries_processed`
   - Count processed vs total

5. **Consolidation**:
   - Check for existence of analysis-level summary files
   - `analysis.analysis_paths.output_tritonswmm_triton_summary`
   - etc.

### Recommendation Logic

```python
def generate_recommendation(status: WorkflowStatus) -> tuple[str, str]:
    """Generate recommended mode and explanation.

    Returns
    -------
    (mode, explanation)
    """
    # Nothing completed yet
    if not status.setup.complete:
        return ("fresh", "No previous work detected, starting fresh")

    # Setup done, but simulations failed
    if status.setup.complete and status.simulations_failed > 0:
        return ("resume", f"Resume to retry {status.simulations_failed} failed simulations")

    # Simulations done, processing not started
    if status.simulation.complete and not status.processing.complete:
        return ("resume", "Continue to output processing phase")

    # Everything complete
    if status.consolidation.complete:
        return ("overwrite", "All phases complete. Use 'overwrite' to regenerate outputs")

    # Partial progress in any phase
    return ("resume", "Continue from last checkpoint")
```

---

## Processing Progress (Exact Percentages)

### Rationale

The original implementation used a coarse 0/50/100 heuristic based on whether
TRITON and SWMM outputs were fully processed. For partially processed workflows
(e.g., 59/71 TRITON processed), users see misleading 50% progress. Instead, we
compute exact progress using model-specific missing-output lists.

### Production-Ready Code Chunk (analysis.get_workflow_status)

```python
# Determine enabled models once for processing counts
enabled_models = self._get_enabled_model_types()
triton_enabled = "triton" in enabled_models or "tritonswmm" in enabled_models
swmm_enabled = "swmm" in enabled_models or "tritonswmm" in enabled_models

# Use model-specific, race-condition-safe log checks
triton_missing = len(self.TRITON_time_series_not_processed) if triton_enabled else 0
swmm_missing = len(self.SWMM_time_series_not_processed) if swmm_enabled else 0

# Total scenarios per model
triton_total = n_total if triton_enabled else 0
swmm_total = n_total if swmm_enabled else 0

# Processed counts per model (guard against negatives)
triton_processed = max(triton_total - triton_missing, 0)
swmm_processed = max(swmm_total - swmm_missing, 0)

processed_total = triton_processed + swmm_processed
total_needed = triton_total + swmm_total
proc_progress = processed_total / total_needed if total_needed else 0.0

triton_proc_complete = triton_missing == 0 if triton_enabled else True
swmm_proc_complete = swmm_missing == 0 if swmm_enabled else True
proc_complete = triton_proc_complete and swmm_proc_complete

proc_phase = PhaseStatus(
    name="processing",
    complete=proc_complete,
    progress=proc_progress,
    details={
        "triton": (
            f"{'✓' if triton_proc_complete else '✗'} TRITON outputs processed: "
            f"{triton_processed}/{triton_total}"
            if triton_enabled
            else "✓ TRITON outputs processed: n/a"
        ),
        "swmm": (
            f"{'✓' if swmm_proc_complete else '✗'} SWMM outputs processed: "
            f"{swmm_processed}/{swmm_total}"
            if swmm_enabled
            else "✓ SWMM outputs processed: n/a"
        ),
    },
)
```

---

## Implementation

### Step 1: Add `WorkflowStatus` dataclass to `orchestration.py`

```python
# orchestration.py

@dataclass
class PhaseStatus:
    name: str
    complete: bool
    progress: float = 0.0
    details: dict = field(default_factory=dict)
    failed_items: List[str] = field(default_factory=list)

    def symbol(self) -> str:
        """Return status symbol for display."""
        if self.complete:
            return "✓"
        elif self.progress > 0:
            return "⚠"
        else:
            return "✗"


@dataclass
class WorkflowStatus:
    """Complete workflow status report."""
    analysis_id: str
    analysis_dir: Path

    setup: PhaseStatus
    preparation: PhaseStatus
    simulation: PhaseStatus
    processing: PhaseStatus
    consolidation: PhaseStatus

    total_simulations: int
    simulations_completed: int
    simulations_failed: int = 0
    simulations_pending: int = 0

    current_phase: str = ""
    recommended_mode: str = "resume"
    recommendation: str = ""

    def __str__(self) -> str:
        """Generate formatted status report."""
        lines = [
            "",
            "Workflow Status Report",
            "═" * 66,
            f"Analysis: {self.analysis_id}",
            f"Directory: {self.analysis_dir}",
            "",
            "Phase Status:",
        ]

        for phase in [self.setup, self.preparation, self.simulation,
                      self.processing, self.consolidation]:
            symbol = phase.symbol()
            progress = f" ({phase.progress*100:.0f}% complete)" if 0 < phase.progress < 1 else ""
            lines.append(f"  {symbol} {phase.name.title()}{progress}")

            for key, value in phase.details.items():
                lines.append(f"    {value}")

            if phase.failed_items:
                n_show = min(3, len(phase.failed_items))
                lines.append(f"    ✗ {len(phase.failed_items)} failed:")
                for item in phase.failed_items[:n_show]:
                    lines.append(f"      - {item}")
                if len(phase.failed_items) > n_show:
                    lines.append(f"      ... and {len(phase.failed_items) - n_show} more")

        lines.extend([
            "",
            "Recommendation:",
            f"  {self.recommendation}",
            "═" * 66,
            "",
        ])

        return "\n".join(lines)
```

### Step 2: Add `get_workflow_status()` method to Analysis class

```python
# analysis.py

def get_workflow_status(self) -> WorkflowStatus:
    """Generate workflow status report.

    Inspects logs and outputs to determine completion state of each phase.

    Returns
    -------
    WorkflowStatus
        Structured status report with phase details and recommendations

    Examples
    --------
    >>> status = analysis.get_workflow_status()
    >>> print(status)
    >>> if not status.simulation.complete:
    ...     print(f"Retry {len(status.simulation.failed_items)} failed sims")
    """
    from .orchestration import WorkflowStatus, PhaseStatus

    # Check setup phase
    system_log = self._system.log
    dem_done = system_log.dem_processed.get()
    mannings_done = (
        self._system.cfg_system.toggle_use_constant_mannings
        or system_log.mannings_processed.get()
    )
    compiled = system_log.compilation_tritonswmm_cpu_successful.get()

    setup_complete = dem_done and mannings_done and compiled
    setup_details = {
        "dem": f"{'✓' if dem_done else '✗'} DEM processed",
        "mannings": f"{'✓' if mannings_done else '✗'} Manning's processed",
        "compiled": f"{'✓' if compiled else '✗'} TRITON-SWMM compiled",
    }

    setup_phase = PhaseStatus(
        name="setup",
        complete=setup_complete,
        progress=1.0 if setup_complete else 0.5 if (dem_done or compiled) else 0.0,
        details=setup_details,
    )

    # Check scenario preparation
    all_prepared = self.all_scenarios_created
    not_prepared = self.scenarios_not_created
    n_total = len(self.df_sims)
    n_prepared = n_total - len(not_prepared)

    prep_phase = PhaseStatus(
        name="preparation",
        complete=all_prepared,
        progress=n_prepared / n_total if n_total > 0 else 0.0,
        details={"scenarios": f"{'✓' if all_prepared else '⚠'} {n_prepared}/{n_total} scenarios created"},
        failed_items=[str(p) for p in not_prepared],
    )

    # Check simulations
    all_run = self.all_sims_run
    not_run = self.scenarios_not_run
    n_run = n_total - len(not_run)

    sim_phase = PhaseStatus(
        name="simulation",
        complete=all_run,
        progress=n_run / n_total if n_total > 0 else 0.0,
        details={"sims": f"{'✓' if all_run else '⚠'} {n_run}/{n_total} simulations completed"},
        failed_items=[str(p) for p in not_run],
    )

    # Check processing (placeholder - would need to check each scenario)
    # For now, use log fields as proxy
    triton_proc = self.log.all_TRITON_timeseries_processed.get()
    swmm_proc = self.log.all_SWMM_timeseries_processed.get()
    proc_complete = triton_proc and swmm_proc

    proc_phase = PhaseStatus(
        name="processing",
        complete=proc_complete,
        progress=1.0 if proc_complete else 0.5 if (triton_proc or swmm_proc) else 0.0,
        details={
            "triton": f"{'✓' if triton_proc else '✗'} TRITON outputs processed",
            "swmm": f"{'✓' if swmm_proc else '✗'} SWMM outputs processed",
        },
    )

    # Check consolidation
    # Check if analysis-level summary files exist
    summaries_exist = (
        self.analysis_paths.output_tritonswmm_triton_summary
        and self.analysis_paths.output_tritonswmm_triton_summary.exists()
    )

    consol_phase = PhaseStatus(
        name="consolidation",
        complete=summaries_exist,
        progress=1.0 if summaries_exist else 0.0,
        details={"summaries": f"{'✓' if summaries_exist else '✗'} Analysis summaries created"},
    )

    # Determine current phase and recommendation
    if not setup_complete:
        current = "setup"
        rec_mode = "fresh"
        rec_text = "Setup incomplete. Recommend 'fresh' mode to process system inputs."
    elif not all_prepared:
        current = "preparation"
        rec_mode = "resume"
        rec_text = f"Resume to create {len(not_prepared)} remaining scenarios."
    elif not all_run:
        current = "simulation"
        rec_mode = "resume"
        rec_text = f"Resume to run {len(not_run)} pending/failed simulations."
    elif not proc_complete:
        current = "processing"
        rec_mode = "resume"
        rec_text = "Resume to process simulation outputs."
    elif not summaries_exist:
        current = "consolidation"
        rec_mode = "resume"
        rec_text = "Resume to consolidate analysis summaries."
    else:
        current = "complete"
        rec_mode = "overwrite"
        rec_text = "All phases complete. Use 'overwrite' to regenerate outputs if needed."

    return WorkflowStatus(
        analysis_id=self.cfg_analysis.analysis_id,
        analysis_dir=self.analysis_paths.analysis_dir,
        setup=setup_phase,
        preparation=prep_phase,
        simulation=sim_phase,
        processing=proc_phase,
        consolidation=consol_phase,
        total_simulations=n_total,
        simulations_completed=n_run,
        simulations_failed=len(not_run),
        simulations_pending=0,  # Would need more logic to distinguish failed vs pending
        current_phase=current,
        recommended_mode=rec_mode,
        recommendation=rec_text,
    )
```

### Step 3: Add `--status` flag to CLI

```python
# cli.py

status_flag: bool = typer.Option(
    False,
    "--status",
    help="Show workflow status and exit (no execution)",
),

# In run_command, after loading configs but before execution:

if status_flag:
    workflow_status = analysis.get_workflow_status()
    console.print(str(workflow_status))
    raise typer.Exit(0)
```

### Step 4: Update `analysis.run()` to support status checking

Remove "auto" mode, add status method reference in docstring:

```python
def run(
    self,
    mode: Literal["fresh", "resume", "overwrite"] = "resume",
    # ... rest of params
):
    """
    ...

    To determine which mode to use, check workflow status first:

    >>> status = analysis.get_workflow_status()
    >>> print(status.recommendation)
    >>> result = analysis.run(mode=status.recommended_mode)

    ...
    """
```

---

## Testing Strategy

1. **Unit tests** for `get_workflow_status()`:
   - Mock log states for each phase
   - Verify correct progress calculations
   - Test recommendation logic

2. **CLI tests** for `--status` flag:
   - Test with empty analysis
   - Test with partially complete analysis
   - Test with fully complete analysis

3. **Integration tests**:
   - Run workflow partway, check status, resume
   - Verify status accurately reflects Snakemake state

---

## Documentation Updates

### priorities.md

Add to Tier 3 Phase 2:
```markdown
- [x] **Shared orchestration core** (Phase 2 of implementation roadmap)
  - ✅ High-level analysis.run() API with mode translation
  - ✅ WorkflowResult structured return type
  - ✅ Workflow status reporting (--status flag)
  - _Ref:_ `docs/planning/implementation_roadmap.md`, `workflow_status_reporting_plan.md`
```

### cli_command_spec.md

Add `--status` flag documentation:
```markdown
## Execution Control Flags

...

### --status

**Type**: boolean flag
**Default**: false (execute workflow)
**Mutually exclusive with**: --dry-run

Display workflow status report and exit without executing.

Shows completion state of all phases:
- Setup (system inputs, compilation)
- Scenario preparation
- Simulation execution
- Output processing
- Result consolidation

Includes recommendation for which mode to use (fresh/resume/overwrite).

**Example**:
```bash
triton-swmm run --status --system-config sys.yaml --analysis-config analysis.yaml
```
```

### shared_orchestration_design.md

Update to remove "auto" mode, add status reporting section.

---

## Benefits

1. **Better UX**: Users see exactly what's done and what's pending
2. **Informed decisions**: Clear recommendations prevent wasteful reruns
3. **Debugging aid**: Status report helps diagnose stuck workflows
4. **API parity**: Same status checking available in notebooks
5. **No guesswork**: Eliminates need for opaque "auto" mode

---

## Future Enhancements

1. **Progress bars**: Show real-time progress during execution
2. **Time estimates**: Predict remaining execution time based on history
3. **Resource usage**: Show CPU/GPU utilization from logs
4. **Error summary**: Aggregate common failure reasons
5. **JSON output**: Add `--status --json` for machine-readable format

---

## References

- `src/TRITON_SWMM_toolkit/analysis.py`: Properties for checking completion state
- `src/TRITON_SWMM_toolkit/log.py`: Log fields used for status detection
- `docs/planning/shared_orchestration_design.md`: Orchestration API design
- `docs/planning/cli_command_spec.md`: CLI flag specifications
