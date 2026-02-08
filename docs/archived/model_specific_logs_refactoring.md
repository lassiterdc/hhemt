# Model-Specific Logs Refactoring

## Problem Statement

Multi-model concurrent execution creates race conditions when all processes write to the same `log.json`:

```
Process A (TRITON):     load log.json → update → write log.json
Process B (TRITONSWMM): load log.json → update → write log.json  ← OVERWRITES A
Process C (SWMM):       load log.json → update → write log.json  ← OVERWRITES A & B
```

**Result:** Successfully completed processing shows as incomplete because log updates are lost.

## Solution: Model-Specific Logs Only

### New File Structure

```
sims/0-event_id.0/
├── log_triton.json       # TRITON-only model log
├── log_tritonswmm.json   # TRITON-SWMM coupled model log
└── log_swmm.json         # SWMM-only model log
```

**No more `log.json`** - all scenario state is in model-specific logs.

### Advantages

1. **No race conditions** - each process writes to its own file
2. **Consistent field names** - same fields across all logs, differentiated by which log file
3. **Clean API** - `scenario.get_log(model_type)`
4. **Simpler** - no backward compatibility complexity
5. **Minimal log files** - only relevant fields are present
   - Pure TRITON logs don't have SWMM noise
   - Pure SWMM logs don't have TRITON noise
   - Easier to read and debug
6. **Type safety** - Optional fields make it clear which are model-specific

### Log Class Design

Each model type has only the fields relevant to it:

```python
class TRITONSWMM_model_log(cfgBaseModel):
    """
    Processing log for a single model type (triton, tritonswmm, or swmm).

    Fields are Optional and only populated for relevant model types.
    This keeps logs clean and prevents confusion.
    """

    # Common fields (all model types)
    simulation_completed: LogField[bool] = Field(default_factory=LogField)
    sim_run_time_minutes: LogField[float] = Field(default_factory=LogField)
    processing_log: TRITONSWMM_scenario_processing_log = Field(
        default_factory=TRITONSWMM_scenario_processing_log
    )
    run_log: TRITONSWMM_run_log = Field(default_factory=TRITONSWMM_run_log)

    # Performance timeseries (triton and tritonswmm only)
    performance_timeseries_written: Optional[LogField[bool]] = None
    performance_summary_written: Optional[LogField[bool]] = None

    # TRITON outputs (triton and tritonswmm only)
    TRITON_timeseries_written: Optional[LogField[bool]] = None
    TRITON_summary_written: Optional[LogField[bool]] = None

    # SWMM outputs (swmm and tritonswmm only)
    SWMM_node_timeseries_written: Optional[LogField[bool]] = None
    SWMM_link_timeseries_written: Optional[LogField[bool]] = None
    SWMM_node_summary_written: Optional[LogField[bool]] = None
    SWMM_link_summary_written: Optional[LogField[bool]] = None
```

**Field population by model type:**

| Field | triton | tritonswmm | swmm |
|-------|--------|------------|------|
| `simulation_completed` | ✓ | ✓ | ✓ |
| `sim_run_time_minutes` | ✓ | ✓ | ✓ |
| `processing_log` | ✓ | ✓ | ✓ |
| `run_log` | ✓ | ✓ | ✓ |
| `performance_timeseries_written` | ✓ | ✓ | - |
| `performance_summary_written` | ✓ | ✓ | - |
| `TRITON_timeseries_written` | ✓ | ✓ | - |
| `TRITON_summary_written` | ✓ | ✓ | - |
| `SWMM_node_timeseries_written` | - | ✓ | ✓ |
| `SWMM_link_timeseries_written` | - | ✓ | ✓ |
| `SWMM_node_summary_written` | - | ✓ | ✓ |
| `SWMM_link_summary_written` | - | ✓ | ✓ |

**Example log contents:**

```json
// log_triton.json (TRITON-only model)
{
  "simulation_completed": true,
  "sim_run_time_minutes": 15.3,
  "performance_timeseries_written": true,
  "performance_summary_written": true,
  "TRITON_timeseries_written": true,
  "TRITON_summary_written": true
  // No SWMM fields
}

// log_swmm.json (SWMM-only model)
{
  "simulation_completed": true,
  "sim_run_time_minutes": 8.2,
  "SWMM_node_timeseries_written": true,
  "SWMM_link_timeseries_written": true,
  "SWMM_node_summary_written": true,
  "SWMM_link_summary_written": true
  // No TRITON or performance fields
}

// log_tritonswmm.json (Coupled model)
{
  "simulation_completed": true,
  "sim_run_time_minutes": 23.7,
  "performance_timeseries_written": true,
  "performance_summary_written": true,
  "TRITON_timeseries_written": true,
  "TRITON_summary_written": true,
  "SWMM_node_timeseries_written": true,
  "SWMM_link_timeseries_written": true,
  "SWMM_node_summary_written": true,
  "SWMM_link_summary_written": true
  // Has all fields
}
```

### API Design

```python
class TRITONSWMM_scenario:
    def get_log(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> TRITONSWMM_model_log:
        """
        Get the log for a specific model type.

        Initializes only the fields relevant to that model type.
        """
        log_file = self.scen_paths.sim_folder / f"log_{model_type}.json"
        if log_file.exists():
            return TRITONSWMM_model_log.from_json(log_file)

        # Create new log with appropriate fields initialized
        log = TRITONSWMM_model_log(
            logfile=log_file,
            simulation_folder=self.scen_paths.sim_folder
        )

        # Initialize model-specific fields
        if model_type in ("triton", "tritonswmm"):
            # TRITON models need performance and TRITON output fields
            log.performance_timeseries_written = LogField()
            log.performance_summary_written = LogField()
            log.TRITON_timeseries_written = LogField()
            log.TRITON_summary_written = LogField()

        if model_type in ("swmm", "tritonswmm"):
            # SWMM models need SWMM output fields
            log.SWMM_node_timeseries_written = LogField()
            log.SWMM_link_timeseries_written = LogField()
            log.SWMM_node_summary_written = LogField()
            log.SWMM_link_summary_written = LogField()

        return log

    @property
    def model_types_enabled(self) -> list[str]:
        """Get list of enabled model types from system config."""
        enabled = []
        if self._system.cfg_system.toggle_triton_model:
            enabled.append("triton")
        if self._system.cfg_system.toggle_tritonswmm_model:
            enabled.append("tritonswmm")
        if self._system.cfg_system.toggle_swmm_model:
            enabled.append("swmm")
        return enabled
```

### Usage Examples

#### In Runner Scripts

```python
# process_timeseries_runner.py
model_log = scenario.get_log(args.model_type)

# Process outputs
proc.write_timeseries_outputs(...)

# Update model log (no race condition!)
model_log.TRITON_timeseries_written.set(True)
model_log.TRITON_summary_written.set(True)
model_log.write()

# Verify
if not model_log.TRITON_timeseries_written.get():
    logger.error("TRITON timeseries not written")
    return 1
```

#### In Analysis Checks

```python
@property
def TRITON_time_series_not_processed(self):
    scens_not_processed = []
    for event_iloc in self.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, self)

        # Check each enabled model type
        if self._system.cfg_system.toggle_tritonswmm_model:
            log = scen.get_log("tritonswmm")
            if not log.TRITON_timeseries_written.get():
                scens_not_processed.append(str(scen.scen_paths.sim_folder))
                continue

        if self._system.cfg_system.toggle_triton_model:
            log = scen.get_log("triton")
            if not log.TRITON_timeseries_written.get():
                scens_not_processed.append(str(scen.scen_paths.sim_folder))
                continue

    return scens_not_processed
```

## Implementation Plan

### 1. Create New Log Class (`log.py`)

- Define `TRITONSWMM_model_log` class
- Consolidate all processing fields from old `TRITONSWMM_scenario_log`
- Remove model-type prefixes (e.g., `TRITON_only_*` becomes just `TRITON_*`)

### 2. Update Scenario Class (`scenario.py`)

- Add `get_log(model_type)` method
- Remove old `log` property
- Update all internal uses to `get_log(model_type)`
- Add `model_types_enabled` property

### 3. Update Path Definitions (`paths.py`)

- Add model-specific log paths:
  ```python
  log_triton: Path = sim_folder / "log_triton.json"
  log_tritonswmm: Path = sim_folder / "log_tritonswmm.json"
  log_swmm: Path = sim_folder / "log_swmm.json"
  ```
- Remove `f_log` field

### 4. Update Runner Scripts

- `run_simulation_runner.py`: Get model log, update simulation completion
- `process_timeseries_runner.py`: Get model log, update processing completion
- Pass model log to processing classes

### 5. Update Processing Classes (`process_simulation.py`, `run_simulation.py`)

- Accept `model_log` parameter
- Write to `model_log` instead of `self.log`
- Update all log field references

### 6. Update Analysis Class (`analysis.py`)

- Update all properties that check scenario logs:
  - `TRITON_time_series_not_processed`
  - `SWMM_time_series_not_processed`
  - `all_TRITON_timeseries_processed`
  - `all_SWMM_timeseries_processed`
- Use `scen.get_log(model_type)` pattern

### 7. Update Tests

- Update all test fixtures
- Modify assertions to use new log structure
- Test multi-model scenarios specifically

### 8. Clean Up

- Delete old `TRITONSWMM_scenario_log` class
- Remove all `*_only_*` prefixed log fields
- Update CLAUDE.md documentation

## Breaking Changes

**All existing scenario logs will be incompatible.**

Since backward compatibility is explicitly not a goal:
- Old `log.json` files will be ignored
- Existing scenarios will need to be re-run to generate new logs
- This is acceptable for a single-developer codebase

## Testing Strategy

1. **Unit tests**: Test `get_log()` method with each model type
2. **Integration tests**: Run multi-model workflow, verify no race conditions
3. **Regression tests**: Ensure all existing test suites pass

## Success Criteria

1. ✅ No `log.json` in scenario directories
2. ✅ Three model-specific log files per scenario (when all enabled)
3. ✅ No race conditions in multi-model concurrent execution
4. ✅ All tests pass, especially `test_PC_04_multisim_with_snakemake.py`
5. ✅ Clean, consistent API: `scenario.get_log(model_type)`

## Timeline Estimate

- Implementation: ~8 hours
- Testing: ~2 hours
- **Total: ~10 hours**

Much simpler than backward-compatible approach!
