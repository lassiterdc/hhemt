# PC04 Review Fixes — Implementation Guide

## Status

- **Phase A (Diagnostic Run)**: COMPLETED — test passed with `clear_raw_outputs=False` (375s).
  - Need to inspect output directories to confirm where files land.
  - The test was reverted back to `clear_raw_outputs=True`.
- **Phase B (Code Changes)**: NOT STARTED

## Remaining Phase A Work

Run this manually to inspect the output directories:

```bash
SCEN_DIR="test_data/norfolk_coastal_flooding/tests/multi_sim/sims/0-event_id.0"

# Key directories to check:
ls -laR "$SCEN_DIR/out_triton/"         # TRITON-only raw outputs
ls -laR "$SCEN_DIR/out_tritonswmm/"     # TRITON-SWMM coupled raw outputs
ls -laR "$SCEN_DIR/out_swmm/"           # SWMM-only raw outputs
ls -laR "$SCEN_DIR/output/"             # Legacy/bug output location
ls -la  "$SCEN_DIR/logs/"               # Log files
```

**If the test data was already cleared**, re-run with `clear_raw_outputs=False` first:
```bash
# In test_PC_04, line ~202, change clear_raw_outputs=True → False, then:
pytest tests/test_PC_04_multisim_with_snakemake.py -k test_snakemake_workflow_end_to_end -q
# Then inspect, then revert the change
```

**What we expect to find:**
1. `out_triton/bin/` — H, QX, QY, MH binary files (TRITON-only raw outputs)
2. `out_tritonswmm/bin/` — H, QX, QY, MH binary files (coupled model raw outputs)
3. `output/swmm/hydraulics.out` and `output/swmm/hydraulics.rpt` — TRITON-SWMM's SWMM outputs (BUG: should go to `out_tritonswmm/`)
4. `output/log.out` — hardcoded by TRITON (BUG: both models overwrite same file)

**Decision tree for Step 2 (path simplification):**
- If `out_tritonswmm/bin/` has TRITON raw outputs → simplify `_triton_swmm_raw_output_directory` to use configured paths directly
- If `output/bin/` has TRITON raw outputs instead → keep the multi-candidate search but document it

---

## Phase B: Code Changes (Complete Specifications)

### Step 1: Create `docs/implementation/triton_output_path_bug.md`

**New file.** Contents:

```markdown
# TRITON Output Path Bug

## Bug Description

TRITON-SWMM (the external C++ executable) writes certain output files to hardcoded paths,
ignoring the `output_folder` directive in the CFG configuration file.

### Affected Outputs

| Output | Expected Location | Actual Location | Status |
|--------|------------------|-----------------|--------|
| TRITON raw outputs (H, QX, QY, MH binaries) | `out_tritonswmm/bin/` | `out_tritonswmm/bin/` | ✅ Correct |
| TRITON-only raw outputs | `out_triton/bin/` | `out_triton/bin/` | ✅ Correct |
| Coupled SWMM outputs (.out, .rpt) | `out_tritonswmm/` | `sim_folder/output/swmm/` | ❌ Bug |
| `log.out` (TRITON log) | Per-model directory | `sim_folder/output/log.out` | ❌ Bug |

### `log.out` Overwrite Issue

When both TRITON-only and TRITON-SWMM models are enabled (multi-model mode), both write
to `sim_folder/output/log.out`. The last model to finish overwrites the other's log. This
means resource-usage parsing (nTasks, OMP threads, GPUs, backend) may be incorrect for
one model type.

## Workaround Convention

All workaround code is tagged with:
```
TODO(TRITON-OUTPUT-PATH-BUG)
```

### Affected Code Locations

- `src/TRITON_SWMM_toolkit/run_simulation.py` — `coupled_swmm_output_file` property
- `src/TRITON_SWMM_toolkit/process_simulation.py` — `_export_SWMM_outputs` tritonswmm branch
- `src/TRITON_SWMM_toolkit/process_simulation.py` — `_clear_raw_SWMM_outputs` tritonswmm branch
- `src/TRITON_SWMM_toolkit/analysis.py` — `log_out_path` (~line 1270)
- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` — `log_out_path` (~line 663)

## Resolution

Once the TRITON-SWMM developer fixes the executable to respect the `output_folder` directive:
1. Search for `TODO(TRITON-OUTPUT-PATH-BUG)` across the codebase
2. Remove all workaround code
3. Delete this document
```

---

### Step 2: Simplify TRITON raw output path resolution

**Depends on Phase A diagnostic results.** If confirmed that TRITON raw outputs go to configured directories:

#### File: `src/TRITON_SWMM_toolkit/run_simulation.py`

**Replace `_triton_swmm_raw_output_directory` (lines 30-46):**

Current code searches 4 candidate directories. Replace with:

```python
@property
def _triton_swmm_raw_output_directory(self):
    """Directory containing raw TRITON outputs from the TRITON-SWMM coupled model."""
    raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type
    out_dir = self._scenario.scen_paths.out_tritonswmm
    if out_dir is not None and out_dir.exists():
        raw_dir = out_dir / raw_type
        if raw_dir.exists() and any(raw_dir.iterdir()):
            return out_dir
    # Fallback for legacy directory structure
    fallback = self._scenario.scen_paths.sim_folder / "output"
    if fallback.exists():
        raw_dir = fallback / raw_type
        if raw_dir.exists() and any(raw_dir.iterdir()):
            return fallback
    return out_dir if out_dir is not None else fallback
```

**Replace `raw_triton_output_dir` (lines 49-55):**

```python
@property
def raw_triton_output_dir(self):
    """Directory containing raw TRITON binary output files (H, QX, QY, MH)."""
    raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type
    base = self._triton_swmm_raw_output_directory
    raw_dir = base / raw_type
    if raw_dir.exists() and any(raw_dir.iterdir()):
        return raw_dir
    return base
```

#### File: `src/TRITON_SWMM_toolkit/process_simulation.py`

**`_export_TRITONSWMM_TRITON_outputs` (lines 517-531):** Replace inline candidate search with:

```python
fldr_out_triton = self._run.raw_triton_output_dir
```

Remove lines 517-531 (the `output_candidates` list and for-loop), replace with:

```python
raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
fldr_out_triton = self._run.raw_triton_output_dir
```

Keep the rest of the method the same but change the error handling at line 533 (see Step 4a).

**`_export_TRITON_only_outputs` (lines 601-617):** Replace inline candidate search with:

```python
raw_out_type = self._analysis.cfg_analysis.TRITON_raw_output_type
out_triton = self._scenario.scen_paths.out_triton
if out_triton is None:
    raise FileNotFoundError(
        "out_triton path is None. Ensure TRITON-only model is enabled in system config."
    )
fldr_out_triton = out_triton / raw_out_type
```

Remove lines 601-636 and replace with the above plus the existing `_already_written` check afterwards.

---

### Step 3: Add `coupled_swmm_output_file` property

#### File: `src/TRITON_SWMM_toolkit/run_simulation.py`

Add after `model_types_enabled` property (after line 98):

```python
@property
def coupled_swmm_output_file(self) -> Path | None:
    """Locate the SWMM output file from a TRITON-SWMM coupled run.

    TODO(TRITON-OUTPUT-PATH-BUG): TRITON-SWMM writes SWMM outputs to
    output/swmm/ regardless of the CFG output_folder directive.
    See docs/implementation/triton_output_path_bug.md
    """
    # Check bug-location first (most common case)
    alt_swmm_dir = self._scenario.scen_paths.sim_folder / "output" / "swmm"
    alt_out = alt_swmm_dir / "hydraulics.out"
    alt_rpt = alt_swmm_dir / "hydraulics.rpt"
    if alt_out.exists():
        return alt_out
    if alt_rpt.exists():
        return alt_rpt
    # Fall back to configured location
    configured = self._scenario.scen_paths.swmm_hydraulics_rpt
    if configured is not None and configured.exists():
        return configured
    return None
```

---

### Step 4: Fail-fast fixes in `process_simulation.py`

#### 4a. `_export_TRITONSWMM_TRITON_outputs` (lines 494-579)

**Remove redundant existence guard (lines 506-511):**

Current code (lines 506-515):
```python
if not overwrite_if_exist and fname_out.exists():  # <-- REMOVE this block
    if verbose:
        print(f"{fname_out.name} already exists. Skipping reprocessing.")
    if clear_raw_outputs:
        self._clear_raw_TRITON_outputs()
    return
if self._already_written(fname_out) and not overwrite_if_exist:  # <-- KEEP this
    if verbose:
        print(f"{fname_out.name} already written. Not overwriting.")
    return
```

Delete lines 506-511 (the `fname_out.exists()` check). Keep only lines 512-515 (the `_already_written` check).

**Line 533 — Change silent return to error:**

Current:
```python
if fldr_out_triton is None:
    if verbose:
        print(
            "Raw TRITON-SWMM outputs missing. Skipping TRITON processing for this run.",
            flush=True,
        )
    return
```

Replace with:
```python
if fldr_out_triton is None or not fldr_out_triton.exists():
    raise FileNotFoundError(
        f"Raw TRITON-SWMM outputs not found at {fldr_out_triton}. "
        "Ensure the TRITON-SWMM coupled simulation completed and wrote outputs to "
        f"the configured output directory."
    )
```

(Note: After Step 2 changes, `fldr_out_triton` comes from `self._run.raw_triton_output_dir` which won't be None, but add the check for safety.)

**Line 549 — Change silent return to error:**

Current:
```python
if df_outputs.empty:
    if verbose:
        print(
            "Raw TRITON-SWMM outputs missing. Skipping TRITON processing for this run.",
            flush=True,
        )
    return
```

Replace with:
```python
if df_outputs.empty:
    raise FileNotFoundError(
        f"No TRITON output files (H, QX, QY, MH) found in {fldr_out_triton}. "
        "Ensure the TRITON-SWMM coupled simulation completed successfully."
    )
```

#### 4b. `_export_SWMM_outputs` tritonswmm branch (lines 756-767)

Current code (lines 756-767):
```python
f_inp = self.scen_paths.swmm_hydraulics_inp
swmm_timeseries_result_file = self.scen_paths.swmm_hydraulics_rpt
alt_swmm_dir = self.scen_paths.sim_folder / "output" / "swmm"
alt_out = alt_swmm_dir / "hydraulics.out"
alt_rpt = alt_swmm_dir / "hydraulics.rpt"
if alt_out.exists():
    swmm_timeseries_result_file = alt_out
elif (
    swmm_timeseries_result_file is None
    or not swmm_timeseries_result_file.exists()
) and alt_rpt.exists():
    swmm_timeseries_result_file = alt_rpt
```

Replace with:
```python
f_inp = self.scen_paths.swmm_hydraulics_inp
# TODO(TRITON-OUTPUT-PATH-BUG): TRITON-SWMM writes SWMM outputs to output/swmm/
# regardless of config. See docs/implementation/triton_output_path_bug.md
swmm_timeseries_result_file = self._run.coupled_swmm_output_file
if swmm_timeseries_result_file is None:
    raise FileNotFoundError(
        "Cannot find SWMM output file from TRITON-SWMM coupled run. "
        f"Checked: output/swmm/hydraulics.out, output/swmm/hydraulics.rpt, "
        f"and configured path {self.scen_paths.swmm_hydraulics_rpt}. "
        "Ensure the TRITON-SWMM simulation completed successfully."
    )
```

#### 4c. `_clear_raw_SWMM_outputs` tritonswmm branch (lines 1037-1045)

Current code (lines 1037-1045):
```python
else:  # model == "tritonswmm"
    swmm_out_file = self.scen_paths.swmm_hydraulics_rpt
    if swmm_out_file is None or not swmm_out_file.exists():
        alt_swmm_dir = self.scen_paths.sim_folder / "output" / "swmm"
        alt_rpt = alt_swmm_dir / "hydraulics.rpt"
        alt_out = alt_swmm_dir / "hydraulics.out"
        swmm_out_file = alt_rpt if alt_rpt.exists() else alt_out
    if swmm_out_file is None:
        return  # No coupled SWMM outputs to clear
```

Replace with:
```python
else:  # model == "tritonswmm"
    # TODO(TRITON-OUTPUT-PATH-BUG): TRITON-SWMM writes SWMM outputs to output/swmm/
    # regardless of config. See docs/implementation/triton_output_path_bug.md
    swmm_out_file = self._run.coupled_swmm_output_file
    if swmm_out_file is None:
        return  # No coupled SWMM outputs to clear
```

Also simplify the `swmm` branch (lines 1033-1036):

Current:
```python
if model == "swmm":
    swmm_out_file = self.scen_paths.swmm_full_out_file
    if swmm_out_file is None:
        return  # No standalone SWMM outputs to clear
```

This is correct as-is (swmm_full_out_file is a required field when SWMM is enabled, but the None check is a reasonable safety guard for this destructive operation). Keep it.

---

### Step 5: Add TODO tags to `log.out` references

#### File: `src/TRITON_SWMM_toolkit/analysis.py` (line ~1268-1270)

Add comment BEFORE the existing comment:
```python
            # TODO(TRITON-OUTPUT-PATH-BUG): log.out is hardcoded to output/ by TRITON.
            # When both TRITON-only and TRITON-SWMM run, last to finish overwrites
            # the other's log.out. See docs/implementation/triton_output_path_bug.md
            # Parse log.out file for actual resource usage
            # log.out is written to the output directory (same location as performance.txt)
            log_out_path = scen.scen_paths.sim_folder / "output" / "log.out"
```

#### File: `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` (line ~661-663)

Same pattern — add comment BEFORE the existing comment:
```python
                # TODO(TRITON-OUTPUT-PATH-BUG): log.out is hardcoded to output/ by TRITON.
                # When both TRITON-only and TRITON-SWMM run, last to finish overwrites
                # the other's log.out. See docs/implementation/triton_output_path_bug.md
                # Parse log.out file for actual resource usage
                # log.out is written to the output directory (same location as performance.txt)
                log_out_path = scen.scen_paths.sim_folder / "output" / "log.out"
```

---

### Step 6: Extract capacity normalization helper

#### File: `tests/utils_for_testing.py`

Add at end of file:

```python
def normalize_swmm_link_vars(link_vars: set[str]) -> set[str]:
    """Normalize SWMM link variable names for cross-parser comparison.

    The .out binary parser (pyswmm) reports 'capacity' while the .rpt text parser
    reports 'capacity_setting'. Both represent the same physical quantity: the
    fraction of conduit filled (0-1 range). This normalizes to 'capacity_setting'.
    """
    if "capacity" in link_vars:
        return (link_vars - {"capacity"}) | {"capacity_setting"}
    return link_vars
```

#### File: `tests/test_PC_01_singlesim.py` (lines 156-162)

Current code:
```python
        # Normalize known naming differences before comparing
        if "capacity" in swmm_link_vars:
            swmm_link_vars = (swmm_link_vars - {"capacity"}) | {"capacity_setting"}
        if "capacity_setting" in tritonswmm_link_vars:
            tritonswmm_link_vars = (tritonswmm_link_vars - {"capacity_setting"}) | {
                "capacity_setting"
            }
```

Note the **no-op bug** on lines 159-162: it checks if `"capacity_setting"` is in the set, then replaces `"capacity_setting"` with `"capacity_setting"` — doing nothing. The intent was to check for `"capacity"` (the binary parser name).

Replace with:
```python
        # Normalize known naming differences before comparing
        swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
        tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)
```

#### File: `tests/test_PC_04_multisim_with_snakemake.py` (lines 289-295)

Current code:
```python
            # Normalize known naming differences before comparing
            if "capacity" in swmm_link_vars:
                swmm_link_vars = (swmm_link_vars - {"capacity"}) | {"capacity_setting"}
            if "capacity" in tritonswmm_link_vars:
                tritonswmm_link_vars = (tritonswmm_link_vars - {"capacity"}) | {
                    "capacity_setting"
                }
```

Replace with:
```python
            # Normalize known naming differences before comparing
            swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
            tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)
```

---

### Step 7: Update CLAUDE.md

#### 7a. Add philosophy section after "Backward Compatibility" (after line 69, before `## Architecture` on line 71)

Insert:

```markdown
### Completion Status: Log-Based Checks over File Existence

**Prefer log-based checks over file existence checks for determining processing completion.**

- `_already_written()` verifies a file was written *successfully*, not just that it exists
- A file may exist but be corrupt, incomplete, or from a previous failed run
- File existence checks are redundant when log checks are available and can mask errors
- Exception: File existence is appropriate for verifying *input* files before reading them
```

#### 7b. Add gotchas 5 and 6 (after line 395, before `## Specialized Agent Documentation`)

Insert:

```markdown
5. **TRITON-SWMM SWMM output path bug** — TRITON-SWMM writes SWMM outputs to `sim_folder/output/swmm/` and `log.out` to `sim_folder/output/` regardless of CFG `output_folder`. Workarounds tagged `TODO(TRITON-OUTPUT-PATH-BUG)`. See `docs/implementation/triton_output_path_bug.md`.

6. **`log.out` overwrite with multi-model** — Both TRITON-only and TRITON-SWMM write to `sim_folder/output/log.out`. Last to finish overwrites the other. Resource-usage parsing may be incorrect for one model type.
```

---

## Verification After All Changes

```bash
# Smoke test (faster)
pytest tests/test_PC_02_multisim.py -q

# Primary E2E (the test that was previously broken)
pytest tests/test_PC_04_multisim_with_snakemake.py -k test_snakemake_workflow_end_to_end -q
```

---

## Files Summary

| File | Change Type | Step |
|------|-------------|------|
| `docs/implementation/triton_output_path_bug.md` | **NEW** | 1 |
| `src/.../run_simulation.py` | Simplify path resolution; add `coupled_swmm_output_file`; TODO tags | 2, 3 |
| `src/.../process_simulation.py` | Remove redundant guard; silent returns → errors; delegate to centralized paths; simplify cleanup; TODO tags | 2, 4 |
| `src/.../analysis.py` | TODO tag on `log.out` reference | 5 |
| `src/.../sensitivity_analysis.py` | TODO tag on `log.out` reference | 5 |
| `tests/utils_for_testing.py` | Add `normalize_swmm_link_vars()` | 6 |
| `tests/test_PC_01_singlesim.py` | Use helper; fix no-op bug | 6 |
| `tests/test_PC_04_multisim_with_snakemake.py` | Use helper | 6 |
| `CLAUDE.md` | Log-based check philosophy; 2 new gotchas | 7 |

---

## Key Context for Fresh Session

### Architecture reminder
- `TRITONSWMM_run` (in `run_simulation.py`) wraps a `TRITONSWMM_scenario` and handles simulation execution
- `TRITONSWMM_sim_post_processing` (in `process_simulation.py`) wraps a `TRITONSWMM_run` and handles output processing
- `self._run.coupled_swmm_output_file` (new property) will be called from `process_simulation.py` via `self._run`
- The `scen_paths` object has all the configured paths (e.g., `out_triton`, `out_tritonswmm`, `swmm_hydraulics_rpt`)

### The no-op bug in test_PC_01 (lines 159-162)
```python
# This does NOTHING — checks for "capacity_setting", then replaces it with itself
if "capacity_setting" in tritonswmm_link_vars:
    tritonswmm_link_vars = (tritonswmm_link_vars - {"capacity_setting"}) | {
        "capacity_setting"
    }
```
Should have checked for `"capacity"` (the binary parser name) and replaced with `"capacity_setting"`.
The `normalize_swmm_link_vars` helper fixes this.

### capacity vs capacity_setting
- `.out` binary parser (pyswmm) → reports as `"capacity"`
- `.rpt` text parser → reports as `"capacity_setting"`
- Both are the same physical quantity: fraction of conduit filled (0-1 range)
- Safe to normalize `"capacity"` → `"capacity_setting"` everywhere
