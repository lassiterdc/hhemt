. # Phase 2 Implementation Prompt for AI Agent

## Context

You are implementing Phase 2 of the SWMM Output Parser Optimization Plan. This phase focuses on core parser optimizations for larger performance gains while keeping output parity.

**Reference Document:** `docs/swmm_output_parser_optimization_plan.md`

---

## Objective

Implement Phase 2 optimizations in `src/TRITON_SWMM_toolkit/swmm_output_parser.py` while preserving output equivalence and warning hygiene established in Phase 1.

---

## ✅ Completed in Phase 1

- Suppressed Zarr V3 string warnings in `utils.write_zarr()`.
- Vectorized `convert_swmm_tdeltas_to_minutes()`.
- Replaced `iterrows()` link_id cleaning with `_clean_link_id()` helper.
- Simplified substring parsing in `return_data_from_rpt()` while preserving newline tokens.
- Removed `Zone.Identifier` artifacts from test data.
- Refactoring and multisim tests pass (including warnings-as-errors).

---

## Phase 2 Tasks: Core Parser Optimization

### Task 1: Optimize `return_data_from_rpt()`

**File:** `src/TRITON_SWMM_toolkit/swmm_output_parser.py`

**Goals:**
- Pre-compile any regex patterns used repeatedly.
- Reduce nested loops where possible.
- Minimize repeated string scanning while keeping error-handling behavior identical.

**Notes:**
- Preserve the newline-token behavior that tests rely on.
- Maintain all existing issue-handling paths (e.g., “two values right next to each other”, orifice conduit handling).

### Task 2: Streamline `return_node_time_series_results_from_rpt()`

**Goals:**
- Reduce redundant passes through `lines`.
- Avoid repeated dictionary lookups when possible.
- Consider using `io.StringIO` or intermediate buffers if it reduces overhead.

### Task 3: Consolidate DataFrame / xarray operations in `return_swmm_outputs()`

**Goals:**
- Reduce number of `to_xarray()` calls where feasible.
- Batch dtype conversions to reduce repeated type coercion overhead.
- Keep join semantics identical (`join="outer"`).

### Task 4: Optimize type conversion helpers

**Goals:**
- Reduce exception-driven loops in `convert_coords_to_dtype()` and `convert_datavars_to_dtype()`.
- Consider pre-determining dtypes for known coordinate/value fields.

---

## Warning Hygiene

Continue to ensure no `Zone.Identifier` files exist in `test_data/swmm_refactoring_reference/*.zarr`:

```bash
find test_data -name '*Zone.Identifier*' -print -delete
```

---

## Testing Requirements

After implementing Phase 2 changes:

### 1. Refactoring Suite
```bash
pytest tests/test_swmm_output_parser_refactoring.py -v
```

### 1a. Baseline Timing
```bash
pytest tests/test_swmm_output_parser_refactoring.py -k retrieve_swmm_outputs_baseline -s
```

### 2. Multi-sim Tests
```bash
pytest tests/test_PC_02_multisim.py -v
```

### 3. Warnings as Errors
```bash
pytest tests/test_PC_02_multisim.py -v -W error::UserWarning
```

---

## Commit Strategy

Make separate commits for each task (or grouped by function):

1. `perf(swmm_output_parser): optimize return_data_from_rpt`
2. `perf(swmm_output_parser): streamline time series rpt parsing`
3. `perf(swmm_output_parser): consolidate dataframe/xarray operations`
4. `perf(swmm_output_parser): reduce dtype conversion overhead`

---

## Success Criteria

- [ ] `pytest tests/test_swmm_output_parser_refactoring.py` passes (all reference comparisons)
- [ ] `pytest tests/test_PC_02_multisim.py` passes with 0 warnings
- [ ] No functional changes to output data
- [ ] Measurable performance improvements vs Phase 1 baseline

**Note:** Phase 2 is complete only when all tests pass and performance improves without altering outputs.

---

## Files Summary

| File | Action |
|------|--------|
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Core optimization changes |
| `docs/swmm_output_parser_optimization_plan.md` | Reference document |
| `docs/next_action_prompt.md` | This prompt |