. # Phase 3 Implementation Prompt for AI Agent

## Context

You are implementing Phase 3 of the SWMM Output Parser Optimization Plan. This phase focuses on advanced parser optimizations for further performance gains while keeping output parity.

**Reference Document:** `docs/swmm_output_parser_optimization_plan.md`

---

## Objective

Implement Phase 3 optimizations in `src/TRITON_SWMM_toolkit/swmm_output_parser.py` while preserving output equivalence and warning hygiene established in Phases 1-2.

---

## ✅ Completed in Phase 1

- Suppressed Zarr V3 string warnings in `utils.write_zarr()`.
- Vectorized `convert_swmm_tdeltas_to_minutes()`.
- Replaced `iterrows()` link_id cleaning with `_clean_link_id()` helper.
- Simplified substring parsing in `return_data_from_rpt()` while preserving newline tokens.
- Removed `Zone.Identifier` artifacts from test data.
- Refactoring and multisim tests pass (including warnings-as-errors).

## ✅ Completed in Phase 2

- Optimized `return_data_from_rpt()` with precompiled regex and cached substring lookups.
- Streamlined `return_node_time_series_results_from_rpt()`.
- Consolidated DataFrame/xarray ops in `return_swmm_outputs()`.
- Reduced exception-driven dtype conversions with numeric prechecks + safe fallbacks.
- Added helper for normal-row selection in RPT parsing.
- Baseline timing: 19.14s vs 20.30s (5.7% savings).

---

## Phase 3 Tasks: Advanced Parser Optimization

### Task 1: Single-pass RPT parser prototype

**Goals:**
- Investigate a state-machine parser that extracts sections in one pass.
- Preserve existing error-handling behavior and newline-token semantics.

### Task 2: Memory usage improvements

**Goals:**
- Explore chunked or memory-mapped reading for large RPT files.
- Ensure behavior matches current parsing logic.

### Task 3: Optional parallelization experiments

**Goals:**
- Evaluate whether node/link time series parsing can be parallelized safely.
- Maintain deterministic outputs and warning hygiene.

---

## Warning Hygiene

Continue to ensure no `Zone.Identifier` files exist in `test_data/swmm_refactoring_reference/*.zarr`:

```bash
find test_data -name '*Zone.Identifier*' -print -delete
```

---

## Testing Requirements

After implementing Phase 3 changes:

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
- [ ] Measurable performance improvements vs Phase 2 baseline

**Note:** Phase 3 is complete only when all tests pass and performance improves without altering outputs.

---

## Files Summary

| File | Action |
|------|--------|
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Core optimization changes |
| `docs/swmm_output_parser_optimization_plan.md` | Reference document |
| `docs/next_action_prompt.md` | This prompt |