 # Phase 4 Implementation Prompt for AI Agent

## Context

You are starting Phase 4 of the SWMM Output Parser Optimization Plan. The focus is on reducing pandas overhead in RPT parsing based on profiling evidence.

**Reference Document:** `docs/swmm_output_parser_optimization_plan.md`

---

## Objective

Refactor dataframe construction in `src/TRITON_SWMM_toolkit/swmm_output_parser.py` to reduce per-row pandas overhead, while preserving output parity.

## Phase 4 Tasks: DataFrame Construction Refactor

### Task 1: Refactor `format_rpt_section_into_dataframe`

**Goals:**
- Replace per-row Series construction with row lists/dicts
- Use `DataFrame.from_records` or `DataFrame` once per section
- Avoid `.to_frame().T` and `.iloc` inside loops

### Task 2: Streamline `return_data_from_rpt`

**Goals:**
- Replace pandas-based mode calculation with `collections.Counter`
- Track problematic rows using list indices
- Preserve error correction behavior and warnings

### Task 3: Optimize `create_tseries_ds`

**Goals:**
- Convert `date_time` in one batch per combined DataFrame
- Minimize repeated `pd.concat` calls

### Task 4: Re-run profiling and report savings

**Goals:**
- Run `python -m cProfile -o profiling/retrieve_profile.out profiling/profile_runner.py`
- Run `kernprof -l -v profiling/profile_runner.py`
- Record updated unprofiled runtime and report savings vs 18.29s baseline

---

## Warning Hygiene

Continue to ensure no `Zone.Identifier` files exist in `test_data/swmm_refactoring_reference/*.zarr`:

```bash
find test_data -name '*Zone.Identifier*' -print -delete
```

---

## Testing Requirements

After Phase 4 changes:

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

1. `perf(swmm_output_parser): reduce pandas overhead in rpt parsing`
2. `perf(swmm_output_parser): optimize tseries dataset creation`
3. `test(swmm_output_parser): confirm refactor parity + timing`

---

## Success Criteria

- [ ] `pytest tests/test_swmm_output_parser_refactoring.py` passes (all reference comparisons)
- [ ] `pytest tests/test_PC_02_multisim.py` passes with 0 warnings
- [ ] No functional changes to output data (excluding removal of static geodataframe vars)
- [ ] Performance should not regress vs Phase 2 baseline (target improvements optional)

**Note:** Phase 4 is complete only when all tests pass and performance improves without altering outputs.

---

## Files Summary

| File | Action |
|------|--------|
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Core optimization changes |
| `docs/swmm_output_parser_optimization_plan.md` | Reference document |
| `docs/next_action_prompt.md` | This prompt |