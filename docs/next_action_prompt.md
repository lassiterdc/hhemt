# Phase 5 Implementation Prompt for AI Agent

## Context

You are starting Phase 5 of the SWMM Output Parser Optimization Plan. The focus is on documentation updates, polish, and optional micro-optimizations based on profiling evidence.

**Reference Document:** `docs/swmm_output_parser_optimization_plan.md`

---

## Objective

Finalize Phase 4 documentation, record performance savings, and decide whether to implement optional low-hanging optimizations.

## Phase 5 Tasks: Documentation + Optional Micro-Optimizations

### Task 1: Update Optimization Plan Docs

**Goals:**
- Update `docs/swmm_output_parser_optimization_plan.md` with Phase 4 timings and completion status
- Include the optional low-hanging optimization ideas for future reference

### Task 2: Update Next Action Prompt

**Goals:**
- Replace the Phase 4 prompt with Phase 5 guidance
- Capture the timing results (1.18s baseline, 94.2% savings)

### Task 3 (Optional): Micro-Optimization Follow-ups

**Goals (optional):**
- Remove `sorted(...)` over row indices in `format_rpt_section_into_dataframe`
- Cache `len(lst_col_headers)` in `format_rpt_section_into_dataframe`
- Pre-split lines once in `parse_rpt_single_pass` to reduce repeated `line.split(" ")` calls
- Use `pd.to_datetime(..., format=...)` if date format is fixed

### Task 4: Re-run profiling and report savings (only if Task 3 executed)

**Goals:**
- Run `python -m cProfile -o profiling/retrieve_profile.out profiling/profile_runner.py`
- Run `kernprof -l -v profiling/profile_runner.py`
- Record updated unprofiled runtime and report savings vs 1.18s baseline

---

## Warning Hygiene

Continue to ensure no `Zone.Identifier` files exist in `test_data/swmm_refactoring_reference/*.zarr`:

```bash
find test_data -name '*Zone.Identifier*' -print -delete
```

---

## Testing Requirements

After Phase 5 changes:

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

1. `docs(swmm_output_parser): record phase 4 timings + next steps`
2. `docs(swmm_output_parser): update next action prompt`
3. `perf(swmm_output_parser): optional micro-optimizations` (if Task 3 executed)

---

## Success Criteria

- [ ] Phase 4 timings and savings recorded in optimization plan
- [ ] Next action prompt reflects Phase 5 scope and new baseline
- [ ] If optional micro-optimizations are applied, profiling and tests re-run

**Note:** Phase 5 is complete once documentation is updated and any optional micro-optimizations are validated.

---

## Files Summary

| File | Action |
|------|--------|
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Core optimization changes |
| `docs/swmm_output_parser_optimization_plan.md` | Reference document |
| `docs/next_action_prompt.md` | This prompt |