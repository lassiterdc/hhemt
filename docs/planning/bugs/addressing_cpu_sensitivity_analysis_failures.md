# Implementation Plan: Addressing CPU Sensitivity Analysis Failures

**Plan Date**: 2026-02-23  
**Related Debug Report**: `.debugging/test_uva_sensitivity_suite_cpu/debugging_report_20260223_213530.md`

## Task Understanding

### Requirements
- Address the `test_uva_sensitivity_suite_cpu` failures where sensitivity simulations fail in `batch_job` mode on UVA.
- Prioritize fixes for the primary failure mode (`TIME LIMIT` and simulation non-completion across all sub-analyses).
- Include follow-on hardening for observed secondary failures (Snakemake status-query/teardown instability and ambiguous model completion detection).
- Keep changes aligned with existing architecture (config -> workflow generation -> runner scripts -> status tracking).

### Assumptions
- The highest-impact immediate fix is resource/time policy and workflow scheduling behavior, not model physics or compilation.
- The debug bundle accurately reflects the failed run even though `workflow_summary.md` was missing.
- We should preserve existing configuration schema where practical, and avoid introducing backwards-compatibility shims.
- HPC validation must be coordinated with the user on UVA (cannot be fully validated locally).

### Success criteria
- CPU sensitivity workflow can complete simulation phase for sub-analyses previously failing from 2-minute runtime limits.
- No broad `TIMEOUT` pattern across simulation rules under realistic walltimes.
- Reduced incidence of post-failure Snakemake plugin/query errors in the same run envelope.
- Clear guardrails exist in code/config/tests to prevent recurrence (especially in fixtures and generated workflows).

## Evidence from Codebase

- `.debugging/test_uva_sensitivity_suite_cpu/debugging_report_20260223_213530.md`
  - Confirms `prepare=20/20`, `simulation=0/20`; primary pattern is `runtime=2` with time-limit cancellations.
- `src/TRITON_SWMM_toolkit/workflow.py`
  - `generate_master_snakefile_content()` and `generate_snakefile_content()` propagate `hpc_time_min_per_sim` directly into simulation rule `resources.runtime`.
  - Sensitivity simulation rules (`simulation_sa*_evt*`) inherit per-sub-analysis runtime but currently allow too-aggressive settings unchanged.
  - `batch_job` uses tmux-orchestrated Snakemake with slurm executor and efficiency report options.
- `src/TRITON_SWMM_toolkit/config/analysis.py`
  - Validates `hpc_time_min_per_sim >= 1` for `batch_job`, but no minimum safety threshold by mode/profile.
- `src/TRITON_SWMM_toolkit/run_simulation.py`
  - Completion check is log-marker based; emits runtime warning for ambiguous completion (`performance.txt exists` while log success false).
  - Runner launches via `bash -lc` and `srun`; timeout/cancellation can cascade into `srun launch failed` errors.
- `src/TRITON_SWMM_toolkit/resource_management.py`
  - Resource manager calculates max demands and concurrency constraints but does not enforce policy limits for per-simulation walltime realism.
- `tests/fixtures/test_case_catalog.py`
  - Explicitly sets `hpc_time_min_per_sim: 2` in UVA CPU/GPU sensitivity suite fixtures, matching observed failure trigger.
- `tests/test_PC_05_sensitivity_analysis_with_snakemake.py`, `tests/test_UVA_03_sensitivity_analysis_with_snakemake.py`, `tests/test_UVA_04_multiCPU_sensitivity_analysis_minirun.py`
  - Existing sensitivity workflow generation/execution tests are appropriate extension points for regression coverage.

## Implementation Strategy

### Chosen approach
Use a **three-layer fix strategy**:
1. **Policy and generation safeguards**: Prevent pathological runtime under-allocation for sensitivity simulation rules in `batch_job`/SLURM contexts.
2. **Execution pressure control**: Improve default or fixture-level concurrency/time realism for CPU sensitivity suites to avoid immediate scheduler timeout churn.
3. **Failure-observability hardening**: Reduce ambiguous simulation completion diagnostics and strengthen post-failure workflow clarity.

This approach directly targets the observed root cause first, then addresses high-value secondary instability without over-scoping.

### Alternatives considered
- **Only adjust test fixtures (`hpc_time_min_per_sim`)**: fastest, but brittle and leaves production users exposed.
- **Only adjust cluster submission behavior (Snakemake/plugin flags)**: may reduce noise but does not fix timeout root cause.
- **Large refactor of runner execution lifecycle**: potentially beneficial long-term, but too large for this focused bug fix.

### Trade-offs
- Adding stronger runtime guardrails may reject formerly allowed but unrealistic configs; this is desirable for fail-fast correctness.
- Conservative defaults can increase queue/runtime cost but significantly improve completion reliability.
- Touching both config validation and workflow generation increases change surface, but reduces recurrence risk.

## File-by-File Change Plan

1. `src/TRITON_SWMM_toolkit/config/analysis.py`
   - Add stricter validation for `hpc_time_min_per_sim` when `multi_sim_run_method == "batch_job"` in sensitivity-heavy contexts (or at minimum add warning/error thresholds configurable by mode).
   - Ensure validation messaging is explicit: tiny runtimes are likely to timeout under MPI/hybrid/high-thread sub-analyses.
   - Impact: catches invalid/unsafe runtime policies before workflow launch.

2. `src/TRITON_SWMM_toolkit/workflow.py`
   - Add centralized helper for simulation runtime normalization (e.g., enforce floor by run mode/resource shape in generated rules).
   - Apply to both regular and sensitivity workflow generation paths (`generate_snakefile_content` and `generate_master_snakefile_content`).
   - Optionally incorporate sub-analysis-aware runtime scaling hints (e.g., higher floor for high `tasks*cpus_per_task`).
   - Impact: generated Snakefiles avoid known-bad runtime settings that cause universal simulation failures.

3. `src/TRITON_SWMM_toolkit/run_simulation.py`
   - Harden completion diagnostics path around ambiguous success (`performance.txt` vs log marker mismatch) to provide deterministic failure reason classification.
   - Ensure timeout/cancellation downstream symptoms are labeled as likely scheduler-time-limit consequences where detectable.
   - Impact: better debugging signal, fewer misleading failure interpretations.

4. `src/TRITON_SWMM_toolkit/resource_management.py` (optional targeted change)
   - If needed, add helper utilities used by workflow runtime normalization (resource-shape classification for serial/openmp/mpi/hybrid).
   - Keep narrowly scoped; avoid broad redesign.
   - Impact: cleaner separation of resource heuristics from Snakefile text construction.

5. `tests/fixtures/test_case_catalog.py`
   - Update failing CPU sensitivity fixture defaults (`test_uva_sensitivity_suite_cpu`) to realistic runtime baseline for CI/HPC validation intent.
   - Consider reducing `hpc_max_simultaneous_sims` for sensitivity stress fixtures where objective is correctness over saturation.
   - Impact: prevents intentionally pathological defaults from masking regressions.

6. `tests/test_PC_05_sensitivity_analysis_with_snakemake.py`
   - Add/extend unit assertions that generated sensitivity simulation rules do not emit unsafe runtime for configured stress profiles.
   - Impact: local regression protection for workflow generation logic.

7. `tests/test_UVA_03_sensitivity_analysis_with_snakemake.py` and/or `tests/test_UVA_04_multiCPU_sensitivity_analysis_minirun.py`
   - Add assertions or lightweight checks for runtime/concurrency policy in generated Snakefile for UVA sensitivity cases.
   - Impact: platform-specific coverage aligned with observed failure mode.

8. `docs/planning/bugs/addressing_cpu_sensitivity_analysis_failures.md` (this file)
   - Track implementation and decisions.

Import/update callouts:
- If runtime normalization helper is introduced in `workflow.py`, update both regular and sensitivity builder call sites immediately.
- If shared heuristics move to `resource_management.py`, update import sites in `workflow.py` accordingly.

## Risks and Edge Cases

- **Risk: Overly strict runtime floor blocks legitimate tiny test runs**
  - Mitigation: scope stricter floors to SLURM `batch_job`/sensitivity contexts or allow explicit override flag with clear warning.
- **Risk: Increased walltime increases queue wait**
  - Mitigation: pair with reduced concurrency defaults for diagnostic suites.
- **Risk: False positive “ambiguous completion” interpretation**
  - Mitigation: classify based on explicit timeout/cancel markers where present in runner/SLURM logs.
- **Edge case: high-resource sub-analyses (sa17/sa18/sa19) with no rule log due to orchestration crash**
  - Validate handling where simulation log is empty/missing and classify as orchestration interruption.
- **Edge case: sensitivity mix of serial/openmp/mpi/hybrid**
  - Validate runtime normalization scales appropriately and does not overfit one mode.

## Validation Plan

Local validation (codegen + unit style):
- `pytest tests/test_PC_01_singlesim.py`
- `pytest tests/test_PC_02_multisim.py -m "not slow"`
- `pytest tests/test_PC_04_multisim_with_snakemake.py`
- `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`
- `pytest tests/test_config_validation.py`

Targeted generation/fixture validation:
- `pytest tests/test_UVA_03_sensitivity_analysis_with_snakemake.py -k "generation or config"`

HPC coordinated validation with user (required for SLURM behavior):
- Re-run `test_uva_sensitivity_suite_cpu` with updated runtime/concurrency policy.
- Confirm:
  - `_status/simulation_sa*_evt0_complete.flag` progresses beyond 0.
  - No widespread `DUE TO TIME LIMIT` signatures in `.snakemake/slurm_logs/rule_simulation_sa*_evt0/*.log`.
  - Workflow reaches processing/consolidation phases.

## Documentation and Tracker Updates

- `docs/planning/reference/priorities.md`
  - Update bug priority/status once implementation is merged and UVA rerun passes.
- `CLAUDE.md` (Gotchas / HPC guidance)
  - Add note about unsafe low `hpc_time_min_per_sim` in sensitivity batch runs if the final implementation introduces policy floors/validation.
- Optional: add a short troubleshooting note in HPC docs for interpreting timeout -> `srun launch failed` cascades.

Conditions to trigger updates:
- If validation rules or runtime normalization behavior changes user-facing config expectations.
- If test fixture defaults are intentionally changed to avoid pathological settings.

## Decisions Needed from User

1. **Runtime policy strictness (blocks implementation finalization)**
   - Option A: hard fail for very low runtime values in `batch_job` sensitivity runs (recommended).
   - Option B: auto-clamp to minimum safe runtime with warning.
   - Option C: warn only (least safe).
   - Assumption if no response: **Option A** (risk: medium).

2. **Scope of fixture updates**
   - Should we update only `test_uva_sensitivity_suite_cpu`, or all similar sensitivity suite fixtures currently set to `2` minutes (CPU/GPU/TRITON-only/SWMM-only)?
   - Assumption if no response: update all analogous suite fixtures for consistency (risk: low).

3. **Concurrency tuning policy in fixtures**
   - Do you want `hpc_max_simultaneous_sims` reduced for sensitivity debug suites to prioritize reliability over throughput?
   - Assumption if no response: keep as-is initially and revisit only if timeouts persist after runtime fix (risk: medium).

## Definition of Done

- [ ] Runtime safety policy for sensitivity `batch_job` simulations is implemented (validation and/or normalization).
- [ ] Generated Snakefiles no longer emit pathological `runtime=2` for the targeted CPU sensitivity suite unless explicitly allowed.
- [ ] Ambiguous completion diagnostics are improved for timeout/cancel scenarios.
- [ ] Relevant fixture defaults are aligned with intended test realism.
- [ ] Local workflow/config tests (PC_01, PC_02, PC_04, PC_05 + config validation) pass.
- [ ] UVA coordinated rerun confirms simulation phase advances and timeout pattern is resolved/reduced.
- [ ] Planning/docs trackers updated to reflect implementation outcome.

---

### Self-Check Results

1. **Header/body alignment check**
- All required section headers are present and content matches each section’s intent.
- File-by-file section maps directly to specific code/test/doc files relevant to observed failures.

2. **Section necessity check**
- All sections are necessary for implementation readiness and approval gating.
- No section can be removed without losing execution-critical guidance.
