# CLI/API Convergence Roadmap (Planning)

## Objective

Implement a Snakemake-first single-command CLI while preserving a direct Python
API that uses the same orchestration semantics.

## Phase 0 — Planning Baseline ✅ COMPLETE

Deliverables:
- `cli_vision.md`
- `cli_command_spec.md`
- `api_vision.md`
- this roadmap

Exit criteria:
- team-aligned interface vision
- agreed v1 command arguments and API shape

---

## Phase 1 — Interface Contracts ✅ COMPLETE (80%)

Scope:
- finalize CLI argument contract and validation matrix
- finalize API high-level facade and result contract
- define error classes and exit code mapping
- finalize profile model: `production`, `testcase`, `case-study`
- finalize `tests_and_case_studies.yaml` schema and merge semantics

Implementation notes:
- keep existing behavior where possible
- document any intentional behavior changes

Exit criteria:
- contract docs approved ✅
- acceptance tests drafted (CLI + API parity) ✅ (45 tests passing)
- ⏸️ Deferred: testcase/case-study profile resolution (blocked)

Completed:
- CLI argument contract with validation matrix
- Error classes and exit code mapping
- Profile model with production/testcase/case-study support
- List actions (--list-testcases, --list-case-studies)
- CLI unit test suite (test_cli_01-03)
- Wired CLI to Analysis orchestration

---

## Phase 2 — Shared Orchestration Core ✅ COMPLETE

Scope:
- consolidate run/setup/processing flow behind one orchestration layer
- make CLI a thin adapter to that layer
- ensure API methods call the same orchestration path

Exit criteria:
- single source of truth for routing and validation ✅
- reduced duplicated control-flow logic ✅

Completed:
- `orchestration.py` module with WorkflowResult dataclass
- `analysis.run()` high-level API with mode translation (fresh/resume/overwrite)
- `translate_mode()` and `translate_phases()` functions
- `WorkflowStatus` and `PhaseStatus` dataclasses
- `analysis.get_workflow_status()` method for status reporting
- `--status` CLI flag integration
- 4 test suite for status reporting (all passing)

---

## Phase 3 — Snakemake-First CLI Implementation (CURRENT)

Scope:
- implement `run` entrypoint with agreed flags
- wire arguments to Snakemake targets/options
- add dry-run and argument-resolution summary output
- add discovery actions for curated profiles (`--list-testcases`, `--list-case-studies`) ✅
- resolve testcase/case-study profile entries into runtime configs

Exit criteria:
- end-to-end run works from command line
- clear non-zero exit codes for failure modes

Next Steps:
- Refactor CLI to use `analysis.run()` API (simplify from 60+ lines to ~30)
- Complete argument wiring to workflow parameters
- Implement dry-run summary output
- Add integration tests for end-to-end CLI execution

---

## Phase 4 — API Facade & Notebook UX

Scope:
- implement or formalize `Toolkit`-style high-level API
- return structured result objects
- expose concise notebook examples

Exit criteria:
- notebook workflow tested on representative scenario subset
- API docs include stage-by-stage and end-to-end usage examples

---

## Phase 5 — Tool Provisioning + Reliability

Scope:
- support `--redownload` behavior with explicit provenance logging
- strengthen resume/from-scratch safeguards
- improve run-manifest outputs for auditability
- ensure HPC override inheritance works for testcase/case-study profiles

Exit criteria:
- provisioning behavior deterministic and test-covered
- resume/fresh/overwrite semantics validated in integration tests

---

## Phase 6 — Documentation & Adoption

Scope:
- update user docs and quickstart guidance
- add migration notes for any interface changes
- include troubleshooting matrix

Exit criteria:
- docs published and linked from main usage entry points
- at least one end-to-end tutorial for CLI and one for API

---

## Open Decisions (to refine)

1. **Executable name**
   - `triton-swmm` vs `tritonswmm` vs existing entrypoint naming

2. **Event subset semantics**
   - exact inclusivity/exclusivity for `--event-range`

3. **Pass-through policy**
   - how broad `--snakemake-arg` support should be in v1

4. **Result object location**
   - dedicated module + dataclass/pydantic style

5. **Compatibility mode**
   - whether to preserve legacy invocation aliases in v1

6. **Profile catalog location policy**
   - default project path for `tests_and_case_studies.yaml` and fallback behavior

7. **How strict curated profile validation should be**
   - fail on unknown fields vs warn and ignore

---

## Test Strategy (high-level)

- Contract tests: CLI argument validation and exit codes
- Parity tests: same configs → equivalent outcomes via CLI and API
- Integration tests: model-specific and combined workflows
- Regression tests: resume/from-scratch/overwrite behavior
