# API Vision (Notebook + Integration Friendly)

## Purpose

Define how users should interact with TRITON-SWMM Toolkit directly from Python,
while preserving behavioral parity with the Snakemake-first CLI.

## Design Principle

Expose **one orchestration core** through two interfaces:

- CLI for reproducible operational runs
- API for interactive analysis and custom workflow composition

The same rules for validation, model routing, processing, and logging should
apply in both interfaces.

---

## API User Types

### 1) Notebook Analyst
- Wants concise setup and quick iteration.
- Needs immediate access to xarray/pandas outputs.

### 2) Pipeline Developer
- Wants explicit control over run/processing stages.
- Needs structured return values and exceptions.

### 3) Advanced Internal Contributor
- Works on lower-level scenario/run/process components.
- Needs stable contracts and minimal hidden side effects.

---

## Proposed API Layers

## 1) High-Level Facade (recommended for most users)

```python
from TRITON_SWMM_toolkit import Toolkit

tk = Toolkit.from_configs(system_config, analysis_config)
result = tk.run(
    model="auto",
    which="both",
    event_ilocs=[0, 1, 2],
    from_scratch=False,
    overwrite=False,
)
```

Characteristics:
- few arguments
- mirrors CLI semantics
- returns a structured `RunResult`

## 2) Composable Mid-Level Workflow

```python
analysis = Analysis(cfg_system=..., cfg_analysis=...)
analysis.prepare()
analysis.run_simulations(event_ilocs=[0, 1, 2])
analysis.process_outputs(which="both")
summary = analysis.collect_summary()
```

Characteristics:
- explicit stage control
- suitable for partial reruns and research notebooks

## 3) Low-Level Expert API

Expose scenario/run/post-processing objects for advanced use, but document these
as power-user paths with stronger assumptions.

---

## Return and Error Contracts

## Result Object

All high-level run methods should return a typed structure, e.g.:

- `success: bool`
- `scenario_ids: list[int]`
- `output_paths: dict[str, str]`
- `warnings: list[str]`
- `timings: dict[str, float]`
- `log_paths: dict[str, str]`

## Exceptions

- Validation errors: raise early with actionable message
- Model compatibility errors: explicit exception class
- Runtime failures: include scenario id and failed stage

---

## CLI ↔ API Parity Requirements

1. Same model/which routing semantics
2. Same precedence and defaults resolution
3. Same completion checks and overwrite behavior
4. Same major log artifacts and status tracking
5. Same profile semantics for `production`, `testcase`, and `case-study`
6. Same inheritance/override behavior for HPC settings in testcase/case-study runs

If behavior diverges, parity tests should fail.

For testcase/case-study usage, API entrypoints should accept either:

- a profile key (e.g., `profile="testcase", testcase="norfolk_smoke"`), or
- an already-resolved profile object loaded from `tests_and_case_studies.yaml`.

---

## Notebook Experience Priorities

- Fast setup from config paths
- Easy subsetting of events
- Direct handles to processed datasets
- Clear progress and status summaries

---

## v1 Non-Goals

- Full abstraction over all Snakemake features
- Hiding all low-level objects from advanced users
- Replacing established internal processing classes immediately
