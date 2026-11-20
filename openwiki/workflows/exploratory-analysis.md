---
type: workflow
title: EDA and sensitivity verification
description: In-process exploratory calculations, exact and magnitude comparison boundaries, persisted verdicts and artifacts, and their reporting and bundle integration.
tags: [eda, sensitivity, validation, reproducibility, reporting]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-16a7e9a20df016cd4209c0dc
    resource: repo://docs/how-to/running-eda.md
  - id: openwiki-source-b9d0ed732962cf4f4ec1584c
    resource: repo://src/hhemt/analysis_validation.py
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-528b694574404109ed10c3da
    resource: repo://src/hhemt/eda/_plotting.py
  - id: openwiki-source-a98b885dc567b03f7525d98f
    resource: repo://src/hhemt/eda/_promote.py
  - id: openwiki-source-d3474cfe2a1a9e0001438024
    resource: repo://src/hhemt/eda/_result.py
  - id: openwiki-source-4c41ea31a3fda8c2af0e56d9
    resource: repo://src/hhemt/eda/compute_sensitivity.py
  - id: openwiki-source-49e56d7729b15ab09c10378e
    resource: repo://src/hhemt/eda/cross_sim_identity.py
  - id: openwiki-source-8d801180266bd00dab7027f9
    resource: repo://src/hhemt/eda/raw_byte_identity.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# EDA and sensitivity verification

## Separate from simulation and consolidation

`analysis.eda()` runs a local, in-process calculation → plot → notebook loop against existing results. It is not the ensemble consolidation DAG and does not submit an allocation or rerun simulations. The operator guide requires consolidated outputs and per-scenario summaries.

```sh
hhemt eda --analysis-dir ANALYSIS_DIR
# Or use explicit input configs:
hhemt eda --system-config system.yaml --analysis-config analysis.yaml
```

An optional `--override-eda-config` supplies a different EDA selection. In a Python session, `tk.analysis.eda()` returns `EdaReportResult`: notebook path, optional HTML report path, emitted plot paths, and calculation verdicts. `Bundle.eda()` is a separate portable render surface whose result has no newly computed verdicts.

The analysis facade calls cross-simulation identity, rank sensitivity, resume sensitivity, cross-hardware magnitude, and raw identity calculations on each invocation. Plot selection does not skip those calculations. It rewrites persisted `cfg_analysis.yaml` and `cfg_system.yaml`, emits `eda_local/`, renders enabled produced kinds, seeds a notebook, and attempts HTML conversion. HTML execution/conversion failure degrades to `report_path=None`, not failure of the calculation loop.

## Artifact and verdict contracts

A calculation returns `EdaResult` with optional `CheckResult`, artifact path, plot ID, and `skipped`. Applicable calculations write backing data under `eda/` plus verdict JSON and provenance sidecars. `raw_byte_identity` is an exception to absence-on-skip: it always writes either its comparison grid or an explicit degraded artifact so its report target can render.

| Artifact | Consumer |
| --- | --- |
| `eda/{stem}.zarr` | Standalone/report figure renderer |
| `eda/{stem}.manifest.json` | Provenance/source harvesting for portable artifacts |
| `eda/{stem}.verdict.json` | Analysis validation aggregation |
| `plots/eda/{kind}.html` | Standalone exploratory figures |
| `eda.ipynb`, then numbered variants | Preserved user exploration and refreshed seed cells |
| `eda_report/eda_report.html` | Best-effort executed notebook export |

Read `CheckResult.applicable` **before** `passed`. Not-applicable checks can have `passed=True`; that is not evidence of a comparison. Read `instrument` and `detection_floor` alongside the result: raw rasters and float32 summary depth cannot support the same precision claim. `details` is optional; raw identity currently leaves it empty even when failing, so gate on `passed` and report `summary`, not detail-list length.

**Current aggregation limitation:** `_read_persisted_eda_verdicts` reconstructs only name, level, passed, summary, and details. It does not restore persisted applicability/instrument/detection-floor fields, so those fields revert to dataclass defaults in the aggregate validation report. Inspect the original EDA verdict JSON for the complete instrument/applicability evidence; do not assume aggregate rendering preserves it.

## Exact equality is scoped, not archive-byte identity

`compare_variable_exact` aligns coordinates exactly, transposes dimensions, checks dtype equality, and compares decoded arrays without tolerance. Floating arrays use `equal_nan=True`; NaN versus a number fails. Coordinate mismatch is non-identical. This is decoded-value equality, not identical Zarr compression or file metadata.

The cross-simulation check normally compares sensitivity members against a reference within their own CPU/GPU family, preferring the serial CPU or minimum-device GPU reference with deterministic fallbacks. Cross-family equality is not asserted by that strict path. An explicit cross-family mode instead records characterized divergence and can return `passed=True` without claiming equality.

Summary checks read through each member's combined-output retrieval, which can resolve a verified consolidated summary slice after per-scenario stores were reclaimed. They label the instrument `summary_tier`, with float32 epsilon as the coarsest disclosed floor when depth is included. They cannot establish raw double-precision identity below that floor.

## Raw identity versus clean/resume comparison

`check_raw_byte_identity` compares the passed primary's **own** configurations against minimum-device references within CPU/GPU device classes. It applies on either a clean or resume primary. Clean-versus-resume between separate primaries belongs on the combined-experiment intercomparison surface, not a fabricated pair inside a single-arm primary.

The reusable raw TRITON kernel enumerates H/MH/QX/QY rasters, decodes them using index coordinates, and applies exact comparison per shared reporting timestep. Timesteps present in only one endpoint are outside that kernel's comparison. A verdict does not attest every file under a run directory: ghost-ring/checkpoint side files are not the raster denominator.

The SWMM helper compares parsed node/link datasets and separately compares `hydraulics.out` bytes with `shallow=False`; report text is not byte-compared because it contains run-date headers. Parse errors are disclosed as data rather than silently treated as equality. Raw identity reports `raw_rasters` and float64 epsilon; degraded data fails its verdict rather than inventing success.

## Rank, resume, and hardware magnitude

- **Rank sensitivity:** compare MPI rank N against rank 1 within its partition family; a family without a rank-1 reference is disclosed/skipped rather than anchored across unrelated modes.
- **Resume sensitivity:** classify clean/resume using `df_status.n_resumes`, pair members with the same compute identity, and skip when no pair exists. It is a within-primary check, distinct from cross-experiment clean/resume comparison.
- **Cross-hardware magnitude:** compare one-GPU endpoints against a one-rank serial CPU reference, characterizing differences without requiring bit identity. `passed=True` means the characterization was emitted, not that arrays were equal.

The magnitude kernel measures maximum absolute difference, wetted-footprint RMSE and extent disagreement over the union of wet cells. Relative differences use baseline depth above a physical threshold and Type-8 percentiles; newly wetted area is reported separately because a baseline ratio cannot represent it. With no wet union, CSI is undefined (`None`), not perfect agreement.

## Reporting, bundles, and promotion

The facade re-persists the whole-tree `validation_report.json` after calculations so new EDA verdicts reach Errors and Warnings. That recomputes core validation too; failure is logged non-fatally. If the Errors-and-Warnings figure already exists, the facade re-renders that single figure after refreshing the read-model, also non-fatally. Calling a report-only `render_report()` is not a substitute for executing stale figure rules.

Rendering validates every enabled kind against the registry before narrowing. Missing backing artifacts ordinarily warn and skip, while enumerated report targets render degradation panels instead of leaving missing outputs. Raw identity is always-emitted; DEM-resolution renderers read the experiment tree directly and bypass the EDA-backing-store gate. `config_diff_maps` reads `eda_cross_sim_identity.zarr`, not a same-named store.

Run EDA before bundling if its figures/data should travel: figure manifests declare the derived source artifacts for harvesting. Emitting a bundle before EDA omits that content.

Promotion APIs do not create a new renderer implementation. `promote_eda_plot_to_static_config` writes a validated base static-plot YAML, but its current header explicitly notes that EDA-ID dispatch in the static consumer is deferred. `register_eda_plot_in_reporting_set` validates names and requires an existing EDA-capable reporting set, then returns an intent record; it does not mutate a live renderer selection or install arbitrary routing. Selecting an existing reporting adapter and registering a new arbitrary plot are different operations.

## Related topics

- [Output data model](../concepts/output-data-model.md)
- [Sensitivity and synthetic experiments](sensitivity-and-synthetic-experiments.md)
- [Diagnostics and in-flight control](../operations/diagnostics-and-in-flight-control.md)
- [Reporting and portable bundles](reporting-and-portable-bundles.md)

Source inspection only; no EDA calculation, notebook kernel, solver, or test was executed.
