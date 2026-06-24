"""Pure-Python structured validator for completed analyses.

Mirrors the assertion logic in `tests/utils_for_testing.py` (the
`assert_analysis_workflow_completed_successfully` chain) but returns
structured `CheckResult` records instead of raising `pytest.fail`. This lets
both pytest tests AND the report renderer (`report_renderers/errors_and_warnings.py`)
share the same validation logic.

Each per-check function returns one `CheckResult` describing pass/fail plus
optional per-scenario detail rows. The aggregator `validate_analysis()` runs
all 7 checks and returns a `ValidationReport`. For sensitivity analyses, the
aggregate per-scenario checks iterate sub-analyses and prefix each detail
row with the sub-analysis id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis


CheckLevel = Literal["system", "aggregate", "scenario", "resource"]


@dataclass
class CheckResult:
    """One pass/fail check result, with optional per-scenario detail rows."""

    name: str
    level: CheckLevel
    passed: bool
    summary: str
    details: list[dict] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation result for a single analysis."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def by_level(self) -> dict[str, list[CheckResult]]:
        out: dict[str, list[CheckResult]] = {"system": [], "aggregate": [], "scenario": [], "resource": []}
        for c in self.checks:
            out.setdefault(c.level, []).append(c)
        return out

    @property
    def granular_failures(self) -> list[dict]:
        """Flat list of per-scenario failure rows across all checks.

        Each row carries `{stage, sa_id (optional), scenario, detail}` so the
        renderer can emit a uniform "scenario × stage × detail" table.
        """
        rows: list[dict] = []
        for c in self.checks:
            if c.passed:
                continue
            if c.level not in ("aggregate", "scenario"):
                continue
            for d in c.details:
                rows.append({"stage": c.name, **d})
        return rows


# ---------------------------------------------------------------------------
# Per-check functions
# ---------------------------------------------------------------------------


def check_system_setup(analysis: TRITONSWMM_analysis) -> CheckResult:
    """System-level: compilation success for enabled models + DEM/Mannings present."""
    cfg_sys = analysis._system.cfg_system
    issues: list[dict] = []
    sys = analysis._system

    if cfg_sys.toggle_tritonswmm_model and not sys.compilation_successful:
        issues.append({"detail": "TRITON-SWMM compilation failed"})
    if cfg_sys.toggle_triton_model and not sys.compilation_triton_only_successful:
        issues.append({"detail": "TRITON-only compilation failed"})
    if cfg_sys.toggle_swmm_model and not sys.compilation_swmm_successful:
        issues.append({"detail": "SWMM compilation failed"})

    dem = sys.processed_dem_rds
    manning = sys.mannings_rds
    if dem is None:
        issues.append({"detail": "DEM not created"})
    if manning is None:
        issues.append({"detail": "Mannings not created"})
    if dem is not None and manning is not None and dem.shape != manning.shape:
        issues.append({"detail": f"DEM shape {dem.shape} != Mannings shape {manning.shape}"})
    if dem is not None and (len(dem.shape) != 3 or dem.shape[0] != 1):
        issues.append({"detail": f"Expected DEM shape (1, rows, cols), got {dem.shape}"})

    passed = not issues
    summary = "System setup OK" if passed else f"System setup FAILED ({len(issues)} issue(s))"
    return CheckResult(name="System setup", level="system", passed=passed, summary=summary, details=issues)


def _iter_subanalyses_or_self(analysis: TRITONSWMM_analysis):
    """Yield (sa_id, sub_analysis) for sensitivity master, else (None, analysis)."""
    sensitivity_on = getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
    sens = getattr(analysis, "sensitivity", None)
    if sensitivity_on and sens is not None:
        yield from sens.sub_analyses.items()
    else:
        yield None, analysis


def _detail_rows_for_failed_scenarios(analysis: TRITONSWMM_analysis, failed_paths: list[str]) -> list[dict]:
    """Convert a list of scenario_dir strings to detail-row dicts (no sa_id)."""
    return [{"scenario": str(Path(p).name), "scenario_dir": str(p), "detail": "did not complete"} for p in failed_paths]


def check_scenarios_setup(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Aggregate: all scenarios were created (per-scenario fails surfaced)."""
    details: list[dict] = []
    total = 0
    failed_count = 0
    for sa_id, sub in _iter_subanalyses_or_self(analysis):
        n = int(sub.n_scenarios)
        total += n
        if not sub._all_scenarios_created:
            failed = list(sub._scenarios_not_created)
            failed_count += len(failed)
            for p in failed:
                row = {"scenario": Path(p).name, "scenario_dir": str(p), "detail": "scenario not created"}
                if sa_id is not None:
                    row["sa_id"] = f"sa_{sa_id}"
                details.append(row)
    passed = failed_count == 0
    summary = (
        f"All {total} scenarios set up" if passed else f"Scenario setup failed for {failed_count} of {total} scenarios"
    )
    return CheckResult(name="Scenarios setup", level="aggregate", passed=passed, summary=summary, details=details)


def check_scenarios_run(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Aggregate: all simulations completed."""
    details: list[dict] = []
    total = 0
    failed_count = 0
    for sa_id, sub in _iter_subanalyses_or_self(analysis):
        try:
            n = len(sub.df_sims)
        except Exception:
            n = 0
        total += n
        if not sub._all_sims_run:
            failed = list(sub._scenarios_not_run)
            failed_count += len(failed)
            for p in failed:
                row = {"scenario": Path(p).name, "scenario_dir": str(p), "detail": "simulation did not complete"}
                if sa_id is not None:
                    row["sa_id"] = f"sa_{sa_id}"
                details.append(row)
    passed = failed_count == 0
    summary = f"All {total} scenarios ran" if passed else f"Simulation failed for {failed_count} of {total} scenarios"
    return CheckResult(name="Scenarios ran", level="aggregate", passed=passed, summary=summary, details=details)


def check_timeseries_processed(
    analysis: TRITONSWMM_analysis,
    which: Literal["both", "TRITON", "SWMM"] = "both",
) -> CheckResult:
    """Aggregate: per-enabled-model timeseries written for every scenario.

    A scenario's timeseries are "processed" iff its per-enabled-model summary
    files are PRESENT ON DISK (a path-only predicate), NOT the clobberable/stale
    ``all_*`` log attributes. The previous implementation ``getattr``'d a wrong
    attribute name (``all_TRITON_timeseries_processed`` — the class actually
    defines ``_all_TRITON_timeseries_processed``) and swallowed the resulting
    ``AttributeError`` under a blanket ``except (AttributeError, Exception)``, so
    it recorded zero failures unconditionally (the R4 bug). On-disk truth fixes
    both halves: the predicate cannot be wrong-named, and any genuine error now
    surfaces instead of being swallowed.

    The ``which`` parameter mirrors the existing ``assert_timeseries_processed``
    pytest helper by restricting the enabled-model set:

    - ``"both"`` (default): every enabled model
    - ``"TRITON"``: TRITONSWMM + TRITON-only
    - ``"SWMM"``: TRITONSWMM + SWMM-only

    Iterates ``_iter_subanalyses_or_self(analysis)`` so the sensitivity
    sub-analysis fan-out is preserved — iterating the master's own ``df_sims``
    would silently pass on a sensitivity analysis (also part of the R4 bug).
    """
    from hhemt.scenario import compute_event_id_slug
    from hhemt.summary_paths import scenario_summaries_present

    details: list[dict] = []
    total = 0
    for sa_id, sub in _iter_subanalyses_or_self(analysis):
        enabled = sub._get_enabled_model_types()
        if which == "TRITON":
            enabled = [m for m in enabled if m in ("tritonswmm", "triton")]
        elif which == "SWMM":
            enabled = [m for m in enabled if m in ("tritonswmm", "swmm")]
        sim_dir = sub.analysis_paths.simulation_directory
        for event_iloc in sub.df_sims.index:
            total += 1
            ev = sub._retrieve_weather_indexer_using_integer_index(event_iloc)
            event_id = compute_event_id_slug(ev)
            if not scenario_summaries_present(sub, event_id, enabled):
                row = {
                    "scenario": event_id,
                    "scenario_dir": str(sim_dir / event_id),
                    "detail": "timeseries not processed",
                }
                if sa_id is not None:
                    row["sa_id"] = f"sa_{sa_id}"
                details.append(row)
    passed = not details
    summary = (
        "All timeseries processed"
        if passed
        else f"Timeseries processing failed for {len(details)} of {total} scenarios"
    )
    return CheckResult(name="Timeseries processed", level="aggregate", passed=passed, summary=summary, details=details)


def check_analysis_summaries_created(analysis: TRITONSWMM_analysis) -> CheckResult:
    """System-level: master DataTree exists on disk (Option B canonical artifact)."""
    missing: list[dict] = []

    def _check_one(a, label_prefix: str = "") -> None:
        dt = a.analysis_paths.analysis_datatree_zarr
        if dt is None or not dt.exists():
            missing.append({"detail": f"{label_prefix}analysis_datatree.zarr missing"})

    sensitivity_on = getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
    if sensitivity_on and getattr(analysis, "sensitivity", None) is not None:
        sens_zarr = analysis.analysis_paths.sensitivity_datatree_zarr
        if sens_zarr is None or not sens_zarr.exists():
            missing.append({"detail": f"Sensitivity DataTree zarr missing at {sens_zarr}"})
        for sa_id, sub in analysis.sensitivity.sub_analyses.items():
            _check_one(sub, label_prefix=f"sa_{sa_id}: ")
    else:
        _check_one(analysis)

    passed = not missing
    summary = "Analysis summaries OK" if passed else f"Analysis summaries missing ({len(missing)} item(s))"
    return CheckResult(
        name="Analysis summaries created",
        level="system",
        passed=passed,
        summary=summary,
        details=missing,
    )


def check_scenario_status_csv(analysis: TRITONSWMM_analysis) -> CheckResult:
    """System-level: scenario_status.csv exists with required resource columns."""
    import pandas as pd

    csv_path = Path(analysis.analysis_paths.analysis_dir) / "scenario_status.csv"
    if not csv_path.exists():
        return CheckResult(
            name="scenario_status.csv created",
            level="system",
            passed=False,
            summary="scenario_status.csv missing",
            details=[{"detail": f"file not found at {csv_path}"}],
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return CheckResult(
            name="scenario_status.csv created",
            level="system",
            passed=False,
            summary="scenario_status.csv unreadable",
            details=[{"detail": f"read error: {e}"}],
        )
    required = [
        "scenario_setup",
        "run_completed",
        "scenario_directory",
        "actual_nTasks",
        "actual_omp_threads",
        "actual_gpus",
        "actual_total_gpus",
        "actual_gpu_backend",
        "actual_build_type",
        "perf_Total",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return CheckResult(
            name="scenario_status.csv created",
            level="system",
            passed=False,
            summary=f"scenario_status.csv missing required columns: {missing_cols}",
            details=[{"detail": f"missing columns: {missing_cols}"}],
        )
    return CheckResult(
        name="scenario_status.csv created",
        level="system",
        passed=True,
        summary=f"scenario_status.csv OK ({len(df)} rows)",
        details=[],
    )


def check_resource_usage(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Resource: actual MPI/OMP/GPU/backend match intended config per scenario."""
    from hhemt.consolidate_workflow import validate_resource_usage

    try:
        passed, issues = validate_resource_usage(analysis, logger=None)
    except Exception as e:
        return CheckResult(
            name="Resource usage matches config",
            level="resource",
            passed=False,
            summary=f"Resource validation crashed: {e}",
            details=[],
        )

    summary = (
        "All scenarios used expected compute resources"
        if passed
        else f"Resource mismatches in {len(issues)} scenario(s)"
    )
    return CheckResult(
        name="Resource usage matches config",
        level="resource",
        passed=passed,
        summary=summary,
        details=issues,
    )


def _read_persisted_eda_verdicts(analysis: TRITONSWMM_analysis) -> list[CheckResult]:
    """Read EDA verdict JSONs from ``{analysis_dir}/eda/*.verdict.json`` (graceful-absent).

    The ADR-9 EDA layer (``eda.check_cross_sim_identity`` et al.) persists each
    verdict as a ``dataclasses.asdict(CheckResult)`` JSON. This reads them back so
    the report's Errors-and-Warnings section surfaces EDA pass/fail families. Absent
    ``eda/`` dir or unreadable files → empty list (the report is unchanged from a
    non-EDA'd analysis).
    """
    import json

    eda_dir = Path(analysis.analysis_paths.analysis_dir) / "eda"
    if not eda_dir.is_dir():
        return []
    verdicts: list[CheckResult] = []
    for vf in sorted(eda_dir.glob("*.verdict.json")):
        try:
            payload = json.loads(vf.read_text())
            verdicts.append(
                CheckResult(
                    name=payload["name"],
                    level=payload["level"],
                    passed=payload["passed"],
                    summary=payload["summary"],
                    details=payload.get("details", []),
                )
            )
        except (OSError, KeyError, ValueError):
            continue
    return verdicts


def validate_analysis(analysis: TRITONSWMM_analysis) -> ValidationReport:
    """Run all 7 checks; return aggregated ValidationReport.

    Order matches the existing `assert_analysis_workflow_completed_successfully`
    chain so the report's check ordering matches what pytest displays.

    Persisted EDA verdicts (``{analysis_dir}/eda/*.verdict.json``, ADR-9) are
    appended after the 7 core checks so the renderer surfaces them by ``level``.
    """
    return ValidationReport(
        checks=[
            check_system_setup(analysis),
            check_scenarios_setup(analysis),
            check_scenarios_run(analysis),
            check_timeseries_processed(analysis),
            check_analysis_summaries_created(analysis),
            check_scenario_status_csv(analysis),
            check_resource_usage(analysis),
        ]
        + _read_persisted_eda_verdicts(analysis)
    )


# ---------------------------------------------------------------------------
# Persist-then-render read-model (Class-Y resolution, Option D, 2026-06-14)
# ---------------------------------------------------------------------------
#
# validate_analysis() reads a whole-tree surface (compilation logs, DEM/Manning
# rasters, Snakefile, per-sim logs + perf-summary zarrs) spanning analysis_dir AND
# system_dir. Running it inside errors_and_warnings.render() put that whole-tree
# read surface in the render path, which (a) the renderer-IO provenance audit could
# not faithfully declare and (b) made the portable render bundle non-re-renderable
# (the bundle ships none of that surface). The fix: run the inspection ONCE at
# consolidation (the compute phase that owns the full tree) and persist its result
# as a single JSON read-model; the renderer reads only that artifact. JSON shape =
# dataclasses.asdict(CheckResult), identical to the ADR-9 eda/*.verdict.json schema.

_VALIDATION_REPORT_FILENAME = "validation_report.json"


def persist_validation_report(analysis: TRITONSWMM_analysis) -> Path:
    """Run validate_analysis and persist it to {analysis_dir}/validation_report.json.

    Called once at consolidation. Overwrites on each consolidate (idempotent, like
    analysis_datatree.zarr). The persisted artifact also carries the eda verdicts
    (validate_analysis already folds them in via _read_persisted_eda_verdicts), so
    the renderer no longer reads eda/ either. Re-stamps the parent DU sentinel per
    the du-sentinels-written-at-every-mutation-site stipulation (Gotcha 38).
    """
    import json
    from dataclasses import asdict

    from hhemt import du_sentinels

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    report = validate_analysis(analysis)
    out = analysis_dir / _VALIDATION_REPORT_FILENAME
    payload = {"checks": [asdict(c) for c in report.checks]}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)  # atomic
    du_sentinels.restamp_parent_sentinels(out, analysis_dir=analysis_dir)
    return out


def load_validation_report(analysis: TRITONSWMM_analysis) -> ValidationReport:
    """Graceful-absent read of {analysis_dir}/validation_report.json.

    Returns an EMPTY ValidationReport when the artifact is absent (a pre-feature
    analysis, or a render that precedes the consolidation write). The absent case is
    deliberately NOT a fallback to validate_analysis(): re-running the whole-tree
    inspection at render time would re-introduce the render-path read surface this
    feature removes AND trip the renderer-IO provenance audit (those reads are
    undeclared). An empty report degrades cleanly, mirroring the eda graceful-absent
    pattern.
    """
    import json

    p = Path(analysis.analysis_paths.analysis_dir) / _VALIDATION_REPORT_FILENAME
    if not p.exists():
        return ValidationReport(checks=[])
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return ValidationReport(checks=[])
    return ValidationReport(
        checks=[
            CheckResult(
                name=c["name"],
                level=c["level"],
                passed=c["passed"],
                summary=c["summary"],
                details=c.get("details", []),
            )
            for c in payload.get("checks", [])
        ]
    )
