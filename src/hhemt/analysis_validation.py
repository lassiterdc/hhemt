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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

logger = logging.getLogger(__name__)


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


def check_invalidating_fixes(analysis: TRITONSWMM_analysis) -> CheckResult:
    """ADR-17 load-time invalidating-fix registry match, surfaced in the report read-model.

    Non-blocking by construction: a matched invalidating fix produces ``passed=False``
    (only when at least one match is ``severity="error"``) + a summary naming the
    count, but ``validate_analysis`` NEVER raises. ``warning``-only matches keep
    ``passed=True``. An absent registry / unstamped tree → no matches → ``passed=True``
    (graceful-absent; D6). ``level="aggregate"`` so the Errors-and-Warnings renderer
    routes it with no renderer edit. Runs at consolidation-time
    ``persist_validation_report``, NOT render time (Gotcha 53 audit-safety) — the
    ADR-15 stamps it reads live in the consolidated datatree the renderer declares,
    and the registry is package-data.
    """
    from hhemt.recompute import match_registry_against_stamps  # ADR-16/17 resolver

    matches = match_registry_against_stamps(analysis)  # [] when registry absent / no hit
    if not matches:
        return CheckResult(
            name="invalidating-fix registry",
            level="aggregate",
            passed=True,
            summary="No registered calculation-invalidating fixes affect this analysis.",
            details=[],
        )
    # match_registry_against_stamps returns only ACTIONABLE matches (non-None
    # recommended_action); the guard narrows the RegistryMatch union for the type
    # checker and stays safe if the contract ever changes.
    rows = [
        {
            "commit_id": m.commit_id,
            "severity": m.severity,
            "recommended_action": m.recommended_action.value if m.recommended_action else None,
            "scenario": m.affected_scope,
            "detail": m.summary,
        }
        for m in matches
    ]
    has_error = any(m.severity == "error" for m in matches)
    return CheckResult(
        name="invalidating-fix registry",
        level="aggregate",
        passed=not has_error,  # warning-only matches keep passed=True
        summary=f"{len(matches)} registered invalidating fix(es) affect this analysis.",
        details=rows,
    )


#: The pinned TRITON coupled-resume-fix commit. Every experiment's TRITON binary is
#: pinned here (D3 clone-gate enforcement); a consolidated tree whose
#: ``triton_has_coupled_resume_fix`` is False was produced by PRE-FIX TRITON. Full
#: 40-char sha (the short pin ``3a832f7d`` resolves to this). Mirrored in
#: ``system.py::_PINNED_TRITON_COUPLED_RESUME_FIX_SHA`` (compile-time capture).
_PINNED_TRITON_COUPLED_RESUME_FIX_SHA = "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4"

#: The positive marker TRITON prints to its stderr log on every SUCCESSFUL exchange-
#: history replay (``swmm_triton.h:672-673`` @ 3a832f7d). Its ABSENCE from a resumed
#: coupled sim's log means the replay never engaged (rank-0-local engage guard, or a
#: purged side-file) and SWMM re-initialized from t=0 — the pre-fix truncation,
#: recurring silently at the fixed commit.
_TRITON_REPLAY_MARKER = "SWMM exchange history replayed to t="

#: The completion marker TRITON prints at the end of a sim that ran to t=end (Gotcha 6:
#: completion is detected from log markers, never return codes). PRODUCER-SIDE SIBLING:
#: ``run_simulation.py:179`` (``model_run_completed``'s raw-marker fallback) tests the
#: IDENTICAL literal -- keep the two in sync until both TRITON marker literals are hoisted
#: to one home. Used here as the replay arm's COMPLETION GATE: because the model log is
#: opened ``"w"`` on every exec (``run_simulation_runner.py``) it holds ONLY the last exec,
#: so a last exec that was walltime-killed BEFORE its replay legitimately carries neither
#: marker. Gating on this literal -- read from the SAME log, in the SAME read, as the
#: replay marker -- keeps both predicates derived from one artifact, so they cannot skew.
#: (``df_status.run_completed`` was rejected for this gate: it resolves from the model-log
#: JSON or the coupled rpt -- DIFFERENT artifacts with DIFFERENT exec granularity than the
#: last-exec-only log -- and it couples this detector to model_run_completed's
#: sticky-False-latch semantics.)
_TRITON_COMPLETION_MARKER = "Simulation ends"

#: The PER-EXEC RESUME DISCRIMINATOR. TRITON prints ``[..] Reading checkpoint files`` then
#: ``[OK] Checkpoint files read`` BEFORE the replay marker, on every exec that resumed from
#: a hotstart cfg. This is what makes the replay arm's claim exact rather than approximate:
#: the log is last-exec-only (opened ``"w"`` per exec) while ``n_resumes`` is CUMULATIVE and
#: is NEVER reset (``run_simulation.py``'s increment is its only writer), so
#: ``n_resumes >= 1`` proves only that the sim resumed at SOME point. Without an in-log
#: per-exec signal, a sim that resumed, lost its checkpoints (clear_raw / delete /
#: force-rerun), and then ran FRESH to completion would retain ``n_resumes >= 1``, carry no
#: replay marker, and be WARNED ON despite being VALID. Reading the discriminator from the
#: SAME log in the SAME read makes all THREE predicates (resumed-this-exec /
#: completed-this-exec / replayed-this-exec) statements about one artifact, so none can skew
#: against the others; ``n_resumes`` degrades to a cheap pre-filter bounding which logs are
#: opened (sound because TRITON reads checkpoints IFF the runner passed a hotstart cfg IFF
#: n_resumes was incremented -- the same ``if`` branch, run_simulation.py:539-544 -- so
#: n_resumes >= 1 is a NECESSARY condition for a checkpoint-reading exec).
#:
#: THE ANCHOR IS THE ``[OK]`` COMPLETION FORM, NOT THE ``[..]`` ATTEMPT FORM. The replay
#: reads the exchange history FROM the checkpoint set, so "the replay should have engaged"
#: is warranted only if the checkpoint read SUCCEEDED. A read that starts and fails, after
#: which the run proceeds fresh and reaches "Simulation ends", would be a FALSE WARN under
#: the attempt form (in scope, complete, no replay marker) and is correctly out of scope
#: under the completion form. The attempt form is retained to separate that case out as
#: INDETERMINATE rather than silently calling it fresh.
#:
#: Empirical basis (read-only against existing artifacts, 2026-07-16, synth_cc_resume at the
#: pin): a clean-vs-resume control over 56 model_tritonswmm_*.log files gave PERFECT
#: separation -- clean arm (n=28, never resumed): 0 checkpoint-reads, 0 replay markers;
#: resume arm (n=28, all resumed): 28/28 both. Exactly ONE replay marker per exec. Both the
#: attempt form and the completion form are corpus-confirmed at 28/28 on the resume arm and
#: 0/28 on the clean arm. The INDETERMINATE branch below is retained as a fail-safe for a
#: partially-read checkpoint set, which the corpus contains no instance of.
_TRITON_CHECKPOINT_READ_MARKER = "Checkpoint files read"
_TRITON_CHECKPOINT_ATTEMPT_MARKER = "Reading checkpoint files"


def _read_triton_provenance(
    analysis: TRITONSWMM_analysis,
) -> tuple[str | None, bool | None]:
    """Graceful-absent read of the consolidated-tree TRITON provenance root attrs.

    Returns ``(triton_producing_sha, triton_has_coupled_resume_fix)``. Mirrors
    ``recompute.py::_iter_scope_stamps``: a sensitivity master resolves
    ``sensitivity_datatree.zarr``, else ``analysis_datatree.zarr``. Either element is
    None when the attr / tree is absent or unreadable (a pre-provenance tree, or an
    off-checkout install) -> the caller treats None as INDETERMINATE, never a false
    pre-fix warn. NEVER raises.
    """
    import xarray as xr

    paths = analysis.analysis_paths
    if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
        zarr_path = getattr(paths, "sensitivity_datatree_zarr", None)
    else:
        zarr_path = getattr(paths, "analysis_datatree_zarr", None)
    if zarr_path is None or not zarr_path.exists():
        return None, None
    # NO chunking: this reader consumes ONLY root attrs, and `chunks="auto"` raises
    # NotImplementedError ("Can not use auto rechunking with object dtype") on any
    # tree carrying an object-dtype variable — which the bare except below then
    # converted into a silent INDETERMINATE, permanently disabling this check's
    # pre-fix warning on every experiment. Empirically confirmed on the
    # synth_cc_resume sensitivity master (2026-07-15): chunks="auto" ->
    # NotImplementedError; no-chunks -> reads triton_producing_sha=3a832f7d...,
    # triton_has_coupled_resume_fix=True.
    try:
        tree = xr.open_datatree(zarr_path, engine="zarr", consolidated=False)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # Genuinely absent / unreadable tree (pre-provenance or off-checkout) —
        # the documented graceful-absent path. Quiet by design.
        return None, None
    except Exception as e:
        # NEVER raise (validation must not abort a run over a diagnostic), but do
        # NOT swallow silently: an unexpected exception here means this check is
        # INERT, which is exactly how the chunks="auto" defect stayed hidden.
        logger.warning(
            f"TRITON provenance read failed unexpectedly for {zarr_path} "
            f"({type(e).__name__}: {e}); coupled-resume validity is INDETERMINATE."
        )
        return None, None
    sha = tree.attrs.get("triton_producing_sha")
    has_fix = tree.attrs.get("triton_has_coupled_resume_fix")
    # zarr may round-trip the bool as numpy bool_/0-1; normalize to Python bool-or-None.
    if has_fix is not None:
        has_fix = bool(has_fix)
    return (str(sha) if sha is not None else None), has_fix


def check_coupled_resume_validity(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Warn when a COMPLETED coupled analysis's resumed data is invalid.

    Two independent invalidity paths, both keyed on a coupled model that resumed:

    (A) PRE-FIX TRITON. When ``triton_has_coupled_resume_fix`` is False the producing
        TRITON predates 3a832f7d, so any ``tritonswmm`` sim with ``n_resumes >= 1``
        re-inited SWMM from t=0 and its max-flow/depth summaries are systematically low.

    (B) POST-FIX but the REPLAY NEVER ENGAGED. Even at the pinned fix, TRITON's exchange-
        replay engage guard is rank-0-LOCAL (``triton.h:435``), so a rank-0 row-strip
        owning no manhole skips the replay and the pre-fix truncation recurs SILENTLY. A
        sha-only check would launder that as "data valid" — worse than emitting no check.
        The sound detector is the positive replay marker TRITON prints on every successful
        replay; its ABSENCE from a log whose exec RESUMED and RAN TO t=end is the failure.
        The log is resolved through ``run_simulation.model_logfile_for`` — the producer's own
        convention — never hand-built: this arm originally hand-built
        ``{scenario_directory}/logs/run_tritonswmm.log`` (the vestigial
        ``ScenarioPaths.log_run_*`` path nothing writes), so every read raised, every row was
        skipped, and the arm passed VACUOUSLY on every experiment.

        THREE PREDICATES, ONE READ, ONE FILE. The log is opened ``"w"`` per exec and holds
        ONLY the last one, so every predicate must describe THAT exec or they skew:
        resumed-this-exec (``"Checkpoint files read"``), completed-this-exec
        (``"Simulation ends"``), replayed-this-exec (the replay marker) — all from a single
        ``read_text()``. This is why ``df_status.run_completed`` is NOT the completion gate
        (it resolves from the model-log JSON or the coupled rpt — different artifacts, different
        exec granularity — and couples this check to ``model_run_completed``'s sticky-False-latch
        semantics), and why ``n_resumes >= 1`` is only a PRE-FILTER bounding which logs are
        opened (it is cumulative and never reset, so it cannot answer a per-exec question; it is
        sound as a necessary condition because TRITON reads checkpoints iff the runner passed a
        hotstart cfg iff n_resumes was incremented — the same branch). Predicates are ordered by
        logical precedence — does the question APPLY, can I ANSWER it, what is the ANSWER — so a
        fresh last exec is OUT OF SCOPE rather than a warn or an INDETERMINATE.

        Reads per-sim LOGS (not the tree) so it stays at consolidation-time
        ``persist_validation_report``, NOT render time (Gotcha 53 audit-safety) — the same logs
        ``check_scenarios_run`` already opens via ``model_run_completed``, so this adds no
        read-surface class.

    Distinct from the removed interim ``check_coupled_hotstart_resume`` (which counted
    sims BLOCKED by the interim guard, ``run_completed`` False); this warns about
    COMPLETED-but-invalid data after the fix ships. Non-blocking (``validate_analysis``
    never raises). Graceful-absent throughout: an unstamped tree / unreadable log ->
    INDETERMINATE INFO (``passed=True``), NEVER a false warn. The pre/post-fix predicate
    reads the stamped BOOLEAN, NOT sha-equality — a DESCENDANT of 3a832f7d (a routine pin
    bump) must not be misclassified pre-fix.
    """
    import pandas as pd

    if not getattr(analysis._system.cfg_system, "toggle_tritonswmm_model", False):
        return CheckResult(
            name="Coupled resume validity",
            level="aggregate",
            passed=True,
            summary="Coupled model not enabled — coupled-resume validity N/A.",
            details=[],
        )
    triton_sha, has_fix = _read_triton_provenance(analysis)
    if has_fix is None:
        return CheckResult(
            name="Coupled resume validity",
            level="aggregate",
            passed=True,
            summary=(
                "Producing-TRITON coupled-resume-fix status unknown (pre-provenance tree "
                "or off-checkout); cannot determine coupled-resume validity. Re-consolidate "
                "to stamp it."
            ),
            details=[],
        )

    # Resume record needed for BOTH the pre-fix arm and the replay-marker arm.
    try:
        df = analysis.df_status
    except Exception:
        df = None
    if df is None or not {"model_type", "n_resumes"}.issubset(getattr(df, "columns", [])):
        return CheckResult(
            name="Coupled resume validity",
            level="aggregate",
            passed=True,
            summary=(
                "Coupled-resume-fix status known but no resume record available — "
                "no coupled-resume invalidity found."
            ),
            details=[],
        )
    # `n_resumes >= 1` is a PRE-FILTER, not a verdict: it bounds which logs we open, and
    # it is SOUND for that because TRITON reads checkpoints IFF the runner passed a hotstart
    # cfg IFF n_resumes was incremented (the same `if` branch, run_simulation.py:539-544), so
    # n_resumes >= 1 is a NECESSARY condition for any checkpoint-reading exec. It is NOT
    # sufficient: n_resumes is CUMULATIVE across execs and is never reset, while the log is
    # last-exec-only. Whether a row's LAST exec actually resumed is decided in-loop below by
    # _TRITON_CHECKPOINT_READ_MARKER. Named `resume_candidates`, not `resumed`, so the
    # distinction cannot be re-collapsed by a future reader.
    n_resumes = pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0)
    resume_candidates = df[(df["model_type"] == "tritonswmm") & (n_resumes >= 1)]

    details: list[dict] = []
    # DISCLOSED DENOMINATOR (R6 hardening). `passed = len(details) == 0` cannot, on its
    # own, distinguish "examined N, found nothing" from "examined 0" — and THAT is the
    # defect this repair closes: Arm B read a path nothing writes, so `except: continue`
    # fired on all 28 rows and the check reported "No coupled-resume invalidity detected"
    # while detecting NOTHING. A verdict is only as good as its denominator, so all three
    # counters are surfaced in `summary` (free-text; no CheckResult schema change, so the
    # persisted validation_report.json / eda verdict shape is untouched). A future reader —
    # or an acceptance gate — can now see a vacuous pass in the artifact itself.
    #
    # The three outcomes are DISTINCT and must not be merged:
    #   examined              — the replay question applied and was answered.
    #   indeterminate         — the question applied (or might have) and could NOT be
    #                           answered (unresolvable sub / unreadable log / resume did not
    #                           take / last exec killed before t=end).
    #   not_resumed_last_exec — the question did NOT apply: the last exec ran fresh, so
    #                           n_resumes is a stale record of superseded execs. This is
    #                           OUT OF SCOPE, not an error and not indeterminate. Folding it
    #                           into `indeterminate` would report a non-problem as a
    #                           partial failure; folding it into `examined` would report a
    #                           row we never tested as tested.
    examined = 0
    indeterminate = 0
    not_resumed_last_exec = 0
    if not has_fix:
        # Arm A — PRE-FIX TRITON: every resumed coupled sim is invalid. The stamped
        # boolean alone decides, so every candidate row is examined by construction.
        #
        # NOTE (bounded, deliberate): Arm A carries the SAME n_resumes-staleness channel
        # Arm B closes below — a sim that resumed pre-fix, lost its checkpoints, and re-ran
        # FRESH to completion is valid data that this arm warns on. The same in-log
        # discriminator would close it, but there is ZERO evidence of what a PRE-FIX binary
        # prints (the entire observed corpus is at the pin, which `_verify_tritonswmm_pin`
        # enforces), and this arm is dormant for the same reason: no new pre-fix tree can be
        # produced. Fixing it on inference rather than evidence is the exact move that
        # produced the original defect. Left as-is until pre-fix data is in hand.
        examined = len(resume_candidates)
        for _, row in resume_candidates.iterrows():
            details.append(
                {
                    "scenario": str(row.get("scenario_directory", "")),
                    "detail": (
                        f"produced by PRE-FIX TRITON ({triton_sha}) with a coupled hotstart "
                        f"resume (n_resumes={int(row.get('n_resumes') or 0)}); coupled SWMM "
                        "re-inits from t=0 on resume so max-flow/depth summaries are "
                        "systematically low. Re-run these sims with the pinned "
                        "coupled-resume-fix TRITON."
                    ),
                }
            )
    else:
        # Arm B — POST-FIX: a coupled sim whose LAST exec resumed, ran to t=end, and yet
        # lacks the replay marker had its replay silently skipped (rank-0-local engage
        # guard, or a purged side-file) -> truncated exactly as pre-fix.
        #
        # THREE PREDICATES, ONE READ, ONE FILE. resumed-this-exec / completed-this-exec /
        # replayed-this-exec all come from a single read_text() of a single log, so none can
        # skew against the others. That atomicity is the whole design: the log is opened "w"
        # on every runner exec (run_simulation_runner.py) and therefore describes ONLY the
        # last exec, so any predicate sourced from a DIFFERENT artifact (df_status.
        # run_completed resolves from the model-log JSON or the coupled rpt; n_resumes is
        # cumulative) is a statement about a different exec granularity and can disagree.
        #
        # Order is logical precedence: does the question APPLY -> can I ANSWER it -> what is
        # the ANSWER. Scope precedes completion deliberately: a FRESH exec that was killed is
        # both "not a resume" and "incomplete", and "not a resume" is the truthful label —
        # the replay question never applied to it, and its incompleteness is
        # check_scenarios_run's business, not this arm's.
        from hhemt.run_simulation import model_logfile_for

        # Resolve the log through the PRODUCER's own convention — never hand-build it.
        # `_iter_subanalyses_or_self` yields (sa_id, sub) for a sensitivity master and
        # (None, analysis) otherwise, which is ALREADY keyed the way `row.get("sa_id")`
        # reads: a non-sensitivity df_status carries no sa_id column (only
        # sensitivity_analysis.df_status adds it), so `.get` returns None and hits the
        # None key. str-normalized per the sa_id-cast-to-string stipulation, mirroring
        # per_analysis_summary.py's `astype(str) == str(sa_id)` precedent.
        subs = {(str(k) if k is not None else None): v for k, v in _iter_subanalyses_or_self(analysis)}
        for _, row in resume_candidates.iterrows():
            scen_dir = str(row.get("scenario_directory", ""))
            _sa = row.get("sa_id")
            sub = subs.get(str(_sa) if _sa is not None else None)
            if sub is None:
                indeterminate += 1
                continue  # INDETERMINATE — cannot resolve the owning (sub-)analysis
            try:
                text = model_logfile_for(sub, int(row["event_iloc"]), "tritonswmm").read_text()
            except Exception:
                indeterminate += 1
                continue  # INDETERMINATE — cannot read the log
            # (1) SCOPE GATE — did THIS exec resume, and did the resume TAKE?
            # The [OK] completion form is the anchor, not the [..] attempt form: the replay
            # reads the exchange history FROM the checkpoint set, so "the replay should have
            # engaged" is warranted only once the read SUCCEEDED. A read that starts and
            # fails, after which the run proceeds fresh to t=end, would be a FALSE WARN under
            # the attempt form and is correctly excluded here.
            if _TRITON_CHECKPOINT_READ_MARKER not in text:
                if _TRITON_CHECKPOINT_ATTEMPT_MARKER in text:
                    indeterminate += 1
                    continue  # INDETERMINATE — resume attempted; checkpoint read did not take
                not_resumed_last_exec += 1
                continue  # OUT OF SCOPE — last exec ran FRESH; n_resumes records superseded execs
            # (2) COMPLETION GATE — did THIS exec reach t=end? A last exec killed at
            # walltime BEFORE its replay legitimately carries no replay marker; warning on it
            # would conflate a benign kill with the rank-0 silent-skip this arm exists to find.
            if _TRITON_COMPLETION_MARKER not in text:
                indeterminate += 1
                continue  # INDETERMINATE — last exec did not run to t=end
            # (3) REPLAY TEST — the question applies and is answerable; answer it.
            examined += 1
            if _TRITON_REPLAY_MARKER not in text:
                details.append(
                    {
                        "scenario": scen_dir,
                        "detail": (
                            f"resumed (n_resumes={int(row.get('n_resumes') or 0)}) at the "
                            "pinned coupled-resume-fix TRITON: this sim's last execution read "
                            "its hotstart checkpoints and ran to t=end, but the exchange-replay "
                            "marker is ABSENT from its TRITON log — the replay never engaged, so "
                            "SWMM re-initialized from t=0 and this sim's max-flow/depth "
                            "summaries are truncated exactly as under pre-fix TRITON. Cause: "
                            "rank 0's row-strip owned no SWMM node (TRITON's engage guard is "
                            "rank-0-local, triton.h:435), or the exchange-replay side-file was "
                            "purged. Re-run from a clean start, or re-run at a rank count whose "
                            "rank-0 strip contains at least one manhole."
                        ),
                    }
                )

    n = len(details)
    passed = n == 0
    _parts = [f"{examined} resumed coupled sim(s) examined"]
    if indeterminate:
        _parts.append(
            f"{indeterminate} INDETERMINATE (unresolvable, unreadable, resume did not take, "
            "or last execution incomplete)"
        )
    if not_resumed_last_exec:
        _parts.append(
            f"{not_resumed_last_exec} out of scope (last execution ran fresh; n_resumes is a "
            "cumulative record of superseded executions)"
        )
    _denom = "; ".join(_parts)
    if passed:
        summary = f"No coupled-resume invalidity detected ({_denom})."
    elif not has_fix:
        summary = (
            f"{n} coupled sim(s) produced by PRE-FIX TRITON WITH a hotstart resume — "
            f"summaries likely invalid ({_denom})."
        )
    else:
        summary = (
            f"{n} resumed coupled sim(s) at the pinned TRITON lack the exchange-replay "
            f"marker — replay did not engage; summaries likely truncated ({_denom})."
        )
    return CheckResult(
        name="Coupled resume validity",
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
    )


def validate_analysis(analysis: TRITONSWMM_analysis) -> ValidationReport:
    """Run all core checks; return aggregated ValidationReport.

    Order matches the existing `assert_analysis_workflow_completed_successfully`
    chain so the report's check ordering matches what pytest displays.

    Persisted EDA verdicts (``{analysis_dir}/eda/*.verdict.json``, ADR-9) are
    appended after the core checks so the renderer surfaces them by ``level``.
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
            check_invalidating_fixes(analysis),  # ADR-17 registry surface
            check_coupled_resume_validity(analysis),  # post-fix retroactive coupled-resume invalidity warning
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
