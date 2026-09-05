"""Pure-Python structured validator for completed analyses.

Mirrors the assertion logic in `tests/utils_for_testing.py` (the
`assert_analysis_workflow_completed_successfully` chain) but returns
structured `CheckResult` records instead of raising `pytest.fail`. This lets
both pytest tests AND the report renderer (`report_renderers/errors_and_warnings.py`)
share the same validation logic.

Each per-check function returns one `CheckResult` describing pass/fail plus
optional per-scenario detail rows. The aggregator `validate_analysis()` runs
all 9 checks and returns a `ValidationReport`. For sensitivity analyses, the
aggregate per-scenario checks iterate members and prefix each detail
row with the member id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hhemt.member_identity import resolve_member_id_column

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

logger = logging.getLogger(__name__)


CheckLevel = Literal["system", "aggregate", "scenario", "resource"]


@dataclass
class CheckResult:
    """One check result, with optional per-scenario detail rows.

    THREE-VALUED ON APPLICABILITY, and SELF-DESCRIBING ON INSTRUMENT. Two separate
    overloads of ``passed`` are removed here, and they are independent axes:

    1. ``passed`` alone cannot distinguish "verified good" from "did not apply": a
       check returning ``passed=True`` with an ``N/A`` summary rendered as a green
       PASS, so a PASS carried no information. ``applicable=False`` is that third
       state; the renderer greys it and shows ``N/A``.
    2. ``passed`` alone cannot distinguish a verdict computed from RAW double
       rasters from one computed from the derived summary tier. Those have
       different detection floors, so an identical-looking PASS means different
       things. Measured on the compute-sensitivity campaign: the summary tier
       stores ``max_wlevel_m`` (the TRITON depth field, the one variable carrying
       the coupled-resume perturbation) as FLOAT32 while every SWMM-side variable
       is float64. float32 eps is 1.1920928955078125e-07, so a true 2.4e-08
       difference quantizes UP to ~1.19e-07 and a true 1.68e-13 difference rounds
       to EXACTLY 0.0 -- manufacturing a false "bit-identical" control. A pass at
       that floor may not render like a pass on raws (principle P7: pass should
       mean something).

    ``instrument`` vocabulary: ``"raw_rasters"`` (raw per-timestep binary output,
    the full-precision instrument), ``"summary_tier"`` (the FLAT per-scenario
    summaries -- NOT the consolidated DataTree, which the flat-summaries
    stipulation forbids these members from reading), or None when the check
    reads neither. It is SELF-REPORTED by the check from the path it actually
    took, and MUST NOT be derived from ``cfg_analysis.clear_raw``: that records
    configured intent, so a run configured to preserve raws whose outputs were
    nonetheless purged would stamp a false provenance claim on the very field
    that exists to make the verdict honest.

    ``detection_floor`` is the smallest absolute difference the instrument can
    resolve at unit magnitude -- the COARSEST floor across the variables actually
    compared, since that is the bound below which the verdict cannot see anything.
    A summary-tier verdict touching ``max_wlevel_m`` therefore reports float32
    eps, not float64 eps, even though its SWMM-side variables are float64.

    All three fields are DEFAULTED so a ``validation_report.json`` written before
    they existed deserialises unchanged. The dataclass crosses the bundle boundary
    via ``dataclasses.asdict`` and is read back by
    ``report_renderers/cross_experiment_errors_and_warnings.py``; a required field
    would break every shipped bundle.
    """

    name: str
    level: CheckLevel
    passed: bool
    summary: str
    details: list[dict] = field(default_factory=list)
    applicable: bool = True
    instrument: str | None = None
    detection_floor: float | None = None


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

        Each row carries `{stage, member_id (optional), scenario, detail}` so the
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
    """System-level: compilation success for enabled models + DEM/Mannings present.

    ADR-1/M-7 (defect-11): the three HOST-compilation assertions are gated on
    ``_native_build``. In container mode no on-cluster compile occurs by design
    (``setup_workflow.py`` skips it -- the SIF carries the binary), so asserting host
    compilation would be asserting a property that is definitionally false. What the
    triad certified in native mode ("a runnable binary exists") is certified MORE
    strongly downstream in container mode: the sim rung actually executes the binary
    out of the SIF, and ``check_scenarios_run`` / ``check_timeseries_processed`` /
    ``check_analysis_summaries_created`` all fail if it did not. The DEM/Mannings half
    below stays UNCONDITIONAL -- those artifacts are produced on the host in BOTH modes
    (``process_system_level_inputs`` runs in container mode too) and nothing downstream
    re-certifies them, so gating them would genuinely weaken this check.
    """
    cfg_sys = analysis._system.cfg_system
    issues: list[dict] = []
    sys = analysis._system

    # Compilation checks apply to NATIVE mode only. In container mode setup skips
    # the on-cluster compile (setup_workflow.py _native_compile) — the SIF carries
    # the binary — so compilation_*_successful is legitimately False and the sim runs
    # `apptainer exec {sif} {exe_in_sif}` (run_simulation.py). Appending a compilation
    # issue here would force validation_report.json overall_passed:false on a valid
    # container run. The DEM/Mannings checks below still run in both modes.
    if analysis.cfg_analysis.execution_environment != "container":
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
    # Gotcha 71(d) disclosed denominator: `passed` derives from len(issues)==0 and
    # therefore cannot distinguish "evaluated and found nothing" from "did not
    # evaluate". A system that declares a GPU backend but resolves NO GPU build dir
    # (the sensitivity-master template, whose GPU hardware varies per member)
    # ABSTAINS on the GPU compile term; name that in the artifact itself.
    if analysis.cfg_analysis.execution_environment != "container" and sys.gpu_compilation_backend:
        _abstained = [
            _label
            for _label, _toggled, _gpu_path in (
                ("TRITON-SWMM", cfg_sys.toggle_tritonswmm_model, sys.sys_paths.compilation_logfile_gpu),
                ("TRITON-only", cfg_sys.toggle_triton_model, sys.sys_paths.TRITON_build_dir_gpu),
            )
            if _toggled and _gpu_path is None
        ]
        if _abstained:
            summary += (
                f" [GPU compile term NOT evaluated for {', '.join(_abstained)}: this system "
                "declares a GPU backend but resolves no GPU build dir (master-level template; "
                "GPU hardware varies per member). CPU term evaluated; per-hardware GPU "
                "compile status is certified downstream by each member's own system.]"
            )
    return CheckResult(name="System setup", level="system", passed=passed, summary=summary, details=issues)


def _iter_members_or_self(analysis: TRITONSWMM_analysis):
    """Yield (member_id, member) for sensitivity master, else (None, analysis)."""
    sensitivity_on = getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
    sens = getattr(analysis, "sensitivity", None)
    if sensitivity_on and sens is not None:
        yield from sens.members.items()
    else:
        yield None, analysis


def _detail_rows_for_failed_scenarios(analysis: TRITONSWMM_analysis, failed_paths: list[str]) -> list[dict]:
    """Convert a list of scenario_dir strings to detail-row dicts (no member_id)."""
    return [{"scenario": str(Path(p).name), "scenario_dir": str(p), "detail": "did not complete"} for p in failed_paths]


def check_scenarios_setup(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Aggregate: all scenarios were created (per-scenario fails surfaced)."""
    details: list[dict] = []
    total = 0
    failed_count = 0
    for member_id, sub in _iter_members_or_self(analysis):
        n = int(sub.n_scenarios)
        total += n
        if not sub._all_scenarios_created:
            failed = list(sub._scenarios_not_created)
            failed_count += len(failed)
            for p in failed:
                # DO NOT RENAME the "sa_id" key written below, here or at :253 :310
                # :1863 :1954. These are CheckResult.details rows, not df_status: the
                # consumer is report_renderers/errors_and_warnings.py:405
                # `d.get("sa_id", "")`, and producer and consumer AGREE. This is a
                # different artifact from the df_status reads this file's other
                # repairs touch, and renaming it breaks a working path.
                row = {"scenario": Path(p).name, "scenario_dir": str(p), "detail": "scenario not created"}
                if member_id is not None:
                    row["sa_id"] = f"member_{member_id}"
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
    for member_id, sub in _iter_members_or_self(analysis):
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
                if member_id is not None:
                    row["sa_id"] = f"member_{member_id}"
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

    Iterates ``_iter_members_or_self(analysis)`` so the sensitivity
    member fan-out is preserved — iterating the master's own ``df_sims``
    would silently pass on a sensitivity analysis (also part of the R4 bug).
    """
    from hhemt.scenario import compute_event_id_slug
    from hhemt.summary_paths import scenario_summaries_present

    details: list[dict] = []
    total = 0
    for member_id, sub in _iter_members_or_self(analysis):
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
                if member_id is not None:
                    row["sa_id"] = f"member_{member_id}"
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
        for member_id, sub in analysis.sensitivity.members.items():
            _check_one(sub, label_prefix=f"member_{member_id}: ")
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


def _read_triton_provenance(analysis: TRITONSWMM_analysis) -> str | None:
    """Graceful-absent read of the consolidated tree's ``triton_producing_sha`` root attr.

    Returns the producing sha, or None when the attr / tree is absent or unreadable (a
    pre-provenance tree, or an off-checkout install) -> the caller treats None as
    INDETERMINATE, never a false pre-fix warn. NEVER raises.

    Returns the SHA ONLY. The two per-defect booleans this used to return are retired;
    applicability is derived from the sha by ``model_defects.resolve``, so a defect added to
    the registry later classifies trees already on disk.
    """
    import xarray as xr

    paths = analysis.analysis_paths
    if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
        zarr_path = getattr(paths, "sensitivity_datatree_zarr", None)
    else:
        zarr_path = getattr(paths, "analysis_datatree_zarr", None)
    if zarr_path is None or not zarr_path.exists():
        return None
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
        return None
    except Exception as e:
        # NEVER raise (validation must not abort a run over a diagnostic), but do
        # NOT swallow silently: an unexpected exception here means this check is
        # INERT, which is exactly how the chunks="auto" defect stayed hidden.
        logger.warning(
            f"TRITON provenance read failed unexpectedly for {zarr_path} "
            f"({type(e).__name__}: {e}); coupled-resume validity is INDETERMINATE."
        )
        return None
    sha = tree.attrs.get("triton_producing_sha")
    return str(sha) if sha is not None else None


#: The stage denominator, and the ONLY place it is defined. `setup` leads because it is
#: the rung that compiles TRITON and standalone SWMM, so it is where the non-hhemt version
#: axes are captured; omitting it made the coverage ratio unable to report on the very
#: stage the version-provenance contract is mostly about. Any caller stating a COUNT must
#: derive it from len() of this tuple -- a literal count in prose goes stale silently, and
#: did (the docstring below said "six" while this tuple held seven).
_PROVENANCE_STAGES = (
    "setup",
    "sim",
    "processing",
    "consolidate",
    "plots",
    "report",
    "bundle",
    "combine",
)


def _collect_stage_stamps(analysis) -> dict[str, dict | None]:
    """Per-stage producing stamps for this analysis; ``None`` where none was captured.

    One entry per named stage, ALWAYS all keys present, so the caller's coverage
    arithmetic has a fixed denominator and a stage that gained a capture site later
    does not silently change the ratio's meaning.

    Never infers one stage's stamp from another's. The temptation is real -- the
    bundle manifest's sha is right there and it is almost certainly the sha the plots
    ran at -- and it is exactly the fabrication this subsystem exists to prevent: a
    back-filled value is indistinguishable from a captured one at read time, which is
    what makes it worse than the absence it replaces. Every read is independently
    graceful-absent; nothing raises.
    """
    import json

    out: dict[str, dict | None] = dict.fromkeys(_PROVENANCE_STAGES)
    try:
        adir = Path(analysis.analysis_paths.analysis_dir)
    except Exception:
        return out

    def _from_json(path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        got = {k: payload[k] for k in ("hhemt_sha", "hhemt_version", "hhemt_dirty") if k in payload}
        return got or None

    def _from_tree_attrs(store: Path) -> dict | None:
        if not store.exists():
            return None
        try:
            import xarray as _xr

            attrs = _xr.open_datatree(str(store), engine="zarr", consolidated=False).attrs
        except Exception:
            return None
        sha = attrs.get("hhemt_producing_sha")
        if not sha:
            return None
        return {
            "hhemt_sha": str(sha),
            "hhemt_version": str(attrs.get("hhemt_producing_version") or ""),
            "hhemt_dirty": str(attrs.get("hhemt_producing_dirty") or "unknown"),
        }

    # consolidate / processing: the consolidated tree root carries the ADR-15 scalar
    # fast-path, and the per-event coordinate underneath it is the processing stage's
    # own capture. Resolved by existence over both tree names rather than by branching
    # on toggle_sensitivity_analysis -- a sensitivity master ships the sensitivity tree
    # and no regular one, and keying on the config would misreport a partially-built
    # tree as uncaptured.
    from hhemt.utils import ROOT_TREE_NAMES

    for name in ROOT_TREE_NAMES:
        got = _from_tree_attrs(adir / name)
        if got:
            out["consolidate"] = got
            out["processing"] = got
            break

    # plots: any figure sidecar. The stamp is per-figure and uniform within a render,
    # so the first readable sidecar is representative; a genuinely mixed render is a
    # different finding and belongs to the caller's build-disagreement branch.
    plots = adir / "plots"
    if plots.exists():
        for sidecar in sorted(plots.rglob("*.manifest.json")):
            got = _from_json(sidecar)
            if got:
                out["plots"] = got
                break

    for stage, rel in (
        ("bundle", "bundle_manifest.json"),
        ("combine", "combined_bundle_manifest.json"),
        ("report", "report_manifest.json"),
    ):
        out[stage] = _from_json(adir / rel)

    # `sim` has no capture site yet -- left as None deliberately. When that site lands it
    # wires here; until then the check reports it uncaptured, which is the true statement
    # about every analysis produced so far. `report` LANDED with the ADR-15 widening
    # (analysis.py render_report tail) and is read above; `bundle` moved from uncaptured
    # to captured in the same change, at the WRITER rather than the reader.
    return out


def check_provenance_completeness(analysis) -> CheckResult:
    """How many of the `_PROVENANCE_STAGES` left a version stamp on this analysis (ADR-15 widening).

    Reports COVERAGE, never a back-filled value. A stage with no stamp is reported as
    "not captured", which is the honest state for any analysis produced before that
    stage gained a capture site -- inferring its version from a sibling stage's stamp
    would launder an assumption into an apparent measurement, which is the EW-3 failure
    shape this subsystem exists to avoid.

    Lives here rather than only in pytest because the audience is the READER of a
    delivered report, not CI: a green build is invisible inside a bundle, whereas an
    Errors-and-Warnings row saying "3 of 6 stages captured" travels with the artifact.
    Graceful-absent throughout -- an unreadable carrier degrades that stage to "not
    captured" and never raises, because a provenance check that can abort
    validate_analysis takes the whole sidebar down with it.
    """
    stages = _collect_stage_stamps(analysis)  # {stage: stamp-dict-or-None}
    captured = sorted(s for s, v in stages.items() if v)
    missing = sorted(s for s, v in stages.items() if not v)
    dirty = sorted(s for s, v in stages.items() if v and v.get("hhemt_dirty") == "true")
    builds = {v.get("hhemt_version") for v in stages.values() if v and v.get("hhemt_version")}

    # `details`, not `detail`, and an explicit `level`. Both were wrong on all four
    # returns, so EVERY path raised TypeError and the function could not execute -- which
    # is why it was never registered, and why the test asserting it "is registered and
    # graceful" could only reach for hasattr. `level="aggregate"` routes it to the
    # per-scenario Stage table and, because tests/test_iter7_check_vocabulary.py scopes
    # its exact-equality guard to {system, resource, aggregate}, obliges the matching
    # _CHECK_VOCABULARY entry that lands in this same change.
    # `revisions` is the deliverable-2 surface: how many DISTINCT builds have produced this
    # stage over the analysis's life. The stamp columns beside it show the LATEST build
    # (contract property 3's "the report shows the LATEST per step"); this column is what
    # says an earlier one existed and was not silently replaced. Graceful-absent -- an
    # analysis predating the history capture reports 0, which is honest rather than 1,
    # because no revision was ever RECORDED even though one certainly occurred.
    from hhemt.provenance import read_stage_provenance_history

    try:
        _history = read_stage_provenance_history(Path(analysis.analysis_paths.analysis_dir))
    except Exception:
        _history = {}
    details = [
        {
            "stage": s,
            "captured": bool(stages[s]),
            "revisions": len(_history.get(s, [])),
            **(stages[s] or {}),
        }
        for s in sorted(stages)
    ]
    if dirty:
        return CheckResult(
            name="provenance_completeness",
            level="aggregate",
            passed=False,
            summary=(
                f"{len(captured)}/{len(stages)} stages stamped, but {len(dirty)} were produced "
                f"from a DIRTY toolkit checkout ({', '.join(dirty)}). The recorded sha names a "
                "commit whose content is not what ran, so a re-run at that sha would NOT "
                "reproduce this product."
            ),
            details=details,
        )
    if len(builds) > 1:
        return CheckResult(
            name="provenance_completeness",
            level="aggregate",
            passed=False,
            summary=(
                f"{len(captured)}/{len(stages)} stages stamped, but they disagree on the hhemt "
                f"build ({', '.join(sorted(builds))}). Some stage was produced by different code "
                "than the others; a single re-run cannot reproduce this mixture."
            ),
            details=details,
        )
    if missing:
        return CheckResult(
            name="provenance_completeness",
            level="aggregate",
            passed=True,
            summary=(
                f"{len(captured)}/{len(stages)} stages stamped; not captured: {', '.join(missing)}. "
                "Informational, not a failure -- an analysis produced before a stage gained its "
                "capture site legitimately has no stamp there, and no value is inferred for it."
            ),
            details=details,
        )
    return CheckResult(
        name="provenance_completeness",
        level="aggregate",
        passed=True,
        summary=f"all {len(stages)} stages stamped at one clean hhemt build.",
        details=details,
    )


def check_known_resume_defects(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Registry-verdict counterpart to ``check_coupled_resume_validity``.

    SEPARATE from that check deliberately, and the separation is measured rather than
    stylistic. That check answers an EVIDENCE question -- did the coupled replay engage
    for these sims -- by reading per-sim logs. This one answers a BUILD question -- does
    the pinned TRITON carry a known resume defect -- from the registry alone. They are
    two questions, not two verdicts on one, and a build can legitimately be clean on one
    and not the other.

    Folding this into that check was tried and reverted: any in-check branch that turns a
    present registry verdict into a non-pass flips fourteen coupled-path tests, because
    TRITON-RESUME-EXTBC-GHOST-RING is PRESENT at all three pins they use
    (15eb18a5 / b3820a44 / 9db367dd) and only 5d2ad1e8 clears it. An early return
    suppressed the refined verdict outright; a merge would have flipped
    test_postfix_with_replay_marker_passes; and firing only when no refined defect is
    present still fires at 9db367dd, which is that very test's pin.

    Trigger scoping mirrors the selection half of the version x selection cross: a
    `resumed_coupled` defect is out of scope for a pure-TRITON arm, while `resumed_any`
    and `always` apply to both. [Q130]: nothing resumed, no record, or an unresolvable
    sha all render N/A rather than a disclosed-denominator PASS.
    """
    import pandas as pd

    from hhemt.model_defects import REGISTRY, resolve

    _name = "Known resume defects"
    try:
        df = analysis.df_status
    except Exception:
        df = None
    if df is None or not {"model_type", "n_resumes"}.issubset(getattr(df, "columns", [])):
        return CheckResult(
            name=_name,
            level="aggregate",
            passed=True,
            applicable=False,
            summary="No resume record available — known-resume-defect status N/A.",
            details=[],
        )
    n_resumed = int((pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0) >= 1).sum())
    if n_resumed == 0:
        return CheckResult(
            name=_name,
            level="aggregate",
            passed=True,
            applicable=False,
            summary="No sim was resumed — known-resume-defect status N/A (0 resumed sim(s)).",
            details=[],
        )
    # Module-global call, NEVER a local import: every test in this module monkeypatches
    # `av._read_triton_provenance`, and a locally-bound name would silently bypass the
    # patch and exercise the real zarr reader.
    sha = _read_triton_provenance(analysis)
    if not sha:
        return CheckResult(
            name=_name,
            level="aggregate",
            passed=True,
            applicable=False,
            summary="Producing-TRITON sha unknown — known-resume-defect status N/A.",
            details=[],
        )
    coupled = bool(getattr(analysis._system.cfg_system, "toggle_tritonswmm_model", False))
    present = [
        d
        for d in REGISTRY
        if not (d.trigger == "resumed_coupled" and not coupled) and resolve(d, sha).status == "present"
    ]
    if present:
        return CheckResult(
            name=_name,
            level="aggregate",
            passed=False,
            summary=(
                f"{n_resumed} resumed sim(s) at a TRITON build carrying {len(present)} known "
                f"resume defect(s): {', '.join(d.defect_id for d in present)} (pin {sha[:12]})."
            ),
            details=[{"scenario": "(analysis-level)", "detail": f"{d.defect_id}: {d.remedy}"} for d in present],
        )
    return CheckResult(
        name=_name,
        level="aggregate",
        passed=True,
        summary=(f"{n_resumed} resumed sim(s) at a build carrying no known resume defect (pin {sha[:12]})."),
        details=[],
    )


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

    # WIDENED (Track 4): applicability is decided by each registry defect's TRIGGER, not by
    # the coupled toggle. `TriggerKind` is documented as a POPULATION SELECTOR whose selected
    # rows ARE the `examined` population, so an empty selection flows into the existing
    # examined-zero gate and renders N/A rather than a warning. The old toggle gate returned
    # N/A for EVERY pure-TRITON analysis, which silently excluded the one registered defect
    # that applies to both model selections (TRITON-RESUME-EXTBC-GHOST-RING, trigger=
    # "resumed_any") -- an entry no code path read at all: `resolve_all` and
    # `resolve_for_tree_attrs` have zero consumers outside model_defects.py.
    _coupled = bool(getattr(analysis._system.cfg_system, "toggle_tritonswmm_model", False))
    # MOVED (Track 4, VMS-9B) from its former position below the registry resolution.
    # `_any_resumed` is the SELECTION half of the version x selection cross, so the record
    # has to be loaded before the predicate rather than after it. Moved, never duplicated:
    # a second `analysis.df_status` read would be a second chance to disagree with the first.
    try:
        df = analysis.df_status
    except Exception:
        df = None
    if df is None or not {"model_type", "n_resumes"}.issubset(getattr(df, "columns", [])):
        # [Q130], second application. This branch previously returned passed=True with no
        # applicable flag -- a PASS asserting "no invalidity found" from a record it could
        # not read, i.e. a disclosed-denominator PASS over an UNKNOWN denominator. It now
        # renders N/A, matching the examined-zero gate below. The two not-verified states
        # share a VERDICT and are distinguished by their SUMMARY, the same way `_denom`
        # already distinguishes the three zero-examined causes.
        return CheckResult(
            name="resume validity",
            level="aggregate",
            passed=True,
            applicable=False,
            summary="No resume record available — resume validity N/A (scenario_status.csv absent or missing "
            "model_type/n_resumes).",
            details=[],
        )
    _n_resumes_col = pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0)
    _any_resumed = bool((_n_resumes_col >= 1).any())
    triton_sha = _read_triton_provenance(analysis)
    # Behavior-preserving mapping of the three retired flag states onto registry verdicts:
    #   has_fix is None  -> replay.status == "indeterminate"   (INDETERMINATE early return)
    #   not has_fix      -> replay.status == "present"         (Arm A, pre-fix TRITON)
    #   has_fix          -> replay.status == "absent"          (Arm B, post-fix)
    #   scatter is False -> scatter.status == "present"        (Arm C, missing node-depth scatter)
    # No live ancestry is attempted here: the read path has no guaranteed clone, and the
    # registry's cached sets are authored for exactly that reason.
    from hhemt.model_defects import (  # noqa: F401  (TriggerKind documents the axis)
        REGISTRY,
        REGISTRY_BY_ID,
        TriggerKind,
        resolve,
    )

    def _trigger_applies(trigger: str, *, any_resumed: bool, coupled: bool) -> bool:
        """The selection half of the version x selection cross.

        `always` is unconditional; `resumed_any` needs a resume on either model selection;
        `resumed_coupled` needs a resume AND the coupled model. Returning False makes the
        defect contribute no rows, which is what routes an unaffected arm to the
        examined-zero N/A gate instead of a vacuous PASS.
        """
        if trigger == "always":
            return True
        if trigger == "resumed_any":
            return any_resumed
        if trigger == "resumed_coupled":
            return any_resumed and coupled
        return False

    # Resolved over EVERY registry entry rather than two by name, so a defect added to the
    # registry later is evaluated without editing this check. `replay` / `scatter` are kept
    # as named locals because the arm-specific summaries below still distinguish them.
    replay = resolve(REGISTRY_BY_ID["TRITON-COUPLED-RESUME-REPLAY"], triton_sha)
    scatter = resolve(REGISTRY_BY_ID["TRITON-RESUME-DEPTH-SCATTER"], triton_sha)
    # TWO gates, not one. TRIGGER decides whether a defect COULD have been exercised by this
    # arm's run shape (did anything resume; is the coupled model selected). STATUS decides
    # whether the pinned build actually CARRIES it. A defect that resolves `absent` must admit
    # no rows at all -- it is certified not-present, and selecting rows for it converts a clean
    # registry verdict into an examined population that the in-loop evidence test then fails.
    #
    # Measured on generation e389264af7b9: synth_cc_resume_triton is stamped 5d2ad1e8, at which
    # all three registry defects resolve ABSENT, yet the shipped read-model reported
    # "28 resumed coupled sim(s) ... lack the exchange-replay marker" -- a FAIL produced by
    # TRITON-RESUME-EXTBC-GHOST-RING (trigger="resumed_any", status="absent") admitting
    # "triton" into _candidate_models on an arm the registry had just cleared.
    #
    # The former name `_applicable` meant trigger-applicable and was read as actually-affected.
    _affected = [
        (d, resolve(d, triton_sha))
        for d in REGISTRY
        if _trigger_applies(d.trigger, any_resumed=_any_resumed, coupled=_coupled)
        and resolve(d, triton_sha).status != "absent"
    ]
    if replay.status == "indeterminate":
        return CheckResult(
            name="resume validity",
            level="aggregate",
            passed=True,
            applicable=False,
            summary=(f"Producing-TRITON resume status unknown ({replay.detail}); resume validity NOT verified."),
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
    # WIDENED with the predicate. A hardcoded "tritonswmm" selects ZERO rows on a pure-TRITON
    # arm, so leaving it would route resume_triton to the examined-zero N/A gate no matter
    # what the trigger logic upstream decided -- a vacuous widening by a second route.
    # `resumed_coupled` defects contribute the coupled model only; `resumed_any` contributes
    # both. The name stays `resume_candidates` (not `resumed`): n_resumes is CUMULATIVE across
    # execs while the log is last-exec-only, so this remains a PRE-FILTER and the in-loop
    # _TRITON_CHECKPOINT_READ_MARKER still decides whether a row's LAST exec resumed.
    #
    # `_n_resumes_col` is the column computed in the moved block above; the former local
    # `n_resumes` was read at exactly this one site and is deleted rather than duplicated.
    # POSITIVE PASS (surface-3 ruling: PASS iff no sim resumed OR the version x selection pair
    # is not affected by a known issue). With _affected empty and sims that DID resume, the
    # registry has positively certified this build clean for this model selection -- that is a
    # verdict with a denominator, not an examined-zero. Falling through to the examined-zero
    # gate would render N/A and understate what was actually established.
    if _any_resumed and not _affected:
        _n_resumed = int((_n_resumes_col >= 1).sum())
        return CheckResult(
            name="resume validity",
            level="aggregate",
            passed=True,
            summary=(
                f"{_n_resumed} resumed sim(s) at a TRITON build carrying no known resume defect "
                f"for this model selection ({triton_sha[:12]}); resume validity verified from the "
                "defect registry."
            ),
            details=[],
        )

    _candidate_models: set[str] = set()
    for _d, _v in _affected:
        if _d.trigger == "resumed_coupled":
            _candidate_models.add("tritonswmm")
        elif _d.trigger in ("resumed_any", "always"):
            _candidate_models.update(("tritonswmm", "triton"))
    resume_candidates = df[df["model_type"].isin(_candidate_models) & (_n_resumes_col >= 1)]

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
    if replay.status == "present":
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
        # `_iter_members_or_self` yields (member_id, sub) for a sensitivity master and
        # (None, analysis) otherwise, which is ALREADY keyed the way `row.get("sa_id")`
        # reads: a non-sensitivity df_status carries no member_id column (only
        # sensitivity_analysis.df_status adds it), so `.get` returns None and hits the
        # None key. str-normalized per the member_id-cast-to-string stipulation, mirroring
        # per_analysis_summary.py's `astype(str) == str(member_id)` precedent.
        subs = {(str(k) if k is not None else None): v for k, v in _iter_members_or_self(analysis)}
        # DURABLE FALLBACK (Q4): the model log is opened "w" per exec and can be purged/cleared,
        # so when read_text() below raises we consult the per-sub replay evidence stamped onto the
        # consolidated tree ROOT at consolidation time (_stamp_coupled_resume_evidence). Read once,
        # best-effort; keyed identically to the log-path branch (str(member_id) else scenario_directory).
        import json

        _tree_ev: dict = {}
        try:
            _p = (
                analysis.analysis_paths.sensitivity_datatree_zarr
                if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
                else analysis.analysis_paths.analysis_datatree_zarr
            )
            if _p is not None and _p.exists():
                import xarray as xr

                _tree_ev = json.loads(
                    xr.open_datatree(_p, engine="zarr", consolidated=False).attrs.get(
                        "coupled_resume_replay_evidence", "{}"
                    )
                )
        except Exception:  # noqa: BLE001 — durable-evidence fallback is best-effort; absence -> log-only Arm B
            _tree_ev = {}
        _id_col = resolve_member_id_column(getattr(resume_candidates, "columns", []))
        for _, row in resume_candidates.iterrows():
            scen_dir = str(row.get("scenario_directory", ""))
            _sa = row.get(_id_col) if _id_col else None
            sub = subs.get(str(_sa) if _sa is not None else None)
            if sub is None:
                indeterminate += 1
                continue  # INDETERMINATE — cannot resolve the owning (sub-)analysis
            try:
                # FOURTH hardcoded-coupled site (Track 4). The widened `_candidate_models`
                # filter now selects pure-TRITON rows, but a literal "tritonswmm" here builds
                # a log path that does not exist for them -- the read raises, the row falls to
                # the stamp fallback, finds nothing, and is counted INDETERMINATE, so the arm
                # is selected and then silently dropped. Measured with a row-model x log-path
                # probe against the applied code: a row declaring model_type="triton" became
                # EXAMINED only when a model_tritonswmm_*.log was present, proving the read
                # ignored the row. Key the path on the row's own model_type.
                _row_model = str(row.get("model_type") or "tritonswmm")
                text = model_logfile_for(sub, int(row["event_iloc"]), _row_model).read_text()
            except Exception:
                # Log gone (purged / cache-cleared): fall back to the durable consolidation-time
                # stamp before conceding INDETERMINATE. A stamped resumed+completed sub is EXAMINED
                # (positive if replayed, a real warn if not); only a sub with no log AND no stamp
                # stays INDETERMINATE.
                _ev = _tree_ev.get(str(_sa) if _sa is not None else scen_dir)
                if _ev and _ev.get("resumed") and _ev.get("completed"):
                    examined += 1
                    if not _ev.get("replayed"):
                        details.append(
                            {
                                "scenario": scen_dir,
                                "detail": (
                                    "resumed coupled sim; durable replay-evidence stamp shows the "
                                    "replay did NOT engage (log purged; tree-stamped at "
                                    "consolidation). Summaries likely truncated."
                                ),
                            }
                        )
                    continue
                indeterminate += 1
                continue  # INDETERMINATE — log gone and no durable stamp
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

    # ---- ARM C (S8): SWMM node-depth scatter absent on the resume path. -------------
    # The defect: replay_exchange_history rebuilds SWMM node depths into rank 0's
    # global_new_depth[], but the global_to_local + MPI_Scatterv that distributes them
    # into the per-rank new_depth[] is ABSENT on the resume path. The first post-resume
    # step evaluates every manhole at new_depth = 0, forces the Case-1 exchange branch,
    # flips the sign of the surface/sewer flux at every junction, and writes a permanent
    # perturbation into TRITON's depth field.
    #
    # Why this arm cannot be folded into A or B: Arm A fires on PRE-3a832f7d code; Arm B
    # fires when the replay marker is ABSENT. This defect occurs at a POST-fix pin WITH the
    # replay marker present — the replay ran, and its result was then discarded by the
    # missing scatter. Arms A and B are both silent on it by construction.
    #
    # Why it is INVISIBLE to the artifacts the existing arms read: measured on this
    # campaign, max_flow_cms is identical on 14/14 coupled configs while max_wlevel_m
    # differs on 14/14, and SWMM's own hydraulics.rpt link/node maxima are unchanged at
    # reported precision because the interface error collapses from 1.45e+02 cfs to
    # 2.1e-03 within ~1000 steps. Only TRITON's own H/MH rasters carry it. No amount of
    # rpt/summary reading can detect it, which is why this arm is PIN-CONDITIONED rather
    # than evidence-conditioned.
    #
    # SUPPRESSION (reconciliation with Arm A, required so a doubly-affected sim does not
    # receive contradictory guidance): when has_fix is False, Arm A has already fired and
    # its remedy — re-run at a pin carrying BOTH fixes — subsumes this one. Arm C is
    # therefore evaluated only on the post-fix branch.
    #
    # `is False`, NOT `not`: an ABSENT or unstampable scatter attr reads None
    # (INDETERMINATE — the clone did not know the sha, system.py:830), and `not None` is
    # True, so the bare-truthiness form turned "unknown" into a positive invalidity claim
    # on every unstamped tree. This mirrors the has_fix is-None early return above.
    if replay.status == "absent" and scatter.status == "present":
        for _, row in resume_candidates.iterrows():
            details.append(
                {
                    "scenario": str(row.get("scenario_directory", "")),
                    "detail": (
                        f"coupled hotstart resume (n_resumes={int(row.get('n_resumes') or 0)}) at a "
                        f"TRITON pin ({triton_sha}) that lacks the SWMM node-depth SCATTER on the "
                        "resume path: replay_exchange_history rebuilds node depths into rank 0's "
                        "global_new_depth[] but never scatters them into the per-rank new_depth[], "
                        "so the first post-resume step evaluates every manhole at new_depth=0 and "
                        "writes a permanent perturbation into TRITON's depth field. TRITON-side "
                        "max_wlevel_m / H / MH from this sim are INVALID; SWMM-side max_flow_cms "
                        "and hydraulics.rpt maxima are unaffected at reported precision and are "
                        "NOT evidence of validity. Re-run once the upstream scatter fix lands "
                        "(_PINNED_TRITON_SWMM_DEPTH_SCATTER_FIX_SHA)."
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
    # EW-2b (principle P7, generalized): a check that EXAMINED NOTHING has not applied.
    # `passed = n == 0` cannot distinguish "examined 28, found nothing" from "examined 0",
    # and the second is a green cell asserting a verification that never ran — measured on
    # the Iteration-5 combined report as clean_tritonswmm rendering PASS with
    # "(0 resumed coupled sim(s) examined)". The disclosed denominator (Gotcha-71(d)) is NOT
    # a substitute: it lives behind a hover title while the grid shows green.
    #
    # `not details` is LOAD-BEARING, not defensive. Arm C (:931) appends a detail row for
    # EVERY resume candidate WITHOUT incrementing `examined` — the counter is touched only at
    # :766 (Arm A) and :847/:881 (Arm B). So a scatter-pin analysis whose logs were purged
    # yields examined == 0 WITH findings, and a bare `examined == 0` test would convert that
    # real FAIL into a grey N/A. Gate on BOTH conjuncts, and see
    # test_armc_zero_examined_finding_is_not_silenced.
    #
    # `_denom` is REUSED rather than replaced so the three zero-examined causes stay
    # distinguishable on hover (nothing in scope / all INDETERMINATE / all out of scope), and
    # so the four existing INDETERMINATE + out-of-scope tests keep passing unchanged.
    if examined == 0 and not details:
        return CheckResult(
            name="resume validity",
            level="aggregate",
            passed=True,
            applicable=False,
            summary=f"No resumed coupled sim was examined — coupled-resume validity N/A ({_denom}).",
            details=[],
        )
    if passed:
        summary = f"No coupled-resume invalidity detected ({_denom})."
    elif replay.status == "present":
        summary = (
            f"{n} coupled sim(s) produced by PRE-FIX TRITON WITH a hotstart resume — "
            f"summaries likely invalid ({_denom})."
        )
    elif scatter.status == "present":
        # Arm C: post-fix pin, replay marker PRESENT, but the replayed depths never reach
        # the per-rank new_depth[]. Distinguished from Arm B because the remedy differs —
        # Arm B is re-runnable now, Arm C waits on an upstream TRITON fix.
        summary = (
            f"{n} resumed coupled sim(s) ran at a pin lacking the SWMM node-depth scatter "
            f"— TRITON depth fields are invalid ({_denom})."
        )
    else:
        summary = (
            f"{n} resumed coupled sim(s) at the pinned TRITON lack the exchange-replay "
            f"marker — replay did not engage; summaries likely truncated ({_denom})."
        )
    return CheckResult(
        name="resume validity",
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
    )


def check_resume_schedule_honored(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Warn when a resumed sim's REALIZED resume did not honor the CONFIGURED schedule.

    DISJOINT from ``check_coupled_resume_validity`` (which tests whether the coupled
    replay ENGAGED — the PRESENCE of the replay marker): this check tests whether the
    realized resume reached the LAST CONFIGURED boundary. The two never count the same
    row: Arm A here fires only when the marker is PRESENT (``replay_matches_schedule``
    is a real bool) but mis-positioned, whereas ``check_coupled_resume_validity`` fires
    only when the marker is ABSENT. Two arms, disclosed as an asymmetry:

    (A) COUPLED (tritonswmm): reads the durable ``coupled_resume_replay_evidence`` stamp
        (``processing_analysis._stamp_coupled_resume_evidence``), which now carries
        ``replay_matches_schedule`` = ``replay_t >= schedule[-1] * reporting_interval_s``
        (coupled-only, unit-matched). ``False`` -> the replay engaged but landed BEFORE
        the last scheduled boundary (schedule truncated / not honored); ``None`` (no
        replay_t or no configured schedule) -> INDETERMINATE, never a warn.
    (B) PURE-TRITON (triton): emits NO replay marker, so ``replay_t`` is structurally
        unavailable; its schedule is instead verified by ``n_resumes == len(schedule)``
        from ``df_status``. A mismatch -> warn. This arm ASYMMETRY is disclosed in every
        pure-TRITON detail row and in the summary.

    ``level="aggregate"``. Non-blocking (``validate_analysis`` never raises). Graceful-
    absent throughout: no stamp / no configured schedule -> INDETERMINATE INFO
    (``passed=True``). Disclosed denominator (Gotcha-71(d)): examined + INDETERMINATE
    counts are named in the summary.
    """
    import json

    import pandas as pd

    cfg_sys = analysis._system.cfg_system
    coupled_on = bool(getattr(cfg_sys, "toggle_tritonswmm_model", False))
    triton_on = bool(getattr(cfg_sys, "toggle_triton_model", False))
    if not (coupled_on or triton_on):
        return CheckResult(
            name="Resume schedule honored",
            level="aggregate",
            passed=True,
            applicable=False,
            summary="Neither coupled nor pure-TRITON model enabled — resume-schedule verification N/A.",
            details=[],
        )

    details: list[dict] = []
    examined = 0
    indeterminate = 0

    # --- Arm A: COUPLED — replay_t vs schedule[-1]*interval, from the durable stamp ---
    if coupled_on:
        _ev: dict = {}
        try:
            _p = (
                analysis.analysis_paths.sensitivity_datatree_zarr
                if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
                else analysis.analysis_paths.analysis_datatree_zarr
            )
            if _p is not None and _p.exists():
                import xarray as xr

                _ev = json.loads(
                    xr.open_datatree(_p, engine="zarr", consolidated=False).attrs.get(
                        "coupled_resume_replay_evidence", "{}"
                    )
                )
        except Exception:  # noqa: BLE001 — durable-stamp read is best-effort; absence -> Arm A N/A
            _ev = {}
        for key, rec in sorted(_ev.items()):
            matches = rec.get("replay_matches_schedule")
            if matches is None:
                indeterminate += 1
                continue  # no replay_t or no configured schedule -> cannot verify position
            examined += 1
            if matches is False:
                details.append(
                    {
                        "scenario": key,
                        "detail": (
                            f"coupled resume replayed to t={rec.get('replay_t')}s but the last "
                            f"configured boundary is t={rec.get('expected_replay_t')}s "
                            "(schedule[-1] * TRITON_reporting_timestep_s); the realized resume did "
                            "not reach the last scheduled interruption, so the coupled state was "
                            "replayed short and later summaries may be truncated."
                        ),
                    }
                )

    # --- Arm B: PURE-TRITON — n_resumes == len(schedule) (arm asymmetry: no replay_t) ---
    if triton_on:
        try:
            df = analysis.df_status
        except Exception:
            df = None
        if df is not None and {"model_type", "n_resumes"}.issubset(getattr(df, "columns", [])):
            n_res = pd.to_numeric(df["n_resumes"], errors="coerce").fillna(0)
            triton_resumed = df[(df["model_type"] == "triton") & (n_res >= 1)]
            subs = {(str(k) if k is not None else None): v for k, v in _iter_members_or_self(analysis)}
            _id_col = resolve_member_id_column(getattr(triton_resumed, "columns", []))
            for _, row in triton_resumed.iterrows():
                _sa = row.get(_id_col) if _id_col else None
                sub = subs.get(str(_sa) if _sa is not None else None)
                sched = getattr(getattr(sub, "cfg_analysis", None), "resume_interruption_schedule", None)
                if not sched:
                    indeterminate += 1
                    continue  # no configured schedule -> cannot verify the resume count
                examined += 1
                n_r = int(row.get("n_resumes") or 0)
                if n_r != len(sched):
                    details.append(
                        {
                            "scenario": str(row.get("scenario_directory", "")),
                            "detail": (
                                f"pure-TRITON resume count n_resumes={n_r} != {len(sched)} configured "
                                "interruption(s) (resume_interruption_schedule). The pure-TRITON arm "
                                "emits no replay marker, so its schedule is verified by the resume "
                                "count (arm asymmetry) — a mismatch means the schedule was not honored."
                            ),
                        }
                    )
                else:
                    # KR-a/KR-b: the COUNT alone cannot distinguish "resumed 3 times at
                    # 36/72/108" from "resumed 3 times at 41/79/113" — the per-config
                    # boundary variance the deterministic prune exists to eliminate. KR-b
                    # persists the REALIZED boundary per resume, so compare it directly.
                    _realized = row.get("resume_reporting_tsteps") or []
                    if not _realized:
                        # ABSENCE OF EVIDENCE IS NOT EVIDENCE OF CORRECTNESS. A resumed sub
                        # with a configured schedule but no realized-boundary list cannot be
                        # verified, and silently passing here is the exact shape of a re-sim
                        # that never re-simulated: re-running the resume arm WITHOUT
                        # start_from_scratch leaves the prior campaign's n_resumes in place,
                        # so no kill arms (the gate is _n_done < len(schedule)), the count
                        # check above passes, and a pre-KR-b tree carries no list to compare
                        # — every detector green over data the deterministic prune never
                        # touched. Surface it as INDETERMINATE (cannot verify) rather than
                        # as a failure: a genuinely legacy pre-KR-b tree is unverifiable-but-
                        # fine, and the honest report of that is "unverifiable", not "ok".
                        indeterminate += 1
                        details.append(
                            {
                                "scenario": str(row.get("scenario_directory", "")),
                                "detail": (
                                    f"pure-TRITON resume count n_resumes={n_r} matches the {len(sched)} "
                                    "configured interruption(s), but NO realized resume boundaries were "
                                    "recorded (resume_reporting_tsteps absent/empty), so the "
                                    "same-timestep claim CANNOT BE VERIFIED. Either this tree predates "
                                    "realized-boundary persistence, or the resume arm was re-run over an "
                                    "existing analysis dir without start_from_scratch — in which case no "
                                    "interruption fired and the data is clean-arm data carrying the "
                                    "resume arm's label."
                                ),
                            }
                        )
                    elif [int(t) for t in _realized] != [int(s) for s in sched]:
                        details.append(
                            {
                                "scenario": str(row.get("scenario_directory", "")),
                                "detail": (
                                    f"pure-TRITON realized resume boundaries {list(_realized)} != "
                                    f"configured schedule {list(sched)}. The resume COUNT is correct, "
                                    "so the harness fired the right number of interruptions, but at "
                                    "least one config resumed from a different reporting step than "
                                    "requested — the b4b same-timestep comparison across configs is "
                                    "NOT valid for this sim."
                                ),
                            }
                        )

    n = len(details)
    passed = n == 0
    _parts = [f"{examined} resumed sim(s) schedule-verified"]
    if indeterminate:
        _parts.append(f"{indeterminate} INDETERMINATE (no replay_t / no configured schedule)")
    _denom = "; ".join(_parts)
    # EW-2b: see the twin gate in check_coupled_resume_validity. Both clean arms rendered a
    # green PASS here carrying "(0 resumed sim(s) schedule-verified)" — this check touches
    # BOTH models, so the vacuous cell appeared on clean_triton as well as clean_tritonswmm.
    # `not details` is kept for symmetry AND because Arm B's unverifiable-boundaries branch
    # (:1129) increments `indeterminate` while ALSO appending a detail row, so details can be
    # non-empty with examined == 0 here too; that row is a real surfaced finding and must not
    # be greyed out.
    if examined == 0 and not details:
        return CheckResult(
            name="Resume schedule honored",
            level="aggregate",
            passed=True,
            applicable=False,
            summary=f"No resumed sim was examined — resume-schedule verification N/A ({_denom}).",
            details=[],
        )
    summary = (
        f"All resumed sims honored their configured resume schedule ({_denom})."
        if passed
        else f"{n} resumed sim(s) did not honor the configured resume schedule ({_denom})."
    )
    return CheckResult(
        name="Resume schedule honored",
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
    )


def _enumerated_eda_templates(analysis: TRITONSWMM_analysis) -> tuple:
    """The EDA rule_spec_templates this analysis ENUMERATES as report targets, or ().

    Mirrors the Snakemake rule-all enumeration gate (workflow.py:7753-7757 and its
    reprocess-master twin at :8495-8499) TERM FOR TERM. All four terms matter:

    1. Only the sensitivity-master generators carry an EDA enumeration site at all.
       ``generate_snakefile_content`` (multisim) and ``reprocess_snakefile_generator``
       contain none, and the multisim plot dispatcher passes no ``predicate_inputs``,
       so a multisim neither emits nor enumerates EDA rules whatever set it names.
    2. The active set must carry an ``eda_compute_sensitivity`` selection. ``default``
       and ``benchmarking`` do not; ``compute-sensitivity``/``dem-resolution``/``b4b``
       do. This is the discriminating term.
    3. ``eda.enabled_plots`` must be non-empty — TRUE by default (non-empty
       default_factory), which is why it cannot carry the predicate alone.
    4. The builder key must not be in ``report_config.disabled_renderers``.

    Set resolution goes through ``resolve_active_reporting_set``, which carries BOTH
    the sentinel rule and the registry membership check and then composes the named
    sets — never an inline re-derivation of the sentinel branch. That shortcut drops
    the membership check, turning a typo'd reporting_set into a bare ``KeyError`` at
    the registry lookup instead of a field-named ``ConfigurationError``; the
    bundle-side harvest took it once and it had to be repaired.
    The ``_active_reporting_set`` / ``_cfg_report`` fallback chain mirrors
    ``workflow.py::_resolve_active_reporting_set`` so validate-time and generate-time
    resolve the same set on a generate-without-run() tree.
    """
    from hhemt.config.report import resolve_active_reporting_set
    from hhemt.report_renderers._reporting_sets import (
        eda_rule_spec_templates,
        renderer_active,
    )

    if not getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
        return ()
    if not list(getattr(getattr(analysis.cfg_analysis, "eda", None), "enabled_plots", []) or []):
        return ()
    cfg_report = getattr(analysis, "_cfg_report", None) or analysis.cfg_analysis.report
    if not renderer_active("eda_compute_sensitivity", list(cfg_report.disabled_renderers)):
        return ()
    active = getattr(analysis, "_active_reporting_set", None)
    if active is None:
        try:
            active = resolve_active_reporting_set(cfg_report, is_sensitivity=True)
        except Exception:  # unresolvable/typo'd set -> run-entry validation owns the error
            return ()
    return tuple(eda_rule_spec_templates(active))


def check_eda_calc_ran(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Fail when EDA figures are ENUMERATED as report targets but the EDA calc never ran.

    The gap this closes: a master can complete, render a report that post-dates its own
    scenario_status.csv, and pass every currency check while ``{master}/eda/`` does not
    exist — because ``analysis.eda()`` was never invoked. Every DoD line naming
    ``eda/*.verdict.json`` as its evidence source is then unevidenceable and the EDA figures
    render as honest-degradation panels.

    Those degradation panels are the enumerate-implies-emit fix working as designed: they
    replaced a workflow-killing MissingOutputException with a survivable panel. That trade is
    correct. Its side effect is that the LOUD failure is gone and no positive signal replaced
    it. This check is the positive signal.

    The predicate mirrors the Snakemake rule-all ENUMERATION GATE term for term, via
    ``_enumerated_eda_templates``: if the workflow enumerated EDA figures, the EDA calc
    owed verdicts. An analysis that enumerates none — a multisim, a ``benchmarking``
    master, a master with ``enabled_plots`` empty, or one that disabled the renderer —
    renders no degradation panels and passes trivially. Keying on ``enabled_plots``
    alone would NOT do: it carries a non-empty default_factory and is therefore true on
    essentially every analysis.
    """
    name = "EDA calc ran"
    templates = _enumerated_eda_templates(analysis)
    if not templates:
        return CheckResult(
            name=name,
            level="aggregate",
            passed=True,
            applicable=False,
            summary="This analysis enumerates no EDA report targets — EDA completeness N/A.",
            details=[],
        )
    n_targets = len(templates)
    eda_dir = Path(analysis.analysis_paths.analysis_dir) / "eda"
    verdicts = sorted(eda_dir.glob("*.verdict.json")) if eda_dir.is_dir() else []
    if verdicts:
        return CheckResult(
            name=name,
            level="aggregate",
            passed=True,
            summary=(
                f"{n_targets} EDA report target(s) enumerated; {len(verdicts)} verdict artifact(s) present "
                f"under {eda_dir.name}/."
            ),
            details=[],
        )
    return CheckResult(
        name=name,
        level="aggregate",
        passed=False,
        summary=(
            f"{n_targets} EDA plot(s) are enumerated as report targets but {eda_dir} carries no "
            "*.verdict.json — analysis.eda() never ran for this analysis. Every EDA figure in the "
            "report is an honest-degradation panel, and any claim sourced from eda/*.verdict.json "
            "is unevidenceable. Run `hhemt eda` (or analysis.eda()) BEFORE render_report() and "
            "before bundle_report_data()."
        ),
        details=[
            {
                "scenario": "(analysis-level)",
                "detail": (
                    f"enumerated_targets={[t.rule_name for t in templates]}; "
                    f"eda_dir_exists={eda_dir.is_dir()}; verdict_count=0"
                ),
            }
        ],
    )


_RECLAIM_LOG_FIELDS: dict[str, str] = {
    "full_TRITON_timeseries_cleared": "TRITON timeseries",
    "full_SWMM_timeseries_cleared": "SWMM node/link timeseries",
    "raw_SWMM_binaries_reclaimed": "coupled raw SWMM .out binaries",
    "coupled_rpt_truncated": "coupled SWMM report body (truncated in place)",
    "hydro_out_reclaimed": "SWMM hydrology output (hydro.out)",
    "prep_inputs_reclaimed": "scenario-prep inputs (dats/, extbc/, sim_weather.nc)",
    "hydrographs_reclaimed": "TRITON inflow hydrographs (strmflow/), captured to zarr first",
    "standalone_rpt_reclaimed": "standalone SWMM reports (full.rpt truncated; hydro.rpt captured then removed)",
}


#: Minutes of slack allowed before a per-cell time-of-maximum counts as post-forcing.
#: The terminal TRITON snapshot lands ON the forced end by construction -- post-trim the
#: simulation duration IS the forcing extent -- so the comparison is an equality case in
#: the normal state and the strict ``>`` is what keeps a healthy run green. That leaves a
#: one-ULP hazard: ``time_of_max_wlevel_min`` is an accumulation of reporting-step
#: multiples and visibly carries float error (measured on the synth fixture: 24.666666666666664
#: for an exact 24.666666666666668), so a terminal snapshot drifting one ULP the other way
#: would fire on EVERY healthy analysis. This tolerance is bounded on both sides rather than
#: picked: ~1e-6 min is eight orders ABOVE the float64 ULP at these magnitudes
#: (2.8e-14 min at 180 min) and eight orders BELOW the smallest reporting step anyone
#: configures (seconds). It cannot mask the regression this check exists to find, which is
#: a whole-simulation-scale overrun, not a microsecond.
_FORCING_TAIL_TOLERANCE_MIN = 1e-6


def _forced_extent_minutes(weather_nc, time_dim: str) -> float | None:
    """Minutes spanned by a per-scenario sim_weather.nc time coordinate, or None.

    THE INTERVAL COMES FROM THE COORDINATE, NEVER FROM A CONFIG FIELD. An earlier
    form of this took the step COUNT from the weather axis and the step INTERVAL
    from ``TRITON_reporting_timestep_s`` -- two different clocks. They coincide on
    the Norfolk campaigns (weather step 120 s, reporting timestep 120 s), which is
    exactly why the error was invisible where it was authored; on the synth fixture
    (weather step 60 s, reporting timestep 10 s) it understated the extent 6x, so a
    healthy 180-minute run reported a forced end of 30 minutes and classified nearly
    the whole simulation as post-forcing. The consumer is the regression detector for
    the window trim, whose whole value is being independent of the config under
    suspicion, so a config-derived interval defeated the instrument rather than
    merely mis-scaling it.

    ``time_dim`` is a schema LABEL, not a clock -- it names which coordinate to read
    and contributes no quantity to the arithmetic.

    Returns None when the extent cannot be derived (missing coordinate, non-datetime
    coordinate, fewer than two steps). The caller counts that as INDETERMINATE rather
    than assuming a value, because a guessed extent in this check manufactures the
    exact false positive it exists to detect.
    """
    import numpy as np
    import xarray as xr

    try:
        with xr.open_dataset(weather_nc, engine="h5netcdf") as ds:
            if time_dim not in ds.coords and time_dim not in ds.variables:
                return None
            values = ds[time_dim].values
    except Exception:
        return None
    if values.ndim != 1 or values.size < 2:
        return None
    if not np.issubdtype(values.dtype, np.datetime64):
        return None
    return float((values[-1] - values[0]) / np.timedelta64(1, "m"))


def check_forcing_tail_influence(analysis) -> CheckResult:
    """Aggregate: did any per-cell maximum occur AFTER that event's forcing ended?

    Post-trim this is unreachable through the normal path, which is the point --
    it is a REGRESSION detector on the window trim, not an ongoing science check.
    Its live reachability case is a hotstart resume against a stale checkpoint cfg
    carrying the pre-fix sim_duration, which prep-time assertions have already
    passed by. The instrument is deliberately independent of the cfg under
    suspicion: the forced extent is re-derived from the per-scenario sim_weather.nc
    time COORDINATE -- both the span and the interval, never a config field.

    Honest limit, restated so it is not lost: a max occurring after the forcing
    ended proves the tail SET that maximum. It does not prove the tail was
    artefactual -- real inland routing continues after real forcing stops.

    DISCLOSED DENOMINATOR: the summary names examined and indeterminate counts, so
    a vacuous pass is legible in the artifact rather than reading as a verified good.
    """
    import numpy as np
    import xarray as xr

    from hhemt.scenario import compute_event_id_slug

    details: list[dict] = []
    examined = 0
    indeterminate = 0

    # Iterate `_iter_members_or_self` like every other sensitivity-aware check in this
    # module. A sensitivity MASTER carries `n_scenarios == n_events * n_subs` while its own
    # `df_sims` holds only the events, so the previous master-scoped `range(n_scenarios)` +
    # `analysis._retrieve_weather_indexer_using_integer_index(i)` raised `KeyError` on the
    # first index past the event count (measured on the synth fixture: n_scenarios=4,
    # df_sims index=[0] -> KeyError: 1). That raise propagated out of `validate_analysis`,
    # so `persist_validation_report` -- whose four call sites ALL swallow it as non-fatal
    # (export_scenario_status.py:444 and :540, consolidate_workflow.py:489,
    # analysis.py:1081) -- silently wrote NO master-level validation_report.json at all,
    # while every sub wrote its own. The master scope was also the wrong sims root: a
    # master's `sims/` is empty (scenarios live under `members/member_N/sims/`), so even
    # without the raise this check examined zero scenarios on every sensitivity run.
    for member_id, sub in _iter_members_or_self(analysis):
        sims_dir = Path(sub.analysis_paths.simulation_directory)
        for i in range(int(sub.n_scenarios)):
            evt = compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(i))
            scen = sims_dir / evt
            wx = scen / "sim_weather.nc"
            summaries = sorted((scen / "processed").glob("*TRITON_summary.zarr"))
            if not wx.exists() or not summaries:
                indeterminate += 1
                continue
            try:
                forced_end_min = _forced_extent_minutes(
                    wx, sub.cfg_analysis.weather_time_series_timestep_dimension_name
                )
                if forced_end_min is None:
                    indeterminate += 1
                    continue
                zarr = xr.open_zarr(summaries[0], consolidated=False)
                tmx = zarr["time_of_max_wlevel_min"].isel(event_iloc=0).values.astype(float)
                mx = zarr["max_wlevel_m"].isel(event_iloc=0).values.astype(float)
            except Exception:
                indeterminate += 1
                continue
            examined += 1
            wet = np.isfinite(tmx) & (mx > 0.01)
            if not wet.any():
                continue
            after = int((tmx[wet] > forced_end_min + _FORCING_TAIL_TOLERANCE_MIN).sum())
            if after:
                row = {
                    "event_id": evt,
                    "cells_max_after_forcing": after,
                    "wet_cells": int(wet.sum()),
                    "forced_end_min": forced_end_min,
                }
                if member_id is not None:
                    row["sa_id"] = f"member_{member_id}"
                details.append(row)

    return CheckResult(
        name="forcing tail influence",
        level="aggregate",
        passed=not details,
        applicable=examined > 0,
        instrument="summary_tier",
        summary=(
            f"examined {examined} scenario(s), {indeterminate} indeterminate; "
            f"{len(details)} with a per-cell maximum attained after their forcing ended"
        ),
        details=details,
    )


def check_data_availability(analysis: TRITONSWMM_analysis) -> CheckResult:
    """Aggregate: which per-scenario artifact classes were DELIBERATELY reclaimed.

    A deliberate reclaim is a fact about data availability, not an error -- so this check
    passes with a descriptive summary in the normal case and fails only in the one state
    that is genuinely wrong: a scenario whose log records a reclaim while its per-model
    summary set is ABSENT. That is the state in which the reclaim stopped being a reclaim
    and became damage.

    Three properties are load-bearing and none is stylistic.

    1. PATH-ONLY. The per-model log is read straight off disk at
       ``sims/{event_id}/log_{model}.json``; ``TRITONSWMM_scenario`` is never instantiated,
       because its constructor mkdir's ``processed/``/``swmm/``/``out_swmm/`` as a side
       effect (the discipline ``summary_paths.py`` codifies). It matters more here than
       there: this runs over 11,394 model-sims on the synthetic ensemble.

    2. IT READS THE LOG, NEVER THE CONFIG. ``CheckResult``'s own docstring forbids deriving
       ``instrument`` from ``cfg_analysis.clear_raw`` because "that records configured
       intent". The symmetric error here would be labelling a scenario reclaimed from
       ``cfg_analysis.remove_after_processing`` -- a scenario that failed before the
       reclaim fired, or one processed by a pre-feature toolkit, would be mislabelled.

    3. DISCLOSED DENOMINATOR. Every ``check_*`` derives ``passed`` from ``len(details) == 0``,
       which cannot distinguish "examined N, found nothing" from "examined 0". The summary
       names the examined and indeterminate counts so a vacuous pass is legible in the
       artifact itself.
    """
    import json

    from hhemt.scenario import compute_event_id_slug
    from hhemt.summary_paths import scenario_summaries_present

    details: list[dict] = []
    examined = 0
    indeterminate = 0
    reclaimed_counts: dict[str, int] = {label: 0 for label in _RECLAIM_LOG_FIELDS.values()}

    for member_id, sub in _iter_members_or_self(analysis):
        try:
            enabled = sub._get_enabled_model_types()
            sim_dir = Path(sub.analysis_paths.simulation_directory)
        except Exception:  # noqa: BLE001 -- a sub we cannot address is indeterminate, not failed
            continue
        for event_iloc in sub.df_sims.index:
            ev = sub._retrieve_weather_indexer_using_integer_index(event_iloc)
            event_id = compute_event_id_slug(ev)
            for model in enabled:
                examined += 1
                logf = sim_dir / event_id / f"log_{model}.json"
                if not logf.exists():
                    indeterminate += 1
                    continue
                try:
                    payload = json.loads(logf.read_text())
                except (OSError, ValueError):
                    indeterminate += 1
                    continue
                classes = [label for field, label in _RECLAIM_LOG_FIELDS.items() if payload.get(field) is True]
                if not classes:
                    continue
                for label in classes:
                    reclaimed_counts[label] += 1
                if not scenario_summaries_present(sub, event_id, [model]):
                    row = {
                        "scenario": event_id,
                        "scenario_dir": str(sim_dir / event_id),
                        "detail": (
                            f"{model}: reclaim recorded ({', '.join(classes)}) but the "
                            "per-model summary set is ABSENT -- the reclaim is damage, not "
                            "a disclosed reclaim"
                        ),
                    }
                    if member_id is not None:
                        row["sa_id"] = f"member_{member_id}"
                    details.append(row)

    passed = not details
    reclaimed_bits = [f"{label}: {n}" for label, n in reclaimed_counts.items() if n]
    if not reclaimed_bits:
        headline = "No per-scenario artifacts were reclaimed"
    else:
        headline = "Reclaimed after processing -- " + "; ".join(reclaimed_bits)
    summary = (
        f"{headline}. Examined {examined} (model, scenario) pair(s); "
        f"{indeterminate} indeterminate (no readable model log)."
    )
    if not passed:
        summary = (
            f"{len(details)} of {examined} (model, scenario) pair(s) record a reclaim with "
            f"their summaries absent. {summary}"
        )
    return CheckResult(
        name="Data availability",
        level="aggregate",
        passed=passed,
        summary=summary,
        details=details,
        instrument="summary_tier",
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
            check_known_resume_defects(analysis),  # registry-verdict counterpart (build-level, no log evidence)
            check_resume_schedule_honored(analysis),  # Phase 5: replay_t / n_resumes vs configured schedule
            check_eda_calc_ran(analysis),  # F4: enumerated EDA figures vs present eda/*.verdict.json
            check_data_availability(analysis),  # reclaim disclosure: which artifact classes were reclaimed
            check_forcing_tail_influence(analysis),  # regression detector: maxima set after the forcing ended
            # Registered ONLY together with the kwarg repair above, never before it. This
            # list is built eagerly and all four persist_validation_report call sites swallow
            # exceptions non-fatally, so a wired-but-raising check leaves the read-model
            # unwritten -- and what that produces FORKS on prior state. On a FRESH tree
            # validation_report.json is missing, workflow.py declares it as an output: of
            # rule export_scenario_status, and Snakemake fails that rule loudly. On a RE-RUN
            # the previous generation's file satisfies the declaration, the rule succeeds,
            # its mtime never advances so the E&W plot rule is classified up to date and
            # skipped, and the report ships the PREVIOUS generation's validation figure.
            # Nothing deletes that file and the one staleness guard
            # (bundle/_emit.py::_assert_report_not_older_than_read_model) tests the opposite
            # mtime direction, so the re-run case is the silent one. That is what this
            # ordering protects against; the fresh case would report itself.
            check_provenance_completeness(analysis),  # ADR-15: per-stage version coverage
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
                # The writer is `dataclasses.asdict` (total over the dataclass); this
                # reader was a hand-listed five-field subset, so `applicable`,
                # `instrument` and `detection_floor` deserialized to their defaults
                # (True / None / None). That made `_status_of`'s N/A branch and both
                # pass-qualified branches unreachable through the persisted path --
                # measured on the delivered generation, where three `applicable=False`
                # checks on each clean arm rendered as unqualified PASS and the whole
                # figure's status-class set was `['pass']`. `.get` with the dataclass
                # default keeps a pre-field validation_report.json loading unchanged.
                applicable=c.get("applicable", True),
                instrument=c.get("instrument"),
                detection_floor=c.get("detection_floor"),
            )
            for c in payload.get("checks", [])
        ]
    )
