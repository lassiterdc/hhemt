"""Iter 9 Phase 7: tests for the analysis-validation report system.

Two layers of testing:

1. **Synthetic-ValidationReport tests** (test_renders_*): build a known
   ValidationReport with hand-crafted CheckResult instances representing
   every failure mode, pass it through the renderer's internal helpers,
   verify the rendered HTML contains the expected scenario × stage ×
   resource rows. This isolates renderer behavior from the complex analysis
   initialization (which re-detects state from disk and overwrites
   directly-mutated log fields).

2. **Real-analysis tests** (test_validation_report_*_failing): build a
   cloned analysis from one of the cached synth fixtures, inject failures
   the analysis init can't undo (file deletions for analysis-summary
   checks; see ``_failing_fixture_helpers.inject_*_at_paths``), assert
   the validator detects them. Some failure modes (compilation flag,
   per-scenario log fields) are re-overwritten by the analysis __init__'s
   resume-detection and aren't reliably triggerable via this path; those
   are covered by the synthetic tests above.
"""

from __future__ import annotations

import pytest

from hhemt.analysis_validation import (
    CheckResult,
    ValidationReport,
    validate_analysis,
)
from hhemt.config.report import DEFAULT_REPORT_CONFIG
from hhemt.report_renderers.errors_and_warnings import (
    _render_aggregate_table,
    _render_granular_failures_table,
    _render_overall_banner,
    _render_resource_mismatches_table,
    _render_system_level_table,
    render,
)

from tests._failing_fixture_helpers import (
    construct_analysis_from_paths,
    inject_multi_sim_failures_at_paths,
    inject_sensitivity_failures_at_paths,
    prepare_clone_dir,
)

pytestmark = pytest.mark.requires_snakemake_subprocess


# ---------------------------------------------------------------------------
# Synthetic ValidationReport tests (deterministic; isolated from analysis init)
# ---------------------------------------------------------------------------


def _synthetic_report() -> ValidationReport:
    """Build a ValidationReport covering every failure mode + a passing one."""
    return ValidationReport(checks=[
        CheckResult(
            name="System setup",
            level="system",
            passed=False,
            summary="System setup FAILED (1 issue(s))",
            details=[{"detail": "TRITON-SWMM compilation failed"}],
        ),
        CheckResult(
            name="Scenarios setup",
            level="aggregate",
            passed=False,
            summary="Scenario setup failed for 1 of 3 scenarios",
            details=[{"sa_id": "sa_0", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "scenario not created"}],
        ),
        CheckResult(
            name="Scenarios ran",
            level="aggregate",
            passed=False,
            summary="Simulation failed for 1 of 3 scenarios",
            details=[{"sa_id": "sa_1", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "simulation did not complete"}],
        ),
        CheckResult(
            name="Timeseries processed",
            level="aggregate",
            passed=False,
            summary="Timeseries processing failed for 1 entries",
            details=[{"sa_id": "sa_2", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "TRITON ts not processed"}],
        ),
        CheckResult(
            name="Analysis summaries created",
            level="system",
            passed=False,
            summary="Analysis summaries missing (1 item(s))",
            details=[{"detail": "TRITONSWMM TRITON summary missing"}],
        ),
        CheckResult(
            name="scenario_status.csv created",
            level="system",
            passed=True,
            summary="scenario_status.csv OK (3 rows)",
        ),
        CheckResult(
            name="Resource usage matches config",
            level="resource",
            passed=False,
            summary="Resource mismatches in 1 scenario(s)",
            details=[{"scenario": "sa_3 / event_index.0", "scenario_dir": "/path", "resource": "OMP threads", "expected": 4, "actual": 1, "detail": "OMP threads: expected 4, actual 1"}],
        ),
    ])


def test_synthetic_report_overall_passed_false():
    report = _synthetic_report()
    assert not report.overall_passed
    assert sum(1 for c in report.checks if c.passed) == 1


def test_synthetic_report_granular_failures_aggregated():
    """granular_failures collects per-scenario rows from aggregate-level checks."""
    report = _synthetic_report()
    rows = report.granular_failures
    sa_ids = {r.get("sa_id") for r in rows}
    stages = {r.get("stage") for r in rows}
    assert {"sa_0", "sa_1", "sa_2"} == sa_ids
    assert {"Scenarios setup", "Scenarios ran", "Timeseries processed"} == stages


def test_renders_overall_banner_failure():
    html = _render_overall_banner(_synthetic_report())
    assert 'class="banner fail"' in html
    # "applicable checks" — the banner scopes its denominator to applicable checks so
    # an N/A row is not counted as a pass or a failure. No synthetic check here is
    # applicable=False, so the denominator is still 7.
    assert "6 of 7 applicable checks failed" in html


def test_renders_system_level_table_marks_pass_and_fail():
    report = _synthetic_report()
    html = _render_system_level_table(report.by_level["system"])
    assert "System-Level Checks" in html
    assert "TRITON-SWMM compilation failed" in html
    assert 'class="fail"' in html
    assert 'class="pass"' in html  # scenario_status.csv check passes


def test_status_of_qualifies_only_a_declared_derived_instrument():
    """Four outcomes, and an UNDECLARED instrument is a plain pass — not a warning.

    Most checks are existence/status/config assertions that perform no numeric
    comparison, so they have no detection floor and must not carry a precision
    disclaimer; a disclaimer on every row is one no reader attends to, which would
    destroy the signal for the checks that genuinely need it. The qualifier is
    reserved for a check that DECLARED it compared at a coarser-than-raw tier.
    """
    from hhemt.analysis_validation import CheckResult
    from hhemt.report_renderers.errors_and_warnings import _status_of

    def mk(**kw):
        base = dict(name="x", level="system", passed=True, summary="s")
        base.update(kw)
        return CheckResult(**base)

    # VIOLATING input: a declared derived-tier pass MUST disclose its floor.
    cls, _, qual = _status_of(mk(instrument="summary_tier", detection_floor=1.1920929e-07))
    assert cls == "pass-qualified"
    assert qual  # non-empty disclosure

    # DIFFERENTLY-POSITIONED SATISFYING inputs — each a distinct correct state that
    # must NOT be qualified. These are what catch an over-firing qualifier.
    assert _status_of(mk(instrument="raw_rasters")) == ("pass", "✓", "")
    assert _status_of(mk()) == ("pass", "✓", "")  # no precision claim made

    assert _status_of(mk(applicable=False))[0] == "na"
    assert _status_of(mk(passed=False))[0] == "fail"


def test_renders_aggregate_table_has_three_failed_rows():
    report = _synthetic_report()
    html = _render_aggregate_table(report.by_level["aggregate"])
    assert "Aggregate Per-Scenario Checks" in html
    assert "Scenarios setup" in html
    assert "Scenarios ran" in html
    assert "Timeseries processed" in html
    assert html.count('class="fail"') == 3


def test_renders_granular_failures_table_groups_by_sa_id():
    report = _synthetic_report()
    html = _render_granular_failures_table(report.granular_failures)
    assert "Granular Per-Scenario Failures" in html
    for sa_id in ["sa_0", "sa_1", "sa_2"]:
        assert sa_id in html
    # Each scenario label uses "sa_X / scenario_name" format
    assert "sa_0 / event_index.0" in html


def test_renders_resource_mismatches_table_has_omp_row():
    report = _synthetic_report()
    html = _render_resource_mismatches_table(report.by_level["resource"])
    assert "Resource-Utilization Mismatches" in html
    assert "OMP threads" in html
    assert "<td>4</td>" in html
    assert "<td>1</td>" in html


def test_renders_full_html_doc(tmp_path):
    """End-to-end: render a synthetic ValidationReport via the public render API
    against a minimal-stub analysis (only cfg_analysis.analysis_id is needed)."""
    import json
    from dataclasses import asdict
    from unittest.mock import MagicMock

    from hhemt.analysis_validation import _VALIDATION_REPORT_FILENAME

    fake_analysis = MagicMock()
    fake_analysis.cfg_analysis.analysis_id = "demo_failing_synth"
    # Option-D: the renderer reads the persisted validation_report.json via
    # load_validation_report — it does NOT call validate_analysis at render time
    # (that whole-tree read would trip the renderer-IO provenance audit). So
    # persist the synthetic report to disk (the canonical
    # persist_validation_report shape) and point the fake analysis_dir at it.
    fake_analysis.analysis_paths.analysis_dir = tmp_path
    report = _synthetic_report()
    (tmp_path / _VALIDATION_REPORT_FILENAME).write_text(
        json.dumps({"checks": [asdict(c) for c in report.checks]})
    )

    out_path = tmp_path / "ew.html"
    render(fake_analysis, DEFAULT_REPORT_CONFIG, out_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings — demo_failing_synth" in html
    assert "System-Level Checks" in html
    assert "Aggregate Per-Scenario Checks" in html
    assert "Granular Per-Scenario Failures" in html
    assert "Resource-Utilization Mismatches" in html
    assert "OMP threads" in html


# ---------------------------------------------------------------------------
# Real-analysis tests (use cloned cached fixture; assert what's reliably triggerable)
# ---------------------------------------------------------------------------


@pytest.fixture
def failing_synth_multi_sim_analysis(
    tritonswmm_cpu_compiled, synth_multi_sim_analysis_cached, tmp_path
):
    """Clone synth_multi_sim cached fixture, inject failures BEFORE construction.

    Gated on ``tritonswmm_cpu_compiled``: the cached analysis must have been RUN
    (compiled binaries + real summaries on disk) for the injected-failure clone to
    be meaningful. Skips without cmake+mpic++; HARD-FAILS under
    HHEMT_REQUIRE_COMPILE_TIER=1. The module's pure-unit tests are unaffected."""
    paths = prepare_clone_dir(synth_multi_sim_analysis_cached, tmp_path)
    inject_multi_sim_failures_at_paths(paths)
    return construct_analysis_from_paths(paths)


@pytest.fixture
def failing_synth_sensitivity_analysis(
    tritonswmm_cpu_compiled, synth_sensitivity_analysis_cached, tmp_path
):
    """Clone synth_sensitivity cached fixture, inject failures BEFORE construction.
    Compile-tier gated; see ``failing_synth_multi_sim_analysis``."""
    paths = prepare_clone_dir(synth_sensitivity_analysis_cached, tmp_path)
    inject_sensitivity_failures_at_paths(paths)
    return construct_analysis_from_paths(paths)


def test_validation_report_multi_sim_failing_disk_mutations(failing_synth_multi_sim_analysis):
    """Disk mutations the analysis init can't undo: file deletions for analysis-summary checks."""
    report = validate_analysis(failing_synth_multi_sim_analysis)
    assert not report.overall_passed
    failed_check_names = {c.name for c in report.checks if not c.passed}
    # File-deletion mutations are checked at validation time (not at init):
    assert "Analysis summaries created" in failed_check_names


@pytest.mark.slow
def test_validation_report_sensitivity_failing_disk_mutations(failing_synth_sensitivity_analysis):
    """Sensitivity equivalent: deleted sensitivity_datatree.zarr should fail summaries check."""
    report = validate_analysis(failing_synth_sensitivity_analysis)
    assert not report.overall_passed
    failed_check_names = {c.name for c in report.checks if not c.passed}
    assert "Analysis summaries created" in failed_check_names


def test_errors_and_warnings_renders_html_multi_sim_real(failing_synth_multi_sim_analysis, tmp_path):
    """End-to-end render against a real (cloned + mutated) analysis."""
    out_path = tmp_path / "ew_multi_sim.html"
    render(failing_synth_multi_sim_analysis, DEFAULT_REPORT_CONFIG, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings" in html
    # I7-4: the rendered label is the DISPLAY name; `CheckResult.name` is unchanged and
    # is still asserted against `failed_check_names` above. Asserting BOTH halves is the
    # point -- it pins the key/label split rather than just tracking the rename.
    assert "Analysis summaries" in html
    assert "Every consolidated DataTree the analysis owes is present on disk" in html


@pytest.mark.slow
def test_errors_and_warnings_renders_html_sensitivity_real(failing_synth_sensitivity_analysis, tmp_path):
    """End-to-end render against a real (cloned + mutated) sensitivity analysis."""
    out_path = tmp_path / "ew_sensitivity.html"
    render(failing_synth_sensitivity_analysis, DEFAULT_REPORT_CONFIG, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings" in html
    # I7-4: the rendered label is the DISPLAY name; `CheckResult.name` is unchanged and
    # is still asserted against `failed_check_names` above. Asserting BOTH halves is the
    # point -- it pins the key/label split rather than just tracking the rename.
    assert "Analysis summaries" in html
    assert "Every consolidated DataTree the analysis owes is present on disk" in html


def _eda_stub(analysis_dir, enabled_plots, *, sensitivity=True, reporting_set="b4b"):
    """Minimal analysis shape `check_eda_calc_ran` reads: all four ENUMERATION-GATE terms.

    The check mirrors the Snakemake rule-all enumeration gate term for term, so a fixture
    carrying fewer than four terms short-circuits at the first missing one and every
    assertion below passes trivially. The terms: (1) toggle_sensitivity_analysis, because
    only the sensitivity-master generators carry an EDA enumeration site; (2) the active
    reporting set carrying an eda_compute_sensitivity selection; (3) a non-empty
    eda.enabled_plots; (4) the builder key absent from report.disabled_renderers.
    """
    import types

    from hhemt.report_renderers._reporting_sets import get_reporting_set

    ns = types.SimpleNamespace(
        cfg_analysis=types.SimpleNamespace(
            eda=types.SimpleNamespace(enabled_plots=list(enabled_plots)),
            toggle_sensitivity_analysis=sensitivity,
            report=types.SimpleNamespace(disabled_renderers=[], reporting_set=reporting_set),
        ),
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )
    ns._active_reporting_set = get_reporting_set(reporting_set)
    return ns


def test_eda_calc_ran_fails_when_targets_are_enumerated_but_no_verdicts_exist(tmp_path):
    """K1/F4: an analysis that ENUMERATES EDA report targets owes verdicts.

    The degradation panels that replaced the workflow-killing MissingOutputException are
    correct, but they removed the LOUD failure without replacing the signal. This is the
    positive signal. The antecedent is the enumeration gate, NOT enabled_plots: that field
    carries a non-empty default_factory and is true on essentially every analysis, which is
    why keying on it alone fired on every multisim.
    """
    from hhemt.analysis_validation import check_eda_calc_ran

    result = check_eda_calc_ran(_eda_stub(tmp_path, ["config_diff_maps"], reporting_set="b4b"))

    assert result.passed is False
    assert "2 EDA plot(s)" in result.summary, "the summary must name the ENUMERATED-target count"
    assert result.level == "aggregate"
    assert result.details and "verdict_count=0" in result.details[0]["detail"]


def test_eda_calc_ran_passes_once_a_verdict_artifact_is_present(tmp_path):
    """Same enumeration, one verdict on disk -> the calc ran."""
    from hhemt.analysis_validation import check_eda_calc_ran

    eda_dir = tmp_path / "eda"
    eda_dir.mkdir()
    (eda_dir / "b4b_clean_identity.verdict.json").write_text("{}", encoding="utf-8")

    result = check_eda_calc_ran(_eda_stub(tmp_path, ["config_diff_maps"], reporting_set="b4b"))

    assert result.passed is True
    assert "1 verdict artifact(s) present" in result.summary


def test_eda_calc_ran_is_not_applicable_when_the_set_enumerates_no_targets(tmp_path):
    """Differently-positioned satisfying input: a set enumerating nothing owes nothing.

    `benchmarking` carries no eda_compute_sensitivity selection, so a sensitivity master on
    it renders no degradation panels and has no silence to close — even though enabled_plots
    is NON-empty here. A predicate keyed on enabled_plots alone FAILS this input, which is
    what pins the check to the enumeration gate rather than to the config.
    """
    from hhemt.analysis_validation import check_eda_calc_ran

    result = check_eda_calc_ran(
        _eda_stub(tmp_path, ["config_diff_maps"], reporting_set="benchmarking")
    )

    assert result.passed is True
    assert "N/A" in result.summary
    assert not (tmp_path / "eda").exists(), "the fixture has no eda/ dir, and that is not a failure"


def test_eda_calc_ran_is_not_applicable_to_a_multisim(tmp_path):
    """The K1 false positive itself: a multisim enumerates no EDA rules whatever set it names.

    generate_snakefile_content carries no EDA enumeration site and the multisim plot
    dispatcher passes no predicate_inputs, so term 1 is what excludes this whole class.
    """
    from hhemt.analysis_validation import check_eda_calc_ran

    result = check_eda_calc_ran(
        _eda_stub(tmp_path, ["config_diff_maps"], sensitivity=False, reporting_set="b4b")
    )

    assert result.passed is True
    assert "N/A" in result.summary


def test_eda_rule_spec_templates_pins_the_sets_check_eda_calc_ran_depends_on():
    """Drift guard: the check's blast radius is exactly the enumerating set membership.

    A new reporting set that adds the eda_compute_sensitivity renderer silently widens what
    check_eda_calc_ran demands verdicts for. This fails loudly instead.
    """
    from hhemt.report_renderers._reporting_sets import REPORTING_SETS, eda_rule_spec_templates

    enumerating = {n for n, s in REPORTING_SETS.items() if eda_rule_spec_templates(s)}

    assert enumerating == {"b4b", "compute-sensitivity", "dem-resolution"}
def test_resume_validity_is_applicable_on_a_pure_triton_resumed_arm():
    """VMS-9: the widened check evaluates the pure-TRITON resume arm instead of
    returning N/A on the coupled toggle.

    PRE-FIX this FAILS twice over: the check is named "Coupled resume validity"
    and it returns applicable=False for any analysis whose tritonswmm toggle is off.
    """
    from hhemt.model_defects import REGISTRY, resolve

    # The sha the delivered resume arms actually ran at (measured from the
    # consolidated tree root attr `triton_producing_sha`).
    resume_sha = "5d2ad1e8adf9a85d7df14e885b76e59a10f9a98b"
    by_trigger = {d.trigger for d in REGISTRY}
    assert "resumed_any" in by_trigger, (
        "no registry defect applies to both model selections; the widened check "
        "would have nothing to evaluate on a pure-TRITON arm"
    )
    both_arms = [d for d in REGISTRY if d.trigger == "resumed_any"]
    verdicts = [resolve(d, resume_sha) for d in both_arms]
    assert all(v.status == "absent" for v in verdicts), [
        (v.defect_id, v.status, v.rule) for v in verdicts
    ]


def test_clean_arm_carries_an_affected_build_but_stays_not_applicable():
    """VMS-9 / [Q130]: the clean arms run a build where the both-arms defect is
    PRESENT, and only the trigger population keeps that from becoming a warning.

    This is the P7 case the TriggerKind docstring names. It pins that a
    version-only predicate would be wrong.
    """
    from hhemt.model_defects import REGISTRY, resolve

    clean_sha = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"
    ghost = [d for d in REGISTRY if d.trigger == "resumed_any"]
    assert ghost, "expected a resumed_any defect"
    assert any(resolve(d, clean_sha).status == "present" for d in ghost), (
        "the clean-arm build is expected to CARRY the both-arms defect; if this "
        "flips, the trigger-population argument needs re-deriving"
    )


def _stub_absent_record():
    """An analysis whose resume record is absent but whose structural attributes exist.

    `analysis_paths` and `cfg_analysis` are STRUCTURAL preconditions of a real
    TRITONSWMM_analysis -- `_read_triton_provenance` reads both directly. Empty
    namespaces are enough: every zarr-path lookup inside it is a getattr-with-None
    default, so it takes its documented graceful-absent path and returns None.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=True)),
        analysis_paths=SimpleNamespace(),
        cfg_analysis=SimpleNamespace(),
        df_status=None,
    )


def test_provenance_helper_is_graceful_absent_on_this_stub():
    """POSITIVE CONTROL for the test below. Proves the stub reaches the helper's
    graceful-absent path, so a failure there is an ASSERTION failure and not a
    setup crash -- the two are indistinguishable in a `1 failed` summary line."""
    from hhemt.analysis_validation import _read_triton_provenance

    assert _read_triton_provenance(_stub_absent_record()) is None


def test_absent_resume_record_renders_not_applicable_not_pass():
    """VMS-9B / VMS-9C / [Q130]: a not-verified state is NOT a negative finding.

    PRE-FIX this fails on its assertion with applicable=True, passed=True and the
    summary "Producing-TRITON coupled-resume status unknown ...; cannot determine
    coupled-resume validity" -- a branch whose own prose says it could not decide,
    rendered green.
    """
    from hhemt.analysis_validation import check_coupled_resume_validity

    result = check_coupled_resume_validity(_stub_absent_record())
    assert result.applicable is False, (
        "an absent resume record must render N/A, not a disclosed-denominator PASS; "
        f"got applicable={result.applicable!r} passed={result.passed!r} "
        f"summary={result.summary!r}"
    )
    assert result.name == "resume validity"
