"""The render records the report settings that render used, and the reconciler and the resets read that record.

`stamp_report_stage` writes `report_settings`, `report_settings_sha256` and
`report_settings_source` into `report_manifest.json` after a render succeeds. The bundle writer,
the reconciler and the tests below read that one record. None of these tests compiles or runs a
solver; the renders use the stand-ins of `tests/test_report_html_mode_wiring.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.analysis import ForceRerunSpec
from hhemt.config.report import DEFAULT_REPORT_CONFIG
from hhemt.config.report import report_config as ReportConfigModel
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from tests.test_report_html_mode_wiring import _stub_whole_render

_REAL_STAMP = provenance.stamp_report_stage  # the render stand-in replaces the function with a no-op
_RENDER = {"analysis": TRITONSWMM_analysis.render_report, "sensitivity": TRITONSWMM_sensitivity_analysis.render_report}
KEYS = ("report_settings", "report_settings_sha256", "report_settings_source")


def _cfg(mode):
    data = DEFAULT_REPORT_CONFIG.model_dump()
    data["interactive"] = {
        **data["interactive"],
        "report_html_mode": mode,
        "plotly_js_mode": "inline",
        "tabulator_js_mode": "inline",
    }
    return ReportConfigModel.model_validate(data)


_REAL_RUN = __import__("subprocess").run


def _let_git_run(monkeypatch):
    """The render stand-in answers every subprocess call; the stamp's git reads need the real one."""
    import subprocess

    stub = subprocess.run
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, *a, **kw: stub(cmd, *a, **kw) if "--report" in cmd else _REAL_RUN(cmd, *a, **kw)
    )


def _manifest(directory):
    return json.loads((Path(directory) / "report_manifest.json").read_text())


@pytest.mark.parametrize("kind", ["analysis", "sensitivity"])
def test_render_with_a_run_entry_config_records_it(monkeypatch, tmp_path, kind):
    stand_in = _stub_whole_render(monkeypatch, kind, tmp_path, "auto")
    monkeypatch.setattr(provenance, "stamp_report_stage", _REAL_STAMP)
    _let_git_run(monkeypatch)
    given = _cfg("zip")
    stand_in._run_entry_report_cfg = given
    _RENDER[kind](stand_in, format="html")
    record = _manifest(tmp_path)
    assert record["report_settings"] == given.model_dump(mode="json")
    assert record["report_settings_sha256"] == provenance.report_settings_sha256(given)
    assert record["report_settings_source"] == "run_entry"


@pytest.mark.parametrize("kind", ["analysis", "sensitivity"])
def test_render_without_a_run_entry_config_records_the_analysis_configs(monkeypatch, tmp_path, kind):
    stand_in = _stub_whole_render(monkeypatch, kind, tmp_path, "auto")
    monkeypatch.setattr(provenance, "stamp_report_stage", _REAL_STAMP)
    _let_git_run(monkeypatch)
    stand_in._run_entry_report_cfg = None
    _RENDER[kind](stand_in, format="html")
    record = _manifest(tmp_path)
    assert record["report_settings"] == stand_in.cfg_analysis.report.model_dump(mode="json")
    assert record["report_settings_source"] == "analysis_config"


@pytest.mark.parametrize("kind", ["analysis", "sensitivity"])
def test_a_failed_render_writes_no_record(monkeypatch, tmp_path, kind):
    import subprocess

    from hhemt.exceptions import WorkflowError

    stand_in = _stub_whole_render(monkeypatch, kind, tmp_path, "auto")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(WorkflowError):
        _RENDER[kind](stand_in, format="html")
    assert not (tmp_path / "report_manifest.json").exists()


def test_a_record_write_that_fails_logs_a_warning_and_removes_the_settings_keys(monkeypatch, tmp_path, caplog):
    """Input class: a record already present, then a write that cannot complete."""
    old = _cfg("zip")
    provenance.stamp_report_stage(tmp_path, tmp_path / "analysis_report.zip", old, "run_entry")
    assert all(k in _manifest(tmp_path) for k in KEYS)

    def _refuse(self, *_a, **_kw):
        raise OSError("manifest path unwritable")

    monkeypatch.setattr(Path, "replace", _refuse)
    with caplog.at_level(logging.WARNING, logger="hhemt.provenance"):
        provenance.stamp_report_stage(tmp_path, tmp_path / "analysis_report.html", _cfg("html"), "analysis_config")
    monkeypatch.undo()
    assert any(r.levelno == logging.WARNING and "report settings record" in r.getMessage() for r in caplog.records)
    assert not any(k in _manifest(tmp_path) for k in KEYS)
    assert provenance.read_report_settings_record(tmp_path) is None
    assert not list(tmp_path.glob("*.tmp"))


# --- The reconciler -----------------------------------------------------------------------------


@pytest.fixture
def recorded(synth_multi_sim_builder):
    """The shared builder with a record written for the analysis config's own settings."""
    analysis = synth_multi_sim_builder
    manifest = analysis.analysis_paths.analysis_dir / "report_manifest.json"
    before = manifest.read_bytes() if manifest.exists() else None
    provenance.stamp_report_stage(
        analysis.analysis_paths.analysis_dir, manifest.parent / "analysis_report.html", analysis.cfg_analysis.report
    )
    yield analysis
    if before is None:
        manifest.unlink(missing_ok=True)
    else:
        manifest.write_bytes(before)


def _different(analysis):
    held = analysis.cfg_analysis.report.interactive.report_html_mode
    return _cfg("zip" if held != "zip" else "html")


def test_reconciler_escalates_to_a_render_stage_force_rerun_on_a_mismatch(recorded, caplog):
    with caplog.at_level(logging.WARNING, logger="hhemt.analysis"):
        got = recorded._reconcile_report_settings_force(None, settings_cfg=_different(recorded))
    assert got == ForceRerunSpec(subject="all", stage="render")
    assert any("report settings mismatch" in r.getMessage() for r in caplog.records)


def test_control_reconciler_leaves_the_value_when_the_settings_are_equal(recorded):
    sentinel = ForceRerunSpec(subject="all", stage="process")
    assert recorded._reconcile_report_settings_force(sentinel, settings_cfg=recorded.cfg_analysis.report) is sentinel


def test_control_reconciler_leaves_the_value_with_no_record(synth_multi_sim_builder):
    analysis = synth_multi_sim_builder
    manifest = analysis.analysis_paths.analysis_dir / "report_manifest.json"
    if manifest.exists():
        pytest.skip("the shared builder holds a report manifest from another test")
    assert analysis._reconcile_report_settings_force(None, settings_cfg=_different(analysis)) is None


def test_control_reconciler_leaves_the_value_on_a_dry_run(recorded):
    assert recorded._reconcile_report_settings_force(None, settings_cfg=_different(recorded), dry_run=True) is None


def test_a_stronger_caller_floor_is_kept(recorded):
    stronger = ForceRerunSpec(subject="all", stage="simulate")
    assert recorded._reconcile_report_settings_force(stronger, settings_cfg=_different(recorded)) is stronger


def test_the_reconciler_is_called_at_the_three_force_rerun_sites():
    """The call sites: submit_workflow, the analysis reprocess and the sensitivity reprocess."""
    import inspect

    for fn in (
        TRITONSWMM_analysis.submit_workflow,
        TRITONSWMM_analysis.reprocess,
        TRITONSWMM_sensitivity_analysis.reprocess,
    ):
        assert "_reconcile_report_settings_force(" in inspect.getsource(fn), fn.__qualname__


# --- A re-process renders with the analysis config's settings --------------------------


def _hold_run_settings(*holders):
    given = _cfg("zip")
    for holder in holders:
        for name in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES:
            setattr(holder, name, given)


def test_reset_clears_the_four_attributes_on_the_analysis(synth_multi_sim_builder, monkeypatch):
    analysis = synth_multi_sim_builder
    for name in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES:
        monkeypatch.setattr(analysis, name, None, raising=False)
    _hold_run_settings(analysis)
    analysis._reset_run_report_settings()
    assert [getattr(analysis, n) for n in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES] == [None] * 4


def test_reset_clears_the_wrapper_and_the_experiment_of_a_sweep(synth_sensitivity_builder, monkeypatch):
    analysis = synth_sensitivity_builder
    wrapper = analysis.sensitivity
    for holder in (analysis, wrapper):
        for name in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES:
            monkeypatch.setattr(holder, name, None, raising=False)
    _hold_run_settings(analysis, wrapper)
    analysis._reset_run_report_settings()
    for holder in (analysis, wrapper):
        assert [getattr(holder, n) for n in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES] == [None] * 4


@pytest.mark.parametrize("sweep", [False, True])
def test_a_reprocess_snakefile_after_a_run_equals_a_fresh_one_and_carries_no_flag(
    sweep, synth_multi_sim_builder, synth_sensitivity_builder, monkeypatch
):
    """Input class: an object that still holds a run-entry config whose reporting set differs."""
    from hhemt.reprocess_snakefile_generator import generate_reprocess_snakefile

    analysis = synth_sensitivity_builder if sweep else synth_multi_sim_builder
    holders = (analysis, analysis.sensitivity) if sweep else (analysis,)
    for holder in holders:
        for name in TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES:
            monkeypatch.setattr(holder, name, None, raising=False)

    def _generate():
        if sweep:
            return analysis.sensitivity._workflow_builder.generate_reprocess_master_snakefile_content(
                which="both", start_with="render"
            )
        return generate_reprocess_snakefile(analysis._workflow_builder, start_with="render")

    fresh = _generate()
    data = _cfg("zip").model_dump()
    data["interactive"] = {**data["interactive"], "static_backend": "matplotlib"}
    _hold_run_settings(*holders)
    for holder in holders:
        holder._cfg_report = ReportConfigModel.model_validate(data)
    analysis._reset_run_report_settings()
    after = _generate()
    assert "--report-config" not in after
    assert after == fresh


def test_reconciler_returns_none_unchanged_when_the_settings_are_equal(recorded):
    """Input class: equal settings and no caller floor. A spurious mismatch would return a render spec."""
    assert recorded._reconcile_report_settings_force(None, settings_cfg=recorded.cfg_analysis.report) is None


def test_two_writers_use_distinct_temporary_names(synth_multi_sim_builder, monkeypatch):
    """Input class: two writes of one content-named file. Concurrent writers need distinct temporaries."""
    from tests.test_report_config_run_entry import _differing_mode, _run_entry_cfg

    analysis = synth_multi_sim_builder
    builder = analysis._workflow_builder
    monkeypatch.setattr(analysis, "_run_entry_report_cfg", _run_entry_cfg(_differing_mode(analysis)), raising=False)
    directory = analysis.analysis_paths.analysis_dir
    flag = builder._report_config_args()
    (target,) = [p for p in directory.glob("report_config_run_entry.*.yaml") if str(p) in flag]
    names = []
    original = Path.write_text

    def _record(self, data, *a, **kw):
        if self.name.startswith(target.name) and self.name.endswith(".tmp"):
            names.append(self.name)
        return original(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", _record)
    for _ in range(2):
        target.unlink()
        builder._report_config_args()
    monkeypatch.undo()
    assert len(names) == 2 and len(set(names)) == 2


def _assigned_reconciler_calls(fn):
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", "") == "_reconcile_report_settings_force"
        and any(isinstance(t, ast.Name) and t.id == "override_force_rerun" for t in n.targets)
    )


def test_each_force_rerun_site_uses_the_reconciler_result():
    """Input class: the three sites. A call whose result is dropped escalates nothing."""
    for fn in (
        TRITONSWMM_analysis.submit_workflow,
        TRITONSWMM_analysis.reprocess,
        TRITONSWMM_sensitivity_analysis.reprocess,
    ):
        assert _assigned_reconciler_calls(fn) == 1, fn.__qualname__


def test_both_reprocess_entries_reset_the_run_attributes():
    """Input class: the two re-process entries. Each must reset before any Snakefile is generated."""
    import inspect

    for fn in (TRITONSWMM_analysis.reprocess, TRITONSWMM_sensitivity_analysis.reprocess):
        assert "_reset_run_report_settings(" in inspect.getsource(fn), fn.__qualname__


def test_the_reset_names_are_the_attributes_run_sets():
    """Input class: `run()`'s body. The reset tuple must contain every report attribute `run()` assigns."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(TRITONSWMM_analysis.run)))
    assigned = {
        t.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
    }
    reportish = {a for a in assigned if "report" in a or "brand" in a}
    assert reportish and reportish <= set(TRITONSWMM_analysis._RUN_REPORT_ATTRIBUTES)
