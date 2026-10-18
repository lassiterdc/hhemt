"""The report renderers let an error from the post-render surgery reach their caller.

After ``snakemake --report`` writes the report, four production callers apply the
React-bundle surgery to it: ``TRITONSWMM_analysis.render_report``,
``TRITONSWMM_sensitivity_analysis.render_report``, ``Bundle.regenerate_report`` and
``render_combined_report_via_snakemake``. The surgery raises ``ProcessingError`` when the
bundle it edits has drifted from the shape it expects. If a caller discards that error,
the render reports success on a report that lacks the row-click delegate and the toggle
fallback. Under ``render_report_runner`` an exception that propagates is logged and fails
the rule instead.

Every step before the surgery call is replaced with a stub: no solver runs, no Snakemake
subprocess runs, and no analysis is constructed. Each method runs as an unbound function
against a stand-in object that carries only the attributes it reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hhemt.provenance as provenance
import hhemt.render_category_order as render_category_order
import hhemt.report_plot_ids as report_plot_ids
import hhemt.report_renderers._react_surgery as react_surgery
import hhemt.workflow as workflow
from hhemt.exceptions import ProcessingError

_FORMATS = ("html", "zip")
_HTML_MARKER = "surgery-stub-html-output"
_ZIP_MARKER = "surgery-stub-zip-output"


def _raise_drift(*_args, **_kwargs):
    raise ProcessingError("react-bundle surgery", reason="stub: anchor drifted")


def _install_raising_surgery(monkeypatch):
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", _raise_drift)
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery_to_zip", _raise_drift)


def _install_writing_surgery(monkeypatch):
    """The html surgery returns a marker; the zip surgery writes a marker to the path."""
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", lambda _text, **_kw: _HTML_MARKER)

    def _zip(path, **_kw):
        path.write_text(_ZIP_MARKER)

    monkeypatch.setattr(react_surgery, "apply_post_process_surgery_to_zip", _zip)


def _stub_label_and_stamp_helpers(monkeypatch):
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(provenance, "stamp_report_stage", lambda *_a, **_kw: None)
    monkeypatch.setattr(report_plot_ids, "member_labels_from_status", lambda _d: {})
    monkeypatch.setattr(report_plot_ids, "event_sidebar_labels_from_status", lambda _d: {})


def _stub_analysis_render_path(monkeypatch, module):
    """Stub everything ``render_report`` in ``module`` calls before the surgery."""
    import subprocess

    _stub_label_and_stamp_helpers(monkeypatch)
    monkeypatch.setattr(workflow, "_assert_snakefile_package_current", lambda _p: None)
    monkeypatch.setattr(workflow, "_brand_theme_css_map", lambda _t: {})
    monkeypatch.setattr(module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    monkeypatch.setattr(render_category_order, "resolve_render_path_category_order", lambda _a: ["Key Results"])
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="", stderr=""))


def _run_analysis(monkeypatch, tmp_path, fmt):
    import hhemt.analysis as analysis
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    _stub_analysis_render_path(monkeypatch, analysis)
    out = tmp_path / f"analysis_report.{fmt}"
    out.write_text("rendered")
    stand_in = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=SimpleNamespace(brand_theme=None, analysis_id="a"),
        _brand_theme=SimpleNamespace(upper_left_text="navbar"),
        _active_reporting_set=get_reporting_set("default"),
    )
    return analysis.TRITONSWMM_analysis.render_report(stand_in, format=fmt), out


def _run_sensitivity(monkeypatch, tmp_path, fmt):
    import hhemt.sensitivity_analysis as sensitivity_analysis
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    _stub_analysis_render_path(monkeypatch, sensitivity_analysis)
    out = tmp_path / f"analysis_report.{fmt}"
    out.write_text("rendered")
    paths = SimpleNamespace(analysis_dir=tmp_path)
    stand_in = SimpleNamespace(
        experiment=SimpleNamespace(analysis_paths=paths),
        analysis_paths=paths,
        cfg_analysis=SimpleNamespace(brand_theme=None, analysis_id="a"),
        _brand_theme=SimpleNamespace(upper_left_text="navbar"),
        _active_reporting_set=get_reporting_set("default"),
    )
    return sensitivity_analysis.TRITONSWMM_sensitivity_analysis.render_report(stand_in, format=fmt), out


def _run_bundle(monkeypatch, tmp_path, fmt):
    import hhemt.bundle as bundle
    import hhemt.bundle.snakefile_generator as snakefile_generator

    _stub_label_and_stamp_helpers(monkeypatch)
    monkeypatch.setattr(workflow, "_emit_report_artifacts", lambda *_a, **_kw: None)
    monkeypatch.setattr(workflow, "_brand_theme_css_map", lambda _t: {})
    monkeypatch.setattr(snakefile_generator, "write_regeneration_snakefile", lambda *_a, **_kw: None)
    monkeypatch.setattr(bundle, "run_subprocess_with_tee", lambda *_a, **_kw: SimpleNamespace(returncode=0))
    out = tmp_path / f"analysis_report.{fmt}"
    out.write_text("rendered")
    stand_in = SimpleNamespace(
        _root=tmp_path,
        _cfg_analysis=None,
        _read_static_backend=lambda: "plotly",
        _assert_render_inputs_present=lambda: None,
    )
    return bundle.Bundle.regenerate_report(stand_in, format=fmt), out


def _run_combined(monkeypatch, tmp_path, fmt):
    import hhemt.bundle.combined_snakefile_generator as combined

    _stub_label_and_stamp_helpers(monkeypatch)
    monkeypatch.setattr(combined, "stage_combined_report_artifacts", lambda _root: None)
    monkeypatch.setattr(combined, "write_combined_snakefile", lambda root: (root / "Snakefile", ()))
    monkeypatch.setattr(combined, "_distinct_child_categories", lambda _root: [])
    monkeypatch.setattr(combined, "run_subprocess_with_tee", lambda *_a, **_kw: SimpleNamespace(returncode=0))
    root = tmp_path.resolve()
    out = root / f"analysis_report.{fmt}"
    out.write_text("rendered")
    return combined.render_combined_report_via_snakemake(root, formats=(fmt,)), out


_CALLERS = {
    "analysis": _run_analysis,
    "sensitivity": _run_sensitivity,
    "bundle": _run_bundle,
    "combined": _run_combined,
}


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_surgery_error_reaches_the_caller(monkeypatch, tmp_path, caller, fmt):
    """An exception the surgery raises propagates out of the render method.

    Property: when the surgery function for the requested format raises
    ``ProcessingError``, the caller raises that same ``ProcessingError``.

    Class: every production caller of the surgery, in each output format; the surgery
    error stands for any exception the surgery raises.

    Why it is required: the surgery raises on a drifted bundle because a warning would be
    lost in the rule log. A caller that discards the error ships a report with no
    row-click delegate and no toggle fallback, and the render rule exits 0.

    What kills it: wrapping the surgery call in ``try`` / ``except Exception: pass`` (or
    any handler that does not re-raise) makes the caller return normally, and the
    ``pytest.raises`` block fails.

    A second correct implementation under which it still passes: catching the error,
    logging it, and re-raising it unchanged.
    """
    _install_raising_surgery(monkeypatch)

    with pytest.raises(ProcessingError, match="stub: anchor drifted"):
        _CALLERS[caller](monkeypatch, tmp_path, fmt)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_surgery_output_is_written_when_it_succeeds(monkeypatch, tmp_path, caller, fmt):
    """With a surgery that succeeds, the caller writes its output and returns normally.

    Property: the html surgery's returned text becomes the report file, the zip surgery's
    in-place write survives, and the caller returns its usual value (the report path, or
    ``None`` for the combined renderer).

    Class: every production caller of the surgery, in each output format, on the path
    where the surgery does not raise.

    Why it is required: it is the control for the propagation test. It shows that the
    stubs reach the surgery call and that removing the error handler did not break the
    normal path.

    What kills it: a caller that skips the surgery call, writes the pre-surgery text back
    over the surgery's output, or raises on the normal path.

    A second correct implementation under which it still passes: the html branch writing
    the surgery's result through a temporary file and an atomic rename.
    """
    _install_writing_surgery(monkeypatch)

    returned, out = _CALLERS[caller](monkeypatch, tmp_path, fmt)

    assert out.read_text() == (_HTML_MARKER if fmt == "html" else _ZIP_MARKER)
    assert returned == (None if caller == "combined" else out)
