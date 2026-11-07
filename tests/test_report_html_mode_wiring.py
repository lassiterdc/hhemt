"""The `report_html_mode` setting reaches every render path that is given no explicit format.

Before this wiring the setting was read nowhere outside its own validator, so
`report_html_mode: auto` promised a size-based choice that no render made. These tests pin
the pieces that carry it: the resolver, the facade parameter name, the render targets a
generated Snakefile declares, and the file the render runner writes beside the report.
None of them compiles or runs a solver.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.report import (
    AUTO_HTML_MAX_BYTES,
    REPORT_FORMAT_MARKER,
    InteractiveBackendConfig,
    report_figure_bytes,
    resolve_report_format,
)
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from hhemt.workflow import _render_target_plan


def _write_figures(root, sizes):
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(sizes):
        (plots / f"figure_{i}.html").write_bytes(b"x" * size)
        # A sidecar of any size must not count toward the figure total.
        (plots / f"figure_{i}.manifest.json").write_bytes(b"{}" * 1000)


def test_default_mode_is_auto():
    assert InteractiveBackendConfig().report_html_mode == "auto"


def test_explicit_modes_ignore_the_figure_size():
    assert resolve_report_format("html", 10 * AUTO_HTML_MAX_BYTES) == "html"
    assert resolve_report_format("zip", 0) == "zip"


def test_auto_follows_the_total_of_the_figure_files(tmp_path):
    """A figure set split over two files crosses the threshold by one byte."""
    below = tmp_path / "below"
    above = tmp_path / "above"
    _write_figures(below, [AUTO_HTML_MAX_BYTES - 1 - 7, 7])
    _write_figures(above, [AUTO_HTML_MAX_BYTES - 7, 7])
    assert report_figure_bytes(below) == AUTO_HTML_MAX_BYTES - 1
    assert report_figure_bytes(above) == AUTO_HTML_MAX_BYTES
    assert resolve_report_format("auto", report_figure_bytes(below)) == "html"
    assert resolve_report_format("auto", report_figure_bytes(above)) == "zip"


def test_control_the_sidecars_do_not_move_the_total(tmp_path):
    """Control: growing only the manifest sidecars leaves the figure total unchanged."""
    _write_figures(tmp_path, [100])
    before = report_figure_bytes(tmp_path)
    (tmp_path / "plots" / "figure_0.manifest.json").write_bytes(b"{}" * 10**6)
    assert report_figure_bytes(tmp_path) == before == 100


def test_a_missing_plots_directory_counts_zero(tmp_path):
    assert report_figure_bytes(tmp_path) == 0


def test_unknown_mode_is_refused():
    with pytest.raises(Exception, match="report_html_mode"):
        resolve_report_format("png", 0)


@pytest.mark.parametrize(
    "method",
    [
        TRITONSWMM_analysis.run,
        TRITONSWMM_analysis.submit_workflow,
        TRITONSWMM_sensitivity_analysis.submit_workflow,
    ],
)
def test_facades_take_override_report_formats_and_no_report_formats(method):
    params = inspect.signature(method).parameters
    assert "override_report_formats" in params
    assert "report_formats" not in params
    assert params["override_report_formats"].default is None


def test_sensitivity_run_path_takes_the_override_name():
    """The sensitivity run entry point is reached through `submit_workflow` of the
    sensitivity analysis; its other public methods carry no report-format parameter."""
    for name, member in inspect.getmembers(TRITONSWMM_sensitivity_analysis, inspect.isfunction):
        assert "report_formats" not in inspect.signature(member).parameters, name


def test_render_methods_default_to_the_setting():
    for method in (TRITONSWMM_analysis.render_report, TRITONSWMM_sensitivity_analysis.render_report):
        assert inspect.signature(method).parameters["format"].default is None


def test_render_targets_for_each_mode():
    auto = _render_target_plan(None, "auto")
    assert auto.rule_all_targets == (f'"{REPORT_FORMAT_MARKER}"',)
    assert _render_target_plan(None, "html").rule_all_targets == ('"analysis_report.html"',)
    assert _render_target_plan(None, "zip").rule_all_targets == ('"analysis_report.zip"',)
    override = _render_target_plan(["zip", "html"], "auto")
    assert override.rule_all_targets == ('"analysis_report.zip"', '"analysis_report.html"')


def test_neutral_render_rule_has_one_output_and_no_format_flag():
    auto = _render_target_plan(None, "auto")
    head = auto.rule_head("logs")
    assert f'"{REPORT_FORMAT_MARKER}"' in head
    assert "wildcard_constraints" not in head and "{format}" not in head
    assert auto.format_shell_line == ""


def test_control_an_explicit_render_rule_keeps_the_format_wildcard():
    explicit = _render_target_plan(None, "zip")
    head = explicit.rule_head("logs")
    assert '"analysis_report.{format}"' in head and 'format="zip|html"' in head
    assert "--format {wildcards.format}" in explicit.format_shell_line


def _run_runner(monkeypatch, tmp_path, argv, *, report_name, writes_marker=False):
    """Run `render_report_runner.main` with fake analysis classes that write `report_name`.

    `writes_marker` makes the stand-in render write the format marker the way the real
    `render_report` does when the setting chose the format, so a test can see whether the
    runner leaves that file alone.
    """
    import hhemt.analysis as analysis_module
    import hhemt.system as system_module
    from hhemt import render_report_runner

    calls = {}

    class FakeAnalysis:
        def __init__(self, *args, **kwargs):
            self.cfg_analysis = types.SimpleNamespace(toggle_sensitivity_analysis=False)

        def render_report(self, format=None, reprocess=False):
            calls["format"] = format
            out = tmp_path / report_name
            out.write_bytes(b"report")
            if writes_marker:
                (tmp_path / REPORT_FORMAT_MARKER).write_text(out.suffix.lstrip(".") + "\n")
            return out

    monkeypatch.setattr(analysis_module, "TRITONSWMM_analysis", FakeAnalysis)
    monkeypatch.setattr(system_module, "TRITONSWMM_system", lambda *a, **k: object())
    monkeypatch.setattr(sys, "argv", ["render_report_runner", *argv])
    return render_report_runner.main(), calls


_OTHER_FORMAT = {"html": "zip", "zip": "html"}


@pytest.mark.parametrize("chosen", ["html", "zip"])
def test_runner_without_format_removes_the_other_formats_report(monkeypatch, tmp_path, chosen):
    """The runner that lets `render_report` choose the format leaves one report file in the directory.

    Input class: every directory that holds the report of the format the setting did not
    choose, whichever format that is. The stand-in `render_report` writes the chosen
    report and the marker, as the real method does, so the marker also shows that the
    runner deletes the stale report file and nothing else.
    """
    stale = tmp_path / f"analysis_report.{_OTHER_FORMAT[chosen]}"
    stale.write_bytes(b"stale")
    code, calls = _run_runner(
        monkeypatch,
        tmp_path,
        ["--system-config", "s.yaml", "--analysis-config", "a.yaml"],
        report_name=f"analysis_report.{chosen}",
        writes_marker=True,
    )
    assert code == 0 and calls["format"] is None
    assert not stale.exists()
    assert (tmp_path / f"analysis_report.{chosen}").read_bytes() == b"report"
    assert (tmp_path / REPORT_FORMAT_MARKER).read_text().strip() == chosen


@pytest.mark.parametrize("chosen", ["html", "zip"])
def test_control_runner_with_an_explicit_format_keeps_the_other_formats_report(monkeypatch, tmp_path, chosen):
    """An explicit `--format` keeps a report file of the other format: both formats can coexist."""
    kept = tmp_path / f"analysis_report.{_OTHER_FORMAT[chosen]}"
    kept.write_bytes(b"kept")
    code, calls = _run_runner(
        monkeypatch,
        tmp_path,
        ["--system-config", "s.yaml", "--analysis-config", "a.yaml", "--format", chosen],
        report_name=f"analysis_report.{chosen}",
    )
    assert code == 0 and calls["format"] == chosen
    assert kept.read_bytes() == b"kept"


def test_runner_leaves_the_marker_to_render_report(monkeypatch, tmp_path):
    """One place writes the marker: a stand-in `render_report` that writes none leaves none."""
    code, _calls = _run_runner(
        monkeypatch,
        tmp_path,
        ["--system-config", "s.yaml", "--analysis-config", "a.yaml"],
        report_name="analysis_report.html",
    )
    assert code == 0
    assert not (tmp_path / REPORT_FORMAT_MARKER).exists()


def test_control_runner_with_an_explicit_format_writes_no_marker(monkeypatch, tmp_path):
    code, calls = _run_runner(
        monkeypatch,
        tmp_path,
        ["--system-config", "s.yaml", "--analysis-config", "a.yaml", "--format", "zip"],
        report_name="analysis_report.zip",
    )
    assert code == 0 and calls["format"] == "zip"
    assert not (tmp_path / REPORT_FORMAT_MARKER).exists()


# --- Every render entry point reads the setting (stand-ins; no solver, no compile) -------------------------------

import ast  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import hhemt.bundle as bundle_module  # noqa: E402
import hhemt.bundle._combine as combine_module  # noqa: E402
import hhemt.bundle.snakefile_generator as snakefile_generator  # noqa: E402
import hhemt.provenance as provenance  # noqa: E402
import hhemt.report_plot_ids as report_plot_ids  # noqa: E402
import hhemt.report_renderers._react_surgery as react_surgery  # noqa: E402
import hhemt.utils as hhemt_utils  # noqa: E402
import hhemt.workflow as workflow_module  # noqa: E402
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config  # noqa: E402


def _report_cfg(mode):
    data = DEFAULT_REPORT_CONFIG.model_dump()
    data["interactive"] = {
        **data["interactive"],
        "report_html_mode": mode,
        "plotly_js_mode": "inline",
        "tabulator_js_mode": "inline",
    }
    return report_config.model_validate(data)


def _cfg_analysis(mode):
    return SimpleNamespace(
        report=_report_cfg(mode), toggle_sensitivity_analysis=False, brand_theme=None, analysis_id="stand-in"
    )


def _sparse_figures(root, total):
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    with open(plots / "figure.html", "wb") as fh:
        fh.truncate(total)  # sparse: st_size is the total and no disk is used


SETTING_CASES = [
    ("html", AUTO_HTML_MAX_BYTES, "html"),
    ("zip", 0, "zip"),
    ("auto", AUTO_HTML_MAX_BYTES - 1, "html"),
    ("auto", AUTO_HTML_MAX_BYTES, "zip"),
]


def _stub_bundle_render(monkeypatch):
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(report_plot_ids, "member_labels_from_status", lambda _d: {})
    monkeypatch.setattr(report_plot_ids, "event_sidebar_labels_from_status", lambda _d: {})
    monkeypatch.setattr(workflow_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    monkeypatch.setattr(workflow_module, "_brand_theme_css_map", lambda _t: {})
    monkeypatch.setattr(snakefile_generator, "write_regeneration_snakefile", lambda *_a, **_kw: None)
    monkeypatch.setattr(bundle_module, "run_subprocess_with_tee", lambda *_a, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", lambda text, **kw: text)
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery_to_zip", lambda _p, **kw: None)


@pytest.mark.parametrize(("mode", "total", "want"), SETTING_CASES)
def test_bundle_regenerate_report_reads_the_setting(monkeypatch, tmp_path, mode, total, want):
    _stub_bundle_render(monkeypatch)
    _sparse_figures(tmp_path, total)
    for fmt in ("html", "zip"):
        (tmp_path / f"analysis_report.{fmt}").write_text("rendered")
    stand_in = SimpleNamespace(
        _root=tmp_path,
        _cfg_analysis=_cfg_analysis(mode),
        _read_static_backend=lambda: "plotly",
        _assert_render_inputs_present=lambda: None,
    )
    assert bundle_module.Bundle.regenerate_report(stand_in).suffix == f".{want}"


def test_control_an_explicit_format_beats_the_bundle_setting(monkeypatch, tmp_path):
    _stub_bundle_render(monkeypatch)
    _sparse_figures(tmp_path, 0)
    for fmt in ("html", "zip"):
        (tmp_path / f"analysis_report.{fmt}").write_text("rendered")
    stand_in = SimpleNamespace(
        _root=tmp_path,
        _cfg_analysis=_cfg_analysis("html"),
        _read_static_backend=lambda: "plotly",
        _assert_render_inputs_present=lambda: None,
    )
    assert bundle_module.Bundle.regenerate_report(stand_in, format="zip").suffix == ".zip"


@pytest.mark.parametrize(
    ("root_total", "child_total", "want"),
    [
        (AUTO_HTML_MAX_BYTES - 1, 0, "html"),
        (AUTO_HTML_MAX_BYTES - 1, 1, "zip"),
        (0, AUTO_HTML_MAX_BYTES, "zip"),
        (0, 0, "html"),
    ],
)
def test_combined_bundle_regenerate_report_sums_root_and_children(monkeypatch, tmp_path, root_total, child_total, want):
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    _sparse_figures(tmp_path, root_total)
    _sparse_figures(tmp_path / "child_crates" / "c1", child_total)
    seen = {}

    def _render(_a, _b, root, formats):
        seen["formats"] = formats
        for fmt in formats:
            (root / f"analysis_report.{fmt}").write_text("rendered")

    monkeypatch.setattr(combine_module, "_render_combined_report", _render)
    out = combine_module.CombinedBundle.regenerate_report(SimpleNamespace(_root=tmp_path, _is_archive=False))
    assert seen["formats"] == (want,) and out.suffix == f".{want}"


class _SnakemakeReached(Exception):
    """Raised by the stand-in `subprocess.run` so a render stops at the `snakemake --report` call."""


def _stop_at_snakemake(monkeypatch):
    def _run(cmd, *args, **kwargs):
        raise _SnakemakeReached(Path(cmd[cmd.index("--report") + 1]))

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(hhemt_utils, "report_rule_figure_stems", lambda _a, **_k: frozenset())
    monkeypatch.setattr(workflow_module, "_assert_snakefile_package_current", lambda _p: None)
    monkeypatch.setattr(workflow_module, "_brand_theme_css_map", lambda _t: {})


def _report_target_of(monkeypatch, method, stand_in):
    _stop_at_snakemake(monkeypatch)
    with pytest.raises(_SnakemakeReached) as reached:
        method(stand_in)
    return reached.value.args[0]


@pytest.mark.parametrize(("mode", "total", "want"), SETTING_CASES)
def test_analysis_render_report_reads_the_setting(monkeypatch, tmp_path, mode, total, want):
    import hhemt.analysis as analysis_module

    monkeypatch.setattr(analysis_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    _sparse_figures(tmp_path, total)
    stand_in = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path), cfg_analysis=_cfg_analysis(mode))
    target = _report_target_of(monkeypatch, TRITONSWMM_analysis.render_report, stand_in)
    assert target == tmp_path / f"analysis_report.{want}"


@pytest.mark.parametrize(("mode", "total", "want"), SETTING_CASES)
def test_sensitivity_render_report_reads_the_setting(monkeypatch, tmp_path, mode, total, want):
    import hhemt.sensitivity_analysis as sensitivity_module

    monkeypatch.setattr(sensitivity_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    _sparse_figures(tmp_path, total)
    stand_in = SimpleNamespace(
        experiment=SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path)),
        cfg_analysis=_cfg_analysis(mode),
    )
    target = _report_target_of(monkeypatch, TRITONSWMM_sensitivity_analysis.render_report, stand_in)
    assert target == tmp_path / f"analysis_report.{want}"


def test_control_an_explicit_format_beats_the_analysis_setting(monkeypatch, tmp_path):
    import hhemt.analysis as analysis_module

    monkeypatch.setattr(analysis_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    _sparse_figures(tmp_path, 0)
    stand_in = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path), cfg_analysis=_cfg_analysis("html")
    )
    _stop_at_snakemake(monkeypatch)
    with pytest.raises(_SnakemakeReached) as reached:
        TRITONSWMM_analysis.render_report(stand_in, format="zip")
    assert reached.value.args[0] == tmp_path / "analysis_report.zip"


def _stub_whole_render(monkeypatch, kind, tmp_path, mode):
    """Stand-ins for a whole `render_report` call; return the stand-in analysis.

    `subprocess.run` writes the file named after `--report`, and every later step of the
    method (surgery, stamp, category order) is replaced, so the call reaches its return
    without Snakemake. `kind` is `analysis` or `sensitivity`.
    """
    import hhemt.analysis as analysis_module
    import hhemt.render_category_order as category_order_module
    import hhemt.sensitivity_analysis as sensitivity_module

    def _run(cmd, *args, **kwargs):
        Path(cmd[cmd.index("--report") + 1]).write_bytes(b"report")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(provenance, "stamp_report_stage", lambda *_a, **_kw: None)
    monkeypatch.setattr(hhemt_utils, "report_rule_figure_stems", lambda _a, **_k: frozenset())
    monkeypatch.setattr(workflow_module, "_assert_snakefile_package_current", lambda _p: None)
    monkeypatch.setattr(workflow_module, "_brand_theme_css_map", lambda _t: {})
    monkeypatch.setattr(category_order_module, "resolve_render_path_category_order", lambda _a: None)
    monkeypatch.setattr(report_plot_ids, "member_labels_from_status", lambda _d: {})
    monkeypatch.setattr(report_plot_ids, "event_sidebar_labels_from_status", lambda _d: {})
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", lambda text, **kw: text)
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery_to_zip", lambda _p, **kw: None)
    monkeypatch.setattr(analysis_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    monkeypatch.setattr(sensitivity_module, "_emit_report_artifacts", lambda *_a, **_kw: None)
    paths = SimpleNamespace(analysis_dir=tmp_path)
    if kind == "sensitivity":
        return SimpleNamespace(
            experiment=SimpleNamespace(analysis_paths=paths), analysis_paths=paths, cfg_analysis=_cfg_analysis(mode)
        )
    return SimpleNamespace(analysis_paths=paths, cfg_analysis=_cfg_analysis(mode))


_RENDER_METHODS = {
    "analysis": TRITONSWMM_analysis.render_report,
    "sensitivity": TRITONSWMM_sensitivity_analysis.render_report,
}


@pytest.mark.parametrize("kind", sorted(_RENDER_METHODS))
@pytest.mark.parametrize("chosen", ["html", "zip"])
def test_render_report_without_format_writes_the_marker(monkeypatch, tmp_path, kind, chosen):
    """A call that chose the format records the choice, so the marker exists however the call was reached.

    Input class: a standard or a sensitivity analysis, rendered with no format, under a
    setting that selects each format. The marker is the file the generated workflow's
    render rule declares as its output.
    """
    stand_in = _stub_whole_render(monkeypatch, kind, tmp_path, chosen)
    out = _RENDER_METHODS[kind](stand_in)
    assert out == tmp_path / f"analysis_report.{chosen}"
    assert (tmp_path / REPORT_FORMAT_MARKER).read_text() == f"{chosen}\n"


@pytest.mark.parametrize("kind", sorted(_RENDER_METHODS))
@pytest.mark.parametrize("explicit", ["html", "zip"])
def test_control_render_report_with_an_explicit_format_writes_no_marker(monkeypatch, tmp_path, kind, explicit):
    """An explicit format is the caller's choice, and no marker records a choice the caller made."""
    other = _OTHER_FORMAT[explicit]
    stand_in = _stub_whole_render(monkeypatch, kind, tmp_path, other)
    out = _RENDER_METHODS[kind](stand_in, format=explicit)
    assert out == tmp_path / f"analysis_report.{explicit}"
    assert not (tmp_path / REPORT_FORMAT_MARKER).exists()


def _invoke_report_from_bundle(monkeypatch, tmp_path, extra_args):
    from typer.testing import CliRunner

    from hhemt.cli import app

    seen = {}

    class _StandInBundle:
        manifest: dict = {}

        @classmethod
        def from_directory(cls, root):
            return cls()

        def regenerate_report(self, *, format=None, declare_stale_plots=False):
            seen["format"] = format
            return tmp_path / f"analysis_report.{format or 'html'}"

    monkeypatch.setattr(bundle_module, "Bundle", _StandInBundle)
    monkeypatch.setattr(bundle_module, "_get_toolkit_git_sha", lambda strict=False: "unknown")
    result = CliRunner().invoke(app, ["report-from-bundle", str(tmp_path), *extra_args])
    assert result.exit_code == 0, result.output
    return seen["format"]


def test_cli_report_from_bundle_without_format_passes_none(monkeypatch, tmp_path):
    assert _invoke_report_from_bundle(monkeypatch, tmp_path, []) is None


def test_control_cli_report_from_bundle_passes_an_explicit_format(monkeypatch, tmp_path):
    assert _invoke_report_from_bundle(monkeypatch, tmp_path, ["--format", "zip"]) == "zip"


# --- The reprocess delete arms remove the marker with the report files ---------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("start_with", ["process", "consolidate", "render"])
def test_analysis_reprocess_removes_the_marker_with_the_report_files(tmp_path, start_with, dry_run):
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    (tmp_path / "_status").mkdir()
    (tmp_path / "plots").mkdir()
    names = ("analysis_report.html", "analysis_report.zip", REPORT_FORMAT_MARKER)
    for name in names:
        (tmp_path / name).write_text("x")
    inst = object.__new__(TRITONSWMM_analysis)
    inst.analysis_paths = SimpleNamespace(analysis_dir=tmp_path, analysis_datatree_zarr=None)
    inst._active_reporting_set = get_reporting_set("default")
    inst._invalidate_downstream_flags(start_with, regenerate_existing=False, dry_run=dry_run)
    assert {n: (tmp_path / n).exists() for n in names} == {n: False for n in names}


def deletion_lists_missing_the_marker(source: str) -> list[int]:
    """Line numbers of `delete_and_account_unless_dry_run` list calls naming `_report_html` but not `_report_marker`."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "delete_and_account_unless_dry_run":
            first = node.args[0] if node.args else None
            if isinstance(first, ast.List):
                names = {e.id for e in first.elts if isinstance(e, ast.Name)}
                if "_report_html" in names and "_report_marker" not in names:
                    bad.append(node.lineno)
    return bad


def test_sensitivity_reprocess_delete_arms_list_the_marker():
    """The sensitivity `reprocess()` needs a built experiment, so its two delete arms are pinned
    structurally: every deletion list that names the html report also names the marker. Residual:
    this reads the call, not the deletion; the behaviour of those arms is covered on Rivanna by
    the sensitivity reprocess tests."""
    source = Path(inspect.getsourcefile(TRITONSWMM_sensitivity_analysis)).read_text(encoding="utf-8")
    assert source.count("delete_and_account_unless_dry_run(") >= 2
    assert deletion_lists_missing_the_marker(source) == []


def test_control_a_deletion_list_without_the_marker_is_found():
    source = (
        "def f():\n    delete_and_account_unless_dry_run([_report_html, _report_zip], scope_dir=d)\n"
        "    delete_and_account_unless_dry_run([_report_html, _report_zip, _report_marker], scope_dir=d)\n"
    )
    assert deletion_lists_missing_the_marker(source) == [2]


# --- The consolidation delete lists the marker ---------------------------------------------------------------------


def report_artifacts_missing(listed) -> set[str]:
    """Report artifacts that the consolidation delete should name and does not."""
    return {"analysis_report.html", "analysis_report.zip", REPORT_FORMAT_MARKER} - set(listed)


def test_delete_consolidation_runner_names_the_marker_with_the_report_files():
    from hhemt.delete_consolidation_runner import _ANALYSIS_LEVEL_ARTIFACTS

    assert report_artifacts_missing(_ANALYSIS_LEVEL_ARTIFACTS) == set()


def test_control_a_list_with_the_report_files_and_no_marker_is_refused():
    from hhemt.delete_consolidation_runner import _ANALYSIS_LEVEL_ARTIFACTS

    without_marker = [a for a in _ANALYSIS_LEVEL_ARTIFACTS if a != REPORT_FORMAT_MARKER]
    assert report_artifacts_missing(without_marker) == {REPORT_FORMAT_MARKER}


# --- The non-sensitivity reprocess Snakefile names the render target the setting or override asks for --------------


def _reprocess_builder(mode, tmp_path):
    """A real `SnakemakeWorkflowBuilder` instance carrying only the attributes
    `generate_reprocess_snakefile` reads; the `_build_*` rule emitters other than the resource
    block are stand-ins returning empty text, so the render rule and `rule all` are the output."""
    from hhemt.config.eda import eda_config
    from hhemt.workflow import SnakemakeWorkflowBuilder

    class _Cfg:
        analysis_id = "stand-in"
        report = _report_cfg(mode)
        toggle_sensitivity_analysis = False
        eda = eda_config()
        clear_raw = "none"

        def __getattr__(self, name):
            return 1

    builder = object.__new__(SnakemakeWorkflowBuilder)
    builder.analysis_paths = SimpleNamespace(analysis_dir=tmp_path, analysis_log_directory=tmp_path / "logs")
    builder.cfg_analysis = _Cfg()
    builder.analysis = SimpleNamespace(df_sims=[], cfg_analysis=builder.cfg_analysis, _active_reporting_set=None)
    builder.system = SimpleNamespace(
        cfg_system=SimpleNamespace(toggle_triton_model=False, toggle_tritonswmm_model=True, toggle_swmm_model=False)
    )
    builder.python_executable = "python"
    builder._get_report_cfg_static_backend = lambda: "plotly"
    builder._get_conda_env_path = lambda: "/conda"
    builder._get_config_args = lambda: "--system-config s.yaml --analysis-config a.yaml"
    for name in dir(SnakemakeWorkflowBuilder):
        if name.startswith("_build_"):
            setattr(builder, name, lambda *a, **k: "")
    builder._build_resource_block = lambda **kw: "        mem_mb=1"
    return builder


def _render_targets(text):
    output = re.search(r"rule render_report:.*?output:\s*\n\s*\"([^\"]+)\"", text, re.S).group(1)
    rule_all = re.search(r"rule all:\s*\n\s*input:(.*?)\n\n", text, re.S).group(1)
    return output, sorted(re.findall(r'"(analysis_report\.[\w{}]+)"', rule_all))


@pytest.mark.parametrize(
    ("mode", "override", "want_output", "want_targets"),
    [
        ("auto", None, REPORT_FORMAT_MARKER, [REPORT_FORMAT_MARKER]),
        ("zip", None, "analysis_report.{format}", ["analysis_report.zip"]),
        ("auto", ["html"], "analysis_report.{format}", ["analysis_report.html"]),
    ],
)
def test_nonsensitivity_reprocess_snakefile_names_the_requested_render_target(
    tmp_path, mode, override, want_output, want_targets
):
    from hhemt.reprocess_snakefile_generator import generate_reprocess_snakefile

    text = generate_reprocess_snakefile(
        _reprocess_builder(mode, tmp_path), start_with="render", override_report_formats=override
    )
    assert _render_targets(text) == (want_output, want_targets)


@pytest.mark.parametrize("override", [None, ["html"], ["zip"]])
def test_reprocess_forwards_the_override_to_the_builder(tmp_path, override):
    """`TRITONSWMM_analysis.reprocess` hands `override_report_formats` to
    `submit_reprocess_workflow`, so an explicit format keeps its meaning on the non-sensitivity
    path. The builder is a stand-in that records the keyword; the call is observed, not its text."""
    from unittest.mock import MagicMock

    stand_in = MagicMock()
    stand_in.cfg_analysis.toggle_sensitivity_analysis = False
    stand_in.cfg_analysis.multi_sim_run_method = "local"
    stand_in.cfg_analysis.clear_raw = "none"
    stand_in.analysis_paths.analysis_dir = tmp_path
    stand_in._all_sim_flags_present.return_value = True
    stand_in._workflow_builder.submit_reprocess_workflow.return_value = {"success": True}
    TRITONSWMM_analysis.reprocess(
        stand_in,
        start_with="render",
        execution_mode="local",
        dry_run=True,
        verbose=False,
        override_report_formats=override,
    )
    call = stand_in._workflow_builder.submit_reprocess_workflow.call_args
    assert call.kwargs["override_report_formats"] == override


class _WriterReached(Exception):
    pass


@pytest.mark.parametrize("override", [None, ["html"], ["zip"]])
def test_submit_reprocess_workflow_forwards_the_override_to_the_writer(monkeypatch, override):
    """The builder hands `override_report_formats` to `write_reprocess_snakefile`: the writer is a
    recorder and the assertion is on the keyword it receives, not on how the call is spelled."""
    import hhemt.reprocess_snakefile_generator as generator

    seen = {}

    def _writer(builder, **kwargs):
        seen.update(kwargs)
        raise _WriterReached

    monkeypatch.setattr(generator, "write_reprocess_snakefile", _writer)
    stand_in = SimpleNamespace(cfg_analysis=SimpleNamespace(multi_sim_run_method="1_job_many_srun_tasks"))
    with pytest.raises(_WriterReached):
        workflow_module.SnakemakeWorkflowBuilder.submit_reprocess_workflow(
            stand_in,
            start_with="render",
            execution_mode="local",
            override_report_formats=override,
            verbose=False,
        )
    assert seen["override_report_formats"] == override
