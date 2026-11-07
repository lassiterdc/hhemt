"""A report config given at run entry governs the whole run, from rule selection to the last plot.

`TRITONSWMM_analysis.run(report_config=...)` chooses the rules to emit from that config. The
render rule and every plot rule then build a fresh analysis from `cfg_analysis.yaml` in their
own process. These tests pin the file the generator writes beside the Snakefile, the
`--report-config` flag on every rule that runs the render runner or the plot command line,
and the two entry points that load the file in place of `cfg_analysis.report`. None of them
compiles or runs a solver.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from hhemt.config.loaders import yaml_to_model
from hhemt.config.report import DEFAULT_REPORT_CONFIG, resolve_report_format
from hhemt.config.report import report_config as ReportConfigModel

RUN_ENTRY_FILE = "report_config_run_entry.yaml"
_CONTENT_NAMED = re.compile(r"report_config_run_entry\.[0-9a-f]{12}\.yaml")


def _run_entry_files(analysis_dir):
    return sorted(analysis_dir.glob("report_config_run_entry.*.yaml"))


_RULE_SPLIT = re.compile(r"(?m)^(?=rule [A-Za-z0-9_]+:)")
_RUNS_THE_REPORT_TAIL = ("report_renderers._cli", "hhemt.render_report_runner")


def _run_entry_cfg(mode):
    data = DEFAULT_REPORT_CONFIG.model_dump()
    data["interactive"] = {
        **data["interactive"],
        "report_html_mode": mode,
        "plotly_js_mode": "inline",
        "tabulator_js_mode": "inline",
    }
    return ReportConfigModel.model_validate(data)


def _differing_mode(analysis):
    """A `report_html_mode` that differs from the one the analysis config holds."""
    return "zip" if analysis.cfg_analysis.report.interactive.report_html_mode != "zip" else "html"


def _report_tail_rules(snakefile_text):
    """Every rule block whose shell runs the render runner or the plot command line."""
    blocks = [b for b in _RULE_SPLIT.split(snakefile_text) if b.startswith("rule ")]
    return [b for b in blocks if any(token in b for token in _RUNS_THE_REPORT_TAIL)]


def _flag_paths(rule_block):
    return re.findall(r"--report-config (\S+)", rule_block)


def _assert_every_report_tail_rule_carries_the_file(text, analysis_dir, mode):
    rules = _report_tail_rules(text)
    assert rules, "the Snakefile emitted no rule that runs the render runner or the plot command line"
    for block in rules:
        name = block.split(":", 1)[0]
        paths = _flag_paths(block)
        assert len(paths) == 1, f"{name} carries {len(paths)} --report-config flags"
        written = Path(paths[0])
        assert written.is_file() and written.parent == analysis_dir and _CONTENT_NAMED.fullmatch(written.name)
        assert yaml_to_model(written, ReportConfigModel).interactive.report_html_mode == mode
    return rules


@pytest.fixture
def multisim(synth_multi_sim_builder, monkeypatch):
    mode = _differing_mode(synth_multi_sim_builder)
    monkeypatch.setattr(synth_multi_sim_builder, "_run_entry_report_cfg", _run_entry_cfg(mode), raising=False)
    return synth_multi_sim_builder, mode


@pytest.fixture
def sensitivity(synth_sensitivity_builder, monkeypatch):
    mode = _differing_mode(synth_sensitivity_builder)
    monkeypatch.setattr(synth_sensitivity_builder, "_run_entry_report_cfg", _run_entry_cfg(mode), raising=False)
    return synth_sensitivity_builder, mode


_MULTISIM_KWARGS = dict(
    process_system_level_inputs=True,
    compile_TRITON_SWMM=True,
    prepare_scenarios=True,
    process_timeseries=True,
)


def test_multisim_snakefile_passes_the_run_entry_config_to_the_render_and_every_plot_rule(multisim):
    analysis, mode = multisim
    text = analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    rules = _assert_every_report_tail_rule_carries_the_file(text, analysis.analysis_paths.analysis_dir, mode)
    names = [b.split(":", 1)[0] for b in rules]
    assert "rule render_report" in names
    assert any(n.startswith("rule plot_") for n in names)


def test_sensitivity_master_snakefile_passes_the_run_entry_config(sensitivity):
    analysis, mode = sensitivity
    text = analysis.sensitivity._workflow_builder.generate_master_snakefile_content(which="both", compression_level=5)
    rules = _assert_every_report_tail_rule_carries_the_file(text, analysis.analysis_paths.analysis_dir, mode)
    assert "rule render_report" in [b.split(":", 1)[0] for b in rules]


def test_sensitivity_reprocess_master_snakefile_passes_the_run_entry_config(sensitivity):
    analysis, mode = sensitivity
    text = analysis.sensitivity._workflow_builder.generate_reprocess_master_snakefile_content(
        which="both", start_with="render"
    )
    _assert_every_report_tail_rule_carries_the_file(text, analysis.analysis_paths.analysis_dir, mode)


def test_reprocess_snakefile_passes_the_run_entry_config(multisim):
    from hhemt.reprocess_snakefile_generator import generate_reprocess_snakefile

    analysis, mode = multisim
    text = generate_reprocess_snakefile(analysis._workflow_builder, start_with="render")
    rules = _assert_every_report_tail_rule_carries_the_file(text, analysis.analysis_paths.analysis_dir, mode)
    assert "rule render_report" in [b.split(":", 1)[0] for b in rules]


def test_control_without_a_run_entry_config_no_flag_and_no_file(synth_multi_sim_builder, monkeypatch):
    """No run-entry config: no flag in any rule and no file in the tree, so the Snakefile is unchanged."""
    monkeypatch.setattr(synth_multi_sim_builder, "_run_entry_report_cfg", None, raising=False)
    before = _run_entry_files(synth_multi_sim_builder.analysis_paths.analysis_dir)
    text = synth_multi_sim_builder._workflow_builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    assert _report_tail_rules(text)
    assert "--report-config" not in text
    assert _run_entry_files(synth_multi_sim_builder.analysis_paths.analysis_dir) == before


def test_run_entry_file_is_written_once_per_content(multisim):
    """A second generation of one config leaves the file's mtime alone: the name holds the content."""
    analysis, _mode = multisim
    builder = analysis._workflow_builder
    builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    (path,) = [
        p for p in _run_entry_files(analysis.analysis_paths.analysis_dir) if str(p) in builder._report_config_args()
    ]
    first = path.stat().st_mtime_ns
    builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    assert path.stat().st_mtime_ns == first


def test_two_different_run_entry_configs_write_two_files_and_two_snakefiles(multisim, monkeypatch):
    analysis, mode = multisim
    builder = analysis._workflow_builder
    first_text = builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    other = "html" if mode == "zip" else "zip"
    monkeypatch.setattr(analysis, "_run_entry_report_cfg", _run_entry_cfg(other), raising=False)
    second_text = builder.generate_snakefile_content(**_MULTISIM_KWARGS)
    named_first = set(re.findall(r"--report-config (\S+)", first_text))
    named_second = set(re.findall(r"--report-config (\S+)", second_text))
    assert len(named_first) == 1 and len(named_second) == 1 and named_first != named_second
    assert all(Path(p).is_file() for p in named_first | named_second)


def test_two_writers_of_one_content_named_file_leave_one_file_and_no_temporary(multisim, monkeypatch):
    """The second writer finds the target present: one file, no `.tmp` left by either, no exception."""
    analysis, _mode = multisim
    builder = analysis._workflow_builder
    directory = analysis.analysis_paths.analysis_dir
    flag = builder._report_config_args()
    (target,) = [p for p in _run_entry_files(directory) if str(p) in flag]
    target.unlink()
    original_write_text = Path.write_text

    def _write_then_let_a_rival_win(self, data, *args, **kwargs):
        # Interleave a second writer: while this one holds its temporary file, the rival
        # puts the same content at the target.
        result = original_write_text(self, data, *args, **kwargs)
        if self.name.startswith(target.name) and self.name.endswith(".tmp"):
            original_write_text(target, data)
        return result

    monkeypatch.setattr(Path, "write_text", _write_then_let_a_rival_win)
    builder._report_config_args()
    monkeypatch.undo()
    assert target.is_file()
    assert not list(directory.glob("report_config_run_entry.*.tmp"))


# --- The two entry points that load the file ---------------------------------------------------


def _write_run_entry_file(tmp_path, mode):
    path = tmp_path / RUN_ENTRY_FILE
    import yaml

    path.write_text(yaml.safe_dump(_run_entry_cfg(mode).model_dump(mode="json"), sort_keys=False))
    return path


def _run_runner(monkeypatch, tmp_path, argv, *, sensitivity=False):
    """Run `render_report_runner.main` with stand-in classes.

    The stand-in render resolves its format from the config the analysis holds.
    """
    import hhemt.analysis as analysis_module
    import hhemt.system as system_module
    from hhemt import render_report_runner

    seen = {}
    analysis_config_mode = "html" if _differing_mode_name(argv) == "zip" else "zip"

    class _Render:
        def render_report(self, format=None, reprocess=False):
            held = getattr(self, "_cfg_report", None) or self.cfg_analysis.report
            chosen = format or resolve_report_format(held.interactive.report_html_mode, 0)
            seen["format"] = chosen
            out = tmp_path / f"analysis_report.{chosen}"
            out.write_bytes(b"report")
            return out

    class FakeAnalysis(_Render):
        def __init__(self, *args, **kwargs):
            self.cfg_analysis = types.SimpleNamespace(
                toggle_sensitivity_analysis=sensitivity, report=_run_entry_cfg(analysis_config_mode)
            )
            if sensitivity:
                self.sensitivity = _Sens(self.cfg_analysis)
            seen["analysis"] = self

    class _Sens(_Render):
        def __init__(self, cfg_analysis):
            self.cfg_analysis = cfg_analysis

    monkeypatch.setattr(analysis_module, "TRITONSWMM_analysis", FakeAnalysis)
    monkeypatch.setattr(system_module, "TRITONSWMM_system", lambda *a, **k: object())
    monkeypatch.setattr(sys, "argv", ["render_report_runner", *argv])
    return render_report_runner.main(), seen


def _differing_mode_name(argv):
    """The mode the run-entry file holds: the stand-in analysis config gets the other one."""
    return "zip" if "--report-config" in argv else "html"


@pytest.mark.parametrize("sensitivity", [False, True])
def test_runner_gives_the_run_entry_config_to_the_analysis_it_builds(monkeypatch, tmp_path, sensitivity):
    """The runner with `--report-config` renders in the format the run-entry value selects."""
    path = _write_run_entry_file(tmp_path, "zip")
    code, seen = _run_runner(
        monkeypatch,
        tmp_path,
        ["--system-config", "s.yaml", "--analysis-config", "a.yaml", "--report-config", str(path)],
        sensitivity=sensitivity,
    )
    assert code == 0
    assert seen["format"] == "zip"  # the stand-in analysis config holds html


def test_control_runner_without_the_flag_uses_the_analysis_config(monkeypatch, tmp_path):
    code, seen = _run_runner(monkeypatch, tmp_path, ["--system-config", "s.yaml", "--analysis-config", "a.yaml"])
    assert code == 0
    assert seen["format"] == "zip"  # the stand-in analysis config holds zip when no file is given


@pytest.fixture
def fresh_cli():
    """Import `_cli` anew and leave `sys.modules` and the package as they were.

    `_cli` binds `TRITONSWMM_analysis` and `TRITONSWMM_system` at import, and
    `tests/test_observer_runner_wiring.py` patches those classes before the test imports the
    module, so an import left behind by this test would hide its patch.
    """
    import importlib

    import hhemt.report_renderers as package

    name = "hhemt.report_renderers._cli"
    original = sys.modules.pop(name, None)
    had_attr = hasattr(package, "_cli")
    original_attr = getattr(package, "_cli", None)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)
    if original is not None:
        sys.modules[name] = original
    if had_attr:
        package._cli = original_attr
    elif hasattr(package, "_cli"):
        del package._cli


def test_plot_command_line_loads_the_run_entry_config_in_place_of_the_analysis_config(monkeypatch, tmp_path, fresh_cli):
    _cli = fresh_cli
    path = _write_run_entry_file(tmp_path, "zip")
    seen = {}

    class FakeAnalysis:
        def __init__(self, *args, **kwargs):
            self.cfg_analysis = types.SimpleNamespace(toggle_sensitivity_analysis=False, report=_run_entry_cfg("html"))
            self.analysis_paths = types.SimpleNamespace(analysis_dir=tmp_path)

    renderer = types.ModuleType("hhemt.report_renderers.fake_run_entry_renderer")
    renderer.render = lambda analysis, report_cfg, output, **kw: seen.update(cfg=report_cfg, holder=analysis)
    monkeypatch.setitem(sys.modules, "hhemt.report_renderers.fake_run_entry_renderer", renderer)
    monkeypatch.setattr(_cli, "TRITONSWMM_analysis", FakeAnalysis)
    monkeypatch.setattr(_cli, "TRITONSWMM_system", lambda *a, **k: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_cli",
            "fake_run_entry_renderer",
            "--system-config",
            "s.yaml",
            "--analysis-config",
            "a.yaml",
            "--output",
            str(tmp_path / "out.html"),
            "--report-config",
            str(path),
        ],
    )
    _cli.main()
    assert seen["cfg"].interactive.report_html_mode == "zip"
    assert seen["holder"]._cfg_report.interactive.report_html_mode == "zip"


def _restore_run_state_after(monkeypatch, analysis):
    """Put back the attributes `run()` sets on the session-shared analysis when the test ends."""
    for name in ("_run_entry_report_cfg", "_active_reporting_set", "_cfg_report", "_brand_theme"):
        monkeypatch.setattr(analysis, name, getattr(analysis, name, None), raising=False)


# --- The run() seam that records the config -----------------------------------------------------


class _StopAfterRecord(Exception):
    """Raised by the stand-in pre-step so `run()` stops right after it records the report config."""


def _stop_run_after_the_recording(monkeypatch, analysis):
    """Replace the `--build-sifs` pre-step, the first call after the recording, with a raise."""

    def _raise(**_kwargs):
        raise _StopAfterRecord

    monkeypatch.setattr(analysis, "_build_sifs_prestep", _raise, raising=False)


def test_run_records_the_run_entry_config(synth_multi_sim_builder, tmp_path, monkeypatch):
    """`run(report_config=...)` records the loaded config for the workflow generator.

    Input class: an analysis run with a report config file whose `report_html_mode` differs
    from the analysis config's. A stand-in for the `--build-sifs` pre-step raises
    `_StopAfterRecord`, the first call after the assignment that records the config, so no
    workflow is built.
    """
    import yaml

    analysis = synth_multi_sim_builder
    _restore_run_state_after(monkeypatch, analysis)
    _stop_run_after_the_recording(monkeypatch, analysis)
    mode = _differing_mode(analysis)
    path = tmp_path / "run_entry.yaml"
    path.write_text(yaml.safe_dump(_run_entry_cfg(mode).model_dump(mode="json"), sort_keys=False))
    with pytest.raises(_StopAfterRecord):
        analysis.run(report_config=path, build_sifs=True)
    assert analysis._run_entry_report_cfg is not None
    assert analysis._run_entry_report_cfg.interactive.report_html_mode == mode


def test_control_run_without_report_config_records_none(synth_multi_sim_builder, monkeypatch):
    """A run given no report config clears a config an earlier run recorded."""
    analysis = synth_multi_sim_builder
    _restore_run_state_after(monkeypatch, analysis)
    _stop_run_after_the_recording(monkeypatch, analysis)
    analysis._run_entry_report_cfg = _run_entry_cfg("zip")
    with pytest.raises(_StopAfterRecord):
        analysis.run(build_sifs=True)
    assert analysis._run_entry_report_cfg is None


# --- The static-plots Snakefile ------------------------------------------------------------------


class _StaticSnakefileSite(Exception):
    """Raised by the stand-in writer so `submit_static_plots_workflow` stops after the call under test."""


def _static_snakefile_text(analysis, tmp_path, monkeypatch):
    """The static-plots Snakefile text built from the config string `submit_static_plots_workflow` passes.

    The stand-in `write_static_snakefile` records the `config_args_str` argument of the real call
    site and stops the submission; the real generator then renders the Snakefile from that string.
    """
    import yaml

    import hhemt.static_snakefile_generator as static_module

    cfg = tmp_path / "system_overview.yaml"
    cfg.write_text(
        yaml.safe_dump({"plot_id": "system_overview", "renderer_kind": "system_overview", "output_format": "pdf"})
    )
    seen = {}

    def _record(_analysis, **kwargs):
        seen.update(kwargs)
        raise _StaticSnakefileSite

    monkeypatch.setattr(static_module, "write_static_snakefile", _record)
    with pytest.raises(_StaticSnakefileSite):
        analysis._workflow_builder.submit_static_plots_workflow(
            resolved_static_plot_configs=[cfg], dry_run=True, verbose=False
        )
    return static_module.generate_static_snakefile(
        analysis,
        static_plot_configs=[cfg],
        config_args_str=seen["config_args_str"],
        static_backend=seen["static_backend"],
    )


def test_static_plots_snakefile_passes_the_run_entry_config_to_every_plot_rule(multisim, tmp_path, monkeypatch):
    analysis, mode = multisim
    text = _static_snakefile_text(analysis, tmp_path, monkeypatch)
    rules = _assert_every_report_tail_rule_carries_the_file(text, analysis.analysis_paths.analysis_dir, mode)
    assert all("report_renderers._cli" in block for block in rules)


def test_control_static_plots_snakefile_without_a_run_entry_config_carries_no_flag(
    synth_multi_sim_builder, tmp_path, monkeypatch
):
    monkeypatch.setattr(synth_multi_sim_builder, "_run_entry_report_cfg", None, raising=False)
    text = _static_snakefile_text(synth_multi_sim_builder, tmp_path, monkeypatch)
    assert _report_tail_rules(text)
    assert "--report-config" not in text


# --- The run-entry config also fixes the shape of the Snakefile --------------------------------


def _holding(monkeypatch, analysis, cfg):
    """Make `analysis` hold `cfg` the way `run()` does: as the run-entry snapshot `_cfg_report`."""
    monkeypatch.setattr(analysis, "_cfg_report", cfg, raising=False)


def _shape_cfg(**interactive):
    data = DEFAULT_REPORT_CONFIG.model_dump()
    data["interactive"] = {
        **data["interactive"],
        "plotly_js_mode": "inline",
        "tabulator_js_mode": "inline",
        **interactive,
    }
    return ReportConfigModel.model_validate(data)


def _analysis_config_says(monkeypatch, analysis, **interactive):
    for name, value in interactive.items():
        monkeypatch.setattr(analysis.cfg_analysis.report.interactive, name, value)


_NEUTRAL_TARGET = '"analysis_report.format"'
_MULTISIM_SHAPE_KWARGS = _MULTISIM_KWARGS


def _plot_extensions(text):
    """The extension of the `system_overview` plot output the Snakefile declares."""
    return sorted(set(re.findall(r"plots/system_overview\.(\w+)", text)) - {"log"})


def test_multisim_render_targets_follow_the_run_entry_html_mode(synth_multi_sim_builder, monkeypatch):
    """An analysis config that says `html` and a run-entry config that says `auto` declare the neutral target.

    Input class: a `report_html_mode` that differs between the analysis config and the config
    given at run entry. The run-entry value decides the render target and the rules alike.
    """
    analysis = synth_multi_sim_builder
    _analysis_config_says(monkeypatch, analysis, report_html_mode="html")
    _holding(monkeypatch, analysis, _shape_cfg(report_html_mode="auto"))
    text = analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_SHAPE_KWARGS)
    assert _NEUTRAL_TARGET in text
    assert '"analysis_report.html"' not in text


def test_control_multisim_render_targets_without_a_run_entry_config_follow_the_analysis_config(
    synth_multi_sim_builder, monkeypatch
):
    analysis = synth_multi_sim_builder
    _analysis_config_says(monkeypatch, analysis, report_html_mode="html")
    _holding(monkeypatch, analysis, None)
    text = analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_SHAPE_KWARGS)
    assert '"analysis_report.html"' in text
    assert _NEUTRAL_TARGET not in text


def test_multisim_plot_extension_follows_the_run_entry_static_backend(synth_multi_sim_builder, monkeypatch):
    """A run-entry `static_backend` that differs from the analysis config's changes the plot output extension."""
    analysis = synth_multi_sim_builder
    assert analysis.cfg_analysis.report.interactive.static_backend == "plotly"
    _holding(monkeypatch, analysis, None)
    plotly = _plot_extensions(analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_SHAPE_KWARGS))
    _holding(monkeypatch, analysis, _shape_cfg(static_backend="matplotlib"))
    matplotlib = _plot_extensions(analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_SHAPE_KWARGS))
    assert plotly == ["html"]
    assert matplotlib and matplotlib != plotly


def test_control_multisim_plot_extension_without_a_run_entry_config_is_the_analysis_configs(
    synth_multi_sim_builder, monkeypatch
):
    analysis = synth_multi_sim_builder
    _holding(monkeypatch, analysis, None)
    text = analysis._workflow_builder.generate_snakefile_content(**_MULTISIM_SHAPE_KWARGS)
    assert _plot_extensions(text) == ["html"]


@pytest.mark.parametrize(
    "generate", ["generate_master_snakefile_content", "generate_reprocess_master_snakefile_content"]
)
def test_sensitivity_master_render_targets_follow_the_run_entry_html_mode(
    synth_sensitivity_builder, monkeypatch, generate
):
    analysis = synth_sensitivity_builder
    builder = analysis.sensitivity._workflow_builder
    kwargs = {"which": "both", "compression_level": 5}
    if generate == "generate_reprocess_master_snakefile_content":
        kwargs = {"which": "both", "start_with": "render"}
    _analysis_config_says(monkeypatch, analysis, report_html_mode="html")
    _holding(monkeypatch, analysis, _shape_cfg(report_html_mode="auto"))
    text = getattr(builder, generate)(**kwargs)
    assert _NEUTRAL_TARGET in text
    assert '"analysis_report.html"' not in text
    # Control: with no run-entry config the same analysis declares the analysis config's target.
    _holding(monkeypatch, analysis, None)
    control = getattr(builder, generate)(**kwargs)
    assert '"analysis_report.html"' in control
    assert _NEUTRAL_TARGET not in control


def test_sensitivity_master_independent_vars_follow_the_run_entry_config(synth_sensitivity_builder, monkeypatch):
    """The `config["independent_vars"]` line of the master Snakefile reads the run-entry sensitivity block."""
    analysis = synth_sensitivity_builder
    builder = analysis.sensitivity._workflow_builder
    data = _shape_cfg().model_dump()
    data["sensitivity"] = {"independent_vars": ["run_entry.only_var"]}
    _holding(monkeypatch, analysis, ReportConfigModel.model_validate(data))
    text = builder.generate_master_snakefile_content(which="both", compression_level=5)
    assert "config[\"independent_vars\"] = ['run_entry.only_var']" in text
    _holding(monkeypatch, analysis, None)
    control = builder.generate_master_snakefile_content(which="both", compression_level=5)
    assert "run_entry.only_var" not in control


def test_reprocess_snakefile_render_targets_follow_the_run_entry_html_mode(synth_multi_sim_builder, monkeypatch):
    """The non-sensitivity reprocess Snakefile declares the render target of the run-entry config."""
    from hhemt.reprocess_snakefile_generator import generate_reprocess_snakefile

    analysis = synth_multi_sim_builder
    builder = analysis._workflow_builder
    _analysis_config_says(monkeypatch, analysis, report_html_mode="html")
    _holding(monkeypatch, analysis, _shape_cfg(report_html_mode="auto"))
    text = generate_reprocess_snakefile(builder, start_with="render")
    assert _NEUTRAL_TARGET in text
    assert '"analysis_report.html"' not in text
    # Control: with no run-entry config the same analysis declares the analysis config's target.
    _holding(monkeypatch, analysis, None)
    control = generate_reprocess_snakefile(builder, start_with="render")
    assert '"analysis_report.html"' in control
    assert _NEUTRAL_TARGET not in control


def test_reprocess_master_independent_vars_follow_the_run_entry_config(synth_sensitivity_builder, monkeypatch):
    """The reprocess master Snakefile reads the run-entry sensitivity block, as the production master does.

    Input class: a run-entry config whose `sensitivity.independent_vars` names a variable the
    analysis config does not, then a run with no run-entry config as the control.
    """
    analysis = synth_sensitivity_builder
    builder = analysis.sensitivity._workflow_builder
    data = _shape_cfg().model_dump()
    data["sensitivity"] = {"independent_vars": ["run_entry.only_var"]}
    _holding(monkeypatch, analysis, ReportConfigModel.model_validate(data))
    text = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    assert "run_entry.only_var" in text
    _holding(monkeypatch, analysis, None)
    control = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    assert "run_entry.only_var" not in control
