"""A bundle carries the report settings its report was rendered with.

`emit_relative_config_pair` writes the `cfg_analysis.yaml` a bundle regenerates its report
from. The `report` section comes from the render's record in `report_manifest.json`; with no
record the section stays the analysis config's, and a comment line at the head of the file names
the source. None of these tests compiles or runs a solver.
"""

from __future__ import annotations

import pytest
import yaml

import hhemt.provenance as provenance
from hhemt.bundle._emit import emit_relative_config_pair
from hhemt.config.report import DEFAULT_REPORT_CONFIG
from hhemt.config.report import report_config as ReportConfigModel


def _cfg(mode, backend):
    data = DEFAULT_REPORT_CONFIG.model_dump()
    data["interactive"] = {
        **data["interactive"],
        "report_html_mode": mode,
        "static_backend": backend,
        "plotly_js_mode": "inline",
        "tabulator_js_mode": "inline",
    }
    return ReportConfigModel.model_validate(data)


def _differing(analysis):
    held = analysis.cfg_analysis.report.interactive
    return ("zip" if held.report_html_mode != "zip" else "html"), (
        "matplotlib" if held.static_backend != "matplotlib" else "plotly"
    )


def _write_pair(analysis, staging):
    staging.mkdir()
    emit_relative_config_pair(analysis, staging)
    return (staging / "cfg_analysis.yaml").read_text()


@pytest.fixture
def analysis(synth_multi_sim_builder):
    """The shared builder with no report manifest, restored at teardown."""
    manifest = synth_multi_sim_builder.analysis_paths.analysis_dir / "report_manifest.json"
    before = manifest.read_bytes() if manifest.exists() else None
    manifest.unlink(missing_ok=True)
    yield synth_multi_sim_builder
    if before is None:
        manifest.unlink(missing_ok=True)
    else:
        manifest.write_bytes(before)


def test_bundle_report_section_equals_the_record(analysis, tmp_path):
    """Input class: a record whose mode and backend differ from the analysis config's."""
    mode, backend = _differing(analysis)
    given = _cfg(mode, backend)
    provenance.stamp_report_stage(analysis.analysis_paths.analysis_dir, tmp_path / "r.html", given, "run_entry")
    text = _write_pair(analysis, tmp_path / "staging")
    loaded = yaml.safe_load(text)["report"]["interactive"]
    assert (loaded["report_html_mode"], loaded["static_backend"]) == (mode, backend)
    assert text.splitlines()[0] == "# report settings: render record, source run_entry"


def test_bundle_without_a_record_keeps_the_analysis_configs_section_and_says_so(analysis, tmp_path):
    text = _write_pair(analysis, tmp_path / "staging")
    held = analysis.cfg_analysis.report.interactive
    loaded = yaml.safe_load(text)["report"]["interactive"]
    assert (loaded["report_html_mode"], loaded["static_backend"]) == (held.report_html_mode, held.static_backend)
    assert text.splitlines()[0] == "# report settings: analysis config, no render record"


def test_two_writes_are_byte_identical(analysis, tmp_path):
    first = _write_pair(analysis, tmp_path / "a")
    second = _write_pair(analysis, tmp_path / "b")
    assert first == second


def test_a_record_with_the_analysis_configs_source_names_that_source(analysis, tmp_path):
    provenance.stamp_report_stage(
        analysis.analysis_paths.analysis_dir, tmp_path / "r.html", analysis.cfg_analysis.report, "analysis_config"
    )
    text = _write_pair(analysis, tmp_path / "staging")
    assert text.splitlines()[0] == "# report settings: render record, source analysis_config"


def test_bundle_loads_a_cfg_analysis_with_the_source_line_and_without_it(analysis, tmp_path):
    """`Bundle.from_directory` reads the file through YAML, so the leading comment is inert.

    Control: the same file without the line loads to the same `report` section.
    """
    from hhemt.bundle._path_policy import load_bundle_config
    from hhemt.config.analysis import analysis_config

    provenance.stamp_report_stage(
        analysis.analysis_paths.analysis_dir, tmp_path / "r.html", _cfg(*_differing(analysis)), "run_entry"
    )
    staging = tmp_path / "staging"
    text = _write_pair(analysis, staging)
    with_line = load_bundle_config(staging / "cfg_analysis.yaml", analysis_config, staging)
    (staging / "cfg_analysis.yaml").write_text("\n".join(text.splitlines()[1:]) + "\n")
    without_line = load_bundle_config(staging / "cfg_analysis.yaml", analysis_config, staging)
    assert with_line.report == without_line.report


# --- Through emit_bundle ------------------------------------------------------------------------


@pytest.fixture
def emitted(analysis, tmp_path):
    """A bundle written by `emit_bundle` from the shared builder, with a stamped record.

    `emit_bundle` needs one plot sidecar under `plots/`; the fixture writes an empty one and
    removes that sidecar afterwards. No solver is compiled or run.
    """
    import json
    import zipfile

    from hhemt.bundle._emit import emit_bundle

    mode, backend = _differing(analysis)
    provenance.stamp_report_stage(
        analysis.analysis_paths.analysis_dir, tmp_path / "r.html", _cfg(mode, backend), "run_entry"
    )
    plots = analysis.analysis_paths.analysis_dir / "plots"
    made_dir = not plots.exists()
    plots.mkdir(exist_ok=True)
    sidecar = plots / "emit_bundle_probe.manifest.json"
    sidecar.write_text(json.dumps({"plot_id": "emit_bundle_probe", "source_paths_relative": []}))
    try:
        archive = emit_bundle(analysis, tmp_path / "bundle.zip")
    finally:
        sidecar.unlink(missing_ok=True)
        if made_dir:
            plots.rmdir()
    root = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    (config,) = list(root.rglob("cfg_analysis.yaml"))
    return config.parent, (mode, backend)


def test_emit_bundle_keeps_the_source_line_across_the_brand_theme_rewrite(emitted):
    """Input class: any bundle `emit_bundle` writes. The first line of the staged file is the source."""
    bundle_root, _ = emitted
    text = (bundle_root / "cfg_analysis.yaml").read_text()
    assert text.splitlines()[0] == "# report settings: render record, source run_entry"
    assert yaml.safe_load(text)["brand_theme"] == "brand_theme.resolved.yaml"


def test_emit_bundle_output_loads_through_bundle_from_directory(emitted):
    from hhemt.bundle import Bundle

    bundle_root, (mode, backend) = emitted
    bundle = Bundle.from_directory(bundle_root)
    interactive = bundle._cfg_analysis.report.interactive
    assert (interactive.report_html_mode, interactive.static_backend) == (mode, backend)


def test_emit_bundle_without_a_record_keeps_the_no_record_source_line(analysis, tmp_path):
    """Input class: `emit_bundle` over an analysis with no report manifest keeps its source line."""
    import json
    import zipfile

    from hhemt.bundle._emit import emit_bundle

    plots = analysis.analysis_paths.analysis_dir / "plots"
    made_dir = not plots.exists()
    plots.mkdir(exist_ok=True)
    sidecar = plots / "emit_bundle_probe.manifest.json"
    sidecar.write_text(json.dumps({"plot_id": "emit_bundle_probe", "source_paths_relative": []}))
    try:
        archive = emit_bundle(analysis, tmp_path / "bundle.zip")
    finally:
        sidecar.unlink(missing_ok=True)
        if made_dir:
            plots.rmdir()
    with zipfile.ZipFile(archive) as zf:
        (name,) = [n for n in zf.namelist() if n.endswith("cfg_analysis.yaml")]
        first = zf.read(name).decode().splitlines()[0]
    assert first == "# report settings: analysis config, no render record"
