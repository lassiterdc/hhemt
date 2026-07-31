"""Tests for the EDA plotting family (eda/_plotting.py)."""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.config.eda import eda_config
from hhemt.eda import check_cross_sim_identity
from hhemt.eda._plotting import render_eda_plots


def test_render_cross_sim_identity_emits_master_rooted_with_source(
    synthetic_sensitivity_completed,
):
    """The config-diff EDA plot emits under MASTER-ROOTED plots/eda/ and declares a
    BUNDLE-CARRIED data source (R3/D1 carriage guard). The redesigned plot reads the
    consolidated sensitivity_datatree.zarr directly (per-cell max_wlevel_m + per-conduit
    max_flow_cms + per-sub compute-config attrs -- no smaller per-plot summary zarr
    suffices), so it declares that tree (the primary consolidated output carried into
    every render bundle) rather than an eda/<plot_id>.zarr artifact."""
    analysis = synthetic_sensitivity_completed.master_analysis  # NOT the wrapper (D4)
    # Run the calc first, mirroring analysis.eda()'s calc->plots order. The config-diff
    # plot itself reads the consolidated tree (not the calc's eda zarr), but the real
    # facade always runs calc before plots, so exercise that order here.
    check_cross_sim_identity(analysis)
    root = Path(analysis.analysis_paths.analysis_dir)

    paths = render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config())
    assert len(paths) == 1
    out = paths[0]
    # MASTER-ROOTED: under {root}/plots/eda/, NOT under plots/sensitivity/per_sim/.
    assert out.parent == root / "plots" / "eda"
    assert out.exists()
    # Declares a BUNDLE-CARRIED data source: the consolidated sensitivity_datatree.zarr,
    # which the config-diff plot reads directly and which the harvest chain carries into
    # the bundle (verified green by test_synth_eda_facade.py::test_bundle_eda_from_bundle,
    # the fresh-emit bundle_report_data() -> Bundle.eda() round-trip).
    manifest = out.parent / f"{out.stem}.manifest.json"
    payload = json.loads(manifest.read_text())
    assert "sensitivity_datatree.zarr" in payload["source_paths_relative"]
    assert payload["output_format"] == "html"


def test_unknown_eda_plot_kind_raises(synthetic_sensitivity_completed):
    """An unknown renderer-kind key fails fast at render_eda_plots."""
    import pytest

    analysis = synthetic_sensitivity_completed.master_analysis
    root = Path(analysis.analysis_paths.analysis_dir)
    with pytest.raises(ValueError, match="unknown EDA plot kind"):
        render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config(enabled_plots=["nope"]))


def test_b4b_absent_backing_zarr_emits_degradation_panel_not_skip(tmp_path):
    """A b4b figure whose backing eda/<stem>.zarr is absent MUST still emit an honest-
    degradation html (Gate-6), so an ENUMERATED b4b report target is never 'marked for report
    but does not exist'. Pre-fix, render_eda_plots' absence gate skipped it (returned []), which
    at report-assembly time raised WorkflowError. b4b renderers ignore cfg_analysis, so a bare
    tmp_path + a b4b enabled_plots entry exercises the path with no completed-run fixture."""
    (tmp_path / "eda").mkdir()  # exists but carries NO b4b_clean_identity.zarr
    paths = render_eda_plots(
        tmp_path,
        cfg_analysis=None,
        eda_cfg=eda_config(enabled_plots=["b4b_clean_identity"]),
    )
    assert len(paths) == 1, "the b4b figure must be emitted (degradation panel), not absence-skipped"
    out = paths[0]
    assert out == tmp_path / "plots" / "eda" / "b4b_clean_identity.html"
    assert out.exists()
    assert "unavailable" in out.read_text().lower()  # the honest-degradation annotation text


def test_eda_config_drops_retired_b4b_clean_vs_resume_kind():
    """A frozen pre-F8 cfg_analysis.yaml carrying enabled_plots: [..., b4b_clean_vs_resume, ...]
    must DROP the retired kind at config-load (mode='before') with a DeprecationWarning, so the
    no-wipe report re-render's render_eda_plots no longer raises 'unknown EDA plot kind' at that
    position (F8: the within-master figure was removed; the result lives on hhemt combine)."""
    import pytest

    from hhemt.config.eda import eda_config

    with pytest.warns(DeprecationWarning, match="retired EDA figure kind"):
        cfg = eda_config(
            enabled_plots=[
                "config_diff_maps",
                "b4b_clean_identity",
                "b4b_clean_vs_resume",
                "eda_rank_sensitivity",
            ]
        )
    assert "b4b_clean_vs_resume" not in cfg.enabled_plots
    assert cfg.enabled_plots == [
        "config_diff_maps",
        "b4b_clean_identity",
        "eda_rank_sensitivity",
    ]


def test_eda_config_renames_legacy_eda_cross_sim_identity_kind():
    """Precedent coverage (previously untested): the eda_cross_sim_identity -> config_diff_maps
    rename still fires at config-load with a DeprecationWarning."""
    import pytest

    from hhemt.config.eda import eda_config

    with pytest.warns(DeprecationWarning):
        cfg = eda_config(enabled_plots=["eda_cross_sim_identity"])
    assert cfg.enabled_plots == ["config_diff_maps"]


def test_enumerated_config_diff_maps_emits_when_backing_artifact_absent(tmp_path):
    """An ENUMERATED report() target must produce its file even when the backing calc member
    legitimately skipped -- the enumerate-implies-emit invariant.

    Measured pre-fix failure (Rivanna smoke job 17609295): the b4b ReportingSet enumerates
    plots/eda/config_diff_maps.html unconditionally, check_cross_sim_identity skips at one
    sub-analysis, render_eda_plots' absence gate skipped the kind, nothing was written, and
    Snakemake raised MissingOutputException -> workflow FAILED.

    Entry is the ADAPTER (not render_eda_plots directly), because the adapter's `output_path`
    IS the enumerated target -- so this asserts the real seam and fails PRE-FIX on BEHAVIOUR
    (no file written) rather than on a parameter name.
    """
    from types import SimpleNamespace

    from hhemt.report_renderers import eda_compute_sensitivity

    (tmp_path / "eda").mkdir()  # exists but carries NO eda_cross_sim_identity.zarr
    analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=SimpleNamespace(eda=eda_config(enabled_plots=["config_diff_maps"])),
    )
    output_path = tmp_path / "plots" / "eda" / "config_diff_maps.html"

    eda_compute_sensitivity.render(analysis, None, output_path)

    assert output_path.exists(), (
        "an enumerated report() target must be produced; a skip yields Snakemake's "
        "MissingOutputException ('marked for report but does not exist')"
    )
    assert "not applicable" in output_path.read_text().lower()
