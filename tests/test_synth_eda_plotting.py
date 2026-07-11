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
