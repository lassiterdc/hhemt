"""Tests for the EDA plotting family (eda/_plotting.py)."""

from __future__ import annotations

import json
from pathlib import Path

from TRITON_SWMM_toolkit.config.eda import eda_config
from TRITON_SWMM_toolkit.eda import check_cross_sim_identity
from TRITON_SWMM_toolkit.eda._plotting import render_eda_plots


def test_render_cross_sim_identity_emits_master_rooted_with_source(
    synthetic_sensitivity_completed,
):
    """The first EDA plot emits under MASTER-ROOTED plots/eda/ and declares the
    eda/<plot_id>.zarr artifact as a source (R3/D1 carriage guard)."""
    analysis = synthetic_sensitivity_completed.master_analysis  # NOT the wrapper (D4)
    # Calc must run first so the eda/<plot_id>.zarr artifact exists.
    check_cross_sim_identity(analysis)
    root = Path(analysis.analysis_paths.analysis_dir)

    paths = render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config())
    assert len(paths) == 1
    out = paths[0]
    # MASTER-ROOTED: under {root}/plots/eda/, NOT under plots/sensitivity/per_sim/.
    assert out.parent == root / "plots" / "eda"
    assert out.exists()
    # Source declared relative to root as eda/<plot_id>.zarr (the carriage chain).
    manifest = out.parent / f"{out.stem}.manifest.json"
    payload = json.loads(manifest.read_text())
    assert any(s.startswith("eda/") and s.endswith(".zarr") for s in payload["source_paths_relative"])
    assert payload["output_format"] == "html"


def test_unknown_eda_plot_kind_raises(synthetic_sensitivity_completed):
    """An unknown renderer-kind key fails fast at render_eda_plots."""
    import pytest

    analysis = synthetic_sensitivity_completed.master_analysis
    root = Path(analysis.analysis_paths.analysis_dir)
    with pytest.raises(ValueError, match="unknown EDA plot kind"):
        render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config(enabled_plots=["nope"]))
