"""Registration + emit + backing-map + absence-gate coverage for the three
compute-sensitivity EDA renderers (Phase 4). The figure bodies are built (see
``tests/test_eda_figures_built.py`` for their content); this module asserts the
emit/source-declaration/registration seam over an artifact of the shape the calc members
write."""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

from hhemt.config.eda import eda_config
from hhemt.eda._plotting import _EDA_RENDERERS, _RENDERER_BACKING_ARTIFACT, render_eda_plots

_SENS_KINDS = ("eda_rank_sensitivity", "eda_resume_sensitivity", "eda_cross_hardware_magnitude")


def test_three_sensitivity_renderers_registered():
    for kind in _SENS_KINDS:
        assert kind in _EDA_RENDERERS
        assert callable(_EDA_RENDERERS[kind])
    # Superset, not equality. This module is scoped to the three compute-sensitivity
    # renderers (see module docstring); asserting the GLOBAL registry contains nothing
    # else makes every renderer registered by any other feature a failure of this test,
    # which is a cross-cutting break no per-branch test run can anticipate.
    assert {"config_diff_maps", *_SENS_KINDS} <= set(_EDA_RENDERERS)


def test_three_members_reexported_from_eda():
    import hhemt.eda as eda

    for name in ("check_rank_sensitivity", "check_resume_sensitivity", "check_cross_hardware_magnitude"):
        assert hasattr(eda, name)


def test_config_diff_backing_stem_maps_to_cross_sim_identity():
    # The friction unit-guard: the gate must key config_diff_maps on eda_cross_sim_identity,
    # NOT config_diff_maps (nothing writes eda/config_diff_maps.zarr).
    assert _RENDERER_BACKING_ARTIFACT.get("config_diff_maps", "config_diff_maps") == "eda_cross_sim_identity"
    # The three sensitivity kinds are 1:1 (identity default).
    for kind in _SENS_KINDS:
        assert _RENDERER_BACKING_ARTIFACT.get(kind, kind) == kind


def _write_artifact(root: Path, plot_id: str) -> None:
    """An artifact of the calc members' shape: float64 ``(sa_id, event_iloc)`` grids."""
    eda_dir = root / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    dims = ("sa_id", "event_iloc")
    xr.Dataset(
        {
            "identical": (dims, [[1.0, 1.0], [1.0, 0.0]]),
            "max_abs_depth_diff": (dims, [[0.0, 0.0], [0.0, 0.012]]),
            "rmse_wetted_depth": (dims, [[0.0, 0.0], [0.0, 0.004]]),
            "relative_depth_diff_p95": (dims, [[0.0, 0.0], [0.0, 1.5]]),
            "n_newly_wet": (dims, [[0.0, 0.0], [0.0, 3.0]]),
            "n_extent_disagree": (dims, [[0.0, 0.0], [0.0, 6.0]]),
        },
        coords={"sa_id": ["sa_1", "sa_2"], "event_iloc": [0, 1]},
    ).to_zarr(eda_dir / f"{plot_id}.zarr", mode="w", consolidated=False)
    (eda_dir / f"{plot_id}.verdict.json").write_text(json.dumps({"passed": True}))


def test_each_renderer_emits_and_declares_both_sources(tmp_path):
    for kind in _SENS_KINDS:
        root = tmp_path / kind
        root.mkdir()
        _write_artifact(root, kind)
        paths = render_eda_plots(
            root, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=[kind], plotly_js_mode="cdn")
        )
        assert len(paths) == 1
        out = paths[0]
        assert out == root / "plots" / "eda" / f"{kind}.html"
        assert out.exists()
        payload = json.loads((out.parent / f"{out.stem}.manifest.json").read_text())
        rel = payload["source_paths_relative"]
        assert f"eda/{kind}.zarr" in rel
        assert f"eda/{kind}.verdict.json" in rel
        assert payload["output_format"] == "html"


def test_absence_gate_skips_when_backing_artifact_missing(tmp_path):
    # No eda/ dir at all -> every enabled sensitivity kind is skipped (warned), not opened.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=list(_SENS_KINDS)))
    assert paths == []
