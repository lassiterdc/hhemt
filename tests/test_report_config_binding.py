"""Binding check: every Pydantic field of the per-sim map configs is referenced
in the per_sim_peak_flood_depth renderer source. Catches config fields added or
renamed without a corresponding renderer consumer (config drift)."""
from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.config.report import PerSimMapConfig, PerSimMapInteractiveConfig

_RENDERER = Path(__file__).parent.parent / "src/hhemt/report_renderers/per_sim_peak_flood_depth.py"

# Fields not referenced in per_sim_peak_flood_depth.py, enumerated explicitly so
# the check stays strict. Two reasons a field appears here:
#  (a) consumed in another renderer (per_sim_conduit_flow): utilization_cmap,
#      peak_flow_cmap, conduit_outline_color, conduit_outline_width, conduit_value_width.
#  (b) config-declared but NOT YET consumed anywhere in src/hhemt/ (aspirational/
#      unimplemented interactive config): time_animation, visible_layers_default,
#      colorbar_range_lock, depth_clip_quantile_lower (lower-clip companion, not
#      yet wired into _shared_depth_max), map_tick_step (axis-tick spacing knob,
#      declared but unconsumed).
_ALLOWED_UNREFERENCED: set[str] = {
    "utilization_cmap",
    "peak_flow_cmap",
    "conduit_outline_color",
    "conduit_outline_width",
    "conduit_value_width",
    "time_animation",
    "visible_layers_default",
    "colorbar_range_lock",
    "depth_clip_quantile_lower",
    "map_tick_step",
}

@pytest.mark.parametrize("model_cls", [PerSimMapConfig, PerSimMapInteractiveConfig])
def test_per_sim_map_config_fields_referenced_in_renderer(model_cls):
    src = _RENDERER.read_text()
    missing = [
        name
        for name in model_cls.model_fields
        if name not in src and name not in _ALLOWED_UNREFERENCED
    ]
    assert not missing, (
        f"{model_cls.__name__} fields not referenced in {_RENDERER.name}: {missing}. "
        "Consume the field in the renderer, add it to _ALLOWED_UNREFERENCED with a "
        "reason, or remove it from the config."
    )


def test_per_sim_depth_dryfill_renders_below_heatmap_not_as_occluding_scatter():
    """Regression guard for the iter-21 plotly occlusion fix.

    Plotly draws SVG vector traces (a go.Scatter fill) ABOVE the go.Heatmap
    raster regardless of add_trace order, so the iter6 dry-cell
    `go.Scatter(fill="toself")` watershed polygon occluded the depth/WSE
    heatmaps entirely (the figure rendered all-grey, hiding all flooding). The
    fix renders the dry-cell base fill as a layout shape with layer="below" so
    it sits beneath the heatmap and shows through only the dry (NaN) cells.

    This static guard catches a revert to the occluding-scatter pattern without
    requiring a TRITON-SWMM compile (the render path is compile-dependent).
    """
    src = _RENDERER.read_text()
    # The fix is present: dry-cell fill is a layer="below" layout shape.
    assert 'layer="below"' in src, (
        "per_sim_peak_flood_depth plotly dry-cell base fill must render via "
        'fig.add_shape(..., layer="below") so it sits BENEATH the go.Heatmap. '
        "Without layer=\"below\" the grey fill occludes the heatmap (all-grey "
        "render, no flooding visible) — see iter-21 occlusion fix."
    )
    # The occluding pattern is gone: the old named dry-fill scatter traces
    # ("dry_watershed_depth"/"dry_watershed_wse") must NOT return.
    assert "dry_watershed" not in src, (
        "The dry-cell base fill must NOT be a named go.Scatter(fill=\"toself\") "
        "trace ('dry_watershed_*') — that SVG fill draws ABOVE the heatmap "
        "raster and occludes it. Use a layer=\"below\" layout shape instead "
        "(iter-21 occlusion fix)."
    )
