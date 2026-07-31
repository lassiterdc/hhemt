"""Watershed-polygon resolution across the two config_diff render roots.

`external/watershed.geojson` is a bundle-EMIT artifact: bundle/_path_policy.py rewrites
cfg_system.watershed_gis_polygon to "external/{filename}" at emit time. A LIVE analysis dir
has no external/ at all -- the polygon sits at the absolute cfg_system path. Without a
fallback the mask degrades to None in every analysis.eda() render, depth_vmax reverts to the
unmasked coastal storm-tide maximum, and no boundary ring is drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import hhemt.eda._config_diff as cd

_GEOJSON = {
    "type": "FeatureCollection",
    "name": "watershed",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
    "features": [
        {
            "type": "Feature",
            "properties": {"id": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[3.5, 17.5], [220.5, 17.5], [220.5, 416.5], [3.5, 416.5], [3.5, 17.5]]
                ],
            },
        }
    ],
}


def _write_geojson(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_GEOJSON))
    return p


def test_bundle_root_resolves_external_relative(tmp_path):
    """Bundle context: external/watershed.geojson resolves (behavior must not change)."""
    root = tmp_path / "bundle"
    _write_geojson(root / "external" / "watershed.geojson")
    (root / "cfg_system.yaml").write_text(
        yaml.safe_dump({"watershed_gis_polygon": "external/watershed.geojson"})
    )
    assert cd._watershed_polygon(root) is not None


def test_live_analysis_root_resolves_via_cfg_system(tmp_path):
    """Live-analysis context: no external/ dir; cfg_system.yaml carries the ABSOLUTE path.

    analysis.eda() writes cfg_system.yaml to the live root BEFORE render_eda_plots(root),
    so the resolver has this input in both contexts.
    """
    root = tmp_path / "analysis_dir"
    root.mkdir()
    abs_poly = _write_geojson(tmp_path / "elsewhere" / "synth_cache" / "watershed.geojson")
    (root / "cfg_system.yaml").write_text(
        yaml.safe_dump({"watershed_gis_polygon": str(abs_poly)})
    )
    assert not (root / "external").exists()
    poly = cd._watershed_polygon(root)
    assert poly is not None, "live-analysis root failed to resolve the watershed polygon"
    assert poly.bounds == (3.5, 17.5, 220.5, 416.5)


def test_genuinely_absent_polygon_still_returns_none(tmp_path):
    """ADR-20 excludable-input case: no external/, cfg_system names a missing file."""
    root = tmp_path / "analysis_dir"
    root.mkdir()
    (root / "cfg_system.yaml").write_text(
        yaml.safe_dump({"watershed_gis_polygon": str(tmp_path / "nope" / "watershed.geojson")})
    )
    assert cd._watershed_polygon(root) is None


def test_no_cfg_system_at_all_returns_none(tmp_path):
    """Neither render context present: resolver degrades quietly rather than raising."""
    root = tmp_path / "bare"
    root.mkdir()
    assert cd._watershed_polygon(root) is None
