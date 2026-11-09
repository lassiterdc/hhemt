"""The DEM-resolution figures draw in the layout a real run produces.

A real run keeps its system configuration OUTSIDE the system directory, so the analysis
directory has no ``system_config.yaml`` beside it and no ``cfg_system.yaml`` inside it; the
report rule's adapter holds the system object and the builders are handed only the analysis
directory. Before the repair every DEM figure failed there with ``FileNotFoundError`` on a
``system_config.yaml`` that only the synthetic fixtures lay out. No test here compiles or
runs a solver or reaches a ``*_compiled`` or ``synthetic_*_completed`` fixture: the figures
draw from a small stored tree.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.config.eda import eda_config
from hhemt.eda._plotting import render_eda_plots
from hhemt.exceptions import ConfigurationError
from hhemt.report_renderers import eda_compute_sensitivity

_EPSG = 32618


def _write_tree(root: Path) -> None:
    """A three-rung experiment tree with the peak-depth, link-flow and performance nodes the figures read."""

    def member(res: float, n: int, seed: int, wall_s: float) -> dict[str, xr.Dataset]:
        rng = np.random.default_rng(seed)
        coords = {
            "event_iloc": [0],
            "y": 4100000.0 + res * (np.arange(n) + 0.5),
            "x": 500000.0 + res * (np.arange(n) + 0.5),
        }
        depth = rng.uniform(0.0, 1.0, (1, n, n)).astype("float32")
        return {
            "triton": xr.Dataset({"max_flood_depth": (("event_iloc", "y", "x"), depth)}, coords=coords),
            "performance": xr.Dataset({"Total": (("Rank",), np.array([wall_s]))}, coords={"Rank": [0]}),
            "swmm_link": xr.Dataset(
                {"max_flow": (("event_iloc", "link_id"), rng.uniform(0.0, 2.0, (1, 2)))},
                coords={"event_iloc": [0], "link_id": ["L1", "L2"]},
            ),
        }

    nodes = {}
    for name, (res, n, seed, wall) in {
        "member_dem_0p5_r1": (0.5, 16, 1, 400.0),
        "member_dem_1_r1": (1.0, 8, 2, 100.0),
        "member_dem_2_r1": (2.0, 4, 3, 30.0),
    }.items():
        for leaf, ds in member(res, n, seed, wall).items():
            nodes[f"/{name}/tritonswmm/{leaf}/summary"] = ds
    xr.DataTree.from_dict(nodes).to_zarr(root / "experiment_datatree.zarr", mode="w", consolidated=False)


def _real_layout(tmp_path: Path) -> Path:
    """The run's layout: the system directory and the analysis directory hold no system config.

    The only ``system_config.yaml`` sits under ``synthetic_test_runs/``, outside the system
    directory, exactly where the measured run put it.
    """
    work = tmp_path / "gb_run"
    root = work / "case" / "synth_dem_tritonswmm"
    root.mkdir(parents=True)
    other = work / "synthetic_test_runs" / "synth_dem_tritonswmm"
    other.mkdir(parents=True)
    (other / "system_config.yaml").write_text(f"crs:\n  horizontal_epsg: {_EPSG}\n")
    _write_tree(root)
    assert not (root.parent / "system_config.yaml").exists()
    assert not (root / "cfg_system.yaml").exists()
    return root


def _analysis(root: Path, kind: str):
    """The stand-in for the analysis the report rule's CLI loads: it holds the system object."""
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=root),
        cfg_analysis=SimpleNamespace(eda=eda_config(enabled_plots=[kind], plotly_js_mode="cdn")),
        _system=SimpleNamespace(cfg_system=SimpleNamespace(crs=SimpleNamespace(horizontal_epsg=_EPSG))),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "dem_resolution_error_ecdf",
        "dem_resolution_cost_error",
        "dem_resolution_coupling_table",
        "dem_resolution_diff_maps",
    ],
)
def test_the_report_rule_adapter_draws_a_dem_figure_where_no_system_config_sits_beside_the_analysis(tmp_path, kind):
    """The adapter route of a real run emits the figure and its manifest.

    Property: with no system configuration file beside or inside the analysis directory, the
    adapter that the report rule calls writes ``plots/eda/{kind}.html`` and its manifest, the
    system object it holds having supplied the EPSG code.

    Class: the layout of every real run (all four builders are drawn here through the adapter;
    they share one helper, covered by the helper tests below). The measured defect was a
    ``FileNotFoundError`` on ``{system directory}/system_config.yaml`` in all four figure rules.

    Why it is required: the fixtures laid the system config out where the old code read it,
    so no test could see that a real run does not.

    What kills it: reading the system config from the directory above the analysis directory,
    or dropping the adapter's pass of the system object to the dispatcher.

    A second correct implementation under which it still passes: reading the code from the
    analysis object through another attribute, or from a persisted copy the adapter writes
    first, provided no file beside the analysis directory is needed.
    """
    root = _real_layout(tmp_path)
    out = root / "plots" / "eda" / f"{kind}.html"
    eda_compute_sensitivity.render(_analysis(root, kind), None, out)
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".manifest.json").exists()


def test_a_root_that_carries_cfg_system_yaml_draws_without_a_system_object(tmp_path):
    """The facade, bundle and notebook routes: the fixed-name copy at the root supplies the code.

    Property: with no system object passed, ``render_eda_plots`` draws a DEM figure from a root
    that holds ``cfg_system.yaml`` with ``crs.horizontal_epsg``.

    Class: the callers that pass no system object (``analysis.eda()`` writes the copy before it
    renders; a render bundle carries it).

    What kills it: ignoring the root's copy.

    A second correct implementation under which it still passes: parsing the file through the
    system-config model instead of the YAML mapping.
    """
    root = _real_layout(tmp_path)
    (root / "cfg_system.yaml").write_text(f"crs:\n  horizontal_epsg: {_EPSG}\n  vertical_epsg: 5703\n")
    paths = render_eda_plots(
        root,
        cfg_analysis=None,
        eda_cfg=eda_config(enabled_plots=["dem_resolution_error_ecdf"], plotly_js_mode="cdn"),
        must_emit={"dem_resolution_error_ecdf"},
    )
    assert [p.name for p in paths] == ["dem_resolution_error_ecdf.html"]


def test_an_absent_epsg_source_raises_naming_the_field_and_both_places(tmp_path):
    """No object and no root copy: a ConfigurationError that names the field and the places.

    Property: ``ConfigurationError`` whose field is ``crs.horizontal_epsg`` and whose message
    names the absent ``cfg_system.yaml`` path; a root copy that lacks the key raises too.

    Class: every caller that supplies neither source; the alternative is a regrid failure far
    from its cause.

    What kills it: a silent fallback (a guessed code, or the old location tried and swallowed).

    A second correct implementation under which it still passes: any message wording that
    keeps the field name and the path.
    """
    from hhemt.eda._dem_resolution_plots import _horizontal_epsg

    root = _real_layout(tmp_path)
    with pytest.raises(ConfigurationError) as caught:
        _horizontal_epsg(root, None)
    assert caught.value.field == "crs.horizontal_epsg"
    assert str(root / "cfg_system.yaml") in str(caught.value)
    (root / "cfg_system.yaml").write_text("crs: {}\n")
    with pytest.raises(ConfigurationError) as caught:
        _horizontal_epsg(root, None)
    assert caught.value.field == "crs.horizontal_epsg"


def test_the_system_object_wins_over_a_stale_root_copy(tmp_path):
    """A supplied system object is authoritative; the root copy is consulted only in its absence.

    Property: with a root copy holding a different code, the helper returns the object's code.

    Class: the two sources disagreeing (a copy written by an earlier run).

    What kills it: preferring the file, which would give a figure from a stale copy.

    A second correct implementation under which it still passes: reading both and asserting
    equality, provided the object's value is the one returned when they agree.
    """
    from hhemt.eda._dem_resolution_plots import _horizontal_epsg

    root = _real_layout(tmp_path)
    (root / "cfg_system.yaml").write_text("crs:\n  horizontal_epsg: 4326\n")
    assert _horizontal_epsg(root, SimpleNamespace(crs=SimpleNamespace(horizontal_epsg=_EPSG))) == _EPSG
    assert _horizontal_epsg(root, None) == 4326


def _polygon_file(path: Path, x0: float, y0: float, x1: float, y1: float) -> Path:
    """A watershed polygon file: a box in the tree's own coordinates (EPSG 32618)."""
    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs=f"EPSG:{_EPSG}").to_file(path, driver="GeoJSON")
    return path


def _coupling_table_cells(root: Path, polygon: Path):
    """Draw the coupling table with a system object that names ``polygon``; return the figure and its cells."""
    from hhemt.eda._dem_resolution_plots import build_dem_resolution_coupling_table_figure

    cfg_system = SimpleNamespace(crs=SimpleNamespace(horizontal_epsg=_EPSG), watershed_gis_polygon=str(polygon))
    fig = build_dem_resolution_coupling_table_figure(root, cfg_system=cfg_system)
    table = next(t for t in fig.data if t.type == "table")
    return fig, [list(col) for col in table.cells.values]


def test_the_watershed_polygon_masks_the_drawn_figure_where_the_analysis_directory_holds_no_system_file(tmp_path):
    """The coupling table's depth columns are restricted to the polygon the system object names.

    Property: in the real layout (no ``cfg_system.yaml`` in the analysis directory), the same
    data drawn with a polygon covering a quarter of the finest grid and drawn with a polygon
    covering all of it gives different depth over-estimate and under-estimate columns, the
    quarter-polygon values never exceed the full-polygon values in magnitude, and the figure
    carries the scope sentence only when the mask applies (a polygon that covers all of the
    grid still applies a mask, so the sentence is present in both; the unmasked absence is
    asserted by the third draw, whose system object names no polygon).

    Class: every analysis directory of a real run, where the old code looked for the polygon
    path in a file that is not there, returned no polygon without an error, and drew the table
    and the diff maps over all cells. Both builders that mask share the one call, so the table
    stands for the pair.

    What kills it: looking the polygon up only in ``{root}/cfg_system.yaml`` (both draws then
    come out unmasked and equal, and the scope sentence is absent).

    A second correct implementation under which it still passes: resolving the polygon path from
    the system object inside the builder instead of inside the shared helper.
    """
    root = _real_layout(tmp_path)
    quarter = _polygon_file(tmp_path / "quarter.geojson", 500000.0, 4100000.0, 500004.0, 4100004.0)
    whole = _polygon_file(tmp_path / "whole.geojson", 499990.0, 4099990.0, 500100.0, 4100100.0)
    fig_q, cells_q = _coupling_table_cells(root, quarter)
    fig_w, cells_w = _coupling_table_cells(root, whole)
    over_q, under_q = cells_q[5], cells_q[6]
    over_w, under_w = cells_w[5], cells_w[6]
    assert (over_q, under_q) != (over_w, under_w)
    for q, w in zip(over_q[1:] + under_q[1:], over_w[1:] + under_w[1:], strict=True):
        assert abs(float(q)) <= abs(float(w))
    scope = "inside the watershed"
    assert any(scope in str(a.text) for a in fig_q.layout.annotations)
    assert any(scope in str(a.text) for a in fig_w.layout.annotations)

    from hhemt.eda._dem_resolution_plots import build_dem_resolution_coupling_table_figure

    no_polygon = SimpleNamespace(crs=SimpleNamespace(horizontal_epsg=_EPSG), watershed_gis_polygon=None)
    fig_n = build_dem_resolution_coupling_table_figure(root, cfg_system=no_polygon)
    assert not any(scope in str(a.text) for a in fig_n.layout.annotations)
