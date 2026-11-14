"""Three repairs the real DEM-resolution run exposed in ``analysis.eda()`` and the plot dispatcher.

The run (``dem --eda``, the DEM-resolution ladder) crossed three defects that no fixture reached,
because every earlier fixture held one grid, one backing store per kind and one layout:

1. ``check_raw_byte_identity`` decoded every member of a hardware family with ONE index stub sized
   from the family reference's header, so a ladder whose members run on different grids raised an
   xarray ``CoordinateValidationError`` and failed the whole ``analysis.eda()`` loop.
2. ``render_eda_plots`` tested every kind for an ``eda/{kind}.zarr`` backing store, and the four
   DEM-resolution kinds have none (they read ``experiment_datatree.zarr`` at the root), so the
   gate warned a false "degradation panel is rendered" under ``must_emit`` and SKIPPED the figures
   on every render-all route.
3. ``build_config_diff_figure`` and ``config_diff_source_paths`` looked the watershed polygon up in
   ``{root}/cfg_system.yaml`` only, a file the analysis directory of a real run does not hold.

No test here compiles or runs a solver or reaches a ``*_compiled`` or ``synthetic_*_completed``
fixture: the raw rasters are a few written bytes and the trees are small stored zarr trees.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.config.eda import eda_config
from hhemt.eda._plotting import (
    _ALWAYS_EMIT_KINDS,
    _EDA_RENDERERS,
    _RENDERER_BACKING_ARTIFACT,
    render_eda_plots,
)
from tests.test_eda_dem_figures_real_layout import _EPSG, _write_tree

_DEM_KINDS = (
    "dem_resolution_cost_error",
    "dem_resolution_coupling_table",
    "dem_resolution_diff_maps",
    "dem_resolution_error_ecdf",
)


# ---------------------------------------------------------------------------------------------
# Repair 1: raw byte identity is defined within one grid
# ---------------------------------------------------------------------------------------------


def _write_bin_raster(path: Path, arr: np.ndarray) -> None:
    """The documented TRITON bin format: a float64 ``[y_dim, x_dim]`` header, then row-major data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = arr.shape
    np.concatenate([[float(y), float(x)], arr.ravel()]).astype(np.float64).tofile(path)


def _raw_sub(tmp_path: Path, name: str, n: int, *, run_mode: str = "serial", n_gpus: int = 0):
    """A stand-in member whose raw bin directory holds one timestep of an ``n`` x ``n`` grid."""
    raw_bin = tmp_path / name / "sims" / "event_index.0" / "out_triton" / "bin"
    base = np.arange(n * n, dtype=np.float64).reshape(n, n)
    for prefix in ("H", "MH"):
        _write_bin_raster(raw_bin / f"{prefix}0", base)
    sub = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path, simulation_directory=tmp_path / name / "sims"),
        cfg_analysis=SimpleNamespace(
            run_mode=run_mode,
            n_gpus=n_gpus,
            n_mpi_procs=1,
            n_omp_threads=1,
            n_nodes=1,
            hpc_ensemble_partition="",
        ),
    )
    return sub, raw_bin


def _run_raw_check(tmp_path: Path, monkeypatch, members: dict[str, tuple]):
    """Run ``check_raw_byte_identity`` over stand-in members; return the written artifact's dataset."""
    import hhemt.analysis_validation as av
    import hhemt.eda.raw_byte_identity as rri
    import hhemt.report_renderers._figure_emission as fe

    subs = {mid: sub for mid, (sub, _raw) in members.items()}
    raws = {id(sub): raw for sub, raw in members.values()}
    monkeypatch.setattr(av, "_iter_members_or_self", lambda m: list(subs.items()))
    monkeypatch.setattr(rri, "_b4b_enabled_model", lambda s: "triton")
    monkeypatch.setattr(rri, "_b4b_n_resumes", lambda m, member_id: 0)
    monkeypatch.setattr(rri, "_b4b_sub_raw_bin_dir", lambda s, model, rot: raws[id(s)])
    monkeypatch.setattr(fe, "emit_data_artifact_with_sources", lambda **kw: kw["artifact_path"])
    cfg = SimpleNamespace(
        TRITON_raw_output_type="bin", TRITON_reporting_timestep_s=60.0, resume_interruption_schedule=None
    )
    master = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path))
    result = rri.check_raw_byte_identity(master, cfg_analysis=cfg, eda_cfg=cfg)
    return result, xr.open_zarr(tmp_path / "eda" / "raw_byte_identity.zarr")


def test_a_family_whose_members_run_on_different_grids_degrades_instead_of_raising(tmp_path, monkeypatch):
    """Two members of one hardware family on a 4 x 4 and an 8 x 8 grid write the degraded marker.

    Property: ``check_raw_byte_identity`` returns, writes ``eda/raw_byte_identity.zarr`` with
    ``degraded == 1``, and its reason says byte identity is defined within one grid and names the
    ``dem_resolution`` reporting set; the verdict is a failure that carries the same reason.

    Class: every sweep whose members differ in grid (the DEM-resolution ladder), in any hardware
    family. The measured defect was an xarray ``CoordinateValidationError`` (``conflicting sizes
    for dimension 'y'``) raised from inside ``analysis.eda()``, which failed the whole loop.

    Why it is required: ``analysis.eda()`` runs this check unconditionally, so the raise was
    reachable from every ladder analysis, whatever ``enabled_plots`` held.

    What kills it: decoding the family with one stub sized from the reference's header, or
    comparing within each grid (``_collapse_replicates`` folds rows by ``config_label``, which
    carries no grid, so that comparison would fold different grids into one row).

    A second correct implementation under which it still passes: any reason wording that keeps
    the two phrases, or reading the grid from the processed tier instead of the raw header.
    """
    fine, fine_raw = _raw_sub(tmp_path, "fine", 8)
    coarse, coarse_raw = _raw_sub(tmp_path, "coarse", 4)
    result, ds = _run_raw_check(tmp_path, monkeypatch, {"fine": (fine, fine_raw), "coarse": (coarse, coarse_raw)})
    assert int(ds.attrs["degraded"]) == 1
    reason = str(ds.attrs["degraded_reason"])
    assert "within one grid" in reason
    assert "dem_resolution" in reason
    assert "4 x 4 cells" in reason and "8 x 8 cells" in reason
    assert result.verdict.passed is False
    assert reason in result.verdict.summary
    assert "compute_config" not in ds["identical"].dims


def test_members_that_share_a_grid_are_still_compared(tmp_path, monkeypatch):
    """Two members of one family on the SAME grid keep the comparison: not degraded, one row each.

    Property: a uniform-grid family produces ``degraded == 0`` and an ``identical`` grid over
    ``compute_config``.

    Class: every ordinary compute sweep (one grid, several compute configurations).

    What kills it: a guard that degrades on any family of more than one member, or on any
    sweep, instead of on a family that holds more than one grid shape.

    A second correct implementation under which it still passes: comparing only the pairs
    whose headers agree, provided a uniform family is compared in full.
    """
    a, a_raw = _raw_sub(tmp_path, "a", 4)
    b, b_raw = _raw_sub(tmp_path, "b", 4)
    _result, ds = _run_raw_check(tmp_path, monkeypatch, {"a": (a, a_raw), "b": (b, b_raw)})
    assert int(ds.attrs["degraded"]) == 0
    assert "compute_config" in ds["identical"].dims


def test_two_families_on_different_grids_are_each_uniform_and_not_degraded(tmp_path, monkeypatch):
    """A cpu family on one grid and a gpu family on another is NOT a mixed-grid family.

    Property: the guard is per family. Each family holds one grid shape, so neither is skipped
    and the sweep is not degraded, although the two families' grids differ.

    Class: a sweep whose hardware families were run on different grids (each family compared only
    within itself, the within-family rule the module states).

    What kills it: comparing the number of distinct grids across ALL members rather than within
    each family.

    A second correct implementation under which it still passes: keying the guard on (family,
    grid) and degrading only a family that holds two grids.
    """
    cpu, cpu_raw = _raw_sub(tmp_path, "cpu", 4)
    gpu, gpu_raw = _raw_sub(tmp_path, "gpu", 8, run_mode="gpu", n_gpus=1)
    _result, ds = _run_raw_check(tmp_path, monkeypatch, {"cpu": (cpu, cpu_raw), "gpu": (gpu, gpu_raw)})
    assert int(ds.attrs["degraded"]) == 0


def test_one_mixed_family_degrades_the_whole_report_target_and_names_only_that_family(tmp_path, monkeypatch):
    """A cpu family on two grids beside a uniform gpu family degrades the target and names the cpu family.

    Property: with one family holding two grid shapes and another holding one, the artifact has
    ``degraded == 1``, the reason names the cpu family's two grids and does not name the gpu
    family, and the verdict is a failure carrying that reason.

    Class: every sweep in which some hardware families are on a mixed grid and others are not.
    The figure draws one panel per report target, so a degraded marker replaces the whole figure;
    the uniform family's rows are computed and not drawn.

    What kills it: ``degraded`` set from ``not per_config`` alone (the uniform family leaves
    ``per_config`` non-empty, so the figure would draw the gpu rows and say nothing of the cpu
    family it skipped), or a reason that lists every family.

    A second correct implementation under which it still passes: degrading only the mixed family
    in the figure, if the reason still names that family and the verdict still fails.
    """
    fine, fine_raw = _raw_sub(tmp_path, "fine", 8)
    coarse, coarse_raw = _raw_sub(tmp_path, "coarse", 4)
    gpu, gpu_raw = _raw_sub(tmp_path, "gpu", 4, run_mode="gpu", n_gpus=1)
    members = {"fine": (fine, fine_raw), "coarse": (coarse, coarse_raw), "gpu": (gpu, gpu_raw)}
    result, ds = _run_raw_check(tmp_path, monkeypatch, members)
    assert int(ds.attrs["degraded"]) == 1
    reason = str(ds.attrs["degraded_reason"])
    assert "cpu family" in reason and "4 x 4 cells" in reason and "8 x 8 cells" in reason
    assert "gpu family" not in reason
    assert result.verdict.passed is False
    assert reason in result.verdict.summary


# ---------------------------------------------------------------------------------------------
# Repair 2: the absence gate skips kinds that have no eda/ backing artifact
# ---------------------------------------------------------------------------------------------


def _root_with_datatree_only(tmp_path: Path) -> Path:
    """A root holding ``experiment_datatree.zarr`` and ``cfg_system.yaml`` and NO ``eda/`` directory."""
    root = tmp_path / "analysis"
    root.mkdir()
    _write_tree(root)
    (root / "cfg_system.yaml").write_text(f"crs:\n  horizontal_epsg: {_EPSG}\n")
    assert not (root / "eda").exists()
    return root


def _backing_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if "backing" in str(w.message)]


@pytest.mark.parametrize("kind", _DEM_KINDS)
def test_a_dem_kind_renders_without_an_eda_backing_store_and_warns_of_nothing(tmp_path, kind):
    """The four DEM-resolution kinds draw from a root that holds no ``eda/`` store, silently.

    Property: with only the datatree and ``cfg_system.yaml`` at the root, ``render_eda_plots``
    returns the figure's path, and emits no backing-artifact warning, both without ``must_emit``
    (``Bundle.eda``, the notebook cell) and with ``must_emit={kind}`` (the report rule).

    Class: the four kinds named in ``_NO_EDA_BACKING_KINDS``; every render route of a DEM-resolution
    analysis. Measured defect: with ``must_emit`` a false "degradation panel is rendered" warning;
    without it the kind was skipped and never drawn.

    What kills it: leaving these kinds under the absence gate, or testing the gate after the kind
    is appended.

    A second correct implementation under which it still passes: a mapping of these kinds to
    ``None`` in ``_RENDERER_BACKING_ARTIFACT`` that the gate honours, provided its other readers
    (the notebook cell) are changed to match.
    """
    root = _root_with_datatree_only(tmp_path)
    eda_cfg = eda_config(enabled_plots=[kind], plotly_js_mode="cdn")
    for must_emit in (None, {kind}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            paths = render_eda_plots(root, cfg_analysis=None, eda_cfg=eda_cfg, must_emit=must_emit)
        assert [p.name for p in paths] == [f"{kind}.html"], must_emit
        assert _backing_warnings(caught) == [], must_emit


def test_a_kind_that_does_have_a_backing_store_keeps_the_absence_gate(tmp_path):
    """A rank-sensitivity kind with no ``eda/`` store is still skipped, with the warning.

    Property: ``render_eda_plots`` returns ``[]`` and warns that the backing artifact is absent.

    Class: every kind OUTSIDE ``_NO_EDA_BACKING_KINDS`` (the calc-member-backed kinds), where an
    absent store is a skipped member or kind/stem drift and the warning is the drift signal.

    What kills it: widening the new set, or skipping the gate for every kind.

    A second correct implementation under which it still passes: any warning wording that keeps
    the word "backing".
    """
    root = _root_with_datatree_only(tmp_path)
    eda_cfg = eda_config(enabled_plots=["eda_rank_sensitivity"], plotly_js_mode="cdn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = render_eda_plots(root, cfg_analysis=None, eda_cfg=eda_cfg)
    assert paths == []
    assert len(_backing_warnings(caught)) == 1


def test_the_no_backing_set_names_four_known_renderers_and_nothing_the_other_sets_own():
    """The set is the four DEM kinds, each a renderer, none in the other two sets.

    Property: ``_NO_EDA_BACKING_KINDS == the four DEM kinds``; it is a subset of
    ``_EDA_RENDERERS``; it shares no kind with ``_ALWAYS_EMIT_KINDS`` (which also promises an
    honest-degradation panel) or with ``_RENDERER_BACKING_ARTIFACT`` (kinds that DO have a store
    under another stem).

    Class: the three kind tables read by the dispatcher, the notebook cell and the kind-table hook.

    What kills it: adding a kind to two of the tables, which changes two different promises.

    A second correct implementation under which it still passes: any structure that keeps the
    same four members.
    """
    from hhemt.eda._plotting import _NO_EDA_BACKING_KINDS

    assert _NO_EDA_BACKING_KINDS == frozenset(_DEM_KINDS)
    assert _NO_EDA_BACKING_KINDS <= set(_EDA_RENDERERS)
    assert not (_NO_EDA_BACKING_KINDS & _ALWAYS_EMIT_KINDS)
    assert not (_NO_EDA_BACKING_KINDS & set(_RENDERER_BACKING_ARTIFACT))


# ---------------------------------------------------------------------------------------------
# Repair 3: the config-diff figure reads the watershed polygon through the system object
# ---------------------------------------------------------------------------------------------


def _write_polygon(path: Path, x0: float, y0: float, x1: float, y1: float) -> Path:
    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs=f"EPSG:{_EPSG}").to_file(path, driver="GeoJSON")
    return path


def _config_diff_root(tmp_path: Path) -> Path:
    """A real-layout root with a two-configuration tree (serial and OpenMP, one 12 x 6 grid, no flow tier).

    Neither ``cfg_system.yaml`` nor ``external/`` sits in the root, as in the analysis directory
    of a real run; only the system object can name the polygon.
    """
    root = tmp_path / "analysis"
    root.mkdir()
    rng = np.random.default_rng(7)
    ny, nx = 12, 6  # taller than wide: the two-column layout refuses a square map at this figure width
    coords = {
        "event_iloc": [0],
        "y": 4100000.0 + np.arange(ny) + 0.5,
        "x": 500000.0 + np.arange(nx) + 0.5,
    }
    base = rng.uniform(0.0, 1.0, (1, ny, nx)).astype("float32")
    nodes: dict[str, xr.Dataset] = {}
    for mid, mode, threads in (("serial_r1", "serial", 1), ("openmp_r1", "openmp", 4)):
        nodes[f"/member_{mid}"] = xr.Dataset(
            attrs={"member_id": mid, "run_mode": mode, "n_omp_threads": threads, "n_mpi_procs": 1, "n_gpus": 0}
        )
        depth = base + (0.0 if mode == "serial" else 0.01)
        nodes[f"/member_{mid}/tritonswmm/triton/summary"] = xr.Dataset(
            {"max_flood_depth": (("event_iloc", "y", "x"), depth)}, coords=coords
        )
    xr.DataTree.from_dict(nodes).to_zarr(root / "experiment_datatree.zarr", mode="w", consolidated=False)
    assert not (root / "cfg_system.yaml").exists() and not (root / "external").exists()
    return root


def _system_naming(polygon: Path | None):
    return SimpleNamespace(
        crs=SimpleNamespace(horizontal_epsg=_EPSG), watershed_gis_polygon=str(polygon) if polygon else None
    )


def test_the_config_diff_declaration_names_the_polygon_the_system_object_holds(tmp_path):
    """``config_diff_source_paths`` declares the polygon path of a supplied system object.

    Property: with no ``cfg_system.yaml`` in the root, the declared sources hold the polygon file
    the object names when it exists, hold none when the object names none or names an absent
    file, and are unchanged (no polygon) when no object is passed.

    Class: every render of a real-layout root through ``render_eda_plots(..., cfg_system=...)``.

    Why it is required: the renderer-IO audit compares what a figure read with what it declared,
    and a polygon read through the object but declared from the file is a mismatch.

    What kills it: leaving the declaration on the file lookup while the build reads the object.

    A second correct implementation under which it still passes: declaring the polygon in the
    renderer instead of in this function, provided the declared set is the same.
    """
    from hhemt.eda._config_diff import config_diff_source_paths

    root = _config_diff_root(tmp_path)
    polygon = _write_polygon(tmp_path / "ws.geojson", 500000.0, 4100000.0, 500004.0, 4100004.0)
    assert polygon in config_diff_source_paths(root, cfg_system=_system_naming(polygon))
    assert polygon not in config_diff_source_paths(root, cfg_system=_system_naming(None))
    assert polygon not in config_diff_source_paths(root, cfg_system=_system_naming(tmp_path / "absent.geojson"))
    assert polygon not in config_diff_source_paths(root)


@pytest.mark.filterwarnings("ignore:EDA plot kind 'config_diff_maps' is a report target")
def test_the_config_diff_figure_reads_the_polygon_it_declares(tmp_path, monkeypatch):
    """The build reads exactly the polygon the declaration names, through the system object.

    Property: rendering ``config_diff_maps`` through ``render_eda_plots(..., cfg_system=obj)`` over a
    root with no ``cfg_system.yaml`` (1) reads the object's polygon file, (2) writes a manifest
    whose declared sources (re-resolved the way the renderer-IO audit re-resolves them) contain
    that file, and (3) draws the watershed swatch only where a polygon was found; a draw with a
    system object that names no polygon reads none and draws none.

    Class: the layout of every real run (the analysis directory holds no system file). A declared
    source outside the root spells ``../..`` in the manifest and re-resolves to the same absolute
    path, which is why the declaration and the read can be compared by path.

    What kills it: reading the polygon from ``{root}/cfg_system.yaml`` only (nothing is read, no
    swatch is drawn), or passing the object to the build but not to the declaration (the read file
    is then absent from the declared set).

    A second correct implementation under which it still passes: resolving the polygon path from
    the object in one shared helper that both the build and the declaration call.
    """
    import geopandas as gpd

    from hhemt.report_renderers._provenance_audit import _declared_set_from_manifest

    root = _config_diff_root(tmp_path)
    polygon = _write_polygon(tmp_path / "ws.geojson", 500000.0, 4100000.0, 500004.0, 4100004.0)
    eda_cfg = eda_config(enabled_plots=["config_diff_maps"], plotly_js_mode="cdn")

    reads: list[Path] = []
    real_read_file = gpd.read_file

    def spy(path, *args, **kwargs):
        reads.append(Path(str(path)).resolve())
        return real_read_file(path, *args, **kwargs)

    monkeypatch.setattr(gpd, "read_file", spy)
    # the root holds no identity artifact: the figure takes its documented "unknown" state, which
    # is not what is under test; the artifact's absence is the layout of the stored tree here.
    out = root / "plots" / "eda" / "config_diff_maps.html"

    def draw(cfg_system):
        reads.clear()
        paths = render_eda_plots(
            root, cfg_analysis=None, eda_cfg=eda_cfg, cfg_system=cfg_system, must_emit={"config_diff_maps"}
        )
        assert [p.name for p in paths] == ["config_diff_maps.html"]
        return out.read_text(), list(reads)

    html_with, read_with = draw(_system_naming(polygon))
    assert polygon.resolve() in read_with
    assert polygon.resolve() in _declared_set_from_manifest(out, root)
    html_without, read_without = draw(_system_naming(None))
    assert polygon.resolve() not in read_without
    assert polygon.resolve() not in _declared_set_from_manifest(out, root)
    assert "watershed" in html_with
    assert "watershed" not in html_without


@pytest.mark.parametrize("kind", ["dem_resolution_diff_maps", "dem_resolution_coupling_table"])
def test_a_dem_figure_declares_the_polygon_its_build_reads_through_the_system_object(tmp_path, monkeypatch, kind):
    """The DEM figures that mask by the watershed declare the polygon file the build opens.

    Property: rendering ``kind`` through ``render_eda_plots(..., cfg_system=obj)`` over a real-layout
    root (no ``cfg_system.yaml``, no ``external/``) (1) reads the polygon the object names and
    (2) writes a manifest whose declared sources, re-resolved the way the renderer-IO audit
    re-resolves them, contain that file; with an object naming no polygon neither holds.

    Class: the two masking DEM figures on every real run, where the polygon is an absolute path
    outside the root. Before the repair the declaration named only ``external/watershed.geojson``
    (the coupling table named no polygon at all), so the audit saw a read that was never declared.

    What kills it: leaving the declaration on the bundle-only path, or resolving the declared file
    by a rule different from the build's.

    A second correct implementation under which it still passes: a declaration that lists the
    polygon through any helper, provided it resolves to the same file the build opens.
    """
    import geopandas as gpd

    from hhemt.report_renderers._provenance_audit import _declared_set_from_manifest

    root = tmp_path / "analysis"
    root.mkdir()
    _write_tree(root)
    polygon = _write_polygon(tmp_path / "ws.geojson", 500000.0, 4100000.0, 500004.0, 4100004.0)
    eda_cfg = eda_config(enabled_plots=[kind], plotly_js_mode="cdn")
    reads: list[Path] = []
    real_read_file = gpd.read_file

    def spy(path, *args, **kwargs):
        reads.append(Path(str(path)).resolve())
        return real_read_file(path, *args, **kwargs)

    monkeypatch.setattr(gpd, "read_file", spy)
    out = root / "plots" / "eda" / f"{kind}.html"
    for named, expect in ((polygon, True), (None, False)):
        reads.clear()
        render_eda_plots(root, cfg_analysis=None, eda_cfg=eda_cfg, cfg_system=_system_naming(named), must_emit={kind})
        assert (polygon.resolve() in reads) is expect
        assert (polygon.resolve() in _declared_set_from_manifest(out, root)) is expect
