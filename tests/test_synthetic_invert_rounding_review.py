"""Reviewer tests for the synthetic invert rounding (DR-9 D2, AC-1.4(b)); each stands on the shape's rule alone.

Three gaps of ``tests/test_synthetic_invert_rounding.py`` are probed:

(1) AC-1.4(b) reads ``dem_rim_findings`` over EVERY node name; the coder passes junction names only, because the
    check reads an invert from a ``[JUNCTIONS]`` row. Here each outfall row is re-stated as a junction row of the
    same invert and the check runs over all names, so both outfalls are judged against SWMM's crown depth.
(2) The coder's outfall depth is the generator's own ``_node_depth``; here an outfall's rim is held against the
    crown of the links that end on it, read from the written model.
(3) The coder sweeps one parameter set; here the rule holds on others, where cells are not whole millimeters.
"""

from __future__ import annotations

import math
import random

import pytest
import rioxarray as rxr

from hhemt.swmm_utils import DEM_RIM_TOLERANCE_M, dem_rim_findings, swmm_assigned_node_depths
from hhemt.synthetic_model import swmm_model
from hhemt.synthetic_model._build import SyntheticModelParams, build_synthetic_case
from hhemt.water_level_boundary import dem_cell_elevations, model_coordinates

_PARAM_SETS = [
    SyntheticModelParams(),
    SyntheticModelParams(slope_ns=0.0137, valley_depth_m=0.37),
    SyntheticModelParams(cell_size_m=7.3, n_rows=41, n_cols=21, slope_ns=0.0213),
    SyntheticModelParams(valley_depth_m=0.4137, slope_ns=0.00731, bc_level_mean_m=1.77),
]


def _section(text: str, name: str) -> tuple[int, int]:
    """``(start, end)`` line indexes of the rows of section ``name``: after the header, up to the next header."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper() == f"[{name}]") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    return start, end


def _rows(text: str, name: str) -> list[list[str]]:
    lines = text.splitlines()
    start, end = _section(text, name)
    return [ln.split() for ln in lines[start:end] if ln.strip() and not ln.lstrip().startswith(";")]


def _outfalls_as_junctions(text: str) -> tuple[str, list[str]]:
    """The model with each ``[OUTFALLS]`` row re-stated as a ``[JUNCTIONS]`` row of the same invert, and every name."""
    lines = text.splitlines()
    out_start, out_end = _section(text, "OUTFALLS")
    outfalls = [(i, ln.split()) for i, ln in enumerate(lines[out_start:out_end], out_start)]
    outfalls = [(i, f) for i, f in outfalls if f and not f[0].startswith(";")]
    new_rows = [f"{f[0]} {f[1]} 1.0 0 0 0" for _i, f in outfalls]
    drop = {i for i, _f in outfalls}
    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    text2 = "\n".join(kept) + "\n"
    j_start, _j_end = _section(text2, "JUNCTIONS")
    lines2 = text2.splitlines()
    lines2[j_start:j_start] = new_rows
    text3 = "\n".join(lines2) + "\n"
    names = [r[0] for r in _rows(text3, "JUNCTIONS")]
    return text3, names


@pytest.fixture(scope="module", params=range(len(_PARAM_SETS)))
def case(request, tmp_path_factory):
    params = _PARAM_SETS[request.param]
    arts = build_synthetic_case(params, tmp_path_factory.mktemp(f"rev_invert_{request.param}"))
    return arts, rxr.open_rasterio(arts.dem), arts.swmm_model.read_text(encoding="utf-8")


def test_dem_rim_check_over_every_node_name_returns_nothing(case):
    """AC-1.4(b) as written: every junction AND outfall name, outfalls read as junction rows, returns ``[]``."""
    _arts, dem, text = case
    text2, names = _outfalls_as_junctions(text)
    assert swmm_model._OUTFALL_NAMES <= set(names)
    assert dem_rim_findings(text2, names, dem) == []


def test_every_outfall_rim_reaches_the_crown_of_the_links_that_end_on_it(case):
    """An outfall's DEM rim is at or above the crown of every link at it, computed from the written model, not from
    the generator's depth table."""
    _arts, dem, text = case
    text2, names = _outfalls_as_junctions(text)
    depths = swmm_assigned_node_depths(text2, names)
    coords = model_coordinates(__import__("hhemt.swmm_runnable", fromlist=["_parse"])._parse(text2))
    inverts = {r[0]: float(r[1]) for r in _rows(text2, "JUNCTIONS")}
    from hhemt.swmm_runnable import _fold

    located = {n: (*coords[_fold(n)], inverts[n]) for n in names}
    cells = dem_cell_elevations(dem, located)
    for name in swmm_model._OUTFALL_NAMES:
        crown = depths[name].crown_depth
        assert crown is not None and crown >= 0, (name, depths[name])
        assert cells[name] - inverts[name] >= crown - DEM_RIM_TOLERANCE_M, (name, cells[name], inverts[name], crown)


def test_every_invert_rim_is_within_one_millimeter_above_the_depth_on_other_parameter_sets(case):
    """On parameter sets whose cells are not whole millimeters, each node's rim lies in ``[depth, depth + 0.001)``
    up to the DEM's single-precision spacing and the rule's micrometer snap, and the rim check agrees."""
    _arts, dem, text = case
    text2, names = _outfalls_as_junctions(text)
    depths = swmm_assigned_node_depths(text2, names)
    from hhemt.swmm_runnable import _fold, _parse

    coords = model_coordinates(_parse(text2))
    inverts = {r[0]: float(r[1]) for r in _rows(text2, "JUNCTIONS")}
    cells = dem_cell_elevations(dem, {n: (*coords[_fold(n)], inverts[n]) for n in names})
    for name in names:
        rim = cells[name] - inverts[name]
        depth = swmm_model._node_depth(name)
        assert depth >= depths[name].crown_depth - 1e-12, (name, depth, depths[name])
        assert depth - DEM_RIM_TOLERANCE_M <= rim < depth + 1e-3 + DEM_RIM_TOLERANCE_M, (name, rim, depth)
        assert abs(inverts[name] * 1000 - round(inverts[name] * 1000)) < 1e-6, (name, inverts[name])


def test_the_rule_never_places_an_invert_above_cell_minus_depth_beyond_the_micrometer_snap():
    """Fuzz: for random cells and depths the invert is a whole millimeter, at most 0.5 micrometer above
    ``cell - depth``, and less than one millimeter plus that below it."""
    rng = random.Random(20261125)
    for _ in range(20000):
        cell = rng.uniform(-5.0, 60.0)
        depth = rng.choice([1.0, 0.5, 1.2, 2.5, rng.uniform(0.1, 4.0)])
        invert = swmm_model._invert_elev(cell, depth)
        assert abs(invert * 1000 - round(invert * 1000)) < 1e-6, (cell, depth, invert)
        gap = (cell - depth) - invert
        assert gap > -5e-7 - 1e-9, (cell, depth, invert)
        assert gap < 1e-3 + 1e-9, (cell, depth, invert)


@pytest.mark.parametrize("millimeters", range(-3000, 3000, 7))
def test_a_cell_that_is_an_exact_millimeter_above_the_depth_keeps_its_invert(millimeters):
    """No whole-millimeter drop when ``cell - depth`` is a whole millimeter that floating point reads as a hair low."""
    depth = 1.0
    cell = millimeters / 1000 + depth
    assert math.isclose(swmm_model._invert_elev(cell, depth), millimeters / 1000, abs_tol=1e-12)
