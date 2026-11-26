"""Reviewer's mutation tests for the DEM-rim check of ``swmm_node_rim_from_dem`` and its opt-in SWMM fallback.

The author's file (``tests/test_dem_rim_validation.py``) pins the change through the cfg generator. This file grades the
same rule from a second direction, so that a mutation the author's file would not notice turns a test here red:
the floor arithmetic is read through ``swmm_assigned_node_depths`` and ``dem_rim_findings`` directly, with the
boundary at exact equality and at the first value below it, and the surrounding contract (what a refusal leaves on
disk, what a second prepare does, which configurations the preflight refuses, and which tables the new field must
appear in) is read from the outside.

The rule under review, in the developer's words: with ``swmm_node_rim_from_dem`` true every coupled node's DEM cell is
at or above the minimum depth SWMM would assign; the floor is the invert plus the crown of the highest connecting
conduit; the SWMM-depth fallback is an override that is not a default. No test compiles or runs a solver, so the file
runs on a laptop.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ConfigurationError
from hhemt.swmm_utils import DEM_RIM_TOLERANCE_M, dem_rim_findings, swmm_assigned_node_depths
from tests.test_dem_rim_validation import (
    _DEPTH,
    _INVERT,
    _READS_NO_EXEMPT,
    _failing_names,
    _ok_cells,
    _refusal,
    _scenario,
)

_NEW_FIELD = "swmm_node_rim_swmm_fallback"


def _model(*, flow_units: str = "CMS", offsets: str = "DEPTH", junctions: str, conduits: str, xsections: str) -> str:
    """A one-outfall SWMM model text with the given junction, conduit and cross-section rows."""
    coordinates = "\n".join(f"{row.split()[0]} {5 + 10 * i} 95" for i, row in enumerate(junctions.splitlines()))
    return (
        f"[OPTIONS]\nFLOW_UNITS {flow_units}\nLINK_OFFSETS {offsets}\n\n[JUNCTIONS]\n{junctions}\n\n"
        f"[OUTFALLS]\nOUT 0.0 FREE NO\n\n[CONDUITS]\n{conduits}\n\n[XSECTIONS]\n{xsections}\n\n"
        f"[COORDINATES]\n{coordinates}\nOUT 95 95\n"
    )


def _raster(cells_by_column: dict[int, float], *, default: float = 100.0):
    """A 10 by 10 DEM of 10 m cells with the given values on its top row, as rioxarray reads one."""
    import rioxarray  # noqa: F401 -- registers the .rio accessor

    values = np.full((10, 10), default, dtype="float32")
    for column, z in cells_by_column.items():
        values[0, column] = z
    xs = 5.0 + 10.0 * np.arange(10)
    ys = 95.0 - 10.0 * np.arange(10)
    da = xr.DataArray(values[None], dims=("band", "y", "x"), coords={"band": [1], "y": ys, "x": xs})
    return da.rio.write_crs("EPSG:32618").rio.write_nodata(-9999.0)


def _one_pipe(invert: float, diameter: float, *, max_depth: float = 0.0, units: str = "CMS", offset: str = "0"):
    """A model with one node A of the given invert and MaxDepth on a pipe of the given diameter and inlet offset."""
    return _model(
        flow_units=units,
        junctions=f"A {invert} {max_depth} 0 0 0",
        conduits=f"cA A OUT 10 0.013 {offset} 0",
        xsections=f"cA CIRCULAR {diameter} 0 0 0 1",
    )


# --- the floor arithmetic ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_depth", [0.0, 0.3, 0.5, 3.0, 50.0])
def test_the_floor_is_the_crown_whatever_the_authored_maxdepth(max_depth):
    """Property: an authored MaxDepth, above or below the crown, never moves the floor off the crown.

    Class: MaxDepth 0, under the crown (0.3), equal to it, above it (3.0) and far above it (50) on one 0.5 m pipe.
    Why: the developer's rule names the crown of the highest connecting conduit and a node deeper than its crown is
    viable. What kills it: taking max(MaxDepth, crown), or MaxDepth when it is greater than 0 (the first scope text).
    A second correct implementation: reading the depth from a SWMM engine run; the assertion reads the depth only.
    """
    depth = swmm_assigned_node_depths(_one_pipe(1.0, 0.5, max_depth=max_depth), ["A"])["A"]
    assert depth.reason is None
    assert depth.crown_depth == pytest.approx(0.5)


def test_a_dem_rim_below_an_authored_maxdepth_but_at_the_crown_does_not_fail():
    """Property: a node with MaxDepth 3.0 and a 0.5 m pipe, cell at invert plus 0.5, is not flagged.

    Class: the cell equal to the crown floor while the authored MaxDepth is six times that. Why: this is the case the
    second commit exists for; a check against MaxDepth refuses a node whose rim is at its pipe's crown. What kills it:
    the pre-984f6ba6 comparison against the larger of MaxDepth and the crown. Must-fire control: the same node with the
    cell 0.01 m under the crown is flagged, so the zero is not a lookup that never fires.
    """
    text = _one_pipe(1.0, 0.5, max_depth=3.0)
    assert dem_rim_findings(text, ["A"], _raster({0: 1.5})) == []
    (finding,) = dem_rim_findings(text, ["A"], _raster({0: 1.49}))
    assert finding.name == "A" and finding.shortfall_m == pytest.approx(0.01, abs=1e-5)


@pytest.mark.parametrize("below", [1e-3, 1e-4])
def test_the_boundary_equal_to_the_floor_passes_with_float32_noise_and_the_first_value_below_fails(below):
    """Property: a cell authored to give exactly the floor passes after float32 storage, and a cell below it fails.

    Class: invert 2.499 and crown 0.381 (the measured F143612 numbers, neither exact in binary), the cell stored as
    float32 as a DEM is, at invert plus crown (passes) and at 1e-3 and 1e-4 m under it (fail). Why: 1e-5 m is the
    tolerance the author chose; the equal case must pass or every Norfolk node authored at its crown would be
    refused, and the below case must fail or the tolerance has swallowed the rule. What kills it: a tolerance above
    1e-4, or a strict less-than that flags the equal case. Measured: a tolerance of zero still passes the equal case
    here (the float32 read-back happens to land on the floor), so this test does not pin the tolerance from below.
    Both outputs are asserted in this one test so neither arm stands alone.
    """
    invert, crown = 2.499, 0.381
    text = _one_pipe(invert, crown)
    assert dem_rim_findings(text, ["A"], _raster({0: invert + crown})) == []
    (finding,) = dem_rim_findings(text, ["A"], _raster({0: invert + crown - below}))
    assert finding.name == "A" and finding.shortfall_m is not None and finding.shortfall_m > DEM_RIM_TOLERANCE_M


def test_the_highest_connecting_conduit_sets_the_floor_over_both_ends_and_an_offset():
    """Property: the floor is the largest offset-plus-diameter over every conduit end at the node, either end.

    Class: node C with its own outflow pipe (0.6 m at offset 0.2 = 0.8), an inflow pipe ending at it (0.3 m at
    offset 0.1 = 0.4) and a second inflow of 0.9 m at offset 0 (0.9). Why: the highest connected link sets it. What
    kills it: reading only the outflow pipe, only the first pipe, or summing the diameters.
    """
    text = _model(
        junctions="C 1.0 0 0 0 0\nX 5.0 0 0 0 0\nY 5.0 0 0 0 0",
        conduits="cOut C OUT 10 0.013 0.2 0\ncIn X C 10 0.013 0 0.1\ncBig Y C 10 0.013 0 0",
        xsections="cOut CIRCULAR 0.6 0 0 0 1\ncIn CIRCULAR 0.3 0 0 0 1\ncBig CIRCULAR 0.9 0 0 0 1",
    )
    assert swmm_assigned_node_depths(text, ["C"])["C"].crown_depth == pytest.approx(0.9)
    lower = text.replace("cBig CIRCULAR 0.9", "cBig CIRCULAR 0.5")
    assert swmm_assigned_node_depths(lower, ["C"])["C"].crown_depth == pytest.approx(0.8)


def test_an_elevation_offset_below_the_invert_adds_nothing_and_a_star_offset_is_zero():
    """Property: under LINK_OFFSETS ELEVATION an offset under the invert counts as 0, and a star offset is 0.

    Class: invert 1.0 with an elevation offset of 0.5 (below the invert, so 0), of 1.3 (0.3 above it) and a star.
    Why: SWMM's link_getOffsetHeight never returns below 0; a negative height would put the floor under the crown.
    What kills it: subtracting without the clamp.
    """
    elev = _model(
        offsets="ELEVATION",
        junctions="A 1.0 0 0 0 0",
        conduits="cA A OUT 10 0.013 0.5 0",
        xsections="cA CIRCULAR 0.4 0 0 0 1",
    )
    assert swmm_assigned_node_depths(elev, ["A"])["A"].crown_depth == pytest.approx(0.4)
    assert swmm_assigned_node_depths(elev.replace(" 0.013 0.5 0", " 0.013 1.3 0"), ["A"])["A"].crown_depth == (
        pytest.approx(0.7)
    )
    assert swmm_assigned_node_depths(elev.replace(" 0.013 0.5 0", " 0.013 * 0"), ["A"])["A"].crown_depth == (
        pytest.approx(0.4)
    )


def test_a_us_unit_model_is_judged_in_meters_against_a_meter_dem():
    """Property: a model in feet gives its invert and floor in meters, so a meter DEM is compared like with like.

    Class: FLOW_UNITS CFS, invert 10 ft (3.048 m), a 2 ft pipe (0.6096 m); the cell at 3.6576 passes, 0.01 m under
    fails. Why: the solver reads such a model through the same 0.3048 factor. What kills it: scaling the invert and
    not the crown, or neither.
    """
    text = _one_pipe(10.0, 2.0, units="CFS")
    assert dem_rim_findings(text, ["A"], _raster({0: 3.048 + 0.6096})) == []
    (finding,) = dem_rim_findings(text, ["A"], _raster({0: 3.048 + 0.6096 - 0.01}))
    assert finding.swmm_depth_m == pytest.approx(0.6096) and finding.invert_m == pytest.approx(3.048)


def test_a_floor_the_toolkit_cannot_read_is_reported_with_a_reason_and_never_guessed():
    """Property: a node whose crown rise is unreadable is flagged with the reason, whatever its cell.

    Class: a node with an IRREGULAR cross-section, and a node that an outlet link leaves, each over a cell far above
    the invert. Why: the criterion says to say so in the failure and not to guess; a guessed floor of 0 would pass
    both silently. What kills it: treating an unreadable cross-section as no floor. Must-fire control: the readable
    node A beside them, over the same high cell, is not flagged.
    """
    text = (
        _model(
            junctions="A 1.0 0 0 0 0\nI 1.0 0 0 0 0\nO 1.0 0 0 0 0",
            conduits="cA A OUT 10 0.013 0 0\ncI I OUT 10 0.013 0 0",
            xsections="cA CIRCULAR 0.5 0 0 0 1\ncI IRREGULAR tr 0 0 0 1",
        )
        + "\n[OUTLETS]\nw1 O OUT 0 TABULAR/DEPTH x\n"
    )
    raster = _raster({0: 50.0, 1: 50.0, 2: 50.0})
    flagged = {f.name: f for f in dem_rim_findings(text, ["A", "I", "O"], raster)}
    assert set(flagged) == {"I", "O"}
    assert all(f.swmm_depth_m is None and "not computable" in f.reason for f in flagged.values())


# --- the refusal and the override, from outside ----------------------------------------------------------------


def test_a_refusal_leaves_no_file_behind_and_the_default_is_the_refusal(tmp_path, monkeypatch):
    """Property: with the field unset a failing node stops preparation and nothing under the scenario is written.

    Class: node A below its floor, the field absent from the stand-in, then False. Why: a refusal that wrote the
    exemption file or the record first would let a retry run with the override half applied. What kills it: writing
    before judging; a default of True. The file the test wrote (``K``) is the baseline.
    """
    for label, fallback in (("absent", None), ("false", False)):
        cells = {**_ok_cells(), "A": _INVERT + _DEPTH["A"] - 0.01}
        scen, cfg, scen_dir = _scenario(tmp_path / label, monkeypatch, cells, fallback=fallback)
        with pytest.raises(ConfigurationError):
            scen._generate_TRITON_SWMM_cfg()
        assert not cfg.exists()
        assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text() == "K\n"
        assert not (scen_dir / "dem_rim_swmm_fallback.json").exists()


def test_a_max_depth_above_the_crown_never_makes_a_scenario_refuse(tmp_path, monkeypatch):
    """Property: the node B of the fixture (MaxDepth 3.0, crown 0.5) at a cell on its crown does not stop preparation.

    Class: every judged node exactly at its floor, B included; then B alone 0.01 under it, which must refuse naming
    only B. Why: the override is for a node SWMM cannot give its rim, not for one SWMM leaves viable. What kills it:
    judging against MaxDepth. The second arm is the must-fire control.
    """
    scen, cfg, _ = _scenario(tmp_path / "ok", monkeypatch, _ok_cells())
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.exists()
    error = _refusal(tmp_path / "low", monkeypatch, {**_ok_cells(), "B": _INVERT + _DEPTH["B"] - 0.01})
    assert _failing_names(error) == ["B"]


def test_the_override_keeps_the_crown_outfall_never_judges_it_and_is_idempotent(tmp_path, monkeypatch):
    """Property: the exemption file is the crown_rim outfalls plus the failing nodes once each, whatever the repeat.

    Class: K the crown_rim outfall far under its invert (exempt already), A failing, the check run twice on one
    scenario directory. Why: the solver keeps SWMM's depth for exactly the nodes in the file; a second prepare that
    appended again, or dropped K, changes the physics of a re-prepared scenario. What kills it: rewriting the file
    from the failing set alone; appending without a membership test; judging K. The record lists A and not K.
    """
    cells = {**_ok_cells(), "A": _INVERT + _DEPTH["A"] - 0.2}
    scen, _cfg, scen_dir = _scenario(tmp_path, monkeypatch, cells, fallback=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scen._generate_TRITON_SWMM_cfg()
        scen._generate_TRITON_SWMM_cfg()
    assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text().split() == ["K", "A"]
    record = json.loads((scen_dir / "dem_rim_swmm_fallback.json").read_text())
    assert [n["name"] for n in record["nodes"]] == ["A"]


def test_the_override_raises_a_user_warning_naming_the_nodes(tmp_path, monkeypatch):
    """Property: the preparation warning is a UserWarning that names every overridden node and the record file.

    Class: A and C failing with the override on. Why: a quiet override is the failure the developer ruled out ("not a
    default, a conscious toggle"). What kills it: a log line instead of a warning; a warning that names the count only.
    """
    cells = {**_ok_cells(), "A": _INVERT, "C": _INVERT + _DEPTH["C"] - 0.05}
    scen, _cfg, _dir = _scenario(tmp_path, monkeypatch, cells, fallback=True)
    with pytest.warns(UserWarning) as caught:
        scen._generate_TRITON_SWMM_cfg()
    text = " ".join(str(w.message) for w in caught if issubclass(w.category, UserWarning))
    assert "A, C" in text and "dem_rim_swmm_fallback.json" in text


def test_the_override_on_a_build_that_cannot_read_the_exemption_key_is_refused_before_any_write(tmp_path, monkeypatch):
    """Property: the build guard fires before the exemption file or the record is written.

    Class: override on, failing node A, a clone that does not read the key. Why: a refusal after the write leaves a
    file naming nodes the solver would not honour. What kills it: guarding at the cfg stage only.
    """
    cells = {**_ok_cells(), "A": _INVERT + _DEPTH["A"] - 0.2}
    scen, cfg, scen_dir = _scenario(tmp_path, monkeypatch, cells, fallback=True, clone_reads=_READS_NO_EXEMPT)
    with pytest.raises(ConfigurationError):
        scen._generate_TRITON_SWMM_cfg()
    assert not cfg.exists() and not (scen_dir / "dem_rim_swmm_fallback.json").exists()
    assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text() == "K\n"


def test_a_record_from_an_earlier_prepare_does_not_survive_a_prepare_that_overrides_nothing(tmp_path, monkeypatch):
    """Property: after the DEM is amended so no node fails, the scenario carries no record of an override.

    Class: prepare with node A failing and the override on (record written), amend A's cell to its floor, prepare
    again in the same directory. Why: the record is how a reader of the results learns a rim is not from the DEM; a
    stale one says an override happened that did not. What kills it: a check that returns early when nothing fails
    and leaves the earlier record in place. This test is RED when the defect is present.
    """
    low = {**_ok_cells(), "A": _INVERT + _DEPTH["A"] - 0.2}
    scen, _cfg, scen_dir = _scenario(tmp_path / "first", monkeypatch, low, fallback=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scen._generate_TRITON_SWMM_cfg()
    assert (scen_dir / "dem_rim_swmm_fallback.json").exists()
    fixed, _cfg2, _dir2 = _scenario(tmp_path / "first", monkeypatch, _ok_cells(), fallback=True)
    fixed._generate_TRITON_SWMM_cfg()
    assert not (scen_dir / "dem_rim_swmm_fallback.json").exists()


# --- the preflight and the config surface ---------------------------------------------------------------------


def _preflight_hit(tmp_path, *, rim, resolution, fallback, coupled=True):
    """Whether ``preflight_problems`` refuses the new field for the given configuration."""
    from types import SimpleNamespace

    from hhemt.water_level_boundary import preflight_problems

    model = tmp_path / "m.inp"
    model.write_text("[OUTFALLS]\nOUT 0 FREE NO\n[COORDINATES]\nOUT 1 1\n")
    cfg_system = SimpleNamespace(
        SWMM_model=model,
        swmm_outfalls_to_couple=[],
        toggle_tritonswmm_model=coupled,
        system_directory=tmp_path,
        target_dem_resolution=10.0,
        DEM_fullres=tmp_path / "absent.tif",
    )
    cfg_analysis = SimpleNamespace(
        toggle_external_boundaries=False,
        swmm_outfalls_to_keep=[],
        swmm_open_water_outfalls=[],
        open_water_outfall_representation="driven_stage",
        swmm_node_rim_from_dem=rim,
        dem_rim_resolution=resolution,
        water_level_boundary_series_source=None,
        swmm_node_rim_swmm_fallback=fallback,
    )
    report = preflight_problems(cfg_system, cfg_analysis)
    return any(p.field == f"analysis.{_NEW_FIELD}" for p in report.problems)


@pytest.mark.parametrize(
    ("rim", "resolution", "fallback", "refused"),
    [
        (False, None, True, True),  # no DEM rim route at all
        (True, "full", True, True),  # the key is not written under the full-resolution DEM
        (True, "processed", True, False),  # the one configuration the field is for
        (True, "processed", False, False),
        (True, "full", False, False),
        (False, None, False, False),
    ],
)
def test_the_override_without_the_dem_rim_route_is_refused_and_only_then(tmp_path, rim, resolution, fallback, refused):
    """Property: the preflight refuses the override exactly when the TRITON cfg will carry no DEM rim.

    Class: the six combinations of the key, the resolution and the override. Why: the criterion refuses a field that
    does nothing, as dem_rim_resolution is refused; the full-resolution arm writes no key so the field would do
    nothing there too. What kills it: refusing only the key-off arm (misses full); refusing False; accepting True with
    the key off. The refused arms and the accepted arms are each other's must-fire controls.
    """
    assert _preflight_hit(tmp_path, rim=rim, resolution=resolution, fallback=fallback) is refused


def test_the_new_field_is_a_defaulted_bool_and_the_resolution_stays_a_closed_optional_literal():
    """Property: the new field is a ``bool`` defaulting False and not required; the resolution is a closed literal.

    Class: the field's annotation, default and requiredness, then ``dem_rim_resolution`` (``Literal`` or None):
    accepts both literals and None, rejects a third spelling. Why: a default of True would make the override the
    default the developer ruled out; an open string would let a typo skip the DEM-rim route silently. What kills it:
    ``bool | None`` with default None; ``str`` for the resolution.
    """
    from pydantic import TypeAdapter, ValidationError

    from hhemt.config.analysis import analysis_config

    field = analysis_config.model_fields[_NEW_FIELD]
    assert field.annotation is bool and field.default is False and not field.is_required()
    adapter = TypeAdapter(analysis_config.model_fields["dem_rim_resolution"].annotation)
    for ok in ("full", "processed", None):
        assert adapter.validate_python(ok) == ok
    with pytest.raises(ValidationError):
        adapter.validate_python("coarse")


def test_the_new_field_is_bucketed_in_the_reprex_taxonomy():
    """Property: every analysis_config field appears in ``_FIELD_BUCKET`` or the path-policy table.

    Class: the one field this change adds. Why: ``tests/test_reprex_taxonomy.py::test_field_bucket_is_total`` and the
    Metadata renderer (``test_realised_ceiling_disclosure``) both fail on an unbucketed field; the override decides
    which rim the node has, so it belongs with ``swmm_node_rim_from_dem`` ("experiment"). What kills it: adding the
    field and not the bucket. This test is RED when the field is missing.
    """
    from hhemt.config.reprex_taxonomy import _FIELD_BUCKET, _PATH_FIELD_POLICY

    assert _NEW_FIELD in set(_FIELD_BUCKET) | set(_PATH_FIELD_POLICY)
