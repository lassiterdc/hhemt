"""Reviewer tests for the toolkit side of the open-water outfall (package 1k, chunk 1k.1).

The problem these tests pin. The open-water outfall list is judged at two moments, run entry and scenario
preparation, by one rule each, and the crown_rim representation hands the solver a diameter it measured from the
outlet conduit. Two properties of that arrangement are not held by the chunk's own tests: that the pre-existing
run-entry judgement of ``manhole_table`` agrees with preparation once a crown_rim list is set, and that the crown the
toolkit records for a node is the crown SWMM raises it to under both ``LINK_OFFSETS`` modes.

Every test reads files, builds stand-ins or calls a function. None compiles or runs a solver.

Tests promoted from the chunk's own test file (their original description follows).

The toolkit side of the open-water outfall: its fields, its rules, its dispositions and the files it writes.

The problem this file pins. An outfall that drains to open water can take water back from it, and the coupled
model has two ways to say so: ``driven_stage`` keeps the outfall an outfall whose stage TRITON sets before every
SWMM step, and ``crown_rim`` converts it to a coupled junction at the crown of its outlet conduit, with the
diameter of a circle of the conduit's flow area. Both are chosen by one list, ``swmm_open_water_outfalls``, and one
representation, so every listed outfall of a simulation receives the same treatment. The list is judged by one
function, ``open_water_outfall_problems``, called at run entry and again at scenario preparation. A TRITON build
that does not read the cfg keys the toolkit writes would ignore them without a word, so preparation refuses it.
With the list empty every byte the toolkit writes is the byte it wrote before the field existed.

Every test reads files, builds stand-ins or calls a pure function. None compiles or runs a solver, so the file
runs on a laptop. The conduit table is the Norfolk one (twin rectangular barrels, both pipes), and every fixture
value that a wrong reader could swap differs from its neighbour (the two diameters, the two crowns).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ConfigurationError
from hhemt.water_level_boundary import BoundaryRecord

_MODEL = """[OPTIONS]
FLOW_UNITS CMS
LINK_OFFSETS {offsets}

[JUNCTIONS]
E143254 0 3 0 0 0
E147008 0 3 0 0 0
J9 1 2 0 0 0

[OUTFALLS]
E143250  -1.079  TIMESERIES water_level  NO
E147007  -1.079  TIMESERIES water_level  NO

[CONDUITS]
1444  E143254 E143250  14.789 0.013 0 {outlet_offset} 0 0
1445  E147008 E147007  50.596 0.013 0 0 0 0
1446  J9 E143254  10 0.013 0 0 0 0

[XSECTIONS]
1444  RECT_CLOSED  1.524  1.676  0  0  2
1445  RECT_CLOSED  1.676  1.676  0  0  2
1446  CIRCULAR  1  0  0  0  1

[TIMESERIES]
water_level 01/01/2020 00:00 0.5

[COORDINATES]
E143254 25 45
E147008 65 45
J9 45 15
E143250 25 55
E147007 65 55
"""


def _model(tmp_path: Path, *, offsets: str = "DEPTH", outlet_offset: str = "0") -> Path:
    path = tmp_path / f"model_{offsets}.inp"
    path.write_text(_MODEL.format(offsets=offsets, outlet_offset=outlet_offset))
    return path


def _cfg_system(model: Path, tmp_path: Path):
    return SimpleNamespace(
        SWMM_model=model,
        swmm_outfalls_to_couple=[],
        toggle_tritonswmm_model=True,
        toggle_swmm_model=False,
        swmm_coupling_mode="all_junctions",
        swmm_coupling_nodes_file=None,
        system_directory=tmp_path,
        target_dem_resolution=10.0,
        DEM_fullres=tmp_path / "dem.tif",
    )


def _cfg_analysis(representation: str = "crown_rim", **extra):
    fields = dict(
        toggle_external_boundaries=False,
        toggle_sensitivity_analysis=False,
        one_coupled_inflow_node_per_dem_cell=False,
        swmm_outfalls_to_keep=[],
        swmm_open_water_outfalls=["E143250", "E147007"],
        open_water_outfall_representation=representation,
        manhole_diameter=1.2,
        manhole_loss_coefficient=0.1,
        manhole_table=None,
        manhole_table_fill_with_constants=False,
        swmm_node_rim_from_dem=False,
        dem_rim_resolution=None,
        water_level_boundary_series_source=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def test_run_entry_judges_a_manhole_table_as_preparation_does_when_a_crown_rim_list_is_set(tmp_path):
    """Property: a manhole table that covers every ordinary coupling node passes run entry under a crown_rim list.

    Class: ``manhole_table`` x {no open-water list, a crown_rim list of two outfalls}, with a table that names the
    three ordinary coupling nodes of an ``all_junctions`` model and fill-with-constants off. The crown_rim outfalls
    are coupling nodes too (they are converted to junctions), so the toolkit writes their rows itself, the table
    may not name them, and preparation (``manhole_table_problems`` given ``crown_rim_nodes``) accepts exactly this
    table. Why this is required: ``Analysis._refuse_manhole_table_mismatch`` is the run-entry site of the same rule
    and builds its own ``ManholeTableContext``; if it asks for a table row of a node preparation refuses a row for,
    the one configuration the table rule allows under crown_rim is refused before any job is queued, and the only
    table the run-entry rule accepts is one preparation then refuses. What kills it: a context at the run-entry
    site that does not carry the crown_rim nodes. A second correct implementation: the run-entry site passing the
    crown_rim nodes from ``swmm_utils.crown_rim_outfalls``, or the rule deriving them from the analysis; the
    assertions read only whether the run-entry call raises. Interface premise: the stand-in analysis carries the
    fields ``_refuse_manhole_table_mismatch`` and ``swmm_role_inp`` read, and it is enforced by the control arm
    below (the same stand-in with no list passes).
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.scenario import ManholeTableContext, manhole_table_problems, read_manhole_table

    model = _model(tmp_path)
    table = tmp_path / "table.csv"
    table.write_text(
        "node_id,manhole_diameter_m,manhole_loss_coefficient\nE143254,0.9,0.15\nE147008,0.8,0.25\nJ9,0.6,0.35\n"
    )
    cfg_system = _cfg_system(model, tmp_path)
    system = SimpleNamespace(cfg_system=cfg_system, sys_paths=SimpleNamespace(dem_processed=tmp_path / "p.dem"))

    def run_entry(cfg_analysis) -> None:
        analysis = SimpleNamespace(cfg_analysis=cfg_analysis, _system=system, analysis_config_yaml=None)
        TRITONSWMM_analysis._refuse_manhole_table_mismatch(analysis)

    # Control arm A: with no open-water list the same table is a table of the three coupling nodes and passes.
    run_entry(_cfg_analysis(swmm_open_water_outfalls=[], manhole_table=table))
    # Control arm B: preparation's rule, on the five coupling nodes a crown_rim list yields, accepts the same table.
    rows = read_manhole_table(table)
    context = ManholeTableContext(
        table=table,
        model=str(model),
        diameter_constant=1.2,
        loss_constant=0.1,
        crown_rim_nodes=("E143250", "E147007"),
    )
    five = ["E143254", "E147008", "J9", "E143250", "E147007"]
    assert manhole_table_problems(rows, five, False, context) == []
    # The arm under test: run entry must not refuse what preparation accepts.
    try:
        run_entry(_cfg_analysis(manhole_table=table))
    except ConfigurationError as exc:  # pragma: no cover - the failing arm
        pytest.fail(f"run entry refused a table that preparation accepts under crown_rim: {exc}")
    # A repeat under the other representation is control arm C: a driven outfall is no coupling node, so the table
    # is a table of the three nodes and passes.
    run_entry(_cfg_analysis("driven_stage", manhole_table=table))


@pytest.mark.parametrize("offsets", ["DEPTH", "ELEVATION"])
def test_the_recorded_crown_is_the_height_swmm_raises_the_node_to_under_both_link_offset_modes(tmp_path, offsets):
    """Property: ``CrownRimOutfall.crown_m`` is the height of the outlet conduit's crown above the outfall's invert.

    Class: ``LINK_OFFSETS`` {DEPTH, ELEVATION} for one outlet conduit whose outlet offset is 0.5 in the model's
    length unit and whose outfall invert is -1.079. Under DEPTH the offset is a height above the node's invert, so
    the crown is 0.5 + 1.524; under ELEVATION the offset is an elevation, and SWMM converts it to a height with
    ``link_convertOffsets`` (``offset -= node invert``) before it takes ``fullDepth = offset + yFull``, so the
    crown above the invert is (0.5 - (-1.079)) + 1.524. Why this is required: ``crown_m`` is written to
    ``crown_rim_outfalls.json`` as ``crown_m_above_invert``, the provenance record of the representation, and a
    reader of that record reads the rim the comparison used; the value written under ELEVATION is the DEPTH value.
    What kills it: adding the raw offset to the height whatever ``LINK_OFFSETS`` says. A second correct
    implementation: leaving ``crown_m`` None for an ELEVATION model, which the assertion allows; and one that
    converts the offset with the outfall's invert, which it also allows. Both-states anchor: the assertion turns on
    the height above the invert, a property the DEPTH arm has before and after any repair. The DEPTH arm is the
    control and must stay green.
    """
    from hhemt.swmm_utils import crown_rim_sizing

    model = _model(tmp_path, offsets=offsets, outlet_offset="0.5")
    cfg_analysis = _cfg_analysis(swmm_open_water_outfalls=["E143250"])
    sized, reasons = crown_rim_sizing(_cfg_system(model, tmp_path), cfg_analysis)
    assert reasons == [] and list(sized) == ["E143250"]
    expected = 0.5 + 1.524 if offsets == "DEPTH" else (0.5 - (-1.079)) + 1.524
    crown = sized["E143250"].crown_m
    assert crown is None and offsets == "ELEVATION" or crown == pytest.approx(expected), (offsets, crown, expected)


_LOSS = 0.1


_DIAMETER = 1.2


#: The Norfolk outfalls as ``hague_model.inp`` holds them (the shape's table): two TIMESERIES outfalls, each ending
#: one conduit of twin rectangular barrels. J9 is an ordinary junction.
_MODEL_promoted = """[OPTIONS]
FLOW_UNITS CMS
LINK_OFFSETS DEPTH

[JUNCTIONS]
E143254 0 3 0 0 0
E147008 0 3 0 0 0
J9 1 2 0 0 0

[OUTFALLS]
E143250  -1.079  TIMESERIES water_level  NO
E147007  -1.079  TIMESERIES water_level  NO

[CONDUITS]
1444  E143254 E143250  14.789 0.013 0 0 0 0
1445  E147008 E147007  50.596 0.013 0 0 0 0
1446  J9 E143254  10 0.013 0 0 0 0

[XSECTIONS]
1444  RECT_CLOSED  1.524  1.676  0  0  2
1445  RECT_CLOSED  1.676  1.676  0  0  2
1446  CIRCULAR  1  0  0  0  1

[TIMESERIES]
water_level 01/01/2020 00:00 0.5
water_level 01/01/2020 01:00 0.7

[COORDINATES]
E143254 25 45
E147008 65 45
J9 45 15
E143250 25 55
E147007 65 55
"""


_WATER_LEVEL_RECORD = BoundaryRecord(0, "water_level", None, ("E143250", "E147007"), ((0.0, 0.0), (1.0, 0.0)))


def _model_promoted(tmp_path: Path, text: str = _MODEL_promoted, name: str = "model.inp") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _cfg_system_promoted(model: Path, tmp_path: Path, **extra):
    system = tmp_path / "system"
    system.mkdir(exist_ok=True)
    return SimpleNamespace(
        SWMM_model=model,
        swmm_outfalls_to_couple=[],
        toggle_tritonswmm_model=True,
        swmm_coupling_mode="all_junctions",
        system_directory=system,
        target_dem_resolution=10.0,
        DEM_fullres=tmp_path / "dem.tif",
        **extra,
    )


def _cfg_analysis_promoted(listed=("E143250", "E147007"), representation="driven_stage", **extra):
    fields = dict(
        toggle_external_boundaries=False,
        swmm_outfalls_to_keep=[],
        swmm_open_water_outfalls=list(listed),
        open_water_outfall_representation=representation,
        manhole_diameter=_DIAMETER,
        manhole_loss_coefficient=_LOSS,
        manhole_table=None,
        manhole_table_fill_with_constants=False,
        swmm_node_rim_from_dem=False,
        dem_rim_resolution=None,
        water_level_boundary_series_source=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _problem_text(problems) -> str:
    return " | ".join(p.message for p in problems)


def _rows(text: str, section: str) -> list[list[str]]:
    """The data rows of ``[section]`` of a SWMM model text, split on blanks, comments dropped."""
    out, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            inside = stripped.upper() == f"[{section}]"
        elif inside and stripped and not stripped.startswith(";"):
            out.append(stripped.split())
    return out


def test_the_two_fields_default_to_an_empty_list_and_the_driven_stage_and_are_experiment_fields():
    """Property: the list is empty and the representation is the driven stage unless the analysis says otherwise.

    Class: the declared defaults and the closed set of the representation, read from the model, and the taxonomy
    entry of each field. Why: an analysis that never heard of the fields must run as it always ran, and the
    developer ruled that the driven stage is the default of a non-empty list. What kills it: a default list
    that is not empty, the other default representation, an open set, or a field with no taxonomy entry (the
    totality test of the taxonomy would then fail). Second correct implementation: ``Literal`` replaced by an
    enum, if the assertions were widened; they read the default and the accepted values.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_taxonomy import all_field_bucket

    fields = analysis_config.model_fields
    assert fields["swmm_open_water_outfalls"].default_factory() == []
    assert fields["open_water_outfall_representation"].default == "driven_stage"
    options = fields["open_water_outfall_representation"].json_schema_extra["options"]
    assert set(options) == {"driven_stage", "crown_rim"}
    assert all_field_bucket("swmm_open_water_outfalls") == "experiment"
    assert all_field_bucket("open_water_outfall_representation") == "experiment"


def test_a_valid_list_has_no_problem_under_either_representation(tmp_path):
    """Property (control): the Norfolk pair is accepted under both representations, with or without a record.

    Class: both outfalls listed, each representation, the water-level record naming both outfalls (as the
    Norfolk bundles do). Why: every refusal below is only informative against an input the same rule accepts,
    and the rule must allow a listed outfall to be named by a record. What kills it: refusing a listed outfall
    because a record names it, or refusing the crown_rim sizing of a valid conduit. Second correct
    implementation: judging the representations in another order; the assertions read the problem lists.
    """
    from hhemt.swmm_utils import open_water_outfall_problems

    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    for representation in ("driven_stage", "crown_rim"):
        assert open_water_outfall_problems(cfg_system, _cfg_analysis_promoted(representation=representation)) == []


@pytest.mark.parametrize(
    ("listed", "expected"),
    [
        (["E143250", "e143250"], "listed more than once"),
        (["E143250", "NOPE"], "'NOPE'"),
    ],
    ids=["a name listed twice after the case fold", "a name that is no [OUTFALLS] row"],
)
def test_a_repeated_or_unknown_outfall_is_refused_naming_the_field_and_the_offender(tmp_path, listed, expected):
    """Property: a repeated name and a name outside ``[OUTFALLS]`` each refuse, naming the offender.

    Class: ``E143250`` and its lower-case spelling, and a junction-like name that is no outfall, each beside a
    valid name, against the valid name alone. Why: SWMM folds case, so a repeat is the same node, and an
    unknown name would otherwise leave an outfall untreated while the configuration says it is listed. What
    kills it: comparing names without the fold, or dropping unknown names. Second correct implementation: raising at the
    first offender; the assertions read the field and the text.
    """
    from hhemt.swmm_utils import open_water_outfall_problems, require_valid_open_water_outfalls

    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    problems = open_water_outfall_problems(cfg_system, _cfg_analysis_promoted(listed=listed))
    assert [p.field for p in problems] == ["analysis.swmm_open_water_outfalls"]
    assert expected in problems[0].message
    with pytest.raises(ConfigurationError) as excinfo:
        require_valid_open_water_outfalls(cfg_system, _cfg_analysis_promoted(listed=listed))
    assert excinfo.value.field == "swmm_open_water_outfalls" and expected in str(excinfo.value)
    assert open_water_outfall_problems(cfg_system, _cfg_analysis_promoted(listed=listed[:1])) == []


@pytest.mark.parametrize(
    ("where", "leaf"),
    [("couple", "swmm_outfalls_to_couple"), ("keep", "swmm_outfalls_to_keep")],
    ids=["also coupled", "also kept"],
)
def test_a_listed_outfall_has_one_disposition(tmp_path, where, leaf):
    """Property: a listed outfall may not also be in ``swmm_outfalls_to_couple`` or ``swmm_outfalls_to_keep``.

    Class: the first outfall listed in each of the two other places, against the second outfall alone listed.
    Why: one disposition per outfall, so that no two of them can disagree about what the node becomes. What
    kills it: checking one of the two places. Second correct implementation: moving the rule into the
    water-level decision pass; the assertions read the problem list of this function.
    """
    from hhemt.swmm_utils import open_water_outfall_problems

    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    cfg_analysis = _cfg_analysis_promoted()
    if where == "couple":
        cfg_system.swmm_outfalls_to_couple = ["E143250"]
    else:
        cfg_analysis.swmm_outfalls_to_keep = ["E143250"]
    problems = open_water_outfall_problems(cfg_system, cfg_analysis)
    assert len(problems) == 1 and leaf in problems[0].message and "E143250" in problems[0].message
    assert "E147007" not in problems[0].message.split("also in")[0]
    assert open_water_outfall_problems(cfg_system, _cfg_analysis_promoted(listed=["E147007"])) == []


@pytest.mark.parametrize("representation", ["driven_stage", "crown_rim"])
def test_a_listed_outfall_may_have_no_row_in_the_manhole_table(tmp_path, representation):
    """Property: a user manhole-table row for a listed outfall is refused under both representations.

    Class: a table naming ``J9`` and ``E143250``, against a table naming ``J9`` alone, under each representation.
    Why: a driven outfall is not a coupling node and has no exchange slot, and a crown_rim node's row is the
    toolkit's, written from the conduit; a user row would either be ignored or collide. What kills it: judging
    only one representation, or ignoring the table. Second correct implementation: refusing in the table's own
    rule; the assertions read this function's problem for the table field.
    """
    from hhemt.swmm_utils import open_water_outfall_problems

    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    table = tmp_path / "table.csv"
    table.write_text("node_id,manhole_diameter_m,manhole_loss_coefficient\nJ9,0.6,0.35\nE143250,0.8,0.25\n")
    problems = open_water_outfall_problems(
        cfg_system, _cfg_analysis_promoted(representation=representation, manhole_table=table)
    )
    assert [p.field for p in problems] == ["analysis.manhole_table"]
    assert "E143250" in problems[0].message and "E147007" not in problems[0].message
    table.write_text("node_id,manhole_diameter_m,manhole_loss_coefficient\nJ9,0.6,0.35\n")
    assert (
        open_water_outfall_problems(
            cfg_system, _cfg_analysis_promoted(representation=representation, manhole_table=table)
        )
        == []
    )


def test_the_norfolk_conduit_table_gives_the_area_equivalent_diameter_and_the_crown(tmp_path):
    """Property: the diameter is ``sqrt(4 A / pi)`` over every barrel's area and the crown is the pipe's full height.

    Class: the two Norfolk conduits (twin ``RECT_CLOSED`` barrels, 1.524 by 1.676 m and 1.676 by 1.676 m, zero
    offsets), whose areas are 5.108 and 5.618 m2. Why: the developer ruled the area-equivalent rule (gate G22
    Q2), and the two pipes differ so that a reader that takes the other outfall's conduit, or one barrel, or the
    perimeter-equivalent diameter (4.074 and 4.268 m), fails. What kills it: ignoring the barrel count, sizing a
    rectangle as a circle of its height, or reading the other conduit. Second correct implementation: computing
    the area in another unit; the assertions read meters.
    """
    from hhemt.swmm_utils import crown_rim_sizing

    sized, reasons = crown_rim_sizing(
        _cfg_system_promoted(_model_promoted(tmp_path), tmp_path), _cfg_analysis_promoted(representation="crown_rim")
    )
    assert reasons == [] and list(sized) == ["E143250", "E147007"]
    first, second = sized["E143250"], sized["E147007"]
    assert (first.conduit, second.conduit) == ("1444", "1445")
    assert (first.barrels, second.barrels) == (2.0, 2.0)
    assert first.area_m2 == pytest.approx(1.524 * 1.676 * 2) and second.area_m2 == pytest.approx(1.676 * 1.676 * 2)
    assert first.diameter_m == pytest.approx(math.sqrt(4 * 1.524 * 1.676 * 2 / math.pi))
    assert second.diameter_m == pytest.approx(math.sqrt(4 * 1.676 * 1.676 * 2 / math.pi))
    assert round(first.diameter_m, 3) == 2.550 and round(second.diameter_m, 3) == 2.675
    assert first.crown_m == pytest.approx(1.524) and second.crown_m == pytest.approx(1.676)
    assert first.loss_coefficient == second.loss_coefficient == _LOSS
    assert first.shape == "RECT_CLOSED"


def test_a_us_unit_model_gives_its_diameter_in_meters(tmp_path):
    """Property: a CFS model holds feet, and the diameter and crown come out in meters.

    Class: a CFS model with one circular conduit of 3 ft and two barrels, against the same dimensions in a CMS
    model. Why: the manhole diameter the solver reads is in meters whatever the model's unit. What kills it:
    returning the model's unit. Second correct implementation: converting the area instead of the lengths; the
    assertions read the meter values.
    """
    from hhemt.swmm_utils import crown_rim_sizing

    text = (
        "[OPTIONS]\nFLOW_UNITS {units}\n\n[JUNCTIONS]\nJ1 0 3 0 0 0\n\n[OUTFALLS]\nO1 0 FREE NO\n\n"
        "[CONDUITS]\nC1 J1 O1 10 0.013 0 0 0 0\n\n[XSECTIONS]\nC1 CIRCULAR 3 0 0 0 2\n"
    )
    for units, scale in (("CFS", 0.3048), ("CMS", 1.0)):
        model = _model_promoted(tmp_path, text.format(units=units), name=f"{units}.inp")
        sized, reasons = crown_rim_sizing(
            _cfg_system_promoted(model, tmp_path), _cfg_analysis_promoted(listed=["O1"], representation="crown_rim")
        )
        assert reasons == []
        # calc_area rounds a circle's area to 2 decimals (3.1415 pi), so the diameter is checked to that rounding.
        assert sized["O1"].diameter_m == pytest.approx(scale * 3 * math.sqrt(2), rel=2e-3)
        assert sized["O1"].crown_m == pytest.approx(scale * 3)


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda t: t.replace("1445  E147008 E147007", "1445  E147008 E143250"), "has no conduit whose outlet node"),
        (
            lambda t: t.replace("1446  J9 E143254  10", "1446  J9 E143250  10"),
            "has 2 conduits whose outlet node is E143250 (1444, 1446)",
        ),
        (
            lambda t: t.replace("1444  RECT_CLOSED  1.524  1.676  0  0  2", "1444  CUSTOM  1.524  1.676  0  0  2"),
            "cannot size",
        ),
    ],
    ids=["no inlet conduit", "two inlet conduits", "a cross-section the toolkit cannot size"],
)
def test_crown_rim_refuses_an_outfall_with_no_single_sizable_outlet_conduit(tmp_path, edit, expected):
    """Property: ``crown_rim`` needs exactly one conduit that ends at the outfall, with a flow area it can size.

    Class: an outfall nothing ends at, an outfall two conduits end at, and an outfall whose conduit is a custom
    shape, each against the unedited model. Why: the rule reads one conduit, and a wrong guess of the pipe is a
    wrong manhole diameter with no message. What kills it: taking the first conduit, summing the conduits, or
    reading a missing area as zero. Second correct implementation: refusing at sizing time instead of in the
    problem list; the assertions read the problem under the representation field, and the unedited model passes.
    """
    from hhemt.swmm_utils import open_water_outfall_problems

    cfg_analysis = _cfg_analysis_promoted(representation="crown_rim")
    problems = open_water_outfall_problems(
        _cfg_system_promoted(_model_promoted(tmp_path, edit(_MODEL_promoted)), tmp_path), cfg_analysis
    )
    assert [p.field for p in problems] == ["analysis.open_water_outfall_representation"], problems
    assert expected in problems[0].message
    assert (
        open_water_outfall_problems(
            _cfg_system_promoted(_model_promoted(tmp_path, name="ok.inp"), tmp_path), cfg_analysis
        )
        == []
    )
    # The conduit rule belongs to crown_rim: the same model is valid for the driven stage.
    assert (
        open_water_outfall_problems(
            _cfg_system_promoted(_model_promoted(tmp_path, edit(_MODEL_promoted), name="d.inp"), tmp_path),
            _cfg_analysis_promoted(),
        )
        == []
    )


def test_with_the_list_empty_the_converted_set_and_every_role_text_are_what_they_were(tmp_path, monkeypatch):
    """Property: an empty list, and an analysis with no such field, change no byte of any derived model.

    Class: the three role texts and the converted set of an analysis with a water-level record, derived with the
    fields absent, with the list empty under each representation, and with the stand-in carrying neither field.
    Why: with the list empty every byte the toolkit writes is today's, so every existing arm keeps its result.
    What kills it: a conversion or a rewrite that runs for an empty list. Second correct implementation:
    caching the derivation differently; the assertions read the texts.
    """
    from hhemt import water_level_boundary
    from hhemt.swmm_utils import converted_outfalls, swmm_role_text

    monkeypatch.setattr(water_level_boundary, "boundary_records_for", lambda *a, **k: ((_WATER_LEVEL_RECORD,), ()))
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    bare = SimpleNamespace(
        toggle_external_boundaries=False,
        swmm_outfalls_to_keep=[],
        dem_rim_resolution=None,
        swmm_node_rim_from_dem=False,
    )
    baseline_converted = converted_outfalls(cfg_system, bare)
    assert baseline_converted == ("E143250", "E147007")  # the record converts both outfalls, as it always did
    for role in ("full", "hydraulics", "hydrology"):
        baseline = swmm_role_text(cfg_system, role, bare, externalize=False)
        for representation in ("driven_stage", "crown_rim"):
            empty = _cfg_analysis_promoted(listed=[], representation=representation)
            assert converted_outfalls(cfg_system, empty) == baseline_converted
            assert swmm_role_text(cfg_system, role, empty, externalize=False) == baseline, (role, representation)


def test_under_driven_stage_a_water_level_record_does_not_convert_the_outfall_and_the_coupled_model_holds_it_fixed(
    tmp_path, monkeypatch
):
    """Property: a listed outfall stays an outfall, written FIXED at its invert in the coupled model only.

    Class: the Norfolk pair listed under ``driven_stage`` while a record names both, against the same record with
    the list empty. Why: the record would convert both outfalls to coupled junctions, which is the earlier
    approach, and the driven stage needs the node to stay an outfall whose stage TRITON sets; the FIXED row's
    stage is the invert, which the solver also uses for a dry cell. What kills it: converting a matched
    outfall that is listed, rewriting the full role or the hydrology role, or keeping the TIMESERIES row in the
    coupled model. Second correct implementation: writing the stage with other column spacing; the assertions
    read the tokens.
    """
    from hhemt import water_level_boundary
    from hhemt.swmm_utils import converted_outfalls, swmm_role_text

    monkeypatch.setattr(water_level_boundary, "boundary_records_for", lambda *a, **k: ((_WATER_LEVEL_RECORD,), ()))
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    driven = _cfg_analysis_promoted(representation="driven_stage")
    assert converted_outfalls(cfg_system, driven) == ()
    hydraulics = swmm_role_text(cfg_system, "hydraulics", driven, externalize=False)
    assert _rows(hydraulics, "OUTFALLS") == [
        ["E143250", "-1.079", "FIXED", "-1.079", "NO"],
        ["E147007", "-1.079", "FIXED", "-1.079", "NO"],
    ]
    assert {r[0] for r in _rows(hydraulics, "JUNCTIONS")} == {
        "E143254",
        "E147008",
        "J9",
    }  # no outfall became a junction
    assert not any(r[0].startswith("hhemt_dummy") for r in _rows(hydraulics, "OUTFALLS"))  # an outfall remains
    full = swmm_role_text(cfg_system, "full", driven, externalize=False)
    assert ["E143250", "-1.079", "TIMESERIES", "water_level", "NO"] in _rows(full, "OUTFALLS")
    hydrology = swmm_role_text(cfg_system, "hydrology", driven, externalize=False)
    assert ["E143250", "-1.079", "TIMESERIES", "water_level", "NO"] in _rows(hydrology, "OUTFALLS")
    # Control that varies the property (the list), not the call: with the list empty the record converts both.
    assert converted_outfalls(cfg_system, _cfg_analysis_promoted(listed=[])) == ("E143250", "E147007")


def test_the_fixed_rewrite_keeps_the_gate_and_route_of_each_kind_of_outfall_and_every_other_line():
    """Property: a FREE, NORMAL, TIDAL and TIMESERIES outfall each become FIXED at the invert, keeping their tail.

    Class: one outfall of each kind with a gate flag and a route-to, CRLF line endings, a comment line and an
    unlisted outfall. Why: SWMM reads the gate flag and the route-to after the stage, and a FREE or NORMAL row
    has no stage token to replace. What kills it: dropping the tail, replacing the wrong token, or touching a
    line that is not listed. Second correct implementation: another column width; the assertions read tokens
    and compare the untouched lines byte for byte.
    """
    from hhemt.swmm_utils import fixed_outfall_text

    text = (
        "[OUTFALLS]\r\n;; header\r\nA 1.5 FREE YES SUB1\r\nB 2 NORMAL NO\r\nC 3 TIDAL tide NO\r\n"
        "D 4 TIMESERIES ts YES\r\nE 5 FREE NO\r\n[JUNCTIONS]\r\nJ1 0 1 0 0 0\r\n"
    )
    out = fixed_outfall_text(text, ("A", "b", "C", "D"))
    assert _rows(out, "OUTFALLS") == [
        ["A", "1.5", "FIXED", "1.5", "YES", "SUB1"],
        ["B", "2", "FIXED", "2", "NO"],
        ["C", "3", "FIXED", "3", "NO"],
        ["D", "4", "FIXED", "4", "YES"],
        ["E", "5", "FREE", "NO"],
    ]
    lines = out.split("\r\n")
    assert lines[0] == "[OUTFALLS]" and lines[1] == ";; header" and lines[6] == "E 5 FREE NO"
    assert out.count("\r\n") == text.count("\r\n") and out.count("\n") == out.count("\r\n")
    with pytest.raises(ConfigurationError, match="NOPE"):
        fixed_outfall_text(text, ("A", "NOPE"))


def test_under_crown_rim_every_listed_outfall_is_converted_whether_or_not_a_record_names_it(tmp_path, monkeypatch):
    """Property: a ``crown_rim`` outfall becomes a coupled junction with SWMM depth 0, record or no record.

    Class: the Norfolk pair listed under ``crown_rim`` with a record naming one of them only, against the list
    empty. Why: the comparison representation converts every listed outfall and writes its depth as 0, which
    SWMM raises to the outlet conduit's crown (the lowest rim it allows). What kills it: converting only the
    matched outfall, or writing the DEM depth. Second correct implementation: a different order of the
    converted names; the assertions compare sets and rows.
    """
    from hhemt import water_level_boundary
    from hhemt.swmm_utils import converted_outfalls, swmm_role_text

    one = BoundaryRecord(0, "water_level", None, ("E143250",), ((0.0, 0.0), (1.0, 0.0)))
    monkeypatch.setattr(water_level_boundary, "boundary_records_for", lambda *a, **k: ((one,), ()))
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    crown = _cfg_analysis_promoted(representation="crown_rim")
    assert sorted(converted_outfalls(cfg_system, crown)) == ["E143250", "E147007"]
    hydraulics = swmm_role_text(cfg_system, "hydraulics", crown, externalize=False)
    depths = {r[0]: (r[1], r[2]) for r in _rows(hydraulics, "JUNCTIONS")}
    assert depths["E143250"] == ("-1.079", "0") and depths["E147007"] == ("-1.079", "0")
    assert not [r for r in _rows(hydraulics, "OUTFALLS") if r[0] in ("E143250", "E147007")]
    # Control: with the list empty and the record naming one outfall, only that outfall converts.
    assert converted_outfalls(cfg_system, _cfg_analysis_promoted(listed=[])) == ("E143250",)


def _write_dem(tmp_path: Path, ground: float) -> Path:
    """A flat 10 by 10 DEM of 10 m cells at ``ground``: DEM_fullres, and the system directory's processed DEM."""
    xs = np.arange(10) * 10.0 + 5.0
    ys = (np.arange(10) * 10.0 + 5.0)[::-1]
    dem = xr.DataArray(np.full((10, 10), ground, dtype="float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
    dem = dem.rio.write_crs("EPSG:32618")
    dem.rio.to_raster(tmp_path / "dem.tif")
    (tmp_path / "system").mkdir(exist_ok=True)
    shutil.copy(tmp_path / "dem.tif", tmp_path / "system" / "elevation_10.00m.dem")
    return tmp_path / "dem.tif"


def test_under_crown_rim_the_dem_check_is_skipped_for_the_listed_node_and_only_for_it(tmp_path):
    """Property: V23 is not asked of a ``crown_rim`` node, and is still asked of every other converted node.

    Class: ground at -1.5 m everywhere (below both outfalls' inverts of -1.079), the Norfolk pair listed under
    ``crown_rim`` with the DEM chosen, and a third outfall converted through ``swmm_outfalls_to_couple`` on the
    same ground; the control moves the first outfall from the list to ``swmm_outfalls_to_couple``. Why: the
    crown_rim node's depth is 0, raised by SWMM to the crown, so no DEM depth enters it, and the refusal that
    stopped the earlier stage S must not fire for it; every other converted node keeps its refusal. What kills it:
    skipping the check for all nodes, or still applying it to the listed node. Second correct implementation:
    returning the problem for the third outfall in another order; the assertions read the problem text.
    """
    from hhemt.swmm_utils import node_depth_problems

    _write_dem(tmp_path, -1.5)
    text = _MODEL_promoted.replace("E147007  -1.079  TIMESERIES water_level  NO", "E147007  -1.079  FREE  NO") + (
        "\n[OUTFALLS]\nE999999  -1.079  FREE  NO\n[COORDINATES]\nE999999 85 55\n"
    )
    model = _model_promoted(tmp_path, text)
    cfg_system = _cfg_system_promoted(model, tmp_path)
    cfg_system.swmm_outfalls_to_couple = ["E999999"]
    listed = _cfg_analysis_promoted(listed=["E143250"], representation="crown_rim", dem_rim_resolution="processed")
    problems, _provisional = node_depth_problems(cfg_system, listed)
    assert len(problems) == 1 and "E999999" in problems[0].message and "E143250" not in problems[0].message
    # Control, the property varied: the same node under swmm_outfalls_to_couple is refused by the same DEM.
    cfg_system.swmm_outfalls_to_couple = ["E999999", "E143250"]
    moved, _ = node_depth_problems(cfg_system, _cfg_analysis_promoted(listed=[], dem_rim_resolution="processed"))
    assert len(moved) == 1 and "E143250" in moved[0].message and "E999999" in moved[0].message


def test_an_open_water_outfall_is_a_decision_and_takes_no_dem_depth_in_the_boundary_preflight(tmp_path):
    """Property: the water-level preflight counts a listed outfall as decided and does not ask the DEM about it.

    Class: the two TIMESERIES outfalls listed under each representation with no record and no keep list, the DEM
    unchosen, against the list empty. Why: with no other disposition the outfalls would be reported as neither
    converted nor kept, and with ``dem_rim_resolution`` unset a crown_rim node would be asked for a DEM it never
    reads. What kills it: leaving the V9 decision pass unaware of the list, or counting a crown_rim node as
    DEM-converted. Second correct implementation: a separate pass for the list; the assertions read the problem
    fields.
    """
    from hhemt.water_level_boundary import preflight_problems

    _write_dem(tmp_path, 5.0)
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    cfg_system.crs = SimpleNamespace(horizontal_epsg=32618)
    for representation in ("driven_stage", "crown_rim"):
        report = preflight_problems(cfg_system, _cfg_analysis_promoted(representation=representation))
        assert [p.message for p in report.problems] == [], representation
    empty = preflight_problems(cfg_system, _cfg_analysis_promoted(listed=[]))
    assert any("no decision" in p.message for p in empty.problems)  # control: the same outfalls, undecided


def test_run_entry_reports_each_rule_and_places_a_driven_outfall_on_the_grid(tmp_path):
    """Property: the preflight pass reports the list's rules, a missing coordinate and a node outside the grid.

    Class: a model with the outfalls listed and a coordinate inside the grid, a listed outfall whose coordinate
    lies outside the 100 m grid, a listed outfall with no coordinate, and a valid list. Why: the solver reads
    the DEM cell that holds each driven outfall and refuses a node it cannot place, and run entry is before any
    job is queued. What kills it: asking only the model rules, or judging a crown_rim node by the grid rule of a
    driven one. Second correct implementation: judging the grid in the water-level pass; the assertions read
    the errors of the one result.
    """
    from hhemt.validation import ValidationResult, _validate_open_water_outfalls

    _write_dem(tmp_path, 5.0)
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    ok = ValidationResult()
    _validate_open_water_outfalls(cfg_system, _cfg_analysis_promoted(), ok)
    assert ok.errors == []
    outside = _model_promoted(tmp_path, _MODEL_promoted.replace("E147007 65 55", "E147007 5000 55"), name="outside.inp")
    result = ValidationResult()
    _validate_open_water_outfalls(_cfg_system_promoted(outside, tmp_path), _cfg_analysis_promoted(), result)
    assert [e.field for e in result.errors] == ["analysis.swmm_open_water_outfalls"]
    assert "E147007" in result.errors[0].message and "outside the TRITON grid" in result.errors[0].message
    unplaced = _model_promoted(tmp_path, _MODEL_promoted.replace("E147007 65 55\n", ""), name="unplaced.inp")
    result = ValidationResult()
    _validate_open_water_outfalls(_cfg_system_promoted(unplaced, tmp_path), _cfg_analysis_promoted(), result)
    assert "have no [COORDINATES] row" in result.errors[0].message and "E147007" in result.errors[0].message
    both = ValidationResult()
    _validate_open_water_outfalls(
        _cfg_system_promoted(outside, tmp_path), _cfg_analysis_promoted(listed=["E143250", "NOPE"]), both
    )
    assert "NOPE" in both.errors[0].message and len(both.errors) == 1  # a refused list is not also placed on the grid
    nothing = ValidationResult()
    _validate_open_water_outfalls(cfg_system, _cfg_analysis_promoted(listed=[]), nothing)
    assert nothing.errors == []


def test_a_crown_rim_outfall_under_legacy_coupling_is_refused_at_run_entry(tmp_path, monkeypatch):
    """Property: the legacy coupling mode, which couples FLOW-inflow nodes, refuses a ``crown_rim`` outfall.

    Class: the Norfolk pair listed under ``crown_rim`` with ``swmm_coupling_mode: legacy`` and no record, against
    ``all_junctions``. Why: a converted outfall that no mode couples would sit in the model as a dry pipe end,
    and the refusal names the mode. What kills it: applying the legacy refusal to matched outfalls only. Second
    correct implementation: refusing in the open-water pass; the assertions read the error of the V8 pass.
    """
    from hhemt.validation import ValidationResult, _validate_converted_outfalls_are_coupled

    _write_dem(tmp_path, 5.0)
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    cfg_analysis = _cfg_analysis_promoted(representation="crown_rim")
    cfg_system.swmm_coupling_mode = "legacy"
    from hhemt.swmm_utils import converted_outfalls

    result = ValidationResult()
    _validate_converted_outfalls_are_coupled(
        cfg_system, cfg_analysis, converted_outfalls(cfg_system, cfg_analysis), result
    )
    assert any("not coupling candidates" in e.message and "E143250" in e.message for e in result.errors), result.errors
    ok = ValidationResult()
    cfg_system.swmm_coupling_mode = "all_junctions"
    _validate_converted_outfalls_are_coupled(cfg_system, cfg_analysis, converted_outfalls(cfg_system, cfg_analysis), ok)
    assert not [e for e in ok.errors if "not coupling candidates" in e.message], ok.errors


_NODES = ["E143254", "E147008", "J9", "E143250"]


def _context(tmp_path: Path, crown=()):
    from hhemt.scenario import ManholeTableContext

    return ManholeTableContext(
        table=tmp_path / "t.csv", model="m.inp", diameter_constant=_DIAMETER, loss_constant=_LOSS, crown_rim_nodes=crown
    )


def test_the_table_rule_does_not_ask_for_a_crown_rim_row_and_refuses_one(tmp_path):
    """Property: with ``crown_rim_nodes`` set the missing-row rule skips them and a row naming one is refused.

    Class: a table covering the three ordinary nodes, against the same table plus a row for the crown_rim node,
    and against the context without ``crown_rim_nodes``. Why: the toolkit writes the crown_rim node's row, so the
    user is neither obliged nor allowed to. What kills it: asking for the row, or letting the row override the
    toolkit's. Second correct implementation: filtering the node list before the call; the assertions read the
    problems of the one function.
    """
    from hhemt.scenario import manhole_table_problems, read_manhole_table

    path = tmp_path / "t.csv"
    path.write_text(
        "node_id,manhole_diameter_m,manhole_loss_coefficient\nE143254,0.6,0.35\nE147008,0.8,0.25\nJ9,0.9,0.15\n"
    )
    rows = read_manhole_table(path)
    assert manhole_table_problems(rows, _NODES, False, _context(tmp_path, ("E143250",))) == []
    assert any(
        "no row" in p or "have no row" in p for p in manhole_table_problems(rows, _NODES, False, _context(tmp_path))
    )
    path.write_text(path.read_text() + "E143250,0.5,0.05\n")
    named = manhole_table_problems(read_manhole_table(path), _NODES, False, _context(tmp_path, ("E143250",)))
    assert len(named) == 1 and "E143250" in named[0] and "toolkit writes itself" in named[0]


def test_the_written_table_gives_a_fixed_node_its_sized_values_over_a_row_and_the_constants(tmp_path):
    """Property: ``fixed`` values take precedence over a row and the constants, in node order, from one float.

    Class: four nodes, one row, one fixed node whose name is spelled in another case, against none fixed. Why:
    the crown_rim node's diameter is the sized one whatever else says. What kills it: applying the constants
    to the fixed node, or writing it out of order. Second correct implementation: building the dict by the
    node's own spelling; the assertions read the whole text.
    """
    from hhemt.scenario import manhole_table_text, read_manhole_table

    path = tmp_path / "t.csv"
    path.write_text("node_id,manhole_diameter_m,manhole_loss_coefficient\nJ9,0.9,0.15\n")
    rows = read_manhole_table(path)
    text = manhole_table_text(rows, _NODES, _DIAMETER, _LOSS, fixed={"e143250": (2.55, 0.1)})
    assert text == "E143254 1.2 0.1\nE147008 1.2 0.1\nJ9 0.9 0.15\nE143250 2.55 0.1\n"
    assert manhole_table_text(rows, _NODES, _DIAMETER, _LOSS) == (
        "E143254 1.2 0.1\nE147008 1.2 0.1\nJ9 0.9 0.15\nE143250 1.2 0.1\n"
    )


def _generator(tmp_path: Path, cfg_analysis, cfg_system):
    from hhemt.scenario_inputs import ScenarioInputGenerator

    gen = ScenarioInputGenerator.__new__(ScenarioInputGenerator)
    gen.scenario = SimpleNamespace(scen_paths=SimpleNamespace(swmm_hydraulics_inp=tmp_path / "hydraulics.inp"))
    gen.cfg_analysis = cfg_analysis
    gen.system = SimpleNamespace(cfg_system=cfg_system)
    return gen


def test_preparation_writes_the_driven_outfall_list_in_model_order_and_refuses_a_coupled_one(tmp_path):
    """Property: ``driven_outfalls.txt`` lists the listed names in ``[OUTFALLS]`` order, and none is coupled.

    Class: the pair listed in reverse order, against a coupling set that also holds a listed outfall, against
    the list empty. Why: the solver's list file takes one name per line, a driven outfall has no exchange slot,
    and with the list empty no file is written. What kills it: writing in list order, writing with an empty
    list, or accepting a coupled driven outfall. Second correct implementation: another spelling of the names
    (the solver folds case); the assertions read the file and the error.
    """
    from hhemt.scenario import driven_outfalls_file_for

    inp = tmp_path / "hydraulics.inp"
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    gen = _generator(tmp_path, _cfg_analysis_promoted(listed=["E147007", "E143250"]), cfg_system)
    gen._write_open_water_outfalls(("E143254", "E147008", "J9"), inp)
    assert driven_outfalls_file_for(inp) == tmp_path / "driven_outfalls.txt"
    assert driven_outfalls_file_for(inp).read_text() == "E143250\nE147007\n"
    assert not (tmp_path / "dem_rim_exempt_nodes.txt").exists() and not (tmp_path / "crown_rim_outfalls.json").exists()
    with pytest.raises(ConfigurationError, match="E143250") as excinfo:
        gen._write_open_water_outfalls(("E143254", "e143250"), inp)
    assert excinfo.value.field == "swmm_open_water_outfalls" and "not a coupling node" in str(excinfo.value)
    driven_outfalls_file_for(inp).unlink()
    _generator(tmp_path, _cfg_analysis_promoted(listed=[]), cfg_system)._write_open_water_outfalls(("J9",), inp)
    assert not driven_outfalls_file_for(inp).exists()


def test_preparation_writes_the_crown_rim_files_and_the_manhole_rows_the_toolkit_sizes(tmp_path):
    """Property: under ``crown_rim`` preparation writes the manhole rows, the sizing record, and the exemption list
    only where the DEM-rim key is written.

    Class: the pair listed under ``crown_rim`` with the processed DEM and ``swmm_node_rim_from_dem`` on, the same
    with it off, and the same with the full DEM (where the toolkit writes the depths and no TRITON key). Why: the
    solver refuses an exemption list while its DEM-rim route is off, so the file is written exactly when the
    key is; the manhole file carries the sized diameter and the loss at ``manhole_loss_coefficient`` for the
    listed nodes and the constants for the others, in coupling order. What kills it: writing the exemption
    list whenever the field is set, skipping the toolkit's rows when the analysis has no table, or sizing from
    the wrong conduit. Second correct implementation: writing the record before the lists; the assertions read
    the files.
    """
    import json

    from hhemt.scenario import dem_rim_exempt_nodes_file_for, manhole_table_file_for

    inp = tmp_path / "hydraulics.inp"
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    nodes = ("E143254", "E147008", "J9", "E143250", "E147007")
    for rim, resolution, exempt_written in (
        (True, "processed", True),
        (False, None, False),
        (True, "full", False),
    ):
        cfg_analysis = _cfg_analysis_promoted(
            representation="crown_rim", swmm_node_rim_from_dem=rim, dem_rim_resolution=resolution
        )
        gen = _generator(tmp_path, cfg_analysis, cfg_system)
        for name in ("dem_rim_exempt_nodes.txt", "manhole_table.txt", "crown_rim_outfalls.json"):
            (tmp_path / name).unlink(missing_ok=True)
        gen._write_open_water_outfalls(nodes, inp)
        gen._write_manhole_table(nodes, inp)
        assert dem_rim_exempt_nodes_file_for(inp).exists() is exempt_written, (rim, resolution)
        if exempt_written:
            assert dem_rim_exempt_nodes_file_for(inp).read_text() == "E143250\nE147007\n"
        table = manhole_table_file_for(inp).read_text().splitlines()
        assert table[:3] == ["E143254 1.2 0.1", "E147008 1.2 0.1", "J9 1.2 0.1"]
        first, second = (line.split() for line in table[3:])
        assert first[0] == "E143250" and float(first[1]) == pytest.approx(2.5503486, rel=1e-6) and first[2] == "0.1"
        assert second[0] == "E147007" and float(second[1]) == pytest.approx(2.6745090, rel=1e-6) and second[2] == "0.1"
    record = json.loads((tmp_path / "crown_rim_outfalls.json").read_text())
    assert [o["outlet_conduit"] for o in record["outfalls"]] == ["1444", "1445"]
    assert [o["barrels"] for o in record["outfalls"]] == [2.0, 2.0]
    assert [round(o["crown_m_above_invert"], 3) for o in record["outfalls"]] == [1.524, 1.676]


def test_preparation_refuses_a_crown_rim_outfall_the_scenario_does_not_couple_and_judges_a_user_table(tmp_path):
    """Property: a ``crown_rim`` outfall must be a coupling node, and a user table keeps its rule for the others.

    Class: the coupling set without one listed outfall, a user table that covers the ordinary nodes and no listed
    one (accepted with the toggle off), and one that also names a listed node (refused). Why: a converted
    outfall that is not coupled exchanges nothing; the table rule is the same function as run entry's and must
    neither demand nor accept a row for a toolkit-sized node. What kills it: skipping the coupling check, or
    asking the user table for the sized nodes. Second correct implementation: judging the coupling set in the
    table writer; the assertions read the exceptions and the file.
    """
    inp = tmp_path / "hydraulics.inp"
    cfg_system = _cfg_system_promoted(_model_promoted(tmp_path), tmp_path)
    cfg_analysis = _cfg_analysis_promoted(representation="crown_rim")
    gen = _generator(tmp_path, cfg_analysis, cfg_system)
    with pytest.raises(ConfigurationError, match="E147007") as excinfo:
        gen._write_open_water_outfalls(("E143254", "E147008", "J9", "E143250"), inp)
    assert excinfo.value.field == "swmm_open_water_outfalls" and "exchanges no water" in str(excinfo.value)
    nodes = ("E143254", "E147008", "J9", "E143250", "E147007")
    table = tmp_path / "t.csv"
    table.write_text(
        "node_id,manhole_diameter_m,manhole_loss_coefficient\nE143254,0.6,0.35\nE147008,0.8,0.25\nJ9,0.9,0.15\n"
    )
    with_table = _cfg_analysis_promoted(representation="crown_rim", manhole_table=table)
    _generator(tmp_path, with_table, cfg_system)._write_manhole_table(nodes, inp)
    assert (tmp_path / "manhole_table.txt").read_text().splitlines()[:3] == [
        "E143254 0.6 0.35",
        "E147008 0.8 0.25",
        "J9 0.9 0.15",
    ]
    table.write_text(table.read_text() + "E147007,0.5,0.05\n")
    with pytest.raises(ConfigurationError, match="toolkit writes itself"):
        _generator(tmp_path, with_table, cfg_system)._write_manhole_table(nodes, inp)


_READS_NODES = 'argsd("swmm_coupling_nodes", argmap, "");\n'


_READS_DRIVEN = _READS_NODES + 'argsd("swmm_driven_outfalls", argmap, "");\n'


_READS_TABLE = _READS_NODES + 'argsd("swmm_manhole_table", argmap, "");\n'


_READS_RIM = _READS_TABLE + 'argsd("swmm_dem_rim_exempt_nodes", argmap, "");\n'


class _AnyField:
    """A stand-in log whose every field reads False and accepts any value."""

    def __getattr__(self, name):
        return SimpleNamespace(get=lambda: False, set=lambda value: None)


def _clone(tmp_path: Path, text: str, name: str = "triton") -> Path:
    src = tmp_path / name / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "config_utils.h").write_text(text)
    return tmp_path / name


def _scenario(
    tmp_path, monkeypatch, *, clone, listed=(), representation="driven_stage", rim=False, environment="native"
):
    """A scenario whose coupled cfg generator runs on fakes; returns (scenario, cfg path)."""
    from hhemt import scenario as scenario_module
    from hhemt.scenario import TRITONSWMM_scenario

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "TRITONSWMM.cfg").write_text("sim_start_time=0\ndem_filename=dem.dem\n")
    extbc = tmp_path / "extbc_loc.csv"
    extbc.write_text("x,y\n")
    options = SimpleNamespace(
        Value=SimpleNamespace(
            START_DATE="01/01/2020", START_TIME="00:00:00", END_DATE="01/01/2020", END_TIME="01:00:00"
        )
    )
    monkeypatch.setattr(
        scenario_module.swmmio, "Model", lambda path: SimpleNamespace(inp=SimpleNamespace(options=options))
    )
    fields = dict(
        manhole_diameter=_DIAMETER,
        manhole_loss_coefficient=_LOSS,
        manhole_table=None,
        manhole_table_fill_with_constants=False,
        TRITON_raw_output_type="bin",
        hydraulic_timestep_s=1.0,
        TRITON_courant_number=0.5,
        TRITON_hextra_m=0.001,
        TRITON_fixed_timestep=False,
        TRITON_domain_decomposition="static",
        TRITON_domain_decomposition_check_factor=None,
        TRITON_progress_print_iterations=10000,
        TRITON_reporting_timestep_s=60,
        open_boundaries=1,
        swmm_snapshot_disable=False,
        swmm_snapshot_keep_all=False,
        swmm_node_rim_from_dem=rim,
        execution_environment=environment,
        one_coupled_inflow_node_per_dem_cell=False,
    )
    if listed:
        fields.update(swmm_open_water_outfalls=list(listed), open_water_outfall_representation=representation)
    cfg_analysis = SimpleNamespace(**fields)
    cfg_system = SimpleNamespace(
        toggle_use_constant_mannings=True,
        constant_mannings=0.05,
        TRITONSWMM_software_directory=clone,
        TRITONSWMM_branch_key=None,
    )
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen._analysis = SimpleNamespace(cfg_analysis=cfg_analysis, cfg_hpc_system=SimpleNamespace(gpu_aware_mpi=False))
    scen._system = SimpleNamespace(
        cfg_system=cfg_system,
        sys_paths=SimpleNamespace(dem_processed=tmp_path / "dem.dem", mannings_processed=tmp_path / "man.dem"),
        log=SimpleNamespace(triton_head_sha=SimpleNamespace(get=lambda: None)),
    )
    cfg = tmp_path / "sim" / "TRITONSWMM.cfg"
    scen.scen_paths = SimpleNamespace(
        swmm_hydraulics_inp=tmp_path / "hydraulics.inp",
        hyg_timeseries=tmp_path / "tseries.hyg",
        hyg_locs=tmp_path / "loc.txt",
        extbc_loc=extbc,
        triton_swmm_cfg=cfg,
    )
    scen.sim_id_str = "evt"
    scen.log = _AnyField()
    scen._swmm_inp_for_sim_duration = lambda: tmp_path / "hydraulics.inp"
    scen._n_hydrograph_sources = lambda: 0
    return scen, cfg


def _generate(tmp_path, monkeypatch, name, **kwargs) -> str:
    sub = tmp_path / name
    scen, cfg = _scenario(sub, monkeypatch, **kwargs)
    scen._generate_TRITON_SWMM_cfg()
    # The scenario's own directory is part of every path the cfg names: name it "<S>" so two scenarios compare.
    return cfg.read_text().replace(str(sub), "<S>")


def test_with_the_list_empty_the_cfg_is_the_cfg_of_an_analysis_that_never_had_the_field(tmp_path, monkeypatch):
    """Property: an empty list adds no cfg line, under either representation, with the DEM-rim key on or off.

    Class: the coupled cfg of a stand-in with neither field, and of one with the list empty under each
    representation, for a clone that reads none of the new keys. Why: with the list empty every byte hhemt writes
    is today's, and a clone that cannot read the keys must not be refused over them. What kills it: writing a
    key or running a guard for an empty list. Second correct implementation: writing a comment line; the
    assertions compare the whole text.
    """
    clone = _clone(tmp_path, _READS_NODES)
    baseline = _generate(tmp_path, monkeypatch, "bare", clone=clone)
    for representation in ("driven_stage", "crown_rim"):
        for rim in (False, True):
            reads = _clone(tmp_path, _READS_NODES + ('argsd("swmm_node_rim_from_dem"' if rim else ""), "rim")
            plain = _generate(tmp_path, monkeypatch, f"p_{representation}_{rim}", clone=reads, rim=rim)
            empty = _generate(
                tmp_path,
                monkeypatch,
                f"e_{representation}_{rim}",
                clone=reads,
                rim=rim,
                listed=[],
                representation=representation,
            )
            assert empty == plain, (representation, rim)
    assert "swmm_driven_outfalls" not in baseline and "swmm_dem_rim_exempt_nodes" not in baseline
    assert "swmm_manhole_table" not in baseline


def test_driven_stage_adds_exactly_one_cfg_line_and_refuses_a_build_that_would_ignore_it(tmp_path, monkeypatch):
    """Property: ``driven_stage`` appends ``swmm_driven_outfalls="{path}"`` and refuses a build that lacks the key.

    Class: a clone that reads the key, a clone that does not, and the container mode. Why: a build that ignores
    the key would run each listed outfall as the model's own outfall while the configuration names a driven
    stage, with no error. What kills it: not guarding, writing the line for a build that cannot read it, or
    admitting a container (no image carries the key). Second correct implementation: a different line position;
    the assertions read the whole text and the exception.
    """
    reads = _clone(tmp_path, _READS_DRIVEN, "reads")
    baseline = _generate(tmp_path, monkeypatch, "base", clone=reads)
    driven = _generate(tmp_path, monkeypatch, "driven", clone=reads, listed=["E143250", "E147007"])
    assert driven == baseline + 'swmm_driven_outfalls="<S>/driven_outfalls.txt"\n'
    assert "swmm_manhole_table" not in driven and "swmm_dem_rim_exempt_nodes" not in driven
    lacks = _clone(tmp_path, _READS_NODES, "lacks")
    with pytest.raises(ConfigurationError) as excinfo:
        _generate(tmp_path, monkeypatch, "lacks", clone=lacks, listed=["E143250"])
    assert excinfo.value.field == "swmm_open_water_outfalls" and 'argsd("swmm_driven_outfalls"' in str(excinfo.value)
    with pytest.raises(ConfigurationError, match="container"):
        _generate(tmp_path, monkeypatch, "ctr", clone=reads, listed=["E143250"], environment="container")


def test_crown_rim_adds_the_table_line_and_the_exemption_line_only_when_the_dem_rim_key_is_written(
    tmp_path, monkeypatch
):
    """Property: ``crown_rim`` writes the manhole-table line always, and the exemption line only with the rim key.

    Class: a clone that reads both keys, the list under ``crown_rim`` with ``swmm_node_rim_from_dem`` off and on,
    the clone that reads the table key only (refused when the exemption is needed, accepted when it is not), and
    one that reads neither. Why: the toolkit writes a table row for every listed node, so the table key is
    always needed; the exemption key is needed only where the solver's DEM-rim route is on, and a clone that
    cannot read it would override the listed rims from the DEM. What kills it: guarding the exemption key when
    the rim route is off, or not guarding the table key when the analysis sets no table of its own. Second
    correct implementation: one combined guard; the assertions read the lines and the exceptions.
    """
    reads = _clone(tmp_path, _READS_RIM + 'argsd("swmm_node_rim_from_dem"\n', "reads")
    off = _generate(tmp_path, monkeypatch, "off", clone=reads, listed=["E143250"], representation="crown_rim")
    assert 'swmm_manhole_table="<S>/manhole_table.txt"\n' in off
    assert "swmm_dem_rim_exempt_nodes" not in off and "swmm_driven_outfalls" not in off
    on = _generate(tmp_path, monkeypatch, "on", clone=reads, listed=["E143250"], representation="crown_rim", rim=True)
    assert on.endswith(
        'swmm_manhole_table="<S>/manhole_table.txt"\nswmm_dem_rim_exempt_nodes="<S>/dem_rim_exempt_nodes.txt"\n'
    )
    table_only = _clone(tmp_path, _READS_TABLE + 'argsd("swmm_node_rim_from_dem"\n', "table_only")
    _generate(tmp_path, monkeypatch, "ok", clone=table_only, listed=["E143250"], representation="crown_rim")
    with pytest.raises(ConfigurationError) as excinfo:
        _generate(
            tmp_path,
            monkeypatch,
            "no_exempt",
            clone=table_only,
            listed=["E143250"],
            representation="crown_rim",
            rim=True,
        )
    assert (
        'argsd("swmm_dem_rim_exempt_nodes"' in str(excinfo.value) and excinfo.value.field == "swmm_open_water_outfalls"
    )
    neither = _clone(tmp_path, _READS_NODES, "neither")
    with pytest.raises(ConfigurationError) as excinfo:
        _generate(tmp_path, monkeypatch, "no_table", clone=neither, listed=["E143250"], representation="crown_rim")
    assert excinfo.value.field == "swmm_open_water_outfalls" and 'argsd("swmm_manhole_table"' in str(excinfo.value)


def test_a_template_line_for_either_new_key_is_replaced_by_the_toolkit_line():
    """Property: an active template line for a key is removed and the toolkit's line is the only one.

    Class: a cfg text carrying each key as an active line, as a comment and indented. Why: a template value must
    never replace the list the toolkit wrote, and a comment is not an active line. What kills it: appending
    without removing, or removing a comment. Second correct implementation: replacing in place; the assertions
    read the set of lines.
    """
    from hhemt.scenario import set_swmm_dem_rim_exempt_nodes, set_swmm_driven_outfalls

    for setter, key in (
        (set_swmm_driven_outfalls, "swmm_driven_outfalls"),
        (set_swmm_dem_rim_exempt_nodes, "swmm_dem_rim_exempt_nodes"),
    ):
        text = f'a=1\n{key}="old"\n  {key} = "older"\n# {key}="note"\nb=2\n'
        assert setter(text, "new.txt").splitlines() == ["a=1", f'# {key}="note"', "b=2", f'{key}="new.txt"']
        assert setter("", "new.txt") == f'{key}="new.txt"\n'
