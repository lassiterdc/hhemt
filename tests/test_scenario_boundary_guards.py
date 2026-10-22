"""Scenario preparation's SWMM-input guards: the inputs TRITON-SWMM couples, and how they are read.

The problem this file pins. Scenario preparation decides which SWMM nodes exchange water
with the 2D surface by deleting ``[INFLOWS]`` lines, sizes pipes to rank them, reads the
SWMM ``.inp`` to find sections and node names, converts the hydrology model's runoff to the
units TRITON applies, and refuses inputs SWMM or TRITON-SWMM would mishandle. Each of those
steps had a defect that produced a wrong coupled set, a wrong value, or an unnamed crash;
the tests below each pin one property of the repaired behaviour.

Every test here generates inputs or calls a pure function; none compiles or runs a solver,
so the file runs on a laptop. The helpers under test are imported INSIDE each test so a run
against code that predates them collects every test and fails each one on its own.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _dem(path: Path, nx: int = 10, ny: int = 10, cell: float = 10.0) -> Path:
    """A flat GeoTIFF DEM with cell centres at 5, 15, ... in both axes."""
    import rioxarray  # noqa: F401 -- registers the .rio accessor
    import xarray as xr

    xs = np.arange(nx) * cell + cell / 2
    ys = (np.arange(ny) * cell + cell / 2)[::-1]
    da = xr.DataArray(np.zeros((ny, nx), dtype="float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.write_crs("EPSG:32618", inplace=True)
    da.rio.to_raster(path)
    return path


def _inp(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _row(**fields):
    """One link row as ``iterrows`` yields it from a one-row frame."""
    return next(pd.DataFrame([fields]).iterrows())[1]


def _tables(nodes: dict, links: dict):
    """A node table (``NodeType``, ``InvertElev``) and a link table, as the resolver reads them."""
    node_df = pd.DataFrame(
        {"NodeType": [v[0] for v in nodes.values()], "InvertElev": [v[1] for v in nodes.values()]},
        index=list(nodes),
    )
    cols = ["InletNode", "OutletNode", "LinkType", "Shape", "Geom1", "Geom2", "Geom3", "Geom4", "Barrels"]
    link_df = pd.DataFrame(
        [{c: spec.get(c, np.nan) for c in cols} for spec in links.values()],
        index=list(links),
        columns=cols,
    )
    return node_df, link_df


def _coupled(resolution, inflow) -> set:
    return set(inflow) - set(resolution.dropped)


_NODES_INP = """
[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
J1 1 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1

[INFLOWS]
{inflows}

[COORDINATES]
J1 5 5
O1 95 95
"""


# ---------------------------------------------------------------------------
# Fixes 1 and 2 -- the boundary-condition and DEM-extent guards
# ---------------------------------------------------------------------------


def test_an_unknown_boundary_side_raises_processing_error():
    """Property: an unknown boundary side raises ProcessingError naming it.

    Class: any ``BC_side`` outside the four named sides. Why: the branch used to print and
    fall through to an extent check that read an unassigned variable, raising an
    UnboundLocalError that named neither the side nor the accepted values. What kills it:
    restoring the print. Second correct implementation: raising before any coordinate is
    read, as a guard at function entry.
    """
    import xarray as xr

    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import find_closest_dem_coord

    dem = xr.DataArray(np.zeros((3, 3)), coords={"y": [25.0, 15.0, 5.0], "x": [5.0, 15.0, 25.0]}, dims=("y", "x"))
    with pytest.raises(ProcessingError, match="middle"):
        find_closest_dem_coord(10.0, 10.0, "middle", dem)


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_each_named_side_returns_a_coordinate_inside_the_dem(side):
    """Property, control: every named side returns a coordinate inside the DEM's extent.

    Class: the four named sides. Why: the two former ``sys.exit`` guards became raises, and a
    valid side must still never reach them. What kills it: a guard that compares against the
    wrong axis. Second correct implementation: clamping the coordinate instead of picking a
    DEM axis value; the assertion reads only containment.
    """
    import xarray as xr

    from hhemt.scenario import find_closest_dem_coord

    dem = xr.DataArray(np.zeros((3, 3)), coords={"y": [25.0, 15.0, 5.0], "x": [5.0, 15.0, 25.0]}, dims=("y", "x"))
    x, y = find_closest_dem_coord(12.0, 13.0, side, dem)
    assert 5.0 <= x <= 25.0 and 5.0 <= y <= 25.0


def test_out_of_bounds_nodes_are_named_in_one_warning(tmp_path):
    """Property: two nodes outside the DEM produce ONE warning naming both.

    Class: any model with nodes outside the DEM. Why: the list was reset inside the loop and
    never read, so out-of-domain nodes vanished with no trace. What kills it: moving the
    initialisation back into the loop, or gating the warning on ``verbose``. Second correct
    implementation: a warning per node; this assertion would then need its count relaxed, so
    it pins the one-warning form.
    """
    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell

    dem = _dem(tmp_path / "dem.tif")
    inp = _inp(
        tmp_path / "m.inp",
        """
[JUNCTIONS]
J1 1 2 0 0 0
FAR1 1 2 0 0 0
FAR2 1 2 0 0 0

[COORDINATES]
J1 5 5
FAR1 500 500
FAR2 -400 5
""",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        return_df_of_nodes_grouped_by_DEM_gridcell(inp, dem)
    hits = [str(w.message) for w in caught if "outside the DEM" in str(w.message)]
    assert len(hits) == 1
    assert "FAR1" in hits[0] and "FAR2" in hits[0]


# ---------------------------------------------------------------------------
# calc_area -- SWMM's own full areas (Fixes 3, 5, 6, 7 and the ellipse/arch rows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prop", "fields", "expected"),
    [
        ("EGG", dict(Shape="EGG", Geom1=2.0, Geom2=0.0, Geom3=0.0, Barrels=1), 0.5105 * 4),
        ("CIRCULAR control", dict(Shape="CIRCULAR", Geom1=2.0, Geom2=0.0, Geom3=0.0, Barrels=1), 3.14),
        ("RECT_OPEN", dict(Shape="RECT_OPEN", Geom1=2.0, Geom2=3.0, Geom3=2.0, Barrels=1), 6.0),
        ("RECT_TRIANGULAR", dict(Shape="RECT_TRIANGULAR", Geom1=2.0, Geom2=3.0, Geom3=1.0, Barrels=1), 4.5),
        ("RECT_ROUND", dict(Shape="RECT_ROUND", Geom1=2.0, Geom2=3.0, Geom3=2.0, Barrels=1), 5.376561799224367),
        (
            "RECT_ROUND clamped",
            dict(Shape="RECT_ROUND", Geom1=2.0, Geom2=3.0, Geom3=1.0, Barrels=1),
            5.034291735288518,
        ),
        ("RECT_CLOSED control", dict(Shape="RECT_CLOSED", Geom1=2.0, Geom2=3.0, Geom3=0.0, Barrels=1), 6.0),
        ("orifice", dict(Shape="RECT_CLOSED", Geom1=1.0, Geom2=1.0, Geom3=0.0, Barrels=np.nan), 1.0),
        ("lower-case", dict(Shape=" egg ", Geom1=2.0, Geom2=0.0, Geom3=0.0, Barrels=1), 2.042),
        ("HORIZ_ELLIPSE", dict(Shape="HORIZ_ELLIPSE", Geom1=1.0, Geom2=1.5, Geom3=0.0, Barrels=1), 1.2692),
        ("ARCH", dict(Shape="ARCH", Geom1=1.0, Geom2=2.0, Geom3=0.0, Barrels=1), 1.5758),
    ],
)
def test_calc_area_returns_swmm_full_area(prop, fields, expected):
    """Property: each shape returns SWMM 5.2.3's full flow area times barrels.

    Class: every recognised shape keyword in any letter case, with or without a barrel count.
    Why: the area ranks a cell's manholes under the one-node-per-cell option, so a wrong area
    couples the wrong manhole; the old arms returned a length for EGG, added Geom3 to the
    height for rectangles, NaN for an orifice, and a circle for an ellipse. What kills it:
    reverting any arm to its old formula, or matching the shape without stripping and
    upper-casing it. Second correct implementation: a table of (keyword -> coefficient
    function) looked up after the same normalisation; the assertion reads only the value.
    The CIRCULAR and RECT_CLOSED rows are controls whose value was already right.
    """
    from hhemt.scenario import calc_area

    assert calc_area(_row(**fields)) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "fields",
    [
        dict(Shape="ARCH", Geom1=5.0, Geom2=0.0, Geom3=0.0, Barrels=1),
        dict(Shape="HORIZ_ELLIPSE", Geom1=5.0, Geom2=0.0, Geom3=0.0, Barrels=1),
        dict(Shape="TRAPEZOIDAL", Geom1=1.0, Geom2=1.0, Geom3=1.0, Barrels=1),
        dict(Shape=np.nan, Geom1=np.nan, Geom2=np.nan, Geom3=np.nan, Barrels=np.nan),
    ],
)
def test_unsizable_shapes_return_none(fields):
    """Property: a size-coded arch or ellipse, an unhandled keyword and a missing shape return None.

    Class: every input whose area this function cannot compute. Why: a size-coded shape's
    Geom1 is a standard-size code, so reading it as a length gave 0 for an arch and 19.63 for
    an ellipse; None routes it to the named unsizable-shape error. What kills it: computing an
    area from the size code. Second correct implementation: raising a dedicated exception that
    the caller converts; the caller contract here is None.
    """
    from hhemt.scenario import calc_area

    assert calc_area(_row(**fields)) is None


# ---------------------------------------------------------------------------
# The coupled-node rule -- eligibility, and the optional one-node-per-cell ranking
# ---------------------------------------------------------------------------


def test_by_default_every_eligible_junction_of_a_cell_stays_coupled():
    """Property: with the option off, two eligible inflow junctions in one cell both stay coupled.

    Class: any cell holding more than one inflow junction. Why: the solver now couples any
    number of nodes per cell, and the default keeps the number of exchange points
    independent of the DEM resolution. What kills it: applying the ranking without the
    option. Second correct implementation: returning only the drop set; the assertion reads
    the coupled set.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "C1": dict(InletNode="J1", OutletNode="O1", LinkType="CONDUIT", Shape="CIRCULAR", Geom1=2.0, Barrels=1),
            "C2": dict(InletNode="J2", OutletNode="O1", LinkType="CONDUIT", Shape="CIRCULAR", Geom1=1.0, Barrels=1),
        },
    )
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"])
    assert _coupled(res, ["J1", "J2"]) == {"J1", "J2"}
    assert res.dropped == ()


def _chain_tables():
    return _tables(
        {"J1": ("JUNCTION", 3.0), "J2": ("JUNCTION", 2.0), "J3": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "C1": dict(InletNode="J1", OutletNode="J2", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=3),
            "C2": dict(InletNode="J2", OutletNode="J3", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=3),
            "C3": dict(InletNode="J3", OutletNode="O1", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=1),
        },
    )


def test_a_tie_at_the_largest_pipe_goes_to_the_cells_exit_node():
    """Property (option on): chain J1->J2->J3->O1 with C1 = C2 = 3 > C3 = 1 keeps J3.

    Class: a cell whose largest-pipe key is tied. Why: the tie rule chooses
    among ALL eligible junctions the one whose outgoing link leaves the cell;
    the old resolver kept none here. What kills it: breaking the tie among the tied nodes
    only (that keeps J2, the lower-invert of J1 and J2). Second correct implementation:
    computing exit nodes from the node-to-cell map rather than the cell's node list.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _chain_tables()
    res = resolve_cell_inflow_nodes(["J1", "J2", "J3"], nodes, links, ["J1", "J2", "J3"], one_per_cell=True)
    assert _coupled(res, ["J1", "J2", "J3"]) == {"J3"}


def test_the_largest_outgoing_pipe_wins():
    """Property, control (option on): areas 3.0 and 1.0 keep the 3.0 node.

    Class: an untied cell. What kills it: ranking by smallest area. Second correct
    implementation: sorting instead of a max.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "C1": dict(InletNode="J1", OutletNode="O1", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=3),
            "C2": dict(InletNode="J2", OutletNode="O1", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=1),
        },
    )
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"], one_per_cell=True)
    assert _coupled(res, ["J1", "J2"]) == {"J1"}


def test_a_junction_without_an_outgoing_pipe_ranks_by_its_largest_incoming_pipe():
    """Property (option on): a junction with only an incoming pipe is ranked by that pipe.

    Class: eligible junctions with no outgoing cross-section link. Why: under
    dynamic wave it drains through that pipe by reverse flow; the old resolver never made it
    a drop candidate, so the cell coupled both. What kills it: ranking such a node lowest
    regardless of its incoming pipe (keeps J1). Second correct implementation: computing
    both directions' keys at once and taking the outgoing one when present.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "X": ("JUNCTION", 5.0), "O1": ("OUTFALL", 0.0)},
        {
            "C1": dict(InletNode="J1", OutletNode="O1", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=1),
            "C2": dict(InletNode="X", OutletNode="J2", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=3),
        },
    )
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"], one_per_cell=True)
    assert _coupled(res, ["J1", "J2"]) == {"J2"}


def test_only_inflow_carrying_nodes_are_candidates():
    """Property (option on): J1 (inflow) -> J2 (no inflow) -> O1 with C1 < C2 couples J1.

    Class: a cell whose largest pipe starts at a node with no ``[INFLOWS]`` line. Why: the old
    resolver kept J2, which has no inflow line, and dropped J1, so the cell coupled none. What
    kills it: ranking every node of the cell rather than the inflow-carrying ones. Second
    correct implementation: filtering after ranking, as long as the keeper is an inflow node.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "C1": dict(InletNode="J1", OutletNode="J2", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=1),
            "C2": dict(InletNode="J2", OutletNode="O1", LinkType="CONDUIT", Shape="RECT_CLOSED", Geom1=1, Geom2=3),
        },
    )
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1"], one_per_cell=True)
    assert _coupled(res, ["J1"]) == {"J1"}


def test_an_orifice_opening_keys_its_junction():
    """Property (option on): a junction draining through an orifice outranks a smaller conduit's.

    Class: junctions whose only outgoing link is an orifice or weir. Why: the orifice row has
    no barrel count, which used to make its area NaN and empty the cell. What kills it:
    excluding orifices from the key's link types. Second correct implementation: keying
    orifices on their opening area through a separate function.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "OR1": dict(InletNode="J1", OutletNode="O1", LinkType="ORIFICE", Shape="RECT_CLOSED", Geom1=2, Geom2=2),
            "C2": dict(InletNode="J2", OutletNode="O1", LinkType="CONDUIT", Shape="CIRCULAR", Geom1=1.0, Barrels=1),
        },
    )
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"], one_per_cell=True)
    assert _coupled(res, ["J1", "J2"]) == {"J1"}


@pytest.mark.parametrize("one_per_cell", [False, True])
def test_a_cell_of_only_ineligible_inflow_nodes_raises_the_eligibility_error(one_per_cell):
    """Property: a lone outfall carrying an inflow line raises the error naming it and its reason.

    Class: cells whose inflow-carrying nodes are all outfalls, storage nodes or dividers.
    Why: TRITON-SWMM cannot couple them; before, the outfall was coupled at a depth of zero.
    What kills it: dropping the line silently. Second correct implementation: raising
    ConfigurationError instead; the assertion reads the shared message substrings.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables({"O1": ("OUTFALL", 0.0)}, {})
    with pytest.raises(ProcessingError) as excinfo:
        resolve_cell_inflow_nodes(["O1"], nodes, links, ["O1"], one_per_cell=one_per_cell)
    assert "Outfall O1" in str(excinfo.value)
    assert "boundary condition" in str(excinfo.value)


def test_a_cell_with_no_inflow_node_couples_none_and_raises_nothing():
    """Property: two outfalls with no inflow line couple none, with no error.

    Class: cells holding no inflow-carrying node. Why: the old branch called the lowest-
    invert helper on an empty set and crashed with an unnamed ValueError. What kills it:
    ranking or erroring on an empty candidate set. Second correct implementation: skipping
    such cells before calling the resolver.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables({"O1": ("OUTFALL", 0.0), "O2": ("OUTFALL", 1.0)}, {})
    for one_per_cell in (False, True):
        res = resolve_cell_inflow_nodes(["O1", "O2"], nodes, links, [], one_per_cell=one_per_cell)
        assert res.kept == () and res.dropped == ()


def test_an_unsizable_shape_names_itself_and_where_shapes_are_added():
    """Property (option on): ranking a TRAPEZOIDAL pipe raises a message naming it and calc_area.

    Class: any cell of two or more eligible junctions whose ranking needs an unsizable shape.
    Why: the message must name the
    shape and the link and say that shapes CAN be added in calc_area in src/hhemt/scenario.py.
    What kills it: dropping any of those parts, or raising on a one-junction cell. Second
    correct implementation: a dedicated exception subclass carrying the same text.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables(
        {"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0), "O1": ("OUTFALL", 0.0)},
        {
            "T1": dict(InletNode="J1", OutletNode="O1", LinkType="CONDUIT", Shape="TRAPEZOIDAL", Geom1=1, Geom2=1),
            "C2": dict(InletNode="J2", OutletNode="O1", LinkType="CONDUIT", Shape="CIRCULAR", Geom1=1.0, Barrels=1),
        },
    )
    with pytest.raises(ProcessingError) as excinfo:
        resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"], one_per_cell=True)
    message = str(excinfo.value)
    assert "TRAPEZOIDAL" in message and "T1" in message
    assert "additional pipe shapes can be added" in message.lower()
    assert "calc_area in src/hhemt/scenario.py" in message
    # One eligible junction needs no area, so nothing is raised.
    res = resolve_cell_inflow_nodes(["J1"], nodes, links, ["J1"], one_per_cell=True)
    assert res.kept == ("J1",)


def test_unlinked_junctions_tie_and_the_lower_invert_wins():
    """Property (option on): two inflow junctions with no links keep the lower-invert one.

    Class: cells whose eligible junctions share the lowest key and none leaves the cell.
    Why: the old no-outgoing-link branch kept a node but dropped none, coupling both. What
    kills it: keeping both, or the higher invert. Second correct implementation: sorting by
    (key, exit, invert) in one pass.
    """
    from hhemt.scenario import resolve_cell_inflow_nodes

    nodes, links = _tables({"J1": ("JUNCTION", 2.0), "J2": ("JUNCTION", 1.0)}, {})
    res = resolve_cell_inflow_nodes(["J1", "J2"], nodes, links, ["J1", "J2"], one_per_cell=True)
    assert _coupled(res, ["J1", "J2"]) == {"J2"}


# ---------------------------------------------------------------------------
# The node universe (Spec D) and the [INFLOWS] writer (Spec F, shared parser)
# ---------------------------------------------------------------------------


_DIVIDER_INP = """
[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
J1 1 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[DIVIDERS]
D1 1 C1 CUTOFF 0.5 0 0 0 0

[CONDUITS]
C1 D1 O1 10 0.013 0 0 0 0
C2 J1 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1
C2 CIRCULAR 1 0 0 0 1

[INFLOWS]
J1 FLOW "" FLOW 1.0 1 0.01
D1 FLOW "" FLOW 1.0 1 0.01

[COORDINATES]
J1 5 5
D1 6 6
O1 95 95
"""


def test_a_divider_is_grouped_into_its_dem_cell(tmp_path):
    """Property: a junction and a divider in one cell are both returned in that cell.

    Class: models with a ``[DIVIDERS]`` section. Why: swmmio's node frame omits dividers, so a
    divider was never grouped, never dropped, and missing from its cell's hydrograph. What
    kills it: reading nodes from ``model.nodes``. Second correct implementation: parsing the
    four sections with the toolkit's own reader instead of swmmio's section accessors.
    """
    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell

    df, outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
        _inp(tmp_path / "m.inp", _DIVIDER_INP), _dem(tmp_path / "dem.tif")
    )
    cells = df.groupby(["dem_x_coord", "dem_y_coord"])["node_key"].apply(set).tolist()
    assert {"J1", "D1"} in cells
    assert outfalls == ["O1"]


def test_a_node_without_coordinates_raises_naming_it(tmp_path):
    """Property: a junction with no ``[COORDINATES]`` row raises ProcessingError naming it.

    Class: any node missing a coordinate. Why: swmmio's geodataframe raised an unnamed
    TypeError. What kills it: skipping the node silently. Second correct implementation:
    checking every node's coordinate before grouping, in one pass.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell

    inp = _inp(tmp_path / "m.inp", "[JUNCTIONS]\nJ1 1 2 0 0 0\nJ2 1 2 0 0 0\n\n[COORDINATES]\nJ1 5 5\n")
    with pytest.raises(ProcessingError, match="J2"):
        return_df_of_nodes_grouped_by_DEM_gridcell(inp, _dem(tmp_path / "dem.tif"))


def test_an_ineligible_node_sharing_a_cell_is_dropped_with_a_named_warning(tmp_path):
    """Property: a divider sharing a cell with a junction loses its inflow line, with a warning naming it.

    Class: ineligible inflow nodes that share a cell with an eligible junction. Why: the
    divider was not in the node universe, so both were coupled. What kills it: keeping the
    divider's line, or dropping it silently. Second correct implementation: one warning per
    dropped node instead of one per scenario; the assertion reads the name in any warning.
    """
    import swmmio

    from hhemt.scenario_inputs import ScenarioInputGenerator

    inp = _inp(tmp_path / "hydraulics.inp", _DIVIDER_INP)
    gen = _generator(tmp_path, inp, one_per_cell=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ScenarioInputGenerator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(gen)
    assert any("D1 (divider)" in str(w.message) for w in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        remaining = set(swmmio.Model(str(inp)).inp.inflows.index)
    assert remaining == {"J1"}


def _generator(tmp_path, inp, *, one_per_cell):
    """A ScenarioInputGenerator bound to a fake scenario that carries only what the resolver step reads."""
    from hhemt.scenario_inputs import ScenarioInputGenerator

    sim_folder = tmp_path / "sim"
    sim_folder.mkdir(exist_ok=True)
    scenario = SimpleNamespace(
        scen_paths=SimpleNamespace(swmm_hydraulics_inp=inp, sim_folder=sim_folder),
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(one_coupled_inflow_node_per_dem_cell=one_per_cell)),
        _system=SimpleNamespace(sys_paths=SimpleNamespace(dem_processed=_dem(tmp_path / "dem.tif"))),
        log=SimpleNamespace(),
    )
    gen = ScenarioInputGenerator.__new__(ScenarioInputGenerator)
    gen.scenario = scenario
    gen.cfg_analysis = scenario._analysis.cfg_analysis
    gen.system = scenario._system
    return gen


_WRITER_INP = """
[OPTIONS]
FLOW_UNITS CMS

[{header}]
;;Node Constituent Series Type Mfactor Sfactor Baseline
J1 FLOW "" FLOW 1.0 1 0.01
J2 FLOW "" FLOW 1.0 1 0.01

[TIMESERIES]
J1 0 1
"""


@pytest.mark.parametrize("header", ["INFLOWS", "INFLOW", "inflows"])
def test_the_writer_finds_the_section_as_swmm_does(tmp_path, header):
    """Property: the drop set's lines are removed under any header SWMM accepts.

    Class: inflow sections headed ``[INFLOW]``, ``[INFLOWS]`` or lower case, and sections
    followed by any header (here ``[TIMESERIES]``, with no ``[CURVES]``). Why: the old writer
    matched the literal ``[INFLOWS]`` and needed a ``[CURVES]`` header after it, raising
    UnboundLocalError otherwise. What kills it: matching the literal header, or ending the
    section at ``[CURVES]`` only. Second correct implementation: a regex-based section
    scanner with the same prefix and fold rule. Every other line is kept byte for byte.
    """
    from hhemt.scenario import remove_inflow_lines

    inp = _inp(tmp_path / "h.inp", _WRITER_INP.format(header=header))
    before = inp.read_text().splitlines()
    removed = remove_inflow_lines(inp, ["J2"])
    after = inp.read_text().splitlines()
    assert [line.split()[0] for line in removed] == ["J2"]
    assert after == [line for line in before if not line.startswith("J2 FLOW")]


def test_a_long_node_name_followed_by_fewer_than_four_spaces_is_removed(tmp_path):
    """Property: an inflow line whose long name leaves fewer than four spaces before the next column is removed.

    Class: node names long enough that a writer's column padding leaves fewer than four
    spaces; swmmio 0.8.2, the version the project pins, writes a 28-character name followed by
    three, which is the line built here. Why: the old writer split names on four spaces, so
    the line survived and the node stayed coupled. What kills it: splitting on a fixed run of
    spaces. Second correct implementation: ``line.split()[0]``; the assertion reads only which
    lines remain. The line is written directly rather than through swmmio, because a later
    swmmio pads differently and would silently remove the input class this test exists for.
    """
    import swmmio

    from hhemt.scenario import remove_inflow_lines

    long_name = "JUNCTION_WITH_A_LONG_NAME_01"
    inp = _inp(
        tmp_path / "h.inp",
        f"""
[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
{long_name} 1 2 0 0 0
J2 1 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[CONDUITS]
C1 J2 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1

[INFLOWS]
;;                             Constituent Time Series Type   Mfactor Sfactor Baseline
{long_name}   FLOW        ""          FLOW   1.0     1       0.01
J2                             FLOW        ""          FLOW   1.0     1       0.01
""",
    )
    assert f"{long_name}   FLOW" in inp.read_text() and f"{long_name}    " not in inp.read_text()
    remove_inflow_lines(inp, [long_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert set(swmmio.Model(str(inp)).inp.inflows.index) == {"J2"}


# ---------------------------------------------------------------------------
# R-4 -- the run-entry refusal of an undefined inflow node
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prop", "inflows", "expected"),
    [
        ("undefined", 'NOSUCH  FLOW  ""  FLOW  1.0 1 0.01 times_zero', ["NOSUCH"]),
        ("case control", 'j1  FLOW  ""  FLOW  1.0 1 0.01 times_zero', []),
        ("ascii-only fold", 'é1  FLOW  ""  FLOW  1.0 1 0.01 times_zero', ["é1"]),
    ],
)
def test_undefined_inflow_nodes_reads_names_as_swmm_does(tmp_path, prop, inflows, expected):
    """Property, with its control: inflow names are matched with SWMM's ASCII-only case fold.

    Class: inflow lines naming defined and undefined nodes in any case. Why: SWMM stops with
    ERROR 209 on an undefined name inside the job; the check refuses it before submission,
    and must accept exactly what SWMM accepts (``j1`` for ``J1``) and refuse what it refuses
    (``é1`` for ``É1``, which ``str.upper`` would wrongly match). What kills it: an exact
    comparison, or ``str.upper()``. Second correct implementation: a dict keyed on the folded
    name. The non-ASCII case's model defines ``É1``.
    """
    from hhemt.scenario import undefined_inflow_nodes

    text = _NODES_INP.format(inflows=inflows)
    if prop.startswith("ascii"):
        text = text.replace("J1 1 2 0 0 0", "É1 1 2 0 0 0")
    assert undefined_inflow_nodes(_inp(tmp_path / "t.inp", text)) == expected


def test_a_template_placeholder_is_skipped_and_named(tmp_path):
    """Property: a node token containing ``$`` is skipped and named in a warning, with no error.

    Class: unfilled template placeholders. Why: its value is not known until preparation, so
    refusing it would refuse a valid template. What kills it: returning it as undefined.
    Second correct implementation: a per-line warning; the assertion reads the line text.
    """
    from hhemt.scenario import undefined_inflow_nodes

    inp = _inp(tmp_path / "t.inp", _NODES_INP.format(inflows='${NODE}  FLOW  ""  FLOW  1.0 1 0.01'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert undefined_inflow_nodes(inp) == []
    assert any("${NODE}" in str(w.message) for w in caught)


def test_singular_section_headers_are_read_as_swmm_reads_them(tmp_path):
    """Property: ``[JUNCTION]`` and ``[INFLOW]`` headers define and read nodes as their plurals do.

    Class: section headers SWMM recognises by prefix. What kills it: matching the exact
    plural header. Second correct implementation: a keyword table of full names matched with
    ``startswith`` on the prefix.
    """
    from hhemt.scenario import undefined_inflow_nodes

    text = _NODES_INP.replace("[JUNCTIONS]", "[JUNCTION]").replace("[INFLOWS]", "[INFLOW]")
    assert undefined_inflow_nodes(_inp(tmp_path / "a.inp", text.format(inflows='J1 FLOW "" FLOW 1 1 0'))) == []
    bad = text.format(inflows='NOSUCH FLOW "" FLOW 1 1 0')
    assert undefined_inflow_nodes(_inp(tmp_path / "b.inp", bad)) == ["NOSUCH"]


def _fake_system(hydraulics=None, full=None, *, coupled=False, standalone=False):
    return SimpleNamespace(
        cfg_system=SimpleNamespace(
            toggle_tritonswmm_model=coupled,
            toggle_swmm_model=standalone,
            SWMM_hydraulics=hydraulics,
            SWMM_full=full,
        )
    )


def _fake_analysis(master, targets=None):
    from hhemt.analysis import TRITONSWMM_analysis

    fake = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=targets is not None),
        _system=master,
        sensitivity=SimpleNamespace(unique_system_targets=[SimpleNamespace(system=s) for s in (targets or [])]),
    )
    fake._inflow_check_templates = lambda: TRITONSWMM_analysis._inflow_check_templates(fake)
    return fake


def test_the_run_entry_check_raises_naming_the_node_and_file(tmp_path):
    """Property (run entry): the run-entry refusal raises ConfigurationError naming the node and template.

    Class: any coupled-model template with an undefined inflow node. Why: refusing before
    submission saves a queued job that would fail in seconds. What kills it: not calling the
    check, or a message without the node or path. Second correct implementation: collecting
    every template's names before raising once.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.exceptions import ConfigurationError

    bad = _inp(tmp_path / "bad.inp", _NODES_INP.format(inflows='NOSUCH FLOW "" FLOW 1 1 0'))
    fake = _fake_analysis(_fake_system(bad, coupled=True))
    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_analysis._refuse_undefined_inflow_nodes(fake)
    assert "NOSUCH" in str(excinfo.value) and str(bad) in str(excinfo.value)


def test_a_target_that_switches_the_coupled_model_on_is_checked(tmp_path):
    """Property: a sensitivity target turning the coupled model on has its template checked.

    Class: ``system.*`` overlay rows that turn a SWMM model on while the master has it off.
    Why: a gate keyed on the master would miss the row. What kills it: reading the master's
    toggles. Second correct implementation: deriving templates from the sub-analyses rather
    than the unique targets.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.exceptions import ConfigurationError

    bad = _inp(tmp_path / "bad.inp", _NODES_INP.format(inflows='NOSUCH FLOW "" FLOW 1 1 0'))
    fake = _fake_analysis(_fake_system(), targets=[_fake_system(bad, coupled=True)])
    assert bad in TRITONSWMM_analysis._inflow_check_templates(fake)
    with pytest.raises(ConfigurationError, match="NOSUCH"):
        TRITONSWMM_analysis._refuse_undefined_inflow_nodes(fake)


def test_the_masters_template_is_not_checked_when_no_target_uses_it(tmp_path):
    """Property, control: a bad master template that no target resolves to is not checked.

    Class: sensitivity analyses whose every target overlays its own template. Why: refusing a
    template no run reads would refuse a valid analysis. What kills it: always adding the
    master's templates. Second correct implementation: filtering by target membership after
    collecting every system's templates.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    bad = _inp(tmp_path / "bad.inp", _NODES_INP.format(inflows='NOSUCH FLOW "" FLOW 1 1 0'))
    good = _inp(tmp_path / "good.inp", _NODES_INP.format(inflows='J1 FLOW "" FLOW 1 1 0'))
    fake = _fake_analysis(_fake_system(bad, coupled=True), targets=[_fake_system(good, coupled=True)])
    assert TRITONSWMM_analysis._inflow_check_templates(fake) == [good]
    TRITONSWMM_analysis._refuse_undefined_inflow_nodes(fake)


def test_the_standalone_models_template_is_checked(tmp_path):
    """Property: with the standalone SWMM model on, its ``SWMM_full`` template is checked.

    Class: analyses running the standalone model. What kills it: checking
    ``SWMM_hydraulics`` only. Second correct implementation: one check per model type rather
    than a shared template list.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.exceptions import ConfigurationError

    full = _inp(tmp_path / "full.inp", _NODES_INP.format(inflows='NOSUCH FLOW "" FLOW 1 1 0'))
    fake = _fake_analysis(_fake_system(full=full, standalone=True))
    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_analysis._refuse_undefined_inflow_nodes(fake)
    assert str(full) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Flow units -- the hydrograph conversion and the preparation-time refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("line", "refused"), [("", True), ("FLOW_UNITS GPM", True), ("flow_units cms", False)])
def test_us_flow_units_are_refused_at_preparation(tmp_path, line, refused):
    """Property, with its control: a US-unit model, or one with no FLOW_UNITS line, is refused.

    Class: SWMM models in any flow unit, the key and value in any case. Why:
    the toolkit reads SWMM output as SI, and an absent line is SWMM's CFS default; lower-case
    ``flow_units cms`` is a CMS model and must pass. What kills it: an exact-case comparison,
    or treating an absent line as CMS. Second correct implementation: reading the unit through
    pyswmm after the hydrology run closes.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_runoff_modeling import refuse_us_flow_units

    inp = _inp(tmp_path / "m.inp", f"[options]\n{line}\n\n[JUNCTIONS]\nJ1 1 2 0 0 0\n")
    if refused:
        with pytest.raises(ProcessingError, match=str(inp)):
            refuse_us_flow_units([inp])
    else:
        refuse_us_flow_units([inp])


@pytest.mark.parametrize(
    ("options", "refused"),
    [
        ("FLOW_UNITS CMS\nFLOW_UNITS CFS", True),
        ("flow_units cfsx", True),
        ("FLOW_UNITS CFS\nFLOW_UNITS CMS", False),
    ],
)
def test_the_unit_check_reads_flow_units_as_swmm_does(tmp_path, options, refused):
    """Property: the preparation check sees the unit SWMM will run the model in.

    Class: ``[OPTIONS]`` blocks whose ``FLOW_UNITS`` lines SWMM resolves differently from a
    first-line, exact-match reading -- a repeated key, and a value SWMM matches by prefix. Why:
    SWMM applies the LAST ``FLOW_UNITS`` line and reads ``cfsx`` as CFS, so a check that reads
    the first line or compares exactly passes a template that then runs in US units. What
    kills it: returning the first matching line, or comparing the value exactly. Second
    correct implementation: collecting every value into a list and resolving its last entry
    through a lookup of the six unit words; the assertion reads only whether the check raises.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_runoff_modeling import refuse_us_flow_units

    inp = _inp(tmp_path / "m.inp", f"[OPTIONS]\n{options}\n\n[JUNCTIONS]\nJ1 1 2 0 0 0\n")
    if refused:
        with pytest.raises(ProcessingError, match=str(inp)):
            refuse_us_flow_units([inp])
    else:
        refuse_us_flow_units([inp])


def _system_config_with_template(tmp_path, field: str, template_text: str):
    """Load a minimal ``system_config`` whose ``field`` template carries ``template_text``.

    The other templates are absent or empty stubs, which the config-load unit check skips, so
    the field under test is the only one it judges. The check reads every template field that
    is set, whatever the model toggles say, so the hydrology toggle stays off (turning it on
    requires the rain-gauge mapping fields, which this property does not concern).
    """
    from hhemt.config.system import system_config

    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    system_dir = tmp_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    def stub(name, text="test"):
        """Write one input file and return its path as a string."""
        path = inputs / name
        path.write_text(text)
        return str(path)

    cfg = {
        "system_directory": str(system_dir),
        "watershed_gis_polygon": stub("watershed.shp"),
        "DEM_fullres": stub("dem.tif"),
        "SWMM_hydraulics": stub("hydraulics.inp"),
        "TRITONSWMM_software_directory": str(tmp_path / "triton"),
        "TRITONSWMM_git_URL": "https://example.com/triton.git",
        "SWMM_git_URL": "https://example.com/swmm.git",
        "triton_swmm_configuration_template": stub("TRITONSWMM.cfg"),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": 5.0,
        "constant_mannings": 0.05,
        "crs": {"horizontal_epsg": 4326},
    }
    cfg[field] = stub(f"{field}.inp", template_text)
    if field == "SWMM_full":
        cfg["toggle_swmm_model"] = True
    return system_config.model_validate(cfg)


@pytest.mark.parametrize(
    ("field", "options", "refused"),
    [
        ("SWMM_hydraulics", "[OPTIONS]\nFLOW_UNITS LPS\n", True),
        ("SWMM_full", "[OPTIONS]\nFLOW_UNITS MLD\n", True),
        ("SWMM_hydrology", "[OPTIONS]\nFLOW_UNITS LPS\n", False),
        ("SWMM_hydraulics", "[OPTIONS]\nFLOW_UNITS CFS\nFLOW_UNITS CMS\n", False),
        ("SWMM_hydraulics", "[JUNCTIONS]\nJ1 1 2 0 0 0\n", False),
    ],
)
def test_the_config_load_unit_check_admits_exactly_what_preparation_admits(tmp_path, field, options, refused):
    """Property: at config load each SWMM template is judged by preparation's unit rule and set.

    Class: templates for each of the three SWMM fields, in a unit the field does or does not
    accept, read as SWMM reads ``FLOW_UNITS``. Why: the hydraulics and standalone models must
    be CMS and the hydrology model may be LPS or MLD; a config-load check that admits LPS for
    the hydraulics model, or reads the first ``FLOW_UNITS`` line where SWMM reads the last,
    disagrees with preparation and either passes a model preparation refuses or refuses a
    valid one. A file with no ``[OPTION`` section is not a SWMM model and is skipped. What kills
    it: one SI set for all three fields, or a first-line reading. Second correct implementation:
    reading every template once and comparing against a per-field table built elsewhere; the
    assertions read only whether loading raises and the field it names.
    """
    from pydantic import ValidationError

    if refused:
        with pytest.raises(ValidationError, match=field) as excinfo:
            _system_config_with_template(tmp_path, field, options)
        assert "accepts CMS" in str(excinfo.value)
    else:
        _system_config_with_template(tmp_path, field, options)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[OPTIONS]\nFLOW_UNITS CMS\nFLOW_UNITS LPS\nFLOW_UNITS\n", "LPS"),
        ("[OPTIONS]\nFLOW_UNITS CMS\n\n[JUNCTIONS]\nJ1 1 2 0 0 0\n\n[OPTIONS]\nFLOW_UNITS LPS\n", "LPS"),
    ],
)
def test_the_unit_read_skips_a_valueless_line_and_spans_every_options_section(tmp_path, text, expected):
    """Property: the unit read ignores a ``FLOW_UNITS`` line with no value and reads every section.

    Class: templates whose last ``FLOW_UNITS`` line carries no value, and templates with more
    than one ``[OPTIONS]`` section. Why: SWMM ignores a line with fewer than two tokens and
    applies the rule across every ``[OPTION`` section, so either difference makes the toolkit
    predict a unit SWMM will not use. What kills it: taking the last line whatever it holds,
    or stopping at the end of the first ``[OPTIONS]`` section. Second correct implementation:
    filtering the option lines first and taking the last two-token one; the assertion reads
    the returned unit word.
    """
    from hhemt.swmm_runoff_modeling import inp_flow_units

    assert inp_flow_units(_inp(tmp_path / "m.inp", text)) == expected


@pytest.mark.parametrize("unit", ["CFS", "LPS", "MLD", "GPM"])
def test_hydraulics_and_standalone_models_must_be_cms_while_hydrology_may_be_lps_or_mld(tmp_path, unit):
    """Property: the hydraulics and standalone check refuses every unit but CMS; hydrology keeps SI.

    Class: SWMM models in each non-CMS flow unit. Why: the toolkit reads the coupled report's
    and the standalone output's flows as m3/s with no conversion, so an LPS model is read 1000
    times too large; the hydrology model's flows are converted, so LPS and MLD stay accepted
    there. What kills it: refusing only US units for the hydraulics and standalone models, or
    applying the CMS-only rule to the hydrology model. Second correct implementation: one
    function taking the accepted unit set as an argument; the assertions read only which
    calls raise.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_runoff_modeling import refuse_non_cms_flow_units, refuse_us_flow_units

    inp = _inp(tmp_path / "m.inp", f"[OPTIONS]\nFLOW_UNITS {unit}\n")
    with pytest.raises(ProcessingError, match=str(inp)):
        refuse_non_cms_flow_units([inp])
    if unit in ("LPS", "MLD"):
        refuse_us_flow_units([inp])
    refuse_non_cms_flow_units([_inp(tmp_path / "cms.inp", "[OPTIONS]\nFLOW_UNITS CMS\n")])


class _StopAfterTheChecks(Exception):
    """Raised by a fake collaborator to end ``prepare_scenario`` once the unit checks have run."""


class _AnyField:
    """A stand-in log whose every field reads False and accepts any value."""

    def __getattr__(self, name):
        return SimpleNamespace(get=lambda: False, set=lambda value: None)


def _scenario_for_preparation(tmp_path, full_options: str):
    """A scenario whose collaborators are fakes, its standalone template carrying ``full_options``.

    Every collaborator ``prepare_scenario`` reaches before the boundary-condition files is
    faked; that step raises ``_StopAfterTheChecks``, so a run that reaches it has passed every
    unit check. ``calls`` records whether the hydrology model was run.
    """
    from hhemt.scenario import TRITONSWMM_scenario

    cms = "[OPTIONS]\nFLOW_UNITS CMS\n"
    templates = tmp_path / "templates"
    templates.mkdir()
    hydraulics = _inp(templates / "hydraulics.inp", cms)
    hydrology = _inp(templates / "hydrology.inp", cms)
    full = _inp(templates / "full.inp", f"[OPTIONS]\n{full_options}\n\n[JUNCTIONS]\nJ1 1 2 0 0 0\n")
    sim = tmp_path / "sim"
    sim.mkdir()
    calls: list[str] = []

    def copy(source, dest):
        """Stand in for a template-to-scenario model builder: the file is copied unchanged."""
        Path(dest).write_text(Path(source).read_text())

    def stop():
        """End preparation at the first step after the unit checks."""
        raise _StopAfterTheChecks

    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen.log = _AnyField()
    scen.event_iloc = 0
    scen.backend = "cpu"
    scen._verify_native_build_or_skip_in_container = lambda: None
    scen._write_sim_weather_nc = lambda: None
    scen.scen_paths = SimpleNamespace(
        swmm_hydraulics_inp=sim / "hydraulics.inp",
        swmm_full_inp=sim / "full.inp",
        swmm_hydro_inp=sim / "hydro.inp",
    )
    scen._system = SimpleNamespace(
        cfg_system=SimpleNamespace(
            SWMM_hydraulics=hydraulics,
            SWMM_full=full,
            SWMM_hydrology=hydrology,
            toggle_swmm_model=True,
            toggle_use_swmm_for_hydrology=True,
        )
    )
    scen._runoff_modeler = SimpleNamespace(
        write_swmm_rainfall_dat_files=lambda: None,
        write_swmm_waterlevel_dat_files=lambda: None,
        create_hydrology_model_from_template=copy,
        run_swmm_hydro_model=lambda **kwargs: calls.append("hydrology run"),
    )
    scen._input_generator = SimpleNamespace(
        create_hydraulics_model_from_template=copy,
        update_swmm_threads_in_inp_file=lambda path: None,
        create_external_boundary_condition_files=stop,
    )
    scen._full_model_builder = SimpleNamespace(create_full_model_from_template=copy)
    return scen, calls


@pytest.mark.parametrize(
    ("full_options", "refused"),
    [("FLOW_UNITS CFS", True), ("", True), ("FLOW_UNITS LPS", True), ("FLOW_UNITS CMS", False)],
)
def test_the_standalone_model_is_unit_checked_at_preparation_before_hydrology_runs(tmp_path, full_options, refused):
    """Property: with the standalone model on, its template's unit is refused at preparation.

    Class: standalone (``SWMM_full``) templates in a unit other than CMS, including one with no
    ``FLOW_UNITS`` line, which SWMM runs as CFS. Why: the standalone model's output is read as
    m3/s unconverted, and a check after its run would come only after that run had spent its
    compute; refusing before the hydrology model runs is what the earlier failure buys. What
    kills it: not checking the standalone model's file, or checking it after the hydrology
    run. Second correct implementation: checking all three model files together right after
    the standalone file is written; the assertions read whether preparation raised naming
    ``full.inp`` and whether the hydrology model ran.
    """
    from hhemt.exceptions import ProcessingError

    scen, calls = _scenario_for_preparation(tmp_path, full_options)
    if refused:
        with pytest.raises(ProcessingError, match="full.inp"):
            scen.prepare_scenario()
        assert calls == []
    else:
        with pytest.raises(_StopAfterTheChecks):
            scen.prepare_scenario()
        assert calls == ["hydrology run"]


class _FakeOutUnits:
    """A pyswmm ``Output`` stand-in reporting ``units`` with one node and one link."""

    def __init__(self, units):
        self.units = units
        self.nodes = {"J1": None}
        self.links = {"C1": None}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSeries:
    """``NodeSeries(out)[id]`` / ``LinkSeries(out)[id]``: every attribute is one 2.0 value."""

    def __init__(self, out):
        pass

    def __getitem__(self, key):
        series = {pd.Timestamp("2020-01-01"): 2.0}
        return SimpleNamespace(
            invert_depth=series,
            hydraulic_head=series,
            total_inflow=series,
            flooding_losses=series,
            flow_rate=series,
            flow_depth=series,
            flow_velocity=series,
            capacity=series,
        )


@pytest.mark.parametrize(
    ("system", "flow", "refused"),
    [("SI", "LPS", True), ("SI", "MLD", True), ("US", "CFS", True), ("SI", "CMS", False)],
)
def test_a_standalone_output_not_in_cms_is_refused_when_read(tmp_path, monkeypatch, system, flow, refused):
    """Property: the standalone ``.out`` reader refuses any flow unit but CMS.

    Class: standalone outputs in each SWMM flow unit, whatever the unit system. Why: the reader
    stores every flow series as m3/s with no conversion, so an SI output in LPS or MLD passes a
    unit-system test and is read 1000 or 86.4 times too large; this is the guard for outputs
    the preparation check never saw, such as a reprocess. What kills it: testing the unit
    system instead of the flow unit. Second correct implementation: converting nothing and
    testing membership in a one-element accepted set; the assertions read whether the read
    raises and, for CMS, the stored inflow value.
    """
    import pyswmm

    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import return_node_time_series_results_from_outfile

    monkeypatch.setattr(pyswmm, "Output", lambda path: _FakeOutUnits({"system": system, "flow": flow}))
    monkeypatch.setattr(pyswmm, "NodeSeries", _FakeSeries)
    monkeypatch.setattr(pyswmm, "LinkSeries", _FakeSeries)
    out = tmp_path / "full.out"
    if refused:
        with pytest.raises(ProcessingError, match="full.out"):
            return_node_time_series_results_from_outfile(out)
    else:
        ds_nodes, _ = return_node_time_series_results_from_outfile(out)
        assert float(ds_nodes["inflow_flow_cms"].values.max()) == 2.0


def _coupled_rpt(path: Path, unit: str) -> Path:
    """The four metadata lines a coupled report's parse requires, with ``unit`` as its flow unit."""
    path.write_text(
        "\n".join(
            [
                f"  Flow Units ............... {unit}",
                "",
                "  **************************        Volume        Volume",
                "  Flow Routing Continuity        hectare-m      10^6 ltr",
                "  **************************     ---------     ---------",
                "  Flooding Loss ............         0.000         0.000",
                "  Continuity Error (%) .....         0.000",
                "",
                "  Analysis ended on:  Sun Aug 17 09:00:00 2026",
            ]
        )
        + "\n"
    )
    return path


@pytest.mark.parametrize(("unit", "refused"), [("LPS", True), ("MLD", True), ("CFS", True), ("CMS", False)])
def test_a_coupled_report_not_in_cms_is_refused_when_read(tmp_path, monkeypatch, unit, refused):
    """Property: the coupled report reader refuses a report whose flow unit is not CMS.

    Class: coupled ``.rpt`` files in each SWMM flow unit. Why: the coupled model's report
    flows are read as m3/s with no conversion, and before this the report's unit was recorded
    and compared with nothing, so a reprocessed US-unit run, or an LPS or MLD one, was read
    with no error. What kills it: placing no comparison on the report path, or comparing
    case-sensitively against the report's own casing. Second correct implementation:
    comparing the raw ``Flow Units`` line after folding it; the assertion reads only whether
    the read raises before the time-series step, which a fake stops.
    """
    from hhemt import swmm_output_parser as sop
    from hhemt.exceptions import ProcessingError

    def stop(*args, **kwargs):
        """End the read at its first step after the unit comparison."""
        raise _StopAfterTheChecks

    monkeypatch.setattr(sop, "convert_datavars_to_dtype", stop)
    inp = _inp(tmp_path / "hydraulics.inp", "[OPTIONS]\nFLOW_UNITS CMS\n")
    rpt = _coupled_rpt(tmp_path / "hydraulics.rpt", unit)
    if refused:
        with pytest.raises(ProcessingError, match="hydraulics.rpt"):
            sop.retrieve_SWMM_outputs_as_datasets(inp, rpt)
    else:
        with pytest.raises(_StopAfterTheChecks):
            sop.retrieve_SWMM_outputs_as_datasets(inp, rpt)


@pytest.mark.parametrize("unit", ["LPS", "MLD"])
def test_a_hydrology_report_in_lps_or_mld_is_still_read(tmp_path, unit):
    """Property (control): the CMS-only refusal does not reach the hydrology report's reader.

    Class: hydrology ``.rpt`` files in an SI unit other than CMS. Why: the hydrology model's
    flows are converted to m3/s, so LPS and MLD stay accepted for it, and its report summary
    shares the system-results builder with the coupled path. What kills it: placing the
    coupled report's unit comparison in that shared builder. Second correct implementation:
    passing the accepted unit set into the builder from each caller; the assertion reads the
    returned unit attribute.
    """
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    ds = parse_hydrology_rpt_summary(_coupled_rpt(tmp_path / "hydro.rpt", unit))
    assert ds.attrs["flow_units"] == unit.lower()


# ---------------------------------------------------------------------------
# The one-node-per-cell option's selection record and its report disclosure
# ---------------------------------------------------------------------------


_TWO_IN_ONE_CELL_INP = """
[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
J1 1 2 0 0 0
J2 1 2 0 0 0
J3 1 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0
C2 J2 O1 10 0.013 0 0 0 0
C3 J3 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 2 0 0 0 1
C2 CIRCULAR 1 0 0 0 1
C3 CIRCULAR 1 0 0 0 1

[INFLOWS]
J1 FLOW "" FLOW 1.0 1 0.01
J2 FLOW "" FLOW 1.0 1 0.01
J3 FLOW "" FLOW 1.0 1 0.01

[COORDINATES]
J1 5 5
J2 6 6
J3 55 55
O1 95 95
"""


def _fake_validated_analysis(sim_dir):
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        analysis_paths=SimpleNamespace(simulation_directory=sim_dir),
        df_sims=pd.DataFrame(index=[0]),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": 0},
    )


def test_a_reduced_cell_is_recorded_and_listed_as_a_passing_disclosure(tmp_path):
    """Property (option on): a two-junction cell is recorded, disclosed by the check, and listed.

    Class: cells the option reduces. Why: every reduction must be visible
    in the report. What kills it: not writing the record, a check that fails or
    flips ``overall_passed``, or a renderer that lists details only for failures. Second
    correct implementation: a record keyed by cell rather than a list; the assertions read
    the node names and the rendered row.
    """
    from hhemt.analysis_validation import ValidationReport, check_inflow_node_reduction
    from hhemt.report_renderers.errors_and_warnings import _render_aggregate_table
    from hhemt.scenario import compute_event_id_slug
    from hhemt.scenario_inputs import ScenarioInputGenerator

    sim_dir = tmp_path / "sims"
    event_dir = sim_dir / compute_event_id_slug({"event_id": 0})
    event_dir.mkdir(parents=True)
    inp = _inp(tmp_path / "hydraulics.inp", _TWO_IN_ONE_CELL_INP)
    gen = _generator(tmp_path, inp, one_per_cell=True)
    gen.scenario.scen_paths.sim_folder = event_dir
    ScenarioInputGenerator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(gen)
    record = json.loads((event_dir / "_status" / "inflow_node_selection.json").read_text())
    assert len(record["cells"]) == 1
    cell = record["cells"][0]
    assert {n["node"] for n in cell["inflow_nodes"]} == {"J1", "J2"}
    assert cell["kept"] == "J1"
    assert cell["dropped"] == [{"node": "J2", "reason": "ranking"}]

    res = check_inflow_node_reduction(_fake_validated_analysis(sim_dir))
    assert res.passed is True and res.applicable is True
    assert len(res.details) == 1 and "J2 (ranking)" in res.details[0]["detail"]
    assert "1 DEM cell(s) reduced" in res.summary
    assert ValidationReport(checks=[res]).overall_passed is True
    assert "J2 (ranking)" in _render_aggregate_table([res])


def test_one_node_cells_leave_an_empty_record_and_a_zero_count(tmp_path):
    """Property, control (option on): no reduced cell writes an empty record and reports zero.

    Class: models with one inflow junction per cell. Why: the record is written for every
    scenario prepared with the option on, so its presence separates "measured, nothing
    reduced" from "not measured". What kills it: skipping the write when no cell was reduced.
    Second correct implementation: writing the record before the resolver runs and filling it
    after.
    """
    from hhemt.analysis_validation import check_inflow_node_reduction
    from hhemt.scenario import compute_event_id_slug
    from hhemt.scenario_inputs import ScenarioInputGenerator

    sim_dir = tmp_path / "sims"
    event_dir = sim_dir / compute_event_id_slug({"event_id": 0})
    event_dir.mkdir(parents=True)
    inp = _inp(tmp_path / "hydraulics.inp", _NODES_INP.format(inflows='J1 FLOW "" FLOW 1 1 0'))
    gen = _generator(tmp_path, inp, one_per_cell=True)
    gen.scenario.scen_paths.sim_folder = event_dir
    ScenarioInputGenerator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(gen)
    assert json.loads((event_dir / "_status" / "inflow_node_selection.json").read_text()) == {"cells": []}
    res = check_inflow_node_reduction(_fake_validated_analysis(sim_dir))
    assert res.applicable is True and res.details == []
    assert "0 DEM cell(s) reduced" in res.summary


def test_a_scenario_without_a_record_is_not_measured(tmp_path):
    """Property: scenarios with no selection record are reported not measured, and the check is N/A.

    Class: analyses prepared without the option, or before it existed. Why: an absent record
    must never read as "no reduced cell". What kills it: counting absent records as zero
    reductions. Second correct implementation: a summary worded differently that still names
    the not-measured count.
    """
    from hhemt.analysis_validation import check_inflow_node_reduction

    res = check_inflow_node_reduction(_fake_validated_analysis(tmp_path / "sims"))
    assert res.applicable is False and res.passed is True
    assert "1 not measured" in res.summary


# ---------------------------------------------------------------------------
# The DEM-rim option: its cfg line and the guard that refuses a build that would ignore it
# ---------------------------------------------------------------------------


def _scenario_for_coupled_cfg(
    tmp_path, monkeypatch, *, rim_from_dem, environment="native", software_dir=None, branch_key=None, recorded_sha=None
):
    """A scenario whose coupled TRITON cfg generator runs on fakes; returns (scenario, cfg path).

    The cfg template carries only a ``dem_filename=`` line, so every line after the generator
    runs is one the generator itself added.
    """
    from hhemt import scenario as scenario_module
    from hhemt.scenario import TRITONSWMM_scenario

    template = tmp_path / "TRITONSWMM.cfg"
    template.write_text("sim_start_time=0\ndem_filename=dem.dem\n")
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
    cfg_analysis = SimpleNamespace(
        manhole_diameter=1.0,
        manhole_loss_coefficient=0.1,
        TRITON_raw_output_type="bin",
        hydraulic_timestep_s=1.0,
        TRITON_reporting_timestep_s=60,
        open_boundaries=1,
        swmm_snapshot_disable=False,
        swmm_snapshot_keep_all=False,
        swmm_node_rim_from_dem=rim_from_dem,
        execution_environment=environment,
    )
    cfg_system = SimpleNamespace(
        toggle_use_constant_mannings=True,
        constant_mannings=0.05,
        triton_swmm_configuration_template=template,
        TRITONSWMM_software_directory=software_dir,
        TRITONSWMM_branch_key=branch_key,
    )
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen._analysis = SimpleNamespace(cfg_analysis=cfg_analysis)
    scen._system = SimpleNamespace(
        cfg_system=cfg_system,
        sys_paths=SimpleNamespace(dem_processed=tmp_path / "dem.dem", mannings_processed=tmp_path / "man.dem"),
        log=SimpleNamespace(triton_head_sha=SimpleNamespace(get=lambda: recorded_sha)),
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


@pytest.mark.parametrize("rim_from_dem", [True, False])
def test_the_dem_rim_option_writes_its_exact_cfg_line_only_when_set(tmp_path, monkeypatch, rim_from_dem):
    """Property, with its control: True puts ``swmm_node_rim_from_dem=1`` on a line of its own.

    Class: coupled TRITON cfgs generated with the option on and off, for a build that reads
    the key (container mode, recorded and configured commit both the first listed commit).
    Why: TRITON reads the value with ``atoi`` and splits ``key=value`` without trimming, so only
    the exact line ``swmm_node_rim_from_dem=1`` turns the option on, and with the option off
    no line beginning with the key may appear. What kills it: not appending the line, writing
    ``True`` or spaces around ``=``, or appending it whatever the field says. Second correct
    implementation: placing the line before the other appended keys; the assertions read the
    set of whole lines and the absence of the key's prefix.
    """
    from hhemt.scenario import SWMM_NODE_RIM_FROM_DEM_FIRST_SHA

    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=rim_from_dem,
        environment="container",
        branch_key=SWMM_NODE_RIM_FROM_DEM_FIRST_SHA,
        recorded_sha=SWMM_NODE_RIM_FROM_DEM_FIRST_SHA,
    )
    scen._generate_TRITON_SWMM_cfg()
    lines = cfg.read_text().splitlines()
    if rim_from_dem:
        assert lines.count("swmm_node_rim_from_dem=1") == 1
    else:
        assert not any(line.startswith("swmm_node_rim_from_dem") for line in lines)


def _two_commit_clone(path: Path):
    """A git repository with two commits; returns (older sha, newer sha), HEAD at the newer."""
    import subprocess

    path.mkdir()

    def git(*args):
        """Run one git command in the repository and return its stripped stdout."""
        return subprocess.run(
            ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (path / "a").write_text("1")
    git("add", "a")
    git("commit", "-q", "-m", "older")
    older = git("rev-parse", "HEAD")
    (path / "a").write_text("2")
    git("commit", "-q", "-am", "newer")
    newer = git("rev-parse", "HEAD")
    return older, newer


def test_the_dem_rim_option_is_refused_for_a_native_build_that_would_ignore_it(tmp_path):
    """Property: native mode refuses the option unless the clone descends from the required commit.

    Class: native TRITON clones whose HEAD is before, at, or after the first commit that reads
    the key, and a required commit the clone does not contain. Why: an older build ignores
    the key silently, so the run would use SWMM's rims while the configuration says DEM rims.
    What kills it: comparing commits for equality (refuses a later descendant), not checking
    at all, or treating an unresolvable ancestry as a pass. Second correct implementation:
    walking ``git rev-list HEAD`` for the required commit; the assertions read only whether
    the guard raises and that the message names the field and the required commit.
    """
    import subprocess

    from hhemt.exceptions import ConfigurationError
    from hhemt.scenario import require_dem_rim_capable_triton

    clone = tmp_path / "triton"
    older, newer = _two_commit_clone(clone)

    def guard(required):
        """Run the native guard against the clone with ``required`` as the first reading commit."""
        require_dem_rim_capable_triton(
            execution_environment="native",
            software_directory=clone,
            branch_key=None,
            recorded_sha=None,
            required_sha=required,
        )

    guard(newer)  # HEAD is the required commit
    guard(older)  # HEAD descends from it
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", older], check=True)
    with pytest.raises(ConfigurationError, match=newer) as excinfo:
        guard(newer)  # HEAD predates it
    assert "swmm_node_rim_from_dem" in str(excinfo.value)
    with pytest.raises(ConfigurationError, match="swmm_node_rim_from_dem"):
        guard("0" * 40)  # the clone does not contain it
    with pytest.raises(ConfigurationError, match="swmm_node_rim_from_dem"):
        require_dem_rim_capable_triton(
            execution_environment="native",
            software_directory=tmp_path / "no_clone",
            branch_key=None,
            recorded_sha=None,
        )


@pytest.mark.parametrize(
    ("recorded", "configured", "passes"),
    [
        ("first", "first", True),
        (None, "first", False),
        ("first", "other", False),
        ("1b8742be088f44f8f9b596a679cb3fb6f98e7e13", "1b8742be088f44f8f9b596a679cb3fb6f98e7e13", False),
    ],
)
def test_the_dem_rim_option_in_container_mode_needs_a_recorded_listed_commit(
    tmp_path, monkeypatch, recorded, configured, passes
):
    """Property: in container mode the option is written only for a recorded, configured, listed commit.

    Class: container runs whose image-recorded TRITON commit is absent, differs from the
    configured ``TRITONSWMM_branch_key``, or is an unlisted older build (``1b8742b``). Why: there
    is no host clone to test ancestry, so the recorded and configured commits must agree and be
    one known to read the key; anything else would let an older image ignore it. What kills
    it: trusting the configured commit alone, accepting any recorded commit, or refusing every
    container run. Second correct implementation: checking the listed set before the equality;
    the assertions read whether preparation raises naming the field and, when it does not,
    the exact cfg line.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.scenario import SWMM_NODE_RIM_FROM_DEM_FIRST_SHA

    sha = {"first": SWMM_NODE_RIM_FROM_DEM_FIRST_SHA, "other": "f" * 40}
    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=True,
        environment="container",
        branch_key=sha.get(configured, configured),
        recorded_sha=sha.get(recorded, recorded),
    )
    if passes:
        scen._generate_TRITON_SWMM_cfg()
        assert "swmm_node_rim_from_dem=1" in cfg.read_text().splitlines()
    else:
        with pytest.raises(ConfigurationError, match="swmm_node_rim_from_dem"):
            scen._generate_TRITON_SWMM_cfg()
