"""Reviewer tests for the SWMM output inventory, written by the reviewing seat.

Every test here runs on the laptop. The two tests that need a binary output run ONE tiny stock SWMM 5.2.4 model
(three nodes, two conduits, a few seconds) through ``swmm.toolkit.solver`` in the test's own temporary directory;
nothing coupled, no TRITON-SWMM compile or run.

Each test names, in its own docstring, the input CLASS it quantifies over, the property and why it is required, the
production change that reds it (B3 field iv) and a second correct implementation that stays green (B5).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

# A tiny Dynamic Wave model in CMS (the toolkit refuses any other flow unit). J1 receives a constant 0.05 m3/s, so
# its LATERAL inflow is nonzero and distinct from every other series; J2 is a storage unit, so a node's stored volume
# is nonzero (a plain junction without ponding stores 0 in the binary output).
_TINY_INP = textwrap.dedent(
    """\
    [TITLE]
    tiny

    [OPTIONS]
    FLOW_UNITS          CMS
    INFILTRATION        HORTON
    FLOW_ROUTING        DYNWAVE
    LINK_OFFSETS        DEPTH
    ALLOW_PONDING       NO
    START_DATE          01/01/2001
    START_TIME          00:00:00
    REPORT_START_DATE   01/01/2001
    REPORT_START_TIME   00:00:00
    END_DATE            01/01/2001
    END_TIME            02:00:00
    REPORT_STEP         00:05:00
    WET_STEP            00:05:00
    DRY_STEP            00:05:00
    ROUTING_STEP        0:00:10

    [JUNCTIONS]
    J1  10.0  3.0  0  0  0

    [STORAGE]
    J2   9.0  3.0  0  FUNCTIONAL  100  0  0  0

    [OUTFALLS]
    O1   8.0  FREE  NO

    [CONDUITS]
    C1  J1  J2  100  0.013  0  0  0  0
    C2  J2  O1  100  0.013  0  0  0  0

    [XSECTIONS]
    C1  CIRCULAR  0.6  0  0  0  1
    C2  CIRCULAR  0.6  0  0  0  1

    [INFLOWS]
    J1  FLOW  ""  FLOW  1.0  1.0  0.05

    [REPORT]
    CONTROLS NO
    NODES ALL
    LINKS ALL

    [COORDINATES]
    J1  0  0
    J2  100  0
    O1  200  0
    """
)


def _run_tiny_model(folder: Path, node_a: str = "J1") -> Path:
    """Run the tiny model in ``folder``; return the path of ``hydraulics.inp`` (the report and output sit beside it)."""
    from swmm.toolkit import solver

    folder.mkdir(parents=True, exist_ok=True)
    inp = folder / "hydraulics.inp"
    inp.write_text(_TINY_INP.replace("J1", node_a))
    solver.swmm_run(str(inp), str(folder / "hydraulics.rpt"), str(folder / "hydraulics.out"))
    assert (folder / "hydraulics.out").stat().st_size > 0
    assert (folder / "hydraulics.rpt").stat().st_size > 0
    return inp


# SWMM 5.2.4 ``enums.h`` (``NodeResultType`` and ``LinkResultType``): the position of each quantity in the binary
# output, written here as integers so the expectation does not borrow the toolkit's own attribute names.
_NODE_INDEX = {"depth": 0, "head": 1, "volume": 2, "lateral_inflow": 3, "inflow_flow": 4, "flooding": 5}
_LINK_INDEX = {"flow": 0, "link_depth": 1, "velocity": 2, "link_volume": 3}


@pytest.mark.parametrize("family", ["swmm", "tritonswmm"])
def test_every_stored_series_is_the_binary_outputs_series_at_its_enums_h_position(tmp_path, family):
    """NEW CAPABILITY. Input class: both families, a REAL binary output written by stock SWMM 5.2.4 (not a stand-in).

    Property: each stored node series equals the binary output's series at the position ``enums.h`` gives its
    quantity (``NODE_VOLUME`` 2, ``NODE_LATFLOW`` 3, ``LINK_VOLUME`` 3, ``LINK_CAPACITY`` 4), and the link capacity
    is named ``capacity_setting`` for the coupled family and ``capacity`` for the standalone one.
    Required because the chunk's own tests read the series through a stand-in file, so a name-to-position slip in the
    pyswmm call (reading ``flooding`` into ``lateral_inflow``, say) is invisible to them and wrong in every store.
    Reds under: ``lateral_inflow`` read from the total-inflow attribute (positions 3 and 4 exchanged).
    A second correct implementation that stays green: reading all five link series in one call instead of five.
    Control (must be nonzero): J1's lateral inflow equals its 0.05 m3/s boundary inflow and the stored volume is
    positive somewhere, so an all-zero store cannot satisfy the equalities.
    """
    from pyswmm import Output
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit import shared_enum

    import hhemt.swmm_output_parser as sp

    inp = _run_tiny_model(tmp_path / family)
    ds_nodes, ds_links = sp.retrieve_SWMM_outputs_as_datasets(inp, inp.with_suffix(".out"), family=family)

    # The expected values are read one TIME INDEX at a time with the toolkit's `get_node_result` / `get_link_result`,
    # not with pyswmm's `Output.node_series`: pyswmm 2.1.0's series accessor drops the final report period, so a
    # series read through it is one value short of the stored series (erratum E43) and cannot be the reference.
    with Output(str(inp.with_suffix(".out"))) as out:
        n_periods = int(swmm_output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
        for name, position in _NODE_INDEX.items():
            for node, index in out.nodes.items():
                expected = np.array(
                    [swmm_output.get_node_result(out.handle, t, index)[position] for t in range(n_periods)],
                    dtype="f4",
                )
                np.testing.assert_allclose(ds_nodes[name].sel(node_id=node).values, expected, atol=1e-6, err_msg=name)
        capacity = "capacity_setting" if family == "tritonswmm" else "capacity"
        for name, position in {**_LINK_INDEX, capacity: 4}.items():
            for link, index in out.links.items():
                expected = np.array(
                    [swmm_output.get_link_result(out.handle, t, index)[position] for t in range(n_periods)],
                    dtype="f4",
                )
                np.testing.assert_allclose(ds_links[name].sel(link_id=link).values, expected, atol=1e-6, err_msg=name)

    other = "capacity" if family == "tritonswmm" else "capacity_setting"
    assert capacity in ds_links.data_vars and other not in ds_links.data_vars
    assert float(ds_nodes["lateral_inflow"].sel(node_id="J1").max()) == pytest.approx(0.05, abs=1e-6)
    assert float(ds_nodes["volume"].max()) > 0.0
    assert float(ds_links["link_volume"].max()) > 0.0


def test_the_run_level_variables_of_a_real_report_are_variables_and_survive_the_zarr_round_trip(tmp_path):
    """NEW CAPABILITY. Input class: the report of a REAL stock SWMM run (not a constructed section), coupled family.

    Property: ``flow_continuity_error`` and ``system_flooding`` are 0-d data variables equal to the report's printed
    figures, ``flow_units`` and ``analysis_end_datetime`` are the only attributes, and every string-valued rank
    variable reads back from zarr unchanged. Required because the chunk's tests build the report text by hand: a
    printed layout the hand-built text does not reproduce (the trailer, the all-clear sentences, an id wider than the
    constructed one) is exactly what a real report adds. Reds under: the figures left as attributes (the pre-chunk
    form, which consolidation drops, M15). A second correct implementation that stays green: storing each 0-d figure
    on a length-one dimension, provided the value and the name are unchanged.
    """
    import re

    import xarray as xr

    import hhemt.swmm_output_parser as sp
    from hhemt.utils import write_zarr

    inp = _run_tiny_model(tmp_path / "run")
    ds_nodes, ds_links = sp.retrieve_SWMM_outputs_as_datasets(inp, inp.with_suffix(".out"), family="tritonswmm")
    report = inp.with_suffix(".rpt").read_text(encoding="latin-1")
    # The report prints two "Continuity Error (%)" rows: runoff first (when there are subcatchments), flow routing last.
    printed_flow_error = float(re.findall(r"Continuity Error \(%\) \.+\s+(-?[\d.]+)", report)[-1])
    printed_flooding = float(re.search(r"Flooding Loss \.+\s+[\d.]+\s+([\d.]+)", report).group(1))

    assert set(ds_nodes.attrs) == {"flow_units", "analysis_end_datetime"}
    assert set(ds_links.attrs) == {"flow_units", "analysis_end_datetime"}
    assert float(ds_nodes["flow_continuity_error"]) == pytest.approx(printed_flow_error)
    assert float(ds_nodes["system_flooding"]) == pytest.approx(printed_flooding)

    for name, ds in (("nodes", ds_nodes), ("links", ds_links)):
        store = tmp_path / f"{name}.zarr"
        write_zarr(ds, store, 5)
        back = xr.open_zarr(store).load()
        for var in ds.data_vars:
            if ds[var].dtype.kind in "US":
                assert back[var].values.astype(str).tolist() == ds[var].values.astype(str).tolist(), var
            else:
                np.testing.assert_allclose(back[var].values, ds[var].values, equal_nan=True, err_msg=var)


@pytest.mark.parametrize("link_id", ["10.1", "0012", "1e3"])
@pytest.mark.parametrize(
    "title, n_tokens",
    [("Flow Classification Summary", 11), ("Conduit Surcharge Summary", 6), ("Pumping Summary", 10)],
)
def test_a_link_table_row_keeps_its_first_token_as_the_id(title, n_tokens, link_id):
    """NEW CAPABILITY. Input class: every link-kind report table (three of them) x link ids that parse as a number
    but are not the integer they print as (a decimal, a zero-padded one, an exponent).

    Property (the reader takes the first token of a row as its id): the
    frame is indexed by the id exactly as the row prints it, which is also the ``link_id`` the binary output carries.
    Required because the series arrive keyed by the binary output's raw ids; an id the table reader rewrites
    (``10.1`` to ``10``, ``0012`` to ``12``, ``1e3`` to ``1000``) joins to the wrong link or to none, and two ids
    that collide after rewriting make the outer join raise on a duplicate index.
    Reds under: the link branch of ``read_report_table`` passing the id through ``_clean_link_id``.
    A second correct implementation that stays green: indexing by the raw token and leaving any normalisation to the
    caller (the node tables already do exactly this).
    Reachability: a model whose link names are numeric-looking but not plain integers (``10.1`` / ``10.2`` branch
    names, zero-padded asset numbers) reaches it; the pump-control sample model with such link names is the artifact.
    """
    from hhemt.swmm_output_parser import read_report_table

    row = "  " + link_id.ljust(20) + " " + " ".join(["1.00"] * (n_tokens - 1))
    frame = read_report_table(title, [row], filepath="<test>")
    assert frame.index.tolist() == [link_id]


def test_a_node_table_row_and_a_plain_integer_link_id_are_unchanged():
    """Control for the test above, in a DIFFERENT correct state: node ids are read raw today, and an integer-looking
    link id such as the reference model's ``1444`` is the same string either way. Green before and after any fix, so
    the red above is about non-integer-looking ids and not about the helper."""
    from hhemt.swmm_output_parser import read_report_table

    node_row = "  " + "10.1".ljust(20) + " JUNCTION     0.10     0.20     0.30     0  01:00        0.20"
    assert read_report_table("Node Depth Summary", [node_row], filepath="<test>").index.tolist() == ["10.1"]
    link_row = "  " + "1444".ljust(20) + " " + " ".join(["1.00"] * 5)
    assert read_report_table("Conduit Surcharge Summary", [link_row], filepath="<test>").index.tolist() == ["1444"]


def test_the_raw_byte_identity_swmm_arm_still_parses_a_real_coupled_layout(tmp_path):
    """REGRESSION (red on the chunk tree, green before it). Input class: a base/test pair laid out as the coupled
    solver lays it out (``{sim}/swmm/hydraulics.inp`` and ``{sim}/out_tritonswmm/swmm/hydraulics.{rpt,out}``), both
    from a real stock SWMM run.

    Property: ``eda.raw_byte_identity.compare_swmm_raw`` returns its three booleans and no ``swmm_parse_error``.
    Required because the ``.rpt`` entry point of ``retrieve_SWMM_outputs_as_datasets`` is refused; the one
    caller outside the module still passes ``.rpt`` paths, and its blanket ``except Exception`` turns the new
    ``ProcessingError`` into a recorded datum, so the raw byte identity artifact silently loses both SWMM bits. A
    search for the removed reader NAMES does not find this retained function's removed ARGUMENT FORM.
    Reds under: the current two calls ``retrieve_SWMM_outputs_as_datasets(inp, rpt)``. A second correct
    implementation that stays green: passing the sibling ``.out`` through ``family="tritonswmm"``, or reading the two
    stores through any other entry point, provided no parse error is recorded and identical runs are reported identical.
    Class note: the file is outside the inventory's write set; this test is
    admissible only if the write set includes ``src/hhemt/eda/raw_byte_identity.py`` in the write set.
    """
    from hhemt.eda.raw_byte_identity import compare_swmm_raw

    outs = {}
    for side in ("base", "test"):
        sim = tmp_path / side
        inp = _run_tiny_model(sim / "scratch")
        (sim / "swmm").mkdir(parents=True)
        (sim / "swmm" / "hydraulics.inp").write_text(inp.read_text())
        coupled = sim / "out_tritonswmm" / "swmm"
        coupled.mkdir(parents=True)
        for suffix in (".rpt", ".out"):
            (coupled / f"hydraulics{suffix}").write_bytes(inp.with_suffix(suffix).read_bytes())
        outs[side] = sim / "out_tritonswmm"

    result = compare_swmm_raw(outs["base"], outs["test"])
    assert "swmm_parse_error" not in result, result.get("swmm_parse_error")
    assert result["swmm_nodes_b4b"] is True and result["swmm_links_b4b"] is True
    assert result["swmm_out_bytes_b4b"] is True
