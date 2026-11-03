"""Reviewer tests for the SWMM-tier names and the three parser rules.

The toolkit stores 50 SWMM, hydrology, runoff-model, water-quality and EDA variables, and
`final_surface_flood_volume`, without a unit token, and applies three parser rules: the Node Inflow Summary's last
field is split by the unit SWMM printed, the Node Flooding Summary's ponded column is a depth only where its header
says `Depth`, and the node tier's maximum flooding rate and the time of its maximum total inflow do not share the link
tier's names. The other tests construct small report text and check one node of the tracked reference report.
These tests are the other instrument: an INDEPENDENT reading of the SWMM report text (plain `str.split`, none of the
toolkit's parser) is compared with every node and every link of the tracked reference stores AND of a fresh parse, and
each parser rule is also driven end to end through the real entry point on a report whose header says the opposite of
the tracked one.

Every test neither compiles nor executes a solver, so all of them run on the laptop. The two real parses of the
3 MB reference report take about twenty seconds each.
"""

from __future__ import annotations

import re
import shutil
import warnings
import zipfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.constants import APP_NAME

REPO = Path(__file__).resolve().parents[1]
REF_DIR = Path(str(files(APP_NAME).parents[1])) / "test_data" / "swmm_refactoring_reference"
REF_INP = REF_DIR / "hydraulics.inp"
REF_RPT = REF_DIR / "hydraulics.rpt"
REF_HYDRO_RPT = REF_DIR / "hydro_runoff_summary.rpt"


# ---------------------------------------------------------------------------------------------
# An independent reader of the SWMM report text (no toolkit parser)
# ---------------------------------------------------------------------------------------------


def _is_rule(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) == {"-"}


def _table_rows(lines: list[str], title: str) -> list[list[str]]:
    """The whitespace-split rows of the table under `title`: after the SECOND dashed rule, to the first blank line."""
    start = next(i for i, ln in enumerate(lines) if ln.strip() == title)
    rules, rows = 0, []
    for ln in lines[start + 1 :]:
        if _is_rule(ln):
            rules += 1
            continue
        if rules < 2:
            continue
        if not ln.strip():
            break
        rows.append(ln.split())
    return rows


def _minutes(days: str, hhmm: str) -> float:
    hh, mm = hhmm.split(":")
    return int(days) * 1440 + int(hh) * 60 + int(mm)


@pytest.fixture(scope="module")
def report_lines() -> list[str]:
    return REF_RPT.read_text(encoding="latin-1").splitlines()


def _expected_from(lines: list[str]):
    """Per node: what the report text says every stored node-summary variable must hold, read with `str.split`."""
    inflow = {}
    for r in _table_rows(lines, "Node Inflow Summary"):
        has_ltr = r[-1] == "ltr"
        t = r[:-1] if has_ltr else r
        assert len(t) == 9, r
        inflow[t[0]] = dict(
            max_lateral_inflow=float(t[2]),
            max_total_inflow=float(t[3]),
            time_of_max_total_inflow=_minutes(t[4], t[5]),
            lateral_inflow_vol=float(t[6]),
            total_inflow_vol=float(t[7]),
            flow_balance_error=np.nan if has_ltr else float(t[8]),
            flow_balance_error_volume=float(t[8]) if has_ltr else np.nan,
        )
    flooding = {}
    for r in _table_rows(lines, "Node Flooding Summary"):
        assert len(r) == 7, r
        flooding[r[0]] = dict(
            hours_flooded=float(r[1]),
            max_flood_rate=float(r[2]),
            time_of_max_flood=_minutes(r[3], r[4]),
            tot_flooded_vol=float(r[5]),
            max_ponded_depth=float(r[6]),
        )
    return inflow, flooding


@pytest.fixture(scope="module")
def expected_node(report_lines):
    return _expected_from(report_lines)


_FLOOD_VARS = ("hours_flooded", "max_flood_rate", "time_of_max_flood", "tot_flooded_vol", "max_ponded_depth")


def _check_node_dataset(ds: xr.Dataset, inflow: dict, flooding: dict) -> None:
    """Every node of `ds` holds, for every variable the two tables define, exactly the number the text printed."""
    if "event_iloc" in ds.dims:
        ds = ds.isel(event_iloc=0)
    ids = list(ds["node_id"].values)
    assert set(inflow) == set(ids), "the stored node set is the Node Inflow Summary's node set"
    for var in next(iter(inflow.values())):
        got = ds[var].sel(node_id=list(inflow)).values.astype(float)
        want = np.array([inflow[n][var] for n in inflow])
        assert np.array_equal(got, want, equal_nan=True), (var, int(np.sum(~np.isclose(got, want, equal_nan=True))))
    for var in _FLOOD_VARS:
        got = ds[var].sel(node_id=list(inflow)).values.astype(float)
        want = np.array([flooding.get(n, {}).get(var, np.nan) for n in inflow])
        assert np.array_equal(got, want, equal_nan=True), (var, int(np.sum(~np.isclose(got, want, equal_nan=True))))


def _controls(inflow: dict, flooding: dict) -> None:
    """Must-fire controls: the fixture separates every confusion the decisions rule out."""
    fbe = [v["flow_balance_error"] for v in inflow.values()]
    fbv = [v["flow_balance_error_volume"] for v in inflow.values()]
    assert 0 < np.isfinite(fbe).sum() < len(inflow) and 0 < np.isfinite(fbv).sum() < len(inflow), "both kinds of row"
    assert np.isfinite(fbe).sum() + np.isfinite(fbv).sum() == len(inflow), "every row is exactly one kind"
    assert 0 < len(flooding) < len(inflow), "some but not all nodes flood"
    both = [n for n in flooding if n in inflow]
    # The flooding rate is a different number from the total inflow. MEASURED: the two TIMES are equal on all
    # 14 flooded nodes of this report, so the reference cannot tell the two node time sources apart; the
    # edited-report test below carries the control that does.
    assert any(flooding[n]["max_flood_rate"] != inflow[n]["max_total_inflow"] for n in both)


def test_independent_text_reading_equals_the_tracked_reference_node_store(expected_node):
    """Property: for EVERY node, the tracked reference node summary holds the numbers the report text printed.

    Class: all 1,135 nodes and all twelve variables of the three rules, including nodes that flood and nodes that
    do not.
    Why required: the other reference check reads one node of the node-and-flooding intersection, and the
    reference store is the fixture every refactoring test pins the parser to, so a store whose `max_flood_rate`
    came from the wrong table, whose litre rows landed in the percent variable, or whose time was taken from the
    flooding table would sit in the repository as the oracle.
    Kills it: swapping the two node time sources; `max_flood_rate` taken from the total-inflow column; `ltr` rows
    stored in `flow_balance_error`; a ponded value stored for a node that did not flood.
    Passes under: a store regenerated by a different parser, provided it holds the printed numbers.
    """
    inflow, flooding = expected_node
    _controls(inflow, flooding)
    ds = xr.open_zarr(REF_DIR / "SWMM_node_summary.zarr")
    _check_node_dataset(ds, inflow, flooding)


@pytest.fixture(scope="module")
def reference_parse():
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return retrieve_SWMM_outputs_as_datasets(REF_INP, REF_RPT)


def test_independent_text_reading_equals_a_fresh_parse_of_the_reference_report(expected_node, reference_parse):
    """Property: the parser, run on the tracked report, returns for every node the numbers the text printed.

    Class: the same 1,135 nodes and twelve variables as the stored-store test, through the real entry point.
    Why required: the stored store and the parser could drift apart behind a loose comparison; this holds the
    parser itself to the text.
    Kills it: any of the mutations the stored-store test kills, applied to the parser; a node-id mangling.
    Passes under: a parser that reads the tables by position rather than by header, provided the numbers agree.
    """
    inflow, flooding = expected_node
    ds_nodes, _ = reference_parse
    _check_node_dataset(ds_nodes, inflow, flooding)


def test_independent_text_reading_equals_the_link_tier_for_every_link(report_lines, reference_parse):
    """Property: for EVERY link, `max_flow`, `time_of_max_flow`, `max_velocity` and the two ratios are the text's.

    Class: all 1,146 links, in the tracked reference store and in a fresh parse.
    Why required: the link tier holds the names `max_flow` and `time_of_max_flow`; a link that took the node
    table's number, or a time read from the wrong column, would be named correctly and be wrong.
    Kills it: link `max_flow` taken from the node flooding table; `time_of_max_flow` from the node inflow table;
    `max_velocity` and `max_over_full_flow` swapped.
    Passes under: a parser that reads the Link Flow Summary by position.
    """
    rows = _table_rows(report_lines, "Link Flow Summary")
    assert len(rows) == 1146, len(rows)
    want = {}
    for r in rows:
        assert len(r) in (6, 8), r  # a conduit prints the two ratios; an orifice, weir or pump does not
        want[r[0]] = dict(
            max_flow=float(r[2]),
            time_of_max_flow=_minutes(r[3], r[4]),
            max_velocity=float(r[5]),
            max_over_full_flow=float(r[6]) if len(r) == 8 else np.nan,
            max_over_full_depth=float(r[7]) if len(r) == 8 else np.nan,
        )
    assert any(len(r) == 6 for r in rows) and any(len(r) == 8 for r in rows), "control: both row shapes occur"
    assert any(v["max_flow"] > 0 for v in want.values()), "control: the report has non-zero flows"
    stored = xr.open_zarr(REF_DIR / "SWMM_link_summary.zarr").isel(event_iloc=0)
    _, parsed = reference_parse
    for label, ds in (("stored", stored), ("parsed", parsed)):
        if "event_iloc" in ds.dims:
            ds = ds.isel(event_iloc=0)
        for var in next(iter(want.values())):
            # The two ratios are printed for conduits only; what the toolkit stores for an orifice, weir or pump
            # (which print neither) is outside what the parser reads, so those rows are compared on the other three.
            ids = [i for i, v in want.items() if not (var.startswith("max_over_full") and np.isnan(v[var]))]
            got = ds[var].sel(link_id=ids).values.astype(float)
            exp = np.array([want[i][var] for i in ids])
            assert np.array_equal(got, exp, equal_nan=True), (label, var)


def test_the_hydrology_summary_names_depth_and_volume_apart_and_holds_the_printed_numbers():
    """Property: `total_runoff_depth` is the mm column and `total_runoff_volume` the 10^6 ltr column, per subcatchment.

    Class: every subcatchment of the tracked hydrology-only report and all ten data columns.
    Why required: removing the suffixes made `total_runoff_mm` and `total_runoff_10e6_ltr` collide on one name;
    it is resolved by a word, and a swap of the two words would keep every test that only checks names.
    Kills it: depth and volume swapped; any column shifted by one; the collision resolved by dropping a column.
    Passes under: a parser that reads the table by header text.
    """
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    lines = REF_HYDRO_RPT.read_text(encoding="latin-1").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.strip() == "Subcatchment Runoff Summary")
    unit_row = next(ln for ln in lines[head + 1 : head + 12] if ln.strip().startswith("Subcatchment ") and "mm" in ln)
    assert unit_row.split()[1:] == ["mm"] * 7 + ["10^6", "ltr", "CMS"], unit_row  # the columns' units, from the text
    rows = _table_rows(lines, "Subcatchment Runoff Summary")
    names = (
        "total_precip total_runon total_evap total_infil imperv_runoff perv_runoff "
        "total_runoff_depth total_runoff_volume peak_runoff runoff_coeff"
    ).split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = parse_hydrology_rpt_summary(REF_HYDRO_RPT, REF_INP)
    assert len(rows) > 10
    for name in names:
        assert name in ds.data_vars, name
        got = ds[name].sel(subcatchment_id=[r[0] for r in rows]).values.astype(float)
        want = np.array([float(r[1 + names.index(name)]) for r in rows])
        assert np.array_equal(got, want), name
    assert any(float(r[7]) != float(r[8]) for r in rows), "control: depth and volume differ somewhere"
    for retired_suffix in ("_mm", "_10e6_ltr", "_cms"):
        assert not [v for v in ds.data_vars if str(v).endswith(retired_suffix)], retired_suffix


# ---------------------------------------------------------------------------------------------
# The ponded-column rule end to end: a report whose header says the opposite of the tracked one
# ---------------------------------------------------------------------------------------------

_DW_THIRD = "Hours       Rate    Occurrence      Volume     Depth"
_KW_THIRD = "Hours       Rate    Occurrence      Volume    Volume"


_FLOOD_ROW = "  E143257                 0.02     0.988      0  00:00       0.014     1.041"
_INFLOW_ROW = "  E143258              JUNCTION     1.045    1.248     0  00:00      0.0403      0.0416      13.059"


def _edited_report(tmp_path: Path) -> tuple[Path, list[str]]:
    """The tracked report with FOUR edits, nothing else: the Node Flooding Summary's third header line (and the
    fourth's unit word) as SWMM prints them under a non-Dynamic-Wave routing; node E143257's flooding-table time
    set to 9 minutes (its inflow-table time stays 0); node E143258's inflow-table time set to 5 minutes (its
    flooding-table time stays 0). The reference report cannot tell the two node time sources apart (every flooded
    node has the same time in both tables), so the times are edited apart here."""
    text = REF_RPT.read_text(encoding="latin-1")
    assert text.count(_DW_THIRD) == 1, "control: the tracked report is a Dynamic-Wave report with one such header"
    assert text.count("Node                 Flooded       CMS   days hr:min    10^6 ltr    Meters") == 1
    assert text.count(_FLOOD_ROW) == 1 and text.count(_INFLOW_ROW) == 1
    text = text.replace(_DW_THIRD, _KW_THIRD)
    text = text.replace(
        "Node                 Flooded       CMS   days hr:min    10^6 ltr    Meters",
        "Node                 Flooded       CMS   days hr:min    10^6 ltr   1000 m3",
    )
    text = text.replace(_FLOOD_ROW, _FLOOD_ROW.replace("0  00:00", "0  00:09"))
    text = text.replace(_INFLOW_ROW, _INFLOW_ROW.replace("0  00:00", "0  00:05"))
    out = tmp_path / "hydraulics.rpt"
    out.write_text(text, encoding="latin-1")
    return out, text.splitlines()


def test_d15_and_d16_through_the_real_parse_of_an_edited_report(tmp_path):
    """Property: through the real parse, a `Volume`-headed report stores NaN `max_ponded_depth` for EVERY node
    and warns, and a node whose two table times differ carries `time_of_max_flood` from the FLOODING table
    and `time_of_max_total_inflow` from the INFLOW table; every other node variable still holds the printed number.

    Class: the tracked 3 MB report, edited as `_edited_report` says, parsed end to end and compared with an
    independent reading of the EDITED text for all 1,135 nodes and all twelve variables.
    Why required: the other tests exercise the helpers on constructed text; nothing else drives the decision through
    `return_swmm_outputs`, where the header is read from the file and the rule is applied before the join.
    The tracked report prints the same time in both tables for every flooded node, so a parser that took
    `time_of_max_total_inflow` from the flooding table (or the reverse) passes every reference-report check.
    Kills it: the ponded-column rule not wired into `return_swmm_outputs` or applied to the wrong frame; a rule
    testing the line for the word `Volume` (the unedited Dynamic-Wave report then stores NaN, see the control);
    swapped node time sources; the rule changing a neighbouring column.
    Passes under: a parser that reads the header word with a regex, or applies the rule per node.
    """
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    rpt, lines = _edited_report(tmp_path)
    try:
        with pytest.warns(UserWarning, match="max_ponded_depth"):
            ds_nodes, _ = retrieve_SWMM_outputs_as_datasets(REF_INP, rpt)
    finally:
        rpt.unlink()
    inflow, flooding = _expected_from(lines)
    # the independent reading of the edited text: the ponded column is not a depth under `Volume`
    assert lines[next(i for i, ln in enumerate(lines) if "Node Flooding Summary" in ln) + 7].split()[-1] == "Volume"
    flooding = {n: {**d, "max_ponded_depth": np.nan} for n, d in flooding.items()}
    assert flooding["E143257"]["time_of_max_flood"] == 9 and inflow["E143257"]["time_of_max_total_inflow"] == 0
    assert flooding["E143258"]["time_of_max_flood"] == 0 and inflow["E143258"]["time_of_max_total_inflow"] == 5
    _check_node_dataset(ds_nodes, inflow, flooding)
    assert np.isnan(ds_nodes["max_ponded_depth"].values.astype(float)).all()
    # control: the unedited Dynamic-Wave report keeps its 14 printed depths in the tracked store
    assert int(xr.open_zarr(REF_DIR / "SWMM_node_summary.zarr")["max_ponded_depth"].notnull().sum()) == 14


@pytest.mark.parametrize(
    "third, fourth, expected",
    [
        (_DW_THIRD, "Node                 Flooded       CMS   days hr:min    10^6 ltr    Meters", "Depth"),
        (_DW_THIRD, "Node                 Flooded       CFS   days hr:min    10^6 gal      Feet", "Depth"),
        (_KW_THIRD, "Node                 Flooded       CMS   days hr:min    10^6 ltr   1000 m3", "Volume"),
        (_KW_THIRD, "Node                 Flooded       CFS   days hr:min    10^6 gal  1000 ft3", "Volume"),
    ],
)
def test_d15_the_third_header_line_decides_and_the_fourth_line_does_not(third, fourth, expected):
    """Property: the field read is the LAST field of the THIRD header line, for both unit systems and both routings.

    Class: four complete four-line headers as SWMM's `statsrpt.c` prints them (SI and US, Dynamic Wave and not),
    including the fourth line, whose last token is `Meters`, `Feet`, `m3` or `ft3`.
    Why required: the other tests' constructed text carries a third header line and a fourth that always ends
    `Meters`; a rule that read the last line of the header block, or the last field of the line AFTER the one
    the rule names, would return the unit word and pass a fixture whose fourth line never changes.
    Kills it: reading header line 4 (or the last non-blank line before the rule); reading the first field of
    line 3; testing line 3 for `Volume`.
    Passes under: a rule that takes the third line's last token by regex.
    """
    from hhemt.swmm_output_parser import node_flooding_ponded_field

    report = [
        "  Node Flooding Summary\n",
        "  *********************\n",
        "  \n",
        "  Flooding refers to all water that overflows a node, whether it ponds or not.\n",
        "  " + "-" * 74 + "\n",
        "                                                             Total   Maximum\n",
        "                                 Maximum   Time of Max       Flood    Ponded\n",
        "                        " + third + "\n",
        "  " + fourth + "\n",
        "  " + "-" * 74 + "\n",
        "  N1                      0.02     0.988      0  00:00       0.014     1.041\n",
        "  \n",
    ]
    assert node_flooding_ponded_field(report) == expected


def test_d15_no_flooded_node_means_no_warning_and_no_unreadable_header_ever_yields_a_depth():
    """Property: a report with no flooded node warns about nothing; a report with rows but no readable header
    never keeps a value as a depth.

    Class: an empty Node Flooding frame with the 'No nodes were flooded.' report; a populated frame whose header
    cannot be read.
    Why required: the unit of the column is unknown when the header is unreadable, and publishing a number
    under `units: m` on a guess is the defect the rule prevents; and a warning on every report that floods no node
    would be noise the Errors-and-Warnings sidebar surfaces.
    Kills it: warning on an empty frame; keeping the positional value when the header is absent.
    Passes under: a parser that raises on an unreadable header instead of storing NaN.
    """
    import pandas as pd

    from hhemt.swmm_output_parser import _apply_ponded_depth_rule, node_flooding_ponded_field

    none_flooded = ["  Node Flooding Summary\n", "  *********************\n", "  \n", "  No nodes were flooded.\n"]
    assert node_flooding_ponded_field(none_flooded) is None
    empty = pd.DataFrame(columns=["node_id", "max_ponded_depth"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _apply_ponded_depth_rule(empty, None, "x.rpt")

    rows = pd.DataFrame({"node_id": ["a", "b"], "max_ponded_depth": ["1.041", "1.325"]})
    with pytest.warns(UserWarning, match="max_ponded_depth"):
        _apply_ponded_depth_rule(rows, node_flooding_ponded_field(["not a report\n"]), "x.rpt")
    assert pd.to_numeric(rows["max_ponded_depth"], errors="coerce").isna().all()


def test_d14_an_empty_node_inflow_table_gives_both_variables_and_no_error():
    """Property: with no Node Inflow Summary rows the split still produces both variables, empty.

    Class: a `.out` run whose `.rpt` is missing, which hands the parser an empty section.
    Why required: the split computes a per-row flag and a row-count check; the empty case is the one
    input where a list-of-rows guard can raise or drop a column, and downstream consolidation expects both names.
    Kills it: an IndexError or ValueError on an empty section; a missing `flow_balance_error_volume` column.
    Passes under: an implementation that returns the shared reader's frame with a column added.
    """
    from hhemt.constants import LST_COL_HEADERS_NODE_FLOW_SUMMARY
    from hhemt.swmm_output_parser import format_node_inflow_summary

    df = format_node_inflow_summary([], list(LST_COL_HEADERS_NODE_FLOW_SUMMARY))
    assert len(df) == 0
    assert {"flow_balance_error", "flow_balance_error_volume"} <= set(df.columns)


def test_d14_a_zero_or_negative_value_and_a_node_id_that_looks_like_a_unit_word_split_by_the_token_only():
    """Property: the split follows the `ltr` TOKEN, never the value: zero, negative and large values go where
    the token puts them, and a node named `ltr_1` is not read as litres.

    Class: four rows of a constructed Node Inflow Summary: `-0.000 ltr`, `-3.5` (percent, negative), `100000.000`,
    and a node id beginning `ltr`.
    Why required: SWMM prints litres when the node's outflow volume is below one cubic foot, which includes
    a zero error and a negative one; a rule keyed on the sign or the magnitude would misfile exactly those rows.
    Kills it: a split on `value < 1` or `value >= 0`; a rule testing the row text for the substring `ltr`.
    Passes under: a rule that reads the last token of each row.
    """
    from hhemt.constants import LST_COL_HEADERS_NODE_FLOW_SUMMARY
    from hhemt.swmm_output_parser import format_node_inflow_summary, return_lines_for_section_of_rpt

    header = [
        "  Node Inflow Summary\n",
        "  *******************\n",
        "  \n",
        "  " + "-" * 97 + "\n",
        "                                  Maximum  Maximum                  Lateral       Total        Flow\n",
        "                                  Lateral    Total  Time of Max      Inflow      Inflow     Balance\n",
        "                                   Inflow   Inflow   Occurrence      Volume      Volume       Error\n",
        "  Node                 Type           CMS      CMS  days hr:min    10^6 ltr    10^6 ltr     Percent\n",
        "  " + "-" * 97 + "\n",
    ]
    body = [
        "  A_ZERO_LTR           JUNCTION     0.000    0.000     0  00:01           0           0      -0.000 ltr\n",
        "  B_NEG_PCT            JUNCTION     0.001    0.002     0  00:01    4.16e-06    4.20e-06      -3.500\n",
        "  C_BIG_PCT            JUNCTION     0.001    0.002     0  00:01    4.16e-06    4.20e-06  100000.000\n",
        "  ltr_1                JUNCTION     0.001    0.002     0  00:01    4.16e-06    4.20e-06       7.250\n",
        "  \n",
    ]
    rows = return_lines_for_section_of_rpt("Node Inflow Summary", lines=header + body)
    assert len(rows) == 4, rows
    df = format_node_inflow_summary(rows, list(LST_COL_HEADERS_NODE_FLOW_SUMMARY)).set_index("node_id")
    pct = df["flow_balance_error"].astype(float)
    vol = df["flow_balance_error_volume"].astype(float)
    assert vol["A_ZERO_LTR"] == 0.0 and np.isnan(pct["A_ZERO_LTR"])
    assert pct["B_NEG_PCT"] == -3.5 and np.isnan(vol["B_NEG_PCT"])
    assert pct["C_BIG_PCT"] == 100000.0 and np.isnan(vol["C_BIG_PCT"])
    assert pct["ltr_1"] == 7.25 and np.isnan(vol["ltr_1"])


# ---------------------------------------------------------------------------------------------
# The tiers share no name, over the committed goldens and the parser output
# ---------------------------------------------------------------------------------------------

_UNIT_TOKEN = re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")


def _extract(kind: str, dest: Path) -> Path:
    from tests.fixtures.golden_trees import extract_golden

    return extract_golden(kind, dest / kind)


def _tier_groups(root: Path, tier: str) -> list[tuple[str, xr.Dataset]]:
    out = []
    for tree in sorted([*root.rglob("analysis_datatree.zarr"), *root.rglob("sensitivity_datatree.zarr")]):
        for group in sorted(p.parent.relative_to(tree).as_posix() for p in tree.rglob(f"{tier}/zarr.json")):
            out.append((f"{tree.name}:{group}", xr.open_zarr(tree, group=group, consolidated=False)))
    return out


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_golden_swmm_node_and_link_tiers_carry_the_ruled_names_and_split_the_flow_balance_error(golden, tmp_path):
    """Property: in every regenerated golden's node tier the flow-balance pair, `max_flood_rate`,
    `time_of_max_total_inflow` and `max_ponded_depth` are present and `max_flow`, `time_of_max_flow` and
    `flow_balance_error_percent` are absent; the link tier carries `max_flow`, `time_of_max_flow` and
    `max_velocity`; and no row of a node tier holds BOTH a percent and a litre error.

    Class: every `swmm_node` and `swmm_link` group of both golden trees (the coupled and the SWMM-only tier).
    Why required: the goldens are the fixture the migration test and the validation read; a golden that still
    carries the shared names would let V0028's test pass against an input the parser can no longer produce.
    Kills it: a stale golden; a generator that fills both flow-balance variables on one row, or neither; the node tier
    keeping a link-tier name.
    Passes under: a golden regenerated by a different generator.
    """
    root = _extract(golden, tmp_path)
    try:
        nodes = _tier_groups(root, "swmm_node")
        links = _tier_groups(root, "swmm_link")
        assert len(nodes) >= 2 and len(links) >= 2, (len(nodes), len(links))  # control: the walk reaches both tiers
        for label, ds in nodes:
            names = set(ds.data_vars)
            assert {
                "flow_balance_error",
                "flow_balance_error_volume",
                "max_flood_rate",
                "time_of_max_total_inflow",
                "max_ponded_depth",
            } <= names, label
            assert not ({"max_flow", "time_of_max_flow", "flow_balance_error_percent", "max_velocity"} & names), label
            pct = ds["flow_balance_error"].values.astype(float)
            vol = ds["flow_balance_error_volume"].values.astype(float)
            assert not (np.isfinite(pct) & np.isfinite(vol)).any(), f"{label}: a row holds both units"
            assert np.isfinite(pct).any() and np.isfinite(vol).any(), f"{label}: the golden exercises both kinds"
        for label, ds in links:
            names = set(ds.data_vars)
            assert {"max_flow", "time_of_max_flow", "max_velocity"} <= names, label
            assert not ({"max_flood_rate", "time_of_max_total_inflow", "flow_balance_error"} & names), label
    finally:
        shutil.rmtree(tmp_path / golden, ignore_errors=True)


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_no_swmm_tier_array_of_a_golden_carries_a_unit_token(golden):
    """Property: no array under a `swmm_node` or `swmm_link` group of a golden has a unit-token name.

    Class: every array leaf under those groups in the committed zips, matched with the unit-token pattern,
    whose must-fire control is a constructed name carrying each token.
    Why required: the text census reads text; the zips are binary, so only a read of their array
    names shows a golden that was not regenerated.
    Kills it: a stale zip; a generator writing `flow_cms` or `depth_m` into a SWMM group.
    Passes under: a golden regenerated by a different generator.
    """
    for token_name in ("flow_cms", "depth_m", "head_m", "max_ponded_depth_m", "time_of_max_flow_min", "x_10e6_ltr"):
        assert _UNIT_TOKEN.search(token_name), token_name  # control: the matcher fires on the property, per token
    for clean in ("max_flood_rate", "hours_flooded", "capacity_setting_last", "flow_balance_error_volume"):
        assert not _UNIT_TOKEN.search(clean), clean
    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    leaves = []
    for name in z.namelist():
        parts = name.split("/")
        if name.endswith("/zarr.json") and ("swmm_node" in parts or "swmm_link" in parts):
            leaves.append(parts[-2])
    assert len(leaves) > 40, len(leaves)  # control: the walk reaches the SWMM arrays at all
    # `type` and the coordinates have no token; `event_iloc` and `node_id` are not matched by the pattern either
    assert [n for n in leaves if _UNIT_TOKEN.search(n)] == []


def test_no_name_is_shared_by_the_triton_tier_the_swmm_node_tier_and_the_swmm_link_tier(reference_parse):
    """Property: the TRITON summary, the SWMM node dataset and the SWMM link dataset write
    pairwise disjoint variable names, apart from the descriptive `type`.

    Class: the variables of a real parse of the reference report (node and link) and of a real TRITON summary.
    Why required: `_CF_VARIABLE_MAP` is applied by name for every mode, so a name two tiers write gets one
    description for two quantities; the node and link tiers must not share one, and dropping the unit tokens
    could create a collision (`depth`, `flow`, `velocity`, `head`).
    Kills it: a node-tier `max_flow` or `time_of_max_flow`; a TRITON name equal to a SWMM name; two tiers both
    writing `velocity_max`-shaped names.
    Passes under: tiers that write the same name with an identical meaning declared in the CF table's overrides.
    """
    from hhemt.process_simulation import TRITON_RAW_VAR_BY_PREFIX, summarize_triton_simulation_results

    ds_nodes, ds_links = reference_parse
    node, link = set(ds_nodes.data_vars), set(ds_links.data_vars)
    dims = ("timestep", "y", "x")
    h = np.random.default_rng(1).uniform(0.2, 1.0, size=(3, 4, 5))  # a 10 m grid, as the summary function requires
    tseries = xr.Dataset(
        {
            "max_flood_depth": (dims, h),
            "flood_depth": (dims, h),
            "unit_discharge_x": (dims, h),
            "unit_discharge_y": (dims, h),
        },
        coords={
            "timestep": [0.0, 10.0, 20.0],
            "y": [35.0, 25.0, 15.0, 5.0],
            "x": [5.0, 15.0, 25.0, 35.0, 45.0],
        },
    )
    triton = set(summarize_triton_simulation_results(tseries, 0, 10.0, chunksize_mb=0.001).data_vars) | set(
        TRITON_RAW_VAR_BY_PREFIX.values()
    )
    assert node and link and triton and "max_flow" in link and "max_flood_rate" in node  # control: all reachable
    assert (node & link) - {"type"} == set(), (node & link) - {"type"}
    assert (triton & node) == set() and (triton & link) == set(), (triton & node, triton & link)


# ---------------------------------------------------------------------------------------------
# The census of unit-token names across the tree, with a control that varies the property
# ---------------------------------------------------------------------------------------------

#: The 50 SWMM, hydrology and EDA names that carry a unit token, and `final_surface_flood_volume_m3`, quoted here
#: BECAUSE this test is their census; this file and the two earlier reviewer files are therefore excluded from it.
_LAYOUT_26_NAMES = (
    "depth_m depth_m_last depth_m_max head_m head_m_last head_m_max inflow_flow_cms inflow_flow_cms_last "
    "inflow_flow_cms_max flooding_cms flooding_cms_last flooding_cms_max max_lateral_inflow_cms max_total_inflow_cms "
    "max_flow_cms time_of_max_flood_min time_of_max_flow_min tot_flooded_vol_10e6_ltr lateral_inflow_vol_10e6_ltr "
    "total_inflow_vol_10e6_ltr max_ponded_depth_m flow_balance_error_percent flow_cms velocity_mps link_depth_m "
    "flow_cms_last flow_cms_max velocity_mps_last velocity_mps_max link_depth_m_last link_depth_m_max "
    "max_velocity_mps total_precip_mm total_runon_mm total_evap_mm total_infil_mm imperv_runoff_mm perv_runoff_mm "
    "total_runoff_mm total_runoff_10e6_ltr peak_runoff_cms inflow_cms total_inflow_volume_m3 "
    "pollutant_time_of_max_conc_min outfall_total_vol_10e6_ltr qual_routing_continuity_error_perc "
    "runoff_qual_continuity_error_perc max_abs_diff_m rmse_wetted_m pct_diff_p95 final_surface_flood_volume_m3"
).split()
_CENSUS = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(sorted(_LAYOUT_26_NAMES, key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
)
_CENSUS_SUFFIXES = {".py", ".md", ".rst", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml", ".j2"}
_CENSUS_SKIP = (
    "docs/reference/bundles",
    "tests/fixtures/legacy_layouts",
    "__pycache__",
    "/.git/",
    "report_templates/vendor",
)
#: The files allowed to quote a forbidden name: the migration and its two tests, plus the two files that must name
#: them (the vocabulary-freeze manifest and the `test_cf_conventions.py` soundness test) and the three reviewer files
#: that assert absence. Measured 2026-11-02: the census is red on the freeze manifest unless it is allowed, because
#: that manifest lists the forbidden tokens.
_MIGRATION_AND_ITS_TESTS = (
    "src/hhemt/version_migration/versions/V0028__flood_depth_and_unit_discharge_names.py",
    "tests/test_version_migration_v0028.py",
    "tests/test_review_v0028_migration.py",
)
_ALSO_ALLOWED = (
    "scripts/vocabulary_freeze.yaml",
    "tests/test_cf_conventions.py",
    "tests/test_review_unit_discharge_relabel.py",
    "tests/test_review_flood_depth_relabel.py",
    "tests/test_review_swmm_tier_names.py",
)


def _census(root: Path, subdirs=("src", "scripts", "tests", "docs", "test_data")) -> list[str]:
    hits = []
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if (
                not p.is_file()
                or p.suffix not in _CENSUS_SUFFIXES
                or any(s in rel for s in _CENSUS_SKIP)
                or rel in _MIGRATION_AND_ITS_TESTS
                or rel in _ALSO_ALLOWED
            ):
                continue
            for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                m = _CENSUS.search(line)
                if m:
                    hits.append(f"{rel}:{n}: {m.group(1)}")
    return hits


def test_census_matcher_fires_on_a_planted_retired_name_and_not_on_its_replacement(tmp_path):
    """Property: the census flags a forbidden name used as a JSON key, a string and a prose word, and passes the new
    name.

    Class: a planted tree with one line per form and one line carrying only a layout-28 name or a longer name that
    CONTAINS a forbidden name as a prefix of another identifier.
    Why required: a census that returns zero is evidence only if the same matcher fires on a violating input that
    differs from the live one in the property under suspicion (a name in a manifest or a doc, not a code token).
    Kills it: a matcher narrowed to code files or to double quotes; a matcher widened to substrings, which would
    flag `depth_max` through `depth_m`.
    Passes under: a matcher implemented over an AST instead of text.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "a.py").write_text('x = ds["flow_cms"]\n')
    (tmp_path / "tests" / "m.json").write_text('{"depth_m_max": 0.1}\n')
    (tmp_path / "docs" / "c.md").write_text("The `max_ponded_depth_m` column is a depth.\n")
    (tmp_path / "docs" / "ok.md").write_text("`depth_max` and `max_flood_rate` and `flood_velocity_min_depth_m`.\n")
    found = _census(tmp_path, subdirs=("src", "docs", "tests"))
    assert sorted(h.split(":")[0] for h in found) == ["docs/c.md", "src/a.py", "tests/m.json"], found


def test_no_layout_26_unit_token_name_remains_outside_the_files_that_quote_them_on_purpose():
    """Property: no text file under src, scripts, tests, docs or test_data
    holds a unit-token variable name as a whole token, outside the three reviewer files that assert their absence.

    Class: every text file of those five trees (json included, so the figure manifests and the reference stores'
    `zarr.json` are read), 51 names.
    Why required: the rename is mechanical over 22 `src` files, 23 test files, 14 manifests and a reference
    store; one missed site is a name the writer no longer produces and a reader still asks for.
    Kills it: any surviving unit-token name in any file of those trees.
    Passes under: a tree in which the name is rewritten or the line removed.
    """
    assert _census(REPO) == []


def test_the_exclusion_list_admits_the_migration_files_only_and_flags_a_retired_name_anywhere_else(tmp_path):
    """Property: the census excludes the migration and its two tests and nothing near them.

    Class: one unit-token SWMM name planted in each allowed path (not flagged) and in four near-miss paths (flagged).
    Why required: `V0028` quotes the 50 names on purpose; the exclusion is exact paths so a stray occurrence
    beside it still reds.
    Kills it: an exclusion by directory or prefix; a dropped file.
    Passes under: any exclusion exact on the relative path.
    """
    name = "flow" + "_cms = 1\n"
    flagged = [
        "src/hhemt/version_migration/versions/V0027__other.py",
        "src/hhemt/version_migration/context.py",
        "tests/test_version_migration_v0027.py",
        "tests/test_review_v0028_migration_extra.py",
    ]
    for rel in [*_MIGRATION_AND_ITS_TESTS, *flagged]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    assert sorted(h.split(":")[0] for h in _census(tmp_path)) == sorted(flagged)
