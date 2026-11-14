"""The three SWMM parser rules for the Node Inflow Summary, the Node Flooding Summary and the tier names.

* Flow balance error: the Node Inflow Summary's last field is a percent where SWMM printed a number alone and an
  absolute error in litres where it printed the number followed by `ltr`. The parser stores them in `flow_balance_error`
  (percent) and `flow_balance_error_volume` (litres), the other NaN on that row.
* Ponded depth: the Node Flooding Summary's sixth column is a depth in meters under Dynamic Wave routing and a ponded
  volume otherwise, and SWMM says which in the LAST field of the table's third header line (`Depth` or
  `Volume`). The parser stores `max_ponded_depth` only for `Depth`; otherwise NaN and a warning.
* Tier names: the node-tier maximum flooding rate and the time of the maximum total inflow are `max_flood_rate` and
  `time_of_max_total_inflow`; the link tier's are `max_flow` and `time_of_max_flow`.

The first two are tested on constructed report text. The third, and the count check, run on the tracked
reference report `test_data/swmm_refactoring_reference/hydraulics.rpt`, parsed once (about twenty seconds).
No solver runs.
"""

from __future__ import annotations

import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from hhemt.constants import APP_NAME, LST_COL_HEADERS_NODE_FLOW_SUMMARY
from hhemt.swmm_output_parser import (
    _apply_ponded_depth_rule,
    format_node_inflow_summary,
    format_rpt_section_into_dataframe,
    node_flooding_ponded_field,
    retrieve_SWMM_outputs_as_datasets,
    return_lines_for_section_of_rpt,
)

REF_DIR = Path(str(files(APP_NAME).parents[1])) / "test_data" / "swmm_refactoring_reference"
REF_INP = REF_DIR / "hydraulics.inp"
REF_RPT = REF_DIR / "hydraulics.rpt"

_RULE = "  " + "-" * 74 + "\n"
_INFLOW_HEADER = [
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


def _flooding_report(third_header_line: str) -> list[str]:
    return [
        "  Node Flooding Summary\n",
        "  *********************\n",
        "  \n",
        "  Flooding refers to all water that overflows a node, whether it ponds or not.\n",
        _RULE,
        "                                                             Total   Maximum\n",
        "                                 Maximum   Time of Max       Flood    Ponded\n",
        third_header_line,
        "  Node                 Flooded       CMS   days hr:min    10^6 ltr    Meters\n",
        _RULE,
        "  N1                      0.02     0.988      0  00:00       0.014     1.041\n",
        "  N2                      0.03     1.299      0  00:00       0.017     1.320\n",
        "  \n",
    ]


_DYNWAVE_THIRD = "                        Hours       Rate    Occurrence      Volume     Depth\n"
_KINWAVE_THIRD = "                        Hours       Rate    Occurrence      Volume    Volume\n"


def test_d14_a_row_ending_ltr_goes_to_the_volume_variable_and_a_row_without_it_to_the_percent_variable():
    """Property: one row ending `ltr` and one not put each value in its own variable and NaN in the other.

    Class: REGRESSION. Why required: the shared section reader dropped the `ltr` token, so one column
    held a percent on some rows and litres on others under a name ending `_percent`.
    Kills it: the old behaviour (one variable holding both), swapping the two variables, dropping the unit.
    Second correct implementation: any parser that reads the token before discarding it; it passes.
    Must-fire control: the SAME text through the shared reader alone yields a single column holding both values,
    so the fixture does distinguish the two behaviours.
    """
    text = _INFLOW_HEADER + [
        "  J_PCT                JUNCTION     0.001    0.002     0  00:01    4.16e-06    4.20e-06       2.930\n",
        "  J_LTR                JUNCTION     0.000    0.000     0  00:01    1.26e-05    1.26e-05      12.563 ltr\n",
        "  \n",
    ]
    rows = return_lines_for_section_of_rpt("Node Inflow Summary", lines=text)
    assert len(rows) == 2, rows
    merged = format_rpt_section_into_dataframe(rows, list(LST_COL_HEADERS_NODE_FLOW_SUMMARY)).set_index("node_id")
    assert merged.loc["J_PCT", "flow_balance_error"] == "2.930"
    assert merged.loc["J_LTR", "flow_balance_error"] == "12.563"

    df = format_node_inflow_summary(rows, list(LST_COL_HEADERS_NODE_FLOW_SUMMARY)).set_index("node_id")
    assert float(df.loc["J_PCT", "flow_balance_error"]) == 2.930
    assert np.isnan(float(df.loc["J_PCT", "flow_balance_error_volume"]))
    assert np.isnan(float(df.loc["J_LTR", "flow_balance_error"]))
    assert float(df.loc["J_LTR", "flow_balance_error_volume"]) == 12.563


def test_d15_the_last_field_of_the_third_header_line_decides_and_the_word_volume_elsewhere_does_not():
    """Property: `Depth` as the last field stores the value; `Volume` as the last field gives NaN and a warning.

    Class: REGRESSION. Why required: under Kinematic Wave the sixth column is a ponded VOLUME, so reading it
    as meters mislabels it; and a test for the word `Volume` anywhere on the line is wrong, because the
    Dynamic-Wave line also contains `Volume` (the flood-volume column's heading).
    Kills it: a rule testing the whole line for `Volume` (NaN on the Dynamic-Wave form), a rule that never
    checks (a value on the Kinematic-Wave form), reading the wrong header line.
    Second correct implementation: a regex on the line's final token; it passes.
    Must-fire control: the Dynamic-Wave line really contains the word `Volume`.
    """
    assert "Volume" in _DYNWAVE_THIRD, "control: a substring test would misread the Dynamic-Wave form"
    assert node_flooding_ponded_field(_flooding_report(_DYNWAVE_THIRD)) == "Depth"
    assert node_flooding_ponded_field(_flooding_report(_KINWAVE_THIRD)) == "Volume"

    cols = [
        "node_id",
        "hours_flooded",
        "max_flood_rate",
        "time_of_max_flood_d_hr_mn",
        "tot_flooded_vol",
        "max_ponded_depth",
    ]
    for third, expect_value in ((_DYNWAVE_THIRD, True), (_KINWAVE_THIRD, False)):
        report = _flooding_report(third)
        rows = return_lines_for_section_of_rpt("Node Flooding Summary", lines=report)
        df = format_rpt_section_into_dataframe(rows, cols)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _apply_ponded_depth_rule(df, node_flooding_ponded_field(report), "x.rpt")
        vals = pd_numeric(df["max_ponded_depth"])
        if expect_value:
            assert list(vals) == [1.041, 1.320]
            assert not any("max_ponded_depth" in str(w.message) for w in caught)
        else:
            assert np.isnan(vals).all()
            assert any("max_ponded_depth" in str(w.message) and "Volume" in str(w.message) for w in caught)


def pd_numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


@pytest.fixture(scope="module")
def reference_parse():
    """The real assembly on the reference report and a stand-in binary output holding that report's series.

    The series come from ``hydraulics.out``, which a laptop cannot produce; the stand-in carries the printed series
    and the ``.out`` path is never opened. The summary variables this module checks come from the report itself.
    """
    from tests.test_swmm_output_inventory import _body_series, _install_stub_pyswmm

    text = REF_RPT.read_text(encoding="latin-1")
    with pytest.MonkeyPatch.context() as mp:
        _install_stub_pyswmm(mp, *_body_series(text))
        ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
            REF_INP, REF_RPT.with_suffix(".out"), family="tritonswmm"
        )
    return ds_nodes, ds_links, text.splitlines()


def _section_rows(lines, header):
    rows = return_lines_for_section_of_rpt(header, lines=[ln + "\n" for ln in lines])
    return [r.split() for r in rows if r.strip()]


def test_d14_the_reference_report_counts_match_between_the_text_and_the_stored_variable(reference_parse):
    """Property: the number of Node Inflow Summary rows ending `ltr` equals the non-NaN count of the volume variable.

    Class: REGRESSION on a real report. Kills it: a parser that sends every value to one variable.
    Second correct implementation: counting the token on the raw text and comparing; this is that.
    """
    ds_nodes, _, lines = reference_parse
    inflow = _section_rows(lines, "Node Inflow Summary")
    n_ltr = sum(1 for r in inflow if r[-1] == "ltr")
    assert 0 < n_ltr < len(inflow), "control: the reference report must have rows of both kinds"
    assert int(ds_nodes["flow_balance_error_volume"].notnull().sum()) == n_ltr
    assert int(ds_nodes["flow_balance_error"].notnull().sum()) == len(inflow) - n_ltr
    assert "flow_balance_error" + "_percent" not in ds_nodes.data_vars


def test_d16_node_and_link_tiers_carry_their_own_names_with_the_values_the_report_printed(reference_parse):
    """Property: a node in both tables carries `max_flood_rate` from the flooding table and
    `time_of_max_total_inflow` from the inflow table, each equal to the printed field; the link tier carries
    `max_flow` and `time_of_max_flow`, and neither tier carries the shared names.

    Class: REGRESSION. Why required: the node tier's maximum flooding rate was stored under the link
    tier's name, one name for two quantities. Kills it: swapping the two node sources, keeping a shared name.
    Second correct implementation: reading each field by column position instead of by header; it passes.
    """
    ds_nodes, ds_links, lines = reference_parse
    flooding = {r[0]: r for r in _section_rows(lines, "Node Flooding Summary")}
    inflow = {r[0]: r for r in _section_rows(lines, "Node Inflow Summary")}
    both = sorted(set(flooding) & set(inflow))
    assert both, "control: the reference report has nodes in both tables"
    node = both[0]
    assert float(ds_nodes["max_flood_rate"].sel(node_id=node)) == pytest.approx(float(flooding[node][2]))
    days, hhmm = inflow[node][4], inflow[node][5]
    hh, mm = hhmm.split(":")
    expected_min = int(days) * 1440 + int(hh) * 60 + int(mm)
    assert float(ds_nodes["time_of_max_total_inflow"].sel(node_id=node)) == pytest.approx(expected_min)
    for gone in ("max_flow" + "_cms", "time_of_max_flow" + "_min", "max_flow", "time_of_max_flow"):
        assert gone not in ds_nodes.data_vars, gone
    for kept in ("max_flow", "time_of_max_flow", "max_velocity"):
        assert kept in ds_links.data_vars, kept
    for gone in ("max_flow" + "_cms", "time_of_max_flow" + "_min", "max_velocity" + "_mps"):
        assert gone not in ds_links.data_vars, gone
    assert "max_ponded_depth" in ds_nodes.data_vars and "max_ponded_depth" + "_m" not in ds_nodes.data_vars
