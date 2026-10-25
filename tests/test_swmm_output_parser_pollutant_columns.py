"""SWMM report parser: extra pollutant columns and block-scoped continuity figures.

Problem these tests guard. When a model declares a pollutant and routes quality, SWMM
appends one concentration column to every row of the node and link time-series blocks and
prints extra continuity tables (`Runoff Quality Continuity`, `Quality Routing Continuity`,
and, for a model with groundwater, `Groundwater Continuity`) after the quantity tables.
The parser must read only the declared hydraulic columns and must take each continuity
figure from its own table.

Class and the two properties each test turns on, both of which the old code can also
exhibit: "a dataset is returned" (the old code raises on the extra column) and "the flow
error equals the flow table's value" (the old code returns the last table's value).
A second correct implementation: a parser that reads all columns and then drops the extras,
or that tracks the table with a regex rather than a header scan, passes every assertion here.
"""

import numpy as np
import pytest

from hhemt.swmm_output_parser import (
    _bulk_parse_tseries,
    parse_rpt_single_pass,
    return_swmm_system_outputs,
)

_RUNOFF_QUANTITY = 0.111
_RUNOFF_QUALITY = 0.222
_GROUNDWATER = 0.333
_FLOW = 0.444
_QUALITY_ROUTING = 0.555
_FLOOD_FLOW = 2.0
_FLOOD_QUALITY = 9.0


def _continuity_table(header, error, flood=None):
    rows = [
        "  **************************        Volume        Volume",
        f"  {header:<31}hectare-m      10^6 ltr",
        "  **************************     ---------     ---------",
        "  Initial Stored Volume ....         0.000         0.000",
    ]
    if flood is not None:
        rows.append(f"  Flooding Loss ............         0.000{flood:14.3f}")
    rows.append(f"  Continuity Error (%) .....{error:14.3f}")
    rows += ["  ", "  "]
    return "\n".join(rows) + "\n"


def _rows(values, n_extra):
    extra = "     0.500" * n_extra
    return "\n".join(f"   01/01/2000 00:00:{10 * (i + 1):02d}{values}{extra}" for i in range(3))


def _tseries_rows(n_extra):
    return _rows("      0.100     0.200     0.300     0.400", n_extra)


def _link_rows(n_extra):
    return _rows("      1.000     2.000     3.000     4.000", n_extra)


def _rpt(n_extra=0, quality_tables=False, groundwater_table=False, runoff_quantity=True):
    parts = [
        "  Element Count\n",
        "  Flow Units ............... CMS\n",
    ]
    if runoff_quantity:
        parts.append(_continuity_table("Runoff Quantity Continuity", _RUNOFF_QUANTITY))
    if quality_tables:
        parts.append(_continuity_table("Runoff Quality Continuity", _RUNOFF_QUALITY))
    if groundwater_table:
        parts.append(_continuity_table("Groundwater Continuity", _GROUNDWATER))
    parts.append(_continuity_table("Flow Routing Continuity", _FLOW, flood=_FLOOD_FLOW))
    if quality_tables:
        parts.append(_continuity_table("Quality Routing Continuity", _QUALITY_ROUTING, flood=_FLOOD_QUALITY))
    parts.append("  ************************\n  Node Time Series Results\n  ************************\n  \n")
    parts.append("  <<< Node N1 >>>\n  " + "-" * 74 + "\n")
    parts.append("                           Inflow  Flooding     Depth      Head\n")
    parts.append("  Date        Time            CMS       CMS    meters    meters\n  " + "-" * 64 + "\n")
    parts.append(_tseries_rows(n_extra) + "\n  \n")
    parts.append("  ************************\n  Link Time Series Results\n  ************************\n  \n")
    parts.append("  <<< Link L1 >>>\n  " + "-" * 74 + "\n")
    parts.append("                             Flow  Velocity     Depth  Capacity/\n")
    parts.append("  Date        Time            CMS     m/sec    meters   Setting \n  " + "-" * 64 + "\n")
    parts.append(_link_rows(n_extra) + "\n  \n")
    parts.append("  Analysis ended on:  Tue Jan 27 20:10:17 2026\n")
    return "".join(parts)


NODE_VARS = {"inflow_flow_cms", "flooding_cms", "depth_m", "head_m"}
LINK_VARS = {"flow_cms", "velocity_mps", "link_depth_m", "capacity_setting"}


def test_extra_pollutant_column_is_not_read():
    ds_node, ds_link = _bulk_parse_tseries(_rpt(n_extra=1))
    assert set(ds_node.data_vars) == NODE_VARS
    assert set(ds_link.data_vars) == LINK_VARS
    assert ds_node.sizes["date_time"] == 3 and ds_link.sizes["date_time"] == 3
    assert not np.isnat(ds_node["date_time"].values).any()


def test_dataset_with_extra_column_equals_dataset_without_it():
    with_col = _bulk_parse_tseries(_rpt(n_extra=1))
    without = _bulk_parse_tseries(_rpt(n_extra=0))
    assert with_col[0].equals(without[0])
    assert with_col[1].equals(without[1])


@pytest.mark.parametrize("n_extra", [0, 1, 2])
def test_any_number_of_extra_columns_is_ignored(n_extra):
    ds_node, ds_link = _bulk_parse_tseries(_rpt(n_extra=n_extra))
    assert float(ds_node["depth_m"].isel(node_id=0, date_time=0)) == pytest.approx(0.3)
    assert float(ds_link["capacity_setting"].isel(link_id=0, date_time=0)) == pytest.approx(4.0)


@pytest.mark.parametrize("quality_tables", [False, True])
def test_flow_figures_come_from_the_flow_table(tmp_path, quality_tables):
    text = _rpt(n_extra=int(quality_tables), quality_tables=quality_tables)
    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text(text)
    system_single = parse_rpt_single_pass(rpt)[3]
    system_scan = return_swmm_system_outputs(text.splitlines(keepends=True))
    for system in (system_single, system_scan):
        assert system["flow_continuity_error_perc"] == pytest.approx(_FLOW)
        assert system["system_flooding_10e6_ltr"] == pytest.approx(_FLOOD_FLOW)
        assert system["runoff_continuity_error_perc"] == pytest.approx(_RUNOFF_QUANTITY)


def test_runoff_error_comes_from_the_quantity_table_not_quality_or_groundwater(tmp_path):
    text = _rpt(quality_tables=True, groundwater_table=True)
    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text(text)
    for system in (parse_rpt_single_pass(rpt)[3], return_swmm_system_outputs(text.splitlines(keepends=True))):
        assert system["runoff_continuity_error_perc"] == pytest.approx(_RUNOFF_QUANTITY)
        assert system["flow_continuity_error_perc"] == pytest.approx(_FLOW)


def test_report_without_quality_tables_keeps_its_values(tmp_path):
    text = _rpt()
    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text(text)
    system = parse_rpt_single_pass(rpt)[3]
    assert system["runoff_continuity_error_perc"] == pytest.approx(_RUNOFF_QUANTITY)
    assert system["flow_continuity_error_perc"] == pytest.approx(_FLOW)
    assert system["system_flooding_10e6_ltr"] == pytest.approx(_FLOOD_FLOW)


def test_hydraulics_only_report_has_nan_runoff_error(tmp_path):
    text = _rpt(runoff_quantity=False)
    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text(text)
    system = parse_rpt_single_pass(rpt)[3]
    assert np.isnan(system["runoff_continuity_error_perc"])
    assert system["flow_continuity_error_perc"] == pytest.approx(_FLOW)
