"""Each SWMM continuity figure is read from the table that owns it, whatever tables the report prints.

Problem. A SWMM status report (`.rpt`) prints up to five continuity tables in a fixed order --
Runoff Quantity, Runoff Quality, Groundwater, Flow Routing, Quality Routing -- and three of them
carry a `Continuity Error (%)` line while two carry a `Flooding Loss` line. Which tables appear
depends on the model: quality tables only when it declares pollutants and routes quality, the
groundwater table only when it has aquifers and groundwater area, the runoff table only when it
has subcatchments. The toolkit's system-level attributes (`runoff_continuity_error_perc`,
`flow_continuity_error_perc`, `system_flooding_10e6_ltr`) must come from the Runoff Quantity and
Flow Routing tables on every one of those reports, and the two scans that read them
(`return_swmm_system_outputs` and the single-pass reader behind `parse_rpt_single_pass`) must agree.

The fixtures below follow the line layout of SWMM's own report writer (`report.c` in SWMM 5.2):
the table header line names the table, the quality tables print one value column per pollutant,
and the flow table's `Flooding Loss` line prints volume then 10^6 ltr (SI) or 10^6 gal (US).
They carry no extra time-series columns, so a failure here is about which table a figure was read
from and never about the time-series reader.
"""

import itertools

import numpy as np
import pytest

from hhemt.swmm_output_parser import parse_rpt_single_pass, return_swmm_system_outputs

RUNOFF_QUANTITY_ERR = 0.111
RUNOFF_QUALITY_ERR = 0.222
GROUNDWATER_ERR = 0.333
FLOW_ERR = 0.444
QUALITY_ROUTING_ERR = 0.555
FLOW_FLOOD_LOSS = 2.0
QUALITY_FLOOD_LOSS = 9.0


def _runoff_quantity(units, from_interface_file):
    if from_interface_file:
        # SWMM's variant when runoff is read from an interface file: a header and no figures.
        return [
            "  **************************",
            "  Runoff Quantity Continuity",
            "  **************************",
            "  Runoff supplied by interface file runoff.dat",
        ]
    vol = "hectare-m            mm" if units == "SI" else "acre-feet        inches"
    return [
        "  **************************        Volume         Depth",
        f"  Runoff Quantity Continuity     {vol}",
        "  **************************     ---------       -------",
        "  Total Precipitation ......         1.000        10.000",
        f"  Continuity Error (%) .....{RUNOFF_QUANTITY_ERR:14.3f}",
    ]


def _quality_columns(base, n_pollutants):
    # A second pollutant prints a different value in its own column, so a reader that took the
    # last token of a quality line would read 0.9xx rather than the first pollutant's figure.
    return "".join(f"{v:14.3f}" for v in [base] + [0.9 + 0.01 * i for i in range(1, n_pollutants)])


def _runoff_quality(n_pollutants):
    names = "".join(f"{name:>14}" for name in ["TSS", "LEAD", "ZINC"][:n_pollutants])
    return [
        "  **************************",
        f"  Runoff Quality Continuity {names}",
        "  **************************",
        f"  Continuity Error (%) .....{_quality_columns(RUNOFF_QUALITY_ERR, n_pollutants)}",
    ]


def _groundwater(units):
    vol = "hectare-m            mm" if units == "SI" else "acre-feet        inches"
    return [
        "  **************************        Volume         Depth",
        f"  Groundwater Continuity         {vol}",
        "  **************************     ---------       -------",
        f"  Continuity Error (%) .....{GROUNDWATER_ERR:14.3f}",
    ]


def _flow_routing(units):
    vol = "hectare-m      10^6 ltr" if units == "SI" else "acre-feet      10^6 gal"
    return [
        "  **************************        Volume        Volume",
        f"  Flow Routing Continuity        {vol}",
        "  **************************     ---------     ---------",
        f"  Flooding Loss ............{0.2:14.3f}{FLOW_FLOOD_LOSS:14.3f}",
        f"  Continuity Error (%) .....{FLOW_ERR:14.3f}",
    ]


def _quality_routing(n_pollutants):
    names = "".join(f"{name:>14}" for name in ["TSS", "LEAD", "ZINC"][:n_pollutants])
    return [
        "  **************************",
        f"  Quality Routing Continuity{names}",
        "  **************************",
        f"  Flooding Loss ............{_quality_columns(QUALITY_FLOOD_LOSS, n_pollutants)}",
        f"  Continuity Error (%) .....{_quality_columns(QUALITY_ROUTING_ERR, n_pollutants)}",
    ]


def _report(*, units, runoff, runoff_from_file, quality, n_pollutants, groundwater):
    lines = ["  Element Count", f"  Flow Units ............... {'CMS' if units == 'SI' else 'CFS'}", "  "]
    if runoff:
        lines += _runoff_quantity(units, runoff_from_file) + ["  "]
    if quality:
        lines += _runoff_quality(n_pollutants) + ["  "]
    if groundwater:
        lines += _groundwater(units) + ["  "]
    lines += _flow_routing(units) + ["  "]
    if quality:
        lines += _quality_routing(n_pollutants) + ["  "]
    lines += [
        "  ************************",
        "  Node Time Series Results",
        "  ************************",
        "  ",
        "  <<< Node N1 >>>",
        "  " + "-" * 74,
        "                           Inflow  Flooding     Depth      Head",
        "  Date        Time            CMS       CMS    meters    meters",
        "  " + "-" * 64,
        "   01/01/2000 00:00:10      0.100     0.200     0.300     0.400",
        "  ",
        "  ************************",
        "  Link Time Series Results",
        "  ************************",
        "  ",
        "  <<< Link L1 >>>",
        "  " + "-" * 74,
        "                             Flow  Velocity     Depth  Capacity/",
        "  Date        Time            CMS     m/sec    meters   Setting ",
        "  " + "-" * 64,
        "   01/01/2000 00:00:10      1.000     2.000     3.000     4.000",
        "  ",
        "  Analysis ended on:  Tue Jan 27 20:10:17 2026",
    ]
    return "\n".join(lines) + "\n"


def _cases():
    for units, runoff, runoff_from_file, quality, groundwater in itertools.product(
        ["SI", "US"], [True, False], [False, True], [False, True], [False, True]
    ):
        if runoff_from_file and not runoff:
            continue
        for n_pollutants in [1, 2] if quality else [0]:
            runoff_id = f"runoff{int(runoff)}{'file' if runoff_from_file else ''}"
            yield pytest.param(
                dict(
                    units=units,
                    runoff=runoff,
                    runoff_from_file=runoff_from_file,
                    quality=quality,
                    n_pollutants=n_pollutants,
                    groundwater=groundwater,
                ),
                id=f"{units}-{runoff_id}-qual{n_pollutants}-gw{int(groundwater)}",
            )


@pytest.mark.parametrize("case", list(_cases()))
def test_each_continuity_figure_comes_from_its_own_table_in_both_scans(case, tmp_path):
    """Property: on every report SWMM can print, both system scans return the Runoff Quantity
    table's error (or NaN when that table has no figure), the Flow Routing table's error, and the
    Flow Routing table's flooding loss, and they return the same three values.

    Class: every combination of the optional continuity tables around the Flow Routing table, in
    SWMM's print order, in SI and US units, with the runoff table absent, read from an interface
    file, or computed, and with one or two pollutants in the quality tables.

    Why it is required: these attributes are reported as the run's hydraulic continuity; a value
    read from a quality or groundwater table is a pollutant-mass or aquifer balance under a
    hydraulic name, and nothing downstream can tell the two apart.

    What kills it: dropping a table header from the set the scan tracks -- with `Quality Routing
    Continuity` untracked, that table's lines are read as the flow table's and the flow error
    becomes 0.555 -- or reverting either scan to taking the last matching line.

    A second correct implementation that still passes: a scan that splits the report into blocks
    at each `****` banner and reads each figure from the block whose header names its table.
    """
    text = _report(**case)
    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text(text)
    scanned = return_swmm_system_outputs(text.splitlines(keepends=True))
    single_pass = parse_rpt_single_pass(rpt)[3]

    expect_runoff = RUNOFF_QUANTITY_ERR if case["runoff"] and not case["runoff_from_file"] else np.nan
    for system in (scanned, single_pass):
        assert system["flow_continuity_error_perc"] == pytest.approx(FLOW_ERR)
        assert system["system_flooding_10e6_ltr"] == pytest.approx(FLOW_FLOOD_LOSS)
        assert system["runoff_continuity_error_perc"] == pytest.approx(expect_runoff, nan_ok=True)
    for key in ("runoff_continuity_error_perc", "flow_continuity_error_perc", "system_flooding_10e6_ltr"):
        assert scanned[key] == pytest.approx(single_pass[key], nan_ok=True)
