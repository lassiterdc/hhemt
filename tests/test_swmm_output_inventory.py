"""The SWMM output inventory: binary-output series, the report tables beyond the first three, and run-level measures.

The coupled and standalone families both read their per-step series from SWMM's binary output (a ``.out``) and
their summary tables, continuity figures and ranked lists from the report beside it. This module holds the readers
against a printed stock SWMM 5.2.4 report (``test_data/swmm_refactoring_reference/hydraulics.rpt``, a 1135-node,
1146-link Dynamic Wave model) and against constructed tables for the rows that report lacks (a pump, a Kinematic Wave
run, a report with no time-step frequencies). No solver is compiled or run; the binary reader is exercised through
a stub of pyswmm's ``Output`` whose series are read from the same stock report, so the assembly (names, dimensions,
ids, families) is held against the committed reference series stores without a ``.out`` file.

Every expected value is read from the PRINTED report line by an independent split in this module (``_printed``), never
from the code under test.
"""

from __future__ import annotations

import functools
import re
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "test_data" / "swmm_refactoring_reference"
REF_INP = REF_DIR / "hydraulics.inp"
REF_RPT = REF_DIR / "hydraulics.rpt"
BANNER = "Node Time Series Results"
LINK_BANNER = "Link Time Series Results"

NAN = float("nan")


# --------------------------------------------------------------------------------------------------- report helpers
def _ref_text() -> str:
    return REF_RPT.read_text(encoding="latin-1")


def _section_lines(text: str, title: str) -> list[str]:
    """The lines after ``title`` up to the next blank-line pair, an independent walk of the printed report."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if title in ln)
    out = []
    for ln in lines[start + 1 :]:
        if out and not ln.strip():
            break
        out.append(ln)
    return out


def _printed(text: str, title: str, first_token: str) -> list[str]:
    """The tokens of the row of ``title`` whose first token is ``first_token``, split independently of the parser."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if title in ln)
    for ln in lines[start + 1 :]:
        tokens = ln.split()
        if tokens and tokens[0] == first_token and not set(tokens[0]) <= {"-", "*"}:
            return tokens
    raise AssertionError(f"no {first_token!r} row under {title!r}")


def _replace_row(text: str, title: str, first_token: str, new_line: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if title in ln)
    for i in range(start + 1, len(lines)):
        tokens = lines[i].split()
        if tokens and tokens[0] == first_token and not set(tokens[0]) <= {"-", "*"}:
            lines[i] = new_line.rstrip("\n") + "\n"
            return "".join(lines)
    raise AssertionError(f"no {first_token!r} row under {title!r}")


def _drop_section(text: str, title: str) -> str:
    """The report without the section titled ``title``: its opening rule, title, rule and body, up to the blank
    pair that precedes the next section's opening rule."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if title in ln)
    first = start - 1 if lines[start - 1].lstrip().startswith("*") else start
    end = start + 1
    while end < len(lines) and not (
        lines[end].lstrip().startswith("*") and not lines[end - 1].strip() and not lines[end - 2].strip()
    ):
        end += 1
    return "".join(lines[:first] + lines[end:])


def _stage(tmp_path: Path, text: str, *, stem: str = "hydraulics") -> tuple[Path, Path]:
    """Write ``text`` as ``{stem}.rpt`` beside an empty ``{stem}.out``; return ``(out, rpt)``."""
    rpt = tmp_path / f"{stem}.rpt"
    rpt.write_text(text, encoding="latin-1")
    out = tmp_path / f"{stem}.out"
    out.write_bytes(b"")
    return out, rpt


# ------------------------------------------------------------------------------- a pyswmm stand-in read from the report
def _body_series(text: str):
    """``({node: {name: {datetime: value}}}, {link: ...})`` read from the report's per-timestep body.

    A test-local reader (split on whitespace) so the stand-in binary output carries the printed values; the
    toolkit's own report series reader is removed and may not be used here.
    """
    nodes: dict[str, dict[str, dict]] = {}
    links: dict[str, dict[str, dict]] = {}
    current = None
    cols = None
    in_body = False
    for ln in text.splitlines():
        if BANNER in ln:
            in_body = True
        if not in_body:
            continue
        m = re.match(r"\s*<<<\s+(Node|Link)\s+(\S+)\s+>>>", ln)
        if m:
            kind, ident = m.groups()
            if kind == "Node":
                current = nodes.setdefault(ident, {})
                cols = ("total_inflow", "flooding_losses", "invert_depth", "hydraulic_head")
            else:
                current = links.setdefault(ident, {})
                cols = ("flow_rate", "flow_velocity", "flow_depth", "capacity")
            for c in cols:
                current[c] = {}
            continue
        tokens = ln.split()
        if current is not None and len(tokens) == 6 and re.match(r"\d\d/\d\d/\d{4}$", tokens[0]):
            when = pd.Timestamp(f"{tokens[0]} {tokens[1]}").to_pydatetime()
            for c, v in zip(cols, tokens[2:], strict=True):
                current[c][when] = float(v)
    return nodes, links


@pytest.fixture(scope="module")
def ref_series():
    return _body_series(_ref_text())


def _install_stub_pyswmm(monkeypatch, nodes, links, *, calls=None, units=None):
    """Replace pyswmm's ``Output``, ``NodeSeries`` and ``LinkSeries`` with stand-ins holding ``nodes`` and ``links``.

    The three series the binary file holds beyond the report's four are synthetic functions of the printed ones so a
    test can tell which pyswmm attribute fed which stored name: ``ponded_volume`` is 10 x depth, ``lateral_inflow``
    is minus the inflow, ``flow_volume`` is 3 x the flow.
    """
    import pyswmm

    class _Out:
        def __init__(self, _path):
            self.nodes = dict.fromkeys(nodes, 0)
            self.links = dict.fromkeys(links, 0)
            self.units = units or {"flow": "CMS", "system": "SI"}
            self.project_size = [0, 0, len(nodes), len(links), 0]
            self.start = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Node:
        def __init__(self, _out):
            self._id = None

        def __getitem__(self, ident):
            self._id = ident
            return self

        def __getattr__(self, attr):
            series = nodes[self._id]
            if calls is not None:
                calls.append(("node", attr))
            if attr == "ponded_volume":
                return {t: 10.0 * v for t, v in series["invert_depth"].items()}
            if attr == "lateral_inflow":
                return {t: -v for t, v in series["total_inflow"].items()}
            return series[attr]

    class _Link:
        def __init__(self, _out):
            self._id = None

        def __getitem__(self, ident):
            self._id = ident
            return self

        def __getattr__(self, attr):
            series = links[self._id]
            if calls is not None:
                calls.append(("link", attr))
            if attr == "flow_volume":
                return {t: 3.0 * v for t, v in series["flow_rate"].items()}
            return series[attr]

    monkeypatch.setattr(pyswmm, "Output", _Out)
    monkeypatch.setattr(pyswmm, "NodeSeries", _Node)
    monkeypatch.setattr(pyswmm, "LinkSeries", _Link)

    # The reader reads through two swmm-toolkit seams, not through pyswmm's series accessors (pyswmm 2.1.0 drops
    # the final report period through them), so the stand-ins answer at the seams: the time axis is the first
    # node's report times and a series is the stand-in's value list for (id, pyswmm attribute).
    import numpy as np

    from hhemt import swmm_output_parser as sop

    first = next((series["invert_depth"] for series in nodes.values()), {})
    monkeypatch.setattr(sop, "_out_time_axis", lambda out, f_outfile: list(first))

    def _values(out, kind, ident, index, attribute):
        element = (_Node if kind == "node" else _Link)(None)[ident]
        return np.array(list(getattr(element, attribute).values()), dtype=float)

    monkeypatch.setattr(sop, "_out_series_values", _values)


@pytest.fixture
def stub_ref(monkeypatch, ref_series):
    """Install the stand-in binary output holding the reference report's series; returns the call log."""
    calls: list = []
    _install_stub_pyswmm(monkeypatch, *ref_series, calls=calls)
    return calls


def _assemble(tmp_path, text, *, family="tritonswmm", inp=REF_INP):
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    out, _rpt = _stage(tmp_path, text)
    return retrieve_SWMM_outputs_as_datasets(inp, out, family=family)


@functools.lru_cache(maxsize=1)
def _ref_sections():
    """The section lines of the reference report, by the real scan. Called inside test bodies, so a name the scan
    module does not yet define fails the test instead of erroring a fixture."""
    from hhemt.swmm_output_parser import (
        _REPORT_SECTION_HEADERS,
        _read_report_head_and_tail,
        _scan_metadata_and_summaries,
    )

    head, tail = _read_report_head_and_tail(REF_RPT)
    return _scan_metadata_and_summaries(head, _REPORT_SECTION_HEADERS, tail_lines=tail)[0]


class _LazySections:
    """``ref_sections[title]`` resolved at access time, in the test body."""

    def __getitem__(self, title):
        return _ref_sections()[title]


@pytest.fixture(scope="module")
def ref_sections():
    return _LazySections()


def _table(title, lines, rpt="<test>"):
    from hhemt.swmm_output_parser import read_report_table

    return read_report_table(title, lines, rpt)


# --------------------------------------------------------------------------------------------------- table readers
def test_node_depth_summary_reads_a_printed_row(ref_sections):
    """Property: one node's average depth, maximum depth, HGL and time of maximum equal its printed row.

    Class: the Node Depth Summary rows of a Dynamic Wave report. Why: the reader takes cells by token position, and
    the row carries a type word, a days/clock pair and a trailing Reported Max Depth the toolkit leaves out, so a
    shifted position reads another column. What kills it: reading the clock cell as the HGL. Second correct
    implementation: a fixed-column reader keyed on character offsets passes. Control: the same row with one token
    removed is refused naming the table.
    """
    from hhemt.exceptions import ProcessingError

    text = _ref_text()
    tokens = _printed(text, "Node Depth Summary", "11-0057")
    frame = _table("Node Depth Summary", ref_sections["Node Depth Summary"])
    row = frame.loc["11-0057"]
    assert (row["average_depth"], row["max_depth"], row["max_hgl"]) == (
        float(tokens[2]),
        float(tokens[3]),
        float(tokens[4]),
    )
    assert row["max_hgl"] == -1.10
    assert row["time_of_max_depth"] == int(tokens[5]) * 1440 + int(tokens[6][:2]) * 60 + int(tokens[6][3:])
    short = [ref_sections["Node Depth Summary"][0].replace(" JUNCTION", "")]
    with pytest.raises(ProcessingError, match="Node Depth Summary"):
        _table("Node Depth Summary", short)


def test_node_surcharge_summary_reads_a_printed_row_and_leaves_an_unlisted_node_nan(tmp_path, stub_ref):
    """Property: a listed node reads its three values; an unlisted junction and the outfall read NaN.

    Class: nodes listed and not listed by the Node Surcharge Summary. Why: SWMM prints a row only for a node that
    surcharged, so the dataset must hold NaN (not 0) for the rest and say what that means. What kills it: filling
    unlisted nodes with 0. Second correct implementation: reading the table through the shared section reader.
    Control: a report whose section reads ``No nodes were surcharged.`` yields all NaN without error.
    """
    text = _ref_text()
    tokens = _printed(text, "Node Surcharge Summary", "E143257")
    ds_nodes, _ = _assemble(tmp_path, text)
    got = ds_nodes.sel(node_id="E143257")
    assert (
        float(got["hours_surcharged"]),
        float(got["max_height_above_crown"]),
        float(got["smallest_depth_below_rim"]),
    ) == (float(tokens[2]), float(tokens[3]), float(tokens[4]))
    for unlisted in ("05-0088", "dummy_of"):
        assert np.isnan(float(ds_nodes["hours_surcharged"].sel(node_id=unlisted))), unlisted
    start = text.index("Surcharging occurs when water rises above the top of the highest conduit.")
    end = text.index("*********************\n  Node Flooding Summary")
    sentence_text = text[:start] + "No nodes were surcharged.\n\n\n  " + text[end:]
    (tmp_path / "none").mkdir()
    ds_none, _ = _assemble(tmp_path / "none", sentence_text)
    assert ds_none["hours_surcharged"].isnull().all()
    assert ds_none["max_height_above_crown"].isnull().all()


def test_storage_volume_summary_reads_eight_values_and_leaves_a_junction_nan(tmp_path, stub_ref):
    """Property: storage unit S1 reads its eight values as printed; a junction reads NaN in all eight.

    Class: storage rows. Why: the row carries a days/clock pair in the middle and the maximum outflow last, so a
    positional reader that drops the pair shifts the outflow. The stock row prints zeros throughout, which cannot
    tell positions apart, so the test rewrites S1's row with eight distinct numbers. What kills it: reading token 7
    as the time. Second correct implementation: a regex per cell.
    """
    text = _replace_row(
        _ref_text(),
        "Storage Volume Summary",
        "S1",
        "  S1                       1.234    5.6    7.8    9.1      10.234   11.2       3  04:05      12.345",
    )
    ds_nodes, _ = _assemble(tmp_path, text)
    s1 = ds_nodes.sel(node_id="S1")
    assert [float(s1[n]) for n in _STORAGE_NAMES] == [1.234, 5.6, 7.8, 9.1, 10.234, 11.2, 3 * 1440 + 4 * 60 + 5, 12.345]
    for name in _STORAGE_NAMES:
        assert np.isnan(float(ds_nodes[name].sel(node_id="05-0088"))), name


_STORAGE_NAMES = (
    "storage_average_volume",
    "storage_average_fill",
    "storage_evaporation_loss",
    "storage_exfiltration_loss",
    "storage_max_volume",
    "storage_max_fill",
    "time_of_storage_max_volume",
    "storage_max_outflow",
)


def test_outfall_loading_summary_reads_flow_columns_without_quality(tmp_path, stub_ref):
    """Property: with quality not routed the outfall reads its four flow columns and the System row's maximum.

    Class: an outfall row of a model that declares a pollutant but routes no quality (6 tokens, one load column).
    Why: the flow columns were read only on the quality path, and the System row's maximum flow is the maximum of the
    SUM of outfall flows, which no per-outfall column gives. What kills it: gating the flow columns on quality, or
    taking the System row's average column. Second correct implementation: reading the rows by regex. Control: a
    quality-routed constructed table still yields ``outfall_total_load`` as before.
    """
    from hhemt.swmm_output_parser import parse_rpt_quality_summaries

    text = _replace_row(
        _ref_text(),
        "Outfall Loading Summary",
        "dummy_of",
        "  dummy_of               12.34     1.234     5.678       9.012         3.456",
    )
    text = _replace_row(
        text,
        "Outfall Loading Summary",
        "System",
        "  System                  6.17     1.234     7.890       9.012         3.456",
    )
    ds_nodes, _ = _assemble(tmp_path, text)
    assert "Water Quality .......... NO" in text
    out = ds_nodes.sel(node_id="dummy_of")
    assert (
        float(out["outfall_flow_frequency"]),
        float(out["outfall_average_flow"]),
        float(out["outfall_max_flow"]),
        float(out["outfall_total_vol"]),
    ) == (12.34, 1.234, 5.678, 9.012)
    assert float(ds_nodes["outfall_system_max_flow"]) == 7.890
    assert "outfall_total_load" not in ds_nodes
    lines = text.splitlines(keepends=True)
    quality = parse_rpt_quality_summaries(lines, [("TSS", "mg/L")], include_runoff_quality=False, filepath="<test>")
    assert float(quality["outfall_total_load"].sel(node_id="dummy_of", pollutant="TSS")) == 3.456


def test_flow_classification_summary_reads_ten_values_and_leaves_a_non_conduit_nan(tmp_path, stub_ref, ref_sections):
    """Property: a conduit's ten values equal its printed row; an orifice reads NaN.

    Class: conduit rows of the Flow Classification Summary. Why: the ten columns are a length ratio, seven flow
    classes and two limit fractions; a miscounted class shifts the limit fractions. What kills it: a ten-token
    row layout. Second correct implementation: a regex per row.
    """
    text = _ref_text()
    tokens = _printed(text, "Flow Classification Summary", "1")
    frame = _table("Flow Classification Summary", ref_sections["Flow Classification Summary"])
    assert list(frame.loc["1"]) == [float(t) for t in tokens[1:]]
    assert frame.shape[1] == 10
    _, ds_links = _assemble(tmp_path, text)
    assert np.isnan(float(ds_links["flow_class_dry"].sel(link_id="756")))
    assert float(ds_links["inlet_control_fraction"].sel(link_id="1")) == float(tokens[10])


def test_conduit_surcharge_summary_reads_five_values_and_leaves_an_unlisted_conduit_nan(tmp_path, stub_ref):
    """Property: a listed conduit reads its five hours; an unlisted conduit reads NaN in all five.

    Class: conduits listed and not listed. Why: SWMM omits a conduit whose first four totals are all 0 and does not
    state the capacity-limited hours of an omitted one, so the stored NaN must carry that comment. What kills it:
    zero-filling. Second correct implementation: a per-cell regex.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    text = _ref_text()
    tokens = _printed(text, "Conduit Surcharge Summary", "439")
    _, ds_links = _assemble(tmp_path, text)
    names = [
        "hours_full_both_ends",
        "hours_full_upstream",
        "hours_full_downstream",
        "hours_above_full_normal_flow",
        "hours_capacity_limited",
    ]
    assert [float(ds_links[n].sel(link_id="439")) for n in names] == [float(t) for t in tokens[1:]]
    for n in names:
        assert np.isnan(float(ds_links[n].sel(link_id="0"))), n
    assert "does not state" in _CF_VARIABLE_MAP["hours_capacity_limited"]["comment"]


_PUMP_ROW = (
    "  P1                      85.50           3      0.00      2.10      4.25     1.234    456.78   10.5    2.5"
)


def _with_pump_section(text: str) -> str:
    section = (
        "  ***************\n  Pumping Summary\n  ***************\n  \n"
        "  ---------------------------------------------------------------------------------------------------------\n"
        "                                                  Min       Avg       Max     Total     Power    % Time Off\n"
        "                        Percent   Number of      Flow      Flow      Flow    Volume     Usage    Pump Curve\n"
        "  Pump                 Utilized   Start-Ups       CMS       CMS       CMS  10^6 ltr     Kw-hr    Low   High\n"
        "  ---------------------------------------------------------------------------------------------------------\n"
        f"{_PUMP_ROW}\n  \n  \n"
    )
    marker = "  ************************\n  Node Time Series Results"
    assert marker in text
    return text.replace(marker, section + marker, 1)


def test_pumping_summary_reads_seven_values_and_names_no_min_or_max_flow(tmp_path, stub_ref):
    """Property: a pump row reads its seven kept values; no pump minimum or maximum flow variable exists; a conduit
    reads NaN.

    Class: a PUMP row in the exact ``statsrpt.c`` format (the stock reference prints none). Why: SWMM 5.2.4 prints
    0.00 as every pump's minimum and repeats the link maximum as the pump maximum, so neither carries information.
    What kills it: storing the minimum or the maximum. Second correct implementation: a fixed-column reader.
    """
    text = _with_pump_section(_ref_text())
    tokens = _PUMP_ROW.split()
    _, ds_links = _assemble(tmp_path, text)
    pump = ds_links.sel(link_id="P1")
    expected = {
        "pump_utilization": float(tokens[1]),
        "pump_startups": float(tokens[2]),
        "pump_average_flow": float(tokens[4]),
        "pump_total_volume": float(tokens[6]),
        "pump_energy": float(tokens[7]),
        "pump_off_curve_low": float(tokens[8]),
        "pump_off_curve_high": float(tokens[9]),
    }
    assert {k: float(pump[k]) for k in expected} == expected
    assert "pump_max_flow" not in ds_links and "pump_min_flow" not in ds_links
    assert np.isnan(float(ds_links["pump_energy"].sel(link_id="1")))


# --------------------------------------------------------------------------------------------- run-level variables
def test_flow_routing_continuity_rows_are_read_from_the_ten_to_the_six_litres_column(ref_sections):
    """Property: each of the ten ``routing_continuity_{term}`` values equals the ``10^6 L`` column of its row.

    Class: the ten rows the toolkit did not keep. Why: each row prints a hectare-meter figure and then a ``10^6 L``
    figure, and the toolkit's unit is ``10^6 L``. What kills it: reading the first column. Second correct
    implementation: reading the last token of the row. Control: the hectare-meter value of External Inflow differs
    from the stored one.
    """
    from hhemt.swmm_output_parser import read_flow_routing_continuity

    text = _ref_text()
    got = read_flow_routing_continuity(ref_sections["Flow Routing Continuity"])
    assert len(got) == 10
    labels = {
        "dry_weather_inflow": "Dry Weather Inflow",
        "wet_weather_inflow": "Wet Weather Inflow",
        "groundwater_inflow": "Groundwater Inflow",
        "rdii_inflow": "RDII Inflow",
        "external_inflow": "External Inflow",
        "external_outflow": "External Outflow",
        "evaporation_loss": "Evaporation Loss",
        "exfiltration_loss": "Exfiltration Loss",
        "initial_stored_volume": "Initial Stored Volume",
        "final_stored_volume": "Final Stored Volume",
    }
    printed = {}
    for line in _section_lines(text, "Flow Routing Continuity"):
        m = re.match(r"\s*([A-Za-z ]+?)\s*\.{2,}\s+(\S+)\s+(\S+)\s*$", line)
        if m:
            printed[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    for stem, label in labels.items():
        assert got[f"routing_continuity_{stem}"] == printed[label][1], stem
    assert printed["External Inflow"] == (0.125, 1.253)
    assert got["routing_continuity_external_inflow"] != printed["External Inflow"][0]


def test_routing_time_step_summary_reads_six_figures_and_five_bins(ref_sections):
    """Property: the six figures and five frequencies equal the printed ones, and the bin bounds equal the printed
    intervals.

    Class: the Routing Time Step Summary of a Dynamic Wave report with a Courant factor. Why: the figures are
    labelled lines and the table is ``upper - lower sec : pct %`` rows. What kills it: swapping the bound
    order. Second correct implementation: parsing the rows with split instead of the regex. Control: a section with
    no ``Time Step Frequencies`` lines yields NaN frequencies without error.
    """
    from hhemt.swmm_output_parser import read_routing_time_step_summary

    scalars, upper, lower, freq = read_routing_time_step_summary(ref_sections["Routing Time Step Summary"])
    assert scalars == {
        "routing_step_smallest": 0.06,
        "routing_step_mean": 0.19,
        "routing_step_largest": 0.26,
        "routing_steady_state_time": 0.0,
        "routing_iterations_mean": 2.67,
        "routing_nonconverging_steps": 0.65,
    }
    assert list(upper) == [0.100, 0.055, 0.030, 0.017, 0.009]
    assert list(lower) == [0.055, 0.030, 0.017, 0.009, 0.005]
    assert list(freq) == [100.0, 0.0, 0.0, 0.0, 0.0]
    no_freq = [
        ln
        for ln in ref_sections["Routing Time Step Summary"]
        if ":" in ln and "Time Step Frequencies" not in ln and not re.match(r"\s*[\d.]+\s*-", ln)
    ]
    assert len(no_freq) == 6
    _, _, _, freq_nan = read_routing_time_step_summary(no_freq)
    assert np.isnan(freq_nan).all() and freq_nan.size == 5


def test_ranked_lists_read_the_printed_elements_and_the_all_clear_sentences_as_nan(ref_sections):
    """Property: the four lists equal their printed elements in rank order; ``None``, ``All links are stable.`` and
    ``Convergence obtained at all time steps.`` read as all-NaN, never as zero.

    Class: the four ranked lists SWMM prints. Why: SWMM prints an all-clear sentence whenever the top element's INDEX
    is 0, which is also the case of a nonzero value at the first node or link of the file, so a sentence does not
    prove the quantity was zero. What kills it: reading a sentence as zeros. Second correct implementation: a
    regex per list.
    """
    from hhemt.swmm_output_parser import read_ranked_list

    kinds, ids, values = read_ranked_list(ref_sections["Highest Flow Instability Indexes"])
    assert list(ids) == ["1444", "597", "1084", "442", "441"] and list(values) == [3, 3, 2, 2, 2]
    _, ids, values = read_ranked_list(ref_sections["Most Frequent Nonconverging Nodes"])
    assert list(ids) == ["dummy_of", "E143704", "E145063", "E147009", "E143257"]
    assert list(values) == [0.65, 0.16, 0.16, 0.16, 0.16]
    _, ids, values = read_ranked_list(ref_sections["Highest Continuity Errors"])
    assert list(ids) == ["05-0088", "E133468", "E133470", "E134203", "E134213"] and list(values) == [100.0] * 5
    kinds, ids, values = read_ranked_list(ref_sections["Time-Step Critical Elements"])
    assert ids.tolist() == [""] * 5 and kinds.tolist() == [""] * 5 and np.isnan(values).all()
    for sentence in ("  All links are stable.\n", "  Convergence obtained at all time steps.\n"):
        _, ids, values = read_ranked_list([sentence])
        assert ids.tolist() == [""] * 5 and np.isnan(values).all(), sentence


def test_a_ranked_line_that_is_not_an_element_is_refused():
    """Property: a ranked-list line that is neither an element nor an all-clear sentence is refused, not skipped.

    Class: malformed ranked lines. Why: a skipped line would shorten a ranking silently. What kills it: a reader
    that ignores unmatched lines. Second correct implementation: a stricter regex. Control: a well-formed line reads.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import read_ranked_list

    with pytest.raises(ProcessingError, match="not 'Node"):
        read_ranked_list(["  Node broken\n"], title="Highest Continuity Errors list")
    _, ids, _ = read_ranked_list(["  Node n1 (1.50%)\n"])
    assert ids[0] == "n1"


def test_the_three_continuity_figures_are_data_variables_that_survive_consolidation(tmp_path, stub_ref):
    """Property: the continuity error, runoff error and flooding loss are data variables, not attributes, and two
    events with different errors keep both values through the consolidation concat.

    Class: the three figures that were dataset attributes. Why: consolidation concatenates events with
    ``combine_attrs="drop_conflicts"``, which deletes an attribute whose value differs across events (measured:
    errors 0.004 and 5.241 lose the attribute and keep ``flow_units``). What kills it: writing them as attributes.
    Second correct implementation: keeping the attributes AND adding variables. Control: the same two values as
    attributes are dropped.
    """
    ds_nodes, _ = _assemble(tmp_path, _ref_text())
    for name in ("flow_continuity_error", "runoff_continuity_error", "system_flooding"):
        assert name in ds_nodes.data_vars and name not in ds_nodes.attrs, name
    assert float(ds_nodes["flow_continuity_error"]) == 5.241
    a = ds_nodes[["flow_continuity_error"]].assign(flow_continuity_error=0.004).expand_dims(event_iloc=[0])
    b = ds_nodes[["flow_continuity_error"]].expand_dims(event_iloc=[1])
    joined = xr.concat([a, b], dim="event_iloc", combine_attrs="drop_conflicts")
    assert list(joined["flow_continuity_error"].values) == [0.004, 5.241]
    attr_a = xr.Dataset(attrs={"flow_continuity_error": 0.004}).expand_dims(event_iloc=[0])
    attr_b = xr.Dataset(attrs={"flow_continuity_error": 5.241}).expand_dims(event_iloc=[1])
    assert (
        "flow_continuity_error"
        not in xr.concat([attr_a, attr_b], dim="event_iloc", combine_attrs="drop_conflicts").attrs
    )


def test_flow_units_and_the_end_datetime_stay_attributes(tmp_path, stub_ref):
    """Property: ``flow_units`` and ``analysis_end_datetime`` stay attributes and survive ``drop_conflicts``.

    Class: the two attributes equal for every event of an analysis. Why: they identify the units and the run, and
    equal values survive the concat. What kills it: moving them to variables. Second correct implementation: none
    needed. Control: an attribute that differs between events is dropped.
    """
    ds_nodes, ds_links = _assemble(tmp_path, _ref_text())
    for ds in (ds_nodes, ds_links):
        assert ds.attrs["flow_units"] == "cms" and "analysis_end_datetime" in ds.attrs
        assert set(ds.attrs) == {"flow_units", "analysis_end_datetime"}
    one = xr.Dataset(attrs={"flow_units": "cms", "x": 1}).expand_dims(event_iloc=[0])
    two = xr.Dataset(attrs={"flow_units": "cms", "x": 2}).expand_dims(event_iloc=[1])
    joined = xr.concat([one, two], dim="event_iloc", combine_attrs="drop_conflicts")
    assert joined.attrs == {"flow_units": "cms"}


def test_run_level_and_ranked_variables_sit_in_the_right_dataset(tmp_path, stub_ref):
    """Property: the flow instability pair is in the link dataset and in no node dataset; every other run-level and
    ranked variable is in the node dataset and in no link dataset.

    Class: all run-level and ranked variables. Why: the instability list names links and the others name nodes or
    the run, and a variable in both datasets would be consolidated twice. What kills it: writing the run-level
    scalars to both. Second correct implementation: a placement table.
    """
    from hhemt import constants as k

    ds_nodes, ds_links = _assemble(tmp_path, _ref_text())
    link_only = set(k.LST_RANKED_LINK_VARIABLES)
    node_only = (
        set(k.LST_RUN_LEVEL_CONTINUITY_VARIABLES)
        | set(k.LST_ROUTING_CONTINUITY_VARIABLES)
        | set(k.LST_COL_HEADERS_ROUTING_TIME_STEP_SUMMARY)
        | set(k.LST_OUTFALL_SYSTEM_VARIABLES)
        | set(k.LST_RANKED_NODE_VARIABLES)
        | {k.ROUTING_STEP_FREQUENCY_VARIABLE}
    )
    assert link_only <= set(ds_links.data_vars) and not link_only & set(ds_nodes.data_vars)
    assert node_only <= set(ds_nodes.data_vars) and not node_only & set(ds_links.data_vars)
    assert ds_nodes["routing_step_frequency"].dims == ("routing_step_bin",)
    assert list(ds_nodes["routing_step_bin_upper"].values) == [0.100, 0.055, 0.030, 0.017, 0.009]


def test_absent_tables_yield_nan_without_error_and_the_reference_yields_values(tmp_path, stub_ref):
    """Property: a Kinematic-Wave-shaped report (no Node Surcharge, Flow Classification, Highest Continuity
    Errors, Time-Step Critical Elements or nonconvergence section) yields NaN for every variable of those tables.

    Class: tables SWMM does not print for the run. Why: an absent table is not zero. What kills it: a reader that
    raises on an absent section or fills zeros. Second correct implementation: building the report from a real
    Kinematic Wave run instead of deleting sections. Control: the Dynamic Wave reference yields values.
    """
    text = _ref_text()
    for title in (
        "Node Surcharge Summary",
        "Flow Classification Summary",
        "Highest Continuity Errors",
        "Time-Step Critical Elements",
        "Most Frequent Nonconverging Nodes",
    ):
        text = _drop_section(text, title)
    assert "Flow Classification Summary" not in text and "Highest Continuity Errors" not in text
    ds_nodes, ds_links = _assemble(tmp_path, text)
    assert ds_nodes["hours_surcharged"].isnull().all()
    assert ds_links["flow_class_dry"].isnull().all()
    for name in ("continuity_error_node_relative", "nonconverging_node_frequency", "courant_critical_frequency"):
        assert ds_nodes[name].isnull().all(), name
    assert (ds_nodes["continuity_error_node_id"] == "").all()
    other = tmp_path / "ref"
    other.mkdir()
    ref_nodes, ref_links = _assemble(other, _ref_text())
    assert ref_nodes["hours_surcharged"].notnull().any() and ref_links["flow_class_dry"].notnull().any()
    assert ref_nodes["continuity_error_node_relative"].notnull().any()


def test_outfall_row_with_a_pollutant_column_reads_and_a_short_row_is_refused(ref_sections):
    """Property: the reference outfall row (6 tokens, pollutant column ``TSS``) reads its four flow cells and the
    System row's maximum without refusal; one token removed is refused naming the table.

    Class: outfall rows of a model that declares one pollutant. Why: the table prints one load column per declared
    pollutant whether or not quality is routed, so the fixed count is five plus the header's pollutant columns.
    What kills it: a fixed count of five. Second correct implementation: deriving the count from the model file's
    ``[POLLUTANTS]``.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import outfall_pollutant_columns, read_outfall_loading_summary

    head = _ref_text().splitlines(keepends=True)
    n = outfall_pollutant_columns(head)
    assert n == 1
    frame, system_max = read_outfall_loading_summary(ref_sections["Outfall Loading Summary"], n)
    assert list(frame.columns) == [
        "outfall_flow_frequency",
        "outfall_average_flow",
        "outfall_max_flow",
        "outfall_total_vol",
    ]
    assert system_max == 0.0 and list(frame.index) == ["dummy_of"]
    with pytest.raises(ProcessingError, match="Outfall Loading Summary"):
        read_outfall_loading_summary(["  dummy_of   0.00   0.000   0.000   0.000\n"], n)
    with pytest.raises(ProcessingError, match="header rule"):
        outfall_pollutant_columns(["  Outfall Loading Summary\n", "  " + "-" * 60 + "\n"])


_TABLES_WITH_PRINTED_ROWS = [
    "Node Depth Summary",
    "Node Surcharge Summary",
    "Storage Volume Summary",
    "Flow Classification Summary",
    "Conduit Surcharge Summary",
]


@pytest.mark.parametrize("title", _TABLES_WITH_PRINTED_ROWS + ["Pumping Summary", "Outfall Loading Summary"])
def test_every_table_reader_refuses_a_row_with_the_wrong_token_count(title, ref_sections):
    """Property: a row with one token removed is refused naming the table, and the unmodified row reads.

    Class: each of the seven fixed-count readers. Why: a row that lost a cell would shift every later column, and a
    shifted column is a wrong number. What kills it: a reader that pads or truncates. Second correct implementation:
    comparing the count to the header's column count.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import outfall_pollutant_columns, read_outfall_loading_summary

    if title == "Pumping Summary":
        lines = [_PUMP_ROW + "\n"]
    else:
        lines = [ref_sections[title][0]]
    broken = [" ".join(lines[0].split()[:-1]) + "\n"]
    if title == "Outfall Loading Summary":
        read_outfall_loading_summary(lines, outfall_pollutant_columns(_ref_text().splitlines(keepends=True)))
        with pytest.raises(ProcessingError, match=title):
            read_outfall_loading_summary(broken, 1)
    else:
        assert len(_table(title, lines)) == 1
        with pytest.raises(ProcessingError, match=title):
            _table(title, broken)


# ------------------------------------------------------------------------------------- the one scan of the report
_NEW_TITLES = (
    "Node Depth Summary",
    "Node Surcharge Summary",
    "Storage Volume Summary",
    "Outfall Loading Summary",
    "Flow Classification Summary",
    "Conduit Surcharge Summary",
    "Flow Routing Continuity",
    "Highest Continuity Errors",
    "Highest Flow Instability Indexes",
    "Most Frequent Nonconverging Nodes",
    "Routing Time Step Summary",
)


@pytest.mark.parametrize("title", _NEW_TITLES)
def test_the_scan_finds_each_new_title_in_a_coupled_report(title):
    """Property: the scan returns the lines of each new title, and a report with one title's banner removed returns
    nothing for that title only.

    Class: the new titles the stock report prints. Why: every reader takes the lines the scan collects, so a title
    the scan misses reads as an unprinted table (NaN). What kills it: a header tuple that lacks a title. Second
    correct implementation: scanning each title with its own pass.
    """
    from hhemt.swmm_output_parser import _REPORT_SECTION_HEADERS, _scan_metadata_and_summaries

    lines = _ref_text().splitlines(keepends=True)
    found = _scan_metadata_and_summaries(lines, _REPORT_SECTION_HEADERS)[0]
    assert found[title], title
    banner = next(i for i, ln in enumerate(lines) if title in ln)
    stripped = lines[:banner] + lines[banner + 1 :]
    again = _scan_metadata_and_summaries(stripped, _REPORT_SECTION_HEADERS)[0]
    assert again[title] == []
    assert again["Node Inflow Summary"] == found["Node Inflow Summary"]


def test_the_scan_stops_at_the_banner_and_takes_the_trailer_from_the_last_lines(tmp_path, stub_ref):
    """Property: replacing the time-series body with unparseable bytes changes no summary or run-level variable,
    for both families; the same corruption before the banner is refused; a report with no ``Analysis ended on``
    line raises through ``_build_system_results``.

    Class: the body between the ``Node Time Series Results`` banner and the final ``Analysis begun on`` line. Why:
    the body is 82 to 99 percent of a coupled report and no reader parses it; a scan that tests every line of it
    reads the trailer from there and is slow, and a scan that parses it is the removed reader. What kills it:
    parsing or testing the body. Second correct implementation: a scan that reads the tail with a backwards seek.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import parse_rpt_single_pass, read_report_table

    text = _ref_text()
    line_start = text.rindex("\n", 0, text.index(BANNER)) + 1
    last_begun = text.rindex("  Analysis begun on")
    garbage = "".join(f"\x00\xff\xfe garbage {i} {{{{\n" for i in range(200))
    corrupted = text[:line_start] + "  " + BANNER + "\n" + garbage + "\n" + text[last_begun:]
    good = tmp_path / "good.rpt"
    bad = tmp_path / "bad.rpt"
    good.write_text(text, encoding="latin-1")
    bad.write_bytes(corrupted.encode("latin-1"))
    g_sec, g_sys, g_valid = parse_rpt_single_pass(good)
    b_sec, b_sys, b_valid = parse_rpt_single_pass(bad)
    assert (g_sec, g_sys, g_valid) == (b_sec, b_sys, b_valid)
    for title in _TABLES_WITH_PRINTED_ROWS:
        assert _table(title, g_sec[title]).equals(_table(title, b_sec[title]))
    # control 1: corruption BEFORE the banner is refused (a Node Depth Summary row loses a token)
    first = g_sec["Node Depth Summary"][0]
    damaged = text.replace(first, " ".join(first.split()[:-1]) + "\n", 1)
    sec, _, _ = parse_rpt_single_pass(_write(tmp_path / "damaged.rpt", damaged))
    with pytest.raises(ProcessingError, match="Node Depth Summary"):
        read_report_table("Node Depth Summary", sec["Node Depth Summary"])
    # control 2: no `Analysis ended on` line raises through _build_system_results
    trailer = "\n".join(ln for ln in text.splitlines() if "Analysis ended on" not in ln)
    with pytest.raises(ValueError, match="Analysis end line"):
        parse_rpt_single_pass(_write(tmp_path / "no_end.rpt", trailer))
    # the assembled datasets of both families are identical whatever the body holds
    for family in ("tritonswmm", "swmm"):
        one, two = tmp_path / f"a_{family}", tmp_path / f"b_{family}"
        one.mkdir()
        two.mkdir()
        good_nodes, good_links = _assemble(one, text, family=family)
        bad_nodes, bad_links = _assemble(two, corrupted, family=family)
        xr.testing.assert_identical(good_nodes, bad_nodes)
        xr.testing.assert_identical(good_links, bad_links)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="latin-1")
    return path


@pytest.mark.parametrize("family", ["tritonswmm", "swmm"])
def test_a_report_the_scan_cannot_read_raises_for_both_families_and_never_warns(tmp_path, stub_ref, family):
    """Property: a report with no ``Flow Units`` line raises through the ``.out`` path and emits no ``UserWarning``.

    Class: both families' assembly over an unreadable report. Why: the ``.out`` path warned and carried on, so
    a coupled report failure would have written a store whose summary variables were silently absent. What kills it:
    wrapping the report read in ``except: warnings.warn``. Second correct implementation: raising a ProcessingError
    instead of the ValueError the builder raises. Control: the complete report reads.
    """
    text = _ref_text()
    broken = "\n".join(ln for ln in text.splitlines() if "Flow Units" not in ln)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Flow units line"):
            _assemble(tmp_path, broken, family=family)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    ok = tmp_path / "ok"
    ok.mkdir()
    ds_nodes, _ = _assemble(ok, text, family=family)
    assert "depth" in ds_nodes


def test_a_truncated_coupled_report_still_yields_the_new_tables(tmp_path):
    """Property: after ``_truncate_coupled_rpt`` every new reader returns what it returns on the untruncated copy.

    Class: the reclaimed (head plus marker plus trailer) form of a coupled report. Why: ``remove_after_processing``
    truncates the report while the series are read from the binary output, so a reprocess reads the truncated head.
    What kills it: a scan that needs the body. Second correct implementation: a truncator that keeps more.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P
    from hhemt.swmm_output_parser import parse_rpt_single_pass

    full = _write(tmp_path / "full.rpt", _ref_text())
    cut = _write(tmp_path / "cut.rpt", _ref_text())
    assert P._truncate_coupled_rpt(cut, False) is True
    assert cut.stat().st_size < full.stat().st_size / 2
    f_sec, f_sys, _ = parse_rpt_single_pass(full)
    c_sec, c_sys, _ = parse_rpt_single_pass(cut)
    assert f_sec == c_sec and f_sys == c_sys
    assert c_sys["analysis_end_datetime"]


def test_the_retired_report_reader_is_gone(tmp_path):
    """Property: no removed report-reader name appears in ``src`` or ``tests`` at the working tree.

    Class: the removed names. Why: the report's per-timestep reader was replaced by the binary-output reader, and a
    surviving definition or caller means the retirement is half done. The matcher is built here from fragments so
    this file does not contain the names it forbids. What kills it: leaving a definition. Second correct
    implementation: an AST walk for definitions and imports. Controls: the same matcher over ``f22181a3`` returns
    lines, and a scratch file holding one removed name is found.
    """
    names = [
        "_bulk" + "_parse_tseries",
        "_bulk" + "_parse_section",
        "_RE_TSERIES" + "_BLOCK",
        "_RE_NODE_TSERIES" + "_SECTION",
        "_RE_LINK_TSERIES" + "_SECTION",
        "parse_rpt_pollutant" + "_tseries",
        "_bulk_parse_pollutant" + "_section",
        "return_node_time_series_results_from" + "_rpt",
        "_build_tseries" + "_datasets",
        "create_tseries" + "_ds",
    ]
    pattern = r"\b(" + "|".join(names) + r")\b"

    def grep(*args, cwd=REPO):
        proc = subprocess.run(["git", "grep", "-nE", pattern, *args], cwd=cwd, capture_output=True, text=True)
        return proc.stdout.splitlines()

    assert grep("--", "src", "tests") == []
    pinned = subprocess.run(["git", "cat-file", "-t", "f22181a3"], cwd=REPO, capture_output=True, text=True)
    assert pinned.stdout.strip() == "commit", "the pre-retirement commit f22181a3 is not in this clone"
    assert grep("f22181a3", "--", "src", "tests"), "the matcher finds nothing at the pre-retirement commit"
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "scratch.py").write_text(f"x = {names[0]}\n")
    found = subprocess.run(
        ["git", "grep", "--no-index", "-nE", pattern], capture_output=True, text=True, cwd=scratch_dir
    ).stdout.splitlines()
    assert len(found) == 1


# --------------------------------------------------------------------------------------- the assembly and its guards
def test_a_rpt_argument_is_refused_naming_the_sibling_out(tmp_path, stub_ref):
    """Property: ``retrieve_SWMM_outputs_as_datasets`` given a ``.rpt`` path raises, naming the sibling ``.out``.

    Class: any non-``.out`` path. Why: the report's series reader is removed, so a ``.rpt`` argument would silently
    mean "no series"; the refusal tells the caller what to pass. What kills it: accepting the report. Second correct
    implementation: a ValueError. Control: the sibling ``.out`` path reads.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    out, rpt = _stage(tmp_path, _ref_text())
    with pytest.raises(ProcessingError, match=r"hydraulics\.out"):
        retrieve_SWMM_outputs_as_datasets(REF_INP, rpt, family="tritonswmm")
    ds_nodes, _ = retrieve_SWMM_outputs_as_datasets(REF_INP, out, family="tritonswmm")
    assert "depth" in ds_nodes


def test_the_coupled_family_reads_every_series_from_the_binary_output(tmp_path, stub_ref):
    """Property: the coupled assembly reads all eleven series through pyswmm and names the capacity series
    ``capacity_setting``; the standalone assembly names it ``capacity``.

    Class: both families. Why: the report holds only four node and four link series, rounded to three decimals,
    and the binary output holds six and five at single precision; the two families name the link capacity series
    differently. What kills it: reading any series from the report, or one capacity name for both. Second correct
    implementation: reading the three new series with a second pass. Control: each family's store holds no variable
    of the other's capacity name.
    """
    ds_n, ds_l = _assemble(tmp_path, _ref_text(), family="tritonswmm")
    assert {a for k, a in stub_ref if k == "node"} == {
        "invert_depth",
        "hydraulic_head",
        "total_inflow",
        "flooding_losses",
        "ponded_volume",
        "lateral_inflow",
    }
    assert {a for k, a in stub_ref if k == "link"} == {
        "flow_rate",
        "flow_velocity",
        "flow_depth",
        "capacity",
        "flow_volume",
    }
    assert "capacity_setting" in ds_l and "capacity" not in ds_l
    other = tmp_path / "swmm"
    other.mkdir()
    _, ds_l2 = _assemble(other, _ref_text(), family="swmm")
    assert "capacity" in ds_l2 and "capacity_setting" not in ds_l2


def test_the_new_series_land_under_their_stored_names_on_the_axis_of_depth(tmp_path, stub_ref):
    """Property: ``volume`` is pyswmm's ponded volume, ``lateral_inflow`` its lateral inflow and ``link_volume`` its
    flow volume, each on the ``date_time`` axis of ``depth``.

    Class: the three series the report cannot supply. Why: the stand-in makes each a distinct function of a printed
    series, so a swapped attribute is visible. What kills it: mapping ``volume`` to a different attribute. Second
    correct implementation: reading by attribute index instead of name.
    """
    ds_n, ds_l = _assemble(tmp_path, _ref_text())
    assert ds_n["volume"].dims == ds_n["depth"].dims == ("node_id", "date_time")
    assert np.allclose(ds_n["volume"].values, 10.0 * ds_n["depth"].values, equal_nan=True)
    assert np.allclose(ds_n["lateral_inflow"].values, -ds_n["inflow_flow"].values, equal_nan=True)
    assert np.allclose(ds_l["link_volume"].values, 3.0 * ds_l["flow"].values, equal_nan=True)
    assert ds_l["link_volume"].dims == ds_l["flow"].dims == ("link_id", "date_time")


def test_pyswmm_attribute_names_are_the_binary_output_variables_they_are_read_as():
    """Property: the pyswmm attributes read for ``volume``, ``lateral_inflow`` and ``link_volume`` are the indexes of
    SWMM's NODE_VOLUME (2), NODE_LATFLOW (3) and LINK_VOLUME (3).

    Class: pyswmm's attribute enums. Why: the stand-in cannot show that ``ponded_volume`` is the node's whole stored
    volume; the enum value can, against `enums.h` (NODE_VOLUME 2, NODE_LATFLOW 3, LINK_VOLUME 3). What kills it:
    a pyswmm release that renumbers. Second correct implementation: reading `swmm.toolkit.shared_enum` directly.
    """
    from swmm.toolkit import shared_enum

    assert shared_enum.NodeAttribute.PONDED_VOLUME.value == 2
    assert shared_enum.NodeAttribute.LATERAL_INFLOW.value == 3
    assert shared_enum.LinkAttribute.FLOW_VOLUME.value == 3


def test_reductions_and_pass_through_in_the_summary(tmp_path, stub_ref):
    """Property: ``summarize_swmm_simulation_results`` gives the six new reductions and carries every run-level and
    ranked variable with ``event_iloc`` and unchanged values.

    Class: the assembled coupled dataset. Why: the summary reduces every variable carrying ``date_time`` and passes
    the others through, so the new series must reduce and the static variables must not. What kills it: a summary
    that drops a variable without ``date_time``. Second correct implementation: carrying them in a second store.
    """
    from hhemt import constants as k
    from hhemt.process_simulation import summarize_swmm_simulation_results

    ds_nodes, ds_links = _assemble(tmp_path, _ref_text())
    summary = summarize_swmm_simulation_results(ds_nodes.copy(deep=True), 3)
    for stem in ("volume", "lateral_inflow"):
        for suffix in ("max", "last"):
            assert f"{stem}_{suffix}" in summary
    link_summary = summarize_swmm_simulation_results(ds_links.copy(deep=True), 3)
    assert "link_volume_max" in link_summary and "link_volume_last" in link_summary
    for name in (*k.LST_RUN_LEVEL_CONTINUITY_VARIABLES, *k.LST_RANKED_NODE_VARIABLES):
        assert summary[name].dims[0] == "event_iloc", name
        assert np.array_equal(
            summary[name].isel(event_iloc=0).values, ds_nodes[name].values, equal_nan=ds_nodes[name].dtype.kind == "f"
        ), name
    assert "depth" not in summary and "volume" not in summary


def test_assembled_datasets_round_trip_through_zarr_with_their_string_ranks(tmp_path, stub_ref):
    """Property: the assembled node and link datasets write to zarr and read back with ranked ids intact and an
    unprinted id as the empty string.

    Class: the assembled datasets, including the five string-valued ranked variables and 0-d scalars. Why: an
    object array cannot be written by ``write_zarr`` and a unicode array cannot hold NaN, so an unprinted rank is
    the empty string. What kills it: padding with NaN into an object array. Second correct implementation: a
    fixed-width bytes dtype.
    """
    from hhemt.utils import write_zarr

    ds_nodes, ds_links = _assemble(tmp_path, _ref_text())
    write_zarr(ds_nodes, tmp_path / "n.zarr", compression_level=5)
    write_zarr(ds_links, tmp_path / "l.zarr", compression_level=5)
    n = xr.open_dataset(tmp_path / "n.zarr", engine="zarr", consolidated=False).load()
    assert n["continuity_error_node_id"].values.tolist()[:2] == ["05-0088", "E133468"]
    assert n["courant_critical_element_id"].values.tolist() == [""] * 5
    assert float(n["flow_continuity_error"]) == 5.241
    lk = xr.open_dataset(tmp_path / "l.zarr", engine="zarr", consolidated=False).load()
    assert lk["flow_instability_link_id"].values.tolist() == ["1444", "597", "1084", "442", "441"]


def test_assembly_reproduces_the_reference_series_from_a_stand_in_binary_output(tmp_path, stub_ref):
    """Property: the assembled series equal the committed reference series stores within the report's print
    rounding, for the coupled family.

    Class: the four node and four link series the reference stores hold. Why: this is the laptop-side check that
    ids, dimensions, names and ordering of the new assembly match the stores the report reader produced; the same
    comparison against the REAL binary output waits on `hydraulics.out` (see
    `tests/test_swmm_output_parser_refactoring.py`). What kills it: a transposed or mislabelled series. Second correct
    implementation: comparing through pandas. Control: one value offset by 0.001 fails.
    """
    ds_nodes, ds_links = _assemble(tmp_path, _ref_text())
    ref_n = xr.open_dataset(REF_DIR / "SWMM_node_tseries.zarr", engine="zarr", consolidated=False)
    ref_l = xr.open_dataset(REF_DIR / "SWMM_link_tseries.zarr", engine="zarr", consolidated=False)
    for name in ("depth", "head", "inflow_flow", "flooding"):
        a = ds_nodes[name].sel(node_id=ref_n["node_id"].values, date_time=ref_n["date_time"].values)
        assert np.allclose(a.values, ref_n[name].values, atol=5e-4 + 1e-9, equal_nan=True), name
    for name in ("flow", "velocity", "link_depth", "capacity_setting"):
        a = ds_links[name].sel(link_id=ref_l["link_id"].values, date_time=ref_l["date_time"].values)
        assert np.allclose(a.values, ref_l[name].values, atol=5e-4 + 1e-9, equal_nan=True), name
    shifted = ds_nodes["depth"].values.copy()
    shifted[0, 0] += 0.001
    assert not np.allclose(
        shifted, ref_n["depth"].sel(node_id=ds_nodes["node_id"].values).values, atol=5e-4, equal_nan=True
    )


# ------------------------------------------------------------------------------------ the coupled export's guard
def test_a_coupled_export_with_no_binary_output_raises_and_writes_nothing(tmp_path):
    """Property: ``_export_SWMM_outputs`` for the coupled model with no ``hydraulics.out`` beside the report raises
    a ProcessingError naming the file and ``raw_swmm_binaries``, and no store is written.

    Class: a coupled scenario whose binary output was reclaimed. Why: every coupled series is read from the file,
    and the one setting that deletes it fires after the summaries are verified, so a reprocess must say why it
    cannot rebuild the series instead of dropping them. What kills it: letting the missing file fall through to a
    FileNotFoundError from pyswmm, or writing the summaries first. Second correct implementation: the check
    inside `retrieve_SWMM_outputs_as_datasets`. Control: with the file present the helper returns its path.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P
    from hhemt.process_simulation import require_coupled_swmm_binary_output

    rpt = tmp_path / "hydraulics.rpt"
    rpt.write_text("x")
    written: list = []
    fake = SimpleNamespace(
        _resolve_clear_raw=lambda override: "none",
        _validate_path=lambda path, name: Path(tmp_path / f"{name}.zarr"),
        scen_paths=SimpleNamespace(
            output_tritonswmm_node_time_series=tmp_path / "n.zarr",
            output_tritonswmm_link_time_series=tmp_path / "l.zarr",
            swmm_hydraulics_inp=REF_INP,
            swmm_hydraulics_rpt=rpt,
        ),
        _swmm_node_outputs_processed=lambda model: False,
        _swmm_link_outputs_processed=lambda model: False,
        _write_output=lambda *a, **k: written.append(a),
        log=SimpleNamespace(deferred_writes=lambda: __import__("contextlib").nullcontext()),
    )
    with pytest.raises(ProcessingError) as excinfo:
        P._export_SWMM_outputs.__wrapped__(fake, "tritonswmm")
    message = str(excinfo.value)
    assert "hydraulics.out" in message and "raw_swmm_binaries" in message
    assert written == []
    (tmp_path / "hydraulics.out").write_bytes(b"")
    assert require_coupled_swmm_binary_output(rpt) == tmp_path / "hydraulics.out"


def test_omitting_the_family_is_a_type_error(tmp_path, stub_ref):
    """Property: ``retrieve_SWMM_outputs_as_datasets`` and ``return_swmm_outputs`` refuse a call without ``family``.

    Class: callers that omit the keyword. Why: a default would name a coupled run's link capacity ``capacity`` with no
    error. What kills it: restoring ``family="swmm"`` as the default. Second correct implementation: a positional
    required argument. Control: the same call with the keyword reads.
    """
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    out, _ = _stage(tmp_path, _ref_text())
    with pytest.raises(TypeError, match="family"):
        retrieve_SWMM_outputs_as_datasets(REF_INP, out)
    ds_nodes, _ = retrieve_SWMM_outputs_as_datasets(REF_INP, out, family="tritonswmm")
    assert "depth" in ds_nodes


def test_a_binary_output_with_no_nodes_or_links_is_refused_naming_the_report_lines(tmp_path, monkeypatch, ref_series):
    """Property: an output holding no nodes (or no links) raises, naming ``NODES ALL`` and ``LINKS ALL``.

    Class: a model whose [REPORT] section lists no nodes or links, so SWMM writes no series. Why: the silent
    alternative is a store with no series at all. What kills it: returning the empty dataset. Second correct
    implementation: checking after assembly. Control: the full stand-in reads.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    nodes, links = ref_series
    out, _ = _stage(tmp_path, _ref_text())
    for n, ln in (({}, links), (nodes, {})):
        _install_stub_pyswmm(monkeypatch, n, ln)
        with pytest.raises(ProcessingError, match="NODES ALL"):
            retrieve_SWMM_outputs_as_datasets(REF_INP, out, family="tritonswmm")
    _install_stub_pyswmm(monkeypatch, nodes, links)
    assert "depth" in retrieve_SWMM_outputs_as_datasets(REF_INP, out, family="tritonswmm")[0]
