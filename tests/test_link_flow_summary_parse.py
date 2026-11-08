"""The Link Flow Summary reader: fixed columns measured from the end of the id field, one row type at a time.

SWMM prints each link row with `fprintf` formats in `statsrpt.c` (`writeLinkFlow`): the id as `%-20s`, the type as
a 9-character cell, the largest flow as `%9.3f` (CMS) or `%9.2f`, the time as `  %4d  %02d:%02d`, then the
velocity (`   %7.2f`, or `    >50.00` above 50), the flow ratio and the depth ratio (`  %6.2f` each). Which of the
last three a row prints depends on its type. The rows below are assembled with those same formats, except the
orifice row and the conduit control row, which are read from a stored report, and the outlet-link row, which stock SWMM
printed. No TRITON-SWMM solver runs here.

The reader is new in this change and is looked up lazily inside each test, so that on a tree without the chunk
each test collects and fails with an AssertionError from this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HEADERS = [
    "link_id",
    "type",
    "max_flow",
    "time_of_max_flow_d_hr_mn",
    "max_velocity",
    "max_over_full_flow",
    "max_over_full_depth",
]
_STORED_REPORT = Path(__file__).resolve().parents[1] / "test_data" / "swmm_refactoring_reference" / "hydraulics.rpt"


def _reader():
    from hhemt import swmm_output_parser

    fn = getattr(swmm_output_parser, "format_link_flow_summary", None)
    assert fn is not None, "swmm_output_parser.format_link_flow_summary does not exist"
    return fn


def _read(*lines: str) -> list[dict]:
    frame = _reader()(list(lines), _HEADERS)
    return frame.to_dict("records")


def _id(link_id: str) -> str:
    return f"  {link_id:<20}"


def _type(kind: str) -> str:
    return {"DUMMY": " DUMMY   ", "CHANNEL": " CHANNEL "}.get(kind, f" {kind:<7} ")


def _flow_time(flow: float, days: int, hh: int, mm: int) -> str:
    return f"{flow:9.3f}  {days:4d}  {hh:02d}:{mm:02d}"


def _velocity(v: float) -> str:
    return "    >50.00" if v > 50.0 else f"   {v:7.2f}"


def _ratio(r: float) -> str:
    return f"  {r:6.2f}"


_BLANK_VELOCITY_AND_RATIO = " " * 18


def _row(link_id, kind, flow, tail) -> str:
    return _id(link_id) + _type(kind) + _flow_time(flow, 0, 1, 30) + tail + "\n"


def _stored_rows() -> dict[str, str]:
    from hhemt.swmm_output_parser import return_lines_for_section_of_rpt

    lines = return_lines_for_section_of_rpt("Link Flow Summary", f_rpt=_STORED_REPORT)
    return {line.split()[0]: line for line in lines}


def test_a_conduit_row_reads_every_column() -> None:
    """Property: a conduit row (the control) lands each printed cell in the column its header names.

    Class: NEW CAPABILITY. Plausible wrong implementation: a reader that mis-measures every row, which the
    non-conduit rows below could not distinguish from a type-specific fault.
    Reachability: the row is built with the solver's own formats; the stored-report conduit row beside it is the
    same case read from a real report. Both-states anchor: asserts printed numbers in named columns.
    Killed by: any offset error in the end-of-id measurement. A second correct implementation splits the row with
    one regular expression; the assertions read the frame, not the method.
    """
    row = _read(_row("C1", "CONDUIT", 1.234, _velocity(0.56) + _ratio(0.12) + _ratio(0.34)))[0]
    assert row == {
        "link_id": "C1",
        "type": "CONDUIT",
        "max_flow": "1.234",
        "time_of_max_flow_d_hr_mn": "0 01:30",
        "max_velocity": "0.56",
        "max_over_full_flow": "0.12",
        "max_over_full_depth": "0.34",
    }
    stored = _stored_rows()
    first = next(line for line in stored.values() if " CONDUIT " in line)
    parsed = _read(first)[0]
    assert parsed["type"] == "CONDUIT" and all(parsed[c] != "" for c in _HEADERS)


def test_a_pump_with_a_curve_reads_its_ratio_in_the_flow_ratio_column() -> None:
    """Property: a `PUMP` row that ends after its ratio puts the ratio in `max_over_full_flow`, and leaves
    `max_velocity` and `max_over_full_depth` empty.

    Class: NEW CAPABILITY. Plausible wrong implementations: the whitespace-positional reader, which puts the ratio
    in `max_velocity`; or right-anchored cells, which put it in `max_over_full_depth`.
    Reachability: the row is `statsrpt.c`'s pump branch (ten blank characters, then `  %6.2f`).
    Both-states anchor: asserts which column holds `0.22` and which are empty.
    Must-fire control: the same text read at right-anchored positions puts `0.22` in the depth column, which the
    assertion below shows would fail this test.
    Killed by: either wrong reader. A second correct implementation measures from the start of the type cell.
    """
    line = _row("P1", "PUMP", 0.5, " " * 10 + _ratio(0.22))
    row = _read(line)[0]
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "0.22", "")
    assert line.rstrip("\n")[-8:].strip() == "0.22"  # the right-anchored depth cell would read the ratio


def test_a_weir_row_reads_only_a_depth_ratio() -> None:
    """Property: a `WEIR` row lands its Max/Full Depth in `max_over_full_depth`; velocity and flow ratio are empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: whitespace tokens, which put the depth ratio in
    `max_velocity`. Reachability: the weir branch prints 18 blanks, then the depth cell.
    Both-states anchor: asserts the three cells. Killed by: whitespace tokenising. A second correct implementation
    reads fixed columns from the left edge of the type cell.
    """
    row = _read(_row("W1", "WEIR", 0.2, _BLANK_VELOCITY_AND_RATIO + _ratio(0.33)))[0]
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "", "0.33")


def test_an_orifice_row_from_a_stored_report_reads_each_printed_cell_in_its_column() -> None:
    """Property: the stored report's `ORIFICE` row lands its flow, time and depth ratio in their columns and
    leaves velocity and flow ratio empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: whitespace tokens put the depth ratio in `max_velocity`.
    Reachability: the row is read from `test_data/swmm_refactoring_reference/hydraulics.rpt` (link `756`).
    Both-states anchor: asserts printed values. Killed by: whitespace tokenising. A second correct implementation
    uses a regular expression with named groups.
    """
    row = _read(_stored_rows()["756"])[0]
    assert row["type"] == "ORIFICE"
    assert (row["max_flow"], row["time_of_max_flow_d_hr_mn"]) == ("0.000", "0 00:00")
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "", "0.00")


# The Link Flow Summary row stock SWMM printed for a model with ONE outlet link
# (`[OUTLETS] OL1 J1 O1 0 TABULAR/HEAD RC1`, a junction feeding a free outfall, 2 hours of dynamic-wave
# routing). Produced with
#   python -c "from swmm.toolkit import solver; solver.swmm_run('outlet.inp', 'outlet.rpt', 'outlet.out')"
# in the hhemt environment (pyswmm 2.0.1, whose bundled engine reports "EPA STORM WATER MANAGEMENT MODEL - VERSION
# 5.2 (Build 5.2.4)"), copying the one data line of the report's Link Flow Summary. No TRITON-SWMM solver ran.
# The printed type word is `DUMMY`, not `OUTLET`: an outlet link has a dummy cross-section, and `statsrpt.c` tests
# the cross-section before the link type.
_STOCK_OUTLET_ROW = "  OL1                  DUMMY       0.107     0  00:00\n"


def test_an_outlet_link_row_printed_by_stock_swmm_reads_flow_and_time_and_nothing_else() -> None:
    """Property: the row stock SWMM printed for an outlet link reads id, type, flow and time; every later cell is
    empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: a reader that expects every row to carry seven cells and
    indexes past the row's end, or that shifts the time cell of a row with no later cells.
    Reachability: the fixture is a line stock SWMM 5.2.4 printed (see the comment above it), not one assembled from
    the engine's format strings. Both-states anchor: asserts the printed cells and the empty trailing cells.
    Not established by this test: that a row with the word OUTLET exists. Stock SWMM prints `DUMMY` for an outlet
    link, so the type word `OUTLET` is never printed and is not tested.
    Killed by: indexing past the row's end. A second correct implementation returns None for the empty cells.
    """
    row = _read(_STOCK_OUTLET_ROW)[0]
    assert (row["link_id"], row["type"], row["max_flow"], row["time_of_max_flow_d_hr_mn"]) == (
        "OL1",
        "DUMMY",
        "0.107",
        "0 00:00",
    )
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "", "")


def test_a_link_id_containing_a_colon_is_read_whole() -> None:
    """Property: a link `C1:C2` reads `link_id` `C1:C2` and `type` `CONDUIT`.

    Class: NEW CAPABILITY. Plausible wrong implementation: the shared whitespace reader joins any token containing
    a colon to the previous cell, so the id would lose its tail and the type would shift.
    Reachability: the colon-bearing id is a single token in the printed row. Both-states anchor: asserts the id
    and type. Killed by: the whitespace reader. A second correct implementation splits on the first two spaces.
    """
    row = _read(_row("C1:C2", "CONDUIT", 1.0, _velocity(0.1) + _ratio(0.2) + _ratio(0.3)))[0]
    assert (row["link_id"], row["type"]) == ("C1:C2", "CONDUIT")
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("0.10", "0.20", "0.30")


def test_a_31_character_id_shifts_nothing() -> None:
    """Property: a conduit with a 31-character id reads its id whole and every later cell in its own column.

    Class: NEW CAPABILITY. Plausible wrong implementation: cells at positions measured from the left edge assume
    the 20-character id field, so every later cell of this row would shift 11 characters.
    Reachability: `%-20s` pads but does not truncate. Both-states anchor: asserts the id and cells.
    Must-fire control: the same row read at left-anchored positions gives different cells, shown below.
    Killed by: left-anchored cells. A second correct implementation measures from the type cell.
    """
    link_id = "LONG_LINK_IDENTIFIER_0123456789"
    assert len(link_id) == 31
    line = _row(link_id, "CONDUIT", 0.0, _velocity(0.0) + _ratio(0.0) + _ratio(0.0))
    row = _read(line)[0]
    assert row["link_id"] == link_id and row["type"] == "CONDUIT"
    assert (row["max_flow"], row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == (
        "0.000",
        "0.00",
        "0.00",
        "0.00",
    )
    left_anchored_type = line.rstrip("\n")[2 + 20 : 2 + 20 + 9].strip()
    assert left_anchored_type != "CONDUIT"  # the left-anchored read would not even find the type


def test_a_pump_with_a_curve_is_not_read_at_right_anchored_positions() -> None:
    """Property: a pump row's ratio is in the flow-ratio column and its depth column is empty, which a
    right-anchored reader cannot say.

    Class: NEW CAPABILITY, a differently-positioned input to the pump test: the row is a conduit's neighbour in
    the same call, so any state carried between rows would show. Plausible wrong implementation: right-anchored
    cells (last 8 characters = depth). Both-states anchor: the assertion is on the cell values. Killed by:
    right anchoring. A second correct implementation anchors on the type cell.
    """
    pump = _row("P1", "PUMP", 0.5, " " * 10 + _ratio(0.22))
    conduit = _row("C9", "CONDUIT", 2.0, _velocity(1.0) + _ratio(0.5) + _ratio(0.6))
    pump_row, conduit_row = _read(pump, conduit)
    assert pump_row["max_over_full_depth"] == "" and pump_row["max_over_full_flow"] == "0.22"
    assert conduit_row["max_over_full_depth"] == "0.60"


def test_a_dummy_row_reads_id_type_flow_and_time_only() -> None:
    """Property: a `DUMMY` row reads id, type, largest flow and time; every later cell is empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: indexing the missing cells of a short row.
    Reachability: the dummy branch `continue`s after the time cell. Both-states anchor: empty trailing cells.
    Killed by: that implementation. A second correct implementation pads with None.
    """
    row = _read(_row("DUM1", "DUMMY", 0.0, ""))[0]
    assert row["type"] == "DUMMY" and row["max_flow"] == "0.000"
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "", "")


def test_a_pump_without_a_curve_ends_after_its_time_cell() -> None:
    """Property: a `PUMP` row that prints no ratio reads id, type, flow and time; later cells are empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: assuming every pump row carries a ratio.
    Reachability: the pump branch prints a ratio only when `qFull > 0`. Both-states anchor: trailing cells empty.
    Killed by: that assumption. A second correct implementation checks the row length first.
    """
    row = _read(_row("P2", "PUMP", 0.1, ""))[0]
    assert row["type"] == "PUMP"
    assert (row["max_velocity"], row["max_over_full_flow"], row["max_over_full_depth"]) == ("", "", "")


def test_a_channel_row_reads_like_a_conduit_row() -> None:
    """Property: a `CHANNEL` row (an irregular cross-section, printed with that word) reads every cell in the
    same columns as a `CONDUIT` row.

    Class: NEW CAPABILITY. Plausible wrong implementation: a type-width assumption that breaks on the 9-character
    ` CHANNEL ` cell. Reachability: `statsrpt.c` prints ` CHANNEL ` for an irregular section. Both-states anchor:
    the two rows differ only in the type word. Killed by: a reader that treats the type as `%-7s`. A second
    correct implementation uses a regular expression.
    """
    tail = _velocity(0.56) + _ratio(0.12) + _ratio(0.34)
    channel, conduit = _read(_row("CH1", "CHANNEL", 1.234, tail), _row("CH2", "CONDUIT", 1.234, tail))
    assert channel["type"] == "CHANNEL"
    assert {k: v for k, v in channel.items() if k not in ("link_id", "type")} == {
        k: v for k, v in conduit.items() if k not in ("link_id", "type")
    }


@pytest.mark.parametrize("velocity", [51.0, 120.5])
def test_a_velocity_above_fifty_reads_as_text_and_leaves_the_ratios_in_place(velocity) -> None:
    """Property: a velocity cell printed `>50.00` reads that text in `max_velocity`, and the ratio and depth cells
    land in their own columns.

    Class: NEW CAPABILITY. Plausible wrong implementation: parsing the cell as a number, or a token reader that
    drops `>50.00` and shifts the ratios left. Reachability: `statsrpt.c` prints `    >50.00` for v above 50.
    Both-states anchor: asserts the text and the following cells; the control row below it prints a plain number.
    Killed by: those. A second correct implementation keeps the velocity as text.
    """
    fast = _read(_row("C1", "CONDUIT", 9.0, _velocity(velocity) + _ratio(0.5) + _ratio(0.6)))[0]
    assert (fast["max_velocity"], fast["max_over_full_flow"], fast["max_over_full_depth"]) == (">50.00", "0.50", "0.60")
    plain = _read(_row("C1", "CONDUIT", 9.0, _velocity(49.0) + _ratio(0.5) + _ratio(0.6)))[0]
    assert plain["max_velocity"] == "49.00"


def test_the_report_pipeline_reads_the_link_flow_summary_through_the_fixed_column_reader(monkeypatch) -> None:
    """Property: `retrieve_SWMM_outputs_as_datasets` builds the link summary with `format_link_flow_summary`, so
    the stored report's `ORIFICE` link keeps its depth ratio in `max_over_full_depth`.
    (The velocity cell of that link is not asserted: the dataset conversion turns every empty cell into 0.0, which
    is how the pipeline behaved before this reader and is outside this chunk.)

    Class: NEW CAPABILITY. Plausible wrong implementation: the reader exists and is tested but the pipeline still
    calls the whitespace reader (a wiring fault the direct tests above cannot see).
    Reachability: the stored `.rpt` beside the `.out` is read by `return_swmm_outputs`. Both-states anchor: the
    wrapped call is observed, and the orifice cell is asserted on the resulting dataset.
    Killed by: the pipeline calling the old reader. A second correct implementation inlines the reader; the
    spy would then see no call, so this test pins the NAME as the integration point, not the behaviour alone.
    """
    from hhemt import swmm_output_parser

    real = _reader()
    calls = []

    def spy(lines, headers):
        """Record the number of section lines and delegate to the real reader."""
        calls.append(len(lines))
        return real(lines, headers)

    monkeypatch.setattr(swmm_output_parser, "format_link_flow_summary", spy)
    folder = _STORED_REPORT.parent
    # The series come from a binary output the laptop cannot produce, so a stand-in holding the stored report's printed
    # series is installed; the `.out` path is never opened, and the report beside it is the stored one.
    from tests.test_swmm_output_inventory import _body_series, _install_stub_pyswmm

    _install_stub_pyswmm(monkeypatch, *_body_series(_STORED_REPORT.read_text(encoding="latin-1")))
    _, ds_links = swmm_output_parser.retrieve_SWMM_outputs_as_datasets(
        folder / "hydraulics.inp", _STORED_REPORT.with_suffix(".out"), family="tritonswmm"
    )
    assert len(calls) == 1 and calls[0] > 1000
    orifice = ds_links.sel(link_id="756")
    assert str(orifice["max_over_full_depth"].values) in ("0.0", "0")
