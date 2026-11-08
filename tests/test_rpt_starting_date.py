"""The SWMM report's Starting Date is found wherever the report places it."""

from __future__ import annotations

from hhemt.swmm_output_parser import read_rpt_starting_date

_OPTIONS = "  Starting Date ............ 08/31/2025 00:00:00\n  Ending Date .............. 09/03/2025 00:00:00\n"
_WARNING = "  WARNING 05: minimum slope used for Conduit {n}\n"


def _report(tmp_path, n_warnings: int):
    body = "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)\n\n"
    body += "".join(_WARNING.format(n=i) for i in range(n_warnings))
    body += "\n  ****************\n  Analysis Options\n  ****************\n" + _OPTIONS
    path = tmp_path / "hydraulics.rpt"
    path.write_text(body, encoding="latin-1")
    return path


def test_starting_date_is_found_after_thousands_of_warning_lines(tmp_path):
    """Property: the value is returned whatever number of warning lines precede the Analysis Options block.

    Class: REGRESSION. Plausible wrong implementation: reading a fixed number of leading lines. A coupled Norfolk
    report holds the line at 4147 of 924694, so a 200-line head read finds no date and processing stops in every
    scenario.
    Why required: the coupling exchange flow is placed on the report's own clock, so a missed date stops processing.
    What kills it: any head limit below the line number of the match (the test puts it past line 4000).
    Second correct implementation: a regular expression over the whole file text, which returns the same value.
    """
    assert read_rpt_starting_date(_report(tmp_path, 4000)) == "08/31/2025 00:00:00"


def test_starting_date_is_found_in_a_short_report(tmp_path):
    """Property: a report with no warnings still returns the value.

    Class: GUARD against over-correction.
    Plausible wrong implementation: a reader that skips a warning block it assumes is there.
    Why required: the small synthetic models print no warnings.
    What kills it: a reader that requires leading warning lines.
    Second correct implementation: a whole-file regular expression.
    """
    assert read_rpt_starting_date(_report(tmp_path, 0)) == "08/31/2025 00:00:00"


def test_a_report_without_the_line_returns_none(tmp_path):
    """Property: a report that names no Starting Date returns None and does not raise.

    Class: GUARD against over-correction.
    Plausible wrong implementation: returning an empty string or raising on a miss.
    Why required: the caller raises a ProcessingError naming the file; it needs a clean None to do so.
    What kills it: any value other than None for a report without the line.
    Second correct implementation: raising a typed error, with the caller changed to match.
    """
    path = tmp_path / "hydraulics.rpt"
    path.write_text(
        "  EPA STORM WATER MANAGEMENT MODEL\n  Ending Date .............. 09/03/2025 00:00:00\n", encoding="latin-1"
    )
    assert read_rpt_starting_date(path) is None
