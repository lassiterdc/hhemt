"""The committed example reports name no user segment, and the figure of the observed-events tutorial is the report's.

A report page is base64 text inside one line of the outer HTML, so a line-based reader never sees it. These tests
decode every page of every tracked `docs/reference/bundles/*.html` and hold one property: no path segment that follows
a `home`, `Users` or `scratch` component is anything but the generic word. The rule needs no private value, which a
tracked test cannot hold, so each value is counted by the coordinator at the commit of a report.

No test runs a solver or renders a report. Each page-side check is one callable that its test and its control both call.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_OBSERVED = "docs/reference/bundles/norfolk-observed-events-report.html"
_FIGURE_PNG = "docs/tutorials/assets/observed-events-irene-result.png"
_IRENE = "year.9_event_type.compound_event_id.1"
_FIGURE_TITLE = "TRITON-SWMM: Peak flood depth"

_spec = importlib.util.spec_from_file_location(
    "prepare_example_report", _REPO / "scripts" / "prepare_example_report.py"
)
per = importlib.util.module_from_spec(_spec)
sys.modules["prepare_example_report"] = per
_spec.loader.exec_module(per)

# Built from fragments, so this file holds no literal root.
_ROOT = "/" + "ho" + "me" + "/"
_PLANTED = "plan" + "ted_seg"
_PRIVATE = "estate_" + "checkout_x"


def tracked_reports() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "-z", "--", "docs/reference/bundles"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = sorted(p for p in out.split("\0") if p.endswith(".html") and (_REPO / p).exists())
    assert paths, "git listed no report; the test would pass vacuously"
    return paths


def synthetic_report(*, with_planted: bool) -> str:
    """A small report: a categories line, a manifest line with one base64 page, and a planted segment in each place."""
    seg = _PLANTED if with_planted else per.GENERIC
    page = f"<html><body>cwd {_ROOT}{seg}/work and {_PRIVATE} here</body></html>"
    entry = {
        "name": f"page {_ROOT}{seg}/x",
        "labels": {"figure": "Simulation results"},
        "data_uri": "data:text/html;charset=utf8;filename=p.html;base64," + base64.b64encode(page.encode()).decode(),
    }
    return "\n".join(
        [
            "<html>",
            f"<p>log {_ROOT}{seg}/run</p>",
            '    var categories = {"Per Simulation Results": {"Other": ["plots/p.html"]}};',
            "    var results = " + json.dumps({"plots/p.html": entry}) + ";",
            "</html>",
        ]
    )


def user_segment_problems(report: str) -> list[str]:
    outer, decoded = per.count_report(report)
    return ([f"{outer} user segments in the outer HTML"] if outer else []) + (
        [f"{decoded} user segments in the decoded pages"] if decoded else []
    )


def test_every_committed_report_names_no_user_segment():
    """Property: rule 1 counts 0 over the outer HTML and the decoded pages of every tracked report.

    Class: REGRESSION on a re-rendered report committed unscrubbed. Reachability: the reports are listed by git and
    decoded through the manifest line. A second correct implementation counts through any other decoder.
    """
    for path in tracked_reports():
        assert user_segment_problems((_REPO / path).read_text(encoding="utf-8")) == [], path


def test_the_count_fires_on_a_planted_segment_in_the_outer_html_and_in_a_page():
    """Control: a synthetic report with a planted segment is counted in both places, and the generic word is not."""
    assert user_segment_problems(synthetic_report(with_planted=True)) != []
    assert per.count_report(synthetic_report(with_planted=True)) == (2, 1)
    assert user_segment_problems(synthetic_report(with_planted=False)) == []


def test_scrub_removes_the_planted_segment_keeps_pages_decodable_and_applies_a_replace_pair(tmp_path):
    """Property: `scrub` leaves rule 1 at 0, keeps each page decodable and replaces a private value by the given word.

    Class: REGRESSION on the scrub. Wrong implementation: scrubbing the outer HTML only, or re-encoding a page that no
    longer decodes. Reachability: the planted segment sits in an outer line and inside a base64 page.
    """
    source, dest = tmp_path / "in.html", tmp_path / "out.html"
    source.write_text(synthetic_report(with_planted=True), encoding="utf-8")
    assert per.main(["scrub", str(source), str(dest), "--replace", f"{_PRIVATE}=estate"]) == 0
    out = dest.read_text(encoding="utf-8")
    assert user_segment_problems(out) == []
    pages = per.decode_pages(out)
    assert list(pages) == ["plots/p.html"]
    assert "estate here" in pages["plots/p.html"] and _PRIVATE not in pages["plots/p.html"]
    assert _PLANTED not in out
    assert "<html>" in out.splitlines()[0]


def test_scrub_refuses_a_report_it_cannot_clear(tmp_path, monkeypatch):
    """Control: when a count stays above 0 after the rules ran, `scrub` writes nothing and exits non-zero."""
    source, dest = tmp_path / "in.html", tmp_path / "out.html"
    source.write_text(synthetic_report(with_planted=True), encoding="utf-8")
    monkeypatch.setattr(per, "count_report", lambda _text: (0, 1))
    assert per.main(["scrub", str(source), str(dest)]) == 1
    assert not dest.exists()


def figure_lines(listing: str, event: str, title: str) -> list[list[str]]:
    """The fields of each `figure` line of a `figure --list` output for the event page and the heading."""
    rows = [ln.split("\t") for ln in listing.splitlines() if ln.startswith("figure\t")]
    return [r for r in rows if r[1].endswith(f"event_page__evt.{event}.html") and r[3] == title]


def test_the_irene_page_has_one_figure_with_the_title_under_per_simulation_results(capsys):
    """Property: `figure --list` finds one figure with the title on the Irene page, in its category, with a meter axis.

    Class: REGRESSION on the report or the figure script. Reachability: the committed report is read through the
    manifest and the categories line.
    """
    assert per.main(["figure", str(_REPO / _OBSERVED), "--list"]) == 0
    rows = figure_lines(capsys.readouterr().out, _IRENE, _FIGURE_TITLE)
    assert len(rows) == 1
    assert rows[0][2] == "Per Simulation Results"
    assert "Northing (m)" in rows[0][4].split("; ")


def test_the_figure_listing_check_finds_no_figure_under_another_title():
    """Control: the same listing filtered by a title that no figure carries returns no row."""
    listing = "figure\tplots/per_sim/x/event_page__evt.e.html\tPer Simulation Results\tA title\tNorthing (m)\n"
    assert figure_lines(listing, "e", "A title") != []
    assert figure_lines(listing, "e", "Another title") == []


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", data[16:24])


def test_the_committed_figure_is_2800_by_1200():
    """Property: the committed PNG header reads 2800 by 1200, the size of a 1400 by 600 figure at scale 2."""
    assert png_size((_REPO / _FIGURE_PNG).read_bytes()) == (2800, 1200)


def test_the_png_size_check_reads_another_size():
    """Control: a header of another size is read as that size, and bytes that are no PNG are refused."""
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 10, 20)
    assert png_size(header) == (10, 20)
    with pytest.raises(AssertionError):
        png_size(b"not a png at all, twenty-four bytes")
