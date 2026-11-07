"""Reviewer tests of chunk 10b.2: the scrub makes every path segment that follows a path component named home, Users
or scratch the generic word, including a segment that ends its path, prints no replaced value, and the committed
figure is the figure of the committed report.

Every private string is built from fragments, so this file holds no real value.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "prepare_example_report_rv", _REPO / "scripts" / "prepare_example_report.py"
)
per = importlib.util.module_from_spec(_spec)
sys.modules["prepare_example_report_rv"] = per
_spec.loader.exec_module(per)

_HOME = "/" + "ho" + "me" + "/"
_SEG = "plan" + "ted_seg"
_PRIVATE = "estate_" + "checkout_x"
_OBSERVED = "docs/reference/bundles/norfolk-observed-events-report.html"
_PNG = "docs/tutorials/assets/observed-events-irene-result.png"
_IRENE = "year.9_event_type.compound_event_id.1"

# Each text ends the path at the segment, or follows it with a character that is not a slash.
_ENDING_TEXTS = [
    f"cwd {_HOME}{_SEG}",
    f"HOME={_HOME}{_SEG}\nnext line",
    f'path "{_HOME}{_SEG}" here',
    f"WorkDir={_HOME}{_SEG} Partition=x",
]


@pytest.mark.parametrize("text", _ENDING_TEXTS)
def test_rule1_counts_a_user_segment_that_ends_its_path(text):
    """Class: every path whose user segment is not followed by a slash (end of line, space, quote).

    The rule is that every path segment that follows a path component named home, Users or scratch is the generic
    word. A matcher that requires a trailing slash never sees `HOME=/home/{user}` printed at the end of a line.
    REGRESSION: red on the tree. Reachability: environment and scheduler dumps print such paths. Both-states anchor:
    the assertion turns on the segment position, not on wording. Wrong implementation: the current matcher. A second
    correct implementation: a matcher that treats a trailing slash as optional. The change that greens it: drop the
    trailing-slash group of `_USER_SEGMENT` (script line 48) and return group 1 plus the generic word (line 108).
    """
    assert per.count_user_segments(text) == 1


def test_rule1_ignores_the_generic_word_and_a_root_with_no_segment():
    """Control: the generic word and a bare root are not counted, at the end of a path or inside one."""
    for text in (
        f"cwd {_HOME}operator",
        f"cwd {_HOME}operator/work",
        f"a {_HOME} b",
        f"a {_HOME}\nb",
    ):
        assert per.count_user_segments(text) == 0


def test_scrub_replaces_a_segment_that_ends_its_path():
    """Class: the scrub's rule 1 over the same ending-position texts, in the outer HTML.

    REGRESSION: red on the tree (the segment survives and rule 1's own after-count is also 0, so the scrub exits 0
    and writes a file that names the segment). Change that greens it: as in the test above.
    """
    outer = "\n".join(
        ["<html>", *_ENDING_TEXTS[:1], '    var results = {"p": {"data_uri": "data:text/html;base64,"}};', "</html>"]
    )
    result, _rules = per.scrub_text(outer, [])
    assert _SEG not in result


def _leaks(output: str, values: list[str]) -> list[str]:
    return [v for v in values if v in output]


def test_scrub_prints_no_replaced_value(tmp_path, capsys):
    """Class: everything `scrub` prints to stdout and stderr, for a source with a planted segment and a rule 2 value.

    NEW CAPABILITY (the command prints a count per rule and never a replaced value). Plausible wrong
    implementation: a rule line that prints OLD (`rule 2 (replace {old}): n replaced`), or a refusal message that
    quotes the segment. Red under it: the control below feeds the checker such a line and it reports the value.
    """
    import json
    from base64 import b64encode

    page = f"<p>{_HOME}{_SEG}/w and {_PRIVATE}</p>"
    entry = {
        "name": "n",
        "labels": {"figure": "f"},
        "data_uri": "data:text/html;charset=utf8;filename=p.html;base64," + b64encode(page.encode()).decode(),
    }
    source = tmp_path / "in.html"
    source.write_text(
        "<html>\n    var results = " + json.dumps({"plots/p.html": entry}) + ";\n</html>", encoding="utf-8"
    )
    assert per.main(["scrub", str(source), str(tmp_path / "out.html"), "--replace", f"{_PRIVATE}=estate"]) == 0
    captured = capsys.readouterr()
    assert _leaks(captured.out + captured.err, [_SEG, _PRIVATE]) == []


def test_the_leak_check_reports_a_printed_value():
    """Control: a rule line that prints the replaced value is reported, a line that prints a count is not."""
    assert _leaks(f"rule 2 (replace {_PRIVATE}): 2 replaced", [_SEG, _PRIVATE]) == [_PRIVATE]
    assert _leaks("rule 2 (replace pair 0): 2 replaced", [_SEG, _PRIVATE]) == []


_PINNED = {"plotly": "5.24.1", "kaleido": "0.2.1"}


def _toolchain_matches() -> bool:
    from importlib import metadata

    try:
        return all(metadata.version(name) == version for name, version in _PINNED.items())
    except metadata.PackageNotFoundError:
        return False


def _render(title: str, dest: Path) -> bytes:
    out = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "prepare_example_report.py"),
            "figure",
            str(_REPO / _OBSERVED),
            "--event",
            _IRENE,
            "--title",
            title,
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return dest.read_bytes()


@pytest.mark.skipif(
    not _toolchain_matches(), reason="the figure bytes are fixed only under plotly 5.24.1 and kaleido 0.2.1"
)
def test_the_committed_figure_is_the_figure_of_the_committed_report(tmp_path):
    """Class: the committed PNG against a fresh `figure` render of the committed report, on the pinned toolchain.

    NEW CAPABILITY. The existing header test reads 2800 by 1200, and the figure committed before this chunk (which
    carried `Northing (ft)`) has the same header, so a stale or wrong-figure PNG passes it. Plausible wrong
    implementation: the previous PNG, or figure 0 of the page, committed at the same size. Red under it: the control
    asserts the figure under another title of the same page renders to different bytes. A second correct implementation
    under another renderer is skipped, not failed.
    """
    committed = (_REPO / _PNG).read_bytes()
    assert _render("TRITON-SWMM: Peak flood depth", tmp_path / "a.png") == committed
    assert _render("TRITON: Peak flood depth", tmp_path / "b.png") != committed
