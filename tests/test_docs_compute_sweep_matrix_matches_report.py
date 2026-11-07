"""The compute-sweep tutorial's embedded matrix against the report it describes.

The tutorial `docs/tutorials/norfolk-compute-config-sweep.md` embeds the sweep's
member table as a CSV block and then embeds the rendered report. Nothing tied the
two together, so a re-render that changed the set of members left the tutorial
describing a different sweep while every docs gate stayed green. These tests
compare the set of `member_id` values in the tutorial's first CSV block with the
set of `member-{id}` labels in the landed report, and write down no count: a count
in the test would go stale in the same way the prose count did.

The report is about 98 MiB, so it is scanned through `mmap` rather than read into
a string. A bare `member-` token (no digits) also occurs in it and is excluded by
requiring digits.

The comparison is one callable, `assert_matrix_matches_report`, that the primary
test and both controls all call. Each control feeds it a mutated input (a tutorial
with one row removed, or a small report file with one label removed) and expects
the refusal, so a comparison weakened to a one-directional subset check, or
removed, turns a control red. No test here runs a solver.
"""

from __future__ import annotations

import mmap
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parent.parent
_TUTORIAL = _REPO / "docs" / "tutorials" / "norfolk-compute-config-sweep.md"
_REPORT = _REPO / "docs" / "reference" / "bundles" / "norfolk-compute-config-sweep-report.html"

_FENCE = re.compile(r"^```text\n(member_id,.*?)^```", re.S | re.M)
_LABEL = re.compile(rb"member-(\d+)")


def _matrix_rows(tutorial_text: str) -> list[str]:
    """Data rows of the tutorial's first fenced block whose first line is `member_id,`."""
    m = _FENCE.search(tutorial_text)
    assert m is not None, "tutorial has no fenced text block starting with `member_id,`"
    lines = m.group(1).splitlines()
    return [ln for ln in lines[1:] if ln.strip()]


def _tutorial_ids(tutorial_text: str) -> set[int]:
    """The `member_id` column of the tutorial's matrix, as ints."""
    return {int(row.split(",", 1)[0]) for row in _matrix_rows(tutorial_text)}


def _report_ids(report_path: Path = _REPORT) -> set[int]:
    """Every distinct integer N for which the report file carries a `member-N` token."""
    with report_path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return {int(g) for g in _LABEL.findall(mm)}


def assert_matrix_matches_report(tutorial_text: str, report_path: Path = _REPORT) -> None:
    """Refuse (AssertionError) unless the tutorial's member ids equal the report's labels.

    Equality, not containment in either direction, because a tutorial that lists
    a member the report lacks and a report that carries a member the tutorial
    omits are both a matrix that does not describe its report.
    """
    tutorial = _tutorial_ids(tutorial_text)
    report = _report_ids(report_path)
    assert tutorial, "the tutorial matrix parsed to no members"
    assert tutorial == report, (
        f"only in tutorial: {sorted(tutorial - report)}; only in report: {sorted(report - tutorial)}"
    )


def test_tutorial_matrix_ids_equal_report_member_labels():
    """Property: the real tutorial matrix and the real report name the same members.

    Class: any pair of (tutorial, report) in which either side has a member the
    other lacks. Why required: a reader takes the matrix as the report's contents.
    Killed by: a tutorial row for a member the report lacks, or a report member
    absent from the matrix. A second correct implementation reads the report with
    `read_bytes` instead of `mmap` and still passes. This test alone does not
    kill a weakened comparison on today's data, where the sets are equal; the two
    controls below do.
    """
    assert_matrix_matches_report(_TUTORIAL.read_text())


@pytest.mark.parametrize("which", ["first", "last"])
def test_comparison_refuses_a_tutorial_with_one_row_removed(which):
    """Control: the comparison refuses a tutorial text missing one matrix row.

    Class: a tutorial that is a strict subset of the report, the row removed at
    either end of the block so a parser that misreads only the first or last row
    is not hidden. Killed by: a comparison weakened to `tutorial <= report` or
    removed, either of which accepts a subset. A second correct implementation is
    any exact set comparison, whatever it reads the report with.
    """
    text = _TUTORIAL.read_text()
    rows = _matrix_rows(text)
    victim = rows[0] if which == "first" else rows[-1]
    assert victim + "\n" in text
    with pytest.raises(AssertionError):
        assert_matrix_matches_report(text.replace(victim + "\n", "", 1))


def test_comparison_refuses_a_report_file_with_one_label_removed(tmp_path):
    """Control: the comparison refuses a report file missing one `member-N` label.

    The file is small and carries the real report's labels minus the smallest, so
    `_report_ids` reads a real file through its `mmap` path. Class: a report that
    is a strict subset of the tutorial matrix. Killed by: a comparison weakened to
    `tutorial >= report` or removed, either of which accepts a report subset. A
    second correct implementation is any exact set comparison.
    """
    ids = sorted(_report_ids())
    small = tmp_path / "report_minus_one.html"
    small.write_text("".join(f"<td>member-{i}</td>\n" for i in ids[1:]))
    with pytest.raises(AssertionError):
        assert_matrix_matches_report(_TUTORIAL.read_text(), small)
