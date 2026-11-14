"""The report page states what `report_html_mode` does, and the code does that.

`docs/reference/example-report.md` documents the container choice: the three modes, the
default, the 15 MB threshold, the byte count and the file a workflow writes under `auto`.
The test reads each of those from the code and compares it with the page by an anchor (a
code span or the sentence that carries the number), so a change to the threshold, the
default, the mode list or the marker name turns it red. It runs no solver.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

from hhemt.config.report import (
    AUTO_HTML_MAX_BYTES,
    REPORT_FORMAT_MARKER,
    InteractiveBackendConfig,
)

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "example-report.md"


def _code_facts():
    field = InteractiveBackendConfig.model_fields["report_html_mode"]
    return {
        "modes": set(typing.get_args(field.annotation)),
        "default": field.default,
        "threshold": AUTO_HTML_MAX_BYTES,
        "marker": REPORT_FORMAT_MARKER,
    }


def check_page(text: str, facts: dict) -> list[str]:
    """Problems between the page text and the code facts, by anchor."""
    problems = []
    spans = set(re.findall(r"`([^`\n]+)`", text))
    flat = " ".join(text.split())
    setting = re.search(r"follows the `report\.interactive\.report_html_mode` setting", flat)
    if not setting:
        problems.append("the page does not name the report_html_mode setting")
    named = {m for m in facts["modes"] if m in spans}
    if named != facts["modes"]:
        problems.append(f"the page names modes {sorted(named)}, the code has {sorted(facts['modes'])}")
    m = re.search(r"defaults to `(\w+)`", flat)
    if m is None or m.group(1) != facts["default"]:
        problems.append("the default mode on the page differs from the code")
    m = re.search(r"below (\d+) MB, or ([\d,]+) bytes", flat)
    if m is None:
        problems.append("the page states no threshold in MB and bytes")
    else:
        stated_bytes = int(m.group(2).replace(",", ""))
        if stated_bytes != facts["threshold"] or int(m.group(1)) * 10**6 != facts["threshold"]:
            problems.append("the threshold on the page differs from the code")
    if facts["marker"] not in spans:
        problems.append("the page does not name the marker file the workflow writes")
    return problems


def test_page_matches_the_code():
    assert check_page(_PAGE.read_text(encoding="utf-8"), _code_facts()) == []


def test_control_a_changed_threshold_or_mode_or_marker_is_refused():
    facts = _code_facts()
    text = _PAGE.read_text(encoding="utf-8")
    assert check_page(text, {**facts, "threshold": facts["threshold"] + 1})
    assert check_page(text, {**facts, "modes": facts["modes"] | {"png"}})
    assert check_page(text, {**facts, "default": "zip"})
    assert check_page(text, {**facts, "marker": "analysis_report.fmt"})


def test_control_a_page_with_the_old_wording_is_refused():
    old = 'The container is chosen by the `format` argument, which defaults to `"zip"`.'
    assert check_page(old, _code_facts())
