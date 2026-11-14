"""Reviewer test of the Reference pages: the flow units that the coupled-inflow-nodes page names.

The page says which SWMM flow units the toolkit converts and which it refuses. The code decides the set, so the
page names every unit the code knows.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAGE = _REPO / "docs" / "reference" / "coupled-inflow-nodes.md"
_HEADING = "## Rims and flow units"


def section_of(text: str, heading: str) -> str:
    """The text from the line that is exactly `heading` to the next heading of the same or higher level."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(rf"#{{1,{level}}}\s", lines[i])),
        len(lines),
    )
    return "\n".join(lines[start:end])


def unit_spans(text: str) -> set[str]:
    """The code spans of `text` that are upper-case words of three letters."""
    return {s for s in re.findall(r"`([A-Z]{3})`", text)}


def missing_units(section: str, units: set[str]) -> list[str]:
    """The units of `units` that no code span of `section` names."""
    return sorted(units - unit_spans(section))


def test_the_page_names_every_flow_unit_the_code_knows():
    """Property: the section on flow units names, as code spans, every unit of the toolkit's conversion table.

    Class: the six SWMM flow unit words of `_FLOW_UNIT_TO_CMS` (three SI units that the toolkit converts or reads,
    three US units that scenario preparation refuses for the hydrology model).
    Why required: a reader who sets a flow unit in the hydrology model learns from this page which units are
    accepted and which are refused, and the page lists the accepted units already; the refused units are the half a
    reader needs before submitting.
    Kills it: a page that names `LPS` and `MLD` only (the staged text), or that drops one of the three US units.
    A second correct implementation under which it still passes: any wording that keeps the six code spans in the
    section, in a table or in a sentence.
    """
    from hhemt.swmm_runoff_modeling import _FLOW_UNIT_TO_CMS

    section = section_of(_PAGE.read_text(encoding="utf-8"), _HEADING)
    assert missing_units(section, set(_FLOW_UNIT_TO_CMS)) == []


def test_control_a_section_without_a_unit_is_refused():
    """Control: the finder reports a unit that the section lacks, and nothing when every unit is named."""
    section = "## Rims and flow units\n\nThe units `CMS`, `LPS`, `MLD`, `CFS`, and `MGD` are named.\n"
    assert missing_units(section, {"CMS", "LPS", "MLD", "CFS", "GPM", "MGD"}) == ["GPM"]
    assert missing_units(section.replace("`MGD`", "`MGD` and `GPM`"), {"CMS", "LPS", "MLD", "CFS", "GPM", "MGD"}) == []
