"""The combined report's landing page names only sidebar sections the report has.

A combined cross-experiment report opens on a static description that tells the
reader which sidebar sections to browse. The sections themselves come from the
`combined` reporting set's `category_order`, so a description written as prose
drifts the moment a section is renamed, and a reader is then sent to a section
that is not there. This test reads the section names from the registry at test
time and checks the description against them in both directions.

It reads two module-level values only. It compiles no solver and runs no simulation.
"""

import re

from hhemt.bundle.combined_snakefile_generator import _COMBINED_WORKFLOW_DESCRIPTION
from hhemt.report_renderers._reporting_sets import get_reporting_set


def test_combined_description_names_exactly_the_combined_sidebar_sections():
    """Property: every section the description names in bold is a section of the
    combined reporting set, and every such section is named.

    Class: every bold (`**...**`) span in the description, against the set's whole
    `category_order` as the registry holds it when the test runs.

    Why required: the description is the landing page's only guide to the sidebar,
    and a bold name with no matching section sends the reader to look for a section
    that does not exist.

    What kills it: a description naming a section the registry does not carry, or
    a registry section renamed or added without the description following.

    A second correct implementation that still passes: a description generated
    from `category_order` at import time, which names every section by construction.
    """
    sections = list(get_reporting_set("combined").category_order)
    assert sections, "the combined set declares no sections, so the check would be vacuous"
    named = re.findall(r"\*\*(.+?)\*\*", _COMBINED_WORKFLOW_DESCRIPTION)
    assert named, "the description names no section in bold, so the check would be vacuous"
    assert [n for n in named if n not in sections] == []
    assert [s for s in sections if s not in named] == []
