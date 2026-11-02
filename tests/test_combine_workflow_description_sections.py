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

from hhemt.bundle.combined_snakefile_generator import _combined_workflow_description
from hhemt.report_renderers._reporting_sets import get_reporting_set


def test_combined_description_names_exactly_the_rendered_combined_sections():
    """Property: for every combination of the two conditions the description depends on, every
    section it names in bold is a section of the combined reporting set and is rendered, and
    every rendered section is named.

    Class: the bold (`**...**`) spans of `_combined_workflow_description(rendered, ...)`, over
    the whole `category_order` as the registry holds it when the test runs, with the
    cross-solver section rendered or not and the clean-and-resume pair present or not (four
    combinations; the full set with a pair is the old static description).

    Why required: the description is the landing page's only guide to the sidebar, and a bold
    name with no matching section sends the reader to look for a section that does not exist.

    What kills it: a description naming a section the report does not render, or a rendered
    section the description leaves out.
    """
    sections = list(get_reporting_set("combined").category_order)
    assert sections, "the combined set declares no sections, so the check would be vacuous"
    for solver in (True, False):
        for pair in (True, False):
            rendered = [s for s in sections if solver or s != "Cross-Solver Comparison"]
            text = _combined_workflow_description(rendered, clean_vs_resume=pair)
            named = re.findall(r"\*\*(.+?)\*\*", text)
            assert named, "the description names no section in bold, so the check would be vacuous"
            assert named == rendered, (solver, pair)
            assert ("clean" in text) == pair, (solver, pair)
            assert ("Cross-Solver Comparison" in text) == solver, (solver, pair)
