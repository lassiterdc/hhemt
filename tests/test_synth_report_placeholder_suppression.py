"""No rendered report carries the empty "Simulation Health (placeholder)" sidebar category.

Snakemake's report page builds its sidebar from the keys of one JavaScript global,
`var categories = {...};`, and post-render surgery used to inject an empty reserved
category into it. That reserved slot was suppressed only for reports regenerated
from a bundle; a report rendered directly from an analysis still showed an empty
category a reader could click into and find nothing. These tests surger a report
fragment through both surgery entry points (the HTML file and the zipped report),
with and without bundle mode, and read the sidebar back from the categories global.

These tests touch only report post-processing. They compile no solver and run no
simulation.
"""

import json
import re
import zipfile

import pytest

from hhemt.report_renderers._react_surgery import (
    apply_post_process_surgery,
    apply_post_process_surgery_to_zip,
)

# The categories global as Snakemake's report template emits it: one JSON object,
# keyed by sidebar category, each mapping subcategory names to result paths.
_CATEGORIES = {
    "Workflow Status": {"Other": ["plots/workflow_status.html"]},
    "Key Results": {"Other": ["plots/key_results.html"]},
}
_REPORT_HTML = (
    "<html><body>\n"
    '<script id="categories">\n'
    f"    var categories = {json.dumps(_CATEGORIES)};\n"
    "</script>\n"
    "<script>x.sort((a, b) => a.localeCompare(b));</script>\n"
    "</body></html>\n"
)
_CATEGORIES_RE = re.compile(r"var categories = (\{.*?\});\n", re.S)


def _sidebar_categories(html_text):
    """Return the sidebar category names a surgered report will show.

    Reads the keys of the `var categories` global, which is the only input the
    report's menu enumerates. A text search over the whole page would also match
    the category-sort comparator, which ranks names but creates no category.
    """
    match = _CATEGORIES_RE.search(html_text)
    assert match is not None, "the fixture's categories global did not survive surgery"
    return list(json.loads(match.group(1)))


def _surger_html(html_text, bundle_mode):
    return apply_post_process_surgery(html_text, bundle_mode=bundle_mode)


def _surger_zip(html_text, bundle_mode, tmp_path):
    zip_path = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("analysis_report/report.html", html_text)
    apply_post_process_surgery_to_zip(zip_path, bundle_mode=bundle_mode)
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read("analysis_report/report.html").decode()


@pytest.mark.parametrize("bundle_mode", [False, True], ids=["analysis-render", "bundle-render"])
@pytest.mark.parametrize("container", ["html", "zip"])
def test_rendered_sidebar_carries_no_placeholder_category(container, bundle_mode, tmp_path):
    """Property: after surgery, the sidebar shows exactly the categories the report's
    figures declared, and no placeholder category.

    Class: every render path, as the cross product of the two surgery entry points
    (the HTML file and the zipped report) and the two report origins (an analysis
    render, bundle mode off; a bundle regeneration or combine, bundle mode on). Every
    report-rendering facade reaches surgery through one of these two functions with
    one of these two flags.

    Why required: an injected category with no figures renders as a sidebar entry
    that opens an empty page, so a reader sees a section the toolkit never fills.

    What kills it: injecting any key into the categories global on any path, for
    example restoring the reserved-slot injection for bundle_mode=False, which reds
    both analysis-render cases.

    A second correct implementation that still passes: injecting the slot and then
    removing every category whose subcategories are all empty before the page is
    written.
    """
    if container == "html":
        out = _surger_html(_REPORT_HTML, bundle_mode)
    else:
        out = _surger_zip(_REPORT_HTML, bundle_mode, tmp_path)

    categories = _sidebar_categories(out)
    assert [c for c in categories if "placeholder" in c.lower()] == []
    assert categories == list(_CATEGORIES)


def test_surgery_is_still_idempotent_on_the_categories_global():
    """Property: surgering an already-surgered report leaves its sidebar unchanged.

    Class: a second surgery pass over the output of a first, the case a report
    re-rendered in place reaches.

    Why required: the surgery entry points promise idempotence, and a pass that
    added or dropped a category on its second run would make the sidebar depend on
    how many times a report had been rendered.

    What kills it: an injection guarded by nothing, which adds a key on every pass.

    A second correct implementation that still passes: rebuilding the categories
    global from parsed JSON on every pass.
    """
    once = apply_post_process_surgery(_REPORT_HTML)
    twice = apply_post_process_surgery(once)
    assert _sidebar_categories(twice) == _sidebar_categories(once) == list(_CATEGORIES)
