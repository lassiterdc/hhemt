"""No render path's sidebar order ranks the retired "Simulation Health (placeholder)" category.

A rendered report's sidebar is sorted by a comparator that post-render surgery writes
into the page as a JavaScript literal, `const ORDER = {...};`, built from a category
order. That order comes from one of two places: the active reporting set's
`category_order` (an analysis or sensitivity render, and the combined report), or the
surgery module's own default when the caller passes none (a bundle regeneration). The
placeholder category is no longer injected on any path, so a name for it in any of those
orders is dead vocabulary that still reaches every rendered page through the comparator,
where a whole-page search for the name finds it on a correct report.

The tests surger a minimal report page once per render-path order and read back the
names the comparator ranks. They compile no solver and run no simulation.
"""

import json
import re

import pytest

from hhemt.report_renderers._react_surgery import apply_post_process_surgery
from hhemt.report_renderers._reporting_sets import REPORTING_SETS

# The two page fragments surgery reads here: the categories global the sidebar is built
# from, and the sort call whose `localeCompare` comparator step 4 replaces.
_REPORT_HTML = (
    "<html><body>\n"
    '<script id="categories">\n'
    '    var categories = {"Key Results": {"Other": ["plots/key_results.html"]}};\n'
    "</script>\n"
    "<script>x.sort((a, b) => a.localeCompare(b));</script>\n"
    "</body></html>\n"
)
_ORDER_RE = re.compile(r"const ORDER = (\{.*?\});")
_CATEGORIES_RE = re.compile(r"var categories = (\{.*?\});\n", re.S)


def _render_path_orders():
    """Every category order a report-rendering caller hands to surgery.

    `None` is what a bundle regeneration passes, which selects the surgery module's
    default. Every registered reporting set's `category_order` is what an analysis,
    sensitivity or combined render passes. The combined render appends experiment ids
    to its set's order, and an experiment id is not a category name the toolkit chooses,
    so the registered order is the whole of what the toolkit itself contributes.
    """
    yield pytest.param(None, id="bundle-regeneration-default")
    for name in sorted(REPORTING_SETS):
        yield pytest.param(REPORTING_SETS[name].category_order, id=f"set-{name}")


@pytest.mark.parametrize("category_order", list(_render_path_orders()))
def test_no_render_path_order_ranks_a_placeholder_category(category_order):
    """Property: the sort comparator surgery writes into a rendered page ranks no
    placeholder category, whichever render path supplied the order.

    Class: the category order of every render path -- the surgery default a bundle
    regeneration falls back to, and each registered reporting set's `category_order`,
    which the analysis, sensitivity and combined renders pass. The assertion reads the
    names in the page's `const ORDER` literal, which surgery writes on every path both
    before and after the placeholder was retired.

    Why required: once no path injects the placeholder category, a ranked name for it
    is text that ships in every report that uses the order, so a check of a rendered
    report for the placeholder by text search fails on a correct report.

    What kills it: putting the name back into any order a render path uses -- for
    example into the registry's standard order, which every one of the six sets built
    on it then carries into its renders, or into the surgery module's default.

    A second correct implementation that still passes: leaving the orders as they are
    and dropping any name containing "placeholder" while building the ORDER literal.
    """
    out = apply_post_process_surgery(_REPORT_HTML, category_order=category_order)

    match = _ORDER_RE.search(out)
    assert match is not None, "surgery wrote no ORDER comparator into the page"
    ranked = list(json.loads(match.group(1)))
    assert ranked, "the ORDER comparator ranks no category, so this case checked nothing"
    assert [name for name in ranked if "placeholder" in name.lower()] == []

    categories = json.loads(_CATEGORIES_RE.search(out).group(1))
    assert [name for name in categories if "placeholder" in name.lower()] == []
