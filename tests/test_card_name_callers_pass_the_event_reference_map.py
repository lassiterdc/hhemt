"""Every report path that names figure cards passes the event map that carries the event number.

A per-event figure's card name is built from its plot ID by `humanize_plot_id`, from an
event map its caller supplies. Two maps exist: one gives each event slug the page's own
event reference, `{label} (event {number})` or `event {number}`; the other gives only the
configured label, so an event with no label falls back to its directory slug and a
labelled event loses its number. Five functions build a card-name map, one for each
way a report is rendered, and only one of them can be driven without Snakemake. This
module reads all five from source and checks that none of them builds the label-only map.

It parses source files only. It compiles no solver and runs no simulation.
"""

import ast
from pathlib import Path

import pytest

import hhemt

_SRC = Path(hhemt.__file__).parent

# The five functions that build the event map handed to card-name rendering: the
# analysis, sensitivity and bundle-regeneration renders (through post-render surgery),
# and the combined report's page harvest and its own render.
_CARD_NAME_CALLERS = (
    ("analysis.py", "TRITONSWMM_analysis.render_report"),
    ("sensitivity_analysis.py", "TRITONSWMM_sensitivity_analysis.render_report"),
    ("bundle/__init__.py", "Bundle.regenerate_report"),
    ("bundle/combined_snakefile_generator.py", "_harvest_per_experiment_rule_specs"),
    ("bundle/combined_snakefile_generator.py", "render_combined_report_via_snakemake"),
)
_LABEL_ONLY_MAP = "event_labels_from_status"


def _function(rel_path, qualname):
    """Return the function definition at `qualname` in the module at `rel_path`, or None."""
    node = ast.parse((_SRC / rel_path).read_text())
    for part in qualname.split("."):
        node = next(
            (
                ch
                for ch in ast.iter_child_nodes(node)
                if isinstance(ch, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and ch.name == part
            ),
            None,
        )
        if node is None:
            return None
    return node


def _called_names(func):
    """Every bare or attribute name called anywhere inside `func`."""
    return {
        getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        for call in ast.walk(func)
        if isinstance(call, ast.Call)
    }


def _feeds_card_names(func):
    """Whether `func` hands an event map to card-name rendering, directly or through surgery.

    True when it calls `humanize_plot_id` or passes an `event_labels=` keyword to any
    call, the two ways an event map reaches a card name. This guards the closed list:
    a listed function that stops feeding card names makes the check below vacuous.
    """
    for call in ast.walk(func):
        if not isinstance(call, ast.Call):
            continue
        if (getattr(call.func, "id", None) or getattr(call.func, "attr", None)) == "humanize_plot_id":
            return True
        if any(kw.arg == "event_labels" for kw in call.keywords):
            return True
    return False


@pytest.mark.parametrize(("rel_path", "qualname"), _CARD_NAME_CALLERS, ids=[q for _, q in _CARD_NAME_CALLERS])
def test_card_name_caller_builds_no_label_only_event_map(rel_path, qualname):
    """Property: none of the five card-name callers builds the label-only event map.

    Class: the five functions that build the event map card names are rendered from,
    one per report path. This is a fitness function over a closed list: a sixth caller
    added later is not covered until it is listed here.

    Why required: a caller passing the label-only map names an unlabelled event by its
    slug and drops the number from a labelled one, so that report path's cards name an
    event differently from its page and sidebar. Four of the five callers run only
    after `snakemake --report`, so no behavioural test reaches them.

    What kills it: any of the five calling `event_labels_from_status`, for example one
    render path reverted to the label-only map.

    A second correct implementation that still passes: callers passing a slug-to-number
    map and `humanize_plot_id` composing the event reference itself, which calls neither
    builder named here at those sites.
    """
    func = _function(rel_path, qualname)
    assert func is not None, f"{qualname} not found in {rel_path}; the caller list is stale"
    names = _called_names(func)
    assert _feeds_card_names(func), f"{qualname} no longer feeds card names; the caller list is stale"
    assert _LABEL_ONLY_MAP not in names, f"{qualname} builds the label-only event map"
