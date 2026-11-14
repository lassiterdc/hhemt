"""The report sidebar names an event the way the per-simulation page does.

Per-simulation figures carry an `event` label in the report's sidebar. The page
itself names an event as `{label} (event {iloc})`, or `event {iloc}` when no label
is configured, and deliberately never by its directory slug, which is unspeakable.
The sidebar used to fall back to `event {slug}` whenever no label was configured,
which is the default, so the two surfaces named one event two ways.

These tests evaluate the sidebar label exactly as a generated Snakefile does: they
execute the label preamble every generator emits, with `workflow.basedir` pointed
at a directory holding a `scenario_status.csv`, then call each registry label
lambda on a wildcard object. They compile no solver and run no simulation.
"""

import types

import pytest

from hhemt.report_plot_ids import LABEL_GLOBALS_BLOCK
from hhemt.report_renderers._reporting_sets import REPORTING_SETS

_SLUG = "year.2_event_id.1"


def _event_label_lambdas():
    """Every distinct registry `labels=` expression that carries an `event` entry."""
    seen = {}
    for rset in REPORTING_SETS.values():
        for sel in rset.renderer_selection:
            for tmpl in sel.rule_spec_template:
                labels = (tmpl.report_kwargs or {}).get("labels") or ""
                if '"event":' in labels:
                    seen.setdefault(tmpl.rule_name, labels)
    return sorted(seen.items())


def _sidebar_event_values(basedir, slug):
    """The `event` sidebar value each label lambda yields for `slug`, keyed by rule name."""
    namespace = {"workflow": types.SimpleNamespace(basedir=str(basedir))}
    exec(LABEL_GLOBALS_BLOCK, namespace)  # noqa: S102 -- the preamble a generated Snakefile runs
    wildcards = types.SimpleNamespace(event_id=slug, member_id="m1")
    return {name: eval(expr, namespace)(wildcards)["event"] for name, expr in _event_label_lambdas()}  # noqa: S307


def _write_status(directory, header, row):
    (directory / "scenario_status.csv").write_text(header + "\n" + row + "\n")


def test_every_event_sidebar_call_site_is_reached():
    """Property: the label lambdas this module exercises are the three per-simulation
    figures the sidebar labels by event.

    Class: the registry's `labels=` expressions that carry an `event` entry.

    Why required: the two behaviour tests below would pass vacuously over an empty
    set if the registry stopped carrying event labels.

    What kills it: a registry change that drops or renames one of the three rules.

    A second correct implementation that still passes: a registry carrying the same
    three rules in a different order or set.
    """
    assert [name for name, _ in _event_label_lambdas()] == [
        "plot_per_sim_event_page",
        "plot_per_sim_per_member_conduit_flow",
        "plot_per_sim_per_member_peak_flood_depth",
    ]


@pytest.mark.parametrize(
    ("header", "row", "expected"),
    [
        (
            "scenario_directory,event_iloc,event_label",
            f"/runs/sims/{_SLUG},3,Isabel",
            "Isabel (event 3)",
        ),
        (
            "scenario_directory,event_iloc",
            f"/runs/sims/{_SLUG},3",
            "event 3",
        ),
    ],
    ids=["labelled", "unlabelled"],
)
def test_sidebar_event_label_is_name_first_with_the_iloc(tmp_path, header, row, expected):
    """Property: when the slug resolves to an `event_iloc` in scenario_status.csv, the
    sidebar's event value is `{label} (event {iloc})`, or `event {iloc}` when no label
    is set, and never carries the slug.

    Class: both label states of a resolvable slug: a configured label, and no label
    column at all, which is every default-configured analysis.

    Why required: the per-simulation page names an event in exactly this form, and a
    sidebar naming the same event by its slug contradicts it.

    What kills it: falling back to `event {slug}` when no label is set (reds the
    unlabelled case), or dropping the iloc from a labelled value (reds the labelled
    case). Both cases assert the integer specifically, so a value of the form
    `event {token}` built from the slug cannot pass.

    A second correct implementation that still passes: resolving the iloc inside the
    label lambda from a slug-to-iloc map instead of a precomposed one.
    """
    _write_status(tmp_path, header, row)
    values = _sidebar_event_values(tmp_path, _SLUG)
    assert values, "no event label lambdas were evaluated"
    for name, value in values.items():
        assert value == expected, (name, value)
        assert _SLUG not in value, (name, value)


def test_unresolvable_slug_keeps_the_characterized_fallback(tmp_path):
    """Property: a slug absent from scenario_status.csv renders `event {slug}`, the
    never-raise fallback, rather than failing the render.

    Class: the one state in which the iloc is unavailable: a CSV that does not carry
    the slug (a missing CSV is the same state).

    Why required: the label reader must never fail a report, so an unresolvable event
    still gets a readable, if raw, sidebar entry.

    What kills it: raising, or rendering an empty value, for an unknown slug.

    A second correct implementation that still passes: any fallback text naming the
    slug under the `event` prefix.
    """
    _write_status(tmp_path, "scenario_directory,event_iloc", "/runs/sims/other_slug,0")
    for name, value in _sidebar_event_values(tmp_path, _SLUG).items():
        assert value == f"event {_SLUG}", (name, value)
