"""Figure-card names name an event the way the page and the sidebar do.

A per-event figure's card name is built from its plot ID, whose event segment is the
event's directory slug. The page and the sidebar name an event as
`{label} (event {number})`, or `event {number}` when no label is configured, and
never by the slug. Card names are built from whatever event map their caller
passes; this test drives the combined report's harvest, which builds that map from
each child bundle's `scenario_status.csv`, and reads the `figure` label it emits.

It writes a minimal child bundle in a temporary directory. It compiles no solver
and runs no simulation.
"""

import json

import pytest

from hhemt.bundle.combined_snakefile_generator import _harvest_per_experiment_rule_specs

_SLUG = "year.2_event_id.1"
_PLOT_ID = f"peak_flood_depth__evt.{_SLUG}"


def _child(bundle_root, status_header, status_row):
    """A child bundle carrying one per-event figure and a scenario_status.csv."""
    child = bundle_root / "child_crates" / "exp_tritonswmm"
    (child / "plots" / "maps").mkdir(parents=True)
    (child / "cfg_analysis.yaml").write_text("{}\n")
    (child / "plots" / "maps" / f"{_PLOT_ID}.html").write_text("<html><body>fig</body></html>")
    (child / "plots" / "maps" / f"{_PLOT_ID}.manifest.json").write_text(
        json.dumps({"plot_id": _PLOT_ID, "source_paths_relative": ["data/x.zarr"]})
    )
    (child / "scenario_status.csv").write_text(f"{status_header}\n{status_row}\n")


@pytest.mark.parametrize(
    ("header", "row", "expected"),
    [
        ("scenario_directory,event_iloc,event_label", f"/runs/sims/{_SLUG},3,Isabel", "Isabel (event 3)"),
        ("scenario_directory,event_iloc", f"/runs/sims/{_SLUG},3", "event 3"),
    ],
    ids=["labelled", "unlabelled"],
)
def test_card_name_names_the_event_by_label_and_number(tmp_path, header, row, expected):
    """Property: the card name of a per-event figure carries the event as
    `{label} (event {number})`, or `event {number}` when unlabelled, never the slug.

    Class: both label states of an event whose slug resolves in scenario_status.csv.

    Why required: the card, the page and the sidebar name one event; a card naming it
    by the slug while the page names it by number reads as two different events.

    What kills it: passing the label-only map to the card-name builder, which names an
    unlabelled event by its slug and drops the number from a labelled one.

    A second correct implementation that still passes: composing the event text inside
    the card-name builder from a slug-to-number map.
    """
    root = tmp_path / "combined"
    _child(root, header, row)
    specs = [s for s in _harvest_per_experiment_rule_specs(root) if "peak_flood_depth" in s.rule_name]
    assert len(specs) == 1, [s.rule_name for s in specs]
    figure = json.loads(specs[0].report_kwargs["labels"])["figure"]
    assert figure.endswith(expected), figure
    assert _SLUG not in figure, figure
