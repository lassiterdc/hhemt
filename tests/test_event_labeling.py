"""Unit tests for the event display-label subsystem (per-sim-event-labeling).

Environment-independent by construction, in the shape tests/test_n_resumes_reporting.py
established for this same surface: they call pure functions on synthetic inputs, so no
analysis instance, config pair, or materialized directory tree is needed.
"""

from __future__ import annotations

import pandas as pd

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.report_plot_ids import (
    EVENT_LABEL_COLUMN,
    event_labels_from_status,
    event_page_reference,
)

_INDICES = ["year", "event_type", "event_id"]


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2, "event_type": "compound", "event_id": 1, "my_label": "2003-09-17 - Isabel"},
            {"year": 5, "event_type": "compound", "event_id": 1, "my_label": ""},
        ]
    )


class _StubAnalysis:
    """Minimal stand-in carrying only the attribute the resolvers read."""

    def __init__(self, df):
        self.df_sims = df


def _projected(label_column):
    return TRITONSWMM_analysis._project_events_table(_events_frame(), _INDICES, label_column)


def test_configured_label_is_projected_under_the_canonical_name():
    out = _projected("my_label")
    assert EVENT_LABEL_COLUMN in out.columns
    assert "my_label" not in out.columns
    assert list(out.columns) == [*_INDICES, EVENT_LABEL_COLUMN]
    assert out.loc[0, EVENT_LABEL_COLUMN] == "2003-09-17 - Isabel"


def test_unconfigured_projection_is_identical_to_the_indexer_only_projection():
    df = _events_frame()
    expected = df.loc[:, _INDICES]
    out = TRITONSWMM_analysis._project_events_table(df, _INDICES, None)
    pd.testing.assert_frame_equal(out, expected)


def test_absent_label_column_falls_back_to_the_indexer_only_projection():
    df = _events_frame()
    expected = df.loc[:, _INDICES]
    out = TRITONSWMM_analysis._project_events_table(df, _INDICES, "not_a_column")
    pd.testing.assert_frame_equal(out, expected)


def test_page_reference_carries_both_label_and_iloc_when_labelled():
    ref = event_page_reference(_StubAnalysis(_projected("my_label")), 0)
    assert ref == "2003-09-17 - Isabel (event 0)"


def test_page_reference_is_byte_identical_to_todays_text_when_unlabelled():
    """The unconfigured degradation contract: every current consumer interpolates
    the literal `event {event_iloc}`, so the reference must reproduce it exactly."""
    unconfigured = event_page_reference(_StubAnalysis(_projected(None)), 0)
    empty_cell = event_page_reference(_StubAnalysis(_projected("my_label")), 1)
    assert unconfigured == "event 0"
    assert empty_cell == "event 1"


def _write_status_csv(tmp_path, rows, header):
    path = tmp_path / "scenario_status.csv"
    lines = [",".join(header)]
    lines.extend(",".join(str(r.get(h, "")) for h in header) for r in rows)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_labels_from_status_keys_on_the_scenario_directory_basename(tmp_path):
    _write_status_csv(
        tmp_path,
        [{"scenario_directory": "/runs/a/sims/year.2_event_id.1", "event_label": "Isabel"}],
        ["scenario_directory", EVENT_LABEL_COLUMN],
    )
    assert event_labels_from_status(tmp_path) == {"year.2_event_id.1": "Isabel"}


def test_labels_from_status_is_empty_when_the_csv_is_absent(tmp_path):
    assert event_labels_from_status(tmp_path) == {}


def test_labels_from_status_is_empty_when_the_label_column_is_absent(tmp_path):
    _write_status_csv(
        tmp_path,
        [{"scenario_directory": "/runs/a/sims/year.2_event_id.1"}],
        ["scenario_directory"],
    )
    assert event_labels_from_status(tmp_path) == {}


def test_report_label_value_resolves_and_falls_back():
    from hhemt.report_plot_ids import report_label_value

    labels = {"year.9": "Ida remnant"}
    assert report_label_value(labels, "year.9", "event") == "Ida remnant"
    assert report_label_value(labels, "year.4", "event") == "event year.4"
    assert report_label_value({}, "year.4", "event") == "event year.4"
    assert report_label_value(None, "year.4", "event") == "event year.4"


def test_report_label_value_escapes_braces_so_apply_wildcards_cannot_fire():
    """snakemake's expand_labels re-expands each resolved value through
    apply_wildcards, so an unescaped brace would raise WorkflowError at render."""
    from hhemt.report_plot_ids import report_label_value

    labels = {"e1": "storm {peak} event"}
    assert report_label_value(labels, "e1", "event") == "storm {{peak}} event"


def test_report_label_value_treats_an_empty_label_as_absent():
    from hhemt.report_plot_ids import report_label_value

    assert report_label_value({"e1": "   "}, "e1", "event") == "event e1"
