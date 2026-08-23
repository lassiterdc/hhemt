"""Structural guard: source-side per-sim labels must carry the label-map call.

`test_reporting_set_cosourcing` compares registry-vs-source VALUES and passes as soon
as the two agree -- including when they agree because someone hand-wrote a matching
copy. That is the state that decayed: `a43c2822` updated the registry and the bundle
path, left the source-side copies behind, and the report the user reviews showed raw
`event_id` / `sa_id` for a merged, approved feature.

This asserts the STRUCTURE the value-comparison cannot: the emitted Snakefile's per-sim
`labels=` must call `_report_label_value`, so a future hand-written copy that drops the
label map fails here even on the day it is written and still matches.

Red before Specs 15-17, green after.
"""

from __future__ import annotations

import pytest


def _labels_lines(snakefile_text: str) -> list[str]:
    return [ln.strip() for ln in snakefile_text.split("\n") if ln.strip().startswith("labels=")]


def _assert_label_map_used(text: str, needle: str) -> None:
    lines = [ln for ln in _labels_lines(text) if needle in ln]
    assert lines, f"no emitted labels= line carries {needle!r}; rules present: {_labels_lines(text)}"
    for ln in lines:
        assert "_report_label_value" in ln, (
            f"labels= line for {needle!r} does not call _report_label_value, so the "
            f"source path emits raw wildcards while the registry emits user-defined "
            f"labels: {ln}"
        )
    assert "_EVENT_LABELS" in text, "the label-globals preamble is absent from the emitted Snakefile"


@pytest.mark.slow
def test_multisim_per_sim_page_labels_use_the_label_map(synth_multi_sim_analysis):
    text = synth_multi_sim_analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    _assert_label_map_used(text, "Simulation results")


@pytest.mark.slow
def test_sensitivity_master_per_sim_labels_use_the_label_map(synth_sensitivity_analysis):
    text = synth_sensitivity_analysis.sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )
    for needle in ("Peak flood depth", "Conduit flow"):
        _assert_label_map_used(text, needle)
