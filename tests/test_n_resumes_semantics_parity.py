"""Every file that names ``n_resumes`` is classified against the field's named semantics.

THE PROBLEM. ``n_resumes`` counts resume ATTEMPTS: the toolkit increments it when it hands
the solver a hotstart cfg, before launch. A consumer that read it as COMPLETED resumes would
be wrong in exactly the case that matters -- a sim whose every resume aborts -- and nothing
would say so. Whether a given consumer reads it as attempts is a judgement over prose and
code, which no test can make; what a test CAN do is notice that the set of consumers changed
and force a person to classify the newcomer. That is this file.

THE TABLE is total over the census below, one row per file, each with one of four
dispositions: ``reads-as-attempts`` (uses the value as a count of resume launches, or as a
``>= 1`` / ``== 0`` "was this resumed" predicate), ``reads-as-completions`` (would contradict
the named semantics -- none may exist), ``asserts-a-predicate`` (states a selection rule in
terms of the field; its row must say whether the statement is true of the code), and
``incidental mention`` (names the field in prose without reading it). A row's second
element names the change that owns its file, when that is not the change that added the
row, by a public commit identity and never by a branch name; otherwise it is ``None``.

THE CENSUS is a command, not a figure: every tracked or untracked-but-not-ignored file under
``src/hhemt/`` and ``tests/`` whose text contains ``n_resumes``. ``git ls-files`` keeps the
gitignored ``*.egg-info`` tree out -- a recursive grep over ``src/`` reaches it and counts one
high -- and ``--others --exclude-standard`` keeps a new, not-yet-committed consumer in, so the
census is the same set before and after the files that introduce it are committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_VOCABULARY = frozenset({"reads-as-attempts", "reads-as-completions", "asserts-a-predicate", "incidental mention"})

#: path -> (disposition, owning change when not this one)
_TABLE: dict[str, tuple[str, str | None]] = {
    "src/hhemt/analysis.py": ("incidental mention", None),
    "src/hhemt/analysis_validation.py": ("reads-as-attempts", None),
    "src/hhemt/bundle/_combine.py": ("reads-as-attempts", None),
    "src/hhemt/bundle/_dependency.py": ("reads-as-attempts", None),
    "src/hhemt/cli.py": ("incidental mention", None),
    "src/hhemt/config/analysis.py": ("incidental mention", None),
    "src/hhemt/config/report.py": ("incidental mention", None),
    "src/hhemt/eda/_config_diff.py": ("reads-as-attempts", None),
    "src/hhemt/eda/compute_sensitivity.py": ("reads-as-attempts", None),
    "src/hhemt/eda/raw_byte_identity.py": ("reads-as-attempts", None),
    "src/hhemt/figure_panels.py": ("incidental mention", None),
    "src/hhemt/log.py": ("reads-as-attempts", None),
    "src/hhemt/process_simulation.py": ("reads-as-attempts", None),
    "src/hhemt/processing_analysis.py": ("reads-as-attempts", None),
    "src/hhemt/report_renderers/_table_presentation.py": ("incidental mention", None),
    "src/hhemt/report_renderers/cross_experiment_compatibility.py": ("reads-as-attempts", None),
    "src/hhemt/report_renderers/cross_experiment_intercomparison_maps.py": ("reads-as-attempts", None),
    "src/hhemt/report_renderers/scenario_status_appendix.py": ("reads-as-attempts", None),
    "src/hhemt/report_templates/captions/raw_byte_identity.rst": ("asserts-a-predicate", None),
    "src/hhemt/resume_events.py": ("reads-as-attempts", None),
    "src/hhemt/run_simulation.py": ("reads-as-attempts", None),
    "src/hhemt/run_simulation_runner.py": ("reads-as-attempts", None),
    "src/hhemt/toolkit.py": ("incidental mention", None),
    "tests/fixtures/legacy_layouts/v0018_unit_test/regenerate_coupled/sims/event_id.0/log_tritonswmm.json": (
        "reads-as-attempts",
        None,
    ),
    "tests/fixtures/legacy_layouts/v0018_unit_test/regenerate_triton_only/sims/event_id.0/log_triton.json": (
        "reads-as-attempts",
        None,
    ),
    "tests/fixtures/legacy_layouts/v0018_unit_test/stamp_stale/sims/event_id.0/log_triton.json": (
        "reads-as-attempts",
        None,
    ),
    "tests/test_check_known_resume_defects.py": ("reads-as-attempts", None),
    "tests/test_clean_restart_facade_thread.py": ("incidental mention", None),
    "tests/test_clean_restart_wipe.py": ("incidental mention", None),
    "tests/test_consecutive_failure_budget_gate.py": ("incidental mention", None),
    "tests/test_coupled_resume_validity.py": ("reads-as-attempts", None),
    "tests/test_debug_flag_keep_previous_snapshots.py": ("reads-as-attempts", None),
    "tests/test_event_labeling.py": ("incidental mention", None),
    "tests/test_known_solver_defects.py": ("reads-as-attempts", None),
    "tests/test_kr_b_resume_reporting_tsteps.py": ("reads-as-attempts", None),
    "tests/test_log_concurrency.py": ("reads-as-attempts", None),
    "tests/test_log_write_stale_instance.py": ("incidental mention", None),
    "tests/test_member_identity.py": ("reads-as-attempts", None),
    "tests/test_n_resumes_reporting.py": ("reads-as-attempts", None),
    "tests/test_n_resumes_semantics_parity.py": ("reads-as-attempts", None),
    "tests/test_raw_resume_identity.py": ("reads-as-attempts", None),
    "tests/test_synth_07_validation_report.py": ("reads-as-attempts", None),
    "tests/test_resume_event_cause_term.py": ("reads-as-attempts", None),
    "tests/test_resume_event_ledger.py": ("reads-as-attempts", None),
    "tests/test_resume_stepback_from_cut_off_snapshot.py": ("reads-as-attempts", None),
    "tests/test_resume_verdict_disclosed_denominator.py": ("reads-as-attempts", None),
    "tests/test_review_quality_snapshot_pin_awareness.py": ("reads-as-attempts", None),
    "tests/test_snapshot_failed_replay.py": ("reads-as-attempts", None),
    "tests/test_swmm_water_quality_outputs.py": ("reads-as-attempts", None),
    "tests/test_synth_combine_pip2.py": ("incidental mention", None),
    "tests/test_synth_compute_sensitivity_eda.py": ("reads-as-attempts", None),
    "tests/test_synth_config_diff_grouping_fungibility.py": ("reads-as-attempts", None),
    "tests/test_synth_cross_bundle_intercomparison.py": ("reads-as-attempts", None),
    "tests/test_synth_experiment_dependency.py": ("reads-as-attempts", None),
    "tests/test_synth_metadata_renderer.py": ("incidental mention", None),
    "tests/test_synth_resume_schedule.py": ("reads-as-attempts", None),
    "tests/test_validate_analysis_single_snapshot.py": ("reads-as-attempts", None),
    "tests/test_wp2b_resume_mechanism_review.py": ("reads-as-attempts", None),
}


def _census() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "--cached", "--others", "--exclude-standard", "--", "src/hhemt", "tests"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    hits = set()
    for rel in filter(None, out):
        path = _REPO / rel
        try:
            if "n_resumes" in path.read_text(errors="replace"):
                hits.add(rel)
        except (OSError, IsADirectoryError):
            continue
    return hits


def test_the_table_is_total_over_the_census():
    """Property: the classification table names exactly the files the census finds.

    Class: every file under ``src/hhemt/`` or ``tests/`` that git tracks or would track and
    whose text names ``n_resumes``.

    Why required, checkable against the code: a new consumer reading the field the wrong way
    is only caught if someone classifies it, and this is what makes someone. A file that
    stops naming the field drops out of the census and must leave the table too.

    What kills it: adding a file that names ``n_resumes`` without a table row, or a row for a
    file that no longer names it.

    A second correct implementation under which it still passes: a census computed with
    ``git grep -l n_resumes`` plus an untracked-file scan, if it yields the same set.
    """
    census = _census()
    assert sorted(census - _TABLE.keys()) == [], "unclassified n_resumes consumers -- add a row"
    assert sorted(_TABLE.keys() - census) == [], "rows for files that no longer name n_resumes -- remove them"


def test_no_consumer_reads_n_resumes_as_completions():
    """Property: every row uses the four-value vocabulary, and none reads the field as completions.

    What kills it: a row classified ``reads-as-completions`` (a consumer contradicting the
    named semantics), or a disposition outside the vocabulary.

    A second correct implementation under which it still passes: an enum in place of strings.
    """
    assert {d for d, _ in _TABLE.values()} <= _VOCABULARY
    assert [p for p, (d, _) in _TABLE.items() if d == "reads-as-completions"] == []


def test_the_caption_states_the_predicate_its_renderer_applies():
    """Property: the clean-identity caption states raw-raster presence as its selection rule, not ``n_resumes == 0``.

    Class: the one ``asserts-a-predicate`` row. Its renderer (``eda/raw_byte_identity.py``)
    keeps members by raw-raster presence and carries ``n_resumes`` only as metadata, so a caption
    describing the panel as the ``n_resumes == 0`` members is false of its own figure as soon as
    a resumed member is re-run clean.

    What kills it: restoring the ``n_resumes == 0`` wording, or dropping every statement of the
    raw-raster selection rule from the body.

    A second correct implementation under which it still passes: a caption whose selection
    sentence reads "Members are selected only by whether they kept their raw output rasters" --
    different words, same rule. So the positive arm asks for one SENTENCE that both speaks of
    selection and names raw rasters (at most one word between "raw" and "raster"), which every
    statement of the rule carries. A bare "raw raster" search is not enough: the caption's
    unrelated "raw rasters are not bit-identical" sentence would satisfy it with no selection
    rule stated, and the old caption carries that sentence too.
    """
    import re

    caption = (_REPO / "src/hhemt/report_templates/captions/raw_byte_identity.rst").read_text()
    assert not re.search(r"n_resumes\W*==\W*0", caption)
    sentences = re.split(r"(?<=[.;])\s+", caption.split("\n", 1)[1])
    assert any(
        re.search(r"\bselect", s, re.IGNORECASE) and re.search(r"\braw\W+(?:\w+\W+)?rasters?\b", s, re.IGNORECASE)
        for s in sentences
    )
    assert _TABLE["src/hhemt/report_templates/captions/raw_byte_identity.rst"][0] == "asserts-a-predicate"
