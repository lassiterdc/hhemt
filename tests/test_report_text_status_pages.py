"""Reader-facing text of the status and health pages: the code facts their wording states.

Package 3b rewrites the text of the status and health pages one page per chunk. Each chunk
adds the tests of its page here (shape binding B3): a test asserts the code fact the text
states, located by a stable anchor (a label, a column name), never by comparing a whole
sentence, writes no typed count, and carries a must-fire control that feeds a mutated
input and expects the failure. Nothing here compiles or runs a solver.

Chunk 3b.1, the P05 summary page (shape item I28, section 7 rows for P05).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt.config.report import report_config
from hhemt.report_renderers import per_analysis_summary

_OPTIONS = 'const tableOptions = {"data": '

# Words the term table retires from labels and column names of the summary page (style brief
# rule 2): "sim" and "simulation" as a count noun, "model type", and the mechanism word
# "derived" that the retired expected-total label carried (brief rule 1).
_RETIRED = re.compile(r"\b(sims?|simulations?|model types?|derived)\b", re.IGNORECASE)


# ---- Stub analyses: inputs only; the renderer under test is the real one ----------------


def _sub(n_events: int) -> SimpleNamespace:
    return SimpleNamespace(df_sims=pd.DataFrame({"event": range(n_events)}))


def _plain(tmp_path: Path, n_events: int, n_status_events: int, model_types: list[str]) -> SimpleNamespace:
    """A plain analysis: `n_events` events in `df_sims`, `n_status_events` in the status record."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {"event_iloc": i, "model_type": mt, "run_completed": True, "scenario_setup": True}
            for i in range(n_status_events)
            for mt in model_types
        ]
    ).to_csv(analysis_dir / "scenario_status.csv", index=False)
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir, simlog_directory=analysis_dir / "logs" / "sims"),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        sensitivity=None,
        df_sims=pd.DataFrame({"event": range(n_events)}),
        _get_enabled_model_types=lambda: model_types,
    )


def _sweep(tmp_path: Path, events_by_member: dict[int, int]) -> SimpleNamespace:
    """A sweep whose members hold `events_by_member[member_id]` events each."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {
                "member_id": m,
                "event_iloc": i,
                "model_type": "tritonswmm",
                "run_completed": True,
                "scenario_setup": True,
            }
            for m, n in events_by_member.items()
            for i in range(n)
        ]
    ).to_csv(analysis_dir / "scenario_status.csv", index=False)
    members = list(events_by_member)
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir, simlog_directory=analysis_dir / "logs" / "sims"),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        sensitivity=SimpleNamespace(
            members={m: _sub(n) for m, n in events_by_member.items()},
            df_setup=pd.DataFrame({"member": members}),
        ),
        df_sims=pd.DataFrame({"event": range(max(events_by_member.values()))}),
        _get_enabled_model_types=lambda: ["tritonswmm"],
    )


def _data(analysis: SimpleNamespace, tmp_path: Path) -> list[dict]:
    """The `data` array of the Tabulator options in the rendered summary page."""
    out = tmp_path / "per_analysis_summary.html"
    per_analysis_summary.render(analysis, report_config(), out)
    html = out.read_text(encoding="utf-8")
    data, _end = json.JSONDecoder().raw_decode(html[html.index(_OPTIONS) + len(_OPTIONS) :])
    return data


def _labels(data: list[dict]) -> list[str]:
    """Every label the reader sees on the page: row labels of a plain page, column names of a sweep."""
    if data and "Metric" in data[0]:
        return [row["Metric"] for row in data]
    return list(data[0])


# ---- 3b.1: the P05 summary page ----------------------------------------------------------


@pytest.mark.parametrize(
    ("n_events", "n_status_events"),
    [(5, 3), (2, 4)],
)
def test_plain_scenarios_row_counts_events_not_status_records(
    tmp_path: Path, n_events: int, n_status_events: int
) -> None:
    """The caption says Scenarios counts events, one scenario per event.

    Input class: a plain analysis whose event list and whose status record hold different numbers
    of events (more events than records; fewer events than records). The row follows `df_sims`.
    Required because the caption's sentence states exactly this. Killed by counting rows of the
    status record. A second correct implementation counts `analysis.df_sims.index` through any
    other accessor and still passes.
    """
    rows = {r["Metric"]: r["Value"] for r in _data(_plain(tmp_path, n_events, n_status_events, ["triton"]), tmp_path)}

    assert rows["Scenarios"] == n_events


def test_plain_scenarios_row_tracks_the_event_list(tmp_path: Path) -> None:
    """Must-fire control for the test above: changing only the event list changes the row."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = {r["Metric"]: r["Value"] for r in _data(_plain(tmp_path / "a", 2, 2, ["triton"]), tmp_path / "a")}
    b = {r["Metric"]: r["Value"] for r in _data(_plain(tmp_path / "b", 6, 2, ["triton"]), tmp_path / "b")}

    assert a["Scenarios"] != b["Scenarios"]


@pytest.mark.parametrize("events_by_member", [{1: 2, 2: 4}, {1: 5, 2: 3, 3: 1}])
def test_sweep_scenarios_column_counts_each_members_events(tmp_path: Path, events_by_member: dict[int, int]) -> None:
    """The sweep form carries a Scenarios column holding each member's own event count.

    Input class: sweeps whose members hold different numbers of events (ascending; descending
    with three members). The column is anchored by its name and the cell by the member id.
    Required because the caption bolds Scenarios as an element both table forms print. Killed by
    printing the longest event list in every row, or by naming the column anything else. A
    second correct implementation reads the count from the status record and still passes.
    """
    data = _data(_sweep(tmp_path, events_by_member), tmp_path)

    assert "Scenarios" in _labels(data)
    assert {row["member"]: row["Scenarios"] for row in data} == events_by_member


def test_sweep_scenarios_column_follows_the_member_sizes(tmp_path: Path) -> None:
    """Must-fire control for the test above: swapping the member sizes swaps the cells."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = {r["member"]: r["Scenarios"] for r in _data(_sweep(tmp_path / "a", {1: 2, 2: 4}), tmp_path / "a")}
    b = {r["member"]: r["Scenarios"] for r in _data(_sweep(tmp_path / "b", {1: 4, 2: 2}), tmp_path / "b")}

    assert a != b


@pytest.mark.parametrize("form", ["plain", "sweep"])
def test_no_label_on_the_summary_page_carries_a_retired_word(tmp_path: Path, form: str) -> None:
    """Every label of both table forms uses the term table's words (brief rule 2).

    Input class: the labels of a plain page (all rows) and of a sweep page (all columns).
    Required because the summary page counted "simulations" and "sims" under two labels the
    term table retires. Killed by restoring "n sims" or "Enabled model types". A second correct
    implementation that words the rows differently passes while it avoids the retired words.
    """
    analysis = _plain(tmp_path, 3, 3, ["triton", "swmm"]) if form == "plain" else _sweep(tmp_path, {1: 2, 2: 2})

    assert [lab for lab in _labels(_data(analysis, tmp_path)) if _RETIRED.search(lab)] == []


@pytest.mark.parametrize(
    "retired_label",
    ["n sims", "Total simulations", "Enabled model types", "Expected total (derived: 6 weather events)"],
)
def test_the_retired_word_matcher_fires_on_each_retired_label(retired_label: str) -> None:
    """Must-fire control for the retired-word test: the matcher returns a hit on each label it retires."""
    assert _RETIRED.search(retired_label) is not None


# ---- 3b.1 reviewer tests: the P05 page-kind markers -------------------------------------------
#
# The rename of the plain-page label "Scenarios (events)" to "Scenarios" turns the P05-plain marker
# of tests/test_caption_page_kinds.py red. The one-string edit that greens it must keep the marker's
# purpose, which that file states: "an element the rendering must print ... these show each stub
# reaches its page". "Scenarios" is now printed by BOTH P05 kinds (plain row, sweep column), so a
# marker with that value no longer separates a plain stub from a sweep stub.

_P05_KINDS = ("P05-plain", "P05-sweep")


def _p05_texts(tmp_path: Path) -> dict[str, str]:
    from tests.test_caption_page_kinds import _by_id, render_page

    return {k: render_page(_by_id(k), tmp_path / k) for k in _P05_KINDS}


def _marker_belongs_to_its_kind_only(marker: str, own_text: str, sibling_texts: list[str]) -> bool:
    return marker in own_text and not any(marker in t for t in sibling_texts)


@pytest.mark.parametrize("kind_id", _P05_KINDS)
def test_each_p05_marker_is_printed_by_its_own_page_kind_and_by_no_sibling_kind(tmp_path: Path, kind_id: str) -> None:
    """A page kind's marker is an element its own rendering prints and its sibling kind does not.

    Input class: the two P05 page kinds, plain and sweep, each rendered from its own stub inputs.
    Required because a marker both kinds print passes against a stub that reached the wrong table
    form, so the marker no longer shows the stub reaches its own page. Killed by setting the
    P05-plain marker to "Scenarios", which the sweep column also prints, or to a string neither
    kind prints. A second correct implementation sets the P05-plain marker to "Model runs" or
    "Expected scenarios", which only the plain form prints, and still passes.
    """
    from tests.test_caption_page_kinds import _MARKERS

    texts = _p05_texts(tmp_path)
    others = [t for k, t in texts.items() if k != kind_id]

    assert _marker_belongs_to_its_kind_only(_MARKERS[kind_id], texts[kind_id], others)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [("Scenarios", False), ("Model runs", True), ("Scenarios (events)", False)],
)
def test_the_marker_matcher_fires_on_a_marker_that_does_not_separate_the_p05_kinds(marker: str, expected: bool) -> None:
    """Must-fire control for the test above, on fixed texts so it does not depend on any renderer wording.

    Input class: a marker printed by both kinds (False), a marker printed by the own kind only (True)
    and a marker the own kind does not print (False). The texts are literals shaped like the two
    renderings, so the control passes and fails identically before and after any label change.
    """
    own_text = "Scenarios 3 Model runs 9 Successful 2"
    sibling_text = "member Scenarios successful pending failed"

    assert _marker_belongs_to_its_kind_only(marker, own_text, [sibling_text]) is expected


# ---- 3b.2: the P06 scenario status appendix ------------------------------------------------

_P06_KINDS = ("P06-plain", "P06-sweep", "P06-csv_missing")


def _p06_page(kind_id: str, tmp_path: Path) -> tuple[str, str]:
    """(page text, raw html) of the appendix rendered from a page-kind stub."""
    from tests.test_caption_page_kinds import _by_id, render_page

    text = render_page(_by_id(kind_id), tmp_path)
    return text, (tmp_path / f"{kind_id.split('-')[0]}.html").read_text(encoding="utf-8")


def _p06_rows(raw_html: str) -> list[dict]:
    data, _end = json.JSONDecoder().raw_decode(raw_html[raw_html.index(_OPTIONS) + len(_OPTIONS) :])
    return data


@pytest.mark.parametrize("kind_id", ["P06-plain", "P06-sweep"])
def test_appendix_has_one_row_per_record_of_the_status_file(tmp_path: Path, kind_id: str) -> None:
    """The caption says each row is one model run: the grid holds exactly the rows of scenario_status.csv.

    Input class: both table forms (plain, sweep), whose status file holds one record per model run.
    Required because the caption states the row unit. Killed by collapsing records to one row per
    event or per member. A second correct implementation that sorts or regroups the rows still passes.
    """
    from tests.test_caption_page_kinds import _by_id

    _text, raw = _p06_page(kind_id, tmp_path)
    csv = _by_id(kind_id).stub_dir / "scenario_status.csv"

    assert len(_p06_rows(raw)) == len(pd.read_csv(csv))


def test_appendix_csv_missing_shows_the_not_shown_sentence_and_no_grid(tmp_path: Path) -> None:
    """The caption says the page states a missing file and shows no rows.

    Input class: the csv_missing kind (no status file). Required because the caption states both
    halves for the missing case. Killed by restoring a blank page or by rendering an empty grid.
    A second correct implementation that words the sentence differently still passes while it
    keeps the form "{element} is not shown: {reason}" and names the file.
    """
    text, raw = _p06_page("P06-csv_missing", tmp_path)

    assert "is not shown:" in text and "scenario_status.csv" in text
    assert _OPTIONS not in raw


def test_appendix_present_file_has_no_not_shown_sentence(tmp_path: Path) -> None:
    """Must-fire control for the test above: a kind with the file prints the grid and not the sentence."""
    text, raw = _p06_page("P06-plain", tmp_path)

    assert "is not shown:" not in text
    assert _OPTIONS in raw


@pytest.mark.parametrize("kind_id", _P06_KINDS)
def test_each_p06_kind_prints_its_own_marker(kind_id: str, tmp_path: Path) -> None:
    """Every P06 kind prints its marker, so each stub reaches its page (all three cases assert).

    Input class: the three P06 kinds. Killed by a stub that renders another kind's page. Separation
    from the sibling kinds is the reviewer's test below.
    """
    from tests.test_caption_page_kinds import _MARKERS

    assert _MARKERS[kind_id] in _p06_page(kind_id, tmp_path)[0]


@pytest.mark.parametrize("kind_id", ["P06-plain", "P06-sweep"])
def test_appendix_table_forms_print_the_completion_column(kind_id: str, tmp_path: Path) -> None:
    """The caption names a completion status on every row: both table forms print run_completed.

    Input class: plain and sweep stubs. Anchored by the column name. Killed by hiding or renaming
    the column. A second correct implementation that reorders columns still passes.
    """
    _text, raw = _p06_page(kind_id, tmp_path)

    assert "run_completed" in _p06_rows(raw)[0]


_PERF_NOTES_CLAIM = "CUMULATIVE wallclock across every allocation"


def _footnote_of_group(label: str) -> str:
    from hhemt.report_renderers.scenario_status_appendix import _build_column_groups

    groups = _build_column_groups(pd.DataFrame({"perf_Total": [1.0]}), [])
    return next(note for lab, _fields, note in groups if lab == label)


def test_performance_footnote_agrees_with_the_cumulative_aggregation_the_code_documents() -> None:
    """The footnote says a resumed run adds up every allocation, as the summary dataset's own notes say.

    Input class: the Performance Breakdown group of a resumed member. Anchored on the code fact
    (the dataset notes written by process_simulation) and the keyword of the footnote ("allocation"),
    not on a sentence. Killed by restoring the claim that the figures cover only the resumed range.
    A second correct implementation words the footnote differently while naming the allocations.
    """
    import inspect

    from hhemt import process_simulation

    assert _PERF_NOTES_CLAIM in inspect.getsource(process_simulation)
    assert "allocation" in _footnote_of_group("Performance Breakdown")
    assert "only the resumed" not in _footnote_of_group("Performance Breakdown")


def test_performance_footnote_check_fires_on_the_retired_claim() -> None:
    """Must-fire control: the two footnote predicates reject the retired wording."""
    retired = "cover only the resumed timestep range when a run was resumed."
    assert "allocation" not in retired
    assert "only the resumed" in retired


# ---- 3b.2 reviewer tests: P06 marker separation and caption markup ----------------------------


@pytest.mark.parametrize("kind_id", _P06_KINDS)
def test_each_p06_marker_is_printed_by_its_own_kind_and_by_no_sibling_kind(kind_id: str, tmp_path: Path) -> None:
    """A page kind's marker is printed by its own rendering and by no sibling P06 kind.

    Input class: the three P06 kinds (plain, sweep, csv_missing), each rendered from its stub inputs.
    Required because a marker that every kind prints cannot show that a stub reached its own page:
    a plain stub that rendered the missing-file placeholder would still print the shared heading and
    pass `test_rendering_reaches_its_page_kind`. Killed by a plain or sweep marker equal to the page
    heading "Scenario Status", which the placeholder also prints. A second correct implementation
    sets the plain marker to "model_type" and the sweep marker to "member_id" (column headers only
    that table form prints) and still passes.
    """
    from tests.test_caption_page_kinds import _MARKERS

    texts = {k: _p06_page(k, tmp_path / k)[0] for k in _P06_KINDS}
    others = [t for k, t in texts.items() if k != kind_id]

    assert _marker_belongs_to_its_kind_only(_MARKERS[kind_id], texts[kind_id], others)


def _caption_html(rst_text: str) -> str:
    """The HTML docutils renders for a caption source, Jinja expressions replaced by a word."""
    import re as _re

    from docutils.core import publish_parts

    masked = _re.sub(r"\{\{.*?\}\}|\{%.*?%\}", "x", rst_text, flags=_re.S)
    return publish_parts(masked, writer_name="html", settings_overrides={"report_level": 5, "halt_level": 5})["body"]


def test_no_caption_names_a_file_in_interpreted_text() -> None:
    """A caption writes a file name or field in literal markup, which renders as code.

    Input class: every `.rst` under report_templates/captions. Required because brief rule 7 sets file
    names and configuration fields in code formatting, and in reStructuredText a single-backtick span
    is interpreted text that renders in the citation style (italic), not as code; the other captions
    all use the double-backtick literal. Killed by writing `scenario_status.csv` with single backticks.
    A second correct implementation writes the name as a literal span, or leaves it plain, and passes.
    """
    from tests.test_caption_page_kinds import CAPTIONS

    offenders = [p.name for p in sorted(CAPTIONS.glob("*.rst")) if "<cite>" in _caption_html(p.read_text("utf-8"))]

    assert offenders == []


@pytest.mark.parametrize(("source", "has_cite"), [("A `file.csv` here.", True), ("A ``file.csv`` here.", False)])
def test_the_interpreted_text_matcher_separates_single_from_double_backticks(source: str, has_cite: bool) -> None:
    """Must-fire control for the test above: the matcher reports a single-backtick span and not a literal one."""
    assert ("<cite>" in _caption_html(source)) is has_cite


# ---- 3b.3: the P07 Errors and Warnings page ------------------------------------------------

_P07_KINDS = ("P07-failures", "P07-all_not_applicable", "P07-record_missing")


def _p07_text(kind_id: str, tmp_path: Path) -> str:
    from tests.test_caption_page_kinds import _by_id, render_page

    return render_page(_by_id(kind_id), tmp_path)


@pytest.mark.parametrize("kind_id", _P07_KINDS)
def test_each_p07_kind_prints_its_own_marker(kind_id: str, tmp_path: Path) -> None:
    """Every P07 kind prints its marker, so each stub reaches its page (all three cases assert).

    Input class: the three P07 kinds. Killed by a stub that renders another kind's page.
    """
    from tests.test_caption_page_kinds import _MARKERS

    assert _MARKERS[kind_id] in _p07_text(kind_id, tmp_path)


@pytest.mark.parametrize("kind_id", _P07_KINDS)
def test_each_p07_marker_is_printed_by_no_sibling_kind(kind_id: str, tmp_path: Path) -> None:
    """A P07 marker is printed by its own kind and by neither sibling kind.

    Input class: the three P07 kinds. Killed by a shared marker such as the banner text "0 of 0",
    which two kinds print. A second correct implementation uses any other string only the own kind prints.
    """
    from tests.test_caption_page_kinds import _MARKERS

    texts = {k: _p07_text(k, tmp_path / k) for k in _P07_KINDS}
    others = [t for k, t in texts.items() if k != kind_id]

    assert _marker_belongs_to_its_kind_only(_MARKERS[kind_id], texts[kind_id], others)


def test_a_record_with_no_checks_shows_the_notice_and_no_pass_banner(tmp_path: Path) -> None:
    """A missing record states itself and does not render a green pass banner.

    Input class: the record_missing kind (no validation_report.json). Anchored on the banner's
    stable phrase "applicable checks". Required because a banner counting zero checks reads as a
    clean analysis. Killed by restoring the banner for an empty report. A second correct
    implementation words the notice differently while naming the file.
    """
    text = _p07_text("P07-record_missing", tmp_path)

    assert "validation_report.json" in text
    assert "applicable checks" not in text


def test_a_record_with_checks_shows_the_banner_and_no_missing_notice(tmp_path: Path) -> None:
    """Must-fire control for the test above: a kind with checks prints the banner and not the notice."""
    text = _p07_text("P07-failures", tmp_path)

    assert "applicable checks" in text
    assert "validation_report.json" not in text


def test_a_check_that_does_not_apply_is_counted_and_has_no_row(tmp_path: Path) -> None:
    """The omission line counts every not-applicable check, so no table row shows one.

    Input class: the all_not_applicable kind, whose two checks are not applicable (one at the
    resource level, which printed a row while the count said both were left out). Anchored on the
    stub's check names. Killed by rendering the resource-level row. A second correct implementation
    that counts differently still passes while no not-applicable check is tabled.
    """
    text = _p07_text("P07-all_not_applicable", tmp_path)

    assert "check_gpu_binding" not in text
    assert "check_raw_outputs" not in text


def test_a_check_that_applies_keeps_its_row(tmp_path: Path) -> None:
    """Must-fire control for the test above: applicable checks of the failures kind keep their rows."""
    text = _p07_text("P07-failures", tmp_path)

    assert "check_scenarios_run" in text


def test_log_recoveries_description_says_each_document_is_counted_once() -> None:
    """The Log recoveries description states the counting unit the check uses.

    Anchored on the code fact that the check counts distinct log documents (tests/test_validation_check_figures.py
    pins that behaviour) and on the keyword "once". Killed by restoring the description that spoke of
    a record carried by a document. A second correct implementation words it differently while stating the unit.
    """
    from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

    assert "once" in _CHECK_VOCABULARY["Log recoveries"][1]
    assert "once" not in "No analysis, system, or per-model scenario log document carries a record"


# ---- 3b.3 reviewer tests: the omission count and the check descriptions ----------------------

_NOT_SHOWN = re.compile(r"(\d+) checks? (?:is|are) not shown")


def test_the_omission_line_counts_every_check_that_does_not_apply(tmp_path: Path) -> None:
    """The page counts the checks that do not apply in one line (K14, rule 6): the figure is that count.

    Input class: the all_not_applicable kind, whose report file holds a mix of levels; the expected
    figure is read from that file (a set compared with a set, no typed count). Required because the
    page states the count and a reader tells "never ran" from "passed" by it. Killed by a line that
    miscounts (one more or one fewer) or that is dropped. A second correct implementation that words
    the line differently still passes while it states the figure before "not shown".
    """
    from tests.test_caption_page_kinds import _by_id

    report = json.loads((_by_id("P07-all_not_applicable").stub_dir / "validation_report.json").read_text())
    expected = sum(1 for c in report["checks"] if not c.get("applicable", True))
    found = _NOT_SHOWN.findall(_p07_text("P07-all_not_applicable", tmp_path))

    assert found == [str(expected)]


def test_the_omission_line_is_absent_when_every_check_applies(tmp_path: Path) -> None:
    """Must-fire control for the test above: the failures kind has no inapplicable check and prints no line."""
    assert _NOT_SHOWN.findall(_p07_text("P07-failures", tmp_path)) == []


_RETIRED_IN_DESCRIPTION = re.compile(r"\bsims?\b|\bmasters?\b|—", re.IGNORECASE)


def test_no_check_description_carries_a_retired_word_or_an_em_dash() -> None:
    """The `Check` column prints each description, so none uses "sim" as a noun or an em dash.

    Input class: the description of every entry of the check vocabulary (the display name is a
    separate, test-pinned string and is not read here). Required because brief rules 2 and 8 apply to
    every string a report reader sees, and this renderer is the only owner of these strings. Killed by
    restoring "sim" in the three resume descriptions or the dash in the two others. A second correct
    implementation that words them differently passes while it avoids both.
    """
    from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

    offenders = sorted(k for k, (_name, desc) in _CHECK_VOCABULARY.items() if _RETIRED_IN_DESCRIPTION.search(desc))

    assert offenders == []


@pytest.mark.parametrize(
    "text", ["Every resumed sim runs", "a regression detector — a guard", "Every Sims", "the master tree"]
)
def test_the_retired_description_matcher_fires(text: str) -> None:
    """Must-fire control for the test above: the matcher fires on a "sim" noun, an em dash and a capital "Sims"."""
    assert _RETIRED_IN_DESCRIPTION.search(text) is not None


def test_the_retired_description_matcher_passes_a_clean_description() -> None:
    """Control for the matcher's other side: "simulation" and a plain hyphen are not hits."""
    assert _RETIRED_IN_DESCRIPTION.search("A simulation of a model-run record") is None


# ---- 3b.3 reviewer tests, round 6: the term table's multi-word retired phrases ---------------

_RETIRED_TERM_PHRASES = re.compile(
    r"peak water level|max(?:imum)? water level|sub-analys[ei]s|sensitivity master|model (?:type|arm)"
    r"|sentinel|sidecar|read-model|flag file|\bcrates?\b|compute-config",
    re.IGNORECASE,
)


def test_no_check_description_carries_a_retired_term_table_phrase() -> None:
    """The `Check` column prints each description, so none uses a multi-word term the style brief retires.

    Input class: the description of every entry of the check vocabulary, against the retired column of
    the brief's term table that is a phrase rather than one word (`peak water level` and `max water
    level` for the per-cell maximum depth, `model type`, `sentinel`, `read-model` and the others). The
    single-word retired set is read by `test_no_check_description_carries_a_retired_word_or_an_em_dash`
    and by the style test; neither reads a phrase. Required because brief rule 2 applies to every string
    a report reader sees. Killed by `forcing tail influence` printing "peak water level" for the
    `max_wlevel_m` value the check reads. A second correct implementation words it "peak flood depth"
    or "maximum depth" and passes.
    """
    from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

    offenders = sorted(k for k, (_name, desc) in _CHECK_VOCABULARY.items() if _RETIRED_TERM_PHRASES.search(desc))

    assert offenders == []


@pytest.mark.parametrize(
    "text",
    [
        "No cell reaches its peak water level",
        "the Max water level",
        "maximum water level",
        "a sensitivity master",
        "the _du.json sentinel",
    ],
)
def test_the_retired_term_phrase_matcher_fires(text: str) -> None:
    """Must-fire control for the test above: the matcher fires on each retired phrase, in either case and long form."""
    assert _RETIRED_TERM_PHRASES.search(text) is not None


def test_the_retired_term_phrase_matcher_passes_the_replacement_terms() -> None:
    """Control for the matcher's other side: the replacement terms and the boundary water level are not hits."""
    assert (
        _RETIRED_TERM_PHRASES.search("No cell reaches its peak flood depth after the boundary water level froze")
        is None
    )


# ---- 3b.4: the P08 Disk use page ------------------------------------------------------------

_P08_KINDS = ("P08-live", "P08-record_missing")


def _p08_html(analysis_dir: Path, tmp_path: Path) -> str:
    from tests.test_disk_utilization_render_breakdown import _render

    return _render(analysis_dir, tmp_path)


def _p08_without_record(tmp_path: Path) -> str:
    analysis_dir = tmp_path / "no_record"
    analysis_dir.mkdir()
    return _p08_html(analysis_dir, tmp_path)


@pytest.mark.parametrize("kind_id", _P08_KINDS)
def test_each_p08_marker_is_printed_by_its_own_kind_and_by_no_sibling_kind(kind_id: str, tmp_path: Path) -> None:
    """Each P08 kind prints its marker and the other kind does not (page-kind coverage, shape B4).

    Input class: the two P08 kinds. Killed by a marker both kinds print, which would stop showing
    that a stub reached its own page. A second correct implementation uses any string only the own
    kind prints.
    """
    from tests.test_caption_page_kinds import _MARKERS

    texts = {k: _p07_text(k, tmp_path / k) for k in _P08_KINDS}
    others = [t for k, t in texts.items() if k != kind_id]

    assert _marker_belongs_to_its_kind_only(_MARKERS[kind_id], texts[kind_id], others)


def test_the_disk_use_time_label_says_totals_recorded_and_prints_the_recorded_time(tmp_path: Path) -> None:
    """The line above the table reads "Totals recorded at {time}", with the time the record holds.

    Input class: a live analysis whose `_status/_du.json` holds a `computed_at`. Required because the
    shape row for P08 replaces "Sentinel computed at" (brief rule 2: record, not sentinel). Killed by
    restoring the old label or by printing another time. A second correct implementation words the
    label differently while it keeps "recorded" and the time.
    """
    from tests.test_disk_utilization_render_breakdown import _live_analysis

    analysis_dir = _live_analysis(tmp_path)
    computed_at = json.loads((analysis_dir / "_status" / "_du.json").read_text())["computed_at"]
    html = _p08_html(analysis_dir, tmp_path)

    assert f"Totals recorded at {computed_at}" in html
    assert "Sentinel computed" not in html


def test_the_disk_use_time_label_check_fires_on_the_retired_label() -> None:
    """Must-fire control for the label test: the retired-phrase matcher fires on the old label."""
    assert _RETIRED_TERM_PHRASES.search("Sentinel computed at 2026-01-01T00:00:00") is not None


_NOTICE_FORM = re.compile(r"^The [^:.]+ is not shown: [^.]*<code>[^<]+</code>[^.]*\.$")


def test_the_missing_record_notice_is_one_not_shown_sentence_naming_the_file_in_code(tmp_path: Path) -> None:
    """With no `_du.json` the page prints one "{Element} is not shown: {reason}." sentence.

    Input class: an analysis directory with no record. Required because brief rules 6 and 7 set the
    notice form and the code formatting of a file name, and rule 8 retires the em dash. Killed by the
    old notice ("Disk utilization sentinel absent", an em dash, a re-run prompt without the form), by a
    file name outside `<code>`, or by a second sentence. A second correct implementation words the
    reason differently and passes.
    """
    html = _p08_without_record(tmp_path)
    notice = re.search(r"<p class='du-missing'>(.*?)</p>", html, re.S)

    assert notice is not None
    assert _NOTICE_FORM.match(notice.group(1))
    assert "<code>_status/_du.json</code>" in notice.group(1)
    assert "—" not in notice.group(1)
    assert _RETIRED_TERM_PHRASES.search(notice.group(1)) is None
    assert _OPTIONS not in html


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("Disk utilization sentinel absent — re-run processing to populate <code>_status/_du.json</code>.", False),
        ("The disk use table is not shown: _status/_du.json is missing.", False),
        ("The disk use table is not shown: <code>_status/_du.json</code> is missing. Run it again.", False),
        ("The disk use table is not shown: <code>_status/_du.json</code> is missing, so re-run processing.", True),
    ],
)
def test_the_notice_form_check_separates_the_old_and_malformed_notices_from_the_form(text: str, matches: bool) -> None:
    """Must-fire control for the notice test: the form matcher rejects the old notice and its malformed variants."""
    assert (_NOTICE_FORM.match(text) is not None) is matches


def test_the_bundle_notice_uses_the_term_table_words_and_still_says_nothing_was_remeasured(tmp_path: Path) -> None:
    """The bundle-root branch keeps its own notice, free of the retired word "sentinel".

    Input class: a bundle-like root (manifest present). Required because the shape keeps this branch
    (K5) and brief rule 2 retires "sentinel". Killed by restoring "the total recorded in the sentinel".
    A second correct implementation words the notice differently while it says the rows were not
    re-measured.
    """
    from tests.test_disk_utilization_render_breakdown import _bundle_root

    html = _p08_html(_bundle_root(tmp_path), tmp_path)
    notice = re.search(r"<p class='du-not-remeasured'>(.*?)</p>", html, re.S)

    assert notice is not None
    assert "re-measured" in notice.group(1)
    assert _RETIRED_TERM_PHRASES.search(notice.group(1)) is None


def test_the_disk_use_caption_describes_only_the_live_page_in_term_table_words() -> None:
    """The caption cuts the bundle clause and the re-run sentence and says "record", not "sentinel".

    Input class: the `disk_utilization.rst` source. Required because shape rows P08 (C-5, K5, K14)
    cut the clause about a route that does not render the page and the sentence the page's own notice
    states. Killed by restoring either sentence or the word "sentinel". A second correct implementation
    words the kept sentences differently and passes.
    """
    from tests.test_caption_page_kinds import CAPTIONS

    text = (CAPTIONS / "disk_utilization.rst").read_text("utf-8")

    assert _RETIRED_TERM_PHRASES.search(text) is None
    assert "bundle" not in text.lower()
    assert "re-run" not in text.lower()


@pytest.mark.parametrize(
    "text",
    [
        "On a page regenerated from a bundle, every row is the total recorded in that sentinel",
        "When the sentinel is absent, the table is replaced by a re-run prompt.",
    ],
)
def test_the_caption_check_fires_on_the_cut_sentences(text: str) -> None:
    """Must-fire control for the caption test: each cut sentence trips a matcher the test applies."""
    lowered = text.lower()
    assert _RETIRED_TERM_PHRASES.search(text) is not None or "bundle" in lowered or "re-run" in lowered


# ---- 3b.5: the P16 Combined health roll-up ---------------------------------------------------


def _p16_derived_finding(tmp_path: Path) -> dict:
    from hhemt.report_renderers.cross_experiment_errors_and_warnings import _swmm_invisible_divergence_finding

    pairs = [{"model": "TRITON-SWMM", "variable": "max_wlevel_m", "identical": False} for _ in range(3)]
    pairs += [{"model": "TRITON-SWMM", "variable": "max_flow_cms", "identical": True} for _ in range(3)]
    (tmp_path / "combined_intercomparison.json").write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    finding = _swmm_invisible_divergence_finding(tmp_path)
    assert finding is not None
    return finding


def test_the_derived_finding_names_the_resume_check_by_the_name_the_check_column_prints(tmp_path: Path) -> None:
    """The derived row points a reader at a check by the display name its own page prints.

    Input class: the clean-vs-resume divergence finding. Required because 3b.3 reworded the
    vocabulary and the Check column prints the display name of `resume validity`, so a pointer in
    the raw key would name a check no page shows. Killed by restoring "'resume validity'" in the
    summary. A second correct implementation words the pointer differently while it carries the
    display name.
    """
    from hhemt.report_renderers.cross_experiment_errors_and_warnings import _vocab_name

    display = _vocab_name("resume validity")
    summary = _p16_derived_finding(tmp_path)["summary"]

    assert display != "resume validity"
    assert display in summary
    assert "'resume validity'" not in summary


def test_the_derived_finding_and_its_row_carry_no_em_dash_capital_emphasis_or_retired_phrase(tmp_path: Path) -> None:
    """The derived row, as the page prints it, follows brief rules 2, 5 and 8.

    Input class: the divergence finding rendered as a matrix row and as the standalone block.
    Killed by the entity `&mdash;` the row printed before, by "SCATTER" or "NOT" as capital
    emphasis, or by a retired term-table phrase. A second correct implementation joins the name and
    the summary with any other punctuation.
    """
    from hhemt.report_renderers.cross_experiment_errors_and_warnings import _derived_block_html, _render_rollup_html

    finding = _p16_derived_finding(tmp_path)
    pages = [
        _render_rollup_html([("exp_a", [{"name": "x", "passed": True}])], derived=[finding]),
        _derived_block_html(finding),
    ]

    for page in pages:
        assert "(derived across experiments): " in page
        assert "&mdash;" not in page and "—" not in page
        assert "SCATTER" not in page and " NOT " not in page
        assert _RETIRED_TERM_PHRASES.search(page) is None


@pytest.mark.parametrize(
    "text", ["(derived across experiments) &mdash; x", "the node-depth SCATTER defect", "is NOT evidence"]
)
def test_the_derived_row_check_fires_on_the_old_wording(text: str) -> None:
    """Must-fire control for the test above: each old form trips one of its assertions."""
    assert "&mdash;" in text or "SCATTER" in text or " NOT " in text


def test_the_empty_roll_up_notice_is_one_not_shown_sentence_naming_the_file_in_code() -> None:
    """With no experiment report the page prints one "{Element} is not shown: {reason}." sentence.

    Input class: a roll-up with no child reports. Required because brief rules 6 and 7 set the form
    and the code formatting of a file name, and rule 2 retires "crate" (the old notice named
    `child_crates/`). Killed by the old notice. A second correct implementation words the reason
    differently and passes.
    """
    from hhemt.report_renderers.cross_experiment_errors_and_warnings import _render_rollup_html

    page = _render_rollup_html([])
    notice = re.search(r"<p class='note'>(.*?)</p>", page, re.S)

    assert notice is not None
    assert _NOTICE_FORM.match(notice.group(1))
    assert "<code>validation_report.json</code>" in notice.group(1)
    assert "child_crates" not in page


def test_the_unaccounted_divergence_finding_states_its_condition_and_counts_with_the_plural_it_takes() -> None:
    """The compatibility finding has no hedge and no "divergence(s)" (brief rules 3 and 4).

    Input class: one unexpected divergence out of two. Killed by the old closing sentence
    ("Figures render, but ... may differ") or by "divergence(s)". The per-field reason text comes from another module
    (`_divergence_message`) and is not read here.
    A second correct implementation words the sentence differently and passes.
    """
    import tempfile

    from hhemt.report_renderers import cross_experiment_errors_and_warnings as mod

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        divs = [
            {
                "field_name": "schemaVersion",
                "bucket": "layout",
                "bundle_a": "a",
                "bundle_b": "b",
                "severity": "warning",
            },
            {"field_name": "x", "bucket": "hpc", "bundle_a": "a", "bundle_b": "b", "severity": "info"},
        ]
        (root / "combined_compatibility.json").write_text(json.dumps({"divergences": divs}), encoding="utf-8")
        finding = mod._compat_divergence_finding(root)

    assert finding is not None
    assert "1 of 2 identity-field divergences are not accounted for" in finding["summary"]
    assert "Figures render, but" not in finding["summary"] and "(s)" not in finding["summary"]


def test_the_combined_health_caption_cuts_capital_emphasis_and_the_pointer_sentence() -> None:
    """The caption keeps the roll-up definition, the hover and the derived rows, in lowercase words.

    Input class: the `cross_experiment_errors_and_warnings.rst` source. Killed by restoring "hover a
    FAIL" or the closing "See the experiment's own" sentence (shape P16 row: capitals cut; the page's
    own paragraph points to the experiment's section). A second correct implementation words the three
    kept sentences differently and passes.
    """
    from tests.test_caption_page_kinds import CAPTIONS

    text = (CAPTIONS / "cross_experiment_errors_and_warnings.rst").read_text("utf-8")

    assert "FAIL" not in text
    assert "See the experiment" not in text
    assert "derived across experiments" in text and "hover" in text


_EMPTY_RECORD_REASON = re.compile(r"missing|no checks?\b|empty|holds no\b|does not hold", re.IGNORECASE)


def test_the_empty_roll_up_notice_does_not_claim_a_file_is_absent_when_the_record_holds_no_check() -> None:
    """The notice states a condition true of every input that reaches it, including a record that holds no check.

    Input class: experiments that are listed but carry no check, which is what the renderer builds when
    an experiment's `validation_report.json` is absent, unreadable or holds an empty list of checks
    (`child_reports.append((child.name, checks))` with `checks = []`). Required because "no experiment
    holds a validation_report.json" is false for the last two. Anchored on the reason naming the
    missing-or-empty condition, not on exact wording. Killed by the notice that says only that no
    experiment holds the file. A second correct implementation says "is missing or holds no checks"
    and passes.
    """
    from hhemt.report_renderers.cross_experiment_errors_and_warnings import _render_rollup_html

    page = _render_rollup_html([("exp_a", [])])
    notice = re.search(r"<p class='note'>(.*?)</p>", page, re.S)

    assert notice is not None
    reason = notice.group(1).split(": ", 1)[1]
    assert _EMPTY_RECORD_REASON.search(reason)


@pytest.mark.parametrize(
    ("reason", "fires"),
    [
        ("no experiment holds a <code>validation_report.json</code>.", False),
        ("<code>validation_report.json</code> is missing or holds no checks in every experiment.", True),
        ("every <code>validation_report.json</code> is empty.", True),
    ],
)
def test_the_empty_record_reason_matcher_separates_the_absence_only_claim(reason: str, fires: bool) -> None:
    """Must-fire control for the test above: the matcher rejects the absence-only wording, accepts two true ones."""
    assert (_EMPTY_RECORD_REASON.search(reason) is not None) is fires
