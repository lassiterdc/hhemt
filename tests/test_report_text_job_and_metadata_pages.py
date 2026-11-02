"""Reader-facing text of the Workflow Performance page and the Metadata page.

This file is the reviewer-test file of the package that rewrites that text. It holds, per
page, the caption page-kind check (the harness of ``tests/test_caption_page_kinds.py``), the
corrections that tie a sentence of the page to the code fact it states, and their must-fire
controls. No test here compiles or runs a solver.

Workflow Performance page (job-count text):
  * the caption's bold names are printed by every page kind it is attached to;
  * the summary sentence counts jobs with the plural of the number it holds;
  * the notice for a missing efficiency report names the file the renderer reads;
  * an unrecognised report column is logged and not printed on the page;
  * no retired word is printed on the page.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from tests.test_caption_page_kinds import CAPTIONS, bold_names, missing_names, page_text
from tests.test_report_text_style import RETIRED
from tests.test_workflow_performance_job_set import (
    _EFF_HEADER,
    RUN_ID_FIRST,
    START_NEVER_STARTED,
    STATE_NEVER_STARTED,
    _render_sweep_page,
    _sweep_analysis,
    _write_log,
    _write_plugin_report,
    _write_recovery,
    _write_sidecars,
)

_P10_CAPTION = (CAPTIONS / "workflow_performance.rst").read_text(encoding="utf-8")

_SIDECARS = {
    "a_setup.flag.json": {"rule_name": "setup_target_0", "slurm_job_id": "101"},
    "b_sim_1.flag.json": {"rule_name": "simulation_member_1_evt_0", "sa_id": "1", "slurm_job_id": "102"},
}


def _write_job_records(analysis_dir: Path) -> None:
    """Two submitted jobs: 101 is in a plugin report, 102 was cancelled before it started."""
    _write_log(
        analysis_dir,
        "2026-11-01T015232.143811.snakemake.log",
        [
            f"SLURM run ID: {RUN_ID_FIRST}",
            "Job 1 has been submitted with SLURM jobid 101 (log: /x/101.log).",
            "Job 2 has been submitted with SLURM jobid 102 (log: /x/102.log).",
        ],
    )
    _write_plugin_report(analysis_dir, ["101"])
    _write_recovery(analysis_dir, {"102": (STATE_NEVER_STARTED, START_NEVER_STARTED)})


def _p10_page(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Render the page under one page kind; return (html, page text)."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    analysis_dir = tmp_path / kind
    analysis_dir.mkdir()
    if kind != "empty":
        _write_sidecars(analysis_dir, _SIDECARS)
        _write_job_records(analysis_dir)
    analysis = _sweep_analysis(analysis_dir, ["1", "2"])
    if kind == "plain":
        analysis.cfg_analysis.toggle_sensitivity_analysis = False
    html = _render_sweep_page(analysis_dir, analysis)
    return html, page_text(html)


#: The page kinds the caption is attached to: a sweep with records, a plain analysis with
#: records, and an analysis that has not written a record yet (the page the workflow produces).
_P10_KINDS = ["sweep", "plain", "empty"]


@pytest.mark.parametrize("kind", _P10_KINDS)
def test_the_workflow_performance_caption_names_only_elements_every_page_kind_prints(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Class: every page kind the caption is attached to. Each bold name of the caption must
    be printed by the rendering of each kind (brief rule 6 and 9)."""
    _html, text = _p10_page(kind, tmp_path, monkeypatch)

    assert bold_names(_P10_CAPTION) != []
    assert missing_names(_P10_CAPTION, text) == []


def test_a_caption_that_bolds_the_cut_toolkit_record_column_fails_on_the_live_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUST-FIRE CONTROL for the page-kind check on this page: the column the page no longer
    prints, set in bold in a caption, is reported missing on a kind that renders the table."""
    _html, text = _p10_page("sweep", tmp_path, monkeypatch)

    assert missing_names("**Toolkit record** names the jobs with a record.", text) == ["Toolkit record"]
    # The control is not vacuous: a column the page does print passes.
    assert missing_names("**Workflow run** names the submitting run.", text) == []


def test_the_slurm_efficiency_bullet_names_no_source_file_the_page_does_not_print() -> None:
    """The bullet states what the section lists. It carries no list of the files the toolkit
    read, which a job reader cannot act on."""
    bullet = _P10_CAPTION[_P10_CAPTION.index("- **SLURM Efficiency**") :]

    assert "job_level_recovery" not in bullet
    assert "slurm_efficiency_report" not in bullet
    assert "Snakemake logs" not in bullet


@pytest.mark.parametrize("kind", _P10_KINDS)
def test_the_page_prints_no_retired_word(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Class: every page kind. A word the term table retires does not appear in a visible
    string or a cell tooltip of the page (brief rule 2)."""
    html, text = _p10_page(kind, tmp_path, monkeypatch)
    tooltips = " ".join(re.findall(r'title=\\?"([^"\\]*)', html))

    assert RETIRED.findall(text) == []
    assert RETIRED.findall(tooltips) == []


def test_the_jobs_are_counted_with_the_plural_of_their_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: ``job(s)`` for every number, or the
    singular form for the plural. The sentence counts what the table holds, so the page with
    two jobs and the page with one job print different forms.

    Input class: two jobs (one cancelled before it started) and one job that ran.
    """
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    two = tmp_path / "two"
    two.mkdir()
    _write_job_records(two)
    html_two = _render_sweep_page(two, _sweep_analysis(two, ["1"]))

    one = tmp_path / "one"
    one.mkdir()
    _write_log(one, "2026-11-01T015232.143811.snakemake.log", [f"SLURM run ID: {RUN_ID_FIRST}"])
    _write_plugin_report(one, ["101"])
    html_one = _render_sweep_page(one, _sweep_analysis(one, ["1"]))

    assert "2 jobs submitted, 1 job cancelled before starting, 1 job ran." in html_two
    assert "1 job submitted, 0 jobs cancelled before starting, 1 job ran." in html_one
    for html in (html_two, html_one):
        assert "job(s)" not in html


def test_the_missing_report_notice_names_the_file_the_renderer_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. The notice names the file the reader would open, and the name is the
    glob the renderer resolves reports with, so a renamed report moves the notice.

    Input class: an analysis with no report and no submit record. Plausible wrong
    implementation: the file name is typed into the notice, so a renamed report leaves a notice
    naming a file the toolkit no longer reads.
    """
    from hhemt.report_renderers import metadata, workflow_performance

    html, _text = _p10_page("empty", tmp_path, monkeypatch)
    notice = f"SLURM Efficiency is not shown: no <code>{metadata._SLURM_EFF_GLOB}</code> record was found."
    assert notice in html

    # Must-fire: change the glob the renderer reads and the notice follows it.
    monkeypatch.setattr(workflow_performance, "_SLURM_EFF_GLOB", "renamed_report_*.csv")
    other = tmp_path / "other"
    other.mkdir()
    moved = _render_sweep_page(other, _sweep_analysis(other, ["1"]))
    assert "<code>renamed_report_*.csv</code>" in moved
    assert metadata._SLURM_EFF_GLOB not in moved


def test_an_unrecognised_report_column_is_logged_and_not_printed(caplog: pytest.LogCaptureFixture) -> None:
    """NEW CAPABILITY. A reader cannot act on a column the plugin newly reports, so the page
    does not print the column name; the maintainer who can add the column reads the log.

    Plausible wrong implementation: the note stays on the page, or the column is dropped
    without a log line. Must-fire control: a report with only known columns logs nothing.
    """
    from hhemt.report_renderers import metadata

    with_new = _EFF_HEADER.rstrip("\n") + ",GPUUtilPct\n"
    with_new += "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0,87.5\n"
    known = _EFF_HEADER + "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n"

    with caplog.at_level(logging.WARNING, logger=metadata.__name__):
        html = metadata._build_slurm_efficiency_html([("r", with_new)], {}, {}, lambda p: "").html
    logged = [r.getMessage() for r in caplog.records if r.name == metadata.__name__]
    assert "GPUUtilPct" not in html
    assert "Unrecognised column" not in html
    assert len(logged) == 1 and "GPUUtilPct" in logged[0]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=metadata.__name__):
        metadata._build_slurm_efficiency_html([("r", known)], {}, {}, lambda p: "")
    assert [r for r in caplog.records if r.name == metadata.__name__] == []


def test_a_report_with_no_job_rows_prints_a_sentence_instead_of_an_empty_table() -> None:
    """NEW CAPABILITY. The caption says the section states why when no job is listed. Two
    branches list no job: no report and no submit record (the notice tested above), and a
    report whose file holds a header and no row.

    Input class: a report with only its header line. Plausible wrong implementation: the
    builder returns an empty table with no sentence. Must-fire control: the same builder given
    one job row prints the table and not the sentence.
    """
    from hhemt.report_renderers import metadata

    empty = metadata._build_slurm_efficiency_html([("r", _EFF_HEADER)], {}, {}, lambda p: "")
    one = metadata._build_slurm_efficiency_html(
        [("r", _EFF_HEADER + "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n")],
        {},
        {},
        lambda p: "",
    )

    assert "The SLURM resource-efficiency reports contain no job rows." in empty.html
    assert empty.fragment is None
    assert "contain no job rows" not in one.html
    assert one.fragment is not None


def test_the_accounting_record_count_agrees_with_its_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: ``have`` for every number, so the page
    with one job reads "1 job have an accounting record".

    Input class: a page whose one job has no accounting record, and a page of two jobs of which
    one has a record.
    """
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    one = tmp_path / "one"
    one.mkdir()
    _write_log(one, "2026-11-01T015232.143811.snakemake.log", [f"SLURM run ID: {RUN_ID_FIRST}"])
    _write_plugin_report(one, ["101"])
    html_one = _render_sweep_page(one, _sweep_analysis(one, ["1"]))
    two = tmp_path / "two"
    two.mkdir()
    _write_job_records(two)
    html_two = _render_sweep_page(two, _sweep_analysis(two, ["1"]))

    assert "0 jobs have an accounting record" in html_one
    assert "1 job has an accounting record" in html_two


def test_the_column_glossary_names_no_function() -> None:
    """NEW CAPABILITY. A function name is not text for a reader (brief rule 7). Plausible wrong
    implementation: the glossary derives a "Method" line from each reducer's name, as it did
    before this change. Must-fire: the control string is what that line looked like."""
    from hhemt.report_renderers import metadata

    caption = metadata._reduction_caption()
    function_call = re.compile(r"\b_?[a-z]+(?:_[a-z]+)+\(\)")

    assert function_call.findall(caption) == []
    assert "Method." not in caption
    assert function_call.findall("Method. _reduce_cpu_sum() sums the steps.") == ["_reduce_cpu_sum()"]


def test_the_column_glossary_carries_no_frequency_hedge() -> None:
    """NEW CAPABILITY. Brief rule 3: a statement true only under a condition states the
    condition. "Often" hedges how frequently the peak-memory step is the runner wrapper, a
    frequency measured on one job. The style test's hedge list does not name it, so this test
    pins the glossary the reader hovers.

    Must-fire: the control string is the sentence this glossary printed before the repair.
    """
    from hhemt.report_renderers import metadata

    frequency = re.compile(r"\b(?:often|sometimes|frequently|commonly|occasionally)\b", re.I)

    assert frequency.findall(re.sub(r"<[^>]+>", " ", metadata._reduction_caption())) == []
    assert frequency.findall("because the peak is often the runner wrapper") == ["often"]


# ---------------------------------------------------------------------------------------
# Metadata page (P09).
# ---------------------------------------------------------------------------------------
_P09_CAPTION = (CAPTIONS / "metadata.rst").read_text(encoding="utf-8")

_ROOT = {
    "@id": "./",
    "@type": "Dataset",
    "name": "case",
    "analysis_id": "A1",
    "system_id": "S1",
    "license": {"@id": "https://spdx.org/licenses/CC0-1.0"},
}
_DESCRIPTOR = {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"}
_APP = {"@id": "#hhemt-app", "@type": "SoftwareApplication", "name": "hhemt", "softwareVersion": "abc123def"}
_SIF = {"@id": "#sif", "@type": "SoftwareApplication", "name": "container", "sha256": "deadbeefcafe"}
_INPUT = {"@id": "inputs/dem.tif", "@type": "File", "sha256": "f00dfeed", "contentSize": "1024"}
_RUN = {
    "@id": "#run-evt0",
    "@type": "CreateAction",
    "name": "run evt0",
    "instrument": [{"@id": "#hhemt-app"}],
    "object": [{"@id": "inputs/dem.tif"}],
    "result": [{"@id": "analysis_datatree.zarr"}],
}


def _crate(*entities: dict) -> dict:
    return {"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": list(entities)}


def _p09_page(tmp_path: Path, kind: str, *, validation: bool = False) -> tuple[str, str]:
    """Render the Metadata page under one page kind; return (html, page text)."""
    import json
    import types

    from hhemt.config.report import report_config
    from hhemt.report_renderers import metadata

    analysis_dir = tmp_path / f"p09_{kind}_{int(validation)}"
    analysis_dir.mkdir()
    doc = {
        "full": _crate(_ROOT, _DESCRIPTOR, _APP, _SIF, _INPUT, _RUN),
        "sweep": _crate(_ROOT, _DESCRIPTOR, _APP),
        "absent": None,
    }[kind]
    if doc is not None:
        (analysis_dir / metadata._SIDECAR_FILENAME).write_text(json.dumps(doc))
    if validation:
        check = {"name": "Data availability", "passed": True, "summary": "all reclaims recorded", "details": []}
        (analysis_dir / metadata._VALIDATION_REPORT_FILENAME).write_text(json.dumps({"checks": [check]}))
    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1"),
    )
    out = analysis_dir / "plots" / "metadata.html"
    metadata.render(analysis, report_config(), out)
    html = out.read_text(encoding="utf-8")
    return html, page_text(html)


_P09_KINDS = ["full", "sweep", "absent"]


@pytest.mark.parametrize("kind", _P09_KINDS)
def test_the_metadata_caption_names_only_elements_every_page_kind_prints(kind: str, tmp_path: Path) -> None:
    """Class: every page kind the caption is attached to (a container run with model runs, a
    sweep crate with none, and an analysis with no provenance record). Each bold name of the
    caption must be printed by each rendering (brief rules 6 and 9)."""
    _html, text = _p09_page(tmp_path, kind)

    assert bold_names(_P09_CAPTION) != []
    assert missing_names(_P09_CAPTION, text) == []


def test_a_caption_that_bolds_the_old_provenance_summary_name_fails_on_every_kind(tmp_path: Path) -> None:
    """MUST-FIRE CONTROL: the caption used to bold `Provenance summary`, a name the page never
    printed (its heading is `Provenance`). The check reports it on a kind that has the section."""
    _html, text = _p09_page(tmp_path, "full")

    assert missing_names("**Provenance summary**: the record.", text) == ["Provenance summary"]
    assert missing_names("**Provenance**: the record.", text) == []


def test_the_missing_provenance_record_is_one_sentence_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. The omission notice has the form "{Element} is not shown: {reason}." and
    names the file the toolkit reads, taken from the code's constant.

    Plausible wrong implementation: the file name is typed into the notice, or the notice runs
    on into a second sentence. Must-fire: change the constant and the notice follows.
    """
    from hhemt.report_renderers import metadata

    html, _text = _p09_page(tmp_path, "absent")
    notice = f"Provenance is not shown: no <code>{metadata._SIDECAR_FILENAME}</code> record was found."
    assert notice in html

    monkeypatch.setattr(metadata, "_SIDECAR_FILENAME", "renamed-record.json")
    moved_root = tmp_path / "moved"
    moved_root.mkdir()
    moved, _ = _p09_page(moved_root, "absent")
    assert "<code>renamed-record.json</code>" in moved


def test_the_missing_validation_record_is_one_sentence_naming_the_file(tmp_path: Path) -> None:
    """NEW CAPABILITY. Data Availability reads the validation record; with none the section
    prints the one-sentence notice naming it. Control: with the record present the section
    prints its status and no notice."""
    from hhemt.report_renderers import metadata

    absent, _ = _p09_page(tmp_path, "full")
    present, _ = _p09_page(tmp_path, "full", validation=True)

    assert (
        f"Data Availability is not shown: <code>{metadata._VALIDATION_REPORT_FILENAME}</code> is not available."
        in absent
    )
    assert "Data Availability is not shown" not in present
    assert "reclaim recorded and consistent" in present


def test_the_process_section_carries_no_hand_count_and_no_chain_note(tmp_path: Path) -> None:
    """NEW CAPABILITY. The table shows its own rows, so a count line above it states nothing a
    reader acts on, and the note under the provenance chain heading restated the row labels.

    Must-fire: the same page still prints the run row and the chain rows.
    """
    html, text = _p09_page(tmp_path, "full")

    assert "run unit(s) recorded" not in html
    assert "can differ from the others" not in html
    assert "run evt0" in text
    assert "Consolidated by" in text and "Rendered by" in text


def test_the_no_model_runs_sentence_is_printed_only_when_the_crate_lists_none(tmp_path: Path) -> None:
    """NEW CAPABILITY. The caption lists model runs only for an analysis that is not a sweep, so
    the page states their absence on a sweep crate and nowhere else (brief rule 6)."""
    sentence = "Model runs are not listed here: this is a sweep, and each member records its own."
    sweep, _ = _p09_page(tmp_path, "sweep")
    full, _ = _p09_page(tmp_path, "full")

    assert sentence in sweep
    assert sentence not in full


@pytest.mark.parametrize("kind", _P09_KINDS)
def test_the_metadata_page_prints_no_retired_word_hedge_or_mechanism_word(kind: str, tmp_path: Path) -> None:
    """Class: every page kind. The visible text outside the data grids carries no retired word,
    hedge, `which` or number word (brief rules 2, 3, 4 and 8; file names in code formatting
    excepted) and none of the mechanism words the earlier text used."""
    import html as html_lib

    from tests.test_report_text_style import PROSE_RULES

    html, _text = _p09_page(tmp_path, kind, validation=True)
    # A file name in code formatting is the thing the reader opens, so it is not scanned.
    prose = re.sub(r"(?is)<(script|style|code)\b.*?</\1\s*>", " ", html)
    visible = html_lib.unescape(re.sub(r"<[^>]+>", " ", prose))

    assert RETIRED.findall(visible) == []
    for rule in ("hedge", "which", "number_word", "filler_adverb", "contraction"):
        assert PROSE_RULES[rule].findall(visible) == [], rule
    for word in ("deliberately", "sidecar", "crate", "drift guard", "Membership is always decided"):
        assert word not in visible, word


def test_a_variable_without_a_descriptor_shows_a_dash_in_the_three_descriptor_columns(tmp_path: Path) -> None:
    """NEW CAPABILITY. The data dictionary note says a dash in Spatial representation, Source
    variables and Operation means the toolkit holds no descriptor. Code fact: a variable name
    the quantity-provenance table does not hold renders those three cells as a dash.

    Must-fire: the note names exactly those three columns, and the page prints them as headers.
    """
    from hhemt.report_renderers import metadata

    var = {"@id": "#var-zz_unknown_quantity", "@type": "PropertyValue", "name": "zz_unknown_quantity", "unitText": "m"}
    zarr = {
        "@id": "analysis_datatree.zarr",
        "@type": "Dataset",
        "name": "tree",
        "variableMeasured": [{"@id": "#var-zz_unknown_quantity"}],
    }
    doc = _crate(_ROOT, _DESCRIPTOR, _APP, var, zarr)
    html = metadata._build_provenance_html(doc, analysis_dir=tmp_path)
    note = html[html.index("Data dictionary.") :].split("</p>", 1)[0]

    for column in ("Spatial representation", "Source variables", "Operation"):
        assert f"<em>{column}</em>" in note
        assert column in html
    row = html[html.index("<code>zz_unknown_quantity</code>") :].split("</tr>", 1)[0]
    cells = re.findall(r"<td>(.*?)</td>", row)
    # Columns after the name: Description, Units, Property ID, then the three descriptor columns.
    assert cells[-3:] == ["—", "—", "—"]


@pytest.mark.parametrize("kind", _P09_KINDS)
def test_the_metadata_page_sets_no_short_word_in_capitals_for_emphasis(kind: str, tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer, brief rule 8). The style test's capitals rule matches four
    letters or more, so a three-letter emphasis word passes it; `Do NOT change these.` is such a
    word and bold carries emphasis. Class: every page kind. Acronyms the page names are listed.

    Must-fire: the control string is the sentence the page printed before this repair.
    """
    import html as html_lib

    html, _text = _p09_page(tmp_path, kind, validation=True)
    prose = re.sub(r"(?is)<(script|style|code)\b.*?</\1\s*>", " ", html)
    visible = html_lib.unescape(re.sub(r"<[^>]+>", " ", prose))
    acronyms = {"CF", "HPC", "ID", "SHA", "SIF", "YAML", "SLURM", "CPU", "GPU", "CSV", "JSON", "HTML", "MPI", "DEM"}
    short_caps = re.compile(r"\b[A-Z]{2,}\b")

    assert [w for w in short_caps.findall(visible) if w not in acronyms] == []
    assert [w for w in short_caps.findall("Do NOT change these.") if w not in acronyms] == ["NOT"]


def test_the_collapsed_runs_summary_counts_member_names_with_the_plural_of_the_number() -> None:
    """NEW CAPABILITY (reviewer, brief rule 4). Plausible wrong implementation: ``member
    name(s)`` for every number. Input class: a run of 28 collapsed siblings and a run of exactly 3,
    the smallest run the page collapses (``_TREE_SENTINEL_MIN_SIBLINGS``; two stay expanded and
    print no summary, so the singular form is unreachable).
    Must-fire: the page-level prose carries a numeral beside its unit in the number's own form.
    """
    from hhemt.report_renderers import metadata

    many = metadata._path_tree_html(
        [f"gpu_{n}_r1/analysis_datatree.zarr/a" for n in range(28)] + [f"gpu_{n}_r1/b.json" for n in range(28)]
    )
    three = metadata._path_tree_html(
        [f"gpu_{n}_r1/analysis_datatree.zarr/a" for n in range(3)] + [f"gpu_{n}_r1/b.json" for n in range(3)]
    )

    assert "member name(s)" not in many + three
    assert "show the 28 member names" in many
    assert "show the 3 member names" in three


def test_the_metadata_page_makes_no_frequency_claim(tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer, brief rule 3; the class found on the Workflow Performance page).
    The style test's hedge list names no frequency word. Must-fire: the control string."""
    import html as html_lib

    frequency = re.compile(r"\b(?:often|sometimes|frequently|commonly|occasionally)\b", re.I)
    for kind in _P09_KINDS:
        html, _text = _p09_page(tmp_path, kind, validation=True)
        prose = re.sub(r"(?is)<(script|style|code)\b.*?</\1\s*>", " ", html)
        assert frequency.findall(html_lib.unescape(re.sub(r"<[^>]+>", " ", prose))) == [], kind
    assert frequency.findall("a varied setting is often reproduced") == ["often"]


# ---------------------------------------------------------------------------------------
# Validation check text (analysis_validation.py).
# ---------------------------------------------------------------------------------------
_EMPHASIS_WORDS = re.compile(r"\b(?:NOT|NO|ALL|ANY|WITH|AND|ONLY|NEVER|MUST|NEITHER|NOR|BEFORE|AFTER)\b")


def _validation_strings() -> list[str]:
    import ast

    from tests.test_report_text_style import PKG, python_strings

    return python_strings(ast.parse((PKG / "analysis_validation.py").read_text(encoding="utf-8")))


def test_no_validation_check_string_uses_a_capitalised_short_word_for_emphasis() -> None:
    """NEW CAPABILITY (brief rule 8, the class the style test's four-letter floor cannot see).
    Bold carries emphasis; a reader-facing check string spells `not`, `no`, `with` in lower case.

    Must-fire: the same matcher reports a planted string.
    """
    hits = [(s[:80], m) for s in _validation_strings() for m in _EMPHASIS_WORDS.findall(s)]

    assert hits == []
    assert _EMPHASIS_WORDS.findall("the arm ran WITH a resume and NOT without") == ["WITH", "NOT"]


def test_a_pre_fix_coupled_resume_summary_and_detail_state_the_same_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. The summary used to say the summaries were "likely invalid" while the
    detail said they were "systematically low". Both now name the measured effect, so a reader
    of the page gets one claim, not a hedge beside a fact.

    Input class: a coupled arm resumed at a pin before the replay fix. Must-fire: an arm that
    never resumed prints neither sentence.
    """
    import hhemt.analysis_validation as av
    from tests.test_coupled_resume_validity import _SHA_PRE_REPLAY, _analysis_stub, _resumed_df

    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_REPLAY)
    res = av.check_coupled_resume_validity(_analysis_stub(df=_resumed_df()))

    assert "systematically low" in res.summary
    assert "systematically low" in res.details[0]["detail"]
    assert "likely" not in res.summary
    assert "is systematically low" not in av.check_coupled_resume_validity(_analysis_stub(df=None)).summary


def test_the_system_setup_failure_names_dimensions_not_a_shape() -> None:
    """NEW CAPABILITY. The detail names the array dimensions a reader compares (planning word
    `shape` retired). Code fact: `dem.shape` is what is printed. Must-fire: a mismatched pair
    prints both tuples."""
    import ast

    from tests.test_report_text_style import PKG

    source = (PKG / "analysis_validation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    texts = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    mismatch = [t for t in texts if "dem.shape" in t and "manning.shape" in t]

    assert len(mismatch) == 1 and "dimensions" in mismatch[0]
    assert "DEM shape" not in source


def _validation_reader_strings() -> list[str]:
    """Reader-facing strings of `analysis_validation.py`, including the short f-string pieces
    (`{n} PRESENT`) that the style test's three-word floor skips."""
    import ast

    from tests.test_report_text_style import PKG, python_strings

    return python_strings(ast.parse((PKG / "analysis_validation.py").read_text(encoding="utf-8")))


_VALIDATION_ACRONYMS = {
    "TRITON",
    "SWMM",
    "GPU",
    "CPU",
    "MPI",
    "DEM",
    "EDA",
    "OK",
    "US",
    "TRITONSWMM",
    "OMP",
    "MH",
    "RIM",
}


def test_the_validation_summaries_set_no_word_in_capitals_for_emphasis() -> None:
    """NEW CAPABILITY (reviewer, brief rule 8). The style test reads a string of fewer than three
    words only for retired words, so a status word such as `PRESENT` in the f-string piece
    `{n} PRESENT` escapes its capitals rule even at seven letters. Class: every reader-facing
    string of the module. Must-fire: the control string is what the summary printed."""
    caps = re.compile(r"(?<![A-Za-z_/.\-])\b[A-Z]{2,}\b(?![_A-Za-z])")

    found = [w for s in _validation_reader_strings() for w in caps.findall(s) if w not in _VALIDATION_ACRONYMS]

    assert found == []
    assert [w for w in caps.findall("{} PRESENT") if w not in _VALIDATION_ACRONYMS] == ["PRESENT"]


def test_the_validation_summaries_write_no_hand_plural_and_no_double_hyphen_dash() -> None:
    """NEW CAPABILITY (reviewer, brief rules 4 and 8). A count the page computes is written with
    the plural its number takes, and `issue(s)` is not written; a ` -- ` dash is the em dash by
    another spelling. Class: every reader-facing string of the module. Must-fire: both patterns
    match the control strings."""
    hand_plural = re.compile(r"\w\(s\)")
    dash = re.compile(r" -- |^-- | --$")
    strings = _validation_reader_strings()

    assert [s[:60] for s in strings if hand_plural.search(s)] == []
    assert [s[:60] for s in strings if dash.search(s)] == []
    assert hand_plural.search("{} issue(s)") and dash.search("not a failure -- an analysis")


# ---- 3g.3 reviewer test: the count helper's two forms ------------------------------------------


@pytest.mark.parametrize(("n", "expected"), [(1, "1 stage"), (0, "0 stages"), (2, "2 stages"), (7, "7 stages")])
def test_the_count_helper_prints_the_numeral_with_the_form_its_number_takes(n: int, expected: str) -> None:
    """`_count` returns "1 {noun}" at one and "{n} {noun}s" at zero, two and many.

    Input class: the counts 0, 1, 2 and 7. Required because every validation summary now takes its
    plural from this helper and no hand-written "(s)" remains; a pin on the plural form alone cannot
    see a wrong singular. Killed by an always-plural helper (fails at 1) and an always-singular helper
    (fails at 0, 2, 7). A second correct implementation builds the string any other way.
    """
    from hhemt.analysis_validation import _count

    assert _count(n, "stage") == expected


def test_a_summary_at_one_known_resume_defect_prints_the_singular() -> None:
    """The pre-fix failure summary at a build with exactly one defect prints "1 known resume defect".

    Input class: the coupled-resume check run on a stub analysis at a build carrying one registered
    defect (the three-defect pin prints the plural and cannot see the singular). Anchored on the
    numeral and the noun followed by the colon the summary prints, so "1 known resume defects" fails.
    """
    import pandas as pd

    from hhemt import analysis_validation as av
    from hhemt.model_defects import SHA_DEPTH_SCATTER_FIX

    class _Cfg:
        toggle_tritonswmm_model = True

    class _Sys:
        cfg_system = _Cfg()

    class _Analysis:
        _system = _Sys()

    original = av._read_triton_provenance
    av._read_triton_provenance = lambda _a: SHA_DEPTH_SCATTER_FIX
    try:
        df = pd.DataFrame({"model_type": ["tritonswmm"], "n_resumes": [1], "event_iloc": [0]})
        result = av.check_known_resume_defects(_Analysis(), df_status=df)
    finally:
        av._read_triton_provenance = original

    assert result.passed is False
    assert "carrying 1 known resume defect:" in result.summary
