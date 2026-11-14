"""Submit-line parser and recovery-roster widening for the Workflow Performance job set.

The submit lines are the only source that holds every submitted job, including jobs
cancelled while pending that no accounting store or plugin report ever received a row for.
This file covers the parser and the recovery roster.

The two log line forms below are copied VERBATIM from a real compute-sweep log,
`2026-11-01T015232.143811.snakemake.log` (line 3 and line 87). Only the path inside the
trailing `(log: ...)` is shortened, because the assertion never reads it.

Test classes: every test here is NEW CAPABILITY -- the function
and the widened roster do not exist on the base tree, so each is red there by construction
(ImportError / missing ids); the plausible wrong implementation is named per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt import slurm_job_recovery as sjr

# Verbatim forms from the real 015232 log.
RUN_ID_LINE_A = "SLURM run ID: 9e62392e-a7d2-476a-a066-cc6fb19708f1"
SUBMIT_LINE_52 = "Job 52 has been submitted with SLURM jobid 22240735 (log: /x/slurm_logs/22240735.log)."

RUN_ID_B = "0c1d2e3f-4a5b-4c6d-8e7f-90a1b2c3d4e5"


def _write_log(analysis_dir: Path, name: str, lines: list[str]) -> Path:
    log_dir = analysis_dir / ".snakemake" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def two_invocations(tmp_path: Path) -> Path:
    """Two logs with different run ids, plus a third log carrying a jobid under no run id."""
    # Invocation A: the run id sits on line 3, as in the real log; submit lines follow.
    _write_log(
        tmp_path,
        "2026-11-01T015232.143811.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            "host: login-node",
            RUN_ID_LINE_A,
            "Building DAG of jobs...",
            SUBMIT_LINE_52,
            "Job 53 has been submitted with SLURM jobid 22240736 (log: /x/.snakemake/slurm_logs/rule_a/22240736.log).",
            # The same job id repeated by a later status line must not duplicate in the id list.
            "Job 52 has been submitted with SLURM jobid 22240735 (log: /x/.snakemake/slurm_logs/rule_a/22240735.log).",
        ],
    )
    # Invocation B: later in time, different run id, a job id numerically SMALLER than A's
    # largest and LARGER than its smallest, so a lexicographic sort would misorder it.
    _write_log(
        tmp_path,
        "2026-11-01T024124.511220.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            f"SLURM run ID: {RUN_ID_B}",
            "Job 7 has been submitted with SLURM jobid 9999999 (log: /x/.snakemake/slurm_logs/rule_b/9999999.log).",
            "Job 8 has been submitted with SLURM jobid 22240800 (log: /x/.snakemake/slurm_logs/rule_b/22240800.log).",
        ],
    )
    # A log carrying a jobid line but NO `SLURM run ID` line anywhere.
    _write_log(
        tmp_path,
        "2026-10-30T000000.000000.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            "Job 1 has been submitted with SLURM jobid 22249999 (log: /x/.snakemake/slurm_logs/rule_c/22249999.log).",
        ],
    )
    return tmp_path


def test_parser_keys_ids_to_the_invocation_that_submitted_them(two_invocations: Path) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: one flat id set, or ids keyed by
    position rather than by the log's own run id -- red under it because the per-invocation
    mapping (run id, ids, file name) is asserted.

    Input class: a log with a `SLURM run ID` line and `has been submitted with SLURM jobid`
    lines. Both-states anchor: turns on the two line forms, present in the real log.
    """
    result = sjr.submitted_job_ids_by_invocation(two_invocations)

    a = result["2026-11-01T015232.143811.snakemake.log"]
    assert a["run_id"] == "9e62392e-a7d2-476a-a066-cc6fb19708f1"
    assert a["job_ids"] == ["22240735", "22240736"]
    assert a["file"] == "2026-11-01T015232.143811.snakemake.log"

    b = result["2026-11-01T024124.511220.snakemake.log"]
    assert b["run_id"] == RUN_ID_B
    # Numeric order, not lexicographic: "9999999" < "22240800" numerically.
    assert b["job_ids"] == ["9999999", "22240800"]


def test_jobid_outside_any_run_id_log_is_attributed_to_no_invocation(two_invocations: Path) -> None:
    """MUST-FAIL ARM, NEW CAPABILITY. Plausible wrong implementation: attribute every
    `with SLURM jobid` line in every log to that log's file name whether or not the log
    announced a run id -- red under it because 22249999 would then appear.

    Input class: a log that contains a submit line and no `SLURM run ID` line. The satisfying
    input is the sibling logs in the same fixture, which DO announce one.
    """
    result = sjr.submitted_job_ids_by_invocation(two_invocations)

    assert "2026-10-30T000000.000000.snakemake.log" not in result
    every_id = {i for entry in result.values() for i in entry["job_ids"]}
    assert "22249999" not in every_id
    # Control: the parser is not simply returning nothing -- the announced invocations are there.
    assert every_id == {"22240735", "22240736", "9999999", "22240800"}


def test_log_with_run_id_and_no_submit_lines_is_an_invocation_with_no_ids(tmp_path: Path) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: drop a log with zero submit lines.
    Red under it because a local or `1_job_many_srun_tasks` invocation must still appear,
    with an empty id list (the page keeps today's behaviour for those run methods).

    Differently-positioned input: the run id is the LAST line, after other noise.
    """
    _write_log(
        tmp_path,
        "2026-11-01T030000.000000.snakemake.log",
        ["Assuming unrestricted shared filesystem usage.", "Building DAG of jobs...", f"SLURM run ID: {RUN_ID_B}"],
    )
    result = sjr.submitted_job_ids_by_invocation(tmp_path)
    assert result["2026-11-01T030000.000000.snakemake.log"]["job_ids"] == []
    assert result["2026-11-01T030000.000000.snakemake.log"]["run_id"] == RUN_ID_B


def test_missing_log_directory_returns_empty(tmp_path: Path) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: raise on a tree with no
    `.snakemake/log` (a bundle, a never-run analysis). Red under it by the exception."""
    assert sjr.submitted_job_ids_by_invocation(tmp_path) == {}


def test_backfill_roster_includes_submitted_ids(two_invocations: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the roster stays the union of the
    plugin CSVs, the job index and the existing store -- red under it because the submitted
    ids, which no store holds, are absent from the ids handed to `recover_rows`.

    A stub `recover_rows` records the roster; `sacct` is never invoked. The job index
    contributes one id so the test also shows the widening is a UNION, not a replacement.
    """
    status_dir = two_invocations / "_status"
    status_dir.mkdir()
    (status_dir / "_job_index.json").write_text('{"11111111": "rule_x"}')

    seen: dict[str, list[str]] = {}

    def _stub_recover_rows(job_ids: list[str], *, timeout_s: float = 60.0) -> list[dict[str, str]]:
        seen["ids"] = list(job_ids)
        return []

    monkeypatch.setattr(sjr, "recover_rows", _stub_recover_rows)
    # The consolidated-store write is best-effort and unrelated; keep it out of this test.
    monkeypatch.setattr("hhemt.slurm_store.consolidate", lambda *a, **k: None)

    report = sjr.backfill(two_invocations)

    assert seen["ids"] == ["9999999", "11111111", "22240735", "22240736", "22240800"]
    assert report["ids_in_csv"] == 5
    # The jobid under no run id is in no roster.
    assert "22249999" not in seen["ids"]


# ---------------------------------------------------------------------------------------
# The Workflow Performance page's job set.
#
# The strings below are exactly what the cluster prints, read from the 2026-11-01 sweep:
# sacct appends the cancelling uid to a cancellation (`CANCELLED by 554635`) and prints a
# never-started job's Start as `None`.
# ---------------------------------------------------------------------------------------
STATE_NEVER_STARTED = "CANCELLED by 554635"
START_NEVER_STARTED = "None"

_EFF_HEADER = (
    ",JobID,JobName,Elapsed,NNodes,NCPUS,MaxRSS,ReqMem,Elapsed_sec,TotalCPU_sec,"
    "MaxRSS_MB,RequestedMem_MB,MainJobID,CPU Efficiency (%),Memory Usage (%)\n"
)


def _eff_step_row(main_job_id: str) -> str:
    return f"0,{main_job_id}.0,python,00:00:23,1,1,710788K,,23.0,0.0,694.12,1000.0,{main_job_id},0.0,69.41\n"


def _write_plugin_report(analysis_dir: Path, main_job_ids: list[str]) -> None:
    """The plugin writes the real CSV INSIDE a `.csv`-named directory."""
    nested = analysis_dir / "logs" / "slurm_efficiency_report" / "slurm_efficiency_report_20260101T000000.csv"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "efficiency_report_deadbeef.csv").write_text(
        _EFF_HEADER + "".join(_eff_step_row(j) for j in main_job_ids)
    )


def _write_recovery(analysis_dir: Path, jobs: dict[str, tuple[str, str]]) -> None:
    """`{main_job_id: (State, Start)}` as job rows of the recovery store."""
    import csv

    eff_dir = analysis_dir / "logs" / "slurm_efficiency_report"
    eff_dir.mkdir(parents=True, exist_ok=True)
    with (eff_dir / sjr.RECOVERY_FILENAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sjr.RECOVERY_HEADER), lineterminator="\n")
        writer.writeheader()
        for job_id, (state, start) in jobs.items():
            row = dict.fromkeys(sjr.RECOVERY_HEADER, "")
            row.update(MainJobID=job_id, StepKind="job", JobID=job_id, State=state, Start=start, Elapsed="00:01:00")
            writer.writerow(row)


def _render_page(analysis_dir: Path) -> tuple[str, dict]:
    import json
    import types

    from hhemt.config.report import report_config
    from hhemt.report_renderers import workflow_performance

    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1"),
    )
    output_path = analysis_dir / "plots" / "workflow_performance.html"
    workflow_performance.render(analysis, report_config(), output_path)
    manifest = json.loads((analysis_dir / "plots" / "workflow_performance.manifest.json").read_text())
    return output_path.read_text(), manifest


def _job_cell(job_id: str) -> str:
    return f'"Job ID (job record)": "{job_id}"'


RUN_ID_FIRST = "9e62392e-a7d2-476a-a066-cc6fb19708f1"
RUN_ID_RENDERING = "0c1d2e3f-4a5b-4c6d-8e7f-90a1b2c3d4e5"


@pytest.fixture
def job_set_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two invocations plus a report-only job and a recovery store.

    - 101: submitted by the first invocation and present in the plugin report.
    - 102: submitted by the first invocation, cancelled while pending: in NO report, and its
      recovery row carries the exact strings the cluster prints for a never-started job.
    - 103: submitted by the first invocation, cancelled AFTER it started (real Start): ran.
    - 201: submitted by the rendering invocation (the second log): in no report.
    - 300: in a plugin report only (its invocation wrote no submit line).
    """
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    _write_log(
        tmp_path,
        "2026-11-01T015232.143811.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            f"SLURM run ID: {RUN_ID_FIRST}",
            "Job 1 has been submitted with SLURM jobid 101 (log: /x/101.log).",
            "Job 2 has been submitted with SLURM jobid 102 (log: /x/102.log).",
            "Job 3 has been submitted with SLURM jobid 103 (log: /x/103.log).",
        ],
    )
    _write_log(
        tmp_path,
        "2026-11-01T024124.511220.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            f"SLURM run ID: {RUN_ID_RENDERING}",
            "Job 9 has been submitted with SLURM jobid 201 (log: /x/201.log).",
        ],
    )
    _write_plugin_report(tmp_path, ["101", "300"])
    _write_recovery(
        tmp_path,
        {
            "102": (STATE_NEVER_STARTED, START_NEVER_STARTED),
            "103": (STATE_NEVER_STARTED, "2026-11-01T01:55:10"),
        },
    )
    return tmp_path


def test_never_started_predicate_is_a_prefix_test_on_state_and_an_absence_test_on_start() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: equality with the bare word
    `CANCELLED`, which counts the cluster's `CANCELLED by 554635` as a job that ran; or a
    State-only test, which counts a job cancelled after it started as never started.

    Input class: (State, Start) pairs. The satisfying inputs are the rows that must NOT match.
    """
    from hhemt.report_renderers.metadata import _is_never_started

    assert _is_never_started(STATE_NEVER_STARTED, START_NEVER_STARTED)
    assert _is_never_started("CANCELLED", "Unknown")
    assert _is_never_started(STATE_NEVER_STARTED, "")
    # Cancelled after it started: a real Start.
    assert not _is_never_started(STATE_NEVER_STARTED, "2026-11-01T01:55:10")
    assert not _is_never_started("COMPLETED", START_NEVER_STARTED)
    assert not _is_never_started("", "")


def test_job_table_is_the_union_of_submit_lines_reports_and_recovery(job_set_tree: Path) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the table keeps the plugin reports as
    its only job source -- red under it because 102, 103 and 201 are in no report.

    MUST-FAIL ARM against the base code: a job present only in a submit line (102, 201) is
    absent from the base table. No invocation is rendering here (`SLURM_JOB_NAME` unset).
    """
    html, _ = _render_page(job_set_tree)

    for job_id in ("101", "102", "103", "201", "300"):
        assert _job_cell(job_id) in html, job_id
    assert "5 jobs submitted, 1 job cancelled before starting, 4 jobs ran." in html


def test_never_started_job_is_counted_apart_and_its_row_shows_its_state(job_set_tree: Path) -> None:
    """NEW CAPABILITY, second must-fail arm. Plausible wrong implementation: the count keys on
    State equality with `CANCELLED` (counts 0 never-started) or on State alone (counts 2).

    102 carries State `CANCELLED by 554635` and Start `None`: never started. 103 carries the
    same State and a real Start: it ran. Both rows show their State.
    """
    html, _ = _render_page(job_set_tree)

    assert "1 job cancelled before starting" in html
    assert "4 jobs ran." in html
    # Each cell is a provenance span; the State cell of job 102's row ends with the state text.
    row_102 = html[html.index(_job_cell("102")) :].split("}", 1)[0]
    assert '"State (job record)": "<span title=' in row_102
    assert f"{STATE_NEVER_STARTED}</span>" in row_102


def test_summary_sentence_is_reworded(job_set_tree: Path) -> None:
    """NEW CAPABILITY. The sentence counts the jobs submitted, the jobs cancelled before they
    started and the jobs that ran, each with its unit in the number's own form. It names no
    source and counts no report: a reader cannot act on which file held a job. The old
    wording counted jobs "across" the reports, which a union no longer is."""
    html, _ = _render_page(job_set_tree)

    assert "5 jobs submitted, 1 job cancelled before starting, 4 jobs ran." in html
    assert "from submit records, plugin reports and the recovery store" not in html
    assert "efficiency report(s) read" not in html
    assert "job(s)" not in html


def test_rendering_invocations_jobs_are_excluded_by_slurm_job_name(
    job_set_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: exclude nothing (201 stays), or
    exclude every invocation's jobs (101 to 103 vanish) -- both red.

    `SLURM_JOB_NAME` is the run uuid the plugin gives every job of the invocation, so the
    jobs of the log whose `SLURM run ID` equals it are in flight and are dropped. Unset
    (the previous test) excludes nothing.
    """
    monkeypatch.setenv("SLURM_JOB_NAME", RUN_ID_RENDERING)
    html, manifest = _render_page(job_set_tree)

    assert _job_cell("201") not in html
    for job_id in ("101", "102", "103", "300"):
        assert _job_cell(job_id) in html, job_id
    assert "4 jobs submitted, 1 job cancelled before starting, 3 jobs ran." in html
    assert manifest["renderer_data"]["excluded_job_count"] == 1


def test_an_unrelated_slurm_job_name_excludes_nothing(job_set_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NEW CAPABILITY. A job name that matches no log's run id (a local shell, an interactive
    allocation) is not an invocation in flight. Differently-positioned input to the exclusion
    arm above: the variable is set, and still nothing is excluded."""
    monkeypatch.setenv("SLURM_JOB_NAME", "interactive")
    html, _ = _render_page(job_set_tree)

    assert _job_cell("201") in html
    assert "5 jobs submitted" in html


def test_table_renders_with_empty_reports_when_submit_lines_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the table branch stays gated on the
    plugin CSVs existing -- red under it, because no report has been written yet on the
    producing run but earlier invocations' submit lines exist."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    _write_log(
        tmp_path,
        "2026-11-01T015232.143811.snakemake.log",
        [f"SLURM run ID: {RUN_ID_FIRST}", "Job 1 has been submitted with SLURM jobid 101 (log: /x/101.log)."],
    )
    html, manifest = _render_page(tmp_path)

    assert _job_cell("101") in html
    assert "SLURM Efficiency is not shown" not in html
    assert manifest["renderer_data"]["slurm_csv_present"] is False


def test_absent_branch_is_unchanged_when_neither_source_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the arm above: no reports and no submit lines is today's absent notice,
    one sentence that names the file the reader opens."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    html, _ = _render_page(tmp_path)

    assert "SLURM Efficiency is not shown: no <code>slurm_efficiency_report_*.csv</code> record was found." in html
    assert "No SLURM resource-efficiency data" not in html


def test_run_method_with_no_submit_lines_keeps_the_report_only_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. A log that announces a run id and writes no submit line (the
    `1_job_many_srun_tasks` pattern) leaves the job set to the reports: a job only the recovery
    store holds does not appear, exactly as before."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    _write_log(tmp_path, "2026-11-01T030000.000000.snakemake.log", [f"SLURM run ID: {RUN_ID_FIRST}"])
    _write_plugin_report(tmp_path, ["101"])
    _write_recovery(tmp_path, {"999": ("COMPLETED", "2026-11-01T01:00:00")})
    html, _ = _render_page(tmp_path)

    assert _job_cell("101") in html
    assert _job_cell("999") not in html
    assert "1 job submitted" in html


def test_every_submit_log_read_is_declared_as_a_source(job_set_tree: Path) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: read the logs and declare none -- red
    under it because the bundle carries only declared sources, so a bundle render would show
    fewer jobs than the page it was cut from."""
    _, manifest = _render_page(job_set_tree)

    declared = manifest["source_paths_relative"]
    assert ".snakemake/log/2026-11-01T015232.143811.snakemake.log" in declared
    assert ".snakemake/log/2026-11-01T024124.511220.snakemake.log" in declared
    assert manifest["renderer_data"]["submit_log_count"] == 2


def test_a_log_the_parser_opens_and_skips_for_carrying_no_run_id_is_still_declared(job_set_tree: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: EVERY `*.snakemake.log` the page opens is a declared
    source, not only the logs the parser attributes to an invocation.

    Input class: a log that holds a `with SLURM jobid` line and no `SLURM run ID` line. The
    parser opens it, finds no run id and drops it from the map, but the open is still a read,
    so the renderer-IO audit reports it undeclared under HHEMT_ENABLE_PROVENANCE_AUDIT=1 and
    the render bundle (which carries only declared sources) omits it. Plausible wrong
    implementation: derive the declared list from the parser's returned map, which lists only
    logs that announced a run id. Reachable: `snakemake` writes a log for an invocation killed
    before the plugin printed its run id line, and the fixture's own sibling logs show the
    announced form.

    Both-states anchor: the assertion turns on membership in `source_paths_relative`, true of
    the glob-derived and the map-derived implementations for the announced logs and differing
    only for the unannounced one. A second correct implementation: declare every file the
    directory listing returns, however the parser later classifies it.
    """
    _write_log(
        job_set_tree,
        "2026-10-30T000000.000000.snakemake.log",
        [
            "Assuming unrestricted shared filesystem usage.",
            "Job 1 has been submitted with SLURM jobid 99 (log: /x/99.log).",
        ],
    )
    html, manifest = _render_page(job_set_tree)

    declared = manifest["source_paths_relative"]
    assert ".snakemake/log/2026-10-30T000000.000000.snakemake.log" in declared
    # Control: the announced logs are declared too, so the assertion above is not vacuous.
    assert ".snakemake/log/2026-11-01T015232.143811.snakemake.log" in declared
    # And the unannounced log contributes no job to the table (attributed to no invocation).
    assert _job_cell("99") not in html


def test_a_job_only_the_recovery_store_holds_still_has_a_row_when_submit_lines_exist(job_set_tree: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: the job set is the union of submit lines, plugin
    reports AND the recovery store's job rows, so a job the store holds and no log or report
    names still has a row.

    Input class: a recovery-store job row for an id that appears in no `.snakemake/log` and no
    plugin report, on a page where submit lines exist. Plausible wrong implementation: the
    universe is the submit-line ids alone, with the recovery store used only to fill columns of
    jobs already in it. Reachable: `slurm_job_recovery._job_ids_from_existing_store` keeps an
    aged-out id in the roster so the store outlives the logs that first named it, and logs are
    rotated or deleted independently of that store.

    Both-states anchor: the assertion turns on the presence of the job's row, not on any
    wording. A second correct implementation: the aggregator, not the caller, unions the
    recovery ids into the population.
    """
    _write_recovery(
        job_set_tree,
        {
            "102": (STATE_NEVER_STARTED, START_NEVER_STARTED),
            "103": (STATE_NEVER_STARTED, "2026-11-01T01:55:10"),
            "400": ("COMPLETED", "2026-10-31T09:00:00"),
        },
    )
    html, _ = _render_page(job_set_tree)

    assert _job_cell("400") in html
    # Control: the other four sources still contribute (101 report+log, 102/103 log+store, 201 log, 300 report).
    for job_id in ("101", "102", "103", "201", "300"):
        assert _job_cell(job_id) in html, job_id


# ---------------------------------------------------------------------------------------
# The Workflow run column and the toolkit-record flag behind the blank-cell note.
# ---------------------------------------------------------------------------------------
INVOCATION_LABEL = "2026-11-01T015232.143811"
REPORT_LABEL = "logs/slurm_efficiency_report/slurm_efficiency_report_20260101T000000.csv/efficiency_report_deadbeef.csv"


def _efficiency_table(
    *,
    report_jobs: list[str],
    universe: set[str] | None = None,
    invocation_by_job: dict[str, str] | None = None,
    recovery: dict | None = None,
    purpose_map: dict | None = None,
) -> str:
    from hhemt.report_renderers import metadata

    csv_texts = [(REPORT_LABEL, _EFF_HEADER + "".join(_eff_step_row(j) for j in report_jobs))] if report_jobs else []
    mounted = metadata._build_slurm_efficiency_html(
        csv_texts,
        purpose_map or {},
        {},
        lambda partition: "",
        recovery,
        universe,
        None,
        invocation_by_job,
    )
    # The rows are in the fragment's script; the markup carries the summary and the notes.
    return mounted.html + (mounted.fragment[1].script if mounted.fragment else "")


def _row(html: str, job_id: str) -> str:
    """The rendered JSON entry of one job's row (cells hold no closing brace)."""
    return html[html.index(_job_cell(job_id)) :].split("}", 1)[0]


def test_every_row_from_a_known_invocation_carries_its_label() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the column keeps its former two-valued
    flag, so no row names an invocation -- red under it because both rows must carry the label.

    Input class: a job with a step row (101) and a job in the universe with no step row (102),
    both named by the submit-record map. Each row's tooltip names that source.
    """
    html = _efficiency_table(
        report_jobs=["101"],
        universe={"101", "102"},
        invocation_by_job={"101": INVOCATION_LABEL, "102": INVOCATION_LABEL},
    )

    for job_id in ("101", "102"):
        row = _row(html, job_id)
        assert f"{INVOCATION_LABEL}</span>" in row, job_id
        assert "read from the Snakemake submit records" in row, job_id


def test_the_submit_record_outranks_the_report_and_recovery_labels() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the report's run id wins over the submit
    record. Red under it because job 101 is named by all three sources and must show the map's."""
    html = _efficiency_table(
        report_jobs=["101"],
        invocation_by_job={"101": INVOCATION_LABEL},
        recovery={"101": {"job": {"JobName": RUN_ID_B}}},
    )

    row = _row(html, "101")
    assert f"{INVOCATION_LABEL}</span>" in row
    assert "deadbeef" not in row
    assert RUN_ID_B not in row


def test_a_job_the_map_does_not_name_falls_back_to_its_reports_run_id() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: no fallback, so the cell is blank.
    The run id is the part of the report's file name after `efficiency_report_`."""
    html = _efficiency_table(report_jobs=["300"], invocation_by_job={"101": INVOCATION_LABEL})

    row = _row(html, "300")
    assert "deadbeef</span>" in row
    assert "run id in the name of the efficiency report" in row


def test_a_recovery_only_row_is_labelled_from_its_job_name() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: only jobs with a report get a label.
    Job 400 is in the universe, in no report and in no map; its recovery job row's `JobName`
    (the plugin names every job with its invocation's run id) is the label."""
    html = _efficiency_table(
        report_jobs=[],
        universe={"400"},
        recovery={"400": {"job": {"JobName": RUN_ID_B, "JobID": "400", "State": "COMPLETED", "Start": "2026-11-01"}}},
    )

    row = _row(html, "400")
    assert f"{RUN_ID_B}</span>" in row
    assert "job name on the job-recovery record" in row


def test_a_job_no_record_names_shows_an_empty_invocation_cell() -> None:
    """The control for the fallbacks: with no map entry, no report and no recovery row the
    cell is the em-dash, not a guessed label."""
    html = _efficiency_table(report_jobs=[], universe={"500"})

    assert '"Workflow run": "\\u2014"' in _row(html, "500")


def test_toolkit_record_flag_is_set_on_every_row_and_counted_by_the_blank_cell_note() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the old values `this run` and
    `historical` stay, or the flag is set only on jobs that have a step row (102 has none), so
    the note under-counts the jobs with no record.

    The flag is no longer a column: a dash in a toolkit-joined column already marks a job with
    no record. The property the earlier test held, that the flag is set on every row including a
    job with no step row, is read here through the note that counts it.

    Input class: a job with a toolkit purpose record (101), one without (300), and a universe
    job with no step row and no record (102).
    """
    html = _efficiency_table(
        report_jobs=["101", "300"],
        universe={"101", "300", "102"},
        purpose_map={"101": {"purpose": "simulate"}},
    )

    assert "show a dash for 2 jobs without a matching record in <code>_status/*.flag.json</code>" in html
    assert '"Toolkit record"' not in html
    assert "historical" not in html
    assert '"this run"' not in html


def test_columns_are_named_workflow_run_and_the_toolkit_record_column_is_cut() -> None:
    """NEW CAPABILITY. The former `Record` heading is gone, its successor is `Workflow run`, and
    the `Toolkit record` column is gone too because a dash already marks an unmatched row."""
    html = _efficiency_table(report_jobs=["101"])

    assert '"Workflow run"' in html
    assert '"Invocation"' not in html
    assert '"Toolkit record"' not in html
    assert '"Record"' not in html


def test_the_rendered_page_labels_each_job_with_the_invocation_whose_log_submitted_it(job_set_tree: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: on the page `render()` builds, a job named by a
    Snakemake log's submit line shows that log's workflow run in its Workflow run cell, ahead of the
    report-file run id.

    Input class: the full page from `job_set_tree`, where 101 is in a log AND in a plugin report
    (`efficiency_report_deadbeef.csv`), 102 and 103 are in a log only, 201 is in the second log,
    and 300 is in a report only. The helper-level tests above hand `_build_slurm_efficiency_html`
    a mapping directly; this one crosses the call from `render()`. Plausible wrong
    implementation: the builder accepts the mapping and the renderer never passes it, so the
    submit records are never the label's source -- 101 then reads `deadbeef` and 102 reads blank.
    Reachable: that is the shipped state when the renderer does not forward the map it already
    built for the job set.

    Both-states anchor: the assertion turns on the log's timestamp appearing in the cell, true of
    any label format that names the invocation by its log, and on the report's run id NOT being
    the label for a job a log names. A second correct implementation: the renderer forwards the
    map under any label spelling that contains the log's timestamp.
    """
    html, _ = _render_page(job_set_tree)

    def _row_of(job_id: str) -> str:
        return html[html.index(_job_cell(job_id)) :].split("}", 1)[0]

    for job_id in ("101", "102", "103"):
        assert "2026-11-01T015232" in _row_of(job_id), job_id
    assert "2026-11-01T024124" in _row_of("201")
    assert "deadbeef" not in _row_of("101")
    # Control: a job no log names still falls back to the report's run id.
    assert "deadbeef" in _row_of("300")


# ---------------------------------------------------------------------------------------
# Member resolution: the member a status record or a rule name belongs to.
# ---------------------------------------------------------------------------------------
#: One rule name per member-bearing family, as the workflow mints them for member 16, and the
#: member each must resolve to. Pinned here so a new family or a renamed one fails a test
#: instead of silently losing its Member cell.
_FAMILY_RULES = {
    "prepare_member_16_evt_3": "16",
    "simulation_member_16_evt_3": "16",
    "process_member_16_evt_3": "16",
    "consolidate_member_16": "16",
    "delete_member_16": "16",
}


def _members_of(index: dict[str, str], member_ids: set[str] | None) -> dict[str, str]:
    from hhemt.report_renderers.metadata import _job_purpose_map

    out = _job_purpose_map([], index, member_ids)
    return {rule: out[job]["sa_id"] for job, rule in index.items()}


def test_each_member_bearing_rule_family_resolves_its_member() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: a family is missing from the anchored
    list, so its jobs lose their Member cell -- red under it, one assertion per family.

    The set is passed and not passed: both resolve the same member for these names.
    """
    index = {str(n): rule for n, rule in enumerate(_FAMILY_RULES)}

    for member_ids in ({"15", "16"}, None):
        assert _members_of(index, member_ids) == _FAMILY_RULES


def test_a_figure_rule_whose_name_contains_member_resolves_to_no_member() -> None:
    """MUST-FIRE ARM, NEW CAPABILITY. Plausible wrong implementation: any `_member_` substring
    names a member. `plot_per_sim_per_member_conduit_flow` runs once per member under one name,
    so it names none, with a member set and without one -- red under the unanchored pattern,
    which captures `conduit_flow`."""
    rules = {"1": "plot_per_sim_per_member_conduit_flow", "2": "delete_member_reprocess_16"}

    assert set(_members_of(rules, {"16", "17"}).values()) == {""}
    assert set(_members_of(rules, None).values()) == {""}


def test_a_member_id_with_a_dot_or_a_dash_matches_its_rule_name() -> None:
    """NEW CAPABILITY. The workflow writes a member id into a rule name with `.` and `-`
    replaced by `_`. Plausible wrong implementation: the parsed token is compared with the id
    as configured, so `1.5` and `a-b` never match and their records read as members no longer
    in the analysis. The matching id comes back in its configured form."""
    index = {"1": "simulation_member_1_5_evt_0", "2": "simulation_member_a_b_evt_0"}

    assert _members_of(index, {"1.5", "a-b"}) == {
        "simulation_member_1_5_evt_0": "1.5",
        "simulation_member_a_b_evt_0": "a-b",
    }


def test_a_sidecar_without_a_member_id_reads_its_member_from_the_rule_name_only_with_a_member_set() -> None:
    """NEW CAPABILITY. Tier 1 reads the rule name of a sidecar that records no member only when
    the member ids are known; without them its Member cell stays empty, as before."""
    from hhemt.report_renderers.metadata import _job_purpose_map

    payload = {"slurm_job_id": "700", "rule_name": "simulation_member_16_evt_3", "event_id": "3"}

    assert _job_purpose_map([payload], None, {"16", "17"})["700"]["sa_id"] == "16"
    assert _job_purpose_map([payload], None, None)["700"]["sa_id"] == ""
    # A sidecar that records its member keeps it either way.
    recorded = {**payload, "sa_id": "17"}
    assert _job_purpose_map([recorded], None, {"16", "17"})["700"]["sa_id"] == "17"


def _timeline(payloads: list[dict], member_ids: set[str] | None):
    from hhemt.report_renderers import metadata

    return metadata._provenance_timeline(payloads, member_ids)


#: A record naming no member, a current member, an orphan that RECORDS its member, and an
#: orphan that is known only by its rule name.
_ORPHAN_PAYLOADS = [
    {"rule_name": "setup_target_0", "written_at": "2026-11-01T01:00:00"},
    {"rule_name": "simulation_member_1_evt_0", "sa_id": "1", "written_at": "2026-11-01T02:00:00"},
    {"rule_name": "simulation_member_99_evt_0", "sa_id": "99", "written_at": "2026-11-01T03:00:00"},
    {"rule_name": "simulation_member_98_evt_0", "written_at": "2026-11-01T04:00:00"},
    {"rule_name": "plot_per_sim_per_member_conduit_flow", "written_at": "2026-11-01T05:00:00"},
]


def test_records_of_members_outside_the_set_are_left_out_and_counted() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: only records that RECORD an outside
    member are left out, so a record known by its rule name (`simulation_member_98_evt_0`)
    shows a removed member's id -- red under it.

    Input class: no member (setup, a figure rule), a current member, and two orphans, one
    recorded and one parsed. The figure rule is kept with an empty Member cell.
    """
    table = _timeline(_ORPHAN_PAYLOADS, {"1"})

    assert table is not None and table.fragment is not None
    page = table.html + table.fragment[1].script
    for kept in ("setup_target_0", "simulation_member_1_evt_0", "plot_per_sim_per_member_conduit_flow"):
        assert kept in page, kept
    for dropped in ("simulation_member_99_evt_0", "simulation_member_98_evt_0"):
        assert dropped not in page, dropped
    assert "2 records left out" in table.html
    assert "3 completed rules" in table.html
    # The figure rule's row has an empty Member cell; the current member's row shows its id.
    figure_row = page[page.index("<code>plot_per_sim_per_member_conduit_flow</code>") :].split("}", 1)[0]
    assert '"Member": ""' in figure_row
    member_row = page[page.index("<code>simulation_member_1_evt_0</code>") :].split("}", 1)[0]
    assert '"Member": "1"' in member_row


def test_the_count_follows_the_member_set() -> None:
    """MUST-FIRE CONTROL for the count: shrinking the member set moves it, so the sentence
    states what the filter did and not a fixed number."""
    both = _timeline(_ORPHAN_PAYLOADS, {"1", "98", "99"})
    none_left = _timeline(_ORPHAN_PAYLOADS, {"1"})

    assert both is not None and none_left is not None
    assert "0 records left out" in both.html
    assert "2 records left out" in none_left.html


def test_the_timeline_leaves_nothing_out_when_there_is_no_member_set() -> None:
    """The control for the exclusion: with no members every record shows, and the note says
    why no count is given."""
    table = _timeline(_ORPHAN_PAYLOADS, None)

    assert table is not None and table.fragment is not None
    page = table.html + table.fragment[1].script
    assert "simulation_member_99_evt_0" in page
    assert "The count of records left out is not shown: this analysis has no members." in table.html
    assert "left out:" not in table.html


def test_a_timeline_whose_every_record_is_an_orphan_states_the_count_instead_of_vanishing() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: an empty row list returns None, so the
    page says no record exists when records exist and all belong to removed members."""
    table = _timeline(_ORPHAN_PAYLOADS[2:4], {"1"})

    assert table is not None
    assert "All 2 records belong to members no longer in this analysis" in table.html
    assert "All 1 record belongs" not in table.html
    one = _timeline(_ORPHAN_PAYLOADS[2:3], {"1"})
    assert one is not None and "All 1 record belong" in one.html


def _sweep_analysis(analysis_dir: Path, member_ids: list[str]):
    """An analysis object whose configuration is a sweep over `member_ids`."""
    import types

    import pandas as pd

    return types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1", toggle_sensitivity_analysis=True),
        sensitivity=types.SimpleNamespace(df_setup=pd.DataFrame(index=pd.Index(member_ids, name="member_id"))),
    )


def test_the_member_set_is_read_from_the_configuration_not_from_scenario_status(tmp_path: Path) -> None:
    """NEW CAPABILITY. The producing render runs before `scenario_status.csv` exists. Plausible
    wrong implementation: the member set comes from that file, so the page the workflow
    produces filters nothing -- red under it because this tree has no such file.

    Printed value on a sweep: `analysis.sensitivity.df_setup.index` is `['0', '1', '2']` for
    the real `TRITONSWMM_sensitivity_analysis` built from a three-row setup table.
    """
    from hhemt.report_renderers.workflow_performance import _configured_member_ids

    assert _configured_member_ids(_sweep_analysis(tmp_path, ["0", "1", "2"])) == {"0", "1", "2"}
    # Not a sweep: no members, and no filter.
    plain = _sweep_analysis(tmp_path, ["0"])
    plain.cfg_analysis.toggle_sensitivity_analysis = False
    assert _configured_member_ids(plain) is None
    # A sweep whose setup table is empty has no members either.
    assert _configured_member_ids(_sweep_analysis(tmp_path, [])) is None


def _render_sweep_page(analysis_dir: Path, analysis) -> str:
    from hhemt.config.report import report_config
    from hhemt.report_renderers import workflow_performance

    output_path = analysis_dir / "plots" / "workflow_performance.html"
    workflow_performance.render(analysis, report_config(), output_path)
    return output_path.read_text()


def _write_sidecars(analysis_dir: Path, sidecars: dict[str, dict]) -> None:
    import json

    status_dir = analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in sidecars.items():
        (status_dir / name).write_text(json.dumps(payload))


def test_the_rendered_page_matches_the_caption_for_every_page_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY, full page, one render per page kind the caption is attached to. Each
    caption clause about the run timeline is asserted against the page that kind builds:

    - a sweep with sidecars of a removed member: the record is left out and the section states
      the count (no `scenario_status.csv` exists, so the member set is the configuration's);
    - a sweep whose only record names no member: shown, with an empty Member cell;
    - a plain analysis: nothing is left out and the section says why no count is given;
    - an analysis with no sidecar: the section says no record exists.
    """
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    sidecars = {
        "a_setup.flag.json": {"rule_name": "setup_target_0", "slurm_job_id": "1"},
        "b_sim_1.flag.json": {"rule_name": "simulation_member_1_evt_0", "sa_id": "1", "slurm_job_id": "2"},
        "c_sim_99.flag.json": {"rule_name": "simulation_member_99_evt_0", "slurm_job_id": "3"},
    }

    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _write_sidecars(sweep, sidecars)
    html = _render_sweep_page(sweep, _sweep_analysis(sweep, ["1", "2"]))
    assert "simulation_member_99_evt_0" not in html
    assert "setup_target_0" in html and "simulation_member_1_evt_0" in html
    assert "1 record left out" in html

    plain = tmp_path / "plain"
    plain.mkdir()
    _write_sidecars(plain, sidecars)
    plain_analysis = _sweep_analysis(plain, ["1"])
    plain_analysis.cfg_analysis.toggle_sensitivity_analysis = False
    html = _render_sweep_page(plain, plain_analysis)
    assert "simulation_member_99_evt_0" in html
    assert "The count of records left out is not shown: this analysis has no members." in html

    empty = tmp_path / "empty"
    empty.mkdir()
    html = _render_sweep_page(empty, _sweep_analysis(empty, ["1"]))
    assert "The run timeline is not shown: no" in html


def test_the_notice_for_a_missing_status_record_prints_no_markdown_backticks(tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: a notice the page prints names a file the reader
    opens as the file name, with no Markdown backtick characters around it. `_banner` escapes
    its text as plain HTML, so a backtick written in the notice is printed to the reader as a
    backtick.

    Input class: the notice shown when no status record exists. Plausible wrong implementation:
    the notice text carries `` ` `` marks as if it were Markdown, so the reader sees
    "no `_status/*.flag.json` record". Reachable: the first render of any analysis, before any
    rule has finished.

    Both-states anchor: the assertion turns on the literal backtick pair being absent and on the
    file name still being present, true of a plain-text and of an `<code>` rendering alike. A
    second correct implementation: the name in plain text.
    """
    html = _render_sweep_page(tmp_path, _sweep_analysis(tmp_path, ["1"]))

    assert "The run timeline is not shown" in html
    assert "_status/*.flag.json" in html
    assert "`_status/*.flag.json`" not in html


# ---------------------------------------------------------------------------------------
# The note on blank cells and the notice for a join that matched no job.
# ---------------------------------------------------------------------------------------
def _blank_cell_note(rows: list[dict]) -> str:
    from hhemt.report_renderers import metadata

    return metadata._uncaptured_note(rows)


def test_the_blank_cell_note_counts_each_kind_from_the_rows() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the former note, whose figures were
    written by hand ("87 of the 88 job ids") and held for one report only -- red under it
    because the counts below follow the rows.

    Input class: rows with every kind of blank cell, one row per count (1 and 2, so the
    singular and the plural of "job" are both asserted).
    """
    rows = [
        {"JobID": "1", "toolkit_record": "no toolkit record"},
        {"JobID": "2", "toolkit_record": "toolkit record", "queue_seconds_this_job": "5"},
        {"JobID": "3", "toolkit_record": "toolkit record", "cpu_eff_pct": "50.0"},
    ]

    note = _blank_cell_note(rows)

    assert "Queue, this job shows a dash for 2 jobs where neither the toolkit nor SLURM recorded" in note
    assert "Billed CPU used shows a dash for 2 jobs where SLURM recorded no CPU time" in note
    assert "show a dash for 1 job without a matching record in <code>_status/*.flag.json</code>" in note
    assert "87 of the 88" not in note


def test_the_blank_cell_note_follows_the_rows_and_omits_a_kind_with_no_blank_cell() -> None:
    """MUST-FIRE CONTROL for the counts: filling the cells removes the sentence and moving a
    row changes the number, so the note states what the rows hold and not a fixed figure."""
    full = {
        "queue_seconds_this_job": "5",
        "cpu_eff_pct": "50.0",
        "toolkit_record": "toolkit record",
    }
    one = _blank_cell_note([{**full, "JobID": "1"}, {**full, "JobID": "2", "queue_seconds_this_job": ""}])
    none = _blank_cell_note([{**full, "JobID": "1"}])

    assert "Queue, this job shows a dash for 1 job where" in one
    assert "Queue, this job" not in none
    assert "Billed CPU used" not in none
    assert "without a matching record" not in none
    # The CPU model has no column, so its sentence does not depend on the rows.
    assert "CPU model is not shown." in none


def test_the_note_on_the_rendered_table_counts_the_rendered_jobs() -> None:
    """NEW CAPABILITY, through the builder. Two report jobs, one with a toolkit record: both
    lack a queue wait and a CPU efficiency (the report row carries no CPU time), and one has no
    matching record."""
    html = _efficiency_table(report_jobs=["101", "300"], purpose_map={"101": {"purpose": "simulate"}})

    assert "Queue, this job shows a dash for 2 jobs" in html
    assert "Billed CPU used shows a dash for 2 jobs" in html
    assert "show a dash for 1 job without a matching record" in html


def test_the_join_rate_sentence_is_gone_and_a_total_miss_is_one_sentence() -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the former join-rate paragraph and its
    multi-sentence banner stay. A miss (a job with a toolkit record whose key matches no
    `scenario_status.csv` row) gets one notice naming the file; a hit and a table with no
    joinable job get none, and no case prints the join-rate paragraph."""
    from hhemt.report_renderers import metadata

    def _html(scenario_map: dict, purpose_map: dict) -> str:
        return metadata._build_slurm_efficiency_html(
            [(REPORT_LABEL, _EFF_HEADER + _eff_step_row("101"))], purpose_map, scenario_map, lambda p: ""
        ).html

    purpose = {"101": {"purpose": "simulate", "model_type": "triton", "event_id": "e1", "sa_id": ""}}
    miss = _html({("", "other", "triton"): {"run_mode": "hybrid"}}, purpose)
    hit = _html({("", "e1", "triton"): {"run_mode": "hybrid"}}, purpose)
    none = _html({}, {})

    notice = "No job matched a record in <code>scenario_status.csv</code>, so the columns read from it are empty."
    assert notice in miss
    assert notice not in hit
    assert notice not in none
    for html in (miss, hit, none):
        assert "Scenario-record join" not in html
        assert "key-vocabulary" not in html


def test_a_queue_wait_slurm_recorded_is_not_counted_as_a_blank_queue_cell() -> None:
    """NEW CAPABILITY (reviewer). Property: the note counts a blank queue cell only where neither
    the toolkit nor SLURM recorded a wait, so a job whose accounting record carries SLURM's
    `Planned` has a queue value and is not in the count. The sentence says exactly that, and
    this is the code fact it states.

    Input class: report jobs 101 and 300 with no toolkit queue map; 101's recovery job row
    carries `Planned` `00:00:54`, 300 has no recovery row. Plausible wrong implementation: the
    table drops the SLURM fallback (a queue value only from the toolkit's map), so both jobs
    read blank and the note counts 2 while the sentence still says SLURM was consulted; or the
    note counts every job. Reachable: any run whose accounting rows are recovered.

    Both-states anchor: the assertion turns on the count of jobs with a dash, 1 and not 2, and
    on the recovered job's cell holding its seconds. A second correct implementation: the
    fallback lives in the aggregator rather than the builder.
    """
    recovery = {
        "101": {"job": {"JobID": "101", "State": "COMPLETED", "Start": "2026-11-01", "Planned": "00:00:54"}},
    }
    html = _efficiency_table(report_jobs=["101", "300"], recovery=recovery)

    assert "Queue, this job shows a dash for 1 job where neither the toolkit nor SLURM recorded" in html
    assert '"Queue, this job (s)": "<span title=' in _row(html, "101")
    assert "54</span>" in _row(html, "101")


def test_a_partial_scenario_join_prints_no_notice() -> None:
    """NEW CAPABILITY (reviewer). Property: the notice says no job matched, so it appears only
    when none did; a join that matched some jobs and missed others prints nothing.

    Input class: two jobs with toolkit records, one whose event key is in `scenario_status.csv`
    and one whose key is not. Plausible wrong implementation: the notice fires on any miss
    (`matched < joinable`), which prints "No job matched" on a table where a job did. Reachable:
    a resumed analysis whose later events are not yet in the file is the routine partial case.

    Both-states anchor: the assertion turns on the notice sentence being absent for the partial
    join and present for the total miss beside it, true of any threshold that means 'none'.
    """
    from hhemt.report_renderers import metadata

    purpose = {
        "101": {"purpose": "simulate", "model_type": "triton", "event_id": "e1", "sa_id": ""},
        "300": {"purpose": "simulate", "model_type": "triton", "event_id": "e2", "sa_id": ""},
    }
    csv = _EFF_HEADER + _eff_step_row("101") + _eff_step_row("300")
    notice = "No job matched a record in <code>scenario_status.csv</code>"

    partial = metadata._build_slurm_efficiency_html(
        [(REPORT_LABEL, csv)], purpose, {("", "e1", "triton"): {"run_mode": "hybrid"}}, lambda p: ""
    ).html
    total = metadata._build_slurm_efficiency_html(
        [(REPORT_LABEL, csv)], purpose, {("", "other", "triton"): {"run_mode": "hybrid"}}, lambda p: ""
    ).html

    assert notice not in partial
    assert notice in total


def test_the_cpu_model_sentence_does_not_claim_the_toolkit_records_no_cpu_model() -> None:
    """NEW CAPABILITY (reviewer). Property: the note makes no statement about what the toolkit
    records that the status table contradicts. `analysis.py` writes `actual_cpu`, the CPU model
    read from the solver log, into `df_status` and so into `scenario_status.csv`; a sentence
    saying the toolkit does not record the CPU model is therefore false while that column
    exists.

    Input class: the note for any rows. Plausible wrong implementation: the constant sentence
    "CPU model is not shown: the toolkit does not record it." The assertion is a conjunction so
    that either repair passes: cut the reason clause (the table simply does not show it) or
    cut the sentence.

    Both-states anchor: the assertion turns on the pair (the code writes `actual_cpu`, the note
    says the toolkit does not record the CPU model), not on the sentence's other wording. A
    second correct implementation: `CPU model is not shown.`
    """
    import inspect

    from hhemt import analysis as analysis_module
    from hhemt.report_renderers import metadata

    note = metadata._uncaptured_note([{"JobID": "1", "toolkit_record": "toolkit record"}])
    code_records_cpu_model = '"actual_cpu",' in inspect.getsource(analysis_module)

    # Control: the fact the sentence would contradict is present in the code today.
    assert code_records_cpu_model
    assert "does not record it" not in note


def test_the_id_set_label_is_exactly_the_ids_present() -> None:
    """NEW CAPABILITY (reviewer). Property: the label's id expression is exactly the ranges of
    the ids present and nothing else, for an unbroken run, a gapped set and a short-run set.

    Input class: three id sets. Plausible wrong implementation: the unbroken-run form carries
    a trailing annotation (a count or a span note) that the substring assertions of the
    package's value tests accept. Reachable: any edit to the single-run branch of `_id_ranges`.

    Both-states anchor: the assertion compares the whole expression, true of every correct
    spelling of the ranges the label uses today.
    """
    from hhemt.report_renderers.metadata import _id_ranges

    assert _id_ranges([1, 2, 3, 4, 5]) == "1…5"
    assert _id_ranges([4, 5, 6, 7, 8, 9, 22, 23, 24, 25, 26, 27, 28, 29, 32]) == "4…9, 22…29, 32"
    assert _id_ranges([1, 2, 4, 5, 6, 7]) == "1, 2, 4…7"


# ---------------------------------------------------------------------------------------
# The sentence about the workflow run that rendered the page.
# ---------------------------------------------------------------------------------------
def test_the_page_names_the_workflow_run_it_left_out_when_slurm_job_name_is_set(
    job_set_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW CAPABILITY. Plausible wrong implementation: the jobs are excluded and the page does
    not say so, so a reader counting jobs against the logs finds a gap with no explanation --
    red under it because the sentence must carry the excluded workflow run's label.

    Input class: `SLURM_JOB_NAME` equal to the run id of the second log. Differently-positioned
    input (the control below): the variable is set to a name that matches no log, or unset.
    """
    monkeypatch.setenv("SLURM_JOB_NAME", RUN_ID_RENDERING)
    html, _ = _render_page(job_set_tree)

    assert "The workflow run that rendered this page is left out: <code>2026-11-01T024124.511220</code>." in html
    assert "No workflow run is left out." not in html


@pytest.mark.parametrize("job_name", [None, "interactive"])
def test_the_page_states_no_exclusion_when_no_workflow_run_was_left_out(
    job_set_tree: Path, monkeypatch: pytest.MonkeyPatch, job_name: str | None
) -> None:
    """MUST-FIRE CONTROL for the sentence: with the variable unset, or set to a name that matches
    no log, nothing is excluded and the page says so; the exclusion wording is absent."""
    if job_name is None:
        monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_NAME", job_name)
    html, _ = _render_page(job_set_tree)

    assert "No workflow run is left out." in html
    assert "rendered this page is left out" not in html
