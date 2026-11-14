"""A failed local Snakemake run names the failing rules, and the resume banner names the incomplete sims.

Problem: a local Snakemake failure used to return only the analysis directory, so a reader (a CI log, a
notebook user) had to find and open the master log to learn which rule failed. Work package 1c, chunk 1c.2
of the release-PR run puts the return code, the log file path and each ``Error in rule R:`` block into the
returned message, and makes the resume banner list the sims that still owe a completion flag.

No solver, compile or Snakemake subprocess runs here. The log is fabricated text in the layout of Snakemake
9.15.0's plain-text log, and the subprocess boundary is replaced by a stand-in that writes that text.

Each test docstring states Property, Class, Why it is required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hhemt import workflow as wf
from hhemt.analysis import _RESUME_BANNER_MAX_NAMED_SIMS, TRITONSWMM_analysis


def _block(rule: str, jobid: int, extra: str = "") -> str:
    """One fabricated ``Error in rule`` block, ending in the blank line Snakemake writes after it."""
    return (
        f"Error in rule {rule}:\n"
        "    message: None\n"
        f"    jobid: {jobid}\n"
        f"    output: _status/{rule}_{jobid}.flag\n"
        "    shell:\n"
        "        python run.py\n"
        "        (one of the commands exited with non-zero exit code; note that snakemake uses bash strict mode!)\n"
        f"{extra}\n"
    )


def _log(*blocks: str) -> str:
    """A fabricated master log: DAG chatter, the blocks with a finished job between them, then the tail."""
    return (
        "Building DAG of jobs...\nSelecting jobs to execute...\n\n"
        + "\nFinished jobid: 1 (Rule: a_prepare)\n\n".join(blocks)
        + "\nRemoving output files of failed job since they might be corrupted:\n_status/x.flag\n"
        "Exiting because a job execution failed. Look below for error messages\n"
    )


# ---------------------------------------------------------------------------
# extract_snakemake_error_blocks
# ---------------------------------------------------------------------------


def test_extract_returns_every_error_block_in_log_order():
    """Property: each ``Error in rule`` block of the log is returned, in log order, with its own body.

    Class: the log's error blocks, a closed population of any size (REGRESSION: the pre-change failure message
    named none of them; NEW CAPABILITY for the extractor itself, whose plausible wrong implementation is one that
    returns only the first or only the last block). Why required: the message is the only text a CI reader sees
    without downloading an artifact, so a rule left out is a rule the reader cannot find. What kills it: returning
    ``blocks[:1]`` or sorting the rule names. Second correct implementation: a single regex over the whole text
    with ``re.finditer`` instead of the line walk, returning the same blocks.
    """
    text = _log(_block("c_run_tritonswmm", 7), _block("d_process_tritonswmm", 9))
    blocks, total = wf.extract_snakemake_error_blocks(text)
    assert total == 2
    assert [b.splitlines()[0] for b in blocks] == [
        "Error in rule c_run_tritonswmm:",
        "Error in rule d_process_tritonswmm:",
    ]
    assert "jobid: 7" in blocks[0] and "jobid: 9" in blocks[1]
    # the neighbouring log lines belong to no block
    assert not any("Finished jobid" in b or "Removing output files" in b for b in blocks)


def test_extract_reads_a_block_at_the_end_of_the_log_with_no_trailing_blank_line():
    """Property: a block that is the last thing in the file is still returned whole.

    Class: a block whose end is end-of-file rather than a blank line (a differently positioned satisfying input
    for the same property as the test above). Why required: a run killed mid-write leaves exactly this tail.
    What kills it: a walk that closes a block only on a blank line and drops one still open at EOF. Second
    correct implementation: splitting the text on the header regex and trimming each piece.
    """
    text = "Building DAG of jobs...\n" + _block("c_run_swmm", 3).rstrip("\n")
    blocks, total = wf.extract_snakemake_error_blocks(text)
    assert total == 1
    assert blocks[0].splitlines()[0] == "Error in rule c_run_swmm:"
    assert "(one of the commands exited with non-zero exit code" in blocks[0]


def test_extract_caps_the_block_count_and_reports_the_true_total():
    """Property: at most ``max_blocks`` blocks are returned and the total counts every block in the log.

    Class: logs with more error blocks than the cap. Why required: an unbounded message would swamp a CI log, and
    a total that stopped at the cap would hide how much is missing. What kills it: dropping the cap, or counting
    only the returned blocks. Second correct implementation: collect all blocks first and slice afterwards.
    """
    text = _log(*[_block(f"rule_{k}", k) for k in range(12)])
    blocks, total = wf.extract_snakemake_error_blocks(text, max_blocks=4)
    assert (len(blocks), total) == (4, 12)
    assert [b.splitlines()[0] for b in blocks] == [f"Error in rule rule_{k}:" for k in range(4)]


def test_extract_truncates_one_long_block_and_says_how_much_was_dropped():
    """Property: a block longer than ``max_block_lines`` is cut and the cut names the number of lines dropped.

    Class: blocks longer than the per-block cap, including a block of exactly the cap (the boundary where nothing
    is dropped). Why required: one runaway block must not displace the other failing rules from the message.
    What kills it: no per-block cap, or a cut that does not name what it dropped. Second correct implementation:
    keep the header and the last lines instead of the first lines, with the same marker.
    """
    long_extra = "\n".join(f"        detail {k}" for k in range(40))
    blocks, _ = wf.extract_snakemake_error_blocks(_block("c_run_triton", 5, long_extra), max_block_lines=10)
    lines = blocks[0].splitlines()
    assert lines[0] == "Error in rule c_run_triton:"
    assert len(lines) == 11
    assert "more line(s) of this block in the log" in lines[-1]

    exact, _ = wf.extract_snakemake_error_blocks(_block("c_run_triton", 5), max_block_lines=9)
    assert "more line(s)" not in exact[0]


def test_extract_on_a_log_with_no_error_block_returns_nothing():
    """Property: a log with no ``Error in rule`` header yields no blocks and a zero total.

    Class: logs of a run that did not fail in a rule (a killed driver, a lock error). Why required: the message
    builder must tell the reader no rule was named instead of printing an empty list. What kills it: returning a
    placeholder block. Second correct implementation: ``re.findall`` over the text returning an empty list.
    """
    assert wf.extract_snakemake_error_blocks("Building DAG of jobs...\nLockException\n") == ([], 0)


# ---------------------------------------------------------------------------
# format_local_snakemake_failure_message
# ---------------------------------------------------------------------------


def test_message_carries_return_code_log_path_and_failing_rule_names(tmp_path: Path):
    """Property: the message names the return code, the log FILE, and each failing rule block.

    Class: failed local runs whose log holds at least one error block. REGRESSION: the pre-change message was
    ``Snakemake workflow failed.\\nSee logs for {dir}``, which carries none of the three; run against that text
    every assertion below reds. Why required: this is the failure the compile-tier CI job showed with no name.
    What kills it: dropping any of the three. Second correct implementation: the same facts laid out as one line
    per rule rather than the verbatim blocks.
    """
    log = tmp_path / "snakemake_master.log"
    log.write_text(_log(_block("c_run_tritonswmm", 7), _block("d_process_tritonswmm", 9)))
    msg = wf.format_local_snakemake_failure_message(1, log, tmp_path / "Snakefile")
    assert "return code 1" in msg
    assert str(log) in msg
    assert "Error in rule c_run_tritonswmm:" in msg and "Error in rule d_process_tritonswmm:" in msg
    assert msg.index("c_run_tritonswmm") < msg.index("d_process_tritonswmm")
    assert f"See logs for {tmp_path}" in msg


def test_message_says_how_many_blocks_it_left_out(tmp_path: Path):
    """Property: when the log holds more blocks than the cap, the message states the count not shown.

    Class: logs above the block cap. Why required: a capped message that is silent about the cap reads as the
    whole story. What kills it: omitting the tail line. Second correct implementation: a header that says
    ``10 of 12`` only, with no tail line (the assertion reads the figure, not the sentence).
    """
    log = tmp_path / "snakemake_master.log"
    log.write_text(_log(*[_block(f"rule_{k}", k) for k in range(wf.LOCAL_FAILURE_MAX_ERROR_BLOCKS + 2)]))
    msg = wf.format_local_snakemake_failure_message(1, log, tmp_path / "Snakefile")
    assert f"{wf.LOCAL_FAILURE_MAX_ERROR_BLOCKS} of {wf.LOCAL_FAILURE_MAX_ERROR_BLOCKS + 2}" in msg
    assert msg.count("Error in rule ") == wf.LOCAL_FAILURE_MAX_ERROR_BLOCKS


def test_message_for_a_log_without_blocks_or_a_missing_log_still_names_code_and_path(tmp_path: Path):
    """Property: the return code and the log path survive a log with no error block and a log that is absent.

    Class: the two ways the extractor has nothing to read. Why required: the diagnostics must not raise inside a
    failure path, which would replace the real failure with an unrelated exception. What kills it: reading the
    log without handling ``OSError``. Second correct implementation: checking ``log.exists()`` before reading.
    """
    present = tmp_path / "snakemake_master.log"
    present.write_text("LockException: directory is locked\n")
    msg = wf.format_local_snakemake_failure_message(2, present, tmp_path / "Snakefile")
    assert "return code 2" in msg and str(present) in msg and "No 'Error in rule' block" in msg

    absent = tmp_path / "absent.log"
    msg = wf.format_local_snakemake_failure_message(3, absent, tmp_path / "Snakefile")
    assert "return code 3" in msg and str(absent) in msg and "could not be read" in msg


def test_run_snakemake_local_returns_the_rule_blocks_it_read_from_the_log(tmp_path: Path, monkeypatch):
    """Property: the dict ``run_snakemake_local`` returns on a nonzero exit carries the failing rule in ``message``.

    Class: nonzero Snakemake exits through the real method body, with the subprocess boundary replaced by a
    stand-in that writes a fabricated log into the file the method opened and exits 1 (REGRESSION: red against the
    pre-change message, which names no rule). The success arm, exit 0, is asserted unchanged. Why required: the
    format function being right is not the same as the failure path calling it. What kills it: leaving the old
    ``error_msg`` line in place. Second correct implementation: building the message in ``analysis.py`` from the
    returned ``snakemake_logfile`` instead of inside the builder.
    """
    log_text = _log(_block("c_run_tritonswmm", 7))

    def fake_run(cmd, cwd, stdout, stderr, text, check):
        stdout.write(log_text if fake_run.returncode else "Workflow finished\n")
        return SimpleNamespace(returncode=fake_run.returncode)

    monkeypatch.setattr(wf.subprocess, "run", fake_run)
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "logs").mkdir(parents=True)
    stub = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir, analysis_log_directory=analysis_dir / "logs"),
        cfg_analysis=SimpleNamespace(local_cpu_cores_for_workflow=1),
        generate_snakemake_config=lambda mode: {},
        write_snakemake_config=lambda config, mode: analysis_dir / "profile",
        _get_snakemake_base_cmd=lambda: ["snakemake"],
        _pre_snakemake_invocation_guards=lambda *a, **k: None,
    )
    snakefile = analysis_dir / "Snakefile"

    fake_run.returncode = 1
    failed = wf.SnakemakeWorkflowBuilder.run_snakemake_local(stub, snakefile, verbose=False)
    assert failed["success"] is False
    assert "Error in rule c_run_tritonswmm:" in failed["message"]
    assert "return code 1" in failed["message"]
    assert str(failed["snakemake_logfile"]) in failed["message"]

    fake_run.returncode = 0
    ok = wf.SnakemakeWorkflowBuilder.run_snakemake_local(stub, snakefile, verbose=False)
    assert ok["success"] is True and ok["message"] == "Workflow completed successfully"


def test_message_shows_a_never_finished_block_even_after_more_retried_blocks_than_the_cap(tmp_path: Path):
    """Property: a never-finished block is in the message and labelled as such, however many retried blocks precede it.

    Class: logs whose retried-to-success blocks outnumber the cap and precede a never-finished block (REGRESSION:
    the first-N-in-log-order form showed 10 of 13 and omitted it; the log was measured by the reviewer's probe).
    Why required: the purpose of the message is that a failed run is diagnosable in one read. What kills it: taking
    the first ``cap`` blocks in log order. Second correct implementation: parsing the retried set instead of the
    never-finished set and showing the complement first.
    """
    retried = "".join(_block("a_prepare", k) + f"Finished jobid: {k} (Rule: a_prepare)\n\n" for k in range(12))
    log = tmp_path / "snakemake_master.log"
    log.write_text("Building DAG of jobs...\n\n" + retried + _block("d_process_tritonswmm", 99))
    msg = wf.format_local_snakemake_failure_message(1, log, tmp_path / "Snakefile")
    assert "Error in rule d_process_tritonswmm:" in msg
    assert "1 never finished, 12 were retried to success" in msg
    assert msg.index("Never-finished rule(s)") < msg.index("d_process_tritonswmm") < msg.index("Retried to success")
    assert msg.count("Error in rule ") == wf.LOCAL_FAILURE_MAX_ERROR_BLOCKS


# ---------------------------------------------------------------------------
# _print_resume_status names the incomplete sims
# ---------------------------------------------------------------------------


def _resume_stub(tmp_path: Path, *, sensitivity: bool, nsims: int, **extra):
    status_dir = tmp_path / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    stub = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(toggle_tritonswmm_model=True, toggle_triton_model=False, toggle_swmm_model=False)
        ),
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=sensitivity,
            analysis_id="testA",
            hpc_restart_times_simulate=2,
            multi_sim_run_method="batch_job",
        ),
        nsims=nsims,
        **extra,
    )
    return stub, status_dir


def test_resume_banner_names_the_events_without_a_completion_flag(capsys, tmp_path: Path, monkeypatch):
    """Property: the banner lists exactly the events whose completion flag is absent, and no completed one.

    Class: non-sensitivity analyses with some, all and no flags present (this test covers some; the next covers
    all). REGRESSION: the pre-change banner printed only the ``k/n`` count. Why required: a resume that says
    ``1/3`` and not which two are owed forces the reader to list ``_status/`` by hand. What kills it: printing
    every event, or none. Second correct implementation: deriving the labels from ``df_status`` rather than from
    the flag files, naming the same events.
    """
    import hhemt.scenario as scenario_mod

    monkeypatch.setattr(scenario_mod, "compute_event_id_slug", lambda ev: f"e{ev}")
    stub, status_dir = _resume_stub(
        tmp_path,
        sensitivity=False,
        nsims=3,
        df_sims=[object(), object(), object()],
        _retrieve_weather_indexer_using_integer_index=lambda i: i,
    )
    (status_dir / "c_run_tritonswmm_evt-e1_complete.flag").write_text("")
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Resuming testA — 1/3 sims complete." in out
    assert "Incomplete sims (2): evt-e0, evt-e2" in out
    assert "evt-e1" not in out.split("Incomplete sims")[1]


def test_resume_banner_names_nothing_when_every_sim_is_complete(capsys, tmp_path: Path, monkeypatch):
    """Property: with every flag present the banner prints no incomplete-sims line.

    Class: fully complete analyses, the state a resume prints ``n/n`` for. Why required: a line naming nothing
    would read as an unexplained problem. What kills it: printing the line unconditionally. Second correct
    implementation: printing the line only when ``n_incomplete > 0`` computed from the counts.
    """
    import hhemt.scenario as scenario_mod

    monkeypatch.setattr(scenario_mod, "compute_event_id_slug", lambda ev: f"e{ev}")
    stub, status_dir = _resume_stub(
        tmp_path,
        sensitivity=False,
        nsims=2,
        df_sims=[object(), object()],
        _retrieve_weather_indexer_using_integer_index=lambda i: i,
    )
    for k in range(2):
        (status_dir / f"c_run_tritonswmm_evt-e{k}_complete.flag").write_text("")
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Resuming testA — 2/2 sims complete." in out
    assert "Incomplete sims" not in out


def test_resume_banner_names_sensitivity_member_event_pairs_and_caps_the_list(capsys, tmp_path: Path, monkeypatch):
    """Property: on a sensitivity primary the banner names ``member-M_evt-E`` pairs, and a long list is capped.

    Class: sensitivity analyses with missing flags, below and above the naming cap. Why required: the compile-tier
    sensitivity failure was the one nobody could name. What kills it: naming events without the member, or
    printing an unbounded list. Second correct implementation: reading the pairs from the sensitivity setup frame
    instead of from each member's ``df_sims``.
    """
    import hhemt.scenario as scenario_mod

    monkeypatch.setattr(scenario_mod, "compute_event_id_slug", lambda ev: f"e{ev}")
    n_events = _RESUME_BANNER_MAX_NAMED_SIMS + 3
    member = SimpleNamespace(df_sims=[object()] * n_events, _retrieve_weather_indexer_using_integer_index=lambda i: i)
    stub, status_dir = _resume_stub(tmp_path, sensitivity=True, nsims=n_events)
    stub.sensitivity = SimpleNamespace(members={"0": member})
    (status_dir / "c_run_tritonswmm_member-0_evt-e0_complete.flag").write_text("")
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert f"Incomplete sims ({n_events - 1}): member-0_evt-e1, member-0_evt-e2" in out
    assert "(+2 more)" in out
    assert "member-0_evt-e0" not in out.split("Incomplete sims")[1]
