"""
Phase 1 retry-policy tests (resume-retry-resilience).

Covers the parts of the two-knob retry policy NOT already pinned by
``test_workflow_snakefile_byte_identity.py`` (per-rule ``retries:`` on the
simulate rules) and ``test_workflow_restart_times_profile.py`` (the global
``restart-times`` key is slurm-only):

- the global ``restart-times`` baseline is sourced from ``hpc_restart_times_other``
  (not the simulate knob), and the ``override_hpc_restart_times_other`` runtime
  override resolves at the emission site;
- ``_resolved_simulate_retries`` resolves override-or-config for the simulate rules;
- ``_emit_wait_for_sim_rule_block`` fail-fasts with ``retries: 0`` (FQ2);
- ``_sweep_failed_rules`` unions the ``_status/_failed/`` markers with the
  ``Error in rule (...)`` Snakemake-log scan (FQ3 / R4-R5);
- ``_augment_result_with_partial_failures`` carries the sweep into the result dict
  and forces ``success=False`` when non-empty.
"""

import json

import pytest

import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def slurm_ready_builder():
    """A multi-sim builder with the minimal HPC config the slurm branch reads.

    Mirrors the fixture in ``test_workflow_restart_times_profile.py``: the
    ``else:  # slurm`` branch of ``generate_snakemake_config`` asserts
    ``hpc_max_simultaneous_sims`` is an int and reads ``hpc_ensemble_partition``/
    ``hpc_account``; the shared top-of-function assert requires
    ``local_cpu_cores_for_workflow`` for every mode.
    """
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
    from hhemt.workflow import SnakemakeWorkflowBuilder

    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis.cfg_analysis.local_cpu_cores_for_workflow = 4
    analysis.cfg_analysis.hpc_ensemble_partition = "standard"
    analysis.cfg_hpc_system = hpc_system_config(
        hpc_name="test-cluster",
        default_account="test_account",
        max_concurrent_jobs=8,
        partitions={"standard": PartitionSpec(max_runtime=120)},
    )
    return SnakemakeWorkflowBuilder(analysis)


# ---- global baseline = the _other knob, with override resolution (R2) ----


def test_global_baseline_is_other_knob(slurm_ready_builder):
    """The slurm profile's ``restart-times`` baseline tracks ``hpc_restart_times_other``
    (the knob directive-less rules inherit), NOT the simulate knob."""
    slurm_ready_builder.cfg_analysis.hpc_restart_times_other = 3
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    config = slurm_ready_builder.generate_snakemake_config(mode="slurm")
    assert config["restart-times"] == 3, (
        "global restart-times baseline must equal hpc_restart_times_other (the "
        f"directive-less-rule baseline), not the simulate knob; got {config.get('restart-times')!r}"
    )


def test_override_other_resolves_at_baseline(slurm_ready_builder):
    """``override_hpc_restart_times_other`` wins over the config knob at the baseline."""
    slurm_ready_builder.cfg_analysis.hpc_restart_times_other = 3
    slurm_ready_builder._override_hpc_restart_times_other = 9
    config = slurm_ready_builder.generate_snakemake_config(mode="slurm")
    assert config["restart-times"] == 9


# ---- per-rule simulate retries resolution (R3) ----


def test_resolved_simulate_retries_uses_config_by_default(slurm_ready_builder):
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    assert slurm_ready_builder._resolved_simulate_retries() == 17


def test_resolved_simulate_retries_prefers_override(slurm_ready_builder):
    slurm_ready_builder.cfg_analysis.hpc_restart_times_simulate = 17
    slurm_ready_builder._override_hpc_restart_times_simulate = 20
    assert slurm_ready_builder._resolved_simulate_retries() == 20


# ---- wait-rule fail-fast (FQ2 / R2-R3) ----


def test_wait_rule_emits_retries_zero(slurm_ready_builder):
    """``_emit_wait_for_sim_rule_block`` must carry ``retries: 0`` so a wait-rule
    observing a ``_failed/`` marker fail-fasts instead of inheriting the global baseline."""
    block = slurm_ready_builder._emit_wait_for_sim_rule_block(
        rule_token="run_triton_evt-0",
        flag_output_path="_status/c_run_triton_evt-0_complete.flag",
        run_rule_inputs=["_status/a_setup_complete.flag"],
        wait_walltime_cap_min=30,
    )
    assert "    retries: 0\n" in block, "wait-rule must emit an explicit retries: 0 (FQ2 fail-fast)"


# ---- failed-rule sweep union (FQ3 / R4-R5) ----


def test_sweep_failed_rules_unions_markers_and_stderr(slurm_ready_builder, tmp_path):
    """The sweep unions _status/_failed/ markers (sim class) with the
    ``Error in rule (...)`` Snakemake-log scan (process/consolidate/plot/render class).

    Property: one failed sim event yields exactly one record, named by its marker.
    Class: a sim failure that both wrote a marker and printed a Snakemake error block,
    where the marker is in the sim runner's real payload shape (no ``rule_token`` key) and
    the log names the RULE (``run_triton``), not the event token. Why: without the stem
    stamp the marker record prints as ``?``, and without the prefix dedup the same failure
    is counted twice. What kills it: dropping the stem stamp, or comparing the log rule to
    marker tokens by equality. A second correct implementation, under which this still
    passes: dedup by parsing the event token back to its rule name; the assertions read
    only the token set and the per-token count.
    """
    failed_dir = tmp_path / "_status" / "_failed"
    failed_dir.mkdir(parents=True)
    # The sim runner's marker payload (run_simulation_runner.py): no rule_token key.
    (failed_dir / "run_triton_evt-0.json").write_text(
        json.dumps(
            {
                "slurm_jobid": "555",
                "run_uuid": "u-1",
                "sa_id": None,
                "model_type": "triton",
                "event_id": "0",
                "status": "failed",
                "finished_at": "2026-10-23T00:00:00",
            }
        )
    )
    (failed_dir / "broken.json").write_text("{ not valid json")  # exercises the parse-fallback
    stderr = (
        "Some snakemake output\n"
        "Error in rule consolidate_zarr:\n"
        "    jobid: 7\n"
        "\n"
        "Error in rule run_triton:\n"  # the same failure as the marker, named by its RULE
        "    jobid: 3\n"
    )

    records = slurm_ready_builder._sweep_failed_rules(tmp_path, snakemake_stderr=stderr)
    tokens = [r.get("rule_token") for r in records]

    assert "run_triton_evt-0" in tokens  # from marker, token stamped from the file stem
    assert "broken" in tokens  # unparseable marker still surfaces by stem
    assert "consolidate_zarr" in tokens  # from stderr scan (no marker)
    # union de-dupes: the log block for the marker's rule adds no second record
    assert "run_triton" not in tokens
    assert tokens.count("run_triton_evt-0") == 1
    assert len(records) == 3


def test_sweep_empty_when_no_failures(slurm_ready_builder, tmp_path):
    assert slurm_ready_builder._sweep_failed_rules(tmp_path, snakemake_stderr="") == []


# ---- the log layer keys on the (jobid, rule) pair ----
#
# These drive the unbound builder methods with a minimal self, so no synthetic case is built.
# The Snakemake log lines follow Snakemake 9.15.0's own layout: an error block is
# ``Error in rule R:`` then ``    jobid: N``, a restart prints ``Trying to restart job N.``,
# and a success prints ``Finished jobid: N (Rule: R)``.


def _r1d_augment(tmp_path, log_text: str) -> dict:
    """Run the real sweep-and-augment over ``log_text`` written to a Snakemake log file."""
    from types import SimpleNamespace

    from hhemt.workflow import SnakemakeWorkflowBuilder

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    logfile = tmp_path / "snakemake_master.log"
    logfile.write_text(log_text)
    fake_self = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        _sweep_failed_rules=lambda d, snakemake_stderr="": SnakemakeWorkflowBuilder._sweep_failed_rules(
            None, d, snakemake_stderr=snakemake_stderr
        ),
    )
    return SnakemakeWorkflowBuilder._augment_result_with_partial_failures(
        fake_self, {"success": True, "message": "Workflow completed successfully", "snakemake_logfile": logfile}
    )


# The two failed attempts of a scheduled-resume run, as the Rivanna run of 2026-10-22 logged them.
_R1D_TWO_FAILED_ATTEMPTS = (
    "localrule run_tritonswmm:\n"
    "    jobid: 4\n"
    "\n"
    "Error in rule run_tritonswmm:\n"
    "    jobid: 4\n"
    "    log: logs/sims/tritonswmm_evt-event_index.0.log (check log file(s) for error details)\n"
    "\n"
    "Trying to restart job 4.\n"
    "Error in rule run_tritonswmm:\n"
    "    jobid: 4\n"
    "\n"
    "Trying to restart job 4.\n"
)


def test_sweep_drops_a_rule_that_snakemake_retried_to_success(tmp_path):
    """Property: a job Snakemake restarted and finished is not a permanent failure, and is disclosed.

    Class: logs in which a ``(jobid, rule)`` pair has one or more error blocks and a later
    ``Finished jobid: N (Rule: R)`` line. Why: every scheduled resume fails its interrupted
    attempts on purpose, so counting their error blocks reports a completed run as failed.
    What kills it: counting error headers without checking for a later finish, or omitting
    the ``retried_to_success`` disclosure. A second correct implementation, under which this
    still passes: track per-jobid state transitions line by line instead of collecting sets;
    the assertions read only ``success``, ``partial_failures`` and the disclosure records.
    """
    result = _r1d_augment(
        tmp_path, _R1D_TWO_FAILED_ATTEMPTS + "Finished jobid: 4 (Rule: run_tritonswmm)\n17 of 17 steps (100%) done\n"
    )
    assert result["partial_failures"] == []
    assert result["success"] is True
    assert result["retried_to_success"] == [{"rule": "run_tritonswmm", "jobid": 4, "attempts": 3}]


def test_sweep_keeps_a_rule_that_errored_and_never_finished(tmp_path):
    """Property: a job whose every attempt errored is still a permanent failure.

    The both-states control for the test above: green before and after the change, so the
    test above cannot be satisfied by dropping the log layer. Class: logs whose error blocks
    for a pair have no later finish. What kills it: discarding log-layer records wholesale.
    A second correct implementation: rejecting a failure only on a matching finish keyed by
    jobid alone; the assertions read only the verdict and the rule token.
    """
    result = _r1d_augment(tmp_path, _R1D_TWO_FAILED_ATTEMPTS)
    assert result["success"] is False
    assert [r.get("rule_token") for r in result["partial_failures"]] == ["run_tritonswmm"]


def test_sweep_reads_the_error_blocks_own_jobid_not_a_preceding_start_block(tmp_path):
    """Property: an error record carries the jobid printed inside its own block.

    Class: logs in which another job's START block (with its own ``jobid:`` line) precedes
    an error block, and that other job later finishes. Why: a backward or nearest search
    binds the other job's id, finds it finished and drops a real failure. What kills it:
    reading the jobid backward from the header, or from the nearest ``jobid:`` line in
    either direction. A second correct implementation: parse whole records split on blank
    lines; the assertion reads only the record's ``jobid`` field.
    """
    log = (
        "localrule plot_z:\n"
        "    output: plots/z.html\n"
        "    jobid: 7\n"
        "\n"
        "Error in rule plot_x:\n"
        "    jobid: 9\n"
        "\n"
        "Finished jobid: 7 (Rule: plot_z)\n"
    )
    result = _r1d_augment(tmp_path, log)
    (record,) = [r for r in result["partial_failures"] if r.get("rule_token") == "plot_x"]
    assert record["jobid"] == 9


# ---- the waited single-job run is swept from the script's log directives ----


def _r1d_waited_single_job_result(tmp_path, script_lines: list[str], logs: dict[str, str]) -> dict:
    """Drive submit_workflow's waited 1_job_many_srun_tasks branch with the submission mocked.

    ``script_lines`` are written to the script the mocked submission reports; ``logs`` maps a
    file name under ``tmp_path`` to its text. Only the submission, Snakefile generation, the
    dry run and the queued-sentinel writers are replaced; the sweep runs for real.
    """
    from unittest.mock import Mock, patch

    from hhemt.workflow import SnakemakeWorkflowBuilder

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    script = analysis_dir / "run_workflow_1job.sh"
    script.write_text("#!/bin/bash\n" + "\n".join(line.format(tmp=tmp_path) for line in script_lines) + "\n")
    for name, text in logs.items():
        (tmp_path / name).write_text(text)

    analysis = Mock()
    analysis.cfg_analysis = Mock()
    analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    analysis.cfg_analysis.hpc_cpus_per_node = 32
    analysis.cfg_analysis.hpc_total_nodes = 1
    analysis.cfg_analysis.mem_gb_per_cpu = 2
    analysis.cfg_analysis.hpc_restart_times_simulate = 2
    analysis.cfg_analysis.hpc_consecutive_failure_budget_simulate = 3
    analysis.analysis_paths = Mock()
    analysis.analysis_paths.analysis_dir = analysis_dir
    analysis._resource_manager = Mock()
    analysis._resource_manager._get_simulation_resource_requirements.return_value = {"n_gpus": 0}
    analysis._system = Mock()
    analysis._python_executable = "python"
    analysis._refresh_log = Mock()

    submitted = {
        "success": True,
        "mode": "single_job",
        "job_id": "123",
        "script_path": script,
        "message": "Job 123 COMPLETED (exit 0)",
        "completed": True,
    }
    builder = SnakemakeWorkflowBuilder(analysis)
    with (
        patch.object(SnakemakeWorkflowBuilder, "_planned_sim_tokens", return_value=[]),
        patch.object(SnakemakeWorkflowBuilder, "_write_queued_sentinels"),
        patch.object(SnakemakeWorkflowBuilder, "_submit_single_job_workflow", return_value=submitted),
        patch.object(builder, "generate_snakefile_content", return_value="# Snakefile"),
        patch.object(builder, "run_snakemake_local", return_value={"success": True}),
        patch("pathlib.Path.write_text"),
    ):
        return builder.submit_workflow(wait_for_completion=True, verbose=False)


_R1D_UNFINISHED_PLOT_Y = "Error in rule plot_y:\n    jobid: 2\n\n"


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     patch.object(SnakemakeWorkflowBuilder, "_submit_single_job_workflow", return_value=submitted),
def test_waited_single_job_run_is_swept(tmp_path):
    """Property: a waited single-job run is swept from the file its script's directives name.

    Class: waited ``1_job_many_srun_tasks`` runs whose log carries a never-finished error
    block. Why: that path has no Snakemake log file of its own, so a sweep that reads only
    ``snakemake_logfile`` sees nothing and reports a failed rule as success. What kills it:
    not assembling the log from the script, or assembling it and not handing it to the
    sweep. A second correct implementation: have the single-job submitter record the log
    path itself; the assertions read only ``success`` and the failed rule's name.
    """
    result = _r1d_waited_single_job_result(
        tmp_path,
        ["#SBATCH --output={tmp}/workflow_x_%j.out", "#SBATCH --error={tmp}/workflow_x_%j.out"],
        {"workflow_x_123.out": _R1D_UNFINISHED_PLOT_Y},
    )
    assert result["success"] is False
    assert "plot_y" in [r.get("rule_token") for r in result["partial_failures"]]


# R2 exclusion: retained, not converted — the run is stubbed, so no analysis tree is consumed and no
# completed-tree fixture applies. Evidence:
#     patch.object(SnakemakeWorkflowBuilder, "_submit_single_job_workflow", return_value=submitted),
# Covers parametrize ids: i-two-timestamps, ii-user-error-line, iii-short-spelling, iv-long-prefix.
@pytest.mark.parametrize(
    "script_lines",
    [
        pytest.param(
            ["#SBATCH --output={tmp}/workflow_a_%j.out", "#SBATCH --error={tmp}/workflow_b_%j.out"],
            id="i-two-timestamps",
        ),
        pytest.param(
            [
                "#SBATCH --output={tmp}/workflow_a_%j.out",
                "#SBATCH --error={tmp}/workflow_a_%j.out",
                "#SBATCH --error={tmp}/user_err_%j.out",
            ],
            id="ii-user-error-line",
        ),
        pytest.param(
            [
                "#SBATCH --output={tmp}/workflow_a_%j.out",
                "#SBATCH --error={tmp}/workflow_a_%j.out",
                "#SBATCH -e {tmp}/user_err_%j.out",
            ],
            id="iii-short-spelling",
        ),
        pytest.param(
            [
                "#SBATCH --output={tmp}/workflow_a_%j.out",
                "#SBATCH --error={tmp}/workflow_a_%j.out",
                "#SBATCH --err={tmp}/user_err_%j.out",
            ],
            id="iv-long-prefix",
        ),
    ],
)
def test_waited_single_job_run_reads_every_error_path_spelling(tmp_path, script_lines):
    """Property: the Snakemake block is found whichever error-path directive holds it.

    Class: scripts naming the error log through a second timestamped line, a user
    ``additional_sbatch_args`` line, the short ``-e`` spelling or a long-option prefix,
    while ``workflow_a`` holds only stdout banner text. Why: sbatch applies the last
    directive in any getopt_long spelling, and Snakemake writes to stderr, so a reader of
    the first or only the ``--error=`` line reads the wrong file. What kills it: reading
    only ``--output``, only the first ``--error``, or only the ``--error=`` spelling. A
    second correct implementation: read only the LAST error-path directive, as sbatch
    applies it; the assertions read only ``success`` and the failed rule's name.
    """
    logs = {"workflow_a_123.out": "stdout banner only\n"}
    target = "user_err_123.out" if any("user_err" in line for line in script_lines) else "workflow_b_123.out"
    logs[target] = _R1D_UNFINISHED_PLOT_Y
    result = _r1d_waited_single_job_result(tmp_path, script_lines, logs)
    assert result["success"] is False
    assert "plot_y" in [r.get("rule_token") for r in result["partial_failures"]]


# ---- result augmentation: partial_failures + success flip ----


def test_augment_sets_partial_failures_and_flips_success(slurm_ready_builder, monkeypatch):
    monkeypatch.setattr(
        slurm_ready_builder,
        "_sweep_failed_rules",
        lambda analysis_dir, snakemake_stderr="": [{"rule_token": "run_triton_evt-0", "reason": "x"}],
    )
    result = slurm_ready_builder._augment_result_with_partial_failures({"success": True, "snakemake_logfile": None})
    assert result["partial_failures"] == [{"rule_token": "run_triton_evt-0", "reason": "x"}]
    assert result["success"] is False, "a non-empty partial_failures sweep must force success=False"


def test_augment_clean_run_keeps_success(slurm_ready_builder, monkeypatch):
    monkeypatch.setattr(
        slurm_ready_builder,
        "_sweep_failed_rules",
        lambda analysis_dir, snakemake_stderr="": [],
    )
    result = slurm_ready_builder._augment_result_with_partial_failures({"success": True, "snakemake_logfile": None})
    assert result["partial_failures"] == []
    assert result["success"] is True


def test_augment_rewrites_message_on_partial_failure(tmp_path):
    """The producer rewrites `message` when it flips `success`, so every consumer tells the
    same truth. Both-states anchor: the message is NOT the input sentence and carries the
    rule token. RED pre-fix: the input message came back unchanged.

    A `SimpleNamespace` self carries the only two attributes the method reads, so no
    builder, no norfolk case and no Snakemake is involved.
    """
    from types import SimpleNamespace

    from hhemt.workflow import SnakemakeWorkflowBuilder

    fake_self = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        _sweep_failed_rules=lambda analysis_dir, snakemake_stderr="": [
            {"rule_token": "run_triton_evt-0", "reason": "x"}
        ],
    )
    result = SnakemakeWorkflowBuilder._augment_result_with_partial_failures(
        fake_self, {"success": True, "message": "Workflow completed successfully", "snakemake_logfile": None}
    )
    assert result["success"] is False
    assert result["message"] != "Workflow completed successfully"
    assert "run_triton_evt-0" in result["message"]

    # Differently-positioned satisfying input: a clean sweep leaves the producer's sentence alone.
    clean_self = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        _sweep_failed_rules=lambda analysis_dir, snakemake_stderr="": [],
    )
    clean = SnakemakeWorkflowBuilder._augment_result_with_partial_failures(
        clean_self, {"success": True, "message": "Workflow completed successfully", "snakemake_logfile": None}
    )
    assert clean["message"] == "Workflow completed successfully"


def test_raise_if_dry_run_failed_helper():
    """The helper's contract. GREEN the moment the helper exists (RED pre-fix as ImportError),
    which is why the site-level node below is kept beside it."""
    from hhemt.exceptions import WorkflowPlanningError
    from hhemt.workflow import _raise_if_dry_run_failed

    assert _raise_if_dry_run_failed({"success": True}, phase="x") is None

    with pytest.raises(WorkflowPlanningError) as excinfo:
        _raise_if_dry_run_failed(
            {"success": False, "message": "reason-token-7f3", "snakemake_logfile": None}, phase="batch_job"
        )
    assert excinfo.value.phase == "batch_job"
    assert "Dry run failed" in str(excinfo.value)
    assert "reason-token-7f3" in str(excinfo.value)


def test_failed_dry_run_raises_workflow_planning_error():
    """SITE-level differential through the unbound `_validate_single_job_dry_run`, driven by a
    `SimpleNamespace` self whose `run_snakemake_local` returns a failed plan (no Snakemake is
    invoked). RED pre-fix: a bare RuntimeError. Both-states anchor: `Dry run failed` in the
    message. The satisfying arm returns the plan when it succeeded."""
    from types import SimpleNamespace

    from hhemt.exceptions import WorkflowPlanningError
    from hhemt.workflow import SnakemakeWorkflowBuilder

    def _analysis(cores: int) -> SimpleNamespace:
        return SimpleNamespace(
            cfg_analysis=SimpleNamespace(hpc_cpus_per_node=4, hpc_total_nodes=2, local_cpu_cores_for_workflow=cores)
        )

    failing_self = SimpleNamespace(
        run_snakemake_local=lambda **kw: {"success": False, "message": "plan failed", "snakemake_logfile": None}
    )
    analysis = _analysis(cores=1)
    with pytest.raises(WorkflowPlanningError) as excinfo:
        SnakemakeWorkflowBuilder._validate_single_job_dry_run(
            failing_self, snakefile_path=None, analysis=analysis, verbose=False
        )
    assert "Dry run failed" in str(excinfo.value)
    assert excinfo.value.phase == "single_job"
    assert analysis.cfg_analysis.local_cpu_cores_for_workflow == 1, "the finally must restore the cores"

    passing_self = SimpleNamespace(run_snakemake_local=lambda **kw: {"success": True, "message": "ok"})
    out = SnakemakeWorkflowBuilder._validate_single_job_dry_run(
        passing_self, snakefile_path=None, analysis=_analysis(cores=1), verbose=False
    )
    assert out["success"] is True
    assert out["mode"] == "single_job"


# ---- a Snakemake finish clears only an error block that precedes it ----


def test_review_a_finished_line_before_the_error_block_does_not_clear_it():
    """Property: an error block is cleared only by a ``Finished jobid: N (Rule: R)`` line AFTER it.

    Class: logs that hold more than one Snakemake run, in which a pair ``(N, R)`` finished in an
    earlier run and then errored, with no finish after the error, in a later run. Such a log is
    reachable on the waited single-job path, which reads a fixed-name ``#SBATCH --error`` file
    whether or not it is fresh: under ``--open-mode=append`` (a user directive, or a cluster's
    append default) the file keeps the earlier run's lines ahead of the current ones. Snakemake
    numbers jobs in DAG order, so the same rule gets the same jobid on a rerun of the same
    workflow. Why it is required: a finish recorded BEFORE the error cannot be that failed
    attempt's retry, so letting it clear the error reports a real failure as clean, the false
    clean the log layer exists to prevent. What kills it: matching error blocks against the SET of
    every finished pair in the text, without regard to position. A second correct implementation,
    under which this still passes: walk the lines once and keep a per-pair state that a finish
    sets to "done" and a later error block resets to "failed"; the assertion reads only which
    rules are reported as failed.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from hhemt.workflow import SnakemakeWorkflowBuilder

    earlier_run = "localrule plot_y:\n    jobid: 4\n\nFinished jobid: 4 (Rule: plot_y)\n5 of 5 steps (100%) done\n"
    current_run = "localrule plot_y:\n    jobid: 4\n\nError in rule plot_y:\n    jobid: 4\n\n4 of 5 steps (80%) done\n"
    records = SnakemakeWorkflowBuilder._sweep_failed_rules(
        SimpleNamespace(), Path("/nonexistent-analysis-dir"), snakemake_stderr=earlier_run + current_run
    )
    assert "plot_y" in [r.get("rule_token") for r in records], (
        f"the current run's plot_y failure has no finish after it, so it must be reported; got {records!r}"
    )


def test_review_an_error_block_after_a_finish_is_failed_again():
    """Property: a pair that finished and then errored again, with no finish after that error, is failed.

    Class: logs in which a ``(jobid, rule)`` pair's LAST error block comes after its last
    ``Finished jobid: N (Rule: R)`` line, with an earlier error block before that finish. On the
    waited single-job path an appended fixed-name log reaches it: an earlier run whose job was
    retried to success, then a later run whose same-numbered job failed. Why it is required: the
    clearing finish must follow the LAST error, not merely some error; a finish that follows only
    an earlier attempt says nothing about the later one, so clearing on it reports a real failure
    as clean. What kills it: clearing a pair when a finish follows its FIRST error block, or when
    any finish exists for it. A second correct implementation, under which this still passes: walk
    the lines once and keep a per-pair state that a finish sets to "done" and every error block
    resets to "failed"; the assertion reads only which rules are reported as failed.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from hhemt.workflow import SnakemakeWorkflowBuilder

    earlier_run = "Error in rule plot_y:\n    jobid: 4\n\nTrying to restart job 4.\nFinished jobid: 4 (Rule: plot_y)\n"
    current_run = "Error in rule plot_y:\n    jobid: 4\n\n4 of 5 steps (80%) done\n"
    records = SnakemakeWorkflowBuilder._sweep_failed_rules(
        SimpleNamespace(), Path("/nonexistent-analysis-dir"), snakemake_stderr=earlier_run + current_run
    )
    assert "plot_y" in [r.get("rule_token") for r in records], (
        f"plot_y errored after its last finish, so it must be reported; got {records!r}"
    )
