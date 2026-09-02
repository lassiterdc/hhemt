"""Regression tests for the _status/_queued/ adoption path.

Each test here fails on the pre-fix source for a stated, specific reason; the
reason is recorded in the test's own docstring so a future reader can tell a
genuine regression from a fixture drift.
"""

import json
import os
import time
from unittest.mock import Mock

import pytest

from hhemt.workflow import SnakemakeWorkflowBuilder


@pytest.fixture
def builder(tmp_path):
    analysis = Mock()
    analysis.cfg_analysis = Mock()
    analysis.cfg_analysis.multi_sim_run_method = "batch_job"
    analysis.cfg_analysis.analysis_id = "adopt"
    analysis.cfg_analysis.hpc_total_job_duration_min = 360
    analysis.cfg_analysis.hpc_time_min_per_sim = 30
    analysis.cfg_analysis.hpc_max_wait_for_inflight_min = 10080
    analysis.cfg_analysis.hpc_max_queue_wait_min = None
    analysis.cfg_analysis.hpc_no_progress_timeout_min = None
    analysis.analysis_paths = Mock()
    analysis.analysis_paths.analysis_dir = tmp_path
    _logs = tmp_path / "logs"
    _logs.mkdir()
    analysis.analysis_paths.analysis_log_directory = _logs
    analysis._refresh_log = Mock()
    (tmp_path / "_status").mkdir()
    return SnakemakeWorkflowBuilder(analysis)


def test_queued_sentinels_are_written_before_the_blocking_wait(builder, monkeypatch):
    """PRE-FIX FAILURE: the _queued/ write lived at the CALLER, behind
    `if isinstance(result, dict) and result.get("success", True)`, and
    _submit_tmux_workflow overwrites result_dict["success"] with
    completion_info["completed"] before returning -- so on a queue-starved run the
    guard is False and nothing is written. events would be ["wait"], not
    ["launch", "wait"].

    The assertion is ORDERED, not a call count: D1's content is that the write
    happens at launch-success BEFORE the optional blocking wait, so moving
    on_launch below the `if wait_for_completion:` block must fail this test. A
    len(calls) == 1 assertion would pass on that regression.
    """
    events = []
    monkeypatch.setattr(type(builder), "_get_module_load_prefix", lambda self: "")
    monkeypatch.setattr(type(builder), "_get_snakemake_pid_from_tmux", lambda self, s: 4242)
    monkeypatch.setattr("hhemt.workflow.time.sleep", lambda *_a, **_k: None)

    def _fake_run(argv, **_k):
        joined = " ".join(argv) if isinstance(argv, list) else str(argv)
        # `tmux has-session` returning 0 means the session already exists, which the
        # method treats as a fatal collision; every other shell-out succeeds.
        return Mock(returncode=1 if "has-session" in joined else 0, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _fake_run)

    def _fake_wait(self, **_k):
        events.append("wait")
        return {"completed": False, "message": "Workflow queue-starved: ..."}

    monkeypatch.setattr(type(builder), "_wait_for_tmux_session_completion", _fake_wait)

    result = builder._submit_tmux_workflow(
        snakefile_path=builder.analysis_paths.analysis_dir / "Snakefile",
        wait_for_completion=True,
        verbose=False,
        on_launch=lambda _jid: events.append("launch"),
    )
    assert result["success"] is False
    assert events == ["launch", "wait"], events


def test_queued_sentinel_survives_a_queue_wait_longer_than_the_sim_walltime(builder):
    """PRE-FIX FAILURE: the age-out bound is _max_plausible_job_lifetime_min =
    hpc_total_job_duration_min + 30 = 390 min (6.5 h). A 10-hour-old sentinel
    exceeds it, so the executor-owns branch unlinks the file and continues --
    the returned list is empty and the file is gone, failing both asserts.
    These are the live Frontier arm's numbers."""
    qdir = builder.analysis_paths.analysis_dir / "_status" / "_queued"
    qdir.mkdir(parents=True)
    tok = "run_triton_evt-e0"
    qpath = qdir / f"{tok}.json"
    qpath.write_text(json.dumps({"rule_token": tok, "slurm_jobid": None}, sort_keys=True))
    old = time.time() - 10 * 3600
    os.utime(qpath, (old, old))
    recovered = builder._recover_pending_from_queued([tok], builder.analysis_paths.analysis_dir)
    assert [t for t, _ in recovered] == [tok], "a job queued 10 h must not be aged out at 6.5 h"
    assert qpath.exists(), "the sentinel must not be unlinked by the age-out"


def test_liveness_query_names_both_the_current_and_the_adopted_run_uuid(builder, monkeypatch):
    """PRE-FIX FAILURE: _workflow_has_live_slurm_jobs takes no extra_run_uuids
    parameter, so the call raises TypeError before any assertion runs. The
    assertion-bearing half is the union FORM of the --name argument, which is
    what makes a resumed driver able to see the previous driver's queued jobs."""
    seen = {}

    def _fake_run(argv, **kwargs):
        seen["argv"] = argv
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _fake_run)
    monkeypatch.setattr(type(builder), "_tmux_slurm_run_uuids", lambda self: ("current-uuid",))
    prior = "cb4cc5fb-b338-409e-8a54-4aba886c3284"
    builder._workflow_has_live_slurm_jobs(extra_run_uuids=(prior,))
    name_arg = seen["argv"][seen["argv"].index("--name") + 1]
    assert "current-uuid" in name_arg and prior in name_arg


def test_queue_starvation_clock_does_not_run_while_adopted_work_is_tracked(builder, monkeypatch):
    """PRE-FIX FAILURE: _wait_for_tmux_session_completion takes no
    adopted_token_count parameter, so the call raises TypeError. Post-fix, a
    driver whose live set reads {'PENDING'} purely because its adopted jobs
    belong to a previous run must not accumulate all_pending_since at all."""
    monkeypatch.setattr(type(builder), "_get_module_load_prefix", lambda self: "")
    monkeypatch.setattr(type(builder), "_workflow_has_live_slurm_jobs", lambda self, **k: {"PENDING"})
    monkeypatch.setattr(type(builder), "_tmux_snakemake_exit_status", lambda self, s: 0)
    seq = iter([Mock(returncode=0, stdout="", stderr=""), Mock(returncode=1, stdout="", stderr="")])
    monkeypatch.setattr("hhemt.workflow.subprocess.run", lambda *a, **k: next(seq))
    builder.cfg_analysis.hpc_max_queue_wait_min = 30
    out = builder._wait_for_tmux_session_completion(
        session_name="s", verbose=False, poll_interval_s=0, adopted_token_count=3
    )
    assert "queue-starved" not in out["message"]


def test_submit_tmux_threads_adopted_token_count_into_the_wait(builder, monkeypatch):
    """PRE-FIX FAILURE: _submit_tmux_workflow takes no adopted_token_count, so the
    call raises TypeError. The assertion-bearing half is that the value REACHES
    _wait_for_tmux_session_completion -- the suppression arm is inert until it does,
    which is the gap a method-level test of the arm alone cannot see."""
    seen = {}
    monkeypatch.setattr(type(builder), "_get_module_load_prefix", lambda self: "")
    monkeypatch.setattr(type(builder), "_get_snakemake_pid_from_tmux", lambda self, s: 1)
    monkeypatch.setattr("hhemt.workflow.time.sleep", lambda *_a, **_k: None)

    def _fake_run(argv, **_k):
        joined = " ".join(argv) if isinstance(argv, list) else str(argv)
        return Mock(returncode=1 if "has-session" in joined else 0, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _fake_run)

    def _fake_wait(self, **kw):
        seen.update(kw)
        return {"completed": True, "message": "ok"}

    monkeypatch.setattr(type(builder), "_wait_for_tmux_session_completion", _fake_wait)
    builder._submit_tmux_workflow(
        snakefile_path=builder.analysis_paths.analysis_dir / "Snakefile",
        wait_for_completion=True,
        verbose=False,
        adopted_token_count=7,
    )
    assert seen.get("adopted_token_count") == 7


def test_liveness_names_the_prior_driver_uuid_from_the_log_set_with_no_caller_capture(builder, monkeypatch):
    """PRE-FIX FAILURE: _workflow_has_live_slurm_jobs read only the NEWEST log's
    LAST uuid, so a resumed driver saw only its own. The adopted uuid then had to be
    captured at the call site BEFORE the new log was created -- correctness by read
    ORDERING, invisible in the code and silently lost to any edit that moves the log
    creation earlier. Sourcing the name set from the whole log SET removes the
    ordering constraint: no caller captures anything and both uuids are named
    whether or not the current driver's log exists yet."""
    seen = {}

    def _fake_run(argv, **_k):
        seen["argv"] = argv
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _fake_run)
    logs = builder.analysis_paths.analysis_log_directory
    prior = "cb4cc5fb-b338-409e-8a54-4aba886c3284"
    current = "11111111-2222-3333-4444-555555555555"
    (logs / "tmux_session_20260101_000000.log").write_text(f"SLURM run ID: {prior}\n")
    (logs / "tmux_session_20260102_000000.log").write_text(f"SLURM run ID: {current}\n")
    builder._workflow_has_live_slurm_jobs()
    name_arg = seen["argv"][seen["argv"].index("--name") + 1]
    assert prior in name_arg and current in name_arg, name_arg


def test_multisim_batch_job_call_site_passes_the_adopted_token_count(builder, monkeypatch):
    """PRE-FIX FAILURE: the multisim batch_job call site passed no
    adopted_token_count, so the parameter defaulted to 0 and the suppression arm
    was unreachable from production however well the arm itself was unit-tested.
    Removing the kwarg makes both asserts below fail on the VALUE, not on a
    TypeError -- the test discriminates what was passed, not merely that the
    parameter exists.

    The four patches sit at genuine seams (reconcile, snakefile generation, the
    dry-run subprocess, the submit). Everything between `alive_by_token = ...`
    and the call site is real source, which is what makes this a test of the code
    rather than of the harness. The second assert is the load-bearing one: it
    proves the cap and the Snakefile read the SAME binding, which is the claim the
    call site's own comment makes."""
    seen = {}
    monkeypatch.setattr(
        type(builder),
        "_reconcile_inflight_submissions",
        lambda self, analysis_dir=None: [("run_triton_evt-a", ""), ("run_triton_evt-b", ""), ("run_triton_evt-c", "")],
    )
    gen = {}

    def _fake_gen(self, **kw):
        gen.update(kw)
        return "rule all:\n    input: []\n"

    monkeypatch.setattr(type(builder), "generate_snakefile_content", _fake_gen)
    monkeypatch.setattr(type(builder), "_validate_batch_job_dry_run", lambda self, **kw: {"success": True})

    def _fake_submit(self, **kw):
        seen.update(kw)
        return {"success": True, "mode": "tmux"}

    monkeypatch.setattr(type(builder), "_submit_tmux_workflow", _fake_submit)
    builder.submit_workflow(dry_run=False, verbose=False)
    assert seen.get("adopted_token_count") == 3, seen
    assert len(gen["alive_by_token"]) == seen["adopted_token_count"]


@pytest.fixture
def sens_builder(tmp_path):
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder

    master = Mock()
    master.cfg_analysis = Mock()
    master.cfg_analysis.multi_sim_run_method = "batch_job"
    master.analysis_paths = Mock()
    master.analysis_paths.analysis_dir = tmp_path
    logs = tmp_path / "logs"
    logs.mkdir()
    master.analysis_paths.analysis_log_directory = logs
    master._resource_manager._get_simulation_resource_requirements.return_value = {"n_gpus": 0}
    sa = Mock()
    sa.experiment = master
    return SensitivityAnalysisWorkflowBuilder(sa)


def test_sensitivity_batch_job_call_site_passes_the_adopted_token_count(sens_builder, monkeypatch):
    """Same contract as the multisim site, and a SEPARATE test because the
    producer differs: the sensitivity alive set comes from
    _reconcile_sensitivity_alive, which sweeps every sub-analysis dir, not from
    the multisim _reconcile_inflight_submissions. A single test of one site would
    leave the other's binding unexercised.

    The two _base_builder seams are patched on the CLASS rather than the instance
    because the sensitivity builder composes a SnakemakeWorkflowBuilder rather
    than inheriting from it."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    seen = {}
    monkeypatch.setattr(
        type(sens_builder),
        "_reconcile_sensitivity_alive",
        lambda self: ({"simulation_sa_0_evt-a": "", "simulation_sa_1_evt-b": ""}, {}),
    )
    gen = {}

    def _fake_gen(self, **kw):
        gen.update(kw)
        return "rule all:\n    input: []\n"

    monkeypatch.setattr(type(sens_builder), "generate_master_snakefile_content", _fake_gen)
    monkeypatch.setattr(SnakemakeWorkflowBuilder, "_validate_batch_job_dry_run", lambda self, **kw: {"success": True})

    def _fake_submit(self, **kw):
        seen.update(kw)
        return {"success": True, "mode": "tmux"}

    monkeypatch.setattr(SnakemakeWorkflowBuilder, "_submit_tmux_workflow", _fake_submit)
    sens_builder.submit_workflow(dry_run=False, verbose=False)
    assert seen.get("adopted_token_count") == 2, seen
    assert len(gen["alive_by_token"]) == seen["adopted_token_count"]
