"""FQ5/FQ6 (2026-09-12): _queued/ re-keying from the tmux orchestrator log, the reconcile's
rewrite/unlink/hold outcomes, and the DU-guard cancel set. HPC-free: no SLURM, no solver.

EXCERPT is a VERBATIM excerpt of a real UVA orchestrator log (tmux_session_20260912_021757.log,
the 2026-09-12 stochastic launch), trimmed to the lines the parser reads: the `jobid:`-before-
`wildcards:` order and the trailing `(log: …)` are what the parser sees. Two whitespace-bearing
lines of the original (`Shell command: ` and its blank continuation) are stripped: the parser reads
only column-0 lines, `jobid:` lines and flag-path lines, so they carry no signal. The account
(`myalloc`) and username (`user`) in the excerpt are scrubbed placeholders (public-repo
anonymization guard, Gotcha 55); the parser reads neither.
"""

# ruff: noqa: E501  (EXCERPT carries verbatim orchestrator-log lines longer than 120 columns)

from __future__ import annotations

import json
import os
import subprocess
import time
import types
from pathlib import Path

import pytest

from hhemt import workflow as wf

EXCERPT = """\
15 of 30396 steps (0.0%) done
Execute 14 jobs...

[Sat Sep 12 02:29:51 2026]
rule run_tritonswmm:
    input: _status/b_prepare_evt-year.223_event_type.surge_event_id.3_complete.flag
    output: _status/c_run_tritonswmm_evt-year.223_event_type.surge_event_id.3_complete.flag
    jobid: 19992
    reason: Missing output files: _status/c_run_tritonswmm_evt-year.223_event_type.surge_event_id.3_complete.flag
    wildcards: event_id=year.223_event_type.surge_event_id.3
    threads: 4
    resources: mem_mb=16000, mem_mib=15259, disk_mb=1000, disk_mib=954, tmpdir=<TBD>, nodes=1, runtime=480, slurm_partition=standard, slurm_account=myalloc, tasks=2, cpus_per_task=2, mpi=True
Job 19992 has been submitted with SLURM jobid 19668222 (log: /sfs/weka/scratch/user/norfolk_hhemt_experiments/event_ensembles/stochastic/norfolk_stochastic/.snakemake/slurm_logs/rule_run_tritonswmm/year.223_event_type.surge_event_id.3/19668222.log).
rule prepare_scenario:
    output: _status/b_prepare_evt-year.5_event_type.rain_event_id.1_complete.flag
    jobid: 41
    wildcards: event_id=year.5_event_type.rain_event_id.1
Job 41 has been submitted with SLURM jobid 19668300 (log: x/rule_prepare_scenario/year.5_event_type.rain_event_id.1/19668300.log).
rule simulation_member_0_evt_event_index_0:
    output: members/member_0/_status/c_run_tritonswmm_member-0_evt-event_index_0_complete.flag
    jobid: 7
Job 7 has been submitted with SLURM jobid 19668400 (log: x/rule_simulation_member_0_evt_event_index_0/19668400.log).
Trying to restart job 19992.
rule run_tritonswmm:
    output: _status/c_run_tritonswmm_evt-year.223_event_type.surge_event_id.3_complete.flag
    jobid: 19992
    wildcards: event_id=year.223_event_type.surge_event_id.3
Job 19992 has been submitted with SLURM jobid 19670001 (log: x/rule_run_tritonswmm/year.223_event_type.surge_event_id.3/19670001.log).
"""

TOK = "run_tritonswmm_evt-year.223_event_type.surge_event_id.3"


def test_parse_tmux_submissions_real_excerpt_newest_wins():
    # Invariant: exactly the SIM tokens the executor submitted map to their NEWEST slurm jobid;
    # prepare/process rules are ignored; a member rule maps to the runner's `_evt-` token.
    got = wf.parse_tmux_submissions(EXCERPT)
    assert got == {
        TOK: "19670001",  # the retry's jobid overwrote 19668222
        "simulation_member_0_evt-event_index_0": "19668400",
    }
    # differently-positioned satisfying input: a submission whose `jobid:` block never appeared
    assert wf.parse_tmux_submissions("Job 5 has been submitted with SLURM jobid 1 (log: x).\n") == {}


def test_parse_tmux_submissions_adversarial_blocks():
    dotted_member = (
        "rule simulation_member_0_evt_year_223_event_type_surge_event_id_3:\n"
        "    output: members/member_0/_status/c_run_tritonswmm_member-0_evt-year.223_event_type.surge_event_id.3_complete.flag\n"
        "    jobid: 12\nJob 12 has been submitted with SLURM jobid 555 (log: x).\n"
    )
    assert (
        wf.parse_tmux_submissions(dotted_member)
        == {
            "simulation_member_0_evt-year.223_event_type.surge_event_id.3": "555"  # the RUNNER's token, not the sanitized rule name
        }
    )
    error_block = (
        "rule run_triton:\n    output: _status/c_run_triton_evt-A_complete.flag\n    jobid: 3\n    wildcards: event_id=A\n"
        "Job 3 has been submitted with SLURM jobid 200 (log: x).\n"
        "Error in rule run_swmm:\n    jobid: 9\n    output: _status/c_run_swmm_evt-B_complete.flag\n"
        "Job 9 has been submitted with SLURM jobid 300 (log: x).\n"
    )
    assert wf.parse_tmux_submissions(error_block) == {"run_triton_evt-A": "200"}  # jobid 9 never attributed
    group_block = (
        "rule run_tritonswmm:\n    output: _status/c_run_tritonswmm_evt-C_complete.flag\n    jobid: 4\n"
        "Group job process_evt_C (jobs in lexicogr. order):\n    rule process_triton:\n"
        "        input: _status/c_run_triton_evt-C_complete.flag\n        jobid: 77\n"
        "Job 77 has been submitted with SLURM jobid 400 (log: x/group_process_evt_C/400.log).\n"
        "Job 4 has been submitted with SLURM jobid 401 (log: x).\n"
    )
    assert wf.parse_tmux_submissions(group_block) == {"run_tritonswmm_evt-C": "401"}
    two_output_and_ignored_rules = (
        "rule run_swmm:\n    input: _status/b_prepare_evt-D_complete.flag\n    output:\n"
        '        c_run="_status/c_run_swmm_evt-D_complete.flag",\n        d_process="_status/d_process_swmm_evt-D_complete.flag"\n'
        "    jobid: 5\nJob 5 has been submitted with SLURM jobid 500 (log: x).\n"
        "rule process_triton:\n    input: _status/c_run_triton_evt-E_complete.flag\n    jobid: 6\n"
        "Job 6 has been submitted with SLURM jobid 600 (log: x).\n"
        "rule wait_for_run_triton_evt_F:\n    output: _status/c_run_triton_evt-F_complete.flag\n    jobid: 8\n"
        "Job 8 has been submitted with SLURM jobid 800 (log: x).\n"
    )
    assert wf.parse_tmux_submissions(two_output_and_ignored_rules) == {"run_swmm_evt-D": "500"}


def _tree(tmp_path: Path, tokens: dict[str, str | None], *, log_text: str | None) -> tuple[Path, Path | None]:
    qdir = tmp_path / "_status" / "_queued"
    qdir.mkdir(parents=True)
    for tok, jid in tokens.items():
        (qdir / f"{tok}.json").write_text(json.dumps({"rule_token": tok, "slurm_jobid": jid}, sort_keys=True))
    log_dir = None
    if log_text is not None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "tmux_session_20260912_021757.log").write_text(log_text)
    return tmp_path, log_dir


def test_rekey_rewrites_unlinks_and_noops(tmp_path: Path):
    planned_only = "run_swmm_evt-year.0_event_type.compound_event_id.1"
    root, log_dir = _tree(tmp_path, {TOK: None, planned_only: None}, log_text=EXCERPT)
    qdir = root / "_status" / "_queued"
    survivors = wf._rekey_queued_from_tmux_logs(qdir, [TOK, planned_only], log_dir)
    assert survivors == [TOK]
    assert json.loads((qdir / f"{TOK}.json").read_text())["slurm_jobid"] == "19670001"  # rewritten
    assert not (qdir / f"{planned_only}.json").exists()  # planned, never submitted -> unlinked
    # NO-OP arms: no log dir, and an empty log dir -> nothing unlinked, nothing rewritten.
    root2, _ = _tree(tmp_path / "b", {planned_only: None}, log_text=None)
    q2 = root2 / "_status" / "_queued"
    assert wf._rekey_queued_from_tmux_logs(q2, [planned_only], None) == [planned_only]
    assert (q2 / f"{planned_only}.json").exists()
    (root2 / "logs").mkdir()
    assert wf._rekey_queued_from_tmux_logs(q2, [planned_only], root2 / "logs") == [planned_only]
    assert (q2 / f"{planned_only}.json").exists()


def _fake_self(tmp_path: Path, log_dir: Path | None, uuids=("de2ee33e-3d23-4a79-8395-c966d88684f1",)):
    return types.SimpleNamespace(
        cfg_analysis=types.SimpleNamespace(hpc_max_wait_for_inflight_min=10080),
        analysis_paths=types.SimpleNamespace(analysis_log_directory=log_dir if log_dir else tmp_path / "nolog"),
        _tmux_slurm_run_uuids=lambda: uuids,
    )


@pytest.mark.parametrize(
    ("sacct_rows", "squeue_live", "expect_held", "expect_file"),
    [
        ({"19670001": ("CANCELLED", "0:0", "None")}, {"19670001"}, False, False),  # sacct DEAD -> unlink
        ({"19670001": ("PENDING", "0:0", "None")}, set(), True, True),  # sacct alive -> hold
        ({}, {"19670001"}, True, True),  # sacct unknown, squeue live -> hold
        ({}, set(), False, False),  # sacct unknown, squeue known-absent -> unlink
        ({}, None, True, True),  # sacct unknown, squeue unavailable -> mtime tier (fresh) -> hold
    ],
)
def test_recover_pending_tiers(tmp_path: Path, monkeypatch, sacct_rows, squeue_live, expect_held, expect_file):
    root, log_dir = _tree(tmp_path, {TOK: None}, log_text=EXCERPT)
    monkeypatch.setattr(wf, "_sacct_states_batched", lambda jids, **kw: sacct_rows)
    monkeypatch.setattr(wf, "_squeue_live_jobids", lambda uuids, **kw: squeue_live)
    self_ = _fake_self(tmp_path, log_dir)
    out = wf.SnakemakeWorkflowBuilder._recover_pending_from_queued(self_, [TOK], root)
    assert ([t for t, _ in out] == [TOK]) is expect_held
    assert (root / "_status" / "_queued" / f"{TOK}.json").exists() is expect_file


def test_recover_pending_ages_out_unrekeyed_token_PIN(tmp_path: Path, monkeypatch):
    # PIN of pre-existing behaviour (green pre- and post-fix): a null-jid token with NO tmux log
    # at all is governed by the hold/mtime tiers; older than the cap -> unlinked.
    tok = "run_triton_evt-old"
    root, _ = _tree(tmp_path, {tok: None}, log_text=None)
    q = root / "_status" / "_queued" / f"{tok}.json"
    stale = time.time() - 10081 * 60
    os.utime(q, (stale, stale))
    monkeypatch.setattr(wf, "_sacct_states_batched", lambda jids, **kw: {})
    monkeypatch.setattr(wf, "_squeue_live_jobids", lambda uuids, **kw: None)
    out = wf.SnakemakeWorkflowBuilder._recover_pending_from_queued(_fake_self(tmp_path, None), [tok], root)
    assert out == [] and not q.exists()


def test_selective_cancel_widens_skips_reclaiming_and_scancels_before_kill(tmp_path: Path, monkeypatch):
    from hhemt import analysis as an

    status = tmp_path / "_status"
    (status / "_submitted").mkdir(parents=True)
    (status / "_submitted" / "run_triton_evt-a.json").write_text(json.dumps({"slurm_jobid": "111"}))
    logs = tmp_path / "logs"
    logs.mkdir()
    # two observed PENDING sims: one still pending (cancel), one whose sim finished (skip)
    (logs / "tmux_session_x.log").write_text(
        "rule run_swmm:\n    output: _status/c_run_swmm_evt-b_complete.flag\n    jobid: 1\n"
        "Job 1 has been submitted with SLURM jobid 222 (log: x).\n"
        "rule run_triton:\n    output: _status/c_run_triton_evt-c_complete.flag\n    jobid: 2\n"
        "Job 2 has been submitted with SLURM jobid 333 (log: x).\n"
    )
    (status / "c_run_triton_evt-c_complete.flag").write_text("")
    calls: list[list[str]] = []

    def _record(argv, **kw):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=0, stdout="")

    # _selective_cancel imports `subprocess` function-locally (analysis.py's module style), so the
    # patch must land on the subprocess MODULE, which that import resolves at call time.
    monkeypatch.setattr(subprocess, "run", _record)
    monkeypatch.setattr(wf, "_squeue_live_jobids", lambda uuids, **kw: {"222", "333"})
    self_ = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path, analysis_log_directory=logs),
        _workflow_builder=types.SimpleNamespace(_tmux_slurm_run_uuids=lambda: ("u",)),
    )
    result = an.TRITONSWMM_analysis._selective_cancel(self_, rule_classes=("run_",), session_name="s", verbose=False)
    assert result["cancelled"] == [("run_triton_evt-a", "111"), ("run_swmm_evt-b", "222")]
    assert calls[0][0] == "scancel" and set(calls[0][1:]) == {"111", "222"}
    assert calls[1][:2] == ["tmux", "kill-session"]  # AFTER scancel
