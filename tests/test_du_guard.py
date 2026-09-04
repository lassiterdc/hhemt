"""The DU guard fires, spares processing, and is ABSENT when unconfigured."""

from __future__ import annotations

import json
import os
import types

from hhemt.workflow import SnakemakeWorkflowBuilder


def _fake_statvfs(free_bytes: int, calls: list):
    def _f(path):
        calls.append(str(path))
        return types.SimpleNamespace(f_bavail=free_bytes, f_frsize=1, f_blocks=10**13)

    return _f


def _builder(tmp_path, *, floor: int | None):
    """Returns (builder, cancels, fp_calls). fp_calls is the POSITIVE CONTROL."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    b = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    b.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    b.cfg_hpc_system = types.SimpleNamespace(halt_sims_below_free_bytes=floor)
    b.cfg_analysis = types.SimpleNamespace(multi_sim_run_method="batch_job")
    cancels: list[dict] = []
    fp_calls: list[int] = []

    def _stub_cancel(verbose=True, **kw):
        cancels.append(kw)
        return {"success": True, "cancelled": [("run_triton_evt-0", "111")]}

    b.analysis = types.SimpleNamespace(cancel=_stub_cancel)

    def _stub_fp():
        fp_calls.append(1)
        return "fp"

    b._status_progress_fingerprint = _stub_fp
    b._workflow_has_live_slurm_jobs = lambda **kw: set()
    b._tmux_snakemake_exit_status = lambda s: 0
    b._get_module_load_prefix = lambda: ""
    return b, cancels, fp_calls


def _drive(monkeypatch, alive_polls: int = 1):
    """tmux has-session: ALIVE for `alive_polls`, then GONE."""
    state = {"n": 0}

    def _run(argv, *a, **kw):
        state["n"] += 1
        rc = 0 if state["n"] <= alive_polls else 1
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _run)


def test_guard_fires_below_the_floor(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    result = b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert result["completed"] is False
    assert "DU guard" in result["message"]
    assert calls, "the guard never called statvfs"
    assert len(cancels) == 1


def test_guard_is_absent_when_the_field_is_unset(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=None)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    # >= 2, NOT truthiness: _status_progress_fingerprint is ALSO called PRE-LOOP at
    # workflow.py:6078, so a non-empty list is consistent with the loop exiting
    # before the guard site. The second call is the in-loop one at :6111, one line
    # after the guard at :6110 -- only that one proves control passed through.
    assert len(fp_calls) >= 2, f"guard site never reached; fp_calls={len(fp_calls)}"
    assert calls == [], f"statvfs was called on the unset path: {calls}"
    assert cancels == []
    assert not (tmp_path / "analysis" / "_status" / "_halted_du.json").exists()


def test_guard_halts_sims_and_names_only_sim_rule_classes(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert cancels[0]["rule_classes"] == ("run_", "simulation_member_")
    assert "process_" not in cancels[0]["rule_classes"]
    assert "consolidate_" not in cancels[0]["rule_classes"]


def test_selective_cancel_spares_processing_jobs(tmp_path, monkeypatch):
    from hhemt.analysis import TRITONSWMM_analysis

    adir = tmp_path / "analysis"
    sub = adir / "_status" / "_submitted"
    sub.mkdir(parents=True)
    for tok, jid in [
        ("run_triton_evt-0", "111"),
        ("simulation_member_3_evt-7", "222"),
        ("process_tritonswmm_evt-0", "333"),
        ("consolidate_scenario_evt-0", "444"),
    ]:
        (sub / f"{tok}.json").write_text(json.dumps({"rule_token": tok, "slurm_jobid": jid}))

    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir)
    # cancel() reads BOTH log fields and analysis_id before reaching the selective
    # arm, so the fake must carry all three. snakemake_pid is read but unused on
    # this arm -- see the ordering finding in the deliverable.
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: "sess"),
        snakemake_pid=types.SimpleNamespace(get=lambda: 4242),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="test_analysis")
    # Step 1 resolves the live PID through the builder BEFORE the selective arm is
    # reached, even though that arm never uses it. Stubbed here; the pass-through
    # dependency is recorded as an ordering finding in the deliverable.
    a._workflow_builder = types.SimpleNamespace(_get_snakemake_pid_from_tmux=lambda s: 4242)

    argvs: list[list[str]] = []

    def _fake_run(argv, *args, **kw):
        argvs.append(list(argv))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    # Patch the subprocess MODULE, not an attribute of hhemt.analysis. Measured,
    # in two steps: the dotted string form makes pytest try to IMPORT
    # `hhemt.analysis.subprocess` (ImportError), and the object form fails with
    # AttributeError because analysis.py has NO module-scope `import subprocess`
    # -- cancel() imports it function-locally. That local import binds the same
    # module object, so patching subprocess.run here reaches it.
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _fake_run)
    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    scancels = [v for v in argvs if v and v[0] == "scancel"]
    assert len(scancels) == 1, f"expected one scancel, got {argvs}"
    cancelled = set(scancels[0][1:])
    assert cancelled == {"111", "222"}, f"wrong job set cancelled: {cancelled}"
    assert "333" not in cancelled, "cancelled a PROCESSING job -- [Q212] violated"
    assert "444" not in cancelled, "cancelled a CONSOLIDATE job -- [Q212] violated"
    assert res["success"] is True
    assert any(v[:2] == ["tmux", "kill-session"] for v in argvs)


def test_the_halt_record_does_not_latch(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    record = tmp_path / "analysis" / "_status" / "_halted_du.json"
    assert record.exists()
    payload = json.loads(record.read_text())
    assert payload["free_bytes_observed"] == 1 * 10**9
    assert payload["floor_bytes"] == 100 * 10**9
    assert payload["du_sentinel_total_bytes"] is None

    b2, cancels2, fp2 = _builder(tmp_path / "second", floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(500 * 10**9, calls))
    _drive(monkeypatch)
    result = b2._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert len(fp2) >= 2, f"guard site never reached on the second evaluation; fp_calls={len(fp2)}"
    assert cancels2 == [], "the guard fired with 500 GB free -- it latched"
    assert result["completed"] is True
    assert record.exists(), "the record was consumed; it is disclosure, not state"
