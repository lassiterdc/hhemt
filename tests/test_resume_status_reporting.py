"""resume-retry-resilience Phase 3 — env-independent tests for the resume
observability surface: ``_print_resume_status`` alignment (#5 canonical-flag
count + #3 per-sim retry-budget print) and the ``_warn_resume_zero_progress``
launch-time advisory (friction Option A).

These exercise the methods directly with lightweight ``SimpleNamespace`` stubs
(no compiled TRITON-SWMM pipeline), substituting for the end-to-end synth_01/04
assertions the plan's Validation Plan names but which require the C++ build
(unrunnable in this worktree — pre-existing environment limit).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hhemt.analysis import TRITONSWMM_analysis

# ---------------------------------------------------------------------------
# _warn_resume_zero_progress (friction Option A) — advisory, no numeric floor
# ---------------------------------------------------------------------------


def _warn_stub(multi_sim_run_method: str, hpc_time_min_per_sim):
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            multi_sim_run_method=multi_sim_run_method,
            hpc_time_min_per_sim=hpc_time_min_per_sim,
        )
    )


@pytest.mark.parametrize(
    "pickup,method,walltime,expect_warn",
    [
        (True, "batch_job", 5, True),
        (True, "1_job_many_srun_tasks", 5, True),
        (False, "batch_job", 5, False),  # not resuming -> no warning
        (True, "local", 5, False),  # not an HPC mode -> no warning
        (True, "batch_job", None, False),  # no per-sim walltime -> no warning
    ],
)
def test_warn_resume_zero_progress(capsys, pickup, method, walltime, expect_warn):
    stub = _warn_stub(method, walltime)
    TRITONSWMM_analysis._warn_resume_zero_progress(stub, pickup)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    if expect_warn:
        assert "WARNING" in out
        assert "ZERO" in out
        assert "pickup_where_leftoff=True" in out
        assert f"hpc_time_min_per_sim={walltime}" in out
    else:
        assert out == ""


# ---------------------------------------------------------------------------
# _print_resume_status — #5 canonical-flag count + #3 attempt-budget print
# ---------------------------------------------------------------------------


def _resume_stub(tmp_path, *, sensitivity, nsims, hpc_restart_times_simulate=2):
    status_dir = tmp_path / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    stub = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(
                toggle_tritonswmm_model=True,
                toggle_triton_model=False,
                toggle_swmm_model=False,
            )
        ),
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=sensitivity,
            analysis_id="testA",
            hpc_restart_times_simulate=hpc_restart_times_simulate,
            # batch_job so the 1_job_many_srun_tasks node-recommendation block
            # (which needs heavier stubs) is skipped.
            multi_sim_run_method="batch_job",
        ),
        nsims=nsims,
    )
    return stub, status_dir


def test_print_resume_status_first_run_silent(capsys, tmp_path):
    """No flags present -> first-run silence (no print)."""
    stub, _ = _resume_stub(tmp_path, sensitivity=True, nsims=3)
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    assert capsys.readouterr().out == ""


def test_print_resume_status_sensitivity_count_and_attempt_budget(capsys, tmp_path):
    """Sensitivity path counts sa completion flags; prints aligned N/M + budget."""
    stub, status_dir = _resume_stub(
        tmp_path, sensitivity=True, nsims=3, hpc_restart_times_simulate=2
    )
    (status_dir / "c_run_tritonswmm_sa-0_evt-0_complete.flag").write_text("")
    (status_dir / "c_run_tritonswmm_sa-1_evt-0_complete.flag").write_text("")
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Resuming testA — 2/3 sims complete." in out
    # attempts = hpc_restart_times_simulate + 1 = 3
    assert "Each sim gets 3 attempt(s)" in out
    assert "hpc_restart_times_simulate=2" in out


def test_print_resume_status_nonsensitivity_canonical_flag_count(capsys, tmp_path, monkeypatch):
    """Non-sensitivity path counts the canonical c_run_{model}_evt-{id}_complete
    flags (the authoritative DAG completion set), not an ad-hoc wildcard glob."""
    import hhemt.scenario as scenario_mod

    monkeypatch.setattr(scenario_mod, "compute_event_id_slug", lambda ev: str(ev))
    stub, status_dir = _resume_stub(tmp_path, sensitivity=False, nsims=2)
    stub.df_sims = [object(), object()]  # len() == 2
    stub._retrieve_weather_indexer_using_integer_index = lambda i: i
    # Seed the canonical evt flag for event 0 only -> 1 of 2 complete.
    (status_dir / "c_run_tritonswmm_evt-0_complete.flag").write_text("")
    TRITONSWMM_analysis._print_resume_status(stub)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Resuming testA — 1/2 sims complete." in out
    assert "Each sim gets 3 attempt(s)" in out
