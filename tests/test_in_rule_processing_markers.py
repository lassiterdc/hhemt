"""Option A terminal-marker contract — MOCKS ONLY. Not simulation-bearing: no compile
fixture, no Toolkit.run()/analysis.run(); the solver launch is a MagicMock asserted never
called.

Contract under the toggle (process_in_sim_rule=True):
  _submitted/{token}   written by the SIM runner at start (unchanged); NOT unlinked by it
  _completed/{token}   written by the PROCESS runner on rc 0
  _failed/{token}      written by the PROCESS runner on rc != 0 (and by the sim runner on a
                       SIM failure, unchanged)
  _submitted unlinked  by the PROCESS runner's finally
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hhemt import process_timeseries_runner as ptr
from hhemt.terminal_markers import rule_token_for, unlink_submitted_sentinel, write_terminal_marker


def _status_tree(tmp_path: Path, token: str) -> Path:
    status = tmp_path / "_status"
    (status / "_submitted").mkdir(parents=True)
    (status / "_submitted" / f"{token}.json").write_text(json.dumps({"slurm_jobid": "1"}))
    return status


def test_rule_token_matches_the_sim_runner_form() -> None:
    assert rule_token_for("run_triton", "year.1_event_type.rain_event_id.1") == (
        "run_triton_evt-year.1_event_type.rain_event_id.1"
    )


def test_write_terminal_marker_is_a_noop_without_a_jobid(tmp_path: Path) -> None:
    assert write_terminal_marker(tmp_path, "tok", status="completed", jobid=None) is None
    assert not (tmp_path / "_completed").exists()


@pytest.mark.parametrize(("rc", "expected"), [(0, "completed"), (1, "failed")])
def test_process_wrapper_owns_the_terminal_markers(tmp_path: Path, monkeypatch, rc: int, expected: str) -> None:
    token = rule_token_for("run_triton", "evt1")
    status = _status_tree(tmp_path, token)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_NAME", "run-uuid")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ptr",
            "--write-terminal-markers",
            "--rule-name",
            "run_triton",
            "--event-id",
            "evt1",
            "--flag-output",
            "_status/d_process_triton_evt-evt1_complete.flag",
        ],
    )
    monkeypatch.setattr(ptr, "main", lambda: rc)
    # During main() the _submitted sentinel is still present (the sim runner deferred).
    assert (status / "_submitted" / f"{token}.json").exists()
    assert ptr._main_with_terminal_markers() == rc
    marker = status / f"_{expected}" / f"{token}.json"
    assert marker.exists(), f"expected {marker}"
    other = "failed" if expected == "completed" else "completed"
    assert not (status / f"_{other}" / f"{token}.json").exists()
    assert not (status / "_submitted" / f"{token}.json").exists(), "process runner unlinks _submitted"
    payload = json.loads(marker.read_text())
    assert payload["status"] == expected and payload["slurm_jobid"] == "4242"


def test_process_wrapper_writes_failed_when_main_raises(tmp_path: Path, monkeypatch) -> None:
    token = rule_token_for("run_swmm", "evt2")
    status = _status_tree(tmp_path, token)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setattr(
        sys, "argv", ["ptr", "--write-terminal-markers", "--rule-name", "run_swmm", "--event-id", "evt2"]
    )

    def _boom():
        raise RuntimeError("processing exploded")

    monkeypatch.setattr(ptr, "main", _boom)
    with pytest.raises(RuntimeError):
        ptr._main_with_terminal_markers()
    assert (status / "_failed" / f"{token}.json").exists()
    assert not (status / "_submitted" / f"{token}.json").exists()


def test_process_wrapper_is_inert_without_the_flag(tmp_path: Path, monkeypatch) -> None:
    token = rule_token_for("run_triton", "evt3")
    status = _status_tree(tmp_path, token)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setattr(sys, "argv", ["ptr", "--rule-name", "run_triton", "--event-id", "evt3"])
    monkeypatch.setattr(ptr, "main", lambda: 0)
    assert ptr._main_with_terminal_markers() == 0
    assert not (status / "_completed").exists()
    assert (status / "_submitted" / f"{token}.json").exists(), "toggle OFF: the sim runner owns the unlink"


def test_unlink_submitted_is_idempotent(tmp_path: Path) -> None:
    unlink_submitted_sentinel(tmp_path / "_status", "absent")  # no raise


def test_retry_after_a_processing_failure_launches_no_solver(tmp_path: Path, monkeypatch) -> None:
    """PIN, not a regression test: the property Option A's safety rests on. A completed sim
    (model_run_completed True) makes prepare_simulation_command return None BEFORE any
    launch, so the combined rule's retry re-runs only the processing pass."""
    from hhemt.run_simulation import TRITONSWMM_run

    run = TRITONSWMM_run.__new__(TRITONSWMM_run)
    scenario = MagicMock()
    scenario.model_run_completed.return_value = True
    run._scenario = scenario
    analysis = MagicMock()
    analysis.cfg_analysis.multi_sim_run_method = "batch_job"
    analysis.cfg_analysis.execution_environment = "native"
    run._analysis = analysis
    # The two seams that would touch the filesystem on a mock: the model-log path
    # convention and the container-spec resolver. Everything else before the
    # already-completed check is attribute reads a MagicMock satisfies.
    monkeypatch.setattr(TRITONSWMM_run, "_analysis_level_model_logfile", lambda self, m: tmp_path / f"{m}.log")
    monkeypatch.setattr("hhemt.run_simulation.resolve_container_spec", lambda cfg: None)
    popen = MagicMock()
    monkeypatch.setattr("subprocess.Popen", popen)
    result = run.prepare_simulation_command(
        pickup_where_leftoff=True, verbose=False, model_type="tritonswmm", execution_locus="local"
    )
    assert result is None
    popen.assert_not_called()
    scenario.model_run_completed.assert_called_with("tritonswmm")
