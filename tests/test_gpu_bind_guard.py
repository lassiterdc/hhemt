"""V-D unit tests for hhemt.gpu_bind_guard and its runner/df_status/validator consumers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hhemt import gpu_bind_guard as gb

# Verbatim per-task records from UVA probe 20350116 (gpu-a6000, 2026-09-21), V-B step.
_VB_TASKS = [
    {
        "procid": 0,
        "localid": 0,
        "host": "udc-an38-1",
        "step_gpus_on_node": "4",
        "device": "GPU-fb0b342c-3c6c-3e21-4b74-a3c4cc896472,00000000:01:00.0",
        "cvd": "0",
    },
    {
        "procid": 1,
        "localid": 1,
        "host": "udc-an38-1",
        "step_gpus_on_node": "4",
        "device": "GPU-77c2670a-cd72-a33e-0aef-791b815d4f75,00000000:81:00.0",
        "cvd": "0",
    },
    {
        "procid": 2,
        "localid": 2,
        "host": "udc-an38-1",
        "step_gpus_on_node": "4",
        "device": "GPU-6169eb83-62ee-7917-8e04-ef9010d6db6e,00000000:A1:00.0",
        "cvd": "0",
    },
    {
        "procid": 3,
        "localid": 3,
        "host": "udc-an38-1",
        "step_gpus_on_node": "4",
        "device": "GPU-ad6176b7-0850-e8cf-6440-14745461aa8f,00000000:C1:00.0",
        "cvd": "0",
    },
]
_VB_L2 = {"ntasks": 4, "records": 4, "distinct_devices": 4, "l2": "evaluated"}
# Member-40 shape (Diagnosis capture): 4 tasks, step exposed 2, ranks 3+1 on two UUIDs.
_M40_TASKS = [
    dict(t, step_gpus_on_node="2", device=("GPU-9a46a7f6-x" if t["procid"] != 1 else "GPU-63093dfb-y"))
    for t in _VB_TASKS
]


def _write(record_dir: Path, tasks, l2=None):
    record_dir.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (record_dir / f"task_{t['procid']}.json").write_text(json.dumps(t))
    if l2 is not None:
        (record_dir / "l2.json").write_text(json.dumps(l2))


def _consolidate(tmp_path, tasks, l2, *, expected=4, rc=0, fired=False):
    rd = tmp_path / "rec"
    _write(rd, tasks, l2)
    return gb.consolidate_records(
        rd,
        expected_per_node=expected,
        ntasks=4,
        artifact_path=tmp_path / "gpu_bind_tritonswmm.json",
        attempt_rc=rc,
        watchdog_fired=fired,
        launch_form="per_task",
    )


def test_pass_on_probe_vb_records(tmp_path):
    assert _consolidate(tmp_path, _VB_TASKS, _VB_L2)["verdict"] == "pass"


def test_short_step_on_probe_vc1_shape(tmp_path):
    # VC1: expected 8 on a 4-GPU step; every task exited 99 (measured rc=99).
    assert _consolidate(tmp_path, _VB_TASKS, None, expected=8, rc=99)["verdict"] == "short_step"


def test_short_step_on_member40_shape(tmp_path):
    assert (
        _consolidate(
            tmp_path, _M40_TASKS, {"ntasks": 4, "records": 4, "distinct_devices": 2, "l2": "evaluated"}, expected=4
        )["verdict"]
        == "short_step"
    )


def test_shared_device_on_positive_l2_only(tmp_path):
    l2 = {"ntasks": 4, "records": 4, "distinct_devices": 3, "l2": "evaluated"}
    assert _consolidate(tmp_path, _VB_TASKS, l2, rc=98)["verdict"] == "shared_device"


def test_not_evaluated_is_not_a_fail(tmp_path):
    v = _consolidate(tmp_path, _VB_TASKS, None)["verdict"]
    assert v == "not_evaluated" and v not in gb.FAILING_VERDICTS


def test_watchdog_precedence(tmp_path):
    assert _consolidate(tmp_path, _VB_TASKS, _VB_L2, fired=True)["verdict"] == "watchdog_bind_error"


def test_artifact_compare_and_write_preserves_mtime(tmp_path):
    p = tmp_path / "gpu_bind_tritonswmm.json"
    _consolidate(tmp_path, _VB_TASKS, _VB_L2)
    m1 = p.stat().st_mtime_ns
    _consolidate(tmp_path, _VB_TASKS, _VB_L2)
    assert p.stat().st_mtime_ns == m1


def test_reader_none_when_absent(tmp_path):
    assert gb.read_gpu_bind_artifact(tmp_path, "tritonswmm") is None


@pytest.mark.parametrize(
    "text,per_task,expected",
    [
        ("[2026-09-21T16:42:01.789] error: Not enough gres to bind 1 per task\n", True, "short_step_by_signature"),
        (
            "--------------------------------------------------------------------------\n"
            "A requested component was not found\n",
            True,
            "pass_by_negative_signature",
        ),
        ("A requested component was not found\n", False, None),
        ("", True, None),
    ],
)
def test_classify_from_signature(tmp_path, text, per_task, expected):
    log = tmp_path / "model.log"
    log.write_text(text)
    assert gb.classify_from_signature(log, launch_form_is_per_task=per_task) == expected


def test_short_step_wins_over_watchdog(tmp_path):
    # A2: on a real short step BOTH the record and the watchdog fire; the RECORD decides the label.
    assert _consolidate(tmp_path, _M40_TASKS, None, expected=4, rc=99, fired=True)["verdict"] == "short_step"


@pytest.mark.parametrize("n_gpus,n_nodes", [(4, 1), (16, 2), (2, 1)])
def test_expected_per_node_matches_resource_block_rule(n_gpus, n_nodes, monkeypatch, tmp_path):
    """S3's HHEMT_GB_EXPECTED_PER_NODE (read from the REAL prepare_simulation_command env) equals
    _build_resource_block's gpus_per_node for the same row (workflow.py: math.ceil(gpus_total / sim_nodes))."""
    import math
    from unittest.mock import patch

    from tests.test_srun_command_construction import _make_run

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run = _make_run(
        run_mode="gpu",
        n_gpus=n_gpus,
        n_omp_threads=1,
        in_slurm=True,
        gpu_alloc_mode="gres",
        multi_sim_run_method="batch_job",
    )
    run._analysis.cfg_analysis.n_nodes = n_nodes
    run._analysis.cfg_hpc_system.container = None
    run._analysis.cfg_analysis.execution_environment = "native"
    run._analysis.cfg_analysis.hpc_ensemble_partition = None
    run._scenario.scen_paths.sim_folder = tmp_path
    with patch.object(run, "_analysis_level_model_logfile", return_value=tmp_path / "run.log"):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            _cmd, env, _log, _ = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    from hhemt.workflow import SnakemakeWorkflowBuilder

    # Both halves from the toolkit: the node rule is the real staticmethod (workflow.py:2445-2448);
    # the per-node division is _build_resource_block's inline `math.ceil(gpus_total / sim_nodes)`
    # (workflow.py:2290), transcribed because it is not factored.
    sim_nodes = max(n_nodes, SnakemakeWorkflowBuilder._calculate_nodes_for_gpus(n_gpus, 8))
    assert int(env["HHEMT_GB_EXPECTED_PER_NODE"]) == math.ceil(n_gpus / sim_nodes)
    assert int(env["HHEMT_GB_NTASKS"]) == n_gpus


def test_watchdog_fires_on_signature_and_not_on_banner(tmp_path):
    log = tmp_path / "model.log"
    log.write_text("x\n")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    log.write_text("error: Not enough gres to bind 1 per task\n")
    rc, fired = gb.wait_with_bind_watchdog(proc, log, window_s=10.0, poll_s=0.2)
    assert fired is True and rc != 0
    log.write_text("A requested component was not found\n")
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    rc, fired = gb.wait_with_bind_watchdog(proc, log, window_s=5.0, poll_s=0.2)
    assert fired is False and rc == 0


def test_df_status_pre_guard_classification_path(tmp_path):
    """The S8a composition on a synthetic logs/sims dir (C-S8a): runner log recorded the per_task
    form + banner-only model log -> pass_by_negative_signature; no runner log -> None. Plus a static
    pin that df_status passes the analysis id from cfg_analysis (TRITONSWMM_analysis has no
    `analysis_id` attribute; iteration 3 got this wrong and would have raised AttributeError)."""
    import inspect

    from hhemt.analysis import TRITONSWMM_analysis

    src = inspect.getsource(TRITONSWMM_analysis.df_status.fget)
    assert "analysis_id=self.cfg_analysis.analysis_id" in src
    assert "self.analysis_id" not in src

    logs = tmp_path / "logs" / "sims"
    logs.mkdir(parents=True)
    event_id = "year.9_event_type.compound_event_id.1"
    model_log = logs / "model_tritonswmm_member_35_evt0.log"
    model_log.write_text(
        "--------------------------------------------------------------------------\n"
        "A requested component was not found\n"
    )
    assert (
        gb.find_runner_command_line(logs, model_type="tritonswmm", analysis_id="member_35", event_id=event_id) is None
    )
    cmd = gb.find_runner_command_line(logs, model_type="tritonswmm", analysis_id="member_35", event_id=event_id)
    assert (
        gb.classify_from_signature(model_log, launch_form_is_per_task=bool(cmd) and "--gpus-per-task=1" in (cmd or ""))
        is None
    )
    (logs / "simulation_member_35_evt_year_9_event_type_compound_event_id_1.log").write_text(
        "2026-09-21 16:34:27,773 [INFO] [0] Command: bash -lc export X=1; srun -N 1 --ntasks=4 "
        "--cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 --mpi=pmix "
        "apptainer exec --nv sif triton.exe cfg\n"
    )
    cmd = gb.find_runner_command_line(logs, model_type="tritonswmm", analysis_id="member_35", event_id=event_id)
    assert cmd is not None and "--gpus-per-task=1" in cmd
    assert gb.classify_from_signature(model_log, launch_form_is_per_task=True) == "pass_by_negative_signature"
    # multisim naming is the other candidate
    (logs / f"tritonswmm_evt-{event_id}.log").write_text(
        "Command: bash -lc srun --ntasks-per-gpu=1 --overlap triton.exe\n"
    )
    cmd2 = gb.find_runner_command_line(logs, model_type="tritonswmm", analysis_id="x", event_id=event_id)
    assert cmd2 is not None and "--gpus-per-task=1" not in cmd2


def test_validator_fails_on_verdict_and_infos_on_missing():
    from hhemt.consolidate_workflow import validate_resource_usage

    base = dict(
        run_completed=True,
        scenario_directory="/s",
        run_mode="gpu",
        n_mpi_procs=4,
        n_omp_threads=1,
        n_gpus=4,
        actual_nTasks=4,
        actual_omp_threads=1,
        actual_total_gpus=4,
        actual_gpu_backend="CUDA",
    )
    bad = pd.DataFrame([dict(base, gpu_binding_verdict="short_step_by_signature", actual_distinct_gpus=None)])
    passed, issues = validate_resource_usage(analysis=None, df_status=bad)
    assert passed is False and issues and issues[0]["resource"] == "GPU binding"
    unmeasured = pd.DataFrame([dict(base, gpu_binding_verdict=None, actual_distinct_gpus=None)])
    passed, issues = validate_resource_usage(analysis=None, df_status=unmeasured)
    assert passed is True and issues and issues[0]["actual"] == "not measured"
    ok = pd.DataFrame([dict(base, gpu_binding_verdict="pass_by_negative_signature", actual_distinct_gpus=None)])
    passed, issues = validate_resource_usage(analysis=None, df_status=ok)
    assert passed is True and len(issues) == 1 and issues[0]["actual"].startswith("pass_by_negative_signature")


def test_validator_mixed_frame_nan_verdict_still_infos():
    # C-S8/C-S10: a row LACKING the key coerces a sibling row's None to NaN; the info row must survive.
    from hhemt.consolidate_workflow import validate_resource_usage

    gpu = dict(
        run_completed=True,
        scenario_directory="/g",
        run_mode="gpu",
        n_mpi_procs=4,
        n_omp_threads=1,
        n_gpus=4,
        actual_nTasks=4,
        actual_omp_threads=1,
        actual_total_gpus=4,
        actual_gpu_backend="CUDA",
        gpu_binding_verdict=None,
        actual_distinct_gpus=None,
    )
    swmm = dict(
        run_completed=True,
        scenario_directory="/s",
        run_mode="serial",
        n_mpi_procs=1,
        n_omp_threads=1,
        n_gpus=0,
        actual_nTasks=1,
        actual_omp_threads=1,
        actual_total_gpus=None,
        actual_gpu_backend="none",
    )
    frame = pd.DataFrame([gpu, swmm])
    assert pd.isna(frame.loc[0, "gpu_binding_verdict"]) and frame.loc[0, "gpu_binding_verdict"] is not None
    passed, issues = validate_resource_usage(analysis=None, df_status=frame)
    assert passed is True and len(issues) == 1 and issues[0]["actual"] == "not measured"


def test_check_resource_usage_summary_discloses_counts(monkeypatch):
    from hhemt import analysis_validation as av

    monkeypatch.setattr(
        "hhemt.consolidate_workflow.validate_resource_usage",
        lambda analysis, logger=None, df_status=None: (
            True,
            [
                {"severity": "info", "actual": "not measured"},
                {"severity": "info", "actual": "pass_by_negative_signature (last exec; pre-guard row)"},
            ],
        ),
    )
    r = av.check_resource_usage(analysis=None)
    assert r.passed is True and "1 row(s) not measured, 1 classified by negative signature" in r.summary
    monkeypatch.setattr(
        "hhemt.consolidate_workflow.validate_resource_usage",
        lambda analysis, logger=None, df_status=None: (
            False,
            [{"severity": "info", "actual": "not measured"}, {"resource": "GPU binding", "actual": "short_step"}],
        ),
    )
    r = av.check_resource_usage(analysis=None)
    assert r.passed is False and r.summary.startswith("Resource mismatches in 1 scenario(s)")
