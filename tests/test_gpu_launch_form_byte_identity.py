"""V-A byte-identity of every srun launch form emitted by prepare_simulation_command.

Goldens captured PRE-CHANGE at hhemt c7620c0b (2026-09-21) through the fixture-free harness in
tests/test_srun_command_construction.py. The two CHANGED keys (UVA gres multi-GPU under the
Snakemake slurm executor) are asserted against the post-change form; every other form MUST stay
byte-identical (GPU-binding workshop, SLURM Ticket 24862; developer mandate [Q445] V-A).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tests.test_srun_command_construction import (  # tests/__init__.py exists: rootdir import mode
    _get_launch_cmd,
    _make_run,
)

# key -> (kwargs for _make_run, execution_locus)
_FORMS: dict[str, tuple[dict, str | None]] = {
    "cpu_mpi_batch_job": (
        dict(run_mode="mpi", n_mpi_procs=4, n_omp_threads=1, in_slurm=True, multi_sim_run_method="batch_job"),
        None,
    ),
    "cpu_hybrid_1job": (
        dict(
            run_mode="hybrid",
            n_mpi_procs=2,
            n_omp_threads=4,
            in_slurm=True,
            multi_sim_run_method="1_job_many_srun_tasks",
        ),
        None,
    ),
    "gpus_mode_multi_1job": (
        dict(
            run_mode="gpu",
            n_gpus=4,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gpus",
            multi_sim_run_method="1_job_many_srun_tasks",
        ),
        None,
    ),
    "gpus_mode_multi_batch": (
        dict(
            run_mode="gpu",
            n_gpus=4,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gpus",
            multi_sim_run_method="batch_job",
        ),
        None,
    ),
    "gres_single_batch_job": (
        dict(
            run_mode="gpu",
            n_gpus=1,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="batch_job",
        ),
        None,
    ),
    "gres_multi_1job": (
        dict(
            run_mode="gpu",
            n_gpus=4,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="1_job_many_srun_tasks",
        ),
        None,
    ),
    "gres_multi_batch_job": (
        dict(
            run_mode="gpu",
            n_gpus=4,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="batch_job",
        ),
        None,
    ),
    "gres_multi_local_slurm": (
        dict(
            run_mode="gpu",
            n_gpus=2,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="local",
        ),
        "slurm",
    ),
    "gres_multi_batch_job_2node": (
        dict(
            run_mode="gpu",
            n_gpus=16,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="batch_job",
        ),
        None,
    ),
    "gres_multi_batch_job_container": (
        dict(
            run_mode="gpu",
            n_gpus=4,
            n_omp_threads=1,
            in_slurm=True,
            gpu_alloc_mode="gres",
            multi_sim_run_method="batch_job",
        ),
        None,
    ),
}

# PRE-CHANGE goldens (c7620c0b). Do NOT edit by hand: a change here is a change to the launch
# contract and needs its own probe on the cluster.
_PRE_CHANGE_GOLDEN: dict[str, str] = {
    "cpu_mpi_batch_job": "srun -N 1 --ntasks=4 --cpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "cpu_hybrid_1job": "srun -N 1 --ntasks=2 --cpus-per-task=4 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gpus_mode_multi_1job": "srun -N 1 --ntasks=4 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gpus_mode_multi_batch": "srun -N 1 --ntasks=4 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_single_batch_job": "srun -N 1 --cpus-per-task=1 --ntasks-per-gpu=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_multi_1job": "srun -N 1 --cpus-per-task=1 --ntasks-per-gpu=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_multi_batch_job": "srun -N 1 --ntasks=4 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_multi_local_slurm": "srun -N 1 --ntasks=2 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_multi_batch_job_container": "srun -N 1 --ntasks=4 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 --mpi=pmix apptainer exec --nv -B /fake/sim/out_tritonswmm:/opt/hhemt/out_tritonswmm /fake/sifs/openmpi-cuda/hhemt_openmpi-cuda_a6000_x_y.sif /opt/hhemt/bin/triton.exe /fake/TRITONSWMM.cfg",  # noqa: E501
    "gres_multi_batch_job_2node": "srun -N 2 --ntasks=16 --cpus-per-task=1 --gpus-per-task=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 /fake/TRITONSWMM /fake/TRITONSWMM.cfg",  # noqa: E501
}

_UNCHANGED_KEYS = (
    "cpu_mpi_batch_job",
    "cpu_hybrid_1job",
    "gpus_mode_multi_1job",
    "gpus_mode_multi_batch",
    "gres_single_batch_job",
    "gres_multi_1job",
    # same elif as the changed form, multi_sim_run_method="local": UNPROBED, byte-identical (A1)
    "gres_multi_local_slurm",
)
_CHANGED_KEYS = ("gres_multi_batch_job", "gres_multi_batch_job_2node", "gres_multi_batch_job_container")


def _build(key: str, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    kwargs, locus = _FORMS[key]
    run = _make_run(**kwargs)
    # A REAL temp sim_folder for every key: the guarded (batch_job) forms write the L1/L2 guard
    # script under {sim_folder}/_status/_gpu_bind/ at emission (never /fake, never the CWD).
    run._scenario.scen_paths.sim_folder = tmp_path / "sim"
    if key == "gres_multi_batch_job_2node":
        run._analysis.cfg_analysis.n_nodes = 2
    if key == "gres_multi_batch_job_container":
        from hhemt.config.hpc_system import ContainerSpec

        run._analysis.cfg_hpc_system.container = ContainerSpec(
            sif_root="/fake/sifs", gpu_flag="--nv", apptainer_module="apptainer/1.5.0", srun_mpi="pmix"
        )
        run._analysis.cfg_analysis.execution_environment = "container"
        run._analysis.cfg_analysis.hpc_ensemble_partition = "gpu"
        run._analysis._sif_identity_for.return_value = SimpleNamespace(
            family="openmpi-cuda", stem="hhemt_openmpi-cuda_a6000_x_y"
        )
        run._analysis._system.cfg_system.system_directory = Path("/fake/system")
        run._scenario._system.cfg_system.system_directory = Path("/fake/system")
        run._scenario.scen_paths.out_tritonswmm = Path("/fake/sim/out_tritonswmm")
    else:
        run._analysis.cfg_hpc_system.container = None
        run._analysis.cfg_analysis.execution_environment = "native"
        run._analysis.cfg_analysis.hpc_ensemble_partition = None
    return run, locus


def _emit(key: str, monkeypatch, tmp_path) -> str:
    run, locus = _build(key, monkeypatch, tmp_path)
    full = _get_launch_cmd(run, execution_locus=locus)
    return full[full.index("srun ") :]


@pytest.mark.parametrize("key", _UNCHANGED_KEYS)
def test_untouched_launch_forms_are_byte_identical(key, monkeypatch, tmp_path):
    assert _emit(key, monkeypatch, tmp_path) == _PRE_CHANGE_GOLDEN[key]


@pytest.mark.parametrize("key", _CHANGED_KEYS)
def test_changed_forms_drop_overlap_and_add_guard(key, monkeypatch, tmp_path):
    pre = _PRE_CHANGE_GOLDEN[key]
    post = _emit(key, monkeypatch, tmp_path)
    assert "--overlap" not in post
    # Everything except the dropped token and the guard prefix is unchanged.
    expected_tail = pre.replace("--overlap ", "")
    guard = re.search(r"bash (\S+/_status/_gpu_bind/[a-z]+\.sh) ", post)
    assert guard is not None, post
    assert post.replace(f"bash {guard.group(1)} ", "") == expected_tail
    assert os.path.basename(guard.group(1)) in ("tritonswmm.sh", "triton.sh")
    if key == "gres_multi_batch_job_container":
        # The guard sits between the srun flags and `apptainer exec`, exactly as probe 20350116 ran it.
        assert post.index(f"bash {guard.group(1)} ") < post.index("apptainer exec")


def test_guard_script_is_written_and_executable(monkeypatch, tmp_path):
    from hhemt.gpu_bind_guard import GUARD_SCRIPT_TEXT

    run, locus = _build("gres_multi_batch_job", monkeypatch, tmp_path)
    _get_launch_cmd(run, execution_locus=locus)
    script = tmp_path / "sim" / "_status" / "_gpu_bind" / "tritonswmm.sh"
    assert script.exists() and os.access(script, os.X_OK)
    assert script.read_text() == GUARD_SCRIPT_TEXT


def _emit_with_env(key: str, monkeypatch, tmp_path):
    from unittest.mock import patch

    run, locus = _build(key, monkeypatch, tmp_path)
    with patch.object(run, "_analysis_level_model_logfile", return_value=tmp_path / "run.log"):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            _cmd, env, _log, _ = run.prepare_simulation_command(
                pickup_where_leftoff=False, verbose=False, execution_locus=locus
            )
    return env


def test_local_in_slurm_form_has_no_guard_side_effects(monkeypatch, tmp_path):
    """AM-4 two-arm: the A1 byte-identical form (gres multi-GPU, multi_sim_run_method="local"
    inside SLURM) must be byte-identical in FILESYSTEM and ENV too -- no guard script, no
    HHEMT_GB_* export -- while the measured batch_job form gets both."""
    env = _emit_with_env("gres_multi_local_slurm", monkeypatch, tmp_path)
    assert not any(k.startswith("HHEMT_GB_") for k in env), sorted(env)
    assert not (tmp_path / "sim").exists(), list((tmp_path / "sim").rglob("*"))

    env = _emit_with_env("gres_multi_batch_job", monkeypatch, tmp_path)
    assert env["HHEMT_GB_RECORD_DIR"] == str(tmp_path / "sim" / "_status" / "_gpu_bind" / "tritonswmm")
    assert (tmp_path / "sim" / "_status" / "_gpu_bind" / "tritonswmm.sh").exists()
