"""DoD-7 container-validation runner.

Drives ``tests/fixtures/container_validation/container_validation_suite.csv`` (the
4-row ``{native,container}×{1-GPU,2-GPU}`` matrix) on UVA or Frontier via the
**partition-derives-hardware** DI pattern — NOT the retired ``gpu_compilation_backend``
backend-DI, and NOT the stale ``scripts/experiments/_matrix_builder.py``.

The single cross-cluster axis is the per-cluster ``hpc_system_config`` yaml: the GPU
hardware/backend and the container ``ContainerSpec`` (Frontier host-Cray-MPICH-ABI
helper-module hybrid vs UVA container-own OpenMPI ``srun_mpi: pmix``) flow entirely
through ``hpc_system_config_yaml``; ``execution_environment`` and the rank/GPU axis are
the only CSV-varying axes. Build the case, then ``tc.analysis.run(execution_mode="local")``
inside a GPU allocation (mirrors the proven ``validate_*`` runners) and read the verdict
with ``check_cross_sim_identity(within_family=True)``.

Run from the repo root (so ``tests/`` is on ``sys.path``). Fill ``{your-allocation}``
and confirm the per-cluster partition/walltime before launching — exactly as
``synth_compute_config.py`` does.
"""

from __future__ import annotations

from pathlib import Path

# Initialize GDAL (rasterio/rioxarray) BEFORE the synthetic-model chain pulls in
# swmm.toolkit — importing the native swmm lib before GDAL inits corrupts GDAL's
# allocator and aborts ("free(): invalid pointer"). Must precede tests.fixtures.
import rioxarray  # noqa: F401  (import-order guard — see synth_compute_config.py)

# Make the repo root importable so `tests.fixtures...` resolves regardless of how this
# file is invoked (`python scripts/experiments/container_validation.py` puts the SCRIPT
# dir on sys.path, not the repo root). Mirrors validate_uva.py's `sys.path.insert(0, ROOT)`.
import sys as _sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

_SUITE = Path("tests/fixtures/container_validation/container_validation_suite.csv")

# Per-cluster knobs. {your-allocation} is the only hard fill-in; confirm the GPU
# partition + walltime against your current allocation before running.
_CLUSTER = {
    "uva": dict(
        yaml=Path("test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml"),
        gpu_partition="gpu-a100-80",            # a100 / CUDA (partition derives hardware)
        multi_sim_run_method="batch_job",       # UVA executor-owns-sbatch
    ),
    "frontier": dict(
        yaml=Path("test_data/norfolk_coastal_flooding/hpc_system_config_frontier.yaml"),
        gpu_partition="batch",                  # mi250x / HIP
        multi_sim_run_method="1_job_many_srun_tasks",  # Frontier toolkit-owns-sbatch
    ),
}


def build_case(
    cluster: str,
    *,
    system_directory: str | None = None,
    start_from_scratch: bool = False,
    multi_sim_run_method: str | None = None,
):
    """Build the container-validation sensitivity test case for ``cluster``.

    ``execution_environment`` is supplied PER ROW by the CSV
    (``analysis.execution_environment``); the master default stays ``native`` so the
    native rows need no override. The ContainerSpec + partition→hardware derivation
    flow through ``hpc_system_config_yaml`` — no per-sub-analysis backend DI.
    """
    if cluster not in _CLUSTER:
        raise ValueError(f"cluster must be one of {sorted(_CLUSTER)}; got {cluster!r}")
    c = _CLUSTER[cluster]
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=f"container_validation_{cluster}",
        sensitivity_csv=_SUITE,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        hpc_system_config_yaml=c["yaml"],
        additional_system_configs=(
            {"system_directory": system_directory} if system_directory else {}
        ),
        additional_analysis_configs={
            # batch_job/1_job_many_srun_tasks = login-node operator submission (default per
            # cluster). The in-allocation validation entry (run_and_verdict) overrides this to
            # "local" so execution_mode="local" runs the sims in-process via the inner srun —
            # no SLURM-executor plugin needed (matches the proven validate_uva/frontier runs).
            "multi_sim_run_method": multi_sim_run_method or c["multi_sim_run_method"],
            "hpc_account": "{your-allocation}",          # FILL: OLCF project / UVA allocation
            "hpc_max_simultaneous_sims": 1000,
            "hpc_total_job_duration_min": 60,
            "hpc_gpus_per_node": 8,
            "hpc_time_min_per_sim": 30,
            "hpc_restart_times_simulate": 2,
            "hpc_restart_times_other": 2,
            "hpc_ensemble_partition": c["gpu_partition"],
            "hpc_setup_and_analysis_processing_partition": c["gpu_partition"],
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(_SUITE),
            "report": {
                "sensitivity": {
                    "mode": "benchmarking",
                    "independent_vars": ["analysis.execution_environment"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "analysis.execution_environment",
                }
            },
        },
    )


def run_and_verdict(cluster: str, *, start_from_scratch: bool = True):
    """Build + run the suite locally on the allocated GPU node, then print the
    within-family native≡container verdict. Call from inside a GPU sbatch allocation
    (the runner submits no jobs itself — execution_mode='local')."""
    from hhemt.eda.cross_sim_identity import check_cross_sim_identity

    # In-allocation validation: multi_sim_run_method="local" so execution_mode="local"
    # runs the sims (incl. the multi-rank container rows via the inner srun) IN-PROCESS
    # within this GPU allocation — not the login-node batch_job submission path.
    tc = build_case(cluster, start_from_scratch=start_from_scratch, multi_sim_run_method="local")
    # DI parity with the proven validate_* runners: ensure master + every sub-analysis carry
    # the GPU partition selector (n_gpus>0 + the GPU-sensitivity validation resolve) and the
    # local orchestration (the subs re-load configs from disk).
    tc.analysis.cfg_analysis.hpc_ensemble_partition = _CLUSTER[cluster]["gpu_partition"]
    tc.analysis.cfg_analysis.multi_sim_run_method = "local"
    sens = getattr(tc.analysis, "sensitivity", None)
    if sens is not None:
        for sub in getattr(sens, "sub_analyses", {}).values():
            try:
                sub.cfg_analysis.hpc_ensemble_partition = _CLUSTER[cluster]["gpu_partition"]
                sub.cfg_analysis.multi_sim_run_method = "local"
            except Exception:
                pass
    tc.analysis.run(from_scratch=True, execution_mode="local", verbose=True)
    v = check_cross_sim_identity(tc.analysis, within_family=True).verdict
    print("VERDICT passed  =", v.passed)
    print("VERDICT summary =", v.summary)
    print("OVERALL:", "native=container within-family" if v.passed else "DIVERGED")
    return v


if __name__ == "__main__":
    import sys

    cluster = sys.argv[1] if len(sys.argv) > 1 else "uva"
    _v = run_and_verdict(cluster)
    # Exit non-zero unless the within-family verdict genuinely passed (so the SLURM job
    # state reflects the result; the log carries the full verdict either way).
    sys.exit(0 if getattr(_v, "passed", False) else 1)
