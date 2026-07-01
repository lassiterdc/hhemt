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

Run from the repo root (so ``tests/`` is on ``sys.path``). The real per-cluster deployment
config (SIF path + account + partitions) is resolved from the PRIVATE ***REMOVED*** estate
via ``_resolve_hpc_system_config`` ($***REMOVED*** / $HHEMT_HPC_SYSTEM_CONFIG / argv[2]) —
NO git-tracked file is edited per run; the in-tree ``test_data/.../hpc_system_config_*.yaml``
are copy-me templates the operator reconstructs into the estate once per cluster.
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
import os
import sys as _sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

# ABSOLUTE paths (rooted at the repo) — the Snakemake SUBPROCESS rules re-read the
# hpc-system-config from the run dir (~/.cache/.../<analysis>), NOT the repo root, so a
# relative --hpc-system-config raises FileNotFoundError at setup_target_0 (mirrors the
# absolute HPC path validate_frontier.py used).
# Suite selection: default single-node suite, overridable via $HHEMT_CV_SUITE (a filename
# under tests/fixtures/container_validation/) so a 2-node variant can be run without forking
# the runner. The 2-node suite adds n_nodes=2 rows to validate the unified config at 2 nodes
# (subsuming 1) per the expanded DoD-7.
_SUITE_DIR = _REPO_ROOT / "tests/fixtures/container_validation"
_SUITE = _SUITE_DIR / (os.environ.get("HHEMT_CV_SUITE") or "container_validation_suite.csv")

# Default desktop checkout of the PRIVATE deployment estate (***REMOVED***), computed
# from the user's home (portable; no hard-coded /home/<user>). Cluster runs set
# $***REMOVED*** to the compute-visible checkout (the resolver reads it).
_***REMOVED***_DEFAULT = str(Path.home() / "dev" / "***REMOVED***")

# In-tree anonymized examples — COPY-ME templates (+ byte-identity-test fixtures), never
# the live config. The operator reconstructs the real config in the estate from these.
_TEMPLATE = {
    "uva": _REPO_ROOT / "test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml",
    "frontier": _REPO_ROOT / "test_data/norfolk_coastal_flooding/hpc_system_config_frontier.yaml",
}

# Per-cluster knobs (NOT secrets — the deployment account/SIF come from the resolved
# estate config; the GPU partition derives the hardware).
_CLUSTER = {
    # execution_mode: UVA runs the suite via the Snakemake SLURM executor (one right-sized
    # job per rule) so multi-node GPU rows get their own -N/gres allocation; Frontier runs
    # in-allocation "local" (proven green, 1_job_many_srun_tasks skips the GPU preflight).
    # gpu_partition="gpu" is the real Rivanna partition (a100 selected via --gres); the
    # former "gpu-a100-80" key was a pseudo-partition that --executor slurm would pass to
    # sbatch -p and fail on.
    "uva": dict(gpu_partition="gpu", multi_sim_run_method="batch_job", execution_mode="slurm"),
    "frontier": dict(gpu_partition="batch", multi_sim_run_method="1_job_many_srun_tasks", execution_mode="local"),
}


def _resolve_hpc_system_config(cluster: str, override: str | None = None) -> Path:
    """Resolve the operator's REAL hpc_system_config for ``cluster`` from the PRIVATE
    ***REMOVED*** estate (the versioned, git-pulled deployment-config home).

    Precedence: explicit ``override`` (argv[2]) > ``$HHEMT_HPC_SYSTEM_CONFIG`` env var >
    ``$***REMOVED***/hpc/hpc_system_config_{cluster}.yaml`` (``$***REMOVED***`` defaults to
    the desktop checkout; set it per-cluster to the compute-visible estate clone). The
    in-tree ``_TEMPLATE`` is the copy-me source, never the live config, so a run edits ZERO
    git-tracked files in the public repo.
    """
    if override:
        path = Path(override).expanduser()
    elif os.environ.get("HHEMT_HPC_SYSTEM_CONFIG"):
        path = Path(os.environ["HHEMT_HPC_SYSTEM_CONFIG"]).expanduser()
    else:
        projects = Path(os.environ.get("***REMOVED***", _***REMOVED***_DEFAULT)).expanduser()
        path = projects / "hpc" / f"hpc_system_config_{cluster}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No hpc_system_config for cluster {cluster!r} at {path}.\n"
            f"  1. On the cluster, set $***REMOVED*** to your compute-visible ***REMOVED*** "
            f"checkout and `git pull` it on the login node.\n"
            f"  2. Reconstruct the config once from the in-tree template:\n"
            f"       cp {_TEMPLATE[cluster]} {path}\n"
            f"     then fill default_account (from hpc/profiles.yaml) and container.sif_path.\n"
            f"  (Or set $HHEMT_HPC_SYSTEM_CONFIG, or pass the path as argv[2].)"
        )
    return path.resolve()


def build_case(
    cluster: str,
    *,
    hpc_system_config_yaml: str | None = None,
    system_directory: str | None = None,
    start_from_scratch: bool = False,
    multi_sim_run_method: str | None = None,
):
    """Build the container-validation sensitivity test case for ``cluster``.

    The real deployment ``hpc_system_config`` (ContainerSpec SIF path + partition→hardware
    + account) is resolved from the PRIVATE ***REMOVED*** estate (see
    ``_resolve_hpc_system_config``) — NO git-tracked file is edited per run. ``hpc_account``
    is sourced from the resolved config's ``default_account``; a fail-fast rejects an
    unfilled template. ``execution_environment`` is supplied PER ROW by the CSV.
    """
    if cluster not in _CLUSTER:
        raise ValueError(f"cluster must be one of {sorted(_CLUSTER)}; got {cluster!r}")
    c = _CLUSTER[cluster]
    from hhemt.config.loaders import load_hpc_system_config

    cfg_path = _resolve_hpc_system_config(cluster, hpc_system_config_yaml)
    cfg_hpc = load_hpc_system_config(cfg_path)
    account = cfg_hpc.default_account or ""
    if (not account) or ("{your-" in account):
        raise ValueError(
            f"{cfg_path}: default_account is unset or still a placeholder ({account!r}). "
            f"Set default_account to your real OLCF project / UVA allocation."
        )
    if cfg_hpc.container is None or "{your-" in (cfg_hpc.container.sif_path or ""):
        raise ValueError(
            f"{cfg_path}: container.sif_path is missing or still a placeholder. "
            f"Set it to the absolute on-cluster path of your transferred, signed SIF."
        )
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=f"container_validation_{cluster}",
        sensitivity_csv=_SUITE,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        hpc_system_config_yaml=cfg_path,
        additional_system_configs=(
            {"system_directory": system_directory} if system_directory else {}
        ),
        additional_analysis_configs={
            # batch_job/1_job_many_srun_tasks = login-node operator submission (default per
            # cluster). The in-allocation validation entry (run_and_verdict) overrides this to
            # "local" so execution_mode="local" runs the sims in-process via the inner srun —
            # no SLURM-executor plugin needed (matches the proven validate_uva/frontier runs).
            "multi_sim_run_method": multi_sim_run_method or c["multi_sim_run_method"],
            "hpc_account": account,
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
                    # n_devices is the scaling x-axis (1 vs 2 GPU); execution_environment
                    # (native vs container) is the group overlay — together a real
                    # strong-scaling + container-overhead figure. n_devices is a derived
                    # column the renderer synthesizes from the (prefixed) compute columns.
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "analysis.execution_environment",
                }
            },
        },
    )


def run_and_verdict(cluster: str, *, start_from_scratch: bool = True, hpc_system_config_yaml: str | None = None):
    """Build + run the suite locally on the allocated GPU node, then print the
    within-family native≡container verdict. Call from inside a GPU sbatch allocation
    (the runner submits no jobs itself — execution_mode='local')."""
    from hhemt.eda.cross_sim_identity import check_cross_sim_identity

    # In-allocation validation: multi_sim_run_method="local" so execution_mode="local"
    # runs the sims (incl. the multi-rank container rows via the inner srun) IN-PROCESS
    # within this GPU allocation — not the login-node batch_job submission path.
    tc = build_case(
        cluster,
        start_from_scratch=start_from_scratch,
        multi_sim_run_method="local",
        hpc_system_config_yaml=hpc_system_config_yaml,
    )
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
    tc.analysis.run(
        from_scratch=True,
        execution_mode=_CLUSTER[cluster]["execution_mode"],
        wait_for_job_completion=True,
        verbose=True,
    )
    result = check_cross_sim_identity(tc.analysis, within_family=True)
    v = result.verdict
    print("VERDICT passed  =", v.passed)
    print("VERDICT summary =", v.summary)
    print("VERDICT details =", v.details)
    # No-data guard (mirrors validate_uva.py): check_cross_sim_identity returns passed=True
    # with an "N/A — no … summaries" verdict when the sims produced NOTHING to compare
    # (a rule failed upstream). That is NOT a real native≡container result.
    details = v.details or []
    summaries_absent = any(
        "summaries absent" in str(d.get("detail", "")) for d in details if isinstance(d, dict)
    )
    no_data = (
        bool(getattr(result, "skipped", False))
        or ("N/A" in (v.summary or ""))
        or ("no sub-analysis" in (v.summary or ""))
        or summaries_absent
    )
    if no_data:
        print("OVERALL: NO-DATA — sims produced no summaries; not a real native=container comparison (run failed upstream)")
        return False
    real_pass = bool(v.passed)
    print("OVERALL:", "native=container within-family" if real_pass else "DIVERGED")
    return real_pass


if __name__ == "__main__":
    import sys

    cluster = sys.argv[1] if len(sys.argv) > 1 else "uva"
    cfg_override = sys.argv[2] if len(sys.argv) > 2 else None
    # run_and_verdict returns True only for a REAL within-family pass (no-data guarded);
    # exit non-zero otherwise so the SLURM job state reflects the result.
    sys.exit(0 if run_and_verdict(cluster, hpc_system_config_yaml=cfg_override) else 1)
