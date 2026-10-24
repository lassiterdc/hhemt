"""Check that a containerized TRITON-SWMM run reproduces the native run, on UVA or Frontier.

The problem: a container image is trustworthy only if a simulation run inside it produces the
same results as the same simulation run natively, at the same rank and GPU count. This script
runs the hand-authored suite ``tests/fixtures/container_validation/container_validation_suite.csv``
(the 4-row ``{native,container}×{1-GPU,2-GPU}`` matrix; ``$HHEMT_CV_SUITE`` selects another
suite in that directory) and reads the verdict with
``check_cross_sim_identity(within_family=True)``, exiting 0 only on a real pass.

Hardware is derived from the partition: the GPU partition the per-cluster
``hpc_system_config`` yaml names selects the GPU hardware and backend — NOT the retired
``gpu_compilation_backend`` field. That yaml is the single cross-cluster axis: the GPU
hardware/backend and the container ``ContainerSpec`` (Frontier host-Cray-MPICH-ABI
helper-module hybrid vs UVA container-own OpenMPI ``srun_mpi: pmix``) flow entirely
through ``hpc_system_config_yaml``; ``execution_environment`` and the rank/GPU axis are
the only CSV-varying axes. Where the simulations run is per cluster (``_CLUSTER``): on
Frontier ``execution_mode="local"`` runs them inside the GPU allocation the script is
launched in; on UVA ``execution_mode="slurm"`` dispatches every rule as its own SLURM job.

Run as ``python scripts/experiments/container_validation.py {cluster} {path to that
cluster's hpc_system_config YAML}``; the script puts the repo root on ``sys.path`` itself, so
``tests/`` is importable however it is invoked. The
real per-cluster deployment config (SIF path + account + partitions) comes ONLY from argv[2]:
it is passed to ``hhemt.experiment_bundle.resolve_hpc_system_config`` as the override, and the
resolver reads no environment variable, so a run without argv[2] exits naming it. NO
git-tracked file is edited per run; the in-tree ``test_data/.../hpc_system_config_*.yaml``
are copy-me templates the operator fills in once per cluster and passes as argv[2].
"""

from __future__ import annotations

# The repo root is made importable below (after the GDAL guard) so `tests.fixtures...`
# resolves regardless of how this file is invoked: `python scripts/experiments/container_validation.py`
# puts the SCRIPT dir on sys.path, not the repo root.
import os
import sys as _sys
from pathlib import Path

# Initialize GDAL (rasterio/rioxarray) BEFORE the synthetic-model chain pulls in
# swmm.toolkit — importing the native swmm lib before GDAL inits corrupts GDAL's
# allocator and aborts ("free(): invalid pointer"). Must precede tests.fixtures.
import rioxarray  # noqa: F401  (import-order guard — see synth_compute_config.py)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# noqa: E402 -- this import MUST follow the _sys.path.insert above: `tests.fixtures`
# is not importable until the repo root is on sys.path. Hoisting it to the top of the
# file (the naive E402 fix) breaks this script at import time. Do not 'clean this up'.
from tests.fixtures.test_case_builder import (  # noqa: E402
    retrieve_synth_TRITON_SWMM_test_case,
)

# ABSOLUTE paths (rooted at the repo) — the Snakemake SUBPROCESS rules re-read the
# hpc-system-config from the run dir (~/.cache/.../<analysis>), NOT the repo root, so a
# relative --hpc-system-config raises FileNotFoundError at setup_target_0.
# Suite selection: default single-node suite, overridable via $HHEMT_CV_SUITE (a filename
# under tests/fixtures/container_validation/) so a 2-node variant can be run without forking
# the runner. The 2-node suite keeps the 1-GPU rows and replaces the 2-GPU rows with
# n_nodes=2 rows, so one run validates the unified config at both 1 and 2 nodes.
_SUITE_DIR = _REPO_ROOT / "tests/fixtures/container_validation"
_SUITE = _SUITE_DIR / (os.environ.get("HHEMT_CV_SUITE") or "container_validation_suite.csv")

# The cluster's real hpc_system_config comes only from argv[2] (see the __main__ guard
# below); pass a path the compute nodes can read. The public repo bakes in no default
# path, so it does not hard-code the operator's private checkout name.

# Per-cluster knobs (NOT secrets — the deployment account/SIF come from the resolved
# estate config; the GPU partition derives the hardware).
_CLUSTER = {
    # execution_mode: UVA runs the suite via the Snakemake SLURM executor (one right-sized
    # job per rule) so multi-node GPU rows get their own -N/gres allocation; Frontier runs
    # in-allocation "local".
    # multi_sim_run_method is build_case's DEFAULT when it is called without an override. For
    # Frontier that is "1_job_many_srun_tasks", the one value under which the sim launcher
    # skips its SLURM GPU-count preflight. run_and_verdict, this file's only run path,
    # overrides it to "local", so on the Frontier run this file performs the preflight is
    # active for GPU rows launched through srun.
    # gpu_partition="gpu" is the real Rivanna partition (a100 selected via --gres); the
    # former "gpu-a100-80" key was a pseudo-partition that --executor slurm would pass to
    # sbatch -p and fail on.
    # cpu_partition: the NON-GPU partition for setup/prepare/process/consolidate/report rules.
    # Under execution_mode="slurm" every rule gets its own sbatch, and a non-GPU rule submitted
    # to the GPU partition is rejected by Rivanna's `gpu` QOS (QOSMinGRES requires >=1 GPU).
    # UVA -> "standard" (CPU-only). Frontier runs in local mode so this value is inert, but set
    # it to "batch" (Frontier's one partition) for correctness if it ever runs per-rule.
    "uva": dict(
        gpu_partition="gpu", cpu_partition="standard", multi_sim_run_method="batch_job", execution_mode="slurm"
    ),
    "frontier": dict(
        gpu_partition="batch",
        cpu_partition="batch",
        multi_sim_run_method="1_job_many_srun_tasks",
        execution_mode="local",
    ),
}


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
    + account) is the file named by ``hpc_system_config_yaml`` (argv[2] when run as a script),
    passed to ``hhemt.experiment_bundle.resolve_hpc_system_config`` as its override; with no
    override the resolver raises, because it reads no environment variable. NO git-tracked
    file is edited per run. ``hpc_account``
    is sourced from the resolved config's ``default_account``; a fail-fast rejects an
    unfilled template. ``execution_environment`` is supplied PER ROW by the CSV.
    """
    if cluster not in _CLUSTER:
        raise ValueError(f"cluster must be one of {sorted(_CLUSTER)}; got {cluster!r}")
    c = _CLUSTER[cluster]
    from hhemt.config.loaders import load_hpc_system_config
    from hhemt.experiment_bundle import resolve_hpc_system_config

    cfg_path = resolve_hpc_system_config(cluster, override=hpc_system_config_yaml)
    cfg_hpc = load_hpc_system_config(cfg_path)
    account = cfg_hpc.default_account or ""
    if (not account) or ("{your-" in account):
        raise ValueError(
            f"{cfg_path}: default_account is unset or still a placeholder ({account!r}). "
            f"Set default_account to your real OLCF project / UVA allocation."
        )
    if cfg_hpc.container is None or "{your-" in str(cfg_hpc.container.sif_root or ""):
        raise ValueError(
            f"{cfg_path}: container.sif_root is missing or still a placeholder. "
            f"Set it to the absolute on-cluster root `hhemt build-sifs` writes images under."
        )
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=f"container_validation_{cluster}",
        sensitivity_csv=_SUITE,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        # SHARED-FILESYSTEM STAGING. `run_and_verdict` reaches `analysis.run(execution_mode=
        # _CLUSTER[cluster]["execution_mode"])`, which for uva is "slurm" -- an EXPLICIT
        # override of `run()`'s auto-resolution ("auto" derives the locus from
        # multi_sim_run_method; "local"/"slurm" override it), so the `local` forcing of
        # multi_sim_run_method does NOT make the run node-local; every rule is dispatched as
        # its own sbatch. This call supplies no `system_directory`, so under
        # `start_from_scratch=True` the builder redirects BOTH the config artifacts and the
        # analysis tree to a `tempfile.mkdtemp()` root (the `not share_scratch_root` branch
        # in test_case_builder.py) that those rules cannot see. That redirect became the
        # default at bc044e89 (2026-08-23), three weeks after this experiment's last green UVA
        # run (35 of 35), so the breakage is real and simply unobserved.
        # `share_scratch_root=True` is the opt-out the branch itself declares for the case
        # where "a LATER reader must see the tree this build produces" -- a rule on another
        # node is such a reader. It restores the
        # pre-bc044e89 staging root, which is the shared slug cache, and re-enables the
        # construction-time wipe against `container_validation_{cluster}` -- an analysis name
        # no pytest fixture and no other campaign uses. `_software_root` is pinned outside
        # this branch (test_case_builder.py, `self._software_root = _slug_runs_root /
        # "_software"`), so the borrow and build-tier invariants are untouched.
        share_scratch_root=True,
        hpc_system_config_yaml=cfg_path,
        additional_system_configs=(
            # PIN THE NATIVE ARM TO THE SIF'S SHA, not to the fixture default.
            # The invariant this experiment rests on is: the native arm's TRITON
            # commit == the container's `org.hhemt.triton_sha` label. That held
            # only by coincidence while the fixture default happened to equal the
            # SIF sha. Once the fixture pin moved forward, inheriting it would
            # make the two arms DIFFERENT SOLVERS, and check_cross_sim_identity
            # would report DIVERGED with nothing in the output naming the cause —
            # the divergence would read as a containerization defect.
            # Bump this only together with a SIF rebuild at the new sha.
            # The pin is on BOTH branches: the conditional governs only whether a
            # system_directory override is supplied, never whether the arm is pinned.
            {
                "TRITONSWMM_branch_key": "15eb18a5d25afe5da295cb4b559a62669dbe5bc3",
                **({"system_directory": system_directory} if system_directory else {}),
            }
        ),
        additional_analysis_configs={
            # batch_job/1_job_many_srun_tasks = login-node operator submission (default per
            # cluster). run_and_verdict overrides this to "local". Under Frontier's
            # execution_mode="local" that runs the sims in-process via the inner srun, with no
            # SLURM-executor plugin; under UVA's execution_mode="slurm" the explicit mode wins
            # and every rule is still its own sbatch (see the staging note above).
            "multi_sim_run_method": multi_sim_run_method or c["multi_sim_run_method"],
            "hpc_account": account,
            "hpc_max_simultaneous_sims": 1000,
            "hpc_total_job_duration_min": 60,
            "hpc_gpus_per_node": 8,
            "hpc_time_min_per_sim": 30,
            "hpc_restart_times_simulate": 2,
            "hpc_restart_times_other": 2,
            "hpc_ensemble_partition": c["gpu_partition"],
            "hpc_setup_and_analysis_processing_partition": c["cpu_partition"],
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(_SUITE),
            "report": {
                "reporting_set": "benchmarking",
                "sensitivity": {
                    # n_devices is the scaling x-axis (1 vs 2 GPU); execution_environment
                    # (native vs container) is the group overlay — together a real
                    # strong-scaling + container-overhead figure. n_devices is a derived
                    # column the renderer synthesizes from the (prefixed) compute columns.
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "analysis.execution_environment",
                },
            },
        },
    )


def run_and_verdict(cluster: str, *, start_from_scratch: bool = True, hpc_system_config_yaml: str | None = None):
    """Build + run the suite on ``cluster``, print the within-family native≡container verdict,
    and return True only for a real pass.

    Where the sims run follows ``_CLUSTER[cluster]["execution_mode"]``: on Frontier
    ("local") they run in-process on the allocated GPU node, so call it from inside a GPU
    sbatch allocation and the runner submits no jobs itself; on UVA ("slurm") every rule is
    submitted as its own SLURM job. Returns False on DIVERGED and on a no-data result, which
    ``check_cross_sim_identity`` reports as passed."""
    from hhemt.eda.cross_sim_identity import check_cross_sim_identity

    # multi_sim_run_method="local" replaces the login-node batch_job submission path. On
    # Frontier, execution_mode="local" then runs the sims (incl. the multi-rank container
    # rows via the inner srun) IN-PROCESS within this GPU allocation; on UVA,
    # execution_mode="slurm" still dispatches each rule as its own sbatch.
    tc = build_case(
        cluster,
        start_from_scratch=start_from_scratch,
        multi_sim_run_method="local",
        hpc_system_config_yaml=hpc_system_config_yaml,
    )
    # Set the GPU partition selector and the local orchestration on the master AND on every
    # member: the members re-load their configs from disk, and the n_gpus>0 rows and the
    # GPU-sensitivity validation resolve through that selector.
    tc.analysis.cfg_analysis.hpc_ensemble_partition = _CLUSTER[cluster]["gpu_partition"]
    tc.analysis.cfg_analysis.multi_sim_run_method = "local"
    sens = getattr(tc.analysis, "sensitivity", None)
    if sens is not None:
        for sub in getattr(sens, "analyses", {}).values():
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
    # No-data guard: check_cross_sim_identity returns passed=True
    # with an "N/A — no … summaries" verdict when the sims produced NOTHING to compare
    # (a rule failed upstream). That is NOT a real native≡container result.
    details = v.details or []
    summaries_absent = any("summaries absent" in str(d.get("detail", "")) for d in details if isinstance(d, dict))
    no_data = (
        bool(getattr(result, "skipped", False))
        or ("N/A" in (v.summary or ""))
        or ("no member" in (v.summary or ""))
        or summaries_absent
    )
    if no_data:
        print(
            "OVERALL: NO-DATA — sims produced no summaries; not a real native=container comparison (run failed "
            "upstream)"
        )
        return False
    real_pass = bool(v.passed)
    print("OVERALL:", "native=container within-family" if real_pass else "DIVERGED")
    return real_pass


if __name__ == "__main__":
    import sys

    cluster = sys.argv[1] if len(sys.argv) > 1 else "uva"
    cfg_override = sys.argv[2] if len(sys.argv) > 2 else None
    # argv[2] is this driver's only HPC-config channel. Without this guard a bare run surfaces
    # the library resolver's refusal, which names --hpc-system-config and experiment.yaml,
    # neither of which this script reads, so the operator is pointed at a fix that cannot work.
    if cfg_override is None:
        sys.exit(
            "container_validation.py: pass your cluster's hpc_system_config YAML as argv[2], e.g.\n"
            f"  python scripts/experiments/container_validation.py {cluster} "
            f"/path/to/hpc_system_config_{cluster}.yaml"
        )
    # run_and_verdict returns True only for a REAL within-family pass (no-data guarded);
    # exit non-zero otherwise so the SLURM job state reflects the result.
    sys.exit(0 if run_and_verdict(cluster, hpc_system_config_yaml=cfg_override) else 1)
