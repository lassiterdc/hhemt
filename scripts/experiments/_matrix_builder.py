"""Compute-config sensitivity-matrix CSV builders for the synth experiment.

Row-count-flexible (the test-fixture ``_write_synth_sensitivity_csv`` is locked to 4/3
rows). Column schema mirrors ``full_benchmarking_experiment_uva.xlsx`` (12 columns).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (run_mode, n_nodes, n_mpi, n_omp, n_gpus, partition, mem_gb_per_cpu, gpu_hardware, backend)
# One representative per byte-group (D1 subset): GPU a6000/a100 x {1,2,3} + CPU
# serial/openmp/mpi/hybrid ladders.
_CLEAN_CONFIGS = [
    ("gpu", 1, 1, 1, 1, "gpu-a6000", 8, None, None),  # a6000 1-GPU (master defaults -> NaN overlay)
    ("gpu", 1, 2, 1, 2, "gpu-a6000", 8, None, None),
    ("gpu", 1, 3, 1, 3, "gpu-a6000", 8, None, None),
    ("gpu", 1, 1, 1, 1, "gpu-a100-80", 8, "a100", "CUDA"),
    ("gpu", 1, 2, 1, 2, "gpu-a100-80", 8, "a100", "CUDA"),
    ("gpu", 1, 3, 1, 3, "gpu-a100-80", 8, "a100", "CUDA"),
    ("serial", 1, 1, 1, 0, "standard", 2, None, None),
    ("openmp", 1, 1, 2, 0, "standard", 2, None, None),
    ("openmp", 1, 1, 8, 0, "standard", 2, None, None),
    ("mpi", 1, 2, 1, 0, "standard", 2, None, None),
    ("mpi", 1, 4, 1, 0, "standard", 2, None, None),
    ("mpi", 1, 8, 1, 0, "standard", 2, None, None),
    ("hybrid", 1, 2, 2, 0, "standard", 2, None, None),
    ("hybrid", 1, 4, 2, 0, "standard", 2, None, None),
]  # 14 unique configs x2 replicates = 28 rows (fixed at 28 — Decision 5; queue is
# uncapped per hpc_max_simultaneous_sims=1000, so the row count is not queue-budget-bound)

_COLS = [
    "sa_id",
    "run_mode",
    "n_nodes",
    "n_mpi_procs",
    "n_omp_threads",
    "n_gpus",
    "hpc_ensemble_partition",
    "mem_gb_per_cpu",
    "hpc_time_min_per_sim",
    "system.target_dem_resolution",
    "system.gpu_hardware",
    "system.gpu_compilation_backend",
]


def _rows(configs, *, walltime_min: int | None, replicates: int = 2):
    """Expand configs x replicates into CSV rows; 3.5m res left NaN.

    ``sa_id = f"{run_mode}_{i}_r{rep}"`` where ``i`` is the GLOBAL enumerate index into
    ``configs`` (NOT a per-run-mode counter) so same-run-mode configs (3 mpi, 2 hybrid,
    2 openmp) stay unique; all tokens are charset-safe (``^[A-Za-z0-9_.]+$``).
    ``walltime_min=None`` => per-row in the caller (Phase 2 resume). 3.5m rows leave
    ``system.target_dem_resolution`` blank (NaN).
    """
    rows = []
    for i, (run_mode, n_nodes, n_mpi, n_omp, n_gpus, part, mem, hw, backend) in enumerate(configs):
        for rep in range(1, replicates + 1):
            rows.append(
                {
                    "sa_id": f"{run_mode}_{i}_r{rep}",
                    "run_mode": run_mode,
                    "n_nodes": n_nodes,
                    "n_mpi_procs": n_mpi,
                    "n_omp_threads": n_omp,
                    "n_gpus": n_gpus,
                    "hpc_ensemble_partition": part,
                    "mem_gb_per_cpu": mem,
                    "hpc_time_min_per_sim": walltime_min,
                    "system.target_dem_resolution": None,
                    "system.gpu_hardware": hw,
                    "system.gpu_compilation_backend": backend,
                }
            )
    return rows


def write_clean_matrix_csv(path: Path) -> None:
    """Clean experiment: generous walltime guaranteeing single-allocation completion (30 min)."""
    df = pd.DataFrame(_rows(_CLEAN_CONFIGS, walltime_min=30), columns=_COLS)
    df.to_csv(path, index=False)


def write_resume_matrix_csv(
    path: Path,
    *,
    runtime_min_by_sa: dict[str, float] | None = None,
    kill_divisor: int = 3,
    min_walltime_min: int = 1,
) -> None:
    """Resume sweep: per-backend walltime sized to force a mid-sim kill AND complete
    within ``restart-times`` from ONE ``analysis.run()``.

    For each row, ``hpc_time_min_per_sim = max(min_walltime_min, round(T_sa / kill_divisor))``
    where ``T_sa`` is that backend's measured full-completion wallclock (minutes) from the
    CLEAN sweep (SLURM ``Elapsed`` via sacct, or ``out_tritonswmm/performance.txt`` Total),
    keyed by ``sa_id``. With ``kill_divisor=3`` each sim is killed ~2x and finishes on
    attempt ~3; set ``hpc_restart_times`` comfortably above
    ``ceil(max(T_sa) / min_walltime_min)`` so even a worst-case slow backend completes.
    When ``runtime_min_by_sa`` is None (off-cluster dry-run only), fall back to a
    conservative GPU=4 min / CPU=18 min estimate by row type — REPLACE with real
    clean-sweep numbers before the production resume run.
    """
    runtimes = runtime_min_by_sa or {}
    rows = _rows(_CLEAN_CONFIGS, walltime_min=None)
    for r in rows:
        t_full = runtimes.get(r["sa_id"], 4.0 if r["n_gpus"] else 18.0)
        r["hpc_time_min_per_sim"] = max(min_walltime_min, round(t_full / kill_divisor))
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False)
