"""Leaf compute-config arithmetic — stdlib-only, NO hhemt imports.

The single definition of a compute config's DEVICE COUNT: the number of parallel
work units a simulation occupies. Extracted so `eda/_config_diff.py::_device_count`
and `report_renderers/sensitivity_benchmarking.py::_ensure_n_devices_column` compute
one quantity instead of two that agree by coincidence. A leaf module (the
`slurm_liveness.py` pattern) because the two callers are peers in different
subpackages and an `eda -> report_renderers` import would be a cycle.

WHAT LIVES HERE: the arithmetic, and nothing else.

WHAT DELIBERATELY DOES NOT, because the two callers differ and MUST keep differing:

1. COERCION. `_device_count` reads a zarr `attrs` dict through `_to_int`, which
   swallows TypeError/ValueError to 0, then floors ranks and threads at 1 via
   `max(x, 1)`. The renderer reads a DataFrame through bare-or-`analysis.`-prefixed
   column resolution and casts the result with `.astype(int)`. Neither coercion is
   correct for the other's input.

2. MISSING-DATA POLICY. The renderer RAISES `ValueError` when a required column is
   absent AND `independent_var == "n_devices"` (the figure's x axis would be
   undefined), and returns the frame unchanged otherwise. `_device_count` never
   raises — an absent key is 0, because it is an ORDERING key and a panel must sort
   rather than abort.

3. THE GPU PREDICATE, which is the divergence most likely to be "unified" by mistake.
   `_device_count` asks `run_mode == "gpu"`. The renderer asks
   `(run_mode == "gpu") | (n_gpus > 0)` -- deliberately BROADER, so a hybrid row that
   also requested a GPU is counted in devices rather than in cores. Folding the
   predicate in here would silently pick one and change the other.

So `is_gpu` arrives already decided. This function cannot express any of the three.
"""

from __future__ import annotations

__all__ = ["n_devices_from"]


def n_devices_from(*, is_gpu: bool, n_gpus: int, n_mpi_procs: int, n_omp_threads: int) -> int:
    """Parallel work units for one compute config.

    GPUs for a GPU config; ranks x threads-per-rank for a CPU config. There is NO
    `n_nodes` factor: `n_mpi_procs` is TOTAL ranks per simulation, not ranks per node
    (`config/analysis.py` declares "Number of MPI ranks per simulation", and
    `run_simulation.py` emits `srun -N {n_nodes} --ntasks={n_mpi_procs}` where SLURM's
    --ntasks is the total), so ranks x threads is already the full parallel width.

    Keyword-only by construction: the four operands are same-typed ints whose order a
    caller cannot get wrong if it cannot pass them positionally.
    """
    return int(n_gpus) if is_gpu else int(n_mpi_procs) * int(n_omp_threads)
