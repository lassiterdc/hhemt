"""Synthetic compute-config sensitivity experiment framework and entry point.

The module owns the compute-config sweep enumeration. The enumeration holds the fixed GPU, serial, openmp, and
hybrid rows plus the mpi-rank rows generated from ``synthetic_experiment_config.rank_sweep``
with the default ``(2, 4, 8)``. ``write_clean_matrix_csv`` and ``write_resume_matrix_csv`` write
the enumeration as the matrix CSVs.

A set ``dem_resolution_ladder`` selects the DEM-resolution rows. One fixed compute
configuration runs at every cell size of the ladder, built by ``dem_resolution_matrix_rows``.

The public surface:

- ``experiment_matrix_rows(cfg) -> list[dict]`` is the shared row enumeration. The
  config ``_validate_caps`` guard also consumes the rows.
- ``build_experiment_matrix(cfg) -> pandas.DataFrame`` runs the coupling-node precheck
  first when ``dem_resolution_ladder`` is set.
- ``write_clean_matrix_csv(path, *, rank_sweep=...)`` writes the clean matrix CSV.
- ``write_resume_matrix_csv(path, *, runtime_min_by_member=None, rank_sweep=..., ...)`` writes the resume matrix CSV.
- ``generate_synthetic_experiment(cfg, dest_dir) -> Path`` builds the synth case.
"""

from __future__ import annotations

# This module must not import from ``scripts/``: a ``src -> scripts`` import would break
# ``pip install -e .``, so the matrix enumeration lives in ``src/``.
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # avoid a runtime config <-> framework import cycle
    from hhemt.config.synthetic_experiment import synthetic_experiment_config

_DEFAULT_RANK_SWEEP: tuple[int, ...] = (2, 4, 8)
_CLEAN_WALLTIME_MIN: int = 30  # generous single-allocation walltime for the clean sweep

#: The two model arms of the compute-config experiment, as (arm_token, system-toggle overlay).
#: The arms run as SIBLING sensitivity primaries, never as one multi-model primary:
#: workflow.py::generate_master_snakefile_content raises
#: ValueError("Sensitivity analysis does not support multi-model execution") on >1
#: enabled model and threads a scalar model_type into every per-member_id flag name.
#: Each arm's matrix is byte-identical; only the system toggles differ, so no matrix
#: column is added and no per-member_id input fingerprint is rewritten.
_MODEL_ARMS: dict[str, dict[str, bool]] = {
    "tritonswmm": {
        "toggle_tritonswmm_model": True,
        "toggle_triton_model": False,
        "toggle_swmm_model": False,
    },
    "triton": {
        "toggle_tritonswmm_model": False,
        "toggle_triton_model": True,
        "toggle_swmm_model": False,
    },
}


def model_arm_toggles(arm: str) -> dict[str, bool]:
    """The system-config toggle overlay for one model arm.

    Raises KeyError on an unknown arm and avoids defaulting, so a typo in the
    deployment config fails at case construction and produces no silently
    coupled run mislabelled as uncoupled.
    """
    return dict(_MODEL_ARMS[arm])


# Canonical partition-as-axis column schema (VMS-1a): the retired
# system.gpu_hardware / system.gpu_compilation_backend overlay columns are dropped
# (GPU hardware DERIVES from the partition's PartitionSpec, Gotcha 54); the
# partition column is the canonical `hpc.partition` alias.
_COLS = [
    "member_id",
    "analysis.run_mode",
    "analysis.n_nodes",
    "analysis.n_mpi_procs",
    "analysis.n_omp_threads",
    "analysis.n_gpus",
    "hpc.partition",
    "analysis.mem_gb_per_cpu",
    "analysis.hpc_time_min_per_sim",
    "system.target_dem_resolution",
]

# Fixed non-mpi configs (partition-as-axis; tuple shape
# (run_mode, n_nodes, n_mpi, n_omp, n_gpus, partition, mem_gb_per_cpu)). The mpi
# rows are generated from rank_sweep and spliced in AFTER the openmp rows so the
# GLOBAL enumerate index (and thus the member_id) matches the historical baseline.
_GPU_CONFIGS = [
    ("gpu", 1, 1, 1, 1, "gpu-a6000", 8),
    ("gpu", 1, 2, 1, 2, "gpu-a6000", 8),
    ("gpu", 1, 3, 1, 3, "gpu-a6000", 8),
    ("gpu", 1, 1, 1, 1, "gpu-a100-80", 8),
    ("gpu", 1, 2, 1, 2, "gpu-a100-80", 8),
    ("gpu", 1, 3, 1, 3, "gpu-a100-80", 8),
]
_SERIAL_CONFIGS = [
    ("serial", 1, 1, 1, 0, "standard", 2),
]
_OPENMP_CONFIGS = [
    ("openmp", 1, 1, 2, 0, "standard", 2),
    ("openmp", 1, 1, 8, 0, "standard", 2),
]
_HYBRID_CONFIGS = [
    ("hybrid", 1, 2, 2, 0, "standard", 2),
    ("hybrid", 1, 4, 2, 0, "standard", 2),
    # The SECOND 8-device hybrid decomposition. With the x axis at total devices,
    # (4 ranks x 2 threads) and (2 ranks x 4 threads) both land at N=8 and share the
    # hybrid colour, symbol and hollow fill -- so the per-point threads-per-rank
    # annotation is the ONLY discriminator, which is what makes the annotation
    # demonstrable in the default matrix rather than only in principle. 8 admits no
    # third non-degenerate hybrid split: 8x1 is pure MPI and 1x8 is pure OpenMP.
    # rank 2 is already in the default rank_sweep, so _validate_coupling_invariant
    # (which iterates rank_sweep only) is unchanged by this row.
    ("hybrid", 1, 2, 4, 0, "standard", 2),
]


def _configs(rank_sweep: tuple[int, ...]) -> list[tuple]:
    """The ordered compute-config sweep. The mpi rows are generated from
    ``rank_sweep`` (one row per rank, ``n_mpi_procs == rank``) and spliced between
    the openmp and hybrid rows so the enumerate index reproduces the baseline
    (``rank_sweep=(2,4,8)`` -> mpi rows at global indices 9/10/11)."""
    mpi = [("mpi", 1, int(r), 1, 0, "standard", 2) for r in rank_sweep]
    return _GPU_CONFIGS + _SERIAL_CONFIGS + _OPENMP_CONFIGS + mpi + _HYBRID_CONFIGS


def _rows(configs: list[tuple], *, walltime_min: int | None, replicates: int = 2) -> list[dict]:
    """Expand configs x replicates into CSV row dicts.

    ``member_id = f"{run_mode}_{i}_r{rep}"`` where ``i`` is the GLOBAL enumerate index
    into ``configs`` (not a per-run-mode counter) so same-run-mode configs stay
    unique; all tokens are charset-safe (``^[A-Za-z0-9_.]+$``). ``walltime_min=None``
    leaves ``hpc_time_min_per_sim`` blank for a per-row caller (resume sizing).
    3.5 m native rows leave ``system.target_dem_resolution`` blank (NaN).
    """
    rows: list[dict] = []
    for i, (run_mode, n_nodes, n_mpi, n_omp, n_gpus, part, mem) in enumerate(configs):
        for rep in range(1, replicates + 1):
            rows.append(
                {
                    "member_id": f"{run_mode}_{i}_r{rep}",
                    "analysis.run_mode": run_mode,
                    "analysis.n_nodes": n_nodes,
                    "analysis.n_mpi_procs": n_mpi,
                    "analysis.n_omp_threads": n_omp,
                    "analysis.n_gpus": n_gpus,
                    "hpc.partition": part,
                    "analysis.mem_gb_per_cpu": mem,
                    "analysis.hpc_time_min_per_sim": walltime_min,
                    "system.target_dem_resolution": None,
                }
            )
    return rows


#: The fixed compute config every DEM-resolution row runs at. The sweep varies
#: ONE thing (cell size); holding the compute config fixed is the experimental
#: control. Shape matches _SERIAL_CONFIGS' tuple:
#: (run_mode, n_nodes, n_mpi, n_omp, n_gpus, partition, mem_gb_per_cpu).
_DEM_FIXED_CONFIG = ("serial", 1, 1, 1, 0, "standard", 2)


def dem_resolution_matrix_rows(
    cfg: synthetic_experiment_config,
    *,
    walltime_min: int | None = None,
    replicates: int = 2,
) -> list[dict]:
    """Expand the DEM-resolution ladder x replicates into CSV row dicts.

    The DEM-resolution experiment is the transpose of the compute-config experiment.
    ONE fixed compute config, ``_DEM_FIXED_CONFIG``, is swept across N cell sizes, whereas
    the compute-config experiment sweeps one fixed cell size across N configs.

    ``member_id = f"dem_{res_token}_r{rep}"`` where ``res_token`` is the cell size with
    '.' -> 'p' for charset safety per ``^[A-Za-z0-9_.]+$``. '.' is legal but 'p' keeps the
    member_id readable as a single token. The ladder comes from ``cfg.dem_resolution_ladder``,
    and the config validates the ladder as divisor-only and constant-ratio.

    The FINEST rung is the reference, and each coarser rung is compared against the finest.
    The finest rung is emitted like any other row, because the reference status of the rung is a figure-time fact
    and no matrix-time fact.
    """
    run_mode, n_nodes, n_mpi, n_omp, n_gpus, part, mem = _DEM_FIXED_CONFIG
    rows: list[dict] = []
    for res in cfg.dem_resolution_ladder:
        res_token = str(res).replace(".", "p")
        for rep in range(1, replicates + 1):
            rows.append(
                {
                    "member_id": f"dem_{res_token}_r{rep}",
                    "analysis.run_mode": run_mode,
                    "analysis.n_nodes": n_nodes,
                    "analysis.n_mpi_procs": n_mpi,
                    "analysis.n_omp_threads": n_omp,
                    "analysis.n_gpus": n_gpus,
                    "hpc.partition": part,
                    "analysis.mem_gb_per_cpu": mem,
                    "analysis.hpc_time_min_per_sim": walltime_min,
                    "system.target_dem_resolution": res,
                }
            )
    return rows


def assert_coupling_nodes_distinct(
    cfg: synthetic_experiment_config,
) -> dict[float, int]:
    """Geometric pre-check that IN-LINE coupling nodes occupy distinct DEM cells.

    The retained in-line count MUST stay >= the largest MPI rank in the experiment.
    A rung that drops the retained count below the rank count is unrunnable, and the
    _N_COUPLING_NODES def in swmm_template.py explains WHY node_count caps
    rank_count. The precheck is the ladder-time enforcement of that cap.

    Returns ``{cell_size_m -> retained_coupling_node_count}`` for every rung in
    ``cfg.dem_resolution_ladder`` -- the same quantity the resolution x coupling-node
    table renders, so the safety check and the disclosure artifact are one
    computation. Monotone non-increasing as the rung coarsens.

    The node set is the ``_N_COUPLING_NODES`` in-line rank-coverage junctions --
    ``swmm_template._node_matrix_rows`` placed on the conduit centerline column
    ``_centerline_col`` -- the same set that ``config.synthetic_experiment.
    _validate_coupling_invariant`` guarantees covers every TRITON row-strip rank at
    native resolution. The OTHER SWMM nodes ``_nodes`` returns are DELIBERATELY
    excluded because none of them can drop the coupling inflow of an in-line node:
      * ``dummy_outfall`` is a DISCONNECTED outfall with no downstream conduit, so the
        most-downstream-per-cell reduction in ``scenario_inputs.
        update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell`` never selects
        the outfall over a connected junction. Iterating the outfall would false-positively reject the
        canonical ladder -- the outfall bins into the cell of ``J1`` at the 14 m rung on
        the default 64x120 grid.
      * each ``_branches`` junction is an UPSTREAM tributary feeding a stem attach, so
        if the junction ever shared a coarse cell with a stem node, the more downstream
        stem node is the one KEPT -- in-line coverage is preserved either way.
      * ``sewer_outflow`` shares the centerline but sits below the sea wall, always
        co-resident in the bottom rank-strip with ``collector``, so a win of the cell of
        ``collector`` by ``sewer_outflow`` leaves the strip non-empty.

    Coarsening the DEM SHRINKS the coupling network.
    ``scenario_inputs.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell``
    groups SWMM nodes by the containing cell in the PROCESSED DEM after coarsening and
    retains the inflow assignment of only the most downstream node per cell. Fewer
    retained in-line nodes than MPI ranks => a node-free top rank => coupling-collective
    deadlock. The ``_N_COUPLING_NODES`` definition in ``swmm_template`` explains the deadlock in a
    comment block that states that every rank must own at least one coupling node or the rank never
    enters the TRITON-SWMM ``ENSIFY_COMM_WORLD`` coupling collective, so every rank
    blocks forever. A deadlock is a HANG and gives no error, the worst failure shape on
    HPC, so the check converts the deadlock into a plan-time constraint on the coarsest rung.

    Raises ProcessingError naming the offending rung and the colliding node pair.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.synthetic_model import SyntheticModelParams
    from hhemt.synthetic_model.swmm_template import (
        _N_COUPLING_NODES,
        _centerline_col,
        _node_matrix_rows,
    )

    params = SyntheticModelParams(n_cols=cfg.n_cols, n_rows=cfg.n_rows, cell_size_m=cfg.cell_size_m)
    col = _centerline_col(params)
    mrs = _node_matrix_rows(params)
    # Reconstruct the in-line chain names + row_from_bottom exactly as ``_nodes`` does
    # (`swmm_template.py::_nodes`): J1..J(n-1) then ``collector`` (southernmost).
    names = [f"J{i + 1}" for i in range(len(mrs) - 1)] + ["collector"]
    inline_nodes = [(names[i], col, params.n_rows - 1 - mrs[i]) for i in range(len(mrs))]

    retained: dict[float, int] = {}
    for res in cfg.dem_resolution_ladder:
        seen: dict[tuple[int, int], str] = {}
        for name, node_col, row_from_bottom in inline_nodes:
            x = cfg.cell_size_m * (node_col + 0.5)
            y = cfg.cell_size_m * (row_from_bottom + 0.5)
            cell = (int(x // res), int(y // res))
            if cell in seen:
                raise ProcessingError(
                    operation="dem_resolution_ladder_precheck",
                    filepath=None,
                    reason=(
                        f"in-line coupling nodes {seen[cell]!r} and {name!r} share DEM cell "
                        f"{cell} at cell_size_m={res}. Coarsening would drop one node's inflow "
                        f"assignment, shrinking the in-line coupling network below "
                        f"_N_COUPLING_NODES={_N_COUPLING_NODES} and risking a coupling-collective "
                        f"deadlock (a rank owning zero coupling nodes -- a HANG). Remove this rung "
                        f"from dem_resolution_ladder or coarsen no further."
                    ),
                )
            seen[cell] = name
        retained[res] = len(seen)
    return retained


def experiment_matrix_rows(cfg: synthetic_experiment_config) -> list[dict]:
    """The shared experiment-matrix row enumeration for ``cfg`` at the clean walltime.

    Consumed by ``build_experiment_matrix`` AND by the config's ``_validate_caps``
    guard, a single enumeration so the validated matrix and the emitted matrix
    always agree.

    A set ``cfg.dem_resolution_ladder`` selects the rows of
    ``dem_resolution_matrix_rows``, and an empty ladder selects the compute-config
    rows. Both carry the clean walltime.
    """
    if cfg.dem_resolution_ladder:
        return dem_resolution_matrix_rows(cfg, walltime_min=_CLEAN_WALLTIME_MIN)
    return _rows(_configs(tuple(cfg.rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN)


def build_experiment_matrix(cfg: synthetic_experiment_config) -> pd.DataFrame:
    """The experiment matrix of ``cfg`` as a DataFrame with the canonical columns.

    A set ``cfg.dem_resolution_ladder`` runs ``assert_coupling_nodes_distinct`` first,
    so a ladder with a rung that merges in-line coupling nodes raises
    ``ProcessingError`` before any row is emitted.
    """
    if cfg.dem_resolution_ladder:
        assert_coupling_nodes_distinct(cfg)
    return pd.DataFrame(experiment_matrix_rows(cfg), columns=_COLS)


def write_clean_matrix_csv(path: Path, *, rank_sweep: tuple[int, ...] = _DEFAULT_RANK_SWEEP) -> None:
    """Clean experiment CSV with a generous walltime guaranteeing single-allocation
    completion. ``rank_sweep`` default reproduces the historical baseline."""
    df = pd.DataFrame(_rows(_configs(tuple(rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN), columns=_COLS)
    df.to_csv(path, index=False)


def write_smoke_matrix_csv(path: Path) -> None:
    """Single-row cheap-confirmation CSV with a serial-CPU row on ``standard``,
    the cheapest compute config, with a single replicate, for the multi-resume
    empirical check. The CSV is no matrix sweep, because the resume harness is confirmed
    per-sim and one row suffices. Consumed by ``synth_compute_config.smoke_case``.
    """
    rows = _rows(_SERIAL_CONFIGS, walltime_min=_CLEAN_WALLTIME_MIN, replicates=1)
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False)


def write_resume_matrix_csv(
    path: Path,
    *,
    rank_sweep: tuple[int, ...] = _DEFAULT_RANK_SWEEP,
) -> None:
    """Resume sweep CSV for the multi-resume deterministic interruption harness.

    The mid-sim kills are DETERMINISTIC and never a walltime expiry. Each kill is a SIGKILL at a cumulative hotstart
    checkpoint index in the schedule. The kill is wired in ``run_simulation_runner`` and armed
    via the analysis config field ``resume_interruption_schedule``. So every row gets the SAME generous
    ``_CLEAN_WALLTIME_MIN`` walltime
    as the clean sweep, and the resume attempt always has budget to finish. The
    resume arm is distinguished from the clean arm ONLY by the analysis-config
    kill field, set in ``resume_case``, and never by a short walltime. The design retires
    the old ``round(T_member / kill_divisor)`` short-walltime sizing and the
    ``runtime_min_by_member`` pass dependency of the sizing, removing the thin-window
    fragility that stalled the earlier resume sweep.
    """
    rows = _rows(_configs(tuple(rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN)
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False)


def generate_synthetic_experiment(cfg: synthetic_experiment_config, dest_dir: Path) -> Path:
    """Build the synthetic TRITON-SWMM case for ``cfg`` under ``dest_dir`` and
    return the case directory.

    Threads the config model knobs for grid and forcing into
    ``SyntheticModelParams`` and delegates to the lifted
    ``hhemt.synthetic_model.build_synthetic_case``. The remaining
    ``SyntheticModelParams`` fields cover event shaping, namely sim duration, compound
    surge, and reporting cadence, and retain their generic defaults. The shaping of the production experiment lives in
    ``scripts/experiments/synth_compute_config.py::_EXPERIMENT_PARAMS`` and not in the config model.
    """
    from hhemt.synthetic_model import SyntheticModelParams, build_synthetic_case

    params = SyntheticModelParams(
        n_cols=cfg.n_cols,
        n_rows=cfg.n_rows,
        cell_size_m=cfg.cell_size_m,
        rainfall_peak_mm_per_hr=cfg.rainfall_peak_mm_per_hr,
    )
    dest_dir = Path(dest_dir)
    build_synthetic_case(params, dest_dir)
    return dest_dir
