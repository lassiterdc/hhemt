"""Synthetic compute-config sensitivity experiment framework (entry point).

SINGLE SOURCE of the experiment matrix (F-B-1): this module owns the
compute-config sweep enumeration — the fixed GPU/serial/openmp/hybrid rows plus
the mpi-rank rows GENERATED from ``synthetic_experiment_config.rank_sweep`` (the
default ``(2, 4, 8)`` reproduces the historical 28-row baseline byte-for-byte).
``scripts/experiments/_matrix_builder.py`` is retired; its sole importer
(``synth_compute_config.py``) re-points at the ``write_clean_matrix_csv`` /
``write_resume_matrix_csv`` writers here. A ``src -> scripts`` import would break
``pip install -e .``, so the enumeration lives in ``src/``.

Public surface:
    experiment_matrix_rows(cfg)   -> list[dict]   (the shared row enumeration; the
                                     config _validate_caps guard also consumes it)
    build_experiment_matrix(cfg)  -> pandas.DataFrame
    write_clean_matrix_csv(path, *, rank_sweep=...)
    write_resume_matrix_csv(path, *, runtime_min_by_sa=None, rank_sweep=..., ...)
    generate_synthetic_experiment(cfg, dest_dir) -> Path   (build the synth case)
    size_resume_walltimes(clean_analysis) -> dict[str, int]  (FQ3 two-pass helper)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # avoid a runtime config <-> framework import cycle
    from hhemt.config.synthetic_experiment import synthetic_experiment_config

_DEFAULT_RANK_SWEEP: tuple[int, ...] = (2, 4, 8)
_CLEAN_WALLTIME_MIN: int = 30  # generous single-allocation walltime for the clean sweep

# Canonical partition-as-axis column schema (VMS-1a): the retired
# system.gpu_hardware / system.gpu_compilation_backend overlay columns are dropped
# (GPU hardware DERIVES from the partition's PartitionSpec, Gotcha 54); the
# partition column is the canonical `hpc.partition` alias.
_COLS = [
    "sa_id",
    "run_mode",
    "n_nodes",
    "n_mpi_procs",
    "n_omp_threads",
    "n_gpus",
    "hpc.partition",
    "mem_gb_per_cpu",
    "hpc_time_min_per_sim",
    "system.target_dem_resolution",
]

# Fixed non-mpi configs (partition-as-axis; tuple shape
# (run_mode, n_nodes, n_mpi, n_omp, n_gpus, partition, mem_gb_per_cpu)). The mpi
# rows are generated from rank_sweep and spliced in AFTER the openmp rows so the
# GLOBAL enumerate index (and thus the sa_id) matches the historical baseline.
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

    ``sa_id = f"{run_mode}_{i}_r{rep}"`` where ``i`` is the GLOBAL enumerate index
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
                    "sa_id": f"{run_mode}_{i}_r{rep}",
                    "run_mode": run_mode,
                    "n_nodes": n_nodes,
                    "n_mpi_procs": n_mpi,
                    "n_omp_threads": n_omp,
                    "n_gpus": n_gpus,
                    "hpc.partition": part,
                    "mem_gb_per_cpu": mem,
                    "hpc_time_min_per_sim": walltime_min,
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

    The DEM-resolution experiment is the compute-config experiment's transpose:
    ONE fixed compute config (_DEM_FIXED_CONFIG) swept across N cell sizes, where
    the compute-config experiment is one fixed cell size swept across N configs.

    ``sa_id = f"dem_{res_token}_r{rep}"`` where ``res_token`` is the cell size with
    '.' -> 'p' (charset-safe per ``^[A-Za-z0-9_.]+$``; '.' is legal but 'p' keeps the
    sa_id readable as one token). The ladder comes from ``cfg.dem_resolution_ladder``,
    which the config validates as divisor-only + constant-ratio.

    The FINEST rung is the reference (D3: each coarser rung is compared vs finest).
    It is emitted like any other row -- its reference status is a figure-time fact,
    not a matrix-time one.
    """
    run_mode, n_nodes, n_mpi, n_omp, n_gpus, part, mem = _DEM_FIXED_CONFIG
    rows: list[dict] = []
    for res in cfg.dem_resolution_ladder:
        res_token = str(res).replace(".", "p")
        for rep in range(1, replicates + 1):
            rows.append(
                {
                    "sa_id": f"dem_{res_token}_r{rep}",
                    "run_mode": run_mode,
                    "n_nodes": n_nodes,
                    "n_mpi_procs": n_mpi,
                    "n_omp_threads": n_omp,
                    "n_gpus": n_gpus,
                    "hpc.partition": part,
                    "mem_gb_per_cpu": mem,
                    "hpc_time_min_per_sim": walltime_min,
                    "system.target_dem_resolution": res,
                }
            )
    return rows


def assert_coupling_nodes_distinct(
    cfg: synthetic_experiment_config,
) -> dict[float, int]:
    """R14 geometric pre-check: no two IN-LINE coupling nodes share a DEM cell.

    The retained in-line count MUST stay >= the largest MPI rank in the experiment:
    a rung that drops it below the rank count is R14-unrunnable (see the
    _N_COUPLING_NODES def in swmm_template.py, which explains WHY node_count caps
    rank_count). This precheck is the ladder-time enforcement of that cap.

    Returns ``{cell_size_m -> retained_coupling_node_count}`` for every rung in
    ``cfg.dem_resolution_ladder`` -- the same quantity the resolution x coupling-node
    table renders, so the safety check and the disclosure artifact are one
    computation (D11). Monotone non-increasing as the rung coarsens.

    Node set: EXACTLY the ``_N_COUPLING_NODES`` in-line rank-coverage junctions --
    ``swmm_template._node_matrix_rows`` placed on the conduit centerline column
    ``_centerline_col`` -- the same set ``config.synthetic_experiment.
    _validate_coupling_invariant`` guarantees covers every TRITON row-strip rank at
    native resolution. The OTHER SWMM nodes ``_nodes`` returns are DELIBERATELY
    excluded because none of them can drop an in-line node's coupling inflow:
      * ``dummy_outfall`` is a DISCONNECTED outfall (no downstream conduit), so the
        most-downstream-per-cell reduction (scenario_inputs.py:157-205) never selects
        it over a connected junction. Iterating it would false-positively reject the
        canonical ladder -- it bins into ``J1``'s cell at the DoD-valid 14 m rung on
        the default 64x120 grid.
      * each ``_branches`` junction is an UPSTREAM tributary feeding a stem attach, so
        if it ever shared a coarse cell with a stem node the stem node (more
        downstream) is the one KEPT -- in-line coverage is preserved either way.
      * ``sewer_outflow`` shares the centerline but sits below the sea wall, always
        co-resident in the bottom rank-strip with ``collector``, so its winning
        ``collector``'s cell can never empty a strip.

    Coarsening the DEM SHRINKS the coupling network:
    ``scenario_inputs.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell``
    groups SWMM nodes by their containing cell in the PROCESSED (coarsened) DEM and
    retains the inflow assignment of only the most downstream node per cell. Fewer
    retained in-line nodes than MPI ranks => a node-free top rank => coupling-collective
    deadlock (swmm_template.py:122-131, triton.h:2363-2404). A deadlock is a HANG, not
    an error -- the worst failure shape on HPC -- so this converts it into a plan-time
    constraint on the coarsest rung.

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
    # (swmm_template.py:241-242): J1..J(n-1) then ``collector`` (southernmost).
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
    """The shared experiment-matrix row enumeration for ``cfg`` (clean walltime).

    Consumed by ``build_experiment_matrix`` AND by the config's ``_validate_caps``
    guard — a single enumeration so the validated matrix and the emitted matrix
    cannot drift.
    """
    return _rows(_configs(tuple(cfg.rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN)


def build_experiment_matrix(cfg: synthetic_experiment_config) -> pd.DataFrame:
    """The partition-as-axis sensitivity matrix as a DataFrame (canonical columns)."""
    return pd.DataFrame(experiment_matrix_rows(cfg), columns=_COLS)


def write_clean_matrix_csv(path: Path, *, rank_sweep: tuple[int, ...] = _DEFAULT_RANK_SWEEP) -> None:
    """Clean experiment CSV: generous walltime guaranteeing single-allocation
    completion. ``rank_sweep`` default reproduces the historical baseline."""
    df = pd.DataFrame(_rows(_configs(tuple(rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN), columns=_COLS)
    df.to_csv(path, index=False)


def write_resume_matrix_csv(
    path: Path,
    *,
    runtime_min_by_sa: dict[str, float] | None = None,
    rank_sweep: tuple[int, ...] = _DEFAULT_RANK_SWEEP,
    kill_divisor: int = 3,
    min_walltime_min: int = 1,
) -> None:
    """Resume sweep CSV: per-row walltime sized to force a mid-sim kill AND
    complete within ``restart-times`` from ONE ``analysis.run()``.

    For each row, ``hpc_time_min_per_sim = max(min_walltime_min, round(T_sa /
    kill_divisor))`` where ``T_sa`` is that sa's measured full-completion wallclock
    (minutes) from the CLEAN sweep, keyed by ``sa_id`` in ``runtime_min_by_sa``.
    When ``runtime_min_by_sa`` is None (off-cluster dry-run only), fall back to a
    conservative GPU=4 min / CPU=18 min estimate by row type — REPLACE with real
    clean-sweep numbers (via ``size_resume_walltimes``) before the production run.
    """
    runtimes = runtime_min_by_sa or {}
    rows = _rows(_configs(tuple(rank_sweep)), walltime_min=None)
    for r in rows:
        t_full = runtimes.get(r["sa_id"], 4.0 if r["n_gpus"] else 18.0)
        r["hpc_time_min_per_sim"] = max(min_walltime_min, round(t_full / kill_divisor))
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False)


def size_resume_walltimes(clean_analysis) -> dict[str, int]:
    """Two-pass (FQ3): read each clean-sweep sa_id's full-completion wallclock
    (minutes) from the completed clean analysis, for feeding
    ``write_resume_matrix_csv(runtime_min_by_sa=...)``.

    Source: ``df_status['perf_Total']`` (cumulative wallclock in seconds -> /60),
    max over each sa's rows. On the CLEAN run this equals SLURM ``Elapsed`` because
    clean is never resumed (perf_Total only exceeds Elapsed when resumes occurred).
    Run AFTER the clean sweep has completed.
    """
    df = clean_analysis.df_status
    return (
        df.dropna(subset=["perf_Total"])
        .assign(_min=lambda d: d["perf_Total"] / 60.0)
        .groupby("sa_id")["_min"]
        .max()
        .round()
        .astype(int)
        .to_dict()
    )


def generate_synthetic_experiment(cfg: synthetic_experiment_config, dest_dir: Path) -> Path:
    """Build the synthetic TRITON-SWMM case for ``cfg`` under ``dest_dir`` and
    return the case directory.

    Threads the four config model knobs (grid + forcing) into
    ``SyntheticModelParams`` and delegates to the lifted
    ``hhemt.synthetic_model.build_synthetic_case``. The remaining
    ``SyntheticModelParams`` fields (event shaping — sim duration, compound
    surge, reporting cadence) retain their generic defaults in this Phase-1 first
    cut; promoting the full compound-event shaping to the config model is a
    separate follow-up (the production experiment's shaping currently lives in
    ``scripts/experiments/synth_compute_config.py::_EXPERIMENT_PARAMS``).
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
