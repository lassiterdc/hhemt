"""UVA synthetic compute-config experiment factories (scripts-side; D5 option a).

Seed of the ADR-8 ``tests/fixtures`` -> ``src/`` synth-machinery lift; coordinate with the
``analysis-test-end-to-end`` predetermined plan before promoting to ``src/``. Imports the
synthetic-model builder from ``tests/`` — run from the repo root (where ``tests/`` is on
sys.path). Emits CSV (not XLSX) sensitivity definitions to sidestep Gotcha 15 (A3).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

# Initialize GDAL (rasterio/rioxarray) BEFORE the synthetic-model chain pulls in
# swmmio/swmm.toolkit. Importing swmm.toolkit's native lib before GDAL is initialized
# corrupts GDAL's allocator and aborts the process ("free(): invalid pointer") at the
# first rioxarray `.rio.to_raster` in the synthetic-model build. Loading rioxarray
# first makes the later swmmio import safe. Must precede the tests.fixtures imports.
import rioxarray  # noqa: F401  (import-order workaround — see comment above)

from tests.fixtures.synthetic_model.cache import SyntheticModelParams
from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case  # delegate target

from ._matrix_builder import write_clean_matrix_csv, write_resume_matrix_csv

_GENERATED = Path(__file__).parent / "_generated"  # gitignored (D3)

# 3.5 m native (identity, no resample); enlarged grid for non-degenerate 2-3 GPU decomp (D4)
_EXPERIMENT_PARAMS = dataclasses.replace(
    SyntheticModelParams(),
    cell_size_m=3.5,
    n_cols=64,
    n_rows=120,
    sim_duration_min=30,
)


@dataclasses.dataclass
class _Case:
    analysis: object  # TRITONSWMM_analysis
    system_directory: str  # resolved on-disk case root (for the analysis tool / Phase-3 discovery)


def _build_case(
    *, analysis_name: str, sensitivity_csv: Path, start_from_scratch: bool, resume: bool, system_directory: str | None
) -> _Case:
    """Materialize the synthetic UVA case and return an object exposing ``.analysis``.

    UVA HPC overrides (A2): system -> CUDA/a6000/3.5m; analysis -> batch_job/***REMOVED***/login/gres.

    ``system_directory``: when given, redirect the case root there (Decision 4 — on Rivanna pass
    ``/project/***REMOVED***/...`` so outputs avoid the small-quota $HOME/.cache and the analysis tool
    has a deterministic read root). When ``None`` (the default), the materializer's natural
    platformdirs cache root is used — required so the case can be materialized + dry-run validated
    off-cluster, where ``/project`` does not exist (the system constructor mkdir's this path).
    """
    system_cfg = {
        "gpu_compilation_backend": "CUDA",
        "gpu_hardware": "a6000",
        "target_dem_resolution": 3.5,
        # HPC module set the generated compile/run scripts must `module load` (system.py:663):
        # without it the field defaults to None, no `module load` is emitted, and `nvcc` is
        # absent on the standard-partition build node -> GPU compile aborts at `which nvcc`.
        # Mirrors the tested UVA platform default (constants.py UVA_DEFAULT_PLATFORM_CONFIG):
        # GCC 11.4 + CUDA 12.4 is a compatible nvcc/host-compiler pairing.
        "additional_modules_needed_to_run_TRITON_SWMM_on_hpc": "miniforge gompi/11.4.0_4.1.4 cuda/12.4.1",
    }
    if system_directory is not None:
        system_cfg["system_directory"] = system_directory

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=analysis_name,
        params=_EXPERIMENT_PARAMS,
        sensitivity_csv=sensitivity_csv,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        additional_system_configs=system_cfg,
        additional_analysis_configs={
            "multi_sim_run_method": "batch_job",
            "hpc_account": "***REMOVED***",
            "hpc_login_node": "login1.hpc.virginia.edu",
            # batch_job REQUIRED fields (analysis_config check_consistency, config/analysis.py
            # lines 620/625/628/633) — all default None and raise at load if omitted:
            "hpc_max_simultaneous_sims": 1000,  # set high so it does NOT cap concurrent sbatch
            # jobs (Decision 2 — ~28 sims/case, so 1000 is an effective no-cap)
            "hpc_total_job_duration_min": 60,  # SBATCH --time; Phase 3 tunes from observed runtimes
            "hpc_gpus_per_node": 8,  # UVA a6000/a100 nodes hold 8 GPUs
            # base-level per-sim walltime (the sensitivity CSV overrides it per sub-analysis;
            # 30 matches the clean-experiment walltime in write_clean_matrix_csv):
            "hpc_time_min_per_sim": 30,
            # base partitions REQUIRED for SLURM resource-block generation (workflow.py:1044):
            # the sim resource block reads hpc_ensemble_partition (CSV overrides it per-row);
            # setup/prepare/process/consolidate jobs read hpc_setup_and_analysis_processing_partition.
            "hpc_ensemble_partition": "standard",
            "hpc_setup_and_analysis_processing_partition": "standard",
            # NOTE: gres GPU allocation is the workflow.run() default (gpu_alloc_mode="gres");
            # it is NOT an analysis_config field, so it must not be set here (extra="forbid").
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(sensitivity_csv),
            # sensitivity report block REQUIRED (validate_sensitivity_independent_vars,
            # analysis.py:1761) — reuse the synth sensitivity report yaml content:
            "report": {
                "sensitivity": {
                    "mode": "benchmarking",
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "run_mode",
                }
            },
        },
    )
    return _Case(analysis=case.analysis, system_directory=str(case.system.cfg_system.system_directory))


def clean_case(start_from_scratch: bool = False, system_directory: str | None = None) -> _Case:
    """Clean determinism experiment: single 3.5m res, 28-config sweep, single-allocation walltime.

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/***REMOVED***/***REMOVED***/norfolk/synth_compute_config/synth_cc_clean"``.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "clean_matrix.csv"
    write_clean_matrix_csv(csv)
    return _build_case(
        analysis_name="synth_cc_clean",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=False,
        system_directory=system_directory,
    )


def resume_case(start_from_scratch: bool = False, system_directory: str | None = None) -> _Case:
    """Resume demo: short walltime forces a mid-sim kill; raised retry cap guarantees completion.

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/***REMOVED***/***REMOVED***/norfolk/synth_compute_config/synth_cc_resume"``.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "resume_matrix.csv"
    write_resume_matrix_csv(csv)
    # NOTE: resume completion is driven by REPEATED DRIVER RE-INVOCATION in Phase 3
    # (analysis.run(from_scratch=False, ...) re-plans the v2 wait-rules and re-dispatches the
    # walltime-killed simulation_sa_* rules from the latest config_NNNN.cfg checkpoint — Gotcha 30,
    # master A5), NOT a config knob. hpc_max_wait_for_inflight_min already defaults to its 10080
    # max (config/analysis.py:147) and is the v2 wait-rule poll backstop, NOT the Snakemake
    # restart-times cap; setting it here was a no-op against the wrong knob and is removed.
    return _build_case(
        analysis_name="synth_cc_resume",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=True,
        system_directory=system_directory,
    )
