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

# 3.5 m native (identity, no resample); enlarged grid for non-degenerate 2-3 GPU decomp (D4).
# Compound coastal-pluvial event + longer runtime (2026-06-15): a triangular rain burst +
# a base tide sinusoid with a co-peaking triangular surge, then a long drainage tail. Shared
# by clean AND resume (via _params_for_resolution) so the two cases are forcing-identical;
# only their per-sim WALLTIME differs (clean = full alloc; resume = 1-min floor -> kills ->
# hotstart-resume). sim_duration is sized so the per-sim wallclock crosses the 1-min floor so
# resume's kills actually fire — TUNE empirically (measure the simulate-rule Elapsed -> 60-120 s).
_EXPERIMENT_PARAMS = dataclasses.replace(
    SyntheticModelParams(),
    cell_size_m=3.5,
    n_cols=64,
    n_rows=120,
    sim_duration_min=1440,        # 24 h event — TUNE: smallest sim_duration whose FASTEST
                                  # GPU config's clean SLURM Elapsed is ~2-4 min (> the 1-min
                                  # walltime floor so resume kills fire), measured in step C4.
    rainfall_peak_min=120,        # rain + surge peak at 2 h
    rainfall_duration_min=720,    # rain over 0..12 h (rise to 2 h, fall to 12 h), then dry
    rainfall_peak_mm_per_hr=100.0,
    stormsurge_peak_m=1.0,        # +1 m surge on the base tide, co-peaking with the rain
    reporting_timestep_s=600.0,   # 10-min dumps -> ~288 over 48 h (manageable output)
    compound_event=True,
)

# Fixed physical domain (m), preserved across resolutions so a model at any cell
# size is the SAME watershed (224 m × 420 m) — only the grid density changes.
_DOMAIN_WIDTH_M = _EXPERIMENT_PARAMS.n_cols * _EXPERIMENT_PARAMS.cell_size_m   # 224.0
_DOMAIN_HEIGHT_M = _EXPERIMENT_PARAMS.n_rows * _EXPERIMENT_PARAMS.cell_size_m  # 420.0


def _params_for_resolution(cell_size_m: float) -> SyntheticModelParams:
    """Return `_EXPERIMENT_PARAMS` re-gridded to `cell_size_m`, preserving the
    physical domain (n_cols/n_rows scale inversely with cell size).

    The generator (`geometry.py` / `swmm_template.py`) is fully grid-driven, so
    any resolution builds and passes the `rim==DEM` and deadlock-safety tripwires.
    A FINER resolution → more cells → longer per-sim wallclock (the lever for
    making the resume sweep's sims exceed the 1-min SLURM walltime floor so a
    kill is actually forced). NOTE: the byte-identity clean-vs-resume comparison
    requires BOTH cases at the SAME resolution — pass the same `cell_size_m` to
    `clean_case` and `resume_case`. Coarser than ~n_rows/`_N_COUPLING_NODES`
    will trip the interior-too-small assertion in `_node_matrix_rows`.
    """
    n_cols = max(round(_DOMAIN_WIDTH_M / cell_size_m), 1)
    n_rows = max(round(_DOMAIN_HEIGHT_M / cell_size_m), 1)
    return dataclasses.replace(
        _EXPERIMENT_PARAMS, cell_size_m=cell_size_m, n_cols=n_cols, n_rows=n_rows
    )


@dataclasses.dataclass
class _Case:
    analysis: object  # TRITONSWMM_analysis
    system_directory: str  # resolved on-disk case root (for the analysis tool / Phase-3 discovery)


def _build_case(
    *, analysis_name: str, sensitivity_csv: Path, start_from_scratch: bool, resume: bool,
    system_directory: str | None, cell_size_m: float = 3.5,
) -> _Case:
    """Materialize the synthetic UVA case and return an object exposing ``.analysis``.

    UVA HPC overrides (A2): system -> CUDA/a6000/3.5m; analysis -> batch_job/{your-allocation}/login/gres.

    ``system_directory``: when given, redirect the case root there (Decision 4 — on Rivanna pass
    ``/project/{your-allocation}/...`` so outputs avoid the small-quota $HOME/.cache and the analysis tool
    has a deterministic read root). When ``None`` (the default), the materializer's natural
    platformdirs cache root is used — required so the case can be materialized + dry-run validated
    off-cluster, where ``/project`` does not exist (the system constructor mkdir's this path).
    """
    system_cfg = {
        "gpu_compilation_backend": "CUDA",
        "gpu_hardware": "a6000",
        # Match the native synth grid resolution (no resample). Tracks cell_size_m
        # so a re-gridded model keeps identity DEM handling.
        "target_dem_resolution": cell_size_m,
        # HPC module set the generated compile/run scripts must `module load` (system.py:663):
        # without it the field is None, no `module load` is emitted, and the build node lacks
        # both `nvcc` and a new-enough libstdc++ -> GPU compile fails (first nvcc-not-found,
        # then a GLIBCXX ABI link error in CMake's TryCompile).
        # gompi/14.2.0_5.0.7 = GCC 14.2, whose libstdc++ provides GLIBCXX_3.4.33 — clears the
        # conda env's GLIBCXX_3.4.30+ floor that CMake's TryCompile needs (it runs before the
        # project's conda-libstdc++ link flag applies). The previously-working set
        # gcc/12.4.0+openmpi/4.1.4+cuda/12.2.2 is no longer on Rivanna; the only other bundle,
        # gompi/11.4.0_4.1.4 (GCC 11.4 / GLIBCXX_3.4.29), is too old. cuda/12.8.0 supports
        # GCC <=14 (cuda/12.4.1 caps at GCC 13). GCC 14.2 already builds the CPU backend OK.
        "additional_modules_needed_to_run_TRITON_SWMM_on_hpc": "miniforge gompi/14.2.0_5.0.7 cuda/12.8.0",
        # GPU SLURM allocation mode. Unset -> defaults to "gpus" (Frontier --gpus-per-task=1,
        # run_simulation.py:651), which on UVA's gres allocation fails at sim launch:
        # "srun: fatal: --gpus-per-task is mutually exclusive with ... SLURM_NTASKS_PER_GPU".
        # UVA requires "gres" (UVA_DEFAULT_PLATFORM_CONFIG). This is the last cfg_system field
        # the hand-built dict was missing vs PlatformConfig.to_system_dict().
        "preferred_slurm_option_for_allocating_gpus": "gres",
    }
    if system_directory is not None:
        system_cfg["system_directory"] = system_directory

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=analysis_name,
        params=_params_for_resolution(cell_size_m),
        sensitivity_csv=sensitivity_csv,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        additional_system_configs=system_cfg,
        additional_analysis_configs={
            "multi_sim_run_method": "batch_job",
            "hpc_account": "{your-allocation}",
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
            # Snakemake retries: high for resume so a walltime-killed sim auto-resumes
            # to completion within ONE analysis.run() (no manual re-run loop); 2 for clean
            # (clean has a single-allocation walltime and is never killed). The simulate
            # knob drives the per-rule retries: on the sim rules; _other seeds the global
            # baseline for the idempotent non-sim rules (old single value seeds both,
            # preserving the prior global-restart-times semantics).
            "hpc_restart_times_simulate": 20 if resume else 2,
            "hpc_restart_times_other": 20 if resume else 2,
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


def clean_case(
    start_from_scratch: bool = False, system_directory: str | None = None, cell_size_m: float = 3.5
) -> _Case:
    """Clean determinism experiment: 28-config sweep, single-allocation walltime.

    ``cell_size_m`` sets the synth DEM resolution (default 3.5 m, physical domain
    preserved). Use a FINER value (e.g. 1.75) to lengthen per-sim wallclock — but
    pass the SAME ``cell_size_m`` to ``resume_case`` or the byte-identity
    clean-vs-resume comparison breaks (different grids → trivially "diverged").

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/{your-allocation}/{username}/norfolk/synth_compute_config/synth_cc_clean"``.
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
        cell_size_m=cell_size_m,
    )


def resume_case(
    start_from_scratch: bool = False, system_directory: str | None = None, cell_size_m: float = 3.5,
    runtime_min_by_sa: dict[str, float] | None = None,
) -> _Case:
    """Resume demo: short walltime forces a mid-sim kill; raised retry cap guarantees completion.

    ``cell_size_m`` MUST match the value passed to ``clean_case`` (same grid → the
    byte-identity clean-vs-resume comparison is valid). A finer resolution makes
    sims run long enough that the 1-min SLURM walltime actually kills them, so the
    hotstart-resume path is genuinely exercised (DoD #3).

    ``runtime_min_by_sa``: per-``sa_id`` full-completion wallclock (minutes) measured
    from the CLEAN sweep; sizes each backend's resume walltime to ~T/3 so the kill
    fires and completion lands within ``hpc_restart_times_simulate`` from a single ``.run()``.

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/{your-allocation}/{username}/norfolk/synth_compute_config/synth_cc_resume"``.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "resume_matrix.csv"
    write_resume_matrix_csv(csv, runtime_min_by_sa=runtime_min_by_sa)
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
        cell_size_m=cell_size_m,
    )
