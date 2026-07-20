"""UVA synthetic compute-config experiment factories (scripts-side; D5 option a).

The synthetic-model generators and the experiment-matrix builder are now lifted to
``src`` (``hhemt.synthetic_model`` + ``hhemt.synthetic_experiment``, PIP-2 Phase 1);
this scripts-side driver composes them into the UVA sensitivity cases and emits CSV
(not XLSX) sensitivity definitions to sidestep Gotcha 15 (A3). Run from the repo root
(where ``tests/`` is on sys.path for the test-case builder).
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

from hhemt.bundle._dependency import ExperimentDependency, ExperimentIdentity
from hhemt.synthetic_experiment import write_clean_matrix_csv, write_resume_matrix_csv
from tests.fixtures.synthetic_model.cache import SyntheticModelParams
from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case  # delegate target

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
    sim_duration_min=1440,  # 24 h event — TUNE: smallest sim_duration whose FASTEST
    # GPU config's clean SLURM Elapsed is ~2-4 min (> the 1-min
    # walltime floor so resume kills fire), measured in step C4.
    rainfall_peak_min=120,  # rain + surge peak at 2 h
    rainfall_duration_min=720,  # rain over 0..12 h (rise to 2 h, fall to 12 h), then dry
    rainfall_peak_mm_per_hr=100.0,
    stormsurge_peak_m=1.0,  # +1 m surge on the base tide, co-peaking with the rain
    reporting_timestep_s=600.0,  # 10-min dumps -> ~288 over 48 h (manageable output)
    compound_event=True,
)

# Fixed physical domain (m), preserved across resolutions so a model at any cell
# size is the SAME watershed (224 m × 420 m) — only the grid density changes.
_DOMAIN_WIDTH_M = _EXPERIMENT_PARAMS.n_cols * _EXPERIMENT_PARAMS.cell_size_m  # 224.0
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
    return dataclasses.replace(_EXPERIMENT_PARAMS, cell_size_m=cell_size_m, n_cols=n_cols, n_rows=n_rows)


@dataclasses.dataclass
class _Case:
    analysis: object  # TRITONSWMM_analysis
    system_directory: str  # resolved on-disk case root (for the analysis tool / Phase-3 discovery)


def _build_case(
    *,
    analysis_name: str,
    sensitivity_csv: Path,
    start_from_scratch: bool,
    resume: bool,
    system_directory: str | None,
    cell_size_m: float = 3.5,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
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
        # ONLY surviving system_config key. GPU hardware/backend, the gres allocation
        # flavor, and the module set moved to hpc_system_config_synth_uva.yaml
        # (partition-as-axis migration, Gotcha 54).
        "target_dem_resolution": cell_size_m,
    }
    if system_directory is not None:
        system_cfg["system_directory"] = system_directory
    # Config-injectable TRITON pin (no hardcoded config -- CLAUDE.md style #9; mirrors the
    # hpc_system_config_yaml estate->toolkit threading below). When set by the estate runner,
    # this OVERRIDES the test-fixture default TRITONSWMM_branch_key (test_case_builder.py:415,
    # "15eb18a5...") via the additional_system_configs merge at test_case_builder.py:450, so the
    # experiment runs under the pinned TRITON while the synth test tier keeps the fixture default.
    if tritonswmm_branch_key is not None:
        system_cfg["TRITONSWMM_branch_key"] = tritonswmm_branch_key

    # Config-injectable (no hardcoded config — CLAUDE.md style #9): callers (the private-estate
    # runner) pass the git-tracked estate config carrying the real account; None preserves the
    # in-toolkit placeholder path (its {your-allocation} is anonymization-safe for the public repo).
    hpc_cfg = (
        hpc_system_config_yaml
        if hpc_system_config_yaml is not None
        else Path(__file__).parent / "hpc_system_config_synth_uva.yaml"
    )

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=analysis_name,
        params=_params_for_resolution(cell_size_m),
        sensitivity_csv=sensitivity_csv,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=start_from_scratch,
        additional_system_configs=system_cfg,
        hpc_system_config_yaml=hpc_cfg,
        additional_analysis_configs={
            "multi_sim_run_method": "batch_job",
            # Opt-in per-scenario SWMM node/link timeseries consolidation. ON for this
            # experiment because the clean-vs-resume over-time MAX-ABSOLUTE-difference
            # figure reads tritonswmm/swmm_{node,link}_timeseries from the consolidated
            # master tree, and the per-config resume vlines read the durable replay_t
            # stamped alongside it. The toolkit-wide default stays False
            # (config/analysis.py) — this is the EXPERIMENT's value for an
            # experiment-policy knob, expressed here beside the other policy literals
            # rather than injected from the estate (the estate carries environment and
            # secrets — the real account — not experiment policy).
            "toggle_consolidate_timeseries": True,
            # batch_job REQUIRED fields (default None -> raise at load if omitted). The retired
            # hpc_account / hpc_login_node / hpc_max_simultaneous_sims / hpc_gpus_per_node keys
            # moved to hpc_system_config_synth_uva.yaml (default_account / login_node /
            # max_concurrent_jobs / partitions.*.gpus_per_node).
            "hpc_total_job_duration_min": 60,  # SBATCH --time; Phase 3 tunes from observed runtimes
            # base-level per-sim walltime (the sensitivity CSV overrides it per sub-analysis;
            # 30 matches the clean-experiment walltime in write_clean_matrix_csv):
            "hpc_time_min_per_sim": 30,
            # Snakemake retries: under the Option-D deterministic single kill the
            # resume arm's attempt-1 is SIGKILLed after N checkpoints and attempt-2
            # resumes-to-completion under the generous walltime, so a LOW cap (3)
            # both completes the sweep and fails a genuinely non-converging config
            # FAST. 2 for clean (never killed).
            "hpc_restart_times_simulate": 3 if resume else 2,
            "hpc_restart_times_other": 3 if resume else 2,
            # Option-D deterministic single-kill resume-test harness: on the RESUME
            # arm the runner SIGKILLs a fresh first-attempt sim after 3 hotstart
            # checkpoints (forcing exactly one mid-sim kill); the retry resumes and
            # completes under the generous walltime. None (clean arm) disables it.
            "deterministic_kill_after_n_checkpoints": 3 if resume else None,
            # base partition selectors (the CSV overrides ensemble per-row). The master ensemble
            # is a GPU partition so the master participates in the GPU-target dedup (Gotcha 54);
            # setup/prepare/process/consolidate run on standard.
            "hpc_ensemble_partition": "gpu-a6000",
            "hpc_setup_and_analysis_processing_partition": "standard",
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(sensitivity_csv),
            # sensitivity report block REQUIRED (validate_sensitivity_independent_vars).
            # reporting_set lives on report_config (top level), NOT on report.sensitivity
            # (ADR-5 ReportingSet: config/report.py::report_config.reporting_set; the
            # sensitivity submodel forbids it as extra and strips a legacy `mode` key).
            "report": {
                "reporting_set": "benchmarking",
                "sensitivity": {
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "run_mode",
                },
            },
        },
    )
    return _Case(analysis=case.analysis, system_directory=str(case.system.cfg_system.system_directory))


def resume_depends_on(tritonswmm_sha: str = "3a832f7d") -> ExperimentDependency:
    """The RESUME experiment's first-class dependency on the CLEAN experiment (P2+V3).

    This is the unmistakeable, version-controlled declaration the reproducible ``intercomparison``
    driver reads to verify + resolve the clean bundle. ``tritonswmm_sha`` is the pinned solver
    (default ``3a832f7d``); ``case_name`` binds once the bundle carries ``case.yaml`` (Phase-5
    ``_emit`` copy) — until then ``read_bundle_identity`` returns ``case_name=None`` and
    ``ExperimentIdentity.matches`` skips it, so the sha+role check still holds. ``compute_config_identity``
    is v1-None (both arms share the SAME 28-config matrix, so it is not a clean/resume discriminator)."""
    return ExperimentDependency(
        dependency_experiment_id="synth_cc_clean",
        role="clean",
        expected_identity=ExperimentIdentity(tritonswmm_sha=tritonswmm_sha),
    )


def clean_case(
    start_from_scratch: bool = False,
    system_directory: str | None = None,
    cell_size_m: float = 3.5,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
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
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
    )


def resume_case(
    start_from_scratch: bool = False,
    system_directory: str | None = None,
    cell_size_m: float = 3.5,
    runtime_min_by_sa: dict[str, float] | None = None,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
) -> _Case:
    """Resume demo (Option-D deterministic single kill): the runner SIGKILLs the
    fresh first attempt mid-sim after N hotstart checkpoints; the Snakemake retry
    resumes-to-completion under a GENEROUS walltime.

    ``cell_size_m`` MUST match the value passed to ``clean_case`` (same grid → the
    byte-identity clean-vs-resume comparison is valid). Under Option D the kill is
    DETERMINISTIC (checkpoint-count SIGKILL), NOT a walltime expiry, so no finer
    resolution / longer runtime is needed to make the kill fire — it works even
    for a ~1.6-min GPU sim (DoD #3 is satisfied by the deterministic kill).

    ``runtime_min_by_sa`` is IGNORED under Option D (the resume walltime is the
    generous clean walltime, not a T/3 short window); it is retained only for
    signature stability with ``build_resume_from_clean_runtimes`` and is a
    candidate for removal in a follow-up cleanup.

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/{your-allocation}/{username}/norfolk/synth_compute_config/synth_cc_resume"``.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "resume_matrix.csv"
    write_resume_matrix_csv(csv)
    # Option-D mechanism: resume completion lands within ONE analysis.run() via
    # Snakemake retries — attempt-1 is deterministically SIGKILLed after N
    # checkpoints (deterministic_kill_after_n_checkpoints on the resume analysis
    # config), and attempt-2 resumes from the latest config_NNNN.cfg checkpoint and
    # completes under the generous per-sim walltime. This supersedes the prior
    # short-walltime + repeated-driver-re-invocation scheme (both retired).
    return _build_case(
        analysis_name="synth_cc_resume",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=True,
        system_directory=system_directory,
        cell_size_m=cell_size_m,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
    )


def build_resume_from_clean_runtimes(
    *,
    clean_system_directory: str,
    system_directory: str | None = None,
    cell_size_m: float = 3.5,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
) -> _Case:
    """Two-pass (FQ3): read each completed clean-sweep sa_id's full-completion
    wallclock and size the resume walltimes to force a mid-sim kill (~T/3), then
    materialize the resume case. Run AFTER the clean sweep has completed.

    Delegates the per-sa runtime read to
    ``hhemt.synthetic_experiment.size_resume_walltimes`` (df_status['perf_Total'];
    on the clean run this equals SLURM Elapsed because clean is never resumed).
    ``cell_size_m`` MUST match the clean sweep's for a valid byte-identity compare.
    """
    from hhemt.synthetic_experiment import size_resume_walltimes

    clean = clean_case(
        system_directory=clean_system_directory,
        cell_size_m=cell_size_m,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
    )
    runtime_min_by_sa = size_resume_walltimes(clean.analysis)
    return resume_case(
        system_directory=system_directory,
        cell_size_m=cell_size_m,
        runtime_min_by_sa=runtime_min_by_sa,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
    )


def _emit_bundle(case: _Case) -> Path:
    """Committed emit step (supersedes the prior inline-heredoc runbook): eda + bundle a materialized
    case, returning the bundle path. Requires df_status all-complete (batch_job runs out-of-band)."""
    a = case.analysis
    a.eda()  # writes eda/{plot_id}.zarr + .verdict.json + plots/eda/*.html
    return Path(a.sensitivity.bundle_report_data())  # harvests plots/eda/*.manifest.json -> carries eda zarr+verdict


def _cli() -> None:
    """First-class committed CLI (FQ3, supersedes the operator inline-heredoc runbook).

    ``clean`` / ``resume --eda --bundle`` emit a per-arm bundle; ``intercomparison`` is the SINGLE
    reproducible entry that resolves+verifies the clean dependency (P2+V3), ensures the resume bundle,
    and combines them (lineage-stamped). ``hhemt combine`` remains the one explicit compare verb; this
    entry drives it from the resume ``depends_on`` so reproduction needs no hand-authored code."""
    import argparse

    p = argparse.ArgumentParser(
        prog="synth_compute_config",
        description="Synth compute-config experiment driver (emit / combine).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("clean", "resume"):
        sp = sub.add_parser(name)
        sp.add_argument("--system-directory", required=True)
        sp.add_argument("--hpc-system-config", type=Path, default=None)
        sp.add_argument("--cell-size-m", type=float, default=3.5)
        sp.add_argument("--tritonswmm-sha", default="3a832f7d")
        sp.add_argument("--eda", action="store_true")
        sp.add_argument("--bundle", action="store_true")
    ip = sub.add_parser("intercomparison")
    ip.add_argument("--clean-system-directory", required=True)
    ip.add_argument("--resume-system-directory", required=True)
    ip.add_argument(
        "--clean-bundle-search-root",
        type=Path,
        action="append",
        required=True,
        help="Dir(s) to search for the clean bundle (repeatable).",
    )
    ip.add_argument("--hpc-system-config", type=Path, default=None)
    ip.add_argument("--cell-size-m", type=float, default=3.5)
    ip.add_argument("--tritonswmm-sha", default="3a832f7d")
    ip.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if args.cmd in ("clean", "resume"):
        factory = clean_case if args.cmd == "clean" else resume_case
        case = factory(
            system_directory=args.system_directory,
            cell_size_m=args.cell_size_m,
            hpc_system_config_yaml=args.hpc_system_config,
            tritonswmm_branch_key=args.tritonswmm_sha,
        )
        if args.eda or args.bundle:
            print("BUNDLE:", _emit_bundle(case))  # eda+bundle folded in (first-class emit)
        else:
            print("MATERIALIZED:", case.system_directory, "-> run analysis.run() then re-invoke with --eda --bundle")
        return

    # intercomparison: the SINGLE reproducible entry (resolve+verify+combine+lineage-stamp).
    from hhemt.bundle._combine import combine_bundle
    from hhemt.bundle._dependency import resolve_dependency

    # AR2 (FQ2): batch_job submission is async (a submission RECEIPT, not completion), so auto-running
    # the 28-config GPU clean sweep here could not be awaited-then-combined in one process anyway; and a
    # large shared-cluster allocation should be operator-authorized. So HALT with the exact committed
    # command (no improvised code -- the command IS the factory entry). auto_satisfy is the opt-in seam a
    # future synchronous local-mode driver could wire.
    # Include --hpc-system-config in the emitted reproduction command ONLY when supplied, so the
    # halt message is always a copy-paste-valid command (no literal "None" path).
    hpc_flag = f"--hpc-system-config {args.hpc_system_config} " if args.hpc_system_config is not None else ""
    clean_root = resolve_dependency(
        resume_depends_on(tritonswmm_sha=args.tritonswmm_sha),
        search_roots=list(args.clean_bundle_search_root),
        auto_satisfy=None,
        emitted_command=(
            f"python -m scripts.experiments.synth_compute_config clean "
            f"--system-directory {args.clean_system_directory} "
            f"{hpc_flag}"
            f"--tritonswmm-sha {args.tritonswmm_sha} --eda --bundle"
        ),
    )
    resume_case_obj = resume_case(
        system_directory=args.resume_system_directory,
        cell_size_m=args.cell_size_m,
        hpc_system_config_yaml=args.hpc_system_config,
        tritonswmm_branch_key=args.tritonswmm_sha,
    )
    resume_root = _emit_bundle(resume_case_obj)  # ensure the resume bundle (df_status must be complete)
    combined = combine_bundle([clean_root, Path(resume_root)], output_path=args.output)
    print("COMBINED:", combined.root)  # lineage stamp lives in combined_of (FILE 3)


if __name__ == "__main__":
    _cli()
