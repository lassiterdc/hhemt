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
from hhemt.synthetic_experiment import (
    model_arm_toggles,
    write_clean_matrix_csv,
    write_resume_matrix_csv,
    write_smoke_matrix_csv,
)
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

# Multi-resume interruption schedule: 25/50/75% of the synth grid's reporting-timestep
# checkpoint count. sim_duration_min=1440 (24 h) / reporting_timestep_s=600 (10 min) = 144
# config_NNNN.cfg checkpoints; each entry is an ABSOLUTE checkpoint index (1:1 with the
# reporting-step index the watcher counts). Strictly increasing; all < 144 so all three
# kills fire and attempt 4 resumes from ~108 to completion -> n_resumes == 3.
_RESUME_INTERRUPTION_SCHEDULE: tuple[int, ...] = (36, 72, 108)

# Phase-2 Empirical-Testing cheap confirmation (one-config, no clean prereq). A
# 2-entry schedule that REUSES two production checkpoint indices, so the kill
# windows stay production-scale (>=36 reporting-step checkpoints between events) and
# the 2 s deterministic-kill watcher (run_simulation.py:390, poll_interval_s=2.0)
# reliably catches both kills regardless of the actual serial wallclock. 144
# checkpoints already exist (sim_duration_min=1440 / reporting_timestep_s=600), so
# both entries < 144: both kills fire -> n_resumes == 2 and attempt 3 completes.
# Do NOT shrink sim_duration for the smoke: on the 64x120 synth grid a short sim
# blows through checkpoints faster than the watcher polls, risking a missed kill
# (a false negative). Cheapness comes from serial + one config + no fan-out.
_SMOKE_SCHEDULE: tuple[int, ...] = (36, 72)


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
    tritonswmm_git_url: str | None = None,
    tritonswmm_software_directory: str | None = None,
    model_arm: str = "tritonswmm",
    resume_interruption_schedule: tuple[int, ...] = _RESUME_INTERRUPTION_SCHEDULE,
    ensemble_partition: str = "gpu-a6000",
    swmm_snapshot_disable: bool = False,
    swmm_snapshot_keep_all: bool = False,
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
    # Config-injectable sibling fields, threaded exactly as the pin above and for the same
    # reason (CLAUDE.md style #9). BOTH are required to express a non-default model source:
    # the URL because the fork carrying a fix is a DIFFERENT REMOTE from ORNL upstream, and
    # the software directory because two remotes both named `triton.git` would otherwise share
    # one clone path -- and `_verify_tritonswmm_pin` compares the COMMIT, never the REMOTE, so
    # nothing else would catch the collision. Keying the directory on the pin is what makes
    # `_software/<dir>` stop silently meaning "whatever version ran last".
    if tritonswmm_git_url is not None:
        system_cfg["TRITONSWMM_git_URL"] = tritonswmm_git_url
    if tritonswmm_software_directory is not None:
        system_cfg["TRITONSWMM_software_directory"] = tritonswmm_software_directory

    # Config-injectable (no hardcoded config — CLAUDE.md style #9): callers (the private-estate
    # runner) pass the git-tracked estate config carrying the real account; None preserves the
    # in-toolkit placeholder path (its {your-allocation} is anonymization-safe for the public repo).
    hpc_cfg = (
        hpc_system_config_yaml
        if hpc_system_config_yaml is not None
        else Path(__file__).parent / "hpc_system_config_synth_uva.yaml"
    )

    # A3's arm selector, and it is injected ONLY when True rather than always.
    # TWO reasons, and the second is the one that is easy to miss.
    # (1) An always-present key would write `swmm_snapshot_disable: false` into the PERSISTED
    #     analysis_config.yaml of every existing arm, so those input configs would stop being
    #     byte-identical to what they are today for no behavioural gain.
    # (2) The persisted config is the ONLY route that reaches the emission. The cfg is written by
    #     prepare_scenario_runner.py, a SUBPROCESS that re-loads `--analysis-config` FROM DISK, so
    #     setting the attribute on an already-constructed analysis object is a silent no-op. It has
    #     to travel as config text, which is what `additional_analysis_configs` is for.
    snapshot_disable_cfg = {"swmm_snapshot_disable": True} if swmm_snapshot_disable else {}

    # THE SNAPSHOT-RETENTION selector, and it is injected ONLY when True for BOTH of the reasons
    # the sibling above gives, which apply here unchanged -- plus a third that is specific to this
    # flag and is the one that would be lost by "simplifying" to an always-present key.
    # (3) THIS CAMPAIGN'S ARM-MEMBERSHIP EVIDENCE IS THE LITERAL IN A MEMBER'S OWN
    #     `config_{k}.cfg` (`src/hhemt/scenario.py:410` records that contract in the toolkit).
    #     An always-present key would write `swmm_snapshot_keep_all: false` into the persisted
    #     analysis_config of EVERY arm in the corpus -- including arms that never elected it --
    #     which that arm never asked for. THAT HARM IS REAL AND IT DOES NOT SHIP -- it is confined
    #     to the IN-TREE INPUT yaml. It is also not (3)'s to justify: reason (1) states this exact
    #     harm for the sibling flag, and this block's opening sentence already adopts BOTH sibling
    #     reasons as applying here unchanged, so the guard is justified before (3) says anything.
    #     (3) EXISTS ONLY TO ADD WHAT IS SPECIFIC TO THIS FLAG -- that THIS arm's membership
    #     evidence is the cfg literal, which is what makes the input yaml's byte-stability worth
    #     protecting here and not merely tidy -- AND IT NOMINATES NO DECISIVE CLAIM OF ITS OWN.
    #     THREE successive drafts each nominated a NEW harm as "sufficient on its own" and all
    #     three were FALSE. The slot is REMOVED rather than refilled, because refilling it is the
    #     mechanism: every one of the three reached a false consequence by one unverified hop from
    #     a TRUE citation, so quoting the source is not the check -- measuring the consequence is.
    #     THREE STRONGER CLAIMS ARE FALSE, and all three are recorded as false because each was
    #     asserted here in an earlier draft of this comment before anyone measured it.
    #     (i) NO member `config_{k}.cfg` would change. `append_swmm_snapshot_keep_all`
    #     (`src/hhemt/scenario.py`) returns its input object UNCHANGED when the field is False and
    #     emits only the literal `=1` otherwise -- asserted on OBJECT IDENTITY by
    #     `tests/test_snapshot_keep_all_cfg_emission.py`, and it has exactly one call site. The
    #     solver cannot add it either: TRITON's checkpoint writer (`src/output.h::output_cfg`)
    #     starts from the original cfg TEXT and in-place-replaces only `sim_start_time=`,
    #     `checkpoint_id=`, `time_step=` and `it_count=`, so it never materializes an absent key.
    #     The membership-by-literal reading is therefore NOT what an always-present key destroys.
    #     (ii) NO compatibility divergence row would fire. `reprex_taxonomy` does class this field
    #     `experiment`, but `bundle/_compatibility.py` selects what it compares from the CLOSED
    #     two-member tuple `_CFG_ANALYSIS_COMPARISON_FIELDS` (`weather_events_to_simulate`,
    #     `sensitivity_analysis`) and never consults the taxonomy to ADD a field -- the taxonomy is
    #     read only to CLASSIFY a field the union already produced (`_field_bucket` is called at
    #     `:338`, INSIDE the `if va != vb` body, so it can change a row's SEVERITY and can never
    #     create one). Grounded STRUCTURALLY rather than on a zero-count grep, which is blind to a
    #     differently-spelled consumer: every row is minted from `(set(ca) | set(cb))` where both
    #     are `_read_jsonld_core` outputs, that function builds its dict through per-tuple
    #     `if ... in` gates plus exactly three literal keys (`case_name`, `schemaVersion`,
    #     `analysis_id`), with no `.keys()`, `.items()` or dict-union anywhere in the module, and it
    #     has exactly two call sites. That exposure is LATENT, not live: it becomes real only if
    #     someone extends that tuple, which the comment heading the tuple block expressly invites.
    #     (iii) NO SHIPPED artifact would change -- and the mechanism an earlier draft cited as the
    #     reason one WOULD is the mechanism that refutes it. BOTH shipped copies of
    #     `cfg_analysis.yaml` are REGENERATED from the model, never copied from the input yaml:
    #     `src/hhemt/publishing.py:188` (the ADR-11 deposit set) and
    #     `src/hhemt/bundle/_emit.py:590` (the render bundle) each write
    #     `cfg.model_dump(mode="json")` with no `exclude_*` flag, and pydantic emits every declared
    #     field at its default. So both ALREADY carry `swmm_snapshot_keep_all: false` today for
    #     every non-electing arm. Measured 2026-10-04 against a persisted config carrying NO
    #     `swmm_snapshot` key: the key is nevertheless present in the dump at `false`, and adding
    #     it to the input yaml leaves the dumped bytes IDENTICAL.
    # KEPT AS A SECOND DICT rather than merged with the sibling: the two flags are independent
    # (`disable` forces the replay fallback; `keep_all` disarms the retention prune that CAUSES
    # the fallback), so a future change to one must not have to read the other.
    snapshot_keep_all_cfg = {"swmm_snapshot_keep_all": True} if swmm_snapshot_keep_all else {}

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=analysis_name,
        params=_params_for_resolution(cell_size_m),
        sensitivity_csv=sensitivity_csv,
        **model_arm_toggles(model_arm),
        start_from_scratch=start_from_scratch,
        # SHARED-FILESYSTEM STAGING. The builder materializes three files under its own
        # `self.system_directory` -- `weather_events_to_simulate.csv` (test_case_builder.py:474)
        # plus `system_config.yaml` and `analysis_config.yaml` (:567-570) -- and threads the
        # analysis yaml into every Snakemake rule shell as `--analysis-config`. Under
        # `start_from_scratch=True` the builder redirects that directory to a
        # `tempfile.mkdtemp()` root (test_case_builder.py:385-387), which is NODE-LOCAL; under
        # `multi_sim_run_method='batch_job'` the `setup_target_*` rules are dispatched as
        # separate SLURM jobs on OTHER nodes, where that path does not exist. Measured
        # 2026-08-24: jobs 18869483 and 18869586 both FAILED 1:0 on
        # "[ERROR] Analysis config not found: /tmp/hhemt-scratch-99nh9f9i/synthetic_test_runs/
        # synth_cc_clean_tritonswmm/analysis_config.yaml". Supplying `runs_root_override`
        # suppresses that redirect -- its guard requires the override to be None -- and stages
        # the three files under the caller's own base on the shared filesystem. It does NOT
        # touch `system_cfg['system_directory']`, which `_write_configs` applies from
        # `additional_system_configs` LAST (test_case_builder.py:524), so the analysis dir and
        # the `run(from_scratch=True)` wipe target are exactly what they were. When the caller
        # names no system_directory the value is None and the tmpdir redirect stands: that
        # caller is off-cluster or local-only, and its analysis tree is node-local too.
        runs_root_override=(Path(system_directory).parent if system_directory is not None else None),
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
            # base-level per-sim walltime (the sensitivity CSV overrides it per member;
            # 30 matches the clean-experiment walltime in write_clean_matrix_csv):
            "hpc_time_min_per_sim": 30,
            # Snakemake retries: a K-entry resume_interruption_schedule needs K+1
            # attempts, so the cap is len(schedule) + 2 (one spare for a genuine
            # transient). 2 for clean (never interrupted).
            "hpc_restart_times_simulate": (len(resume_interruption_schedule) + 2) if resume else 2,
            "hpc_restart_times_other": (len(resume_interruption_schedule) + 2) if resume else 2,
            # Multi-resume interruption schedule: absolute hotstart-checkpoint indices
            # at which the runner SIGKILLs the sim (one kill per attempt); each retry
            # hotstart-resumes from the latest checkpoint. None (clean arm) disables it.
            "resume_interruption_schedule": resume_interruption_schedule if resume else None,
            # base partition selectors (the CSV overrides ensemble per-row). The master ensemble
            # is a GPU partition so the master participates in the GPU-target dedup (Gotcha 54);
            # setup/prepare/process/consolidate run on standard.
            "hpc_ensemble_partition": ensemble_partition,
            "hpc_setup_and_analysis_processing_partition": "standard",
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(sensitivity_csv),
            # eda.enabled_plots must include the b4b figures so analysis.eda() computes AND
            # renders them (config/eda.py default omits b4b_clean_identity/b4b_clean_vs_resume).
            # Without this the reporting_set='b4b' report ships honest-degradation placeholders
            # instead of the real b4b_clean_identity grid; the render is robust to a mis-set
            # enabled_plots (the b4b always-emit kinds), but only this config makes the grids real.
            "eda": {
                "enabled_plots": [
                    "config_diff_maps",
                    "b4b_clean_identity",
                    "b4b_clean_vs_resume",
                    "eda_rank_sensitivity",
                    "eda_cross_hardware_magnitude",
                ],
            },
            # sensitivity report block REQUIRED (validate_sensitivity_independent_vars).
            # reporting_set lives on report_config (top level), NOT on report.sensitivity
            # (ADR-5 ReportingSet: config/report.py::report_config.reporting_set; the
            # sensitivity submodel forbids it as extra and strips a legacy `mode` key).
            "report": {
                "reporting_set": "b4b",
                "sensitivity": {
                    "independent_vars": ["n_devices"],
                    "dependent_var": "performance.Total",
                    "aggregation": "mean",
                    "group_by_var": "run_mode",
                },
            },
            **snapshot_disable_cfg,
            **snapshot_keep_all_cfg,
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
    tritonswmm_git_url: str | None = None,
    tritonswmm_software_directory: str | None = None,
    model_arm: str = "tritonswmm",
    variant: str = "",
) -> _Case:
    """Clean determinism experiment: 28-config sweep, single-allocation walltime.

    ``cell_size_m`` sets the synth DEM resolution (default 3.5 m, physical domain
    preserved). Use a FINER value (e.g. 1.75) to lengthen per-sim wallclock — but
    pass the SAME ``cell_size_m`` to ``resume_case`` or the byte-identity
    clean-vs-resume comparison breaks (different grids → trivially "diverged").

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/{your-allocation}/{username}/norfolk/synth_compute_config/synth_cc_clean"``.

    ``variant`` is an INFIX on the analysis name: ``synth_cc_clean{variant}_{model_arm}``. Default
    ``""`` reproduces ``synth_cc_clean_{model_arm}`` byte-for-byte, so every existing caller is
    unaffected. ``variant="P"`` yields ``synth_cc_cleanP_tritonswmm`` — the SAME-PIN clean reference
    arm (``A1``) of the four-arm b4b re-run.

    WHY AN INFIX RATHER THAN AN ANALYSIS-NAME OVERRIDE, and it is the arm dimension that decides it.
    A full ``analysis_name`` override lets a caller supply a name whose arm disagrees with
    ``model_arm``, and ``model_arm`` is what drives ``model_arm_toggles`` — so the name would say
    ``tritonswmm`` while the enabled model was ``triton``, with nothing raising. That is the same
    arm-disappears-silently class the estate runner's own ``_run_reprocess`` docstring records paying
    for. Composing the name from ``model_arm`` makes the disagreement unwritable instead of merely
    discouraged.

    THE NAME IS LOAD-BEARING, not cosmetic. ``A1`` MUST NOT land at ``synth_cc_clean_{model_arm}``:
    that path is already occupied by a DIFFERENT-PIN clean tree (declared
    ``TRITONSWMM_branch_key: a38338b0...``, measured on its own ``cfg_system.yaml``) which carries 30
    members, 30 ``hydraulics.out`` and member ids byte-identical to the resume arm's. It therefore
    passes every cardinality guard the b4b criterion has and is discriminated from ``A1`` ONLY by the
    solver pin. A distinct root is what makes that tree unreachable as a comparand rather than
    merely refused after the fact.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "clean_matrix.csv"
    write_clean_matrix_csv(csv)
    return _build_case(
        analysis_name=f"synth_cc_clean{variant}_{model_arm}",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=False,
        system_directory=system_directory,
        cell_size_m=cell_size_m,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
        tritonswmm_git_url=tritonswmm_git_url,
        tritonswmm_software_directory=tritonswmm_software_directory,
        model_arm=model_arm,
    )


def resume_case(
    start_from_scratch: bool = False,
    system_directory: str | None = None,
    cell_size_m: float = 3.5,
    runtime_min_by_member: dict[str, float] | None = None,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
    tritonswmm_git_url: str | None = None,
    tritonswmm_software_directory: str | None = None,
    model_arm: str = "tritonswmm",
    variant: str = "",
    swmm_snapshot_disable: bool = False,
    swmm_snapshot_keep_all: bool = False,
) -> _Case:
    """Resume demo (Option-D deterministic single kill): the runner SIGKILLs the
    fresh first attempt mid-sim after N hotstart checkpoints; the Snakemake retry
    resumes-to-completion under a GENEROUS walltime.

    ``cell_size_m`` MUST match the value passed to ``clean_case`` (same grid → the
    byte-identity clean-vs-resume comparison is valid). Under Option D the kill is
    DETERMINISTIC (checkpoint-count SIGKILL), NOT a walltime expiry, so no finer
    resolution / longer runtime is needed to make the kill fire — it works even
    for a ~1.6-min GPU sim (DoD #3 is satisfied by the deterministic kill).

    ``runtime_min_by_member`` is IGNORED under Option D (the resume walltime is the
    generous clean walltime, not a T/3 short window); it is retained only for
    signature stability with ``build_resume_from_clean_runtimes`` and is a
    candidate for removal in a follow-up cleanup.

    Pass ``system_directory`` on Rivanna to root the case under project space (Decision 4), e.g.
    ``"/project/{your-allocation}/{username}/norfolk/synth_compute_config/synth_cc_resume"``.

    ``variant`` is an INFIX on the analysis name: ``synth_cc_resume{variant}_{model_arm}``, the exact
    mirror of ``clean_case``'s. Default ``""`` reproduces ``synth_cc_resume_{model_arm}``
    byte-for-byte, so every existing caller is unaffected. ``variant="_ornl"`` yields
    ``synth_cc_resume_ornl_tritonswmm`` — the ORNL-UPSTREAM-pinned resume arm of the solver-version
    control, whose comparand is the ORNL-pinned clean arm rather than either fork-pinned arm.

    WHY THIS PARAMETER EXISTS AT ALL, since the CLI deliberately does not expose it. The analysis
    name is composed HERE and nowhere else, so it is the only place a caller can vary it; an estate
    driver that needs two resume arms at two different solver pins can distinguish their OUTER
    folders via ``system_directory`` but could not distinguish their INNER analysis names without
    this. The path is DOUBLED (``{system_directory}/{analysis_name}``), and the inner layer is the
    one the b4b design calls load-bearing: its measured lesson is that a wrong-pin arm at a
    plausible path passed every structural guard and was discriminated from the reference arm by
    its PIN alone. Two resume arms sharing the inner name ``synth_cc_resume_{model_arm}`` reproduce
    exactly that ambiguity one level down.

    THE SAME ARM-AGREEMENT ARGUMENT APPLIES AS ON ``clean_case`` and is the reason this is an infix
    rather than a full ``analysis_name`` override: a free-form name could claim ``tritonswmm`` while
    ``model_arm`` enabled ``triton``, with nothing raising. Composing from ``model_arm`` makes that
    disagreement unwritable.
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "resume_matrix.csv"
    write_resume_matrix_csv(csv)
    # Multi-resume mechanism: resume completion lands within ONE analysis.run() via
    # Snakemake retries — each attempt is deterministically SIGKILLed at the next
    # resume_interruption_schedule checkpoint index (on the resume analysis config),
    # and the following attempt hotstart-resumes from the latest config_NNNN.cfg
    # checkpoint; the final attempt completes under the generous per-sim walltime.
    # This supersedes the prior short-walltime + repeated-driver-re-invocation
    # scheme (both retired).
    return _build_case(
        analysis_name=f"synth_cc_resume{variant}_{model_arm}",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=True,
        system_directory=system_directory,
        cell_size_m=cell_size_m,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
        tritonswmm_git_url=tritonswmm_git_url,
        tritonswmm_software_directory=tritonswmm_software_directory,
        model_arm=model_arm,
        swmm_snapshot_disable=swmm_snapshot_disable,
        swmm_snapshot_keep_all=swmm_snapshot_keep_all,
    )


def smoke_case(
    start_from_scratch: bool = True,
    system_directory: str | None = None,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
    tritonswmm_git_url: str | None = None,
    tritonswmm_software_directory: str | None = None,
) -> _Case:
    """Phase-2 Empirical-Testing cheap confirmation of the multi-resume interruption
    harness. ONE serial-CPU member on ``standard`` (pure-TRITON / uncoupled
    arm), PRODUCTION sim sizing (1440 min / 600 s -> 144 checkpoints), a 2-entry
    ``_SMOKE_SCHEDULE``. Runs STANDALONE (no clean-sweep prerequisite — Option D
    ignores the clean-derived walltime). ``ensemble_partition="standard"`` keeps the
    master off the GPU-target fast path (sensitivity_analysis.py:268-303), so the
    compile is CPU-only and the serial sim runs a CPU binary. The caller MUST pass
    ``override_pickup_where_leftoff=True`` to ``analysis.run()`` or every retry
    restarts from t=0 and the sim never completes (run_simulation.py:618).
    """
    _GENERATED.mkdir(parents=True, exist_ok=True)
    csv = _GENERATED / "smoke_matrix.csv"
    write_smoke_matrix_csv(csv)
    return _build_case(
        analysis_name="synth_cc_smoke_triton",
        sensitivity_csv=csv,
        start_from_scratch=start_from_scratch,
        resume=True,
        system_directory=system_directory,
        cell_size_m=3.5,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
        tritonswmm_git_url=tritonswmm_git_url,
        tritonswmm_software_directory=tritonswmm_software_directory,
        model_arm="triton",
        resume_interruption_schedule=_SMOKE_SCHEDULE,
        ensemble_partition="standard",
    )


def build_resume_from_clean_runtimes(
    *,
    clean_system_directory: str,
    system_directory: str | None = None,
    cell_size_m: float = 3.5,
    hpc_system_config_yaml: Path | None = None,
    tritonswmm_branch_key: str | None = None,
    tritonswmm_git_url: str | None = None,
    tritonswmm_software_directory: str | None = None,
    model_arm: str = "tritonswmm",
    variant: str = "",
    clean_variant: str | None = None,
    swmm_snapshot_disable: bool = False,
    swmm_snapshot_keep_all: bool = False,
) -> _Case:
    """Two-pass (FQ3): read each completed clean-sweep member_id's full-completion
    wallclock and size the resume walltimes to force a mid-sim kill (~T/3), then
    materialize the resume case. Run AFTER the clean sweep has completed.

    Delegates the per-member runtime read to
    ``hhemt.synthetic_experiment.size_resume_walltimes`` (df_status['perf_Total'];
    on the clean run this equals SLURM Elapsed because clean is never resumed).
    ``cell_size_m`` MUST match the clean sweep's for a valid byte-identity compare.

    ``variant`` is forwarded to BOTH the internal ``clean_case`` read AND the returned
    ``resume_case``, which is a DELIBERATE COUPLING and not an oversight. The internal read exists
    only to locate the clean arm this resume arm pairs with, so its analysis name MUST be the one
    the clean arm was actually minted under: a resume arm at ``variant="_ornl"`` whose internal read
    reconstructed ``synth_cc_clean_{model_arm}`` would point at a directory that does not exist
    under ``clean_system_directory``. One parameter therefore enforces the invariant that the pair
    shares an infix, and an asymmetric pair is UNWRITABLE rather than merely discouraged. A future
    caller genuinely needing asymmetric infixes must add a second parameter and argue for it; do
    not reach for that to work around a mis-set ``clean_system_directory``.

    ``clean_variant`` IS that second parameter, and this is the argument for it. It defaults to
    ``None``, which resolves to ``variant`` and reproduces the coupling above EXACTLY -- every
    existing caller is byte-unaffected, and the asymmetric pair stays unwritable unless a caller
    names it. The caller that needs it is the b4b re-run's THIRD arm, ``A3``: two resume arms sit at
    the SAME solver pin and differ only in which resume path they exercise -- ``A2`` takes the
    full-precision state snapshot, ``A3`` is forced onto the old exchange-replay fallback by
    ``swmm_snapshot_disable=1`` -- and BOTH are compared against the ONE same-pin clean arm ``A1``,
    which is minted at ``variant="P"``. So ``A3`` needs the clean infix ``P`` (to locate ``A1``) and
    a resume infix that is NOT ``P`` (because ``synth_cc_resumeP_tritonswmm`` is already ``A2``, and
    the docstring above gives the reason that collision is unacceptable: two resume arms sharing one
    inner analysis name reproduce the wrong-pin-at-a-plausible-path ambiguity one level down). One
    parameter cannot express that pair, which is exactly the case this parameter is added for.

    WHAT IT DOES NOT LICENSE. It does not make the clean arm optional or the pairing advisory: the
    internal read is still LIVE against ``clean_system_directory``, so naming a clean infix whose
    arm is absent or incomplete still raises, which is the stop described below and is deliberately
    preserved. Do not reach for this parameter to work around a mis-set ``clean_system_directory``
    either -- that is the same misuse the paragraph above warns about, and it now has a second door.

    WHAT THAT INTERNAL READ IS AND IS NOT, because its name oversells it. ``size_resume_walltimes``
    is called and its result is DISCARDED — ``resume_case`` accepts ``runtime_min_by_member`` and
    forwards it nowhere (``_build_case`` has no such parameter, and ``write_resume_matrix_csv`` is
    called bare). Under Option D the kill is a deterministic checkpoint-count SIGKILL and every row
    carries the generous clean walltime, so nothing is sized from the clean sweep any more. What
    SURVIVES is that the read is LIVE: it touches ``clean_analysis.df_status`` and
    ``dropna(subset=["perf_Total"])``, so it can raise against an incomplete or absent clean tree.
    That is the real reason ``clean_system_directory`` must name the pair's OWN clean arm — not
    walltime provenance, which no longer exists.
    """
    from hhemt.synthetic_experiment import size_resume_walltimes

    # None resolves to `variant`, so the symmetric pairing above is preserved byte-for-byte for
    # every caller that does not name an asymmetric pair. An empty STRING is a real, distinct
    # value here -- it names the un-infixed clean arm `synth_cc_clean_{model_arm}` -- so the
    # resolution tests `is None` rather than falsiness.
    clean_infix = variant if clean_variant is None else clean_variant
    clean = clean_case(
        system_directory=clean_system_directory,
        cell_size_m=cell_size_m,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
        tritonswmm_git_url=tritonswmm_git_url,
        tritonswmm_software_directory=tritonswmm_software_directory,
        model_arm=model_arm,
        variant=clean_infix,
    )
    runtime_min_by_member = size_resume_walltimes(clean.analysis)
    return resume_case(
        system_directory=system_directory,
        cell_size_m=cell_size_m,
        runtime_min_by_member=runtime_min_by_member,
        hpc_system_config_yaml=hpc_system_config_yaml,
        tritonswmm_branch_key=tritonswmm_branch_key,
        tritonswmm_git_url=tritonswmm_git_url,
        tritonswmm_software_directory=tritonswmm_software_directory,
        model_arm=model_arm,
        variant=variant,
        # Forwarded to the RESUME arm only. The clean read above must NEVER carry it: the clean
        # arm never resumes, so the key would have no effect on its behaviour but WOULD land in
        # its reconstructed config, and `A1` is the shared comparand for both resume arms.
        swmm_snapshot_disable=swmm_snapshot_disable,
        # Forwarded to the RESUME arm only, for the identical reason: the retention prune this
        # flag disarms runs on the RESUME restore path, so the key is inert on a clean arm and
        # would still land in the clean arm's reconstructed config and destroy its own
        # membership-by-literal reading.
        swmm_snapshot_keep_all=swmm_snapshot_keep_all,
    )


#: Member-directory globs the stamp read walks, in order. BOTH are required and neither is
#: redundant: live trees carry ``members/member_*`` while trees minted before the members-rename
#: carry ``subanalyses/sa_*``, and the per-member store is the ONE stamp location uniform across
#: the two eras. A MASTER-ONLY read (``sensitivity_datatree.zarr`` at the arm root) returns a FALSE
#: ABSENT on a live arm, because a live arm has no master store at all — which is exactly the
#: false-pass instrument this function exists instead of.
_MEMBER_STORE_GLOBS: tuple[str, ...] = ("members/member_*", "subanalyses/sa_*")

#: The attribute the stamp lives in. Named as a constant because a NEIGHBOURING attribute,
#: ``hhemt_producing_sha``, IS populated on exactly the trees whose ``triton_producing_sha`` is
#: absent — measured on all 30 stores of the different-pin clean tree — so any read loose enough to
#: accept a ``*_producing_sha`` sibling reports a stamped arm on an unstamped one.
_STAMP_ATTR = "triton_producing_sha"


def _read_member_stamp(store: Path) -> str | None:
    """Read one per-member store's ``triton_producing_sha``, or None when it carries no value.

    ZARR V3, AND THE V2 READ IS A FALSE-ABSENT RATHER THAN AN ERROR YOU NOTICE. These stores are
    ``zarr_format: 3`` (measured), so the attributes live in ``zarr.json`` under ``attributes`` and
    there is NO ``.zattrs`` file. A ``.zattrs`` read raises ``FileNotFoundError`` on EVERY store,
    including a correctly stamped one — so an instrument that catches that exception and maps it to
    "no stamp" reports a uniform definite absence and can never observe a stamp that is present.
    Measured on a real store: ``.zattrs`` present=False, read raised ``FileNotFoundError``, while the
    ``zarr.json`` read returned the attribute namespace.

    Returns None for: no ``zarr.json``, unparseable JSON, no ``attributes``, attribute absent, or an
    attribute present but empty/whitespace. An EMPTY string is a non-stamp, not a stamp — treating a
    falsy value as present is the other direction of the same false-pass.
    """
    import json

    zarr_json = store / "zarr.json"
    if not zarr_json.is_file():
        return None
    try:
        attributes = json.loads(zarr_json.read_text()).get("attributes")
    except (ValueError, OSError):
        return None
    if not isinstance(attributes, dict):
        return None
    value = attributes.get(_STAMP_ATTR)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _shas_agree(stamped: str, expected: str) -> bool:
    """True iff two commit-ish strings name the same commit under abbreviation.

    The stamp is written full-length (40 chars) while an operator supplies a pin in either form, so
    an equality test on the raw strings reports a MISMATCH on a correct arm whenever the two
    lengths differ. Prefix agreement with a 7-character floor is the comparison git itself uses for
    an abbreviated rev; below 7 the prefix is not discriminating and is refused rather than
    accepted loosely.
    """
    a, b = stamped.strip().lower(), expected.strip().lower()
    if len(a) < 7 or len(b) < 7:
        return False
    return a.startswith(b) or b.startswith(a)


def declared_member_count(*, rank_sweep: tuple[int, ...] | None = None) -> int:
    """The clean/resume sweep's DECLARED member count — its sensitivity-matrix row count.

    This is the ``expected_members`` input to ``check_arm_pin_stamp`` for a caller that has no live
    case to count (the stand-alone ``stamp-check`` phase). It is DERIVED from the matrix builder
    rather than written as a literal ``30``, so it moves with ``rank_sweep`` instead of silently
    disagreeing with an arm minted under a different sweep.

    Serves BOTH sweeps: ``write_clean_matrix_csv`` and ``write_resume_matrix_csv`` both emit
    ``_rows(_configs(rank_sweep))``, differing only in per-row walltime, so their row counts are
    equal by construction (measured: 30 and 30 at the default sweep, with 30 unique ``member_id``
    in each). The count goes through the PUBLIC writer that ``clean_case`` itself calls, so it
    cannot drift from what the mint actually declares.

    A caller that DOES hold the minted case should pass ``len(case.analysis.sensitivity.members)``
    instead — that is the realized population, and it is strictly better evidence than a
    re-derivation of the matrix.
    """
    import tempfile

    import pandas as pd

    kwargs = {} if rank_sweep is None else {"rank_sweep": tuple(rank_sweep)}
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "matrix.csv"
        write_clean_matrix_csv(csv, **kwargs)
        return len(pd.read_csv(csv))


def check_arm_pin_stamp(arm_root: Path, expected_sha: str, *, expected_members: int) -> dict:
    """MINT-TIME stamp check for one comparand arm. Returns a verdict dict; raises nothing.

    This is the TERM of §10 item 12's routed mechanics: immediately after ``A1`` is minted and
    BEFORE the resume arms are launched, every per-member store must carry a non-empty
    ``triton_producing_sha`` equal to the pin the arm was built at. An absent or differing stamp is a
    STOP before 60 more members are spent — not a NOT-EVALUATED discovered after them.

    WHY A POSITIVE CHECK RATHER THAN TRUSTING THE GATE. The gate's ``G3`` is fail-closed and
    correct, but it runs at ACCEPTANCE time. The capture is intermittent rather than
    deterministically absent — measured across 22 ``system_log.json`` under the run root, 7 carry a
    null ``triton_head_sha`` (31.8%), and the split is NOT clean-versus-resume (5 of 10 clean runs DO
    capture it while 2 resume arms do not). An intermittent ~32% failure with an unisolated cause
    cannot be shown not to recur, which is the argument FOR a positive check at mint time.

    THE DENOMINATOR IS PART OF THE VERDICT, and it comes from the DECLARED population rather than
    the FOUND one. A verdict derived as "no bad stores were found" cannot distinguish "examined 30,
    all good" from "examined 0". The first repair of that gap guarded only the zero case, and zero is
    the one value of the family a glob-derived denominator gets right: a member directory that exists
    and carries NO ``analysis_datatree.zarr`` is matched by no glob, so it is neither examined nor
    reported missing, and the arm returns ``PASS`` on a population it never looked at. Measured on
    the landed instrument before this argument existed: 30 member directories, 3 consolidated,
    verdict ``PASS``, ``examined`` 3, ``missing`` ``[]``, reason "all 3 store(s) stamped and equal
    to e53c2fa0…" — a true sentence answering a different question than the one asked.

    So ``expected_members`` is REQUIRED and keyword-only, and ``examined != expected_members`` is
    ``NOT-EVALUATED`` in EITHER direction. Required rather than defaulted because a default of
    "do not check" leaves the defect live at exactly the call sites that reach it, and the omission
    is then unwritable instead of silent. The number is free at every call site — it is the matrix
    row count, which is the member count by construction (``write_clean_matrix_csv`` emits one row
    per ``member_id`` and ``_create_members`` makes one member per row).

    A NON-POSITIVE ``expected_members`` is itself ``NOT-EVALUATED``, and that branch is load-bearing
    rather than defensive: without it ``expected_members=0`` against an empty glob satisfies
    ``examined == expected_members`` and returns ``PASS`` with "all 0 store(s) stamped" — the
    original false pass, re-entered through the argument added to prevent it.

    THE CARDINALITY GUARD ALSO SUBSUMES THE BOTH-LAYOUTS HAZARD, which is why no de-duplication of
    the two globs is performed. A tree carrying both member containers yields two DISTINCT stores per
    logical member, so a path-keyed de-duplication can never fire (the globs' first segments are
    disjoint) and a suffix-keyed one would guard a shape ``V0019`` REFUSES to produce — it
    ``move_dir``s ``subanalyses/`` onto ``members/`` with ``merge_policy="error"``, so the source
    container ceases to exist and a pre-existing destination raises. Under this guard such a tree
    reports ``NOT-EVALUATED`` because its store count exceeds its declared member count, which is the
    refusing direction. The hazard is handled here, by the denominator, not by a de-duplication.

    DELIBERATELY NOT A WHOLE-TREE GREP. Measured on the different-pin clean tree:
    ``grep -rl triton_producing_sha`` returns 4 files — three binary ``render_bundle/*.zip`` and
    ``validation_report.json`` — against ZERO stores carrying a value. A grep-based check reports
    that unstamped tree as stamped.
    """
    # A plain UNION of the two globs, deliberately un-de-duplicated. The two patterns differ in
    # their first path segment, so no path can be produced by both and a path-keyed de-duplication
    # is unreachable by construction; the both-layouts tree a suffix-keyed one would address is
    # refused by V0019 rather than produced by it, and is caught here by the cardinality guard
    # below in the refusing direction. See the docstring's final paragraph.
    stores: list[Path] = []
    for glob in _MEMBER_STORE_GLOBS:
        stores.extend(sorted(arm_root.glob(f"{glob}/analysis_datatree.zarr")))

    stamped: dict[str, str] = {}
    missing: list[str] = []
    mismatched: dict[str, str] = {}
    for store in stores:
        member = store.parent.name
        value = _read_member_stamp(store)
        if value is None:
            missing.append(member)
            continue
        stamped[member] = value
        if not _shas_agree(value, expected_sha):
            mismatched[member] = value

    examined = len(stores)
    # THE ONE CARDINALITY GUARD. It replaces the `examined == 0` branch rather than sitting beside
    # it: a second guard for a case this one already decides would be dead the moment it was
    # written, which is the defect this commit repairs elsewhere in this function. The zero case
    # keeps its distinct path-typo diagnostic inside the shared branch, because a wrong arm root and
    # a short arm are different operator errors with different remedies.
    if expected_members < 1:
        verdict = "NOT-EVALUATED"
        reason = (
            f"expected_members={expected_members} declares no population, so every verdict over it "
            "is vacuous — pass the arm's matrix row count"
        )
    elif examined != expected_members:
        verdict = "NOT-EVALUATED"
        if examined == 0:
            detail = (
                f"no per-member analysis_datatree.zarr store found under {arm_root} — a wrong arm "
                "root is the likeliest cause, the path is DOUBLED as {run_root}/{name}/{name}"
            )
        elif examined < expected_members:
            detail = (
                f"{expected_members - examined} declared member(s) carry no store; an unconsolidated "
                "member is matched by no glob, so it is neither examined nor reported missing"
            )
        else:
            detail = (
                f"{examined - expected_members} store(s) beyond the declared population — a stale "
                "member directory or both member containers present on one tree"
            )
        reason = f"examined {examined} store(s) against {expected_members} declared member(s): {detail}"
    elif missing or mismatched:
        verdict = "STOP"
        parts = []
        if missing:
            parts.append(f"{len(missing)} store(s) carry no {_STAMP_ATTR}: {sorted(missing)[:5]}")
        if mismatched:
            parts.append(f"{len(mismatched)} store(s) differ from {expected_sha}: {sorted(mismatched.items())[:5]}")
        reason = "; ".join(parts)
    else:
        verdict = "PASS"
        # The denominator is named in the PASS reason too, not only in the refusals. The retired
        # wording ("all 3 store(s) stamped and equal to …") was TRUE on the 3-of-30 arm and answered
        # a different question than the one asked; saying "all 30 of 30 declared" is the same
        # sentence with the thing a reader needs in order to tell those two states apart.
        reason = f"all {examined} of {expected_members} declared member store(s) stamped and equal to {expected_sha}"
    return {
        "verdict": verdict,
        "arm_root": str(arm_root),
        "expected_sha": expected_sha,
        "examined": examined,
        "expected_members": expected_members,
        "stamped": len(stamped),
        "missing": sorted(missing),
        "mismatched": mismatched,
        "reason": reason,
    }


def _emit_bundle(case: _Case) -> Path:
    """Committed emit step (supersedes the prior inline-heredoc runbook): eda + bundle a materialized
    case, returning the bundle path. Requires df_status all-complete (batch_job runs out-of-band)."""
    a = case.analysis
    # WHAT THIS FUNCTION IS: an out-of-band emit step. The workflow ran ELSEWHERE
    # (docstring: "Requires df_status all-complete (batch_job runs out-of-band)"), so there
    # is no _run_workflow on this path and no DAG executes here. The three calls are:
    # eda() computes the EDA members, render_report() produces analysis_report.{html,zip},
    # and bundle_report_data() harvests plots/eda/*.manifest.json into the bundle.
    #
    # WHY render_report() STAYS: it is the ONLY producer of analysis_report.{html,zip} on
    # this path. Deleting it is a regression, not dead-code removal.
    #
    # WHAT IT DOES NOT DO, and the earlier comment here claimed otherwise: it does NOT
    # rebuild any figure. render_report() invokes `snakemake --report`, which renders and
    # executes no rules (analysis.py:2899, :2970). Measured with a two-rule Snakefile whose
    # output was older than its input -- under `--report` the rule did not execute; under a
    # real invocation with the same --rerun-triggers mtime it did. The retired claim that
    # "the now-newer JSON re-fires exactly this rule" was false and cost three cluster runs.
    #
    # KNOWN RESIDUAL: eda() re-persists validation_report.json as its last act
    # (analysis.py:1053), so on this path the Errors-and-Warnings figure can end up older
    # than the read-model it transcribes, and emit_bundle refuses that tree
    # (StaleReadModelError, bundle/_emit.py). There is no actuator here that can fix it --
    # rebuilding the figure needs a rule to execute, and nothing on this path executes rules.
    # Closing it is a design decision about what _emit_bundle is for (hoist eda() above the
    # out-of-band run, or add a rule-executing step here), NOT another mechanism bolted on.
    a.eda()  # writes eda/{plot_id}.zarr + .verdict.json + plots/eda/*.html
    a.render_report()  # the only producer of analysis_report.{html,zip} here; rebuilds no figures
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
    _by_name: dict[str, object] = {}
    for name in ("clean", "resume"):
        sp = sub.add_parser(name)
        _by_name[name] = sp
        sp.add_argument("--system-directory", required=True)
        sp.add_argument("--hpc-system-config", type=Path, default=None)
        sp.add_argument("--cell-size-m", type=float, default=3.5)
        # FAIL CLOSED. This flag decides what is CLONED AND BUILT, not merely what is
        # asserted, so a default silently builds a version nobody chose -- the previous
        # "3a832f7d" default was stale against every pin used since it was written and
        # nothing surfaced that.
        sp.add_argument(
            "--tritonswmm-sha",
            required=True,
            help="TRITON commit to BUILD for this case. No default: an invocation that names no version fails.",
        )
        # Siblings of --tritonswmm-sha. Default None so every existing invocation keeps
        # today's behaviour: the case builder only writes a key when the value is not None.
        sp.add_argument("--tritonswmm-git-url", default=None)
        sp.add_argument("--tritonswmm-software-directory", default=None)
        sp.add_argument("--eda", action="store_true")
        sp.add_argument("--bundle", action="store_true")
    # CLEAN ONLY ON THIS CLI SURFACE, and the reason is no longer that the resume arm lacks the
    # parameter — `resume_case` and `build_resume_from_clean_runtimes` BOTH accept `variant` now,
    # because the ORNL-pinned solver-version control needs a resume arm whose INNER analysis name
    # differs from the fork-pinned one. What stays CLI-clean-only is the FLAG, and that is a
    # surface decision rather than a capability one: a resume arm's variant must MATCH its paired
    # clean arm's (the internal clean read resolves `synth_cc_clean{variant}_{arm}`), and this CLI
    # exposes no way to state the pair, so an operator-supplied `--variant` on `resume` could
    # silently name a clean tree that does not exist. The estate driver calls the Python factories
    # directly and sets both halves of the pair at one site, which is where the invariant is
    # expressible. Default "" keeps every existing `clean` invocation byte-identical.
    _by_name["clean"].add_argument(
        "--variant",
        default="",
        help="Infix on the clean analysis name: synth_cc_clean{VARIANT}_{arm}. "
        "Use 'P' for the SAME-PIN clean reference arm (A1) of the four-arm b4b re-run; "
        "the default '' is the historical synth_cc_clean_{arm}.",
    )
    # MINT-TIME stamp check (a TERM of the routed b4b re-run mechanics, not a diagnostic). Exits
    # 0 only on PASS: a STOP and a NOT-EVALUATED both exit non-zero, because the whole point is to
    # halt before the resume arms are launched and a zero exit on "nothing examined" would not.
    stp = sub.add_parser("stamp-check")
    stp.add_argument(
        "--arm-root",
        type=Path,
        required=True,
        help="The arm's DOUBLED analysis dir, i.e. {run_root}/{name}/{name} — the parent of members/ or subanalyses/.",
    )
    stp.add_argument("--expected-sha", required=True, help="The pin the arm was BUILT at.")
    # FAIL CLOSED, for the same reason --tritonswmm-sha carries no default. The denominator is
    # part of the verdict, so a default silently asserts a population nobody declared for THIS
    # arm — and the value a default could plausibly take (today's matrix row count) is wrong for
    # any arm minted under a different rank_sweep, which is precisely the drift a default hides.
    # A wrong operator value halts (NOT-EVALUATED); an absent declaration cannot pass.
    stp.add_argument(
        "--expected-members",
        type=int,
        required=True,
        help="How many members the arm DECLARES, i.e. its sensitivity-matrix row count (30 for the "
        "clean/resume sweeps at the default rank_sweep). A store count that disagrees in either "
        "direction is NOT-EVALUATED, never PASS: an unconsolidated member is matched by no glob, "
        "so without this the arm passes on a population it never looked at.",
    )
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
    # TWO SHAS, because under a split pin they are different versions and one flag cannot
    # mean both. --tritonswmm-sha is the version the RESUME case BUILDS; --clean-tritonswmm-sha
    # is the version the CLEAN bundle being depended on was BUILT WITH, which resolve_dependency
    # verifies rather than builds. Both fail closed.
    ip.add_argument(
        "--tritonswmm-sha",
        required=True,
        help="TRITON commit to BUILD for the resume case.",
    )
    ip.add_argument(
        "--clean-tritonswmm-sha",
        required=True,
        help="TRITON commit the CLEAN bundle was built with; verified, not built.",
    )
    # Siblings of --tritonswmm-sha, MISSING from this subparser until now while the code path
    # below read args.tritonswmm_git_url / args.tritonswmm_software_directory -- an AttributeError
    # on every invocation that got past dependency resolution, latent only because
    # resolve_dependency halts first when the clean bundle is absent.
    ip.add_argument("--tritonswmm-git-url", default=None)
    ip.add_argument("--tritonswmm-software-directory", default=None)
    ip.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if args.cmd == "stamp-check":
        import json as _json
        import sys as _sys

        verdict = check_arm_pin_stamp(args.arm_root, args.expected_sha, expected_members=args.expected_members)
        print(_json.dumps(verdict, indent=2, sort_keys=True))
        print(f"STAMP-CHECK {verdict['verdict']}: {verdict['reason']}")
        _sys.exit(0 if verdict["verdict"] == "PASS" else 1)

    if args.cmd in ("clean", "resume"):
        factory = clean_case if args.cmd == "clean" else resume_case
        # `resume_case` ALSO accepts `variant` now, so this guard is no longer about capability —
        # it mirrors the parser, which defines `--variant` on the `clean` subparser only (see the
        # CLI-surface comment above for why). Passing it only where the parser defined it keeps the
        # resume path's call signature byte-identical to before and keeps `args.variant` from being
        # read on a namespace that has no such attribute.
        extra = {"variant": args.variant} if args.cmd == "clean" else {}
        case = factory(
            system_directory=args.system_directory,
            cell_size_m=args.cell_size_m,
            hpc_system_config_yaml=args.hpc_system_config,
            tritonswmm_branch_key=args.tritonswmm_sha,
            tritonswmm_git_url=args.tritonswmm_git_url,
            tritonswmm_software_directory=args.tritonswmm_software_directory,
            **extra,
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
        resume_depends_on(tritonswmm_sha=args.clean_tritonswmm_sha),
        search_roots=list(args.clean_bundle_search_root),
        auto_satisfy=None,
        emitted_command=(
            f"python -m scripts.experiments.synth_compute_config clean "
            f"--system-directory {args.clean_system_directory} "
            f"{hpc_flag}"
            f"--tritonswmm-sha {args.clean_tritonswmm_sha} --eda --bundle"
        ),
    )
    resume_case_obj = resume_case(
        system_directory=args.resume_system_directory,
        cell_size_m=args.cell_size_m,
        hpc_system_config_yaml=args.hpc_system_config,
        tritonswmm_branch_key=args.tritonswmm_sha,
        tritonswmm_git_url=args.tritonswmm_git_url,
        tritonswmm_software_directory=args.tritonswmm_software_directory,
    )
    resume_root = _emit_bundle(resume_case_obj)  # ensure the resume bundle (df_status must be complete)
    combined = combine_bundle([clean_root, Path(resume_root)], output_path=args.output)
    print("COMBINED:", combined.root)  # lineage stamp lives in combined_of (FILE 3)


if __name__ == "__main__":
    _cli()
