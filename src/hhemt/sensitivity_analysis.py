# %%
import hashlib
import json
import os
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
import yaml  # type: ignore

import hhemt.analysis as anlysis
from hhemt import orchestrator_sentinels as _osent
from hhemt.cf_conventions import apply_global_attributes
from hhemt.config.analysis import ClearRawValue, ForceRerunValue
from hhemt.config.hpc_system import resolve_additional_modules, resolve_gpu_target
from hhemt.exceptions import ConfigurationError
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.utils import current_datetime_string, write_datatree_zarr
from hhemt.validation import assert_configs_visible_cross_node
from hhemt.workflow import (
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeDiagnostics,
    _emit_report_artifacts,
)

#: The retired sensitivity id-column spellings, kept ONLY so the reader can tell an
#: operator holding a pre-rename experiments/*.xlsx which header to change. Exact
#: match on the lowercased, stripped column name -- never a substring, because a
#: legitimate column such as "analysis.sa_id_note" contains the retired token and
#: must not trip the guard.
_RETIRED_ID_COLUMNS = frozenset({"sa_id", "sa-id", "sa id"})

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis
    from .orchestration import RunOverrides
    from .system import TRITONSWMM_system


__all__ = ["TRITONSWMM_sensitivity_analysis"]


@dataclass
class UniqueSystemTarget:
    target_id: int
    system_config_yaml: Path
    system: "TRITONSWMM_system"
    analysis_ids: list[str] = field(default_factory=list)
    # Phase 6 (DQ7a): the ensemble partition this build target compiles for.
    # All member_ids in a target share (hw, backend) by dedup-key construction, so any
    # member's partition yields the correct GPU build; the first member's is stored.
    # Threaded to the setup rule's --target-partition so the GPU compile resolves
    # the right PartitionSpec hardware per build target (not the master partition).
    target_partition: str | None = None


_SYSTEM_COLUMN_PREFIX = "system."
_ANALYSIS_COLUMN_PREFIX = "analysis."
_HPC_COLUMN_PREFIX = "hpc."
# hpc.{key} columns are a provenance-rooted ALIAS surface for HPC-resource
# sensitivity axes. The single supported alias today is hpc.partition ->
# analysis.hpc_ensemble_partition (the partition selector stays an analysis_config
# field per D-A; only the column SPELLING gains an hpc. root). DQ1 Option 3.
_HPC_ALIAS_TO_ANALYSIS_FIELD: dict[str, str] = {
    "partition": "hpc_ensemble_partition",
    "setup_partition": "hpc_setup_and_analysis_processing_partition",
}


def _is_hpc_overlay_column(col: str) -> bool:
    """True if `col` is `hpc.{key}` where key is a supported HPC-resource alias."""
    if not col.startswith(_HPC_COLUMN_PREFIX):
        return False
    return col[len(_HPC_COLUMN_PREFIX) :] in _HPC_ALIAS_TO_ANALYSIS_FIELD


def _resolve_hpc_alias_to_analysis_field(col: str) -> str:
    """Map an `hpc.{key}` column to its target analysis_config field name."""
    assert col.startswith(_HPC_COLUMN_PREFIX), f"expected hpc.* column, got {col!r}"
    return _HPC_ALIAS_TO_ANALYSIS_FIELD[col[len(_HPC_COLUMN_PREFIX) :]]


def _is_system_overlay_column(col: str) -> bool:
    """True if `col` is `system.{field}` where field is in system_config.model_fields."""
    if not col.startswith(_SYSTEM_COLUMN_PREFIX):
        return False
    from hhemt.config.system import system_config

    field_name = col[len(_SYSTEM_COLUMN_PREFIX) :]
    return field_name in system_config.model_fields


def _is_analysis_overlay_column(col: str) -> bool:
    """True if `col` is `analysis.{field}` where field is in analysis_config.model_fields."""
    if not col.startswith(_ANALYSIS_COLUMN_PREFIX):
        return False
    from hhemt.config.analysis import analysis_config

    field_name = col[len(_ANALYSIS_COLUMN_PREFIX) :]
    return field_name in analysis_config.model_fields


def _strip_system_prefix(col: str) -> str:
    """Return the system_config field name from a `system.{field}` column."""
    assert col.startswith(_SYSTEM_COLUMN_PREFIX), f"expected system.* column, got {col!r}"
    return col[len(_SYSTEM_COLUMN_PREFIX) :]


def _strip_analysis_prefix(col: str) -> str:
    """Return the analysis_config field name from an `analysis.{field}` column."""
    assert col.startswith(_ANALYSIS_COLUMN_PREFIX), f"expected analysis.* column, got {col!r}"
    return col[len(_ANALYSIS_COLUMN_PREFIX) :]


def _resolve_row_ensemble_partition(row, master_partition: str | None) -> str | None:
    """Per-row ensemble partition from the partition-overlay cell, else master.

    Recognizes the canonical `hpc.partition` alias and the legacy
    `analysis.hpc_ensemble_partition` spelling (both resolve to the ensemble
    selector). Returns the master ensemble partition when neither cell is set.
    Used by `_build_unique_system_targets` to derive the per-row compile-dedup
    hardware BEFORE `_create_members` materializes the per-sub cfg.
    """
    for col in ("hpc.partition", "analysis.hpc_ensemble_partition"):
        if col in row.index:
            val = row.get(col)
            if val is not None and not pd.isna(val) and str(val).strip() != "":
                return str(val)
    return master_partition


def _to_native_attr(value):
    """Cast pandas / numpy scalars to JSON-safe native Python types for zarr attrs."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _unlink_dprocess_flags_for_regenerate(targets: list[str], status_dir: Path) -> None:
    """FIX-2b — regenerate_existing rebuild parity (restores f31a0eb's
    unconditional per-member d_process unlink that d5d0084 dropped).

    The existence-keyed self-heal (_reconcile_stale_process_flags_against_summaries)
    repairs a PRE-EXISTING divergence (summaries already absent at reprocess
    entry — the regenerate_existing=False case). It is a NO-OP on the
    regenerate_existing=True path because the per-member summaries are still present
    at self-heal time; the deletion that creates the divergence runs LATER
    (_delete_processed_outputs_for_reprocess / the SLURM reprocess-delete
    workflow). So for regenerate_existing the stale d_process flag would survive,
    the master generator's emit gate (workflow.py:6810, `not d_process_path.exists()`)
    would skip the rebuild rule, and consolidate would fan in against the
    just-deleted summary -> the silent-partial / FileNotFoundError class. Unlink
    the per-member d_process flags unconditionally on this arm so the gate re-emits
    the rule. The per-model LOG-clear (Gate 2, Gotcha 28) is handled by the later
    _delete_processed_outputs_for_reprocess (scope="all"); only the flag (Gate 1)
    is cleared here. Mirrors the non-sensitivity arm's blanket unlink at
    analysis.py:2989.

    Extracted to a free function so the fast (no-compile, no-fixture-mutation)
    unit test exercises EXACTLY this loop against a synthetic _status/ dir without
    entering reprocess()'s destructive body (D1 Option A).
    """
    for member_id in targets:
        for f in status_dir.glob(f"d_process_*_member-{member_id}_*"):
            # EXEMPT-DU: status-flag
            f.unlink(missing_ok=True)


def _should_materialize_analysis_yaml(cfg_yaml: Path, is_main_orchestrator: bool) -> bool:
    """Decide whether this construction should (re)write a member config YAML.

    member YAML materialization is a DRIVER-only side effect. Every
    `TRITONSWMM_analysis` construction with `toggle_sensitivity_analysis=True`
    reaches `_create_members`, including the ~63 per-figure renderer
    subprocesses spawned by the report tail (`report_renderers/_cli.py`
    constructs with `is_main_orchestrator=False`). Those renderers previously
    rewrote all N member YAMLs each, putting every destination name under
    continuous concurrent-rename churn on the shared filesystem for the ~12 s a
    full pass takes. Concurrent cross-client renames onto a shared destination
    name were observed to expose a window in which a third client's `open()`
    returns ENOENT -- a renderer failing to read the very file it had just
    renamed into place, killing the workflow at the report tail.

    The writes were redundant, not merely racy: each sub config is derived
    deterministically from the same master `cfg_analysis.model_dump()` plus the
    same sensitivity-CSV row, so all writers produce byte-identical content.
    Gating on `is_main_orchestrator` therefore removes N-1 writers with no
    content change -- the same convention `_build_unique_system_targets` already
    applies to its `_generated/` purge.

    The absent-target fallback keeps the change safe on any construction path
    that is genuinely first-to-materialize: a missing YAML is always written,
    whoever is constructing, so no consumer can encounter a missing file.
    """
    return bool(is_main_orchestrator) or not cfg_yaml.exists()


class TRITONSWMM_sensitivity_analysis:
    """
    Manages sensitivity analysis by creating and orchestrating multiple members.

    This class creates a separate TRITONSWMM_analysis instance for each row in a
    sensitivity analysis configuration table (CSV or Excel). Each member runs
    with different parameter values, and results are consolidated at the master level.

    The sensitivity analysis workflow:
    1. Reads sensitivity configuration (CSV/Excel with parameter combinations)
    2. Creates member for each configuration row
    3. Runs simulations for all members
    4. Consolidates outputs across all parameter combinations
    5. Produces multi-dimensional datasets with sensitivity dimensions

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        Master analysis instance that contains the sensitivity configuration

    Attributes
    ----------
    experiment : TRITONSWMM_analysis
        Reference to the master analysis
    analyses : dict
        Dictionary mapping member index to TRITONSWMM_analysis instances
    df_setup : pd.DataFrame
        Sensitivity configuration table with parameter combinations
    independent_vars : list
        List of parameters being varied in the sensitivity analysis
    """

    def __init__(
        self,
        analysis: "TRITONSWMM_analysis",
        is_main_orchestrator: bool = True,
        skip_log_update: bool = False,
    ) -> None:
        """
        Initialize a sensitivity analysis orchestrator.

        Creates members for each parameter combination defined in the sensitivity
        configuration file, enabling systematic exploration of parameter space.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            Master analysis instance containing sensitivity configuration

        Raises
        ------
        ValueError
            If sensitivity configuration mixes GPU and non-GPU run modes
        """
        self.experiment = analysis
        self._system = analysis._system
        self._skip_log_update = skip_log_update
        self._is_main_orchestrator = is_main_orchestrator
        self.analysis_paths = analysis.analysis_paths
        self.cfg_analysis = analysis.cfg_analysis
        # BundleableAnalysis delegation (_protocol.py): emit_bundle reads these off its input, so a
        # sensitivity master (the real experiment topology) must expose them too — else the combine
        # identity surfaces silently drop: case.yaml (BLOCKING case_name) + hpc_system_config.identity.yaml
        # (INFORMATIONAL compute-config). Mirrors the cfg_analysis/analysis_paths delegation above.
        self.cfg_hpc_system = analysis.cfg_hpc_system
        self.case_manifest_yaml = analysis.case_manifest_yaml
        self._case_manifest = analysis._case_manifest
        self.member_prefix = "member_"
        self.members_dir = self.experiment.analysis_paths.analysis_dir / "members"
        df_setup_full = self._retrieve_df_setup()
        self._df_setup_full = df_setup_full
        self._has_per_member_system_configs = "system_config_yaml" in df_setup_full.columns
        self._has_per_member_system_overlay_columns = any(_is_system_overlay_column(c) for c in df_setup_full.columns)
        # Phase 6 (DQ7): a per-row ensemble-partition axis (hpc.partition canonical or
        # analysis.hpc_ensemble_partition legacy) that varies across rows resolves to
        # DISTINCT gpu_hardware per row, so it must route through the per-target build
        # path (one UniqueSystemTarget per distinct hardware) — not the master fast
        # path. A single distinct partition collapses to the master target as before.
        _partition_axis_cols = [
            c for c in df_setup_full.columns if c == "hpc.partition" or c == "analysis.hpc_ensemble_partition"
        ]
        _distinct_row_partitions: set = set()
        for _c in _partition_axis_cols:
            _distinct_row_partitions |= {str(v) for v in df_setup_full[_c].dropna().tolist() if str(v).strip() != ""}
        self._has_per_row_partition_variation = len(_distinct_row_partitions) > 1
        if (
            self._has_per_member_system_overlay_columns
            or self._has_per_member_system_configs
            or self._has_per_row_partition_variation
        ):
            self.unique_system_targets = self._build_unique_system_targets(
                df_setup_full,
                is_main_orchestrator=is_main_orchestrator,
            )
        else:
            # Fast path: no row varies system_config; reuse master self._system. A
            # same-partition GPU sensitivity suite (e.g. the native/container
            # within-family suite, which shares ONE partition across rows so the
            # per-row-partition gate above is False) lands here. The single target
            # MUST carry the master ensemble partition so `setup_target_N` emits
            # `--target-partition` (-> resolve_gpu_target -> GPU compile backend) AND
            # self._system gets the GPU pair injected (the workflow.py GPU-sensitivity
            # validation reads self.system.gpu_compilation_backend). Guarded on a
            # resolved backend so CPU/null-selector fast paths stay byte-identical
            # (every Snakefile golden builds with hpc_ensemble_partition null ->
            # _fast_backend is None -> no injection, no target_partition).
            _master_partition = self.experiment.cfg_analysis.hpc_ensemble_partition
            _fast_hw, _fast_backend = resolve_gpu_target(self.experiment.cfg_hpc_system, _master_partition)
            _fast_target_partition = None
            if _fast_backend is not None:
                # synth_cc friction fix (2026-07-08): same invariant as the build-path
                # site — inject the master-ensemble GPU pair into self._system only on
                # the DRIVER. On a setup_target runner (is_main_orchestrator=False)
                # self._system already carries the correct --target-partition pair (the
                # fast path fires only for single-partition suites, so runner partition
                # == master ensemble; this is a no-op-same-value today, gated to keep the
                # "runner never mutates the compile target's gpu fields" invariant explicit).
                if is_main_orchestrator:
                    self._system.gpu_hardware = _fast_hw
                    self._system.gpu_compilation_backend = _fast_backend
                    self._system.additional_modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
                _fast_target_partition = _master_partition
            self.unique_system_targets = [
                UniqueSystemTarget(
                    target_id=0,
                    system_config_yaml=self._system.system_config_yaml,
                    system=self._system,
                    analysis_ids=list(df_setup_full.index.astype(str)),
                    target_partition=_fast_target_partition,
                )
            ]
        from hhemt.config.analysis import analysis_config as _analysis_config_for_df_setup

        analysis_cols = [
            c
            for c in df_setup_full.columns
            if c in _analysis_config_for_df_setup.model_fields
            or _is_analysis_overlay_column(c)
            or _is_hpc_overlay_column(c)  # Phase 6 (DQ5): retain hpc.* alias columns so
            # the per-sub overlay application in _create_members can resolve them.
        ]
        self.df_setup = df_setup_full.loc[:, analysis_cols]
        self.members = self._create_members()

        # Initialize workflow builder for sensitivity analysis
        self._workflow_builder = SensitivityAnalysisWorkflowBuilder(self)

    def prepare_scenarios_in_each_analysis(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        concurrent: bool = True,
        verbose: bool = False,
    ):
        """Prepare every scenario across every member (workflow phase 2a).

        Generates per-event SWMM `.inp` files, boundary conditions and TRITON
        `.cfg` files for each member defined by the sensitivity table.

        Scenario preparation always forks into subprocesses: PySwmm raises
        ``MultiSimulationError`` if two SWMM simulations are instantiated in one
        Python process, so ``concurrent`` controls how many subprocesses run at
        once, never whether they are used.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool
            Re-prepare scenarios whose outputs already exist.
        rerun_swmm_hydro_if_outputs_exist : bool
            Re-run the SWMM hydrology step even when its outputs are present.
        concurrent : bool
            Prepare members in parallel rather than serially.
        verbose : bool
            Emit per-scenario progress.
        """
        if self.experiment.cfg_analysis.multi_sim_run_method in [
            "local",
            "1_job_many_srun_tasks",
        ]:
            prepare_scenario_launchers = []
            for _analysis_iloc, analysis in self.members.items():
                prepare_scenario_launchers += analysis.retrieve_prepare_scenario_launchers(
                    overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                    rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                    verbose=verbose,
                )
            if concurrent:
                self.experiment.run_python_functions_concurrently(prepare_scenario_launchers, verbose=verbose)
            else:
                for launcher in prepare_scenario_launchers:
                    launcher()

            if self.all_scenarios_created is not True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(f"Preparation failed for the following scenarios:\n{scens_not_created}")
            self._update_experiment_log()
        elif self.experiment.cfg_analysis.multi_sim_run_method in ["batch_job"]:
            raise ValueError("prepare scenarios is not currently executable as batch_job.")

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        # setup stuff
        process_system_level_inputs: bool = True,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        # ensemble run stuff
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        overrides: "RunOverrides | None" = None,
        report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        This orchestrates multiple member workflows and a final master
        consolidation step that combines all member outputs.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings)
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        overrides : RunOverrides | None
            Runtime override carrier (``clear_raw``, ``force_rerun``,
            ``hpc_total_nodes``, ``hpc_restart_times_simulate/other``); ``None``
            reads every value from the config.
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str
            - snakefile_path: Path
            - message: str
        """
        # OE-A login-node preflight, mirroring the Analysis.submit_workflow twin.
        # This facade is reachable DIRECTLY (see the force-rerun comment below, which
        # names the same caller class), so a guard only at the dispatch site would
        # miss it. Pure predicate: invoking it twice on the dispatch path is free.
        _m = self.experiment
        assert_configs_visible_cross_node(
            _m._system.cfg_system,
            _m.cfg_analysis,
            {
                "--system-config": _m._system.system_config_yaml,
                "--analysis-config": _m.analysis_config_yaml,
                "--hpc-system-config": _m.hpc_system_config_yaml,
            },
            mode=mode,
        )
        if overrides is None:
            from .orchestration import RunOverrides

            overrides = RunOverrides()
        # Force-rerun pre-delete for direct sensitivity.submit_workflow callers.
        # Idempotent when Analysis.submit_workflow already applied it on the
        # dispatch path (matched flags would be absent by now).
        #
        # THE IDEMPOTENCY ARGUMENT ABOVE HOLDS FOR FLAGS AND NOT FOR FIGURES, and the
        # difference only became observable once the dispatch-path call was gated. At a
        # render floor the pre-delete deletes NO flag (the floor's prefix tuple is empty)
        # and instead deletes every figure under `plots/` except `plots/eda/`. On a dry
        # run the gated first invocation now leaves those figures in place, so an
        # ungated second invocation here finds them present and deletes them -- the
        # symptom is unchanged and the fix at analysis.py:3578 does nothing on the
        # sensitivity path, which is the path every sensitivity master takes. Measured
        # over the modelled chain: gating only the first call leaves a dry run at
        # 7 figures -> 2, identical to the unfixed behaviour.
        self.experiment._apply_force_rerun(overrides.force_rerun, dry_run=dry_run)

        # Driver-start orchestrator-liveness sentinel (Phase 2), keyed on the
        # MASTER analysis_dir. This is the sensitivity-master submit path and
        # always owns its sentinel (the Analysis.submit_workflow guard leaves
        # _driver_id None there and delegates here). Blocking-local drivers
        # remove on return; detached drivers leave a durable sentinel reclaimed
        # by the gate's liveness probes.
        #
        # A DRY RUN writes NO sentinel — exact mirror of the non-sensitivity
        # twin in analysis.py::submit_workflow. A rehearsal submits nothing and
        # writes no zarr, and on a detached mode its sentinel would keep null
        # identity fields (the enrich below has no job_id / session_name to
        # merge), which the tri-state gate holds as UNKNOWN until the
        # mtime-age fail-safe expires. Suppressed at the WRITE, not at the
        # removal condition: that condition is a LIVE-driver dichotomy whose
        # premise a dry run never satisfies.
        _master_dir = self.experiment.analysis_paths.analysis_dir
        _eff_mode = self.experiment.cfg_analysis.multi_sim_run_method
        # GATE-AND-CLAIM (see the twin in analysis.py::submit_workflow). The
        # claim is keyed on the MASTER analysis_dir and routed through the
        # master's own base builder, whose cfg_analysis is what
        # _max_plausible_job_lifetime_min must read. Dry-run suppression now
        # lives inside the helper, so the shape is equivalent for a rehearsal.
        _driver_id = self.experiment._workflow_builder._acquire_submit_driver_claim(
            _master_dir,
            workflow_submission_mode=_eff_mode,
            dry_run=dry_run,
            override_live_driver=overrides.live_driver,
        )
        try:
            result = self._workflow_builder.submit_workflow(
                mode=mode,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                wait_for_completion=wait_for_completion,
                dry_run=dry_run,
                verbose=verbose,
                overrides=overrides,
                report_formats=report_formats,
                extra_sbatch_args=extra_sbatch_args,
                snakemake_diagnostics=snakemake_diagnostics,
            )
        finally:
            if _driver_id is not None and _eff_mode == "local":
                _osent.remove_orchestrator_sentinel(_master_dir, _driver_id)

        if _driver_id is not None and _eff_mode != "local" and isinstance(result, dict):
            _osent.enrich_orchestrator_sentinel(
                _master_dir,
                driver_id=_driver_id,
                slurm_jobid=result.get("job_id"),
                tmux_session_name=result.get("session_name"),
            )

        return result

    def _invalidate_processing_log_for_member_ids(self, member_id_tokens: tuple[str, ...]) -> None:
        """Per-member_id dispatch for processing-log invalidation under
        ``override_force_rerun={"member_id": [...]}``.

        For each requested member_id, looks up its member and calls the
        per-member ``Analysis._invalidate_processing_log_for_force_rerun``
        with a ``scope="all"`` spec — which invalidates every scenario in
        that member. members are full Analysis instances and
        own their own scenario list (cf. CLAUDE.md Gotcha 11: "Sensitivity
        analysis members are full TRITONSWMM_analysis instances").

        Per cleanup-rerun-delete-redesign Phase 4 + B-mechanism.
        """
        from hhemt.workflow import ResolvedForceRerunSpec

        all_spec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="simulate")
        for member_id in member_id_tokens:
            analysis = self.members.get(member_id)
            if analysis is None:
                # _validate_force_rerun_targets already filtered unknown
                # member_ids; reaching here means the members dict is
                # out of sync with df_setup — surface loudly.
                raise RuntimeError(
                    f"members missing entry for member_id={member_id!r} after "
                    f"validation passed; df_setup/members are out of sync"
                )
            analysis._invalidate_processing_log_for_force_rerun(all_spec)

    def reprocess(
        self,
        start_with: Literal["process", "consolidate", "render"] = "consolidate",
        member_ids: list[str] | None = None,
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        verbose: bool = True,
        dry_run: bool = False,
        report_formats: list[str] | None = None,
        *,
        regenerate_existing: bool = False,
        delete_via_slurm: bool | None = None,
        override_force_rerun: ForceRerunValue | None = None,
    ) -> dict:
        """Master-level reprocess for sensitivity analyses.

        Invalidates per-member consolidate flags (subset via ``member_ids``
        or all members by default) plus the master consolidate flag,
        then emits a scoped master Snakefile via
        :meth:`SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content`
        and submits it via
        :meth:`SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow`.

        Unlike :meth:`TRITONSWMM_analysis.reprocess`, this method does NOT
        invoke the ``override_clear_raw`` orphan/abort gate (R12) — sensitivity
        master reprocess is a downstream-only refresh of consolidation +
        plotting + rendering against existing per-member sim outputs and does
        not need the in-flight reconciliation logic that the analysis-level
        ``override_clear_raw`` flow uses.

        Parameters
        ----------
        start_with
            Stage to re-fire from. ``"consolidate"`` (default) deletes per-member
            ``e_consolidate_member-{id}_complete.flag`` files and the master
            ``f_consolidate_experiment_complete.flag``, then re-runs the consolidate
            + experiment_consolidation + plot/render rule chain. ``"render"``
            invalidates only the report artifacts. ``"process"`` reconciles
            stale ``d_process`` flags against summary existence and re-emits the
            per-(member, event) rebuild rules (Gotcha 34/40); it does NOT collapse
            onto the ``"consolidate"`` Snakefile.
        member_ids
            Optional subset of member IDs (string-cast) to invalidate.
            When ``None`` (default), every member's per-member consolidate
            flag is invalidated. IDs not in ``members`` are silently
            ignored at the unlink call (``missing_ok=True``).
        execution_mode
            ``"auto"`` detects SLURM context; ``"local"`` / ``"slurm"`` force
            the mode.
        which
            ``"both"`` / ``"TRITON"`` / ``"SWMM"`` — threaded into the
            consolidate rule shells' ``--which`` flag.
        compression_level
            Compression level (0-9) for the consolidate rule shells.
        verbose
            If True, print progress messages.
        dry_run
            If True, runs ``snakemake --dry-run`` only.

        Returns
        -------
        dict
            Status dictionary from
            :meth:`SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow`.
        """
        # OE-A login-node preflight, sensitivity reprocess entry. Reachable DIRECTLY --
        # analysis.py's clear-raw refusal tells operators to call this method by name --
        # so a guard only at the dispatch site would be bypassed by following the
        # toolkit's own printed instruction. FIRST statement, above stamp_new_target and
        # above every destructive step, for the same reason as the non-sensitivity twin.
        _m = self.experiment
        assert_configs_visible_cross_node(
            _m._system.cfg_system,
            _m.cfg_analysis,
            {
                "--system-config": _m._system.system_config_yaml,
                "--analysis-config": _m.analysis_config_yaml,
                "--hpc-system-config": _m.hpc_system_config_yaml,
            },
            mode=execution_mode,
        )
        # Lazy-stamp _version.json at LAYOUT_VERSION (PI-1 pattern). Idempotent.
        from hhemt.version_migration import LAYOUT_VERSION
        from hhemt.version_migration.state import stamp_new_target

        stamp_new_target(self.experiment.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Force-rerun pre-delete (login-node responsibility). Per
        # cleanup-rerun-delete-redesign Phase 4 + R10. Resolves + validates +
        # deletes matched flags before Snakemake plans the reprocess DAG.
        # Skipped on dry_run — it deletes flags and clears per-scenario
        # processing-log records, both filesystem mutations the dry-run
        # no-destructive-mutation contract forbids.
        # Orchestrator-liveness CLAIM — hoisted ahead of every destructive step
        # below (this _apply_force_rerun, the inline flag/report/zarr
        # invalidation, the scoped reprocess-delete workflow, and the per-sub
        # processed-output deletion). Mirrors the non-sensitivity twin in
        # analysis.py::reprocess: until 2026-08-16 the gate lived inside
        # submit_reprocess_workflow, so a REFUSED reprocess had already
        # destroyed the report it was refusing to rebuild. The claim is keyed
        # on the MASTER analysis_dir and taken on the BASE builder, which is
        # where _orchestrator_liveness_gate lives (workflow.py:9799 reaches it
        # the same way). Returns None on dry_run.
        _reprocess_claim = self._workflow_builder._base_builder._acquire_reprocess_driver_claim(
            self.experiment.analysis_paths.analysis_dir,
            dry_run=dry_run,
        )

        if not dry_run:
            self.experiment._apply_force_rerun(override_force_rerun)

        # Resolve invalidation target set. ``None`` → all members; explicit
        # list → subset. String-cast preserves alignment with members dict
        # iteration keys regardless of source type (int / str / numpy scalar).
        if member_ids is None:
            targets = [str(member_id) for member_id in self.members.keys()]
        else:
            targets = [str(s) for s in member_ids]

        # Invalidate per-member consolidate flags + master flag. start_with controls
        # which flags get unlinked; per-member flag deletion is the entry point for
        # both "consolidate" and "process" (the master generator does not emit
        # process rules, so process invalidation is treated as consolidate
        # invalidation). "render" leaves consolidate flags intact and only
        # invalidates the rendered report artifact.
        from hhemt.du_sentinels import (
            decrement_scope_sentinel,
            restamp_parent_sentinels,
            sum_child_sentinels,
        )
        from hhemt.utils import fast_rmtree as _fast_rmtree

        experiment_dir = self.experiment.analysis_paths.analysis_dir
        status_dir = experiment_dir / "_status"
        if start_with in ("consolidate", "process"):
            for member_id in targets:
                # EXEMPT-DU: status-flag
                (status_dir / f"e_consolidate_member-{member_id}_complete.flag").unlink(missing_ok=True)
            # EXEMPT-DU: status-flag
            (status_dir / "f_consolidate_experiment_complete.flag").unlink(missing_ok=True)
            # R7 (D2 Option a) — consolidate-stage divergence preflight. Login-node
            # fail-fast that converts the SILENT-partial-master-tree hazard into a
            # clear ConfigurationError. On start_with="consolidate" the generator's
            # _sub_included_for_reprocess (Gotcha 37) SILENTLY EXCLUDES any sub whose
            # summaries are absent, and master consolidation's allow_incomplete=True
            # default (Gotcha 36) then assembles the master tree over the COMPLETED
            # subset and returns success — so a sub whose sim completed (c_run
            # present) but whose summary was deleted is silently dropped from
            # sensitivity_datatree.zarr with only a buried log warning. This is the
            # symmetric analogue of the process-stage
            # _assert_reprocess_rebuild_sources_present preflight. Fires ONLY on the
            # divergence signature (c_run present AND summary absent) so Gotcha 36's
            # tolerance for genuinely-never-ran subs is preserved. Read-only (no
            # flag/mtime touch — zero rerun-trigger surface).
            if start_with == "consolidate" and not dry_run:
                from hhemt.constants import sim_run_flag_per_member
                from hhemt.scenario import compute_event_id_slug
                from hhemt.workflow import _scenario_summaries_present

                _enabled = self.experiment._get_enabled_model_types()
                _diverged: list[str] = []
                for member_id in targets:
                    sub = self.members.get(member_id)
                    if sub is None:
                        continue
                    for _evt_iloc in sub.df_sims.index:
                        _evt = compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(_evt_iloc))
                        _c_run = experiment_dir / sim_run_flag_per_member(_enabled[0], str(member_id), _evt)
                        if _c_run.exists() and not _scenario_summaries_present(sub, _evt, _enabled):
                            _diverged.append(f"{self.member_prefix}{member_id}@evt-{_evt}")
                if _diverged:
                    from hhemt.exceptions import ConfigurationError

                    raise ConfigurationError(
                        field="start_with",
                        message=(
                            "reprocess(start_with='consolidate') cannot consolidate "
                            f"{sorted(_diverged)} — the simulation completed (c_run flag "
                            "present) but the per-scenario summary outputs are absent, so "
                            "the master consolidation would silently drop these "
                            "members from sensitivity_datatree.zarr. Re-run with "
                            "start_with='process', regenerate_existing=True to rebuild the "
                            "missing summaries first."
                        ),
                    )
            # FIX 2 — divergence self-heal runs on the process path on EVERY
            # route REGARDLESS of regenerate_existing (D2). Each member
            # reconciles its own d_process flags + per-model processing_log
            # against on-disk summary presence (D3): where a flag survives but
            # the enabled-model summary set is absent (the May-31 divergence:
            # 72 d_process flags vs 0 summary zarrs), unlink the flag + clear
            # the log so the master generator's emit gate (workflow.py:6810)
            # re-emits the per-(member,evt) process rule and _already_written
            # (Gotcha 28) lets it write. No-op for any sub whose summaries are
            # all present (healthy). members are full Analysis instances
            # and own their scenarios (Gotcha 11); the helper resolves the
            # per-member flag-token shape from each sub's is_experiment_member context.
            if start_with == "process" and not dry_run:
                for member_id in targets:
                    analysis = self.members.get(member_id)
                    if analysis is None:
                        continue
                    _reconciled = analysis._reconcile_stale_process_flags_against_summaries(
                        member_id=member_id, master_dir=experiment_dir
                    )
                    analysis._assert_reprocess_rebuild_sources_present(_reconciled)
            # FIX 2b — regenerate_existing rebuild parity (D1 Option A: logic in the
            # extracted free function so the fast unit test exercises it without
            # entering reprocess()'s destructive body). Gated on the
            # regenerate_existing arm; no-op otherwise.
            if start_with == "process" and regenerate_existing and not dry_run:
                _unlink_dprocess_flags_for_regenerate(targets, status_dir)
            # Report+plot deletion ALWAYS runs (toggle-independent) — the report
            # regenerates from the preserved zarr on the default path (FQ1 parity).
            _report_html = experiment_dir / "analysis_report.html"
            _report_zip = experiment_dir / "analysis_report.zip"
            # EXEMPT-DU: du-handled-by-decrement
            _report_html.unlink(missing_ok=True)
            # EXEMPT-DU: du-handled-by-decrement
            _report_zip.unlink(missing_ok=True)
            # FIX 3 — when regenerate_existing (and not dry_run), a LATER
            # deletion restamps the master _du.json anyway (SLURM route: the
            # reprocess-delete workflow's per-sub + master rules; in-process
            # route: compute_and_write_scope_sentinel(master, scope="analysis")
            # below). The early report-restamp here would otherwise force a
            # full-tree GPFS stat() walk on the login node before any SLURM
            # offload — the observed multi-minute stall. The default
            # (regenerate_existing=False) path still restamps (no later deletion).
            if not dry_run and not regenerate_existing:
                restamp_parent_sentinels(_report_html, analysis_dir=experiment_dir)  # PATTERN B (FIX 3 gate)
            # Consolidated-zarr deletion + batched DU restamp are the EXPENSIVE
            # GPFS work — gate behind regenerate_existing. Default path preserves
            # the zarrs (consolidate stays inert) and runs NO restamp walk.
            # R8 routing (D-scope Option C) — computed once from the MASTER
            # multi_sim_run_method. None auto-resolves to slurm-offload on HPC
            # modes (D6 refinement 1). When the SLURM path runs, ONE scoped
            # reprocess-delete workflow fans out per-sub (each sub's processed/
            # across events + its analysis_datatree.zarr) + a master rule for
            # sensitivity_datatree.zarr — replacing BOTH the in-process zarr
            # deletions AND the per-sub processed-output delegation below.
            _hpc = self.experiment.cfg_analysis.multi_sim_run_method in (
                "batch_job",
                "1_job_many_srun_tasks",
            )
            _resolved_delete_via_slurm = _hpc if delete_via_slurm is None else delete_via_slurm
            # An explicit execution_mode="local" from the caller must not be silently
            # overridden by the config's multi_sim_run_method: before this guard, a
            # caller forcing local execution still had the scoped delete offloaded to
            # SLURM (2026-07-20). execution_mode="auto" preserves the prior routing.
            route_delete_via_slurm = (
                regenerate_existing
                and _resolved_delete_via_slurm
                and not dry_run
                and _hpc
                and execution_mode != "local"
            )
            if route_delete_via_slurm:
                # The result was previously DISCARDED, so a delete workflow that
                # failed (or ran zero rules) was indistinguishable from success and
                # reprocess fell through to submit the consolidate workflow against
                # trees it believed were gone. Observed 2026-07-20: the delete
                # workflow failed 112 times, raised, and the exception was thrown
                # away. regenerate_existing invalidates by DELETION; if the deletion
                # did not happen the flag is a lie, and the correct response is to
                # fail loudly rather than consolidate stale.
                _del_result = self._workflow_builder._base_builder.submit_reprocess_delete_workflow(
                    start_with=start_with,
                    override_in_flight=False,
                )
                if not _del_result.get("success", False):
                    from hhemt.exceptions import WorkflowError

                    raise WorkflowError(
                        phase="reprocess-delete",
                        return_code=_del_result.get("returncode", -1),
                        stderr=(
                            "regenerate_existing=True requested a scoped reprocess-delete "
                            "workflow, but it did not complete successfully. The consolidated "
                            "trees were NOT invalidated, so continuing would silently "
                            "re-consolidate stale artifacts. See "
                            f"{_del_result.get('snakemake_logfile')} for the workflow-level log. "
                            "Per-rule stderr for each failed delete job is written to "
                            "{analysis_dir}/logs/delete_reprocess/<rule>.log — this capture is "
                            "executor-independent and is present whether the rules ran locally "
                            "or on SLURM. If the rules DID reach SLURM, per-job logs may also "
                            "exist under {analysis_dir}/.snakemake_reprocess_delete/.snakemake/"
                            "slurm_logs/ (retained on failure, deleted on success); that "
                            "directory is absent when submission itself failed and no job was "
                            "ever created. Re-run with delete_via_slurm=False to delete "
                            "in-process instead."
                        ),
                    )
            elif regenerate_existing and not dry_run:
                # In-process path (local / delete_via_slurm=False) — per-sub +
                # master zarr deletion, plus the Phase-3 per-sub processed-output
                # delegation (FQ2; members own their scenarios). The
                # delegated helper internally guards on
                # `start_with == "process"` (analysis.py
                # _delete_processed_outputs_for_reprocess, ~L2979), so a
                # consolidate/render reprocess PRESERVES each sub's processed/
                # (the rebuild source consolidate reads from) — only a
                # process-stage reprocess deletes it. Do NOT inline the
                # processed/ rmtree here without that guard: dropping it makes a
                # consolidate-stage regenerate delete the rebuild source and the
                # consolidate Snakemake step then fails (FIX-1 Phase-1 regression,
                # 2026-05-31). For the process stage the per-model LOG-clear
                # already ran above (FIX 1, hunk 2a) on both routes; the helper's
                # idempotent re-clear here is harmless.
                affected_sub_dirs: set = set()
                for member_id in targets:
                    analysis = self.members.get(member_id)
                    if analysis is None:
                        continue
                    # FQ2 processed-output deletion (Phase 3) — delegate to the
                    # member's own helper (members own their scenarios;
                    # the helper's start_with guard protects consolidate/render).
                    analysis._delete_processed_outputs_for_reprocess(
                        start_with, regenerate_existing=regenerate_existing, dry_run=dry_run
                    )
                    _sub_zarr = analysis.analysis_paths.analysis_datatree_zarr
                    if _sub_zarr is not None and _sub_zarr.exists():
                        _fast_rmtree(_sub_zarr, analysis_dir=None)  # batched-restamp
                        affected_sub_dirs.add(analysis.analysis_paths.analysis_dir)
                _master_zarr = self.analysis_paths.sensitivity_datatree_zarr
                if _master_zarr is not None and _master_zarr.exists():
                    _fast_rmtree(_master_zarr, analysis_dir=None)  # batched-restamp
                for _sub_dir in affected_sub_dirs:
                    sum_child_sentinels(_sub_dir, scope="member", child_scope_dirs=["sims"])
                sum_child_sentinels(experiment_dir, scope="analysis", child_scope_dirs=["members", "sims"])
        elif start_with == "render":
            # No _status flag for render — re-fire by deleting the report
            # artifacts so Snakemake's mtime trigger sees the output as absent.
            # The report-artifact unlink is the flag-equivalent trigger, so it
            # runs even on dry_run (see D6); only the DU restamp is gated.
            _report_html = experiment_dir / "analysis_report.html"
            _report_zip = experiment_dir / "analysis_report.zip"
            # D3 — capture sizes BEFORE unlink so the O(1) decrement has the bytes.
            _html_bytes = _report_html.stat().st_size if _report_html.exists() else 0
            _zip_bytes = _report_zip.stat().st_size if _report_zip.exists() else 0
            # EXEMPT-DU: du-handled-by-decrement
            _report_html.unlink(missing_ok=True)
            # EXEMPT-DU: du-handled-by-decrement
            _report_zip.unlink(missing_ok=True)
            if not dry_run:
                # D3 — O(1) decrement of the two report children (no plots on the
                # sensitivity render arm). Mirrors the non-sensitivity render path;
                # routes through write_du_sentinel (compare-and-write mtime invariant).
                _child_deltas: dict[str, int] = {}
                if _html_bytes:
                    _child_deltas["analysis_report.html"] = _html_bytes
                if _zip_bytes:
                    _child_deltas["analysis_report.zip"] = _zip_bytes
                if _child_deltas:
                    decrement_scope_sentinel(experiment_dir, scope="analysis", child_deltas=_child_deltas)
        else:
            raise ValueError(f"start_with must be one of 'process', 'consolidate', 'render'; got {start_with!r}")

        # Delegate to the sensitivity workflow builder.
        return self._workflow_builder.submit_reprocess_workflow(
            start_with=start_with,
            execution_mode=execution_mode,
            which=which,
            compression_level=compression_level,
            claimed_driver_id=_reprocess_claim,
            dry_run=dry_run,
            verbose=verbose,
            report_formats=report_formats,
        )

    def delete(
        self,
        override_in_flight: bool = False,
        *,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> None:
        """Distributed delete workflow for the sensitivity master analysis.

        Refuses by default when ``_status/_submitted/*.json`` sentinels
        indicate live SLURM jobs. Pass ``override_in_flight=True`` to bypass
        the guard.

        Per cleanup-rerun-delete-redesign Phase 2 (D-DeleteSentinelInteraction
        + D-DeleteBoundary resolutions) and distributed-delete-and-du-
        recording Phase 3 (SLURM lift; ``override_multi_sim_run_method``
        mirrors the run-mode override pattern).
        """
        from hhemt.utils import fast_rmtree

        analysis_dir = self.experiment.analysis_paths.analysis_dir

        # 1. Clear any stale sentinels from a prior failed delete attempt.
        stale_dir = analysis_dir / "_status" / "_deleting"
        if stale_dir.exists():
            # EXEMPT-DU: status-dir-cleanup
            fast_rmtree(stale_dir)

        # 2. Submit the distributed sensitivity-delete workflow. Guards run
        # inside the builder; orchestrator does not invoke _pre_delete_guards
        # directly.
        self._workflow_builder.submit_delete_workflow_sensitivity(
            override_in_flight=override_in_flight,
            override_multi_sim_run_method=override_multi_sim_run_method,
        )

        # 3. Verify all expected sentinels present; remove analysis_dir atomically.
        expected = self._enumerate_expected_delete_sentinels()
        deleting_dir = analysis_dir / "_status" / "_deleting"
        actual = set(deleting_dir.glob("*.flag")) if deleting_dir.exists() else set()
        missing = expected - actual
        if missing:
            print(
                f"[delete] {len(missing)} per-member-rule sentinels missing — preserving analysis_dir for debugging.",
                flush=True,
            )
            print(f"[delete] missing: {sorted(p.name for p in missing)}", flush=True)
            return
        print(
            f"[delete] all {len(expected)} per-member-rule sentinels present — removing analysis_dir.",
            flush=True,
        )
        # EXEMPT-DU: full-analysis-root-wipe
        fast_rmtree(analysis_dir)

    def _enumerate_expected_delete_sentinels(self) -> set[Path]:
        """Compute the set of ``_status/_deleting/*.flag`` paths the
        sensitivity delete workflow will produce on full success.

        One per member row in ``self.df_setup.index`` plus one for the
        analysis-level consolidation rule.
        """
        delete_dir = self.experiment.analysis_paths.analysis_dir / "_status" / "_deleting"
        expected = {delete_dir / "analysis_consolidation.flag"}
        for member_id in self.df_setup.index.astype(str):
            expected.add(delete_dir / f"member_member-{member_id}.flag")
        return expected

    def render_report(self, format: Literal["html", "zip"] = "zip", *, reprocess: bool = False) -> "Path":
        """Render the master report for the sensitivity analysis.

        Idempotent: invokes ``snakemake --report`` against the master Snakefile
        without re-executing any rules. Renders only the master-level report;
        per-member reports are not generated (R13).

        Parameters
        ----------
        format : Literal["html", "zip"], default "zip"
            Output format. ``"html"`` produces a single self-contained
            ``analysis_report.html`` with all figures inlined as base64, plus
            React-bundle post-process surgery (title, navbar, sidebar order,
            click-to-figure shim). ``"zip"`` produces ``analysis_report.zip``
            containing the unbundled report tree (separate HTML + assets);
            no post-process surgery is applied (the zip layout differs from
            the single-file HTML).
        reprocess : bool, default False
            When ``True``, render against ``Snakefile.reprocess`` (the filtered
            reprocess DAG) instead of the production ``Snakefile``, so the
            ``snakemake --report`` step only expects the figures the reprocess
            DAG built. Keyword-only; set by the reprocess ``render_report`` rule
            shell. Default ``False`` keeps the production render path
            byte-identical.
        """
        import subprocess
        import sys

        from .exceptions import WorkflowError
        from .workflow import _assert_snakefile_package_current

        master_dir = self.experiment.analysis_paths.analysis_dir
        snakefile_name = "Snakefile.reprocess" if reprocess else "Snakefile"
        snakefile = master_dir / snakefile_name
        _assert_snakefile_package_current(snakefile)
        out = master_dir / f"analysis_report.{format}"
        css_path = master_dir / "report" / "report.css"
        # Brand-theme resolution (ADR-7 layer 2) — symmetric to
        # analysis.render_report. The render_report_runner path builds a FRESH
        # instance without _brand_theme; resolve once via getattr-fallback to
        # serve BOTH the CSS emit and the navbar surgery (D-6; plan-review SE Flag 1).
        from .config.brand_theme import DEFAULT_BRAND_THEME
        from .config.loaders import load_brand_theme
        from .workflow import _brand_theme_css_map

        _theme = getattr(self, "_brand_theme", None)
        if _theme is None:
            _theme = (
                load_brand_theme(self.cfg_analysis.brand_theme)
                if self.cfg_analysis.brand_theme is not None
                else DEFAULT_BRAND_THEME
            )
        # Re-emit report artifacts from package resources so render_report
        # picks up edits made to the source-tree report_templates/.
        _emit_report_artifacts(master_dir, brand_theme=_brand_theme_css_map(_theme))
        cmd = [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--directory",
            str(master_dir),
            "--report",
            str(out),
            "--report-stylesheet",
            str(css_path),
            "--cores",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-50:])
            raise WorkflowError(
                phase="render_report",
                return_code=result.returncode,
                stderr=f"snakemake --report exit {result.returncode}; last 50 lines:\n{tail}",
            )
        # Apply React-bundle post-process surgery (title, navbar, sort order,
        # placeholder category, showCategory auto-pop, row-click delegate).
        # Both formats need the surgery:
        #  - HTML: edit the single rendered file in place.
        #  - Zip: extract, edit `analysis_report/report.html` inside, re-zip.
        # Without surgery in zip mode, the eye-icon-hiding CSS in report.css
        # leaves figure tables with no clickable affordance (the JS click
        # delegate that makes rows clickable lives only in the surgery).
        from .report_renderers._react_surgery import (
            apply_post_process_surgery,
            apply_post_process_surgery_to_zip,
        )

        # Navbar upper-left brand text: brand_theme.upper_left_text (ADR-7),
        # defaulting to analysis_id when None (D-6). _theme is resolved above.
        _navbar = _theme.upper_left_text or self.cfg_analysis.analysis_id
        # Resolve the active set's category_order. render_report() is dominantly
        # invoked from render_report_runner.main() on a FRESH analysis that never
        # called run() (see the _brand_theme getattr-fallback above for the
        # identical hazard), so self._active_reporting_set may not exist. getattr-
        # fallback to a config-only resolution (no CSV cross-validation at render
        # time) mirroring the _theme fallback above. Never let the bare attribute
        # AttributeError be swallowed by the surrounding `except Exception: pass`.
        _active_set = getattr(self, "_active_reporting_set", None)
        if _active_set is None:
            # render-without-run() fallback. Fail SOFT (SE F-I-3): the render path
            # bypasses validate_active_reporting_set, so a stale/unknown
            # reporting_set would raise here and surface as an opaque Snakemake
            # rule failure. Degrade to the historical "default" sidebar order + a
            # one-line warning instead of crashing the render rule.
            import logging

            from .config.report import resolve_active_reporting_set_name
            from .report_renderers._reporting_sets import get_reporting_set

            try:
                _cfg_report = getattr(self, "_cfg_report", None)
                if _cfg_report is None:
                    _cfg_report = self.cfg_analysis.report
                _set_name = resolve_active_reporting_set_name(
                    _cfg_report,
                    is_sensitivity=self.cfg_analysis.toggle_sensitivity_analysis,
                )
                _active_set = get_reporting_set(_set_name)
            except Exception as _e:
                logging.getLogger(__name__).warning(
                    "render-path reporting_set resolution failed (%s); falling back to 'default' category order",
                    _e,
                )
                _active_set = get_reporting_set("default")
        _category_order = list(_active_set.category_order)
        # S4: resolve member_id card names to derived compute-config labels. Threaded to
        # BOTH branches -- the html and the zip carry the same card names, and
        # resolving one alone would ship a divergence between two delivered artifacts.
        from .report_plot_ids import event_labels_from_status, member_labels_from_status

        _member_labels = member_labels_from_status(self.analysis_paths.analysis_dir)
        _event_labels = event_labels_from_status(self.analysis_paths.analysis_dir)
        try:
            if format == "html":
                out.write_text(
                    apply_post_process_surgery(
                        out.read_text(),
                        navbar_text=_navbar,
                        category_order=_category_order,
                        member_labels=_member_labels,
                        event_labels=_event_labels,
                    )
                )
            else:
                apply_post_process_surgery_to_zip(
                    out,
                    navbar_text=_navbar,
                    category_order=_category_order,
                    member_labels=_member_labels,
                    event_labels=_event_labels,
                )
        except Exception:
            pass
        if format != "html":
            return out
        out_html = out
        # Snap-confined browsers (Ubuntu Firefox snap) cannot read files under
        # ~/.cache/. If the rendered report lands there, surface a one-line
        # workaround so the user does not hit "Access to the file was denied".
        try:
            if "/.cache/" in str(out_html):
                print(
                    f"[render_report] {out_html}\n"
                    f"[render_report] Note: snap-confined browsers cannot read ~/.cache; "
                    f"copy to ~/Downloads to view: cp {out_html} ~/Downloads/",
                    flush=True,
                )
        except Exception:
            pass
        return out_html

    # Conforms to hhemt.bundle._protocol.BundleableAnalysis
    # via duck typing — attributes delegated to self.experiment in
    # __init__ (lines 91-94).
    def bundle_report_data(
        self,
        output_path: "Path | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> "Path":
        """Emit a portable render bundle for the sensitivity master analysis.

        Opt-in only — NEVER invoked from analysis.run() or
        submit_workflow(). The bundle includes the sensitivity master's
        consolidated outputs plus the union of source paths declared by
        every renderer in the master's render_report(), including per-sim
        renderers wildcarded over (member_id, event_id).

        Args:
            output_path: Optional target path for the bundle tar.
            container_defs: ADR-19 (multi-SIF) — one Apptainer .def per distinct arch to
                carry. Required (repeatable) for a container-mode analysis (nothing in the
                config names one); ignored for native.

        Returns:
            Path to the emitted bundle tar.
        """
        from hhemt.bundle import emit_bundle

        return emit_bundle(self, output_path, container_defs=container_defs)

    def reprex_bundle(
        self,
        output_path: "Path | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> "Path":
        """Emit a reprex-ready Workflow-Run-Crate bundle for the sensitivity master and
        return its extracted directory root (ADR-10, D3).

        Parity peer of ``bundle_report_data()`` — the sensitivity master is the PRIMARY
        reprex surface (the ``(member_id, column)`` problem-pair emission is intrinsically a
        sensitivity concept). ``emit_bundle`` already carries the reprex runnable-template
        set + WRC crate (Phase 2); this facade extracts the emitted zip to a sibling
        directory so the round-trip consumes a directory root directly
        (``Bundle.from_directory(...).reprex(...)``). Opt-in only.

        Args:
            container_defs: ADR-19 (multi-SIF) — one Apptainer .def per distinct arch to
                carry. Required (repeatable) for a container-mode analysis (nothing in the
                config names one); ignored for native.

        Returns:
            Path to the extracted reprex-bundle directory.
        """
        from hhemt.bundle import emit_bundle
        from hhemt.bundle._reprex import extract_reprex_bundle

        return extract_reprex_bundle(emit_bundle(self, output_path, container_defs=container_defs))

    def publish(
        self,
        target: "Literal['hydroshare', 'zenodo']",
        *,
        override_dataset_license: "Literal['CC0-1.0', 'CC-BY-NC-4.0'] | None" = None,
        software_doi: "str | None" = None,
    ) -> dict:
        """Deposit the sensitivity MASTER tree to a DOI-minting repo (C6, ADR-11).

        Opt-in only — NEVER invoked from run()/submit_workflow(), mirroring
        render_report()/bundle_report_data(). Deposits the master
        sensitivity_datatree.zarr + master-rooted ro-crate sidecar; the license is
        read from the emitted crate. Returns {"target","data_doi","software_doi","record_url"}.
        """
        from hhemt.publishing import publish_analysis

        return publish_analysis(
            self.experiment,
            target=target,
            override_dataset_license=override_dataset_license,
            software_doi=software_doi,
            consolidated_zarr_relpath="sensitivity_datatree.zarr",
        )

    def publish_reprex_bundle(
        self,
        target: "Literal['hydroshare', 'zenodo']",
        *,
        exclude_config: "Path | None" = None,
        override_dataset_license: "Literal['CC0-1.0', 'CC-BY-NC-4.0'] | None" = None,
        software_doi: "str | None" = None,
        container_defs: "list[Path] | None" = None,
    ) -> dict:
        """Deposit the RUNNABLE reprex bundle for the sensitivity master (D6, R5).

        Sensitivity parallel of ``TRITONSWMM_analysis.publish_reprex_bundle`` — the emit
        half of the DOI round-trip. Opt-in only; NEVER invoked from ``run()``.

        Args:
            exclude_config: The ADR-20 governed opt-out (see the analysis-tier facade).
                Omit it and the deposited bundle is SELF-CONTAINED.
        """
        return self.experiment.publish_reprex_bundle(
            target,
            exclude_config=exclude_config,
            override_dataset_license=override_dataset_license,
            software_doi=software_doi,
            container_defs=container_defs,
        )

    def run_all_sims(
        self,
        pickup_where_leftoff,
        concurrent: bool = False,
        process_outputs_after_sim_completion: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose=False,
    ):
        """Execute every simulation across every member (workflow phase 2b).

        Parameters
        ----------
        pickup_where_leftoff : bool
            Resume from existing completion state instead of re-running
            simulations already recorded as complete.
        concurrent : bool
            Run simulations in parallel. Concurrency is bounded by the resolved
            execution strategy — CPU, GPU and memory limits on a workstation,
            and the SLURM allocation's own limits on a cluster.
        process_outputs_after_sim_completion : bool
            Convert raw solver output to Zarr/NetCDF as each simulation
            finishes, rather than in a separate later pass.
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        compression_level : int
            Zarr/NetCDF compression level for any processing done here.

        Notes
        -----
        Completion is detected from solver LOG MARKERS, not from exit codes — a
        simulation can exit 0 having failed partway. A run that reports complete
        with an implausibly short elapsed time is the signature to check.
        """
        if concurrent:
            raise RuntimeError(
                "Running sensitivity analyses concurrently requires"
                "more intelligent handling of compute resource availability"
                "tracking. Update run_simulations_concurrently function"
                "in analysis.py to enable this."
            )
            launch_functions = []
            for _analysis_iloc, analysis in self.members.items():
                launch_functions += analysis._create_launchable_sims(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                )
            self.experiment.run_simulations_concurrently(launch_functions, verbose=verbose)
        else:
            for _analysis_iloc, analysis in self.members.items():
                analysis.run_sims_in_sequence(
                    pickup_where_leftoff=pickup_where_leftoff,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    override_clear_raw=override_clear_raw,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_experiment_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Convert raw solver output to Zarr/NetCDF across all members (phase 2c).

        Parameters
        ----------
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        override_clear_raw : ClearRawValue or None
            Override ``analysis_config.clear_raw`` for this invocation only.
            ``None`` reads the config. Per the override-prefix convention this
            never falls through silently: if both are absent, it raises.
        verbose : bool
            Emit per-scenario progress.
        compression_level : int
            Zarr/NetCDF compression level.
        """
        scenario_timeseries_processing_launchers = []
        for _analysis_iloc, analysis in self.members.items():
            launchers = analysis.retrieve_scenario_timeseries_processing_launchers(
                which=which,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.experiment.run_python_functions_concurrently(scenario_timeseries_processing_launchers)
        return

    def _consolidate_outputs_in_each_analysis(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for _analysis_iloc, analysis in self.members.items():
            analysis._consolidate_analysis_outputs(
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_experiment_log()
        return

    @property
    def TRITON_analyses_outputs_consolidated(self):
        """True when every member has a completed TRITON-side summary.

        Which flag is checked depends on the enabled model: the coupled
        TRITON-SWMM summary when ``toggle_tritonswmm_model`` is set, otherwise
        the TRITON-only summary. An aggregate over all members — one
        incomplete member makes this False.
        """
        cfg_sys = self.experiment._system.cfg_system
        success = True
        for _analysis_iloc, analysis in self.members.items():
            if cfg_sys.toggle_tritonswmm_model:
                success = success and analysis._tritonswmm_triton_analysis_summary_created
            elif cfg_sys.toggle_triton_model:
                success = success and analysis._triton_only_analysis_summary_created
        return success

    @property
    def SWMM_analyses_outputs_consolidated(self):
        """True when every member has completed SWMM node AND link summaries.

        Both must be present; a member with nodes but not links reads
        False. An aggregate over all members.
        """
        cfg_sys = self.experiment._system.cfg_system
        node_success = True
        link_success = True
        for _analysis_iloc, analysis in self.members.items():
            if cfg_sys.toggle_tritonswmm_model:
                node_success = node_success and analysis._tritonswmm_node_analysis_summary_created
                link_success = link_success and analysis._tritonswmm_link_analysis_summary_created
            elif cfg_sys.toggle_swmm_model:
                node_success = node_success and analysis._swmm_only_node_analysis_summary_created
                link_success = link_success and analysis._swmm_only_link_analysis_summary_created
        return node_success and link_success

    def consolidate_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = True,
        compression_level: int = 5,
    ):
        """Consolidate per-scenario summaries into analysis-level datasets (phase 3).

        Calls :meth:`create_analysis_summaries` for each member, then
        assembles the master ``sensitivity_datatree.zarr``.

        Parameters
        ----------
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        verbose : bool
            Emit per-member progress.
        compression_level : int
            Zarr/NetCDF compression level.

        Notes
        -----
        members whose per-scenario summaries are absent are SKIPPED with a
        warning rather than raising, so the master tree assembles over the
        completed subset. The root ``parameters`` dataset still lists every
        DEFINED member (the experiment definition); only completed ones
        appear as tree nodes (the realized results). A partial master tree is
        therefore an expected state, not a defect.
        """
        self.create_analysis_summaries(
            which=which,
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_analysis_outputs(
            which=which,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_analysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Consolidate members into a hierarchical sensitivity DataTree zarr.

        Replaces the previous per-mode flat ``xr.concat`` path. Each member
        first builds its per-analysis DataTree (``analysis_datatree.zarr``); then
        the master assembles all members into a single
        ``sensitivity_datatree.zarr`` at the master analysis dir.
        """
        self.consolidate_sensitivity_datatree(
            compression_level=compression_level,
            verbose=verbose,
        )
        return

    def build_sensitivity_datatree(self) -> "xr.DataTree":
        """Assemble the master sensitivity DataTree lazily from member trees.

        Each member's consolidated DataTree (``analysis_datatree.zarr``) is
        opened lazily and grafted under a ``member_{member_id}/`` subtree. Sensitivity
        parameters for each member are attached as ``.attrs`` on the
        ``member_{member_id}`` node. A parameter-summary Dataset is written at the root
        under ``parameters`` for tabular queries.
        """
        tree_dict: dict[str, xr.Dataset] = {}

        tree_dict["/"] = xr.Dataset(
            attrs={
                "Conventions": "CF-1.13",
                "title": "TRITON-SWMM sensitivity analysis results",
                "analysis_id": str(self.experiment.cfg_analysis.analysis_id),
                "output_creation_date": current_datetime_string(),
            }
        )

        tree_dict["parameters"] = xr.Dataset.from_dataframe(self.df_setup)

        for member_id, analysis in self.members.items():
            node_name = f"{self.member_prefix}{member_id}"
            # Refresh the member's in-memory log from disk before reading its
            # consolidation state: open_datatree() gates on the in-memory
            # datatree_consolidation_complete flag, which a run (often in another
            # process) sets on disk but not in this long-lived sub object.
            # Previously the sensitivity aggregators' _update_log() refreshed these
            # sub logs as a side effect; that call was dropped in the
            # log-write-race-fix compute-on-read change, so refresh explicitly here
            # at the cross-analysis read site (read-only observer; safe).
            analysis._refresh_log()
            try:
                sub_tree = analysis.process.open_datatree()
            except ValueError:
                continue

            for path, node in sub_tree.subtree_with_keys:
                if not node.has_data:
                    continue
                rel = path.lstrip("/")
                if not rel:
                    continue
                tree_dict[f"{node_name}/{rel}"] = node.dataset

            setup_row = self.df_setup.loc[member_id]
            attrs = {k: _to_native_attr(v) for k, v in setup_row.to_dict().items()}
            attrs["member_id"] = str(member_id)
            tree_dict[node_name] = xr.Dataset(attrs=attrs)

        tree = xr.DataTree.from_dict(tree_dict)
        apply_global_attributes(tree, analysis_id=str(self.experiment.cfg_analysis.analysis_id))

        # ADR-15 Phase 1: the per-event hhemt_producing_sha/version coordinates ride
        # up automatically via the verbatim node.dataset graft above (no new
        # transmission code). Re-derive the MASTER-level scalar fast-path here by
        # scanning the grafted sub-nodes for master-wide uniformity (uniform across
        # ALL members' events -> scalar; else absent + divergent breadcrumb).
        from hhemt.cf_conventions import apply_producing_stamp

        _sha_vals: list[str] = []
        _semver_vals: list[str] = []
        for _key, _ds in tree_dict.items():
            _coords = getattr(_ds, "coords", {})
            if "hhemt_producing_sha" in _coords:
                _sha_vals.extend(str(v) for v in _ds["hhemt_producing_sha"].values.tolist())
            if "hhemt_producing_version" in _coords:
                _semver_vals.extend(str(v) for v in _ds["hhemt_producing_version"].values.tolist())
        apply_producing_stamp(tree, _sha_vals, _semver_vals)
        return tree

    def consolidate_sensitivity_datatree(
        self,
        compression_level: int = 5,
        verbose: bool = False,
        allow_incomplete: bool = True,
    ) -> Path:
        """Build and write the master sensitivity DataTree zarr.

        Ensures each member has its own consolidated ``analysis_datatree.zarr``
        first, then assembles them into a single hierarchical store at
        ``sensitivity_datatree.zarr``.

        When ``allow_incomplete`` is True (the default), members whose
        per-scenario summaries are not all present on disk are SKIPPED (with a
        logged warning naming the ``member_id``) instead of crashing the master
        assembly with ``FileNotFoundError``/``ValueError``. The skip is
        whole-member because ``_retrieve_combined_output`` concatenates
        per-scenario summaries along ``event_iloc`` and is all-or-nothing per
        member. Pass ``allow_incomplete=False`` to restore fail-fast.

        NOTE (Decision D2, 2026-06-02, "for now"): the default is ``True`` so
        reprocess AND canonical runs tolerate partial-completion sensitivity
        suites by default. To restore strict fail-fast on canonical runs, flip
        this default back to ``False``.
        """
        fname_out = self.analysis_paths.sensitivity_datatree_zarr
        if fname_out is None:
            raise ValueError("sensitivity_datatree_zarr path is not configured on AnalysisPaths.")

        # Per D5/R8: bare .exists() is an unreliable completion signal — a
        # present-but-corrupt sensitivity_datatree.zarr (a write that crashed
        # mid-stream) .exists() as True, and would be returned as a healthy
        # result. Align with the master log's canonical signal: "already
        # consolidated" iff it exists AND sensitivity_datatree_consolidation_complete
        # is True (set only on a successful full write below). Present-but-incomplete
        # falls through to a clean rebuild. This is the master-sensitivity analogue
        # of consolidate_to_datatree's D5 log-keyed early-return; do NOT touch the
        # read-path .exists() guard in open_sensitivity_datatree (a read-path
        # existence check is correct).
        self.experiment._refresh_log()
        _log_complete = (
            hasattr(self.experiment.log, "sensitivity_datatree_consolidation_complete")
            and self.experiment.log.sensitivity_datatree_consolidation_complete.get() is True
        )
        # The master tree's inputs ARE the sub trees, so its staleness is the
        # disjunction of theirs: recompute each sub's current fingerprint and compare
        # it to what that sub actually has stamped. Any sub that is stale (or has no
        # stamp) makes the master stale, which is what carries a newly-added per-sub
        # node (e.g. the timeseries nodes) up through build_sensitivity_datatree's
        # subtree_with_keys graft. No separate master-side toggle read is needed.
        _subs_stale = False
        for _member_id, _sub in self.members.items():
            _sub._refresh_log()
            _cur = _sub.process._consolidation_inputs_fingerprint()
            _stamped = (
                _sub.log.consolidation_inputs_fingerprint.get()
                if hasattr(_sub.log, "consolidation_inputs_fingerprint")
                else None
            )
            if _stamped != _cur:
                _subs_stale = True
                break
        if fname_out.exists() and _log_complete and _subs_stale:
            from hhemt.utils import fast_rmtree

            fast_rmtree(fname_out, analysis_dir=self.analysis_paths.analysis_dir)
            if verbose:
                print(
                    f"Sensitivity DataTree zarr present at {fname_out} but at least one "
                    "member's consolidation inputs changed — rebuilding."
                )
        elif fname_out.exists() and _log_complete:
            if verbose:
                print(f"Sensitivity DataTree zarr already present at {fname_out} and log complete. Not overwriting.")
            # Ensure the master analysis-scope DU sentinel exists even on the
            # already-consolidated early-return path. This materializes the
            # sentinel on trees consolidated before this write site existed, and
            # is cheap/idempotent via compare-and-write.
            self._write_master_du_sentinel()
            return fname_out
        if fname_out.exists() and not _log_complete:
            from hhemt.utils import fast_rmtree

            fast_rmtree(fname_out, analysis_dir=self.analysis_paths.analysis_dir)
            if verbose:
                print(
                    f"Sensitivity DataTree zarr present at {fname_out} but log incomplete — "
                    "rebuilding (treating as corrupt)."
                )

        # Ensure each member has its analysis_datatree.zarr built.
        for member_id, analysis in self.members.items():
            sub_path = analysis.analysis_paths.analysis_datatree_zarr
            if sub_path is None:
                continue
            # Gate on the POSITIVE completion marker, not bare .exists(): a sub
            # whose zarr is on disk but whose datatree_consolidation_complete flag
            # is null/False (set by a consolidate job in another process but not
            # reflected in this long-lived sub's in-memory log) must be
            # (re)consolidated — otherwise build_sensitivity_datatree's
            # open_datatree() raises ValueError and silently drops the sub.
            # consolidate_to_datatree is idempotent on an already-complete sub
            # (its own _log_complete early-return), so this self-heals.
            analysis._refresh_log()
            # Always delegate: consolidate_to_datatree owns the single completeness
            # AND staleness decision (its own early-return is idempotent on a healthy,
            # current tree). The former `if not (sub_path.exists() and sub_complete)`
            # pre-gate was a SECOND, weaker gate that skipped a present-and-complete-
            # but-STALE sub before the real guard could notice — the exact shape that
            # kept the timeseries nodes out of the tree.
            #
            # The `sub_complete` computation that fed that pre-gate is DELETED rather
            # than retained: the log-refresh side effect belongs to the
            # `_refresh_log()` call above (a separate statement), and nothing else
            # read the variable, so keeping it would be dead code carrying an F841.
            if True:
                try:
                    analysis.process.consolidate_to_datatree(
                        compression_level=compression_level,
                        verbose=verbose,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    if not allow_incomplete:
                        raise
                    print(
                        f"[sensitivity-consolidate] Skipping incomplete member "
                        f"{self.member_prefix}{member_id} under allow_incomplete=True: {exc}",
                        flush=True,
                    )
                    continue

        tree = self.build_sensitivity_datatree()
        from hhemt.cf_conventions import apply_provenance_core
        from hhemt.metadata import write_rocrate_sidecar
        from hhemt.provenance import emit_provenance

        _sub_relpaths = [f"members/member_{member_id}/analysis_datatree.zarr" for member_id in self.members]
        _emitted_vars = {str(v) for _n in tree.subtree for v in _n.dataset.data_vars}
        _core_json, _graph_json = emit_provenance(
            self.experiment,
            consolidated_zarr_relpath="sensitivity_datatree.zarr",
            sub_dataset_relpaths=_sub_relpaths,
            with_run_units=False,
            emitted_vars=_emitted_vars,
        )
        apply_provenance_core(tree, core_json_str=_core_json)

        # TRITON provenance (D2/R5): stamp the producing-TRITON sha + coupled-resume-fix
        # ancestry onto the sensitivity-MASTER tree root as plain attrs. This is a
        # SEPARATE wiring site from consolidate_to_datatree: check_coupled_resume_validity's
        # reader resolves sensitivity_datatree.zarr for a sensitivity master, so omitting
        # this site would leave the reader permanently INDETERMINATE on the primary
        # experiment shape (the c2c3 dual-wiring seam).
        from hhemt.processing_analysis import _stamp_coupled_resume_evidence, _stamp_triton_provenance

        _stamp_triton_provenance(tree, self.experiment)
        _stamp_coupled_resume_evidence(tree, self.experiment)
        write_datatree_zarr(tree, fname_out, compression_level=compression_level)
        write_rocrate_sidecar(self.experiment.analysis_paths.analysis_dir, graph_json=_graph_json)

        self.experiment._refresh_log()
        if hasattr(self.experiment.log, "sensitivity_datatree_consolidation_complete"):
            self.experiment.log.sensitivity_datatree_consolidation_complete.set(True)

        if verbose:
            print(f"Wrote sensitivity DataTree zarr to {fname_out}")
        self._write_master_du_sentinel()
        return fname_out

    def _write_master_du_sentinel(self) -> None:
        """Write the master analysis-scope ``_du.json`` DU sentinel.

        The sensitivity mirror of the multisim analysis-scope write in
        ``processing_analysis.py`` (``consolidate_to_datatree``): the sensitivity
        master-consolidate path is otherwise the ONLY consolidation path that
        never writes an analysis-scope ``_du.json``, leaving the master root
        unsentineled so a ``delete --dry-run`` falls back to a full tree walk.

        Uses ``sum_child_sentinels`` (Gotcha 38 / the DU-rollup decision): the
        master total is the Σ of the per-sub ``_du.json`` sentinels (written by
        the D6 fold) + a bounded own-files walk excluding the child-scope dirs —
        NEVER a full-tree ``compute_and_write_scope_sentinel`` walk on the
        largest tree in the system. Ordering is structurally safe: the
        ``experiment_consolidation`` rule fans in on every per-sub completion flag,
        so all per-sub sentinels exist before this runs. Compare-and-write keeps
        the call idempotent (mtime preserved on unchanged bytes), so it is safe
        to invoke on the already-consolidated early-return path too.
        """
        from hhemt.du_sentinels import sum_child_sentinels

        sum_child_sentinels(
            self.experiment.analysis_paths.analysis_dir,
            scope="analysis",
            child_scope_dirs=["members", "sims"],
        )

    def open_sensitivity_datatree(self) -> "xr.DataTree":
        """Open the consolidated sensitivity DataTree zarr lazily."""
        path = self.analysis_paths.sensitivity_datatree_zarr
        if path is None or not path.exists():
            raise ValueError("Sensitivity DataTree zarr not found. Run consolidate_sensitivity_datatree() first.")
        return xr.open_datatree(path, engine="zarr", chunks="auto", consolidated=False)

    def create_analysis_summaries(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Build each member's own analysis-level summary datasets.

        The per-member half of :meth:`consolidate_outputs`, which calls
        this before assembling the master tree. Useful on its own to
        re-consolidate one tier without rebuilding the master.

        Parameters
        ----------
        which : {"TRITON", "SWMM", "both"}
            Restrict to one side of the coupled pair.
        verbose : bool
            Emit per-member progress.
        compression_level : int
            Zarr/NetCDF compression level.
        """
        if which in ["TRITON", "both"]:
            self._consolidate_outputs_in_each_analysis(
                which="TRITON",
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self._consolidate_outputs_in_each_analysis(
                which="SWMM",
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    @property
    def tritonswmm_SWMM_node_summary(self):
        """The master analysis's consolidated coupled-model SWMM NODE summary.

        Node-level results (depths, flooding) across every event and
        member, from the coupled TRITON-SWMM run.
        """
        return self.experiment._tritonswmm_SWMM_node_summary

    @property
    def tritonswmm_SWMM_link_summary(self):
        """The master analysis's consolidated coupled-model SWMM LINK summary.

        Conduit-level results (flow, capacity utilisation) across every event
        and member, from the coupled TRITON-SWMM run.
        """
        return self.experiment._tritonswmm_SWMM_link_summary

    @property
    def tritonswmm_TRITON_summary(self):
        """The master analysis's consolidated coupled-model TRITON summary.

        The 2D surface results — peak depth and water-surface elevation with
        their companion variables — across every event and member.
        """
        return self.experiment._tritonswmm_TRITON_summary

    # @property
    # def TRITONSWMM_runtimes(self):
    #     return self.experiment.TRITONSWMM_runtimes

    @property
    def analysis_independent_vars(self) -> list[str]:
        """Phase 2 — analysis-config attributes varied across members.

        Returns the canonical (stripped) field name for each varied analysis-config
        column. Recognizes both `analysis.{field}` (canonical) and bare `field`
        names (deprecated; emits DeprecationWarning at member construction
        time via `_create_members`).
        """
        from hhemt.config.analysis import analysis_config

        seen: list[str] = []
        for col in self._df_setup_full.columns:
            if col == "system_config_yaml":
                continue
            if _is_system_overlay_column(col):
                continue
            if _is_hpc_overlay_column(col):
                field_name = _resolve_hpc_alias_to_analysis_field(col)
            elif _is_analysis_overlay_column(col):
                field_name = _strip_analysis_prefix(col)
            elif col in analysis_config.model_fields:
                field_name = col  # bare name; DeprecationWarning fires at member construction time
            else:
                continue  # Defensive — should be caught by _retrieve_df_setup allowlist
            if field_name not in seen:
                seen.append(field_name)
        return seen

    @property
    def system_independent_vars(self) -> list[str]:
        """Phase 2 — system-config attributes varied across members.

        Recognizes only `system.{field}` columns (no bare names — Phase 1 R1
        rejects bare-name system_config columns at the allowlist gate).
        """
        seen: list[str] = []
        for col in self._df_setup_full.columns:
            if _is_system_overlay_column(col):
                field_name = _strip_system_prefix(col)
                if field_name not in seen:
                    seen.append(field_name)
        return seen

    @property
    def df_setup_with_system_overlays(self) -> pd.DataFrame:
        """`df_setup` unioned with the `system.*` overlay columns from the full frame.

        `df_setup` is filtered to analysis-config columns only (:173-180); the
        `system.*` overlay columns are retained on `_df_setup_full`. Report
        renderers that plot a system-axis independent variable
        (`system.target_dem_resolution`, `system.gpu_hardware`, ...) need both
        sets in one frame. This accessor unions them on the shared `member_id`
        index, preserving the PREFIXED column names so a renderer's
        `independent_var="system.gpu_hardware"` lookup resolves directly.

        The frame is analysis-columns + system-overlay-columns only — NOT the
        raw `_df_setup_full` (which also carries `system_config_yaml` and
        non-overlay annotation columns), keeping the renderer's column
        membership check scoped to the resolvable independent-var set.
        """
        overlay_cols = [c for c in self._df_setup_full.columns if _is_system_overlay_column(c)]
        if not overlay_cols:
            return self.df_setup
        return pd.concat(
            [self.df_setup, self._df_setup_full.loc[:, overlay_cols]],
            axis=1,
        )

    @property
    def independent_vars(self) -> list[str]:
        """BC alias — Phase 2 retains this name for downstream callers that haven't migrated.

        Returns the union of `analysis_independent_vars` and
        `system_independent_vars` (latter prefixed with `system.` to
        disambiguate). Downstream callers should migrate to the explicit
        `analysis_independent_vars` and `system_independent_vars` properties;
        this alias may be deprecated in a future release.

        Contract for prefixed-name entries: every entry of the returned list is
        an opaque label suitable for Snakemake wildcards (charset
        `^[A-Za-z0-9_.]+$`). Consumers MUST NOT deconstruct entries by `.`-split
        or by Pydantic-field lookup against a single model — entries may name
        either an `analysis_config` field (bare) OR a `system.{field}` overlay
        column. Verified consumers (`analysis.py`, `workflow.py`,
        `report_templates/workflow_description.rst.j2`, `config/report.py`,
        `bundle/snakefile_generator.py`) treat entries as opaque labels and
        tolerate the prefixed form without modification.
        """
        return self.analysis_independent_vars + [f"system.{f}" for f in self.system_independent_vars]

    def _retrieve_df_setup(self) -> pd.DataFrame:
        import re as _re

        snstivity_definition = self.experiment.cfg_analysis.sensitivity_analysis
        f_extension = snstivity_definition.name.lower().split(".")[-1]  # type: ignore
        if f_extension == "csv":
            df_setup = pd.read_csv(snstivity_definition)  # type: ignore
        elif f_extension == "xlsx":
            df_setup = pd.read_excel(snstivity_definition)
        else:
            raise ValueError("File extension not recognized for file defining sensitivity analysis.")
        retired = sorted(c for c in df_setup.columns if str(c).strip().lower() in _RETIRED_ID_COLUMNS)
        if retired:
            raise ConfigurationError(
                field="sensitivity_analysis",
                config_path=snstivity_definition,
                message=(
                    f"Sensitivity definition still uses the retired id column {retired}. "
                    f"Rename it to 'member_id' -- the identifier was renamed from sub-analysis "
                    f"to member and this toolkit carries no backwards compatibility for the old "
                    f"spelling. Only the header cell changes; every value stays as it is."
                ),
            )
        if "member_id" not in df_setup.columns:
            raise ValueError(
                "sensitivity_analysis file must contain a required 'member_id' column. "
                "Values may be integer or string but must be unique and match "
                "^[A-Za-z0-9_.]+$ to be safe for Snakemake wildcards."
            )
        df_setup["member_id"] = df_setup["member_id"].astype(str)
        if not df_setup["member_id"].is_unique:
            dupes = df_setup["member_id"][df_setup["member_id"].duplicated()].tolist()
            raise ValueError(f"member_id values must be unique. Duplicates: {dupes}")
        pat = _re.compile(r"^[A-Za-z0-9_.]+$")
        bad = [v for v in df_setup["member_id"] if not pat.match(v)]
        if bad:
            raise ValueError(
                f"member_id values must match ^[A-Za-z0-9_.]+$ (Snakemake-wildcard safe). Offending values: {bad}"
            )
        df_setup = df_setup.set_index("member_id")
        # Phase 1 — column allowlist enforcement (post-set_index so member_id excluded).
        from hhemt.config.analysis import analysis_config
        from hhemt.config.system import system_config

        KNOWN_BARE_COLS = {"system_config_yaml"}
        valid_columns = (
            KNOWN_BARE_COLS
            | set(analysis_config.model_fields)
            | {_SYSTEM_COLUMN_PREFIX + f for f in system_config.model_fields}
            | {_ANALYSIS_COLUMN_PREFIX + f for f in analysis_config.model_fields}
            | {_HPC_COLUMN_PREFIX + k for k in _HPC_ALIAS_TO_ANALYSIS_FIELD}
        )
        unknown = set(df_setup.columns) - valid_columns
        if unknown:
            # Phase 6 (DQ4): a direct `hpc.gpu_hardware` axis is rejected — gpu_hardware
            # is derived-only (R7); cross-hardware variation is expressed via the
            # `hpc.partition` selector (hardware derives from the partition spec).
            hpc_gpu_hint = (
                " To vary GPU hardware across rows, use `hpc.partition` "
                "(gpu_hardware derives from the partition spec); a direct "
                "`hpc.gpu_hardware` axis is not supported."
                if any(c.startswith(_HPC_COLUMN_PREFIX) for c in unknown)
                else ""
            )
            raise ConfigurationError(
                field="sensitivity_analysis.csv_columns",
                message=(
                    f"Unknown sensitivity-CSV columns: {sorted(unknown)}. "
                    f"Valid columns: member_id (required, becomes index), system_config_yaml, "
                    f"bare analysis_config field names, `system.{{field}}` for system_config fields, "
                    f"`analysis.{{field}}` for analysis_config fields, and the HPC-resource "
                    f"aliases `hpc.partition` / `hpc.setup_partition` (resolving to the "
                    f"analysis_config partition selectors)." + hpc_gpu_hint
                ),
                config_path=snstivity_definition,
            )
        return df_setup

    def export_sensitivity_definition_csv(self) -> Path:
        """Export sensitivity analysis definition to analysis directory as CSV.

        Exports only the fields that vary across members (self.df_setup columns)
        to a standardized 'sensitivity_analysis_definition.csv' file in the analysis directory.
        This allows easier inspection of the sensitivity analysis configuration during debugging.

        Returns:
            Path to the exported CSV file.
        """
        output_path = self.analysis_paths.analysis_dir / "sensitivity_analysis_definition.csv"
        df_export = self.df_setup.copy()
        df_export.to_csv(output_path, index=True)
        return output_path

    def find_orphan_member_dirs(self) -> list[Path]:
        """Return member directories on disk whose member_id is absent from the current CSV.

        The authoritative set of expected member directory names is derived
        from ``self.df_setup.index`` (the sensitivity CSV's ``member_id`` column) and
        the ``self.member_prefix`` constant. This ties orphan detection to
        the CSV directly, so a partially-constructed ``self.members`` dict
        cannot cause legitimate directories to be misclassified as orphans.
        On-disk ``member_*`` directories whose suffix fails the Snakemake-wildcard-safe
        charset ``^[A-Za-z0-9_.]+$`` are skipped — they were not created by this
        toolkit and must not be deleted by it. If ``self.members_dir`` does
        not exist, returns ``[]``.
        """
        import re as _re

        if not self.members_dir.exists():
            return []
        expected_names = {f"{self.member_prefix}{member_id}" for member_id in self.df_setup.index.astype(str)}
        charset = _re.compile(r"^[A-Za-z0-9_.]+$")
        orphans: list[Path] = []
        for entry in self.members_dir.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(self.member_prefix):
                continue
            suffix = entry.name[len(self.member_prefix) :]
            if not charset.match(suffix):
                continue
            if entry.name not in expected_names:
                orphans.append(entry)
        return sorted(orphans)

    def cleanup_orphan_analysis_dirs(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> list[Path]:
        """Identify and optionally delete orphaned member directories.

        Uses :meth:`find_orphan_member_dirs` to locate directories under
        ``members/`` whose ``member_id`` no longer appears in the current CSV.

        Parameters
        ----------
        dry_run : bool
            If True (default), only reports orphans without deleting.
        force : bool
            Required when ``dry_run=False``. Without it, the method raises
            ``ValueError`` to guard against accidental deletion of expensive
            HPC outputs.
        verbose : bool
            If True, prints each orphan path via ``print(..., flush=True)``.

        Returns
        -------
        list[Path]
            The orphan directories (either deleted or proposed for deletion).

        Raises
        ------
        ValueError
            If ``dry_run=False`` and ``force=False``.
        """
        from hhemt.utils import fast_rmtree

        orphans = self.find_orphan_member_dirs()
        if verbose:
            if orphans:
                print(f"[cleanup-orphans] Found {len(orphans)} orphan member directories:", flush=True)
                for p in orphans:
                    print(f"  {p}", flush=True)
            else:
                print("[cleanup-orphans] No orphan member directories found.", flush=True)
        if dry_run:
            return orphans
        if not force:
            raise ValueError(
                "cleanup_orphan_analysis_dirs called with dry_run=False but "
                "force=False. Pass force=True to perform deletion."
            )
        experiment_dir = self.experiment.analysis_paths.analysis_dir
        deleted: list[Path] = []
        failed: list[tuple[Path, Exception]] = []
        for p in orphans:
            if verbose:
                print(f"[cleanup-orphans] Deleting {p}", flush=True)
            try:
                fast_rmtree(p, analysis_dir=experiment_dir)  # PATTERN A
                deleted.append(p)
            except Exception as exc:
                failed.append((p, exc))
                if verbose:
                    print(f"[cleanup-orphans] FAILED to delete {p}: {exc}", flush=True)
        if failed:
            summary = "; ".join(f"{p}: {exc}" for p, exc in failed)
            raise RuntimeError(
                f"cleanup_orphan_analysis_dirs deleted {len(deleted)} of {len(orphans)} orphans; failures: {summary}"
            )
        return deleted

    def find_orphan_status_flags(self) -> list[Path]:
        """Return _status/ flag files whose embedded member_id is absent from df_setup.index.

        Matches against the four Snakemake rule-output flag families that embed
        a member_id (verified against workflow.py rule generation):

        - ``b_prepare_member-{member_id}_evt-{event_id}_complete.flag``
        - ``c_run_{model_type}_member-{member_id}_evt-{event_id}_complete.flag``
        - ``d_process_{model_type}_member-{member_id}_evt-{event_id}_complete.flag``
        - ``e_consolidate_member-{member_id}_complete.flag``

        The member_id charset is constrained to ``^[A-Za-z0-9_.]+$`` per the
        project stipulation. Returns an empty list if the ``_status/``
        directory does not exist.
        """
        import re as _re

        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        # Anchored to the four known rule-name prefixes so unrelated 'member-'
        # substrings (or future non-sensitivity rules that happen to contain
        # 'member-') cannot trigger a false orphan.
        pat = _re.compile(
            r"^(?:b_prepare|c_run_[A-Za-z0-9]+|d_process_[A-Za-z0-9]+|e_consolidate)_member-([A-Za-z0-9_.]+?)(?:_evt-[A-Za-z0-9_.]+|_complete|)\.flag$"
        )
        orphans: list[Path] = []
        for entry in status_dir.glob("*.flag"):
            m = pat.match(entry.name)
            if m is None:
                continue
            member_id = m.group(1)
            if member_id not in expected_member_ids:
                orphans.append(entry)
        return sorted(orphans)

    def find_orphan_input_fingerprints(self) -> list[Path]:
        """Return _status/member-{member_id}_inputs.json fingerprint files whose
        member_id is absent from df_setup.index.

        The per-member_id input-fingerprint files (Gotcha 17) are written by
        ``SensitivityAnalysisWorkflowBuilder`` at Snakefile-build time and named
        ``member-{member_id}_inputs.json`` under ``_status/``. When a member_id is removed
        from the sensitivity CSV its fingerprint is orphaned just like its
        dirs/flags/datatree groups. Returns an empty list if ``_status/`` is absent.
        """
        import re as _re

        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        pat = _re.compile(r"^member-([A-Za-z0-9_.]+?)_inputs\.json$")
        orphans: list[Path] = []
        for entry in status_dir.glob("member-*_inputs.json"):
            m = pat.match(entry.name)
            if m is None:
                continue
            member_id = m.group(1)
            if member_id not in expected_member_ids:
                orphans.append(entry)
        return sorted(orphans)

    def find_orphan_datatree_groups(self) -> list[str]:
        """Return member_id strings present as subgroups in sensitivity_datatree.zarr but absent from df_setup.index.

        Inspects on-disk subdirectories of ``sensitivity_datatree.zarr/`` matching
        ``{prefix}{member_id}`` where ``prefix`` is ``self.member_prefix``. Returns
        the member_id strings (without prefix). Returns an empty list if the zarr
        does not exist.
        """
        zarr_path = self.analysis_paths.sensitivity_datatree_zarr
        if zarr_path is None or not zarr_path.exists():
            return []
        expected_member_ids = set(self.df_setup.index.astype(str))
        prefix = self.member_prefix
        orphans: list[str] = []
        for entry in zarr_path.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix):
                continue
            member_id = entry.name[len(prefix) :]
            if member_id and member_id not in expected_member_ids:
                orphans.append(member_id)
        return sorted(orphans)

    def cleanup_all_orphans(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Detect and (optionally) delete orphan member dirs, flags, datatree groups, and input fingerprints.

        When any orphan is detected and deletion proceeds, the entire
        ``sensitivity_datatree.zarr`` is removed (rebuild approach — see plan
        D-SURGICAL) and the master-consolidation status flag is also removed so
        Snakemake re-runs the experiment_consolidation rule on the next workflow run.

        Parameters
        ----------
        dry_run : bool
            If True (default), only reports without deleting.
        force : bool
            Required when ``dry_run=False``.
        verbose : bool
            If True, prints each deletion via ``print(..., flush=True)``.

        Returns
        -------
        dict[str, list | bool]
            Keys: ``"dirs"`` (list[Path]), ``"status_flags"`` (list[Path]),
            ``"datatree_groups"`` (list[str]), ``"input_fingerprints"`` (list[Path]),
            and (after deletion only)
            ``"sensitivity_datatree_removed"`` (bool) and
            ``"master_flag_removed"`` (bool) reporting whether the
            rebuild-trigger artifacts were actually removed.

        Raises
        ------
        ValueError
            If ``dry_run=False`` and ``force=False``.
        """
        from hhemt.utils import fast_rmtree

        result = {
            "dirs": self.find_orphan_member_dirs(),
            "status_flags": self.find_orphan_status_flags(),
            "datatree_groups": self.find_orphan_datatree_groups(),
            "input_fingerprints": self.find_orphan_input_fingerprints(),
        }
        any_orphan = bool(
            result["dirs"] or result["status_flags"] or result["datatree_groups"] or result["input_fingerprints"]
        )
        if verbose:
            if any_orphan:
                print(
                    f"[cleanup-orphans] dirs={len(result['dirs'])} "
                    f"status_flags={len(result['status_flags'])} "
                    f"datatree_groups={len(result['datatree_groups'])}",
                    flush=True,
                )
                for p in result["dirs"]:
                    print(f"  dir: {p}", flush=True)
                for p in result["status_flags"]:
                    print(f"  flag: {p}", flush=True)
                for member_id in result["datatree_groups"]:
                    print(f"  datatree-group: member_{member_id}", flush=True)
                for p in result["input_fingerprints"]:
                    print(f"  input-fingerprint: {p}", flush=True)
            else:
                print("[cleanup-orphans] No orphans detected.", flush=True)
        if dry_run:
            return result
        if not force:
            raise ValueError(
                "cleanup_all_orphans called with dry_run=False but force=False. Pass force=True to perform deletion."
            )
        experiment_dir = self.experiment.analysis_paths.analysis_dir
        for p in result["dirs"]:
            if verbose:
                print(f"[cleanup-orphans] Deleting dir {p}", flush=True)
            fast_rmtree(p, analysis_dir=experiment_dir)  # PATTERN A
        for p in result["status_flags"]:
            if verbose:
                print(f"[cleanup-orphans] Unlinking flag {p}", flush=True)
            # EXEMPT-DU: status-flag
            p.unlink()
        for p in result["input_fingerprints"]:
            if verbose:
                print(f"[cleanup-orphans] Unlinking input-fingerprint {p}", flush=True)
            # EXEMPT-DU: status-dir-cleanup
            p.unlink()
        result["sensitivity_datatree_removed"] = False
        result["master_flag_removed"] = False
        if any_orphan:
            zarr_path = self.analysis_paths.sensitivity_datatree_zarr
            if zarr_path is not None and zarr_path.exists():
                if verbose:
                    print(
                        f"[cleanup-orphans] Deleting sensitivity_datatree.zarr (rebuild on next run): {zarr_path}",
                        flush=True,
                    )
                fast_rmtree(zarr_path, analysis_dir=experiment_dir)  # PATTERN A
                result["sensitivity_datatree_removed"] = True
            master_flag = self.analysis_paths.analysis_dir / "_status" / "f_consolidate_experiment_complete.flag"
            if master_flag.exists():
                if verbose:
                    print(
                        f"[cleanup-orphans] Unlinking master-consolidation flag {master_flag}",
                        flush=True,
                    )
                # EXEMPT-DU: status-flag
                master_flag.unlink()
                result["master_flag_removed"] = True
        return result

    def _build_unique_system_targets(
        self,
        df_setup_full: pd.DataFrame,
        is_main_orchestrator: bool = True,
    ) -> list[UniqueSystemTarget]:
        """Resolve per-member_id system targets and materialize per-target synthesized YAMLs.

        Handles three per-row mechanisms:

        1. ``system_config_yaml`` column (path to a per-member system YAML).
        2. ``system.{field}`` overlay columns (Phase 1 prefixed-column mechanism).
        3. Neither — fall back to master ``self._system``.

        Mutual exclusion: a single row may use mechanism 1 OR mechanism 2, never both;
        violation raises ``ConfigurationError``.

        ``is_main_orchestrator=True`` purges ``_generated/`` before emission.
        Runner subprocesses pass ``False`` to skip the purge.
        """
        import pydantic

        from hhemt.config.system import system_config
        from hhemt.system import TRITONSWMM_system
        from hhemt.utils import fast_rmtree

        sensitivity_csv = self.experiment.cfg_analysis.sensitivity_analysis
        analysis_dir = self.experiment.analysis_paths.analysis_dir
        generated_dir = analysis_dir / "_generated"

        if is_main_orchestrator:
            # PATTERN A (_generated is DU-counted; not _status*-prefixed)
            fast_rmtree(generated_dir, missing_ok=True, analysis_dir=analysis_dir)
            generated_dir.mkdir(parents=True, exist_ok=True)

        has_yaml_col = "system_config_yaml" in df_setup_full.columns
        overlay_col_names = sorted(c for c in df_setup_full.columns if _is_system_overlay_column(c))

        # Group members by their compile-key tuple.
        groups: dict[tuple, dict] = {}

        for member_id, row in df_setup_full.iterrows():
            member_id_str = str(member_id)
            yaml_cell = row.get("system_config_yaml") if has_yaml_col else None
            yaml_specified = yaml_cell is not None and not pd.isna(yaml_cell) and str(yaml_cell).strip() != ""
            overlay_cells = {_strip_system_prefix(c): row[c] for c in overlay_col_names if not pd.isna(row[c])}
            if overlay_cells and yaml_specified:
                raise ConfigurationError(
                    field=f"sensitivity_analysis.row[{member_id_str}]",
                    message=(
                        f"member_id={member_id_str}: row specifies both system_config_yaml "
                        f"({yaml_cell}) and system.* overlay column(s) "
                        f"{sorted(overlay_cells)}; mutually exclusive — "
                        f"use one mechanism per row."
                    ),
                    config_path=sensitivity_csv,
                )

            if overlay_cells:
                try:
                    # NOTE: the base model_dump() carries TRITONSWMM/SWMM software-dir
                    # Paths that may not yet exist on disk (created by system.py's
                    # clone/build gate at run/setup). They are exempt from
                    # _check_paths_exist via json_schema_extra={"toolkit_owned_output":
                    # True}. Do NOT add an existence assertion here.
                    cfg = system_config.model_validate(
                        {
                            **self._system.cfg_system.model_dump(),
                            **overlay_cells,
                        }
                    )
                except pydantic.ValidationError as exc:
                    raise ConfigurationError(
                        field=f"sensitivity_analysis.row[{member_id_str}]",
                        message=(
                            f"member_id={member_id_str}: system.* overlay-column values "
                            f"failed SystemConfig validation: {exc}"
                        ),
                        config_path=sensitivity_csv,
                    ) from exc
            elif yaml_specified:
                yaml_path = Path(yaml_cell).resolve()
                if not yaml_path.is_file():
                    raise ConfigurationError(
                        field="sensitivity_analysis.system_config_yaml",
                        message=(f"member_id={member_id_str}: system_config_yaml does not exist at {yaml_path}."),
                        config_path=sensitivity_csv,
                    )
                cfg = TRITONSWMM_system(yaml_path).cfg_system
            else:
                cfg = self._system.cfg_system

            # Phase 6 (DQ7a): resolve the ensemble partition PER ROW from the
            # overlay (analysis.hpc_ensemble_partition / hpc.partition), falling back
            # to the master. The dedup key STAYS hardware-derived (two same-hardware
            # partitions collapse to one build target) — NOT partition-name-keyed —
            # but the hardware now derives from each row's partition, so a6000 + a100
            # rows produce DISTINCT build targets.
            _row_partition = _resolve_row_ensemble_partition(row, self.experiment.cfg_analysis.hpc_ensemble_partition)
            _gpu_hardware, _gpu_backend = resolve_gpu_target(
                self.experiment.cfg_hpc_system,
                _row_partition,
            )
            key = (
                cfg.target_dem_resolution,
                _gpu_hardware,
                _gpu_backend,
            )
            if key not in groups:
                groups[key] = {"cfg": cfg, "member_ids": [], "partition": _row_partition}
            groups[key]["member_ids"].append(member_id_str)

        # Phase-4 (4c): GPU hardware/backend + modules are injected into each target
        # system (retired off system_config), resolved from the master ensemble
        # partition (uniform in 4c). The master self._system represents that same
        # ensemble target, so populate its injected attrs too (so the reuse branch
        # below is GPU-correct, not a None-injected CPU system).
        _gpu_hardware, _gpu_backend = resolve_gpu_target(
            self.experiment.cfg_hpc_system,
            self.experiment.cfg_analysis.hpc_ensemble_partition,
        )
        _modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
        # synth_cc friction fix (2026-07-08): inject the master-ensemble GPU pair into
        # self._system ONLY on the DRIVER (is_main_orchestrator=True). It is a
        # driver-only template feeding the reuse-branch below + the workflow.py GPU-
        # sensitivity validation (`not self.system.gpu_compilation_backend`). In a
        # setup_target_N RUNNER subprocess (is_main_orchestrator=False) self._system IS
        # the compile target that setup_workflow.py already constructed correctly from
        # --target-partition (backend=None for a CPU/standard partition, with
        # sys_paths.compilation_script_gpu frozen to None). Overwriting it here with the
        # master ensemble backend flips the CPU target into the GPU branch of
        # compile_TRITON_SWMM (system.py:522) against a None compile-script path ->
        # AttributeError at system.py:813; it also silently rewrites a non-master GPU
        # target's arch (a100 -> master a6000). sys_paths is frozen at construction and
        # is NOT rebuilt by this assignment, so the runner MUST keep its constructed pair.
        if is_main_orchestrator:
            self._system.gpu_hardware = _gpu_hardware
            self._system.gpu_compilation_backend = _gpu_backend
            self._system.additional_modules = _modules

        targets: list[UniqueSystemTarget] = []
        for target_id, group in enumerate(groups.values()):
            cfg = group["cfg"]
            member_ids = group["member_ids"]
            _target_partition = group["partition"]
            # Phase 6 (DQ7a): resolve THIS target's (hw, backend, modules) from its
            # own partition — distinct targets (a6000 vs a100) inject distinct pairs.
            _t_hw, _t_backend = resolve_gpu_target(self.experiment.cfg_hpc_system, _target_partition)
            _t_modules = resolve_additional_modules(self.experiment.cfg_hpc_system)
            generated_yaml = generated_dir / f"target_{target_id}.yaml"
            if is_main_orchestrator:
                # Temp-file-rename for atomicity (PID-keyed per Gotcha 17 pattern).
                tmp_yaml = generated_dir / f"target_{target_id}.{os.getpid()}.tmp.yaml"
                with tmp_yaml.open("w") as fh:
                    yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
                tmp_yaml.rename(generated_yaml)
            # Reuse master self._system only when BOTH the resolved cfg AND the
            # injected GPU pair match the master's (else a same-cfg, different-hw
            # target would wrongly reuse the master CPU/hardware system).
            if (
                cfg.model_dump_json() == self._system.cfg_system.model_dump_json()
                and _t_hw == self._system.gpu_hardware
                and _t_backend == self._system.gpu_compilation_backend
            ):
                target_system = self._system
            else:
                target_system = TRITONSWMM_system(
                    generated_yaml,
                    gpu_hardware=_t_hw,
                    gpu_compilation_backend=_t_backend,
                    additional_modules=_t_modules,
                    # ADR-1/M-7: this per-UniqueSystemTarget system DRIVES the
                    # sensitivity compile (compile_TRITON_SWMM below). In container
                    # mode the libstdc++ exact-soname patch must be suppressed (the
                    # SIF carries a self-consistent toolchain); the False default in
                    # native mode keeps the link line byte-identical.
                    execution_container_mode=(self.experiment.cfg_analysis.execution_environment == "container"),
                )
            targets.append(
                UniqueSystemTarget(
                    target_id=target_id,
                    system_config_yaml=generated_yaml,
                    system=target_system,
                    analysis_ids=member_ids,
                    target_partition=_target_partition,
                )
            )

        return targets

    def _materialize_target_yamls(self) -> None:
        """Re-write the per-target ``_generated/target_*.yaml`` files from the
        already-resolved in-memory :attr:`unique_system_targets`.

        ``_build_unique_system_targets`` writes these YAMLs at CONSTRUCTION
        (``is_main_orchestrator=True``), and the ``setup_target_N`` rules emitted by
        :meth:`SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content`
        reference their ABSOLUTE paths. But ``analysis.run(from_scratch=True)``
        ``fast_rmtree``s the analysis_dir AFTER construction, deleting ``_generated/``
        with nothing re-materializing it, so the setup rules fail with
        ``System config not found: .../_generated/target_0.yaml``. The master-Snakefile
        generator therefore calls this at generation time (post-wipe, before the setup
        rules are written) so the referenced files always exist.

        Only targets whose ``system_config_yaml`` lives under ``analysis_dir/_generated``
        are (re)written — the fast-path target points at the master system config
        (outside analysis_dir, never wiped) and is skipped. Idempotent: a byte-identical
        re-write on the resume path is harmless because the target YAMLs are shell ARGs,
        not declared Snakemake ``input:`` (no rerun trigger).
        """
        analysis_dir = self.experiment.analysis_paths.analysis_dir
        generated_dir = (analysis_dir / "_generated").resolve()
        for target in self.unique_system_targets:
            yaml_path = Path(target.system_config_yaml)
            try:
                under_generated = yaml_path.resolve().is_relative_to(generated_dir)
            except (OSError, ValueError):
                under_generated = False
            if not under_generated:
                continue
            generated_dir.mkdir(parents=True, exist_ok=True)
            # Atomic temp-file rename (PID-keyed; mirrors the construction-time write).
            tmp_yaml = generated_dir / f"{yaml_path.stem}.{os.getpid()}.tmp.yaml"
            with tmp_yaml.open("w") as fh:
                yaml.safe_dump(target.system.cfg_system.model_dump(mode="json"), fh, sort_keys=False)
            tmp_yaml.rename(yaml_path)

    def _create_members(self):
        member_id_to_system: dict = {}
        for target in self.unique_system_targets:
            for member_id in target.analysis_ids:
                member_id_to_system[member_id] = target.system

        from hhemt.config.analysis import analysis_config

        dic_sensitivity_analyses = dict()
        for idx, row in self.df_setup.iterrows():
            member_id = str(idx)
            overlay_cells: dict = {}
            for k, v in row.items():
                if pd.isna(v):
                    continue
                if _is_hpc_overlay_column(k):
                    overlay_cells[_resolve_hpc_alias_to_analysis_field(k)] = v
                elif _is_analysis_overlay_column(k):
                    overlay_cells[_strip_analysis_prefix(k)] = v
                elif k in analysis_config.model_fields:
                    warnings.warn(
                        f"Bare-name analysis-config column `{k}` is deprecated; "
                        f"rename to `analysis.{k}` for the canonical prefixed-column form. "
                        f"Bare-name support will be removed in a future release.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    overlay_cells[k] = v
                else:
                    # Defensive — `_retrieve_df_setup`'s column allowlist plus the
                    # analysis-only `self.df_setup` projection should have filtered
                    # `member_id`/`system_config_yaml`/`system.*` already.
                    raise ConfigurationError(
                        field="sensitivity_analysis.unknown_column",
                        message=f"Column `{k}` is not a recognized analysis-config field.",
                    )
            cfg_snstvty_analysis = analysis_config.model_validate(
                {
                    **self.experiment.cfg_analysis.model_dump(),
                    **overlay_cells,
                }
            )
            analysis_id = f"{self.member_prefix}{member_id}"
            cfg_snstvty_analysis.analysis_id = analysis_id  # type: ignore
            analysis_directory = self.members_dir / str(cfg_snstvty_analysis.analysis_id)
            analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False
            cfg_snstvty_analysis.is_experiment_member = True

            cfg_anlysys_yaml = analysis_directory / f"{analysis_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = analysis_directory

            cfg_snstvty_analysis.experiment_cfg_yaml = self.experiment.analysis_config_yaml

            # DRIVER-only write (see `_should_materialize_analysis_yaml`). The
            # renderer subprocesses that dominate the report tail construct with
            # `is_main_orchestrator=False` and are pure readers here; only a
            # genuinely-absent target is written on those paths.
            #
            # Atomic write via temp-file + rename is retained for the remaining
            # writers. `Path.write_text` truncates the target before writing, so a
            # concurrent reader would otherwise catch the truncated-empty state and
            # fail with `model_validate(None)`; `rename(2)` (Path.replace) swaps a
            # fully-written file into place instead.
            #
            # Temp filename is keyed on a uuid4, NOT on `os.getpid()`. PID is unique
            # only per NODE, and these subprocesses span multiple compute nodes
            # writing to one shared filesystem, so a PID-keyed tmp path can collide
            # across nodes -- one writer's `replace()` moves the tmp out from under
            # another, raising FileNotFoundError on the second writer's replace.
            # uuid4 subsumes both PID and hostname.
            #
            # Note that atomic rename is NOT sufficient on its own here: concurrent
            # cross-client renames onto a SHARED destination name were observed to
            # expose a window in which another client's `open()` on that name
            # returns ENOENT. The gate above -- not the rename -- is what closes
            # that exposure, by ensuring there is normally a single writer.
            #
            # The deeper fix remains lifting member yaml materialization out
            # of `__init__` into an explicit setup phase; that would let this
            # collapse back to a single `cfg_anlysys_yaml.write_text(...)`.
            if _should_materialize_analysis_yaml(cfg_anlysys_yaml, self._is_main_orchestrator):
                _tmp = cfg_anlysys_yaml.with_suffix(cfg_anlysys_yaml.suffix + f".{uuid.uuid4().hex}.tmp")
                _tmp.write_text(
                    yaml.safe_dump(
                        cfg_snstvty_analysis.model_dump(mode="json"),
                        sort_keys=False,
                    )
                )
                _tmp.replace(cfg_anlysys_yaml)
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=member_id_to_system[member_id],
                skip_log_update=self._skip_log_update,
                # Phase 6 (DQ7): thread the master's hpc_system_config to each sub so
                # the per-sub partition -> gpu_hardware / gpus_per_node resolution
                # (resolve_gpu_target / resolve_gpus_per_node, consumed by the per-member
                # GRES emission) works for cross-hardware sensitivity. Subs previously
                # carried cfg_hpc_system=None, which was latent only because the GRES
                # block is gated on n_gpus>0 (the synth CPU fixtures never tripped it).
                hpc_system_config_yaml=self.experiment.hpc_system_config_yaml,
            )
            dic_sensitivity_analyses[member_id] = anlsys
        return dic_sensitivity_analyses

    def _compute_member_id_fingerprint_payload(self, analysis: "anlysis.TRITONSWMM_analysis") -> dict[str, object]:
        """Compute the deterministic fingerprint payload for one member.

        Projects the member's post-Pydantic ``analysis_config.model_dump(mode="json")``
        onto the canonical field names from ``self.analysis_independent_vars``
        (sorted). Adds a ``__schema_version__`` sentinel so future
        serializer-format changes are themselves observable. Excludes ``member_id``
        (the path already disambiguates).

        Stability contract: every ``analysis_config`` field that may appear in
        ``self.analysis_independent_vars`` must be JSON-stable under ``model_dump(mode="json")``
        — that is, two invocations on the same member instance must produce
        byte-identical ``json.dumps(..., sort_keys=True)`` output. The currently-known
        sensitivity-CSV columns (``cpus_per_sim``, ``n_omp_threads``, ``hpc_total_nodes``,
        ``hpc_max_simultaneous_sims``, ``hpc_total_job_duration_min``, ``run_mode``) are
        all native Python int/str/Literal types and meet the contract. Adding a
        new ``analysis_config`` field that may legitimately become a sensitivity-CSV
        column requires re-checking JSON stability and may require bumping
        ``__schema_version__``.

        Returns a plain dict suitable for ``json.dumps`` with ``sort_keys=True``.
        """
        cfg_dump = analysis.cfg_analysis.model_dump(mode="json")
        # KeyError on missing key — surfaces config-schema drift loudly rather than
        # producing fingerprints that silently project None for an absent field.
        # Phase 2 — project against `analysis_independent_vars` (canonical stripped
        # names) rather than the BC alias `independent_vars` (which includes
        # `system.*` entries that have no key in `cfg_dump`).
        payload: dict[str, object] = {
            "__schema_version__": 1,
            "fields": {k: cfg_dump[k] for k in sorted(self.analysis_independent_vars)},
        }
        # When the sensitivity CSV declares a `system_config_yaml` column, bump the
        # schema and attach a SHA-1 of the member's resolved cfg_system. This
        # invalidates any member_id whose system config changes between runs. The
        # schema bump intentionally invalidates every member_id on the first run that
        # introduces per-member system configs (Gotcha 17 cascade — see Phase 1 doc).
        if self._has_per_member_system_configs:
            payload["__schema_version__"] = 2
            cfg_system_json = analysis._system.cfg_system.model_dump_json(by_alias=False, exclude_none=False)
            payload["system_cfg_hash"] = hashlib.sha1(cfg_system_json.encode("utf-8")).hexdigest()

        # Phase 1 — attach system_overlay key when any system.* overlay columns
        # are declared on the master sensitivity df (un-projected).
        from hhemt.config.system import system_config

        df = self.experiment.sensitivity._df_setup_full
        overlay_col_names = [c for c in df.columns if _is_system_overlay_column(c)]
        if overlay_col_names:
            member_id_str = analysis.cfg_analysis.analysis_id.removeprefix(self.experiment.sensitivity.member_prefix)
            overlay_cells = {
                _strip_system_prefix(c): df.loc[member_id_str, c]
                for c in overlay_col_names
                if not pd.isna(df.loc[member_id_str, c])
            }
            if overlay_cells:
                resolved = system_config.model_validate(
                    {
                        **self.experiment._system.cfg_system.model_dump(),
                        **overlay_cells,
                    }
                )
                resolved_overlay = {k: resolved.model_dump(mode="json")[k] for k in overlay_cells}
            else:
                resolved_overlay = {}
            payload["__schema_version__"] = 3
            payload["system_overlay"] = resolved_overlay
        return payload

    def _write_member_id_fingerprint(
        self,
        analysis: "anlysis.TRITONSWMM_analysis",
        fingerprint_path: Path,
    ) -> bool:
        """Write the per-member_id fingerprint file via compare-and-write.

        Reads ``fingerprint_path`` if it exists, serializes the new payload with
        ``sort_keys=True`` and stable separators, and only rewrites the file when
        content differs. This preserves mtime when content is unchanged — the
        mechanism on which Snakemake's per-rule rerun gating depends.

        Returns ``True`` if the file was (re)written, ``False`` if skipped because
        content matched the existing file.
        """
        payload = self._compute_member_id_fingerprint_payload(analysis)
        new_text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        # Treat unreadable existing content (zero-byte, corrupted, encoding error
        # from a crashed prior workflow) as "not equal to new content" and proceed
        # to overwrite. This preserves the compare-and-write contract under the
        # one failure mode the contract cannot otherwise diagnose.
        try:
            existing = fingerprint_path.read_text() if fingerprint_path.exists() else None
        except (OSError, UnicodeDecodeError):
            existing = None
        if existing == new_text:
            return False
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(new_text)
        return True

    def compile_and_preprocess_all_targets(
        self,
        overwrite_system_inputs: bool = False,
        recompile_if_already_done_successfully: bool = False,
        verbose: bool = True,
    ):
        """Process system-level inputs and compile TRITON-SWMM for each unique target.

        Iterates ``self.unique_system_targets`` (populated in ``__init__``) and runs
        ``process_system_level_inputs()`` + ``compile_TRITON_SWMM()`` once per target.
        In the no-per-member-config case the list contains a single target
        wrapping the master system, so this method is the unified entry point for
        non-Snakemake direct execution regardless of whether per-member configs are used.
        """
        for target in self.unique_system_targets:
            if verbose:
                print(
                    f"[Setup] Processing target {target.target_id} ({len(target.analysis_ids)} members)",
                    flush=True,
                )
            target.system.process_system_level_inputs(
                overwrite_outputs_if_already_created=overwrite_system_inputs,
                verbose=verbose,
            )
            target.system.compile_TRITON_SWMM(
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                verbose=verbose,
            )
        self._update_experiment_log()

    def compile_TRITON_SWMM_for_sensitivity_analysis(
        self,
        verbose: bool = False,
        recompile_if_already_done_successfully: bool = False,
    ):
        """Compile the solver once per unique system target (workflow phase 1).

        members that agree on the compile-relevant tuple
        ``(target_dem_resolution, gpu_hardware, gpu_compilation_backend)``
        collapse into a single ``UniqueSystemTarget`` and share one build, so
        this compiles once per DISTINCT target rather than once per row. A
        sweep whose rows differ only in rank count therefore compiles once.

        Parameters
        ----------
        verbose : bool
            Stream compiler output.
        recompile_if_already_done_successfully : bool
            Rebuild even when the compilation log already records success.
            Compilation success is detected by log markers rather than by exit
            code, so a partially-failed build can exit 0 and be skipped.
        """
        for target in self.unique_system_targets:
            target.system.compile_TRITON_SWMM(
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                verbose=verbose,
            )
        self._update_experiment_log()
        return

    @property
    def scenarios_not_created(self):
        """Scenario directories whose PREPARATION never completed.

        Returns
        -------
        list of str
            One directory path per scenario whose scenario-level log does not
            record ``scenario_creation_complete``. Empty when preparation is
            complete everywhere.

        Notes
        -----
        Preparation only. A scenario can appear here having been prepared and
        then had its outputs cleared; it says nothing about simulation state,
        for which see :attr:`scenarios_not_run`.
        """
        scenarios_not_created = []
        for _analysis_iloc, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                if scen.log.scenario_creation_complete.get() is not True:
                    scenarios_not_created.append(str(scen.log.logfile.parent))
        return scenarios_not_created

    @property
    def scenarios_not_run(self):
        """Scenario directories where at least one ENABLED model has not completed.

        Returns
        -------
        list of str
            One directory path per scenario for which any enabled model type is
            missing a completed run. Empty when every enabled model has
            completed for every scenario.

        Notes
        -----
        Checks every enabled model, so a scenario whose TRITON run finished but
        whose SWMM run did not still appears here. Completion is read from the
        per-model logs, which are separate from the scenario preparation log.
        """
        scens_not_run = []
        for _analysis_iloc, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                # Check if all enabled models completed
                enabled_models = scen.run.model_types_enabled
                all_models_completed = all(scen.model_run_completed(model_type) for model_type in enabled_models)
                if not all_models_completed:
                    scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def classify_incomplete_sim_failures(self) -> dict[str, str]:
        """Scan model logs for all incomplete simulations across members and classify each failure.

        Aggregates ``_classify_model_log_failure()`` across all members.
        Works for both ``"1_job_many_srun_tasks"`` and ``"batch_job"`` execution
        methods — the SLURM cancellation marker appears in the model log in both cases.

        Returns
        -------
        dict[str, str]
            Maps scenario identifier (e.g. ``"sa1_0"``) to failure class:

            - ``"timeout"`` — log contains ``DUE TO TIME LIMIT``
            - ``"unclassified"`` — log exists but no known failure marker found
            - ``"no_log"`` — model log file does not exist
        """
        results: dict[str, str] = {}
        for member_id, analysis in self.members.items():
            for event_iloc in analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, analysis)
                enabled_models = scen.run.model_types_enabled
                for model_type in enabled_models:
                    if not scen.model_run_completed(model_type):
                        event_id = scen.event_id
                        key = f"member-{member_id}_evt-{event_id}"
                        results[key] = scen.run._classify_model_log_failure(model_type)
        return results

    @property
    def is_timeout_only_failure(self) -> bool:
        """True iff all incomplete simulations across members have timeout-classified failures.

        Returns False if there are no incomplete sims (all done), or if any
        incomplete sim has an unclassified or no_log failure.
        """
        failures = self.classify_incomplete_sim_failures()
        if not failures:
            return False
        return all(v == "timeout" for v in failures.values())

    @property
    def df_status(self):
        """
        Get status DataFrame for all scenarios across all members.

        Returns
        -------
        pd.DataFrame
            Concatenated status table from all members. This includes
            member-specific setup columns plus the canonical status
            schema from ``TRITONSWMM_analysis.df_status`` (e.g. ``scenario_setup``
            and ``run_completed``), as well as:

            - sub_analysis_iloc: int - member index
        """
        status_frames = []

        for member_id, analysis in self.members.items():
            assert (
                analysis.cfg_analysis.is_experiment_member
            ), "is_experiment_member attribute not true in member.cfg_analysis.is_experiment_member"
            sub_df_status = analysis.df_status.copy()

            setup_row = self.df_setup.loc[member_id, :]
            for key, val in setup_row.items():
                sub_df_status[key] = val

            sub_df_status["member_id"] = member_id
            sub_df_status = sub_df_status[["member_id"] + [c for c in sub_df_status.columns if c != "member_id"]]

            status_frames.append(sub_df_status)

        if len(status_frames) == 0:
            return pd.DataFrame()

        return pd.concat(status_frames, ignore_index=True)

    @property
    def all_scenarios_created(self):
        """
        Check if all scenarios across all members have been created.

        Returns
        -------
        bool
            True if all scenarios in all members are created successfully
        """
        all_scenarios_created = True
        for _key, analysis in self.members.items():
            all_scenarios_created = all_scenarios_created and analysis._all_scenarios_created
        return all_scenarios_created is True

    @property
    def all_sims_run(self):
        """
        Check if all simulations across all members have completed.

        Returns
        -------
        bool
            True if all simulations in all members completed successfully
        """
        all_sims_run = True
        for _key, analysis in self.members.items():
            all_sims_run = all_sims_run and analysis._all_sims_run
        return all_sims_run is True

    @property
    def all_TRITON_timeseries_processed(self):
        """
        Check if all TRITON timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all TRITON outputs in all members are processed
        """
        all_TRITON_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_TRITON_timeseries_processed = (
                all_TRITON_timeseries_processed and analysis._all_TRITON_timeseries_processed
            )
        return all_TRITON_timeseries_processed is True

    @property
    def all_SWMM_timeseries_processed(self):
        """
        Check if all SWMM timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all SWMM outputs in all members are processed
        """
        all_SWMM_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_SWMM_timeseries_processed = all_SWMM_timeseries_processed and analysis._all_SWMM_timeseries_processed
        return all_SWMM_timeseries_processed is True

    @property
    def all_TRITONSWMM_performance_timeseries_processed(self):
        """
        Check if all performance timeseries across all members have been processed.

        Returns
        -------
        bool
            True if all performance outputs in all members are processed
        """
        all_TRITONSWMM_performance_timeseries_processed = True
        for _key, analysis in self.members.items():
            all_TRITONSWMM_performance_timeseries_processed = (
                all_TRITONSWMM_performance_timeseries_processed
                and analysis._all_TRITONSWMM_performance_timeseries_processed
            )
        return all_TRITONSWMM_performance_timeseries_processed is True

    @property
    def TRITONSWMM_performance_time_series_not_processed(self):
        """Scenarios whose coupled-model PERFORMANCE timeseries has not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.

        Notes
        -----
        Performance timeseries are the per-rank timing records used for the
        benchmarking figures; they are processed separately from the physical
        results, so this can be non-empty while the flood results are complete.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._TRITONSWMM_performance_time_series_not_processed
        return lst_scens

    @property
    def TRITON_time_series_not_processed(self):
        """Scenarios whose TRITON result timeseries have not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._TRITON_time_series_not_processed
        return lst_scens

    @property
    def SWMM_time_series_not_processed(self):
        """Scenarios whose SWMM result timeseries have not been processed.

        Returns
        -------
        list
            Per-scenario entries aggregated across all members. Empty when
            processing is complete everywhere.
        """
        lst_scens = []
        for _key, analysis in self.members.items():
            lst_scens += analysis._SWMM_time_series_not_processed
        return lst_scens

    @property
    def all_raw_TRITON_outputs_cleared(self):
        """
        Check if all raw TRITON outputs across all members have been cleared.

        Returns
        -------
        bool
            True if all raw TRITON outputs in all members are cleared
        """
        all_raw_TRITON_outputs_cleared = True
        for _key, analysis in self.members.items():
            all_raw_TRITON_outputs_cleared = all_raw_TRITON_outputs_cleared and analysis._all_raw_TRITON_outputs_cleared
        return all_raw_TRITON_outputs_cleared is True

    @property
    def all_raw_SWMM_outputs_cleared(self):
        """
        Check if all raw SWMM outputs across all members have been cleared.

        Returns
        -------
        bool
            True if all raw SWMM outputs in all members are cleared
        """
        all_raw_SWMM_outputs_cleared = True
        for _key, analysis in self.members.items():
            all_raw_SWMM_outputs_cleared = all_raw_SWMM_outputs_cleared and analysis._all_raw_SWMM_outputs_cleared
        return all_raw_SWMM_outputs_cleared is True

    def _update_experiment_log(self):
        self.experiment._update_log()
        return
