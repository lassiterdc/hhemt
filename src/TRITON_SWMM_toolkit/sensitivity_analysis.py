# %%
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
import yaml  # type: ignore

import TRITON_SWMM_toolkit.analysis as anlysis
from TRITON_SWMM_toolkit import orchestrator_sentinels as _osent
from TRITON_SWMM_toolkit.cf_conventions import apply_global_attributes
from TRITON_SWMM_toolkit.config.analysis import ClearRawValue, ForceRerunValue
from TRITON_SWMM_toolkit.exceptions import ConfigurationError
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.utils import current_datetime_string, write_datatree_zarr
from TRITON_SWMM_toolkit.workflow import (
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeDiagnostics,
    _emit_report_artifacts,
)

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis
    from .system import TRITONSWMM_system


@dataclass
class UniqueSystemTarget:
    target_id: int
    system_config_yaml: Path
    system: "TRITONSWMM_system"
    sub_analysis_ids: list[str] = field(default_factory=list)


_SYSTEM_COLUMN_PREFIX = "system."
_ANALYSIS_COLUMN_PREFIX = "analysis."


def _is_system_overlay_column(col: str) -> bool:
    """True if `col` is `system.{field}` where field is in system_config.model_fields."""
    if not col.startswith(_SYSTEM_COLUMN_PREFIX):
        return False
    from TRITON_SWMM_toolkit.config.system import system_config

    field_name = col[len(_SYSTEM_COLUMN_PREFIX) :]
    return field_name in system_config.model_fields


def _is_analysis_overlay_column(col: str) -> bool:
    """True if `col` is `analysis.{field}` where field is in analysis_config.model_fields."""
    if not col.startswith(_ANALYSIS_COLUMN_PREFIX):
        return False
    from TRITON_SWMM_toolkit.config.analysis import analysis_config

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


def _to_native_attr(value):
    """Cast pandas / numpy scalars to JSON-safe native Python types for zarr attrs."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class TRITONSWMM_sensitivity_analysis:
    """
    Manages sensitivity analysis by creating and orchestrating multiple sub-analyses.

    This class creates a separate TRITONSWMM_analysis instance for each row in a
    sensitivity analysis configuration table (CSV or Excel). Each sub-analysis runs
    with different parameter values, and results are consolidated at the master level.

    The sensitivity analysis workflow:
    1. Reads sensitivity configuration (CSV/Excel with parameter combinations)
    2. Creates sub-analysis for each configuration row
    3. Runs simulations for all sub-analyses
    4. Consolidates outputs across all parameter combinations
    5. Produces multi-dimensional datasets with sensitivity dimensions

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        Master analysis instance that contains the sensitivity configuration

    Attributes
    ----------
    master_analysis : TRITONSWMM_analysis
        Reference to the master analysis
    sub_analyses : dict
        Dictionary mapping sub-analysis index to TRITONSWMM_analysis instances
    df_setup : pd.DataFrame
        Sensitivity configuration table with parameter combinations
    independent_vars : list
        List of parameters being varied in the sensitivity analysis
    """

    def __init__(
        self,
        analysis: "TRITONSWMM_analysis",
        is_main_orchestrator: bool = True,
    ) -> None:
        """
        Initialize a sensitivity analysis orchestrator.

        Creates sub-analyses for each parameter combination defined in the sensitivity
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
        self.master_analysis = analysis
        self._system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        self.cfg_analysis = analysis.cfg_analysis
        self.sub_analyses_prefix = "sa_"
        self.subanalysis_dir = self.master_analysis.analysis_paths.analysis_dir / "subanalyses"
        df_setup_full = self._retrieve_df_setup()
        self._df_setup_full = df_setup_full
        self._has_per_sa_system_configs = "system_config_yaml" in df_setup_full.columns
        self._has_per_sa_system_overlay_columns = any(_is_system_overlay_column(c) for c in df_setup_full.columns)
        if self._has_per_sa_system_overlay_columns or self._has_per_sa_system_configs:
            self.unique_system_targets = self._build_unique_system_targets(
                df_setup_full,
                is_main_orchestrator=is_main_orchestrator,
            )
        else:
            # Fast path: no row varies system_config; reuse master self._system.
            self.unique_system_targets = [
                UniqueSystemTarget(
                    target_id=0,
                    system_config_yaml=self._system.system_config_yaml,
                    system=self._system,
                    sub_analysis_ids=list(df_setup_full.index.astype(str)),
                )
            ]
        from TRITON_SWMM_toolkit.config.analysis import analysis_config as _analysis_config_for_df_setup

        analysis_cols = [
            c
            for c in df_setup_full.columns
            if c in _analysis_config_for_df_setup.model_fields or _is_analysis_overlay_column(c)
        ]
        self.df_setup = df_setup_full.loc[:, analysis_cols]
        self.sub_analyses = self._create_sub_analyses()

        # Initialize workflow builder for sensitivity analysis
        self._workflow_builder = SensitivityAnalysisWorkflowBuilder(self)

    def prepare_scenarios_in_each_subanalysis(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        concurrent: bool = True,
        verbose: bool = False,
    ):
        if self.master_analysis.cfg_analysis.multi_sim_run_method in [
            "local",
            "1_job_many_srun_tasks",
        ]:
            prepare_scenario_launchers = []
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                prepare_scenario_launchers += sub_analysis.retrieve_prepare_scenario_launchers(
                    overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                    rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                    verbose=verbose,
                )
            if concurrent:
                self.master_analysis.run_python_functions_concurrently(prepare_scenario_launchers, verbose=verbose)
            else:
                for launcher in prepare_scenario_launchers:
                    launcher()

            if self.all_scenarios_created is not True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(f"Preparation failed for the following scenarios:\n{scens_not_created}")
            self._update_master_analysis_log()
        elif self.master_analysis.cfg_analysis.multi_sim_run_method in ["batch_job"]:
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
        override_clear_raw: ClearRawValue | None = None,
        override_force_rerun: ForceRerunValue | None = None,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        This orchestrates multiple sub-analysis workflows and a final master
        consolidation step that combines all sub-analysis outputs.

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
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` (None reads YAML).
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If set, overrides `hpc_total_nodes` in the generated SBATCH script without
            mutating the config. Only valid for `multi_sim_run_method="1_job_many_srun_tasks"`.

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str
            - snakefile_path: Path
            - message: str
        """
        # Force-rerun pre-delete for direct sensitivity.submit_workflow callers.
        # Idempotent when Analysis.submit_workflow already applied it on the
        # dispatch path (matched flags would be absent by now).
        self.master_analysis._apply_force_rerun(override_force_rerun)

        # Driver-start orchestrator-liveness sentinel (Phase 2), keyed on the
        # MASTER analysis_dir. This is the sensitivity-master submit path and
        # always owns its sentinel (the Analysis.submit_workflow guard leaves
        # _driver_id None there and delegates here). Blocking-local drivers
        # remove on return; detached drivers leave a durable sentinel reclaimed
        # by the gate's liveness probes.
        _master_dir = self.master_analysis.analysis_paths.analysis_dir
        _eff_mode = self.master_analysis.cfg_analysis.multi_sim_run_method
        _driver_id = _osent.new_driver_id()
        _osent.write_orchestrator_sentinel(
            _master_dir,
            driver_id=_driver_id,
            workflow_submission_mode=_eff_mode,
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
                override_clear_raw=override_clear_raw,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                wait_for_completion=wait_for_completion,
                dry_run=dry_run,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
                report_formats=report_formats,
                extra_sbatch_args=extra_sbatch_args,
                snakemake_diagnostics=snakemake_diagnostics,
            )
        finally:
            if _eff_mode == "local":
                _osent.remove_orchestrator_sentinel(_master_dir, _driver_id)

        if _eff_mode != "local" and isinstance(result, dict):
            _osent.enrich_orchestrator_sentinel(
                _master_dir,
                driver_id=_driver_id,
                slurm_jobid=result.get("job_id"),
                tmux_session_name=result.get("session_name"),
            )

        return result

    def _invalidate_processing_log_for_sa_ids(
        self, sa_id_tokens: tuple[str, ...]
    ) -> None:
        """Per-sa_id dispatch for processing-log invalidation under
        ``override_force_rerun={"sa_id": [...]}``.

        For each requested sa_id, looks up its sub-analysis and calls the
        per-sub-analysis ``Analysis._invalidate_processing_log_for_force_rerun``
        with a ``scope="all"`` spec — which invalidates every scenario in
        that sub-analysis. Sub-analyses are full Analysis instances and
        own their own scenario list (cf. CLAUDE.md Gotcha 11: "Sensitivity
        analysis sub-analyses are full TRITONSWMM_analysis instances").

        Per cleanup-rerun-delete-redesign Phase 4 + B-mechanism.
        """
        from TRITON_SWMM_toolkit.workflow import ResolvedForceRerunSpec

        all_spec = ResolvedForceRerunSpec(scope="all", tokens=())
        for sa_id in sa_id_tokens:
            sub_analysis = self.sub_analyses.get(sa_id)
            if sub_analysis is None:
                # _validate_force_rerun_targets already filtered unknown
                # sa_ids; reaching here means the sub_analyses dict is
                # out of sync with df_setup — surface loudly.
                raise RuntimeError(
                    f"sub_analyses missing entry for sa_id={sa_id!r} after "
                    f"validation passed; df_setup/sub_analyses are out of sync"
                )
            sub_analysis._invalidate_processing_log_for_force_rerun(all_spec)

    def reprocess(
        self,
        start_with: Literal["process", "consolidate", "render"] = "consolidate",
        sa_ids: list[str] | None = None,
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        verbose: bool = True,
        dry_run: bool = False,
        report_formats: list[str] | None = None,
        *,
        regenerate_existing: bool = False,
        override_force_rerun: ForceRerunValue | None = None,
    ) -> dict:
        """Master-level reprocess for sensitivity analyses.

        Invalidates per-sub-analysis consolidate flags (subset via ``sa_ids``
        or all sub-analyses by default) plus the master consolidate flag,
        then emits a scoped master Snakefile via
        :meth:`SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content`
        and submits it via
        :meth:`SensitivityAnalysisWorkflowBuilder.submit_reprocess_workflow`.

        Unlike :meth:`TRITONSWMM_analysis.reprocess`, this method does NOT
        invoke the ``override_clear_raw`` orphan/abort gate (R12) — sensitivity
        master reprocess is a downstream-only refresh of consolidation +
        plotting + rendering against existing per-sa sim outputs and does
        not need the in-flight reconciliation logic that the analysis-level
        ``override_clear_raw`` flow uses.

        Parameters
        ----------
        start_with
            Stage to re-fire from. ``"consolidate"`` (default) deletes per-sa
            ``e_consolidate_sa-{id}_complete.flag`` files and the master
            ``f_consolidate_master_complete.flag``, then re-runs the consolidate
            + master_consolidation + plot/render rule chain. ``"render"``
            invalidates only the report artifacts. ``"process"`` is accepted
            but maps onto the same Snakefile as ``"consolidate"`` (the master
            generator does not emit ``process_*`` rules).
        sa_ids
            Optional subset of sub-analysis IDs (string-cast) to invalidate.
            When ``None`` (default), every sub-analysis's per-sa consolidate
            flag is invalidated. IDs not in ``sub_analyses`` are silently
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
        # Lazy-stamp _version.json at LAYOUT_VERSION (PI-1 pattern). Idempotent.
        from TRITON_SWMM_toolkit.version_migration import LAYOUT_VERSION
        from TRITON_SWMM_toolkit.version_migration.state import stamp_new_target

        stamp_new_target(self.master_analysis.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Force-rerun pre-delete (login-node responsibility). Per
        # cleanup-rerun-delete-redesign Phase 4 + R10. Resolves + validates +
        # deletes matched flags before Snakemake plans the reprocess DAG.
        # Skipped on dry_run — it deletes flags and clears per-scenario
        # processing-log records, both filesystem mutations the dry-run
        # no-destructive-mutation contract forbids.
        if not dry_run:
            self.master_analysis._apply_force_rerun(override_force_rerun)

        # Resolve invalidation target set. ``None`` → all sub-analyses; explicit
        # list → subset. String-cast preserves alignment with sub_analyses dict
        # iteration keys regardless of source type (int / str / numpy scalar).
        if sa_ids is None:
            targets = [str(sa_id) for sa_id in self.sub_analyses.keys()]
        else:
            targets = [str(s) for s in sa_ids]

        # Invalidate per-sa consolidate flags + master flag. start_with controls
        # which flags get unlinked; per-sa flag deletion is the entry point for
        # both "consolidate" and "process" (the master generator does not emit
        # process rules, so process invalidation is treated as consolidate
        # invalidation). "render" leaves consolidate flags intact and only
        # invalidates the rendered report artifact.
        from TRITON_SWMM_toolkit.du_sentinels import (
            compute_and_write_scope_sentinel,
            restamp_parent_sentinels,
        )
        from TRITON_SWMM_toolkit.utils import fast_rmtree as _fast_rmtree

        master_analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        status_dir = master_analysis_dir / "_status"
        if start_with in ("consolidate", "process"):
            for sa_id in targets:
                (status_dir / f"e_consolidate_sa-{sa_id}_complete.flag").unlink(missing_ok=True)
            (status_dir / "f_consolidate_master_complete.flag").unlink(missing_ok=True)
            # Report+plot deletion ALWAYS runs (toggle-independent) — the report
            # regenerates from the preserved zarr on the default path (FQ1 parity).
            _report_html = master_analysis_dir / "analysis_report.html"
            _report_zip = master_analysis_dir / "analysis_report.zip"
            _report_html.unlink(missing_ok=True)
            _report_zip.unlink(missing_ok=True)
            if not dry_run:
                restamp_parent_sentinels(_report_html, analysis_dir=master_analysis_dir)  # PATTERN B
            # Consolidated-zarr deletion + batched DU restamp are the EXPENSIVE
            # GPFS work — gate behind regenerate_existing. Default path preserves
            # the zarrs (consolidate stays inert) and runs NO restamp walk.
            if regenerate_existing and not dry_run:
                affected_sub_dirs: set = set()
                for sa_id in targets:
                    sub_analysis = self.sub_analyses.get(sa_id)
                    if sub_analysis is None:
                        continue
                    _sub_zarr = sub_analysis.analysis_paths.analysis_datatree_zarr
                    if _sub_zarr is not None and _sub_zarr.exists():
                        _fast_rmtree(_sub_zarr, analysis_dir=None)  # batched-restamp
                        affected_sub_dirs.add(sub_analysis.analysis_paths.analysis_dir)
                _master_zarr = self.analysis_paths.sensitivity_datatree_zarr
                if _master_zarr is not None and _master_zarr.exists():
                    _fast_rmtree(_master_zarr, analysis_dir=None)  # batched-restamp
                for _sub_dir in affected_sub_dirs:
                    compute_and_write_scope_sentinel(_sub_dir, scope="sub_analysis")
                compute_and_write_scope_sentinel(master_analysis_dir, scope="analysis")
        elif start_with == "render":
            # No _status flag for render — re-fire by deleting the report
            # artifacts so Snakemake's mtime trigger sees the output as absent.
            # The report-artifact unlink is the flag-equivalent trigger, so it
            # runs even on dry_run (see D6); only the DU restamp is gated.
            _report_html = master_analysis_dir / "analysis_report.html"
            _report_zip = master_analysis_dir / "analysis_report.zip"
            _report_html.unlink(missing_ok=True)
            _report_zip.unlink(missing_ok=True)
            if not dry_run:
                restamp_parent_sentinels(_report_html, analysis_dir=master_analysis_dir)  # PATTERN B
        else:
            raise ValueError(f"start_with must be one of 'process', 'consolidate', 'render'; got {start_with!r}")

        # Delegate to the sensitivity workflow builder.
        return self._workflow_builder.submit_reprocess_workflow(
            start_with=start_with,
            execution_mode=execution_mode,
            which=which,
            compression_level=compression_level,
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
        from TRITON_SWMM_toolkit.utils import fast_rmtree

        analysis_dir = self.master_analysis.analysis_paths.analysis_dir

        # 1. Clear any stale sentinels from a prior failed delete attempt.
        stale_dir = analysis_dir / "_status" / "_deleting"
        if stale_dir.exists():
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
                f"[delete] {len(missing)} per-sa-rule sentinels missing — "
                f"preserving analysis_dir for debugging.",
                flush=True,
            )
            print(f"[delete] missing: {sorted(p.name for p in missing)}", flush=True)
            return
        print(
            f"[delete] all {len(expected)} per-sa-rule sentinels present — "
            f"removing analysis_dir.",
            flush=True,
        )
        fast_rmtree(analysis_dir)

    def _enumerate_expected_delete_sentinels(self) -> set[Path]:
        """Compute the set of ``_status/_deleting/*.flag`` paths the
        sensitivity delete workflow will produce on full success.

        One per sub-analysis row in ``self.df_setup.index`` plus one for the
        analysis-level consolidation rule.
        """
        delete_dir = self.master_analysis.analysis_paths.analysis_dir / "_status" / "_deleting"
        expected = {delete_dir / "analysis_consolidation.flag"}
        for sa_id in self.df_setup.index.astype(str):
            expected.add(delete_dir / f"subanalysis_sa-{sa_id}.flag")
        return expected

    def render_report(self, format: Literal["html", "zip"] = "zip") -> "Path":
        """Render the master report for the sensitivity analysis.

        Idempotent: invokes ``snakemake --report`` against the master Snakefile
        without re-executing any rules. Renders only the master-level report;
        per-sub-analysis reports are not generated (R13).

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
        """
        import subprocess
        import sys

        from .exceptions import WorkflowError

        master_dir = self.master_analysis.analysis_paths.analysis_dir
        snakefile = master_dir / "Snakefile"
        out = master_dir / f"analysis_report.{format}"
        css_path = master_dir / "report" / "report.css"
        # Re-emit report artifacts from package resources so render_report
        # picks up edits made to the source-tree report_templates/.
        _emit_report_artifacts(master_dir)
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

        try:
            if format == "html":
                out.write_text(apply_post_process_surgery(out.read_text()))
            else:
                apply_post_process_surgery_to_zip(out)
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

    # Conforms to TRITON_SWMM_toolkit.bundle._protocol.BundleableAnalysis
    # via duck typing — attributes delegated to self.master_analysis in
    # __init__ (lines 91-94).
    def bundle_report_data(
        self,
        output_path: "Path | None" = None,
    ) -> "Path":
        """Emit a portable render bundle for the sensitivity master analysis.

        Opt-in only — NEVER invoked from analysis.run() or
        submit_workflow(). The bundle includes the sensitivity master's
        consolidated outputs plus the union of source paths declared by
        every renderer in the master's render_report(), including per-sim
        renderers wildcarded over (sa_id, event_id).

        Args:
            output_path: Optional target path for the bundle tar.

        Returns:
            Path to the emitted bundle tar.
        """
        from TRITON_SWMM_toolkit.bundle import emit_bundle

        return emit_bundle(self, output_path)

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
        if concurrent:
            raise RuntimeError(
                "Running sensitivity analyses concurrently requires"
                "more intelligent handling of compute resource availability"
                "tracking. Update run_simulations_concurrently function"
                "in analysis.py to enable this."
            )
            launch_functions = []
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                launch_functions += sub_analysis._create_launchable_sims(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                )
            self.master_analysis.run_simulations_concurrently(launch_functions, verbose=verbose)
        else:
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                sub_analysis.run_sims_in_sequence(
                    pickup_where_leftoff=pickup_where_leftoff,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    override_clear_raw=override_clear_raw,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_master_analysis_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        scenario_timeseries_processing_launchers = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            launchers = sub_analysis.retrieve_scenario_timeseries_processing_launchers(
                which=which,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.master_analysis.run_python_functions_concurrently(scenario_timeseries_processing_launchers)
        return

    def _consolidate_outputs_in_each_subanalysis(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            sub_analysis.consolidate_analysis_outputs(
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_master_analysis_log()
        return

    @property
    def TRITON_subanalyses_outputs_consolidated(self):
        cfg_sys = self.master_analysis._system.cfg_system
        success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            if cfg_sys.toggle_tritonswmm_model:
                success = success and sub_analysis.tritonswmm_triton_analysis_summary_created
            elif cfg_sys.toggle_triton_model:
                success = success and sub_analysis.triton_only_analysis_summary_created
        return success

    @property
    def SWMM_subanalyses_outputs_consolidated(self):
        cfg_sys = self.master_analysis._system.cfg_system
        node_success = True
        link_success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            if cfg_sys.toggle_tritonswmm_model:
                node_success = node_success and sub_analysis.tritonswmm_node_analysis_summary_created
                link_success = link_success and sub_analysis.tritonswmm_link_analysis_summary_created
            elif cfg_sys.toggle_swmm_model:
                node_success = node_success and sub_analysis.swmm_only_node_analysis_summary_created
                link_success = link_success and sub_analysis.swmm_only_link_analysis_summary_created
        return node_success and link_success

    def consolidate_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = True,
        compression_level: int = 5,
    ):
        self.create_subanalysis_summaries(
            which=which,
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_subanalysis_outputs(
            which=which,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_subanalysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """Consolidate sub-analyses into a hierarchical sensitivity DataTree zarr.

        Replaces the previous per-mode flat ``xr.concat`` path. Each sub-analysis
        first builds its per-analysis DataTree (``analysis_datatree.zarr``); then
        the master assembles all sub-analyses into a single
        ``sensitivity_datatree.zarr`` at the master analysis dir.
        """
        self.consolidate_sensitivity_datatree(
            compression_level=compression_level,
            verbose=verbose,
        )
        return

    def build_sensitivity_datatree(self) -> "xr.DataTree":
        """Assemble the master sensitivity DataTree lazily from sub-analysis trees.

        Each sub-analysis's consolidated DataTree (``analysis_datatree.zarr``) is
        opened lazily and grafted under a ``sa_{sa_id}/`` subtree. Sensitivity
        parameters for each sub-analysis are attached as ``.attrs`` on the
        ``sa_{sa_id}`` node. A parameter-summary Dataset is written at the root
        under ``parameters`` for tabular queries.
        """
        tree_dict: dict[str, xr.Dataset] = {}

        tree_dict["/"] = xr.Dataset(
            attrs={
                "Conventions": "CF-1.13",
                "title": "TRITON-SWMM sensitivity analysis results",
                "analysis_id": str(self.master_analysis.cfg_analysis.analysis_id),
                "output_creation_date": current_datetime_string(),
            }
        )

        tree_dict["parameters"] = xr.Dataset.from_dataframe(self.df_setup)

        for sa_id, sub_analysis in self.sub_analyses.items():
            node_name = f"{self.sub_analyses_prefix}{sa_id}"
            try:
                sub_tree = sub_analysis.process.open_datatree()
            except ValueError:
                continue

            for path, node in sub_tree.subtree_with_keys:
                if not node.has_data:
                    continue
                rel = path.lstrip("/")
                if not rel:
                    continue
                tree_dict[f"{node_name}/{rel}"] = node.dataset

            setup_row = self.df_setup.loc[sa_id]
            attrs = {k: _to_native_attr(v) for k, v in setup_row.to_dict().items()}
            attrs["sa_id"] = str(sa_id)
            tree_dict[node_name] = xr.Dataset(attrs=attrs)

        tree = xr.DataTree.from_dict(tree_dict)
        apply_global_attributes(tree, analysis_id=str(self.master_analysis.cfg_analysis.analysis_id))
        return tree

    def consolidate_sensitivity_datatree(
        self,
        compression_level: int = 5,
        verbose: bool = False,
    ) -> Path:
        """Build and write the master sensitivity DataTree zarr.

        Ensures each sub-analysis has its own consolidated ``analysis_datatree.zarr``
        first, then assembles them into a single hierarchical store at
        ``sensitivity_datatree.zarr``.
        """
        fname_out = self.analysis_paths.sensitivity_datatree_zarr
        if fname_out is None:
            raise ValueError("sensitivity_datatree_zarr path is not configured on AnalysisPaths.")

        if fname_out.exists():
            if verbose:
                print(f"Sensitivity DataTree zarr already present at {fname_out}. Not overwriting.")
            return fname_out

        # Ensure each sub-analysis has its analysis_datatree.zarr built.
        for sa_id, sub_analysis in self.sub_analyses.items():
            sub_path = sub_analysis.analysis_paths.analysis_datatree_zarr
            if sub_path is None:
                continue
            if not sub_path.exists():
                sub_analysis.process.consolidate_to_datatree(
                    compression_level=compression_level,
                    verbose=verbose,
                )

        tree = self.build_sensitivity_datatree()
        write_datatree_zarr(tree, fname_out, compression_level=compression_level)

        self.master_analysis._refresh_log()
        if hasattr(self.master_analysis.log, "sensitivity_datatree_consolidation_complete"):
            self.master_analysis.log.sensitivity_datatree_consolidation_complete.set(True)
        if verbose:
            print(f"Wrote sensitivity DataTree zarr to {fname_out}")
        return fname_out

    def open_sensitivity_datatree(self) -> "xr.DataTree":
        """Open the consolidated sensitivity DataTree zarr lazily."""
        path = self.analysis_paths.sensitivity_datatree_zarr
        if path is None or not path.exists():
            raise ValueError("Sensitivity DataTree zarr not found. Run consolidate_sensitivity_datatree() first.")
        return xr.open_datatree(path, engine="zarr", chunks="auto", consolidated=False)

    def create_subanalysis_summaries(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="TRITON",
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="SWMM",
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    @property
    def tritonswmm_SWMM_node_summary(self):
        return self.master_analysis.tritonswmm_SWMM_node_summary

    @property
    def tritonswmm_SWMM_link_summary(self):
        return self.master_analysis.tritonswmm_SWMM_link_summary

    @property
    def tritonswmm_TRITON_summary(self):
        return self.master_analysis.tritonswmm_TRITON_summary

    # @property
    # def TRITONSWMM_runtimes(self):
    #     return self.master_analysis.TRITONSWMM_runtimes

    @property
    def analysis_independent_vars(self) -> list[str]:
        """Phase 2 — analysis-config attributes varied across sub-analyses.

        Returns the canonical (stripped) field name for each varied analysis-config
        column. Recognizes both `analysis.{field}` (canonical) and bare `field`
        names (deprecated; emits DeprecationWarning at sub-analysis construction
        time via `_create_sub_analyses`).
        """
        from TRITON_SWMM_toolkit.config.analysis import analysis_config

        seen: list[str] = []
        for col in self._df_setup_full.columns:
            if col == "system_config_yaml":
                continue
            if _is_system_overlay_column(col):
                continue
            if _is_analysis_overlay_column(col):
                field_name = _strip_analysis_prefix(col)
            elif col in analysis_config.model_fields:
                field_name = col  # bare name; DeprecationWarning fires at sub-analysis construction time
            else:
                continue  # Defensive — should be caught by _retrieve_df_setup allowlist
            if field_name not in seen:
                seen.append(field_name)
        return seen

    @property
    def system_independent_vars(self) -> list[str]:
        """Phase 2 — system-config attributes varied across sub-analyses.

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

        snstivity_definition = self.master_analysis.cfg_analysis.sensitivity_analysis
        f_extension = snstivity_definition.name.lower().split(".")[-1]  # type: ignore
        if f_extension == "csv":
            df_setup = pd.read_csv(snstivity_definition)  # type: ignore
        elif f_extension == "xlsx":
            df_setup = pd.read_excel(snstivity_definition)
        else:
            raise ValueError("File extension not recognized for file defining sensitivity analysis.")
        if "sa_id" not in df_setup.columns:
            raise ValueError(
                "sensitivity_analysis file must contain a required 'sa_id' column. "
                "Values may be integer or string but must be unique and match "
                "^[A-Za-z0-9_.]+$ to be safe for Snakemake wildcards."
            )
        df_setup["sa_id"] = df_setup["sa_id"].astype(str)
        if not df_setup["sa_id"].is_unique:
            dupes = df_setup["sa_id"][df_setup["sa_id"].duplicated()].tolist()
            raise ValueError(f"sa_id values must be unique. Duplicates: {dupes}")
        pat = _re.compile(r"^[A-Za-z0-9_.]+$")
        bad = [v for v in df_setup["sa_id"] if not pat.match(v)]
        if bad:
            raise ValueError(
                f"sa_id values must match ^[A-Za-z0-9_.]+$ (Snakemake-wildcard safe). Offending values: {bad}"
            )
        df_setup = df_setup.set_index("sa_id")
        # Phase 1 — column allowlist enforcement (post-set_index so sa_id excluded).
        from TRITON_SWMM_toolkit.config.analysis import analysis_config
        from TRITON_SWMM_toolkit.config.system import system_config

        KNOWN_BARE_COLS = {"system_config_yaml"}
        valid_columns = (
            KNOWN_BARE_COLS
            | set(analysis_config.model_fields)
            | {_SYSTEM_COLUMN_PREFIX + f for f in system_config.model_fields}
            | {_ANALYSIS_COLUMN_PREFIX + f for f in analysis_config.model_fields}
        )
        unknown = set(df_setup.columns) - valid_columns
        if unknown:
            raise ConfigurationError(
                field="sensitivity_analysis.csv_columns",
                message=(
                    f"Unknown sensitivity-CSV columns: {sorted(unknown)}. "
                    f"Valid columns: sa_id (required, becomes index), system_config_yaml, "
                    f"bare analysis_config field names, `system.{{field}}` for system_config fields, "
                    f"`analysis.{{field}}` for analysis_config fields. "
                    f"If you previously used `gpu_hardware_override`, replace with `system.gpu_hardware`."
                ),
                config_path=snstivity_definition,
            )
        return df_setup

    def export_sensitivity_definition_csv(self) -> Path:
        """Export sensitivity analysis definition to analysis directory as CSV.

        Exports only the fields that vary across sub-analyses (self.df_setup columns)
        to a standardized 'sensitivity_analysis_definition.csv' file in the analysis directory.
        This allows easier inspection of the sensitivity analysis configuration during debugging.

        Returns:
            Path to the exported CSV file.
        """
        output_path = self.analysis_paths.analysis_dir / "sensitivity_analysis_definition.csv"
        df_export = self.df_setup.copy()
        df_export.to_csv(output_path, index=True)
        return output_path

    def find_orphan_subanalysis_dirs(self) -> list[Path]:
        """Return sub-analysis directories on disk whose sa_id is absent from the current CSV.

        The authoritative set of expected sub-analysis directory names is derived
        from ``self.df_setup.index`` (the sensitivity CSV's ``sa_id`` column) and
        the ``self.sub_analyses_prefix`` constant. This ties orphan detection to
        the CSV directly, so a partially-constructed ``self.sub_analyses`` dict
        cannot cause legitimate directories to be misclassified as orphans.
        On-disk ``sa_*`` directories whose suffix fails the Snakemake-wildcard-safe
        charset ``^[A-Za-z0-9_.]+$`` are skipped — they were not created by this
        toolkit and must not be deleted by it. If ``self.subanalysis_dir`` does
        not exist, returns ``[]``.
        """
        import re as _re

        if not self.subanalysis_dir.exists():
            return []
        expected_names = {f"{self.sub_analyses_prefix}{sa_id}" for sa_id in self.df_setup.index.astype(str)}
        charset = _re.compile(r"^[A-Za-z0-9_.]+$")
        orphans: list[Path] = []
        for entry in self.subanalysis_dir.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(self.sub_analyses_prefix):
                continue
            suffix = entry.name[len(self.sub_analyses_prefix) :]
            if not charset.match(suffix):
                continue
            if entry.name not in expected_names:
                orphans.append(entry)
        return sorted(orphans)

    def cleanup_orphan_subanalysis_dirs(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> list[Path]:
        """Identify and optionally delete orphaned sub-analysis directories.

        Uses :meth:`find_orphan_subanalysis_dirs` to locate directories under
        ``subanalyses/`` whose ``sa_id`` no longer appears in the current CSV.

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
        from TRITON_SWMM_toolkit.utils import fast_rmtree

        orphans = self.find_orphan_subanalysis_dirs()
        if verbose:
            if orphans:
                print(f"[cleanup-orphans] Found {len(orphans)} orphan sub-analysis directories:", flush=True)
                for p in orphans:
                    print(f"  {p}", flush=True)
            else:
                print("[cleanup-orphans] No orphan sub-analysis directories found.", flush=True)
        if dry_run:
            return orphans
        if not force:
            raise ValueError(
                "cleanup_orphan_subanalysis_dirs called with dry_run=False but "
                "force=False. Pass force=True to perform deletion."
            )
        master_analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        deleted: list[Path] = []
        failed: list[tuple[Path, Exception]] = []
        for p in orphans:
            if verbose:
                print(f"[cleanup-orphans] Deleting {p}", flush=True)
            try:
                fast_rmtree(p, analysis_dir=master_analysis_dir)  # PATTERN A
                deleted.append(p)
            except Exception as exc:
                failed.append((p, exc))
                if verbose:
                    print(f"[cleanup-orphans] FAILED to delete {p}: {exc}", flush=True)
        if failed:
            summary = "; ".join(f"{p}: {exc}" for p, exc in failed)
            raise RuntimeError(
                f"cleanup_orphan_subanalysis_dirs deleted {len(deleted)} of {len(orphans)} orphans; failures: {summary}"
            )
        return deleted

    def find_orphan_status_flags(self) -> list[Path]:
        """Return _status/ flag files whose embedded sa_id is absent from df_setup.index.

        Matches against the four Snakemake rule-output flag families that embed
        an sa_id (verified against workflow.py rule generation):

        - ``b_prepare_sa-{sa_id}_evt-{event_id}_complete.flag``
        - ``c_run_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag``
        - ``d_process_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag``
        - ``e_consolidate_sa-{sa_id}_complete.flag``

        The sa_id charset is constrained to ``^[A-Za-z0-9_.]+$`` per the
        project stipulation. Returns an empty list if the ``_status/``
        directory does not exist.
        """
        import re as _re

        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return []
        expected_sa_ids = set(self.df_setup.index.astype(str))
        # Anchored to the four known rule-name prefixes so unrelated 'sa-'
        # substrings (or future non-sensitivity rules that happen to contain
        # 'sa-') cannot trigger a false orphan.
        pat = _re.compile(
            r"^(?:b_prepare|c_run_[A-Za-z0-9]+|d_process_[A-Za-z0-9]+|e_consolidate)_sa-([A-Za-z0-9_.]+?)(?:_evt-[A-Za-z0-9_.]+|_complete|)\.flag$"
        )
        orphans: list[Path] = []
        for entry in status_dir.glob("*.flag"):
            m = pat.match(entry.name)
            if m is None:
                continue
            sa_id = m.group(1)
            if sa_id not in expected_sa_ids:
                orphans.append(entry)
        return sorted(orphans)

    def find_orphan_datatree_groups(self) -> list[str]:
        """Return sa_id strings present as subgroups in sensitivity_datatree.zarr but absent from df_setup.index.

        Inspects on-disk subdirectories of ``sensitivity_datatree.zarr/`` matching
        ``{prefix}{sa_id}`` where ``prefix`` is ``self.sub_analyses_prefix``. Returns
        the sa_id strings (without prefix). Returns an empty list if the zarr
        does not exist.
        """
        zarr_path = self.analysis_paths.sensitivity_datatree_zarr
        if zarr_path is None or not zarr_path.exists():
            return []
        expected_sa_ids = set(self.df_setup.index.astype(str))
        prefix = self.sub_analyses_prefix
        orphans: list[str] = []
        for entry in zarr_path.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix):
                continue
            sa_id = entry.name[len(prefix) :]
            if sa_id and sa_id not in expected_sa_ids:
                orphans.append(sa_id)
        return sorted(orphans)

    def cleanup_all_orphans(
        self,
        dry_run: bool = True,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Detect and (optionally) delete orphan subanalysis dirs, status flags, and datatree groups.

        When any orphan is detected and deletion proceeds, the entire
        ``sensitivity_datatree.zarr`` is removed (rebuild approach — see plan
        D-SURGICAL) and the master-consolidation status flag is also removed so
        Snakemake re-runs the master_consolidation rule on the next workflow run.

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
            ``"datatree_groups"`` (list[str]), and (after deletion only)
            ``"sensitivity_datatree_removed"`` (bool) and
            ``"master_flag_removed"`` (bool) reporting whether the
            rebuild-trigger artifacts were actually removed.

        Raises
        ------
        ValueError
            If ``dry_run=False`` and ``force=False``.
        """
        from TRITON_SWMM_toolkit.utils import fast_rmtree

        result = {
            "dirs": self.find_orphan_subanalysis_dirs(),
            "status_flags": self.find_orphan_status_flags(),
            "datatree_groups": self.find_orphan_datatree_groups(),
        }
        any_orphan = bool(result["dirs"] or result["status_flags"] or result["datatree_groups"])
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
                for sa_id in result["datatree_groups"]:
                    print(f"  datatree-group: sa_{sa_id}", flush=True)
            else:
                print("[cleanup-orphans] No orphans detected.", flush=True)
        if dry_run:
            return result
        if not force:
            raise ValueError(
                "cleanup_all_orphans called with dry_run=False but force=False. Pass force=True to perform deletion."
            )
        master_analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        for p in result["dirs"]:
            if verbose:
                print(f"[cleanup-orphans] Deleting dir {p}", flush=True)
            fast_rmtree(p, analysis_dir=master_analysis_dir)  # PATTERN A
        for p in result["status_flags"]:
            if verbose:
                print(f"[cleanup-orphans] Unlinking flag {p}", flush=True)
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
                fast_rmtree(zarr_path, analysis_dir=master_analysis_dir)  # PATTERN A
                result["sensitivity_datatree_removed"] = True
            master_flag = self.analysis_paths.analysis_dir / "_status" / "f_consolidate_master_complete.flag"
            if master_flag.exists():
                if verbose:
                    print(
                        f"[cleanup-orphans] Unlinking master-consolidation flag {master_flag}",
                        flush=True,
                    )
                master_flag.unlink()
                result["master_flag_removed"] = True
        return result

    def _build_unique_system_targets(
        self,
        df_setup_full: pd.DataFrame,
        is_main_orchestrator: bool = True,
    ) -> list[UniqueSystemTarget]:
        """Resolve per-sa_id system targets and materialize per-target synthesized YAMLs.

        Handles three per-row mechanisms:

        1. ``system_config_yaml`` column (path to a per-sa system YAML).
        2. ``system.{field}`` overlay columns (Phase 1 prefixed-column mechanism).
        3. Neither — fall back to master ``self._system``.

        Mutual exclusion: a single row may use mechanism 1 OR mechanism 2, never both;
        violation raises ``ConfigurationError``.

        ``is_main_orchestrator=True`` purges ``_generated/`` before emission.
        Runner subprocesses pass ``False`` to skip the purge.
        """
        import pydantic

        from TRITON_SWMM_toolkit.config.system import system_config
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.utils import fast_rmtree

        sensitivity_csv = self.master_analysis.cfg_analysis.sensitivity_analysis
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        generated_dir = analysis_dir / "_generated"

        if is_main_orchestrator:
            fast_rmtree(generated_dir, missing_ok=True)
            generated_dir.mkdir(parents=True, exist_ok=True)

        has_yaml_col = "system_config_yaml" in df_setup_full.columns
        overlay_col_names = sorted(c for c in df_setup_full.columns if _is_system_overlay_column(c))

        # Group sub-analyses by their compile-key tuple.
        groups: dict[tuple, dict] = {}

        for sa_id, row in df_setup_full.iterrows():
            sa_id_str = str(sa_id)
            yaml_cell = row.get("system_config_yaml") if has_yaml_col else None
            yaml_specified = yaml_cell is not None and not pd.isna(yaml_cell) and str(yaml_cell).strip() != ""
            overlay_cells = {_strip_system_prefix(c): row[c] for c in overlay_col_names if not pd.isna(row[c])}
            if overlay_cells and yaml_specified:
                raise ConfigurationError(
                    field=f"sensitivity_analysis.row[{sa_id_str}]",
                    message=(
                        f"sa_id={sa_id_str}: row specifies both system_config_yaml "
                        f"({yaml_cell}) and system.* overlay column(s) "
                        f"{sorted(overlay_cells)}; mutually exclusive — "
                        f"use one mechanism per row."
                    ),
                    config_path=sensitivity_csv,
                )

            if overlay_cells:
                try:
                    cfg = system_config.model_validate(
                        {
                            **self._system.cfg_system.model_dump(),
                            **overlay_cells,
                        }
                    )
                except pydantic.ValidationError as exc:
                    raise ConfigurationError(
                        field=f"sensitivity_analysis.row[{sa_id_str}]",
                        message=(
                            f"sa_id={sa_id_str}: system.* overlay-column values failed SystemConfig validation: {exc}"
                        ),
                        config_path=sensitivity_csv,
                    ) from exc
            elif yaml_specified:
                yaml_path = Path(yaml_cell).resolve()
                if not yaml_path.is_file():
                    raise ConfigurationError(
                        field="sensitivity_analysis.system_config_yaml",
                        message=(f"sa_id={sa_id_str}: system_config_yaml does not exist at {yaml_path}."),
                        config_path=sensitivity_csv,
                    )
                cfg = TRITONSWMM_system(yaml_path).cfg_system
            else:
                cfg = self._system.cfg_system

            key = (
                cfg.target_dem_resolution,
                cfg.gpu_hardware,
                cfg.gpu_compilation_backend,
            )
            if key not in groups:
                groups[key] = {"cfg": cfg, "sa_ids": []}
            groups[key]["sa_ids"].append(sa_id_str)

        targets: list[UniqueSystemTarget] = []
        for target_id, group in enumerate(groups.values()):
            cfg = group["cfg"]
            sa_ids = group["sa_ids"]
            generated_yaml = generated_dir / f"target_{target_id}.yaml"
            if is_main_orchestrator:
                # Temp-file-rename for atomicity (PID-keyed per Gotcha 17 pattern).
                tmp_yaml = generated_dir / f"target_{target_id}.{os.getpid()}.tmp.yaml"
                with tmp_yaml.open("w") as fh:
                    yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
                tmp_yaml.rename(generated_yaml)
            # Reuse master self._system when the resolved cfg matches; avoids re-running
            # _check_paths_exist against possibly-HPC paths in the synthesized YAML.
            if cfg.model_dump_json() == self._system.cfg_system.model_dump_json():
                target_system = self._system
            else:
                target_system = TRITONSWMM_system(generated_yaml)
            targets.append(
                UniqueSystemTarget(
                    target_id=target_id,
                    system_config_yaml=generated_yaml,
                    system=target_system,
                    sub_analysis_ids=sa_ids,
                )
            )

        return targets

    def _create_sub_analyses(self):
        sa_id_to_system: dict = {}
        for target in self.unique_system_targets:
            for sa_id in target.sub_analysis_ids:
                sa_id_to_system[sa_id] = target.system

        from TRITON_SWMM_toolkit.config.analysis import analysis_config

        dic_sensitivity_analyses = dict()
        for idx, row in self.df_setup.iterrows():
            sa_id = str(idx)
            overlay_cells: dict = {}
            for k, v in row.items():
                if pd.isna(v):
                    continue
                if _is_analysis_overlay_column(k):
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
                    # `sa_id`/`system_config_yaml`/`system.*` already.
                    raise ConfigurationError(
                        field="sensitivity_analysis.unknown_column",
                        message=f"Column `{k}` is not a recognized analysis-config field.",
                    )
            cfg_snstvty_analysis = analysis_config.model_validate(
                {
                    **self.master_analysis.cfg_analysis.model_dump(),
                    **overlay_cells,
                }
            )
            analysis_id = f"{self.sub_analyses_prefix}{sa_id}"
            cfg_snstvty_analysis.analysis_id = analysis_id  # type: ignore
            sub_analysis_directory = self.subanalysis_dir / str(cfg_snstvty_analysis.analysis_id)
            sub_analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False
            cfg_snstvty_analysis.is_subanalysis = True

            cfg_anlysys_yaml = sub_analysis_directory / f"{analysis_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = sub_analysis_directory

            cfg_snstvty_analysis.master_analysis_cfg_yaml = self.master_analysis.analysis_config_yaml

            # Atomic write via temp-file + rename. `Path.write_text` truncates the
            # target before writing; concurrent readers in other plot subprocesses
            # would catch the truncated-empty state and fail with
            # `model_validate(None)`. POSIX `rename(2)` (Path.replace) is atomic
            # on the same filesystem, so readers always see a complete file.
            #
            # Temp filename is keyed on PID so concurrent writers do not collide
            # on the same `*.tmp` path (one writer's `replace()` would otherwise
            # move the tmp out from under another, raising FileNotFoundError on
            # the second writer's replace). PID is unique per OS process within
            # a job's lifetime; subprocess A and subprocess B always write to
            # distinct tmp files before swapping into the target.
            #
            # Once the deeper fix lands (sub-analysis yaml materialization lifted
            # out of `__init__` into the setup phase), this temp-file dance can
            # collapse back to a single `cfg_anlysys_yaml.write_text(...)`.
            _tmp = cfg_anlysys_yaml.with_suffix(cfg_anlysys_yaml.suffix + f".{os.getpid()}.tmp")
            _tmp.write_text(
                yaml.safe_dump(
                    cfg_snstvty_analysis.model_dump(mode="json"),
                    sort_keys=False,
                )
            )
            _tmp.replace(cfg_anlysys_yaml)
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=sa_id_to_system[sa_id],
            )
            dic_sensitivity_analyses[sa_id] = anlsys
        return dic_sensitivity_analyses

    def _compute_sa_id_fingerprint_payload(self, sub_analysis: "anlysis.TRITONSWMM_analysis") -> dict[str, object]:
        """Compute the deterministic fingerprint payload for one sub-analysis.

        Projects the sub-analysis's post-Pydantic ``analysis_config.model_dump(mode="json")``
        onto the canonical field names from ``self.analysis_independent_vars``
        (sorted). Adds a ``__schema_version__`` sentinel so future
        serializer-format changes are themselves observable. Excludes ``sa_id``
        (the path already disambiguates).

        Stability contract: every ``analysis_config`` field that may appear in
        ``self.analysis_independent_vars`` must be JSON-stable under ``model_dump(mode="json")``
        — that is, two invocations on the same sub_analysis instance must produce
        byte-identical ``json.dumps(..., sort_keys=True)`` output. The currently-known
        sensitivity-CSV columns (``cpus_per_sim``, ``n_omp_threads``, ``hpc_total_nodes``,
        ``hpc_max_simultaneous_sims``, ``hpc_total_job_duration_min``, ``run_mode``) are
        all native Python int/str/Literal types and meet the contract. Adding a
        new ``analysis_config`` field that may legitimately become a sensitivity-CSV
        column requires re-checking JSON stability and may require bumping
        ``__schema_version__``.

        Returns a plain dict suitable for ``json.dumps`` with ``sort_keys=True``.
        """
        cfg_dump = sub_analysis.cfg_analysis.model_dump(mode="json")
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
        # schema and attach a SHA-1 of the sub-analysis's resolved cfg_system. This
        # invalidates any sa_id whose system config changes between runs. The
        # schema bump intentionally invalidates every sa_id on the first run that
        # introduces per-sa system configs (Gotcha 17 cascade — see Phase 1 doc).
        if self._has_per_sa_system_configs:
            payload["__schema_version__"] = 2
            cfg_system_json = sub_analysis._system.cfg_system.model_dump_json(by_alias=False, exclude_none=False)
            payload["system_cfg_hash"] = hashlib.sha1(cfg_system_json.encode("utf-8")).hexdigest()

        # Phase 1 — attach system_overlay key when any system.* overlay columns
        # are declared on the master sensitivity df (un-projected).
        from TRITON_SWMM_toolkit.config.system import system_config

        df = self.master_analysis.sensitivity._df_setup_full
        overlay_col_names = [c for c in df.columns if _is_system_overlay_column(c)]
        if overlay_col_names:
            sa_id_str = sub_analysis.cfg_analysis.analysis_id.removeprefix(
                self.master_analysis.sensitivity.sub_analyses_prefix
            )
            overlay_cells = {
                _strip_system_prefix(c): df.loc[sa_id_str, c]
                for c in overlay_col_names
                if not pd.isna(df.loc[sa_id_str, c])
            }
            if overlay_cells:
                resolved = system_config.model_validate(
                    {
                        **self.master_analysis._system.cfg_system.model_dump(),
                        **overlay_cells,
                    }
                )
                resolved_overlay = {k: resolved.model_dump(mode="json")[k] for k in overlay_cells}
            else:
                resolved_overlay = {}
            payload["__schema_version__"] = 3
            payload["system_overlay"] = resolved_overlay
        return payload

    def _write_sa_id_fingerprint(
        self,
        sub_analysis: "anlysis.TRITONSWMM_analysis",
        fingerprint_path: Path,
    ) -> bool:
        """Write the per-sa_id fingerprint file via compare-and-write.

        Reads ``fingerprint_path`` if it exists, serializes the new payload with
        ``sort_keys=True`` and stable separators, and only rewrites the file when
        content differs. This preserves mtime when content is unchanged — the
        mechanism on which Snakemake's per-rule rerun gating depends.

        Returns ``True`` if the file was (re)written, ``False`` if skipped because
        content matched the existing file.
        """
        payload = self._compute_sa_id_fingerprint_payload(sub_analysis)
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
        In the no-per-sub-analysis-config case the list contains a single target
        wrapping the master system, so this method is the unified entry point for
        non-Snakemake direct execution regardless of whether per-sa configs are used.
        """
        for target in self.unique_system_targets:
            if verbose:
                print(
                    f"[Setup] Processing target {target.target_id} ({len(target.sub_analysis_ids)} sub-analyses)",
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
        self._update_master_analysis_log()

    def compile_TRITON_SWMM_for_sensitivity_analysis(
        self,
        verbose: bool = False,
        recompile_if_already_done_successfully: bool = False,
    ):
        for target in self.unique_system_targets:
            target.system.compile_TRITON_SWMM(
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                verbose=verbose,
            )
        self._update_master_analysis_log()
        return

    @property
    def scenarios_not_created(self):
        scenarios_not_created = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                if scen.log.scenario_creation_complete.get() is not True:
                    scenarios_not_created.append(str(scen.log.logfile.parent))
        return scenarios_not_created

    @property
    def scenarios_not_run(self):
        scens_not_run = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                # Check if all enabled models completed
                enabled_models = scen.run.model_types_enabled
                all_models_completed = all(scen.model_run_completed(model_type) for model_type in enabled_models)
                if not all_models_completed:
                    scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def classify_incomplete_sim_failures(self) -> dict[str, str]:
        """Scan model logs for all incomplete simulations across sub-analyses and classify each failure.

        Aggregates ``_classify_model_log_failure()`` across all sub-analyses.
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
        for sa_id, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                enabled_models = scen.run.model_types_enabled
                for model_type in enabled_models:
                    if not scen.model_run_completed(model_type):
                        event_id = scen.event_id
                        key = f"sa-{sa_id}_evt-{event_id}"
                        results[key] = scen.run._classify_model_log_failure(model_type)
        return results

    @property
    def is_timeout_only_failure(self) -> bool:
        """True iff all incomplete simulations across sub-analyses have timeout-classified failures.

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
        Get status DataFrame for all scenarios across all sub-analyses.

        Returns
        -------
        pd.DataFrame
            Concatenated status table from all sub-analyses. This includes
            sub-analysis-specific setup columns plus the canonical status
            schema from ``TRITONSWMM_analysis.df_status`` (e.g. ``scenario_setup``
            and ``run_completed``), as well as:

            - sub_analysis_iloc: int - Sub-analysis index
        """
        status_frames = []

        for sa_id, sub_analysis in self.sub_analyses.items():
            assert sub_analysis.cfg_analysis.is_subanalysis, (
                "is_subanalysis attribute not true in sub_analysis.cfg_analysis.is_subanalysis"
            )
            sub_df_status = sub_analysis.df_status.copy()

            setup_row = self.df_setup.loc[sa_id, :]
            for key, val in setup_row.items():
                sub_df_status[key] = val

            sub_df_status["sa_id"] = sa_id
            sub_df_status = sub_df_status[["sa_id"] + [c for c in sub_df_status.columns if c != "sa_id"]]

            status_frames.append(sub_df_status)

        if len(status_frames) == 0:
            return pd.DataFrame()

        return pd.concat(status_frames, ignore_index=True)

    @property
    def all_scenarios_created(self):
        """
        Check if all scenarios across all sub-analyses have been created.

        Returns
        -------
        bool
            True if all scenarios in all sub-analyses are created successfully
        """
        all_scenarios_created = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_scenarios_created = all_scenarios_created and sub_analysis.log.all_scenarios_created.get()
        return all_scenarios_created is True

    @property
    def all_sims_run(self):
        """
        Check if all simulations across all sub-analyses have completed.

        Returns
        -------
        bool
            True if all simulations in all sub-analyses completed successfully
        """
        all_sims_run = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_sims_run = all_sims_run and sub_analysis.log.all_sims_run.get()
        return all_sims_run is True

    @property
    def all_TRITON_timeseries_processed(self):
        """
        Check if all TRITON timeseries across all sub-analyses have been processed.

        Returns
        -------
        bool
            True if all TRITON outputs in all sub-analyses are processed
        """
        all_TRITON_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_TRITON_timeseries_processed = (
                all_TRITON_timeseries_processed and sub_analysis.log.all_TRITON_timeseries_processed.get()
            )
        return all_TRITON_timeseries_processed is True

    @property
    def all_SWMM_timeseries_processed(self):
        """
        Check if all SWMM timeseries across all sub-analyses have been processed.

        Returns
        -------
        bool
            True if all SWMM outputs in all sub-analyses are processed
        """
        all_SWMM_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_SWMM_timeseries_processed = (
                all_SWMM_timeseries_processed and sub_analysis.log.all_SWMM_timeseries_processed.get()
            )
        return all_SWMM_timeseries_processed is True

    @property
    def all_TRITONSWMM_performance_timeseries_processed(self):
        """
        Check if all performance timeseries across all sub-analyses have been processed.

        Returns
        -------
        bool
            True if all performance outputs in all sub-analyses are processed
        """
        all_TRITONSWMM_performance_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_TRITONSWMM_performance_timeseries_processed = (
                all_TRITONSWMM_performance_timeseries_processed
                and sub_analysis.log.all_TRITONSWMM_performance_timeseries_processed.get()
            )
        return all_TRITONSWMM_performance_timeseries_processed is True

    @property
    def TRITONSWMM_performance_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.TRITONSWMM_performance_time_series_not_processed
        return lst_scens

    @property
    def TRITON_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.TRITON_time_series_not_processed
        return lst_scens

    @property
    def SWMM_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.SWMM_time_series_not_processed
        return lst_scens

    @property
    def all_raw_TRITON_outputs_cleared(self):
        """
        Check if all raw TRITON outputs across all sub-analyses have been cleared.

        Returns
        -------
        bool
            True if all raw TRITON outputs in all sub-analyses are cleared
        """
        all_raw_TRITON_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_TRITON_outputs_cleared = (
                all_raw_TRITON_outputs_cleared and sub_analysis.log.all_raw_TRITON_outputs_cleared.get()
            )
        return all_raw_TRITON_outputs_cleared is True

    @property
    def all_raw_SWMM_outputs_cleared(self):
        """
        Check if all raw SWMM outputs across all sub-analyses have been cleared.

        Returns
        -------
        bool
            True if all raw SWMM outputs in all sub-analyses are cleared
        """
        all_raw_SWMM_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_SWMM_outputs_cleared = (
                all_raw_SWMM_outputs_cleared and sub_analysis.log.all_raw_SWMM_outputs_cleared.get()
            )
        return all_raw_SWMM_outputs_cleared is True

    def _update_master_analysis_log(self):
        self.master_analysis._update_log()
        return
