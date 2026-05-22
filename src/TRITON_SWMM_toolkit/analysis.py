# %%

import math
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from TRITON_SWMM_toolkit.config.analysis import ClearRawValue, ForceRerunValue
from TRITON_SWMM_toolkit.config.loaders import load_analysis_config
from TRITON_SWMM_toolkit.execution import (
    LocalConcurrentExecutor,
    SerialExecutor,
    SlurmExecutor,
)
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log
from TRITON_SWMM_toolkit.paths import AnalysisPaths
from TRITON_SWMM_toolkit.plot_analysis import TRITONSWMM_analysis_plotting
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing
from TRITON_SWMM_toolkit.processing_analysis import TRITONSWMM_analysis_post_processing
from TRITON_SWMM_toolkit.resource_management import ResourceManager
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from TRITON_SWMM_toolkit.snakemake_dry_run_report import (
    generate_dry_run_report_markdown,
)
from TRITON_SWMM_toolkit.snakemake_snakefile_parsing import (
    parse_regular_workflow_model_allocations,
    parse_sensitivity_analysis_workflow_model_allocations,
)
from TRITON_SWMM_toolkit.swmm_output_parser import (
    retrieve_swmm_performance_stats_from_rpt,
)
from TRITON_SWMM_toolkit.utils import fast_rmtree, parse_triton_log_file
from TRITON_SWMM_toolkit.validation import ValidationResult, preflight_validate
from TRITON_SWMM_toolkit.workflow import (
    SnakemakeDiagnostics,
    SnakemakeWorkflowBuilder,
    _emit_report_artifacts,
)

if TYPE_CHECKING:
    from .config.globus import PostRunTransferConfig
    from .orchestration import WorkflowResult, WorkflowStatus
    from .system import TRITONSWMM_system
    from .workflow import ResolvedForceRerunSpec  # noqa: F401

# All variable names present in TRITON performance.txt files.
# Used to build all-None dicts for model types with no performance dataset (SWMM).
PERF_VARS: list[str] = [
    "Compute",
    "MPI",
    "IO",
    "Resize",
    "SWMM",
    "Other",
    "Simulation",
    "Init",
    "Total",
]

# Display order for perf_* columns in df_status / scenario_status.csv.
# Total is first for quick scanning; breakdown follows in descending importance.
PERF_VARS_ORDERED: list[str] = [
    "Total",
    "Compute",
    "SWMM",
    "MPI",
    "Simulation",
    "IO",
    "Resize",
    "Other",
    "Init",
]


class TRITONSWMM_analysis:
    def __init__(
        self,
        analysis_config_yaml: Path,
        system: "TRITONSWMM_system",
        skip_log_update: bool = False,
        verbose: bool = True,
        is_main_orchestrator: bool = True,
    ) -> None:
        """
        Initialize a TRITON-SWMM analysis orchestrator.

        This class manages the complete lifecycle of a TRITON-SWMM analysis including
        scenario preparation, simulation execution, output processing, and result
        consolidation. It supports multiple execution strategies (serial, local
        concurrent, SLURM) and workflow management via Snakemake.

        Parameters
        ----------
        analysis_config_yaml : Path
            Path to the analysis configuration YAML file
        system : TRITONSWMM_system
            The TRITON-SWMM system object containing system configuration
        skip_log_update : bool, optional
            If True, skip initial log update (default: False)
        verbose : bool, optional
            If True, print a resume status summary when prior ``_status/`` flags
            are detected (default: True)
        """
        self._system = system
        self.analysis_config_yaml = analysis_config_yaml
        cfg_analysis = load_analysis_config(analysis_config_yaml)
        self.cfg_analysis = cfg_analysis
        if cfg_analysis.analysis_dir:
            analysis_dir = cfg_analysis.analysis_dir
        else:
            analysis_dir = self._system.cfg_system.system_directory / self.cfg_analysis.analysis_id

        ext = self.cfg_analysis.target_processed_output_type
        cfg_sys = self._system.cfg_system

        analysis_log_directory = analysis_dir / "logs"
        simlog_directory = analysis_log_directory / "sims"

        analysis_paths_kwargs = dict(
            f_log=analysis_dir / "log.json",
            analysis_dir=analysis_dir,
            simulation_directory=analysis_dir / "sims",
            simlog_directory=simlog_directory,
            analysis_log_directory=analysis_log_directory,
        )

        # TRITON-SWMM coupled model consolidated outputs
        if cfg_sys.toggle_tritonswmm_model:
            analysis_paths_kwargs["output_tritonswmm_triton_summary"] = analysis_dir / f"TRITONSWMM_TRITON.{ext}"
            analysis_paths_kwargs["output_tritonswmm_node_summary"] = analysis_dir / f"TRITONSWMM_SWMM_nodes.{ext}"
            analysis_paths_kwargs["output_tritonswmm_link_summary"] = analysis_dir / f"TRITONSWMM_SWMM_links.{ext}"
            analysis_paths_kwargs["output_tritonswmm_performance_summary"] = (
                analysis_dir / f"TRITONSWMM_performance.{ext}"
            )

        # TRITON-only consolidated outputs
        if cfg_sys.toggle_triton_model:
            analysis_paths_kwargs["output_triton_only_summary"] = analysis_dir / f"TRITON_only.{ext}"
            analysis_paths_kwargs["output_triton_only_performance_summary"] = (
                analysis_dir / f"TRITON_only_performance.{ext}"
            )

        # SWMM-only consolidated outputs
        if cfg_sys.toggle_swmm_model:
            analysis_paths_kwargs["output_swmm_only_node_summary"] = analysis_dir / f"SWMM_only_nodes.{ext}"
            analysis_paths_kwargs["output_swmm_only_link_summary"] = analysis_dir / f"SWMM_only_links.{ext}"

        # Hierarchical DataTree consolidation (Phase 2)
        analysis_paths_kwargs["analysis_datatree_zarr"] = analysis_dir / "analysis_datatree.zarr"

        # Sensitivity-level DataTree zarr (Phase 3) — aggregates sub-analyses.
        if cfg_analysis.toggle_sensitivity_analysis:
            analysis_paths_kwargs["sensitivity_datatree_zarr"] = analysis_dir / "sensitivity_datatree.zarr"

        self.analysis_paths = AnalysisPaths(**analysis_paths_kwargs)

        self.df_sims = pd.read_csv(self.cfg_analysis.weather_events_to_simulate).loc[
            :, self.cfg_analysis.weather_event_indices
        ]
        self._sim_run_objects: dict = {}
        self._sim_run_processing_objects: dict = {}
        self.backend = "gpu" if self.cfg_analysis.run_mode == "gpu" else "cpu"

        # self._system.compilation_successful = False
        self.in_slurm = "SLURM_JOB_ID" in os.environ.copy() or (
            cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks"
        )
        self._execution_strategy = self._select_execution_strategy()
        if self.cfg_analysis.python_path is not None:
            python_executable = str(self.cfg_analysis.python_path)
        else:
            python_executable = "python"
        self._python_executable = python_executable
        self._workflow_builder = SnakemakeWorkflowBuilder(self)
        self.process = TRITONSWMM_analysis_post_processing(self)
        self.plot = TRITONSWMM_analysis_plotting(self)
        self.nsims = len(self.df_sims)

        if self.cfg_analysis.toggle_sensitivity_analysis is True:
            self.sensitivity = TRITONSWMM_sensitivity_analysis(self, is_main_orchestrator=is_main_orchestrator)
            self.nsims *= len(self.sensitivity.df_setup)
        if not skip_log_update:
            # self._add_all_scenarios()
            self._refresh_log()

            # Record available backends at analysis creation time
            self.log.cpu_backend_available.set(self._system.compilation_cpu_successful)
            self.log.gpu_backend_available.set(self._system.compilation_gpu_successful)

            self._update_log()
        self._resource_manager = ResourceManager(self)
        if verbose:
            self._print_resume_status()

    def _print_resume_status(self) -> None:
        """Print a resume status summary if prior _status/ flags are detected.

        Fires at the end of ``__init__()`` when ``verbose=True``. Skips silently
        on first runs (no flags present). For ``1_job_many_srun_tasks`` analyses
        with incomplete sims, also prints a node recommendation.
        """
        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists() or not any(status_dir.glob("*.flag")):
            return  # first run — no flags yet

        # Determine primary model type for counting c_run_* flags
        cfg_sys = self._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            primary_model_type = "tritonswmm"
        elif cfg_sys.toggle_triton_model:
            primary_model_type = "triton"
        else:
            primary_model_type = "swmm"

        # Count completed simulations via glob — fast, no scenario instantiation
        if self.cfg_analysis.toggle_sensitivity_analysis:
            sim_flags = list(status_dir.glob(f"c_run_{primary_model_type}_sa*_complete.flag"))
        else:
            sim_flags = list(status_dir.glob(f"c_run_{primary_model_type}_*_complete.flag"))

        total_sims = self.nsims
        n_complete = len(sim_flags)
        n_incomplete = total_sims - n_complete

        analysis_id = self.cfg_analysis.analysis_id
        print(f"[Analysis] Resuming {analysis_id} — {n_complete}/{total_sims} sims complete.", flush=True)

        if n_incomplete == 0:
            return

        # Node recommendation — only for 1_job_many_srun_tasks
        if self.cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks":
            # Compute per-sim node requirement for each incomplete sub-analysis
            failures = self.classify_incomplete_sim_failures()
            if self.cfg_analysis.toggle_sensitivity_analysis:
                incomplete_nodes: list[int] = []
                incomplete_sa_ids = {re.search(r"sa-(.+?)_evt-", k).group(1) for k in failures}
                for sa_id in incomplete_sa_ids:
                    sa = self.sensitivity.sub_analyses[sa_id]
                    n_gpus = sa.cfg_analysis.n_gpus or 0
                    gpus_per_node = sa.cfg_analysis.hpc_gpus_per_node or 1
                    if n_gpus > 0:
                        nodes = math.ceil(n_gpus / gpus_per_node)
                    else:
                        nodes = sa.cfg_analysis.n_nodes or 1
                    incomplete_nodes.append(nodes)
                max_per_sim_nodes = max(incomplete_nodes) if incomplete_nodes else 1
                recommended_nodes = max_per_sim_nodes
            else:
                n_nodes = self.cfg_analysis.n_nodes or 1
                max_per_sim_nodes = n_nodes
                recommended_nodes = n_incomplete * n_nodes

            current_nodes = self.cfg_analysis.hpc_total_nodes
            print(
                f"[Analysis] Node recommendation for re-run:\n"
                f"  Max per-sim nodes (across incomplete sims): {max_per_sim_nodes}\n"
                f"  Recommended override_hpc_total_nodes={recommended_nodes}\n"
                f"  (Current hpc_total_nodes={current_nodes})",
                flush=True,
            )

            if failures:
                if self.is_timeout_only_failure:
                    print("[Analysis] All failures are SLURM time limits — increase --time and re-run.", flush=True)
                else:
                    print(
                        "[Analysis] Some failures are not time limits — see debugging docs for root cause.",
                        flush=True,
                    )

    def _enumerate_stale_metadata_paths(self) -> list[str]:
        """Return Snakemake-output-path strings whose ``.snakemake/metadata/``
        records are known stale due to past rule-output renames.

        Currently enumerates the four Phase 8 rule-rename orphans:

        - ``plots/system_overview.png``
        - ``plots/per_sim/{event_id}/peak_flood_depth.png`` (one per event_iloc)
        - ``plots/per_sim/{event_id}/conduit_flow.png`` (one per event_iloc)
        - ``plots/sensitivity/benchmarking/{independent_var}_vs_total.svg``
          (one per ``sensitivity.independent_vars`` when sensitivity is enabled
          at the master analysis level)

        The enumeration is deterministic — paths are constructed from
        ``self.df_sims.index`` (event_ilocs) plus the canonical event-id
        slug (``compute_event_id_slug``) and, when sensitivity is enabled,
        ``self.sensitivity.independent_vars``. No filesystem inspection;
        Snakemake's ``cleanup_metadata`` is idempotent on non-existent records.

        Returns paths as strings (relative to ``analysis_dir``).
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        orphans: list[str] = ["plots/system_overview.png"]
        for event_iloc in self.df_sims.index:
            ev = self._retrieve_weather_indexer_using_integer_index(event_iloc)
            event_id = compute_event_id_slug(ev)
            orphans.append(f"plots/per_sim/{event_id}/peak_flood_depth.png")
            orphans.append(f"plots/per_sim/{event_id}/conduit_flow.png")
        if (
            self.cfg_analysis.toggle_sensitivity_analysis
            and not self.cfg_analysis.is_subanalysis
        ):
            for ind_var in self.sensitivity.independent_vars:
                orphans.append(
                    f"plots/sensitivity/benchmarking/{ind_var}_vs_total.svg"
                )
        return orphans

    def _invoke_snakemake_cleanup_metadata(self, orphan_paths: list[str]) -> None:
        """Subprocess-invoke ``snakemake --cleanup-metadata`` against orphan paths.

        Snakemake's ``cleanup_metadata`` is idempotent (no-op without error on
        non-existent records), so passing paths that have no record on disk is
        safe — the cost is one subprocess call per ``analysis.run()`` when the
        gate fires.

        Raises ``WorkflowError`` on non-zero subprocess exit, capturing the
        last 50 lines of combined stdout+stderr in the ``stderr`` field.
        """
        import subprocess

        from TRITON_SWMM_toolkit.exceptions import WorkflowError

        cmd = [
            "snakemake",
            "--cleanup-metadata",
            *orphan_paths,
            "--directory",
            str(self.analysis_paths.analysis_dir),
            "--cores",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            combined = (result.stdout + "\n" + result.stderr)
            tail = "\n".join(combined.splitlines()[-50:])
            # Best-effort hygiene: "No Snakefile found" is a benign no-op for
            # analyses whose Snakefile has been removed or never written —
            # there is no metadata to interpret without a Snakefile, but
            # there is also no harm in skipping cleanup in that case.
            if "No Snakefile found" in combined:
                return
            raise WorkflowError(
                phase="cleanup_stale_metadata",
                return_code=result.returncode,
                stderr=(
                    f"snakemake --cleanup-metadata exit {result.returncode}; "
                    f"last 50 lines:\n{tail}"
                ),
            )

    def validate(self) -> ValidationResult:
        """Run preflight validation on system and analysis configurations.

        This method performs comprehensive validation of both system and analysis
        configurations before launching expensive simulation work. It checks:

        - System config: paths, toggle dependencies, model selection
        - Analysis config: weather data, run-mode consistency, HPC settings
        - Data consistency: event alignment, storm tide data, units

        Returns
        -------
        ValidationResult
            Validation result with any errors and warnings. Use result.is_valid
            to check if validation passed, or result.raise_if_invalid() to raise
            ConfigurationError if any errors exist.

        Examples
        --------
        >>> analysis = system.analysis
        >>> result = analysis.validate()
        >>> if not result.is_valid:
        >>>     print(result)  # Show all errors and warnings
        >>>     result.raise_if_invalid()  # Raise ConfigurationError

        >>> # Or validate and raise in one step:
        >>> analysis.validate().raise_if_invalid()

        Notes
        -----
        Validation is NOT automatically called in __init__ to avoid breaking
        existing workflows. Users should explicitly call validate() before
        launching simulations, or CLI/API entry points can call it automatically.
        """
        return preflight_validate(
            cfg_system=self._system.cfg_system,
            cfg_analysis=self.cfg_analysis,
        )

    def _refresh_log(self):
        if self.analysis_paths.f_log.exists():
            self.log = TRITONSWMM_analysis_log.from_json(self.analysis_paths.f_log)
        else:
            self.log = TRITONSWMM_analysis_log(logfile=self.analysis_paths.f_log)

    def _select_execution_strategy(self):
        """
        Select the appropriate execution strategy based on configuration.

        Returns
        -------
        ExecutionStrategy
            The appropriate executor (SerialExecutor, LocalConcurrentExecutor, or SlurmExecutor)
        """
        method = self.cfg_analysis.multi_sim_run_method
        if method == "1_job_many_srun_tasks":
            return SlurmExecutor(self)
        elif method == "local":
            return LocalConcurrentExecutor(self)
        else:
            # Default to serial execution for safety
            return SerialExecutor(self)

    def print_cfg(self, which: Literal["system", "analysis", "both"] = "both"):
        """
        Print configuration settings in tabular format.

        Parameters
        ----------
        which : Literal["system", "analysis", "both"], optional
            Which configuration to print (default: "both")
        """
        if which in ["system", "both"]:
            print("=== System Configuration ===", flush=True)
            self._system.cfg_system.display_tabulate_cfg()
        if which == "both":
            print("\n", flush=True)
        if which in ["analysis", "both"]:
            print("=== analysis Configuration ===", flush=True)
            self.cfg_analysis.display_tabulate_cfg()

    def globus_to_local(self, transfer_yaml: "Path") -> str:
        """Transfer HPC results to local machine via Globus.

        Args:
            transfer_yaml: Path to a transfer spec YAML in configs/transfers/.
                           See configs/transfers/template_transfer.yaml.

        Returns:
            Globus task ID. Pass to ``GlobusTransferManager().wait(task_id)``
            to block until complete, or monitor at app.globus.org.

        Example::

            task_id = analysis.globus_to_local(
                Path("configs/transfers/my_frontier_run.yaml")
            )
        """
        from pathlib import Path as _Path

        from TRITON_SWMM_toolkit.config.loaders import load_transfer_config
        from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager

        spec = load_transfer_config(_Path(transfer_yaml))
        manager = GlobusTransferManager(collection_uuids=[spec.endpoints.source_uuid])
        return manager.transfer(spec)

    def globus_to_hpc(self, transfer_yaml: "Path") -> str:
        """Transfer local inputs to HPC via Globus.

        Args:
            transfer_yaml: Path to a transfer spec YAML in configs/transfers/.

        Returns:
            Globus task ID.
        """
        from pathlib import Path as _Path

        from TRITON_SWMM_toolkit.config.loaders import load_transfer_config
        from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager

        spec = load_transfer_config(_Path(transfer_yaml))
        manager = GlobusTransferManager(collection_uuids=[spec.endpoints.destination_uuid])
        return manager.transfer(spec)

    def transfer_results(
        self,
        config: "PostRunTransferConfig",
    ) -> str:
        """Transfer analysis results to local machine via Globus.

        This is a standalone method — it does not require ``run()`` to have
        been called first.  Use it for the "submit on HPC, poll squeue,
        transfer when done" workflow.

        Args:
            config: User-facing transfer configuration.  See
                :class:`~TRITON_SWMM_toolkit.config.globus.PostRunTransferConfig`.

        Returns:
            Globus task ID.

        Raises:
            GlobusTransferError: If the transfer fails or is cancelled
                (only when ``config.wait_for_transfer`` is True).

        Example::

            from TRITON_SWMM_toolkit.config.globus import PostRunTransferConfig

            config = PostRunTransferConfig(
                destination_root=r"D:\\Dropbox\\_GradSchool\\repos\\TRITON-SWMM_toolkit\\frontier",
                system="frontier",
            )
            task_id = analysis.transfer_results(config)
        """
        from TRITON_SWMM_toolkit.config.globus import (
            _get_endpoint_uuids,
            _normalize_wsl_path,
        )
        from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager

        spec = config.to_transfer_spec(
            analysis_dir=self.analysis_paths.analysis_dir,
            analysis_id=self.cfg_analysis.analysis_id,
        )

        # Handle destination conflict
        dest_path = _normalize_wsl_path(config.destination_root).rstrip("/")
        dest_dir = Path(f"{dest_path}/{self.cfg_analysis.analysis_id}")
        if dest_dir.exists():
            self._handle_destination_conflict(dest_dir, config.conflict_policy)

        # Only pass collection_uuids for endpoints that need data_access consent;
        # pass session_required_domains for domain-restricted endpoints (e.g. OLCF).
        _uuid, _base, needs_data_access, session_domain = _get_endpoint_uuids(config.system)
        consent_uuids = [spec.endpoints.source_uuid] if needs_data_access else []
        session_domains = [session_domain] if session_domain else None
        manager = GlobusTransferManager(
            collection_uuids=consent_uuids,
            session_required_domains=session_domains,
        )
        task_id = manager.transfer(spec, exclude_dirs=config.exclude_patterns)

        if config.wait_for_transfer:
            manager.wait(task_id, timeout_minutes=config.timeout_minutes)

        return task_id

    # Conforms to TRITON_SWMM_toolkit.bundle._protocol.BundleableAnalysis
    # via duck typing (Protocol is structural; no registration needed).
    def bundle_report_data(
        self,
        output_path: "Path | None" = None,
    ) -> "Path":
        """Emit a portable render bundle for local renderer iteration.

        Opt-in only — NEVER invoked from analysis.run() or
        submit_workflow(). The bundle is a self-contained tar including
        every source path declared via prov.artist().add_channel(...)
        during the most recent render_report() execution, plus configs
        with relative paths, the Snakefile, and the HPC-baseline
        analysis_report.{html,zip} under bundle_baseline/.

        Args:
            output_path: Optional target path for the bundle tar.
                Defaults to
                {analysis_dir}/render_bundle/{analysis_id}_{git_sha}_v{schema}.tar.

        Returns:
            Path to the emitted bundle tar.

        Raises:
            FileNotFoundError: If render_report() has not been invoked
                on this analysis (no *.manifest.json sidecars exist).
        """
        from TRITON_SWMM_toolkit.bundle import emit_bundle

        return emit_bundle(self, output_path)

    @staticmethod
    def _handle_destination_conflict(
        dest_dir: Path,
        policy: str,
    ) -> None:
        """Handle an existing destination directory before transfer.

        Args:
            dest_dir: The local destination directory that already exists.
            policy: One of ``"prompt"``, ``"archive"``, ``"clear"``.

        Raises:
            ConfigurationError: If *policy* is ``"prompt"`` and stdin is
                not a TTY.
        """
        import shutil
        import sys

        from TRITON_SWMM_toolkit.exceptions import ConfigurationError

        if policy == "archive":
            import datetime

            suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = dest_dir.parent / "archived"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archived = archive_dir / f"{dest_dir.name}_{suffix}"
            print(
                f"[Transfer] Archiving existing destination → {archived}",
                flush=True,
            )
            shutil.move(str(dest_dir), str(archived))

        elif policy == "clear":
            print(
                f"[Transfer] Clearing existing destination: {dest_dir}",
                flush=True,
            )
            shutil.rmtree(dest_dir)

        elif policy == "prompt":
            if not sys.stdin.isatty():
                raise ConfigurationError(
                    field="conflict_policy",
                    message=(
                        "conflict_policy='prompt' requires an interactive terminal. "
                        "Use 'archive' or 'clear' for non-interactive contexts."
                    ),
                )
            print(
                f"\n[Transfer] Destination already exists: {dest_dir}",
                flush=True,
            )
            print("  (a) Archive to archived/ subfolder", flush=True)
            print("  (c) Clear and overwrite", flush=True)
            print("  (s) Skip — proceed with sync_level transfer", flush=True)
            choice = input("  Choice [a/c/s]: ").strip().lower()
            if choice == "a":
                TRITONSWMM_analysis._handle_destination_conflict(dest_dir, "archive")
            elif choice == "c":
                TRITONSWMM_analysis._handle_destination_conflict(dest_dir, "clear")
            # "s" or anything else: skip, let Globus handle via sync_level

    def print_all_yaml_defined_input_files(self):
        print_json_file_tree(self.dict_of_exp_and_sys_config())

    def dict_of_exp_and_sys_config(self):
        dic_exp = self._system.cfg_system.model_dump()
        dic_sys = self.cfg_analysis.model_dump()
        return dic_exp | dic_sys

    def dict_of_all_sim_files(self, event_iloc):
        dic_syspaths = self._system.sys_paths.as_dict()
        dic_analysis_paths = self.analysis_paths.as_dict()
        scen = TRITONSWMM_scenario(event_iloc, self)
        dic_sim_paths = scen.scen_paths.as_dict()
        dic_all_paths = dic_syspaths | dic_analysis_paths | dic_sim_paths
        return dic_all_paths

    def print_all_sim_files(self, event_iloc):
        dic_all_paths = self.dict_of_all_sim_files(event_iloc)
        print_json_file_tree(dic_all_paths)

    def _retrieve_weather_indexer_using_integer_index(
        self,
        event_iloc,
    ):
        row = self.df_sims.loc[event_iloc, self.cfg_analysis.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    @property
    def scenarios_not_created(self):
        """
        Get list of scenarios that have not been created successfully.

        Returns
        -------
        list of str
            Paths to scenario directories where creation is incomplete
        """
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.scenarios_not_created
        scens_not_created = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scen.log.refresh()
            if scen.log.scenario_creation_complete.get() is not True:
                scens_not_created.append(str(scen.log.logfile.parent))
        return scens_not_created

    @property
    def scenarios_not_run(self):
        """
        Get list of scenarios that have not been run successfully.

        Returns
        -------
        list of str
            Paths to scenario directories where simulation is incomplete
        """
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.scenarios_not_run
        scens_not_run = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            # Check if all enabled models completed for this scenario
            enabled_models = scen.run.model_types_enabled
            all_models_completed = all(scen.model_run_completed(model_type) for model_type in enabled_models)
            if not all_models_completed:
                scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def classify_incomplete_sim_failures(self) -> dict[str, str]:
        """Scan model logs for all incomplete simulations and classify each failure.

        Reads the analysis-level model log for each incomplete simulation and
        searches for known SLURM failure markers. Works for both
        ``"1_job_many_srun_tasks"`` and ``"batch_job"`` execution methods —
        the SLURM cancellation marker appears in the model log in both cases.

        Returns
        -------
        dict[str, str]
            Maps scenario identifier (e.g. ``"sa1_0"``) to failure class:

            - ``"timeout"`` — log contains ``DUE TO TIME LIMIT``
            - ``"unclassified"`` — log exists but no known failure marker found
            - ``"no_log"`` — model log file does not exist
        """
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.classify_incomplete_sim_failures()

        results: dict[str, str] = {}
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            enabled_models = scen.run.model_types_enabled
            for model_type in enabled_models:
                if not scen.model_run_completed(model_type):
                    key = f"{event_iloc}"
                    results[key] = scen.run._classify_model_log_failure(model_type)
        return results

    @property
    def is_timeout_only_failure(self) -> bool:
        """True iff all incomplete simulations have timeout-classified failures.

        Returns False if there are no incomplete sims (all done), or if any
        incomplete sim has an unclassified or no_log failure.
        """
        failures = self.classify_incomplete_sim_failures()
        if not failures:
            return False
        return all(v == "timeout" for v in failures.values())

    @property
    def all_scenarios_created(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.all_scenarios_created
        return bool(self.log.all_scenarios_created.get())

    @property
    def all_sims_run(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.all_sims_run
        return bool(self.log.all_sims_run.get())

    @property
    def all_TRITONSWMM_performance_timeseries_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.all_TRITONSWMM_performance_timeseries_processed
        return bool(self.log.all_TRITONSWMM_performance_timeseries_processed.get())

    @property
    def TRITONSWMM_performance_time_series_not_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.TRITONSWMM_performance_time_series_not_processed
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            # Check model-specific logs (race-condition free!)
            perf_ok = True
            if self._system.cfg_system.toggle_tritonswmm_model:
                log = scen.get_log("tritonswmm")
                perf_ok = perf_ok and bool(
                    log.performance_timeseries_written and log.performance_timeseries_written.get()
                )
            if self._system.cfg_system.toggle_triton_model:
                log = scen.get_log("triton")
                perf_ok = perf_ok and bool(
                    log.performance_timeseries_written and log.performance_timeseries_written.get()
                )
            if not perf_ok:
                scens_not_processed.append(str(scen.scen_paths.sim_folder))
        return scens_not_processed

    @property
    def all_SWMM_timeseries_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.all_SWMM_timeseries_processed
        # Uses model-specific logs - race-condition free!
        return len(self.SWMM_time_series_not_processed) == 0

    @property
    def TRITON_time_series_not_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.TRITON_time_series_not_processed
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            # Check model-specific logs (race-condition free!)
            triton_ok = True
            if self._system.cfg_system.toggle_tritonswmm_model:
                log = scen.get_log("tritonswmm")
                triton_ok = triton_ok and (log.TRITON_timeseries_written and bool(log.TRITON_timeseries_written.get()))
            if self._system.cfg_system.toggle_triton_model:
                log = scen.get_log("triton")
                triton_ok = triton_ok and (log.TRITON_timeseries_written and bool(log.TRITON_timeseries_written.get()))
            if not triton_ok:
                scens_not_processed.append(str(scen.scen_paths.sim_folder))
        return scens_not_processed

    @property
    def SWMM_time_series_not_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.SWMM_time_series_not_processed
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            # Check model-specific logs (race-condition free!)
            swmm_ok = True
            if self._system.cfg_system.toggle_tritonswmm_model:
                log = scen.get_log("tritonswmm")
                node_ok = log.SWMM_node_timeseries_written and bool(log.SWMM_node_timeseries_written.get())
                link_ok = log.SWMM_link_timeseries_written and bool(log.SWMM_link_timeseries_written.get())
                swmm_ok = swmm_ok and (node_ok and link_ok)
            if self._system.cfg_system.toggle_swmm_model:
                log = scen.get_log("swmm")
                node_ok = log.SWMM_node_timeseries_written and bool(log.SWMM_node_timeseries_written.get())
                link_ok = log.SWMM_link_timeseries_written and bool(log.SWMM_link_timeseries_written.get())
                swmm_ok = swmm_ok and (node_ok and link_ok)
            if not swmm_ok:
                scens_not_processed.append(str(scen.scen_paths.sim_folder))
        return scens_not_processed

    @property
    def all_TRITON_timeseries_processed(self):
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.all_TRITON_timeseries_processed
        # Uses model-specific logs - race-condition free!
        return len(self.TRITON_time_series_not_processed) == 0

    def _update_log(self):
        self._refresh_log()
        # dict_all_logs = {}
        all_scens_created = True
        all_sims_run = True
        all_TRITON_outputs_processed = True
        all_SWMM_outputs_processed = True
        all_TRITONSWMM_performance_outputs_processed = True
        all_raw_TRITON_outputs_cleared = True
        all_raw_SWMM_outputs_cleared = True
        if self.cfg_analysis.toggle_sensitivity_analysis is True:
            sens = self.sensitivity
            all_scens_created = sens.all_scenarios_created
            all_sims_run = sens.all_sims_run
            all_TRITON_outputs_processed = sens.all_TRITON_timeseries_processed
            all_SWMM_outputs_processed = sens.all_SWMM_timeseries_processed
            all_TRITONSWMM_performance_outputs_processed = sens.all_TRITONSWMM_performance_timeseries_processed
            all_raw_TRITON_outputs_cleared = sens.all_raw_TRITON_outputs_cleared
            all_raw_SWMM_outputs_cleared = sens.all_raw_SWMM_outputs_cleared
        else:
            for event_iloc in self.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, self)
                # sim run status - check if all enabled models completed
                enabled_models = scen.run.model_types_enabled
                scen_all_models_completed = all(scen.model_run_completed(model_type) for model_type in enabled_models)
                all_sims_run = all_sims_run and scen_all_models_completed

                # Scenario creation status comes exclusively from scenario prep log
                scen.log.refresh()
                scen_created = bool(scen.log.scenario_creation_complete.get())
                all_scens_created = all_scens_created and scen_created

                # Check output processing status for each enabled model
                for model_type in enabled_models:
                    model_log = scen.get_log(model_type)

                    # TRITON outputs (triton and tritonswmm models)
                    if model_type in ("triton", "tritonswmm"):
                        triton_ok = bool(
                            model_log.TRITON_timeseries_written and model_log.TRITON_timeseries_written.get()
                        )
                        all_TRITON_outputs_processed = all_TRITON_outputs_processed and triton_ok

                        perf_ok = bool(
                            model_log.performance_timeseries_written and model_log.performance_timeseries_written.get()
                        )
                        all_TRITONSWMM_performance_outputs_processed = (
                            all_TRITONSWMM_performance_outputs_processed and perf_ok
                        )

                        cleared = bool(
                            model_log.raw_TRITON_outputs_cleared and model_log.raw_TRITON_outputs_cleared.get()
                        )
                        all_raw_TRITON_outputs_cleared = all_raw_TRITON_outputs_cleared and cleared

                    # SWMM outputs (swmm and tritonswmm models)
                    if model_type in ("swmm", "tritonswmm"):
                        node_ok = bool(
                            model_log.SWMM_node_timeseries_written and model_log.SWMM_node_timeseries_written.get()
                        )
                        link_ok = bool(
                            model_log.SWMM_link_timeseries_written and model_log.SWMM_link_timeseries_written.get()
                        )
                        swmm_ok = node_ok and link_ok
                        all_SWMM_outputs_processed = all_SWMM_outputs_processed and swmm_ok

                        cleared = bool(model_log.raw_SWMM_outputs_cleared and model_log.raw_SWMM_outputs_cleared.get())
                        all_raw_SWMM_outputs_cleared = all_raw_SWMM_outputs_cleared and cleared
        self.log.all_scenarios_created.set(all_scens_created)
        self.log.all_sims_run.set(all_sims_run)
        self.log.all_TRITON_timeseries_processed.set(all_TRITON_outputs_processed)
        self.log.all_SWMM_timeseries_processed.set(all_SWMM_outputs_processed)
        self.log.all_TRITONSWMM_performance_timeseries_processed.set(all_TRITONSWMM_performance_outputs_processed)
        self.log.all_raw_TRITON_outputs_cleared.set(all_raw_TRITON_outputs_cleared)
        self.log.all_raw_SWMM_outputs_cleared.set(all_raw_SWMM_outputs_cleared)
        return

    def retrieve_prepare_scenario_launchers(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Create subprocess-based launchers for scenario preparation.

        Each launcher runs scenario preparation in an isolated subprocess to avoid
        PySwmm's MultiSimulationError when preparing multiple scenarios concurrently.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of launcher functions that execute scenario preparation in subprocesses
        """
        prepare_scenario_launchers = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)

            # Create a subprocess-based launcher
            launcher = scen._create_subprocess_prepare_scenario_launcher(
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                verbose=verbose,
            )
            prepare_scenario_launchers.append(launcher)

        return prepare_scenario_launchers

    def retrieve_scenario_timeseries_processing_launchers(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create subprocess-based launchers for scenario timeseries processing.

        Each launcher runs timeseries processing in an isolated subprocess to avoid
        potential conflicts when processing multiple scenarios' outputs concurrently.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process: TRITON, SWMM, or both
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw``. ``None`` (the default)
            reads from the YAML config; a concrete value overrides for this run.
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)

        Returns
        -------
        list
            List of launcher functions that execute timeseries processing in subprocesses
        """
        scenario_timeseries_processing_launchers = []
        for event_iloc in self.df_sims.index:
            proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)

            # Create a subprocess-based launcher
            launcher = proc._create_subprocess_timeseries_processing_launcher(
                which=which,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers.append(launcher)

        return scenario_timeseries_processing_launchers

    def calculate_effective_max_parallel(
        self,
        min_memory_per_function_MiB: int | None = 1024,
        max_concurrent: int | None = None,
        verbose: bool = False,
    ) -> int:
        """
        Calculate the effective maximum parallelism based on CPU, GPU, memory, and SLURM constraints.

        This method delegates to ResourceManager for resource allocation calculations.

        Parameters
        ----------
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        max_concurrent : int | None
            CPU-based upper bound on parallelism (e.g., based on cores/threads per task).
            If None, defaults to physical CPU count - 1 (or SLURM allocation if in SLURM).
        verbose : bool
            Print progress messages.

        Returns
        -------
        int
            The effective maximum number of parallel tasks.
        """
        return self._resource_manager.calculate_effective_max_parallel(
            min_memory_per_function_MiB=min_memory_per_function_MiB,
            max_concurrent=max_concurrent,
            verbose=verbose,
        )

    def run_python_functions_concurrently(
        self,
        function_launchers: list[Callable[[], None]],
        min_memory_per_function_MiB: int | None = 1024,
        max_parallel: int | None = None,
        verbose: bool = True,
    ) -> list[int]:
        """
        Run Python functions concurrently, limiting parallelism by CPU and memory.

        Parameters
        ----------
        function_launchers : List[Callable[[], None]]
            Functions to execute concurrently.
        max_parallel : int | None
            Upper bound on parallelism (defaults to CPU count).
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        verbose : bool
            Print progress messages.

        Returns
        -------
        List[int]
            Indices of functions that completed successfully.
        """

        effective_max_parallel = self.calculate_effective_max_parallel(
            min_memory_per_function_MiB=min_memory_per_function_MiB,
            max_concurrent=max_parallel,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Running {len(function_launchers)} functions (max parallel = {effective_max_parallel})",
                flush=True,
            )

        results: list[int] = []
        batch_start = time.time()  # Reference point for all tasks

        def wrapper(idx: int, launcher: Callable[[], None]):
            task_start = time.time()
            launcher()
            task_end = time.time()

            duration = task_end - task_start
            completed_at = task_end - batch_start

            if verbose:
                print(
                    f"Function {idx}: duration={duration:.2f}s, completed_at={completed_at:.2f}s",
                    flush=True,
                )
            return idx

        # ----------------------------
        # Execute
        # ----------------------------
        with ThreadPoolExecutor(max_workers=effective_max_parallel) as executor:
            futures = {executor.submit(wrapper, idx, launcher): idx for idx, launcher in enumerate(function_launchers)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    if verbose:
                        print(f"Function {idx} failed with error: {e}", flush=True)

        self._update_log()
        return results

    def run_prepare_scenarios_serially(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Prepare all scenarios sequentially.

        Executes scenario preparation for all scenarios in serial order, updating
        logs after each scenario completes.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool, optional
            If True, overwrite existing scenarios (default: False)
        rerun_swmm_hydro_if_outputs_exist : bool, optional
            If True, rerun SWMM hydrology model even if outputs exist (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        """
        prepare_scenario_launchers = self.retrieve_prepare_scenario_launchers(
            overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            verbose=verbose,
        )
        for launcher in prepare_scenario_launchers:
            launcher()
            self._update_log()  # update logs
        self._update_log()
        return

    def print_logfile_for_scenario(self, event_iloc):
        scen = TRITONSWMM_scenario(event_iloc, self)
        scen.log.print()

    def _get_enabled_model_types(self) -> list[str]:
        """
        Return enabled model types based on system toggles.

        Returns
        -------
        list[str]
            Enabled model types: "triton", "tritonswmm", and/or "swmm"
        """
        cfg_sys = self._system.cfg_system
        models = []
        if cfg_sys.toggle_triton_model:
            models.append("triton")
        if cfg_sys.toggle_tritonswmm_model:
            models.append("tritonswmm")
        if cfg_sys.toggle_swmm_model:
            models.append("swmm")
        return models

    def _retrieve_snakemake_allocations(
        self,
    ) -> tuple[dict[str, dict[str, int]], str | None]:
        """Retrieve parsed per-model Snakemake allocations.

        Routing is strict and context-aware:
        - regular analysis: parse `run_<model>` rules from this analysis Snakefile
        - sensitivity sub-analysis: parse `simulation_sa*_evt*` rules from the
          parent/master sensitivity Snakefile and select this sub-analysis id

        Raises
        ------
        FileNotFoundError
            If the workflow Snakefile does not exist.
        SnakefileParsingError
            If allocations cannot be parsed from the Snakefile.
        """
        enabled_models = self._get_enabled_model_types()

        if self.cfg_analysis.toggle_sensitivity_analysis:
            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            expected_sa_ids = sorted(self.sensitivity.sub_analyses.keys())
            sa_allocations = parse_sensitivity_analysis_workflow_model_allocations(
                snakefile_path=snakefile_path,
                expected_subanalysis_ids=expected_sa_ids,
            )
            allocations = {
                model_type: alloc.copy() for model_type in enabled_models for alloc in sa_allocations.values()
            }
        else:
            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            allocations = parse_regular_workflow_model_allocations(
                snakefile_path=snakefile_path,
                enabled_model_types=enabled_models,
            )

        return allocations, None

    def run_sim(
        self,
        event_iloc: int,
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool,
        which: Literal["TRITON", "SWMM", "both"],
        compression_level: int,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose=False,
        model_type: Literal["triton", "tritonswmm", "swmm"] = "tritonswmm",
    ):
        """
        Run a single simulation for the specified scenario.

        Executes the TRITON-SWMM simulation for a specific weather event scenario,
        optionally processing outputs after completion.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario in df_sims
        pickup_where_leftoff : bool
            If True, resume simulation from last checkpoint
        process_outputs_after_sim_completion : bool
            If True, process timeseries outputs after simulation completes
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process (only used if process_outputs_after_sim_completion=True)
        compression_level : int
            Compression level for output files, 0-9
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` (None reads from YAML).
        verbose : bool, optional
            If True, print progress messages (default: False)
        model_type : Literal["triton", "tritonswmm", "swmm"], optional
            Model type to run (default: "tritonswmm")

        Raises
        ------
        ValueError
            If scenario creation is incomplete or TRITONSWMM is not compiled
        """
        scen = TRITONSWMM_scenario(event_iloc, self)

        if not scen.log.scenario_creation_complete.get():
            print("Log file:", flush=True)
            print(scen.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        valid_types = ("triton", "tritonswmm", "swmm")
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {model_type}")

        if model_type == "triton":
            if not self._system.compilation_triton_only_successful:
                print("Log file:", flush=True)
                print(scen.log.print())
                raise ValueError("TRITON-only has not been compiled")
        elif model_type == "tritonswmm":
            if not self._system.compilation_successful:
                print("Log file:", flush=True)
                print(scen.log.print())
                raise ValueError("TRITONSWMM has not been compiled")
        elif model_type == "swmm":
            if not self._system.compilation_swmm_successful:
                print("Log file:", flush=True)
                print(scen.log.print())
                raise ValueError("SWMM has not been compiled")
        run = self._retrieve_sim_runs(event_iloc)
        if verbose:
            print("run instance instantiated", flush=True)

        self.analysis_paths.simlog_directory.mkdir(parents=True, exist_ok=True)
        # Use the subprocess launcher pattern, mirroring process_sim_timeseries
        launcher, finalize_sim = run._create_subprocess_sim_run_launcher(
            pickup_where_leftoff=pickup_where_leftoff,
            verbose=verbose,
            model_type=model_type,
        )
        # Launch the simulation (non-blocking)
        proc, start_time, sim_logfile, lf = launcher()
        # Wait for simulation to complete and update simlog
        finalize_sim(proc, start_time, sim_logfile, lf)

        # self._update_log()  # updates analysis log
        if process_outputs_after_sim_completion and run._scenario.model_run_completed(model_type):
            if model_type == "triton":
                outputs_to_process = "TRITON"
            elif model_type == "swmm":
                outputs_to_process = "SWMM"
            else:
                outputs_to_process = which
            self.process_sim_timeseries(
                event_iloc,
                outputs_to_process,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    def process_sim_timeseries(
        self,
        event_iloc,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Process and write timeseries outputs for a single simulation.

        Converts raw TRITON and/or SWMM outputs into processed timeseries files,
        optionally clearing raw outputs after processing.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario in df_sims
        which : Literal["TRITON", "SWMM", "both"], optional
            Which outputs to process (default: "both")
        override_clear_raw : ClearRawValue | None, optional
            Runtime override for ``cfg_analysis.clear_raw``. ``None`` (default)
            reads the YAML; a concrete value overrides for this invocation.
        verbose : bool, optional
            If True, print progress messages (default: False)
        compression_level : int, optional
            Compression level for output files, 0-9 (default: 5)
        """
        proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)
        proc.write_timeseries_outputs(
            which=which,
            override_clear_raw=override_clear_raw,
            verbose=verbose,
            compression_level=compression_level,
        )
        proc.write_summary_outputs(
            which=which,
            verbose=verbose,
            compression_level=compression_level,
        )

    def process_all_sim_timeseries_serially(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for event_iloc in self.df_sims.index:
            self.process_sim_timeseries(
                event_iloc=event_iloc,
                which=which,
                override_clear_raw=override_clear_raw,
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_log()
        return

    def consolidate_analysis_outputs(
        self,
        *,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        self.process.consolidate_to_datatree(
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def _retrieve_sim_runs(self, event_iloc):
        scen = TRITONSWMM_scenario(event_iloc, self)
        run = scen.run
        self._sim_run_objects[event_iloc] = run
        return run

    def _retrieve_sim_run_processing_object(self, event_iloc):
        run = self._retrieve_sim_runs(event_iloc)
        proc = TRITONSWMM_sim_post_processing(run)
        self._sim_run_processing_objects[event_iloc] = proc
        return proc

    def _create_launchable_sims(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
    ):
        """
        Create launcher functions for all simulations.

        Uses the consolidated _create_subprocess_sim_run_launcher pattern
        which handles the complete simulation lifecycle including simlog updates.

        The execution method (local, batch_job, or 1_job_many_srun_tasks) is
        determined by self.cfg_analysis.multi_sim_run_method.

        Parameters
        ----------
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of launcher functions
        """
        launch_and_finalize_functions_tuples = []
        enabled_model_types = self._get_enabled_model_types()
        scenario_locks = {event_iloc: threading.Lock() for event_iloc in self.df_sims.index}

        for event_iloc in self.df_sims.index:
            run = self._retrieve_sim_runs(event_iloc)
            lock = scenario_locks[event_iloc]
            for model_type in enabled_model_types:
                launch_and_finalize_functions_tuple = run._create_subprocess_sim_run_launcher(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                    model_type=model_type,
                )
                if launch_and_finalize_functions_tuple is None:
                    continue
                launcher, finalize_sim = launch_and_finalize_functions_tuple

                def locked_launcher(
                    _launcher=launcher,
                    _lock=lock,
                ):
                    _lock.acquire()
                    try:
                        return _launcher()
                    except Exception:
                        _lock.release()
                        raise

                def locked_finalize(
                    proc,
                    start_time,
                    sim_logfile,
                    lf,
                    _finalize=finalize_sim,
                    _lock=lock,
                ):
                    try:
                        _finalize(proc, start_time, sim_logfile, lf)
                    finally:
                        _lock.release()

                launch_and_finalize_functions_tuples.append((locked_launcher, locked_finalize))

        return launch_and_finalize_functions_tuples

    def run_simulations_concurrently(
        self,
        launch_functions: list[tuple],
        max_concurrent: int | None = None,
        verbose: bool = True,
    ):
        """
        Run simulations concurrently using the configured execution strategy.

        Automatically selects the appropriate executor based on cfg_analysis.multi_sim_run_method:
        - "1_job_many_srun_tasks": Uses SlurmExecutor for HPC execution
        - "local": Uses LocalConcurrentExecutor for parallel local execution
        - Other: Uses SerialExecutor for sequential execution

        Parameters
        ----------
        launch_functions : list[tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher()
        max_concurrent : Optional[int]
            Maximum number of concurrent simulations
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of simulation statuses
        """
        return self._execution_strategy.execute_simulations(launch_functions, max_concurrent, verbose)

    def run_sims_in_sequence(
        self,
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool = False,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        *,
        override_clear_raw: ClearRawValue | None = None,
        verbose=False,
    ):
        """
        Arguments passed to run:
            - mode: Mode | Literal["single_core"]
            - pickup_where_leftoff
        Arguments passed to processing process_sim_timeseriess
        (only needed if process_outputs_after_sim_completion=True):
            - which: Literal["TRITON", "SWMM", "both"]
            - override_clear_raw: ClearRawValue | None
            - compression_level: int
        """
        if verbose:
            print("Running all sims in series...", flush=True)
        enabled_model_types = self._get_enabled_model_types()
        for event_iloc in self.df_sims.index:
            for model_type in enabled_model_types:
                if verbose:
                    print(
                        f"Running sim {event_iloc} ({model_type}) and pickup_where_leftoff = {pickup_where_leftoff}",
                        flush=True,
                    )
                self.run_sim(
                    event_iloc=event_iloc,
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    override_clear_raw=override_clear_raw,
                    compression_level=compression_level,
                    model_type=model_type,  # type: ignore
                )
        self._update_log()

    def run(
        self,
        from_scratch: bool = False,
        dry_run: bool = False,
        events: list[int] | None = None,
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        verbose: bool = True,
        wait_for_job_completion: bool | None = None,
        override_clear_raw: ClearRawValue | None = None,
        override_force_rerun: ForceRerunValue | None = None,
        override_hpc_total_nodes: int | None = None,
        transfer_config: "PostRunTransferConfig | None" = None,
        report_config: "Path | None" = None,
        report_formats: list[Literal["html", "zip"]] | None = None,
        cleanup_orphans: bool = False,
        cleanup_stale_metadata: bool = True,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> "WorkflowResult":
        """
        High-level orchestration method for running TRITON-SWMM workflows.

        Parameters
        ----------
        from_scratch : bool
            If True, delete all analysis artifacts and start fresh.
            If False (default), resume from last completed checkpoint.
        dry_run : bool
            If True, validate workflow but don't execute.
        events : list[int] | None
            Subset of event_ilocs to process. If None, processes all events.
        execution_mode : Literal["auto", "local", "slurm"]
            Where to execute: auto-detect (default), force local, or force SLURM.
        verbose : bool
            If True, print progress messages.
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw``. ``None`` (default)
            reads the YAML; pass ``"none"`` / ``"all"`` / a list of model types
            (e.g. ``["tritonswmm", "swmm"]``) to override for this invocation.
            Per the ``override_`` prefix convention introduced by
            cleanup-rerun-delete-redesign Phase 1.
        wait_for_job_completion : bool | None
            If True, block until the SLURM job finishes. Mainly for tests.
        override_hpc_total_nodes : int | None
            Overrides `hpc_total_nodes` in the SBATCH script without mutating
            the config. Only valid for `multi_sim_run_method="1_job_many_srun_tasks"`.
        transfer_config : PostRunTransferConfig | None
            If provided, automatically transfer results to the local machine
            after successful completion (requires ``wait_for_job_completion=True``).
        cleanup_orphans : bool, default False
            When True, deletes orphan sub-analysis artifacts (subanalysis dirs,
            status flags, sensitivity_datatree.zarr groups) detected when an
            ``sa_id`` is removed from the sensitivity CSV/XLSX. Opt-in because
            the blast radius is irrecoverable (subanalysis data deleted).
        cleanup_stale_metadata : bool, default True
            When True (default), subprocess-invokes ``snakemake --cleanup-metadata``
            against orphaned ``.snakemake/metadata/`` records left by past
            rule-output renames (e.g., Phase 8's ``.png``/``.svg`` → ``.html``
            flip). When False, skips the cleanup; users may experience a
            one-shot full plot rebuild on first post-rename invocation per
            Phase 8 Risks. Asymmetric with ``cleanup_orphans`` default (False)
            because metadata-cleanup blast radius is bounded to records, not
            data; safe to auto-apply.
        extra_sbatch_args : list[str] | None
            Optional list of additional SBATCH directive strings (e.g.,
            ``["--qos=debug"]`` to route the job to Frontier's debug queue) to
            append to the generated ``run_workflow_1job.sh`` script. Each list
            element is emitted as one ``#SBATCH <element>`` line, after every
            other source of ``#SBATCH`` directives in the script — both the
            always-emitted directives derived from config fields
            (``--job-name``, ``--partition`` from
            ``cfg_analysis.hpc_ensemble_partition``, ``--account`` from
            ``cfg_analysis.hpc_account``, ``--nodes`` from
            ``cfg_analysis.hpc_total_nodes`` (or the
            ``override_hpc_total_nodes`` runtime kwarg), ``--exclusive``,
            ``--gres`` from ``cfg_analysis.hpc_gpus_per_node`` +
            ``cfg_system.gpu_hardware``, ``--time`` from
            ``cfg_analysis.hpc_total_job_duration_min``, ``--output``,
            ``--error``) AND the directives in the
            ``cfg_analysis.additional_SBATCH_params`` config list.

            **Override behavior**: any flag in ``extra_sbatch_args`` that
            matches a flag emitted earlier in the script — whether that earlier
            directive came from a top-level ``cfg_analysis`` field
            (``hpc_ensemble_partition``, ``hpc_account``, ``hpc_total_nodes``,
            ``hpc_total_job_duration_min``, ``hpc_gpus_per_node``, etc.) or
            from the ``cfg_analysis.additional_SBATCH_params`` list — WILL
            OVERRIDE the config-derived value via SLURM's last-directive-wins
            parser semantics. When such an override is detected, an
            informational ``[extra_sbatch_args] OVERRIDE: ...`` message is
            printed naming the flag, the origin of the original value
            (e.g. ``cfg_analysis.hpc_ensemble_partition``), and the new
            runtime value, so the user can confirm the override took effect
            as intended.

            Only valid for ``multi_sim_run_method="1_job_many_srun_tasks"``;
            raises ``ConfigurationError`` otherwise (a fail-fast guard
            preventing the user from believing they are controlling the
            experiment when the kwarg would silently no-op in another mode).

        Returns
        -------
        WorkflowResult
            Structured result object with success status and execution details.

        Examples
        --------
        Resume (default):

        >>> result = analysis.run()

        Fresh start:

        >>> result = analysis.run(from_scratch=True)

        Dry-run validation:

        >>> result = analysis.run(dry_run=True, verbose=True)

        Auto-transfer after completion:

        >>> from TRITON_SWMM_toolkit.config.globus import PostRunTransferConfig
        >>> result = analysis.run(
        ...     wait_for_job_completion=True,
        ...     transfer_config=PostRunTransferConfig(
        ...         destination_root=r"D:\\Dropbox\\results",
        ...         system="frontier",
        ...     ),
        ... )

        See Also
        --------
        submit_workflow : Lower-level workflow submission (15+ parameters)
        transfer_results : Standalone transfer method
        """
        # TODO - if from_scratch = True, user should be prompted for manual input to
        # type something like 'y' 'yes' or 'proceed' if the status of the
        # analysis shows that some steps have been completed. This should be
        # accompanied by a print statement of the current status.

        import time

        from .config.loaders import yaml_to_model
        from .config.report import (
            report_config as ReportConfigModel,
        )
        from .config.report import (
            validate_sensitivity_independent_vars,
        )
        from .exceptions import ConfigurationError
        from .orchestration import WorkflowResult, translate_mode, translate_phases

        # Pre-run report_config resolution (post-F2 v2 — 2-step, fail-fast).
        # Resolution order:
        #   (a) explicit `report_config=` argument → load and use
        #   (b) self.cfg_analysis.report (guaranteed non-None by R1 — required
        #       Pydantic field; loading a cfg_analysis.yaml without `report:`
        #       raises ValidationError before this code is reached)
        # No DEFAULT_REPORT_CONFIG fallback — the field is required.
        if report_config is not None:
            report_config = Path(report_config)
            try:
                cfg_report = yaml_to_model(report_config, ReportConfigModel)
            except Exception as e:
                raise ConfigurationError(
                    field="report_config",
                    message=f"Failed to load/validate {report_config}: {e}",
                    config_path=report_config,
                ) from e
        else:
            cfg_report = self.cfg_analysis.report

        sa_csv = self.cfg_analysis.sensitivity_analysis if self.cfg_analysis.toggle_sensitivity_analysis else None
        validate_sensitivity_independent_vars(cfg_report, sa_csv)
        self._cfg_report = cfg_report

        # Pre-run transfer validation — fail fast before submitting the workflow
        if transfer_config is not None:
            transfer_config.to_transfer_spec(
                analysis_dir=self.analysis_paths.analysis_dir,
                analysis_id=self.cfg_analysis.analysis_id,
            )

        start_time = time.time()

        # Event filtering not yet implemented - validate parameter
        if events is not None:
            raise NotImplementedError(
                "Event filtering via events parameter not yet implemented. "
                "For now, all events in analysis will be processed."
            )

        if from_scratch:
            # remove analysis folder
            fast_rmtree(self.cfg_analysis.analysis_dir)

        # Orphan detection gate (sensitivity-only; non-sensitivity covered by
        # follow-up plan per D-EVENT-PARITY).
        if not from_scratch and self.cfg_analysis.toggle_sensitivity_analysis and not self.cfg_analysis.is_subanalysis:
            from TRITON_SWMM_toolkit.exceptions import ConfigurationError as _CfgErr

            _dirs = self.sensitivity.find_orphan_subanalysis_dirs()
            _flags = self.sensitivity.find_orphan_status_flags()
            _groups = self.sensitivity.find_orphan_datatree_groups()
            _has_orphans = bool(_dirs or _flags or _groups)
            if _has_orphans and not cleanup_orphans:
                raise _CfgErr(
                    field="cleanup_orphans",
                    message=(
                        "Detected orphan sub-analysis artifacts on disk that are "
                        "absent from the current sensitivity CSV: "
                        f"{len(_dirs)} subanalysis dir(s), "
                        f"{len(_flags)} _status flag(s), "
                        f"{len(_groups)} datatree group(s). "
                        "Re-invoke analysis.run(cleanup_orphans=True) to delete them, "
                        "or run `triton-swmm cleanup-orphans --apply --force` from the CLI."
                    ),
                    config_path=str(self.analysis_config_yaml),
                )
            if _has_orphans and cleanup_orphans:
                self.sensitivity.cleanup_all_orphans(
                    dry_run=False,
                    force=True,
                    verbose=verbose,
                )

        # Stale-metadata cleanup gate — analysis-level, not sensitivity-specific
        # (per Phase 8.5 of interactive_report_renderers plan). Asymmetric with
        # cleanup_orphans default: cleanup_stale_metadata defaults to True
        # because metadata-cleanup blast radius is bounded to .snakemake/metadata/
        # records — no data is deleted; worst-case auto-apply result is the same
        # one-shot full plot rebuild Phase 8 Risks documents.
        # Precondition for the subprocess invocation: `snakemake
        # --cleanup-metadata` requires BOTH a Snakefile in the working
        # directory (to interpret path arguments) AND a
        # `.snakemake/metadata/` directory (the records to clean). The
        # Snakefile is generated by `submit_workflow()` later in this
        # method, so at this gate site it exists only on resumed analyses
        # (the use case cleanup_stale_metadata targets — fresh analyses
        # have no stale metadata to clean).
        _snakefile = self.analysis_paths.analysis_dir / "Snakefile"
        _metadata_dir = self.analysis_paths.analysis_dir / ".snakemake" / "metadata"
        if (
            cleanup_stale_metadata
            and not from_scratch
            and _snakefile.exists()
            and _metadata_dir.exists()
        ):
            orphan_paths = self._enumerate_stale_metadata_paths()
            if orphan_paths:
                if verbose:
                    print(
                        f"[cleanup-stale-metadata] Cleaning {len(orphan_paths)} "
                        f"orphan metadata record(s) from "
                        f"{self.analysis_paths.analysis_dir}/.snakemake/metadata/",
                        flush=True,
                    )
                    for p in orphan_paths:
                        print(f"  orphan: {p}", flush=True)
                self._invoke_snakemake_cleanup_metadata(orphan_paths)

        # Stamp _version.json at LAYOUT_VERSION on first materialization (lazy
        # stamp per version_migration_system master plan PI-1). Idempotent
        # under concurrent writers.
        from TRITON_SWMM_toolkit.version_migration import LAYOUT_VERSION
        from TRITON_SWMM_toolkit.version_migration.state import stamp_new_target

        stamp_new_target(self.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Translate user-friendly parameters to workflow parameters
        mode_params = translate_mode("resume")  # TODO - hardcoded while troubleshooting
        phase_params = translate_phases(None)  # TODO - hardcoded while troubleshooting

        # Detect system input processing needs

        swmm_used = False
        triton_used = False
        for model_used in self._get_enabled_model_types():
            if "swmm" in model_used.lower():
                swmm_used = True
            if "triton" in model_used.lower():
                triton_used = True
        if swmm_used and triton_used:
            which = "both"
        elif swmm_used and not triton_used:
            which = "SWMM"
        else:
            which = "TRITON"

        # Determine execution mode
        if execution_mode == "auto":
            if (
                self.in_slurm
                or self.cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks"
                or self.cfg_analysis.multi_sim_run_method == "batch_job"
            ):
                exec_mode = "slurm"
            else:
                exec_mode = "local"
        else:
            exec_mode = execution_mode

        if wait_for_job_completion is None:
            wait_for_job_completion = exec_mode != "slurm"

        # Build complete parameter dict for submit_workflow
        workflow_params = {
            **mode_params,
            **phase_params,
            "mode": exec_mode,
            "which": which,
            "override_clear_raw": override_clear_raw,
            "override_force_rerun": override_force_rerun,
            "compression_level": 5,
            "wait_for_completion": wait_for_job_completion,
            "dry_run": dry_run,
            "verbose": verbose,
            "override_hpc_total_nodes": override_hpc_total_nodes,
            "report_formats": report_formats,
            "extra_sbatch_args": extra_sbatch_args,
            "snakemake_diagnostics": snakemake_diagnostics,
        }

        if verbose:
            print("Submitting workflow with args:")
            print(workflow_params)

        # Call underlying submit_workflow
        result_dict = self.submit_workflow(**workflow_params)

        # Calculate execution time
        elapsed = time.time() - start_time

        # Determine which phases were completed based on parameters
        phases_completed = []
        if workflow_params["process_system_level_inputs"] or workflow_params["compile_TRITON_SWMM"]:
            phases_completed.append("setup")
        if workflow_params["prepare_scenarios"]:
            phases_completed.append("prepare")
        if workflow_params["prepare_scenarios"]:  # Simulate always runs if scenarios prepared
            phases_completed.append("simulate")
        if workflow_params["process_timeseries"]:
            phases_completed.append("process")
        if workflow_params["process_timeseries"]:  # Consolidate happens after processing
            phases_completed.append("consolidate")

        # Get event list (all events in analysis)
        events_processed = list(self.df_sims.index)

        # Post-completion auto-transfer
        if transfer_config is not None and wait_for_job_completion and result_dict.get("success", False):
            if verbose:
                print("[Transfer] Workflow succeeded — initiating Globus transfer...", flush=True)
            self.transfer_results(transfer_config)

        # Build WorkflowResult
        return WorkflowResult(
            success=result_dict.get("success", False),
            mode=result_dict.get("mode", exec_mode),
            execution_time=(elapsed if result_dict.get("success") and exec_mode == "local" else None),
            phases_completed=phases_completed if result_dict.get("success") else [],
            events_processed=events_processed if result_dict.get("success") else [],
            snakefile_path=result_dict.get("snakefile_path"),
            job_id=result_dict.get("job_id"),
            message=result_dict.get("message", ""),
        )

    def render_report(self, format: "Literal['html','zip']" = "zip") -> "Path":
        """Render the report from already-completed workflow outputs.

        Idempotent: invokes ``snakemake --report`` against the existing Snakefile
        without re-executing any rules. Requires the workflow to have completed
        (so the report() outputs exist) and the Snakefile to be on disk.

        Parameters
        ----------
        format : Literal["html", "zip"], default "zip"
            Output format. ``"html"`` produces a single self-contained
            ``analysis_report.html`` with all figures inlined as base64, plus
            React-bundle post-process surgery. ``"zip"`` produces
            ``analysis_report.zip`` containing the unbundled report tree;
            no post-process surgery is applied.

        Returns
        -------
        Path
            Path to the rendered ``analysis_report.{format}``.
        """
        import subprocess
        import sys

        from .exceptions import WorkflowError

        snakefile = self.analysis_paths.analysis_dir / "Snakefile"
        out = self.analysis_paths.analysis_dir / f"analysis_report.{format}"
        css_path = self.analysis_paths.analysis_dir / "report" / "report.css"
        # Re-emit report artifacts (report.css + workflow_description template)
        # from package resources so render_report picks up edits made to the
        # source-tree report_templates/ since the analysis was last run.
        _emit_report_artifacts(self.analysis_paths.analysis_dir)
        # --cores 1 is required by Snakemake's CLI even though --report is a
        # post-execution render that does not execute rules.
        cmd = [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--directory",
            str(self.analysis_paths.analysis_dir),
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

    @property
    def n_scenarios(self):
        sensitivity_scenario = 1
        if self.cfg_analysis.toggle_sensitivity_analysis:
            sens = self.sensitivity
            sensitivity_scenario = len(sens.df_setup)

        n_total = len(self.df_sims) * sensitivity_scenario
        return n_total

    @property
    def n_sims(self):
        sensitivity_scenario = 1
        if self.cfg_analysis.toggle_sensitivity_analysis:
            sens = self.sensitivity
            sensitivity_scenario = len(sens.df_setup)

        n_total = len(self.df_sims) * len(self._get_enabled_model_types()) * sensitivity_scenario
        return n_total

    def get_workflow_status(self) -> "WorkflowStatus":
        """Generate workflow status report.

        Inspects logs and outputs to determine completion state of each phase,
        providing actionable recommendations for which execution mode to use.

        Returns
        -------
        WorkflowStatus
            Structured status report with phase details and recommendations

        Examples
        --------
        Check status before running:

        >>> status = analysis.get_workflow_status()
        >>> print(status)
        >>> if not status.simulation.complete:
        ...     print(f"Retry {len(status.simulation.failed_items)} failed sims")

        Use recommended mode:

        >>> status = analysis.get_workflow_status()
        >>> result = analysis.run(mode=status.recommended_mode)

        Notes
        -----
        This method is read-only and does not modify any state. It provides
        transparency into workflow progress to help users make informed
        decisions about execution modes.

        See Also
        --------
        run : High-level workflow execution method
        """
        from .orchestration import PhaseStatus, WorkflowStatus

        # Check setup phase
        system_log = self._system.log
        dem_done = system_log.dem_processed.get()
        mannings_done = self._system.cfg_system.toggle_use_constant_mannings or system_log.mannings_processed.get()
        compiled = system_log.compilation_tritonswmm_cpu_successful.get()

        setup_complete = dem_done and mannings_done and compiled
        setup_progress = 1.0 if setup_complete else 0.5 if (dem_done or compiled) else 0.0
        setup_details = {
            "dem": f"{'✓' if dem_done else '✗'} DEM processed",
            "mannings": f"{'✓' if mannings_done else '✗'} Manning's processed",
            "compiled": f"{'✓' if compiled else '✗'} TRITON-SWMM compiled",
        }

        setup_phase = PhaseStatus(
            name="setup",
            complete=setup_complete,
            progress=setup_progress,
            details=setup_details,
        )

        # Check scenario preparation
        all_prepared = self.all_scenarios_created
        not_prepared = self.scenarios_not_created

        n_total = self.n_sims

        n_prepared = n_total - len(not_prepared)

        prep_phase = PhaseStatus(
            name="preparation",
            complete=all_prepared,
            progress=n_prepared / n_total if n_total > 0 else 0.0,
            details={"scenarios": f"{'✓' if all_prepared else '⚠'} {n_prepared}/{n_total} scenarios created"},
            failed_items=[str(p) for p in not_prepared],
        )

        # Check simulations
        all_run = self.all_sims_run
        not_run = self.scenarios_not_run
        n_run = n_total - len(not_run)

        sim_phase = PhaseStatus(
            name="simulation",
            complete=all_run,
            progress=n_run / n_total if n_total > 0 else 0.0,
            details={"sims": f"{'✓' if all_run else '⚠'} {n_run}/{n_total} simulations completed"},
            failed_items=[str(p) for p in not_run],
        )

        # Check processing
        enabled_models = self._get_enabled_model_types()
        triton_enabled = "triton" in enabled_models or "tritonswmm" in enabled_models
        swmm_enabled = "swmm" in enabled_models or "tritonswmm" in enabled_models

        triton_missing = len(self.TRITON_time_series_not_processed) if triton_enabled else 0
        swmm_missing = len(self.SWMM_time_series_not_processed) if swmm_enabled else 0

        triton_total = n_total if triton_enabled else 0
        swmm_total = n_total if swmm_enabled else 0

        triton_processed = max(triton_total - triton_missing, 0)
        swmm_processed = max(swmm_total - swmm_missing, 0)

        processed_total = triton_processed + swmm_processed
        total_needed = triton_total + swmm_total
        proc_progress = processed_total / total_needed if total_needed else 0.0

        triton_proc_complete = triton_missing == 0 if triton_enabled else True
        swmm_proc_complete = swmm_missing == 0 if swmm_enabled else True
        proc_complete = triton_proc_complete and swmm_proc_complete

        proc_phase = PhaseStatus(
            name="processing",
            complete=proc_complete,
            progress=proc_progress,
            details={
                "triton": (
                    f"{'✓' if triton_proc_complete else '✗'} TRITON outputs processed: "
                    f"{triton_processed}/{triton_total}"
                    if triton_enabled
                    else "✓ TRITON outputs processed: n/a"
                ),
                "swmm": (
                    f"{'✓' if swmm_proc_complete else '✗'} SWMM outputs processed: {swmm_processed}/{swmm_total}"
                    if swmm_enabled
                    else "✓ SWMM outputs processed: n/a"
                ),
            },
        )

        # Check consolidation: under Option B (render_bundle plan), the
        # canonical master-level artifact is analysis_datatree.zarr; the
        # per-mode flat zarrs no longer exist. The log marker
        # `datatree_consolidation_complete` is the single canonical signal
        # for "consolidation has completed."
        summaries_exist = (
            hasattr(self.log, "datatree_consolidation_complete")
            and self.log.datatree_consolidation_complete.get() is True
        )

        consol_details = {
            "datatree": (
                f"{'✓' if summaries_exist else '✗'} analysis_datatree.zarr "
                f"({'present' if summaries_exist else 'not yet built'})"
            )
        }

        consol_phase = PhaseStatus(
            name="consolidation",
            complete=summaries_exist,
            progress=1.0 if summaries_exist else 0.0,
            details=consol_details,
        )

        # Determine current phase and recommendation
        if not setup_complete:
            current = "setup"
            rec_mode = "fresh"
            rec_text = "Setup incomplete. Use 'fresh' mode to process system inputs."
        elif not all_prepared:
            current = "preparation"
            rec_mode = "resume"
            rec_text = f"Use 'resume' to create {len(not_prepared)} remaining scenarios."
        elif not all_run:
            current = "simulation"
            rec_mode = "resume"
            rec_text = f"Use 'resume' to run {len(not_run)} pending/failed simulations."
        elif not proc_complete:
            current = "processing"
            rec_mode = "resume"
            rec_text = "Use 'resume' to process simulation outputs."
        elif not summaries_exist:
            current = "consolidation"
            rec_mode = "resume"
            rec_text = "Use 'resume' to consolidate analysis summaries."
        else:
            current = "complete"
            rec_mode = "n/a"
            rec_text = "All phases complete. Use 'fresh' if you want to redo the analysis."

        return WorkflowStatus(
            analysis_id=self.cfg_analysis.analysis_id,
            analysis_dir=self.analysis_paths.analysis_dir,
            setup=setup_phase,
            preparation=prep_phase,
            simulation=sim_phase,
            processing=proc_phase,
            consolidation=consol_phase,
            total_simulations=n_total,
            simulations_completed=n_run,
            simulations_failed=len(not_run),
            simulations_pending=0,  # Would need more logic to distinguish failed vs pending
            current_phase=current,
            recommended_mode=rec_mode,
            recommendation=rec_text,
        )

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        override_clear_raw: ClearRawValue | None = None,
        override_force_rerun: ForceRerunValue | None = None,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit workflow using Snakemake (replaces submit_SLURM_job_array).

        Automatically detects execution context (local vs. HPC) and submits accordingly.

        Delegates to SnakemakeWorkflowBuilder.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each simulation will prepare its scenario before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after each simulation
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process (only used if process_timeseries=True)
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw``. ``None`` (default)
            reads from YAML; concrete values follow the override-prefix convention.
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        wait_for_completion : bool
            If True, wait for workflow completion (relevant for slurm jobs only)
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
            - success: bool - Whether workflow succeeded
            - mode: str - "local" or "slurm"
            - snakefile_path: Path - Path to generated Snakefile
            - job_id: str | None - Job ID (only for slurm mode)
            - message: str - Status message
        """
        # Stamp _version.json at LAYOUT_VERSION on first materialization (lazy
        # stamp per version_migration_system master plan PI-1). Idempotent
        # under concurrent writers.
        from TRITON_SWMM_toolkit.version_migration import LAYOUT_VERSION
        from TRITON_SWMM_toolkit.version_migration.state import stamp_new_target

        stamp_new_target(self.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Force-rerun pre-delete (login-node responsibility per master plan
        # Strategy). Resolve + validate + delete BEFORE Snakemake plans the DAG
        # so MTIME-input triggers cascade re-fire automatically.
        self._apply_force_rerun(override_force_rerun)

        if self.cfg_analysis.toggle_sensitivity_analysis:
            result = self.sensitivity.submit_workflow(
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
                override_force_rerun=override_force_rerun,
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
        else:
            # NOTE: override_force_rerun is NOT threaded into the inner builder
            # — the pre-delete already happened at this layer
            # (self._apply_force_rerun above) and the builder's
            # submit_workflow does not need a runtime force-rerun parameter.
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

        if dry_run and result.get("success"):
            snakemake_logfile = result.get("snakemake_logfile")
            if snakemake_logfile is not None:
                report_path = generate_dry_run_report_markdown(
                    snakemake_logfile=Path(snakemake_logfile),
                    analysis_dir=self.analysis_paths.analysis_dir,
                    verbose=verbose,
                )
                result["dry_run_report_markdown"] = report_path

        return result

    def reprocess(
        self,
        start_with: "Literal['process','consolidate','render']" = "consolidate",
        execution_mode: "Literal['auto','local','slurm']" = "auto",
        which: "Literal['TRITON','SWMM','both']" = "both",
        *,
        override_clear_raw: ClearRawValue | None = "none",
        override_force_rerun: ForceRerunValue | None = None,
        verbose: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """Re-run downstream stages against existing sim outputs.

        Re-runs processing / consolidation / plotting / report rendering
        without re-running the simulation rules. Runs the Phase-1
        reconciliation guard against ``_status/_submitted/`` before
        submitting, so a parallel live sim driver cannot be double-submitted.
        Emits a scope-limited Snakefile at
        ``{analysis_dir}/Snakefile.reprocess`` and runs it against a
        sibling ``.snakemake_reprocess/`` working directory so the reprocess
        driver does not collide with the main ``.snakemake/`` state.

        Parameters
        ----------
        start_with
            Stage to re-fire from. ``"consolidate"`` is the common case —
            re-aggregates the analysis datatree zarr and re-renders the
            report against existing sim outputs.
        execution_mode
            ``"auto"`` (default) detects SLURM context; ``"local"`` /
            ``"slurm"`` force the mode.
        which
            ``"both"`` (default) / ``"TRITON"`` / ``"SWMM"`` — passes through
            to ``rule consolidate``'s ``--which`` flag.
        override_clear_raw
            **Hard-default "none"** to preserve historic ``reprocess`` semantics
            (reprocess never auto-clears unless the caller explicitly opts in).
            Pass ``None`` to read ``cfg_analysis.clear_raw``; pass ``"all"`` /
            ``"none"`` / a list of model types to override. When the resolved
            value is anything other than ``"none"``, two guards must both pass:
            (a) every enabled sim's ``c_run_*`` flag must exist (no
            never-started sims); (b) no ``_status/_submitted/`` sentinel
            may be present (no in-flight / just-died sims). Cites
            stipulation ``clear raw triton outputs deferred until last allocation``
            (under ``library/docs/stipulations/TRITON-SWMM_toolkit/``).
        verbose
            If True, print progress messages.
        dry_run
            If True, runs ``snakemake --dry-run`` only.

        Returns
        -------
        dict
            Status dictionary from
            :meth:`SnakemakeWorkflowBuilder.submit_reprocess_workflow`.

        Raises
        ------
        ConfigurationError
            When the resolved ``clear_raw`` would clear and either guard fails.
        """
        resolved_clear_raw = override_clear_raw if override_clear_raw is not None else self.cfg_analysis.clear_raw
        # True iff the resolved value would trigger any cleanup for any model.
        would_clear = resolved_clear_raw != "none"
        # Lazy-stamp _version.json at LAYOUT_VERSION (PI-1 pattern, mirroring
        # run() and submit_workflow). Idempotent under concurrent writers;
        # if _version.json is missing or stamped at an older version, this
        # writes a fresh stamp at the current LAYOUT_VERSION.
        from TRITON_SWMM_toolkit.version_migration import LAYOUT_VERSION
        from TRITON_SWMM_toolkit.version_migration.state import stamp_new_target

        from .exceptions import ConfigurationError

        stamp_new_target(self.analysis_paths.analysis_dir, LAYOUT_VERSION)

        # Dispatch to the sensitivity-master reprocess path for sensitivity-toggled
        # analyses. The non-sensitivity reprocess generator emits a `rule consolidate`
        # that consumes from `analysis_dir/sims/`, which for sensitivity layouts does
        # not exist — sims live under `subanalyses/sa_*/sims/`. The sensitivity-master
        # generator (SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content)
        # emits per-sa consolidate rules + a master_consolidation rule that consume
        # from the correct paths. Pattern mirrors analysis.py:683-801 property
        # dispatches and the bundle CLI dispatch at cli.py:1026.
        if self.cfg_analysis.toggle_sensitivity_analysis:
            if would_clear:
                raise ConfigurationError(
                    field="override_clear_raw",
                    message=(
                        "TRITONSWMM_analysis.reprocess does not support clearing raw outputs "
                        "for sensitivity-toggled analyses (resolved clear_raw="
                        f"{resolved_clear_raw!r}). The sensitivity-master reprocess "
                        "path deliberately omits the clear-raw gate (see "
                        "TRITONSWMM_sensitivity_analysis.reprocess docstring). Invoke "
                        "self.sensitivity.reprocess(...) directly with explicit sa_ids if "
                        "raw-output clearing is required."
                    ),
                    config_path=str(self.analysis_config_yaml),
                )
            return self.sensitivity.reprocess(
                start_with=start_with,
                execution_mode=execution_mode,
                which=which,
                override_force_rerun=override_force_rerun,
                verbose=verbose,
                dry_run=dry_run,
            )

        if would_clear:
            # Guard (a): every enabled sim must have a c_run_* flag.
            if not self._all_sim_flags_present():
                raise ConfigurationError(
                    field="override_clear_raw",
                    message=(
                        "reprocess refuses raw-output clearing while c_run_* flags are absent "
                        f"(resolved clear_raw={resolved_clear_raw!r}; some sims have not completed). "
                        "See stipulation `clear raw triton outputs deferred until last allocation`."
                    ),
                    config_path=str(self.analysis_config_yaml),
                )
            # Guard (b): no in-flight or unreconciled _submitted/ sentinel.
            submitted_dir = self.analysis_paths.analysis_dir / "_status" / "_submitted"
            if submitted_dir.exists() and any(submitted_dir.glob("*.json")):
                raise ConfigurationError(
                    field="override_clear_raw",
                    message=(
                        "reprocess refuses raw-output clearing while _submitted/ sentinels are present "
                        f"(resolved clear_raw={resolved_clear_raw!r}; simulations may still be in flight "
                        "or recently died). Run the Phase-1 reconciliation guard or `scancel` outstanding "
                        "jobs first."
                    ),
                    config_path=str(self.analysis_config_yaml),
                )

        # Reprocess overrides 1_job_many_srun_tasks → batch_job at submission
        # time. 1_job_many_srun_tasks reserves an exclusive multi-node SLURM
        # allocation that the downstream-only reprocess does not need, and the
        # method cannot decouple driver-cancel from job-cancel (master plan
        # Assumptions + FQ1 research). The override is local — the analysis's
        # original cfg_analysis.multi_sim_run_method is not mutated.
        effective_method: str | None = None
        if self.cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks":
            effective_method = "batch_job"
            if verbose:
                print(
                    "[reprocess] NOTE: 1_job_many_srun_tasks reprocess overridden to "
                    "batch_job (per-rule sbatch). The original analysis_config is unchanged.",
                    flush=True,
                )

        # Invalidate from start_with onward — deletes the upstream flag/artifact
        # that triggers Snakemake's mtime-driven re-fire. Per D-INVALIDATE
        # option 1: delete flags + rely on the generator's baked overwrite.
        self._invalidate_downstream_flags(start_with)

        # Force-rerun pre-delete (login-node responsibility). Resolve +
        # validate + delete BEFORE Snakemake plans the reprocess DAG. Per
        # cleanup-rerun-delete-redesign Phase 4 + R10.
        self._apply_force_rerun(override_force_rerun)

        # Delegate to the workflow builder. The submit method writes the
        # reprocess Snakefile and orchestrates the snakemake invocation with
        # `--directory .snakemake_reprocess --rerun-triggers mtime
        # --snakefile Snakefile.reprocess`.
        result = self._workflow_builder.submit_reprocess_workflow(
            start_with=start_with,
            execution_mode=execution_mode,
            multi_sim_run_method_override=effective_method,
            dry_run=dry_run,
            verbose=verbose,
        )
        return result

    def delete(self, override_in_flight: bool = False) -> None:
        """Distributed Snakemake workflow that deletes the entire analysis_dir.

        Refuses by default when ``_status/_submitted/*.json`` sentinels
        indicate live SLURM jobs. Pass ``override_in_flight=True`` to bypass
        the guard.

        Mirrors the dispatch pattern of :meth:`reprocess` — sensitivity-toggled
        analyses dispatch to
        :meth:`TRITONSWMM_sensitivity_analysis.delete`.

        Per cleanup-rerun-delete-redesign Phase 2 (D-DeleteSentinelInteraction
        + D-DeleteBoundary resolutions).
        """
        if self.cfg_analysis.toggle_sensitivity_analysis:
            return self.sensitivity.delete(override_in_flight=override_in_flight)

        analysis_dir = self.analysis_paths.analysis_dir

        # 1. Clear any stale sentinels from a prior failed delete attempt.
        # Without this, the post-check at step 3 could falsely pass on a
        # half-completed previous delete and fast_rmtree a partially-deleted
        # tree.
        stale_dir = analysis_dir / "_status" / "_deleting"
        if stale_dir.exists():
            fast_rmtree(stale_dir)

        # 2. Submit the distributed delete workflow. The workflow builder's
        # _pre_delete_guards (live-sentinel refusal + scoped lock-check) runs
        # inside submit_delete_workflow; orchestrator does not invoke it
        # directly.
        self._workflow_builder.submit_delete_workflow(override_in_flight=override_in_flight)

        # 3. Verify all expected sentinels present; remove analysis_dir atomically.
        expected = self._enumerate_expected_delete_sentinels()
        deleting_dir = analysis_dir / "_status" / "_deleting"
        actual = set(deleting_dir.glob("*.flag")) if deleting_dir.exists() else set()
        missing = expected - actual
        if missing:
            print(
                f"[delete] {len(missing)} per-rule sentinels missing — "
                f"preserving analysis_dir for debugging.",
                flush=True,
            )
            print(f"[delete] missing: {sorted(p.name for p in missing)}", flush=True)
            return
        print(
            f"[delete] all {len(expected)} per-rule sentinels present — "
            f"removing analysis_dir.",
            flush=True,
        )
        fast_rmtree(analysis_dir)

    def _enumerate_expected_delete_sentinels(self) -> set[Path]:
        """Compute the set of ``_status/_deleting/*.flag`` paths the delete
        workflow will produce on full success.

        One per scenario for regular analyses (sensitivity-master analyses
        delegate to :meth:`TRITONSWMM_sensitivity_analysis._enumerate_expected_delete_sentinels`
        before reaching this method); plus one for the consolidation rule.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        delete_dir = self.analysis_paths.analysis_dir / "_status" / "_deleting"
        expected = {delete_dir / "analysis_consolidation.flag"}
        for i in range(len(self.df_sims)):
            event_id = compute_event_id_slug(self._retrieve_weather_indexer_using_integer_index(i))
            expected.add(delete_dir / f"scenario_evt-{event_id}.flag")
        return expected

    def _all_sim_flags_present(self) -> bool:
        """True iff every enabled sim's ``c_run_*`` completion flag exists.

        Used as the *flag-presence* component of the ``override_clear_raw``
        guard; the sentinel-presence component (in-flight detection) is
        checked separately at the :meth:`reprocess` call site.

        Enumeration contract
        --------------------
        For a non-sensitivity analysis: for each enabled model_type (from
        cfg_system's ``toggle_*_model`` fields) and each event_id in the
        analysis's event set, expect
        ``{_status}/c_run_{model_type}_evt-{event_id}.flag``.

        For a sensitivity master analysis: recurse into each sub-analysis's
        ``_status/`` directory and check
        ``c_run_{model_type}_sa-{sa_id}_evt-{event_id}.flag`` for every
        (sa_id, event_id, enabled_model_type) tuple.

        Returns True only if every expected flag exists. Missing → False.
        Does NOT consult ``_submitted/`` sentinels; that signal is the
        in-flight guard layered on top of this method at the
        :meth:`reprocess` call site.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        cfg_sys = self._system.cfg_system
        enabled_models: list[str] = []
        if cfg_sys.toggle_triton_model:
            enabled_models.append("triton")
        if cfg_sys.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if cfg_sys.toggle_swmm_model:
            enabled_models.append("swmm")
        if not enabled_models:
            return False  # No models enabled — nothing to attest.

        # Non-sensitivity path (sensitivity paths handled by
        # TRITONSWMM_sensitivity_analysis.reprocess in Phase 3).
        if getattr(self.cfg_analysis, "toggle_sensitivity_analysis", False):
            # Sensitivity master analyses are out of scope for Phase 2's
            # reprocess; the sensitivity-master reprocess is Phase 3. Until
            # then, conservatively return False so the guard short-circuits
            # rather than admitting a false-positive.
            return False

        status_dir = self.analysis_paths.analysis_dir / "_status"
        for i in range(len(self.df_sims)):
            event_id = compute_event_id_slug(self._retrieve_weather_indexer_using_integer_index(i))
            for model in enabled_models:
                flag = status_dir / f"c_run_{model}_evt-{event_id}_complete.flag"
                if not flag.exists():
                    return False
        return True

    def _invalidate_downstream_flags(self, start_with: str) -> None:
        """Delete ``_status`` flags from ``start_with`` onward.

        Never deletes ``c_run_*`` (sim) flags. The render layer is
        file-driven — deleting plot/report artifacts is the trigger for
        re-firing (Snakemake's mtime check sees the output as absent).

        Per cleanup-rerun-delete-redesign Phases 3 + 4, the legacy
        rule-shell-level overwrite toggle is fully retired. This helper
        is the *reprocess* entry-point (start-from-a-named-stage); the
        parallel *force-rerun* entry-point is
        ``_apply_force_rerun(override_force_rerun)`` which invokes
        ``_workflow_builder._delete_flags_for_force_rerun(spec)`` and
        ``_invalidate_processing_log_for_force_rerun(spec)`` to ensure
        ``process_simulation.py::_already_written`` returns False for
        targeted outputs. This helper additionally deletes the
        consolidated datatree zarr when ``start_with`` is ``"process"``
        or ``"consolidate"`` so the rebuild path is unconditional.

        Parameters
        ----------
        start_with
            One of ``"process"``, ``"consolidate"``, ``"render"``.
        """
        sd = self.analysis_paths.analysis_dir / "_status"
        if start_with == "process":
            for f in sd.glob("d_process_*"):
                f.unlink(missing_ok=True)
            (sd / "e_consolidate_complete.flag").unlink(missing_ok=True)
            _zarr = self.analysis_paths.analysis_datatree_zarr
            if _zarr is not None and _zarr.exists():
                fast_rmtree(_zarr)
        elif start_with == "consolidate":
            (sd / "e_consolidate_complete.flag").unlink(missing_ok=True)
            _zarr = self.analysis_paths.analysis_datatree_zarr
            if _zarr is not None and _zarr.exists():
                fast_rmtree(_zarr)
        elif start_with == "render":
            # No _status flag for render — re-fire by deleting the report
            # artifacts so Snakemake's mtime trigger sees the output as
            # absent. Plot PNGs/HTML are left in place; the plot rules
            # only re-fire if a plot's inputs are newer (the intended
            # surgical behavior for `start_with="render"`).
            (self.analysis_paths.analysis_dir / "analysis_report.html").unlink(missing_ok=True)
            (self.analysis_paths.analysis_dir / "analysis_report.zip").unlink(missing_ok=True)
        else:
            raise ValueError(f"start_with must be one of 'process', 'consolidate', 'render'; got {start_with!r}")

    def _validate_force_rerun_targets(self, resolved_force_rerun) -> None:
        """Validate that requested ``sa_id`` / ``event_iloc`` values exist in the analysis.

        Per cleanup-rerun-delete-redesign Phase 4, R11 + D-ForceRerunValidatesSaId
        Option 1 (hard error at API entry). Unknown values raise
        ``ConfigurationError`` before any filesystem touch.
        """
        from .exceptions import ConfigurationError

        if resolved_force_rerun in ("all", "none"):
            return
        if not isinstance(resolved_force_rerun, dict):
            raise ValueError(f"Unexpected force_rerun shape: {resolved_force_rerun!r}")
        key = next(iter(resolved_force_rerun))
        # Cross-check against toggle_sensitivity_analysis — mirrors the cfg-load
        # validator in config/analysis.py but applies to the override path too.
        if key == "sa_id" and not self.cfg_analysis.toggle_sensitivity_analysis:
            raise ConfigurationError(
                field="override_force_rerun",
                message=(
                    "override_force_rerun.sa_id requires toggle_sensitivity_analysis=True"
                ),
            )
        if key == "event_iloc" and self.cfg_analysis.toggle_sensitivity_analysis:
            raise ConfigurationError(
                field="override_force_rerun",
                message=(
                    "override_force_rerun.event_iloc requires toggle_sensitivity_analysis=False; "
                    "sensitivity-toggled analyses must use override_force_rerun.sa_id instead"
                ),
            )
        requested = set(map(str, resolved_force_rerun[key]))
        if key == "sa_id":
            known = set(self.sensitivity.df_setup.index.astype(str))
        else:  # event_iloc
            known = set(map(str, self.df_sims.index))
        unknown = requested - known
        if unknown:
            raise ConfigurationError(
                field="override_force_rerun",
                message=(
                    f"override_force_rerun.{key} contains unknown values: "
                    f"{sorted(unknown)}. Known {key} values: {sorted(known)}."
                ),
            )

    def _build_force_rerun_spec(self, resolved_force_rerun):
        """Resolve a ``ForceRerunValue`` into a ``ResolvedForceRerunSpec``.

        For the ``event_iloc`` scope, resolves event_iloc integers to event_id
        slugs via ``compute_event_id_slug`` (V0001's stable event-slug
        invariant); the builder helper consumes only slugs/sa_ids.
        """
        from TRITON_SWMM_toolkit.workflow import ResolvedForceRerunSpec
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        if resolved_force_rerun == "all":
            return ResolvedForceRerunSpec(scope="all", tokens=())
        if resolved_force_rerun == "none":
            return ResolvedForceRerunSpec(scope="none", tokens=())
        assert isinstance(resolved_force_rerun, dict)
        key = next(iter(resolved_force_rerun))
        values = resolved_force_rerun[key]
        if key == "sa_id":
            return ResolvedForceRerunSpec(scope="sa", tokens=tuple(str(v) for v in values))
        # event_iloc → event_id slug per V0001 stable slug invariant.
        slugs = tuple(
            compute_event_id_slug(self._retrieve_weather_indexer_using_integer_index(int(iloc)))
            for iloc in values
        )
        return ResolvedForceRerunSpec(scope="event", tokens=slugs)

    def _apply_force_rerun(self, override_force_rerun) -> None:
        """Resolve, validate, and pre-delete flags + per-scenario log records
        for the force-rerun override.

        Called at workflow-submission boundaries (run / submit_workflow /
        reprocess / submit_reprocess_workflow). Pre-delete happens on the
        login node BEFORE Snakemake plans the DAG so MTIME-input triggers see
        the deleted flags and cascade re-fire automatically. Per cleanup-
        rerun-delete-redesign Phase 4 + R10.

        Two-layer invalidation per the FQ0 trace (post-Phase-4):

        1. ``_delete_flags_for_force_rerun(spec)`` removes ``_status/*.flag``
           markers so Snakemake re-plans the affected rules.
        2. ``_invalidate_processing_log_for_force_rerun(spec)`` clears the
           per-scenario per-model log ``processing_log.outputs`` so each
           runner subprocess's ``_already_written`` gate returns False and
           the write paths actually re-execute. Without (2), step (1) alone
           produces fresh flags but stale outputs.
        """
        resolved = (
            override_force_rerun
            if override_force_rerun is not None
            else self.cfg_analysis.force_rerun
        )
        self._validate_force_rerun_targets(resolved)
        spec = self._build_force_rerun_spec(resolved)
        self._workflow_builder._delete_flags_for_force_rerun(spec)
        self._invalidate_processing_log_for_force_rerun(spec)

    def _invalidate_processing_log_for_force_rerun(
        self, spec: "ResolvedForceRerunSpec"
    ) -> None:
        """Invalidate per-scenario log ``processing_log.outputs`` entries
        that match the force-rerun spec.

        Per cleanup-rerun-delete-redesign Phase 4 + B-mechanism: flag
        deletion triggers Snakemake to re-plan the DAG, but the runner
        subprocess's ``process_simulation.py::_already_written`` gate
        consults each per-model log's ``processing_log.outputs`` dict, NOT
        the flag files. Without log-record invalidation, the re-fired rule
        subprocess executes ``write_timeseries_outputs(...)`` but every
        internal ``_export_*`` early-returns on ``_already_written``
        because the log still records ``success=True`` from the prior run
        — net result is fresh flags but stale zarrs. This helper closes
        that gap: for each scope matched by ``spec``, it clears the
        corresponding per-model log files' ``processing_log.outputs`` (so
        the next runner pass writes the outputs fresh) and resets the
        ``raw_*_outputs_cleared`` markers so the clear-raw step re-runs.

        For the ``"all"`` scope: invalidates every scenario's per-model
        log. For ``"sa"`` scope: dispatches via
        ``sensitivity._invalidate_processing_log_for_sa_ids``. For
        ``"event"`` scope: invalidates only the scenarios whose
        ``event_id`` slug matches the tokens.

        On-disk side effect: the per-scenario log JSON file
        (``log_tritonswmm.json`` / ``log_triton.json`` / ``log_swmm.json``)
        is rewritten with ``processing_log.outputs = {}``. No zarr/nc
        artifact is touched — the runner's ``_write_output`` path
        overwrites the existing zarr on re-execution.
        """
        if spec.scope == "none":
            return

        if spec.scope == "sa":
            # Sensitivity dispatch — sub-analyses own their scenarios.
            self.sensitivity._invalidate_processing_log_for_sa_ids(spec.tokens)
            return

        if spec.scope == "all":
            target_event_ids = set(self._all_event_id_slugs())
        elif spec.scope == "event":
            target_event_ids = set(spec.tokens)
        else:
            raise ValueError(f"Unrecognized spec.scope: {spec.scope!r}")

        for event_iloc in range(len(self.df_sims)):
            scen = TRITONSWMM_scenario(event_iloc, self)
            if scen.event_id not in target_event_ids:
                continue
            for model_type in scen.run.model_types_enabled:
                model_log = scen.get_log(model_type)
                # Clear the processing_log dict and persist.
                model_log.processing_log.outputs.clear()
                # Also reset raw-outputs-cleared markers so the next
                # processing pass re-runs the clear_raw step on top of
                # the re-written outputs.
                if model_log.raw_TRITON_outputs_cleared is not None:
                    model_log.raw_TRITON_outputs_cleared.set(False)
                if model_log.raw_SWMM_outputs_cleared is not None:
                    model_log.raw_SWMM_outputs_cleared.set(False)
                model_log.write()

    def _all_event_id_slugs(self) -> list[str]:
        """Helper: enumerate every scenario's event_id slug for ``"all"`` scope.

        Uses the same V0001 stable-slug derivation as ``_build_force_rerun_spec``
        — no scenarios attribute exists on Analysis (per Phase 2 audit row), so
        slugs are computed from df_sims index via
        ``compute_event_id_slug(self._retrieve_weather_indexer_using_integer_index(i))``.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug
        return [
            compute_event_id_slug(
                self._retrieve_weather_indexer_using_integer_index(i)
            )
            for i in range(len(self.df_sims))
        ]

    # TODO - fix or delete
    # @property
    # def TRITONSWMM_runtimes(self):
    #     return (
    #         self.tritonswmm_TRITON_summary["compute_time_min"]
    #         .to_dataframe()
    #         .dropna()["compute_time_min"]
    #     )

    @property
    def tritonswmm_performance_analysis_summary_created(self):
        return bool(self.log.tritonswmm_performance_analysis_summary_created.get())

    @property
    def tritonswmm_triton_analysis_summary_created(self):
        return bool(self.log.tritonswmm_triton_analysis_summary_created.get())

    @property
    def tritonswmm_node_analysis_summary_created(self):
        return bool(self.log.tritonswmm_node_analysis_summary_created.get())

    @property
    def tritonswmm_link_analysis_summary_created(self):
        return bool(self.log.tritonswmm_link_analysis_summary_created.get())

    @property
    def triton_only_analysis_summary_created(self):
        return bool(self.log.triton_only_analysis_summary_created.get())

    @property
    def swmm_only_node_analysis_summary_created(self):
        return bool(self.log.swmm_only_node_analysis_summary_created.get())

    @property
    def swmm_only_link_analysis_summary_created(self):
        return bool(self.log.swmm_only_link_analysis_summary_created.get())

    @property
    def df_snakemake_allocations(self) -> pd.DataFrame:
        enabled_models_untyped = self._get_enabled_model_types()
        enabled_models: list[Literal["triton", "tritonswmm", "swmm"]] = [
            m for m in ("triton", "tritonswmm", "swmm") if m in enabled_models_untyped
        ]

        if self.cfg_analysis.toggle_sensitivity_analysis:
            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            expected_sa_ids = sorted(self.sensitivity.sub_analyses.keys())
            sa_allocations = parse_sensitivity_analysis_workflow_model_allocations(
                snakefile_path=snakefile_path,
                expected_subanalysis_ids=expected_sa_ids,
            )
            rows: list[dict] = []
            for sa_id, sub_analysis in self.sensitivity.sub_analyses.items():
                if sa_id not in sa_allocations:
                    raise ValueError(
                        "Parsed sensitivity allocations missing subanalysis id: "
                        f"sa_{sa_id}. Available ids: {sorted(sa_allocations.keys())}"
                    )
                alloc = sa_allocations[sa_id]
                for event_iloc in sub_analysis.df_sims.index:
                    scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                    scen.log.refresh()
                    scenario_dir = str(scen.log.logfile.parent)
                    for model_type in enabled_models:
                        row = {
                            "event_iloc": event_iloc,
                            "model_type": model_type,
                            "scenario_directory": scenario_dir,
                            "snakemake_allocation_parse_error": None,
                        }
                        row.update(alloc)
                        rows.append(row)
            return pd.DataFrame(rows)

        model_allocations, parse_error = self._retrieve_snakemake_allocations()
        rows = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scen.log.refresh()
            scenario_dir = str(scen.log.logfile.parent)

            for model_type in enabled_models:
                row = {
                    "event_iloc": event_iloc,
                    "model_type": model_type,
                    "scenario_directory": scenario_dir,
                    "snakemake_allocation_parse_error": parse_error,
                }

                if model_type not in model_allocations:
                    raise ValueError(
                        "Parsed Snakemake allocations are missing model_type "
                        f"'{model_type}'. Available keys: {list(model_allocations.keys())}"
                    )

                alloc = model_allocations[model_type]
                row.update(alloc)
                rows.append(row)
        return pd.DataFrame(rows)

    def _get_performance_summary_row(
        self,
        event_iloc: int,
        model_type: Literal["triton", "tritonswmm", "swmm"],
    ) -> dict[str, float | None]:
        """
        Extract per-category timing totals from the performance summary dataset for one scenario.

        Returns a dict keyed by ``perf_<VarName>`` for all variables in PERF_VARS.
        Values are ``None`` for SWMM rows (no TRITON performance dataset) and for
        TRITON/TRITONSWMM rows where the performance summary has not been written yet.

        Parameters
        ----------
        event_iloc : int
            The event index for the scenario.
        model_type : Literal["triton", "tritonswmm", "swmm"]
            The model type for this row.

        Returns
        -------
        dict[str, float | None]
            Keyed by ``perf_<VarName>`` for each variable in PERF_VARS.
        """
        null_row: dict[str, float | None] = {f"perf_{v}": None for v in PERF_VARS}

        if model_type == "swmm":
            return null_row

        # Gate on log flag — performance summary only written when processing completes
        scen = TRITONSWMM_scenario(event_iloc, self)
        model_log = scen.get_log(model_type)
        if model_log.performance_summary_written is None:
            return null_row
        if model_log.performance_summary_written.get() is not True:
            return null_row

        proc = self._retrieve_sim_run_processing_object(event_iloc)
        if model_type == "tritonswmm":
            ds = proc.TRITONSWMM_performance_summary
        else:  # triton
            ds = proc.TRITON_only_performance_summary
        return {f"perf_{v}": float(ds[v].values.item()) for v in PERF_VARS}

    @staticmethod
    def _reorder_df_status_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder df_status columns into a canonical reader-friendly layout.

        Groups: identity/status → weather/setup → sensitivity params →
        performance breakdown → expected resources → actual resources → snakemake alloc.

        Any columns present in the DataFrame but not in the canonical list are
        appended at the end so no data is silently dropped.

        Parameters
        ----------
        df : pd.DataFrame
            The raw df_status DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns in canonical order.
        """
        fixed_identity = [
            "subanalysis_id",
            "sub_analysis_iloc",
            "event_iloc",
            "model_type",
            "scenario_setup",
            "run_completed",
            "scenario_directory",
        ]
        fixed_perf = [f"perf_{v}" for v in PERF_VARS_ORDERED]
        fixed_resources = [
            "run_mode",
            "n_mpi_procs",
            "n_omp_threads",
            "n_gpus",
            "n_nodes",
            "backend_used",
        ]
        fixed_actual = [
            "actual_nTasks",
            "actual_omp_threads",
            "actual_gpus",
            "actual_total_gpus",
            "actual_gpu_backend",
            "actual_build_type",
        ]
        fixed_snakemake = [
            "snakemake_allocated_nTasks",
            "snakemake_allocated_omp_threads",
            "snakemake_allocated_total_cpus",
            "snakemake_allocation_parse_error",
        ]
        all_fixed = fixed_identity + fixed_perf + fixed_resources + fixed_actual + fixed_snakemake
        # Columns not in any fixed group are weather/setup or sensitivity params —
        # place them between identity and performance (groups 2 & 3).
        dynamic_cols = [c for c in df.columns if c not in all_fixed]
        ordered = [c for c in fixed_identity if c in df.columns]
        ordered += dynamic_cols
        ordered += [c for c in fixed_perf if c in df.columns]
        ordered += [c for c in fixed_resources if c in df.columns]
        ordered += [c for c in fixed_actual if c in df.columns]
        ordered += [c for c in fixed_snakemake if c in df.columns]
        # Append any unexpected columns that slipped through
        ordered += [c for c in df.columns if c not in ordered]
        return df[ordered]

    @property
    def df_status(self):
        """
        Get status DataFrame for all scenarios in the analysis.

        Returns
        -------
        pd.DataFrame
            Long-format status table with one row per (event_iloc, model_type),
            including scenario setup status, model run completion status,
            parsed Snakemake allocated resources, and actual runtime details
            (where available from model logs / reports).
        """
        if self.cfg_analysis.toggle_sensitivity_analysis:
            df_status = self.sensitivity.df_status
            df_status_joined = df_status.merge(
                self.df_snakemake_allocations,
                on=["model_type", "scenario_directory", "event_iloc"],
                how="left",
            )
            allocation_columns = [
                col
                for col in df_status_joined.columns
                if col.startswith("snakemake_") and col != "snakemake_allocation_parse_error"
            ]
            if allocation_columns and df_status_joined[allocation_columns].isna().any().any():
                missing = df_status_joined.loc[
                    df_status_joined[allocation_columns].isna().any(axis=1),
                    ["model_type", "scenario_directory", "event_iloc"],
                ]
                raise ValueError(
                    "Missing Snakemake allocations after join for sensitivity status rows. "
                    f"First missing rows: {missing.head().to_dict(orient='records')}"
                )
            return self._reorder_df_status_columns(df_status_joined)

        enabled_models_untyped = self._get_enabled_model_types()
        enabled_models: list[Literal["triton", "tritonswmm", "swmm"]] = [
            m for m in ("triton", "tritonswmm", "swmm") if m in enabled_models_untyped
        ]

        rows: list[dict] = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scen.log.refresh()
            scenario_setup = scen.log.scenario_creation_complete.get() is True
            scenario_dir = str(scen.log.logfile.parent)

            weather_row = self.df_sims.loc[event_iloc].to_dict()

            for model_type in enabled_models:
                row = dict(weather_row)
                row["event_iloc"] = event_iloc
                row["model_type"] = model_type
                row["scenario_setup"] = scenario_setup
                row["run_completed"] = scen.model_run_completed(model_type)
                row["scenario_directory"] = scenario_dir

                # Provide model-specific expected resources to downstream validators.
                if model_type == "swmm":
                    row["run_mode"] = "serial" if self.cfg_analysis.n_omp_threads == 1 else "openmp"
                    row["n_mpi_procs"] = 1
                    row["n_omp_threads"] = self.cfg_analysis.n_omp_threads or 1
                    row["n_gpus"] = 0
                    row["backend_used"] = "cpu"
                else:
                    row["run_mode"] = self.cfg_analysis.run_mode
                    row["n_mpi_procs"] = self.cfg_analysis.n_mpi_procs or 1
                    row["n_omp_threads"] = self.cfg_analysis.n_omp_threads or 1
                    row["n_gpus"] = (self.cfg_analysis.n_gpus or 0) if self.cfg_analysis.run_mode == "gpu" else 0
                    row["backend_used"] = scen.log.triton_backend_used.get()

                if self.in_slurm:
                    row["n_nodes"] = 1 if model_type == "swmm" else self.cfg_analysis.n_nodes or 1

                # Actual resources (model-dependent availability)
                if model_type == "tritonswmm":
                    log_out_path = (scen.scen_paths.out_tritonswmm or scen.scen_paths.sim_folder) / "log.out"
                    log_data = parse_triton_log_file(log_out_path)
                    row["actual_nTasks"] = log_data["nTasks"]
                    row["actual_omp_threads"] = log_data["omp_threads_per_task"]
                    row["actual_gpus"] = log_data["gpus_per_task"]
                    row["actual_total_gpus"] = log_data["total_gpus"]
                    row["actual_gpu_backend"] = log_data["gpu_backend"]
                    row["actual_build_type"] = log_data["build_type"]
                elif model_type == "triton":
                    log_out_path = (scen.scen_paths.out_triton or scen.scen_paths.sim_folder) / "log.out"
                    log_data = parse_triton_log_file(log_out_path)
                    row["actual_nTasks"] = log_data["nTasks"]
                    row["actual_omp_threads"] = log_data["omp_threads_per_task"]
                    row["actual_gpus"] = log_data["gpus_per_task"]
                    row["actual_total_gpus"] = log_data["total_gpus"]
                    row["actual_gpu_backend"] = log_data["gpu_backend"]
                    row["actual_build_type"] = log_data["build_type"]
                else:  # swmm
                    swmm_report_data = retrieve_swmm_performance_stats_from_rpt(scen.scen_paths.swmm_full_rpt_file)
                    row["actual_nTasks"] = 1
                    row["actual_omp_threads"] = swmm_report_data.get("actual_omp_threads")
                    row["actual_gpus"] = None
                    row["actual_total_gpus"] = None
                    row["actual_gpu_backend"] = "none"
                    row["actual_build_type"] = "SWMM"

                # Performance breakdown from processed summary dataset
                row.update(self._get_performance_summary_row(event_iloc, model_type))

                rows.append(row)

        df_status = pd.DataFrame(rows)
        if self.cfg_analysis.is_subanalysis:
            return self._reorder_df_status_columns(df_status)
        else:
            df_status_joined = df_status.merge(
                self.df_snakemake_allocations,
                on=["model_type", "scenario_directory", "event_iloc"],
                how="left",
            )
            allocation_columns = [
                col
                for col in df_status_joined.columns
                if col.startswith("snakemake_") and col != "snakemake_allocation_parse_error"
            ]
            if allocation_columns and df_status_joined[allocation_columns].isna().any().any():
                missing = df_status_joined.loc[
                    df_status_joined[allocation_columns].isna().any(axis=1),
                    ["model_type", "scenario_directory", "event_iloc"],
                ]
                raise ValueError(
                    "Missing Snakemake allocations after join for status rows. "
                    f"First missing rows: {missing.head().to_dict(orient='records')}"
                )
            return self._reorder_df_status_columns(df_status_joined)

    # TRITON-SWMM model accessors
    @property
    def tritonswmm_TRITON_summary(self):
        return self.process.tritonswmm_TRITON_summary

    @property
    def tritonswmm_performance_summary(self):
        return self.process.tritonswmm_performance_summary

    @property
    def tritonswmm_SWMM_node_summary(self):
        return self.process.tritonswmm_SWMM_node_summary

    @property
    def tritonswmm_SWMM_link_summary(self):
        return self.process.tritonswmm_SWMM_link_summary

    # TRITON-only model accessors
    @property
    def triton_only_summary(self):
        return self.process.triton_only_summary

    @property
    def triton_only_performance_summary(self):
        return self.process.triton_only_performance_summary

    # SWMM-only model accessors
    @property
    def swmm_only_node_summary(self):
        return self.process.swmm_only_node_summary

    @property
    def swmm_only_link_summary(self):
        return self.process.swmm_only_link_summary

    def cancel(self, verbose: bool = True, wait_timeout: int = 120, debug: bool = False) -> dict:
        """
        Cancel ongoing tmux workflow for this analysis.

        This method sends SIGINT to the Snakemake process running in the tmux session,
        which triggers Snakemake's built-in cancel_jobs() to cleanly cancel all worker jobs.

        The method uses persistent log data to identify the session, so it works across
        terminal sessions (close terminal, reopen, reinitialize analysis, call cancel).

        **Key features:**
        - Checks if session is actually running before attempting cancellation
        - Sends SIGINT to Snakemake process for clean cancellation
        - Waits for Snakemake to finish canceling worker jobs
        - Verifies all worker jobs are terminated
        - Gracefully handles already-completed workflows

        Parameters
        ----------
        verbose : bool, default=True
            Print progress messages
        wait_timeout : int, default=120
            Maximum seconds to wait for Snakemake process exit
        debug : bool, default=False
            Print detailed per-iteration diagnostics during the wait loop

        Returns
        -------
        dict
            Cancellation status with keys:
            - success: bool (True if cancellation succeeded or no session running)
            - session_canceled: bool
            - workers_canceled: bool
            - jobs_were_running: bool (False if no session found to cancel)
            - message: str
            - session_name: str | None
            - errors: list[str] (any errors encountered)

        Examples
        --------
        Cancel from same session:
        >>> result = analysis.submit_workflow(wait_for_completion=False)
        >>> # ... later decide to cancel ...
        >>> cancel_result = analysis.cancel()

        Cancel from new terminal session:
        >>> analysis = TRITONSWMM_analysis("config.yaml", system)
        >>> cancel_result = analysis.cancel()  # Loads session name from log
        """
        import datetime
        import subprocess

        if verbose:
            print(
                f"[Cancel] Checking workflow status for analysis '{self.cfg_analysis.analysis_id}'",
                flush=True,
            )

        # Load session info from persistent log
        session_name = self.log.tmux_session_name.get()
        snakemake_pid = self.log.snakemake_pid.get()
        analysis_id = self.cfg_analysis.analysis_id

        # Step 0: Check if tmux session exists
        if not session_name:
            if verbose:
                print(
                    f"[Cancel] No tmux session recorded for analysis '{analysis_id}'",
                    flush=True,
                )
            return {
                "success": True,
                "session_canceled": False,
                "workers_canceled": False,
                "jobs_were_running": False,
                "session_name": None,
                "analysis_id": analysis_id,
                "message": f"No workflow session found for analysis '{analysis_id}'",
                "errors": [],
            }

        # Check if session still exists
        session_check = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            text=True,
        )

        if session_check.returncode != 0:
            if verbose:
                print(
                    f"[Cancel] Tmux session '{session_name}' no longer exists (workflow already completed)",
                    flush=True,
                )
            return {
                "success": True,
                "session_canceled": False,
                "workers_canceled": False,
                "jobs_were_running": False,
                "session_name": session_name,
                "analysis_id": analysis_id,
                "message": f"Tmux session '{session_name}' already completed",
                "errors": [],
            }

        # Session exists, proceed with cancellation
        if verbose:
            print(
                f"[Cancel] Canceling workflow in tmux session '{session_name}'",
                flush=True,
            )

        errors = []

        # Step 1: Get current Snakemake PID (may have changed since submission)
        current_pid = self._workflow_builder._get_snakemake_pid_from_tmux(session_name)
        if current_pid:
            snakemake_pid = current_pid
        elif not snakemake_pid:
            # Could not find PID - try killing session directly
            if verbose:
                print(
                    "[Cancel] WARNING: Could not find Snakemake PID. Killing tmux session directly.",
                    flush=True,
                )
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
            )
            self.log.workflow_canceled.set(True)
            self.log.workflow_cancellation_time.set(datetime.datetime.now().isoformat())

            return {
                "success": True,
                "session_canceled": True,
                "workers_canceled": False,  # Unknown
                "jobs_were_running": True,
                "session_name": session_name,
                "analysis_id": analysis_id,
                "message": "Tmux session killed (PID not found - worker cleanup uncertain)",
                "errors": ["Could not find Snakemake PID for clean cancellation"],
            }

        # Step 2: Send SIGINT to Snakemake process
        if verbose:
            print(
                f"[Cancel] Sending SIGINT to Snakemake (PID {snakemake_pid})...",
                flush=True,
            )

        try:
            os.kill(snakemake_pid, signal.SIGINT)
            if verbose:
                print("[Cancel]   ✓ SIGINT sent", flush=True)
        except ProcessLookupError:
            if verbose:
                print(
                    f"[Cancel]   ⚠ Process {snakemake_pid} already exited",
                    flush=True,
                )
        except PermissionError as e:
            error_msg = f"Permission denied sending SIGINT to PID {snakemake_pid}: {e}"
            errors.append(error_msg)
            if verbose:
                print(f"[Cancel]   ✗ {error_msg}", flush=True)

        # Step 3: Wait for Snakemake to finish canceling jobs
        if verbose:
            print(
                f"[Cancel] Waiting for Snakemake to cancel worker jobs (timeout: {wait_timeout}s)...",
                flush=True,
            )

        start_time = time.time()
        process_exited = False

        while time.time() - start_time < wait_timeout:
            time.sleep(2)

            # Check if process still exists using ps (works across permission boundaries)
            ps_check = subprocess.run(
                ["ps", "-p", str(snakemake_pid)],
                capture_output=True,
            )
            elapsed = int(time.time() - start_time)
            if debug:
                print(
                    f"[Cancel]   [debug] ps returncode={ps_check.returncode} at {elapsed}s",
                    flush=True,
                )
            if ps_check.returncode != 0:
                process_exited = True
                if verbose:
                    print(
                        "[Cancel]   ✓ Snakemake process exited",
                        flush=True,
                    )
                break

        if not process_exited:
            error_msg = f"Snakemake process {snakemake_pid} did not exit within {wait_timeout}s"
            errors.append(error_msg)
            if verbose:
                print(f"[Cancel]   ⚠ {error_msg}", flush=True)
                print(
                    "[Cancel]   (Killing tmux session anyway)",
                    flush=True,
                )

        # Step 4: Verify worker jobs are canceled
        if verbose:
            print("[Cancel] Verifying worker jobs are canceled...", flush=True)

        worker_count = 0
        try:
            result = subprocess.run(
                ["squeue", "-u", "$(whoami)", "-o", "%j", "-h"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=5,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if analysis_id in line:
                        worker_count += 1

            if worker_count > 0:
                error_msg = f"{worker_count} worker jobs still running (Snakemake may not have canceled them)"
                errors.append(error_msg)
                if verbose:
                    print(f"[Cancel]   ⚠ {error_msg}", flush=True)
            elif verbose:
                print("[Cancel]   ✓ All worker jobs canceled", flush=True)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            if verbose:
                print(
                    "[Cancel]   ⚠ Could not verify worker job status",
                    flush=True,
                )

        # Step 5: Kill tmux session
        if verbose:
            print("[Cancel] Cleaning up tmux session...", flush=True)

        kill_result = subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
            text=True,
        )

        if kill_result.returncode == 0:
            if verbose:
                print("[Cancel]   ✓ Tmux session terminated", flush=True)
        else:
            error_msg = f"Failed to kill tmux session: {kill_result.stderr.strip()}"
            errors.append(error_msg)
            if verbose:
                print(f"[Cancel]   ✗ {error_msg}", flush=True)

        # Step 6: Update analysis log
        self.log.workflow_canceled.set(True)
        self.log.workflow_cancellation_time.set(datetime.datetime.now().isoformat())

        success = len(errors) == 0 and worker_count == 0

        if verbose:
            if success:
                print("[Cancel] ✓ Workflow canceled successfully", flush=True)
            else:
                print(
                    "[Cancel] ✗ Cancellation completed with warnings/errors",
                    flush=True,
                )

        return {
            "success": success,
            "session_canceled": True,
            "workers_canceled": worker_count == 0,
            "jobs_were_running": True,
            "session_name": session_name,
            "analysis_id": analysis_id,
            "message": ("Workflow canceled" if success else f"Cancellation issues: {'; '.join(errors)}"),
            "errors": errors,
        }


# %%
