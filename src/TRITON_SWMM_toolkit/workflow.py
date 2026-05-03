"""
Snakemake Workflow Generation Module

This module handles the generation and execution of Snakemake workflows for
TRITON-SWMM simulations. It provides a clean interface for creating workflow
files and submitting them to either local or SLURM execution environments.

Key Components:
- SnakemakeWorkflowBuilder: Main class for workflow generation and submission
- SensitivityAnalysisWorkflowBuilder: Specialized builder for sensitivity analysis workflows
"""

import datetime
import math
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml  # type: ignore

from TRITON_SWMM_toolkit.exceptions import ConfigurationError, WorkflowError
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis
    from .sensitivity_analysis import TRITONSWMM_sensitivity_analysis


class SnakemakeWorkflowBuilder:
    """
    Builder class for generating and executing Snakemake workflows.

    This class encapsulates all Snakemake-related functionality including:
    - Snakefile content generation
    - Dynamic configuration generation
    - Local execution
    - SLURM/HPC execution

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The parent analysis object containing configuration and paths
    """

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        """
        Initialize the workflow builder.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            The parent analysis object containing configuration and paths
        """
        self.analysis = analysis
        self.cfg_analysis = analysis.cfg_analysis
        self.system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        # Prefer an explicit interpreter path for generated shell commands.
        # If analysis stores a generic command ("python"/"python3"), use
        # the current interpreter running this process to avoid PATH issues.
        configured_python = str(analysis._python_executable)
        if configured_python in {"python", "python3"}:
            self.python_executable = sys.executable
        else:
            self.python_executable = configured_python

    def _get_conda_env_path(self) -> Path:
        """Get absolute path to conda environment file.

        The path is embedded in generated Snakefiles via the 'conda:' directive, but
        --use-conda is not currently passed to Snakemake, so the directive is inert.
        The two-environment split is aspirational; this file is currently the single
        working environment for all toolkit work.
        """
        triton_toolkit_root = Path(__file__).parent.parent.parent
        return triton_toolkit_root / "workflow" / "envs" / "triton_swmm.yaml"

    def _get_snakemake_base_cmd(self) -> list[str]:
        """Return command prefix for invoking Snakemake.

        Prefer `python -m snakemake` so execution works even when the
        `snakemake` console script is not on PATH.
        """
        return [sys.executable, "-m", "snakemake"]

    def _check_and_clear_snakemake_lock(self, snakefile_path: Path, dry_run: bool, verbose: bool = True) -> None:
        """Check for a stale Snakemake lock and prompt the user to clear it.

        Snakemake leaves lock files in .snakemake/locks/ when a workflow is
        killed (e.g. SLURM time limit). If not cleared before the next run,
        Snakemake exits immediately with LockException, wasting any queued
        compute allocation.

        Skipped when dry_run=True — dry runs don't submit anything, so a lock
        is not dangerous, and the real submission call will check again.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile (used to build the --unlock command).
        dry_run : bool
            If True, skip the lock check entirely.
        verbose : bool
            If True, print status messages.

        Raises
        ------
        WorkflowError
            If lock files are found and the user declines to unlock, or if
            snakemake --unlock itself fails.
        """
        if dry_run:
            return
        locks_dir = self.analysis_paths.analysis_dir / ".snakemake" / "locks"
        lock_files = list(locks_dir.glob("*.lock")) if locks_dir.exists() else []
        if not lock_files:
            return

        lock_names = ", ".join(f.name for f in lock_files)
        print(
            f"[Snakemake] WARNING: Stale lock files detected in {locks_dir}:",
            flush=True,
        )
        print(f"[Snakemake]   {lock_names}", flush=True)
        print(
            "[Snakemake] This usually means a previous job was killed before Snakemake "
            "could clean up.\n"
            "[Snakemake] Only unlock if no other Snakemake process is currently running "
            "in this directory.",
            flush=True,
        )

        response = input("[Snakemake] Run snakemake --unlock and proceed? [y/N]: ").strip()
        if response.lower() != "y":
            manual_cmd = f"{sys.executable} -m snakemake --unlock --snakefile {snakefile_path}"
            raise WorkflowError(
                phase="pre-submission lock check",
                return_code=1,  # sentinel: user aborted (WorkflowError requires int)
                stderr=(
                    "Workflow submission aborted. If no other Snakemake process is "
                    f"running, unlock manually and retry:\n  {manual_cmd}"
                ),
            )

        unlock_cmd = self._get_snakemake_base_cmd() + [
            "--unlock",
            "--snakefile",
            str(snakefile_path),
        ]
        if verbose:
            print(f"[Snakemake] Running: {' '.join(unlock_cmd)}", flush=True)

        result = subprocess.run(
            unlock_cmd,
            cwd=str(self.analysis_paths.analysis_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorkflowError(
                phase="snakemake --unlock",
                return_code=result.returncode,
                stderr=result.stderr,
            )
        if verbose:
            print("[Snakemake] Unlock successful. Proceeding.", flush=True)

    def _get_config_args(
        self,
        analysis_config_yaml: Path | None = None,
        include_report_config: bool = False,
    ) -> str:
        """
        Generate common config path arguments.

        Parameters
        ----------
        analysis_config_yaml : Path | None
            If provided, use this analysis config instead of self.analysis.analysis_config_yaml
        include_report_config : bool
            Emit ``--report-config`` when ``self._report_config_path`` is set. Default False
            because only renderer rules (the ``_cli`` dispatcher) accept that flag — the
            setup / prepare / run / process / consolidate runners would error on it.

        Returns
        -------
        str
            Config arguments string
        """
        analysis_cfg = analysis_config_yaml or self.analysis.analysis_config_yaml
        args = (
            f"--system-config {self.system.system_config_yaml} \\\n"
            f"            --analysis-config {analysis_cfg}"
        )
        if include_report_config:
            report_cfg_path = getattr(self, "_report_config_path", None)
            if report_cfg_path is not None:
                args += f" \\\n            --report-config {report_cfg_path}"
        return args

    def _build_resource_block(
        self,
        partition: str | None,
        runtime_min: int,
        mem_mb: int,
        nodes: int,
        tasks: int,
        cpus_per_task: int,
        gpus_total: int = 0,
        gpus_per_node_config: int = 0,
        gpu_hardware: str | None = None,
        gpu_alloc_mode: Literal["gres", "gpus"] = "gres",
        mpi: bool = False,
    ) -> str:
        """
        Build a Snakemake resources block.

        Parameters
        ----------
        partition : str | None
            SLURM partition name (defaults to "standard" if None)
        runtime_min : int
            Runtime limit in minutes
        mem_mb : int
            Memory in MB
        nodes : int
            Number of nodes
        tasks : int
            Number of MPI tasks
        cpus_per_task : int
            CPUs per task (OpenMP threads)
        gpus_total : int
            Total GPUs per job (0 if no GPUs)
        gpus_per_node_config : int
            GPUs per node configured for the cluster (0 if no GPUs)
        gpu_hardware : str | None
            GPU model name for SLURM gres/gpus specification
        gpu_alloc_mode : Literal["gres", "gpus"]
            Which SLURM GPU directive to emit in resources
        mpi : bool
            If True, adds mpi=True to resources (required for SLURM executor to set --ntasks > 1)

        Returns
        -------
        str
            Formatted resources block
        """
        if partition is None and (self.cfg_analysis.multi_sim_run_method != "local"):
            raise ValueError("hpc partition must be set when generating SLURM resources")
        partition_name = partition
        if gpus_total > 0 and gpus_per_node_config < 1:
            raise ValueError("hpc_gpus_per_node must be set when requesting GPUs")

        nodes_from_gpu = self._calculate_nodes_for_gpus(gpus_total, gpus_per_node_config)
        sim_nodes = max(nodes, nodes_from_gpu)
        gpus_per_node = math.ceil(gpus_total / sim_nodes) if gpus_total > 0 else 0

        block = f"""        slurm_partition=\"{partition_name}\",
        runtime={runtime_min},"""

        # For GPU jobs: set tasks=1 (1 task per GPU, SLURM executor uses --ntasks-per-gpu)
        # For non-GPU jobs: set tasks=<actual MPI rank count>
        if gpus_total > 0:
            block += "\n        tasks=1,"  # 1:1 GPU-to-task mapping
        else:
            block += f"\n        tasks={tasks},"  # Use actual task count

        block += f"""
        cpus_per_task={cpus_per_task},
        mem_mb={mem_mb},
        nodes={sim_nodes}"""

        # Only set mpi=True for non-GPU MPI jobs (has no effect on GPU jobs)
        if mpi and gpus_total == 0:
            block += ",\n        mpi=True"

        if gpus_total > 0:
            if gpu_alloc_mode == "gpus":
                block += f",\n        gpu={gpus_total}"
                if gpu_hardware:
                    block += f',\n        gpu_model="{gpu_hardware}"'
            else:
                if gpu_hardware:
                    block += f',\n        gres="gpu:{gpu_hardware}:{gpus_per_node}"'
                else:
                    block += f',\n        gres="gpu:{gpus_per_node}"'
        return block

    @staticmethod
    def _calculate_nodes_for_gpus(total_gpus: int, gpus_per_node: int) -> int:
        if total_gpus <= 0:
            return 1
        return max(1, math.ceil(total_gpus / gpus_per_node))

    def _build_plot_rule_block_system_overview(self, input_flag: str = "_status/e_consolidate_complete.flag") -> str:
        """Generate the Snakemake rule for the 2-panel system-overview plot.

        Left panel is the SWMM model elements view (R5); right panel is the
        DEM elevation raster. Combined into one figure per iteration-4
        feedback from the Phase 2 STOP gate.

        ``input_flag`` defaults to the regular multisim consolidation flag
        (`e_consolidate_complete`); the sensitivity master Snakefile passes
        `f_consolidate_master_complete.flag` instead.
        """
        import os as _os

        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args(include_report_config=True)
        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        cfg_ana = self.analysis.cfg_analysis
        if getattr(cfg_ana, "toggle_sensitivity_analysis", False):
            subs = self.analysis.sensitivity.sub_analyses
            first_sub = subs[next(iter(subs))]
            repr_scen_paths = first_sub._retrieve_sim_runs(0)._scenario.scen_paths
        else:
            repr_scen_paths = (
                self.analysis._retrieve_sim_runs(0)._scenario.scen_paths
            )
        # System overview reads: DEM raster + BOTH SWMM .inp files (hydro and
        # hydraulics — the renderer constructs separate swmmio.Model instances
        # for each) + optional BC shapefile. Each source enumerates the variables
        # / sections actually consumed by the renderer.
        source_paths = [
            {
                "path": _os.path.relpath(
                    str(self.system.sys_paths.dem_processed.resolve()), analysis_root
                ),
                "variables": [],  # single-band raster, no subset/indexer enumeration
            },
            {
                "path": _os.path.relpath(
                    str(Path(repr_scen_paths.swmm_hydro_inp).resolve()), analysis_root
                ),
                "variables": ["[SUBCATCHMENTS]", "[JUNCTIONS]", "[OUTFALLS]"],
            },
            {
                "path": _os.path.relpath(
                    str(Path(repr_scen_paths.swmm_hydraulics_inp).resolve()), analysis_root
                ),
                "variables": ["[CONDUITS]", "[JUNCTIONS]", "[POLYGONS]"],
            },
        ]
        if cfg_ana.toggle_storm_tide_boundary and cfg_ana.storm_tide_boundary_line_gis:
            source_paths.append({
                "path": _os.path.relpath(
                    str(Path(cfg_ana.storm_tide_boundary_line_gis).resolve()), analysis_root
                ),
                "variables": [],  # single LineString feature, no subset
            })
        return f'''
rule plot_system_overview:
    input:
        consolidated = "{input_flag}",
    output:
        report(
            "plots/system_overview.png",
            caption="report/captions/system_map.rst",
            category="System Information",
            labels={{"figure": "System map"}},
        )
    params:
        source_paths = {source_paths!r},
        source_paths_rst = {format_sources_rst(source_paths)!r},
    log: "logs/plots/system_overview.log"
    conda: "{conda_env_path}"
    resources: mem_mb=2000, time_min=10
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli system_overview \\
            {config_args} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def _emit_report_artifacts(self, analysis_dir: Path) -> None:
        """Copy report_templates/ -> {analysis_dir}/report/.

        Uses importlib.resources for package-resource resolution (robust across
        editable and site-packages installs). Falls back to Path(__file__) arithmetic
        only when importlib.resources is unavailable. Requires report_templates/
        to ship as package data under src/TRITON_SWMM_toolkit/ via pyproject.toml's
        [tool.setuptools.package-data] entry.

        The Jinja2 workflow_description.rst.j2 template is renamed to
        workflow_description.rst on copy because Snakemake's report engine
        renders all .rst files through Jinja2 — the .j2 extension is a
        repo-side convention, not a Snakemake one.
        """
        try:
            from importlib.resources import files as _resource_files
            src_templates = Path(str(_resource_files("TRITON_SWMM_toolkit") / "report_templates"))
        except (ImportError, ModuleNotFoundError):
            src_templates = Path(__file__).parent / "report_templates"

        dst_report = analysis_dir / "report"
        dst_report.mkdir(parents=True, exist_ok=True)
        (dst_report / "report.css").write_text((src_templates / "report.css").read_text())
        captions_dst = dst_report / "captions"
        captions_dst.mkdir(exist_ok=True)
        for cap in (src_templates / "captions").glob("*.rst"):
            (captions_dst / cap.name).write_text(cap.read_text())
        (dst_report / "workflow_description.rst").write_text(
            (src_templates / "workflow_description.rst.j2").read_text()
        )

    def generate_snakefile_content(
        self,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: str = "TRITON",
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        report_config_path: "Path | None" = None,
    ) -> str:
        """
        Generate Snakefile content with separate rules for prep, simulation, and processing.

        This creates a five-phase workflow:
        1. Setup: System inputs processing and compilation
        2. Scenario preparation: SWMM model generation (lightweight, 1 CPU)
        3. Simulation execution: TRITON-SWMM runs (resource-intensive, GPUs/CPUs)
        4. Output processing: Timeseries extraction and compression (I/O bound, 1-2 CPUs)
        5. Consolidation: Analysis-level output aggregation

        Parameters
        ----------
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
        which : str
            Which outputs to process: "TRITON", "SWMM", or "both"
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_outputs_if_already_created : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Complete Snakefile content as a string
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        self._report_config_path = report_config_path

        # Emit report templates (CSS, captions, Jinja2 workflow description) into
        # {analysis_dir}/report/ so Snakemake's --report engine can resolve the
        # caption= / report: paths inside generated rules at report-render time.
        self._emit_report_artifacts(self.analysis_paths.analysis_dir)

        n_sims = len(self.analysis.df_sims)
        event_ids = [
            compute_event_id_slug(
                self.analysis._retrieve_weather_indexer_using_integer_index(i)
            )
            for i in range(n_sims)
        ]
        iloc_by_event_id = {event_ids[i]: i for i in range(n_sims)}
        hpc_time_min = self.cfg_analysis.hpc_time_min_per_sim or 30

        mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
        omp_threads = self.cfg_analysis.n_omp_threads or 1
        n_gpus = self.cfg_analysis.n_gpus or 0
        cpus_per_sim = mpi_ranks * omp_threads

        # CRITICAL: Snakemake's SLURM executor uses max(threads, tasks×cpus_per_task) for --ntasks
        # We must set threads = total CPUs needed to ensure correct SLURM allocation
        # Even though we also set resources.tasks and resources.cpus_per_task correctly,
        # Snakemake will underallocate if threads < required CPUs
        snakemake_threads = cpus_per_sim

        # Conservative estimate: 2GB per CPU (can be made configurable later)
        mem_mb_per_sim = self.cfg_analysis.mem_gb_per_cpu * cpus_per_sim * 1000
        n_nodes = self.cfg_analysis.n_nodes or 1
        gpus_per_node_config = self.cfg_analysis.hpc_gpus_per_node or 0
        gpu_alloc_mode = self.system.cfg_system.preferred_slurm_option_for_allocating_gpus or "gpus"

        # Get absolute path to conda environment file using helper
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args()
        skip_setup = not (process_system_level_inputs or compile_TRITON_SWMM)

        # Make log dirs
        analysis_dir = self.analysis_paths.analysis_dir
        log_dir = self.analysis_paths.analysis_log_directory
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "sims").mkdir(parents=True, exist_ok=True)

        if skip_setup:
            setup_shell = '''"""
        touch {output}
        """
        '''
        else:
            tritonswmm_model = self.system.cfg_system.toggle_tritonswmm_model
            setup_shell = f'''"""
        {self.python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            {config_args} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {"--compile-triton-swmm " if compile_TRITON_SWMM and tritonswmm_model else ""}\\
            {"--compile-triton-only " if compile_TRITON_SWMM and self.system.cfg_system.toggle_triton_model else ""}\\
            {"--compile-swmm " if compile_TRITON_SWMM and self.system.cfg_system.toggle_swmm_model else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """'''

        # Build resource blocks using helper
        setup_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=30,
            mem_mb=self.cfg_analysis.mem_gb_per_cpu * 1000,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
        )

        # Scenario preparation: lightweight (1 CPU, minimal memory)
        prep_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=30,
            mem_mb=self.cfg_analysis.mem_gb_per_cpu * 1000,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
        )

        # Simulation: resource-intensive (multi-CPU, GPUs, high memory)
        sim_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_ensemble_partition,
            runtime_min=hpc_time_min,
            mem_mb=mem_mb_per_sim,
            nodes=n_nodes,
            tasks=mpi_ranks,
            cpus_per_task=omp_threads,
            gpus_total=n_gpus,
            gpus_per_node_config=gpus_per_node_config,
            gpu_hardware=self.system.cfg_system.gpu_hardware,
            gpu_alloc_mode=gpu_alloc_mode,
            mpi=(self.cfg_analysis.run_mode in ["hybrid", "mpi"]),
        )

        # Output processing: I/O bound (1-2 CPUs for compression)
        process_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=120,
            mem_mb=self.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
            nodes=1,
            tasks=1,
            cpus_per_task=2,  # Parallel compression
        )

        # Consolidation resources
        consolidate_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=30,
            mem_mb=self.cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
            nodes=1,
            tasks=1,
            cpus_per_task=2,
        )

        log_dir_str = str(log_dir)
        analysis_id_str = str(self.cfg_analysis.analysis_id)
        snakefile_content = f'''# Auto-generated by TRITONSWMM_analysis

import os
import glob
import subprocess
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = {analysis_id_str!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {n_sims}
config["is_sensitivity"] = False
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

report: "report/workflow_description.rst"

SIM_IDS = {event_ids!r}
ILOC_BY_EVENT_ID = {iloc_by_event_id!r}

rule all:
    input:
        "_status/e_consolidate_complete.flag",
        "plots/system_overview.png",
        expand("plots/per_sim/{{event_id}}/peak_flood_depth.png", event_id=SIM_IDS),
        expand("plots/per_sim/{{event_id}}/conduit_flow.png",     event_id=SIM_IDS),
        "plots/per_analysis/summary_table.svg",
        "plots/appendix/scenario_status.html",
        "plots/errors_and_warnings/validation_report.html",

onsuccess:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)

onerror:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)

rule setup:
    output: "_status/a_setup_complete.flag"
    log: "{log_dir_str}/setup.log"
    conda: "{conda_env_path}"
    resources:
{setup_resources}
    shell:
        {setup_shell}
'''

        # Add scenario preparation rule if requested
        if prepare_scenarios:
            snakefile_content += f'''
rule prepare_scenario:
    input: "_status/a_setup_complete.flag"
    output: "_status/b_prepare_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/prepare_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{prep_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.prepare_scenario_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            {"--overwrite-scenario-if-already-set-up " if overwrite_scenario_if_already_set_up else ""}\\
            {"--rerun-swmm-hydro " if rerun_swmm_hydro_if_outputs_exist else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """
'''

        # Add simulation rules (separate rules per model type)
        sim_input = (
            "_status/b_prepare_evt-{event_id}_complete.flag" if prepare_scenarios else "_status/a_setup_complete.flag"
        )

        # Determine which model types are enabled
        enabled_models = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")

        if not enabled_models:
            raise ValueError(
                "No model types enabled! Enable at least one of: toggle_triton_model, toggle_tritonswmm_model, toggle_swmm_model"  # noqa: E501
            )

        # Generate separate simulation rule for each enabled model type
        for model_type in enabled_models:
            # For SWMM, use fixed CPU-only resources (no GPU, limited threads)
            if model_type == "swmm":
                swmm_cpus = self.cfg_analysis.n_omp_threads or 1
                swmm_resources = self._build_resource_block(
                    partition=self.cfg_analysis.hpc_ensemble_partition,
                    runtime_min=hpc_time_min,
                    mem_mb=self.cfg_analysis.mem_gb_per_cpu * swmm_cpus * 1000,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=swmm_cpus,
                    gpus_total=0,  # SWMM has no GPU support
                    gpus_per_node_config=0,
                )
                model_resources = swmm_resources
                model_threads = swmm_cpus
            else:
                # TRITON and TRITON-SWMM use configured resources
                model_resources = sim_resources
                model_threads = snakemake_threads

            snakefile_content += f'''
rule run_{model_type}:
    input: "{sim_input}"
    output: "_status/c_run_{model_type}_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/{model_type}_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    threads: {model_threads}
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{model_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.run_simulation_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            --model-type {model_type} \\
            {"--pickup-where-leftoff " if pickup_where_leftoff else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """
'''

        # Add output processing rules (one per model type) if requested
        if process_timeseries:
            for model_type in enabled_models:
                # Determine --which flag based on model type
                if model_type == "triton":
                    which_arg = "TRITON"
                elif model_type == "tritonswmm":
                    which_arg = "both"
                elif model_type == "swmm":
                    which_arg = "SWMM"
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                snakefile_content += f'''
rule process_{model_type}:
    input: "_status/c_run_{model_type}_evt-{{event_id}}_complete.flag"
    output: "_status/d_process_{model_type}_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/process_{model_type}_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{process_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.process_timeseries_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            --model-type {model_type} \\
            --which {which_arg} \\
            {"--clear-raw-outputs " if clear_raw_outputs else ""}\\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """
'''

        # Consolidation rule depends on final output of each model type
        # Build list of all output flags from all enabled models
        consolidate_inputs = []
        for model_type in enabled_models:
            if process_timeseries:
                flag_pattern = f"d_process_{model_type}_evt-{{event_id}}_complete.flag"
            else:
                flag_pattern = f"c_run_{model_type}_evt-{{event_id}}_complete.flag"
            consolidate_inputs.append(f'expand("_status/{flag_pattern}", event_id=SIM_IDS)')

        # Join all input patterns
        consolidate_input_str = " + ".join(consolidate_inputs)

        snakefile_content += f'''
rule consolidate:
    input: {consolidate_input_str}
    output: "_status/e_consolidate_complete.flag"
    log: "{log_dir_str}/consolidate.log"
    conda: "{conda_env_path}"
    resources:
{consolidate_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {config_args} \\
            --compression-level {compression_level} \\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --which {which} \\
            > {{log}} 2>&1
        touch {{output}}
        """
'''
        snakefile_content += self._build_plot_rule_block_system_overview()
        snakefile_content += self._build_plot_rule_block_per_sim()
        snakefile_content += self._build_plot_rule_block_per_analysis_summary()
        snakefile_content += self._build_plot_rule_block_scenario_status_appendix()
        snakefile_content += self._build_plot_rule_block_errors_and_warnings()
        return snakefile_content

    def _collect_per_analysis_summary_source_paths(self) -> list[dict]:
        """Return analysis-dir-relative .rpt + TRITON-log descriptors the renderer reads.

        Per Gotcha 5: dispatch on enabled model types — `swmm_hydraulics_rpt`
        for TRITON-SWMM coupled mode, `swmm_full_rpt_file` for SWMM-only;
        `log_run_tritonswmm` / `log_run_triton` for TRITON-side logs.

        Each returned dict has the schema ``{"path": str, "variables": list[str]}``
        — the variable list names which fields the renderer parses from each
        source (e.g., "Flow Routing Continuity error" from SWMM .rpt). Caption
        RSTs render the dict as a path bullet with variable sub-bullets, with a
        backward-compat shim for callers still returning ``list[str]``.

        Sensitivity-master detection: if the analysis is a sensitivity master,
        iterate per-sub-analysis scenarios so the master per_analysis_summary
        table has provenance for every sub-analysis's status counts (per
        Iteration 6 "show all sub-analyses" scope).
        """
        import os as _os

        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(Path(analysis_dir).resolve())
        enabled = self.analysis._get_enabled_model_types()
        sources: list[dict] = []
        # Sensitivity-master scope: iterate every sub-analysis's scenarios.
        is_sensitivity_master = (
            getattr(self.analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
            and getattr(self.analysis, "sensitivity", None) is not None
        )
        if is_sensitivity_master:
            scenario_objs = []
            for sub in self.analysis.sensitivity.sub_analyses.values():
                for event_iloc in sub.df_sims.index:
                    try:
                        scenario_objs.append(
                            sub._retrieve_sim_run_processing_object(event_iloc).scen_paths
                        )
                    except Exception:
                        continue
        else:
            scenario_objs = []
            for event_iloc in self.analysis.df_sims.index:
                scenario_objs.append(
                    self.analysis._retrieve_sim_run_processing_object(event_iloc).scen_paths
                )
        for scen_paths in scenario_objs:
            if "tritonswmm" in enabled and scen_paths.swmm_hydraulics_rpt:
                sources.append({
                    "path": _os.path.relpath(
                        str(Path(scen_paths.swmm_hydraulics_rpt).resolve()),
                        analysis_root,
                    ),
                    "variables": ["Flow Routing Continuity error (%)"],
                })
            elif "swmm" in enabled and scen_paths.swmm_full_rpt_file:
                sources.append({
                    "path": _os.path.relpath(
                        str(Path(scen_paths.swmm_full_rpt_file).resolve()),
                        analysis_root,
                    ),
                    "variables": ["Flow Routing Continuity error (%)"],
                })
            # Per-model-type model-state JSON logs (sim_folder/log_{mt}.json) are
            # what the renderer's _is_scenario_successful / _is_scenario_pending
            # actually read for status counts — NOT the simulation execution
            # logs (log_run_*) that the renderer never opens. Enumerate one entry
            # per enabled model type per scenario.
            sim_folder = getattr(scen_paths, "sim_folder", None)
            if sim_folder is not None:
                for mt in enabled:
                    log_file = Path(sim_folder) / f"log_{mt}.json"
                    if log_file.exists():
                        sources.append({
                            "path": _os.path.relpath(
                                str(log_file.resolve()), analysis_root
                            ),
                            "variables": [
                                f"model_run_completed[{mt}] (status flag for n_successful / n_pending counts)",
                            ],
                        })
        return sources

    def _build_plot_rule_block_per_analysis_summary(self, input_flag: str = "_status/e_consolidate_complete.flag") -> str:
        """Generate the Snakemake rule for the per-analysis summary table (R7).

        Produces `plots/per_analysis/summary_table.svg` — a deterministic
        metrics table (n sims, n successful/pending/failed, average TRITON +
        SWMM flow-routing continuity errors). Per Phase 5 plan: SWMM .rpt
        parsing delegates to swmm_output_parser.return_swmm_system_outputs,
        not in-renderer regex.

        ``input_flag`` defaults to the regular multisim consolidation flag
        (`e_consolidate_complete`); the sensitivity master Snakefile passes
        `f_consolidate_master_complete.flag` instead.
        """
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args(include_report_config=True)
        source_paths = self._collect_per_analysis_summary_source_paths()
        return f'''
rule plot_per_analysis_summary_table:
    input:
        consolidated = "{input_flag}",
    output:
        report(
            "plots/per_analysis/summary_table.svg",
            caption="report/captions/per_analysis_summary_table.rst",
            category="Workflow Status",
            subcategory="Workflow Health Summary",
            labels={{"figure": "Summary table"}},
        )
    params:
        source_paths = {source_paths!r},
        source_paths_rst = {format_sources_rst(source_paths)!r},
    log: "logs/plots/per_analysis_summary_table.log"
    conda: "{conda_env_path}"
    resources: mem_mb=2000, time_min=5
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli per_analysis_summary \\
            {config_args} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def _build_plot_rule_block_scenario_status_appendix(self, input_flag: str = "_status/e_consolidate_complete.flag") -> str:
        """Generate the Snakemake rule for the scenario_status.csv Appendix table.

        Iter 8 agenda item 3: produces `plots/appendix/scenario_status.html` —
        an inline-styled HTML table rendered from `analysis_dir / scenario_status.csv`
        (written by `export_scenario_status.py` as a Snakemake onsuccess/onerror
        hook). Sidebar category is "Appendix"; the comparator-fallback in the
        category-order post-process places it after all known categories
        alphabetically.

        ``input_flag`` defaults to the regular multisim consolidation flag;
        the sensitivity master Snakefile passes the master flag instead.
        """
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args(include_report_config=True)
        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        import os as _os
        # Source: scenario_status.csv. The file is written by
        # export_scenario_status.py at workflow close (onsuccess/onerror hook).
        # Even though the renderer also handles a missing CSV, declare it as
        # the source so the caption shows what the renderer reads.
        csv_rel = _os.path.relpath(str((analysis_dir / "scenario_status.csv").resolve()), analysis_root)
        source_paths = [{
            "path": csv_rel,
            "variables": ["event_id", "model_type", "status", "runtime_s", "continuity_error_pct", "notes"],
        }]
        return f'''
rule plot_scenario_status_appendix:
    input:
        consolidated = "{input_flag}",
    output:
        report(
            "plots/appendix/scenario_status.html",
            caption="report/captions/scenario_status_appendix.rst",
            category="Appendix",
            subcategory="Scenario Status",
            labels={{"figure": "Per-scenario status table"}},
        )
    params:
        source_paths = {source_paths!r},
        source_paths_rst = {format_sources_rst(source_paths)!r},
    log: "logs/plots/scenario_status_appendix.log"
    conda: "{conda_env_path}"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli scenario_status_appendix \\
            {config_args} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def _build_plot_rule_block_errors_and_warnings(self, input_flag: str = "_status/e_consolidate_complete.flag") -> str:
        """Generate the Snakemake rule for the Errors and Warnings validation report.

        Iter 9 agenda: produces `plots/errors_and_warnings/validation_report.html`
        — calls `analysis_validation.validate_analysis(analysis)` and renders
        a structured pass/fail report organized into 4 sections (system-level
        checks; aggregate per-scenario; granular per-scenario failures;
        resource-utilization mismatches). Replaces the placeholder injection
        for "Errors and Warnings" added in Subiteration 8.1.

        ``input_flag`` defaults to the regular multisim consolidation flag;
        the sensitivity master Snakefile passes the master flag instead.
        """
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args(include_report_config=True)
        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        import os as _os
        # Sources the renderer actually reads: per-scenario JSON logs (status
        # / setup / run completion), scenario_status.csv (resource-usage
        # validation), and the system_log.json (compilation status).
        csv_rel = _os.path.relpath(str((analysis_dir / "scenario_status.csv").resolve()), analysis_root)
        source_paths = [
            {
                "path": csv_rel,
                "variables": [
                    "scenario_setup", "run_completed",
                    "actual_nTasks", "actual_omp_threads", "actual_total_gpus", "actual_gpu_backend",
                ],
            },
            {
                "path": "sims/<event_id>/log_<model_type>.json",
                "variables": ["simulation_completed (per scenario × model_type)"],
            },
            {
                "path": "../system_log.json",
                "variables": ["compilation_successful", "compilation_triton_only_successful", "compilation_swmm_successful"],
            },
        ]
        return f'''
rule plot_errors_and_warnings:
    input:
        consolidated = "{input_flag}",
    output:
        report(
            "plots/errors_and_warnings/validation_report.html",
            caption="report/captions/errors_and_warnings.rst",
            category="Errors and Warnings",
            subcategory="Validation Report",
            labels={{"figure": "Validation report"}},
        )
    params:
        source_paths = {source_paths!r},
        source_paths_rst = {format_sources_rst(source_paths)!r},
    log: "logs/plots/errors_and_warnings.log"
    conda: "{conda_env_path}"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli errors_and_warnings \\
            {config_args} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def _build_plot_rule_block_per_sim(self) -> str:
        """Generate two per-sim plot rules wildcarded over event_id (Phase 3, R6).

        `params.source_paths` for each rule is a function-based lookup
        (`_per_sim_*_sources`) that reads event-scoped paths at rule-schedule
        time (not Snakefile-emit time) — keeps the generated Snakefile
        readable even for large scenario counts. The wildcards' `event_id`
        is mapped to the integer `event_iloc` argument expected by the
        renderer CLI via the Snakefile-level `ILOC_BY_EVENT_ID` dict (set
        alongside `SIM_IDS` in `generate_snakefile_content`).

        System-level paths (DEM, watershed shapefile) used by the
        peak_flood_depth renderer are computed at emit time and baked into
        the closure as default kwargs — they are constant per analysis
        regardless of event wildcard.
        """
        import os as _os
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args(include_report_config=True)
        analysis_root = str(self.analysis.analysis_paths.analysis_dir.resolve())
        dem_rel = _os.path.relpath(
            str(self.system.sys_paths.dem_processed.resolve()), analysis_root
        )
        watershed_path = self.system.cfg_system.watershed_gis_polygon
        watershed_rel = _os.path.relpath(
            str(Path(watershed_path).resolve()), analysis_root
        ) if watershed_path else None
        rainfall_datavar = self.analysis.cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
        storm_tide_datavar = self.analysis.cfg_analysis.weather_time_series_storm_tide_datavar
        return f'''
def _per_sim_flood_depth_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "peak_flood_depth",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        dem_rel_path={dem_rel!r},
        watershed_rel_path={watershed_rel!r},
    )

def _per_sim_conduit_flow_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "conduit_flow",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        dem_rel_path={dem_rel!r},
        watershed_rel_path={watershed_rel!r},
    )

rule plot_per_sim_peak_flood_depth:
    input:
        consolidated = "_status/e_consolidate_complete.flag",
    output:
        report(
            "plots/per_sim/{{event_id}}/peak_flood_depth.png",
            caption="report/captions/per_sim_peak_flood_depth.rst",
            category="Per Simulation Results",
            labels={{"event_id": "{{event_id}}", "figure": "Peak flood depth"}},
        )
    params:
        source_paths = _per_sim_flood_depth_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_flood_depth_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID[w.event_id],
    log: "logs/plots/per_sim_peak_flood_depth_{{event_id}}.log"
    conda: "{conda_env_path}"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli per_sim_peak_flood_depth \\
            {config_args} \\
            --event-iloc {{params.event_iloc}} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """

rule plot_per_sim_conduit_flow:
    input:
        consolidated = "_status/e_consolidate_complete.flag",
    output:
        report(
            "plots/per_sim/{{event_id}}/conduit_flow.png",
            caption="report/captions/per_sim_conduit_flow.rst",
            category="Per Simulation Results",
            labels={{"event_id": "{{event_id}}", "figure": "Conduit flow"}},
        )
    params:
        source_paths = _per_sim_conduit_flow_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_conduit_flow_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID[w.event_id],
    log: "logs/plots/per_sim_conduit_flow_{{event_id}}.log"
    conda: "{conda_env_path}"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli per_sim_conduit_flow \\
            {config_args} \\
            --event-iloc {{params.event_iloc}} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def generate_snakemake_config(self, mode: Literal["local", "slurm", "single_job"]) -> dict:
        """
        Generate dynamic snakemake config based on analysis_config and system_config.

        Supports three execution modes:
        - local: Uses cores based on system capabilities
        - slurm: Uses 'executor: slurm' with job steps (many SLURM jobs)
        - single_job: Behaves like local execution but respects SLURM allocation
          (one SLURM job with many srun tasks inside)

        Parameters
        ----------
        mode : Literal["local", "slurm", "single_job"]
            Execution mode (local, slurm, or single_job)

        Returns
        -------
        dict
            Snakemake configuration dictionary
        """
        # Base config shared by all modes
        config = {
            "use-conda": False,
            "conda-frontend": "mamba",
            "printshellcmds": True,
            "rerun-incomplete": True,
            "keep-going": True,
            "rerun-triggers": ["mtime", "input"],
        }
        assert isinstance(self.cfg_analysis.local_cpu_cores_for_workflow, int), (
            "local_cpu_cores_for_workflow must be specified for local runs"
        )
        if mode == "local":
            config.update(
                {
                    "cores": self.cfg_analysis.local_cpu_cores_for_workflow,
                    "keep-going": True,
                }
            )
        elif mode == "single_job":
            # Single-job mode: cores and GPU resources set dynamically via CLI in SBATCH script
            # Don't set cores or resources here - will be passed via CLI args in SBATCH script
            config.update(
                {
                    "keep-going": True,  # Continue other sims if one fails
                    "latency-wait": 60,
                }
            )
        else:  # slurm
            # SLURM mode: support both modern executor and legacy cluster modes
            slurm_partition = self.cfg_analysis.hpc_ensemble_partition
            max_concurrent = self.cfg_analysis.hpc_max_simultaneous_sims
            assert isinstance(max_concurrent, int), (
                "hpc_max_simultaneous_sims is required for generate_snakemake_config"
            )
            # Modern executor mode: uses 'executor: slurm' with job steps
            config.update(
                {
                    "executor": "slurm",
                    "jobs": max_concurrent,
                    "latency-wait": 60,
                    "max-jobs-per-second": 5,
                    "max-status-checks-per-second": 10,
                    "default-resources": [
                        "nodes=1",
                        "mem_mb=2000",
                        "runtime=30",
                        f"slurm_partition={slurm_partition}",
                        f"slurm_account={self.cfg_analysis.hpc_account}",
                    ],
                    "slurm": {
                        "sbatch": {
                            "partition": "{resources.slurm_partition}",
                            "account": "{resources.slurm_account}",
                        }
                    },
                }
            )

        return config

    def write_snakemake_config(self, config: dict, mode: Literal["local", "slurm", "single_job"]) -> Path:
        """
        Write snakemake config to analysis directory.

        Parameters
        ----------
        config : dict
            Snakemake configuration dictionary
        mode : Literal["local", "slurm", "single_job"]
            Execution mode (local, slurm, or single_job)

        Returns
        -------
        Path
            Path to the written config directory
        """
        config_dir = self.analysis_paths.analysis_dir / ".snakemake_profile" / mode
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                width=float("inf"),  # Prevent YAML from breaking long lines
            )

        return config_dir

    def _generate_single_job_submission_script(
        self,
        snakefile_path: Path,
        config_dir: Path,
        override_hpc_total_nodes: int | None = None,
    ) -> Path:
        """
        Generate SLURM batch script that runs Snakemake.

        For 1_job_many_srun_tasks mode, this requests exclusive access to nodes
        specified by hpc_total_nodes. Concurrency is determined dynamically from
        the SLURM allocation rather than being pre-calculated.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        config_dir : Path
            Path to the Snakemake profile config directory
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for this submission
            without mutating the config object. Only valid for 1_job_many_srun_tasks mode.

        Returns
        -------
        Path
            Path to the generated batch script
        """
        import TRITON_SWMM_toolkit.utils as ut

        batch_log_path = self.analysis.analysis_paths.analysis_log_directory / "_slurm_logs"
        batch_log_path.mkdir(exist_ok=True, parents=True)
        # Get per-simulation resource requirements (without requiring totals)
        sim_resources = self.analysis._resource_manager._get_simulation_resource_requirements()

        # Get total nodes — use override if provided, otherwise fall back to config
        total_nodes = override_hpc_total_nodes if override_hpc_total_nodes is not None else self.cfg_analysis.hpc_total_nodes  # noqa: E501
        assert isinstance(total_nodes, int), "hpc_total_nodes required for 1_job_many_srun_tasks mode"

        # Get job duration
        job_time = self.cfg_analysis.hpc_total_job_duration_min
        assert isinstance(job_time, int), "hpc_total_job_duration_min required"

        assert self.analysis.in_slurm, "_generate_submission_script only makes sense to run in a SLURM environment."

        # Convert to HH:MM:SS format
        hours = job_time // 60
        minutes = job_time % 60
        estimated_time = f"{hours:02d}:{minutes:02d}:00"

        additional_sbatch_args = ""
        if self.cfg_analysis.additional_SBATCH_params:
            additional_sbatch_args = "#SBATCH "
            additional_sbatch_args += "\n#SBATCH ".join(self.cfg_analysis.additional_SBATCH_params)

        modules = self.analysis._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        module_load_cmd = ""
        if modules:
            module_load_cmd = f"module load {modules}"

        # Conda initialization for non-interactive shells
        # In SLURM batch scripts, conda's shell integration is not automatically available
        # Strategy: After module load sets CONDA_EXE, use conda's shell hook to initialize
        conda_init_cmd = """
# Initialize conda for non-interactive shell (required in SLURM batch scripts)
# After 'module load miniforge3', CONDA_EXE is set by the module system
# Use conda's shell hook for robust initialization
if [ -n "${CONDA_EXE}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization. CONDA_EXE and CONDA_PREFIX are both unset."
    echo "  CONDA_EXE=${CONDA_EXE:-<not set>}"
    echo "  CONDA_PREFIX=${CONDA_PREFIX:-<not set>}"
    exit 1
fi

conda activate triton_swmm_toolkit

# Fix for Frontier: conda activate in SLURM batch scripts doesn't add lib to LD_LIBRARY_PATH
# Explicitly add conda lib directory to ensure shared libraries (like libproj.so.25) are found
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "Added ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"
else
    echo "WARNING: CONDA_PREFIX not set after conda activate"
fi

# Fix for Frontier: GPFS /ccs/home compute node mounts do not support HDF5 POSIX byte-range
# locking. Without this, xr.open_dataset() on NetCDF-4 files fails with errno 524 on compute
# nodes. This env var is inherited by all srun child steps via SLURM's default --export=ALL.
export HDF5_USE_FILE_LOCKING=FALSE

# ===================================================================
# DIAGNOSTIC OUTPUT - Environment state after LD_LIBRARY_PATH fix
# ===================================================================
echo "=========================================="
echo "DIAGNOSTICS: Environment after LD_LIBRARY_PATH fix"
echo "=========================================="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<not set>}"
echo ""
echo "LD_LIBRARY_PATH (line-by-line):"
echo "${LD_LIBRARY_PATH:-<not set>}" | tr ':' '\n' | sed 's/^/  /'
echo ""
echo "Python executable:"
which python
echo ""
echo "Checking for libproj.so.25 in conda env:"
if [ -n "${CONDA_PREFIX}" ]; then
    ls -la ${CONDA_PREFIX}/lib/libproj.so* 2>&1 || echo "  libproj.so* not found"
else
    echo "  CONDA_PREFIX not set, cannot check"
fi
echo ""
echo "Verification: Is conda lib in LD_LIBRARY_PATH?"
if [[ "${LD_LIBRARY_PATH}" == *"${CONDA_PREFIX}/lib"* ]]; then
    echo "  ✓ YES - ${CONDA_PREFIX}/lib is in LD_LIBRARY_PATH"
else
    echo "  ✗ NO - ${CONDA_PREFIX}/lib is NOT in LD_LIBRARY_PATH"
fi
echo "=========================================="
echo ""
"""

        # Build GPU directive if needed
        # Check if any simulation uses GPUs (handles sensitivity analysis)
        n_gpus_per_sim = sim_resources["n_gpus"]
        gpu_directive = ""
        gpu_calculation = ""
        gpu_cli_arg = ""

        if n_gpus_per_sim > 0:
            gpus_per_node = self.cfg_analysis.hpc_gpus_per_node
            assert isinstance(gpus_per_node, int), (
                "hpc_gpus_per_node required when using GPUs in 1_job_many_srun_tasks mode"
            )
            # --gres/--gpus-per-node are per-node, SLURM will multiply by --nodes automatically
            gpu_hardware = self.system.cfg_system.gpu_hardware
            if gpu_hardware:
                gpu_directive = f"#SBATCH --gres=gpu:{gpu_hardware}:{gpus_per_node}\n"
            else:
                gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}\n"
            # Calculate total GPUs dynamically in bash script
            gpu_calculation = f"\n# Calculate total GPUs from SLURM allocation\nTOTAL_GPUS=$((SLURM_JOB_NUM_NODES * {gpus_per_node}))\n"  # noqa: E501
            gpu_cli_arg = " --resources gpu=$TOTAL_GPUS"

        script_content = f"""#!/bin/bash
#SBATCH --job-name=triton_workflow
#SBATCH --partition={self.cfg_analysis.hpc_ensemble_partition}
#SBATCH --account={self.cfg_analysis.hpc_account}
#SBATCH --nodes={total_nodes}
#SBATCH --exclusive
{gpu_directive}#SBATCH --time={estimated_time}
#SBATCH --output={str(batch_log_path)}/workflow_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
#SBATCH --error={str(batch_log_path)}/workflow_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
{additional_sbatch_args}

module purge

# Load required modules
{module_load_cmd}

{conda_init_cmd}

# Calculate total CPUs dynamically from SLURM allocation
if [ -z "$SLURM_CPUS_ON_NODE" ]; then
    echo "ERROR: SLURM_CPUS_ON_NODE not set. Cannot determine CPU allocation."
    exit 1
fi
TOTAL_CPUS=$((SLURM_CPUS_ON_NODE * SLURM_JOB_NUM_NODES))
{gpu_calculation}
# Run Snakemake with dynamic resource limits
${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} --snakefile {snakefile_path} \\
    --cores $TOTAL_CPUS{gpu_cli_arg}
"""

        script_path = self.analysis_paths.analysis_dir / "run_workflow_1job.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def run_snakemake_local(
        self,
        snakefile_path: Path,
        verbose: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Run Snakemake workflow on local machine.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages
        dry_run : bool
            If True, perform a Snakemake dry run only

        Returns
        -------
        dict
            Status dictionary
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow locally with Snakemake",
                    flush=True,
                )
                if dry_run:
                    print(
                        "[Snakemake] DRY RUN",
                        flush=True,
                    )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="local")
            config_dir = self.write_snakemake_config(config, mode="local")

            if verbose:
                print(f"[Snakemake] Using dynamic config from: {config_dir}", flush=True)

            # Create log directory and file for Snakemake output
            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile_name = "snakemake_master_dry_run.log" if dry_run else "snakemake_master.log"
            snakemake_logfile = logs_dir / logfile_name

            if verbose:
                print(
                    f"[Snakemake] Snakemake output will be logged to: {snakemake_logfile}",
                    flush=True,
                )

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
            ]

            # Explicitly pass --cores for multicore local runs
            # (ensures CLI-level cores setting when profile behavior varies)
            local_cores = self.cfg_analysis.local_cpu_cores_for_workflow
            assert isinstance(local_cores, int), "local_cpu_cores_for_workflow must be specified for local runs"
            if local_cores > 1:
                cmd_args.extend(["--cores", str(local_cores)])

            # Add dry-run flag last
            if dry_run:
                cmd_args.append("--dry-run")

            # Check for stale lock before running Snakemake locally (skipped on dry runs)
            self._check_and_clear_snakemake_lock(snakefile_path, dry_run=dry_run, verbose=verbose)

            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if verbose:
                cmd = " ".join(cmd_args)
                print(f"[Snakemake] command: \n     {cmd}")

            if result.returncode != 0:
                error_msg = f"Snakemake workflow failed.\nSee logs for {snakefile_path.parent}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": error_msg,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Workflow completed successfully", flush=True)

            return {
                "success": True,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Workflow completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to run Snakemake: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "snakemake_logfile": snakemake_logfile,
            }

    def _validate_single_job_dry_run(
        self,
        snakefile_path: Path,
        analysis: "TRITONSWMM_analysis",
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
    ) -> dict:
        """
        Perform dry-run validation for 1_job_many_srun_tasks mode.

        Computes expected resource allocations and validates the workflow DAG
        using the same CLI arguments that will be used in the SBATCH script.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        analysis : TRITONSWMM_analysis
            The analysis object (regular or master sensitivity analysis)
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for the CPU budget
            calculation. Must match the value passed to _generate_single_job_submission_script.

        Returns
        -------
        dict
            Status dictionary with 'success' and 'mode' keys
        """
        # Compute expected resources to match SBATCH script (--cores $TOTAL_CPUS)
        hpc_cpus_per_node = getattr(analysis.cfg_analysis, "hpc_cpus_per_node", None)
        hpc_total_nodes = (
            override_hpc_total_nodes
            if override_hpc_total_nodes is not None
            else getattr(analysis.cfg_analysis, "hpc_total_nodes", None)
        )
        if not isinstance(hpc_cpus_per_node, int) or not isinstance(hpc_total_nodes, int):
            if verbose:
                print(
                    "[Snakemake] Skipping single-job dry-run validation: "
                    "hpc_cpus_per_node or hpc_total_nodes missing in config",
                    flush=True,
                )
            return {
                "success": True,
                "mode": "single_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Dry run skipped (missing hpc_cpus_per_node or hpc_total_nodes)",
            }

        expected_total_cpus = hpc_cpus_per_node * hpc_total_nodes

        # Temporarily align local dry-run cores with expected SLURM allocation.
        # This keeps run_snakemake_local config-driven while validating the DAG
        # under expected single-job CPU availability.
        original_local_cores = analysis.cfg_analysis.local_cpu_cores_for_workflow
        analysis.cfg_analysis.local_cpu_cores_for_workflow = expected_total_cpus
        try:
            dry_run_result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        finally:
            analysis.cfg_analysis.local_cpu_cores_for_workflow = original_local_cores

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        # Override mode to indicate intended execution context
        dry_run_result["mode"] = "single_job"
        return dry_run_result

    # TODO - since we are unlikely to run models as detached processes, this and all calls to it can probably be deleted
    def _run_snakemake_slurm_detached(
        self,
        snakefile_path: Path,
        verbose: bool = True,
        wait_for_completion: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """
        Run Snakemake workflow on SLURM HPC system.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages
        wait_for_completion : bool
            If True, block and wait for workflow completion. If False (default),
            return immediately after submission (non-blocking).
        dry_run : bool
            If True, perform a Snakemake dry run only

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool - Did submission succeed?
            - mode: str - "slurm"
            - snakefile_path: Path - Path to Snakefile
            - job_id: str | None - Always None (job ID not extracted)
            - message: str - Status message
            - process: Popen - Process object
            - wait_for_completion: bool - Whether we waited
            - completed: bool - True only if wait_for_completion=True and job finished
            - completion_status: str | None - "success"/"failed" (only if waited)
            - snakemake_logfile: Path - Path to snakemake output log
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow on SLURM with Snakemake",
                    flush=True,
                )
                if dry_run:
                    print(
                        "[Snakemake] DRY RUN",
                        flush=True,
                    )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            if verbose:
                print(f"[Snakemake] Using config from: {config_dir}", flush=True)

            # Create log directory and file for Snakemake output
            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile_name = "snakemake_master_dry_run.log" if dry_run else "snakemake_master.log"
            snakemake_logfile = logs_dir / logfile_name

            # Create SLURM efficiency report directory and set timestamped filename
            import TRITON_SWMM_toolkit.utils as ut

            efficiency_report_dir = logs_dir / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            if verbose:
                print(
                    f"[Snakemake] Snakemake output will be logged to: {snakemake_logfile}",
                    flush=True,
                )
                print(
                    f"[Snakemake] SLURM efficiency report will be written to: {efficiency_report_path}",
                    flush=True,
                )

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
                "--executor",
                "slurm",
                "--printshellcmds",
                "--slurm-efficiency-report",
                "--slurm-efficiency-report-path",
                str(efficiency_report_path),
            ]
            if dry_run:
                cmd_args.append("--dry-run")
            if verbose:
                cmd_args.append("--verbose")

            with open(snakemake_logfile, "w") as log_f:
                proc = subprocess.Popen(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            if not wait_for_completion:
                if verbose:
                    print(
                        f"[Snakemake] Workflow submitted to background (PID: {proc.pid})",
                        flush=True,
                    )
                    print(
                        f"[Snakemake] Monitor progress with: tail -f {snakemake_logfile}",
                        flush=True,
                    )
                return {
                    "success": True,
                    "mode": "slurm",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": "Workflow submitted to background",
                    "process": proc,
                    "wait_for_completion": False,
                    "completed": False,
                    "completion_status": None,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Waiting for workflow completion...", flush=True)
            proc.wait()
            success = proc.returncode == 0
            completion_status = "success" if success else "failed"

            if verbose:
                print(
                    f"[Snakemake] Workflow completed with status: {completion_status}",
                    flush=True,
                )
                print(
                    f"[Snakemake] Full output available in: {snakemake_logfile}",
                    flush=True,
                )

            return {
                "success": success,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": f"Workflow completed with status: {completion_status}",
                "process": proc,
                "wait_for_completion": True,
                "completed": True,
                "completion_status": completion_status,
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to submit workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "process": None,
                "wait_for_completion": wait_for_completion,
                "completed": False,
                "completion_status": None,
                "snakemake_logfile": None,
            }

    def _validate_batch_job_dry_run(
        self,
        snakefile_path: Path,
        verbose: bool = True,
    ) -> dict:
        """
        Perform a dry-run validation for batch_job mode using the SLURM profile.

        This validates the Snakemake DAG/resources before submitting the
        orchestration SBATCH job.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Dry-run status dictionary
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running batch_job dry-run validation",
                    flush=True,
                )

            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            snakemake_logfile = logs_dir / "snakemake_master_dry_run.log"

            # Create SLURM efficiency report directory for dry run validation
            import TRITON_SWMM_toolkit.utils as ut

            efficiency_report_dir = logs_dir / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_dry_run_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
                "--executor",
                "slurm",
                "--printshellcmds",
                "--slurm-efficiency-report",
                "--slurm-efficiency-report-path",
                str(efficiency_report_path),
                "--dry-run",
            ]
            if verbose:
                cmd_args.append("--verbose")

            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if result.returncode != 0:
                error_msg = f"Snakemake batch_job dry run failed. See logs for {snakefile_path.parent}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "batch_job",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": error_msg,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Batch-job dry run completed successfully", flush=True)

            return {
                "success": True,
                "mode": "batch_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Batch-job dry run completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to run batch-job dry run: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "batch_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "snakemake_logfile": None,
            }

    def _wait_for_slurm_job_completion(
        self,
        job_id: str,
        poll_interval: int = 1,
        timeout: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Wait for SLURM job to complete by polling job status.

        Uses squeue for active jobs and sacct for completed jobs.

        Parameters
        ----------
        job_id : str
            SLURM job ID to monitor
        poll_interval : int, default=1
            Seconds between status checks
        timeout : int | None, default=None
            Maximum seconds to wait (None = indefinite)
        verbose : bool, default=True
            Print status updates

        Returns
        -------
        dict
            Job completion info:
            - completed: bool - True if job finished successfully
            - state: str - SLURM job state (COMPLETED, FAILED, etc.)
            - exit_code: int | None - Job exit code
            - message: str - Human-readable status
        """
        import time

        start_time = time.time()
        last_state = None

        if verbose:
            print(f"[Snakemake] Waiting for SLURM job {job_id} to complete...", flush=True)

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                msg = f"Job {job_id} timed out after {timeout}s"
                if verbose:
                    print(f"[Snakemake] ERROR: {msg}", flush=True)
                return {
                    "completed": False,
                    "state": "TIMEOUT",
                    "exit_code": None,
                    "message": msg,
                }

            # Query squeue for running/pending jobs
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip()

                # Print status update if changed
                if verbose and state != last_state:
                    elapsed = int(time.time() - start_time)
                    print(
                        f"[Snakemake] [{elapsed}s] Job {job_id}: {state}",
                        flush=True,
                    )
                    last_state = state

                if state in ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING"]:
                    time.sleep(poll_interval)
                    continue

            # Job not in squeue - check sacct for completion
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State,ExitCode"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split()
                state = parts[0]
                exit_code_str = parts[1] if len(parts) > 1 else "0:0"
                exit_code = int(exit_code_str.split(":")[0])

                completed = state == "COMPLETED" and exit_code == 0

                if verbose:
                    elapsed = int(time.time() - start_time)
                    status = "✓" if completed else "✗"
                    print(
                        f"[Snakemake] [{elapsed}s] Job {job_id}: {state} {status}",
                        flush=True,
                    )

                return {
                    "completed": completed,
                    "state": state,
                    "exit_code": exit_code,
                    "message": f"Job {job_id} {state} (exit {exit_code})",
                }

            # Job not found yet - might be starting up
            time.sleep(poll_interval)

    def _submit_single_job_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
    ) -> dict:
        """
        Submit workflow as a single SLURM batch job.

        This method generates a batch script that submits a single SLURM job
        which runs Snakemake inside the allocation using the single_job profile.
        Each simulation is then launched via srun within that allocation.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        wait_for_completion : bool, default=False
            If True, wait for job completion
        verbose : bool, default=True
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for this submission
            without mutating the config object. Only valid for 1_job_many_srun_tasks mode.

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str ("single_job")
            - job_id: str | None
            - script_path: Path
            - message: str
            - completed: bool (only if wait_for_completion=True)
            - state: str (only if wait_for_completion=True)
            - exit_code: int | None (only if wait_for_completion=True)
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Preparing single-job workflow submission",
                    flush=True,
                )

            # Check for stale lock before consuming a SLURM allocation
            self._check_and_clear_snakemake_lock(snakefile_path, dry_run=False, verbose=verbose)

            # Generate single_job profile
            config = self.generate_snakemake_config(mode="single_job")
            config_dir = self.write_snakemake_config(config, mode="single_job")

            # Generate submission script
            script_path = self._generate_single_job_submission_script(
                snakefile_path, config_dir, override_hpc_total_nodes=override_hpc_total_nodes
            )

            if verbose:
                print(
                    f"[Snakemake] Generated submission script: {script_path}",
                    flush=True,
                )

            # Submit with sbatch
            if verbose:
                print(f"[Snakemake] Submitting with sbatch: {script_path}", flush=True)

            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            # Parse job ID from sbatch output
            job_id = None
            if result.returncode == 0 and result.stdout:
                # sbatch output typically: "Submitted batch job 12345"
                parts = result.stdout.strip().split()
                if len(parts) >= 4 and parts[0] == "Submitted":
                    job_id = parts[-1]

            if result.returncode != 0:
                error_msg = f"sbatch submission failed: {result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "single_job",
                    "job_id": None,
                    "script_path": script_path,
                    "message": error_msg,
                }

            if verbose:
                print(
                    f"[Snakemake] Single-job workflow submitted successfully (Job ID: {job_id})",
                    flush=True,
                )

            # Base result
            result_dict = {
                "success": True,
                "mode": "single_job",
                "job_id": job_id,
                "script_path": script_path,
                "message": f"Single-job workflow submitted (Job ID: {job_id})",
            }

            # Wait for completion if requested
            if wait_for_completion:
                if job_id:
                    completion_info = self._wait_for_slurm_job_completion(
                        job_id=job_id,
                        timeout=None,
                        verbose=verbose,
                    )

                    result_dict.update(completion_info)
                    result_dict["success"] = completion_info["completed"]
                else:
                    if verbose:
                        print(
                            "[Snakemake] ERROR: Failed to parse job ID for wait",
                            flush=True,
                        )
                    result_dict["success"] = False
                    result_dict["completed"] = False
                    result_dict["message"] = "Failed to parse job ID"

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit single-job workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "single_job",
                "job_id": None,
                "script_path": None,
                "message": error_msg,
            }

    def _deprecated_submit_batch_job_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        DEPRECATED: Submit Snakemake workflow as an SLURM sbatch orchestration job.

        **WARNING**: This method runs Snakemake inside an sbatch job, which causes
        orphaned worker jobs when the orchestrator is canceled. This approach is
        deprecated in favor of tmux-based orchestration.

        This method is kept for backward compatibility but should not be used.
        The batch_job mode now uses _submit_tmux_workflow() instead.

        Parameters
        ----------
        snakefile_path : Path
            Path to the generated Snakefile
        wait_for_completion : bool, default=False
            If True, block until orchestration job completes
        verbose : bool, default=True
            Print progress messages

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str ("batch_job")
            - job_id: str | None
            - script_path: Path | None
            - message: str
            - completed/state/exit_code when wait_for_completion=True
        """
        import warnings

        warnings.warn(
            "The sbatch orchestrator approach is deprecated due to orphaned job issues. "
            "This method should not be called directly. batch_job mode now uses tmux orchestration.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            if verbose:
                print(
                    "[Snakemake] Preparing batch_job orchestration submission",
                    flush=True,
                )

            # Build and write slurm profile used by the orchestration job
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            # Long-duration walltime for orchestration job
            job_time = self.cfg_analysis.hpc_total_job_duration_min
            assert isinstance(job_time, int), "hpc_total_job_duration_min required for multi_sim_run_method='batch_job'"

            hours = job_time // 60
            minutes = job_time % 60
            estimated_time = f"{hours:02d}:{minutes:02d}:00"

            # Lightweight orchestration resources (single-core process)
            mem_mb = self.cfg_analysis.mem_gb_per_cpu * 1000
            orchestration_partition = (
                self.cfg_analysis.hpc_setup_and_analysis_processing_partition
                or self.cfg_analysis.hpc_ensemble_partition
            )

            if orchestration_partition is None:
                raise ValueError(
                    "Either hpc_setup_and_analysis_processing_partition or "
                    "hpc_ensemble_partition must be set for batch_job orchestration"
                )

            # Logs for sbatch script stdout/stderr
            import TRITON_SWMM_toolkit.utils as ut

            batch_log_path = self.analysis.analysis_paths.analysis_log_directory / "_slurm_logs"
            batch_log_path.mkdir(exist_ok=True, parents=True)

            additional_sbatch_args = ""
            if self.cfg_analysis.additional_SBATCH_params:
                additional_sbatch_args = "#SBATCH "
                additional_sbatch_args += "\n#SBATCH ".join(self.cfg_analysis.additional_SBATCH_params)

            modules = self.analysis._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
            module_load_cmd = ""
            if modules:
                module_load_cmd = f"module load {modules}"

            # Conda initialization for non-interactive SLURM shell
            conda_init_cmd = """
# Initialize conda for non-interactive shell
if [ -n "${CONDA_EXE}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization. CONDA_EXE and CONDA_PREFIX are both unset."
    exit 1
fi

conda activate triton_swmm_toolkit

# Ensure conda libs are discoverable (important on some HPC systems)
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# Diagnostics: confirm activation and Snakemake availability
echo "=========================================="
echo "DIAGNOSTICS: Conda activation + Snakemake"
echo "=========================================="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<not set>}"
echo "Python (PATH): $(which python)"
echo "PATH (head):"
echo "${PATH}" | tr ':' '\n' | head -n 10 | sed 's/^/  /'
echo "=========================================="
"""

            account_directive = ""
            if self.cfg_analysis.hpc_account:
                account_directive = f"#SBATCH --account={self.cfg_analysis.hpc_account}\n"

            # Create SLURM efficiency report directory and set timestamped filename
            efficiency_report_dir = self.analysis.analysis_paths.analysis_log_directory / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            # The orchestration job runs snakemake; snakemake then submits worker jobs via executor=slurm
            script_content = f"""#!/bin/bash
#SBATCH --job-name={self.cfg_analysis.analysis_id}_orchestrator
#SBATCH --partition={orchestration_partition}
{account_directive}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem_mb}
#SBATCH --time={estimated_time}
#SBATCH --output={str(batch_log_path)}/workflow_batch_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
#SBATCH --error={str(batch_log_path)}/workflow_batch_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
{additional_sbatch_args}


module purge
{module_load_cmd}

{conda_init_cmd}

${{CONDA_PREFIX}}/bin/python -V
${{CONDA_PREFIX}}/bin/python -m snakemake --version

# Capture Snakemake plugin stack versions and environment size for debugging.
mkdir -p {str(self.analysis_paths.analysis_log_directory)}
{{
    echo "captured: $(date -Iseconds)"
    echo "env_size_bytes: $(env | wc -c)"
    echo "path_length_chars: ${{#PATH}}"
    ${{CONDA_PREFIX}}/bin/python -m snakemake --version 2>/dev/null | sed 's/^/snakemake: /'
    ${{CONDA_PREFIX}}/bin/pip show \\
        snakemake-executor-plugin-slurm \\
        snakemake-executor-plugin-slurm-jobstep \\
        snakemake-interface-executor-plugins \\
        snakemake-interface-common \\
        2>/dev/null | grep -E "^(Name|Version):"
    ${{CONDA_PREFIX}}/bin/python --version 2>&1 | sed 's/^/python: /'
}} > {str(self.analysis_paths.analysis_log_directory)}/snakemake_versions.txt

# Trim PATH before launching Snakemake to prevent ARG_MAX overflow in scontrol calls.
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")
env PATH="${{CONDA_PREFIX}}/bin:${{SLURM_BIN}}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \\
    LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib" \\
    ${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} \\
    --snakefile {snakefile_path} \\
    --executor slurm \\
    --printshellcmds \\
    --slurm-efficiency-report \\
    --slurm-efficiency-report-path {efficiency_report_path}
"""

            script_path = self.analysis_paths.analysis_dir / "run_workflow_batch_job.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            if verbose:
                print(
                    f"[Snakemake] Generated batch orchestration script: {script_path}",
                    flush=True,
                )
                print(f"[Snakemake] Submitting with sbatch: {script_path}", flush=True)

            submit_result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            # Parse job id from: "Submitted batch job 12345"
            job_id = None
            if submit_result.returncode == 0 and submit_result.stdout:
                parts = submit_result.stdout.strip().split()
                if len(parts) >= 4 and parts[0] == "Submitted":
                    job_id = parts[-1]

                    # Persist job ID to analysis log (batch_job mode - deprecated)
                    import datetime

                    # Note: batch_job mode is deprecated; use tmux mode instead
                    self.analysis.log.workflow_submission_time.set(datetime.datetime.now().isoformat())
                    self.analysis.log.workflow_submission_mode.set("batch_job")

            if submit_result.returncode != 0:
                error_msg = f"sbatch submission failed: {submit_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "batch_job",
                    "job_id": None,
                    "script_path": script_path,
                    "message": error_msg,
                }

            if verbose:
                print(
                    f"[Snakemake] Batch orchestration job submitted successfully (Job ID: {job_id})",
                    flush=True,
                )

            result_dict = {
                "success": True,
                "mode": "batch_job",
                "job_id": job_id,
                "script_path": script_path,
                "message": f"Batch orchestration workflow submitted (Job ID: {job_id})",
            }

            if wait_for_completion:
                if job_id:
                    completion_info = self._wait_for_slurm_job_completion(
                        job_id=job_id,
                        timeout=None,
                        verbose=verbose,
                    )
                    result_dict.update(completion_info)
                    result_dict["success"] = completion_info["completed"]
                else:
                    if verbose:
                        print(
                            "[Snakemake] ERROR: Failed to parse job ID for wait",
                            flush=True,
                        )
                    result_dict["success"] = False
                    result_dict["completed"] = False
                    result_dict["message"] = "Failed to parse job ID"

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit batch-job workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "batch_job",
                "job_id": None,
                "script_path": None,
                "message": error_msg,
            }

    def _submit_tmux_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Submit Snakemake workflow in detached tmux session.

        This approach runs Snakemake on the login node in a persistent tmux session,
        avoiding the orphaned jobs problem with sbatch orchestration. Snakemake's
        SIGINT handler properly cancels all worker jobs when the session receives SIGINT.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        wait_for_completion : bool
            If True, block until tmux session exits
        verbose : bool
            Print status messages

        Returns
        -------
        dict
            - success: bool
            - mode: str ("tmux")
            - session_name: str
            - snakemake_pid: int
            - message: str
        """
        try:
            # Build module load prefix for HPC systems
            module_load_prefix = self._get_module_load_prefix()

            # Check if tmux is available (with module load on HPC)
            tmux_check_cmd = f"{module_load_prefix}which tmux" if module_load_prefix else "which tmux"
            tmux_check = subprocess.run(
                ["bash", "-c", tmux_check_cmd],
                capture_output=True,
                text=True,
            )
            if tmux_check.returncode != 0:
                raise OSError(
                    "tmux is required for tmux workflow mode but not found in PATH. "
                    "Please install tmux or use multi_sim_run_method='local'."
                )

            # Check for stale lock before launching tmux session
            self._check_and_clear_snakemake_lock(snakefile_path, dry_run=False, verbose=verbose)

            # Generate unique session name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"triton_swmm_{self.cfg_analysis.analysis_id}_{timestamp}"

            # Check if session already exists (with module load on HPC)
            has_session_cmd = f"{module_load_prefix}tmux has-session -t {session_name}"
            session_check = subprocess.run(
                ["bash", "-c", has_session_cmd],
                capture_output=True,
                text=True,
            )
            if session_check.returncode == 0:
                raise RuntimeError(
                    f"Tmux session '{session_name}' already exists. "
                    "Please check if another workflow is running or kill the session manually."
                )

            # Build Snakemake command with absolute paths
            config_dir = self.analysis_paths.analysis_dir / ".snakemake_profile" / "slurm"

            # Create SLURM efficiency report directory and set timestamped filename
            from TRITON_SWMM_toolkit import utils as ut

            efficiency_report_dir = self.analysis.analysis_paths.analysis_log_directory / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            # Build module load commands for inside tmux session (reuse the same modules)
            # This ensures the Snakemake process has access to required modules
            module_load_cmd = module_load_prefix.removesuffix(" && ") if module_load_prefix else ""

            # Build the full command that will run inside tmux
            # Write output to a timestamped log file for debugging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tmux_log = self.analysis_paths.analysis_log_directory / f"tmux_session_{timestamp}.log"

            # Create THE workflow script - this is what actually gets executed
            workflow_script = self.analysis_paths.analysis_dir / "run_workflow_tmux.sh"
            workflow_content = f"""#!/bin/bash
# TRITON-SWMM Tmux Workflow Script
# This is the ACTUAL script executed inside the tmux session
# Generated by TRITON-SWMM toolkit

{{
set -e  # Exit on error

echo "=== Tmux session started at $(date) ==="

# Load required modules (including tmux if needed)
{module_load_cmd}

echo "=== Modules loaded ==="

# Initialize conda
if [ -n "${{CONDA_EXE}}" ]; then
    eval "$(${{CONDA_EXE}} shell.bash hook)"
elif [ -n "${{CONDA_PREFIX}}" ] && [ -f "${{CONDA_PREFIX}}/../etc/profile.d/conda.sh" ]; then
    source "${{CONDA_PREFIX}}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization"
    exit 1
fi

echo "=== Conda initialized ==="

# Activate environment
conda activate triton_swmm_toolkit

echo "=== Environment activated ==="

# Ensure conda libs are discoverable
if [ -n "${{CONDA_PREFIX}}" ]; then
    export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}"
fi

# Diagnostics: confirm activation and Snakemake availability
echo "=========================================="
echo "DIAGNOSTICS: Conda activation + Snakemake"
echo "=========================================="
echo "CONDA_PREFIX: ${{CONDA_PREFIX:-<not set>}}"
echo "CONDA_DEFAULT_ENV: ${{CONDA_DEFAULT_ENV:-<not set>}}"
echo "Python (PATH): $(which python)"
echo "Python (conda): ${{CONDA_PREFIX}}/bin/python"
echo "PATH (head):"
echo "${{PATH}}" | tr ':' '\\n' | head -n 10 | sed 's/^/  /'
echo "=========================================="

${{CONDA_PREFIX}}/bin/python -V
${{CONDA_PREFIX}}/bin/python -m snakemake --version

# Capture Snakemake plugin stack versions and environment size for debugging.
# Written before the PATH trim so env_size_bytes reflects the pre-trim state.
mkdir -p {self.analysis_paths.analysis_log_directory}
{{
    echo "captured: $(date -Iseconds)"
    echo "env_size_bytes: $(env | wc -c)"
    echo "path_length_chars: ${{#PATH}}"
    ${{CONDA_PREFIX}}/bin/python -m snakemake --version 2>/dev/null | sed 's/^/snakemake: /'
    ${{CONDA_PREFIX}}/bin/pip show \\
        snakemake-executor-plugin-slurm \\
        snakemake-executor-plugin-slurm-jobstep \\
        snakemake-interface-executor-plugins \\
        snakemake-interface-common \\
        2>/dev/null | grep -E "^(Name|Version):"
    ${{CONDA_PREFIX}}/bin/python --version 2>&1 | sed 's/^/python: /'
}} > {self.analysis_paths.analysis_log_directory}/snakemake_versions.txt

# Trim PATH and LD_LIBRARY_PATH before launching Snakemake.
# After module load and conda activate, PATH can exceed Linux ARG_MAX limits.
# The snakemake-executor-plugin-slurm calls scontrol inheriting the full
# environment; if the env is too large, it crashes with OSError: [Errno 7]
# Argument list too long. We scope the trim to just the Snakemake process
# using `env` so the surrounding tmux script is unaffected.
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")

# Run Snakemake
cd {self.analysis_paths.analysis_dir}
echo "=== Starting Snakemake ==="
set +e
env PATH="${{CONDA_PREFIX}}/bin:${{SLURM_BIN}}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \\
    LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib" \\
    ${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} \\
    --snakefile {snakefile_path} \\
    --executor slurm \\
    --printshellcmds \\
    --slurm-efficiency-report \\
    --slurm-efficiency-report-path {efficiency_report_path}
snakemake_status=$?
echo "=== Snakemake completed at $(date) (exit: $snakemake_status) ==="
tmux kill-session -t {session_name}
exit $snakemake_status
}} >> {tmux_log} 2>&1
"""
            workflow_script.write_text(workflow_content)
            workflow_script.chmod(0o755)

            # Create detached tmux session (with module load on HPC)
            new_session_cmd = f"{module_load_prefix}tmux new-session -d -s {session_name} bash"

            tmux_result = subprocess.run(
                ["bash", "-c", new_session_cmd],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            if tmux_result.returncode != 0:
                error_msg = f"Failed to create tmux session: {tmux_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "tmux",
                    "session_name": None,
                    "snakemake_pid": None,
                    "message": error_msg,
                }

            # Execute THE workflow script in the tmux session
            exec_cmd = f"bash {workflow_script}"
            send_keys_cmd = f"{module_load_prefix}tmux send-keys -t {session_name} {shlex.quote(exec_cmd)} Enter"

            send_cmd_result = subprocess.run(
                ["bash", "-c", send_keys_cmd],
                capture_output=True,
                text=True,
            )

            if send_cmd_result.returncode != 0:
                # Clean up the session (with module load on HPC)
                kill_session_cmd = f"{module_load_prefix}tmux kill-session -t {session_name}"
                subprocess.run(["bash", "-c", kill_session_cmd], capture_output=True)
                error_msg = f"Failed to send command to tmux session: {send_cmd_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "tmux",
                    "session_name": None,
                    "snakemake_pid": None,
                    "message": error_msg,
                }

            # Wait a moment for process to start
            time.sleep(2)

            # Extract Snakemake PID from tmux session
            snakemake_pid = self._get_snakemake_pid_from_tmux(session_name)

            # Note: snakemake_pid may be None if Snakemake hasn't started yet

            # Capture the login node hostname for node-pinned reattach commands
            submission_node = socket.gethostname()

            # Persist session info to analysis log
            self.analysis.log.tmux_session_name.set(session_name)
            if snakemake_pid:
                self.analysis.log.snakemake_pid.set(snakemake_pid)
            self.analysis.log.workflow_submission_time.set(datetime.datetime.now().isoformat())
            self.analysis.log.workflow_submission_mode.set("tmux")
            self.analysis.log.workflow_submission_node.set(submission_node)

            # Determine the node to use in reattach commands:
            # prefer explicit config value, fall back to auto-detected hostname
            reattach_node = self.cfg_analysis.hpc_login_node or submission_node

            # Build node-pinned reattach commands (required when cluster uses
            # round-robin login load balancers, e.g. login.hpc.virginia.edu)
            module_load_cmd = self._get_module_load_prefix()
            if module_load_cmd:
                # On HPC: include module load tmux so a fresh SSH session can attach
                reattach_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux attach -t {session_name}'"
                kill_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux kill-session -t {session_name}'"
                list_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux list-sessions'"
            else:
                reattach_cmd = f"tmux attach -t {session_name}"
                kill_cmd = f"tmux kill-session -t {session_name}"
                list_cmd = "tmux list-sessions"

            if verbose:
                print(
                    "[Snakemake] Tmux workflow submitted successfully",
                    flush=True,
                )
                print(f"[Snakemake] Session name: {session_name}", flush=True)
                print(f"[Snakemake] Submission node: {submission_node}", flush=True)
                if snakemake_pid:
                    print(f"[Snakemake] Snakemake PID: {snakemake_pid}", flush=True)
                print(f"[Snakemake] Log file: {tmux_log}", flush=True)
                print("", flush=True)
                print("[Snakemake] Useful commands:", flush=True)
                print(f"[Snakemake]   Monitor log:      tail -f {tmux_log}", flush=True)
                print(
                    f"[Snakemake]   Attach to session: {reattach_cmd}",
                    flush=True,
                )
                print("[Snakemake]   Detach from session: Ctrl+B, then D", flush=True)
                print(
                    f"[Snakemake]   Kill this session: {kill_cmd}",
                    flush=True,
                )
                print(
                    f"[Snakemake]   List all sessions: {list_cmd}",
                    flush=True,
                )

            result_dict = {
                "success": True,
                "mode": "tmux",
                "session_name": session_name,
                "snakemake_pid": snakemake_pid,
                "message": f"Tmux workflow submitted (session: {session_name})",
            }

            if wait_for_completion:
                if verbose:
                    print("[Snakemake] Waiting for workflow completion...", flush=True)
                completion_info = self._wait_for_tmux_session_completion(
                    session_name=session_name,
                    verbose=verbose,
                )
                result_dict.update(completion_info)
                result_dict["success"] = completion_info["completed"]

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit tmux workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "tmux",
                "session_name": None,
                "snakemake_pid": None,
                "message": error_msg,
            }

    def _get_module_load_prefix(self) -> str:
        """
        Build module load prefix for HPC tmux commands.

        Returns
        -------
        str
            Shell command prefix to load modules, or empty string if not on HPC
        """
        modules_str = self.system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc

        # If we're in SLURM or using batch_job mode, always try to load tmux
        # Even if no other modules are specified, tmux might not be in default PATH
        if self.analysis.in_slurm or self.cfg_analysis.multi_sim_run_method == "batch_job":
            if modules_str:
                # modules_str is a space-separated string, e.g., "gcc/11.2.0 openmpi/4.1.1"
                return f"module purge && module load tmux {modules_str} && "
            else:
                # No other modules, but still load tmux on HPC
                return "module load tmux && "

        return ""

    def _get_snakemake_pid_from_tmux(self, session_name: str) -> int | None:
        """
        Extract Snakemake process ID from tmux session.

        Parameters
        ----------
        session_name : str
            Name of the tmux session

        Returns
        -------
        int | None
            Snakemake PID if found, None otherwise
        """
        module_load_prefix = self._get_module_load_prefix()
        try:
            # Get the shell PID in the tmux pane (with module load on HPC)
            list_panes_cmd = f"{module_load_prefix}tmux list-panes -t {session_name} -F '#{{pane_pid}}'"
            pane_pid_result = subprocess.run(
                ["bash", "-c", list_panes_cmd],
                capture_output=True,
                text=True,
            )

            if pane_pid_result.returncode != 0:
                return None

            shell_pid = int(pane_pid_result.stdout.strip())

            # Recursively search for Snakemake process in descendant tree
            # ps --ppid only shows direct children, so we need to recurse manually
            def find_snakemake_in_descendants(parent_pid: int) -> int | None:
                # Get direct children of this parent
                children_result = subprocess.run(
                    ["ps", "-o", "pid", "--ppid", str(parent_pid), "--no-headers"],
                    capture_output=True,
                    text=True,
                )

                if children_result.returncode != 0:
                    return None

                child_pids = [
                    int(pid.strip()) for pid in children_result.stdout.strip().split("\n") if pid.strip().isdigit()
                ]

                # Check each child process
                for child_pid in child_pids:
                    # Get the command line for this child
                    cmd_result = subprocess.run(
                        ["ps", "-o", "cmd", "-p", str(child_pid), "--no-headers"],
                        capture_output=True,
                        text=True,
                    )

                    if cmd_result.returncode == 0:
                        cmd = cmd_result.stdout.strip()
                        # Check if this is the Snakemake process
                        if "snakemake" in cmd and "python" in cmd:
                            return child_pid

                    # Recurse into this child's descendants
                    found_pid = find_snakemake_in_descendants(child_pid)
                    if found_pid:
                        return found_pid

                return None

            return find_snakemake_in_descendants(shell_pid)

        except Exception:
            return None

    def _wait_for_tmux_session_completion(
        self,
        session_name: str,
        verbose: bool = True,
    ) -> dict:
        """
        Wait for tmux session to exit.

        Parameters
        ----------
        session_name : str
            Name of the tmux session
        verbose : bool
            Print status messages

        Returns
        -------
        dict
            - completed: bool
            - message: str
        """
        module_load_prefix = self._get_module_load_prefix()
        try:
            while True:
                # Check if session still exists (with module load on HPC)
                has_session_cmd = f"{module_load_prefix}tmux has-session -t {session_name}"
                check_result = subprocess.run(
                    ["bash", "-c", has_session_cmd],
                    capture_output=True,
                    text=True,
                )

                if check_result.returncode != 0:
                    # Session no longer exists - workflow completed
                    if verbose:
                        print(
                            "[Snakemake] Tmux session exited - workflow complete",
                            flush=True,
                        )
                    return {
                        "completed": True,
                        "message": "Workflow completed successfully",
                    }

                # Session still exists, wait and check again
                time.sleep(5)

        except KeyboardInterrupt:
            if verbose:
                print("\n[Snakemake] Wait interrupted by user", flush=True)
            return {
                "completed": False,
                "message": "Wait interrupted by user",
            }
        except Exception as e:
            return {
                "completed": False,
                "message": f"Error while waiting: {str(e)}",
            }

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
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        wait_for_completion: bool = False,
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_config_path: "Path | None" = None,
    ) -> dict:
        """
        Submit workflow using Snakemake.

        Automatically detects execution context (local vs. HPC) and submits accordingly.
        If multi_sim_run_method is "1_job_many_srun_tasks", submits as a single SLURM
        job with multiple srun tasks inside.

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
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_outputs_if_already_created : bool
            If True, overwrite existing processed outputs
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
            If provided, overrides cfg_analysis.hpc_total_nodes for SLURM script generation
            without mutating the config object. Only valid when multi_sim_run_method is
            "1_job_many_srun_tasks"; raises ConfigurationError otherwise.

        Returns
        -------
        dict
            Status dictionary with keys defined by run_snakemake_local or run_snakemake_slurm
        """
        self._report_config_path = report_config_path

        # Check if we should use 1-job mode based on config
        multi_sim_method = self.cfg_analysis.multi_sim_run_method

        if override_hpc_total_nodes is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="override_hpc_total_nodes",
                message=(
                    f"override_hpc_total_nodes is only valid when multi_sim_run_method='1_job_many_srun_tasks', "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if multi_sim_method == "1_job_many_srun_tasks":
            # Always submit a batch job for 1-job mode
            if verbose:
                print(
                    "[Snakemake] Using 1-job many-srun-tasks mode",
                    flush=True,
                )

            # Generate Snakefile content
            snakefile_content = self.generate_snakefile_content(
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                report_config_path=report_config_path,
            )

            # Write Snakefile to disk
            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            snakefile_path.write_text(snakefile_content)

            if verbose:
                print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

            # Always perform a dry run validation first
            dry_run_result = self._validate_single_job_dry_run(
                snakefile_path=snakefile_path,
                analysis=self.analysis,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            if dry_run:
                # Override mode to indicate intended execution context
                dry_run_result["mode"] = "single_job"
                self.analysis._refresh_log()
                return dry_run_result

            result = self._submit_single_job_workflow(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            self.analysis._refresh_log()
            return result

        if multi_sim_method == "batch_job":
            if verbose:
                print(
                    "[Snakemake] Using batch_job mode (tmux orchestration)",
                    flush=True,
                )

            snakefile_content = self.generate_snakefile_content(
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                report_config_path=report_config_path,
            )

            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            snakefile_path.write_text(snakefile_content)

            if verbose:
                print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

            # Use batch_job dry run validation (same SLURM profile)
            dry_run_result = self._validate_batch_job_dry_run(
                snakefile_path=snakefile_path,
                verbose=verbose,
            )

            if not dry_run_result.get("success"):
                raise RuntimeError("Dry run failed; workflow submission aborted.")

            if dry_run:
                self.analysis._refresh_log()
                return dry_run_result

            result = self._submit_tmux_workflow(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
            )

            self.analysis._refresh_log()
            return result

        # Standard workflow submission (existing logic)
        if mode == "auto":
            mode = "slurm" if self.analysis.in_slurm else "local"

        if verbose:
            print(f"[Snakemake] Submitting workflow in {mode} mode", flush=True)

        # Generate Snakefile content
        snakefile_content = self.generate_snakefile_content(
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            compression_level=compression_level,
            pickup_where_leftoff=pickup_where_leftoff,
            report_config_path=report_config_path,
        )

        # Write Snakefile to disk
        snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)

        if verbose:
            print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

        # Always perform a dry run first
        if mode == "local":
            dry_run_result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        else:  # slurm
            dry_run_result = self._run_snakemake_slurm_detached(
                snakefile_path=snakefile_path,
                wait_for_completion=True,
                verbose=verbose,
                dry_run=True,
            )

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        if dry_run:
            self.analysis._refresh_log()
            return dry_run_result

        # Submit workflow based on mode
        if mode == "local":
            result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
            )
        else:  # slurm
            result = self._run_snakemake_slurm_detached(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
            )

        self.analysis._refresh_log()
        return result


class SensitivityAnalysisWorkflowBuilder:
    """
    Builder class for generating and executing Snakemake workflows for sensitivity analysis.

    This class handles the unique requirements of sensitivity analysis workflows,
    which involve a hierarchical structure (master analysis → sub-analyses → simulations)
    with multiple consolidation steps. It composes SnakemakeWorkflowBuilder to reuse
    common workflow patterns while adding sensitivity-specific logic.

    Key Features:
    - Generates flattened master Snakefile with all simulation rules
    - Handles dynamic resource allocation per sub-analysis
    - Supports multiple consolidation levels (per-subanalysis + master)
    - Delegates workflow submission to base SnakemakeWorkflowBuilder

    Parameters
    ----------
    sensitivity_analysis : TRITONSWMM_sensitivity_analysis
        The parent sensitivity analysis object containing configuration and sub-analyses
    """

    def __init__(self, sensitivity_analysis: "TRITONSWMM_sensitivity_analysis"):
        """
        Initialize the sensitivity analysis workflow builder.

        Parameters
        ----------
        sensitivity_analysis : TRITONSWMM_sensitivity_analysis
            The parent sensitivity analysis object containing configuration and sub-analyses
        """
        self.sensitivity_analysis = sensitivity_analysis
        self.master_analysis = sensitivity_analysis.master_analysis
        self.system = self.master_analysis._system
        self.analysis_paths = self.master_analysis.analysis_paths
        self.python_executable = self.master_analysis._python_executable

        # Compose base workflow builder for common patterns
        self._base_builder = SnakemakeWorkflowBuilder(self.master_analysis)

    def generate_master_snakefile_content(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_outputs_if_already_created: bool = False,
        compression_level: int = 5,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        clear_raw_outputs: bool = True,
        pickup_where_leftoff: bool = True,
        report_config_path: "Path | None" = None,
    ) -> str:
        """
        For sensitivity analyses.

        Generate flattened master Snakefile with individual simulation rules.

        This method generates a single Snakefile with all simulation rules
        flattened directly into it (no nested Snakemake calls). Each simulation
        gets its own rule with exact resource requirements from its sub-analysis config.

        This avoids resource contention issues where sub-analyses with different
        CPU/GPU requirements would fail due to incorrect resource allocation.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        overwrite_outputs_if_already_created : bool
            If True, overwrite existing consolidated outputs
        compression_level : int
            Compression level for output files (0-9)
        process_system_level_inputs : bool
            If True, process system-level inputs in master setup rule
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in master setup rule
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
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Master Snakefile content
        """
        from TRITON_SWMM_toolkit.config.loaders import yaml_to_model
        from TRITON_SWMM_toolkit.config.report import DEFAULT_REPORT_CONFIG, report_config
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        # Emit report templates into the master analysis_dir/report/ so the
        # snakemake --report engine can resolve caption= paths.
        self._base_builder._emit_report_artifacts(
            self.master_analysis.analysis_paths.analysis_dir
        )

        # Get absolute path to conda environment file using helper
        conda_env_path = self._base_builder._get_conda_env_path()
        master_config_args = self._base_builder._get_config_args(
            analysis_config_yaml=self.master_analysis.analysis_config_yaml
        )

        # Resolve report_config to get the sensitivity benchmarking independent_vars
        # so the master Snakefile can wildcard the plot rule per independent_var.
        if report_config_path is not None:
            _report_cfg = yaml_to_model(report_config_path, report_config)
        else:
            _report_cfg = DEFAULT_REPORT_CONFIG
        _independent_vars: list[str] = (
            list(_report_cfg.sensitivity.independent_vars)
            if _report_cfg.sensitivity is not None
            else []
        )
        _group_by_var: str | None = (
            _report_cfg.sensitivity.group_by_var
            if _report_cfg.sensitivity is not None
            else None
        )

        # Determine the single enabled model type for sensitivity analysis
        # Sensitivity analysis doesn't support multi-model (would explode parameter space)
        enabled_models = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")

        if len(enabled_models) == 0:
            raise ValueError("No model types enabled in system configuration")
        if len(enabled_models) > 1:
            raise ValueError(
                f"Sensitivity analysis does not support multi-model execution. "
                f"Enabled models: {enabled_models}. Please enable only one model type."
            )

        model_type = enabled_models[0]

        log_dir_str = str(self.master_analysis.analysis_paths.analysis_log_directory)
        master_analysis_id = str(self.master_analysis.cfg_analysis.analysis_id)
        n_sub_analyses = len(self.sensitivity_analysis.sub_analyses)
        # Total scenarios across all sub-analyses (best-effort; matches per-sub-analysis n_sims sum)
        try:
            total_n_sims = sum(
                len(sub.df_sims) for sub in self.sensitivity_analysis.sub_analyses.values()
            )
        except Exception:
            total_n_sims = n_sub_analyses

        # Compute paired (sa_id, event_id) lists for per-sa per-event plot rules.
        # Used by `_build_plot_rule_block_per_sim_per_sa` and the master `rule all`
        # via `expand(..., zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)`.
        # Event IDs derived via the canonical slug helper used elsewhere in workflow.py.
        sa_event_pairs_sa: list[str] = []
        sa_event_pairs_evt: list[str] = []
        try:
            for sa_id_pair, sub_pair in self.sensitivity_analysis.sub_analyses.items():
                for event_iloc in sub_pair.df_sims.index:
                    ev = sub_pair._retrieve_weather_indexer_using_integer_index(event_iloc)
                    sa_event_pairs_sa.append(str(sa_id_pair))
                    sa_event_pairs_evt.append(compute_event_id_slug(ev))
        except Exception:
            # Best-effort: if any sub-analysis can't materialize event ids, leave the
            # paired lists empty — per-sa per-event plot rules will simply not emit
            # any wildcarded outputs and the master report will skip Per-Simulation panels.
            sa_event_pairs_sa = []
            sa_event_pairs_evt = []

        # Start building the Snakefile
        snakefile_content = f'''# Auto-generated flattened master Snakefile for sensitivity analysis
# Each sub-analysis simulation phase gets its own rule with appropriate resources

import os
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = {master_analysis_id!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {total_n_sims}
config["is_sensitivity"] = True
config["n_sub_analyses"] = {n_sub_analyses}
config["independent_vars"] = {_independent_vars!r}
config["group_by_var"] = {_group_by_var!r}
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

# Paired (sa_id, event_id) lists for per-sa per-event plot rules.
# Used by `expand(..., zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)`
# in the master `rule all` and in the per-sa per-event plot rule definitions.
SA_EVENT_PAIRS_SA = {sa_event_pairs_sa!r}
SA_EVENT_PAIRS_EVT = {sa_event_pairs_evt!r}

report: "report/workflow_description.rst"

onstart:
    shell("mkdir -p _status {log_dir_str}/sims {log_dir_str}")

onsuccess:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {master_config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)

onerror:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {master_config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)


'''

        # Build the rule all with all dependencies
        consolidation_flags = []
        for sa_id in self.sensitivity_analysis.sub_analyses.keys():  # type: ignore
            consolidation_flags.append(
                f"_status/e_consolidate_sa-{sa_id}_complete.flag"  # type: ignore
            )

        rule_all_inputs = [f'"{flag}"' for flag in consolidation_flags]
        rule_all_inputs.append('"_status/f_consolidate_master_complete.flag"')
        # System-overview at master scope: the DEM and SWMM topology are shared
        # across sub-analyses, so a single system_overview.png in the master
        # report is the natural place to surface them. Per-analysis summary at
        # master scope renders one row per sub-analysis (Iteration 6 "show all
        # sub-analyses" scope). Per-sim plots wildcarded over (sa_id, event_id)
        # pairs (Iteration 7 Change 3b — "show all" panel parity per the user's
        # scope expansion: identical-looking panels across sub-analyses are a QC
        # signal; expected variation is also visible).
        rule_all_inputs.append('"plots/system_overview.png"')
        rule_all_inputs.append('"plots/per_analysis/summary_table.svg"')
        rule_all_inputs.append('"plots/appendix/scenario_status.html"')
        rule_all_inputs.append('"plots/errors_and_warnings/validation_report.html"')

        if sa_event_pairs_sa:
            rule_all_inputs.append(
                'expand("plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/peak_flood_depth.png", '
                'zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
            rule_all_inputs.append(
                'expand("plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/conduit_flow.png", '
                'zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
        if _independent_vars:
            rule_all_inputs.append(
                'expand("plots/sensitivity/benchmarking/{independent_var}_vs_total.svg", '
                f'independent_var={_independent_vars!r})'
            )

        snakefile_content += f'''rule all:
    input:
        {", ".join(rule_all_inputs)}

rule setup:
    output: "_status/a_setup_complete.flag"
    log: "{log_dir_str}/setup.log"
    conda: "{conda_env_path}"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=self.master_analysis.cfg_analysis.mem_gb_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            {master_config_args} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {
            "--compile-triton-swmm " if compile_TRITON_SWMM and self.system.cfg_system.toggle_tritonswmm_model else ""
        }\\
            {"--compile-triton-only " if compile_TRITON_SWMM and self.system.cfg_system.toggle_triton_model else ""}\\
            {"--compile-swmm " if compile_TRITON_SWMM and self.system.cfg_system.toggle_swmm_model else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

        # Generate simulation rules for each sub-analysis
        subanalysis_flags = []
        for sa_id, sub_analysis in self.sensitivity_analysis.sub_analyses.items():  # type: ignore
            # Extract resource requirements from sub-analysis config
            n_mpi = sub_analysis.cfg_analysis.n_mpi_procs or 1
            n_omp = sub_analysis.cfg_analysis.n_omp_threads or 1
            n_gpus = sub_analysis.cfg_analysis.n_gpus or 0
            n_nodes = sub_analysis.cfg_analysis.n_nodes or 1
            hpc_time = sub_analysis.cfg_analysis.hpc_time_min_per_sim or 30
            mem_per_cpu = sub_analysis.cfg_analysis.mem_gb_per_cpu or 2
            gpus_per_node_config = sub_analysis.cfg_analysis.hpc_gpus_per_node or 0
            cpus_per_sim = n_mpi * n_omp
            run_mode = sub_analysis.cfg_analysis.run_mode

            sub_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=sub_analysis.analysis_config_yaml
            )

            gpu_alloc_mode = self.system.cfg_system.preferred_slurm_option_for_allocating_gpus or "gpus"

            # Build resource blocks for this sub-analysis
            prep_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=mem_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )

            # CRITICAL: Snakemake's SLURM executor uses max(threads, tasks×cpus_per_task) for allocation
            # Always set threads = total CPUs to ensure correct SLURM --ntasks value
            snakemake_threads = cpus_per_sim

            gpu_hw_override = getattr(
                sub_analysis.cfg_analysis, "gpu_hardware_override", None
            )
            gpu_hw = gpu_hw_override or self.system.cfg_system.gpu_hardware
            sim_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_ensemble_partition,
                runtime_min=hpc_time,
                mem_mb=int(mem_per_cpu * n_mpi * n_omp * 1000),
                nodes=n_nodes,
                tasks=n_mpi,
                cpus_per_task=n_omp,
                gpus_total=n_gpus,
                gpus_per_node_config=gpus_per_node_config,
                gpu_hardware=gpu_hw,
                gpu_alloc_mode=gpu_alloc_mode,
                mpi=(run_mode in ["hybrid", "mpi"]),
            )

            process_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=120,
                mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=2,
            )

            # For each simulation in this sub-analysis
            sub_analysis_sim_flags = []
            for event_iloc in sub_analysis.df_sims.index:
                event_id = compute_event_id_slug(
                    sub_analysis._retrieve_weather_indexer_using_integer_index(
                        event_iloc
                    )
                )
                # Rule names must be valid Python identifiers (no `.`, `-`).
                # Flag paths keep the hyphen-delimited format for wildcard parsing.
                sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
                event_id_rule = event_id.replace(".", "_").replace("-", "_")
                # Phase 1: Scenario preparation (if enabled)
                if prepare_scenarios:
                    prep_rule_name = f"prepare_sa_{sa_id_rule}_evt_{event_id_rule}"
                    prep_outflag = f"_status/b_prepare_sa-{sa_id}_evt-{event_id}_complete.flag"

                    snakefile_content += f'''rule {prep_rule_name}:
    input: "_status/a_setup_complete.flag"
    output: "{prep_outflag}"
    log: "{log_dir_str}/sims/{prep_rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{prep_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.prepare_scenario_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            {"--overwrite-scenario-if-already-set-up " if overwrite_scenario_if_already_set_up else ""}\\
            {"--rerun-swmm-hydro " if rerun_swmm_hydro_if_outputs_exist else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

                # Phase 2: Simulation execution
                sim_rule_name = f"simulation_sa_{sa_id_rule}_evt_{event_id_rule}"
                sim_outflag = f"_status/c_run_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"
                sim_input = f'"{prep_outflag}"' if prepare_scenarios else '"_status/a_setup_complete.flag"'

                snakefile_content += f'''rule {sim_rule_name}:
    input: {sim_input}
    output: "{sim_outflag}"
    log: "{log_dir_str}/sims/{sim_rule_name}.log"
    conda: "{conda_env_path}"
    threads: {snakemake_threads}
    resources:
{sim_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.run_simulation_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            --model-type {model_type} \\
            {"--pickup-where-leftoff " if pickup_where_leftoff else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

                # Phase 3: Output processing (if enabled)
                if process_timeseries:
                    process_rule_name = f"process_sa_{sa_id_rule}_evt_{event_id_rule}"
                    process_outflag = f"_status/d_process_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"

                    snakefile_content += f'''rule {process_rule_name}:
    input: "{sim_outflag}"
    output: "{process_outflag}"
    log: "{log_dir_str}/sims/{process_rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{process_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.process_timeseries_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            --model-type {model_type} \\
            --which {which} \\
            {"--clear-raw-outputs " if clear_raw_outputs else ""}\\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """

'''
                    final_flag = process_outflag
                else:
                    final_flag = sim_outflag

                sub_analysis_sim_flags.append(final_flag)

            subanalysis_flag = f"_status/e_consolidate_sa-{sa_id}_complete.flag"
            subanalysis_flags.append(subanalysis_flag)

            # Consolidate outputs after all sims have been run. Sanitize for
            # use as a Snakemake rule identifier.
            prefix = self.sensitivity_analysis.sub_analyses_prefix  # type: ignore
            snakefile_content += f'''rule consolidate_{prefix}{sa_id_rule}:
    input: {", ".join([f'"{flag}"' for flag in sub_analysis_sim_flags])}
    output: "{subanalysis_flag}"
    log: "{log_dir_str}/sims/consolidate_{prefix}{sa_id}.log"
    conda: "{conda_env_path}"
    resources:
{
                self._base_builder._build_resource_block(
                    partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                    runtime_min=30,
                    mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=1,
                )
            }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {sub_config_args} \\
            --which {which} \\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

        # Generate master consolidation rule
        snakefile_content += f'''rule master_consolidation:
    input: {", ".join([f'"{flag}"' for flag in subanalysis_flags])}
    output: "_status/f_consolidate_master_complete.flag"
    log: "{log_dir_str}/master_consolidation.log"
    conda: "{conda_env_path}"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {master_config_args} \\
            --consolidate-sensitivity-analysis-outputs \\
            --which {which} \\
            {"--overwrite-outputs-if-already-created " if overwrite_outputs_if_already_created else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """
'''

        # Append system_overview + per_analysis_summary rules at master scope (match rule_all above).
        # Master uses f_consolidate_master_complete.flag (NOT the multisim e_consolidate_complete flag).
        snakefile_content += self._base_builder._build_plot_rule_block_system_overview(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_per_analysis_summary(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_scenario_status_appendix(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_errors_and_warnings(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        # Per-sa per-event plot rules (Iteration 7 Change 3b — "show all" parity).
        # Only emit when sa_event_pairs are populated (best-effort guarded above).
        if sa_event_pairs_sa:
            snakefile_content += self._build_plot_rule_block_per_sim_per_sa()

        if _independent_vars:
            snakefile_content += self._build_plot_rule_block_sensitivity_benchmarking(
                _independent_vars
            )

        return snakefile_content

    def _build_plot_rule_block_sensitivity_benchmarking(
        self, independent_vars: list[str]
    ) -> str:
        """Generate the sensitivity benchmarking plot rule, wildcarded over independent_var.

        Charset validation for independent_var names is upstream, at Phase 1's
        ``validate_sensitivity_independent_vars()``; names reaching here are guaranteed
        Snakemake-safe.

        SWMM-only sub-analyses' .rpt paths are computed at emit time and baked
        into the closure as a list, so the collector can declare them as
        provenance even though they are conditional on enabled-model-types.
        """
        import os as _os
        conda_env_path = self._base_builder._get_conda_env_path()
        config_args = self._base_builder._get_config_args(
            analysis_config_yaml=self.master_analysis.analysis_config_yaml,
            include_report_config=True,
        )
        # Collect SWMM-only sub-analyses' .rpt paths (relative to master analysis_dir).
        # These are read by the renderer's parse_total_elapsed fallback for
        # SWMM-only sub-analyses; declaring them here makes the provenance
        # surface complete even though the read is conditional at runtime.
        master_root = str(self.master_analysis.analysis_paths.analysis_dir.resolve())
        swmm_only_rpt_rels: list[str] = []
        for sub in self.sensitivity_analysis.sub_analyses.values():
            sub_enabled = sub._get_enabled_model_types()
            if sub_enabled == ["swmm"] or sub_enabled == ("swmm",):
                for event_iloc in sub.df_sims.index:
                    try:
                        scen_paths = sub._retrieve_sim_run_processing_object(event_iloc).scen_paths
                        rpt = getattr(scen_paths, "swmm_full_rpt_file", None)
                        if rpt:
                            swmm_only_rpt_rels.append(
                                _os.path.relpath(str(Path(rpt).resolve()), master_root)
                            )
                    except Exception:
                        continue
        return f'''
INDEPENDENT_VARS = {independent_vars!r}

def _sensitivity_source_paths(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_sensitivity_source_paths,
    )
    return collect_sensitivity_source_paths(
        wildcards.independent_var,
        swmm_only_rpt_rel_paths={swmm_only_rpt_rels!r},
    )

rule plot_sensitivity_benchmarking:
    input:
        master = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/sensitivity/benchmarking/{{independent_var}}_vs_total.svg",
            caption="report/captions/sensitivity_benchmarking.rst",
            category="Key Results",
            subcategory="Benchmarking",
            labels={{"independent_var": "{{independent_var}}", "figure": "vs Total runtime"}},
        )
    params:
        source_paths = _sensitivity_source_paths,
        source_paths_rst = lambda w: _fmt_sources_rst(_sensitivity_source_paths(w)),
    log: "logs/plots/sensitivity_benchmarking_{{independent_var}}.log"
    conda: "{conda_env_path}"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli sensitivity_benchmarking \\
            {config_args} \\
            --independent-var {{wildcards.independent_var}} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

    def _build_plot_rule_block_per_sim_per_sa(self) -> str:
        """Generate per-sa per-event plot rules for the sensitivity master Snakefile.

        Realizes Iteration 7 Change 3b ("show all sub-analyses" panel parity per
        the user's scope expansion). For each (sa_id, event_id) pair in
        SA_EVENT_PAIRS, emits two plot rules: peak_flood_depth + conduit_flow.
        Both rules dispatch the per-sim renderer with `--sa-id {wildcards.sa_id}`
        + `--event-iloc {params.event_iloc}` so the renderer (via _cli.py
        sub-analysis routing) resolves the sub-analysis from the master and
        operates on per-sa-scoped scenario data.

        The `report(...)` annotation uses `category="Per Simulation Results"` +
        `subcategory="sa-{sa_id}"` so the master report's sidebar groups per-sim
        plots by sub-analysis. Identical-looking panels across sub-analyses are
        a QC signal (per the user's scope-expansion rationale); expected
        variation is also visible.

        ILOC_BY_EVENT_ID_BY_SA is emitted as a master-Snakefile global to map
        (sa_id, event_id) -> event_iloc for the renderer dispatch.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        conda_env_path = self._base_builder._get_conda_env_path()
        config_args = self._base_builder._get_config_args(
            analysis_config_yaml=self.master_analysis.analysis_config_yaml,
            include_report_config=True,
        )

        # Build ILOC_BY_EVENT_ID_BY_SA mapping at emit time so the rule can
        # resolve event_iloc from (sa_id, event_id) wildcards.
        iloc_by_event_id_by_sa: dict[str, dict[str, int]] = {}
        for sa_id, sub in self.sensitivity_analysis.sub_analyses.items():
            iloc_by_event_id_by_sa[str(sa_id)] = {}
            for event_iloc in sub.df_sims.index:
                ev = sub._retrieve_weather_indexer_using_integer_index(event_iloc)
                event_id = compute_event_id_slug(ev)
                iloc_by_event_id_by_sa[str(sa_id)][event_id] = int(event_iloc)

        rainfall_datavar = self.master_analysis.cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
        storm_tide_datavar = self.master_analysis.cfg_analysis.weather_time_series_storm_tide_datavar

        return f'''
ILOC_BY_EVENT_ID_BY_SA = {iloc_by_event_id_by_sa!r}

def _per_sim_per_sa_flood_depth_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "peak_flood_depth",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        sa_id=wildcards.sa_id,
    )

def _per_sim_per_sa_conduit_flow_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "conduit_flow",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        sa_id=wildcards.sa_id,
    )

rule plot_per_sim_per_sa_peak_flood_depth:
    input:
        master = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/peak_flood_depth.png",
            caption="report/captions/per_sim_peak_flood_depth.rst",
            category="Per Simulation Results",
            labels={{"sa_id": "{{sa_id}}", "event_id": "{{event_id}}", "figure": "Peak flood depth"}},
        )
    params:
        source_paths = _per_sim_per_sa_flood_depth_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_per_sa_flood_depth_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID_BY_SA[w.sa_id][w.event_id],
    log: "logs/plots/per_sim_per_sa_peak_flood_depth_sa-{{sa_id}}_{{event_id}}.log"
    conda: "{conda_env_path}"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli per_sim_peak_flood_depth \\
            {config_args} \\
            --sa-id {{wildcards.sa_id}} \\
            --event-iloc {{params.event_iloc}} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """

rule plot_per_sim_per_sa_conduit_flow:
    input:
        master = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/conduit_flow.png",
            caption="report/captions/per_sim_conduit_flow.rst",
            category="Per Simulation Results",
            labels={{"sa_id": "{{sa_id}}", "event_id": "{{event_id}}", "figure": "Conduit flow"}},
        )
    params:
        source_paths = _per_sim_per_sa_conduit_flow_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_per_sa_conduit_flow_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID_BY_SA[w.sa_id][w.event_id],
    log: "logs/plots/per_sim_per_sa_conduit_flow_sa-{{sa_id}}_{{event_id}}.log"
    conda: "{conda_env_path}"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli per_sim_conduit_flow \\
            {config_args} \\
            --sa-id {{wildcards.sa_id}} \\
            --event-iloc {{params.event_iloc}} \\
            --output {{output}} \\
            > {{log}} 2>&1
        """
'''

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
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_config_path: "Path | None" = None,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        This orchestrates multiple sub-analysis workflows and a final master
        consolidation step that combines all sub-analysis outputs.
        If multi_sim_run_method is "1_job_many_srun_tasks", submits as a single SLURM
        job with multiple srun tasks inside.

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
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_outputs_if_already_created : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for SLURM script generation
            without mutating the config object. Only valid when multi_sim_run_method is
            "1_job_many_srun_tasks"; raises ConfigurationError otherwise.

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str
            - snakefile_path: Path
            - message: str
        """
        self._report_config_path = report_config_path
        self._base_builder._report_config_path = report_config_path

        # Check if we should use 1-job mode based on config
        multi_sim_method = self.master_analysis.cfg_analysis.multi_sim_run_method

        sim_resources = self.master_analysis._resource_manager._get_simulation_resource_requirements()
        n_gpus_per_sim = sim_resources["n_gpus"]
        if n_gpus_per_sim > 0 and not self.system.cfg_system.gpu_compilation_backend:
            raise ConfigurationError(
                field="gpu_compilation_backend",
                message=(
                    "Sensitivity analysis requests GPUs (n_gpus > 0) but system config "
                    "has gpu_compilation_backend unset. Set gpu_compilation_backend to "
                    "CUDA/HIP or set n_gpus: 0 in sub-analyses."
                ),
                config_path=self.system.system_config_yaml,
            )

        if override_hpc_total_nodes is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="override_hpc_total_nodes",
                message=(
                    f"override_hpc_total_nodes is only valid when multi_sim_run_method='1_job_many_srun_tasks', "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if multi_sim_method == "1_job_many_srun_tasks":
            # Always submit a batch job for 1-job mode
            if verbose:
                print(
                    "[Snakemake] Using 1-job many-srun-tasks mode for sensitivity analysis",
                    flush=True,
                )

            # Generate master Snakefile
            master_snakefile_content = self.generate_master_snakefile_content(
                which=which,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                compression_level=compression_level,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                clear_raw_outputs=clear_raw_outputs,
                pickup_where_leftoff=pickup_where_leftoff,
            )

            master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
            master_snakefile_path.write_text(master_snakefile_content)

            if verbose:
                print(
                    f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                    flush=True,
                )

            # Create required directories
            analysis_dir = self.master_analysis.analysis_paths.analysis_dir
            (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
            self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
            (self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)

            # Always perform a dry run validation first
            dry_run_result = self._base_builder._validate_single_job_dry_run(
                snakefile_path=master_snakefile_path,
                analysis=self.master_analysis,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            if dry_run:
                # Override mode to indicate intended execution context
                dry_run_result["mode"] = "single_job"
                self.sensitivity_analysis._update_master_analysis_log()
                return dry_run_result

            result = self._base_builder._submit_single_job_workflow(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            self.sensitivity_analysis._update_master_analysis_log()
            return result

        if multi_sim_method == "batch_job":
            if verbose:
                print(
                    "[Snakemake] Using batch_job orchestration mode for sensitivity analysis",
                    flush=True,
                )

            master_snakefile_content = self.generate_master_snakefile_content(
                which=which,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                compression_level=compression_level,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                clear_raw_outputs=clear_raw_outputs,
                pickup_where_leftoff=pickup_where_leftoff,
            )

            master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
            master_snakefile_path.write_text(master_snakefile_content)

            if verbose:
                print(
                    f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                    flush=True,
                )

            analysis_dir = self.master_analysis.analysis_paths.analysis_dir
            (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
            self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
            (self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)

            dry_run_result = self._base_builder._validate_batch_job_dry_run(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
            )

            if not dry_run_result.get("success"):
                raise RuntimeError("Dry run failed; workflow submission aborted.")

            if dry_run:
                self.sensitivity_analysis._update_master_analysis_log()
                return dry_run_result

            result = self._base_builder._submit_tmux_workflow(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
            )

            self.sensitivity_analysis._update_master_analysis_log()
            return result

        # Standard workflow submission (existing logic)
        # Detect execution mode
        if mode == "auto":
            mode = "slurm" if self.master_analysis.in_slurm else "local"

        if verbose:
            print(
                f"[Snakemake] Submitting sensitivity analysis workflow in {mode} mode",
                flush=True,
            )

        # Generate master Snakefile with flattened hierarchy
        # (no nested Snakemake calls - all rules in one file)
        master_snakefile_content = self.generate_master_snakefile_content(
            which=which,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            compression_level=compression_level,
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            clear_raw_outputs=clear_raw_outputs,
            pickup_where_leftoff=pickup_where_leftoff,
            report_config_path=report_config_path,
        )

        master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
        master_snakefile_path.write_text(master_snakefile_content)

        if verbose:
            print(
                f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                flush=True,
            )

        # Create required directories BEFORE Snakemake DAG construction
        # (onstart: in Snakefile runs AFTER DAG parsing, too late for file validation)
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        self.master_analysis.analysis_paths.simlog_directory.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[Snakemake] Created required directories (_status, {self.master_analysis.analysis_paths.simlog_directory})",  # noqa: E501
                flush=True,
            )

        # Always perform a dry run first
        if mode == "local":
            dry_run_result = self._base_builder.run_snakemake_local(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        else:  # slurm
            dry_run_result = self._base_builder._run_snakemake_slurm_detached(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=True,
                dry_run=True,
            )

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        if dry_run:
            self.sensitivity_analysis._update_master_analysis_log()
            return dry_run_result

        # Submit workflow based on mode
        if mode == "local":
            result = self._base_builder.run_snakemake_local(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                dry_run=False,
            )
        else:  # slurm
            result = self._base_builder._run_snakemake_slurm_detached(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
                dry_run=False,
            )

        # Print snakemake log file location if available
        if verbose and result.get("snakemake_logfile") is not None and not wait_for_completion:
            print(
                "[Snakemake] Sensitivity analysis workflow submitted in background.",
                flush=True,
            )
            print(
                f"[Snakemake] Monitor progress with: tail -f {result.get('snakemake_logfile')}",
                flush=True,
            )

        self.sensitivity_analysis._update_master_analysis_log()
        return result
