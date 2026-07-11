"""High-level API facade for H&H Ensemble Modeling Toolkit.

This module provides a simplified, notebook-friendly interface to the toolkit.
For more control, use the underlying Analysis class directly.

Example:
    >>> from hhemt import Toolkit
    >>>
    >>> # Simple setup
    >>> tk = Toolkit.from_configs(
    ...     system_config="system.yaml",
    ...     analysis_config="analysis.yaml"
    ... )
    >>>
    >>> # Run workflow
    >>> result = tk.run(mode="fresh")
    >>> print(f"Success: {result.success}")
    >>> print(f"Processed {len(result.events_processed)} events")
    >>>
    >>> # Check status
    >>> status = tk.get_status()
    >>> print(status)
"""

from pathlib import Path
from typing import Literal

from .orchestration import WorkflowResult, WorkflowStatus
from .system import TRITONSWMM_system

__all__ = ["Toolkit"]


class Toolkit:
    """High-level API facade for TRITON-SWMM workflow orchestration.

    This class provides a simplified interface to the toolkit, wrapping the
    underlying Analysis class with notebook-friendly methods and sensible defaults.

    Attributes:
        system: The TRITONSWMM_system instance containing system configuration
        analysis: The TRITONSWMM_analysis instance for workflow orchestration

    Example:
        Basic workflow execution:

        >>> from hhemt import Toolkit
        >>>
        >>> # Load configurations
        >>> tk = Toolkit.from_configs(
        ...     system_config="configs/system.yaml",
        ...     analysis_config="configs/analysis.yaml"
        ... )
        >>>
        >>> # Run from scratch
        >>> result = tk.run(mode="fresh")
        >>> if result.success:
        ...     print(f"✓ Workflow complete: {len(result.events_processed)} events")
        ... else:
        ...     print(f"✗ Workflow failed: {result.message}")

        Resume interrupted workflow:

        >>> # Check current status
        >>> status = tk.get_status()
        >>> print(status)
        >>> print(f"Recommendation: {status.recommendation}")
        >>>
        >>> # Resume from last checkpoint
        >>> result = tk.run(mode=status.recommended_mode)

        Run specific events only:

        >>> # Process events 0-4 only
        >>> result = tk.run(mode="resume", events=list(range(5)))

        Run specific workflow phases:

        >>> # Only run simulation phase (skip setup/preparation)
        >>> result = tk.run(
        ...     mode="resume",
        ...     phases=["simulation"]
        ... )
    """

    def __init__(self, system: "TRITONSWMM_system"):
        """Initialize Toolkit with a system instance.

        Args:
            system: Initialized TRITONSWMM_system instance

        Note:
            Prefer using Toolkit.from_configs() for simpler initialization.
        """
        self.system = system
        self.analysis = system.analysis

    @classmethod
    def from_configs(
        cls,
        system_config: str | Path,
        analysis_config: str | Path,
        hpc_system_config: str | Path | None = None,
        validate: bool = True,
    ) -> "Toolkit":
        """Create Toolkit instance from configuration files.

        This is the recommended way to initialize the toolkit. It handles:
        - Loading and validating configurations
        - Instantiating system and analysis objects
        - Running preflight validation checks (if validate=True)

        Args:
            system_config: Path to system configuration YAML file
            analysis_config: Path to analysis configuration YAML file
            hpc_system_config: Optional path to the per-HPC-system configuration
                YAML file (``hpc_system_config.yaml``). When None (default),
                behavior is byte-identical to today — the HPC config consumers
                wire in later phases.
            validate: Whether to run preflight validation (default: True).
                Raises ConfigurationError if validation fails.

        Returns:
            Initialized Toolkit instance ready for workflow execution

        Raises:
            ConfigurationError: If configuration files are invalid or validation fails
            FileNotFoundError: If configuration files don't exist

        Example:
            >>> from hhemt import Toolkit
            >>>
            >>> tk = Toolkit.from_configs(
            ...     system_config="configs/system.yaml",
            ...     analysis_config="configs/analysis.yaml"
            ... )
            >>>
            >>> # Toolkit is ready - system inputs processed, executables compiled
            >>> print(f"Analysis directory: {tk.analysis.analysis_dir}")
            >>> print(f"Total simulations: {tk.analysis.n_simulations}")
        """
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        # Load system and analysis
        system = TRITONSWMM_system(Path(system_config))
        analysis = TRITONSWMM_analysis(
            Path(analysis_config),
            system,
            hpc_system_config_yaml=(Path(hpc_system_config) if hpc_system_config is not None else None),
        )
        system._analysis = analysis  # Link back

        # Run preflight validation if requested
        if validate:
            validation_result = system.analysis.validate()
            validation_result.raise_if_invalid()

        return cls(system)

    @classmethod
    def synthetic_experiment(
        cls,
        config: str | Path,
        *,
        hpc_system_config: str | Path | None = None,
        dest_dir: str | Path | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Load-smoke facade for the synthetic compute-config experiment (PIP-2 Phase 1).

        Loads and validates the ``synthetic_experiment_config`` (firing the
        coupling-invariant AND partition-cap validators — the latter loads the
        per-cluster ``hpc_system_config``), builds the partition-as-axis experiment
        matrix, and — unless ``dry_run`` — writes the matrix CSV and generates the
        synthetic model under ``dest_dir``.

        Args:
            config: Path to a ``synthetic_experiment_config`` YAML.
            hpc_system_config: Optional path to the per-cluster ``hpc_system_config``
                YAML; when given it overrides the config's ``hpc_system_config_yaml``.
            dest_dir: Output dir for the matrix CSV + generated model (non-dry-run).
                Defaults to ``{config parent}/synth_experiment_out``.
            dry_run: When True, validate the config + build the matrix in memory and
                return WITHOUT writing files or generating the model (the DoD
                "load-smoke").

        Returns:
            ``{"config": synthetic_experiment_config, "n_matrix_rows": int,
               "matrix_csv": Path | None, "model_dir": Path | None}``.

        Note:
            This Phase-1 scaffold does NOT compose and run a full analysis ensemble;
            that composition currently lives in
            ``scripts/experiments/synth_compute_config.py`` and is promoted into the
            framework in a later phase (see the deferred follow-up).
        """
        import yaml

        from .config.synthetic_experiment import synthetic_experiment_config
        from .synthetic_experiment import build_experiment_matrix, generate_synthetic_experiment

        config = Path(config)
        cfg_dict = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if hpc_system_config is not None:
            cfg_dict["hpc_system_config_yaml"] = str(hpc_system_config)
        cfg = synthetic_experiment_config.model_validate(cfg_dict)  # fires both validators

        matrix = build_experiment_matrix(cfg)
        result: dict = {
            "config": cfg,
            "n_matrix_rows": len(matrix),
            "matrix_csv": None,
            "model_dir": None,
        }
        if not dry_run:
            dest = Path(dest_dir) if dest_dir is not None else config.parent / "synth_experiment_out"
            dest.mkdir(parents=True, exist_ok=True)
            matrix_csv = dest / "experiment_matrix.csv"
            matrix.to_csv(matrix_csv, index=False)
            model_dir = dest / "model"
            generate_synthetic_experiment(cfg, model_dir)
            result["matrix_csv"] = matrix_csv
            result["model_dir"] = model_dir
        return result

    def run(
        self,
        mode: Literal["fresh", "resume", "overwrite"] = "resume",
        phases: list[str] | None = None,
        events: list[int] | None = None,
        dry_run: bool = False,
        verbose: bool = True,
        report_config: Path | None = None,
        override_force_rerun=None,
    ) -> WorkflowResult:
        """Run TRITON-SWMM workflow.

        This is the main entry point for workflow execution. It handles:
        - System input processing (DEM, Manning's coefficients)
        - TRITON/SWMM compilation
        - Scenario preparation (SWMM model generation)
        - Simulation execution (TRITON-SWMM runs)
        - Output processing (timeseries extraction, compression)
        - Consolidation (analysis-level aggregation)

        Args:
            mode: Execution mode controlling checkpoint behavior:
                - "fresh": Start from scratch, overwrite all outputs
                - "resume": Resume from last checkpoint (default)
                - "overwrite": Rerun existing scenarios without full reset
            phases: Optional list of phases to run. If None, runs all phases.
                Valid phases: ["setup", "preparation", "simulation",
                              "processing", "consolidation"]
            events: Optional list of event indices to process. If None,
                processes all events in the analysis.
            dry_run: If True, print workflow plan without executing
            verbose: If True, print progress messages during execution

        Returns:
            WorkflowResult with execution details:
                - success (bool): Whether workflow completed successfully
                - mode (str): Mode used for execution
                - execution_time (float): Total runtime in seconds
                - phases_completed (List[str]): Phases that finished
                - events_processed (List[int]): Event indices processed
                - snakefile_path (Path): Path to generated Snakefile
                - job_id (str): SLURM job ID (if HPC execution)
                - message (str): Status message or error description

        Raises:
            ConfigurationError: If configuration is invalid
            WorkflowError: If workflow execution fails

        Example:
            Fresh run (overwrite everything):

            >>> result = tk.run(mode="fresh")
            >>> print(f"Success: {result.success}")
            >>> print(f"Runtime: {result.execution_time:.1f}s")
            >>> print(f"Events: {result.events_processed}")

            Resume interrupted workflow:

            >>> # Check what's done
            >>> status = tk.get_status()
            >>> print(f"Progress: {status.simulations_completed}/{status.total_simulations}")
            >>>
            >>> # Continue from checkpoint
            >>> result = tk.run(mode="resume")

            Run specific events:

            >>> # Process only hurricane Irene and Sandy
            >>> result = tk.run(mode="resume", events=[5, 12])

            Run only simulation phase:

            >>> # Skip setup/preparation, just run simulations
            >>> result = tk.run(
            ...     mode="resume",
            ...     phases=["simulation"]
            ... )

            Dry run (preview without executing):

            >>> result = tk.run(mode="fresh", dry_run=True)
            >>> print(result.message)  # Shows what would be executed

        Notes:
            - Execution mode (local vs SLURM) is auto-detected from configuration
            - Use get_status() to check current progress before resuming
            - For fine-grained control, use analysis.run() directly
        """
        # Auto-detect execution mode
        execution_mode = self._detect_execution_mode()

        # Delegate to analysis.run()
        return self.analysis.run(
            mode=mode,
            phases=phases,
            events=events,
            execution_mode=execution_mode,
            dry_run=dry_run,
            verbose=verbose,
            report_config=report_config,
            override_force_rerun=override_force_rerun,
        )

    def get_status(self) -> WorkflowStatus:
        """Get current workflow status report.

        This method inspects logs and outputs to determine the completion state
        of each workflow phase, providing:
        - Per-phase completion status
        - Number of simulations completed/pending/failed
        - Current workflow phase
        - Recommended mode for next run()
        - Actionable recommendation message

        Returns:
            WorkflowStatus with detailed phase information and recommendations

        Example:
            Check status and decide next action:

            >>> status = tk.get_status()
            >>> print(status)
            Workflow Status Report
            ════════════════════════════════════════
            Analysis: norfolk_coastal_flooding
            Directory: /path/to/analysis

            Phase Status:
            ✓ Setup (complete)
            ✓ Scenario Preparation (complete)
            ⚠ Simulation (in progress: 12/24 complete)
            ✗ Output Processing (not started)
            ✗ Consolidation (not started)

            Progress: 12/24 simulations complete (0 failed)
            Current Phase: simulation

            Recommendation:
            Use mode='resume' to continue simulation execution.
            12 simulations have completed successfully.
            >>>
            >>> # Follow the recommendation
            >>> if not status.simulation.complete:
            ...     result = tk.run(mode=status.recommended_mode)

            Inspect phase details:

            >>> status = tk.get_status()
            >>> print(f"Setup complete: {status.setup.complete}")
            >>> print(f"Simulations done: {status.simulations_completed}")
            >>> print(f"Simulations failed: {status.simulations_failed}")
            >>> print(f"Recommended mode: {status.recommended_mode}")

            Check if workflow is fully complete:

            >>> status = tk.get_status()
            >>> if all([
            ...     status.setup.complete,
            ...     status.preparation.complete,
            ...     status.simulation.complete,
            ...     status.processing.complete,
            ...     status.consolidation.complete,
            ... ]):
            ...     print("✓ Workflow fully complete!")

        Notes:
            - Status is determined by inspecting actual outputs, not cached state
            - Use recommended_mode for next run() to follow best practices
            - Check simulations_failed to detect partial failures
        """
        return self.analysis.get_workflow_status()

    def _detect_execution_mode(self) -> Literal["auto", "local", "slurm"]:
        """Detect appropriate execution mode from configuration and environment.

        Returns:
            "slurm" if in SLURM context or configured for SLURM,
            "local" otherwise
        """
        if self.analysis.in_slurm or self.analysis.cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks":
            return "slurm"
        return "local"

    @property
    def analysis_dir(self) -> Path:
        """Get analysis directory path.

        Returns:
            Path to analysis output directory

        Example:
            >>> tk = Toolkit.from_configs(system_cfg, analysis_cfg)
            >>> print(f"Outputs at: {tk.analysis_dir}")
            Outputs at: /path/to/norfolk_coastal_flooding_2024-01-15_143022
        """
        return self.analysis.analysis_paths.analysis_dir

    @property
    def n_simulations(self) -> int:
        """Get total number of simulations in analysis.

        Returns:
            Number of scenarios/events to be processed

        Example:
            >>> tk = Toolkit.from_configs(system_cfg, analysis_cfg)
            >>> print(f"Total simulations: {tk.n_simulations}")
            Total simulations: 24
        """
        return self.analysis.nsims

    def __repr__(self) -> str:
        """Return string representation of Toolkit instance."""
        return (
            f"Toolkit(analysis='{self.analysis.cfg_analysis.analysis_id}', "
            f"n_simulations={self.n_simulations}, "
            f"dir='{self.analysis_dir}')"
        )
