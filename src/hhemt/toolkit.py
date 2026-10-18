"""Notebook-friendly entry point for building and running a TRITON-SWMM analysis.

Running an analysis by hand means constructing a ``TRITONSWMM_system`` and a
``TRITONSWMM_analysis`` from their config files, linking the two, running preflight
validation, and translating a run mode into ``analysis.run()``'s ``from_scratch``
flag. ``Toolkit`` does each of those in one call. For a ``run()`` option it does not
expose, call ``tk.analysis.run()`` directly.

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
    >>> print(f"{len(result.events_processed)} events in scope")
    >>>
    >>> # Check status
    >>> status = tk.get_status()
    >>> print(status)
"""

from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError
from .orchestration import RunMode, WorkflowResult, WorkflowStatus
from .system import TRITONSWMM_system

__all__ = ["Toolkit"]


class Toolkit:
    """High-level API facade for TRITON-SWMM workflow orchestration.

    This class provides a simplified interface to the toolkit, wrapping the
    underlying Analysis class with notebook-friendly methods and sensible defaults.

    Attributes
    ----------
    system : TRITONSWMM_system
        The system instance carrying the system configuration.
    analysis : TRITONSWMM_analysis
        The analysis instance used for workflow orchestration.

    Examples
    --------
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
    ...     print(f"✓ Workflow succeeded: {len(result.events_processed)} events in scope")
    ... else:
    ...     print(f"✗ Workflow failed: {result.message}")

    Resume interrupted workflow:

    >>> # Check current status
    >>> status = tk.get_status()
    >>> print(status)
    >>> print(f"Recommendation: {status.recommendation}")
    >>>
    >>> # Check current_phase first: recommended_mode may be "fresh", which wipes
    >>> result = tk.run(mode=status.recommended_mode)
    """

    def __init__(self, system: "TRITONSWMM_system"):
        """Wrap a system whose analysis is already linked.

        Args:
            system: A ``TRITONSWMM_system`` whose analysis is linked, as
                ``from_configs`` leaves it. On an unlinked system, reading
                ``system.analysis`` here raises ``RuntimeError("No analysis defined.
                Call add_analysis() first.")``; no ``add_analysis`` method exists, so
                build through ``from_configs``, which does the linking.

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
        case_manifest_yaml: str | Path | None = None,
        validate: bool = True,
    ) -> "Toolkit":
        """Create Toolkit instance from configuration files.

        This is the recommended way to initialize the toolkit. It handles:
        - Loading and validating configurations
        - Instantiating system and analysis objects
        - Running preflight validation checks (if validate=True)

        Parameters
        ----------
        system_config : str or Path
            Path to the system configuration YAML file.
        analysis_config : str or Path
            Path to the analysis configuration YAML file.
        hpc_system_config : str or Path, optional
            Path to the per-HPC-system configuration YAML file
            (``hpc_system_config.yaml``). When None, the default, the analysis is
            built without one, so nothing it supplies is available: the
            per-partition caps ``validate()`` checks, the partition GPU hardware the
            SLURM emitters resolve, and the SBATCH account.
        case_manifest_yaml : str or Path, optional
            Path to the case study's ``case.yaml`` manifest. Forwarded to the
            analysis constructor so ``case_name`` is bundled, which is a blocking
            combine-compatibility field, and the RO-Crate root name resolves from
            the real manifest. When None, the default, no ``case.yaml`` is bundled
            and the RO-Crate root name falls back to the ``analysis_id``; a manifest
            that fails to load falls back for the name too.
        validate : bool, default True
            Whether to run preflight validation. Raises ConfigurationError if
            validation fails.

        Returns
        -------
        Toolkit
            Initialized Toolkit instance ready for workflow execution.

        Raises
        ------
        ConfigurationError
            If configuration files are invalid or validation fails.
        FileNotFoundError
            If configuration files do not exist.

        Examples
        --------
        >>> from hhemt import Toolkit
        >>>
        >>> tk = Toolkit.from_configs(
        ...     system_config="configs/system.yaml",
        ...     analysis_config="configs/analysis.yaml"
        ... )
        >>>
        >>> # Toolkit is ready - configs loaded, validated, and bound
        >>> print(f"Analysis directory: {tk.analysis_dir}")
        >>> print(f"Scenarios: {tk.n_simulations}")
        """
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system
        from .validation import assert_both_configs_load

        # Meant to report BOTH documents' load errors in one round, but today it does
        # not: config/loaders.py::_load_config turns every pydantic ValidationError into
        # a ConfigurationError, which assert_both_configs_load does not catch, so the
        # first failing document's error propagates alone.
        assert_both_configs_load(Path(system_config), Path(analysis_config))

        system = TRITONSWMM_system(Path(system_config))
        analysis = TRITONSWMM_analysis(
            Path(analysis_config),
            system,
            hpc_system_config_yaml=(Path(hpc_system_config) if hpc_system_config is not None else None),
            case_manifest_yaml=(Path(case_manifest_yaml) if case_manifest_yaml is not None else None),
        )
        system._analysis = analysis  # Link back

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
        """Load-smoke facade for the synthetic compute-config experiment.

        Loads and validates the ``synthetic_experiment_config``, which runs every one
        of its validators: the ``model_arms`` field check and the coupling-invariant,
        partition-cap and DEM-ladder model checks. The partition-cap check is the one
        that loads the per-cluster ``hpc_system_config``. Builds the partition-as-axis
        experiment matrix. Unless ``dry_run`` is set, writes the matrix CSV and
        generates the synthetic model under ``dest_dir``.

        Parameters
        ----------
        config : str or Path
            Path to a ``synthetic_experiment_config`` YAML.
        hpc_system_config : str or Path, optional
            Path to the per-cluster ``hpc_system_config`` YAML. When given it
            overrides the config's own ``hpc_system_config_yaml``.
        dest_dir : str or Path, optional
            Output directory for the matrix CSV and generated model on a non-dry
            run. Defaults to ``{config parent}/synth_experiment_out``.
        dry_run : bool, default False
            Validate the config and build the matrix in memory, then return
            without writing files or generating the model.

        Returns
        -------
        dict
            ``{"config": synthetic_experiment_config, "n_matrix_rows": int,
            "matrix_csv": Path | None, "model_dir": Path | None}``.

        Notes
        -----
        It does not compose or run the analysis ensemble the matrix describes;
        ``scripts/experiments/synth_compute_config.py`` composes and runs it.
        """
        import yaml

        from .config.synthetic_experiment import synthetic_experiment_config
        from .synthetic_experiment import build_experiment_matrix, generate_synthetic_experiment

        config = Path(config)
        cfg_dict = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if hpc_system_config is not None:
            cfg_dict["hpc_system_config_yaml"] = str(hpc_system_config)
        cfg = synthetic_experiment_config.model_validate(cfg_dict)  # runs every validator

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
        mode: RunMode | str = RunMode.resume,
        events: list[int] | None = None,
        dry_run: bool = False,
        verbose: bool = True,
        report_config: Path | None = None,
        override_force_rerun=None,
        wait_for_completion: bool | None = None,
        override_wipe_nonempty: bool = False,
    ) -> WorkflowResult:
        """Run TRITON-SWMM workflow.

        This is the main entry point for workflow execution. It handles:
        - System input processing (DEM, Manning's coefficients)
        - Asserting that every enabled model already has a successful build (native mode)
        - Scenario preparation (SWMM model generation)
        - Simulation execution (TRITON-SWMM runs)
        - Output processing (timeseries extraction, compression)
        - Consolidation (analysis-level aggregation)

        The solver is not built here. In native mode, build it once per
        machine and GPU target before the first run; see
        [Compile the solver](../how-to/compiling-the-solver.md).

        Parameters
        ----------
        mode : RunMode or {'fresh', 'resume'}, default 'resume'
            Execution mode controlling checkpoint behavior. ``fresh`` deletes the
            analysis directory and rebuilds everything (refused while it holds
            completed work unless ``override_wipe_nonempty=True``); ``resume``
            continues from the last checkpoint. Any other value raises
            ``ConfigurationError``. To re-run completed scenarios without a wipe,
            pass ``override_force_rerun`` instead.
        events : list of int, or None
            Not yet supported: any value other than None makes ``analysis.run()``
            raise ``NotImplementedError``. None, the default, runs every event in the
            analysis.
        dry_run : bool, default False
            Print the workflow plan without executing it.
        verbose : bool, default True
            Print progress messages during execution.
        report_config : Path, optional
            A report config YAML used instead of ``cfg_analysis.report`` for this
            run. None, the default, uses the config's own.
        override_force_rerun : {'all', 'none'} or dict, optional
            Runtime override for ``cfg_analysis.force_rerun`` for this invocation
            only; ``"all"`` re-runs every completed scenario and everything
            downstream without deleting the analysis directory. None, the default,
            reads the config field.
        wait_for_completion : bool, optional
            Whether to block until the workflow finishes. None, the default, defers
            to ``analysis.run()``'s own default, which blocks unless the resolved
            locus is SLURM; a bool overrides it.
        override_wipe_nonempty : bool, default False
            With ``mode="fresh"``, permit deleting an analysis directory that holds
            completed work; without it that wipe is refused.

        Returns
        -------
        WorkflowResult
            Execution details: ``success``, ``mode``, ``execution_time`` in seconds,
            ``phases_completed``, ``events_processed``, ``snakefile_path`` to the
            generated Snakefile, ``job_id``, a ``message`` and ``partial_failures``.
            On a ``batch_job`` tmux launch that is not waited on -- the default on a
            SLURM locus, when ``wait_for_completion`` is None or False --
            ``success=True`` means the launch was made, not that every rule
            succeeded; ``WorkflowResult`` states what each field holds.

        Raises
        ------
        ConfigurationError
            If the configuration is invalid, or ``mode`` is not a ``RunMode`` member.
        WorkflowError
            If workflow execution fails.

        Examples
        --------
        Fresh run, overwriting everything:

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

        Dry run (preview without executing):

        >>> result = tk.run(mode="fresh", dry_run=True)
        >>> print(result.message)  # a one-line summary of the dry run

        Notes
        -----
        - Execution mode, local or SLURM, is auto-detected from the configuration.
        - Use ``get_status()`` to check current progress before resuming.
        - For fine-grained control, use ``analysis.run()`` directly.
        """
        # Always "auto": analysis.run() resolves local vs SLURM from the config.
        execution_mode = self._detect_execution_mode()

        # Refuse an out-of-set mode HERE, the layer that owns the reduction below. The
        # CLI closes --mode at parse time, but a notebook calls this method with the bare
        # word, and an unknown value must not silently resume (measured: it did).
        try:
            run_mode = RunMode(mode)
        except ValueError as exc:
            raise ConfigurationError(
                field="mode",
                message=f"must be one of {[m.value for m in RunMode]}; got {mode!r}",
            ) from exc

        # Map the public ``mode`` knob to ``analysis.run()``'s ``from_scratch`` flag:
        # ``analysis.run()`` accepts neither ``mode`` nor ``phases``, and
        # ``from_scratch=True`` is what makes it delete the analysis directory.
        from_scratch = run_mode == RunMode.fresh

        # wait_for_completion=None preserves analysis.run()'s own default
        # (wait = exec_mode != "slurm"); an explicit bool overrides it. run_experiment()
        # passes an explicit value so a batch_job hosted inside an sbatch blocks.
        return self.analysis.run(
            from_scratch=from_scratch,
            override_wipe_nonempty=override_wipe_nonempty,
            events=events,
            execution_mode=execution_mode,
            dry_run=dry_run,
            verbose=verbose,
            report_config=report_config,
            override_force_rerun=override_force_rerun,
            wait_for_job_completion=wait_for_completion,
        )

    def get_status(self) -> WorkflowStatus:
        """Get current workflow status report.

        This method reads the system, scenario and per-model logs to determine the
        completion state of each workflow phase. It writes no log, flag or output,
        but it does create any missing per-scenario directories, because each
        ``TRITONSWMM_scenario`` it constructs creates them. It provides:
        - Per-phase completion status
        - Counts of scenarios completed, failed and not yet started (scenarios, not
          model runs, despite the ``simulations_*`` field names)
        - Current workflow phase
        - Recommended mode for next run()
        - Actionable recommendation message

        Returns
        -------
        WorkflowStatus
            Detailed phase information and recommendations.

        Examples
        --------
        Check status and decide next action:

        >>> status = tk.get_status()
        >>> print(status)  # abbreviated: per-phase detail lines elided as "..."
        Workflow Status Report
        ══════════════════════════════════════════════════════════════════
        Analysis: my_analysis
        Directory: /path/to/analysis
        Phase Status:
          ✓ Setup
            ...
          ✓ Preparation
            ...
          ⚠ Simulation (50% complete)
            ...
          ✗ Processing
            ...
          ✗ Consolidation
            ...
        Recommendation:
          ...
        >>>
        >>> # Follow the recommendation
        >>> if not status.simulation.complete:
        ...     result = tk.run(mode=status.recommended_mode)

        Inspect phase details:

        >>> status = tk.get_status()
        >>> print(f"Setup complete: {status.setup.complete}")
        >>> print(f"Scenarios run: {status.simulations_completed}")
        >>> print(f"Scenarios failed: {status.simulations_failed}")
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

        Notes
        -----
        - Status is read from the logs on every call; nothing is cached between calls.
        - ``recommended_mode`` is ``fresh`` both when setup is incomplete and when every
          phase is done, and ``fresh`` deletes the analysis directory, so check
          ``current_phase`` before passing it to ``run()``.
        - ``simulations_failed`` counts scenarios with a model run that started and did
          not finish; ``simulations_pending`` counts those with no run started.
        """
        return self.analysis.get_workflow_status()

    def _detect_execution_mode(self) -> Literal["auto", "local", "slurm"]:
        """Return the execution-mode value to hand to ``analysis.run()``.

        Always returns ``"auto"``. This method does NOT read the environment and
        does NOT itself choose between local and SLURM: execution locus is resolved
        in exactly one place, ``orchestration.resolve_execution_locus``, which
        ``analysis.run()`` calls with this value. The comment below records why this
        method no longer chooses.

        Returns:
            ``"auto"``, always. ``"local"`` and ``"slurm"`` remain in the return
            annotation because the value is passed straight through to
            ``analysis.run(execution_mode=...)``, whose parameter accepts all three;
            this method simply never selects the other two.
        """
        # Delegates: execution locus is resolved in exactly ONE place,
        # orchestration.resolve_execution_locus, called from analysis.run(). This method
        # previously carried its own copy of the rule, and that copy was BOTH
        # env-reading (`self.analysis.in_slurm`, which promotes a local-configured
        # analysis inside an allocation) AND incomplete (it omitted the batch_job
        # arm, so a batch_job config resolved "local"). Returning "auto" removes
        # both defects by construction rather than repairing them in parallel.
        # "auto" is already in this method's declared return type.
        return "auto"

    @property
    def analysis_dir(self) -> Path:
        """Get analysis directory path.

        Returns
        -------
        Path
            Path to analysis output directory.

        Examples
        --------
        >>> tk = Toolkit.from_configs(system_cfg, analysis_cfg)
        >>> print(f"Outputs at: {tk.analysis_dir}")
        Outputs at: /path/to/system_directory/my_analysis
        """
        return self.analysis.analysis_paths.analysis_dir

    @property
    def n_simulations(self) -> int:
        """Number of scenarios in the analysis.

        Returns
        -------
        int
            Events times sensitivity members (members are 1 without a sensitivity
            analysis). Despite the name it counts scenarios, not model runs;
            ``tk.analysis.n_sims`` counts model runs.

        Examples
        --------
        >>> tk = Toolkit.from_configs(system_cfg, analysis_cfg)
        >>> print(f"Scenarios: {tk.n_simulations}")
        Scenarios: 24
        """
        return self.analysis.nsims

    def __repr__(self) -> str:
        """Return string representation of Toolkit instance."""
        return (
            f"Toolkit(analysis='{self.analysis.cfg_analysis.analysis_id}', "
            f"n_simulations={self.n_simulations}, "
            f"dir='{self.analysis_dir}')"
        )
