# %%
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
import yaml  # type: ignore

import TRITON_SWMM_toolkit.analysis as anlysis
from TRITON_SWMM_toolkit.cf_conventions import apply_global_attributes
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.utils import current_datetime_string, write_datatree_zarr
from TRITON_SWMM_toolkit.workflow import SensitivityAnalysisWorkflowBuilder

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


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
        self.sub_analyses_prefix = "sa_"
        self.subanalysis_dir = (
            self.master_analysis.analysis_paths.analysis_dir / "subanalyses"
        )
        self.independent_vars = self._attributes_varied_for_analysis()
        self.df_setup = self._retrieve_df_setup().loc[:, self.independent_vars]  # type: ignore
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
                self.master_analysis.run_python_functions_concurrently(
                    prepare_scenario_launchers, verbose=verbose
                )
            else:
                for launcher in prepare_scenario_launchers:
                    launcher()

            if self.all_scenarios_created is not True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(
                    f"Preparation failed for the following scenarios:\n{scens_not_created}"
                )
            self._update_master_analysis_log()
        elif self.master_analysis.cfg_analysis.multi_sim_run_method in ["batch_job"]:
            raise ValueError(
                "prepare scenarios is not currently executable as batch_job."
            )

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
        return self._workflow_builder.submit_workflow(
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
            clear_raw_outputs=clear_raw_outputs,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            compression_level=compression_level,
            pickup_where_leftoff=pickup_where_leftoff,
            wait_for_completion=wait_for_completion,
            dry_run=dry_run,
            verbose=verbose,
            override_hpc_total_nodes=override_hpc_total_nodes,
        )

    def run_all_sims(
        self,
        pickup_where_leftoff,
        concurrent: bool = False,
        process_outputs_after_sim_completion: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        compression_level: int = 5,
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
            self.master_analysis.run_simulations_concurrently(
                launch_functions, verbose=verbose
            )
        else:
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                sub_analysis.run_sims_in_sequence(
                    pickup_where_leftoff=pickup_where_leftoff,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    clear_raw_outputs=clear_raw_outputs,
                    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_master_analysis_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        scenario_timeseries_processing_launchers = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            launchers = sub_analysis.retrieve_scenario_timeseries_processing_launchers(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.master_analysis.run_python_functions_concurrently(
            scenario_timeseries_processing_launchers
        )
        return

    def _consolidate_outputs_in_each_subanalysis(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            sub_analysis.consolidate_analysis_outputs(
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
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
                success = (
                    success and sub_analysis.tritonswmm_triton_analysis_summary_created
                )
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
                node_success = (
                    node_success
                    and sub_analysis.tritonswmm_node_analysis_summary_created
                )
                link_success = (
                    link_success
                    and sub_analysis.tritonswmm_link_analysis_summary_created
                )
            elif cfg_sys.toggle_swmm_model:
                node_success = (
                    node_success
                    and sub_analysis.swmm_only_node_analysis_summary_created
                )
                link_success = (
                    link_success
                    and sub_analysis.swmm_only_link_analysis_summary_created
                )
        return node_success and link_success

    def consolidate_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = True,
        compression_level: int = 5,
    ):
        self.create_subanalysis_summaries(
            which=which,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_subanalysis_outputs(
            which=which,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_subanalysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_outputs_if_already_created: bool = False,
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
            overwrite_if_already_created=overwrite_outputs_if_already_created,
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
        apply_global_attributes(
            tree, analysis_id=str(self.master_analysis.cfg_analysis.analysis_id)
        )
        return tree

    def consolidate_sensitivity_datatree(
        self,
        overwrite_if_already_created: bool = False,
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
            raise ValueError(
                "sensitivity_datatree_zarr path is not configured on AnalysisPaths."
            )

        if (not overwrite_if_already_created) and fname_out.exists():
            if verbose:
                print(
                    f"Sensitivity DataTree zarr already present at {fname_out}. "
                    "Not overwriting."
                )
            return fname_out

        # Ensure each sub-analysis has its analysis_datatree.zarr built.
        for sa_id, sub_analysis in self.sub_analyses.items():
            sub_path = sub_analysis.analysis_paths.analysis_datatree_zarr
            if sub_path is None:
                continue
            if overwrite_if_already_created or not sub_path.exists():
                sub_analysis.process.consolidate_to_datatree(
                    overwrite_if_already_created=overwrite_if_already_created,
                    compression_level=compression_level,
                    verbose=verbose,
                )

        tree = self.build_sensitivity_datatree()
        write_datatree_zarr(tree, fname_out, compression_level=compression_level)

        self.master_analysis._refresh_log()
        if hasattr(
            self.master_analysis.log, "sensitivity_datatree_consolidation_complete"
        ):
            self.master_analysis.log.sensitivity_datatree_consolidation_complete.set(
                True
            )
        if verbose:
            print(f"Wrote sensitivity DataTree zarr to {fname_out}")
        return fname_out

    def open_sensitivity_datatree(self) -> "xr.DataTree":
        """Open the consolidated sensitivity DataTree zarr lazily."""
        path = self.analysis_paths.sensitivity_datatree_zarr
        if path is None or not path.exists():
            raise ValueError(
                "Sensitivity DataTree zarr not found. "
                "Run consolidate_sensitivity_datatree() first."
            )
        return xr.open_datatree(
            path, engine="zarr", chunks="auto", consolidated=False
        )

    def create_subanalysis_summaries(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_outputs_if_already_created: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="TRITON",
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="SWMM",
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
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

    def _attributes_varied_for_analysis(self):
        df_setup = self._retrieve_df_setup()
        keys_targeted_for_sensitivity = []
        for key, val in self.master_analysis.cfg_analysis.model_dump().items():
            # print(key)
            if key in df_setup.columns:
                keys_targeted_for_sensitivity.append(key)
        return keys_targeted_for_sensitivity

    def _retrieve_df_setup(self) -> pd.DataFrame:
        import re as _re

        snstivity_definition = self.master_analysis.cfg_analysis.sensitivity_analysis
        f_extension = snstivity_definition.name.lower().split(".")[-1]  # type: ignore
        if f_extension == "csv":
            df_setup = pd.read_csv(snstivity_definition)  # type: ignore
        elif f_extension == "xlsx":
            df_setup = pd.read_excel(snstivity_definition)
        else:
            raise ValueError(
                "File extension not recognized for file defining sensitivity analysis."
            )
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
                f"sa_id values must match ^[A-Za-z0-9_.]+$ (Snakemake-wildcard safe). "
                f"Offending values: {bad}"
            )
        df_setup = df_setup.set_index("sa_id")
        return df_setup

    def export_sensitivity_definition_csv(self) -> Path:
        """Export sensitivity analysis definition to analysis directory as CSV.

        Exports only the fields that vary across sub-analyses (self.df_setup columns)
        to a standardized 'sensitivity_analysis_definition.csv' file in the analysis directory.
        This allows easier inspection of the sensitivity analysis configuration during debugging.

        Returns:
            Path to the exported CSV file.
        """
        output_path = (
            self.analysis_paths.analysis_dir / "sensitivity_analysis_definition.csv"
        )
        df_export = self.df_setup.copy()
        df_export.to_csv(output_path, index=True)
        return output_path

    def _create_sub_analyses(self):
        dic_sensitivity_analyses = dict()
        for idx, row in self.df_setup.iterrows():
            sa_id = str(idx)
            cfg_snstvty_analysis = self.master_analysis.cfg_analysis.model_copy()

            for key, val in row.items():
                if key == "gpu_hardware_override":
                    if pd.isna(val) or val == "":
                        continue
                    setattr(cfg_snstvty_analysis, key, str(val))
                    continue
                setattr(cfg_snstvty_analysis, key, val)  # type: ignore
            analysis_id = f"{self.sub_analyses_prefix}{sa_id}"
            cfg_snstvty_analysis.analysis_id = analysis_id  # type: ignore
            sub_analysis_directory = self.subanalysis_dir / str(
                cfg_snstvty_analysis.analysis_id
            )
            sub_analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False
            cfg_snstvty_analysis.is_subanalysis = True

            cfg_anlysys_yaml = sub_analysis_directory / f"{analysis_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = sub_analysis_directory

            cfg_snstvty_analysis.master_analysis_cfg_yaml = (
                self.master_analysis.analysis_config_yaml
            )

            cfg_anlysys_yaml.write_text(
                yaml.safe_dump(
                    cfg_snstvty_analysis.model_dump(mode="json"),
                    sort_keys=False,
                )
            )
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=self._system,
            )
            dic_sensitivity_analyses[sa_id] = anlsys
        return dic_sensitivity_analyses

    def compile_TRITON_SWMM_for_sensitivity_analysis(
        self,
        verbose: bool = False,
        recompile_if_already_done_successfully: bool = False,
    ):
        self._system.compile_TRITON_SWMM(
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
                all_models_completed = all(
                    scen.model_run_completed(model_type)
                    for model_type in enabled_models
                )
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
            assert (
                sub_analysis.cfg_analysis.is_subanalysis
            ), "is_subanalysis attribute not true in sub_analysis.cfg_analysis.is_subanalysis"
            sub_df_status = sub_analysis.df_status.copy()

            setup_row = self.df_setup.loc[sa_id, :]
            for key, val in setup_row.items():
                sub_df_status[key] = val

            sub_df_status["sa_id"] = sa_id
            sub_df_status = sub_df_status[
                ["sa_id"] + [c for c in sub_df_status.columns if c != "sa_id"]
            ]

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
            all_scenarios_created = (
                all_scenarios_created and sub_analysis.log.all_scenarios_created.get()
            )
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
                all_TRITON_timeseries_processed
                and sub_analysis.log.all_TRITON_timeseries_processed.get()
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
                all_SWMM_timeseries_processed
                and sub_analysis.log.all_SWMM_timeseries_processed.get()
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
                all_raw_TRITON_outputs_cleared
                and sub_analysis.log.all_raw_TRITON_outputs_cleared.get()
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
                all_raw_SWMM_outputs_cleared
                and sub_analysis.log.all_raw_SWMM_outputs_cleared.get()
            )
        return all_raw_SWMM_outputs_cleared is True

    def _update_master_analysis_log(self):
        self.master_analysis._update_log()
        return
