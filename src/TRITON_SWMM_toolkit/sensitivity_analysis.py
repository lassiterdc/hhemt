# %%
import TRITON_SWMM_toolkit.utils as ut
import pandas as pd
from typing import Literal, TYPE_CHECKING
import time
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.workflow import SensitivityAnalysisWorkflowBuilder
import yaml  # type: ignore
import TRITON_SWMM_toolkit.analysis as anlysis
import xarray as xr

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


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

    def _select_triton_summary(self, sub_analysis: "TRITONSWMM_analysis"):
        cfg_sys = self.master_analysis._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            return sub_analysis.tritonswmm_TRITON_summary
        if cfg_sys.toggle_triton_model:
            return sub_analysis.triton_only_summary
        raise ValueError(
            "TRITON outputs requested, but neither TRITONSWMM nor TRITON-only models are enabled."
        )

    def _select_swmm_node_summary(self, sub_analysis: "TRITONSWMM_analysis"):
        cfg_sys = self.master_analysis._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            return sub_analysis.tritonswmm_SWMM_node_summary
        if cfg_sys.toggle_swmm_model:
            return sub_analysis.swmm_only_node_summary
        raise ValueError(
            "SWMM node outputs requested, but neither TRITONSWMM nor SWMM-only models are enabled."
        )

    def _select_swmm_link_summary(self, sub_analysis: "TRITONSWMM_analysis"):
        cfg_sys = self.master_analysis._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            return sub_analysis.tritonswmm_SWMM_link_summary
        if cfg_sys.toggle_swmm_model:
            return sub_analysis.swmm_only_link_summary
        raise ValueError(
            "SWMM link outputs requested, but neither TRITONSWMM nor SWMM-only models are enabled."
        )

    def _triton_output_mode(self) -> str:
        cfg_sys = self.master_analysis._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            return "tritonswmm_triton"
        if cfg_sys.toggle_triton_model:
            return "triton_only"
        raise ValueError(
            "TRITON outputs requested, but no TRITON model is enabled in system config."
        )

    def _swmm_output_modes(self) -> tuple[str, str]:
        cfg_sys = self.master_analysis._system.cfg_system
        if cfg_sys.toggle_tritonswmm_model:
            return "tritonswmm_swmm_node", "tritonswmm_swmm_link"
        if cfg_sys.toggle_swmm_model:
            return "swmm_only_node", "swmm_only_link"
        raise ValueError(
            "SWMM outputs requested, but no SWMM model is enabled in system config."
        )

    def _combine_TRITON_outputs_per_subanalysis(self):
        assert self.TRITON_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = self._select_triton_summary(sub_analysis)
            ds = ds.assign_coords(coords={"sub_analysis_iloc": sub_analysis_iloc})
            ds = ds.expand_dims("sub_analysis_iloc")
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)
            lst_ds.append(ds)

        ds_triton_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_triton_outputs

    def _combine_SWMM_node_outputs_per_subanalysis(self):
        assert self.SWMM_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = self._select_swmm_node_summary(sub_analysis)
            ds = ds.assign_coords(coords={"sub_analysis_iloc": sub_analysis_iloc})
            ds = ds.expand_dims("sub_analysis_iloc")
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_node_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_node_outputs

    def _combine_SWMM_link_outputs_outputs_per_subanalysis(self):
        assert self.SWMM_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = self._select_swmm_link_summary(sub_analysis)
            ds = ds.assign_coords(coords={"sub_analysis_iloc": sub_analysis_iloc})
            ds = ds.expand_dims("sub_analysis_iloc")
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_link_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_link_outputs

    def _combine_TRITONSWMM_performance_per_subanalysis(self):
        """
        Combine TRITONSWMM performance summaries from all sub-analyses.

        Returns
        -------
        xr.Dataset
            Combined performance dataset with sensitivity analysis dimensions
        """
        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            # Access the performance summary from the sub-analysis
            if self.master_analysis._system.cfg_system.toggle_tritonswmm_model:
                ds = sub_analysis.process.tritonswmm_performance_summary
            elif self.master_analysis._system.cfg_system.toggle_triton_model:
                ds = sub_analysis.process.triton_only_performance_summary
            else:
                raise ValueError(
                    "Performance summaries requested, but no TRITON model is enabled."
                )
            ds = ds.assign_coords(coords={"sub_analysis_iloc": sub_analysis_iloc})
            ds = ds.expand_dims("sub_analysis_iloc")

            # Add sensitivity analysis dimensions
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_performance = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_performance

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
        cfg_sys = self.master_analysis._system.cfg_system
        if which in ["TRITON", "both"]:
            ds_combined_outputs = self._combine_TRITON_outputs_per_subanalysis()
            triton_mode = self._triton_output_mode()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode=triton_mode,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            ds_combined_outputs = self._combine_SWMM_node_outputs_per_subanalysis()
            swmm_node_mode, swmm_link_mode = self._swmm_output_modes()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode=swmm_node_mode,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                verbose=verbose,
                compression_level=compression_level,
            )
            ds_combined_outputs = (
                self._combine_SWMM_link_outputs_outputs_per_subanalysis()
            )
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode=swmm_link_mode,
                overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
                verbose=verbose,
                compression_level=compression_level,
            )

        # Consolidate performance summaries using MODE_CONFIG pipeline
        # (independent of 'which' parameter - performance is always combined)
        if cfg_sys.toggle_tritonswmm_model:
            perf_mode = "tritonswmm_performance"
        elif cfg_sys.toggle_triton_model:
            perf_mode = "triton_only_performance"
        else:
            # No TRITON-based model enabled, skip performance consolidation
            return

        # Combine performance from all sub-analyses
        ds_performance = self._combine_TRITONSWMM_performance_per_subanalysis()

        # Use unified consolidation pipeline (includes fail-fast validation)
        self.master_analysis.process._consolidate_outputs(
            ds_performance,
            mode=perf_mode,
            overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
            verbose=verbose,
            compression_level=compression_level,
        )

        return

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
        return df_setup

    def _create_sub_analyses(self):
        # create sub analyses
        dic_sensitivity_analyses = dict()
        for idx, row in self.df_setup.iterrows():
            cfg_snstvty_analysis = self.master_analysis.cfg_analysis.model_copy()

            for key, val in row.items():
                setattr(cfg_snstvty_analysis, key, val)  # type: ignore
            sa_id = f"{self.sub_analyses_prefix}{idx}"
            cfg_snstvty_analysis.analysis_id = sa_id  # type: ignore
            sub_analysis_directory = self.subanalysis_dir / str(
                cfg_snstvty_analysis.analysis_id
            )
            sub_analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False
            cfg_snstvty_analysis.is_subanalysis = True

            cfg_anlysys_yaml = sub_analysis_directory / f"{sa_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = sub_analysis_directory

            cfg_anlysys_yaml.write_text(
                yaml.safe_dump(
                    cfg_snstvty_analysis.model_dump(mode="json"),
                    sort_keys=False,  # .dict() for pydantic v1
                )
            )
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=self._system,
            )
            # Mark sub-analysis instances with parent sensitivity context so
            # status/allocation parsing can route to the master Snakefile.
            dic_sensitivity_analyses[idx] = anlsys
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

        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            assert (
                sub_analysis.cfg_analysis.is_subanalysis
            ), "is_subanalysis attribute not true in sub_analysis.cfg_analysis.is_subanalysis"
            sub_df_status = sub_analysis.df_status.copy()

            # Add sensitivity parameter columns for this sub-analysis row
            setup_row = self.df_setup.iloc[sub_analysis_iloc, :]
            for key, val in setup_row.items():
                sub_df_status[key] = val

            # Preserve existing naming convention while adding a singular alias
            sub_df_status["sub_analysis_iloc"] = sub_analysis_iloc

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
