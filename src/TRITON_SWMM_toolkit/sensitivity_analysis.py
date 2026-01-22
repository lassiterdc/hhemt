# %%
import subprocess
import shutil
import TRITON_SWMM_toolkit.utils as ut
from pathlib import Path
from TRITON_SWMM_toolkit.config import analysis_config
import pandas as pd
from typing import Literal

# from TRITON_SWMM_toolkit.paths import SensitivityAnalysisPaths
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing

from TRITON_SWMM_toolkit.processing_analysis import TRITONSWMM_analysis_post_processing
from TRITON_SWMM_toolkit.constants import Mode
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log
from TRITON_SWMM_toolkit.plot_analysis import TRITONSWMM_analysis_plotting
import yaml
from pprint import pprint
import json
import TRITON_SWMM_toolkit.analysis as anlysis
import xarray as xr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_sensitivity_analysis:
    """
    Docstring for TRITONSWMM_sensitivity_analysis
    - Creates subanalyses for each sensitivity analysis table row
    - Consolidates results at the 'master_analysis' level
    """

    def __init__(
        self,
        analysis: "TRITONSWMM_analysis",
    ) -> None:
        self.master_analysis = analysis
        self._system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        self.independent_vars = self._attributes_varied_for_analysis()
        self.df_setup = self._retieve_df_setup().loc[:, self.independent_vars]  # type: ignore
        self.sub_analyses = self._create_sub_analyses()

    def prepare_scenarios_in_each_subanalysis(
        self,
        overwrite_scenarios: bool = False,
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
                    overwrite_scenario=overwrite_scenarios,
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

            if self.all_scenarios_created != True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(
                    f"Preparation failed for the following scenarios:\n{scens_not_created}"
                )
            self._update_master_analysis_log()
        elif self.master_analysis.cfg_analysis.multi_sim_run_method in ["batch_job"]:
            raise ValueError(
                "prepare scenarios is not currently executable as batch_job."
            )

    def run_sensitivity_analysis_as_batch_job(
        self,
        # setup stuff
        process_system_level_inputs: bool = True,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        # ensemble run stuff
        prepare_scenarios: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        # other
        verbose: bool = True,
    ):
        if "TRITON_SWMM_make_command" in self.df_setup.columns:
            raise ValueError(
                "Currently sensitivity analysis run as batch jobs can only use 1 compiled TRITON-SWMM version. "
                "If CPU and GPU sensitivity analysis needs to be conducted, they should be submitted as "
                "separate analyses."
            )
        subanalysis_consolidation_jobs = []
        setup_script = self.master_analysis.generate_setup_workflow_script(
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            verbose=verbose,
        )
        setup_job_id = ut.run_bash_script(setup_script, verbose=verbose)
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            ensemble_script = sub_analysis.generate_SLURM_job_array_script(
                prepare_scenarios=prepare_scenarios,
                process_timeseries=process_timeseries,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                overwrite_scenario=overwrite_scenario,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                verbose=verbose,
            )
            ensemble_jobid = ut.run_bash_script(
                ensemble_script, dependent_job_id=setup_job_id, verbose=verbose
            )
            subanalysis_consolidation_script = (
                sub_analysis.generate_consolidation_workflow_script(
                    overwrite_if_exist=overwrite_if_exist,
                    compression_level=compression_level,
                    verbose=verbose,
                    which=which,
                )
            )
            subanalysis_consolidation_jobid = ut.run_bash_script(
                subanalysis_consolidation_script,
                dependent_job_id=ensemble_jobid,
                verbose=verbose,
            )
            subanalysis_consolidation_jobs.append(subanalysis_consolidation_jobid)

        # Generate master consolidation script to consolidate subanalysis outputs
        master_consolidation_script = (
            self.master_analysis.generate_consolidation_workflow_script(
                overwrite_if_exist=overwrite_if_exist,
                compression_level=compression_level,
                verbose=verbose,
                which=which,
                consolidate_sensitivity_analysis_outputs=True,
            )
        )

        # Submit master consolidation job with dependencies on all subanalysis consolidation jobs
        master_consolidation_jobid = ut.run_bash_script(
            master_consolidation_script,
            dependent_job_id=subanalysis_consolidation_jobs,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Sensitivity analysis batch job workflow submitted successfully",
                flush=True,
            )
            print(
                f"Master consolidation job ID: {master_consolidation_jobid}", flush=True
            )
        return master_consolidation_jobid

    def run_all_sims(
        self,
        pickup_where_leftoff,
        concurrent: bool = False,
        process_outputs_after_sim_completion: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
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
                    overwrite_if_exist=overwrite_if_exist,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_master_analysis_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        scenario_timeseries_processing_launchers = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            launchers = sub_analysis.retreive_scenario_timeseries_processing_launchers(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.master_analysis.run_python_functions_concurrently(
            scenario_timeseries_processing_launchers
        )
        return

    def _combine_TRITON_outputs_per_subanalysis(self):
        assert self.TRITON_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = sub_analysis.TRITON_summary
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
            ds = sub_analysis.SWMM_node_summary
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
            ds = sub_analysis.SWMM_link_summary
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_link_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_link_outputs

    def _consolidate_outputs_in_each_subanalysis(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            sub_analysis.consolidate_analysis_outptus(
                which=which,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_master_analysis_log()
        return

    @property
    def TRITON_subanalyses_outputs_consolidated(self):
        success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            success = success and sub_analysis.TRITON_analysis_summary_created
        return success

    @property
    def SWMM_subanalyses_outputs_consolidated(self):
        node_success = True
        link_success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            node_success = (
                node_success and sub_analysis.SWMM_node_analysis_summary_created
            )
            link_success = (
                link_success and sub_analysis.SWMM_link_analysis_summary_created
            )
        return node_success and link_success

    def consolidate_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        self.create_subanalysis_summaries(
            which=which,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_subanalysis_outputs(
            which=which,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_subanalysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            ds_combined_outputs = self._combine_TRITON_outputs_per_subanalysis()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="TRITON",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            ds_combined_outputs = self._combine_SWMM_node_outputs_per_subanalysis()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="SWMM_node",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
            ds_combined_outputs = (
                self._combine_SWMM_link_outputs_outputs_per_subanalysis()
            )
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="SWMM_link",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    def create_subanalysis_summaries(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="TRITON",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="SWMM",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    @property
    def SWMM_node_summary(self):
        return self.master_analysis.SWMM_node_summary

    @property
    def SWMM_link_summary(self):
        return self.master_analysis.SWMM_link_summary

    @property
    def TRITON_summary(self):
        return self.master_analysis.TRITON_summary

    @property
    def TRITONSWMM_runtimes(self):
        return self.master_analysis.TRITONSWMM_runtimes

    def _attributes_varied_for_analysis(self):
        df_setup = self._retieve_df_setup()
        keys_targeted_for_sensitivity = []
        for key, val in self.master_analysis.cfg_analysis.model_dump().items():
            # print(key)
            if key in df_setup.columns:
                if len(df_setup[key].unique()) > 1:
                    keys_targeted_for_sensitivity.append(key)
        return keys_targeted_for_sensitivity

    def _retieve_df_setup(self) -> pd.DataFrame:
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
            cfg_snstvty_analysis.analysis_id = f"subanalysis_{idx}"  # type: ignore
            sub_analysis_directory = (
                self.master_analysis.analysis_paths.analysis_dir
                / str(cfg_snstvty_analysis.analysis_id)
            )
            sub_analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False

            cfg_anlysys_yaml = sub_analysis_directory / f"subanalysis_{idx}.yaml"

            cfg_snstvty_analysis.analysis_dir = sub_analysis_directory

            if "TRITON_SWMM_make_command" in self.df_setup.columns:
                raise ValueError("Benchmarking on GPU and CPU configs is not possible.")

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
                if scen.log.scenario_creation_complete.get() != True:
                    scenarios_not_created.append(str(scen.log.logfile.parent))
        return scenarios_not_created

    @property
    def scenarios_not_run(self):
        scens_not_run = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                if scen.sim_run_completed != True:
                    scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    @property
    def all_scenarios_created(self):
        all_scenarios_created = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_scenarios_created = (
                all_scenarios_created and sub_analysis.log.all_scenarios_created.get()
            )
        return all_scenarios_created == True

    @property
    def all_sims_run(self):
        all_sims_run = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_sims_run = all_sims_run and sub_analysis.log.all_sims_run.get()
        return all_sims_run == True

    @property
    def all_TRITON_timeseries_processed(self):
        all_TRITON_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_TRITON_timeseries_processed = (
                all_TRITON_timeseries_processed
                and sub_analysis.log.all_TRITON_timeseries_processed.get()
            )
        return all_TRITON_timeseries_processed == True

    @property
    def all_SWMM_timeseries_processed(self):
        all_SWMM_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_SWMM_timeseries_processed = (
                all_SWMM_timeseries_processed
                and sub_analysis.log.all_SWMM_timeseries_processed.get()
            )
        return all_SWMM_timeseries_processed == True

    @property
    def TRITON_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.TRITON_time_series_not_processed()
        return lst_scens

    @property
    def SWMM_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.SWMM_time_series_not_processed()
        return lst_scens

    @property
    def all_raw_TRITON_outputs_cleared(self):
        all_raw_TRITON_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_TRITON_outputs_cleared = (
                all_raw_TRITON_outputs_cleared
                and sub_analysis.log.all_raw_TRITON_outputs_cleared.get()
            )
        return all_raw_TRITON_outputs_cleared == True

    @property
    def all_raw_SWMM_outputs_cleared(self):
        all_raw_SWMM_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_SWMM_outputs_cleared = (
                all_raw_SWMM_outputs_cleared
                and sub_analysis.log.all_raw_SWMM_outputs_cleared.get()
            )
        return all_raw_SWMM_outputs_cleared == True

    def _update_master_analysis_log(self):
        self.master_analysis._update_log()
        return
