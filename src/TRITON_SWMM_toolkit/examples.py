import xarray as xr
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.constants import (
    APP_NAME,
    NORFOLK_EX,
    NORFOLK_SYSTEM_CONFIG,
    NORFOLK_ANALYSIS_CONFIG,
    # NORFOLK_sensitivity_EXP_CONFIG,
    NORFOLK_CASE_CONFIG,
)
import sys
import warnings
from typing import Iterable, Union
from typing import Optional

with warnings.catch_warnings():
    # Only ignore the pkg_resources deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*pkg_resources is deprecated.*",
    )
    try:
        from hsclient import HydroShare
    except ImportError:
        HydroShare = None
from importlib.resources import files
import yaml
from zipfile import ZipFile
import bagit
import shutil

from TRITON_SWMM_toolkit.config import (
    load_system_config,
    load_system_config_from_dict,
    load_analysis_config,
    analysis_config,
)
from TRITON_SWMM_toolkit.utils import (
    get_package_root,
    get_package_data_root,
    fill_template,
    read_yaml,
    write_yaml,
)
import numpy as np
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis


#  define test case class
class TRITON_SWMM_example:
    # LOADING FROM SYSTEM CONFIG
    def __init__(self, cfg_system_yaml: Path, cfg_analysis_yaml: Path):
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.system.add_analysis(cfg_analysis_yaml)


class TRITON_SWMM_testcase:
    """
    - everything is based on the single sim analysis so there is only 1 template yaml to keep up to date for testing
    - retrieves one or more events and subsets time series to simulate a short duration
    - writes a .csv file with event indices targeted for simulation
    - creates a version of the analysis config file with a revised analysis name field and pointing to the subset weather index csv
    """

    n_tsteps: int = 5

    # LOADING FROM SYSTEM CONFIG
    def __init__(
        self,
        cfg_system_yaml: Path,
        analysis_name: str,
        n_events: int,
        n_reporting_tsteps_per_sim: int,
        TRITON_reporting_timestep_s: int,
        test_system_dirname: str,
        analysis_description: str = "",
        start_from_scratch: bool = False,
        additional_analysis_configs: dict = dict(),
        additional_system_configs: dict = dict(),
    ):
        # load system
        self.system = TRITONSWMM_system(cfg_system_yaml)

        for key, val in additional_system_configs.items():
            setattr(self.system.cfg_system, key, val)
        # update system directory
        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / test_system_dirname
        )
        anlysys_dir = self.system.cfg_system.system_directory / analysis_name

        if start_from_scratch and anlysys_dir.exists():
            shutil.rmtree(anlysys_dir)
        anlysys_dir.mkdir(parents=True, exist_ok=True)

        new_system_config_yaml = (
            self.system.cfg_system.system_directory / f"{test_system_dirname}.yaml"
        )

        new_system_config_yaml.write_text(
            yaml.safe_dump(
                self.system.cfg_system.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )

        self.system = TRITONSWMM_system(new_system_config_yaml)

        # load single sime analysis
        single_sim_anlysys_yaml = TRITON_SWMM_examples.load_norfolk_template_analysis()
        anlysys = load_analysis_config(single_sim_anlysys_yaml)
        # update analysis attributes
        anlysys.analysis_id = analysis_name
        anlysys.analysis_description = analysis_description
        f_weather_indices = anlysys_dir / "weather_indices.csv"
        anlysys.weather_events_to_simulate = f_weather_indices
        event_index_name = "event_id"
        anlysys.weather_event_indices = [event_index_name]
        anlysys.TRITON_reporting_timestep_s = TRITON_reporting_timestep_s
        # create weather indexer dataset
        df_weather_indices = pd.DataFrame({event_index_name: np.arange(n_events)})
        df_weather_indices.to_csv(f_weather_indices)
        # add additional fields
        for key, val in additional_analysis_configs.items():
            setattr(anlysys, key, val)

        f_weather_tseries = anlysys_dir / "weather_tseries.nc"
        anlysys.weather_timeseries = f_weather_tseries

        cfg_anlysys = analysis_config.model_validate(anlysys)
        # write analysis as yaml
        cfg_anlysys_yaml = anlysys_dir / f"{analysis_name}.yaml"
        cfg_anlysys_yaml.write_text(
            yaml.safe_dump(
                cfg_anlysys.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )

        self.system.add_analysis(cfg_anlysys_yaml)
        self.create_short_intense_weather_timeseries(
            f_weather_tseries, n_reporting_tsteps_per_sim, n_events, event_index_name
        )
        self.system.process_system_level_inputs(overwrite_if_exists=start_from_scratch)
        # write weather time series and update weather time series path

        # self.system.analysis.cfg_analysis.weather_timeseries = f_weather_tseries  # type: ignore
        # add analysis to the system

    # create weather time series dataset
    def create_short_intense_weather_timeseries(
        self,
        f_out,
        n_reporting_tsteps_per_sim,
        n_events,
        event_index_name,
        rain_intensity=50,
        storm_tide=3,
    ):
        wlevel_name = (
            self.system.analysis.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        tstep_coord_name = (
            self.system.analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        )
        df_raingage_mapping = pd.read_csv(self.system.cfg_system.subcatchment_raingage_mapping)  # type: ignore
        gage_colname = (
            self.system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        gages = df_raingage_mapping[gage_colname].unique()

        reporting_tstep_sec = (
            self.system.analysis.cfg_analysis.TRITON_reporting_timestep_s
        )

        timesteps = pd.date_range(
            start="2000-01-01",
            periods=n_reporting_tsteps_per_sim + 1,
            freq=f"{int(reporting_tstep_sec)}s",
        )
        columns = list(gages) + [wlevel_name]
        df_tseries = pd.DataFrame(index=timesteps, columns=columns)
        df_tseries.loc[:, wlevel_name] = storm_tide  # type: ignore
        df_tseries.loc[:, gages] = rain_intensity
        df_tseries.index.name = tstep_coord_name
        df_tseries.columns = df_tseries.columns.astype(str)
        lst_df = []
        for event_idx in np.arange(n_events):
            df = df_tseries.copy()
            df[event_index_name] = event_idx
            lst_df.append(df)
        df_tseries = pd.concat(lst_df)
        df_tseries = df_tseries.reset_index().set_index(
            [event_index_name, tstep_coord_name]
        )

        ds_weather_tseries = df_tseries.to_xarray()
        ds_weather_tseries.to_netcdf(f_out)
        return

    def _create_reduced_weather_file_for_testing_if_it_does_not_exist(self):
        og_weather_timeseries = self.system.analysis.cfg_analysis.weather_timeseries
        new_weather_timeseries = (
            self.system.cfg_system.system_directory / "weather_subset.nc"
        )
        # weather_events_to_simulate = self.system.analysis.cfg_analysis.weather_events_to_simulate
        # weather_event_indices = self.system.analysis.cfg_analysis.weather_event_indices
        weather_time_series_timestep_dimension_name = (
            self.system.analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        )

        ds_event_weather_series = xr.open_dataset(
            og_weather_timeseries, engine="h5netcdf"
        )

        ds_event_weather_series = ds_event_weather_series.isel(
            {weather_time_series_timestep_dimension_name: slice(0, self.n_tsteps)}
        )

        tsteps_new = ds_event_weather_series[
            weather_time_series_timestep_dimension_name
        ].to_series()

        new_weather_timeseries.parent.mkdir(parents=True, exist_ok=True)

        if new_weather_timeseries.exists():
            with xr.open_dataset(
                new_weather_timeseries, engine="h5netcdf"
            ) as ds_existing:
                tsteps_existing = ds_existing[
                    weather_time_series_timestep_dimension_name
                ].to_series()
            if len(tsteps_new) == len(tsteps_existing):
                if (
                    tsteps_new == tsteps_existing
                ).all():  # don't rewrite if it already matches
                    return new_weather_timeseries
            else:  # if they are not identical, remove the file and rerewrite
                new_weather_timeseries.unlink()
        ds_event_weather_series.to_netcdf(new_weather_timeseries)
        # print(f"created weather netcdf {new_weather_timeseries}")
        return new_weather_timeseries


class TRITON_SWMM_examples:
    def __init__(self) -> None:
        pass

    @classmethod
    def retrieve_norfolk_irene_example(
        cls,
        download_if_exists=False,
        example_data_dir: Optional[Path] = None,
    ):
        norfolk_system_yaml = load_norfolk_system_config(
            download_if_exists=download_if_exists, example_data_dir=example_data_dir
        )
        norfolk_analysis_yaml = cls.load_norfolk_template_analysis()
        norfolk_irene = TRITON_SWMM_example(norfolk_system_yaml, norfolk_analysis_yaml)
        return norfolk_irene

    # @classmethod
    # def load_norfolk_sensitivity_config(cls):
    #     return cls._load_example_analysis_config(
    #         NORFOLK_EX, NORFOLK_sensitivity_EXP_CONFIG
    # )

    @classmethod
    def load_norfolk_template_analysis(cls):
        return cls._load_example_analysis_config(NORFOLK_EX, NORFOLK_ANALYSIS_CONFIG)

    @classmethod
    def _fill_analysis_yaml(
        cls,
        system_name: str,
        analysis_config_filename: str,
        example_data_dir: Optional[Path] = None,
    ):
        mapping = get_norfolk_data_and_package_directory_mapping_dict(
            example_data_dir=example_data_dir
        )
        cfg_template = load_config_filepath(system_name, analysis_config_filename)
        filled_yaml_data = return_filled_template_yaml_dictionary(cfg_template, mapping)
        return filled_yaml_data

    @classmethod
    def _load_example_analysis_config(
        cls,
        system_name: str,
        analysis_config_filename: str,
        example_data_dir: Optional[Path] = None,
    ):
        filled_yaml_data = cls._fill_analysis_yaml(
            system_name=system_name,
            analysis_config_filename=analysis_config_filename,
            example_data_dir=example_data_dir,
        )
        cfg_system_yaml = load_norfolk_system_config(
            download_if_exists=False, example_data_dir=example_data_dir
        )
        cfg_system = load_system_config(cfg_system_yaml)
        analysis_id = filled_yaml_data["analysis_id"]
        cfg_yaml = (
            Path(cfg_system.system_directory) / f"config_analysis_{analysis_id}.yaml"
        )
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml


class GetTS_TestCases:
    test_system_dirname = "tests"
    n_reporting_tsteps_per_sim = 12
    TRITON_reporting_timestep_s = 10
    test_data_dir = files(APP_NAME).parents[1].joinpath(f"test_data/{NORFOLK_EX}/")  # type: ignore
    cpu_sensitivity = test_data_dir / "cpu_benchmarking_analysis.xlsx"
    # hpc_bash_script_ensemble_template = test_data_dir / "ensemble_template.sh"
    sensitivity_frontier_all_configs = test_data_dir / "benchmarking_frontier.xlsx"
    sensitivity_UVA_cpu = test_data_dir / "benchmarking_uva_cpus.xlsx"
    frontier_compilation_script = (
        test_data_dir / "template_compile_triton_swmm_frontier.sh"
    )
    frontier_modules_to_load_for_srun = "PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a"  # additional_modules_needed_to_run_TRITON_SWMM_on_hpc
    # UVA
    UVA_compilation_script = test_data_dir / "template_compile_triton_swmm_UVA.sh"
    UVA_modules_to_load_for_srun = "gompi/14.2.0_5.0.7 miniforge"
    UVA_norfolk_data_dir = Path("/scratch/***REMOVED***/triton_swmm_toolkit_data")

    def __init__(self) -> None:
        pass

    @classmethod
    def _retrieve_norfolk_case(
        cls,
        analysis_name: str,
        n_events: int,
        n_reporting_tsteps_per_sim: int,
        TRITON_reporting_timestep_s: int,
        start_from_scratch: bool,
        download_if_exists=False,
        additional_analysis_configs=dict(),
        additional_system_configs=dict(),
        example_data_dir: Optional[Path] = None,
    ) -> TRITON_SWMM_testcase:
        norfolk_system_yaml = load_norfolk_system_config(
            download_if_exists, example_data_dir=example_data_dir
        )
        nrflk_test = TRITON_SWMM_testcase(
            norfolk_system_yaml,
            analysis_name=analysis_name,
            n_events=n_events,
            n_reporting_tsteps_per_sim=n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=TRITON_reporting_timestep_s,
            test_system_dirname=cls.test_system_dirname,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs=additional_analysis_configs,
            additional_system_configs=additional_system_configs,
        )
        return nrflk_test

    @classmethod
    def retreive_norfolk_UVA_multisim_1cpu_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "UVA_multisim"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=8,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
            additional_analysis_configs=dict(
                TRITON_SWMM_make_command="hpc_swmm_omp",
                hpc_time_min_per_sim=30,
                hpc_partition="standard",
                hpc_allocation="***REMOVED***",
                run_mode="serial",
                n_mpi_procs=1,
                n_omp_threads=1,
                n_gpus=0,
                n_nodes=1,
                multi_sim_run_method="batch_job",
                python_path="/home/***REMOVED***/.conda/envs/triton_swmm_toolkit/bin/python",
                additional_bash_lines=[
                    'eval "$(conda shell.bash hook)"',
                    "conda activate triton_swmm_toolkit",
                    "export PYTHONNOUSERSITE=1",
                    # "pip install -e .",
                ],
            ),
            additional_system_configs=dict(
                TRITON_SWMM_software_compilation_script=cls.UVA_compilation_script,
                additional_modules_needed_to_run_TRITON_SWMM_on_hpc=cls.UVA_modules_to_load_for_srun,
            ),
            example_data_dir=cls.UVA_norfolk_data_dir,
        )

    @classmethod
    def retreive_norfolk_UVA_sensitivtiy_CPU(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "UVA_sensitivtiy_CPU"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
            additional_analysis_configs=dict(
                toggle_sensitivity_analysis=True,
                sensitivity_analysis=cls.sensitivity_UVA_cpu,
                TRITON_SWMM_make_command="hpc_swmm_omp",
                hpc_time_min_per_sim=30,
                hpc_partition="standard",
                hpc_allocation="***REMOVED***",
                run_mode="serial",
                n_mpi_procs=1,
                n_omp_threads=1,
                n_gpus=0,
                n_nodes=1,
                multi_sim_run_method="batch_job",
                python_path="/home/***REMOVED***/.conda/envs/triton_swmm_toolkit/bin/python",
                additional_bash_lines=[
                    'eval "$(conda shell.bash hook)"',
                    "conda activate triton_swmm_toolkit",
                    "export PYTHONNOUSERSITE=1",
                    # "pip install -e .",
                ],
            ),
            additional_system_configs=dict(
                TRITON_SWMM_software_compilation_script=cls.UVA_compilation_script,
                additional_modules_needed_to_run_TRITON_SWMM_on_hpc=cls.UVA_modules_to_load_for_srun,
            ),
            example_data_dir=cls.UVA_norfolk_data_dir,
        )

    @classmethod
    def retreive_norfolk_frontier_multisim_cpu_serial_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "frontier_multisim"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=128,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
            additional_analysis_configs=dict(
                TRITON_SWMM_make_command="frontier_swmm_omp",
                multi_sim_run_method="1_job_many_srun_tasks",
            ),
            additional_system_configs=dict(
                TRITON_SWMM_software_compilation_script=cls.frontier_compilation_script,
                additional_modules_needed_to_run_TRITON_SWMM_on_hpc=cls.frontier_modules_to_load_for_srun,
            ),
        )

    @classmethod
    def retreive_norfolk_frontier_all_configs(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "frontier_all_configs_sensitivity"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
            additional_analysis_configs=dict(
                TRITON_SWMM_make_command="frontier_swmm_omp",
                toggle_sensitivity_analysis=True,
                sensitivity_analysis=cls.sensitivity_frontier_all_configs,
                multi_sim_run_method="1_job_many_srun_tasks",
            ),
            additional_system_configs=dict(
                TRITON_SWMM_software_compilation_script=cls.frontier_compilation_script,
                additional_modules_needed_to_run_TRITON_SWMM_on_hpc=cls.frontier_modules_to_load_for_srun,
            ),
        )

    # @classmethod
    # def retreive_norfolk_hcp_cpu_sensitivity_case(
    #     cls, start_from_scratch: bool = False, download_if_exists: bool = False
    # ):
    #     analysis_name = "cpu_config_sensitivity"
    #     return cls._retrieve_norfolk_case(
    #         analysis_name=analysis_name,
    #         start_from_scratch=start_from_scratch,
    #         download_if_exists=download_if_exists,
    #         n_events=1,
    #         n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
    #         TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
    #         additional_analysis_configs=dict(
    #             toggle_sensitivity_analysis=True,
    #             toggle_run_ensemble_with_bash_script=True,
    #             hpc_bash_script_ensemble_template=cls.hpc_bash_script_ensemble_template,
    #             sensitivity_analysis=cls.cpu_sensitivity,
    #             hpc_allocation="***REMOVED***",
    #             hpc_time_min=120,
    #             hpc_partition="batch",
    #             hpc_n_nodes=1,
    #         ),
    #     )

    @classmethod
    def retreive_norfolk_cpu_config_sensitivity_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "cpu_config_sensitivity"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
            additional_analysis_configs=dict(
                toggle_sensitivity_analysis=True,
                sensitivity_analysis=cls.cpu_sensitivity,
            ),
        )

    @classmethod
    def retreive_norfolk_single_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "single_sim"
        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
        )

    @classmethod
    def retreive_norfolk_multi_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_name = "multi_sim"

        return cls._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=2,
            n_reporting_tsteps_per_sim=cls.n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=cls.TRITON_reporting_timestep_s,
        )


def load_config_filepath(case_study_name: str, filename: str) -> Path:
    return files(APP_NAME).parents[1].joinpath(f"test_data/{case_study_name}/{filename}")  # type: ignore


def load_config_file_as_dic(case_study_name: str, filename: str) -> dict:
    path = load_config_filepath(case_study_name, filename)
    return read_yaml(path)


def download_data_from_hydroshare(
    res_identifier: str,
    target: Path,
    hs,
    download_if_exists=False,
    validate=False,
):
    if target.exists() and download_if_exists:
        shutil.rmtree(target)
    if target.exists() and not download_if_exists:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    hs_resource = hs.resource(res_identifier)
    zip_path = Path(hs_resource.download(target.parent))
    extract_to = target.parent

    with ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    # Detect the actual top-level folder
    with ZipFile(zip_path, "r") as z:
        top_level_dirs = {Path(f).parts[0] for f in z.namelist() if Path(f).parts}
        if len(top_level_dirs) == 1:
            unzipped_folder = extract_to / next(iter(top_level_dirs))
        else:
            raise RuntimeError(
                "ZIP has multiple top-level folders; cannot determine Bag root."
            )
    # unzipped_folder = Path(extract_to).joinpath(zip_path.name.split(".")[0])
    if validate:
        bag = bagit.Bag(unzipped_folder)
        if bag.is_valid():
            print("Bag verified! All checksums match.")
        else:
            print("Bag is invalid!")

    outdir = unzipped_folder.rename(target)
    zip_path.unlink()


def sign_into_hydroshare():
    if HydroShare is None:
        raise RuntimeError(
            "hsclient is not installed. Install optional dependencies with `pip install .[tests]`. Alternatively, you can download the data manually if you have issues installing this package with pip. Link: https://www.hydroshare.org/resource/a4aace329b8c401a93e94ce2a761fe1b/"
        )
    hs = HydroShare()
    print("Please log into hydroshare to download example.", flush=True)
    hs.sign_in()
    print("signed into Hydroshare successfully.")
    return hs


def return_filled_template_yaml_dictionary(cfg_template: Path, mapping: dict):
    cfg_filled = fill_template(cfg_template, mapping)
    try:
        cfg_filled_yaml = yaml.safe_load(cfg_filled)
    except:
        print(cfg_filled)
        sys.exit("failed to load yaml")
    return cfg_filled_yaml


def get_norfolk_data_and_package_directory_mapping_dict(
    example_data_dir: Optional[Path] = None,
):
    if example_data_dir:
        root = example_data_dir
    else:
        root = get_package_data_root(APP_NAME)
    hydroshare_root_dir = root / "examples" / NORFOLK_EX
    data_dir = hydroshare_root_dir / "data" / "contents"
    package_dir = get_package_root(APP_NAME).parents[1] / "test_data" / NORFOLK_EX

    mapping = dict(
        DATA_DIR=str(data_dir),
        PACKAGE_DIR=str(package_dir),
        HYDROSHARE_ROOT=str(hydroshare_root_dir),
    )
    return mapping


def load_norfolk_system_config(
    download_if_exists: bool, example_data_dir: Optional[Path] = None
):
    case_details = load_config_file_as_dic(NORFOLK_EX, NORFOLK_CASE_CONFIG)
    res_identifier = case_details["res_identifier"]  # will come from the case yaml
    mapping = get_norfolk_data_and_package_directory_mapping_dict(
        example_data_dir=example_data_dir
    )
    cfg_template = load_config_filepath(NORFOLK_EX, NORFOLK_SYSTEM_CONFIG)
    filled_yaml_data = return_filled_template_yaml_dictionary(cfg_template, mapping)
    cfg_system = load_system_config_from_dict(filled_yaml_data)
    # download data if it doesn't exist
    if Path(mapping["DATA_DIR"]).exists() and not download_if_exists:
        pass
    else:
        hs = sign_into_hydroshare()
        download_data_from_hydroshare(
            res_identifier,
            Path(mapping["HYDROSHARE_ROOT"]),
            hs,
            download_if_exists=download_if_exists,
        )
        # zipped_software = Path(mapping["DATA_DIR"]) / "triton_swmm.zip"
        # with ZipFile(zipped_software, "r") as z:
        #     z.extractall(zipped_software.parent)
        # zipped_software.unlink()
    cfg_yaml = Path(filled_yaml_data["system_directory"]) / "config_system.yaml"
    cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(filled_yaml_data, cfg_yaml)
    return cfg_yaml
