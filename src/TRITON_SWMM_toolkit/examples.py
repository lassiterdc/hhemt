# %%
import xarray as xr
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.constants import (
    APP_NAME,
    NORFOLK_EX,
    NORFOLK_SYSTEM_CONFIG,
    NORFOLK_SINGLE_SIM_EXP_CONFIG,
    NORFOLK_BENCHMARKING_EXP_CONFIG,
    NORFOLK_CASE_CONFIG,
)
import warnings

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
)
from TRITON_SWMM_toolkit.utils import (
    get_package_root,
    get_package_data_root,
    fill_template,
    read_yaml,
    write_yaml,
)

from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.experiment import TRITONSWMM_experiment


#  define test case class
class TRITON_SWMM_example:
    # LOADING FROM SYSTEM CONFIG
    def __init__(self, cfg_system_yaml: Path, cfg_exp_1sim_yaml: Path):
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.system.add_experiment(cfg_exp_1sim_yaml)
        # self.system.experiment = TRITONSWMM_experiment(cfg_exp_1sim_yaml)


class TRITON_SWMM_testcase:
    dur_min: int = 10
    event_iloc_for_subsetting = 0

    # LOADING FROM SYSTEM CONFIG
    def __init__(
        self, cfg_system_yaml: Path, cfg_exp_1sim_yaml: Path, test_dirname: str
    ):
        self.system = TRITONSWMM_system(cfg_system_yaml)

        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / test_dirname
        )
        self.system.add_experiment(cfg_exp_1sim_yaml)
        new_weather_timeseries = (
            self._create_reduced_weather_file_for_testing_if_it_does_not_exist()
        )
        self.system.experiment.cfg_exp.weather_timeseries = new_weather_timeseries  # type: ignore

    def _create_reduced_weather_file_for_testing_if_it_does_not_exist(self):
        og_weather_timeseries = self.system.experiment.cfg_exp.weather_timeseries
        new_weather_timeseries = (
            self.system.cfg_system.system_directory / "weather_subset.nc"
        )
        # weather_events_to_simulate = self.system.experiment.cfg_exp.weather_events_to_simulate
        # weather_event_indices = self.system.experiment.cfg_exp.weather_event_indices
        weather_time_series_timestep_dimension_name = (
            self.system.experiment.cfg_exp.weather_time_series_timestep_dimension_name
        )
        dur_min = self.dur_min

        weather_event_indexers = (
            self.system.experiment._retrieve_weather_indexer_using_integer_index(
                self.event_iloc_for_subsetting
            )
        )

        ds_event_weather_series = xr.open_dataset(og_weather_timeseries)
        ds_event_ts = ds_event_weather_series.sel(weather_event_indexers)

        peak_idx = ds_event_ts["mm_per_hr"].to_series().dropna().idxmax()
        # compute 6 min window around peak rainfall
        first_idx = peak_idx - pd.Timedelta(f"{dur_min/2} minutes")  # type: ignore
        last_idx = first_idx + pd.Timedelta(f"{dur_min} minutes")

        ds_event_weather_series = ds_event_weather_series.sel(
            {weather_time_series_timestep_dimension_name: slice(first_idx, last_idx)}
        )

        tsteps_new = ds_event_weather_series[
            weather_time_series_timestep_dimension_name
        ].to_series()

        new_weather_timeseries.parent.mkdir(parents=True, exist_ok=True)

        if new_weather_timeseries.exists():
            with xr.open_dataset(new_weather_timeseries) as ds_existing:
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
        print(f"created weather netcdf {new_weather_timeseries}")
        return new_weather_timeseries


def retrieve_norfolk_irene_example(download_if_exists=False):
    norfolk_system_yaml = load_norfolk_system_config(download_if_exists)
    norfolk_1sim_1core_experiment_yaml = load_norfolk_single_sim_experiment()
    norfolk_irene = TRITON_SWMM_example(
        norfolk_system_yaml, norfolk_1sim_1core_experiment_yaml
    )
    return norfolk_irene


def retrieve_norfolk_testcase(download_if_exists=False):
    norfolk_system_yaml = load_norfolk_system_config(download_if_exists)
    norfolk_1sim_1core_experiment_yaml = load_norfolk_single_sim_experiment()
    nrflk_test = TRITON_SWMM_testcase(
        norfolk_system_yaml,
        norfolk_1sim_1core_experiment_yaml,
        test_dirname="_test_norfolk",
    )
    return nrflk_test


def load_config_filepath(case_study_name: str, filename: str) -> Path:
    return files(APP_NAME).joinpath(f"examples/{case_study_name}/{filename}")  # type: ignore


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
    hs.sign_in()
    print("signed into Hydroshare successfully.")
    return hs


import sys


def return_filled_template_yaml_dictionary(cfg_template: Path, mapping: dict):
    cfg_filled = fill_template(cfg_template, mapping)
    try:
        cfg_filled_yaml = yaml.safe_load(cfg_filled)
    except:
        print(cfg_filled)
        sys.exit("failed to load yaml")
    return cfg_filled_yaml


def get_norfolk_data_and_package_directory_mapping_dict():
    hydroshare_root_dir = get_package_data_root(APP_NAME) / "examples" / NORFOLK_EX
    data_dir = hydroshare_root_dir / "data" / "contents"
    package_dir = get_package_root(APP_NAME) / "examples" / NORFOLK_EX
    mapping = dict(
        DATA_DIR=str(data_dir),
        PACKAGE_DIR=str(package_dir),
        HYDROSHARE_ROOT=str(hydroshare_root_dir),
    )
    return mapping


def load_norfolk_system_config(
    download_if_exists,
):
    case_details = load_config_file_as_dic(NORFOLK_EX, NORFOLK_CASE_CONFIG)
    res_identifier = case_details["res_identifier"]  # will come from the case yaml
    mapping = get_norfolk_data_and_package_directory_mapping_dict()
    cfg_template = load_config_filepath(NORFOLK_EX, NORFOLK_SYSTEM_CONFIG)
    filled_yaml_data = return_filled_template_yaml_dictionary(cfg_template, mapping)
    cfg_yaml = Path(filled_yaml_data["system_directory"]) / "config_system.yaml"
    write_yaml(filled_yaml_data, cfg_yaml)
    cfg_system = load_system_config(cfg_yaml)
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
        zipped_software = Path(str(cfg_system.TRITONSWMM_software_directory) + ".zip")
        with ZipFile(zipped_software, "r") as z:
            z.extractall(cfg_system.TRITONSWMM_software_directory.parent)
        zipped_software.unlink()
    return cfg_yaml


def load_example_experiment_config(system_name: str, experiment_config_filename: str):
    cfg_system_yaml = load_norfolk_system_config(download_if_exists=False)
    cfg_system = load_system_config(cfg_system_yaml)
    mapping = get_norfolk_data_and_package_directory_mapping_dict()
    cfg_template = load_config_filepath(system_name, experiment_config_filename)
    filled_yaml_data = return_filled_template_yaml_dictionary(cfg_template, mapping)
    experiment_id = filled_yaml_data["experiment_id"]
    cfg_yaml = (
        Path(cfg_system.system_directory) / f"config_experiment_{experiment_id}.yaml"
    )
    write_yaml(filled_yaml_data, cfg_yaml)
    return cfg_yaml


def load_norfolk_benchmarking_config():
    return load_example_experiment_config(NORFOLK_EX, NORFOLK_BENCHMARKING_EXP_CONFIG)


def load_norfolk_single_sim_experiment():
    return load_example_experiment_config(NORFOLK_EX, NORFOLK_SINGLE_SIM_EXP_CONFIG)
