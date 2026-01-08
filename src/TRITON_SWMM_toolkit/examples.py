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
import sys
import warnings
from typing import Iterable, Union

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
    load_experiment_config,
    experiment_config,
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
    def __init__(self, cfg_system_yaml: Path, cfg_exp_yaml: Path):
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.system.add_experiment(cfg_exp_yaml)


class TRITON_SWMM_testcase:
    """
    - everything is based on the single sim experiment so there is only 1 template yaml to keep up to date for testing
    - retrieves one or more events and subsets time series to simulate a short duration
    - writes a .csv file with event indices targeted for simulation
    - creates a version of the experiment config file with a revised experiment name field and pointing to the subset weather index csv
    """

    n_tsteps: int = 5

    # LOADING FROM SYSTEM CONFIG
    def __init__(
        self,
        cfg_system_yaml: Path,
        experiment_name: str,
        target_dates: Iterable[Union[str, pd.Timestamp]],
        experiment_description: str = "",
        test_system_dirname: str = "norfolk_tests",
        start_from_scratch: bool = False,
    ):
        # load system
        self.system = TRITONSWMM_system(cfg_system_yaml)
        # update system directory
        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / test_system_dirname
        )
        # load single sime experiment
        single_sim_exp_yaml = TRITON_SWMM_examples.load_norfolk_single_sim_experiment()
        exp = load_experiment_config(single_sim_exp_yaml)
        # update experiment attributes
        exp.experiment_id = experiment_name
        exp.experiment_description = experiment_description
        exp_dir = self.system.cfg_system.system_directory / experiment_name
        if start_from_scratch and exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        f_weather_indices = exp_dir / "weather_indices.csv"
        exp.weather_events_to_simulate = f_weather_indices
        # create weather indexer dataset
        write_subset_weather_events_to_simulate(
            weather_event_summaries=exp.weather_event_summary_csv,  # type: ignore
            target_dates=target_dates,
            start_date_col="event_start",
            end_date_col="event_end",
            f_out=f_weather_indices,
            weather_indexers=exp.weather_event_indices,
        )

        cfg_exp = experiment_config.model_validate(exp)
        # write experiment as yaml
        cfg_exp_yaml = exp_dir / f"{experiment_name}.yaml"
        cfg_exp_yaml.write_text(
            yaml.safe_dump(
                cfg_exp.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )

        # add experiment to the system
        self.system.add_experiment(cfg_exp_yaml)
        # udpate weather time series so simulations are short
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

        ds_event_weather_series = xr.open_dataset(og_weather_timeseries)

        ds_event_weather_series = ds_event_weather_series.isel(
            {weather_time_series_timestep_dimension_name: slice(0, self.n_tsteps)}
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
        # print(f"created weather netcdf {new_weather_timeseries}")
        return new_weather_timeseries


class TRITON_SWMM_examples:
    def __init__(self) -> None:
        pass

    @classmethod
    def retrieve_norfolk_irene_example(cls, download_if_exists=False):
        norfolk_system_yaml = load_norfolk_system_config(download_if_exists)
        norfolk_experiment_yaml = cls.load_norfolk_single_sim_experiment()
        norfolk_irene = TRITON_SWMM_example(
            norfolk_system_yaml, norfolk_experiment_yaml
        )
        return norfolk_irene

    @classmethod
    def load_norfolk_benchmarking_config(cls):
        return cls._load_example_experiment_config(
            NORFOLK_EX, NORFOLK_BENCHMARKING_EXP_CONFIG
        )

    @classmethod
    def load_norfolk_single_sim_experiment(cls):
        return cls._load_example_experiment_config(
            NORFOLK_EX, NORFOLK_SINGLE_SIM_EXP_CONFIG
        )

    @classmethod
    def _fill_experiment_yaml(cls, system_name: str, experiment_config_filename: str):
        mapping = get_norfolk_data_and_package_directory_mapping_dict()
        cfg_template = load_config_filepath(system_name, experiment_config_filename)
        filled_yaml_data = return_filled_template_yaml_dictionary(cfg_template, mapping)
        return filled_yaml_data

    @classmethod
    def _load_example_experiment_config(
        cls, system_name: str, experiment_config_filename: str
    ):
        filled_yaml_data = cls._fill_experiment_yaml(
            system_name, experiment_config_filename
        )
        cfg_system_yaml = load_norfolk_system_config(download_if_exists=False)
        cfg_system = load_system_config(cfg_system_yaml)
        experiment_id = filled_yaml_data["experiment_id"]
        cfg_yaml = (
            Path(cfg_system.system_directory)
            / f"config_experiment_{experiment_id}.yaml"
        )
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml


class TRITON_SWMM_testcases:
    test_system_dirname = "norfolk_tests"

    def __init__(self) -> None:
        pass

    @classmethod
    def _retrieve_norfolk_case(
        cls,
        experiment_name: str,
        target_dates: Iterable,
        start_from_scratch: bool,
        experiment_description: str = "",
        download_if_exists=False,
    ) -> TRITON_SWMM_testcase:
        norfolk_system_yaml = load_norfolk_system_config(download_if_exists)
        nrflk_test = TRITON_SWMM_testcase(
            norfolk_system_yaml,
            experiment_name=experiment_name,
            target_dates=target_dates,
            experiment_description=experiment_description,
            test_system_dirname=cls.test_system_dirname,
            start_from_scratch=start_from_scratch,
        )
        return nrflk_test

    @classmethod
    def retreive_norfolk_single_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        experiment_name = "single_sim"
        target_dates = ["2011-08-28"]
        experiment_description = "hurricane irene"
        return cls._retrieve_norfolk_case(
            experiment_name,
            target_dates,
            start_from_scratch,
            experiment_description,
            download_if_exists,
        )

    @classmethod
    def retreive_norfolk_multi_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        experiment_name = "multi_sim"
        target_dates = [
            "2011-08-28",
            "2015-07-11",
            "2015-10-04",
            # "2016-09-03",
            # "2016-09-21",
            # "2017-08-29",
        ]
        experiment_description = "events wtih 311 flood reports"
        return cls._retrieve_norfolk_case(
            experiment_name,
            target_dates,
            start_from_scratch,
            experiment_description,
            download_if_exists,
        )


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
    cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
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


def write_subset_weather_events_to_simulate(
    weather_event_summaries: Path,
    target_dates: Iterable[Union[str, pd.Timestamp]],
    start_date_col: str,
    end_date_col: str,
    f_out: Path,
    weather_indexers: Iterable[str],
):

    df = pd.read_csv(weather_event_summaries)
    df_subset = select_events_near_dates(
        df,
        start_date_col,
        end_date_col,
        target_dates,
    )
    df_subset.loc[:, weather_indexers].to_csv(f_out)  # type: ignore


def select_events_near_dates(
    df: pd.DataFrame,
    start_date_col: str,
    end_date_col: str,
    target_dates: Iterable[Union[str, pd.Timestamp]],
):
    """
    Select events matching or closest to target dates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    start_date_col : str
        Column name for event start datetime
    end_date_col : str
        Column name for event end datetime
    target_dates : array-like
        Iterable of datetime-like values

    Returns
    -------
    pandas.DataFrame
        Subset of selected events (one per target date)
    """

    df = df.copy()
    df[start_date_col] = pd.to_datetime(df[start_date_col])
    df[end_date_col] = pd.to_datetime(df[end_date_col])
    target_dates = pd.to_datetime(target_dates)  # type: ignore

    selected_events = []

    for date in target_dates:
        # 1) Events spanning the target date
        spanning = df[(df[start_date_col] <= date) & (df[end_date_col] >= date)]
        if len(spanning == 1):
            selected_events.append(spanning.iloc[0, :])
        elif len(spanning > 1):
            sys.exit("multiple events were returned")
        else:
            # 2) Otherwise warn and find closest by start and end
            print(f"WARNING: No event spans {date.date()}")  # type: ignore

            start_diffs = (df[start_date_col] - date).abs()  # type: ignore
            end_diffs = (df[end_date_col] - date).abs()  # type: ignore

            idx_start = start_diffs.idxmin()
            idx_end = end_diffs.idxmin()

            # 3) Enforce agreement
            if idx_start != idx_end:
                raise RuntimeError(
                    f"Closest events disagree for {date.date()}:\n"  # type: ignore
                    f"  Closest by {start_date_col}: index {idx_start}\n"
                    f"  Closest by {end_date_col}:   index {idx_end}"
                )

            # 4) Diagnostics
            event = df.loc[idx_start]
            print(
                f"  Selected event:\n"
                f"    {start_date_col} = {event[start_date_col]}\n"
                f"    {end_date_col}   = {event[end_date_col]}\n"
                f"    |date - {start_date_col}| = {abs(event[start_date_col] - date)}\n"  # type: ignore
                f"    |date - {end_date_col}|   = {abs(event[end_date_col] - date)}"  # type: ignore
            )

            selected_events.append(event)

    return pd.DataFrame(selected_events).reset_index(drop=True)
