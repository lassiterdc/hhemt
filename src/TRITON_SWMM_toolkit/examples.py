# %%
import requests
from pathlib import Path
from TRITON_SWMM_toolkit.constants import (
    APP_NAME,
    NORFOLK_EX,
    NORFOLK_SYSTEM_CONFIG,
    NORFOLK_SINGLE_SIM_EXP_CONFIG,
    NORFOLK_BENCHMARKING_EXP_CONFIG,
    NORFOLK_CASE_CONFIG,
)

from importlib.resources import files
import yaml
from zipfile import ZipFile
import bagit
import shutil
from TRITON_SWMM_toolkit.config import (
    load_system_config,
    load_experiment_config,
)
from TRITON_SWMM_toolkit.utils import (
    get_package_root,
    get_package_data_root,
    create_from_template,
    fill_template,
)

try:
    from hsclient import HydroShare
except ImportError:
    HydroShare = None


def load_config_filepath(case_study_name: str, filename: str) -> Path:
    return files(APP_NAME).joinpath(f"examples/{case_study_name}/{filename}")  # type: ignore


def load_config_file_as_dic(case_study_name: str, filename: str) -> dict:
    path = load_config_filepath(case_study_name, filename)
    return yaml.safe_load(path.read_text())


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
    cfg_filled_yaml = return_filled_template_yaml_dictionary(cfg_template, mapping)
    model = load_system_config(cfg_filled_yaml)
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
        zipped_software = Path(str(model.TRITONSWMM_software_directory) + ".zip")
        with ZipFile(zipped_software, "r") as z:
            z.extractall(model.TRITONSWMM_software_directory.parent)
        zipped_software.unlink()
    return model


def load_example_experiment_config(system_name: str, experiment_config_filename: str):
    mapping = get_norfolk_data_and_package_directory_mapping_dict()
    cfg_template = load_config_filepath(system_name, experiment_config_filename)
    cfg_filled_yaml = return_filled_template_yaml_dictionary(cfg_template, mapping)
    return load_experiment_config(cfg_filled_yaml)


def load_norfolk_benchmarking_config():
    return load_example_experiment_config(NORFOLK_EX, NORFOLK_BENCHMARKING_EXP_CONFIG)


def load_norfolk_single_sim_experiment():
    return load_example_experiment_config(NORFOLK_EX, NORFOLK_SINGLE_SIM_EXP_CONFIG)
