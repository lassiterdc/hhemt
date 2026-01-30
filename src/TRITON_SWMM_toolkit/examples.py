"""
Production examples for TRITON-SWMM toolkit.

This module provides utilities for loading and running production examples.
Examples are organized as case studies (e.g., Norfolk coastal flooding) with
associated HydroShare data and configuration templates.

Generic API (for any case study):
    from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example
    from TRITON_SWMM_toolkit.constants import *

    # Load any case study
    example = TRITON_SWMM_example.from_case_study(
        app_name=APP_NAME,
        case_name="norfolk_coastal_flooding",
        system_config_template="template_system_config.yaml",
        analysis_config_template="template_analysis_config.yaml",
        case_config_filename="case.yaml",
    )
    system = example.system

Case-Specific API (convenience wrappers):
    from TRITON_SWMM_toolkit.examples import NorfolkExample

    # Load Norfolk example (convenience wrapper)
    norfolk = NorfolkExample.load()
    system = norfolk.system

Adding New Case Studies:
    To add a new case study, create a thin wrapper class:

    class MiamiExample:
        @classmethod
        def load(cls, download_if_exists=False, example_data_dir=None):
            return TRITON_SWMM_example.from_case_study(
                app_name=APP_NAME,
                case_name="miami_flooding",
                system_config_template="template_system_config.yaml",
                analysis_config_template="template_analysis_config.yaml",
                case_config_filename="case.yaml",
                download_if_exists=download_if_exists,
                example_data_dir=example_data_dir,
            )

For test infrastructure (synthetic weather, isolated test directories),
see tests/fixtures/ instead.
"""

from pathlib import Path
from typing import Optional

from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
import TRITON_SWMM_toolkit.constants as cnst

from TRITON_SWMM_toolkit.utils import (
    get_package_root,
    get_package_data_root,
    fill_template,
    read_yaml,
    write_yaml,
)
import sys
import yaml
import bagit
import shutil
from zipfile import ZipFile

from pathlib import Path
from typing import Optional
import warnings
from importlib.resources import files

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

from TRITON_SWMM_toolkit.config import (
    load_system_config,
    load_system_config_from_dict,
)


class TRITON_SWMM_example:
    """
    Generic example loader for TRITON-SWMM case studies.

    Provides both low-level initialization from YAML paths and high-level
    class methods for loading case studies from configuration templates.

    Attributes:
        system: Configured TRITONSWMM_system instance with analysis loaded
    """

    def __init__(self, cfg_system_yaml: Path, cfg_analysis_yaml: Path, case_name: str):
        """
        Initialize example from system and analysis configuration files.

        Args:
            cfg_system_yaml: Path to system configuration YAML
            cfg_analysis_yaml: Path to analysis configuration YAML
        """
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.analysis = TRITONSWMM_analysis(
            analysis_config_yaml=cfg_analysis_yaml, system=self.system
        )
        self.test_case_directory = self.retrieve_test_data_directory(case_name)
        #   # type: ignore

    @classmethod
    def retrieve_test_data_directory(cls, case_name: str):
        test_data_dir = files(cnst.APP_NAME).parents[1].joinpath(f"test_data/{case_name}/")  # type: ignore
        return test_data_dir

    @classmethod
    def from_case_study(
        cls,
        case_name: str,
        system_config_template: str,
        analysis_config_template: str,
        case_config_filename: str,
        download_if_exists: bool = False,
        example_data_dir: Optional[Path] = None,
    ):
        """
        Load a case study example from configuration templates.

        This is the high-level API for loading any case study. It handles:
        - Template filling with data directory paths
        - HydroShare data download
        - System and analysis configuration generation

        Args:
            app_name: Application package name (e.g., "TRITON_SWMM_toolkit")
            case_name: Case study name (e.g., "norfolk_coastal_flooding")
            system_config_template: System config filename (e.g., "template_system_config.yaml")
            analysis_config_template: Analysis config filename (e.g., "template_analysis_config.yaml")
            case_config_filename: Case metadata filename (e.g., "case.yaml")
            download_if_exists: If True, re-download HydroShare data even if it exists
            example_data_dir: Optional override for data storage location

        Returns:
            TRITON_SWMM_example instance with loaded system and analysis

        Example:
            from TRITON_SWMM_toolkit.constants import *
            example = TRITON_SWMM_example.from_case_study(
                app_name=APP_NAME,
                case_name=NORFOLK_EX,
                system_config_template=NORFOLK_SYSTEM_CONFIG,
                analysis_config_template=NORFOLK_ANALYSIS_CONFIG,
                case_config_filename=NORFOLK_CASE_CONFIG,
            )
        """
        cfg_system_yaml = cls._load_case_system_config(
            case_name=case_name,
            system_config_template=system_config_template,
            case_config_filename=case_config_filename,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )
        cfg_analysis_yaml = cls._load_case_analysis_config(
            case_name=case_name,
            analysis_config_template=analysis_config_template,
            cfg_system_yaml=cfg_system_yaml,
            example_data_dir=example_data_dir,
        )

        return cls(cfg_system_yaml, cfg_analysis_yaml, case_name)

    @classmethod
    def _load_case_analysis_config(
        cls,
        case_name: str,
        analysis_config_template: str,
        cfg_system_yaml: Path,
        example_data_dir: Optional[Path] = None,
    ):
        """
        Load analysis config for any case study.

        Args:
            app_name: Application package name
            case_name: Case study name
            analysis_config_template: Analysis config template filename
            example_data_dir: Optional data directory override

        Returns:
            Path to generated analysis configuration YAML
        """
        filled_yaml_data = cls._fill_case_analysis_yaml(
            app_name=cnst.APP_NAME,
            case_name=case_name,
            analysis_config_template=analysis_config_template,
            example_data_dir=example_data_dir,
        )
        cfg_system = load_system_config(cfg_system_yaml)
        analysis_id = filled_yaml_data["analysis_id"]
        cfg_yaml = (
            Path(cfg_system.system_directory) / f"config_analysis_{analysis_id}.yaml"
        )
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml

    @classmethod
    def _fill_case_analysis_yaml(
        cls,
        app_name: str,
        case_name: str,
        analysis_config_template: str,
        example_data_dir: Optional[Path] = None,
    ):
        """
        Fill analysis YAML template for any case study.

        Args:
            app_name: Application package name
            case_name: Case study name
            analysis_config_template: Analysis config template filename
            example_data_dir: Optional data directory override

        Returns:
            Filled YAML data as dictionary
        """
        mapping = cls._get_case_data_and_package_directory_mapping_dict(
            case_name=case_name,
            example_data_dir=example_data_dir,
        )
        cfg_template = cls._load_config_filepath(case_name, analysis_config_template)
        filled_yaml_data = cls._return_filled_template_yaml_dictionary(
            cfg_template, mapping
        )
        return filled_yaml_data

    @classmethod
    def _load_case_system_config(
        cls,
        case_name: str,
        system_config_template: str,
        case_config_filename: str,
        download_if_exists: bool,
        example_data_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Load system configuration for any case study.

        Handles template filling, HydroShare download, and config generation.

        Args:
            app_name: Application package name
            case_name: Case study name
            system_config_template: System config template filename
            case_config_filename: Case metadata filename
            download_if_exists: If True, re-download HydroShare data
            example_data_dir: Optional data directory override
            verbose: If True, print download messages

        Returns:
            Path to generated system configuration YAML
        """
        case_details = cls._load_config_file_as_dic(case_name, case_config_filename)
        res_identifier = case_details["res_identifier"]  # will come from the case yaml
        mapping = cls._get_case_data_and_package_directory_mapping_dict(
            case_name=case_name,
            example_data_dir=example_data_dir,
        )
        cfg_template = cls._load_config_filepath(case_name, system_config_template)
        filled_yaml_data = cls._return_filled_template_yaml_dictionary(
            cfg_template, mapping
        )
        cfg_system = load_system_config_from_dict(filled_yaml_data)

        # download data if it doesn't exist
        if Path(mapping["DATA_DIR"]).exists() and not download_if_exists:
            pass
        else:
            if verbose:
                print(
                    f"Download example data to {mapping['DATA_DIR']} using Hydroshare"
                )
            hs = cls._sign_into_hydroshare()
            cls._download_data_from_hydroshare(
                res_identifier,
                Path(mapping["HYDROSHARE_ROOT"]),
                hs,
                download_if_exists=download_if_exists,
            )

        cfg_yaml = Path(filled_yaml_data["system_directory"]) / "config_system.yaml"
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml

    @classmethod
    def _get_case_data_and_package_directory_mapping_dict(
        cls,
        case_name: str,
        example_data_dir: Optional[Path] = None,
    ):
        """
        Get directory mappings for any case study.

        Args:
            app_name: Application package name
            case_name: Case study name
            example_data_dir: Optional data directory override

        Returns:
            Dictionary with DATA_DIR, PACKAGE_DIR, HYDROSHARE_ROOT paths
        """
        if example_data_dir:
            root = example_data_dir
        else:
            root = get_package_data_root(cnst.APP_NAME)
        hydroshare_root_dir = root / "examples" / case_name
        data_dir = hydroshare_root_dir / "data" / "contents"
        package_dir = (
            get_package_root(cnst.APP_NAME).parents[1] / "test_data" / case_name
        )

        mapping = dict(
            DATA_DIR=str(data_dir),
            PACKAGE_DIR=str(package_dir),
            HYDROSHARE_ROOT=str(hydroshare_root_dir),
        )
        return mapping

    @classmethod
    def _load_config_filepath(cls, case_study_name: str, filename: str) -> Path:
        return files(cnst.APP_NAME).parents[1].joinpath(f"test_data/{case_study_name}/{filename}")  # type: ignore

    @classmethod
    def _load_config_file_as_dic(cls, case_study_name: str, filename: str) -> dict:
        path = cls._load_config_filepath(case_study_name, filename)
        return read_yaml(path)

    @classmethod
    def _download_data_from_hydroshare(
        cls,
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

    @classmethod
    def _sign_into_hydroshare(
        cls,
    ):
        if HydroShare is None:
            raise RuntimeError(
                "hsclient is not installed. Install optional dependencies with `pip install .[tests]`. Alternatively, you can download the data manually if you have issues installing this package with pip. Link: https://www.hydroshare.org/resource/a4aace329b8c401a93e94ce2a761fe1b/"
            )
        hs = HydroShare()
        print("Please log into hydroshare to download example.", flush=True)
        hs.sign_in()
        print("signed into Hydroshare successfully.")
        return hs

    @classmethod
    def _return_filled_template_yaml_dictionary(cls, cfg_template: Path, mapping: dict):
        cfg_filled = fill_template(cfg_template, mapping)
        try:
            cfg_filled_yaml = yaml.safe_load(cfg_filled)
        except Exception:
            print(cfg_filled)
            sys.exit("failed to load yaml")
        return cfg_filled_yaml


class NorfolkExample:
    """
    Convenience wrapper for Norfolk coastal flooding case study.

    This is a thin wrapper around TRITON_SWMM_example that provides
    Norfolk-specific defaults. Makes it easy to load the Norfolk example
    without remembering all the constant names.

    Example:
        from TRITON_SWMM_toolkit.examples import NorfolkExample

        # Load Norfolk example with Hurricane Irene data
        norfolk = NorfolkExample.load()
        system = norfolk.system

        # Or just load the analysis template
    """

    @classmethod
    def load(
        cls,
        download_if_exists: bool = False,
        example_data_dir: Optional[Path] = None,
    ) -> TRITON_SWMM_example:
        """
        Load Norfolk coastal flooding example.

        Args:
            download_if_exists: If True, re-download HydroShare data
            example_data_dir: Optional override for data directory

        Returns:
            TRITON_SWMM_example instance with Norfolk system loaded
        """

        return TRITON_SWMM_example.from_case_study(
            case_name=cnst.NORFOLK_EX,
            system_config_template=cnst.NORFOLK_SYSTEM_CONFIG,
            analysis_config_template=cnst.NORFOLK_ANALYSIS_CONFIG,
            case_config_filename=cnst.NORFOLK_CASE_CONFIG,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )
