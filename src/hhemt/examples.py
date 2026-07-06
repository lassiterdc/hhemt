"""
Production examples for TRITON-SWMM toolkit.

This module provides utilities for loading and running production examples.
Examples are organized as case studies (e.g., Norfolk coastal flooding) with
associated HydroShare data and configuration templates.

Generic API (for any case study):
    from hhemt.examples import TRITON_SWMM_example
    from hhemt.constants import *

    # Load any case study
    example = TRITON_SWMM_example.from_case_study(
        case_name="norfolk_coastal_flooding",
        system_config_template="template_system_config.yaml",
        analysis_config_template="template_analysis_config.yaml",
        case_config_filename="case.yaml",
    )
    system = example.system

Case-Specific API (convenience wrappers):
    from hhemt.examples import NorfolkIreneExample

    # Load Norfolk example (convenience wrapper)
    norfolk = NorfolkIreneExample.load()
    system = norfolk.system

Adding New Case Studies:
    To add a new case study, create a thin wrapper class:

    class MiamiExample:
        @classmethod
        def load(cls, download_if_exists=False, example_data_dir=None):
            return TRITON_SWMM_example.from_case_study(
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

from hhemt.system import TRITONSWMM_system
from hhemt.analysis import TRITONSWMM_analysis
import hhemt.constants as cnst

from hhemt.utils import (
    fast_rmtree,
    get_package_root,
    get_package_data_root,
    fill_template,
    read_yaml,
    write_yaml,
)
import sys
import yaml
import bagit
from zipfile import ZipFile

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

from hhemt.config.loaders import (
    load_system_config,
    load_system_config_from_dict,
)

import hashlib

from hhemt.config.case_manifest import CaseManifest
from hhemt.exceptions import ProcessingError


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
        weather_events_to_simulate: str,
        analysis_description: str,
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
            case_name: Case study name (e.g., "norfolk_coastal_flooding")
            system_config_template: System config filename (e.g., "template_system_config.yaml")
            analysis_config_template: Analysis config filename (e.g., "template_analysis_config.yaml")
            case_config_filename: Case metadata filename (e.g., "case.yaml")
            download_if_exists: If True, re-download HydroShare data even if it exists
            example_data_dir: Optional override for data storage location

        Returns:
            TRITON_SWMM_example instance with loaded system and analysis

        Example:
            from hhemt.constants import *
            example = TRITON_SWMM_example.from_case_study(
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
            weather_events_to_simulate=weather_events_to_simulate,
            analysis_description=analysis_description,
            example_data_dir=example_data_dir,
        )

        return cls(cfg_system_yaml, cfg_analysis_yaml, case_name)

    @classmethod
    def _load_case_analysis_config(
        cls,
        case_name: str,
        analysis_config_template: str,
        cfg_system_yaml: Path,
        weather_events_to_simulate: str,
        analysis_description: str,
        example_data_dir: Optional[Path] = None,
    ):
        """
        Load analysis config for any case study.

        Args:
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
        placeholder_weather_path = Path(filled_yaml_data["weather_events_to_simulate"])
        weatherpath = placeholder_weather_path.parent / weather_events_to_simulate
        filled_yaml_data["weather_events_to_simulate"] = str(weatherpath)
        filled_yaml_data["analysis_description"] = analysis_description
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
            case_name: Case study name
            system_config_template: System config template filename
            case_config_filename: Case metadata filename
            download_if_exists: If True, re-download HydroShare data
            example_data_dir: Optional data directory override
            verbose: If True, print download messages

        Returns:
            Path to generated system configuration YAML
        """
        case_manifest = cls._load_case_manifest(case_name, case_config_filename)
        res_identifier = case_manifest.res_identifier
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
            hs = cls._connect_to_hydroshare(res_identifier)
            cls._download_data_from_hydroshare(
                res_identifier,
                Path(mapping["HYDROSHARE_ROOT"]),
                hs,
                download_if_exists=download_if_exists,
                expected_manifest=case_manifest.manifest,
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
    def _load_case_manifest(cls, case_study_name: str, filename: str) -> CaseManifest:
        path = cls._load_config_filepath(case_study_name, filename)
        return CaseManifest.model_validate(read_yaml(path))

    @classmethod
    def _download_data_from_hydroshare(
        cls,
        res_identifier: str,
        target: Path,
        hs,
        download_if_exists=False,
        validate=True,
        expected_manifest: dict[str, str] | None = None,
    ):
        if target.exists() and download_if_exists:
            # EXEMPT-DU: test-example-fixture
            fast_rmtree(target)
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
                raise ProcessingError(
                    operation="hydroshare_bag_extract",
                    filepath=str(zip_path),
                    reason="ZIP has multiple top-level folders; cannot determine Bag root.",
                )
        # unzipped_folder = Path(extract_to).joinpath(zip_path.name.split(".")[0])
        if validate:
            bag = bagit.Bag(str(unzipped_folder))
            if bag.is_valid():
                print("Bag verified! All bagit checksums match.", flush=True)
            else:
                raise ProcessingError(
                    operation="hydroshare_bag_validation",
                    filepath=str(unzipped_folder),
                    reason="bagit manifest validation failed (bag is not self-consistent).",
                )

        cls._verify_manifest(unzipped_folder, expected_manifest)

        outdir = unzipped_folder.rename(target)
        # EXEMPT-DU: test-example-fixture
        zip_path.unlink()

    @classmethod
    def _verify_manifest(cls, bag_root: Path, expected_manifest: dict[str, str] | None) -> None:
        """Raise ProcessingError if any manifest-declared file is absent or sha256-mismatched.

        Keys in ``expected_manifest`` are POSIX paths relative to ``bag_root`` (the
        extracted bag root — the dir containing ``data/`` and the bagit control files,
        i.e. the dir ``_download_data_from_hydroshare`` names ``unzipped_folder``). This
        is the SAME key base ``generate_case_manifest.compute_manifest`` uses, so a
        manifest computed against the bag root verifies against the bag root. An empty
        or ``None`` manifest is a no-op (the Norfolk pre-population ``manifest: {}`` state).
        """
        if not expected_manifest:
            return
        for rel_name, expected_sha in expected_manifest.items():
            fpath = bag_root / rel_name
            if not fpath.exists():
                raise ProcessingError(
                    operation="case_manifest_verification",
                    filepath=str(fpath),
                    reason=f"file declared in case.yaml manifest is absent from the downloaded bag: {rel_name}",
                )
            h = hashlib.sha256()
            with fpath.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            actual_sha = h.hexdigest()  # streaming read bounds peak RSS on multi-GB assets
            if actual_sha != expected_sha:
                raise ProcessingError(
                    operation="case_manifest_verification",
                    filepath=str(fpath),
                    reason=(
                        f"sha256 mismatch for {rel_name}: expected {expected_sha}, got {actual_sha}. "
                        "The Hydroshare resource may have been reorganized; "
                        "regenerate case.yaml via generate_case_manifest if intended."
                    ),
                )

    @classmethod
    def _connect_to_hydroshare(cls, res_identifier: str):
        """Return a HydroShare client able to read ``res_identifier``.

        Anonymous-first: a public resource is downloadable with NO credentials
        (verified: GET /hsapi/resource/{id}/ returns 200 binary/octet-stream for a
        public resource). Only fall back to the interactive OAuth sign-in when the
        anonymous resource read fails (private/unshared resource).
        """
        if HydroShare is None:
            raise ProcessingError(
                operation="hydroshare_connect",
                filepath=None,
                reason=(
                    "hsclient is not installed. Install optional dependencies with "
                    "`pip install .[tests]`. Alternatively, download the data manually: "
                    f"https://www.hydroshare.org/resource/{res_identifier}/"
                ),
            )
        hs = HydroShare()  # no args -> unauthenticated requests.Session, no userInfo call
        try:
            hs.resource(res_identifier, validate=True)  # public read -> 200, no auth
            return hs
        except Exception as exc:  # noqa: BLE001 - hsclient raises bare Exception on non-200
            # Broad catch is deliberate: hsclient does not raise a typed auth error.
            # Preserve the original cause so a network failure or a misspelled
            # res_identifier is diagnosable rather than silently masked by the
            # interactive sign-in prompt.
            print(
                f"Anonymous read of {res_identifier} failed ({exc!r}); "
                "signing in to HydroShare.",
                flush=True,
            )
            try:
                hs.sign_in()
            except Exception as sign_in_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"HydroShare sign-in failed after anonymous read of "
                    f"{res_identifier} failed ({exc!r})."
                ) from sign_in_exc
            print("Signed into HydroShare successfully.", flush=True)
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


class NorfolkIreneExample:
    """
    Convenience wrapper for Norfolk coastal flooding case study.

    This is a thin wrapper around TRITON_SWMM_example that provides
    Norfolk-specific defaults. Makes it easy to load the Norfolk example
    without remembering all the constant names.

    Example:
        from hhemt.examples import NorfolkIreneExample

        # Load Norfolk example with Hurricane Irene data
        norfolk = NorfolkIreneExample.load()
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

        weather_events_to_simulate = "hurricane_irene_event_index.csv"
        analysis_description = "Single Simulation of Hurricane Irene 8-27-2011"

        return TRITON_SWMM_example.from_case_study(
            case_name=cnst.NORFOLK_EX,
            system_config_template=cnst.NORFOLK_SYSTEM_CONFIG,
            analysis_config_template=cnst.NORFOLK_ANALYSIS_CONFIG,
            case_config_filename=cnst.NORFOLK_CASE_CONFIG,
            weather_events_to_simulate=weather_events_to_simulate,
            analysis_description=analysis_description,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )


class NorfolkObservedExample:
    """
    Convenience wrapper for observed event ensemble simulation.

    This is a thin wrapper around TRITON_SWMM_example that provides
    Norfolk-specific defaults. Makes it easy to load the Norfolk example
    without remembering all the constant names.

    Example:
        from hhemt.examples import NorfolkObservedExample

        # Load Norfolk example with Hurricane Irene data
        norfolk = NorfolkObservedExample.load()
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

        # this method just changes the weather_events_to_simulate
        # for analysis config

        weather_events_to_simulate = (
            "obs_event_summaries_from_yrs_with_complete_coverage.csv"
        )
        analysis_description = "Observed event ensemble"
        return TRITON_SWMM_example.from_case_study(
            case_name=cnst.NORFOLK_EX,
            system_config_template=cnst.NORFOLK_SYSTEM_CONFIG,
            analysis_config_template=cnst.NORFOLK_ANALYSIS_CONFIG,
            case_config_filename=cnst.NORFOLK_CASE_CONFIG,
            weather_events_to_simulate=weather_events_to_simulate,
            analysis_description=analysis_description,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )
