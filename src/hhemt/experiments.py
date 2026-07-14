"""
Production examples for TRITON-SWMM toolkit.

This module provides utilities for loading and running production examples.
Examples are organized as case studies (e.g., Norfolk coastal flooding) with
associated HydroShare data and configuration templates.

Generic API (for any case study):
    from hhemt.experiments import TRITON_SWMM_experiment
    from hhemt.constants import *

    # Load any case study
    example = TRITON_SWMM_experiment.from_case_study(
        case_name="norfolk_coastal_flooding",
        system_config_template="template_system_config.yaml",
        analysis_config_template="template_analysis_config.yaml",
        case_config_filename="case.yaml",
    )
    system = example.system

Case-Specific API (convenience wrappers):
    from hhemt.experiments import NorfolkIreneExperiment

    # Load Norfolk example (convenience wrapper)
    norfolk = NorfolkIreneExperiment.load()
    system = norfolk.system

Adding New Case Studies:
    To add a new case study, create a thin wrapper class:

    class MiamiExample:
        @classmethod
        def load(cls, download_if_exists=False, example_data_dir=None):
            return TRITON_SWMM_experiment.from_case_study(
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
import json
import re

from hhemt.config.case_manifest import CaseManifest
from hhemt.exceptions import ConfigurationError, ProcessingError


class TRITON_SWMM_experiment:
    """
    Generic example loader for TRITON-SWMM case studies.

    Provides both low-level initialization from YAML paths and high-level
    class methods for loading case studies from configuration templates.

    Attributes:
        system: Configured TRITONSWMM_system instance with analysis loaded
    """

    def __init__(
        self,
        cfg_system_yaml: Path,
        cfg_analysis_yaml: Path,
        case_name: str | None = None,
    ):
        """
        Initialize an experiment from system and analysis configuration files.

        Args:
            cfg_system_yaml: Path to system configuration YAML
            cfg_analysis_yaml: Path to analysis configuration YAML
            case_name: In-repo case-study name, used to resolve the packaged
                ``test_data/{case_name}/`` tree. ``None`` for a DOI-ingested bundle
                (``from_doi``), which carries no in-repo test-data tree — the
                ``retrieve_test_data_directory`` lookup is skipped and
                ``test_case_directory`` is ``None``; ``from_doi`` sets
                ``bundle_root`` instead.
        """
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.analysis = TRITONSWMM_analysis(
            analysis_config_yaml=cfg_analysis_yaml, system=self.system
        )
        self.test_case_directory = (
            self.retrieve_test_data_directory(case_name)
            if case_name is not None
            else None
        )
        self.bundle_root: Path | None = None

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
            TRITON_SWMM_experiment instance with loaded system and analysis

        Example:
            from hhemt.constants import *
            example = TRITON_SWMM_experiment.from_case_study(
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
    def from_doi(
        cls,
        doi: str | None = None,
        *,
        pid: str | None = None,
        host: str,
        expected_sha256: str | None = None,
        target_dir: Path | None = None,
        software_dir: Path | None = None,
    ) -> "TRITON_SWMM_experiment":
        """Fetch a published reprex bundle by DOI/PID, reconstitute it, and return a
        runnable experiment (ADR-13 C9; R1-R4).

        Under the ADR-9 self-contained-by-default contract the emitted bundle carries
        every cfg-declared simulation input at its bundle-relative location, so a
        self-contained bundle reconstitutes to inputs that all exist on disk — a
        from-scratch runnable experiment, dependent externally only on the reproducer's
        USER-specific + HPC-specific configs and the container SIF (built on ingest from
        the carried ``.def`` recipe, or transferred per ADR-2 — a separate phase). A
        bundle emitted with a ``bundle_exclude_config`` opt-out carries the excluded
        inputs by-reference via an ``input_deposit`` block fetched on ingest.

        Trust boundary: ingesting a DOI and running it executes shell derived from the
        fetched config — ingest only deposits you trust. ``expected_sha256`` pins the
        fetched bundle-zip integrity.

        Args:
            doi: The bundle's DOI (e.g. ``'10.5281/zenodo.123456'``). Either ``doi`` or ``pid``.
            pid: The host-native id (Zenodo record id / HydroShare resource id).
            host: ``'zenodo'`` or ``'hydroshare'`` — REQUIRED, no default (the sibling
                case-study fetch defaults to hydroshare; two DOI entry points with
                opposite host defaults is a trap).
            expected_sha256: Optional sha256 pin on the fetched bundle zip.
            target_dir: Optional directory to fetch into (default: a fresh temp dir).
            software_dir: Target-side directory for the toolkit-owned SWMM/TRITON build
                dirs (default: ``{bundle_root}/software``). These are build OUTPUTS the
                toolkit creates at setup, not bundled inputs; they must be non-null for
                ``TRITONSWMM_system`` to construct.
        """
        from hhemt.bundle import Bundle
        from hhemt.bundle._emit import (
            reconstitute_runnable_analysis_config,
            reconstitute_runnable_config,
        )
        from hhemt.bundle._reprex import extract_reprex_bundle

        if not (doi or pid):
            raise ConfigurationError(
                field="doi",
                message="from_doi requires either `doi` or `pid`.",
                config_path=None,
            )
        if host not in ("zenodo", "hydroshare"):
            raise ConfigurationError(
                field="host",
                message=f"host must be 'zenodo' or 'hydroshare', got {host!r}.",
                config_path=None,
            )

        if target_dir is None:
            import tempfile

            target_dir = Path(tempfile.mkdtemp(prefix="hhemt_ingest_"))
        else:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

        zip_path = cls._fetch_bundle_zip(
            host,
            doi=doi,
            pid=pid,
            res_identifier=pid,
            dest=target_dir,
            expected_sha256=expected_sha256,
        )
        bundle_root = extract_reprex_bundle(zip_path)

        # Schema guard: raises BundleSchemaError on version skew, FileNotFoundError when
        # the directory carries no bundle_manifest.json.
        Bundle.from_directory(bundle_root)

        # R3: fail closed unless the crate names a runnable workflow (mainEntity).
        cls._assert_bundle_has_workflow(bundle_root)

        # Reconstitute the runnable config pair with bundle-relative paths resolved
        # absolute under bundle_root (system + analysis). software_dir points the
        # toolkit-owned build dirs at a writable target-side location (default
        # {bundle_root}/software) so TRITONSWMM_system constructs and run() can build
        # there — the render-path null-ing would raise at construction.
        system_config_path = reconstitute_runnable_config(
            bundle_root, software_dir=software_dir or (bundle_root / "software")
        )
        analysis_config_path = reconstitute_runnable_analysis_config(bundle_root)

        # Materialize-or-fail: a self-contained bundle carries every declared input, so
        # this passes; it fails closed (naming every absent input) on a malformed/partial
        # bundle. This is the only real gate — the load-time existence check is inert on
        # the reconstituted YAML string values.
        cls._assert_declared_inputs_exist(system_config_path, analysis_config_path)

        exp = cls(system_config_path, analysis_config_path, case_name=None)
        exp.bundle_root = bundle_root
        return exp

    @classmethod
    def _fetch_bundle_zip(
        cls,
        host: str,
        *,
        doi: str | None = None,
        pid: str | None = None,
        res_identifier: str | None = None,
        dest: Path,
        expected_sha256: str | None = None,
    ) -> Path:
        """Fetch a deposit and locate the single reprex-bundle ``.zip`` within its
        payload root. Raises ``ProcessingError`` when zero or more than one candidate
        zip is present."""
        payload_root = cls._fetch_deposit_files(
            host,
            doi=doi,
            pid=pid,
            res_identifier=res_identifier,
            dest=Path(dest),
            download_if_exists=True,
        )
        candidates = sorted(payload_root.rglob("*.zip"))
        if len(candidates) != 1:
            rel = [str(c.relative_to(payload_root)) for c in candidates]
            raise ProcessingError(
                operation="doi_ingest_bundle_zip",
                filepath=str(payload_root),
                reason=(
                    f"expected exactly one bundle .zip in the fetched deposit, found "
                    f"{len(candidates)}: {rel}"
                ),
            )
        zip_path = candidates[0]
        if expected_sha256 is not None:
            cls._verify_sha256(zip_path, expected_sha256)
        return zip_path

    @staticmethod
    def _verify_sha256(path: Path, expected_sha256: str) -> None:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)  # streaming read bounds peak RSS on multi-GB assets
        actual = h.hexdigest()
        if actual != expected_sha256:
            raise ProcessingError(
                operation="doi_ingest_sha256",
                filepath=str(path),
                reason=(
                    f"sha256 mismatch on the fetched bundle: expected "
                    f"{expected_sha256}, got {actual}"
                ),
            )

    @staticmethod
    def _assert_bundle_has_workflow(bundle_root: Path) -> None:
        """R3: fail closed unless the bundle's RO-Crate names a runnable workflow.

        Reads the crate ROOT entity's ``mainEntity`` ``@id`` (the placement
        ``metadata.upgrade_doc_to_workflow_run_crate`` writes) and verifies the
        referenced workflow file exists under ``bundle_root``. A crate that is absent,
        carries no ``mainEntity``, or points at a missing workflow file is a mis-pointed
        ADR-11 data deposit rather than a runnable reprex bundle — a key-presence test
        alone is a weaker gate than "this bundle carries a runnable workflow"."""
        crate_path = bundle_root / "ro-crate-metadata.json"
        if not crate_path.exists():
            raise ProcessingError(
                operation="doi_ingest_crate",
                filepath=str(crate_path),
                reason=(
                    "the fetched bundle carries no ro-crate-metadata.json; it is not a "
                    "runnable reprex bundle (likely a mis-pointed ADR-11 data deposit). "
                    "Ingest the reprex-bundle DOI, not the data-deposit DOI."
                ),
            )
        graph = json.loads(crate_path.read_text()).get("@graph", [])
        root = next((e for e in graph if e.get("@id") == "./"), None)
        main_entity = (root or {}).get("mainEntity")
        workflow_id = main_entity.get("@id") if isinstance(main_entity, dict) else None
        if not workflow_id:
            raise ProcessingError(
                operation="doi_ingest_crate",
                filepath=str(crate_path),
                reason=(
                    "the fetched crate has no root `mainEntity` (not a Workflow-Run-"
                    "Crate); it is a plain RO-Crate or a mis-pointed ADR-11 data "
                    "deposit, not a runnable reprex bundle."
                ),
            )
        workflow_path = bundle_root / workflow_id
        if not workflow_path.exists():
            raise ProcessingError(
                operation="doi_ingest_crate",
                filepath=str(workflow_path),
                reason=(
                    f"the crate names a workflow mainEntity ({workflow_id}) but that "
                    f"file is absent from the bundle — the bundle does not carry a "
                    f"runnable workflow."
                ),
            )

    @staticmethod
    def _assert_declared_inputs_exist(
        system_config_path: Path, analysis_config_path: Path
    ) -> None:
        """Fail-closed materialize gate: raise ``ProcessingError`` naming EVERY
        reconstituted input Path that does not exist on disk.

        The reconstituted configs carry ABSOLUTE resolved paths under ``bundle_root``.
        ``cfgBaseModel._check_paths_exist`` is ``mode='before'`` + ``isinstance(v, Path)``
        -gated, so it never fires on these YAML strings — this gate is the only thing
        standing between a partial bundle and a silent hours-later setup failure.

        Only CARRIED-INPUT fields are checked (the same ``BUNDLE_RELATIVE`` family the
        self-contained harvest carries). The toolkit-owned build dirs
        (``IS_NONE_ACCEPTABLE`` — set by ``reconstitute_runnable_config`` to a
        not-yet-existing, existence-exempt target-side location) and the ``FORCED_DOT``
        bundle-root markers (``analysis_dir`` / ``system_directory``) are NOT inputs and
        are skipped — checking them would false-fail on a runnable bundle."""
        from hhemt.bundle._path_policy import (
            _PATH_FIELD_POLICY,
            PathPolicy,
            enumerate_path_fields,
        )
        from hhemt.config.analysis import analysis_config
        from hhemt.config.system import system_config

        carried = {
            PathPolicy.BUNDLE_RELATIVE,
            PathPolicy.BUNDLE_RELATIVE_OR_NONE,
            PathPolicy.BUNDLE_RELATIVE_LIST,
        }
        missing: list[str] = []
        for cfg_path, cfg_model in (
            (system_config_path, system_config),
            (analysis_config_path, analysis_config),
        ):
            data = yaml.safe_load(Path(cfg_path).read_text())
            for name in enumerate_path_fields(cfg_model):
                if _PATH_FIELD_POLICY.get(name) not in carried:
                    continue
                value = data.get(name)
                if value is None:
                    continue
                for v in value if isinstance(value, list) else [value]:
                    if v is None:
                        continue
                    if not Path(v).exists():
                        missing.append(f"{name}: {v}")
        if missing:
            raise ProcessingError(
                operation="doi_ingest_inputs",
                filepath=None,
                reason=(
                    "the reconstituted experiment declares inputs that do not exist on "
                    "disk — the bundle is not self-contained (or was emitted with an "
                    "exclude-config whose input_deposit fetch is unavailable). Place "
                    "these under {bundle_root}/external/ and re-run, or re-emit the "
                    "bundle self-contained. Missing inputs:\n  "
                    + "\n  ".join(sorted(missing))
                ),
            )

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
            if case_manifest.host == "zenodo":
                cls._download_data_from_zenodo(
                    case_manifest,
                    Path(mapping["HYDROSHARE_ROOT"]),
                    download_if_exists=download_if_exists,
                    expected_manifest=case_manifest.manifest,
                )
            else:
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
    def _fetch_deposit_files(
        cls,
        host: str,
        *,
        doi: str | None = None,
        pid: str | None = None,
        res_identifier: str | None = None,
        dest: Path,
        expected_manifest: dict[str, str] | None = None,
        download_if_exists: bool = False,
        hs=None,
        validate: bool = True,
    ) -> Path:
        """Generic host-dispatched deposit fetch (ADR-12/C7; R4) — the single
        implementation the case-study data fetch AND the bundle-zip ingestion both route
        through. Returns the PAYLOAD ROOT: ``dest`` for Zenodo (flat files under their
        record keys), the extracted BAG ROOT (renamed to ``dest``) for HydroShare
        (deposited files under ``data/contents/``). Both hosts run the host-agnostic
        streaming-sha256 ``_verify_manifest`` against the returned root, preserving the
        ``case manifest sha256 uses streaming chunked read`` stipulation.

        Contract points the seam owns: (i) host-key resolution — Zenodo resolves
        ``pid or doi.rsplit('zenodo.', 1)[-1]`` and rejects a non-numeric record id
        (it is interpolated into the records API URL); HydroShare keys on
        ``res_identifier or pid``. (ii) client construction — the seam connects via
        ``_connect_to_hydroshare`` internally unless a pre-connected ``hs`` is passed.
        (iii) caching — ``download_if_exists`` reproduces the early-return-if-present /
        ``fast_rmtree``-and-refetch behavior so ``from_case_study`` stays byte-behavioral.
        """
        dest = Path(dest)
        if host == "zenodo":
            recid = (pid or "").strip()
            if not recid and doi:
                recid = doi.rsplit("zenodo.", 1)[-1].strip()
            if not recid:
                raise ProcessingError(
                    operation="zenodo_resolve",
                    filepath=None,
                    reason=(
                        f"cannot resolve a Zenodo record id from doi={doi!r} pid={pid!r}"
                    ),
                )
            if not re.fullmatch(r"[0-9]+", recid):
                raise ProcessingError(
                    operation="zenodo_resolve",
                    filepath=None,
                    reason=(
                        f"resolved Zenodo record id {recid!r} is not numeric; refusing "
                        f"to interpolate it into the records API URL"
                    ),
                )
            if dest.exists() and download_if_exists:
                fast_rmtree(dest)  # EXEMPT-DU: test-example-fixture
            if dest.exists() and not download_if_exists:
                return dest
            dest.mkdir(parents=True, exist_ok=True)
            import requests  # explicit dep declared in pyproject

            resp = requests.get(
                f"https://zenodo.org/api/records/{recid}", timeout=60
            )
            if resp.status_code != 200:
                raise ProcessingError(
                    operation="zenodo_fetch",
                    filepath=None,
                    reason=f"Zenodo record {recid} returned HTTP {resp.status_code}",
                )
            for entry in resp.json().get("files", []):
                cls._fetch_file_by_url(entry["links"]["self"], dest / entry["key"])
            if validate:
                # host-agnostic sha256 (Zenodo md5 is advisory)
                cls._verify_manifest(dest, expected_manifest)
            return dest

        if host == "hydroshare":
            key = res_identifier or pid
            if not key:
                raise ProcessingError(
                    operation="hydroshare_resolve",
                    filepath=None,
                    reason="HydroShare fetch requires res_identifier or pid",
                )
            if dest.exists() and download_if_exists:
                fast_rmtree(dest)  # EXEMPT-DU: test-example-fixture
            if dest.exists() and not download_if_exists:
                return dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            if hs is None:
                hs = cls._connect_to_hydroshare(key)
            hs_resource = hs.resource(key)
            zip_path = Path(hs_resource.download(dest.parent))
            with ZipFile(zip_path, "r") as z:
                z.extractall(dest.parent)
            with ZipFile(zip_path, "r") as z:
                top_level_dirs = {
                    Path(f).parts[0] for f in z.namelist() if Path(f).parts
                }
            if len(top_level_dirs) == 1:
                bag_root = dest.parent / next(iter(top_level_dirs))
            else:
                raise ProcessingError(
                    operation="hydroshare_bag_extract",
                    filepath=str(zip_path),
                    reason=(
                        "ZIP has multiple top-level folders; cannot determine Bag root."
                    ),
                )
            if validate:
                bag = bagit.Bag(str(bag_root))
                if bag.is_valid():
                    print("Bag verified! All bagit checksums match.", flush=True)
                else:
                    raise ProcessingError(
                        operation="hydroshare_bag_validation",
                        filepath=str(bag_root),
                        reason=(
                            "bagit manifest validation failed (bag is not "
                            "self-consistent)."
                        ),
                    )
            cls._verify_manifest(bag_root, expected_manifest)
            bag_root.rename(dest)
            zip_path.unlink()  # EXEMPT-DU: test-example-fixture
            return dest

        raise ProcessingError(
            operation="deposit_fetch",
            filepath=None,
            reason=f"unknown deposit host {host!r} (expected 'zenodo' or 'hydroshare')",
        )

    @classmethod
    def _fetch_file_by_url(
        cls, url: str, dest: Path, *, expected_sha256: str | None = None
    ) -> Path:
        """Streaming 1 MiB-chunk download of a single URL to ``dest`` (the digest core).

        Used by the Zenodo per-file branch of ``_fetch_deposit_files`` AND by the crate's
        by-reference SIF fetch (``downloadUrl`` + ``sha256``), which is a URL fetch that
        cannot be expressed through the ``(host, doi|pid)`` deposit signature. When
        ``expected_sha256`` is given, the downloaded file is sha256-verified."""
        import requests  # explicit dep declared in pyproject

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=600) as fr:
            fr.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in fr.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        if expected_sha256 is not None:
            cls._verify_sha256(dest, expected_sha256)
        return dest

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
        """Thin wrapper preserving the case-study call site; delegates to the single
        ``_fetch_deposit_files`` seam (passing the already-connected ``hs`` so the
        seam does not reconnect)."""
        return cls._fetch_deposit_files(
            "hydroshare",
            res_identifier=res_identifier,
            dest=Path(target),
            expected_manifest=expected_manifest,
            download_if_exists=download_if_exists,
            hs=hs,
            validate=validate,
        )

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
    def _download_data_from_zenodo(
        cls,
        case_manifest: "CaseManifest",
        target: Path,
        download_if_exists: bool = False,
        expected_manifest: dict[str, str] | None = None,
    ):
        """Thin wrapper preserving the case-study call site; delegates to the single
        ``_fetch_deposit_files`` seam. Resolves the Zenodo record id from
        ``case_manifest.doi`` (suffix after ``'zenodo.'``) or ``case_manifest.pid``,
        downloads each file, then runs the host-agnostic ``_verify_manifest``. Prefer a
        versioned DOI (pins one version)."""
        return cls._fetch_deposit_files(
            "zenodo",
            doi=case_manifest.doi,
            pid=case_manifest.pid,
            dest=Path(target),
            expected_manifest=expected_manifest,
            download_if_exists=download_if_exists,
        )

    @classmethod
    def _return_filled_template_yaml_dictionary(cls, cfg_template: Path, mapping: dict):
        cfg_filled = fill_template(cfg_template, mapping)
        try:
            cfg_filled_yaml = yaml.safe_load(cfg_filled)
        except Exception:
            print(cfg_filled)
            sys.exit("failed to load yaml")
        return cfg_filled_yaml


class NorfolkIreneExperiment:
    """
    Convenience wrapper for Norfolk coastal flooding case study.

    This is a thin wrapper around TRITON_SWMM_experiment that provides
    Norfolk-specific defaults. Makes it easy to load the Norfolk example
    without remembering all the constant names.

    Example:
        from hhemt.experiments import NorfolkIreneExperiment

        # Load Norfolk example with Hurricane Irene data
        norfolk = NorfolkIreneExperiment.load()
        system = norfolk.system

        # Or just load the analysis template
    """

    @classmethod
    def load(
        cls,
        download_if_exists: bool = False,
        example_data_dir: Optional[Path] = None,
    ) -> TRITON_SWMM_experiment:
        """
        Load Norfolk coastal flooding example.

        Args:
            download_if_exists: If True, re-download HydroShare data
            example_data_dir: Optional override for data directory

        Returns:
            TRITON_SWMM_experiment instance with Norfolk system loaded
        """

        weather_events_to_simulate = "hurricane_irene_event_index.csv"
        analysis_description = "Single Simulation of Hurricane Irene 8-27-2011"

        return TRITON_SWMM_experiment.from_case_study(
            case_name=cnst.NORFOLK_EX,
            system_config_template=cnst.NORFOLK_SYSTEM_CONFIG,
            analysis_config_template=cnst.NORFOLK_ANALYSIS_CONFIG,
            case_config_filename=cnst.NORFOLK_CASE_CONFIG,
            weather_events_to_simulate=weather_events_to_simulate,
            analysis_description=analysis_description,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )


class NorfolkObservedExperiment:
    """
    Convenience wrapper for observed event ensemble simulation.

    This is a thin wrapper around TRITON_SWMM_experiment that provides
    Norfolk-specific defaults. Makes it easy to load the Norfolk example
    without remembering all the constant names.

    Example:
        from hhemt.experiments import NorfolkObservedExperiment

        # Load Norfolk example with Hurricane Irene data
        norfolk = NorfolkObservedExperiment.load()
        system = norfolk.system

        # Or just load the analysis template
    """

    @classmethod
    def load(
        cls,
        download_if_exists: bool = False,
        example_data_dir: Optional[Path] = None,
    ) -> TRITON_SWMM_experiment:
        """
        Load Norfolk coastal flooding example.

        Args:
            download_if_exists: If True, re-download HydroShare data
            example_data_dir: Optional override for data directory

        Returns:
            TRITON_SWMM_experiment instance with Norfolk system loaded
        """

        # this method just changes the weather_events_to_simulate
        # for analysis config

        weather_events_to_simulate = (
            "obs_event_summaries_from_yrs_with_complete_coverage.csv"
        )
        analysis_description = "Observed event ensemble"
        return TRITON_SWMM_experiment.from_case_study(
            case_name=cnst.NORFOLK_EX,
            system_config_template=cnst.NORFOLK_SYSTEM_CONFIG,
            analysis_config_template=cnst.NORFOLK_ANALYSIS_CONFIG,
            case_config_filename=cnst.NORFOLK_CASE_CONFIG,
            weather_events_to_simulate=weather_events_to_simulate,
            analysis_description=analysis_description,
            download_if_exists=download_if_exists,
            example_data_dir=example_data_dir,
        )
