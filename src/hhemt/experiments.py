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

import sys
import warnings
from importlib.resources import files
from pathlib import Path
from zipfile import ZipFile

import bagit
import yaml

import hhemt.constants as cnst
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.system import TRITONSWMM_system
from hhemt.utils import (
    fast_rmtree,
    fill_template,
    get_package_data_root,
    get_package_root,
    read_yaml,
    write_yaml,
)

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

import hashlib
import json
import re

from hhemt.config.case_manifest import CaseManifest
from hhemt.config.loaders import (
    load_system_config,
    load_system_config_from_dict,
)
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError, ProcessingError

#: The published surface of this module. Declared because `docs/reference/api.md`
#: renders `::: hhemt.experiments` and `scripts/check_autodoc_coverage.py` derives
#: its expected set from each rendered module's `__all__`: without this the
#: directive contributes ZERO expected symbols and the gate certifies a surface
#: that excludes the class every tutorial's first line imports.
__all__ = [
    "TRITON_SWMM_experiment",
    "NorfolkIreneExperiment",
    "NorfolkObservedExperiment",
]


class TRITON_SWMM_experiment:
    """
    Generic example loader for TRITON-SWMM case studies.

    Provides both low-level initialization from YAML paths and high-level
    class methods for loading case studies from configuration templates.

    Attributes
    ----------
    system : TRITONSWMM_system
        Configured system instance with the analysis loaded.
    """

    def __init__(
        self,
        cfg_system_yaml: Path,
        cfg_analysis_yaml: Path,
        case_name: str | None = None,
        hpc_system_config_yaml: Path | None = None,
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
            hpc_system_config_yaml: Path to the per-cluster ``hpc_system_config.yaml``
                (ADR-6). The bundle NEVER carries this — ADR-9 makes the reproducer's
                HPC-specific config an external input by design — so ``from_doi``
                acquires it and passes it here. REQUIRED for a container-mode
                experiment (``execution_environment: container``): the ContainerSpec
                that renders ``apptainer exec {sif_path}`` lives on it
                (``config/hpc_system.py::ContainerSpec``). ``None`` (default) keeps
                today's native behavior byte-identical.
        """
        self.system = TRITONSWMM_system(cfg_system_yaml)
        self.analysis = TRITONSWMM_analysis(
            analysis_config_yaml=cfg_analysis_yaml,
            system=self.system,
            hpc_system_config_yaml=hpc_system_config_yaml,
        )
        # Link back, mirroring the link-back in `Toolkit.from_configs` and in the
        # test-case-builder fixture.
        # Without it `system.analysis` raises RuntimeError (system.py:218-221) and
        # `Toolkit(exp.system)` is impossible (Toolkit.__init__ reads system.analysis).
        # This constructor was the sole system+analysis pair-builder omitting it.
        self.system._analysis = self.analysis
        self.test_case_directory = self.retrieve_test_data_directory(case_name) if case_name is not None else None
        self.bundle_root: Path | None = None

    @classmethod
    def retrieve_test_data_directory(cls, case_name: str):
        """Locate the on-disk directory holding a case study's example data.

        Parameters
        ----------
        case_name : str
            Case-study directory name under ``test_data``, e.g.
            ``"norfolk_coastal_flooding"``.

        Returns
        -------
        Path
            The ``test_data/{case_name}/`` directory beside the installed
            package. The path is returned whether or not it exists; callers that
            need the data present are responsible for downloading it.
        """
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
        example_data_dir: Path | None = None,
    ):
        """
        Load a case study example from configuration templates.

        This is the high-level API for loading any case study. It handles:
        - Template filling with data directory paths
        - HydroShare data download
        - System and analysis configuration generation

        Parameters
        ----------
        case_name : str
            Case study name, for example ``norfolk_coastal_flooding``.
        system_config_template : str
            System config filename, for example ``template_system_config.yaml``.
        analysis_config_template : str
            Analysis config filename, for example ``template_analysis_config.yaml``.
        case_config_filename : str
            Case metadata filename, for example ``case.yaml``.
        download_if_exists : bool, default False
            Re-download the HydroShare data even when it is already present.
        example_data_dir : Path or None
            Override for the data storage location, when one is given.

        Returns
        -------
        TRITON_SWMM_experiment
            Instance with the system and analysis loaded.

        Examples
        --------
        >>> from hhemt.constants import *
        >>> example = TRITON_SWMM_experiment.from_case_study(
        ...     case_name=NORFOLK_EX,
        ...     system_config_template=NORFOLK_SYSTEM_CONFIG,
        ...     analysis_config_template=NORFOLK_ANALYSIS_CONFIG,
        ...     case_config_filename=NORFOLK_CASE_CONFIG,
        ... )
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
        hpc_system_config_yaml: Path | None = None,
        validate: bool = True,
        sif_build_config_yaml: Path | None = None,
    ) -> "TRITON_SWMM_experiment":
        """Fetch a published reprex bundle by DOI or PID, reconstitute it, and return
        a runnable experiment.

        Under the self-contained-by-default contract the emitted bundle carries every
        config-declared simulation input at its bundle-relative location, so a
        self-contained bundle reconstitutes to inputs that all exist on disk. That is
        a from-scratch runnable experiment, dependent externally only on the
        reproducer's user-specific and HPC-specific configs and on the container SIF,
        which is built on ingest from the carried ``.def`` recipe or transferred in a
        separate phase. A bundle emitted with a ``bundle_exclude_config`` opt-out
        carries the excluded inputs by reference, via an ``input_deposit`` block
        fetched on ingest.

        Trust boundary: ingesting a DOI and running it executes shell derived from the
        fetched config, so ingest only deposits you trust. ``expected_sha256`` pins the
        integrity of the fetched bundle zip.

        Parameters
        ----------
        doi : str or None
            The bundle's DOI, for example ``'10.5281/zenodo.123456'``. Give either
            ``doi`` or ``pid``.
        pid : str or None
            The host-native id: a Zenodo record id or a HydroShare resource id.
        host : {'zenodo', 'hydroshare'}
            Required, with no default. The sibling case-study fetch defaults to
            hydroshare, and two DOI entry points with opposite host defaults would be
            a trap.
        expected_sha256 : str or None
            Sha256 pin on the fetched bundle zip.
        target_dir : Path or None
            Directory to fetch into. Defaults to a fresh temporary directory.
        software_dir : Path or None
            Target-side directory for the toolkit-owned SWMM and TRITON build
            directories. Defaults to ``{bundle_root}/software``. These are build
            outputs the toolkit creates at setup rather than bundled inputs, and they
            must be non-null for ``TRITONSWMM_system`` to construct.
        hpc_system_config_yaml : Path or None
            Path to your cluster's ``hpc_system_config.yaml``. Resolution precedence
            is this argument, then ``$HHEMT_HPC_SYSTEM_CONFIG``, then None. The bundle
            never carries it, because a DOI-downloaded bundle is contracted to run
            with no external dependency beyond the reproducer's user-specific and
            HPC-specific configs. It is required for a container-mode bundle, because
            the ContainerSpec that renders ``apptainer exec {sif}`` lives on it. Start
            from a worked in-tree example under
            ``test_data/norfolk_coastal_flooding/`` or from the shape sketched in the
            bundle's ``hpc_system_config.template.yaml``.

            On the container path ``from_doi`` writes a derived copy at
            ``{software_dir}/hpc_system_config.resolved.yaml`` whose
            ``container.sif_path`` names the built or transferred SIF, and hands the
            analysis that path. A derived file rather than an in-memory edit is
            required: every downstream consumer re-loads the YAML from the path,
            including ``run_simulation_runner.main``, so an in-memory repoint would
            never reach the ``apptainer exec`` invocation in
            ``run_simulation.prepare_simulation_command``. Your original config is
            never modified.
        validate : bool, default True
            Run preflight validation on the reconstituted experiment before returning,
            mirroring ``Toolkit.from_configs``. This is what makes a container-mode
            bundle fail closed rather than silently degrade to a native run:
            ``preflight_validate`` calls ``_validate_container_config``, which errors
            when ``execution_environment == "container"`` and no ContainerSpec
            resolves. Without it the workflow builder sets an empty container prefix
            (``SnakemakeWorkflowBuilder.__init__``) and the experiment runs natively
            while reporting success. Pass False only to inspect a bundle you do not
            intend to run.
        sif_build_config_yaml : Path or None, default None
            The reproducer's ``sif_build_config`` (build venue + resources). REQUIRED when the
            bundle's carried images are not already present under the reproducer's
            ``container.sif_root``: each missing image is REBUILT through the same transaction
            the producer used, on THIS checkout — which must be at the carried ``hhemt_sha``
            (a different commit, or a wheel install, is refused; item 11).
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

        # ADR-6/ADR-9: the HPC config is the reproducer's, never bundle-carried.
        hpc_cfg_path = cls._resolve_hpc_system_config(hpc_system_config_yaml)

        # ADR-19: build (or fall back to transfer) the SIF, then repoint container.sif_path
        # at it via a DERIVED config copy. Container-mode only — a native bundle skips this
        # entirely and its behavior is byte-identical to today (R9).
        cfg_analysis_dict = read_yaml(analysis_config_path)
        if cfg_analysis_dict.get("execution_environment") == "container":
            if hpc_cfg_path is None:
                raise ConfigurationError(
                    field="hpc_system_config_yaml",
                    message=(
                        "This bundle is container-mode (execution_environment='container') "
                        "but no hpc_system_config was supplied. The bundle deliberately does "
                        "not carry one (ADR-9: the reproducer's HPC-specific config is an "
                        "external input). Pass hpc_system_config_yaml=... (or `hhemt ingest "
                        "--hpc-system-config ...`), or set $HHEMT_HPC_SYSTEM_CONFIG. Start "
                        "from a worked example: "
                        "test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml, or "
                        f"the shape sketched at {bundle_root}/hpc_system_config.template.yaml."
                    ),
                    config_path=None,
                )
            resolved_software_dir = software_dir or (bundle_root / "software")
            # SIF quest (ADR-21): the bundle carries the PRODUCER MANIFEST of every image its matrix
            # resolved to (bundle_manifest.json["sif_manifests"]). Place each at its identity path
            # under the reproducer's container.sif_root, rebuilding any that is absent through the
            # same transaction on THIS checkout (which must be at the carried hhemt_sha), then write
            # a DERIVED config that carries sif_root. No .def and no source tree are carried.
            from hhemt.config.hpc_system import hpc_system_config as _hpc_model
            from hhemt.config.loaders import yaml_to_model

            _hpc_dict = read_yaml(hpc_cfg_path) or {}
            _sif_root = Path(
                (_hpc_dict.get("container") or {}).get("sif_root") or (resolved_software_dir / "sifs")
            ).resolve()
            _mf = bundle_root / "bundle_manifest.json"
            _carried = (json.loads(_mf.read_text()) or {}).get("sif_manifests") if _mf.is_file() else None
            if not _carried:
                raise ConfigurationError(
                    field="bundle_manifest.json",
                    message=(
                        "container-mode bundle carries no sif_manifests: it was emitted by a pre-ADR-21 "
                        "toolkit (BUNDLE_SCHEMA_VERSION < 6) or before the producer ran hhemt build-sifs. "
                        "Re-emit from a producer whose images are built."
                    ),
                    config_path=None,
                )
            _cfg_hpc = yaml_to_model(hpc_cfg_path, _hpc_model)
            cls._assert_carried_covers_matrix(
                carried=_carried, cfg_hpc_system=_cfg_hpc, analysis_config_path=analysis_config_path
            )
            cls._place_or_rebuild_sifs(
                carried=_carried,
                sif_root=_sif_root,
                cfg_hpc_system=_cfg_hpc,
                sif_build_config_yaml=sif_build_config_yaml,
            )
            hpc_cfg_path = cls._set_sif_root(
                hpc_cfg_path,
                sif_root=_sif_root,
                target_path=resolved_software_dir / "hpc_system_config.resolved.yaml",
            )

        exp = cls(
            system_config_path,
            analysis_config_path,
            case_name=None,
            hpc_system_config_yaml=hpc_cfg_path,
        )
        exp.bundle_root = bundle_root
        # Fail closed on an unrunnable reconstitution — notably a container-mode bundle
        # whose ContainerSpec does not resolve, which would otherwise run NATIVELY and
        # report success, because `SnakemakeWorkflowBuilder.__init__` leaves the
        # container prefix empty in that case. Mirrors the preflight-validation block
        # in `Toolkit.from_configs`; sanctioned by `analysis.validate`, whose docstring
        # states that "CLI/API entry points can call it automatically".
        if validate:
            exp.analysis.validate().raise_if_invalid()
        return exp

    @classmethod
    def _resolve_hpc_system_config(cls, override: Path | None = None) -> Path | None:
        """Resolve the reproducer's hpc_system_config (ADR-6/ADR-9 external input).

        Precedence: explicit ``override`` > ``$HHEMT_HPC_SYSTEM_CONFIG`` > None. Mirrors
        the proven-green operator chain in
        ``scripts/experiments/container_validation.py::_resolve_hpc_system_config`` (:86-119),
        MINUS its ``$HHEMT_DEPLOYMENT_CONFIG/hpc/hpc_system_config_{cluster}.yaml`` tier —
        that tier needs a ``cluster`` name ``from_doi`` does not have, and the private-estate
        layout must not leak into library code.

        Returns None when neither source is set (the native path — unchanged behavior).
        Raises FileNotFoundError when a source names a path that does not exist, so a typo
        fails loudly rather than silently degrading to native.
        """
        import os

        if override is not None:
            path = Path(override).expanduser()
            source = "hpc_system_config_yaml argument"
        elif os.environ.get("HHEMT_HPC_SYSTEM_CONFIG"):
            path = Path(os.environ["HHEMT_HPC_SYSTEM_CONFIG"]).expanduser()
            source = "$HHEMT_HPC_SYSTEM_CONFIG"
        else:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"hpc_system_config not found at {path} (from {source}).")
        print(f"[Ingest] hpc_system_config: {path} (from {source})", flush=True)
        return path.resolve()

    @classmethod
    def _set_sif_root(cls, hpc_cfg_path: Path, *, sif_root: Path, target_path: Path) -> Path:
        """ADR-21: write a DERIVED hpc_system_config whose ``container.sif_root`` names the
        directory the carried images were placed under; the retired pointer keys are dropped so
        a carried pre-quest config does not trip the hard refusal. A derived FILE is required:
        every downstream consumer re-loads the YAML from the path it is handed. The user's
        original config is never modified."""
        cfg = read_yaml(hpc_cfg_path)
        container = dict(cfg.get("container") or {})
        container["sif_root"] = str(Path(sif_root).resolve())
        for _retired in ("sif_path", "sif_paths_by_arch", "sif_sha256"):
            container.pop(_retired, None)
        cfg["container"] = container
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(cfg, target_path)
        print(
            f"[Ingest] container.sif_root -> {sif_root}\n"
            f"[Ingest]   derived config: {target_path} (your original is unmodified)",
            flush=True,
        )
        return target_path

    @classmethod
    def _assert_carried_covers_matrix(cls, *, carried: list[dict], cfg_hpc_system, analysis_config_path: Path) -> None:
        """Fail closed BEFORE any build: every partition the reproducer's matrix requires must be
        covered by a carried identity with the same ``(mpi_family, accel, gpu_hardware)`` — the
        exact replacement of the retired cross-family arch guard (no override flag: an image built
        for other hardware is a wrong image, not a warning)."""
        from hhemt.bundle._emit import _matrix_required_partitions
        from hhemt.config.analysis import analysis_config
        from hhemt.config.hpc_system import resolve_gpu_target
        from hhemt.config.loaders import yaml_to_model
        from hhemt.sif.identity import SifIdentity, derive_family

        cfg_analysis = yaml_to_model(analysis_config_path, analysis_config)
        cspec = getattr(cfg_hpc_system, "container", None)
        if cspec is None:
            raise ConfigurationError(
                field="container",
                message="container-mode ingest with no container: block in your hpc_system_config",
                config_path=None,
            )
        mpi, accel = derive_family(cspec)
        have = {(i.mpi_family, i.accel, i.gpu_hardware) for i in (SifIdentity.from_carried(m) for m in carried)}
        missing = []
        for part in sorted(_matrix_required_partitions(cfg_analysis, cfg_hpc_system)):
            hw = resolve_gpu_target(cfg_hpc_system, part)[0] if accel != "cpu" else None
            if (mpi, accel, hw) not in have:
                missing.append(f"partition {part!r} needs ({mpi}, {accel}, {hw or 'cpu'})")
        if missing:
            raise ConfigurationError(
                field="hpc_system_config",
                message=(
                    "the carried images do not cover your matrix: "
                    + "; ".join(missing)
                    + f". Carried: {sorted(have)}. Build for your hardware with hhemt build-sifs (a fresh "
                    "identity), or ingest on a partition whose hardware the bundle covers."
                ),
                config_path=None,
            )

    @classmethod
    def _place_or_rebuild_sifs(
        cls, *, carried: list[dict], sif_root: Path, cfg_hpc_system, sif_build_config_yaml: Path | None
    ) -> None:
        """For every carried manifest: keep an image already at its identity path under
        ``sif_root`` whose manifest records the carried digest; otherwise REBUILD it through the
        standard build DAG (Specs 9-13) on this checkout, refusing when the running toolkit is not
        at the carried ``hhemt_sha``. Runs the DAG in the FOREGROUND (an ingest is an operator
        action and `hhemt ingest` is its login-node driver); no experiment is submitted here."""
        from hhemt.config.loaders import yaml_to_model
        from hhemt.config.sif_build import sif_build_config
        from hhemt.sif.driver import run_build_dag
        from hhemt.sif.identity import SifIdentity, manifest_path, resolve_sif
        from hhemt.sif.plan import SifBuildPlan, live_sentinels
        from hhemt.sif.snakefile_generator import reconcile_sif_root, write_sif_snakefile
        from hhemt.validation import _running_toolkit_sha_full

        plan = SifBuildPlan()
        for m in carried:
            ident = SifIdentity.from_carried(m)
            sif = resolve_sif(sif_root, ident)
            man = manifest_path(sif)
            if sif.is_file() and man.is_file() and json.loads(man.read_text()).get("sha256") == m.get("sha256"):
                continue
            plan.entries[ident.key] = ident
            plan.covers[ident.key] = [("carried", ident.gpu_hardware or "cpu")]
        if not plan.entries:
            print(f"[Ingest] every carried image is present under {sif_root}", flush=True)
            return
        running = _running_toolkit_sha_full()
        wrong = sorted({i.hhemt_sha for i in plan.entries.values() if i.hhemt_sha != running})
        if wrong:
            raise ConfigurationError(
                field="toolkit",
                message=(
                    f"the carried images were built from hhemt {wrong[0][:12]} but the running toolkit is "
                    f"{(running or 'a wheel install (no sha)')[:12]}. Rebuilding an identical image requires a git "
                    f"checkout at that commit: git checkout {wrong[0]} && pip install -e . ; then re-run the ingest."
                ),
                config_path=None,
            )
        if sif_build_config_yaml is None:
            raise ConfigurationError(
                field="sif_build_config_yaml",
                message=(
                    f"{len(plan.entries)} carried image(s) are absent under {sif_root} and must be rebuilt; "
                    "pass sif_build_config_yaml=<your sif_build_config> (build partition/account/resources)."
                ),
                config_path=None,
            )
        cfg = yaml_to_model(Path(sif_build_config_yaml), sif_build_config)
        for key in plan.entries:
            if live_sentinels(sif_root, key):
                raise ConfigurationError(
                    field="sif_root",
                    message=f"identity {key} has a LIVE producer under {sif_root}; wait for it.",
                    config_path=None,
                )
        reconcile_sif_root(sif_root, walltime_min=cfg.walltime_min, planned_keys=set(plan.entries))
        snakefile = write_sif_snakefile(plan, sif_root=sif_root, build_host=cfg_hpc_system, sif_build_cfg=cfg)
        rc = run_build_dag(snakefile, sif_root=sif_root, keys=set(plan.entries))
        if rc != 0:
            raise ProcessingError(
                operation="ingest SIF build",
                filepath=snakefile,
                reason=f"the build DAG exited {rc}; see {sif_root}/_build and the SLURM logs",
            )
        still = [k for k, i in plan.entries.items() if not manifest_path(resolve_sif(sif_root, i)).is_file()]
        if still:
            raise ProcessingError(
                operation="ingest SIF build",
                filepath=sif_root,
                reason=f"the build DAG exited 0 but no manifest landed for {still}",
            )

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
                reason=(f"expected exactly one bundle .zip in the fetched deposit, found {len(candidates)}: {rel}"),
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
                reason=(f"sha256 mismatch on the fetched bundle: expected {expected_sha256}, got {actual}"),
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

    @classmethod
    def _materialize_input_deposits(cls, bundle_root: Path, missing: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Fetch the excluded inputs a bundle carries BY REFERENCE (ADR-20, as amended).

        This is outcome 2 of the three-outcome materialize gate. For each declared-but-absent
        input, look up its ``input_deposit`` block in ``bundle_manifest.json``:

        - block present WITH a ``contentUrl`` -> fetch to ``bundle_root/{relpath}`` and
          sha256-verify. A fetch error or a digest mismatch is a HARD failure — never a
          silent skip, because a silently-wrong input would produce plausible-looking
          garbage results hours later.
        - block present WITHOUT a ``contentUrl`` -> **referenced-but-unfetchable-by-design**
          (outcome 3). Left in the returned still-missing list, to be reported with its
          citation. This is the CORRECT terminal state for licensed/IP data, not a bug.
        - no block at all -> still missing (a genuinely broken bundle).

        The fetch routes through ``_fetch_file_by_url`` — the per-file URL seam — and NOT
        through ``_fetch_deposit_files``, which is a WHOLE-RECORD fetcher (its Zenodo arm
        loops every entry in ``files[]``; its HydroShare arm downloads the entire bag and
        returns a directory). Routing a per-input fetch through it would force a consumer to
        download the ENTIRE deposit to obtain ONE excluded input — defeating the purpose of
        an opt-out that exists precisely because the data is too large to ship.

        Returns:
            The inputs still missing after fetching, as ``(field, path)`` pairs.
        """
        manifest_path = bundle_root / "bundle_manifest.json"
        if not manifest_path.exists():
            return missing
        deposits = json.loads(manifest_path.read_text()).get("input_deposit") or []
        if not deposits:
            return missing

        # Key by the ABSOLUTE on-disk path the reconstituted cfg points at, so the lookup
        # matches what the gate found missing. `relpath` is emitted via the same
        # _rewrite_absolute_to_relative the cfg rewrite uses, so this join is exact.
        by_abspath = {str((bundle_root / d["relpath"]).resolve()): d for d in deposits}

        still_missing: list[tuple[str, str]] = []
        for field_name, path_str in missing:
            block = by_abspath.get(str(Path(path_str).resolve()))
            if block is None:
                still_missing.append((field_name, path_str))
                continue
            url = block.get("contentUrl")
            if not url:
                still_missing.append((field_name, path_str))  # outcome 3 — by design
                continue
            dest = bundle_root / block["relpath"]
            try:
                cls._fetch_file_by_url(url, dest, expected_sha256=block.get("sha256"))
            except Exception as exc:  # noqa: BLE001 — any fetch/digest failure is terminal
                raise ProcessingError(
                    operation="doi_ingest_input_deposit",
                    filepath=dest,
                    reason=(
                        f"the bundle carries '{field_name}' BY REFERENCE, but fetching it "
                        f"failed: {exc}\n"
                        f"  contentUrl: {url}\n"
                        f"  citation:   {block.get('citation', '(none)')}\n"
                        "The reference may have rotted. Obtain the file from the citation "
                        f"above and place it at {dest}, then re-run."
                    ),
                ) from exc
        return still_missing

    @classmethod
    def _assert_declared_inputs_exist(cls, system_config_path: Path, analysis_config_path: Path) -> None:
        """Fail-closed materialize gate: raise ``ProcessingError`` naming EVERY
        reconstituted input Path that does not exist on disk.

        The reconstituted configs carry ABSOLUTE resolved paths under ``bundle_root``.
        ``cfgBaseModel._check_paths_exist`` now runs in ``mode='after'`` and WOULD fire on
        these paths — but nothing constructs a model over them before this gate, for two
        independent reasons, and BOTH must hold for that to stay true. (1) The functions
        that produce them, ``reconstitute_runnable_config`` and
        ``reconstitute_runnable_analysis_config``, read the bundle archive with a bare
        ``yaml.safe_load`` and write it back with ``yaml.safe_dump`` — they construct no
        model, so no validator sees the values. (2) Each file's first model construction
        happens LATER: ``system_config.yaml`` in ``TRITONSWMM_system(...)`` and
        ``analysis_config.yaml`` in ``TRITONSWMM_analysis(...)``, both after this gate has
        run. This gate is therefore the first and only check they meet, and it is the only
        one that fetches via ``contentUrl``, sha256-verifies, and reports citation +
        place-it-at. Do NOT read this as "every bundle-archive read is covered" — it is
        not, and the two bare reads above are the counterexamples. It is an ORDERING
        property of THIS call path.

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
        missing: list[tuple[str, str]] = []
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
                        missing.append((name, str(v)))

        # Outcome 2 (fetchable): an absent input with an input_deposit block carrying a
        # contentUrl is fetched + sha256-verified here, BEFORE the fail-closed enumeration.
        # bundle_root is the configs' parent (both reconstituted configs sit at the root).
        if missing:
            bundle_root = Path(system_config_path).parent
            missing = cls._materialize_input_deposits(bundle_root, missing)

        if missing:
            # Outcome 3 (referenced-but-unfetchable-by-design) + genuinely-broken bundles.
            # Report each with its citation so the operator can actually obtain the file —
            # a bare "missing input" message is useless for licensed data, which is exactly
            # the case the exclude-config exists to serve.
            blocks: dict[str, dict] = {}
            manifest_path = Path(system_config_path).parent / "bundle_manifest.json"
            if manifest_path.exists():
                root = Path(system_config_path).parent
                for d in json.loads(manifest_path.read_text()).get("input_deposit") or []:
                    blocks[str((root / d["relpath"]).resolve())] = d

            lines: list[str] = []
            for name, path_str in sorted(missing):
                block = blocks.get(str(Path(path_str).resolve()))
                if block is None:
                    lines.append(f"{name}: {path_str}\n      (not carried, and no input_deposit record)")
                    continue
                lines.append(
                    f"{name}: {path_str}\n"
                    f"      REFERENCED, not carried (no direct download is available for it).\n"
                    f"      how to obtain: {block.get('citation', '(no citation supplied)')}\n"
                    + (f"      landing page:  {block['url']}\n" if block.get("url") else "")
                    + (f"      identifier:    {block['identifier']}\n" if block.get("identifier") else "")
                    + f"      sha256:        {block.get('sha256', '(none)')}\n"
                    f"      place it at:   {path_str}"
                )

            raise ProcessingError(
                operation="doi_ingest_inputs",
                filepath=None,
                reason=(
                    "the reconstituted experiment declares inputs that do not exist on "
                    "disk. Inputs marked REFERENCED were deliberately excluded from the bundle "
                    "(licensed, restricted, or oversized data the depositor could not "
                    "redistribute) — obtain each from its citation below and place it at "
                    "the stated path, then re-run. This is a fail-closed stop, not a "
                    "corrupt bundle.\n  " + "\n  ".join(lines)
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
        example_data_dir: Path | None = None,
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
        cfg_yaml = Path(cfg_system.system_directory) / f"config_analysis_{analysis_id}.yaml"
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml

    @classmethod
    def _fill_case_analysis_yaml(
        cls,
        app_name: str,
        case_name: str,
        analysis_config_template: str,
        example_data_dir: Path | None = None,
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
        filled_yaml_data = cls._return_filled_template_yaml_dictionary(cfg_template, mapping)
        return filled_yaml_data

    @classmethod
    def _load_case_system_config(
        cls,
        case_name: str,
        system_config_template: str,
        case_config_filename: str,
        download_if_exists: bool,
        example_data_dir: Path | None = None,
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
        filled_yaml_data = cls._return_filled_template_yaml_dictionary(cfg_template, mapping)
        # PRE-DOWNLOAD: schema/enum/toggle-dependency validity only. The declared inputs
        # legitimately do not exist yet -- this call PRECEDES the download branch below
        # that creates them, so a `runnable` intent here is a deadlock: validating the
        # pre-download config is what would block the download that makes it valid.
        # The post-download `runnable` load two blocks down is the discharge.
        system_config.model_validate(filled_yaml_data, context={"existence": "template"})

        # download data if it doesn't exist
        if Path(mapping["DATA_DIR"]).exists() and not download_if_exists:
            pass
        else:
            if verbose:
                print(f"Download example data to {mapping['DATA_DIR']} using Hydroshare")
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

        # POST-DOWNLOAD: the deferred existence obligation is discharged here. This is
        # the check that the fetch delivered what the config declares -- it has no
        # counterpart in the pre-fix code, where the validator was inert on the str
        # values a YAML load produces.
        load_system_config_from_dict(filled_yaml_data)

        cfg_yaml = Path(filled_yaml_data["system_directory"]) / "config_system.yaml"
        cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(filled_yaml_data, cfg_yaml)
        return cfg_yaml

    @classmethod
    def _get_case_data_and_package_directory_mapping_dict(
        cls,
        case_name: str,
        example_data_dir: Path | None = None,
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
        package_dir = get_package_root(cnst.APP_NAME).parents[1] / "test_data" / case_name

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
                    reason=(f"cannot resolve a Zenodo record id from doi={doi!r} pid={pid!r}"),
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
            import os  # host resolution mirrors publishing._ZenodoTarget.publish

            import requests  # explicit dep declared in pyproject

            base = os.environ.get("HHEMT_ZENODO_BASE_URL", "https://zenodo.org").rstrip("/")
            resp = requests.get(f"{base}/api/records/{recid}", timeout=60)
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
                top_level_dirs = {Path(f).parts[0] for f in z.namelist() if Path(f).parts}
            if len(top_level_dirs) == 1:
                bag_root = dest.parent / next(iter(top_level_dirs))
            else:
                raise ProcessingError(
                    operation="hydroshare_bag_extract",
                    filepath=str(zip_path),
                    reason=("ZIP has multiple top-level folders; cannot determine Bag root."),
                )
            if validate:
                bag = bagit.Bag(str(bag_root))
                if bag.is_valid():
                    print("Bag verified! All bagit checksums match.", flush=True)
                else:
                    raise ProcessingError(
                        operation="hydroshare_bag_validation",
                        filepath=str(bag_root),
                        reason=("bagit manifest validation failed (bag is not self-consistent)."),
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
    def _fetch_file_by_url(cls, url: str, dest: Path, *, expected_sha256: str | None = None) -> Path:
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

        Three tiers, cheapest and least-privileged first:

        1. **Anonymous** — a PUBLIC resource is downloadable with NO credentials
           (verified: GET /hsapi/resource/{id}/ returns 200 for a public resource).
        2. **Env-credentialed (non-interactive)** — when the anonymous read fails
           (a private/unshared resource) AND ``HHEMT_HYDROSHARE_USERNAME`` +
           ``HHEMT_HYDROSHARE_PASSWORD`` are set, authenticate with them. This is the
           symmetric read-side counterpart to ``publishing._HydroShareTarget.publish``
           (which already constructs ``HydroShare(username=…, password=…)`` from the same
           env vars), and it is what makes PRIVATE-resource retrieval work HEADLESSLY — in
           a batch job, on an HPC node, or in an automated test — none of which can answer
           an interactive prompt.
        3. **Interactive sign-in** — the last resort, for a human at a terminal with no env
           credentials configured.
        """
        import os

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
            # res_identifier is diagnosable rather than silently masked below.
            username = os.environ.get("HHEMT_HYDROSHARE_USERNAME")
            password = os.environ.get("HHEMT_HYDROSHARE_PASSWORD")
            if username and password:
                # Tier 2: non-interactive credentialed auth (private + headless).
                print(
                    f"Anonymous read of {res_identifier} failed ({exc!r}); "
                    "authenticating with HHEMT_HYDROSHARE_* credentials.",
                    flush=True,
                )
                try:
                    hs_auth = HydroShare(username=username, password=password)
                    hs_auth.resource(res_identifier, validate=True)
                except Exception as auth_exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"HydroShare credentialed read of {res_identifier} failed "
                        f"({auth_exc!r}) after the anonymous read failed ({exc!r}). Check "
                        "HHEMT_HYDROSHARE_USERNAME / HHEMT_HYDROSHARE_PASSWORD and that the "
                        "account has access to the resource. (The username may be your bare "
                        "id or your full institutional email — try the other if one fails.)"
                    ) from auth_exc
                print("Authenticated to HydroShare with env credentials.", flush=True)
                return hs_auth
            # Tier 3: interactive sign-in — a human at a terminal, no env creds set.
            print(
                f"Anonymous read of {res_identifier} failed ({exc!r}); no "
                "HHEMT_HYDROSHARE_* credentials set — falling back to interactive sign-in.",
                flush=True,
            )
            try:
                hs.sign_in()
            except Exception as sign_in_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"HydroShare sign-in failed after anonymous read of "
                    f"{res_identifier} failed ({exc!r}). For a headless/automated context, "
                    "set HHEMT_HYDROSHARE_USERNAME / HHEMT_HYDROSHARE_PASSWORD to avoid the "
                    "interactive prompt entirely."
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

    Examples
    --------
    >>> from hhemt.experiments import NorfolkIreneExperiment
    >>> norfolk = NorfolkIreneExperiment.load()
    >>> system = norfolk.system
    """

    @classmethod
    def load(
        cls,
        download_if_exists: bool = False,
        example_data_dir: Path | None = None,
    ) -> TRITON_SWMM_experiment:
        """
        Load Norfolk coastal flooding example.

        Parameters
        ----------
        download_if_exists : bool, default False
            Re-download the HydroShare data even when it is already present.
        example_data_dir : Path or None
            Override for the data directory, when one is given.

        Returns
        -------
        TRITON_SWMM_experiment
            Instance with the Norfolk system loaded.
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

    Examples
    --------
    >>> from hhemt.experiments import NorfolkObservedExperiment
    >>> norfolk = NorfolkObservedExperiment.load()
    >>> system = norfolk.system
    """

    @classmethod
    def load(
        cls,
        download_if_exists: bool = False,
        example_data_dir: Path | None = None,
    ) -> TRITON_SWMM_experiment:
        """
        Load Norfolk coastal flooding example.

        Parameters
        ----------
        download_if_exists : bool, default False
            Re-download the HydroShare data even when it is already present.
        example_data_dir : Path or None
            Override for the data directory, when one is given.

        Returns
        -------
        TRITON_SWMM_experiment
            Instance with the Norfolk system loaded.
        """

        # this method just changes the weather_events_to_simulate
        # for analysis config

        weather_events_to_simulate = "obs_event_summaries_from_yrs_with_complete_coverage.csv"
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
