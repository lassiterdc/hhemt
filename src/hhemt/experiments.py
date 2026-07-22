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
        # Link back, mirroring toolkit.py:152 and tests/fixtures/test_case_builder.py:189.
        # Without it `system.analysis` raises RuntimeError (system.py:218-221) and
        # `Toolkit(exp.system)` is impossible (toolkit.py:96 reads system.analysis).
        # This constructor was the sole system+analysis pair-builder omitting it.
        self.system._analysis = self.analysis
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
        hpc_system_config_yaml: Path | None = None,
        validate: bool = True,
        allow_cross_family_sif: bool = False,
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
            hpc_system_config_yaml: Path to YOUR cluster's ``hpc_system_config.yaml``
                (ADR-6). Resolution precedence: this argument > ``$HHEMT_HPC_SYSTEM_CONFIG``
                > None. The bundle never carries it (ADR-9: a DOI-downloaded bundle runs
                "with no external dependency aside from the reproducer's USER-specific and
                HPC-specific configs"), and it is REQUIRED for a container-mode bundle —
                the ContainerSpec that renders ``apptainer exec {sif}`` lives on it.
                Start from a worked in-tree example
                (``test_data/norfolk_coastal_flooding/hpc_system_config_{uva,frontier}.yaml``)
                or from the shape sketched in the bundle's
                ``hpc_system_config.template.yaml``.
                ADR-19 repoint: on the container path ``from_doi`` writes a DERIVED copy at
                ``{software_dir}/hpc_system_config.resolved.yaml`` whose
                ``container.sif_path`` names the built (or transferred) SIF, and hands the
                analysis THAT path. A derived FILE — not an in-memory edit — is required:
                every downstream consumer re-loads the YAML from the path
                (``analysis.test()`` at analysis.py:2728; the sim runner at
                run_simulation_runner.py:219), so an in-memory repoint would never reach
                ``run_simulation.py:421``'s ``apptainer exec {cspec.sif_path}``. Your
                original config is never modified.
            validate: Run preflight validation on the reconstituted experiment before
                returning (default True), mirroring ``Toolkit.from_configs``
                (toolkit.py:154-157). This is what makes a container-mode bundle FAIL
                CLOSED rather than silently degrade to a native run: ``preflight_validate``
                → ``_validate_container_config`` (validation.py:1552) errors when
                ``execution_environment == "container"`` and no ContainerSpec resolves.
                Without it, ``workflow.py:807`` sets an empty container prefix and the
                experiment runs NATIVELY while reporting success. Pass False only to
                inspect a bundle you do not intend to run.
            allow_cross_family_sif: Override the ADR-19(vii) cross-family SIF guard
                (default False = fail closed). When True, a bundle whose baked GPU arch
                (``container_build.target_arch``) does not match your target partition's
                ``gpu_hardware`` is built + run anyway with only a warning. Set True ONLY
                when you have confirmed the baked arch is run-compatible with your GPU.
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
            # defect-8 guard (ADR-19 dogfood): a container-mode ingest on a SLURM host submits
            # a `sbatch --wait` SIF build that `cd`s into bundle_root (container_build.py:156)
            # from a COMPUTE node. If bundle_root sits under the system temp dir (the node-local
            # mkdtemp default), that compute node cannot see it and the build dies instantly with
            # a cryptic `cd: No such file or directory`. Refuse loud, BEFORE the ~1.6 h build.
            # Inert on the fixed path: a shared target_dir is not under gettempdir(); a non-SLURM
            # host does a same-node local build and is unaffected.
            import tempfile as _tempfile

            from hhemt.container_build import _slurm_available

            _sys_tmp = Path(_tempfile.gettempdir()).resolve()
            if _slurm_available() and bundle_root.resolve().is_relative_to(_sys_tmp):
                raise ConfigurationError(
                    field="target_dir",
                    message=(
                        f"The reprex bundle was extracted under the system temp dir "
                        f"({bundle_root}), which is node-local on an HPC cluster. This "
                        f"container-mode ingest will submit a SLURM SIF build that `cd`s into "
                        f"the bundle from a COMPUTE node, which cannot see a login/orchestrator "
                        f"temp dir. Pass target_dir=... on a SHARED filesystem "
                        f"(e.g. /scratch/$USER/...) — or set $TMPDIR to shared scratch — so the "
                        f"build context is visible cluster-wide."
                    ),
                    config_path=None,
                )
            # The build account is the REPRODUCER's, read from their own config — never
            # defaulted (a default would submit against the producer's allocation). The
            # apptainer BUILD module is sourced from the same config's ContainerSpec (no
            # src/ site literal); None => the build script emits no `module load` and the
            # in-script `command -v apptainer` guard governs.
            _hpc_dict = read_yaml(hpc_cfg_path) or {}
            account = _hpc_dict.get("default_account")
            apptainer_module = (_hpc_dict.get("container") or {}).get("apptainer_module")
            # ADR-19(vii): fail closed BEFORE the ~1.6 h build if the bundle's baked GPU
            # arch does not match the reproducer's target partition (silent wrong-physics).
            from hhemt.config.hpc_system import hpc_system_config as _hpc_model
            from hhemt.config.loaders import yaml_to_model

            _mf = bundle_root / "bundle_manifest.json"
            _cb_raw = (
                (json.loads(_mf.read_text()) or {}).get("container_build")
                if _mf.is_file()
                else None
            )
            # Back-compat: a pre-multi-SIF bundle carries a single dict; normalize to a list.
            _blocks = (
                _cb_raw if isinstance(_cb_raw, list) else ([_cb_raw] if _cb_raw else [])
            )
            _cfg_hpc_model = yaml_to_model(hpc_cfg_path, _hpc_model)
            # ADR-19(vii) D-E guard, generalized to SET-CONTAINMENT (multi-SIF, Option A):
            # every distinct GPU arch the matrix requires (gpu_hardware namespace) must be
            # covered by a carried .def BEFORE any ~1.6 h build.
            cls._assert_container_arch_set_covers_matrix(
                carried_blocks=_blocks,
                cfg_hpc_system=_cfg_hpc_model,
                analysis_config_path=analysis_config_path,
                bundle_root=bundle_root,
                allow_cross_family=allow_cross_family_sif,
            )
            # Build every carried .def into its arch-keyed cache slot; construct the
            # {gpu_hardware: sif} map for the SIM rung.
            sif_paths_by_arch: dict[str, str] = {}
            _cpu_slot_sif = None  # the CPU/no-arch def's SIF (block with no target_arch)
            _first_sif = None  # any built SIF (GPU-only-bundle fallback)
            for _blk in _blocks:
                _sif = cls._build_or_fetch_sif(
                    bundle_root,
                    account=account,
                    apptainer_module=apptainer_module,
                    container_block=_blk,
                )
                _first_sif = _first_sif or _sif
                _arch = _blk.get("target_arch")
                if _arch:
                    sif_paths_by_arch[_arch] = str(_sif)  # gpu_hardware-keyed (a100/a6000)
                else:
                    _cpu_slot_sif = _sif  # the CPU/no-arch def
            # DELTA-4: sif_path serves BOTH the arch-agnostic PROCESS rung AND the CPU-SIM
            # fallback (run_simulation.py: _row_hw is None -> _sif = sif_path). A GPU SIF
            # cannot run a CPU row (Kokkos DefaultExecutionSpace=Cuda, OpenMP OFF), so prefer
            # the CPU-slot SIF as the representative when the bundle carries one; a GPU-only
            # bundle has no CPU rows, so any SIF serves the (arch-agnostic) process rung.
            _representative_sif = _cpu_slot_sif or _first_sif
            hpc_cfg_path = cls._repoint_sif_paths(
                hpc_cfg_path,
                sif_path=_representative_sif,
                sif_paths_by_arch=sif_paths_by_arch,
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
        # report success (workflow.py:795-807). Mirrors toolkit.py:154-157; sanctioned by
        # analysis.py:622-624 ("CLI/API entry points can call it automatically").
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
            raise FileNotFoundError(
                f"hpc_system_config not found at {path} (from {source})."
            )
        print(f"[Ingest] hpc_system_config: {path} (from {source})", flush=True)
        return path.resolve()

    @classmethod
    def _repoint_sif_paths(
        cls,
        hpc_cfg_path: Path,
        *,
        sif_path: Path,
        sif_paths_by_arch: dict[str, str],
        target_path: Path,
    ) -> Path:
        """ADR-19 repoint (multi-SIF, Option A): write a DERIVED hpc_system_config whose
        ``container.sif_path`` names a representative on-ingest-built SIF (the arch-agnostic
        PROCESS/default SIF, read by the process rung workflow.py:805) AND whose
        ``container.sif_paths_by_arch`` maps each gpu_hardware ("a100"/"a6000") to its built
        SIF (consumed at the SIM rung run_simulation.py:421). Return the derived path.

        A derived FILE is required, not an in-memory edit: every downstream consumer
        re-loads the YAML from the path it was handed — ``analysis.test()`` passes
        ``self.hpc_system_config_yaml`` to each ``_test/`` sub (analysis.py:2728), and the
        sim runner rebuilds the analysis from ``--hpc-system-config`` in a fresh subprocess
        (run_simulation_runner.py:214-219) before ``run_simulation.py:421`` renders
        ``apptainer exec {sif}``. An in-memory mutation is invisible to all of it.

        Mirrors ``bundle/_emit.py::reconstitute_runnable_config`` (:659-721): read → repoint
        the fields knowable only at target-side ingest → ``yaml.safe_dump`` → return the path.
        The write is UNCONDITIONAL, so a resolved config from a prior ingest cannot go stale.
        The user's original config is never modified. With a single-arch bundle the map has
        one entry and sif_path IS that SIF — byte-identical to the pre-multi-SIF repoint.
        """
        cfg = read_yaml(hpc_cfg_path)
        container = dict(cfg.get("container") or {})
        container["sif_path"] = str(Path(sif_path).resolve())  # process/default (arch-agnostic)
        container["sif_paths_by_arch"] = {
            a: str(Path(p).resolve()) for a, p in sif_paths_by_arch.items()
        }
        cfg["container"] = container
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(cfg, target_path)
        print(
            f"[Ingest] repointed container.sif_path -> {sif_path}\n"
            f"[Ingest]   per-arch SIM SIFs -> {sorted(sif_paths_by_arch)}\n"
            f"[Ingest]   derived config: {target_path} (your original is unmodified)",
            flush=True,
        )
        return target_path

    @classmethod
    def _build_or_fetch_sif(
        cls,
        bundle_root: Path,
        *,
        account: str | None = None,
        apptainer_module: str | None = None,
        container_block: dict | None = None,
    ) -> Path:
        """ADR-19: build the SIF from the bundle's carried ``.def``.

        Reads the bundle manifest's ``container_build`` block for the digest-pinned recipe
        and delegates to ``container_build.get_or_build_sif`` (content-addressed cache
        OUTSIDE ``bundle_root`` — that tree is rmtree'd on every ingest, ``_reprex.py:156-159``).

        Raises ``ConfigurationError`` when the SIF cannot be built here (no rootless-fakeroot
        capability), naming the exact ``hhemt build-sif`` command and the manual ADR-2
        recourse (build off-site, point ``container.sif_path`` at the result).
        """
        from hhemt.container_build import SifBuildUnavailable, get_or_build_sif

        manifest_path = bundle_root / "bundle_manifest.json"
        # Multi-SIF (Option A): callers pass the specific per-arch block via container_block.
        # container_block=None is the legacy single-block path (read the manifest's one block;
        # normalize a list-carrying manifest to its first block for back-compat).
        block: dict = container_block if container_block is not None else {}
        if container_block is None and manifest_path.is_file():
            import json

            _cb = (json.loads(manifest_path.read_text()) or {}).get("container_build")
            block = (
                _cb
                if isinstance(_cb, dict)
                else (_cb[0] if isinstance(_cb, list) and _cb else {})
            )
        def_relpath = block.get("def_relpath")
        if not def_relpath:
            raise ConfigurationError(
                field="container_build.def_relpath",
                message=(
                    "This bundle is container-mode but its manifest carries no "
                    "`container_build` block, so there is no .def recipe to build from and "
                    "no ADR-2 transfer reference to fall back to. It was emitted before the "
                    "ADR-19 carriage landed. Build a SIF yourself and point "
                    "hpc_system_config.container.sif_path at it:\n"
                    "  hhemt build-sif --def <your.def> --sif-out <path>"
                ),
                config_path=str(manifest_path),
            )
        def_path = bundle_root / def_relpath
        lock_rel = block.get("source_tree_relpath")
        lock_path = (
            bundle_root / lock_rel / (block.get("source_tree_lock") or "uv.lock")
            if lock_rel
            else bundle_root / "uv.lock"
        )
        try:
            return get_or_build_sif(
                def_path=def_path,
                base_image_digest=block.get("base_image_digest", ""),
                lock_path=lock_path,
                target_arch=block.get("target_arch", "unknown"),
                account=account,
                apptainer_module=apptainer_module,
            )
        except SifBuildUnavailable as exc:
            # ADR-19 D-A: there is NO automatic transfer fallback. The producer-side keys
            # it would have read (transfer_url / sif_download_url / recorded_sif_sha256) are
            # never emitted by any code path (_emit.py:844-849), and the branch's sha256
            # gate was fail-OPEN (skipped whenever the never-written key was absent). The
            # reproducer's real recourse is the manual ADR-2 path: build the SIF off-site
            # and point container.sif_path at it. (SifBuildUnavailable is now dormant — the
            # preflight that raised it is deleted; a real build failure raises ProcessingError
            # and propagates LOUD, the correct fail-closed for a reproducibility tool.)
            raise ConfigurationError(
                field="container_build",
                message=(
                    f"Cannot build a container for this bundle here.\n\n{exc}\n\n"
                    "Build the SIF on a host with rootless-fakeroot capability and point "
                    "hpc_system_config.container.sif_path at the result:\n"
                    f"  hhemt build-sif --def {def_path} --sif-out <path>"
                ),
                config_path=None,
            ) from exc

    @classmethod
    def _assert_container_arch_set_covers_matrix(
        cls,
        *,
        carried_blocks: list[dict],
        cfg_hpc_system,
        analysis_config_path: Path,
        bundle_root: Path,
        allow_cross_family: bool = False,
    ) -> None:
        """ADR-19(vii) D-E guard, generalized to SET-CONTAINMENT (multi-SIF, Option A):
        every distinct GPU arch the reproducer's matrix requires (gpu_hardware namespace,
        e.g. "a100"/"a6000") must be covered by a carried .def BEFORE any ~1.6 h build.

        The bundle carries one ``container_build`` block per .def, each with a
        ``target_arch`` in the gpu_hardware namespace (DELTA-1: sourced from the .def's
        ``org.hhemt.gpu_hardware`` label). The ``.def`` cross-compiles the Kokkos/CUDA
        binary FOR that arch (``nvcc`` needs no device), so running a SIF on a different-arch
        GPU is a silent wrong-physics path. This guard recomputes the SAME
        ``resolve_gpu_target`` over the reproducer's matrix partitions and asserts the
        required arch set is a subset of the carried arch set BEFORE any build.

        REQUIRED = {resolve_gpu_target(reproducer_cfg, p)[0]} over every matrix partition p —
        the master ``hpc_ensemble_partition`` plus, for a sensitivity analysis, each distinct
        per-row ``hpc.partition`` / ``analysis.hpc_ensemble_partition`` value in the carried
        setup CSV. CARRIED = {block.target_arch} over the carried blocks. None/CPU arches
        (partitions with no GPU) are skipped — a CPU row runs the CPU/no-arch SIF via the SIM
        rung's ``sif_path`` fallback, not the per-arch map. ``allow_cross_family=True``
        downgrades a gap to a warning. Zero-user-info: the message names only arch strings +
        the reproducer's own partition names, never a producer path.

        The single-arch case (a100->a100, the [Q8] gating chain) is required={"a100"} ⊆
        carried={"a100"} — passes, exactly as the prior equality guard did. The per-row
        enumeration is best-effort: if the carried setup CSV cannot be read, the guard
        degrades to the master-partition check (never a false-close), and the [Q8] per-arch
        PASS gate remains the ultimate routing check.
        """
        from hhemt.config.hpc_system import resolve_gpu_target

        carried = {
            b["target_arch"]
            for b in carried_blocks
            if b.get("target_arch") and b["target_arch"] != "unknown"
        }
        # Enumerate the matrix's partitions (master + best-effort sensitivity sub-rows).
        cfg = read_yaml(analysis_config_path) or {}
        partitions: set[str] = set()
        _master = cfg.get("hpc_ensemble_partition")
        if _master:
            partitions.add(str(_master))
        if cfg.get("toggle_sensitivity_analysis"):
            try:
                _ref = cfg.get("sensitivity_analysis")
                if _ref:
                    _setup = Path(_ref)
                    if not _setup.is_absolute():
                        _setup = bundle_root / _ref
                    if _setup.is_file():
                        import pandas as pd

                        _df = pd.read_csv(_setup)
                        for _col in ("hpc.partition", "analysis.hpc_ensemble_partition"):
                            if _col in _df.columns:
                                partitions |= {
                                    str(v)
                                    for v in _df[_col].dropna().tolist()
                                    if str(v).strip()
                                }
            except Exception:
                pass  # degrade to master-only — never false-close on a setup-read failure
        required = {
            hw
            for hw in (resolve_gpu_target(cfg_hpc_system, p)[0] for p in partitions)
            if hw
        }
        missing = required - carried
        if not missing:
            return  # every required arch is covered (incl. the a100->a100 [Q8] chain)

        _missing = ", ".join(sorted(missing))
        _carried = ", ".join(sorted(carried)) or "(none)"
        detail = (
            f"This cross-hardware bundle carries container .defs for GPU arch(es) "
            f"[{_carried}], but your matrix's partitions require [{_missing}] that no "
            f"carried .def provides. On-ingest SIF building is per-arch; a row whose arch "
            f"has no matching SIF cannot run its own binary."
        )
        if allow_cross_family:
            warnings.warn(
                detail + " Proceeding anyway (allow_cross_family_sif=True).",
                stacklevel=2,
            )
            return
        raise ConfigurationError(
            field="container_build.target_arch",
            message=(
                detail
                + "\n\nSupply one .def per required arch at emit time "
                "(hhemt bundle ... --container-defs <a.def> --container-defs <b.def>), "
                "point your partitions at hardware the carried .defs cover, or — if you "
                "have confirmed run-compatibility — re-ingest with allow_cross_family_sif="
                "True (Python) / --allow-cross-family-sif (CLI)."
            ),
            config_path=None,
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

    @classmethod
    def _materialize_input_deposits(
        cls, bundle_root: Path, missing: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
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
    def _assert_declared_inputs_exist(
        cls, system_config_path: Path, analysis_config_path: Path
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
            import os  # host resolution mirrors publishing._ZenodoTarget.publish

            import requests  # explicit dep declared in pyproject

            base = os.environ.get("HHEMT_ZENODO_BASE_URL", "https://zenodo.org").rstrip("/")
            resp = requests.get(
                f"{base}/api/records/{recid}", timeout=60
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
