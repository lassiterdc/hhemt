"""SifIdentity: the one object that names, fills, labels, resolves and verifies a SIF.

Two SOURCES, one type: ``derive_identity`` recomputes it from the same Pydantic objects
preflight reads (git-checkout drivers); ``SifIdentity.from_carried`` reads it back from a
producer manifest (``{sif}.manifest.json``, carried in a bundle by ``bundle/_emit.py``).
``resolve_sif`` is the ONLY way an image path is produced anywhere in the toolkit: no
pointer field exists. ``find_by_sha256`` is the digest-keyed lookup ``reprex()`` uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hhemt.config.hpc_system import ContainerSpec, hpc_system_config, resolve_gpu_target
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError
from hhemt.sif.pins import is_full_sha, resolve_full_sha

MpiFamily = Literal["openmpi", "cray-mpich"]
Accel = Literal["cpu", "cuda", "rocm"]

_IDENTITY_DOMAIN = b"hhemt-sif-identity-v1\x00"

# The toolkit has no HIP arch resolver (system.py's map is CUDA-only, and the compile scripts
# for HIP carry the arch in the generated cmake line). This is the ONE ROCm table; a second
# Frontier GPU model is added here and nowhere else.
ROCM_ARCH_MAP: dict[str, str] = {"mi250x": "Kokkos_ARCH_AMD_GFX90A"}


def derive_family(cspec: ContainerSpec) -> tuple[MpiFamily, Accel]:
    """DERIVED, never declared. Measured on the 8 estate configs: pmix/None -> openmpi/cpu
    (uva_cpu, uva_cpu_stochastic); pmix/--nv -> openmpi/cuda (a100, a6000); cray/None ->
    cray-mpich/cpu (frontier_cpu); cray/--rocm -> cray-mpich/rocm (frontier_gpu)."""
    if cspec.cray_mpich_abi_module and cspec.srun_mpi:
        raise ConfigurationError(
            field="container.srun_mpi", message="cray_mpich_abi_module=true XOR srun_mpi='pmix'; both set"
        )
    mpi: MpiFamily = "cray-mpich" if cspec.cray_mpich_abi_module else "openmpi"
    if not cspec.cray_mpich_abi_module and cspec.srun_mpi not in ("pmix", None):
        raise ConfigurationError(
            field="container.srun_mpi",
            message=f"unknown MPI launch flavour {cspec.srun_mpi!r}; no recipe family",
        )
    accel: Accel = {None: "cpu", "--nv": "cuda", "--rocm": "rocm"}[cspec.gpu_flag]
    return mpi, accel


class SifIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mpi_family: MpiFamily
    accel: Accel
    gpu_hardware: str | None
    gpu_compilation_backend: str | None
    kokkos_arch: str | None  # the Kokkos_ARCH_* cmake token, e.g. Kokkos_ARCH_AMPERE86
    triton_url: str
    triton_sha: str  # 40-hex, resolved
    swmm_tag: str
    hhemt_sha: str  # 40-hex; the RUNNING toolkit (recomputed) or the CARRIED value
    base_digest: str  # the family recipe's `From: ...@sha256:` reference
    recipe_sha256: str  # sha256 of the family recipe text

    @property
    def family(self) -> str:
        return f"{self.mpi_family}-{self.accel}"

    @property
    def key(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(_IDENTITY_DOMAIN + payload).hexdigest()[:16]

    @property
    def stem(self) -> str:
        hw = f"_{self.gpu_hardware}" if self.gpu_hardware else ""
        return f"hhemt_{self.family}{hw}_{self.triton_sha[:8]}_{self.hhemt_sha[:8]}"

    def labels(self) -> dict[str, str]:
        """The org.hhemt.* labels the recipe bakes (static half; %post appends the measured half)."""
        return {
            "org.hhemt.identity": self.key,
            "org.hhemt.family": self.family,
            "org.hhemt.accel": self.accel,
            "org.hhemt.gpu_hardware": self.gpu_hardware or "none",
            "org.hhemt.gpu_arch": self.kokkos_arch or "none",
            "org.hhemt.gpu_compilation_backend": self.gpu_compilation_backend or "none",
            "org.hhemt.triton_url": self.triton_url,
            "org.hhemt.base_digest": self.base_digest,
            "org.hhemt.recipe_sha256": self.recipe_sha256,
            "org.hhemt.toolkit_root": "/opt/hhemt-src",
        }

    @classmethod
    def from_carried(cls, manifest: dict) -> SifIdentity:
        """The CARRIED source: a producer's ``{sif}.manifest.json`` (``manifest["identity"]``)."""
        return cls.model_validate(manifest["identity"])


def kokkos_arch_for(gpu_hardware: str | None, backend: str | None) -> str | None:
    """The Kokkos_ARCH_* token for a partition's hardware, from the ONE table per backend."""
    if not gpu_hardware:
        return None
    if str(backend).upper() == "CUDA":
        from hhemt.system import CUDA_ARCH_MAP  # lifted from _resolve_cuda_arch_flags (one table)

        try:
            return CUDA_ARCH_MAP[gpu_hardware.lower()][1]
        except KeyError as exc:
            raise ConfigurationError(
                field="gpu_hardware",
                message=f"Unknown CUDA gpu_hardware {gpu_hardware!r}; known: {sorted(CUDA_ARCH_MAP)}",
            ) from exc
    if str(backend).upper() == "HIP":
        try:
            return ROCM_ARCH_MAP[gpu_hardware.lower()]
        except KeyError as exc:
            raise ConfigurationError(
                field="gpu_hardware",
                message=f"Unknown HIP gpu_hardware {gpu_hardware!r}; known: {sorted(ROCM_ARCH_MAP)}",
            ) from exc
    raise ConfigurationError(field="gpu_compilation_backend", message=f"unknown backend {backend!r}")


def derive_identity(
    cfg_hpc: hpc_system_config,
    cfg_system: system_config,
    partition: str,
    running_sha: str | None,
    *,
    recipes_dir: Path | None = None,
) -> SifIdentity:
    """The RECOMPUTED source. ``running_sha`` MUST be the 40-hex sha of the RUNNING toolkit
    (``validation.running_identity().sha`` — the one identity source, [Q331]). None is NOT legal
    here: a wheel driver has no identity and ``running_identity`` refuses it before this is reached;
    the check below stays as the fail-closed floor for any caller that hands a bare value."""
    if not is_full_sha(running_sha):
        raise ConfigurationError(
            field="container",
            message=(
                f"identity cannot be recomputed: the running toolkit's commit is {running_sha!r}, not a 40-hex "
                "sha (a `pip install hhemt` wheel has none). Container mode requires the driver to be a git "
                "checkout of hhemt — check out the commit the image was built at (its manifest's "
                "identity.hhemt_sha) and re-run."
            ),
        )
    cspec = cfg_hpc.container
    if cspec is None:
        raise ConfigurationError(field="container", message="container mode with no container: block")
    mpi, accel = derive_family(cspec)
    hw, backend = resolve_gpu_target(cfg_hpc, partition) if accel != "cpu" else (None, None)
    from hhemt.sif.recipes import family_recipe

    _recipe_path, recipe_text, base_digest = family_recipe(f"{mpi}-{accel}", recipes_dir)
    return SifIdentity(
        mpi_family=mpi,
        accel=accel,
        gpu_hardware=hw,
        gpu_compilation_backend=backend,
        kokkos_arch=kokkos_arch_for(hw, backend),
        triton_url=str(cfg_system.TRITONSWMM_git_URL),
        triton_sha=resolve_full_sha(cfg_system.TRITONSWMM_software_directory, str(cfg_system.TRITONSWMM_branch_key)),
        swmm_tag=str(cfg_system.SWMM_tag_key),
        hhemt_sha=str(running_sha),
        base_digest=base_digest,
        recipe_sha256=hashlib.sha256(recipe_text.encode()).hexdigest(),
    )


def resolve_sif(sif_root: Path | str, identity: SifIdentity) -> Path:
    """THE resolver. ``{sif_root}/{family}/{stem}.sif``. Pure; never stats the filesystem."""
    return Path(sif_root) / identity.family / f"{identity.stem}.sif"


def manifest_path(sif: Path) -> Path:
    return sif.with_name(sif.name[: -len(".sif")] + ".manifest.json")


def find_by_sha256(sif_root: Path | str, sha256: str) -> tuple[Path, dict] | None:
    """Digest-keyed lookup over ``{sif_root}/*/*.manifest.json`` (the reprex path): returns
    ``(sif_path, manifest)`` for the manifest whose ``sha256`` equals ``sha256``, else None."""
    for man in sorted(Path(sif_root).glob("*/*.manifest.json")):
        try:
            m = json.loads(man.read_text())
        except (OSError, ValueError):
            continue
        if m.get("sha256") == sha256:
            return man.with_name(man.name[: -len(".manifest.json")] + ".sif"), m
    return None
