"""render_build_args + expected_measured_labels: the --build-arg-file body and the read-back operands."""

from __future__ import annotations

import re

from hhemt.sif.identity import SifIdentity

_VAR = re.compile(r"{{\s*(\w+)\s*}}")  # Apptainer's template variable form (reader.go at v1.4.5)
_CUDA_SM = re.compile(r"(\d+)$")  # Kokkos_ARCH_AMPERE86 -> 86
_GFX = re.compile(r"AMD_GFX(\w+)$")  # Kokkos_ARCH_AMD_GFX90A -> gfx90a


def build_arg_values(ident: SifIdentity) -> dict[str, str]:
    v = {
        "BASE_REF": ident.base_digest,
        "TRITON_URL": ident.triton_url,
        "TRITON_SHA": ident.triton_sha,
        "HHEMT_SHA": ident.hhemt_sha,
        "SWMM_TAG": ident.swmm_tag,
        "LBL_IDENTITY": ident.key,
        "LBL_FAMILY": ident.family,
        "LBL_ACCEL": ident.accel,
        "LBL_GPU_HARDWARE": ident.gpu_hardware or "none",
        "LBL_GPU_ARCH": ident.kokkos_arch or "none",
        "LBL_GPU_BACKEND": ident.gpu_compilation_backend or "none",
        "LBL_RECIPE_SHA256": ident.recipe_sha256,
    }
    if ident.accel == "cuda":
        m = _CUDA_SM.search(ident.kokkos_arch or "")
        if not m:
            raise ValueError(f"cuda identity with non-CUDA kokkos_arch {ident.kokkos_arch!r}")
        v["CUDA_ARCH"] = m.group(1)
        v["KOKKOS_ARCH"] = str(ident.kokkos_arch)
    if ident.accel == "rocm":
        m = _GFX.search(ident.kokkos_arch or "")
        if not m:
            raise ValueError(f"rocm identity with non-HIP kokkos_arch {ident.kokkos_arch!r}")
        v["GFX"] = "gfx" + m.group(1).lower()
        v["KOKKOS_ARCH"] = str(ident.kokkos_arch)
    return v


def render_build_args(ident: SifIdentity, recipe_text: str) -> str:
    vals = build_arg_values(ident)
    used = set(_VAR.findall(recipe_text))
    missing, unused = used - vals.keys(), vals.keys() - used
    if missing or unused:  # Apptainer would fail the build for either; fail here in microseconds instead
        raise ValueError(
            f"recipe/args parity: recipe uses {sorted(missing)} unfilled; args carry {sorted(unused)} unused"
        )
    return "".join(f"{k}={vals[k]}\n" for k in sorted(vals))


def expected_measured_labels(ident: SifIdentity) -> dict[str, str]:
    """Labels %post MEASURES that have an INTENT operand; the read-back gate asserts equality.
    swmm_*/mpi/version are measured-only (recorded, never compared) and are NOT here."""
    return {"org.hhemt.triton_sha": ident.triton_sha, "org.hhemt.hhemt_sha": ident.hhemt_sha}
