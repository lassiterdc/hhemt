"""SIF identity (ADR-21), Tier 1 / HPC-free: family derivation on the four measured estate
container blocks, 40-hex refusal, arch parity over the four families, resolver purity,
retired-key refusal, digest lookup. Self-contained: every fixture is inline."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from hhemt.config.hpc_system import ContainerSpec
from hhemt.exceptions import ConfigurationError
from hhemt.sif.args import build_arg_values, render_build_args
from hhemt.sif.identity import (
    SifIdentity,
    derive_family,
    derive_identity,
    find_by_sha256,
    kokkos_arch_for,
    manifest_path,
    resolve_sif,
)
from hhemt.sif.recipes import family_recipe

_SHA40 = "a" * 40
_TRITON = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"

# The four measured estate `container:` blocks (the private estate's hpc/hpc_system_config_*.yaml,
# read 2026-09-11), minus the retired pointer keys (sif_path/sif_sha256) and with sif_root.
_UVA_CPU = dict(
    sif_root="/scratch/user/sifs",
    apptainer_module="apptainer/1.5.0",
    gpu_flag=None,
    srun_mpi="pmix",
    binds=["/scratch", "/sfs"],
)
_UVA_CPU_STOCHASTIC = dict(_UVA_CPU)  # identical container block; only hpc_name differed
_FRONTIER_CPU = dict(
    sif_root="/lustre/orion/proj/sifs",
    gpu_flag=None,
    pre_exec_modules=["olcf-container-tools", "apptainer-enable-mpi"],
    cray_mpich_abi_module=True,
    # The LIVE after-validator (hpc_system.py `_check_mpi_flavor_exclusive`) requires this when
    # cray_mpich_abi_module=True; the estate's frontier_cpu config carries the real value.
    apptainerenv_ld_library_path="${CRAY_MPICH_DIR}/lib-abi-mpich:${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}",
    srun_mpi=None,
)
_UVA_A100 = dict(
    sif_root="/scratch/user/sifs",
    apptainer_module="apptainer/1.5.0",
    gpu_flag="--nv",
    srun_mpi="pmix",
    binds=["/scratch", "/sfs"],
)


def _ident(family: str, hw: str | None = None, *, hhemt_sha: str = _SHA40) -> SifIdentity:
    mpi, accel = family.rsplit("-", 1)
    backend = {"cpu": None, "cuda": "CUDA", "rocm": "HIP"}[accel]
    _p, text, base = family_recipe(family)
    return SifIdentity(
        mpi_family=mpi,
        accel=accel,
        gpu_hardware=hw,
        gpu_compilation_backend=backend,
        kokkos_arch=kokkos_arch_for(hw, backend),
        triton_url="https://code.ornl.gov/hydro/triton.git",
        triton_sha=_TRITON,
        swmm_tag="v5.2.4",
        hhemt_sha=hhemt_sha,
        base_digest=base,
        recipe_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_family_is_derived_not_declared():
    assert derive_family(ContainerSpec(**_UVA_CPU)) == ("openmpi", "cpu")
    assert derive_family(ContainerSpec(**_UVA_CPU_STOCHASTIC)) == ("openmpi", "cpu")  # [Q312] case-2 collapse
    assert derive_family(ContainerSpec(**_FRONTIER_CPU)) == ("cray-mpich", "cpu")  # MUST NOT collapse with UVA CPU
    assert derive_family(ContainerSpec(**_UVA_A100)) == ("openmpi", "cuda")


def test_family_refuses_both_mpi_flavours():
    # Arm 1: the CONSTRUCTOR refuses the XOR block first (the live `_check_mpi_flavor_exclusive`
    # after-validator) — a config file carrying both never reaches derive_family.
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ContainerSpec(sif_root="/x", cray_mpich_abi_module=True, srun_mpi="pmix", apptainerenv_ld_library_path="/o")
    # Arm 2: derive_family's OWN refusal is not vacuous behind the validator. model_construct is
    # the only way to hand it an XOR spec (it bypasses validation by design), which is why it is used.
    xor = ContainerSpec.model_construct(sif_root="/x", cray_mpich_abi_module=True, srun_mpi="pmix", gpu_flag=None)
    with pytest.raises(ConfigurationError, match="XOR"):
        derive_family(xor)


def test_identity_refuses_a_non_40_hex_running_sha():
    # The refusal is the FIRST statement of derive_identity: no config is touched (a wheel driver).
    for bad in ("c86daaa5", "c86daaa5c86daaa5c86daaa5c86daaa5c86daaa", None):
        with pytest.raises(ConfigurationError, match="40-hex"):
            derive_identity(None, None, "standard", running_sha=bad)  # type: ignore[arg-type]


def test_hpc_name_never_enters_the_key():
    a, b = _ident("openmpi-cpu"), _ident("openmpi-cpu")
    assert a.key == b.key
    assert "hpc" not in " ".join(a.labels())
    assert _ident("openmpi-cpu", hhemt_sha="b" * 40).key != a.key  # the toolkit commit DOES


@pytest.mark.parametrize(
    "family,hw",
    [
        ("openmpi-cpu", None),
        ("openmpi-cuda", "a100"),
        ("openmpi-cuda", "a6000"),
        ("cray-mpich-cpu", None),
        ("cray-mpich-rocm", "mi250x"),
    ],
)
def test_recipe_args_parity(family, hw):
    _p, text, _base = family_recipe(family)
    rendered = render_build_args(_ident(family, hw), text)  # raises on any unfilled/unused variable
    assert rendered.count("\n") == len(build_arg_values(_ident(family, hw)))


def test_cuda_and_rocm_tokens_derive_from_the_kokkos_arch():
    assert build_arg_values(_ident("openmpi-cuda", "a6000"))["CUDA_ARCH"] == "86"
    assert build_arg_values(_ident("openmpi-cuda", "a100"))["CUDA_ARCH"] == "80"
    assert build_arg_values(_ident("cray-mpich-rocm", "mi250x"))["GFX"] == "gfx90a"
    with pytest.raises(ConfigurationError, match="Unknown CUDA gpu_hardware"):
        kokkos_arch_for("rtx5090", "CUDA")


def test_resolver_is_pure(tmp_path):
    i = _ident("openmpi-cuda", "a6000")
    root = tmp_path / "never-created"
    assert resolve_sif(root, i) == root / "openmpi-cuda" / f"{i.stem}.sif"
    assert not root.exists()  # the resolver never stats or creates
    assert manifest_path(resolve_sif(root, i)).name == f"{i.stem}.manifest.json"


def test_retired_pointer_fields_refuse():
    for retired in ({"sif_path": "/x.sif"}, {"sif_paths_by_arch": {"a100": "/x.sif"}}, {"sif_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="RETIRED"):
            ContainerSpec.model_validate({"sif_root": "/scratch/user/sifs", **retired})


def test_find_by_sha256_scans_manifests(tmp_path):
    i = _ident("openmpi-cpu")
    sif = resolve_sif(tmp_path, i)
    sif.parent.mkdir(parents=True)
    sif.write_bytes(b"IMAGE-BYTES")
    digest = hashlib.sha256(b"IMAGE-BYTES").hexdigest()
    manifest_path(sif).write_text(json.dumps({"identity": i.model_dump(), "sha256": digest}))
    found = find_by_sha256(tmp_path, digest)
    assert found is not None and found[0] == sif and SifIdentity.from_carried(found[1]).key == i.key
    assert find_by_sha256(tmp_path, "0" * 64) is None


def test_build_sifs_experiment_id_is_the_bundle_directory_name(tmp_path, monkeypatch):
    import inspect

    from hhemt import cli
    from hhemt.experiment_bundle import bundle_experiment_id, load_bundle

    bundle_dir = tmp_path / "my_experiment"
    bundle_dir.mkdir()
    (bundle_dir / "experiment.yaml").write_text(
        "description: minimal\n"
        "system_config: configs/system.yaml\n"
        "analysis_config: configs/analysis.yaml\n"
        "toolkit_pin:\n  version: '0.1.0'\n"
    )
    bundle = load_bundle(bundle_dir)
    # Premise: the descriptor carries NO id field (extra='forbid'); reading one is the defect.
    assert not hasattr(bundle, "experiment_id")
    # Derivation: the containing directory's name, robust to a trailing slash and a relative '.'.
    assert bundle_experiment_id(bundle_dir) == "my_experiment"
    assert bundle_experiment_id(str(bundle_dir) + "/") == "my_experiment"
    monkeypatch.chdir(bundle_dir)
    assert bundle_experiment_id(".") == "my_experiment"
    # Binding: the CLI's --experiment arm constructs ExperimentInputs from that derivation, never
    # from a descriptor attribute (a source assertion — the arm needs loadable configs to run).
    src = inspect.getsource(cli.build_sifs_command)
    assert "ExperimentInputs(bundle_experiment_id(exp_dir)" in src
    assert "bundle.experiment_id" not in src
