"""`_emit_sif_manifests` carries the image the SIMULATIONS RAN IN, resolved by the digest setup
recorded, and refuses rather than recomputing an identity from the running toolkit.

Solver-free: no compile, no simulation, no container. The analysis is a stub carrying only the
attributes the function reads, and `sif_root` is a `tmp_path` tree holding a manifest and an
empty `.sif` beside it -- the layout `sif.identity.find_by_sha256` globs.

The production failure this pins: a campaign ran inside an image built at toolkit X, the operator
advanced the checkout to Y to land an unrelated fix, and `hhemt bundle` then demanded an image at
Y's identity that nothing had ever run -- after a three-hour staging walk it did not need.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.exceptions import ConfigurationError

_TRITON = "a38338b09e62e57c936f51516bbdbe495d89a546"
_RAN_AT = "6f8f397982f7295e52f2e5b10aa45bab3e306910"
_RUNNING = "50a91da590b4c0b105c233fd88f899432a6484c8"
_DIGEST = "41328743f95ffe5737e39d3f98abb0d1f946d7ff3ae3c883cf0d4d39b26f2cd1"


def _write_image(sif_root: Path, *, digest: str = _DIGEST, **identity_overrides) -> dict:
    """Write `{sif_root}/{family}/{stem}.manifest.json` + an empty `.sif`, return the manifest."""
    identity = {
        "mpi_family": "openmpi",
        "accel": "cpu",
        "gpu_hardware": None,
        "gpu_compilation_backend": None,
        "kokkos_arch": None,
        "triton_url": "https://code.ornl.gov/hydro/triton.git",
        "triton_sha": _TRITON,
        "swmm_tag": "v5.2.4",
        "hhemt_sha": _RAN_AT,
        "base_digest": "ubuntu@sha256:" + "4" * 64,
        "recipe_sha256": "6" * 64,
    }
    identity.update(identity_overrides)
    family = f"{identity['mpi_family']}-{identity['accel']}"
    hw = f"_{identity['gpu_hardware']}" if identity["gpu_hardware"] else ""
    stem = f"hhemt_{family}{hw}_{identity['triton_sha'][:8]}_{identity['hhemt_sha'][:8]}"
    d = sif_root / family
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"sha256": digest, "key": "0" * 16, "identity": identity}
    (d / f"{stem}.manifest.json").write_text(json.dumps(manifest))
    (d / f"{stem}.sif").write_bytes(b"")
    return manifest


class _Field:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value


class _Log:
    def __init__(self, logfile: Path, digest: str | None):
        self.logfile = logfile
        self.sif_sha256 = _Field(digest)
        self.refreshed = False

    def refresh(self):
        self.refreshed = True


def _stub(tmp_path: Path, *, digest: str | None = _DIGEST, partitions=("standard",), swmm_tag="v5.2.4"):
    sif_root = tmp_path / "sifs"
    sif_root.mkdir(exist_ok=True)
    partition_specs = {p: SimpleNamespace(gpu_hardware=None, gpu_compilation_backend=None) for p in partitions}
    return SimpleNamespace(
        cfg_hpc_system=SimpleNamespace(
            container=SimpleNamespace(sif_root=sif_root, gpu_flag=None, srun_mpi="pmix", cray_mpich_abi_module=False),
            partitions=partition_specs,
        ),
        cfg_analysis=SimpleNamespace(
            execution_environment="container",
            hpc_ensemble_partition=partitions[0],
            toggle_sensitivity_analysis=False,
            sensitivity_analysis=None,
            hhemt_sha=_RUNNING,
        ),
        _system=SimpleNamespace(
            log=_Log(tmp_path / "system_log.json", digest),
            cfg_system=SimpleNamespace(
                SWMM_tag_key=swmm_tag,
                TRITONSWMM_branch_key=_TRITON,
                TRITONSWMM_software_directory=tmp_path / "no_clone_here",
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _recompute_is_a_hard_error(monkeypatch):
    """Any surviving recompute path is a test failure, not a silent fallback."""

    def _boom(*_a, **_k):
        raise AssertionError("recompute reached: the emit derived an identity from the running toolkit")

    monkeypatch.setattr("hhemt.validation.running_identity", _boom, raising=False)
    monkeypatch.setattr("hhemt.sif.identity.derive_identity", _boom, raising=False)


def test_carries_the_recorded_image_not_the_running_toolkits(tmp_path):
    """THE INVARIANT. The carried manifest is the one whose digest setup recorded, and its
    `hhemt_sha` is the image's, not the running toolkit's.

    THE TWO DECOYS ARE LOAD-BEARING -- DO NOT DELETE THEM AS SETUP NOISE, DO NOT KEEP ONLY THE
    FIRST, AND DO NOT ADD A CROSS-FAMILY ONE. A production `sif_root` holds several images, and
    with only ONE image on disk this test cannot tell digest-keyed selection from "take whatever
    manifest is there": both return the same object, so a take-first implementation passes. The two
    decoys kill DIFFERENT wrong implementations and neither substitutes for the other. Decoy 1
    differs in digest and is written after the real image, so it kills take-newest and take-largest.
    Decoy 2 is the SAME family as the real image and differs only in the stem's trailing toolkit
    segment, so it sorts first inside `openmpi-cpu/` and kills take-first-sorted.

    DECOY 2 IS DELIBERATELY SAME-FAMILY, AND THE ALTERNATIVE IS WORSE RATHER THAN EQUIVALENT. A
    cross-family decoy also kills the take-first mutant, but it kills it at arm (d) -- `family` is
    `f"{mpi_family}-{accel}"` and no accel sorts before `cpu`, so anything sorting before
    `openmpi-cpu/` must differ in `mpi_family`, which arm (d) compares. The selection assertion
    below then never runs, and this test's selection coverage becomes contingent on arm (d) keeping
    `mpi_family` in its field list. Decoy 2 as written passes every arm-(d) field, so the mutant
    reaches `assert out == [written]` and the kill lands where it is aimed. Measured on the
    take-first mutant: one image -> 7 passed (survives); decoy 1 only -> 7 passed (still survives,
    `openmpi-cpu` sorts before `openmpi-cuda` and take-first is right by accident of this host's
    directory names); decoys 1+2 -> 1 failed at `assert out == [written]`; decoys 1+2 plus a
    cross-family third -> 1 failed at arm (d) instead, i.e. adding it SHADOWS decoy 2 and restores
    the contingency. This pair also reproduces the real `sif_root` that produced this fix: two
    images in one family directory differing in exactly the toolkit segment.

    RED PRE-APPLY: the pre-fix body calls `analysis._sif_identity_for(part)`, which this stub does
    not define (AttributeError) and whose real implementation calls the monkeypatched
    `running_identity` (AssertionError). Either way it cannot return, and it could not find an
    image if it did: the fixture writes only the `_RAN_AT` stem.
    """
    stub = _stub(tmp_path)
    sif_root = stub.cfg_hpc_system.container.sif_root
    written = _write_image(sif_root)
    # Decoy 1: a DIFFERENT digest, built at the RUNNING toolkit -- the image the pre-fix code
    # would have demanded. Selection must not prefer it.
    _write_image(sif_root, digest="c" * 64, accel="cuda", gpu_hardware="a6000", hhemt_sha=_RUNNING)
    # Decoy 2: SAME family, differing only in the stem's toolkit segment, so it sorts first
    # inside openmpi-cpu/ AND passes every arm-(d) field -- the kill lands on the selection
    # assertion rather than on a config-disagreement refusal. Cross-family would not.
    _write_image(sif_root, digest="e" * 64, hhemt_sha=_RUNNING)

    out = _emit._emit_sif_manifests(stub)

    assert out == [written]
    assert out[0]["identity"]["hhemt_sha"] == _RAN_AT != _RUNNING
    assert stub._system.log.refreshed, "setup and emit are different processes; the log must be refreshed"


def test_no_recorded_digest_refuses_and_names_both_remedies(tmp_path):
    """ARM (a). RED PRE-APPLY: pre-fix there is no digest read at all, so this input reaches the
    recompute and raises AssertionError/AttributeError rather than ConfigurationError."""
    stub = _stub(tmp_path, digest=None)
    _write_image(stub.cfg_hpc_system.container.sif_root)

    with pytest.raises(ConfigurationError) as exc:
        _emit._emit_sif_manifests(stub)

    msg = str(exc.value)
    assert "system_log.json" in msg
    assert "check out the toolkit commit" in msg and "build an image at the running commit" in msg
    assert "recurs unchanged" in msg, "the precondition must be stated, not just the two remedies"
    assert "build-sifs" not in msg


def test_recorded_digest_with_no_manifest_refuses_naming_digest_and_root(tmp_path):
    """ARM (b). RED PRE-APPLY: pre-fix the digest is never read, so the message names a recomputed
    identity and tells the operator to run `hhemt build-sifs` -- the assertion below forbids it."""
    stub = _stub(tmp_path)  # no image written

    with pytest.raises(ConfigurationError) as exc:
        _emit._emit_sif_manifests(stub)

    msg = str(exc.value)
    assert _DIGEST in msg
    assert str(stub.cfg_hpc_system.container.sif_root) in msg
    assert "build-sifs" not in msg, "a rebuild yields a different blob and digest; it is not the remedy"


def test_matrix_partition_needing_different_hardware_refuses(tmp_path):
    """ARM (c), coverage predicate. A GPU container spec whose matrix partition resolves to a100
    while the recorded image is an a6000 build."""
    stub = _stub(tmp_path, partitions=("gpu",))
    stub.cfg_hpc_system.container.gpu_flag = "--nv"
    stub.cfg_hpc_system.partitions["gpu"] = SimpleNamespace(gpu_hardware="a100", gpu_compilation_backend="CUDA")
    _write_image(
        stub.cfg_hpc_system.container.sif_root,
        accel="cuda",
        gpu_hardware="a6000",
        gpu_compilation_backend="CUDA",
    )

    with pytest.raises(ConfigurationError) as exc:
        _emit._emit_sif_manifests(stub)

    msg = str(exc.value)
    assert "a100" in msg and "a6000" in msg
    assert "ONE SIF digest" in msg


def test_carried_solver_pin_disagreeing_with_config_refuses(tmp_path):
    """ARM (d), and the residual neither predicate reaches: the coverage tuple AGREES
    (openmpi, cpu, None) while the solver pin does not, on a host with no TRITON clone.

    RED AGAINST THE BEST-EFFORT IMPLEMENTATION, which is the point of this arm: a version that
    skips the triton_sha comparison when no clone is present returns the wrong manifest here and
    discloses nothing, because the function returns list[dict] and has no channel for a skip.
    `TRITONSWMM_software_directory` deliberately does not exist; `resolve_full_sha` returns a
    40-hex ref without touching it.
    """
    stub = _stub(tmp_path)
    _write_image(stub.cfg_hpc_system.container.sif_root, triton_sha="b" * 40)

    with pytest.raises(ConfigurationError) as exc:
        _emit._emit_sif_manifests(stub)

    msg = str(exc.value)
    assert "triton_sha" in msg and _TRITON in msg
    assert "system_directory" in msg, "the CAUSE clause is what sends the operator to the right fix"


def test_explicit_null_swmm_tag_is_not_a_false_refusal(tmp_path):
    """ARM (d) negative control, and the differently-positioned satisfying input: both Frontier
    arms set `SWMM_tag_key: null`, which the identity now carries as a real None rather than as
    the string 'None'. Both operands are sourced the same way, so absence compares equal to
    absence and those two experiments stay bundleable."""
    stub = _stub(tmp_path, swmm_tag=None)
    written = _write_image(stub.cfg_hpc_system.container.sif_root, swmm_tag=None)

    assert _emit._emit_sif_manifests(stub) == [written]


def test_pre_fix_string_none_manifest_is_refused_against_a_null_pin(tmp_path):
    """The forward-incompatibility cost, ASSERTED rather than left silent.

    A manifest written by a PRE-fix toolkit serialized an absent pin as the string 'None'. The
    expected operand is now the config value itself, so such a manifest no longer matches a
    null-pin config. No such manifest can exist in production -- a pre-fix null-pin build died
    in the recipe's `git clone --branch None` under `set -eu` before any manifest was written --
    so this pins the contract rather than a live state, and it is RED pre-fix because the old
    code compared 'None' against str(None) and found them equal.
    """
    stub = _stub(tmp_path, swmm_tag=None)
    _write_image(stub.cfg_hpc_system.container.sif_root, swmm_tag="None")

    with pytest.raises(ConfigurationError) as exc:
        _emit._emit_sif_manifests(stub)

    assert "swmm_tag" in str(exc.value)


def test_the_refusal_precedes_the_prune_and_the_staging_walk(tmp_path, monkeypatch):
    """ORDERING. `_prune_undeclared_figures` DELETES orphan figures from the LIVE analysis tree;
    wasted staging I/O is recoverable and that deletion is not.

    Anchored on WHICH exception type escapes -- a property present in both the pre-fix and
    post-fix worlds, so the assertion discriminates on behaviour rather than on wording. RED
    PRE-APPLY: pre-fix the SIF check sits inside the `with _staging_dir(...)` block, so the prune
    runs first and the `AssertionError` sentinel escapes instead of the `ConfigurationError`.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "plots").mkdir(parents=True)
    (analysis_dir / "plots" / "x.manifest.json").write_text("{}")
    stub = _stub(tmp_path)  # no image written -> arm (b) refuses
    stub.analysis_paths = SimpleNamespace(analysis_dir=analysis_dir)

    def _sentinel(*_a, **_k):
        raise AssertionError("reached a destructive or expensive step before the SIF gate")

    monkeypatch.setattr(_emit, "_prune_undeclared_figures", _sentinel)
    monkeypatch.setattr(_emit, "harvest_source_paths", _sentinel)

    with pytest.raises(ConfigurationError):
        _emit.emit_bundle(stub, tmp_path / "out.zip")
