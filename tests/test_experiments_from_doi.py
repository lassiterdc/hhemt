"""Phase 1 — ``from_doi`` ingestion round-trip + fail-closed gates (ADR-13 C9; R1-R4/R10).

Exercises the REAL self-contained emit -> ingest path: a rendered synth analysis is
emitted via the actual ``emit_bundle`` (with the self-contained harvest that carries every
cfg-declared input), then ``TRITON_SWMM_experiment.from_doi`` reconstitutes it with the
network fetch mocked. Coverage:

  (a) happy path — a self-contained bundle yields a runnable experiment whose
      ``analysis_dir`` resolves to ``bundle_root`` (R1).
  (d) self-containment — every reconstituted input Path exists on disk in-bundle (R10).
  (b) no-``mainEntity`` crate -> fail closed (R3).
  (c) a carried input deleted from the bundle -> the materialize-or-fail gate fails
      closed naming that path (the silent-failure this phase closes).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from hhemt.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    PathPolicy,
    enumerate_path_fields,
)
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment

_CARRIED = {
    PathPolicy.BUNDLE_RELATIVE,
    PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    PathPolicy.BUNDLE_RELATIVE_LIST,
}


@pytest.fixture(scope="module")
def self_contained_bundle(request, tmp_path_factory):
    """Emit a self-contained reprex bundle from the rendered multi_sim synth analysis,
    extract it, and return (zip_path, bundle_root)."""
    from hhemt.bundle._reprex import extract_reprex_bundle

    analysis = request.getfixturevalue("rendered_synth_multi_sim")
    out = tmp_path_factory.mktemp("from_doi_bundle")
    bundle_path = analysis.bundle_report_data(out / "bundle.zip")
    bundle_root = extract_reprex_bundle(bundle_path)
    return bundle_path, bundle_root


def _rezip(src_dir: Path, dest_zip: Path) -> Path:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))
    return dest_zip


def _patch_fetch(monkeypatch, zip_path: Path) -> None:
    """Mock the network fetch: _fetch_bundle_zip returns the pre-built (or tampered) zip;
    from_doi still runs the real extract + reconstitute + fail-closed path against it."""
    monkeypatch.setattr(
        TRITON_SWMM_experiment,
        "_fetch_bundle_zip",
        classmethod(lambda cls, *args, **kwargs: Path(zip_path)),
    )


def _first_carried_input_in_bundle(bundle_root: Path) -> tuple[str, Path]:
    """Return (field_name, absolute_path_under_bundle_root) for the first cfg-declared,
    carried (BUNDLE_RELATIVE family) input whose bundle-relative value resolves to a real
    FILE in the bundle. Used to pick an input to delete for the fail-closed test."""
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    for cfg_name, model in (
        ("cfg_analysis.yaml", analysis_config),
        ("cfg_system.yaml", system_config),
    ):
        data = yaml.safe_load((bundle_root / cfg_name).read_text())
        for name in enumerate_path_fields(model):
            if _PATH_FIELD_POLICY.get(name) not in _CARRIED:
                continue
            value = data.get(name)
            if value is None:
                continue
            for elem in value if isinstance(value, list) else [value]:
                if not isinstance(elem, str) or Path(elem).is_absolute():
                    continue
                candidate = (bundle_root / elem).resolve()
                if candidate.is_file():
                    return name, candidate
    raise AssertionError(
        "no carried cfg-declared input file found in the emitted bundle — the "
        "self-contained harvest did not carry any input (unexpected for multi_sim)."
    )


def test_from_doi_happy_path_is_runnable_and_self_contained(
    self_contained_bundle, monkeypatch, tmp_path
):
    zip_path, _ = self_contained_bundle
    _patch_fetch(monkeypatch, zip_path)

    exp = TRITON_SWMM_experiment.from_doi(
        doi="10.5281/zenodo.123456", host="zenodo", target_dir=tmp_path / "ingest"
    )

    # R1: a runnable experiment (system + analysis constructed).
    assert exp.system is not None
    assert exp.analysis is not None
    # analysis_dir resolves to bundle_root (NOT the caller's CWD).
    assert (
        exp.analysis.analysis_paths.analysis_dir.resolve() == exp.bundle_root.resolve()
    )

    # R10: every reconstituted CARRIED input Path resolves under bundle_root AND exists
    # on disk (self-contained). Toolkit-owned build dirs (IS_NONE_ACCEPTABLE, set to a
    # not-yet-built target-side location) and FORCED_DOT dir markers are not inputs.
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    checked_any = False
    for cfg_name, model in (
        ("analysis_config.yaml", analysis_config),
        ("system_config.yaml", system_config),
    ):
        data = yaml.safe_load((exp.bundle_root / cfg_name).read_text())
        for name in enumerate_path_fields(model):
            if _PATH_FIELD_POLICY.get(name) not in _CARRIED:
                continue
            value = data.get(name)
            if value is None:
                continue
            for elem in value if isinstance(value, list) else [value]:
                if isinstance(elem, str):
                    checked_any = True
                    assert Path(
                        elem
                    ).exists(), f"reconstituted input {name} does not exist: {elem}"
    assert checked_any, "no carried input was checked — self-containment not exercised"


def test_from_doi_no_main_entity_fails_closed(
    self_contained_bundle, monkeypatch, tmp_path
):
    _, bundle_root = self_contained_bundle
    tampered = tmp_path / "no_main_entity"
    shutil.copytree(bundle_root, tampered)
    crate = tampered / "ro-crate-metadata.json"
    doc = json.loads(crate.read_text())
    for entity in doc["@graph"]:
        if entity.get("@id") == "./":
            entity.pop("mainEntity", None)
    crate.write_text(json.dumps(doc))
    zip_path = _rezip(tampered, tmp_path / "no_main_entity.zip")
    _patch_fetch(monkeypatch, zip_path)

    with pytest.raises(ProcessingError, match="mainEntity"):
        TRITON_SWMM_experiment.from_doi(
            doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "ing_b"
        )


def test_from_doi_missing_input_fails_closed(
    self_contained_bundle, monkeypatch, tmp_path
):
    _, bundle_root = self_contained_bundle
    field_name, victim = _first_carried_input_in_bundle(bundle_root)
    tampered = tmp_path / "missing_input"
    shutil.copytree(bundle_root, tampered)
    # Delete the victim input from the tampered copy (path-mirror under the copytree).
    (tampered / victim.relative_to(bundle_root)).unlink()
    zip_path = _rezip(tampered, tmp_path / "missing_input.zip")
    _patch_fetch(monkeypatch, zip_path)

    with pytest.raises(ProcessingError) as excinfo:
        TRITON_SWMM_experiment.from_doi(
            doi="10.5281/zenodo.2", host="zenodo", target_dir=tmp_path / "ing_c"
        )
    msg = str(excinfo.value)
    assert "do not exist on disk" in msg
    assert field_name in msg  # the gate names the missing field


def test_from_doi_requires_doi_or_pid():
    from hhemt.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        TRITON_SWMM_experiment.from_doi(host="zenodo")


# --- ADR-19(vii) cross-family SIF guard (D-E) ------------------------------------------


def _hpc_cfg(partition_name, gpu_hardware):
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config

    return hpc_system_config(
        system_name="test",
        partitions={
            partition_name: PartitionSpec(
                gpu_hardware=gpu_hardware, gpu_compilation_backend="CUDA"
            )
        },
    )


def _analysis_cfg(tmp_path, partition_name):
    # Minimal non-sensitivity analysis config: the matrix is the single ensemble partition.
    p = tmp_path / "analysis_config.yaml"
    p.write_text(
        f"hpc_ensemble_partition: {partition_name}\ntoggle_sensitivity_analysis: false\n"
    )
    return p


def test_arch_set_guard_covered_passes(tmp_path):
    # required = {a100} (the matrix partition maps to a100); a carried a100 .def covers it.
    TRITON_SWMM_experiment._assert_container_arch_set_covers_matrix(
        carried_blocks=[{"target_arch": "a100"}],
        cfg_hpc_system=_hpc_cfg("gpu", "a100"),
        analysis_config_path=_analysis_cfg(tmp_path, "gpu"),
        bundle_root=tmp_path,
    )  # no raise


def test_arch_set_guard_uncovered_fails_closed(tmp_path):
    from hhemt.exceptions import ConfigurationError

    # matrix requires a6000, but only an a100 .def is carried -> fail closed.
    with pytest.raises(ConfigurationError, match="a6000"):
        TRITON_SWMM_experiment._assert_container_arch_set_covers_matrix(
            carried_blocks=[{"target_arch": "a100"}],
            cfg_hpc_system=_hpc_cfg("gpu", "a6000"),
            analysis_config_path=_analysis_cfg(tmp_path, "gpu"),
            bundle_root=tmp_path,
        )


def test_arch_set_guard_over_provisioned_passes(tmp_path):
    # carries BOTH a100 + a6000; matrix requires only a100 -> the extra a6000 def is unused.
    TRITON_SWMM_experiment._assert_container_arch_set_covers_matrix(
        carried_blocks=[{"target_arch": "a100"}, {"target_arch": "a6000"}],
        cfg_hpc_system=_hpc_cfg("gpu", "a100"),
        analysis_config_path=_analysis_cfg(tmp_path, "gpu"),
        bundle_root=tmp_path,
    )  # no raise


def test_arch_set_guard_cpu_matrix_passes(tmp_path):
    # the matrix partition declares no GPU -> required set is empty -> a carried GPU .def is
    # not required (a CPU row resolves to the CPU/sif_path fallback, not the per-arch map).
    TRITON_SWMM_experiment._assert_container_arch_set_covers_matrix(
        carried_blocks=[{"target_arch": "a100"}],
        cfg_hpc_system=_hpc_cfg("cpu", None),
        analysis_config_path=_analysis_cfg(tmp_path, "cpu"),
        bundle_root=tmp_path,
    )  # no raise


def test_arch_set_guard_override_warns_not_raises(tmp_path):
    with pytest.warns(UserWarning, match="allow_cross_family"):
        TRITON_SWMM_experiment._assert_container_arch_set_covers_matrix(
            carried_blocks=[{"target_arch": "a100"}],
            cfg_hpc_system=_hpc_cfg("gpu", "a6000"),
            analysis_config_path=_analysis_cfg(tmp_path, "gpu"),
            bundle_root=tmp_path,
            allow_cross_family=True,
        )


# --- ADR-19 D-A: the dead transfer branch is deleted -----------------------------------


def test_build_unavailable_raises_no_silent_transfer(tmp_path, monkeypatch):
    from hhemt import container_build as cb
    from hhemt.exceptions import ConfigurationError

    bundle_root = tmp_path / "b"
    bundle_root.mkdir()
    (bundle_root / "recipe.def").write_text("Bootstrap: docker\nFrom: x\n")
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(
            {"container_build": {"def_relpath": "recipe.def", "target_arch": "a100"}}
        )
    )

    def _raise(**kw):
        raise cb.SifBuildUnavailable(reason="no subuid", remediation="ask admin")

    monkeypatch.setattr(cb, "get_or_build_sif", _raise)
    # Any accidental network fetch is a HARD failure — the transfer branch is dead.
    monkeypatch.setattr(
        TRITON_SWMM_experiment,
        "_fetch_file_by_url",
        classmethod(lambda cls, *a, **k: pytest.fail("transfer branch must be dead")),
    )
    with pytest.raises(ConfigurationError, match="build-sif"):
        TRITON_SWMM_experiment._build_or_fetch_sif(bundle_root, account="acct")


# --- multi-SIF (Option A) net-new plumbing ---------------------------------------------


def test_build_or_fetch_sif_uses_passed_container_block(tmp_path, monkeypatch):
    # Multi-SIF: from_doi passes each per-arch block via container_block; that block must
    # win over the manifest's (a passed a6000 block must build the a6000 def, not the
    # manifest's first a100 block).
    from hhemt import container_build as cb

    bundle_root = tmp_path / "b"
    bundle_root.mkdir()
    (bundle_root / "a100.def").write_text("Bootstrap: docker\nFrom: x\n")
    (bundle_root / "a6000.def").write_text("Bootstrap: docker\nFrom: x\n")
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "container_build": [
                    {"def_relpath": "a100.def", "target_arch": "a100"},
                    {"def_relpath": "a6000.def", "target_arch": "a6000"},
                ]
            }
        )
    )
    captured: dict = {}

    def _fake_build(**kw):
        captured.update(kw)
        return tmp_path / f"hhemt-{kw['target_arch']}.sif"

    monkeypatch.setattr(cb, "get_or_build_sif", _fake_build)
    out = TRITON_SWMM_experiment._build_or_fetch_sif(
        bundle_root,
        account="acct",
        container_block={"def_relpath": "a6000.def", "target_arch": "a6000"},
    )
    assert captured["target_arch"] == "a6000"  # the PASSED block, not the manifest's a100
    assert captured["def_path"].name == "a6000.def"
    assert out.name == "hhemt-a6000.sif"


def test_repoint_sif_paths_writes_both_keys(tmp_path):
    # DELTA-4-adjacent: the repoint writes BOTH the arch-agnostic sif_path (process rung)
    # AND the per-arch sif_paths_by_arch map (SIM rung), resolving each path.
    import yaml

    src = tmp_path / "hpc.yaml"
    src.write_text("container:\n  sif_path: /orig/placeholder.sif\n  gpu_flag: --nv\n")
    a100_sif = tmp_path / "hhemt-a100.sif"
    a100_sif.write_text("x")
    a6000_sif = tmp_path / "hhemt-a6000.sif"
    a6000_sif.write_text("x")
    target = tmp_path / "derived" / "hpc_system_config.resolved.yaml"

    out = TRITON_SWMM_experiment._repoint_sif_paths(
        src,
        sif_path=a100_sif,
        sif_paths_by_arch={"a100": str(a100_sif), "a6000": str(a6000_sif)},
        target_path=target,
    )
    cfg = yaml.safe_load(out.read_text())
    assert cfg["container"]["sif_path"] == str(a100_sif.resolve())
    assert cfg["container"]["sif_paths_by_arch"] == {
        "a100": str(a100_sif.resolve()),
        "a6000": str(a6000_sif.resolve()),
    }
    assert cfg["container"]["gpu_flag"] == "--nv"  # other fields preserved


def test_matrix_required_arches_cross_hardware_sensitivity(tmp_path):
    # VMS-8 helper: the required arch set is the distinct per-row partitions' gpu_hardware;
    # a CPU partition (no gpu_hardware) is excluded (CPU rows use the sif_path fallback).
    from types import SimpleNamespace

    from hhemt.bundle._emit import _matrix_required_arches
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config

    cfg_hpc = hpc_system_config(
        system_name="t",
        partitions={
            "gpu-a100": PartitionSpec(gpu_hardware="a100", gpu_compilation_backend="CUDA"),
            "gpu-a6000": PartitionSpec(
                gpu_hardware="a6000", gpu_compilation_backend="CUDA"
            ),
            "standard": PartitionSpec(),
        },
    )
    setup = tmp_path / "setup.csv"
    setup.write_text("sa_id,hpc.partition\n0,gpu-a100\n1,gpu-a6000\n2,standard\n")
    analysis = SimpleNamespace(
        cfg_hpc_system=cfg_hpc,
        cfg_analysis=SimpleNamespace(
            hpc_ensemble_partition="gpu-a100",
            toggle_sensitivity_analysis=True,
            sensitivity_analysis=setup,
        ),
    )
    assert _matrix_required_arches(analysis) == {"a100", "a6000"}


# --- defect-8: node-local /tmp build-context guard -------------------------------------


def _minimal_reproducer_hpc_cfg(tmp_path: Path) -> Path:
    """A minimal EXISTING hpc_system_config path. from_doi's _resolve_hpc_system_config only
    RESOLVES the path (existence), it does not validate content, and the defect-8 guard fires
    before any content validation — so the ingest merely needs a real file here."""
    p = tmp_path / "reproducer_hpc.yaml"
    p.write_text(
        "system_name: test\ndefault_account: test_acct\npartitions:\n  standard: {}\n"
    )
    return p


def _container_flipped_bundle_zip(bundle_root: Path, dest_dir: Path, tag: str) -> Path:
    """Copy an emitted native bundle and flip it to container-mode (execution_environment=
    container) so from_doi enters the ADR-19 container branch that carries the defect-8 guard."""
    src = dest_dir / f"container_bundle_{tag}"
    shutil.copytree(bundle_root, src)
    cfg_path = src / "cfg_analysis.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["execution_environment"] = "container"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return _rezip(src, dest_dir / f"container_bundle_{tag}.zip")


def test_from_doi_node_local_build_context_fails_closed(
    self_contained_bundle, monkeypatch, tmp_path
):
    """defect-8: a container-mode ingest whose bundle_root lands under the node-local system
    temp dir on a SLURM host must fail LOUD before submitting the compute-node `sbatch --wait`
    SIF build (which would `cd` into the invisible bundle_root and die with a cryptic
    `cd: No such file or directory`). target_dir=None makes from_doi mkdtemp bundle_root under
    gettempdir(); _slurm_available()->True makes the guard's SLURM precondition hold."""
    from hhemt import container_build as cb
    from hhemt.exceptions import ConfigurationError

    _, bundle_root = self_contained_bundle
    zip_path = _container_flipped_bundle_zip(bundle_root, tmp_path, "slurm")
    _patch_fetch(monkeypatch, zip_path)
    monkeypatch.setattr(cb, "_slurm_available", lambda: True)

    # (?i): the guard's remediation phrase is "on a SHARED filesystem" — match
    # case-insensitively so an innocuous message re-capitalization does not break the test.
    with pytest.raises(ConfigurationError, match="(?i)shared filesystem"):
        TRITON_SWMM_experiment.from_doi(
            doi="10.5281/zenodo.123456",
            host="zenodo",
            hpc_system_config_yaml=_minimal_reproducer_hpc_cfg(tmp_path),
            target_dir=None,
        )


def test_from_doi_node_local_guard_inert_on_non_slurm(
    self_contained_bundle, monkeypatch, tmp_path
):
    """The defect-8 guard is inert on a non-SLURM host: a same-node local build sees the
    node-local bundle_root fine, so the guard must NOT fire even though bundle_root is under
    the system temp dir. from_doi proceeds past the guard and fails later for an unrelated
    reason (no real SIF build in the test env) — the point is that the failure is NOT the
    defect-8 'shared filesystem' refusal (no false positive)."""
    from hhemt import container_build as cb

    _, bundle_root = self_contained_bundle
    zip_path = _container_flipped_bundle_zip(bundle_root, tmp_path, "local")
    _patch_fetch(monkeypatch, zip_path)
    monkeypatch.setattr(cb, "_slurm_available", lambda: False)

    with pytest.raises(Exception) as exc_info:
        TRITON_SWMM_experiment.from_doi(
            doi="10.5281/zenodo.123456",
            host="zenodo",
            hpc_system_config_yaml=_minimal_reproducer_hpc_cfg(tmp_path),
            target_dir=None,
        )
    # Lowercase both sides: the guard's message says "SHARED filesystem", so a case-SENSITIVE
    # check here would pass vacuously (it could never fail) and would not actually test the guard.
    assert "shared filesystem" not in str(
        exc_info.value
    ).lower(), "the defect-8 guard fired on a non-SLURM host (false positive)"
