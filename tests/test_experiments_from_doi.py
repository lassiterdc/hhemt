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
