"""Phase-1 checker tests — classification, report shape, the reserved-label
exclusion, and the real ``_read_jsonld_core`` read against a self-contained
bundle fixture (Option A: the read is inlined against the as-built foundation,
not a monkeypatched stub)."""

import json
from pathlib import Path

import yaml

from hhemt.bundle import _compatibility as C
from hhemt.bundle._compatibility import CompatibilitySeverity as S


def test_classify_buckets():
    assert C._classify("account", "user") is S.BLOCKING
    # The hpc branch is unit-tested DIRECTLY (no HPC field reaches the checker via
    # a real read, so it must not be exercised through a _field_bucket path).
    assert C._classify("gpu_hardware", "hpc") is S.INFORMATIONAL
    assert C._classify("toggle_triton_model", "experiment") is S.BLOCKING  # model-family identity
    assert C._classify("case_name", "experiment") is S.BLOCKING  # defensive identity entry
    assert C._classify("target_dem_resolution", "experiment") is S.WARNING  # sensitivity axis


def test_report_blocking_property():
    rep = C.CompatibilityReport(
        divergences=[
            C.CompatibilityDivergence("toggle_triton_model", "experiment", S.BLOCKING, "a", "b", True, False),
            C.CompatibilityDivergence("gpu_hardware", "hpc", S.INFORMATIONAL, "a", "b", "x", "y"),
        ]
    )
    assert not rep.is_compatible
    assert len(rep.blocking) == 1


def test_check_pairwise_warning(monkeypatch):
    # Two bundles differing only on a sensitivity axis -> one WARNING, compatible.
    # analysis_id differs (always) but is reserved -> must NOT produce a divergence.
    cores = {
        "a": {"analysis_id": "a", "toggle_triton_model": True, "target_dem_resolution": 1.0},
        "b": {"analysis_id": "b", "toggle_triton_model": True, "target_dem_resolution": 2.0},
    }
    monkeypatch.setattr(C, "_read_jsonld_core", lambda r: cores[r.name])
    rep = C.check_bundle_compatibility([Path("/x/a"), Path("/x/b")])
    assert [d.severity for d in rep.divergences] == [S.WARNING]
    assert {d.field_name for d in rep.divergences} == {"target_dem_resolution"}
    assert rep.is_compatible


def test_check_pairwise_blocking(monkeypatch):
    # Two bundles differing on the model family -> BLOCKING (different experiment).
    cores = {
        "a": {"analysis_id": "a", "toggle_triton_model": True},
        "b": {"analysis_id": "b", "toggle_triton_model": False},
    }
    monkeypatch.setattr(C, "_read_jsonld_core", lambda r: cores[r.name])
    rep = C.check_bundle_compatibility([Path("/x/a"), Path("/x/b")])
    assert not rep.is_compatible
    assert rep.blocking[0].field_name == "toggle_triton_model"


def _write_bundle(root: Path, *, sysd: dict, anad: dict, analysis_id: str, schema_version: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(yaml.safe_dump(sysd, sort_keys=False))
    (root / "cfg_analysis.yaml").write_text(yaml.safe_dump(anad, sort_keys=False))
    (root / "bundle_manifest.json").write_text(json.dumps({"analysis_id": analysis_id}))
    if schema_version is not None:
        (root / "ro-crate-metadata.json").write_text(
            json.dumps(
                {
                    "@context": "https://w3id.org/ro/crate/1.2/context",
                    "@graph": [{"@id": "./", "schemaVersion": schema_version, "analysis_id": analysis_id}],
                }
            )
        )
    return root


def test_read_jsonld_core_sources_bundled_surface(tmp_path):
    root = _write_bundle(
        tmp_path / "exp_a",
        sysd={
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "target_dem_resolution": 5.0,
            "constant_mannings": 0.03,  # not a comparison field -> must be dropped
        },
        anad={
            "weather_events_to_simulate": "weather.csv",
            "sensitivity_analysis": "sens.csv",
            "brand_theme": "brand_theme.resolved.yaml",  # not a comparison field -> dropped
        },
        analysis_id="exp_a_id",
        schema_version="3",
    )
    core = C._read_jsonld_core(root)
    assert core["toggle_tritonswmm_model"] is True
    assert core["target_dem_resolution"] == 5.0
    assert core["weather_events_to_simulate"] == "weather.csv"
    assert core["sensitivity_analysis"] == "sens.csv"
    assert core["schemaVersion"] == "3"
    assert core["analysis_id"] == "exp_a_id"  # reserved LABEL, from bundle_manifest.json
    # Non-curated fields and the never-bundled case_name are absent:
    assert "constant_mannings" not in core
    assert "brand_theme" not in core
    assert "case_name" not in core


def test_read_jsonld_core_tolerates_missing_rocrate(tmp_path):
    root = _write_bundle(
        tmp_path / "exp_b",
        sysd={"toggle_triton_model": True, "toggle_tritonswmm_model": False, "toggle_swmm_model": False},
        anad={"weather_events_to_simulate": "w.csv"},
        analysis_id="exp_b_id",
        schema_version=None,  # no ro-crate sidecar
    )
    core = C._read_jsonld_core(root)
    assert "schemaVersion" not in core  # absent sidecar -> no version field
    assert core["analysis_id"] == "exp_b_id"
    assert core["toggle_triton_model"] is True


def test_check_bundle_compatibility_end_to_end(tmp_path):
    # Real read (no monkeypatch): same model+weather, different sensitivity + version
    # -> WARNING only, compatible; always-divergent analysis_id is not flagged.
    a = _write_bundle(
        tmp_path / "a",
        sysd={
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "target_dem_resolution": 1.0,
        },
        anad={"weather_events_to_simulate": "w.csv", "sensitivity_analysis": "sensA.csv"},
        analysis_id="a_id",
        schema_version="3",
    )
    b = _write_bundle(
        tmp_path / "b",
        sysd={
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "target_dem_resolution": 2.0,
        },
        anad={"weather_events_to_simulate": "w.csv", "sensitivity_analysis": "sensB.csv"},
        analysis_id="b_id",
        schema_version="4",
    )
    rep = C.check_bundle_compatibility([a, b])
    assert rep.is_compatible  # only WARNING-level divergences
    diverged = {d.field_name for d in rep.divergences}
    assert diverged == {"target_dem_resolution", "sensitivity_analysis", "schemaVersion"}
    assert all(d.severity is S.WARNING for d in rep.divergences)
    assert "analysis_id" not in diverged  # reserved label, never compared
