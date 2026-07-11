"""Phase 3 (reproducibility C8, ADR-10): the consumer-facing reprex round-trip.

Exercises ``TRITONSWMM_sensitivity_analysis.reprex_bundle()`` +
``Bundle.reprex(reprex_config, target_hpc_profile)`` on a rendered synth sensitivity
master (the PRIMARY reprex surface), plus the degenerate single-analysis dispatch via
``TRITONSWMM_analysis.reprex_bundle()``.

Covers R14-R18:
- SIF verify: the ``synthetic`` fixtures are NATIVE CPU runs, so their crate carries no
  SIF entity — reprex reports ``sif_reference_present=False`` and vacuously passes. The
  container mandatory-digest-match path is exercised by injecting a SIF ``SoftwareApplication``
  + a matching fake SIF into the emitted bundle (a native test box cannot build a real
  signed SIF); a mismatch is fail-closed (``ProcessingError``).
- Preflight re-aim + per-``(sa_id, column)`` problem pairs for a seeded resource-exceeding row.
- Per-field amendments, each labelled validated-vs-advisory.
- Both emit-side facades dispatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from hhemt.bundle import Bundle
from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
from hhemt.config.reprex_config import reprex_config
from hhemt.exceptions import ProcessingError


def _target_profile(*, max_gpu: int = 1) -> hpc_system_config:
    """A minimal reproducer HPC profile: one declared 'gpu' partition with a low GPU cap."""
    return hpc_system_config(
        system_name="target-cluster",
        partitions={"gpu": PartitionSpec(max_gpu=max_gpu, max_runtime=60)},
    )


def _reprex_cfg(sif_path: Path) -> reprex_config:
    return reprex_config(
        default_account="target-alloc",
        sif_path=sif_path,
        target_ensemble_partition="gpu",
    )


def _seed_resource_exceeding_row(bundle_dir: Path) -> None:
    """Point the bundle's cfg_analysis at a minimal one-row sensitivity CSV whose
    ``n_gpus`` request (8) exceeds the target 'gpu' partition cap (1), so the per-row
    cap scan emits a ``(sa_id, column)`` problem pair when reprex re-aims at the target."""
    cfg_path = bundle_dir / "cfg_analysis.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["toggle_sensitivity_analysis"] = True
    cfg["sensitivity_analysis"] = "sensitivity_analysis_definition.csv"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (bundle_dir / "sensitivity_analysis_definition.csv").write_text("n_gpus\n8\n")


def _inject_sif(bundle_dir: Path, *, sif_bytes: bytes = b"REFERENCE-SIF-BYTES") -> Path:
    """Write a fake SIF into the bundle and register it in the crate as a by-reference
    SoftwareApplication with its sha256 (the container-run crate shape). Returns the SIF path."""
    sif_path = bundle_dir / "tritonswmm.sif"
    sif_path.write_bytes(sif_bytes)
    digest = hashlib.sha256(sif_bytes).hexdigest()
    crate_path = bundle_dir / "ro-crate-metadata.json"
    doc = json.loads(crate_path.read_text())
    doc["@graph"].append(
        {
            "@id": "tritonswmm.sif",
            "@type": "SoftwareApplication",
            "name": "TRITON-SWMM Apptainer container",
            "softwareVersion": "test",
            "sha256": digest,
            "downloadUrl": "https://example.org/tritonswmm.sif",
        }
    )
    crate_path.write_text(json.dumps(doc, indent=2))
    return sif_path


def test_reprex_roundtrip_native_problem_pairs_and_amendments(
    rendered_synth_sensitivity, tmp_path: Path
) -> None:
    """R14-R17: emit a reprex bundle from the sensitivity master, then round-trip it.

    The native fixture has no SIF in its crate (sif_reference_present is False, verify is
    a vacuous pass). Re-aiming preflight at a low-cap target profile with a seeded
    resource-exceeding row emits the (sa_id, column) problem pair and marks the run not
    runnable; amendments are labelled validated-vs-advisory."""
    sensitivity = rendered_synth_sensitivity.sensitivity
    bundle_dir = sensitivity.reprex_bundle(output_path=tmp_path / "reprex_native.zip")
    assert bundle_dir.is_dir()

    _seed_resource_exceeding_row(bundle_dir)

    result = Bundle.from_directory(bundle_dir).reprex(
        _reprex_cfg(tmp_path / "unused.sif"), _target_profile(max_gpu=1)
    )

    # Native bundle: nothing to verify, vacuous pass.
    assert result.sif_reference_present is False
    assert result.sif_verified is True
    assert result.sif_signature_ok is None

    # Per-(sa_id, column) problem pair for the seeded row.
    assert result.problem_pairs, "expected a problem pair for the resource-exceeding row"
    assert any(i.field.endswith(".n_gpus") for i in result.problem_pairs)
    assert result.runnable is False

    # Amendments carry both validated (declared target partition selector) and advisory labels.
    assert result.amendments
    assert all(a.status in {"validated", "advisory"} for a in result.amendments)
    assert any(a.status == "validated" for a in result.amendments)
    assert any(a.status == "advisory" for a in result.amendments)

    # The consume-side zero-user-info scan is informational (never fatal) this phase.
    assert isinstance(result.zero_user_info_leaks, list)


def test_reprex_sif_digest_match_and_mismatch(
    rendered_synth_sensitivity, tmp_path: Path
) -> None:
    """R14: when the crate references a SIF (container bundle), reprex does a mandatory
    fail-closed sha256 digest match; PGP warns (sif_signature_ok is None) when apptainer
    is absent. A digest mismatch raises ProcessingError before any validation."""
    sensitivity = rendered_synth_sensitivity.sensitivity
    bundle_dir = sensitivity.reprex_bundle(output_path=tmp_path / "reprex_container.zip")
    sif_path = _inject_sif(bundle_dir, sif_bytes=b"REFERENCE-SIF-BYTES")

    bundle = Bundle.from_directory(bundle_dir)

    # Digest match: sif verified; PGP best-effort (None when apptainer unavailable).
    ok = bundle.reprex(_reprex_cfg(sif_path), _target_profile())
    assert ok.sif_reference_present is True
    assert ok.sif_verified is True
    assert ok.sif_signature_ok in (True, False, None)

    # Digest MISMATCH: a different-bytes SIF at the target path is fail-closed.
    wrong = bundle_dir / "wrong.sif"
    wrong.write_bytes(b"TAMPERED-SIF-BYTES")
    with pytest.raises(ProcessingError):
        bundle.reprex(_reprex_cfg(wrong), _target_profile())


def test_analysis_reprex_bundle_degenerate_dispatch(
    rendered_synth_multi_sim, tmp_path: Path
) -> None:
    """R18: the flat single-analysis facade dispatches; the zero-sensitivity degenerate
    case has no rows to cap-scan, so it is runnable with no problem pairs."""
    bundle_dir = rendered_synth_multi_sim.reprex_bundle(
        output_path=tmp_path / "reprex_single.zip"
    )
    assert bundle_dir.is_dir()

    result = Bundle.from_directory(bundle_dir).reprex(
        _reprex_cfg(tmp_path / "unused.sif"), _target_profile()
    )
    assert result.sif_reference_present is False
    assert result.problem_pairs == []
    assert result.runnable is True
