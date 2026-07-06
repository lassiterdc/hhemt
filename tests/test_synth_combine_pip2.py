"""PIP-2 proving test — combine two synthetic MPI/resume bundles end-to-end (R10).

Hermetic: reuses the cached rendered-sensitivity synth fixture (no HPC). File-scope
requires_snakemake_subprocess marker because the session fixture invokes snakemake
once at first resolution (pytest-xdist nested-parallelism guard — same as synth_08).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.bundle import CombinedBundle, combine_bundle

pytestmark = pytest.mark.requires_snakemake_subprocess


def test_combine_pip2_roundtrip(synthetic_two_bundle_fixture):
    a, b = synthetic_two_bundle_fixture
    cb = combine_bundle([a, b])  # raises if a BLOCKING divergence exists (none here)
    assert (cb.root / "bundle_manifest.json").exists()
    crate = json.loads((cb.root / "ro-crate-metadata.json").read_text())
    # flat hasPart-by-reference over the two intact child crates (NOT N mainEntity):
    root_ds = next(e for e in crate["@graph"] if e.get("@id") == "./")
    haspart = root_ds.get("hasPart", [])
    assert len(haspart) >= 2, f"expected >=2 child crates in hasPart, got {haspart}"
    # round-trip: reconstruct + regenerate the combined report:
    report = CombinedBundle.from_directory(cb.root).regenerate_report()
    assert Path(report).exists()
