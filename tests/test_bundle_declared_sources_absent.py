"""F4b: a renderer-declared source that is absent at harvest time is RECORDED, not lost.

`_harvest_and_copy_sources` skips a declared-but-absent source and warns — ADR-6 D3 permits
renderers to declare an expected source unconditionally (e.g. `disk_utilization` declares
`_status/_du.json`, absent by design on a sensitivity master), so hard-raising would break
correct bundles. The skip is right; the evidence disappearing is not. Before F4b the only
trace was a `warnings.warn` on stderr inside a Snakemake rule log, gone by the time anyone
opened the bundle. These tests pin the accumulation and the manifest key.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from hhemt.bundle._emit import _harvest_and_copy_sources, _write_bundle_manifest


def test_harvest_records_the_absent_source_and_still_copies_the_present_one(tmp_path):
    """Emission must SUCCEED (non-fatal) while returning the absent path."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "eda").mkdir(parents=True)
    present = analysis_dir / "eda" / "present.zarr"
    present.write_text("x", encoding="utf-8")
    absent = analysis_dir / "eda" / "b4b_clean_identity.zarr"  # never created

    staging = tmp_path / "staging"
    staging.mkdir()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declared_absent = _harvest_and_copy_sources(
            {"eda_compute_sensitivity": [present, absent]}, analysis_dir, staging
        )

    assert declared_absent == ["eda/b4b_clean_identity.zarr"], "exactly the absent source"
    assert (staging / "eda" / "present.zarr").exists(), "the present source must still be copied"
    assert not (staging / "eda" / "b4b_clean_identity.zarr").exists()


def test_harvest_returns_empty_when_every_declared_source_resolves(tmp_path):
    """Differently-positioned satisfying input: a clean harvest records nothing.

    This is what keeps the manifest key ABSENT rather than empty on a correct bundle, so a
    self-contained manifest stays byte-identical to what it was before F4b existed.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "eda").mkdir(parents=True)
    a = analysis_dir / "eda" / "a.zarr"
    a.write_text("x", encoding="utf-8")

    staging = tmp_path / "staging"
    staging.mkdir()

    assert _harvest_and_copy_sources({"r": [a]}, analysis_dir, staging) == []


def test_manifest_carries_declared_sources_absent_only_when_non_empty(tmp_path):
    """The manifest that made a false claim also carries the record of it."""
    staging = tmp_path / "staging"
    staging.mkdir()

    _write_bundle_manifest(
        staging,
        sources_by_renderer={"r": [Path("eda/a.zarr")]},
        analysis_id="aid",
        git_sha="deadbeef",
        declared_sources_absent=["eda/b4b_clean_identity.zarr"],
    )
    manifest = json.loads((staging / "bundle_manifest.json").read_text())
    assert manifest["declared_sources_absent"] == ["eda/b4b_clean_identity.zarr"]

    staging2 = tmp_path / "staging2"
    staging2.mkdir()
    _write_bundle_manifest(
        staging2,
        sources_by_renderer={"r": [Path("eda/a.zarr")]},
        analysis_id="aid",
        git_sha="deadbeef",
        declared_sources_absent=[],
    )
    clean = json.loads((staging2 / "bundle_manifest.json").read_text())
    assert "declared_sources_absent" not in clean, "absent, not empty, on a clean bundle"
