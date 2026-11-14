"""Member bundles and combined bundles carry separate schema constants.

Problem: one constant gated both layouts, so the combined layout's bump (commit 930c37eb) made
``hhemt combine`` refuse every schema-7 member bundle, which it reads through ``Bundle.from_directory``.
These tests pin the split: members at ``BUNDLE_SCHEMA_VERSION`` (7) and combined bundles at
``COMBINED_BUNDLE_SCHEMA_VERSION`` (8), each read and stamped by its own constant.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hhemt.bundle import BundleSchemaError
from hhemt.bundle._combine import CombinedBundle, _write_combined_bundle_manifest
from hhemt.bundle._combine_merge import _experiment_id
from hhemt.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    COMBINED_BUNDLE_SCHEMA_VERSION,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "bundles" / "multi_sim"


def _member(tmp_path: Path, version: int) -> Path:
    """A member bundle directory with the fixture cfg and a manifest stamped ``version``."""
    root = tmp_path / f"member_{version}"
    root.mkdir()
    shutil.copy(_FIXTURE / "cfg_analysis.yaml", root / "cfg_analysis.yaml")
    manifest = json.loads((_FIXTURE / BUNDLE_MANIFEST_FILENAME).read_text())
    manifest["bundle_schema_version"] = version
    (root / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return root


def _combined(tmp_path: Path, version: int) -> Path:
    root = tmp_path / f"combined_{version}"
    root.mkdir()
    (root / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps({"bundle_schema_version": version}))
    return root


def test_a_schema_7_member_bundle_is_read_by_the_member_reader_and_by_combine(tmp_path: Path) -> None:
    """Property: a member manifest at 7 loads through ``Bundle.from_directory`` and the combine's id read.

    Input class: member bundle directories at the member constant's value. Why required: the combine reads
    every input through ``_experiment_id`` -> ``Bundle.from_directory`` (job 22392651 failed there).
    Class: REGRESSION, red on the pre-change tree where the member constant is 8. What kills it: setting
    the member constant back to 8. Second correct implementation: a member reader accepting a set of
    versions containing 7 passes too, since the test turns on 7 being accepted.
    """
    from hhemt.bundle import Bundle

    member = _member(tmp_path, 7)
    assert BUNDLE_SCHEMA_VERSION == 7
    assert Bundle.from_directory(member).root == member.resolve()
    assert _experiment_id(member)


@pytest.mark.parametrize("version", [6, 9])
def test_a_member_bundle_with_an_unknown_version_is_still_refused(tmp_path: Path, version: int) -> None:
    """Property: member versions other than the member constant raise ``BundleSchemaError``, both directions.

    Input class: member manifests at 6 (below) and 9 (above), positioned either side of 7. Class:
    REGRESSION. What kills it: a member reader that stops comparing. Second correct implementation: an
    inequality guard in place of the two ordered comparisons.
    """
    from hhemt.bundle import Bundle

    with pytest.raises(BundleSchemaError):
        Bundle.from_directory(_member(tmp_path, version))


@pytest.mark.parametrize("version", [BUNDLE_SCHEMA_VERSION, COMBINED_BUNDLE_SCHEMA_VERSION + 1])
def test_the_combined_reader_refuses_the_old_combined_version_and_a_newer_one(tmp_path: Path, version: int) -> None:
    """Property: ``CombinedBundle.from_directory`` refuses 7 (the old combined version) and 9.

    Input class: combined manifests at 7 and 9. Class: REGRESSION for 7 (red if the combined reader tested
    the member constant), NEW CAPABILITY for 9. What kills it: a combined guard that compares the member
    constant. Second correct implementation: any exact-match check on the combined constant.
    """
    with pytest.raises(BundleSchemaError):
        CombinedBundle.from_directory(_combined(tmp_path, version))


def test_the_combined_reader_accepts_the_combined_version(tmp_path: Path) -> None:
    """Property: a combined manifest at 8 loads. Class: NEW CAPABILITY; what kills it: a guard on 7."""
    root = _combined(tmp_path, COMBINED_BUNDLE_SCHEMA_VERSION)
    assert CombinedBundle.from_directory(root).root == root.resolve()


def test_each_emitter_stamps_its_own_constant(tmp_path: Path) -> None:
    """Property: the member writer stamps the member constant and the combined writer the combined one.

    Input class: the two manifest writers. Class: REGRESSION. What kills it: either writer stamping the
    other constant. Second correct implementation: any pair of writers whose stamps equal the constants.
    """
    from hhemt.bundle._emit import _write_bundle_manifest

    member_staging = tmp_path / "member_out"
    member_staging.mkdir()
    _write_bundle_manifest(member_staging, {}, "a1", "deadbeef")
    member = json.loads((member_staging / BUNDLE_MANIFEST_FILENAME).read_text())
    assert member["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION

    combined_out = tmp_path / "combined_out"
    combined_out.mkdir()
    _write_combined_bundle_manifest(combined_out, experiment_ids=[], child_crates=[], git_sha="deadbeef", roots=[])
    combined = json.loads((combined_out / BUNDLE_MANIFEST_FILENAME).read_text())
    assert combined["bundle_schema_version"] == COMBINED_BUNDLE_SCHEMA_VERSION
