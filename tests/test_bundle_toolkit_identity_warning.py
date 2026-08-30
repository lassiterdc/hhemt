"""The consume-side toolkit-identity comparison in ``Bundle.from_directory``.

The bundle manifest has always carried ``toolkit_git_sha`` (written at
``bundle/_emit.py:1413``); until the comparison landed, nothing on the consume side
read it back, so re-rendering a bundle on a drifted checkout was silent.

The POSITIVE arm below is the discriminating one: it fails against pre-fix source,
where ``from_directory`` emits no warning at all. The two NEGATIVE arms cannot fail
pre-fix -- with no warning ever emitted, "does not warn" is true vacuously -- and are
present as REGRESSION guards on the two guard clauses that shape the comparison, each
of which a naive ``!=`` implementation gets wrong.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import pytest

from hhemt.bundle import Bundle
from hhemt.bundle._emit import _get_toolkit_git_sha

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bundles"
MULTI_SIM_FIXTURE = FIXTURES_DIR / "multi_sim"

# A valid-looking sha that is not this checkout's. Fixed rather than random so a
# failure names the same value every time.
FOREIGN_SHA = "deadbeefcafe"


def _local_sha() -> str:
    sha = _get_toolkit_git_sha(strict=False)
    if sha == "unknown":
        pytest.skip("toolkit is not a git checkout; the comparison is guarded off and the arms would pass vacuously")
    return sha


def _bundle_with_sha(tmp_path: Path, sha: str | None) -> Path:
    """Copy the checked-in fixture and set its manifest ``toolkit_git_sha``."""
    dest = tmp_path / "bundle"
    shutil.copytree(MULTI_SIM_FIXTURE, dest)
    manifest_path = dest / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha is None:
        manifest.pop("toolkit_git_sha", None)
    else:
        manifest["toolkit_git_sha"] = sha
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dest


def _warnings_from_open(bundle_dir: Path) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Bundle.from_directory(bundle_dir)
    return [str(w.message) for w in caught]


# --------------------------------------------------------------------------- #
# POSITIVE -- the discriminating arm. FAILS against pre-fix source.
# --------------------------------------------------------------------------- #
def test_drifted_producing_toolkit_warns(tmp_path: Path) -> None:
    local = _local_sha()
    assert not local.startswith(FOREIGN_SHA), "test constant collides with the live sha"
    messages = _warnings_from_open(_bundle_with_sha(tmp_path, FOREIGN_SHA))
    assert any(FOREIGN_SHA in m and local in m for m in messages), (
        f"expected a provenance warning naming both {FOREIGN_SHA} and {local}; got {messages}"
    )


# --------------------------------------------------------------------------- #
# NEGATIVE -- regression guards. Both pass vacuously pre-fix, by construction.
# --------------------------------------------------------------------------- #
def test_non_sha_sentinel_does_not_warn(tmp_path: Path) -> None:
    """The checked-in fixtures ship ``toolkit_git_sha: "fixture"``.

    A bare ``!=`` fires here, and would do so at every ``from_directory`` call site
    in the suite. The ``looks_like_sha`` guard is what prevents that.
    """
    _local_sha()
    messages = _warnings_from_open(_bundle_with_sha(tmp_path, "fixture"))
    assert messages == [], f"non-sha sentinel must not warn; got {messages}"


def test_shorter_matching_prefix_does_not_warn(tmp_path: Path) -> None:
    """An 8-char manifest abbreviation of the SAME commit is not a divergence.

    ``_get_toolkit_git_sha`` emits ``--short=12``; a bundle abbreviated differently
    names the same object, and a bare ``!=`` reports it as drift. The prefix test
    must run in BOTH directions for this to hold.
    """
    local = _local_sha()
    messages = _warnings_from_open(_bundle_with_sha(tmp_path, local[:8]))
    assert messages == [], f"a shorter abbreviation of the same commit must not warn; got {messages}"


def test_absent_toolkit_sha_does_not_warn(tmp_path: Path) -> None:
    """A legacy manifest with no ``toolkit_git_sha`` key is skipped, not flagged."""
    _local_sha()
    messages = _warnings_from_open(_bundle_with_sha(tmp_path, None))
    assert messages == [], f"a manifest without the key must not warn; got {messages}"
