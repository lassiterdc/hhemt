"""Operator-gated, short HydroShare round trip: stage a tiny zip -> fetch it back -> compare the bytes.

The toolkit fetches the bundle of a HydroShare deposit through ``TRITON_SWMM_experiment._fetch_bundle_zip``, which is
the call ``from_doi(host='hydroshare')`` makes. This test stages a zip of a few kilobytes, built here, into the
operator's PRIVATE HydroShare resource, fetches it back through that function with the bag wait bounded to 120 s, and
asserts that the bytes are equal. It runs no simulation, compiles nothing and needs no fixture. It never publishes the
resource, never mints a DOI and never changes the sharing of the resource. The finally clause removes the staged file.

The reconstitute-and-run half of the round trip is covered without a network by
``tests/test_synth_09_from_doi_run_proof.py``.

To run it, set these three variables in your own shell or in ``~/.config/hhemt/e2e.env``, an env file that lives outside
the repository (never paste a value into a chat or a commit)::

    HHEMT_HYDROSHARE_USERNAME      # bare id OR institutional email; try the other form if authentication fails
    HHEMT_HYDROSHARE_PASSWORD      # hsclient has no token support, so this is the account password
    HHEMT_E2E_HYDROSHARE_RESOURCE  # the 32-hex id of a resource you own; the test clears its files

Then::

    source ~/.config/hhemt/e2e.env
    conda run -n hhemt python -m pytest tests/test_hydroshare_roundtrip_light_e2e.py -v

With any of the three blank the module skips and names the variable that is missing.
"""

from __future__ import annotations

import os
import time
import uuid
import zipfile
from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from hhemt.experiments import TRITON_SWMM_experiment
from hhemt.suite.aggregate import LIVE_DEPOSIT_SKIP_REASON

E2E_ENV_FILE = "~/.config/hhemt/e2e.env"

_BAG_DEADLINE_S = "120"
_WALL_LIMIT_S = 120

pytestmark = [
    pytest.mark.publish_e2e,
    pytest.mark.skipif(tst_ut.in_suite_array_chunk(), reason=LIVE_DEPOSIT_SKIP_REASON),
]


def _require_hydroshare() -> str:
    """Skip unless the credentials and a resource id are all present; return the resource id."""
    missing = [
        v
        for v in ("HHEMT_HYDROSHARE_USERNAME", "HHEMT_HYDROSHARE_PASSWORD", "HHEMT_E2E_HYDROSHARE_RESOURCE")
        if not os.environ.get(v)
    ]
    if missing:
        pytest.skip(
            f"the HydroShare round trip needs {', '.join(missing)}; set them in {E2E_ENV_FILE}. "
            "The test stages a tiny zip into your PRIVATE resource and never publishes or mints a DOI."
        )
    return os.environ["HHEMT_E2E_HYDROSHARE_RESOURCE"]


def _clear_resource_files(resource) -> None:
    """Best-effort: remove every file from the resource so it holds exactly the one staged zip."""
    for f in resource.files():
        try:
            resource.file_delete(path=f.path)
        except Exception:  # noqa: BLE001 - cleanup is best-effort; a stuck file is not fatal
            pass


def _write_tiny_zip(path: Path) -> None:
    """A zip of a few kilobytes with content that differs on every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("probe/run_id.txt", uuid.uuid4().hex)
        zf.writestr("probe/payload.bin", os.urandom(2048))
        zf.writestr("probe/notes.txt", "round-trip probe\n" * 20)


def test_a_staged_zip_round_trips_through_the_hydroshare_fetch_in_under_two_minutes(tmp_path, monkeypatch) -> None:
    resource_id = _require_hydroshare()
    monkeypatch.setenv("HHEMT_HYDROSHARE_BAG_TIMEOUT_S", _BAG_DEADLINE_S)

    staged = tmp_path / "staged" / "roundtrip_probe.zip"
    _write_tiny_zip(staged)

    # Authenticated staging is test-side setup, not the code under test. The toolkit's own connection helper gives the
    # credentialed session.
    hs = TRITON_SWMM_experiment._connect_to_hydroshare(resource_id)
    resource = hs.resource(resource_id)
    _clear_resource_files(resource)
    resource.file_upload(str(staged))

    try:
        started = time.monotonic()
        fetched = TRITON_SWMM_experiment._fetch_bundle_zip(
            "hydroshare", pid=resource_id, res_identifier=resource_id, dest=tmp_path / "ingest"
        )
        elapsed = time.monotonic() - started
        assert fetched.read_bytes() == staged.read_bytes()
        assert elapsed < _WALL_LIMIT_S, f"the fetch took {elapsed:.0f} s"
    finally:
        _clear_resource_files(resource)
