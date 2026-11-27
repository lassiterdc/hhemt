"""Operator-gated, short Zenodo round trip: deposit a tiny zip -> mint a DOI -> fetch it by DOI -> compare the bytes.

The test builds a zip of a few kilobytes, deposits it through the toolkit's Zenodo adapter
(``publishing._ZenodoTarget.publish``, the code that ``publish_analysis`` hands every deposit to), and fetches it
back by the minted DOI through ``TRITON_SWMM_experiment._fetch_bundle_zip``, the call that
``from_doi(host='zenodo')`` makes. Neither side is mocked. The test asserts that the fetched bytes equal the deposited
bytes. It builds no analysis, compiles nothing, runs no simulation and requests no fixture, so it finishes in well
under two minutes. ``publish_analysis`` itself, which reads the license and title from the sidecar of an analysis, is
not exercised here; ``tests/test_doi_roundtrip_e2e.py`` leg 1 covers it on a machine that can compile and simulate.

WHAT THE TEST LEAVES BEHIND. A Zenodo publish mints a DOI that cannot be withdrawn. Against the sandbox
(``https://sandbox.zenodo.org``) the DOI is a disposable ``10.5072/...`` DOI, and the test leaves one published sandbox
record that holds the probe zip. The test prints the record URL. To remove the record, open that URL while signed in to
your sandbox account and use the delete action on the record page, or leave it, because the sandbox exists for such
records.

PRODUCTION IS REFUSED. The base URL comes from ``HHEMT_ZENODO_BASE_URL``. The toolkit defaults to production Zenodo when
the variable is unset, so the test fails, rather than publishes, unless the base URL is a sandbox or
``HHEMT_E2E_ALLOW_PRODUCTION=1`` is set as well.

To run it, set these variables in your own shell or in ``~/.config/hhemt/e2e.env``, an env file that lives outside the
repository (never paste a value into a chat or a commit)::

    HHEMT_ZENODO_TOKEN            # personal access token of the SANDBOX account; scopes deposit:write, deposit:actions
    HHEMT_ZENODO_BASE_URL         # https://sandbox.zenodo.org

Then::

    source ~/.config/hhemt/e2e.env
    conda run -n hhemt python -m pytest tests/test_zenodo_roundtrip_light_e2e.py -v

Without ``HHEMT_ZENODO_TOKEN`` the module skips and names the variable.
"""

from __future__ import annotations

import os
import time
import uuid
import zipfile
from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from hhemt import publishing
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment
from hhemt.suite.aggregate import LIVE_DEPOSIT_SKIP_REASON

E2E_ENV_FILE = "~/.config/hhemt/e2e.env"

_PRODUCTION_HOSTS = ("https://zenodo.org", "http://zenodo.org", "zenodo.org")
_WALL_LIMIT_S = 120
_RECORD_APPEARS_WITHIN_S = 60

pytestmark = [
    pytest.mark.publish_e2e,
    pytest.mark.skipif(tst_ut.in_suite_array_chunk(), reason=LIVE_DEPOSIT_SKIP_REASON),
]


def _require_token() -> None:
    if not os.environ.get("HHEMT_ZENODO_TOKEN"):
        pytest.skip(
            f"missing HHEMT_ZENODO_TOKEN; set it in {E2E_ENV_FILE} and source the file. "
            "The variable is HHEMT_ZENODO_TOKEN, not ZENODO_TOKEN."
        )


def _refuse_production() -> None:
    """Fail, never skip, when the base URL is production Zenodo and production was not allowed on purpose."""
    base = os.environ.get("HHEMT_ZENODO_BASE_URL", "https://zenodo.org").rstrip("/")
    if base in _PRODUCTION_HOSTS and os.environ.get("HHEMT_E2E_ALLOW_PRODUCTION") != "1":
        pytest.fail(
            "REFUSING to run the live round trip against PRODUCTION Zenodo.\n"
            f"  HHEMT_ZENODO_BASE_URL = {base}\n"
            "  HHEMT_E2E_ALLOW_PRODUCTION is not '1'.\n\n"
            "A published Zenodo record mints a PERMANENT, IRREVOCABLE DOI. Point the test at the sandbox:\n"
            "    export HHEMT_ZENODO_BASE_URL='https://sandbox.zenodo.org'\n"
            "(the sandbox needs its own account and its own token), or set HHEMT_E2E_ALLOW_PRODUCTION=1 when you "
            "mean to mint a real DOI."
        )


def _write_tiny_zip(path: Path) -> None:
    """A zip of a few kilobytes with content that differs on every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("probe/run_id.txt", uuid.uuid4().hex)
        zf.writestr("probe/payload.bin", os.urandom(2048))
        zf.writestr("probe/notes.txt", "round-trip probe\n" * 20)


def _fetch_when_the_record_is_readable(doi: str, dest: Path) -> Path:
    """Fetch by DOI, retrying for up to ``_RECORD_APPEARS_WITHIN_S`` while the records API answers an HTTP error.

    Zenodo can take a moment to serve a record that it just published. Any other failure raises at once.
    """
    deadline = time.monotonic() + _RECORD_APPEARS_WITHIN_S
    while True:
        try:
            return TRITON_SWMM_experiment._fetch_bundle_zip("zenodo", doi=doi, dest=dest)
        except ProcessingError as exc:
            if exc.operation != "zenodo_fetch" or time.monotonic() >= deadline:
                raise
            time.sleep(3)


def test_a_deposited_zip_round_trips_through_the_zenodo_fetch_by_doi(tmp_path) -> None:
    _require_token()
    _refuse_production()

    probe = tmp_path / "deposit" / "roundtrip_probe.zip"
    _write_tiny_zip(probe)

    started = time.monotonic()
    result = publishing._TARGETS["zenodo"]().publish(
        deposit=[probe],
        license_spdx="CC0-1.0",
        software_doi=None,
        analysis_dir=tmp_path,
    )
    doi = result.get("data_doi")
    print(f"zenodo record: {result.get('record_url')}  doi: {doi}", flush=True)
    assert doi, f"no DOI minted; publish returned {result}"

    fetched = _fetch_when_the_record_is_readable(doi, tmp_path / "ingest")
    elapsed = time.monotonic() - started

    assert fetched.name == probe.name
    assert fetched.read_bytes() == probe.read_bytes()
    assert elapsed < _WALL_LIMIT_S, f"the round trip took {elapsed:.0f} s"
