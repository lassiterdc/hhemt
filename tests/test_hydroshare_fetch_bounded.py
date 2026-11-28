"""The HydroShare fetch never waits without a limit and fetches a one-zip resource without a bag.

HydroShare builds the BagIt archive of a resource asynchronously. ``Resource.download`` of ``hsclient`` answers
"not ready" by sleeping and calling itself again with no limit, so an archive that never gets built hangs the fetch.
These tests hold the toolkit's replacement to a deadline, and hold the direct download of the single bundle zip,
which asks the server for no archive. Every test uses a fake HydroShare client, so none touches the network,
compiles anything or runs a simulation.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment
from tests.hydroshare_fakes import FakeBagHS, FakeFilesHS

_RID = "0123456789abcdef0123456789abcdef"
_ZIP_BYTES = b"PK-not-a-real-zip-but-the-bytes-are-what-is-compared" * 40


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()  # noqa: S324 - HydroShare lists md5 as the checksum of a file


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _connect_to(monkeypatch, fake) -> None:
    monkeypatch.setattr(TRITON_SWMM_experiment, "_connect_to_hydroshare", classmethod(lambda cls, _key: fake))


class _FakeClock:
    """A clock whose ``sleep`` advances ``monotonic``, and whose ``monotonic`` fails the test after ``max_reads`` reads.

    A deadline that never fires reads the clock without end, so the read cap turns that hang into an assertion.
    """

    def __init__(self, max_reads: int = 200):
        self.now = 0.0
        self.reads = 0
        self.max_reads = max_reads

    def monotonic(self) -> float:
        self.reads += 1
        assert self.reads <= self.max_reads, f"the deadline never fired in {self.max_reads} clock reads"
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, f"slept a negative length: {seconds}"
        self.now += seconds


def test_a_bag_that_is_never_ready_raises_at_the_deadline_with_the_server_status(tmp_path, monkeypatch):
    """The deadline fires after a bounded number of polls, and the message names what an operator needs.

    Input class: a server that answers "Not ready" to every poll, on a fake clock that sleep advances and that fails
    the test after 200 reads. A wait with no limit trips the read cap, and an error that names none of the resource,
    the status and the task id sends the operator to the library source.
    """
    clock = _FakeClock()
    monkeypatch.setattr("hhemt.experiments.time", clock)
    monkeypatch.setenv("HHEMT_HYDROSHARE_BAG_TIMEOUT_S", "1")
    monkeypatch.setattr(TRITON_SWMM_experiment, "_HYDROSHARE_BAG_POLL_INTERVAL_S", 0.1)
    fake = FakeBagHS(None)
    with pytest.raises(ProcessingError) as ei:
        TRITON_SWMM_experiment._fetch_deposit_files(
            "hydroshare", res_identifier=_RID, dest=tmp_path / "out" / "case", hs=fake, validate=False
        )
    assert ei.value.operation == "hydroshare_bag_wait"
    reason = str(ei.value.reason)
    assert _RID in reason
    assert "Not ready" in reason
    assert "task-0001" in reason
    assert "1 s" in reason
    assert "HHEMT_HYDROSHARE_BAG_TIMEOUT_S" in reason
    assert len(fake.session.get_calls) > 1, "the fetch must poll more than once before it gives up"
    assert clock.now == pytest.approx(1.0), "the wait must end at the deadline, not before or after"


@pytest.mark.parametrize("bad", ["0", "-5", "soon", "inf", "nan"])
def test_a_bad_deadline_setting_is_refused_and_names_the_variable(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("HHEMT_HYDROSHARE_BAG_TIMEOUT_S", bad)
    with pytest.raises(ConfigurationError) as ei:
        TRITON_SWMM_experiment._fetch_deposit_files(
            "hydroshare", res_identifier=_RID, dest=tmp_path / "out" / "case", hs=FakeBagHS(None), validate=False
        )
    assert "HHEMT_HYDROSHARE_BAG_TIMEOUT_S" in str(ei.value)


def test_a_resource_with_one_zip_is_fetched_as_that_file_with_no_bag(tmp_path, monkeypatch):
    """Property: the bytes come back equal and the archive endpoint is never asked for.

    The fake session of ``FakeFilesHS`` fails the test when the fetch polls the bag endpoint.
    """
    fake = FakeFilesHS({"bundle.zip": (_ZIP_BYTES, _md5(_ZIP_BYTES)), "notes.txt": (b"x", _md5(b"x"))})
    _connect_to(monkeypatch, fake)
    got = TRITON_SWMM_experiment._fetch_bundle_zip("hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "in")
    assert got.read_bytes() == _ZIP_BYTES
    assert got.parent == tmp_path / "in"
    assert fake.resource_obj.downloaded == ["bundle.zip"]


def test_a_download_that_differs_from_the_listed_md5_is_refused(tmp_path, monkeypatch):
    fake = FakeFilesHS({"bundle.zip": (_ZIP_BYTES, _md5(b"other bytes"))})
    _connect_to(monkeypatch, fake)
    with pytest.raises(ProcessingError) as ei:
        TRITON_SWMM_experiment._fetch_bundle_zip("hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "in")
    assert ei.value.operation == "hydroshare_file_verification"
    assert "md5 mismatch" in str(ei.value.reason)


def test_a_multipart_etag_is_refused_without_a_sha256_and_accepted_with_one(tmp_path, monkeypatch):
    fake = FakeFilesHS({"bundle.zip": (_ZIP_BYTES, _md5(_ZIP_BYTES) + "-2")})
    _connect_to(monkeypatch, fake)
    with pytest.raises(ProcessingError) as ei:
        TRITON_SWMM_experiment._fetch_bundle_zip("hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "a")
    assert ei.value.operation == "hydroshare_file_verification"
    assert "sha256" in str(ei.value.reason)
    assert fake.resource_obj.downloaded == [], "an unverifiable file must be refused before it is downloaded"

    got = TRITON_SWMM_experiment._fetch_bundle_zip(
        "hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "b", expected_sha256=_sha256(_ZIP_BYTES)
    )
    assert got.read_bytes() == _ZIP_BYTES

    with pytest.raises(ProcessingError) as ei2:
        TRITON_SWMM_experiment._fetch_bundle_zip(
            "hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "c", expected_sha256="0" * 64
        )
    assert ei2.value.operation == "doi_ingest_sha256"


def test_a_resource_with_two_zips_goes_to_the_bag_path(tmp_path, monkeypatch):
    """Two zips are not the single-bundle shape, so the fetch asks for the bag (the fake fails on that ask)."""
    fake = FakeFilesHS({"a.zip": (b"a", _md5(b"a")), "b.zip": (b"b", _md5(b"b"))})
    _connect_to(monkeypatch, fake)
    with pytest.raises(AssertionError, match="BagIt archive"):
        TRITON_SWMM_experiment._fetch_bundle_zip("hydroshare", pid=_RID, res_identifier=_RID, dest=tmp_path / "in")
    assert fake.resource_obj.downloaded == []


def test_the_installed_hsclient_still_exposes_what_the_bounded_bag_download_uses():
    """Fail loudly here, not in a three-hour hang, if ``hsclient`` drops what ``_download_hydroshare_bag`` relies on.

    The bounded bag download reads ``_hs_session`` and ``_hsapi_path`` of a ``Resource`` and calls the session's
    ``get`` (with ``status_code`` and request keywords) and ``write_file``. The single-zip fetch calls
    ``Resource.files`` and ``Resource.file_download`` and reads ``checksum`` of each listed file.
    """
    from hsclient.hydroshare import Aggregation, File, HydroShareSession, Resource

    session = HydroShareSession("www.hydroshare.org", "https", 443)
    # Resource.__init__ asks the server for the resource, so the test builds the object without it and reads what the
    # base initializer stores. The path property answers from the server too, so the test checks only that it exists.
    resource = Resource.__new__(Resource)
    Aggregation.__init__(resource, f"/resource/{_RID}/data/resourcemap.xml", session)
    assert resource._hs_session is session
    assert isinstance(Resource._hsapi_path, property)

    get_params = inspect.signature(HydroShareSession.get).parameters
    assert list(get_params)[:3] == ["self", "path", "status_code"]
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in get_params.values()), (
        "session.get must forward request keywords such as timeout and allow_redirects"
    )
    write_params = list(inspect.signature(HydroShareSession.write_file).parameters)
    assert write_params[:4] == ["self", "path", "content", "save_path"]

    assert "save_path" in inspect.signature(Resource.file_download).parameters
    assert callable(Resource.files)
    assert isinstance(File.checksum, property)
