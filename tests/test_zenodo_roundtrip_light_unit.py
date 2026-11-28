"""Token-free tests of the short Zenodo deposit-and-fetch test: its retry, its production guard and its byte comparison.

``tests/test_zenodo_roundtrip_light_e2e.py`` skips without ``HHEMT_ZENODO_TOKEN``, so a regression in its retry loop,
its guard or its comparison goes unseen on a machine without a token. These tests call that module's own functions
against a fake Zenodo that stands in for ``requests``, the HTTP library every Zenodo call of the toolkit uses, so the
real ``_ZenodoTarget.publish``, ``_fetch_bundle_zip`` and ``_fetch_file_by_url`` run and nothing touches the network.
"""

from __future__ import annotations

import pytest

import tests.test_zenodo_roundtrip_light_e2e as light
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment

_BASE = "https://sandbox.zenodo.org"


class _Resp:
    def __init__(self, *, status=200, body=None, content=b""):
        self.status_code = status
        self._body = body if body is not None else {}
        self._content = content
        self.text = str(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeZenodo:
    """Serves create/upload/publish for a deposit and the records API and file URL for a fetch."""

    def __init__(self, *, corrupt=False, record_not_ready_polls=0, record_status_forever=None):
        self.corrupt = corrupt
        self.not_ready = record_not_ready_polls
        self.forever = record_status_forever
        self.files: dict[str, bytes] = {}
        self.key = None
        self.record_gets = 0

    # --- requests.Session surface used by publish
    def session(self):
        outer = self

        class S:
            headers: dict = {}

            def post(self, url, json=None, **kw):
                if url.endswith("/actions/publish") or "publish" in url.rsplit("/", 1)[-1]:
                    return _Resp(body={"pids": {"doi": {"identifier": "10.5072/zenodo.777"}}, "links": {}})
                if json and isinstance(json, list):
                    outer.key = json[0]["key"]
                    return _Resp(body={})
                if "commit" in url:
                    return _Resp(body={})
                return _Resp(body={"id": 777, "links": {}})

            def put(self, url, json=None, data=None, **kw):
                if data is not None:
                    outer.files[outer.key] = data.read()
                return _Resp(body={})

            def get(self, url, **kw):
                return _Resp(body={})

        return S()

    # --- requests.get surface used by the fetch
    def get(self, url, stream=False, timeout=None, **kw):
        if "/api/records/" in url:
            self.record_gets += 1
            if self.forever is not None:
                return _Resp(status=self.forever)
            if self.record_gets <= self.not_ready:
                return _Resp(status=404)
            return _Resp(body={"files": [{"key": k, "links": {"self": f"{_BASE}/files/{k}"}} for k in self.files]})
        key = url.rsplit("/", 1)[-1]
        data = self.files[key]
        return _Resp(content=(data + b"X") if self.corrupt else data)


@pytest.fixture
def zenodo(monkeypatch):
    def make(**kw):
        fake = _FakeZenodo(**kw)
        monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "t")
        monkeypatch.setenv("HHEMT_ZENODO_BASE_URL", _BASE)
        monkeypatch.delenv("HHEMT_E2E_ALLOW_PRODUCTION", raising=False)
        monkeypatch.setattr("hhemt.publishing.requests.Session", lambda: fake.session())
        monkeypatch.setattr("requests.get", fake.get)
        monkeypatch.setattr(light.time, "sleep", lambda s: None)
        return fake

    return make


def test_the_light_round_trip_passes_against_a_fake_zenodo_and_the_doi_parses_to_the_record_id(tmp_path, zenodo):
    fake = zenodo()
    light.test_a_deposited_zip_round_trips_through_the_zenodo_fetch_by_doi(tmp_path)
    assert fake.record_gets == 1


def test_a_record_that_is_not_served_yet_is_retried_until_it_appears(tmp_path, zenodo):
    fake = zenodo(record_not_ready_polls=3)
    light.test_a_deposited_zip_round_trips_through_the_zenodo_fetch_by_doi(tmp_path)
    assert fake.record_gets == 4


def test_a_byte_mismatch_is_never_swallowed_by_the_retry(tmp_path, zenodo):
    zenodo(corrupt=True)
    with pytest.raises(AssertionError):
        light.test_a_deposited_zip_round_trips_through_the_zenodo_fetch_by_doi(tmp_path)


def test_the_retry_gives_up_at_its_deadline_and_raises_the_fetch_error(tmp_path, zenodo, monkeypatch):
    fake = zenodo(record_status_forever=500)
    clock = {"t": 0.0}
    monkeypatch.setattr(light.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(light.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    with pytest.raises(ProcessingError) as ei:
        light._fetch_when_the_record_is_readable("10.5072/zenodo.777", tmp_path / "x")
    assert ei.value.operation == "zenodo_fetch"
    assert fake.record_gets > 1
    assert clock["t"] >= light._RECORD_APPEARS_WITHIN_S - 3


def test_an_error_that_is_not_a_missing_record_raises_at_once(tmp_path, monkeypatch):
    calls = []
    clock = {"t": 0.0}
    monkeypatch.setattr(light.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(light.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))

    def boom(*a, **k):
        calls.append(1)
        raise ProcessingError(operation="doi_ingest_bundle_zip", filepath=None, reason="two zips")

    monkeypatch.setattr(TRITON_SWMM_experiment, "_fetch_bundle_zip", classmethod(lambda cls, *a, **k: boom()))
    with pytest.raises(ProcessingError):
        light._fetch_when_the_record_is_readable("10.5072/zenodo.1", tmp_path / "x")
    assert calls == [1]


@pytest.mark.parametrize(
    "env", [{}, {"HHEMT_ZENODO_BASE_URL": "https://zenodo.org"}, {"HHEMT_ZENODO_BASE_URL": "https://zenodo.org/"}]
)
def test_production_is_refused_by_failing_not_skipping(monkeypatch, env):
    monkeypatch.delenv("HHEMT_ZENODO_BASE_URL", raising=False)
    monkeypatch.delenv("HHEMT_E2E_ALLOW_PRODUCTION", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(pytest.fail.Exception):
        light._refuse_production()


def test_production_is_allowed_only_on_purpose_and_the_sandbox_passes(monkeypatch):
    monkeypatch.setenv("HHEMT_ZENODO_BASE_URL", "https://zenodo.org")
    monkeypatch.setenv("HHEMT_E2E_ALLOW_PRODUCTION", "1")
    light._refuse_production()
    monkeypatch.delenv("HHEMT_E2E_ALLOW_PRODUCTION")
    monkeypatch.setenv("HHEMT_ZENODO_BASE_URL", "https://sandbox.zenodo.org")
    light._refuse_production()
