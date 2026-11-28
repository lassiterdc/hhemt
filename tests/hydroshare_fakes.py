"""Stand-ins for the parts of ``hsclient`` that the toolkit's HydroShare fetch touches.

``FakeBagHS`` serves a prepared BagIt zip through the session calls ``_download_hydroshare_bag`` makes, or answers
"Not ready" forever, as HydroShare does while it builds the archive. ``FakeFilesHS`` serves the files of a resource one
by one, as ``Resource.files`` and ``Resource.file_download`` do, and fails the test if the fetch asks for the bag.
"""

from __future__ import annotations

from pathlib import Path

_RESOURCE_ID = "0123456789abcdef0123456789abcdef"


class FakeResponse:
    """The slice of a ``requests.Response`` that the bag download reads."""

    def __init__(self, *, content: bytes = b"", content_type: str = "application/json", body: dict | None = None):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self._body = body if body is not None else {}

    def json(self) -> dict:
        return self._body


class FakeBagSession:
    """A ``HydroShareSession`` stand-in: ``get`` serves the bag zip, or "Not ready" when no zip is given."""

    def __init__(self, zip_path: Path | None = None, *, not_ready_body: dict | None = None):
        self._zip = zip_path
        self._not_ready_body = not_ready_body or {
            "bag_status": "Not ready",
            "task_id": "task-0001",
            "status": "Not ready",
        }
        self.get_calls: list[str] = []

    def get(self, path: str, status_code: int, **kwargs) -> FakeResponse:
        self.get_calls.append(path)
        if self._zip is None:
            return FakeResponse(body=self._not_ready_body)
        return FakeResponse(content=self._zip.read_bytes(), content_type="application/zip")

    def write_file(self, path: str, content: bytes, save_path: str = "") -> str:
        out = Path(save_path) / path.split("/")[-1]
        out.write_bytes(content)
        return str(out)


class FakeBagResource:
    def __init__(self, session: FakeBagSession, resource_id: str = _RESOURCE_ID):
        self._hs_session = session
        self._hsapi_path = f"hsapi/resource/{resource_id}"


class FakeBagHS:
    """A ``HydroShare`` stand-in whose every resource is served from one ``FakeBagSession``."""

    def __init__(self, zip_path: Path | None = None, *, not_ready_body: dict | None = None):
        self.session = FakeBagSession(zip_path, not_ready_body=not_ready_body)

    def resource(self, _key: str, validate: bool = True) -> FakeBagResource:
        return FakeBagResource(self.session, _key)


class FakeListedFile(str):
    """A file path string with the ``checksum`` that ``hsclient.File`` carries."""

    def __new__(cls, value: str, checksum: str):
        return super().__new__(cls, value)

    def __init__(self, value: str, checksum: str):
        self.checksum = checksum


class _FailingSession:
    def get(self, path: str, status_code: int, **kwargs):
        raise AssertionError(f"the fetch asked for the BagIt archive ({path}) when single files were enough")


class FakeFilesResource:
    """A resource that lists files with checksums and serves each through ``file_download``."""

    def __init__(self, files: dict[str, tuple[bytes, str]]):
        self._payloads = {name: payload for name, (payload, _checksum) in files.items()}
        self._listed = [FakeListedFile(name, checksum) for name, (_payload, checksum) in files.items()]
        self._hs_session = _FailingSession()
        self._hsapi_path = f"hsapi/resource/{_RESOURCE_ID}"
        self.downloaded: list[str] = []

    def files(self) -> list[FakeListedFile]:
        return self._listed

    def file_download(self, path: str, save_path: str = "") -> str:
        self.downloaded.append(path)
        out = Path(save_path) / Path(path).name
        out.write_bytes(self._payloads[path])
        return str(out)


class FakeFilesHS:
    def __init__(self, files: dict[str, tuple[bytes, str]]):
        self.resource_obj = FakeFilesResource(files)

    def resource(self, _key: str, validate: bool = True) -> FakeFilesResource:
        return self.resource_obj
