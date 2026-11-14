"""Reviewer tests for chunk 1k.2: the HydroShare bag the toolkit fetches versus its sha256 case manifest.

The 1k.2 coder replaced the Norfolk HydroShare resource and found that the bag HydroShare serves
fails ``bagit.Bag.is_valid()`` on exactly the one file above 5 MiB, because HydroShare records a
multipart-upload etag (a ``-N`` suffix) where bagit expects a plain md5. ``_fetch_deposit_files``
validates the bag by default, so a default fetch raises ``hydroshare_bag_validation`` even though
the sha256 case manifest verifies every declared file.

These tests reproduce that on a synthetic bag, with a fake HydroShare resource object. None compiles
or runs a solver, none touches the network, and all run on a laptop. Run from the chunk worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_review_1k2_resource_verify.py``.

Two groups. The CHARACTERIZATION tests pin what the code does and pass. The one TARGET test pins the
behaviour the reviewer recommended (a pinned sha256 manifest is authoritative over the bag's md5 for
the files it declares). It was a strict xfail until chunk 1k.3 landed the toolkit change
(``_validate_hydroshare_bag``) and removed the marker, so it now runs as an ordinary test. The
characterization that pinned the DEFECT itself (default fetch raising ``hydroshare_bag_validation`` on
a bag a pinned manifest fully verifies) was removed in the same chunk: it asserted the inverse of the
target test, so it could not stay green once the defect was repaired.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import bagit
import pytest

from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment

_BIG = b"big-dem-bytes" * 500
_SMALL = b"small-csv\n" * 20
_RID = "abc123"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()  # noqa: S324 - bag fixture mirrors HydroShare's md5 manifest


def _make_bag_zip(tmp_path: Path, *, multipart_etag_on: str | None, corrupt: str | None = None) -> Path:
    """Build a HydroShare-shaped bag zip (one top-level dir, payload under data/contents/external).

    ``multipart_etag_on`` rewrites that payload file's manifest-md5 line to ``<md5>-2`` (the S3
    multipart etag shape) and refreshes tagmanifest-md5 so the ONLY inconsistency is the etag.
    ``corrupt`` flips one payload byte AFTER the manifests were written (a real integrity failure).
    """
    bag_dir = tmp_path / "bagsrc" / _RID
    ext = bag_dir / "data" / "contents" / "external"
    ext.mkdir(parents=True)
    payload = {"fullres_dem_m.tif": _BIG, "table.csv": _SMALL}
    for name, body in payload.items():
        (ext / name).write_bytes(body)
    # Hand-built so the payload sits at data/contents/... exactly as HydroShare serves it
    # (bagit.make_bag would move it under data/data/).
    (bag_dir / "bagit.txt").write_text("BagIt-Version: 0.97\nTag-File-Character-Encoding: UTF-8\n")
    mlines = []
    for name, body in payload.items():
        digest = _md5(body) + ("-2" if name == multipart_etag_on else "")
        mlines.append(f"{digest}  data/contents/external/{name}")
    mf = bag_dir / "manifest-md5.txt"
    mf.write_text("\n".join(mlines) + "\n")
    tm = bag_dir / "tagmanifest-md5.txt"
    tm.write_text(
        f"{_md5((bag_dir / 'bagit.txt').read_bytes())}  bagit.txt\n{_md5(mf.read_bytes())}  manifest-md5.txt\n"
    )
    if corrupt is not None:
        target = ext / corrupt
        data = bytearray(target.read_bytes())
        data[0] ^= 0xFF
        target.write_bytes(bytes(data))
    zip_path = tmp_path / "bag.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in sorted(bag_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(bag_dir.parent).as_posix())
    return zip_path


class _Res:
    def __init__(self, zip_path: Path):
        self._zip = zip_path

    def download(self, dest_dir):
        out = Path(dest_dir) / "served_bag.zip"
        out.write_bytes(self._zip.read_bytes())
        return str(out)


class _FakeHS:
    def __init__(self, zip_path: Path):
        self._zip = zip_path

    def resource(self, rid, validate=True):
        return _Res(self._zip)


_MANIFEST = {
    "data/contents/external/fullres_dem_m.tif": _sha(_BIG),
    "data/contents/external/table.csv": _sha(_SMALL),
}


def _fetch(tmp_path: Path, zip_path: Path, *, manifest, validate: bool = True):
    return TRITON_SWMM_experiment._fetch_deposit_files(
        "hydroshare",
        res_identifier=_RID,
        dest=tmp_path / "out" / "case",
        expected_manifest=manifest,
        hs=_FakeHS(zip_path),
        validate=validate,
    )


def test_fixture_etag_bag_fails_bagit_on_exactly_the_etag_file_and_the_manifest_passes(tmp_path):
    """Characterization: the synthetic bag reproduces the coder's measured defect shape.

    ``bagit`` rejects exactly the multipart-etag entry; the sha256 manifest verifies both files.
    The control arm (no etag rewrite) is a valid bag, so the failure varies the property
    under suspicion (an etag suffix), not the token.
    """
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif")
    root = tmp_path / "x"
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    bag = bagit.Bag(str(root / _RID))
    with pytest.raises(bagit.BagValidationError) as ei:
        bag.validate()
    failing = [d for d in ei.value.details if "md5 validation failed" in str(d)]
    assert len(failing) == 1 and "fullres_dem_m.tif" in str(failing[0])
    assert "-2" in str(failing[0])
    TRITON_SWMM_experiment._verify_manifest(root / _RID, _MANIFEST)  # sha256 arm passes

    ctl = tmp_path / "ctl"
    ctl.mkdir()
    z2 = _make_bag_zip(ctl, multipart_etag_on=None)
    root2 = ctl / "x"
    with zipfile.ZipFile(z2) as zf:
        zf.extractall(root2)
    assert bagit.Bag(str(root2 / _RID)).is_valid() is True  # must-be-nonzero control


def test_validate_false_with_pinned_manifest_returns_the_payload_root(tmp_path):
    """Characterization of option 3: validate=False returns, because _verify_manifest still runs for HydroShare."""
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif")
    got = _fetch(tmp_path, z, manifest=_MANIFEST, validate=False)
    assert (got / "data" / "contents" / "external" / "fullres_dem_m.tif").read_bytes() == _BIG


def test_validate_false_without_a_manifest_accepts_a_corrupted_payload(tmp_path):
    """Why option 3 alone is unsafe: with no manifest, validate=False leaves NO integrity check.

    The control arm (validate=True, same corrupted bag, plain md5) raises, so the bagit check is the
    only thing standing between a corrupted bundle deposit and the caller when no manifest is pinned
    (``_fetch_bundle_zip`` passes none).
    """
    z = _make_bag_zip(tmp_path, multipart_etag_on=None, corrupt="table.csv")
    got = _fetch(tmp_path, z, manifest=None, validate=False)
    assert (got / "data" / "contents" / "external" / "table.csv").exists()

    ctl = tmp_path / "ctl"
    ctl.mkdir()
    z2 = _make_bag_zip(ctl, multipart_etag_on=None, corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(ctl, z2, manifest=None, validate=True)
    assert ei.value.operation == "hydroshare_bag_validation"


def test_a_corrupted_file_against_a_pinned_manifest_is_refused_whatever_validate_says(tmp_path):
    """The sha256 arm is the real gate: a corrupted declared file raises case_manifest_verification."""
    z = _make_bag_zip(tmp_path, multipart_etag_on=None, corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=_MANIFEST, validate=False)
    assert ei.value.operation == "case_manifest_verification"


def test_absent_manifest_plus_multipart_etag_stays_refused_by_default(tmp_path):
    """Pin for the other half of the recommended behaviour: with no manifest, an unverifiable etag is refused."""
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif")
    with pytest.raises(ProcessingError):
        _fetch(tmp_path, z, manifest=None, validate=True)


def test_default_fetch_with_pinned_manifest_tolerates_a_multipart_etag(tmp_path):
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif")
    got = _fetch(tmp_path, z, manifest=_MANIFEST, validate=True)
    assert (got / "data" / "contents" / "external" / "fullres_dem_m.tif").read_bytes() == _BIG
