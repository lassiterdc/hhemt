"""Coder tests for chunk 1k.3: the HydroShare bag-validation rule of ``_fetch_deposit_files``,
and the agreement of the Norfolk templates and loaders with the pinned ``case.yaml`` manifest.

Shape item: package 1k (K2), chunk 1k.3, bag-validation ruling recorded in the run hub entry
"1k.2 result, the HydroShare bag-validation design call, and its ruling". The rule under test
(``TRITON_SWMM_experiment._validate_hydroshare_bag``):

* manifest PINNED (non-empty ``expected_manifest``): bagit's structural arm only, then the
  sha256 ``_verify_manifest``. A multipart etag in manifest-md5 no longer refuses the fetch.
* manifest ABSENT (``None`` or ``{}``): full bagit validation, and a manifest-md5 entry that
  matches ``-\\d+$`` is refused as unverifiable with a reason that tells the caller to pin a
  case.yaml manifest.

The last two tests (group 2) pin that the Norfolk templates and the two loader-built CSV names name
only files the pinned ``case.yaml`` manifest declares, at the ``external/`` layout the dataset has.

No test compiles or runs a solver or touches the network; each builds a synthetic
HydroShare-shaped bag zip and serves it through a fake HydroShare client. Run from the chunk
tree as ``PYTHONPATH=${PWD}/src pytest tests/test_experiments_hydroshare_bag_validation.py``.

CLASS and DISCRIMINATION of each test are in its own docstring. The helpers are copied in
shape from ``tests/test_review_1k2_resource_verify.py`` (the reviewer's file, not imported
from, so this file stays independent of it).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from hhemt.exceptions import ProcessingError
from hhemt.experiments import NorfolkIreneExperiment, NorfolkObservedExperiment, TRITON_SWMM_experiment

_BIG = b"big-dem-bytes" * 500
_SMALL = b"small-csv\n" * 20
_RID = "feedbeef"
_REL = "data/contents/external"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()  # noqa: S324 - the bag fixture mirrors HydroShare's md5 manifest


def _make_bag_zip(
    tmp_path: Path,
    *,
    etag_on: str | None = None,
    corrupt: str | None = None,
    drop_payload: str | None = None,
) -> Path:
    """HydroShare-shaped bag zip. ``etag_on`` rewrites one manifest-md5 line to ``<md5>-2``;
    ``corrupt`` flips a payload byte after the manifests were written; ``drop_payload`` deletes a
    payload file the manifest still lists (a structurally incomplete bag)."""
    bag_dir = tmp_path / "bagsrc" / _RID
    ext = bag_dir / _REL
    ext.mkdir(parents=True)
    payload = {"fullres_dem_m.tif": _BIG, "table.csv": _SMALL}
    for name, body in payload.items():
        (ext / name).write_bytes(body)
    (bag_dir / "bagit.txt").write_text("BagIt-Version: 0.97\nTag-File-Character-Encoding: UTF-8\n")
    lines = [f"{_md5(body)}{'-2' if name == etag_on else ''}  {_REL}/{name}" for name, body in payload.items()]
    (bag_dir / "manifest-md5.txt").write_text("\n".join(lines) + "\n")
    if corrupt is not None:
        data = bytearray((ext / corrupt).read_bytes())
        data[0] ^= 0xFF
        (ext / corrupt).write_bytes(bytes(data))
    if drop_payload is not None:
        (ext / drop_payload).unlink()
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


_MANIFEST = {f"{_REL}/fullres_dem_m.tif": _sha(_BIG), f"{_REL}/table.csv": _sha(_SMALL)}


def _fetch(tmp_path: Path, zip_path: Path, *, manifest):
    return TRITON_SWMM_experiment._fetch_deposit_files(
        "hydroshare",
        res_identifier=_RID,
        dest=tmp_path / "out" / "case",
        expected_manifest=manifest,
        hs=_FakeHS(zip_path),
        validate=True,
    )


@pytest.mark.parametrize("absent", [None, {}], ids=["none", "empty-dict"])
def test_manifest_absent_with_a_multipart_etag_is_refused_and_says_to_pin_a_manifest(tmp_path, absent):
    """NEW CAPABILITY. Plausible wrong implementation: skip the md5 arm whenever any ``-N`` entry
    is seen, whatever the manifest state (the blanket ``validate=False`` shape). Under it this
    test is red because the fetch returns instead of raising.

    Input class: a HydroShare bag whose manifest-md5 carries a ``-\\d+$`` entry and whose
    ``expected_manifest`` is absent, spelled both ways (``None`` and ``{}``), since both spell
    "no pin" to ``_verify_manifest``. The reason must tell the caller to pin a case.yaml manifest
    and name the unverifiable file, so the operator can act on the message.

    Why required: with no manifest the md5 arm is the only integrity check, so tolerating the
    etag here would turn refusal of a corrupted deposit into acceptance.
    A second correct implementation: raise from a pre-pass over ``bag.entries`` or from a custom
    bagit subclass; both satisfy the same assertions.
    """
    z = _make_bag_zip(tmp_path, etag_on="fullres_dem_m.tif")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=absent)
    assert ei.value.operation == "hydroshare_bag_validation"
    reason = str(ei.value.reason)
    assert "no checkable md5" in reason
    assert "case.yaml" in reason and "manifest" in reason
    assert "fullres_dem_m.tif" in reason


def test_manifest_absent_without_an_etag_and_a_plain_md5_bag_still_returns(tmp_path):
    """Both-states control for the refusal above: it varies the PROPERTY (an etag suffix), not the
    token. The same bag with plain md5 entries and no manifest must still fetch, so the refusal
    is keyed on the etag, not on 'no manifest' alone."""
    z = _make_bag_zip(tmp_path)
    got = _fetch(tmp_path, z, manifest=None)
    assert (got / _REL / "fullres_dem_m.tif").read_bytes() == _BIG


def test_manifest_absent_and_a_plain_md5_corruption_is_still_refused_by_full_validation(tmp_path):
    """REGRESSION arm (the pre-change tree also raised here). Full bagit validation is kept when
    the manifest is absent, so a corrupted payload with a plain md5 is refused with the bagit
    reason, not the pin-a-manifest reason (it is a corruption, not an unverifiable file)."""
    z = _make_bag_zip(tmp_path, corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=None)
    assert ei.value.operation == "hydroshare_bag_validation"
    assert "no checkable md5" not in str(ei.value.reason)


def test_pinned_manifest_with_a_corrupted_declared_file_still_raises_case_manifest_verification(tmp_path):
    """NEW CAPABILITY. Plausible wrong implementation: with a pinned manifest, skip bagit entirely
    AND skip ``_verify_manifest`` too (treating the pin as sufficient on its face). Under it the
    corrupted file would be accepted and this test red.

    Input class: a pinned manifest and a bag whose declared file ``table.csv`` is corrupted. The
    sha256 arm is the real gate and must raise ``case_manifest_verification``. The md5 arm is
    skipped on the pinned path, so the operation names the arm that did the refusing rather than
    ``hydroshare_bag_validation``.
    A second correct implementation: run the sha256 check before bagit and keep the structural
    arm; the operation is still ``case_manifest_verification``.
    """
    z = _make_bag_zip(tmp_path, corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=_MANIFEST)
    assert ei.value.operation == "case_manifest_verification"


def test_pinned_manifest_with_an_etag_and_an_intact_payload_returns(tmp_path):
    """Both-states anchor for the pinned path: the multipart etag alone no longer refuses, and the
    payload root is returned with the DEM bytes intact. Differs from the reviewer's target test in
    carrying a corrupted-then-fixed control: the same bag with the declared file corrupted raises
    (previous test), so a return here is not vacuous."""
    z = _make_bag_zip(tmp_path, etag_on="fullres_dem_m.tif")
    got = _fetch(tmp_path, z, manifest=_MANIFEST)
    assert (got / _REL / "fullres_dem_m.tif").read_bytes() == _BIG


def test_pinned_manifest_still_refuses_a_structurally_incomplete_bag(tmp_path):
    """NEW CAPABILITY. Plausible wrong implementation: pinned path skips bagit completely (the
    ``validate=False`` shape for HydroShare). Under it a bag whose manifest-md5 lists a payload
    file that is absent from the zip would still be refused only by ``_verify_manifest`` when the
    case manifest happens to declare that file. Here the case manifest declares ONLY ``table.csv``,
    so the missing ``fullres_dem_m.tif`` is invisible to the sha256 arm and only bagit's
    structural arm can refuse the bag.

    Input class: a pinned manifest that does not declare a file the bag's own manifest-md5 lists
    and the zip lacks. Required because the ruling keeps bagit's completeness arm on the pinned
    path ('bag structure only')."""
    z = _make_bag_zip(tmp_path, drop_payload="fullres_dem_m.tif")
    only_table = {f"{_REL}/table.csv": _sha(_SMALL)}
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=only_table)
    assert ei.value.operation == "hydroshare_bag_validation"
    assert "structural" in str(ei.value.reason)


# ---------------------------------------------------------------------------------------------
# Group 2: templates and loaders versus the pinned case manifest
# ---------------------------------------------------------------------------------------------

_CASE_DIR = Path(__file__).resolve().parents[1] / "test_data" / "norfolk_coastal_flooding"
_PLACEHOLDER = "PLACEHOLDER.csv"


def _manifest_keys() -> set[str]:
    return set(yaml.safe_load((_CASE_DIR / "case.yaml").read_text())["manifest"])


def test_every_template_data_dir_path_resolves_to_a_declared_manifest_key():
    """REGRESSION. Red on the pre-change tree (``shp_watershed/``, ``swmm/`` and flat names).

    Input class: every uncommented ``${DATA_DIR}/...`` value of the two Norfolk templates. Each
    expands to ``data/contents/{rest}`` under the extracted bag root, and must equal a key of
    the pinned manifest EXACTLY (folder included), which the basename-only pin of the 1k.1
    reviewer cannot see. The placeholder's folder is checked through the loader test below.
    Both-states anchor: the property holds for any dataset layout the manifest declares.
    A second correct implementation: any layout, so long as templates and manifest agree.
    """
    keys = _manifest_keys()
    checked = 0
    for name in ("template_system_config.yaml", "template_analysis_config.yaml"):
        cfg = yaml.safe_load((_CASE_DIR / name).read_text())
        for field, value in cfg.items():
            if not (isinstance(value, str) and value.startswith("${DATA_DIR}/")):
                continue
            rel = value[len("${DATA_DIR}/") :]
            if Path(rel).name == _PLACEHOLDER:
                continue
            checked += 1
            assert f"data/contents/{rel}" in keys, f"{name}:{field} -> data/contents/{rel} is not a manifest key"
    assert checked >= 11, f"matcher read {checked} paths; the two templates carry 11 dataset paths"


@pytest.mark.parametrize("loader", [NorfolkIreneExperiment, NorfolkObservedExperiment], ids=["irene", "observed"])
def test_each_loader_built_event_csv_resolves_beside_the_placeholder_to_a_manifest_key(loader, monkeypatch):
    """REGRESSION. Red on the pre-change tree (``obs_event_summaries_from_yrs_with_complete_coverage.csv``
    is in no manifest) and on a placeholder left at the dataset root.

    The loader hands ``from_case_study`` a bare CSV name; ``_load_case_analysis_config`` then
    resolves it beside the template's placeholder path. The resolved file, as ``data/contents/...``,
    must be a manifest key, so the CSV a loader asks for is a file the fetch delivers.
    """
    seen: dict = {}

    def _capture(cls, **kwargs):
        seen.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(TRITON_SWMM_experiment, "from_case_study", classmethod(_capture))
    loader.load(hhemt_sha="0" * 40)
    cfg = yaml.safe_load((_CASE_DIR / "template_analysis_config.yaml").read_text())
    placeholder_rel = cfg["weather_events_to_simulate"][len("${DATA_DIR}/") :]
    assert Path(placeholder_rel).name == _PLACEHOLDER
    resolved = (Path(placeholder_rel).parent / seen["weather_events_to_simulate"]).as_posix()
    assert f"data/contents/{resolved}" in _manifest_keys()
