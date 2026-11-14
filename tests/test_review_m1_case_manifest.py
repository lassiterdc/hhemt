"""Reviewer's own checks for chunk 1k.3 (the Norfolk case manifest pin, the case templates, the bag rule).

None of these touches the network, compiles a solver or runs one. Run from the chunk worktree as
``PYTHONPATH=${PWD}/src:${PWD} pytest tests/test_review_m1_case_manifest.py``.

Four groups: the pinned case.yaml, the case templates against the manifest keys, the loader names,
and a two-arm differential over the bag rule.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import bagit
import pytest
import yaml

from hhemt import constants as cnst
from hhemt.exceptions import ProcessingError
from hhemt.experiments import NorfolkIreneExperiment, NorfolkObservedExperiment, TRITON_SWMM_experiment
from tests.test_review_1k2_resource_verify import _BIG, _MANIFEST, _RID, _SMALL, _fetch, _make_bag_zip

_REPO = Path(__file__).resolve().parent.parent
_CASE = _REPO / "test_data" / "norfolk_coastal_flooding"
_PLACEHOLDER = "PLACEHOLDER.csv"


def _manifest() -> dict[str, str]:
    return yaml.safe_load((_CASE / "case.yaml").read_text())["manifest"]


# --------------------------------------------------------------------------------------------
# Group 1: the pinned case.yaml
# --------------------------------------------------------------------------------------------


def test_case_yaml_manifest_is_the_23_entry_external_block_with_lowercase_sha256_digests():
    m = _manifest()
    assert len(m) == 23
    assert all(k.startswith("data/contents/external/") for k in m)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in m.values())


def test_case_yaml_resource_version_is_a_nonnull_timestamp():
    rv = yaml.safe_load((_CASE / "case.yaml").read_text())["resource_version"]
    assert isinstance(rv, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+Z", rv)


def test_pin_guard_exits_zero_on_the_repository_and_fires_on_an_emptied_manifest(tmp_path):
    ok = subprocess.run([sys.executable, str(_REPO / "scripts" / "check_case_manifest_pinned.py")], cwd=_REPO)
    assert ok.returncode == 0
    # control: the same guard over a tracked case.yaml with an empty manifest must exit 1
    root = tmp_path / "r"
    (root / "test_data" / "c").mkdir(parents=True)
    (root / "test_data" / "c" / "case.yaml").write_text("case_name: c\nmanifest: {}\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    bad = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "check_case_manifest_pinned.py"), "--root", str(root)],
        capture_output=True,
    )
    assert bad.returncode == 1


# --------------------------------------------------------------------------------------------
# Group 2: the case templates against the manifest keys
# --------------------------------------------------------------------------------------------


def _data_dir_values(template: str) -> dict[str, str]:
    cfg = yaml.safe_load((_CASE / template).read_text())
    return {k: v for k, v in cfg.items() if isinstance(v, str) and v.startswith("${DATA_DIR}/")}


def _key(v: str) -> str:
    return "data/contents/" + v.removeprefix("${DATA_DIR}/")


@pytest.mark.parametrize("template", ["template_system_config.yaml", "template_analysis_config.yaml"])
def test_every_data_dir_path_in_a_template_is_a_manifest_key_or_the_placeholder(template):
    m = _manifest()
    vals = _data_dir_values(template)
    assert vals, "must-be-nonzero control: the template names ${DATA_DIR} paths"
    unresolved = {k: v for k, v in vals.items() if _key(v) not in m and Path(v).name != _PLACEHOLDER}
    assert unresolved == {}


def test_the_resolution_check_is_live_an_old_layout_path_is_not_a_manifest_key():
    # varies the PROPERTY (path layout), not the token: the retired folder layout must not resolve
    assert _key("${DATA_DIR}/shp_watershed/norfolk_wshed_epsg32147_state_plane_m.shp") not in _manifest()
    assert _key("${DATA_DIR}/swmm/hague_full_model_template.inp") not in _manifest()


def test_the_placeholder_parent_is_external_so_the_loader_appends_into_the_manifest_directory():
    # _load_case_analysis_config sets weather_events_to_simulate = Path(placeholder).parent / name
    cfg = yaml.safe_load((_CASE / "template_analysis_config.yaml").read_text())
    parent = Path(cfg["weather_events_to_simulate"]).parent
    assert str(parent) == "${DATA_DIR}/external"
    # and with the pre-change parent (${DATA_DIR}) the loader names would NOT be manifest keys
    m = _manifest()
    for name in ("hurricane_irene_event_index.csv", "observed_event_events.csv"):
        assert _key(f"{parent}/{name}") in m
        assert _key(f"${{DATA_DIR}}/{name}") not in m


def test_windows_wiring_names_declared_config_fields_and_the_two_window_columns():
    from hhemt.config.analysis import analysis_config

    cfg = yaml.safe_load((_CASE / "template_analysis_config.yaml").read_text())
    for f in ("weather_event_windows_csv", "weather_event_start_column", "weather_event_end_column"):
        assert f in analysis_config.model_fields
    assert cfg["weather_event_start_column"] == "event_start"
    assert cfg["weather_event_end_column"] == "event_end"
    assert _key(cfg["weather_event_windows_csv"]) in _manifest()


def test_the_analysis_template_validates_once_every_data_dir_file_exists(tmp_path):
    from hhemt.config.analysis import analysis_config

    cfg = yaml.safe_load((_CASE / "template_analysis_config.yaml").read_text())

    def fill(base: Path, skip: str | None = None) -> dict:
        out = dict(cfg)
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith("${DATA_DIR}/"):
                p = base / v.removeprefix("${DATA_DIR}/")
                p.parent.mkdir(parents=True, exist_ok=True)
                if k != skip:
                    p.write_text("x")
                out[k] = str(p)
        out["hhemt_sha"] = "0" * 40
        return out

    analysis_config.model_validate(fill(tmp_path / "ok"))
    with pytest.raises(Exception, match="does not exist"):  # control: the validator is live on the new field
        analysis_config.model_validate(fill(tmp_path / "ctl", skip="weather_event_windows_csv"))


# --------------------------------------------------------------------------------------------
# Group 3: the loader names
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "expected"),
    [
        (NorfolkIreneExperiment, "hurricane_irene_event_index.csv"),
        (NorfolkObservedExperiment, "observed_event_events.csv"),
    ],
)
def test_each_norfolk_loader_asks_for_a_file_the_manifest_declares(loader, expected, monkeypatch):
    seen: dict = {}

    def fake(cls, **kw):
        seen.update(kw)
        return "sentinel"

    monkeypatch.setattr(TRITON_SWMM_experiment, "from_case_study", classmethod(fake))
    assert loader.load(hhemt_sha="0" * 40) == "sentinel"
    assert seen["weather_events_to_simulate"] == expected
    assert seen["case_name"] == cnst.NORFOLK_EX
    assert f"data/contents/external/{expected}" in _manifest()


# --------------------------------------------------------------------------------------------
# Group 4: two-arm differential over the bag rule
# --------------------------------------------------------------------------------------------


def test_pinned_arm_corrupted_declared_file_with_validate_true_is_case_manifest_verification(tmp_path):
    # arm A (violating): a declared file is corrupted AND another carries a multipart etag
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif", corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=_MANIFEST, validate=True)
    assert ei.value.operation == "case_manifest_verification"
    # arm B (differently-positioned satisfying input): the etag alone, nothing corrupted, returns
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    z2 = _make_bag_zip(ctl, multipart_etag_on="fullres_dem_m.tif")
    assert (
        _fetch(ctl, z2, manifest=_MANIFEST, validate=True) / "data" / "contents" / "external" / "table.csv"
    ).exists()


def test_pinned_arm_keeps_the_structural_checks_an_undeclared_payload_file_is_refused(tmp_path):
    z = _make_bag_zip(tmp_path, multipart_etag_on=None)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, zin.read(n))
        zout.writestr(f"{_RID}/data/contents/external/stowaway.csv", b"extra")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, bad, manifest=_MANIFEST, validate=True)
    assert ei.value.operation == "hydroshare_bag_validation"
    assert "structural" in ei.value.reason


def test_pinned_arm_keeps_the_structural_checks_a_missing_declared_file_is_refused(tmp_path):
    z = _make_bag_zip(tmp_path, multipart_etag_on=None)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            if not n.endswith("table.csv"):
                zout.writestr(n, zin.read(n))
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, bad, manifest=_MANIFEST, validate=True)
    assert ei.value.operation == "hydroshare_bag_validation"


def test_pinned_arm_does_not_content_check_a_payload_file_the_manifest_does_not_declare(tmp_path):
    """Characterization of the residual the rule accepts: skipping the md5 arm leaves an UNDECLARED file unchecked.

    The bag lists table.csv; the pinned manifest declares only the DEM. Corrupting table.csv used to be
    caught by bagit's md5 arm; under the pinned arm it passes. The manifest is the content pin, so the
    manifest must declare every file whose content matters (the Norfolk manifest declares 23 files).
    """
    z = _make_bag_zip(tmp_path, multipart_etag_on=None, corrupt="table.csv")
    partial = {"data/contents/external/fullres_dem_m.tif": _MANIFEST["data/contents/external/fullres_dem_m.tif"]}
    got = _fetch(tmp_path, z, manifest=partial, validate=True)
    assert (got / "data" / "contents" / "external" / "table.csv").exists()
    # control: the same bag, fetched with NO manifest (full bagit arm), IS refused for the corrupted file
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    z2 = _make_bag_zip(ctl, multipart_etag_on=None, corrupt="table.csv")
    with pytest.raises(ProcessingError):
        _fetch(ctl, z2, manifest=None, validate=True)


@pytest.mark.parametrize("manifest", [None, {}])
def test_absent_arm_refuses_an_etag_with_the_pin_a_manifest_reason_and_accepts_a_clean_bag(tmp_path, manifest):
    z = _make_bag_zip(tmp_path, multipart_etag_on="fullres_dem_m.tif")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=manifest, validate=True)
    assert ei.value.operation == "hydroshare_bag_validation"
    assert "no checkable md5" in ei.value.reason and "manifest" in ei.value.reason
    assert "fullres_dem_m.tif" in ei.value.reason
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    z2 = _make_bag_zip(ctl, multipart_etag_on=None)
    assert (_fetch(ctl, z2, manifest=manifest, validate=True) / "data" / "contents" / "external" / "table.csv").exists()


def test_absent_arm_still_full_validates_a_corrupted_clean_etag_free_bag_with_the_old_reason(tmp_path):
    z = _make_bag_zip(tmp_path, multipart_etag_on=None, corrupt="table.csv")
    with pytest.raises(ProcessingError) as ei:
        _fetch(tmp_path, z, manifest=None, validate=True)
    assert ei.value.operation == "hydroshare_bag_validation"
    assert "no checkable md5" not in ei.value.reason  # the etag refusal is not what fired


def test_etag_regex_property_a_plain_md5_never_matches_and_a_dash_digit_suffix_always_does():
    rx = TRITON_SWMM_experiment._MULTIPART_ETAG_RE
    assert rx.search(hashlib.md5(b"x").hexdigest()) is None  # noqa: S324
    assert rx.search(hashlib.md5(b"x").hexdigest() + "-2") is not None  # noqa: S324
    assert rx.search(hashlib.md5(b"x").hexdigest() + "-17") is not None  # noqa: S324
    assert rx.search("") is None
    assert bagit.Bag  # bagit is the real dependency under test
    assert _BIG and _SMALL
