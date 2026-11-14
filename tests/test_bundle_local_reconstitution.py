"""Reconstitution from a LOCAL bundle zip, and the comparison of a re-run with the bundle's reference outputs.

``TRITON_SWMM_experiment.from_bundle_zip`` extracts a local zip and then takes the one post-extraction path
``from_doi`` takes (``_reconstitute_from_bundle_root``); ``compare_reference_outputs`` compares a reproducer's
``_test/`` summaries with the bundle's ``reference_outputs/``.

No solver is compiled or run. The reconstitution arms use a REAL ``emit_bundle`` over the bare analysis of
``tests/test_publish_deposit_paths.py`` and replace only the experiment constructor (which would build a system and
an analysis over the reconstituted configs) with a recorder, so what is compared is what the shared path wrote.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment
from tests.test_publish_deposit_paths import _world

# ---- reconstitution --------------------------------------------------------------------------------------------


def _write_crate_with_workflow(ad: Path) -> None:
    """A plain crate and a Snakefile: the emit upgrades the crate to a Workflow-Run-Crate whose ``mainEntity`` is the
    bundled ``Snakefile.source``, which is what a runnable bundle carries."""
    (ad / "Snakefile").write_text("rule all:\n    input: []\n")
    crate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "name": "T"},
        ],
    }
    (ad / "ro-crate-metadata.json").write_text(json.dumps(crate))


def _emitted_zip(tmp_path: Path) -> Path:
    from hhemt.bundle import _emit

    world = _world(tmp_path)
    _write_crate_with_workflow(world.ad)
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    return _emit.emit_bundle(world.analysis, tmp_path / "out" / "bundle.zip")


@pytest.fixture
def recorded_constructor(monkeypatch):
    """Replace the experiment constructor with a recorder of the config paths the shared path hands it."""
    calls: list[dict] = []

    def _init(self, cfg_system_yaml, cfg_analysis_yaml, case_name=None, hpc_system_config_yaml=None):
        self.cfg_system_yaml = Path(cfg_system_yaml)
        self.cfg_analysis_yaml = Path(cfg_analysis_yaml)
        calls.append(
            {"system": self.cfg_system_yaml, "analysis": self.cfg_analysis_yaml, "hpc": hpc_system_config_yaml}
        )

    monkeypatch.setattr(TRITON_SWMM_experiment, "__init__", _init)
    return calls


def _stub_fetch(monkeypatch, zip_path: Path) -> None:
    monkeypatch.setattr(
        TRITON_SWMM_experiment, "_fetch_bundle_zip", classmethod(lambda cls, *args, **kwargs: Path(zip_path))
    )


def _config_bytes(exp, bundle_root_of: Path) -> dict[str, dict]:
    """The two reconstituted configs, parsed, with the extraction root replaced so two roots compare equal."""
    out = {}
    for name, path in (("system", exp.cfg_system_yaml), ("analysis", exp.cfg_analysis_yaml)):
        text = Path(path).read_text().replace(str(bundle_root_of), "ROOTDIR")
        out[name] = yaml.safe_load(text)
    return out


def test_from_bundle_zip_reconstitutes_the_same_configs_as_from_doi_for_the_same_zip(
    tmp_path, monkeypatch, recorded_constructor
):
    zip_path = _emitted_zip(tmp_path)
    _stub_fetch(monkeypatch, zip_path)
    via_doi = TRITON_SWMM_experiment.from_doi(
        doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "doi", validate=False
    )
    doi_root = via_doi.bundle_root
    via_zip = TRITON_SWMM_experiment.from_bundle_zip(zip_path, target_dir=tmp_path / "local", validate=False)
    zip_root = via_zip.bundle_root
    assert doi_root != zip_root, "control: the two extractions are different directories"
    assert zip_root == tmp_path / "local" / "bundle"
    configs_doi = _config_bytes(via_doi, doi_root)
    configs_zip = _config_bytes(via_zip, zip_root)
    assert configs_zip == configs_doi
    assert configs_zip["system"], "control: the configs are not empty"
    assert len(recorded_constructor) == 2


def test_from_bundle_zip_runs_the_shared_gates_a_from_doi_runs(tmp_path, recorded_constructor):
    """A bundle whose crate names no workflow is refused by the local entry point exactly as by ``from_doi``."""
    zip_path = _emitted_zip(tmp_path)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(broken, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "ro-crate-metadata.json":
                crate = json.loads(data)
                for entity in crate["@graph"]:
                    entity.pop("mainEntity", None)
                data = json.dumps(crate).encode()
            dst.writestr(info.filename, data)
    with pytest.raises(ProcessingError):
        TRITON_SWMM_experiment.from_bundle_zip(broken, target_dir=tmp_path / "t", validate=False)
    assert recorded_constructor == [], "control: no experiment was built from the refused bundle"


def test_from_bundle_zip_refuses_a_missing_zip_and_leaves_the_zip_alone(tmp_path):
    with pytest.raises(FileNotFoundError):
        TRITON_SWMM_experiment.from_bundle_zip(tmp_path / "nope.zip", validate=False)


def test_from_bundle_zip_does_not_modify_the_zip(tmp_path, recorded_constructor):
    import hashlib

    zip_path = _emitted_zip(tmp_path)
    before = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    TRITON_SWMM_experiment.from_bundle_zip(zip_path, target_dir=tmp_path / "t", validate=False)
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("validate", [True, False])
def test_validate_reaches_the_shared_path_from_both_entry_points(tmp_path, monkeypatch, validate):
    """``validate=True`` runs the experiment's validation once, from either entry point; False runs none."""
    from types import SimpleNamespace

    zip_path = _emitted_zip(tmp_path)
    _stub_fetch(monkeypatch, zip_path)
    ran: list[str] = []

    def _init(self, *args, **kwargs):
        self.bundle_root = None
        self.analysis = SimpleNamespace(
            validate=lambda: SimpleNamespace(raise_if_invalid=lambda: ran.append("validated"))
        )

    monkeypatch.setattr(TRITON_SWMM_experiment, "__init__", _init)
    TRITON_SWMM_experiment.from_bundle_zip(zip_path, target_dir=tmp_path / "l", validate=validate)
    TRITON_SWMM_experiment.from_doi(doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "d", validate=validate)
    assert ran == (["validated", "validated"] if validate else [])


def test_both_entry_points_call_the_one_shared_function(tmp_path, monkeypatch):
    zip_path = _emitted_zip(tmp_path)
    _stub_fetch(monkeypatch, zip_path)
    seen: list[Path] = []

    def _shared(cls, bundle_root, **kwargs):
        seen.append(Path(bundle_root))
        return "experiment"

    monkeypatch.setattr(TRITON_SWMM_experiment, "_reconstitute_from_bundle_root", classmethod(_shared))
    assert TRITON_SWMM_experiment.from_doi(doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "d") == (
        "experiment"
    )
    assert TRITON_SWMM_experiment.from_bundle_zip(zip_path, target_dir=tmp_path / "l") == "experiment"
    assert len(seen) == 2 and all((root / "bundle_manifest.json").is_file() for root in seen)


# ---- compare_reference_outputs --------------------------------------------------------------------------------

EVENT = "event_id.0"
GROUP = "group_0"
STORE = "TRITONSWMM_TRITON_summary.zarr"


def _summary(max_depth: float = 1.5, nan_cell: bool = True) -> xr.Dataset:
    depth = np.array([[0.1, 0.2], [0.3, max_depth]])
    if nan_cell:
        depth[0, 0] = np.nan
    return xr.Dataset(
        {
            "max_flood_depth": (("y", "x"), depth),
            "time_of_max_flood_depth": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]])),
            "label": (("y",), np.array(["a", "b"])),
        },
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )


def _put_store(root: Path, ds: xr.Dataset, *, reference: bool, group: str = GROUP, name: str = STORE) -> Path:
    base = root / ("reference_outputs" if reference else "_test") / group / "sims" / EVENT / "processed"
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    if name.endswith(".zarr"):
        ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)
    else:
        ds.to_netcdf(path, engine="h5netcdf")
    return path


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    bundle_root, analysis_dir = tmp_path / "bundle", tmp_path / "analysis"
    bundle_root.mkdir(parents=True)
    analysis_dir.mkdir()
    return bundle_root, analysis_dir


def test_identical_stores_pass(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(), reference=True)
    _put_store(analysis_dir, _summary(), reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert result.passed
    assert [(v.store, v.status) for v in result.verdicts] == [(f"{GROUP}/sims/{EVENT}/processed/{STORE}", "equal")]
    assert result.extra_stores == ()
    result.raise_if_failed()
    # control for NaN-awareness: the compared variable holds a NaN, so equal_nan was exercised
    assert np.isnan(_summary()["max_flood_depth"].values).any()


def test_one_differing_value_fails_naming_the_store_and_the_variable(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(1.5), reference=True)
    _put_store(analysis_dir, _summary(1.5000001), reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed
    (verdict,) = result.verdicts
    assert verdict.status == "differs"
    assert list(verdict.variables) == ["max_flood_depth"], "only the changed variable is named"
    text = result.summary()
    assert f"{GROUP}/sims/{EVENT}/processed/{STORE}" in text and "max_flood_depth" in text
    with pytest.raises(ProcessingError) as caught:
        result.raise_if_failed()
    assert "max_flood_depth" in str(caught.value)


def test_a_tolerance_is_an_explicit_parameter_and_exact_is_the_default(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(1.5), reference=True)
    _put_store(analysis_dir, _summary(1.5000001), reference=False)
    assert not compare_reference_outputs(bundle_root, analysis_dir).passed
    assert compare_reference_outputs(bundle_root, analysis_dir, rtol=1e-5).passed
    assert not compare_reference_outputs(bundle_root, analysis_dir, rtol=1e-9).passed


def test_a_nan_where_the_reference_has_a_value_fails(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(nan_cell=False), reference=True)
    _put_store(analysis_dir, _summary(nan_cell=True), reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed and "max_flood_depth" in result.summary()


def test_a_store_the_bundle_carries_and_the_rerun_lacks_fails(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(), reference=True)
    (analysis_dir / "_test").mkdir()
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed
    assert result.verdicts[0].status == "missing_in_rerun"
    assert STORE in result.summary()


def test_a_store_only_the_rerun_has_is_reported_and_does_not_fail(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(), reference=True)
    _put_store(analysis_dir, _summary(), reference=False)
    _put_store(analysis_dir, _summary(), reference=False, name="SWMM_only_node_summary.zarr")
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert result.passed
    assert result.extra_stores == (f"{GROUP}/sims/{EVENT}/processed/SWMM_only_node_summary.zarr",)


def test_a_variable_absent_from_the_rerun_fails_and_an_extra_one_is_reported(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(), reference=True)
    _put_store(analysis_dir, _summary().drop_vars("time_of_max_flood_depth"), reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed
    assert result.verdicts[0].variables == {"time_of_max_flood_depth": "is absent from the re-run"}
    bundle_root2, analysis_dir2 = _roots(tmp_path / "second")
    _put_store(bundle_root2, _summary().drop_vars("time_of_max_flood_depth"), reference=True)
    _put_store(analysis_dir2, _summary(), reference=False)
    again = compare_reference_outputs(bundle_root2, analysis_dir2)
    assert again.passed and again.verdicts[0].extra_variables == ("time_of_max_flood_depth",)


def test_a_text_variable_that_differs_fails_and_a_tolerance_does_not_loosen_it(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    changed = _summary()
    changed["label"] = ("y", np.array(["a", "z"]))
    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, _summary(), reference=True)
    _put_store(analysis_dir, changed, reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir, rtol=1.0, atol=1.0)
    assert not result.passed and list(result.verdicts[0].variables) == ["label"]


def test_a_netcdf_store_is_compared(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    name = "TRITONSWMM_TRITON_summary.nc"
    _put_store(bundle_root, _summary(1.5), reference=True, name=name)
    _put_store(analysis_dir, _summary(2.5), reference=False, name=name)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed and list(result.verdicts[0].variables) == ["max_flood_depth"]
    _put_store(analysis_dir, _summary(1.5), reference=False, name=name)
    assert compare_reference_outputs(bundle_root, analysis_dir).passed


def test_a_bundle_without_reference_outputs_cannot_support_a_comparison(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(analysis_dir, _summary(), reference=False)
    with pytest.raises(ProcessingError):
        compare_reference_outputs(bundle_root, analysis_dir)


def test_the_groups_map_by_their_shared_basename(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    for group, depth in (("group_0", 1.5), ("group_1", 2.5)):
        _put_store(bundle_root, _summary(depth), reference=True, group=group)
        _put_store(analysis_dir, _summary(depth if group == "group_0" else 9.0), reference=False, group=group)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    status = {v.store.split("/")[0]: v.status for v in result.verdicts}
    assert status == {"group_0": "equal", "group_1": "differs"}


def test_each_group_is_compared_with_its_own_group_when_the_groups_differ_from_each_other(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bundle_root, analysis_dir = _roots(tmp_path)
    for group, depth in (("group_0", 1.5), ("group_1", 2.5)):
        _put_store(bundle_root, _summary(depth), reference=True, group=group)
        _put_store(analysis_dir, _summary(depth), reference=False, group=group)
    assert compare_reference_outputs(bundle_root, analysis_dir).passed


def test_a_variable_whose_shape_differs_fails_and_says_so(tmp_path):
    from hhemt.bundle._reprex import compare_reference_outputs

    bigger = xr.Dataset(
        {"max_flood_depth": (("y", "x"), np.zeros((3, 2)))}, coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0]}
    )
    smaller = xr.Dataset({"max_flood_depth": (("y", "x"), np.zeros((2, 2)))}, coords={"y": [0.0, 1.0], "x": [0.0, 1.0]})
    bundle_root, analysis_dir = _roots(tmp_path)
    _put_store(bundle_root, smaller, reference=True)
    _put_store(analysis_dir, bigger, reference=False)
    result = compare_reference_outputs(bundle_root, analysis_dir)
    assert not result.passed
    assert "shape" in result.verdicts[0].variables["max_flood_depth"]


# ---- the destination an extraction may replace ------------------------------------------------------------------


def _tiny_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("bundle_manifest.json", "{}")
    return path


def test_extraction_replaces_an_old_extraction_but_refuses_a_destination_that_holds_the_cwd_or_is_home(
    tmp_path, monkeypatch
):
    from hhemt.bundle._reprex import extract_reprex_bundle

    zip_path = _tiny_zip(tmp_path / "zips" / "b.zip")
    old = tmp_path / "old"
    old.mkdir()
    (old / "stale.txt").write_text("from a previous extraction")
    out = extract_reprex_bundle(zip_path, dest=old)
    assert out == old and not (old / "stale.txt").exists() and (old / "bundle_manifest.json").is_file(), (
        "control: an ordinary existing destination is still replaced"
    )

    cwd_dir = tmp_path / "work"
    (cwd_dir / "inside").mkdir(parents=True)
    (cwd_dir / "keep.txt").write_text("k")
    monkeypatch.chdir(cwd_dir / "inside")
    with pytest.raises(ProcessingError):
        extract_reprex_bundle(zip_path, dest=cwd_dir)
    assert (cwd_dir / "keep.txt").exists()

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home_dir"
    home.mkdir()
    (home / "keep.txt").write_text("k")
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(ProcessingError):
        extract_reprex_bundle(zip_path, dest=home)
    assert (home / "keep.txt").exists()


def test_a_filesystem_root_is_refused_as_holding_every_zip(tmp_path):
    from hhemt.bundle._reprex import _refuse_to_replace_unsafe_destination

    with pytest.raises(ProcessingError):
        _refuse_to_replace_unsafe_destination(Path("/"), tmp_path / "x.zip")
