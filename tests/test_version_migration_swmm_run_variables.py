"""V0030 (layout 29 to 30): the SWMM run-level attributes become scalar data variables.

The committed ``v0030_unit_test`` fixture is a layout-29 tree whose per-scenario node and link zarr stores hold
``flow_continuity_error``, ``runoff_continuity_error`` and ``system_flooding`` as dataset attributes (with their
``_units`` companions), exactly as the layout-29 parser wrote them. Further stores (a NetCDF file, an unindexed
store, a store that already holds a variable, a format 2 store) are built in ``tmp_path`` because ``*.nc`` is
gitignored and the other shapes are cheap to build.

No solver is compiled or run. The migration is driven through the runner so the test reads what a user's tree would
become.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"
NAMES = ("flow_continuity_error", "runoff_continuity_error", "system_flooding")
STEMS = (
    "TRITONSWMM_SWMM_node_tseries",
    "TRITONSWMM_SWMM_link_tseries",
)


def _migrate(work: Path, *, target: int = 30):
    return runner.run_migration(
        work,
        target=target,
        apply=True,
        cfg_paths={
            "system": work / "cached_configs" / "system.yaml",
            "analysis": work / "cached_configs" / "analysis.yaml",
        },
    )


def _tree_digest(root: Path) -> dict[str, str]:
    """Unit: files. Path -> sha256 of every file under ``root``, ``_version.json`` and filelock sidecars excluded."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "_version.json" and not p.name.endswith(".lock")
    }


def _legacy_attrs() -> dict:
    return {
        "analysis_end_datetime": "2020-01-01 00:06:00",
        "flow_units": "cms",
        "runoff_continuity_error": float("nan"),
        "runoff_continuity_error_units": "percent",
        "flow_continuity_error": -0.37,
        "flow_continuity_error_units": "percent",
        "system_flooding": 12.5,
        "system_flooding_units": "10^6 L",
    }


def _node_dataset(attrs: dict | None = None) -> xr.Dataset:
    t = np.array(["2020-01-01T00:02:00", "2020-01-01T00:04:00"], dtype="datetime64[ns]")
    return xr.Dataset(
        {"flooding": (("date_time", "node_id"), np.array([[0.0, 1.0], [0.5, 2.0]]), {"units": "m3 s-1"})},
        coords={"date_time": t, "node_id": ["n1", "n2"]},
        attrs=_legacy_attrs() if attrs is None else attrs,
    )


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    work = tmp_path / "v29"
    shutil.copytree(FIXTURE_ROOT / "v0030_unit_test", work)
    # Stamped, so the runner plans V0030 alone: an unstamped fixture is inference-dated and would also run the
    # earlier migrations over the stores this test builds.
    runner.baseline(work, 29)
    return work


def _processed(work: Path) -> Path:
    return work / "sims" / "event_id.0" / "processed"


def test_fixture_stores_hold_the_three_figures_as_attributes_before_the_migration(fixture_tree: Path):
    """Control for every test below: the committed fixture really is the pre-conversion shape."""
    for stem in STEMS:
        group = zarr.open_group(str(_processed(fixture_tree) / f"{stem}.zarr"), mode="r")
        assert all(name in group.attrs for name in NAMES), stem
        assert not any(name in set(group.array_keys()) for name in NAMES), stem


def test_the_migration_writes_scalar_variables_with_equal_values_and_removes_the_attributes(fixture_tree: Path):
    before = {}
    for stem in STEMS:
        before[stem] = dict(zarr.open_group(str(_processed(fixture_tree) / f"{stem}.zarr"), mode="r").attrs)
    result = _migrate(fixture_tree)
    assert result.migrations_applied == ["V0030__swmm_run_attributes_to_variables"]
    for stem in STEMS:
        path = _processed(fixture_tree) / f"{stem}.zarr"
        group = zarr.open_group(str(path), mode="r")
        for name in NAMES:
            assert name not in group.attrs, (stem, name)
            assert f"{name}_units" not in group.attrs, (stem, name)
        # opens through a consolidated read: the index was rebuilt
        ds = xr.open_zarr(path, consolidated=True, decode_timedelta=False)
        for name in NAMES:
            assert name in ds.data_vars and ds[name].dims == (), (stem, name)
            assert ds[name].attrs["units"] == before[stem][f"{name}_units"], (stem, name)
            assert np.array_equal(float(ds[name].values), before[stem][name], equal_nan=True), (stem, name)
        # the attributes the migration does not own are untouched, and the time series is untouched
        assert ds.attrs == {k: v for k, v in before[stem].items() if k in ("analysis_end_datetime", "flow_units")}
        assert "date_time" in ds.dims


def test_a_converted_zarr_store_keeps_its_modification_time(fixture_tree: Path):
    import os

    paths = [_processed(fixture_tree) / f"{stem}.zarr" for stem in STEMS]
    for path in paths:
        os.utime(path, (1_600_000_000, 1_600_000_000))
    _migrate(fixture_tree)
    assert [p.stat().st_mtime for p in paths] == [1_600_000_000.0, 1_600_000_000.0]
    # control: the stores were converted, so a changed mtime would have been a real rewrite's side effect
    assert NAMES[0] in set(zarr.open_group(str(paths[0]), mode="r").array_keys())


def test_a_second_run_changes_no_byte(fixture_tree: Path):
    _migrate(fixture_tree)
    first = _tree_digest(fixture_tree)
    again = _migrate(fixture_tree)
    assert again.migrations_applied == []
    assert _tree_digest(fixture_tree) == first
    # control: the digest instrument sees a change when one is made
    node = _processed(fixture_tree) / f"{STEMS[0]}.zarr"
    zarr.open_group(str(node), mode="r+").attrs["flow_units"] = "cfs"
    assert _tree_digest(fixture_tree) != first


def test_a_store_without_the_attributes_is_untouched(fixture_tree: Path):
    clean = _processed(fixture_tree) / "SWMM_only_link_tseries.zarr"
    ds = xr.Dataset(
        {"flow": (("date_time",), np.array([1.0, 2.0]))},
        coords={"date_time": np.array(["2020-01-01T00:02:00", "2020-01-01T00:04:00"], dtype="datetime64[ns]")},
        attrs={"flow_units": "cms"},
    )
    ds.to_zarr(clean, mode="w", consolidated=True)
    clean_before = _tree_digest(clean)
    mtime = clean.stat().st_mtime_ns
    _migrate(fixture_tree)
    assert _tree_digest(clean) == clean_before
    assert clean.stat().st_mtime_ns == mtime
    # control: the same migration did convert a store that carries the attributes
    assert NAMES[0] in set(zarr.open_group(str(_processed(fixture_tree) / f"{STEMS[0]}.zarr"), mode="r").array_keys())


def test_a_cleared_store_is_skipped(fixture_tree: Path):
    shutil.rmtree(_processed(fixture_tree) / f"{STEMS[1]}.zarr")
    result = _migrate(fixture_tree)
    assert result.migrations_applied[-1] == "V0030__swmm_run_attributes_to_variables"
    assert not (_processed(fixture_tree) / f"{STEMS[1]}.zarr").exists()
    assert NAMES[2] in set(zarr.open_group(str(_processed(fixture_tree) / f"{STEMS[0]}.zarr"), mode="r").array_keys())


def test_a_store_holding_both_keeps_the_variable_and_loses_the_attribute(fixture_tree: Path):
    path = _processed(fixture_tree) / "SWMM_only_node_tseries.zarr"
    ds = _node_dataset()
    ds["system_flooding"] = xr.DataArray(np.float64(99.0), attrs={"units": "10^6 L"})
    ds.to_zarr(path, mode="w", consolidated=True)
    _migrate(fixture_tree)
    out = xr.open_zarr(path, consolidated=True, decode_timedelta=False)
    assert float(out["system_flooding"]) == 99.0  # the variable, not the attribute's 12.5
    assert "system_flooding" not in out.attrs
    assert float(out["flow_continuity_error"]) == pytest.approx(-0.37)


def test_an_unindexed_store_stays_unindexed(fixture_tree: Path):
    path = _processed(fixture_tree) / "SWMM_only_node_tseries.zarr"
    _node_dataset().to_zarr(path, mode="w", consolidated=False)
    assert "consolidated_metadata" not in json.loads((path / "zarr.json").read_text())
    _migrate(fixture_tree)
    assert "consolidated_metadata" not in json.loads((path / "zarr.json").read_text())
    out = xr.open_zarr(path, consolidated=False, decode_timedelta=False)
    assert float(out["system_flooding"]) == 12.5


def test_a_format_2_store_is_converted(fixture_tree: Path):
    path = _processed(fixture_tree) / "SWMM_only_node_tseries.zarr"
    _node_dataset().to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    assert (path / ".zgroup").is_file()
    _migrate(fixture_tree)
    out = xr.open_zarr(path, consolidated=True, decode_timedelta=False)
    assert out["system_flooding"].dims == () and float(out["system_flooding"]) == 12.5
    assert "system_flooding" not in out.attrs


def test_a_netcdf_store_is_rewritten_through_a_temporary_sibling_and_keeps_its_mtime(fixture_tree: Path):
    path = _processed(fixture_tree) / "SWMM_only_node_tseries.nc"
    _node_dataset().to_netcdf(path, engine="h5netcdf")
    import os

    os.utime(path, (1_600_000_000, 1_600_000_000))
    _migrate(fixture_tree)
    assert path.stat().st_mtime == 1_600_000_000
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".v0030tmp")] == []
    with xr.open_dataset(path, engine="h5netcdf") as out:
        assert out["system_flooding"].dims == () and float(out["system_flooding"]) == 12.5
        assert np.isnan(float(out["runoff_continuity_error"]))
        assert out["flow_continuity_error"].attrs["units"] == "percent"
        assert not any(name in out.attrs for name in NAMES)
        assert out.attrs["flow_units"] == "cms"
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    _migrate(fixture_tree)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first


def test_a_netcdf_conversion_that_fails_leaves_the_original_byte_identical_and_no_sibling(
    fixture_tree: Path, monkeypatch
):
    path = _processed(fixture_tree) / "SWMM_only_node_tseries.nc"
    _node_dataset().to_netcdf(path, engine="h5netcdf")
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def boom(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr("hhemt.utils.write_netcdf", boom)
    with pytest.raises(RuntimeError):
        from hhemt.version_migration.context import MigrationContext

        ctx = MigrationContext(target_dir=fixture_tree, dry_run=False, migration_id="V0030")
        ctx.netcdf_attrs_to_scalar_variables(path, NAMES)
        ctx.execute()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".v0030tmp")] == []


def test_a_dry_run_changes_nothing(fixture_tree: Path):
    before = _tree_digest(fixture_tree)
    runner.run_migration(
        fixture_tree,
        target=30,
        apply=False,
        cfg_paths={
            "system": fixture_tree / "cached_configs" / "system.yaml",
            "analysis": fixture_tree / "cached_configs" / "analysis.yaml",
        },
    )
    assert _tree_digest(fixture_tree) == before


def test_the_layout_version_is_30_and_the_v30_fixture_is_a_byte_copy_of_v29():
    import filecmp

    from hhemt.version_migration.constants import LAYOUT_VERSION

    assert LAYOUT_VERSION == 31
    cmp = filecmp.dircmp(FIXTURE_ROOT / "v29", FIXTURE_ROOT / "v30")

    def walk(c: filecmp.dircmp) -> list[str]:
        bad = list(c.left_only) + list(c.right_only) + list(c.diff_files) + list(c.funny_files)
        for sub in c.subdirs.values():
            bad += walk(sub)
        return bad

    assert walk(cmp) == []
    assert len(list((FIXTURE_ROOT / "v30").rglob("*"))) > 90, "control: the copy is not empty"
