"""Tests for the V0030 migration that the chunk's own file does not reach.

The migration is driven through the runner on a copy of the committed layout-29 fixture, and through the context
operations directly where a failure has to be injected. Inputs here: stores under member and sub-analysis
trees, a NetCDF file whose other variables and time axis must come through unchanged, a conversion that stops
between its two writes, a leftover temporary sibling, integer and full-precision attribute values, a store of the
same name outside a scenario's `processed` directory, and the units the converted variables carry against the
toolkit's own unit table. No solver is compiled or run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"
NAMES = ("flow_continuity_error", "runoff_continuity_error", "system_flooding")
NODE_STEM = "TRITONSWMM_SWMM_node_tseries"
LINK_STEM = "TRITONSWMM_SWMM_link_tseries"


def _migrate(work: Path):
    return runner.run_migration(
        work,
        target=30,
        apply=True,
        cfg_paths={
            "system": work / "cached_configs" / "system.yaml",
            "analysis": work / "cached_configs" / "analysis.yaml",
        },
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A stamped copy of the layout-29 fixture, so the runner plans V0030 alone."""
    work = tmp_path / "v29"
    shutil.copytree(FIXTURE_ROOT / "v0030_unit_test", work)
    runner.baseline(work, 29)
    return work


def _processed(work: Path) -> Path:
    return work / "sims" / "event_id.0" / "processed"


def _has_variables(path: Path) -> bool:
    return all(name in set(zarr.open_group(str(path), mode="r", use_consolidated=False).array_keys()) for name in NAMES)


def _has_attributes(path: Path) -> bool:
    attrs = dict(zarr.open_group(str(path), mode="r", use_consolidated=False).attrs)
    return any(name in attrs for name in NAMES)


@pytest.mark.parametrize("container, member", [("members", "member_0"), ("subanalyses", "sa_1")])
def test_stores_under_member_and_legacy_sub_analysis_trees_are_converted(tree, container, member):
    """Property: a layout-29 node and link store under `members/member_*` or the `subanalyses/sa_*` of
    the tree is converted exactly as one under the tree's own `sims`, and the tree's own stores still are.

    Input class: the two container spellings `collect_sims_dirs` widens over. Plausible wrong implementation: a
    store finder that walks only `{tree}/sims`, so every member of a sensitivity analysis keeps its attributes and a
    re-consolidation of the master loses the three figures the migration exists to keep.
    Must-fire control: the same stores before the migration carry attributes and no variables.
    Killed by: restricting the finder to the top-level `sims`. Passes under a second correct implementation that
    walks every `processed` directory under the tree.
    """
    member_processed = tree / container / member / "sims" / "event_id.9" / "processed"
    member_processed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_processed(tree), member_processed)
    member_stores = [member_processed / f"{stem}.zarr" for stem in (NODE_STEM, LINK_STEM)]
    assert all(_has_attributes(p) and not _has_variables(p) for p in member_stores)
    _migrate(tree)
    for store in member_stores + [_processed(tree) / f"{stem}.zarr" for stem in (NODE_STEM, LINK_STEM)]:
        assert _has_variables(store) and not _has_attributes(store), store


def test_a_store_with_a_migrated_name_outside_a_scenario_processed_directory_is_left_alone(tree):
    """Property: a zarr store whose name matches a per-scenario SWMM store but which sits outside
    `sims/*/processed/` keeps its attributes and gains no variable.

    Input class: a store of the same stem under `plots/`. Plausible wrong implementation: a finder that matches by
    file name anywhere under the tree and rewrites a store the migration does not own.
    Must-fire control: the real store of the same stem in `processed/` is converted by the same run.
    Killed by: a name-only walk. Passes under a second correct implementation that anchors on the scenario layout.
    """
    stray = tree / "plots" / f"{NODE_STEM}.zarr"
    stray.parent.mkdir(parents=True)
    shutil.copytree(_processed(tree) / f"{NODE_STEM}.zarr", stray)
    _migrate(tree)
    assert _has_attributes(stray) and not _has_variables(stray)
    assert _has_variables(_processed(tree) / f"{NODE_STEM}.zarr")


def _nc_dataset(attrs: dict) -> xr.Dataset:
    t = np.array(["2020-01-01T00:02:00", "2020-01-01T00:04:00", "2020-01-01T00:06:00"], dtype="datetime64[ns]")
    return xr.Dataset(
        {
            "depth": (("date_time", "node_id"), np.arange(6, dtype="float32").reshape(3, 2), {"units": "m"}),
            "is_coupling_node": (("node_id",), np.array([1, 0], dtype="int8"), {"flag_meanings": "a b"}),
            "max_flow": (("node_id",), np.array([0.5, 1.5]), {"units": "m3 s-1"}),
        },
        coords={"date_time": t, "node_id": ["n1", "n2"]},
        attrs=attrs,
    )


_LEGACY = {
    "flow_units": "cms",
    "runoff_continuity_error": float("nan"),
    "runoff_continuity_error_units": "percent",
    "flow_continuity_error": -0.37,
    "flow_continuity_error_units": "percent",
    "system_flooding": 12.5,
    "system_flooding_units": "10^6 L",
}


def test_a_netcdf_conversion_keeps_every_other_variable_and_the_time_axis_identical(tree):
    """Property: after the NetCDF conversion every variable that is not one of the three has the same values, dtype and
    attributes as before, the time coordinate is unchanged, and the dataset attributes the migration does not own
    remain.

    Input class: an `nc` store with a float32 series, an int8 flag, a float64 static variable and a datetime axis.
    Plausible wrong implementations: a rewrite that widens float32 to float64 or drops variable attributes (the
    summary was computed from the original dtype), re-encodes the time axis, or drops the attributes it does not own.
    Must-fire control: the three figures are present as variables with the stored values and units.
    Killed by: any lossy rewrite. Passes under a second correct implementation that edits the file in place.
    """
    path = _processed(tree) / "SWMM_only_node_tseries.nc"
    _nc_dataset(dict(_LEGACY)).to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        before = opened.load()
    _migrate(tree)
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        after = opened.load()
    for name in ("depth", "is_coupling_node", "max_flow"):
        assert after[name].dtype == before[name].dtype, name
        assert after[name].attrs == before[name].attrs, name
        np.testing.assert_array_equal(after[name].values, before[name].values)
    np.testing.assert_array_equal(after["date_time"].values, before["date_time"].values)
    assert after.attrs == {"flow_units": "cms"}
    assert float(after["flow_continuity_error"]) == -0.37 and after["system_flooding"].attrs["units"] == "10^6 L"


def test_a_netcdf_store_holding_both_keeps_the_variable_and_its_units_and_loses_the_attribute(tree):
    """Property: a NetCDF file that holds `system_flooding` as a variable and as an attribute keeps the variable (value
    and units) and loses the attribute, while the other two attributes are converted.

    Input class: the both-present shape for the NetCDF arm, which the chunk's own file builds for zarr only.
    Plausible wrong implementation: overwriting the existing variable with the attribute's value or dropping its
    units, or leaving the attribute in place so the next run plans the file again.
    Killed by: removing the existing-variable guard of the NetCDF arm. Passes under a second correct implementation
    that merges by taking the variable first.
    """
    path = _processed(tree) / "SWMM_only_node_tseries.nc"
    ds = _nc_dataset(dict(_LEGACY))
    ds["system_flooding"] = xr.DataArray(np.float64(99.0), attrs={"units": "10^6 L", "long_name": "kept"})
    ds.to_netcdf(path, engine="h5netcdf")
    _migrate(tree)
    with xr.open_dataset(path, engine="h5netcdf") as out:
        assert float(out["system_flooding"]) == 99.0 and out["system_flooding"].attrs["long_name"] == "kept"
        assert "system_flooding" not in out.attrs and "system_flooding_units" not in out.attrs
        assert float(out["flow_continuity_error"]) == -0.37


def test_an_interrupted_zarr_conversion_is_finished_by_a_rerun_and_ends_as_a_clean_run_does(tree, monkeypatch):
    """Property: a conversion that stops after the variables are written and before the attributes are removed is
    finished by running the migration again, and the store then equals one converted without interruption.

    Input class: a failure between the two writes of one store. Plausible wrong implementation: a conversion that
    cannot be re-entered (creating a variable that already exists fails the rerun, leaving the store half converted
    for good), or one that leaves the attribute so the figure exists twice.
    Must-fire control: the interrupted state really holds the variables and still holds the attributes.
    Killed by: removing the existing-variable guard. Passes under a second correct implementation that writes the
    attribute removal first and the variables second, provided the rerun reaches the same state.
    """
    from zarr.core.attributes import Attributes

    from hhemt.version_migration.context import MigrationContext

    store = _processed(tree) / f"{NODE_STEM}.zarr"
    twin = tree.parent / "twin.zarr"
    shutil.copytree(store, twin)
    real_put = Attributes.put
    calls = []

    def failing_put(self, new_attributes):
        """Raise on the rewrite that removes the three attributes, delegate any other."""
        calls.append(1)
        if not any(name in new_attributes for name in NAMES):
            raise OSError("killed between the variables and the attribute removal")
        return real_put(self, new_attributes)

    monkeypatch.setattr(Attributes, "put", failing_put)
    ctx = MigrationContext(target_dir=tree, dry_run=False, migration_id="V0030")
    ctx.zarr_attrs_to_scalar_variables(store, NAMES)
    with pytest.raises(OSError):
        ctx.execute()
    assert _has_variables(store) and _has_attributes(store), "precondition: a half-converted store"
    monkeypatch.undo()
    for target in (store, twin):
        again = MigrationContext(target_dir=tree, dry_run=False, migration_id="V0030")
        again.zarr_attrs_to_scalar_variables(target, NAMES)
        again.execute()
    assert _has_variables(store) and not _has_attributes(store)
    a = xr.open_zarr(store, consolidated=True, decode_timedelta=False).load()
    b = xr.open_zarr(twin, consolidated=True, decode_timedelta=False).load()
    xr.testing.assert_identical(a, b)


def test_a_leftover_temporary_file_does_not_block_or_survive_a_netcdf_conversion(tree):
    """Property: a `.v0030tmp` sibling left by an earlier crashed run neither stops the next conversion nor remains
    after it.

    Input class: a stale temporary of the migration's own name beside an unconverted file. Plausible wrong
    implementation: refusing to overwrite the temporary, or converting and leaving the stale file for the next
    reader of the directory.
    Must-fire control: the file is converted and the temporary's stale bytes are gone.
    Killed by: opening the temporary exclusively. Passes under a second correct implementation that uses a unique
    temporary name and removes any stale one.
    """
    path = _processed(tree) / "SWMM_only_node_tseries.nc"
    _nc_dataset(dict(_LEGACY)).to_netcdf(path, engine="h5netcdf")
    stale = path.with_name(path.name + ".v0030tmp")
    stale.write_bytes(b"stale bytes from a crashed run")
    _migrate(tree)
    with xr.open_dataset(path, engine="h5netcdf") as out:
        assert float(out["system_flooding"]) == 12.5
    assert not stale.exists()


def test_integer_and_full_precision_attribute_values_are_carried_exactly(tree):
    """Property: an integer-typed `system_flooding` attribute and a `flow_continuity_error` of full float precision
    become float64 variables equal to the attribute value, bit for bit.

    Input class: values a parser can write that are not short floats (the integer 0 and `0.1 + 0.2`). Plausible
    wrong implementation: a conversion through `float32` or through a string, which changes the digits the summary
    and the validation compare.
    Killed by: any narrowing. Passes under a second correct implementation that stores the value unchanged.
    """
    path = _processed(tree) / "SWMM_only_node_tseries.zarr"
    attrs = dict(_LEGACY, system_flooding=0, flow_continuity_error=0.1 + 0.2)
    _nc_dataset(attrs).drop_vars("date_time").to_zarr(path, mode="w", consolidated=True)
    _migrate(tree)
    out = xr.open_zarr(path, consolidated=True, decode_timedelta=False)
    assert out["flow_continuity_error"].dtype == np.float64 and float(out["flow_continuity_error"]) == 0.1 + 0.2
    assert out["system_flooding"].dtype == np.float64 and float(out["system_flooding"]) == 0.0


def test_the_converted_units_equal_the_toolkits_unit_table_for_all_three_names(tree):
    """Property: the unit a converted variable carries equals the unit the toolkit's CF table gives that name.

    Input class: the three names, converted from the layout-29 companion attributes. Plausible wrong implementation:
    a conversion that takes a unit from a different source than the table the rest of the toolkit stamps from, so a
    migrated store and a freshly written one disagree on a unit (the unit-presence tests read only that a unit
    exists).
    Must-fire control: the table has an entry for each name.
    Killed by: a wrong or missing unit on any of the three. Passes under a second correct implementation that reads
    the unit from the table directly.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    _migrate(tree)
    out = xr.open_zarr(_processed(tree) / f"{NODE_STEM}.zarr", consolidated=True, decode_timedelta=False)
    for name in NAMES:
        assert name in _CF_VARIABLE_MAP, name
        assert out[name].attrs["units"] == _CF_VARIABLE_MAP[name]["units"], name
