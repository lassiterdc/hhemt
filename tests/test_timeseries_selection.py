"""Time-series variable selection: the domain table, ``resolve_keep``, the three keep-fields, the preflight refusal,
the prune of a per-scenario store (zarr and NetCDF), its restartability, and how it meets the consolidation.

Small stores are built under ``tmp_path``; the golden-backed rows extract the committed multi-sim golden (as
``test_synth_timeseries_consolidation.py`` does). No solver is compiled or run. Everything this change creates
(the module, the config fields) is looked up inside the test bodies behind an assertion that it exists, so on a tree
without the change each test collects and fails with an AssertionError from this file.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis


def _sel():
    """The selection module, or an AssertionError saying it does not exist."""
    try:
        return importlib.import_module("hhemt.timeseries_selection")
    except ModuleNotFoundError as exc:
        raise AssertionError("src/hhemt/timeseries_selection.py is absent") from exc


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_none_and_all_give_the_whole_domain_of_every_kind_and_family():
    """Property: ``None`` and ``"all"`` return every variable of the (kind, family) domain, in the domain's order.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``None`` treated as an empty selection (everything pruned),
    or ``"all"`` dropping the coupled-only ``coupling_exchange_flow``.
    Reachability: ``resolve_keep`` is the one resolver. Both-states anchor: asserts equality with ``DOMAIN``.
    Killed by: either. A second correct implementation derives the domain from the writers at import.
    """
    sel = _sel()
    for (kind, family), domain in sel.DOMAIN.items():
        assert sel.resolve_keep(kind, family, None) == domain
        assert sel.resolve_keep(kind, family, "all") == domain
    assert "coupling_exchange_flow" in sel.resolve_keep("swmm_node", "tritonswmm", "all")
    assert "coupling_exchange_flow" not in sel.resolve_keep("swmm_node", "swmm_only", "all")


def test_default_is_family_conditional():
    """Property: ``"default"`` drops ``flooding`` for the coupled node store only, drops the three gridded names for
    both gridded families, and equals ``"all"`` everywhere else.

    Class: NEW CAPABILITY. Plausible wrong implementations: the coupled exclusion applied to standalone SWMM too; the
    gridded exclusion applied to one family only; a link exclusion.
    Reachability: all six (kind, family) pairs. Both-states anchor: asserts set differences against "all".
    Must-fire control: the standalone family keeps ``flooding``. Killed by: a wrong family default.
    A second correct implementation stores the sets explicitly; the differences are the same.
    """
    sel = _sel()
    full = lambda k, f: set(sel.resolve_keep(k, f, "all"))  # noqa: E731
    default = lambda k, f: set(sel.resolve_keep(k, f, "default"))  # noqa: E731
    assert full("swmm_node", "tritonswmm") - default("swmm_node", "tritonswmm") == {"flooding"}
    assert default("swmm_node", "swmm_only") == full("swmm_node", "swmm_only")
    assert "flooding" in default("swmm_node", "swmm_only")
    for family in ("tritonswmm", "swmm_only"):
        assert default("swmm_link", family) == full("swmm_link", family)
    for family in ("tritonswmm", "triton_only"):
        assert full("triton_gridded", family) - default("triton_gridded", family) == {
            "max_flood_depth",
            "unit_discharge_x",
            "unit_discharge_y",
        }
        assert "flood_depth" in default("triton_gridded", family)


def test_the_swmm_system_kind_belongs_to_the_standalone_family_and_its_default_equals_all():
    """Property: ``resolve_keep("swmm_system", "swmm_only", "default")`` equals the whole domain that ``"all"``,
    ``None`` and a blank string give, the domain is the six system series in the order the writer stores them, and
    no keep-field exists for the kind (its setting reads ``None`` from any config).

    Class: NEW CAPABILITY. Plausible wrong implementations: a ``"default"`` that drops a series (the pruned system
    summary would then lose statistics silently), the kind offered to the coupled family, or a keep-field read from a
    config that does not carry one. Reachability: ``resolve_keep``, ``DOMAIN`` and ``keep_setting`` are the module's
    one table and its two readers. Both-states anchor: the other kinds of the same table are checked by the tests
    above. Must-fire control: the coupled family has no ``swmm_system`` domain, so the pair is refused naming the
    families by kind. Killed by: those. Passes under a second correct implementation that stores the kind's domain
    explicitly, if it keeps the order.
    """
    sel = _sel()
    whole = sel.resolve_keep("swmm_system", "swmm_only", "all")
    assert sel.resolve_keep("swmm_system", "swmm_only", "default") == whole
    assert sel.resolve_keep("swmm_system", "swmm_only", None) == whole
    assert sel.resolve_keep("swmm_system", "swmm_only", "  ") == whole
    assert whole == (
        "system_flooding_rate",
        "system_outfall_flow",
        "system_stored_volume",
        "system_lateral_inflow",
        "system_runoff",
        "system_rainfall",
    )
    assert sel.FAMILIES_BY_KIND["swmm_system"] == ("swmm_only",)
    assert sel.TIME_DIM_BY_KIND["swmm_system"] == "date_time"
    assert sel.keep_setting(SimpleNamespace(), "swmm_system") is None
    for family in ("tritonswmm", "triton_only"):
        assert ("swmm_system", family) not in sel.DOMAIN
        with pytest.raises(ValueError, match="swmm_system"):
            sel.resolve_keep("swmm_system", family, "default")


def test_the_series_the_inventory_added_are_in_default_for_both_families():
    """Property: ``volume`` and ``lateral_inflow`` (node) and ``link_volume`` (link) are in ``"default"`` and in
    ``"all"`` for both SWMM families; the capacity series is named per family.

    Class: NEW CAPABILITY. Plausible wrong implementation: the three series left out of the table (they were added by
    the SWMM output inventory after the table's first draft), or one capacity name for both families.
    Reachability: the table is the domain. Both-states anchor: membership. Killed by: omission.
    A second correct implementation adds them to the exclusions; the shape says include.
    """
    sel = _sel()
    for family in ("tritonswmm", "swmm_only"):
        for name in ("volume", "lateral_inflow"):
            assert name in sel.resolve_keep("swmm_node", family, "default")
            assert name in sel.resolve_keep("swmm_node", family, "all")
        assert "link_volume" in sel.resolve_keep("swmm_link", family, "default")
    assert "capacity_setting" in sel.DOMAIN[("swmm_link", "tritonswmm")]
    assert "capacity" not in sel.DOMAIN[("swmm_link", "tritonswmm")]
    assert "capacity" in sel.DOMAIN[("swmm_link", "swmm_only")]
    assert "capacity_setting" not in sel.DOMAIN[("swmm_link", "swmm_only")]


def test_a_list_returns_exactly_its_names_in_order_and_may_force_in_an_excluded_one():
    """Property: a list resolves to exactly its names, in the order written, repeats dropped, and may name a variable
    the default set excludes.

    Class: NEW CAPABILITY. Plausible wrong implementations: returning domain order; intersecting with the default
    set (silently dropping a forced-in ``max_flood_depth``); not de-duplicating.
    Reachability: the list branch of ``resolve_keep``. Both-states anchor: tuple equality.
    Killed by: those. A second correct implementation returns a frozenset; the shape fixes a tuple.
    """
    sel = _sel()
    assert sel.resolve_keep(
        "triton_gridded", "tritonswmm", ["unit_discharge_x", "flood_depth", "unit_discharge_x"]
    ) == (
        "unit_discharge_x",
        "flood_depth",
    )
    assert sel.resolve_keep("triton_gridded", "tritonswmm", ["max_flood_depth"]) == ("max_flood_depth",)
    # A blank keep-list keeps everything (it used to resolve to () and delete every domain series; see the blank test).
    assert sel.resolve_keep("swmm_node", "tritonswmm", []) == sel.DOMAIN[("swmm_node", "tritonswmm")]
    # A name another enabled family writes is admitted and filtered: the standalone family ignores a coupled-only name.
    assert sel.resolve_keep("swmm_link", "swmm_only", ["flow", "capacity_setting"]) == ("flow",)
    assert sel.resolve_keep("swmm_link", "tritonswmm", ["flow", "capacity_setting"]) == ("flow", "capacity_setting")


def test_unknown_names_and_impossible_pairs_are_refused_naming_the_domain():
    """Property: an unknown set name, a variable outside the domain and a (kind, family) no model writes each raise
    ``ValueError`` whose message names the domain.

    Class: NEW CAPABILITY. Plausible wrong implementation: silently ignoring a misspelling (the user then keeps
    nothing they meant to keep). Reachability: the three refusal branches. Both-states anchor: the exception type
    and a domain member in the message. Killed by: removing a branch. A second correct implementation raises a
    subclass.
    """
    sel = _sel()
    with pytest.raises(ValueError, match="depth"):
        sel.resolve_keep("swmm_node", "tritonswmm", "everything")
    with pytest.raises(ValueError, match="flood_depth"):
        sel.resolve_keep("triton_gridded", "tritonswmm", ["flood_dept"])
    with pytest.raises(ValueError, match="no time-series store"):
        sel.resolve_keep("swmm_node", "triton_only", "all")


# ---------------------------------------------------------------------------
# Configuration, taxonomy, preflight
# ---------------------------------------------------------------------------
def test_the_three_keep_fields_exist_default_none_and_are_experiment_bucketed():
    """Property: the three ``timeseries_keep_*`` fields exist on ``analysis_config`` with default ``None``, and the
    reproducer taxonomy buckets each ``experiment``.

    Class: NEW CAPABILITY. Plausible wrong implementation: a default other than ``None`` (today's behaviour keeps
    everything), or a missing taxonomy row (``test_field_bucket_is_total`` then reds).
    Reachability: the schema and the table. Both-states anchor: names, defaults and a table value.
    Killed by: either. A second correct implementation derives buckets from an annotation.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_taxonomy import _FIELD_BUCKET

    for kind in ("swmm_node", "swmm_link", "triton_gridded"):
        name = f"timeseries_keep_{kind}"
        assert name in analysis_config.model_fields, f"{name} is not an analysis field"
        assert analysis_config.model_fields[name].default is None
        assert _FIELD_BUCKET.get(name) == "experiment"


def _preflight_errors(cfg_system_kwargs: dict, **keeps) -> list:
    from hhemt import validation

    check = getattr(validation, "_validate_timeseries_keep", None)
    assert check is not None, "validation._validate_timeseries_keep does not exist"
    cfg_analysis = SimpleNamespace(
        timeseries_keep_swmm_node=keeps.get("node"),
        timeseries_keep_swmm_link=keeps.get("link"),
        timeseries_keep_triton_gridded=keeps.get("gridded"),
    )
    result = validation.ValidationResult(context="test")
    check(SimpleNamespace(**cfg_system_kwargs), cfg_analysis, result)
    return list(result.errors)


_COUPLED = dict(toggle_tritonswmm_model=True, toggle_swmm_model=False, toggle_triton_model=False)


def test_preflight_refuses_unknown_sets_and_out_of_domain_names_and_admits_the_rest():
    """Property: preflight errors on an unknown set name and on a listed name outside the domain of every enabled
    family, naming the domain; it admits ``None``, both sets, and an in-domain list.

    Class: NEW CAPABILITY. Plausible wrong implementations: no check; a check against the union of ALL families
    including disabled ones (admits ``capacity`` for a coupled-only run).
    Reachability: the five rows are the branches. Both-states anchor: error counts per row.
    Must-fire control: ``capacity`` (standalone-only) is refused when only the coupled family is enabled and admitted
    when standalone SWMM is enabled too. Killed by: either wrong implementation.
    A second correct implementation raises instead of accumulating.
    """
    assert _preflight_errors(_COUPLED) == []
    assert _preflight_errors(_COUPLED, node="all", link="default", gridded="default") == []
    assert _preflight_errors(_COUPLED, node=["depth", "coupling_exchange_flow"], link=["capacity_setting"]) == []
    bad_set = _preflight_errors(_COUPLED, node="everything")
    assert len(bad_set) == 1 and "everything" in bad_set[0].message and "depth" in bad_set[0].message
    bad_name = _preflight_errors(_COUPLED, gridded=["flood_dept"])
    assert len(bad_name) == 1 and "flood_dept" in bad_name[0].message and "flood_depth" in bad_name[0].message
    assert len(_preflight_errors(_COUPLED, link=["capacity"])) == 1
    both = dict(toggle_tritonswmm_model=True, toggle_swmm_model=True, toggle_triton_model=False)
    assert _preflight_errors(both, link=["capacity"]) == []


# ---------------------------------------------------------------------------
# The prune, on small stores
# ---------------------------------------------------------------------------
def _node_ds() -> xr.Dataset:
    t = np.arange(1, 5) * 10
    rng = np.random.default_rng(3)
    dyn = lambda: (("node_id", "date_time"), rng.normal(size=(3, 4)))  # noqa: E731
    return xr.Dataset(
        {
            "depth": dyn(),
            "head": dyn(),
            "flooding": dyn(),
            "inflow_flow": dyn(),
            "mystery_series": dyn(),
            "is_coupling_node": (("node_id",), np.array([1, 0, 1], dtype="int8")),
            "max_ponded_depth": (("node_id",), np.array([1.0, 2.0, 3.0])),
        },
        coords={"node_id": ["a", "b", "c"], "date_time": t},
    )


def _write_zarr(path: Path, ds: xr.Dataset, *, consolidated: bool = True) -> Path:
    ds.to_zarr(path, mode="w", consolidated=consolidated)
    return path


def _open(path: Path) -> xr.Dataset:
    from hhemt.utils import zarr_store_has_index

    return xr.open_zarr(path, consolidated=zarr_store_has_index(path)).load()


def test_a_zarr_prune_keeps_the_listed_and_the_static_variables_and_records_the_names(tmp_path):
    """Property: after the prune the store holds the listed variables, every static variable, and any time-variant
    variable outside the domain; it records the removed names; the removed arrays' files are gone; the index lists
    exactly what is left.

    Class: NEW CAPABILITY. Plausible wrong implementations: pruning static variables (they have no ``date_time``);
    pruning ``mystery_series`` (outside the domain, not the selection's to remove); not recording the names, or
    recording them unsorted; leaving the deleted arrays in the consolidated index (a reader then fails or lies).
    Reachability: the node store above holds all four classes. Both-states anchor: asserts the surviving set and
    the attribute. Control: a keep-list resolving to the whole domain prunes nothing and records nothing.
    Killed by: those. A second correct implementation rewrites the store; the same state results.
    """
    sel = _sel()
    store = _write_zarr(tmp_path / "node.zarr", _node_ds())
    removed = sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth", "flooding"])
    assert sorted(removed) == ["head", "inflow_flow"]
    left = _open(store)
    assert set(left.data_vars) == {"depth", "flooding", "mystery_series", "is_coupling_node", "max_ponded_depth"}
    assert list(left.attrs[sel.PRUNED_ATTR]) == ["head", "inflow_flow"]
    assert sel.pruned_variables(store) == ("head", "inflow_flow")
    assert not (store / "head").exists() and not (store / "inflow_flow").exists(), "chunk files must go too"
    assert (store / "depth").exists()
    np.testing.assert_array_equal(left["depth"].values, _node_ds()["depth"].values)

    untouched = _write_zarr(tmp_path / "all.zarr", _node_ds())
    assert (
        sel.prune_timeseries_store(
            untouched, "swmm_node", "tritonswmm", sel.resolve_keep("swmm_node", "tritonswmm", None)
        )
        == ()
    )
    assert sel.PRUNED_ATTR not in _open(untouched).attrs
    assert set(_open(untouched).data_vars) == set(_node_ds().data_vars)


def test_a_variable_without_the_time_dimension_is_static_even_when_it_carries_a_domain_name(tmp_path):
    """Property: the static rule is the PROPERTY (no time dimension), not a list of names: a variable named ``head``
    that has no ``date_time`` dimension is kept by a keep-list that omits ``head``, in a zarr and a NetCDF store.

    Class: NEW CAPABILITY. Plausible wrong implementation: pruning by domain membership alone, which would delete a
    per-node value that a later chunk stores under a domain name. Reachability: the store is built with ``head``
    reduced over time. Both-states anchor: the surviving set. Must-fire control: ``flooding``, which does have the
    time dimension and is not listed, is removed in the same call. Killed by: dropping the time-dimension test.
    A second correct implementation tests the dimension through the variable's encoding; the outcome is the same.
    """
    sel = _sel()
    ds = _node_ds()
    ds["head"] = ds["head"].max("date_time")
    zarr_store = _write_zarr(tmp_path / "s.zarr", ds)
    nc_store = tmp_path / "s.nc"
    ds.to_netcdf(nc_store, engine="h5netcdf")
    for store in (zarr_store, nc_store):
        removed = sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
        assert "head" not in removed and "flooding" in removed, store.name
    assert "head" in _open(zarr_store).data_vars
    with xr.open_dataset(nc_store, engine="h5netcdf") as left:
        assert "head" in left.data_vars and "flooding" not in left.data_vars


def test_a_second_prune_is_a_no_op_and_a_wider_second_prune_unions_the_names(tmp_path):
    """Property: pruning twice with the same list removes nothing the second time; a second, narrower keep-list
    removes more and the attribute becomes the sorted union.

    Class: NEW CAPABILITY. Plausible wrong implementation: overwriting the attribute with only the latest names,
    which hides an earlier removal from the rebuild guard. Reachability: two calls on one store.
    Both-states anchor: the attribute value after each call. Killed by: overwrite instead of union.
    A second correct implementation keeps a list per call; the union is what a reader needs.
    """
    sel = _sel()
    store = _write_zarr(tmp_path / "node.zarr", _node_ds())
    first = sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth", "flooding", "head"])
    assert first == ("inflow_flow",)
    assert sel.pruned_variables(store) == ("inflow_flow",)
    assert sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth", "flooding", "head"]) == ()
    sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
    assert sel.pruned_variables(store) == ("flooding", "head", "inflow_flow")
    assert set(_open(store).data_vars) >= {"depth", "is_coupling_node"} and "flooding" not in _open(store).data_vars


def test_an_unindexed_store_is_pruned_without_gaining_an_index(tmp_path):
    """Property: a store with no consolidated index is pruned and stays unindexed.

    Class: NEW CAPABILITY. Plausible wrong implementation: always consolidating, which would hand a reader inside a
    deferred-consolidation block an index it was told it could not trust. Reachability: ``consolidated=False``
    write. Both-states anchor: ``zarr_store_has_index`` before and after. Killed by: unconditional consolidation.
    A second correct implementation consolidates only the touched group; the property is the absence of an index.
    """
    from hhemt.utils import zarr_store_has_index

    sel = _sel()
    store = _write_zarr(tmp_path / "n.zarr", _node_ds(), consolidated=False)
    assert not zarr_store_has_index(store)
    sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
    assert not zarr_store_has_index(store)
    assert "head" not in _open(store).data_vars


def test_an_interrupted_zarr_prune_is_recognisable_and_a_rerun_ends_in_the_uninterrupted_state(tmp_path, monkeypatch):
    """Property: if the prune dies after the attribute write and after deleting one array, the attribute already
    names every doomed variable, and re-running the same prune ends in exactly the state of an uninterrupted prune
    (variables, attribute, values).

    Class: NEW CAPABILITY. Plausible wrong implementation: writing the attribute after the deletes (an interrupted
    prune then leaves arrays missing with no record, so a summary rebuild would lose statistics silently).
    Reachability: ``_delete_zarr_array`` is patched to raise on its second call. Both-states anchor: the stored
    names and values. Killed by: deleting before recording. A second correct implementation records per variable;
    the all-names-first check below is then the one to adjust.
    """
    sel = _sel()
    clean = _write_zarr(tmp_path / "clean.zarr", _node_ds())
    sel.prune_timeseries_store(clean, "swmm_node", "tritonswmm", ["depth"])

    hurt = _write_zarr(tmp_path / "hurt.zarr", _node_ds())
    real = sel._delete_zarr_array
    calls = []

    def die_on_second(group, name):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("killed mid-prune")
        real(group, name)

    monkeypatch.setattr(sel, "_delete_zarr_array", die_on_second)
    with pytest.raises(RuntimeError, match="killed"):
        sel.prune_timeseries_store(hurt, "swmm_node", "tritonswmm", ["depth"])
    assert sel.pruned_variables(hurt) == ("flooding", "head", "inflow_flow"), "the names are recorded before deleting"
    monkeypatch.setattr(sel, "_delete_zarr_array", real)
    sel.prune_timeseries_store(hurt, "swmm_node", "tritonswmm", ["depth"])
    a, b = _open(clean), _open(hurt)
    xr.testing.assert_identical(a, b)


def _write_nc(path: Path) -> Path:
    _node_ds().to_netcdf(path, engine="h5netcdf")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_netcdf_prune_keeps_listed_and_static_variables_and_records_the_names(tmp_path):
    """Property: an uninterrupted NetCDF prune leaves exactly the listed, the static and the out-of-domain
    variables and records the removed names; no temporary sibling remains.

    Class: NEW CAPABILITY. Plausible wrong implementation: dropping static variables, or leaving the temporary file
    behind. Reachability: a small h5netcdf file. Both-states anchor: surviving names and attribute.
    Killed by: those. A second correct implementation rewrites in place after copying; the end state is the same.
    """
    sel = _sel()
    store = _write_nc(tmp_path / "node.nc")
    removed = sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth", "flooding"])
    assert sorted(removed) == ["head", "inflow_flow"]
    with xr.open_dataset(store, engine="h5netcdf") as ds:
        assert set(ds.data_vars) == {"depth", "flooding", "mystery_series", "is_coupling_node", "max_ponded_depth"}
    assert sel.pruned_variables(store) == ("head", "inflow_flow")
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_netcdf_prune_whose_temporary_write_fails_leaves_the_original_byte_identical(tmp_path, monkeypatch):
    """Property: if writing the temporary file raises, the original store is byte-identical to its pre-prune bytes
    and no temporary sibling remains.

    Class: NEW CAPABILITY. Plausible wrong implementation: writing the pruned data over the original (a crash
    leaves a partial file where the only copy of the series was). Reachability: ``write_netcdf`` is patched to
    write a partial file and raise. Both-states anchor: file hash and directory listing.
    Must-fire control: the unpatched call on a copy does change the bytes. Killed by: in-place writing.
    A second correct implementation writes to a temp directory; the sibling check is then adapted.
    """
    sel = _sel()
    store = _write_nc(tmp_path / "node.nc")
    control = _write_nc(tmp_path / "control.nc")
    before = _sha(store)
    sel.prune_timeseries_store(control, "swmm_node", "tritonswmm", ["depth"])
    assert _sha(control) != before, "control: a real prune does change the bytes"

    def partial_then_raise(ds, fname_out, *a, **k):
        Path(fname_out).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(sel, "write_netcdf", partial_then_raise)
    with pytest.raises(OSError, match="disk full"):
        sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
    assert _sha(store) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["control.nc", "node.nc"]


def test_a_netcdf_prune_whose_replace_fails_leaves_the_original_byte_identical(tmp_path, monkeypatch):
    """Property: if ``os.replace`` raises, the original is byte-identical and the temporary sibling is removed.

    Class: NEW CAPABILITY. Plausible wrong implementation: no cleanup on a failed replace, leaving a stray full-size
    file. Reachability: ``os.replace`` patched inside the module. Both-states anchor: hash and listing.
    Killed by: skipping cleanup. A second correct implementation renames the original aside first; the bytes at
    the original path would then differ during the call but not after.
    """
    sel = _sel()
    store = _write_nc(tmp_path / "node.nc")
    before = _sha(store)

    def refuse(src, dst):
        raise OSError("cannot replace")

    monkeypatch.setattr(sel.os, "replace", refuse)
    with pytest.raises(OSError, match="cannot replace"):
        sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
    assert _sha(store) == before
    assert [p.name for p in tmp_path.iterdir()] == ["node.nc"]


def test_the_summary_rebuild_guard_refuses_a_pruned_store_and_passes_an_unpruned_one(tmp_path):
    """Property: the guard raises ``ProcessingError`` naming the pruned variables and the store for a pruned store,
    and does nothing for an unpruned one.

    Class: NEW CAPABILITY. Plausible wrong implementation: no guard (a rebuild from fewer series returns fewer
    statistics with no error), or a guard that refuses every store. Reachability: the static guard method.
    Both-states anchor: exception type, names in the message. Must-fire control: the unpruned store passes.
    Killed by: either. A second correct implementation raises at open time; the property is the refusal.
    """
    sel = _sel()
    from hhemt.exceptions import ProcessingError
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as pp

    guard = getattr(pp, "_refuse_summary_rebuild_from_pruned_store", None)
    assert guard is not None, "the rebuild guard does not exist"
    clean = _write_zarr(tmp_path / "clean.zarr", _node_ds())
    guard(clean)
    pruned = _write_zarr(tmp_path / "pruned.zarr", _node_ds())
    sel.prune_timeseries_store(pruned, "swmm_node", "tritonswmm", ["depth"])
    with pytest.raises(ProcessingError) as err:
        guard(pruned)
    assert "head" in str(err.value) and "pruned.zarr" in str(err.value)


# ---------------------------------------------------------------------------
# Golden-backed rows: the prune in a real analysis, and how it meets the time-series consolidation
# ---------------------------------------------------------------------------
def _golden(golden_extractor, *, extra: str = ""):
    from hhemt.config.analysis import analysis_config

    assert "timeseries_keep_swmm_node" in analysis_config.model_fields, "the keep fields do not exist"
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    # An explicit false keeps the per-scenario stores these rows read back; a row whose extra sets the key owns it.
    keep_copies = (
        ""
        if "toggle_clear_scenario_timeseries_after_consolidation" in extra
        else "toggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + keep_copies + extra)
    analysis = construct_golden_analysis(system_dir)
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        nodes = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)["node_id"].values[
            :3
        ]
        f = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(f"{n}\n" for n in nodes))
    return analysis


def _node_store(analysis, i: int) -> Path:
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(i, analysis).scen_paths.output_tritonswmm_node_time_series


def test_the_prune_in_a_real_analysis_follows_the_config_and_leaves_the_summary_untouched(golden_extractor):
    """Property: with ``timeseries_keep_swmm_node: [depth]`` in the persisted config, ``write_summary_outputs``
    prunes each scenario's node store to ``depth`` plus its static variables, leaves the link store alone (its field
    is ``None``), and does not change the node summary store.

    Class: NEW CAPABILITY. Plausible wrong implementation: pruning before the summary is written, or pruning a kind
    whose field is ``None``. Reachability: the golden carries node, link and summary stores per scenario.
    Both-states anchor: the summary's bytes (file tree digest) and the link store's variable set before and after.
    Killed by: pruning the wrong kind or the summary. A second correct implementation prunes inside the export call.
    """
    a = _golden(golden_extractor, extra="timeseries_keep_swmm_node: [depth]\n")
    proc = a._retrieve_sim_run_processing_object(0)
    summary = proc.scen_paths.output_tritonswmm_node_summary
    link = proc.scen_paths.output_tritonswmm_link_time_series

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
        return h.hexdigest()

    summary_before, link_before = digest(summary), digest(link)
    proc.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    node = xr.open_zarr(_node_store(a, 0), consolidated=False)
    time_variant = {v for v in node.data_vars if "date_time" in node[v].dims}
    assert time_variant == {"depth"}
    assert "max_ponded_depth" in node.data_vars, "static report variables stay"
    assert digest(summary) == summary_before
    assert digest(link) == link_before


def test_a_summary_rebuild_from_the_pruned_scenario_store_is_refused(golden_extractor):
    """Property: after the prune, deleting the node summary and calling the build pass raises ``ProcessingError``
    naming a pruned variable; the same call on an unpruned scenario builds.

    Class: NEW CAPABILITY. Plausible wrong implementation: a rebuild that returns fewer statistics silently.
    Reachability: the summary file is deleted, so ``_summary_usable`` is False and the build pass runs.
    Both-states anchor: exception type, names, and the control. Killed by: no guard.
    A second correct implementation rebuilds from the consolidated store; the shape refuses instead.
    """
    import shutil

    from hhemt.exceptions import ProcessingError

    a = _golden(golden_extractor, extra="timeseries_keep_swmm_node: [depth]\n")
    proc0 = a._retrieve_sim_run_processing_object(0)
    proc0.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    shutil.rmtree(proc0.scen_paths.output_tritonswmm_node_summary)
    with pytest.raises(ProcessingError) as err:
        a._retrieve_sim_run_processing_object(0).write_summary_outputs(which="SWMM", model_type="tritonswmm")
    assert "head" in str(err.value)
    proc1 = a._retrieve_sim_run_processing_object(1)  # never pruned: control
    shutil.rmtree(proc1.scen_paths.output_tritonswmm_node_summary)
    proc1.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    assert proc1.scen_paths.output_tritonswmm_node_summary.exists()


def _prune_all(a, keep) -> None:
    for i in a.df_sims.index:
        a._retrieve_sim_run_processing_object(i).write_summary_outputs(which="SWMM", model_type="tritonswmm")
    return None


def test_consolidating_pruned_scenarios_stores_exactly_the_kept_variables_and_clearing_still_verifies(golden_extractor):
    """Property: when every scenario is pruned the same way, the consolidated store holds exactly the kept
    time-variant variables plus the flag, the variable-set guard does not trip, and with clearing on every
    scenario's slice still passes its read-back and its stores are cleared.

    Class: NEW CAPABILITY. Plausible wrong implementations: the append guard comparing against the unpruned
    variable set (it would refuse every pruned scenario), or the read-back comparing against a variable the prune
    removed. Reachability: golden scenarios pruned through ``write_summary_outputs``, then consolidated.
    Both-states anchor: the stored variable set and the existence of the cleared stores' absence.
    Killed by: either. A second correct implementation consolidates before pruning; the stored set would then
    hold every variable, which the shape forbids.
    """
    a = _golden(
        golden_extractor,
        extra="timeseries_keep_swmm_node: [depth, flooding]\n"
        "toggle_clear_scenario_timeseries_after_consolidation: true\n",
    )
    _prune_all(a, None)
    a.process.consolidate_timeseries()
    stored = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group="tritonswmm/swmm_node/timeseries")
    assert set(stored.data_vars) == {"depth", "flooding", "is_coupling_node"}
    assert not any(_node_store(a, i).exists() for i in a.df_sims.index)
    link = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group="tritonswmm/swmm_link/timeseries")
    assert "flow" in link.data_vars, "the link field is None, so every link series is kept"


def test_a_scenario_pruned_differently_is_refused_by_the_append_guard_naming_the_variable(golden_extractor):
    """Property: if one scenario's store was pruned differently from its siblings (a changed keep-list), the append
    raises ``ProcessingError`` naming the variable that differs, before writing it.

    Class: NEW CAPABILITY. Plausible wrong implementation: tolerating a differing variable set, which makes the
    consolidated store unreadable (measured). Reachability: event 2's store is pruned further by hand.
    Both-states anchor: exception type, name and the stored event list. Killed by: no guard.
    A second correct implementation pads the missing variable with NaN; the shape raises.
    """
    a = _golden(golden_extractor, extra="timeseries_keep_swmm_node: [depth, flooding]\n")
    _prune_all(a, None)
    sel = _sel()
    sel.prune_timeseries_store(_node_store(a, 2), "swmm_node", "tritonswmm", ["depth"])
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError" and "flooding" in str(err.value)
    stored = xr.open_zarr(
        a.analysis_paths.analysis_datatree_zarr, group="tritonswmm/swmm_node/timeseries", consolidated=False
    )
    assert [int(v) for v in stored["event_iloc"].values] == [0, 1]


def test_the_keep_lists_join_the_consolidation_fingerprint_by_what_they_resolve_to(golden_extractor):
    """Property: changing what a keep-list resolves to changes the fingerprint; reordering the list, or writing
    ``"all"`` for ``None``, does not.

    Class: NEW CAPABILITY. Plausible wrong implementations: keep-lists absent from the fingerprint (a tree built
    before the prune is reused after it); hashing the raw field text (a reordered list or ``"all"`` for ``None``
    would rebuild every tree for no change). Reachability: the fingerprint is computed from the analysis config.
    Both-states anchor: hash equality and inequality. Killed by: either. A second correct implementation hashes the
    resolved tuples unsorted; the reorder assertion would then be the one to drop, so it is a requirement.
    """
    a = _golden(golden_extractor)
    cfg = a.cfg_analysis
    base = a.process._consolidation_inputs_fingerprint()
    cfg.timeseries_keep_swmm_node = "all"
    assert a.process._consolidation_inputs_fingerprint() == base, "None and 'all' resolve alike"
    cfg.timeseries_keep_swmm_node = ["depth", "flooding"]
    one = a.process._consolidation_inputs_fingerprint()
    assert one != base
    cfg.timeseries_keep_swmm_node = ["flooding", "depth"]
    assert a.process._consolidation_inputs_fingerprint() == one
    cfg.timeseries_keep_swmm_node = ["depth"]
    assert a.process._consolidation_inputs_fingerprint() != one
    cfg.timeseries_keep_swmm_node = None
    cfg.timeseries_keep_triton_gridded = "default"
    assert a.process._consolidation_inputs_fingerprint() != base


def test_a_blank_keep_field_keeps_everything_and_prunes_nothing(tmp_path):
    """Property: ``[]``, ``""`` and a whitespace-only string resolve exactly as ``None`` does (the whole domain), pass
    preflight, and prune nothing; a one-name list still prunes, and ``None`` still prunes nothing.

    Class: NEW CAPABILITY. Plausible wrong implementations: ``[]`` resolving to an empty keep-list (measured: it
    deleted every domain time series), or ``""`` raising as an unknown set name. The config description says "None or
    blank keeps everything" and scope S6 says "blank meaning all".
    Reachability: the three blank spellings reach ``resolve_keep``, preflight and the prune. Both-states anchor:
    equality with ``None``'s result and the surviving variable set. Must-fire control: the one-name list prunes.
    Killed by: either wrong implementation. A second correct implementation normalises blanks in the config model;
    the observable results are the same.
    """
    sel = _sel()
    whole = sel.resolve_keep("swmm_node", "tritonswmm", None)
    for blank in ([], "", "   ", "\t"):
        assert sel.resolve_keep("swmm_node", "tritonswmm", blank) == whole, repr(blank)
        assert _preflight_errors(_COUPLED, node=blank, link=blank, gridded=blank) == [], repr(blank)
        store = _write_zarr(tmp_path / f"b{abs(hash(repr(blank)))}.zarr", _node_ds())
        assert (
            sel.prune_timeseries_store(
                store, "swmm_node", "tritonswmm", sel.resolve_keep("swmm_node", "tritonswmm", blank)
            )
            == ()
        )
        assert set(_open(store).data_vars) == set(_node_ds().data_vars)
    assert sel.resolve_keep("swmm_node", "tritonswmm", " all ") == whole
    one = _write_zarr(tmp_path / "one.zarr", _node_ds())
    assert sorted(sel.prune_timeseries_store(one, "swmm_node", "tritonswmm", ["depth"])) == [
        "flooding",
        "head",
        "inflow_flow",
    ]


def test_the_keep_fields_treat_a_blank_as_none_in_the_fingerprint_and_the_real_prune(golden_extractor):
    """Property: a blank keep-list (``[]`` and ``""``) neither changes the consolidation fingerprint relative to
    ``None`` nor prunes a scenario store when ``write_summary_outputs`` runs.

    Class: NEW CAPABILITY. Plausible wrong implementation: the blank handled in ``resolve_keep`` but not at the
    prune call site (a guard on ``value is None`` only), or the fingerprint hashing the raw value.
    Reachability: the golden analysis, config set in memory, one scenario processed. Both-states anchor: the
    fingerprint equals the ``None`` fingerprint and the store's time-variant set is unchanged.
    Killed by: either. A second correct implementation normalises in the config model.
    """
    a = _golden(golden_extractor)
    cfg = a.cfg_analysis
    base = a.process._consolidation_inputs_fingerprint()
    before = set(xr.open_zarr(_node_store(a, 0), consolidated=False).data_vars)
    for blank in ([], ""):
        cfg.timeseries_keep_swmm_node = blank
        cfg.timeseries_keep_swmm_link = blank
        cfg.timeseries_keep_triton_gridded = blank
        assert a.process._consolidation_inputs_fingerprint() == base, repr(blank)
        a._retrieve_sim_run_processing_object(0).write_summary_outputs(which="SWMM", model_type="tritonswmm")
        assert set(xr.open_zarr(_node_store(a, 0), consolidated=False).data_vars) == before, repr(blank)


def test_a_zarr_prune_inside_a_scenario_folder_leaves_the_disk_use_sentinel_equal_to_a_fresh_walk(tmp_path):
    """Property: pruning a store that lives in a scenario folder, with ``scope_dir`` given, removes the arrays
    through the toolkit's deletion tool, so the folder's ``_du.json`` afterwards equals the bytes a fresh walk of
    the folder measures, and the store holds exactly the kept and static variables.

    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the arrays without telling the sentinel (it
    would keep reading the pre-prune size, so a capacity estimate built on it counts bytes already freed), or
    accounting a different byte count than was removed.
    Reachability: the folder is seeded with a sentinel by ``compute_and_write_scope_sentinel`` as a scenario is.
    Both-states anchor: asserts equality with an independent walk and that the sentinel moved. Must-fire control:
    without ``scope_dir`` the same prune leaves a stale sentinel, which shows the assertion can fail.
    Killed by: not routing through the tool. A second correct implementation re-walks the folder after the prune;
    the equality would still hold, it would only cost more.
    """
    from hhemt.du_sentinels import compute_and_write_scope_sentinel, read_du_sentinel

    sel = _sel()

    def scenario(name: str) -> tuple[Path, Path]:
        folder = tmp_path / name
        folder.mkdir()
        big = _node_ds()
        big["head"] = big["head"].expand_dims(extra=range(40)).transpose("node_id", "date_time", "extra")
        store = _write_zarr(folder / "node.zarr", big)
        compute_and_write_scope_sentinel(folder, scope="scenario")
        return folder, store

    def recorded(folder: Path) -> int:
        return int(read_du_sentinel(folder / "_status" / "_du.json")["disk_utilization_bytes"])

    folder, store = scenario("scen_a")
    before = recorded(folder)
    sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"], scope_dir=folder)
    after = recorded(folder)
    assert after < before, "the sentinel must have lost the removed bytes"
    compute_and_write_scope_sentinel(folder, scope="scenario")  # an independent walk, overwriting the record
    # `after` was written by the tool's decrement; a fresh walk differs only by the sentinel file itself changing
    # size, so compare with that slack and nothing more.
    assert abs(recorded(folder) - after) <= 4096
    assert {v for v in _open(store).data_vars if "date_time" in _open(store)[v].dims} == {"depth", "mystery_series"}
    assert "is_coupling_node" in _open(store).data_vars

    stale_folder, stale_store = scenario("scen_b")
    stale_before = recorded(stale_folder)
    sel.prune_timeseries_store(stale_store, "swmm_node", "tritonswmm", ["depth"])
    assert recorded(stale_folder) == stale_before, "control: without scope_dir nothing tells the sentinel"
