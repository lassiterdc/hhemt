"""Tests for time-series variable selection that the chunk's own file does not reach.

The domain and the two sets are pinned against literal tables written from the measured variable lists, the domain
is checked against the variables the committed golden stores really hold, and the prune is driven in the states
the chunk's rows skip: summaries absent when the prune runs, a prune that fails after the summaries were written,
an interrupted prune followed by a wider keep-list, a keep-list changed after the consolidated store exists, a list
that names a variable only one of two enabled families writes, and a NetCDF prune checked for what it keeps.
No solver is compiled or run.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_NODE_COMMON = {"depth", "head", "inflow_flow", "flooding", "volume", "lateral_inflow", "pollutant_conc"}
_LINK_COMMON = {"flow", "velocity", "link_depth", "link_volume", "link_pollutant_conc"}
_GRIDDED = {"flood_depth", "max_flood_depth", "unit_discharge_x", "unit_discharge_y"}

# (kind, family) -> (domain, default), written out from the measured variable table and the developer's rulings
# for the default set, not computed from the module.
_EXPECTED = {
    ("swmm_node", "tritonswmm"): (
        _NODE_COMMON | {"coupling_exchange_flow"},
        _NODE_COMMON - {"flooding"} | {"coupling_exchange_flow"},
    ),
    ("swmm_node", "swmm_only"): (_NODE_COMMON, _NODE_COMMON),
    ("swmm_link", "tritonswmm"): (_LINK_COMMON | {"capacity_setting"}, _LINK_COMMON | {"capacity_setting"}),
    ("swmm_link", "swmm_only"): (_LINK_COMMON | {"capacity"}, _LINK_COMMON | {"capacity"}),
    ("triton_gridded", "tritonswmm"): (_GRIDDED, {"flood_depth"}),
    ("triton_gridded", "triton_only"): (_GRIDDED, {"flood_depth"}),
}


def _sel():
    from hhemt import timeseries_selection

    return timeseries_selection


@pytest.mark.parametrize("pair", sorted(_EXPECTED))
def test_domain_and_default_equal_the_measured_tables_for_every_kind_and_family(pair):
    """Property: for each of the six (kind, family) pairs the domain, `"all"`, `None` and `"default"` equal the tables
    written out from the measured variable lists and the developer's default rulings.

    Input class: all six pairs, including the two that differ only in family (gridded for the coupled and the TRITON
    only model) and the link pair whose capacity series is named differently per family. Plausible wrong
    implementations: a variable missing from or added to a domain, a default that drops or keeps the wrong series
    (the node default of the standalone model dropping `flooding`, the link default dropping `velocity`), or the
    capacity name swapped between families.
    Killed by: any such edit. Passes under a second correct implementation that derives the sets from a different
    table layout.
    """
    domain, default = _EXPECTED[pair]
    kind, family = pair
    sel = _sel()
    assert set(sel.DOMAIN[pair]) == domain
    assert set(sel.resolve_keep(kind, family, None)) == domain
    assert set(sel.resolve_keep(kind, family, "all")) == domain
    assert set(sel.resolve_keep(kind, family, "default")) == default
    assert len(sel.resolve_keep(kind, family, "default")) == len(default), "a name is listed twice"


def test_the_domain_covers_every_time_variant_variable_the_golden_stores_hold(golden_extractor):
    """Property: every variable with the kind's time dimension in the committed golden node, link and gridded stores
    is in the coupled domain of its kind, and the time dimension the module names for the kind is the one those
    stores use.

    Input class: the real stores of a coupled scenario, which is the population the domain is a table of. Plausible
    wrong implementations: a time dimension named wrongly for the gridded kind (every gridded variable then reads as
    static and nothing is ever pruned), or a series a writer produces that no table names (it can be neither
    selected nor pruned). Must-fire control: the time-variant sets are non-empty, so a wrong time dimension that
    finds none fails the assertion that they are.
    Killed by: a wrong `TIME_DIM_BY_KIND` entry or a missing domain name. Passes under a second correct implementation
    that finds the time dimension from the store.
    """
    from hhemt.scenario import TRITONSWMM_scenario

    sel = _sel()
    a = construct_golden_analysis(golden_extractor("multisim"))
    paths = TRITONSWMM_scenario(0, a).scen_paths
    stores = {
        "swmm_node": paths.output_tritonswmm_node_time_series,
        "swmm_link": paths.output_tritonswmm_link_time_series,
        "triton_gridded": paths.output_tritonswmm_triton_timeseries,
    }
    for kind, path in stores.items():
        ds = xr.open_zarr(path, consolidated=False)
        time_variant = {v for v in ds.data_vars if sel.TIME_DIM_BY_KIND[kind] in ds[v].dims}
        assert time_variant, f"{kind}: no variable carries {sel.TIME_DIM_BY_KIND[kind]}; dims are {dict(ds.sizes)}"
        assert time_variant <= set(sel.DOMAIN[(kind, "tritonswmm")]), (kind, sorted(time_variant))


def _two_family_cfg_system():
    return SimpleNamespace(toggle_tritonswmm_model=True, toggle_swmm_model=True, toggle_triton_model=False)


def _cfg_analysis(**keep):
    values = {
        "timeseries_keep_swmm_node": None,
        "timeseries_keep_swmm_link": None,
        "timeseries_keep_triton_gridded": None,
    }
    values.update(keep)
    return SimpleNamespace(**values)


def test_a_list_the_preflight_admits_for_two_enabled_families_does_not_crash_the_fingerprint():
    """Property: with the coupled and the standalone SWMM model both enabled, a link keep-list naming `capacity_setting`
    (a coupled-only series) passes preflight, and the consolidation fingerprint is then computed without raising.

    Input class: a list drawn from the union of the enabled families' domains, which is exactly what preflight
    admits. Plausible wrong implementation (the present one): preflight admits the name because one enabled family
    writes it, and the fingerprint resolves the same list against each family separately, so the standalone family's
    resolution raises `ValueError`. Both configs pass preflight and the run then fails at the first consolidation.
    Must-fire control: the same list with only the coupled family enabled computes.
    Killed by: resolving a family's list without first restricting it to that family's domain. Passes under a second
    correct implementation that drops a name the family does not write from that family's resolved list.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.validation import ValidationResult, _validate_timeseries_keep

    cfg_analysis = _cfg_analysis(timeseries_keep_swmm_link=["flow", "capacity_setting"])
    result = ValidationResult(context="test")
    _validate_timeseries_keep(_two_family_cfg_system(), cfg_analysis, result)
    assert not result.errors, "precondition: preflight must admit the list"

    def fingerprint_keep(models):
        """The resolved keep-lists for a stand-in analysis with the given enabled model types."""
        stub = pp.__new__(pp)
        stub._analysis = SimpleNamespace(cfg_analysis=cfg_analysis, _get_enabled_model_types=lambda: set(models))
        return stub._timeseries_keep_for_fingerprint()

    assert fingerprint_keep({"tritonswmm"})["swmm_link/tritonswmm"] == ["capacity_setting", "flow"]
    both = fingerprint_keep({"tritonswmm", "swmm"})
    assert "capacity_setting" not in both["swmm_link/swmm_only"] and "flow" in both["swmm_link/swmm_only"]


def test_a_list_the_preflight_admits_for_two_enabled_families_does_not_crash_the_standalone_prune(tmp_path):
    """Property: with the coupled and the standalone SWMM model both enabled, pruning the standalone model's link store
    with a keep-list that names `capacity_setting` keeps `flow` and removes the other domain series, instead of
    raising.

    Input class: the same union-admitted list as the fingerprint test, at the prune that runs after each standalone
    scenario's summaries. Plausible wrong implementation: resolving the list against the standalone family's domain
    raises `ValueError` from inside processing, after the summaries were written.
    Killed by: not restricting the list to the family's domain. Passes under a second correct implementation that
    resolves the list once over the union and prunes only what the family's domain holds.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as pp

    store = tmp_path / "link.zarr"
    n = 4
    ds = xr.Dataset(
        {
            "flow": (("link_id", "date_time"), np.ones((2, n))),
            "velocity": (("link_id", "date_time"), np.ones((2, n))),
            "capacity": (("link_id", "date_time"), np.ones((2, n))),
            "max_flow": (("link_id",), np.ones(2)),
        },
        coords={"link_id": ["a", "b"], "date_time": np.arange(n)},
    )
    ds.to_zarr(store, mode="w")
    proc = pp.__new__(pp)
    proc._analysis = SimpleNamespace(cfg_analysis=_cfg_analysis(timeseries_keep_swmm_link=["flow", "capacity_setting"]))
    proc.scen_paths = SimpleNamespace(output_swmm_only_link_time_series=store)
    proc._prune_timeseries_stores("swmm", "SWMM")
    kept = xr.open_zarr(store, consolidated=True)
    assert {v for v in kept.data_vars if "date_time" in kept[v].dims} == {"flow"}
    assert "max_flow" in kept.data_vars


def _golden(golden_extractor, *, extra: str = ""):
    """Golden multi-sim analysis with consolidation on, a coupling list per scenario, and extra config lines."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + extra)
    analysis = construct_golden_analysis(system_dir)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _paths(analysis, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, analysis).scen_paths


def test_the_summary_is_built_from_the_full_store_before_the_prune_removes_anything(golden_extractor):
    """Property: when the node and link summaries are absent, `write_summary_outputs` builds them with every statistic
    of the unpruned series and only then prunes the stores.

    Input class: a scenario whose summaries were deleted, with a node keep-list of `[depth]`. The chunk's own
    real-analysis row runs on summaries that already exist, so the summary build never runs in it and the order of
    prune and build is not observed. Plausible wrong implementations: the prune before the build (the build then
    refuses on the pruned store, or in a variant that skips the refusal returns fewer statistics with no error),
    or the prune between the node and link builds.
    Must-fire control: the rebuilt node summary has the variable set of the original summary, and the node store
    afterwards has `depth` as its only time-variant variable.
    Killed by: moving the call above the summary exports. Passes under a second correct implementation that prunes
    inside each export after its summary is written.
    """
    a = _golden(golden_extractor, extra="timeseries_keep_swmm_node: [depth]\n")
    proc = a._retrieve_sim_run_processing_object(0)
    paths = _paths(a, 0)
    original = set(xr.open_zarr(paths.output_tritonswmm_node_summary, consolidated=False).data_vars)
    for summary in (paths.output_tritonswmm_node_summary, paths.output_tritonswmm_link_summary):
        shutil.rmtree(summary)
    proc.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    rebuilt = set(xr.open_zarr(paths.output_tritonswmm_node_summary, consolidated=False).data_vars)
    assert rebuilt == original
    node = xr.open_zarr(paths.output_tritonswmm_node_time_series, consolidated=False)
    assert {v for v in node.data_vars if "date_time" in node[v].dims} == {"depth"}


def test_a_prune_that_fails_after_the_summaries_were_written_is_completed_by_the_next_call(
    golden_extractor, monkeypatch
):
    """Property: if the prune raises after the summaries are on disk, calling `write_summary_outputs` again leaves the
    summaries byte-identical and completes the prune, so the store ends pruned exactly as an uninterrupted call
    leaves it.

    Input class: a failure between the summary write and the prune (here an `OSError` raised by the first call to the
    prune), then a second call whose summary exports return early because their summaries exist. Plausible wrong
    implementation: running the prune only on the path that built a summary, so the second call prunes nothing and
    the unkept variables stay on disk for good with no error.
    Killed by: moving the prune into the build branch of the exports. Passes under a second correct implementation
    that prunes in a finally block.
    """
    import hhemt.timeseries_selection as sel

    a = _golden(golden_extractor, extra="timeseries_keep_swmm_node: [depth]\n")
    proc = a._retrieve_sim_run_processing_object(0)
    paths = _paths(a, 0)
    shutil.rmtree(paths.output_tritonswmm_node_summary)
    shutil.rmtree(paths.output_tritonswmm_link_summary)
    real = sel.prune_timeseries_store
    calls = []

    def failing_once(*args, **kwargs):
        """Raise on the first call, delegate afterwards."""
        calls.append(args[1])
        if len(calls) == 1:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(sel, "prune_timeseries_store", failing_once)
    with pytest.raises(OSError):
        proc.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    node_summary = xr.open_zarr(paths.output_tritonswmm_node_summary, consolidated=False).load()
    node = xr.open_zarr(paths.output_tritonswmm_node_time_series, consolidated=False)
    assert "head" in node.data_vars, "precondition: nothing was pruned before the failure"
    a._retrieve_sim_run_processing_object(0).write_summary_outputs(which="SWMM", model_type="tritonswmm")
    after = xr.open_zarr(paths.output_tritonswmm_node_time_series, consolidated=False)
    assert {v for v in after.data_vars if "date_time" in after[v].dims} == {"depth"}
    xr.testing.assert_identical(
        xr.open_zarr(paths.output_tritonswmm_node_summary, consolidated=False).load(), node_summary
    )


def test_the_pruned_record_names_only_variables_that_are_absent_after_an_interrupted_prune_and_a_wider_list(
    tmp_path, monkeypatch
):
    """Property: after a prune interrupted between its attribute write and its deletes, a second prune with a keep-list
    that keeps one of the variables the attribute named leaves a record that names only variables absent from the
    store.

    Input class: an interrupted prune, which the attribute makes recognisable, followed by the user widening the
    list before the
    re-run. Plausible wrong implementation (the present one): the record is the union of the old record and the new
    removals, so it still names a variable the new list keeps and which is on disk. A reader then reads a present
    variable as pruned, and the summary rebuild guard refuses a store that holds every series.
    Must-fire control: the record also names a variable that really was removed.
    Killed by: a union that is not intersected with the variables the store lacks. Passes under a second correct
    implementation that rewrites the record from the store's actual variables after the deletes.
    """
    import hhemt.timeseries_selection as sel

    store = tmp_path / "node.zarr"
    n = 4
    xr.Dataset(
        {name: (("node_id", "date_time"), np.ones((2, n))) for name in ("depth", "head", "flooding", "inflow_flow")},
        coords={"node_id": ["a", "b"], "date_time": np.arange(n)},
    ).to_zarr(store, mode="w")
    real_delete = sel._delete_zarr_array
    count = []

    def interrupt_on_second(group, name):
        """Delete the first array, then raise on the second."""
        count.append(name)
        if len(count) == 2:
            raise OSError("killed")
        real_delete(group, name)

    monkeypatch.setattr(sel, "_delete_zarr_array", interrupt_on_second)
    with pytest.raises(OSError):
        sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"])
    monkeypatch.undo()
    present = {v for v in xr.open_zarr(store, consolidated=False).data_vars}
    survivor = sorted({"head", "flooding", "inflow_flow"} & present)[0]
    removed = sorted({"head", "flooding", "inflow_flow"} - present)
    assert removed, "precondition: the interrupted call deleted one array"
    sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth", survivor])
    final = {v for v in xr.open_zarr(store, consolidated=False).data_vars}
    recorded = set(sel.pruned_variables(store))
    assert survivor in final
    assert recorded <= (set(("depth", "head", "flooding", "inflow_flow")) - final), (recorded, sorted(final))
    assert set(removed) <= recorded


def test_a_zarr_prune_changes_nothing_but_the_removed_arrays_and_the_record(golden_extractor, tmp_path):
    """Property: pruning a copy of a real scenario store leaves every kept and static variable identical (values,
    dtype, attributes, coordinates), lists only the kept variables in the consolidated index, and changes the root
    attributes only by adding the record.

    Input class: a real coupled node store, with a keep-list of `[depth, flooding]`. Plausible wrong implementations:
    a prune that rewrites the store (dtype or attribute loss), or one that leaves the deleted arrays in the
    consolidated index (an indexed open then fails or lists them).
    Killed by: rewriting through a copy or skipping the re-consolidation. Passes under a second correct
    implementation that deletes through the zarr API in any order.
    """
    a = construct_golden_analysis(golden_extractor("multisim"))
    source = _paths(a, 0).output_tritonswmm_node_time_series
    copy = tmp_path / "node.zarr"
    shutil.copytree(source, copy)
    from hhemt.utils import consolidate_zarr_store, zarr_store_has_index

    consolidate_zarr_store(copy)
    assert zarr_store_has_index(copy), "precondition: the copy must carry an index for the prune to keep current"
    before = xr.open_zarr(copy, consolidated=False).load()
    removed = _sel().prune_timeseries_store(copy, "swmm_node", "tritonswmm", ["depth", "flooding"])
    assert set(removed) == {"head", "inflow_flow"}
    after = xr.open_zarr(copy, consolidated=True).load()
    expected = before.drop_vars(list(removed))
    after.attrs.pop("timeseries_pruned_variables")
    xr.testing.assert_identical(after, expected)
    for name in after.data_vars:
        assert after[name].dtype == before[name].dtype, name


def _nc_store(path):
    from hhemt.utils import write_netcdf

    n = 4
    ds = xr.Dataset(
        {
            "depth": (("node_id", "date_time"), np.arange(8, dtype="float32").reshape(2, n), {"units": "m"}),
            "head": (("node_id", "date_time"), np.ones((2, n), dtype="float32"), {"units": "m"}),
            "is_coupling_node": (("node_id",), np.array([1, 0], dtype="int8"), {"flag_meanings": "a b"}),
            "max_flow": (("node_id",), np.array([0.5, 1.5]), {"units": "m3 s-1"}),
        },
        coords={
            "node_id": ["a", "b"],
            "date_time": np.array(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
        },
    )
    write_netcdf(ds, path)
    return path


def test_a_netcdf_prune_keeps_dtype_attributes_and_time_axis_of_what_it_keeps(tmp_path):
    """Property: after a NetCDF prune, every kept time-variant and static variable has the same values, dtype, units and
    flag attributes as before, the time coordinate is unchanged, and an unpruned keep-list leaves the file's bytes
    alone.

    Input class: an `nc` store with a float32 series, an int8 flag, a float64 static variable and a datetime axis,
    pruned to `[depth]`. Plausible wrong implementations: the rewrite widens float32 to float64 or drops variable
    attributes (the kept series then differs from what the summary was computed from), re-encodes the time axis, or
    rewrites the file when nothing is removed.
    Must-fire control: the `head` series is removed and the keep-everything call is the only one that must leave the
    bytes identical.
    Killed by: any lossy rewrite. Passes under a second correct implementation that deletes variables in place.
    """
    import hashlib

    sel = _sel()
    path = _nc_store(tmp_path / "node.nc")
    before = xr.open_dataset(path).load()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sel.prune_timeseries_store(path, "swmm_node", "tritonswmm", sel.DOMAIN[("swmm_node", "tritonswmm")]) == ()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "keeping everything must not rewrite the file"
    assert sel.prune_timeseries_store(path, "swmm_node", "tritonswmm", ["depth"]) == ("head",)
    after = xr.open_dataset(path).load()
    assert "head" not in after.data_vars
    for name in ("depth", "is_coupling_node", "max_flow"):
        assert after[name].dtype == before[name].dtype, name
        assert after[name].attrs == before[name].attrs, name
        np.testing.assert_array_equal(after[name].values, before[name].values)
    np.testing.assert_array_equal(after["date_time"].values, before["date_time"].values)


def test_a_keep_list_changed_after_the_consolidated_store_exists_is_refused_and_leaves_the_store_unchanged(
    golden_extractor,
):
    """Property: consolidating, then pruning the scenario stores with a narrower keep-list and consolidating again,
    raises `ProcessingError` and leaves every stored variable and value as it was.

    Input class: the keep-list changed after `analysis_timeseries.zarr` holds the full variable set. Plausible
    wrong implementations: a silent pass that leaves the store holding variables the new list drops while the
    fingerprint says the tree matches, or a rewrite that deletes arrays from the only consolidated copy.
    Must-fire control: the stored `flooding` series is present and equal before and after the refused pass.
    Killed by: skipping the variable-set comparison of the read-back, or repairing by deleting store arrays. Passes
    under a second correct implementation that raises a subclass of `ProcessingError`.
    """
    from hhemt.exceptions import ProcessingError

    a = _golden(golden_extractor, extra="")
    a.process.consolidate_timeseries()
    store = a.analysis_paths.analysis_timeseries_zarr
    group = "tritonswmm/swmm_node_timeseries"
    before = xr.open_zarr(store, group=group, consolidated=False).load()
    a.cfg_analysis.timeseries_keep_swmm_node = ["depth"]
    for i in a.df_sims.index:
        a._retrieve_sim_run_processing_object(i).write_summary_outputs(which="SWMM", model_type="tritonswmm")
    with pytest.raises(ProcessingError):
        a.process.consolidate_timeseries()
    xr.testing.assert_identical(xr.open_zarr(store, group=group, consolidated=False).load(), before)
    assert "flooding" in before.data_vars


def test_the_prune_wiring_reads_each_kind_from_its_own_field(golden_extractor):
    """Property: with a keep-list only on the link field, `_prune_timeseries_stores` prunes the link store and
    leaves the node store and the gridded store untouched, and with a list only on the gridded field it prunes the
    gridded
    store alone.

    Input class: one field set at a time on a real scenario. Plausible wrong implementations: a kind reading another
    kind's field (a link list applied to the node store removes nothing in the domain but a node list applied to
    the gridded store would remove its series), or all kinds pruned by any field.
    Killed by: swapping the field names. Passes under a second correct implementation that reads one dict of fields.
    """
    a = _golden(golden_extractor, extra="")
    proc = a._retrieve_sim_run_processing_object(0)
    paths = _paths(a, 0)

    def variables(path):
        """The time-variant variable names of the zarr store at `path`."""
        ds = xr.open_zarr(path, consolidated=False)
        return {v for v in ds.data_vars if v in ds and ({"date_time", "timestep"} & set(ds[v].dims))}

    node0, link0, grid0 = (
        variables(p)
        for p in (
            paths.output_tritonswmm_node_time_series,
            paths.output_tritonswmm_link_time_series,
            paths.output_tritonswmm_triton_timeseries,
        )
    )
    a.cfg_analysis.timeseries_keep_swmm_link = ["flow"]
    proc._prune_timeseries_stores("tritonswmm", "both")
    assert variables(paths.output_tritonswmm_link_time_series) == {"flow"}
    assert variables(paths.output_tritonswmm_node_time_series) == node0
    assert variables(paths.output_tritonswmm_triton_timeseries) == grid0
    a.cfg_analysis.timeseries_keep_triton_gridded = ["flood_depth"]
    proc._prune_timeseries_stores("tritonswmm", "both")
    assert variables(paths.output_tritonswmm_triton_timeseries) == {"flood_depth"}
    assert variables(paths.output_tritonswmm_node_time_series) == node0
    assert link0 > {"flow"}


def test_a_padded_set_name_and_a_blank_list_element_are_judged_alike_by_preflight_and_resolution():
    """Property: a set name padded with spaces is accepted by both preflight and `resolve_keep` and means the set, and a
    list holding a blank element is refused by both, so neither path turns it into a selection of nothing.

    Input class: `" default "` and `[" "]` for the node field, which sit on the edge of the blank rule. Plausible
    wrong implementations: preflight stripping a name that resolution does not (the run passes preflight and then
    raises), or resolution reading a blank element as an empty list and pruning every series.
    Must-fire control: the unpadded `"default"` is accepted by both, and the refusals name the offending value.
    Killed by: stripping in one place only. Passes under a second correct implementation that normalises once in
    the config model.
    """
    from hhemt.validation import ValidationResult, _validate_timeseries_keep

    sel = _sel()
    system = SimpleNamespace(toggle_tritonswmm_model=True, toggle_swmm_model=False, toggle_triton_model=False)

    def refused(value):
        """Whether preflight refuses `value` for the node field, and whether resolution does."""
        result = ValidationResult(context="test")
        _validate_timeseries_keep(system, _cfg_analysis(timeseries_keep_swmm_node=value), result)
        try:
            sel.resolve_keep("swmm_node", "tritonswmm", value)
            resolved_refuses = False
        except ValueError:
            resolved_refuses = True
        return bool(result.errors), resolved_refuses

    assert refused("default") == (False, False)
    assert refused(" default ") == (False, False)
    assert sel.resolve_keep("swmm_node", "tritonswmm", " default ") == sel.resolve_keep(
        "swmm_node", "tritonswmm", "default"
    )
    assert refused([" "]) == (True, True)
