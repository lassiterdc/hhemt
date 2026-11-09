"""Every model family that writes series is consolidated and cleared like the coupled family.

No solver and no Snakemake. The committed multi-sim golden (a finished analysis tree written by the independent
generator, three events, the coupled, SWMM-only and TRITON-only arms, every per-scenario series store on a 10 s step)
is extracted per test and its system configuration is edited so the analysis enables only the family a test is about,
or all three where an arm compares families. Each arm names its control: the input class that makes a broken
implementation read as correct.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_SWMM_NODE = "swmm_only/swmm_node/timeseries"
_SWMM_LINK = "swmm_only/swmm_link/timeseries"
_TRI_ONLY = "triton_only/triton/timeseries"
_CPL_NODE = "tritonswmm/swmm_node/timeseries"
_CPL_LINK = "tritonswmm/swmm_link/timeseries"
_CPL_TRI = "tritonswmm/triton/timeseries"
_ALL_GROUPS = (_CPL_NODE, _CPL_LINK, _CPL_TRI, _SWMM_NODE, _SWMM_LINK, _TRI_ONLY)
# What the coupled model attaches and the SWMM-only model never does (D3, D5, D18): the flag, the exchange series and
# the eight exchange statistics.
_COUPLED_ONLY = {
    "is_coupling_node",
    "coupling_exchange_flow",
    "coupling_exchange_peak_to_network",
    "coupling_exchange_peak_to_surface",
    "coupling_exchange_volume_to_network",
    "coupling_exchange_volume_to_surface",
    "coupling_exchange_volume_net",
    "coupling_exchange_time_of_peak_to_network",
    "coupling_exchange_time_of_peak_to_surface",
    "coupling_exchange_duration_to_surface",
}
_FAMILY_TOGGLES = {
    "tritonswmm": "toggle_tritonswmm_model",
    "swmm": "toggle_swmm_model",
    "triton": "toggle_triton_model",
}


def _ready(golden_extractor, *, families, clear: bool, consolidate: bool = True):
    """Golden analysis enabling exactly ``families``, with series consolidation and the clearing switch as given.

    A coupling-node list is written beside every scenario, because the coupled node store of the golden carries no
    ``is_coupling_node`` flag and the consolidation derives it from that list.
    """
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    system_cfg = system_dir / "system_config.yaml"
    text = system_cfg.read_text()
    for family, key in _FAMILY_TOGGLES.items():
        text = re.sub(rf"^{key}: .*$", f"{key}: {'true' if family in families else 'false'}", text, flags=re.M)
    system_cfg.write_text(text)
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + f"toggle_consolidate_timeseries: {'true' if consolidate else 'false'}\n"
        + f"toggle_clear_scenario_timeseries_after_consolidation: {'true' if clear else 'false'}\n"
    )
    analysis = construct_golden_analysis(system_dir)
    if "tritonswmm" in families:
        for event_iloc in analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, analysis)
            ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
            nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
            nodes_file.parent.mkdir(parents=True, exist_ok=True)
            nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _scen(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _store(a) -> Path:
    return Path(a.analysis_paths.analysis_datatree_zarr)


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(_store(a), group=name, consolidated=False)


def _group_names(a) -> set[str]:
    tree = xr.open_datatree(_store(a), engine="zarr", consolidated=False)
    return {g.lstrip("/") for g in tree.groups}


def _own(a, event_iloc: int, attr: str) -> Path:
    return Path(getattr(_scen(a, event_iloc).scen_paths, attr))


def _flag(a, event_iloc: int, model: str, name: str):
    field = getattr(_scen(a, event_iloc).get_log(model), name, None)
    return None if field is None else field.get()


def _source_equals_slice(a, event_iloc: int, attr: str, group: str, variable: str) -> bool:
    """Whether the stored slice of ``variable`` equals the scenario's own store, NaN-aware, on the source's times."""
    source = xr.open_zarr(_own(a, event_iloc, attr), consolidated=False).load()
    stored = _group(a, group).load()
    got = stored[variable].sel(event_iloc=event_iloc)
    want = source[variable]
    if "date_time" in want.dims:
        want = want.rename(date_time="elapsed_s")
        got = got.transpose(*want.dims)
        return bool(
            np.array_equal(got.isel(elapsed_s=slice(0, want.sizes["elapsed_s"])).values, want.values, equal_nan=True)
        )
    seconds = np.rint(source["timestep"].values * 60.0).astype("int64")
    placed = got.sel(elapsed_s=seconds)
    return bool(np.array_equal(placed.values, want.values, equal_nan=True))


def test_a_swmm_only_analysis_consolidates_its_node_and_link_series_without_a_coupling_flag(golden_extractor):
    """Property: an analysis that enables only the SWMM-only model writes its node and link series to
    ``swmm_only/swmm_node/timeseries`` and ``swmm_only/swmm_link/timeseries`` on the shared ``elapsed_s`` axis, every
    event present, each variable equal to the scenario's own store, and the node group carries no
    ``is_coupling_node`` and no ``coupling_exchange_flow``; no coupled group is written.

    Class: NEW CAPABILITY. Plausible wrong implementations: a node group that keeps the coupled flag because the slice
    builder derives it for every node store (a SWMM-only scenario has no coupling-node list, so the derivation
    raises), a SWMM-only family left out of the mode table (the groups are absent), or the SWMM-only series written
    under the coupled group name. Reachability: the real ``consolidate_timeseries`` over the golden with the coupled
    and TRITON-only arms switched off. Both-states anchor: ``test_the_coupled_node_group_keeps_its_flag_beside_the_
    swmm_only_group`` below shows the coupled group of the same golden carrying the flag. Control: see
    ``test_with_consolidation_off_no_family_writes_a_timeseries_branch``.
    Killed by: those. Passes under a second correct implementation that writes the groups by a region write.
    """
    a = _ready(golden_extractor, families=("swmm",), clear=False)
    a.process.consolidate_timeseries()
    names = _group_names(a)
    assert {_SWMM_NODE, _SWMM_LINK} <= names, names
    assert not names & {_CPL_NODE, _CPL_LINK, _CPL_TRI, _TRI_ONLY}, "a family the analysis does not run was written"
    node, link = _group(a, _SWMM_NODE), _group(a, _SWMM_LINK)
    assert [int(v) for v in node["event_iloc"].values] == [0, 1, 2]
    assert not set(node.data_vars) & _COUPLED_ONLY, sorted(node.data_vars)
    assert set(node.data_vars) >= {"depth", "head", "flooding", "inflow_flow"}, sorted(node.data_vars)
    assert "elapsed_s" in node.dims and "elapsed_s" in link.dims
    for event_iloc in (0, 1, 2):
        for variable in node.data_vars:
            assert _source_equals_slice(a, event_iloc, "output_swmm_only_node_time_series", _SWMM_NODE, variable), (
                event_iloc,
                variable,
            )
        for variable in link.data_vars:
            assert _source_equals_slice(a, event_iloc, "output_swmm_only_link_time_series", _SWMM_LINK, variable), (
                event_iloc,
                variable,
            )


def test_the_coupled_node_group_keeps_its_flag_beside_the_swmm_only_group(golden_extractor):
    """Property (anchor of the arm above): with the coupled and SWMM-only arms both enabled, the coupled node group
    carries ``is_coupling_node`` and the SWMM-only node group of the SAME consolidation does not.

    Class: NEW CAPABILITY. Plausible wrong implementation: deciding the flag by node-store shape instead of by family,
    which gives both groups or neither the flag. Reachability: one real ``consolidate_timeseries`` over both arms.
    Killed by: either shape. Passes under a second correct implementation that stores the flag as a coordinate, if
    the variable-set comparison below is made on variables and coordinates together.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm"), clear=False)
    a.process.consolidate_timeseries()
    assert "is_coupling_node" in _group(a, _CPL_NODE).data_vars
    assert "is_coupling_node" not in _group(a, _SWMM_NODE).data_vars


def test_with_consolidation_off_no_family_writes_a_timeseries_branch(golden_extractor):
    """Property (control of the SWMM-only arm): the same golden with consolidation off consolidates its summaries and
    writes no ``timeseries`` group for any family, the SWMM-only summary groups being present.

    Class: NEW CAPABILITY. Plausible wrong implementation: a new family mode that is consolidated whether or not the
    toggle is on. Reachability: ``consolidate_to_datatree`` over all three arms with ``toggle_consolidate_timeseries``
    false. Both-states anchor: the arm above writes the groups with the toggle on. Killed by: an unconditional
    SWMM-only or TRITON-only write.
    Passes under a second correct implementation that gates each family's write on the same toggle in a different place.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=False, consolidate=False)
    a.process.consolidate_to_datatree()
    names = _group_names(a)
    assert {"swmm_only/swmm_node/summary", "swmm_only/swmm_link/summary", "triton_only/triton/summary"} <= names
    assert not [n for n in names if n.endswith("/timeseries")], sorted(n for n in names if n.endswith("/timeseries"))


def test_the_triton_only_gridded_series_is_written_in_blocks_and_read_back_block_by_block(
    golden_extractor, monkeypatch
):
    """Property: the TRITON-only gridded stores land on ``triton_only/triton/timeseries`` chunked by the block size
    (shrunk to four steps so a slice spans several chunks), with every explicit load of a Dataset or DataArray
    refused, each variable equal to the scenario's own store on the source's own times, and the coupled gridded group
    absent from this TRITON-only analysis.

    Class: NEW CAPABILITY. Plausible wrong implementations: the whole-slice ``.load()`` the SWMM builder uses (refused
    here), a TRITON-only family left out of the mode table (the group is absent), or a chunk shape that ignores the
    block size. Reachability: the real ``consolidate_timeseries`` over the golden with only the TRITON-only arm
    enabled. Both-states anchor: the control below changes one value after the write and the same read-back fails.
    Killed by: those. Passes under a second correct implementation that streams through another writer, if its
    chunks along ``elapsed_s`` are no larger than the stated block.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    monkeypatch.setattr(cls, "_TIMESERIES_BLOCK_BYTES", 400)
    a = _ready(golden_extractor, families=("triton",), clear=False)

    def refuse(self, *args, **kwargs):
        raise AssertionError("the TRITON-only consolidation loaded a whole slice")

    with monkeypatch.context() as patched:
        for owner in (xr.Dataset, xr.DataArray):
            for method in ("load", "compute"):
                patched.setattr(owner, method, refuse)
        a.process.consolidate_timeseries()
    names = _group_names(a)
    assert _TRI_ONLY in names and _CPL_TRI not in names, names
    stored = _group(a, _TRI_ONLY)
    assert [int(v) for v in stored["event_iloc"].values] == [0, 1, 2]
    chunks = stored["flood_depth"].encoding["chunks"]
    dims = list(stored["flood_depth"].dims)
    assert chunks[dims.index("elapsed_s")] < stored.sizes["elapsed_s"], (chunks, stored.sizes["elapsed_s"])
    for event_iloc in (0, 1, 2):
        for variable in stored.data_vars:
            assert _source_equals_slice(a, event_iloc, "output_triton_only_timeseries", _TRI_ONLY, variable)


def test_one_value_changed_after_the_triton_only_write_fails_the_readback_and_keeps_that_store(
    golden_extractor, monkeypatch
):
    """Property (must-fire control of the TRITON-only read-back): one value changed in the LAST block of the stored
    TRITON-only slice right after its write makes the read-back raise ``ProcessingError`` and keeps the scenario's
    TRITON-only store with its cleared record unset.

    Class: NEW CAPABILITY. Plausible wrong implementation: a TRITON-only clearing that deletes the store without a
    read-back, or a read-back that compares only the first block. Reachability: ``append_zarr`` is wrapped to
    overwrite one stored value through the zarr API right after the real append, block size shrunk to 4 steps.
    Both-states anchor: the arm above clears nothing (switch off) and the arm
    ``test_every_family_is_cleared_after_its_slices_pass`` clears with the same code unwrapped. Killed by: those.
    Passes under a second correct implementation that damages the stored slice through a region write instead of the
    zarr array handle.
    """
    import zarr

    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    monkeypatch.setattr(cls, "_TIMESERIES_BLOCK_BYTES", 400)
    a = _ready(golden_extractor, families=("triton",), clear=True)
    real_append = pa.append_zarr

    def damaging_append(ds, store, **kw):
        real_append(ds, store, **kw)
        if kw.get("group") == _TRI_ONLY and kw.get("append_dim") == "event_iloc":
            arr = zarr.open_group(store, path=_TRI_ONLY, mode="r+")["flood_depth"]
            arr[arr.shape[0] - 1, 10, ...] = 999.0

    monkeypatch.setattr(pa, "append_zarr", damaging_append)
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert _own(a, 1, "output_triton_only_timeseries").exists(), "the store whose slice was damaged must be kept"
    assert _flag(a, 1, "triton", "consolidated_triton_timeseries_cleared") is not True
    assert not _own(a, 0, "output_triton_only_timeseries").exists(), (
        "control: the event before it passed and was cleared"
    )


_STORES = {
    "tritonswmm": (
        ("output_tritonswmm_node_time_series", "consolidated_node_timeseries"),
        ("output_tritonswmm_link_time_series", "consolidated_link_timeseries"),
        ("output_tritonswmm_triton_timeseries", "consolidated_triton_timeseries"),
    ),
    "swmm": (
        ("output_swmm_only_node_time_series", "consolidated_node_timeseries"),
        ("output_swmm_only_link_time_series", "consolidated_link_timeseries"),
    ),
    "triton": (("output_triton_only_timeseries", "consolidated_triton_timeseries"),),
}


def test_every_family_is_cleared_after_its_slices_pass(golden_extractor):
    """Property: with the clearing switch on and all three arms enabled, every scenario's six series stores are
    deleted, each model's own log records its stores as cleared (the cleared field True, the pending record gone),
    and every group still holds every event.

    Class: NEW CAPABILITY. Plausible wrong implementations: a clearing that covers only the coupled family (the
    SWMM-only and TRITON-only stores survive), a cleared record written to the coupled log for every family (the
    SWMM-only and TRITON-only logs read unset), or a record that stays pending. Reachability: the real
    ``consolidate_timeseries``. Both-states anchor: ``test_a_failed_readback_keeps_the_failing_familys_store_and_
    leaves_the_verified_families_cleared`` below.
    Killed by: those. Passes under a second correct implementation that clears all families in one delete.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=True)
    a.process.consolidate_timeseries()
    for group in _ALL_GROUPS:
        assert [int(v) for v in _group(a, group)["event_iloc"].values] == [0, 1, 2], group
    for event_iloc in (0, 1, 2):
        for model, stores in _STORES.items():
            for attr, field in stores:
                assert not _own(a, event_iloc, attr).exists(), (event_iloc, model, attr)
                assert _flag(a, event_iloc, model, f"{field}_cleared") is True, (event_iloc, model, field)
                assert not _flag(a, event_iloc, model, f"{field}_clearing_pending"), (event_iloc, model, field)


def test_a_failed_readback_keeps_the_failing_familys_store_and_leaves_the_verified_families_cleared(
    golden_extractor, monkeypatch
):
    """Property (must-fire control of the clearing): when the read-back of the SWMM-only node group fails for event 0,
    the consolidation raises ``ProcessingError``, the SWMM-only stores of event 0 exist and their cleared records are
    not True, the coupled stores of event 0 (verified before it) are cleared, and the TRITON-only store (never
    reached) exists.

    Class: NEW CAPABILITY. Plausible wrong implementations: one clearing act for every family after the whole scenario
    (the verified coupled family would survive, which the first assertion pair separates from the next), or a
    clearing that does not wait for its own family's slices (the SWMM-only store would be gone). Reachability: the
    real consolidation with ``_stored_slice_equals`` answering False for one group. Killed by: those.
    Passes under a second correct implementation that processes the models in another order, provided the failing
    model's own stores are kept and no store is cleared before its model's slices pass.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=True)
    real = cls._stored_slice_equals

    def failing(store, group, expected, event_iloc):
        if group == _SWMM_NODE:
            return False
        return real(store, group, expected, event_iloc)

    monkeypatch.setattr(cls, "_stored_slice_equals", staticmethod(failing))
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    for attr, field in _STORES["swmm"]:
        assert _own(a, 0, attr).exists(), f"the failing family lost {attr}"
        assert _flag(a, 0, "swmm", f"{field}_cleared") is not True
    for attr, field in _STORES["tritonswmm"]:
        assert not _own(a, 0, attr).exists(), f"the verified coupled family kept {attr}"
        assert _flag(a, 0, "tritonswmm", f"{field}_cleared") is True
    assert _own(a, 0, "output_triton_only_timeseries").exists(), (
        "control: the family after the failure was never reached"
    )


def _proc(a, event_iloc: int, model: str):
    """A post-processing object over the scenario's real log for ``model`` and its real scenario paths."""
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    scen = _scen(a, event_iloc)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.log = scen.get_log(model)
    proc.scen_paths = scen.scen_paths
    return proc


def _forget_processing(proc) -> None:
    """Remove the log's own processed records, so the cleared field is the only evidence a predicate can read."""
    for name in ("TRITON_timeseries_written", "SWMM_node_timeseries_written", "SWMM_link_timeseries_written"):
        field = getattr(proc.log, name, None)
        if field is not None:
            field.set(False)
    proc.log.processing_log.outputs.clear()


def test_the_processed_state_checks_read_the_cleared_field_for_every_family(golden_extractor):
    """Property: after a clearing consolidation, a SWMM-only scenario whose node and link stores are gone reads as
    processed for both, a TRITON-only scenario whose gridded store is gone reads as processed, and the coupled
    scenario still does, each from its own model's cleared record alone (the log's processed records are removed
    first, so a predicate that reads them or the filesystem would answer False).

    Class: NEW CAPABILITY. Plausible wrong implementation: the cleared record honoured for the coupled model only,
    which re-derives a SWMM-only or TRITON-only scenario from raw outputs the clearing may already have removed.
    Reachability: the real consolidation, then the real predicates over the real logs. Must-fire control: with the
    cleared record set False and nothing else recorded, every predicate reads False, so a predicate that always
    answers True fails.
    Killed by: those. Passes under a second correct implementation that reads one combined record.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=True)
    a.process.consolidate_timeseries()
    swmm, triton, coupled = _proc(a, 1, "swmm"), _proc(a, 1, "triton"), _proc(a, 1, "tritonswmm")
    for proc in (swmm, triton, coupled):
        _forget_processing(proc)
    assert swmm._swmm_node_outputs_processed("swmm") is True
    assert swmm._swmm_link_outputs_processed("swmm") is True
    assert triton.TRITON_outputs_processed is True
    assert coupled._swmm_node_outputs_processed("tritonswmm") is True
    assert coupled.TRITON_outputs_processed is True
    # control: the record unset and nothing else recorded reads not processed
    for proc, names in (
        (swmm, ("consolidated_node_timeseries_cleared", "consolidated_link_timeseries_cleared")),
        (triton, ("consolidated_triton_timeseries_cleared",)),
    ):
        for name in names:
            getattr(proc.log, name).set(False)
    assert swmm._swmm_node_outputs_processed("swmm") is False
    assert swmm._swmm_link_outputs_processed("swmm") is False
    assert triton.TRITON_outputs_processed is False


def _gap(a) -> dict[str, set[str]]:
    """The variable-name differences between the coupled and the SWMM-only family, per kind and branch."""
    tree = xr.open_datatree(_store(a), engine="zarr", consolidated=False)
    out: dict[str, set[str]] = {}
    for kind in ("swmm_node", "swmm_link"):
        for branch in ("summary", "timeseries"):
            coupled = set(tree[f"tritonswmm/{kind}/{branch}"].to_dataset().data_vars)
            alone = set(tree[f"swmm_only/{kind}/{branch}"].to_dataset().data_vars)
            if kind == "swmm_link":
                alone = {re.sub(r"^capacity(?=$|_)", "capacity_setting", v) for v in alone}
            out[f"{kind}/{branch}: coupled only"] = coupled - alone
            out[f"{kind}/{branch}: swmm_only only"] = alone - coupled
    return out


def test_the_swmm_only_and_coupled_variable_sets_differ_exactly_by_the_coupled_only_set(golden_extractor):
    """Property: the SWMM-only and coupled node ``summary`` and ``timeseries`` variable sets differ exactly by the
    coupled-only set (``is_coupling_node``, ``coupling_exchange_flow``, the exchange statistics), and the link sets
    differ only by the capacity name (``capacity`` against ``capacity_setting``, with ``_max`` and ``_last``); the
    SWMM-only family has no variable the coupled family lacks except the five run-level ``runoff_continuity_*`` rows,
    which only the node ``summary`` group may carry (``swmm_only_node_vars()``: the coupled model routes no runoff).

    Class: NEW CAPABILITY. Plausible wrong implementation: a SWMM-only reader that drifts from the coupled one (one
    reader serves both families, so a variable added to one family's store alone is the realistic defect).
    Reachability: all three arms consolidated, the summaries by ``consolidate_to_datatree`` and the series by the
    clearing-off consolidation it runs. Must-fire control: ``is_coupling_node`` IS in the coupled-only difference of
    the node series, which proves the comparison saw both families; and one variable added to the SWMM-only node
    stores alone appears as a SWMM-only-only difference, which the assertion of this test would reject.
    Killed by: a reader drift in either direction. Passes under a second correct implementation with other
    coupled-only variables, provided they are in the stated set.
    """
    from tests.utils_for_testing import swmm_only_node_vars

    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=False)
    a.process.consolidate_to_datatree()
    gap = _gap(a)
    assert gap["swmm_node/timeseries: coupled only"] == {"is_coupling_node"}, "control: the comparison saw both"
    for key, names in gap.items():
        if key.endswith("coupled only"):
            assert names <= _COUPLED_ONLY, (key, names)
        else:
            allowed = swmm_only_node_vars() if key == "swmm_node/summary: swmm_only only" else set()
            assert names <= allowed, (key, names)
    assert gap["swmm_link/summary: coupled only"] == set() and gap["swmm_link/timeseries: coupled only"] == set()


def test_a_variable_added_to_one_family_reader_alone_is_a_difference_the_parity_assertion_rejects(golden_extractor):
    """Property (must-fire control of the parity arm): one time-variant variable added to every SWMM-only node store
    appears in the SWMM-only group and in the SWMM-only-only difference, so the parity assertion would fail.

    Class: NEW CAPABILITY. Plausible wrong implementation: a comparison that cannot see a one-sided variable (for
    instance one that intersects the two sets before differencing). Reachability: the store of each of the three
    events is rewritten with an extra variable, then consolidated for real. Killed by: an intersecting comparison.
    Passes under a second correct implementation of the comparison that works on variable and coordinate sets together.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm"), clear=False)
    for event_iloc in (0, 1, 2):
        path = _own(a, event_iloc, "output_swmm_only_node_time_series")
        ds = xr.open_zarr(path, consolidated=False).load()
        ds["extra_one_sided"] = ds["depth"] * 2.0
        import shutil

        shutil.rmtree(path)
        ds.to_zarr(path, mode="w")
    a.process.consolidate_to_datatree()
    assert _gap(a)["swmm_node/timeseries: swmm_only only"] == {"extra_one_sided"}


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".lock"):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _v30_family_tree(work: Path) -> None:
    """A stamped layout-30 analysis tree with a summary at ``{family}/{kind}`` for the SWMM-only and TRITON-only
    families beside the coupled one, and no separate series store."""
    from hhemt.utils import write_datatree_zarr
    from hhemt.version_migration import runner

    def summary(scale: float, entity: str, ids: list[str]) -> xr.Dataset:
        return xr.Dataset(
            {"depth_max": (("event_iloc", entity), scale * np.arange(2 * len(ids), dtype=float).reshape(2, -1))},
            coords={"event_iloc": [0, 1], entity: ids, "event_id": ("event_iloc", ["e0", "e1"])},
        )

    groups = {
        "tritonswmm/swmm_node": summary(1.0, "node_id", ["a", "b", "c"]),
        "swmm_only/swmm_node": summary(2.0, "node_id", ["a", "b", "c"]),
        "swmm_only/swmm_link": summary(3.0, "link_id", ["l1", "l2"]),
        "triton_only/triton": summary(4.0, "x", ["p", "q"]),
        "triton_only/performance": xr.Dataset({"Total": ("event_iloc", [1.0, 2.0])}, coords={"event_iloc": [0, 1]}),
    }
    nodes = {"/": xr.Dataset(attrs={"analysis_id": "t"})}
    nodes.update({f"/{g}": ds for g, ds in groups.items()})
    write_datatree_zarr(xr.DataTree.from_dict(nodes), work / "analysis_datatree.zarr")
    runner.baseline(work, 30)


def test_migration_31_moves_the_swmm_only_and_triton_only_summary_groups_and_leaves_no_series_store(tmp_path):
    """Property: migrating a layout-30 tree moves every SWMM-only and TRITON-only summary group to ``{kind}/summary``
    with identical values, leaves nothing at the old path, folds no series (layout 30 consolidated none for those
    families, so no series group appears for them), reads back through a default consolidated open, and a second run
    changes no byte.

    Class: NEW CAPABILITY over the families (the module's own tests build the coupled family only). Plausible wrong
    implementation: a family list that stops at the coupled family, which leaves the SWMM-only and TRITON-only
    summaries at the layout-30 path where the new consolidation then refuses to write (a hybrid tree). Reachability:
    the real runner over a constructed layout-30 tree. Must-fire control: before the migration each group holds its
    arrays at the old path, so an implementation that moves nothing fails the first assertion after it.
    Killed by: a coupled-only family list. Passes under a second correct implementation that moves groups by copy.
    """
    from hhemt.version_migration import runner

    work = tmp_path / "v30"
    work.mkdir()
    _v30_family_tree(work)
    kinds = ("swmm_only/swmm_node", "swmm_only/swmm_link", "triton_only/triton", "triton_only/performance")
    before_tree = xr.open_datatree(work / "analysis_datatree.zarr", engine="zarr", consolidated=True)
    before = {k: before_tree[k].to_dataset().load() for k in kinds}
    assert all(before[k].data_vars for k in kinds), "control: the arrays sit at the layout-30 path"
    result = runner.run_migration(work, target=31, apply=True)
    assert result.migrations_applied == ["V0031__datatree_summary_and_timeseries_branches"]
    after = xr.open_datatree(work / "analysis_datatree.zarr", engine="zarr", consolidated=True)
    for kind in kinds:
        assert "summary" in after[kind].children, kind
        assert not after[kind].to_dataset().data_vars, f"{kind}: nothing may stay at the old path"
        xr.testing.assert_identical(after[f"{kind}/summary"].to_dataset().load(), before[kind])
    assert not [g for g in after.groups if g.endswith("/timeseries")], "no series group for a family with no series"
    assert not (work / "analysis_timeseries.zarr").exists()
    digest = _digest(work / "analysis_datatree.zarr")
    runner.run_migration(work, target=31, apply=True)
    assert _digest(work / "analysis_datatree.zarr") == digest, "a second run changed a byte"
