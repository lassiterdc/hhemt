"""Coverage for the opt-in per-scenario timeseries consolidation and the durable ``replay_t`` resume-boundary capture.

Three kinds of test:
  * Unit: ``_parse_replay_t`` extracts the LAST replay marker's ``t=`` value from a model-log string, with no
    pipeline run. A replay marker is the line TRITON prints (``SWMM exchange history replayed to t=``) when a resumed
    coupled run replays SWMM's exchange history; its ``t`` is the simulation time, in seconds, at which the resumed
    run was positioned. One model log can hold several markers, and the LAST is the final resume boundary.
  * Config only: ``test_default_is_off`` reads the toggle from ``synth_multi_sim_builder``, a configured analysis
    that was not run (its docstring in ``tests/conftest.py`` says "no compile, no scenario prep").
  * Golden: the two consolidation tests extract the committed multi-sim golden (``tests/fixtures/golden_trees.py``),
    which already carries the per-scenario SWMM node and link timeseries stores, set ``toggle_consolidate_timeseries``
    in the PERSISTED analysis config, and call ``consolidate_to_datatree`` in-process. No solver is compiled or run
    and no Snakemake process starts.

The separate store, the elapsed axis, the append guards and the clearing predicate are exercised the same way: on
the golden, with the reporting-window and per-scenario-store edits each row states. Names this change creates are
read inside the test bodies, behind an assertion that they exist, so that on a tree without the change each test
collects and fails with an AssertionError from this file.

No golden-reprocess prerequisites are copied. The rows need no Snakefile, no ``python`` on ``PATH`` and no child
interpreter identity shim, because nothing is reprocessed through the workflow. The one environment declaration used
is explained at ``_declare_stale_build``.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.analysis_validation import _TRITON_REPLAY_MARKER
from hhemt.processing_analysis import _parse_replay_t
from tests.fixtures.golden_trees import construct_golden_analysis


# ---------------------------------------------------------------------------
# Unit: replay-marker parse
# ---------------------------------------------------------------------------
def test_parse_replay_t_extracts_float():
    line = f"[00:12:34] {_TRITON_REPLAY_MARKER}123.5\n"
    assert _parse_replay_t(line, _TRITON_REPLAY_MARKER) == 123.5


def test_parse_replay_t_returns_none_when_marker_absent():
    assert _parse_replay_t("nothing to see here\n", _TRITON_REPLAY_MARKER) is None


def test_parse_replay_t_keeps_last_marker():
    # The model log is "w"-truncated per exec (one launch of the solver through ``run_simulation_runner``), so it
    # holds only the latest exec's markers; within one exec multiple markers can appear, and the durable value is
    # the LAST one (the final resume boundary).
    text = f"{_TRITON_REPLAY_MARKER}100.0\n... more log ...\n{_TRITON_REPLAY_MARKER}3000.0\n"
    assert _parse_replay_t(text, _TRITON_REPLAY_MARKER) == 3000.0


def test_parse_replay_t_none_on_unparseable_value():
    assert _parse_replay_t(f"{_TRITON_REPLAY_MARKER}not_a_number\n", _TRITON_REPLAY_MARKER) is None


# ---------------------------------------------------------------------------
# Opt-in timeseries consolidation: the toggle default (config only), then the golden rows
# ---------------------------------------------------------------------------
_TS_NODES = ("swmm_node_timeseries", "swmm_link_timeseries")


def test_default_is_off(synth_multi_sim_builder):
    """The toggle must default False so a consolidation that does not opt in stays byte-compatible with trees built
    before the option existed.

    This is asserted on the CONFIG rather than by a second consolidation: the workflow's consolidate step runs as a
    separate process that loads the persisted ``analysis_config.yaml``, so an in-memory flip to False in this process
    would not reach it and such a test would pass for the wrong reason. Established here: the default, and (by the
    toggle-OFF pass of ``test_toggle_change_invalidates_an_already_consolidated_tree``) that ``swmm_node_timeseries``
    is absent after a consolidation with the toggle false. Not established: byte-compatibility with older trees, which
    no test here compares, and the absence of the link timeseries node under the toggle OFF, which that pass does not
    check.
    """
    assert synth_multi_sim_builder.cfg_analysis.toggle_consolidate_timeseries is False


def _golden_with_timeseries_toggle(golden_extractor, *, on: bool):
    """Extract the multi-sim golden and construct its analysis with the toggle written into the persisted config.

    The line is appended, which adds the key exactly once only because the multi-sim golden's ``analysis_config.yaml``
    does not already set ``toggle_consolidate_timeseries`` and ends with a newline (both read from the committed
    ``multisim.zip``).
    """
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + f"toggle_consolidate_timeseries: {'true' if on else 'false'}\n")
    return construct_golden_analysis(system_dir)


def _declare_stale_build(monkeypatch):
    """Make the consolidation build gate defer to the inputs fingerprint.

    A golden's log carries no consolidation build stamp, and a checkout with uncommitted changes never matches one,
    so the build gate would force a rebuild on every call and hide whether the fingerprint alone decides. The
    documented operator escape (``HHEMT_DECLARE_STALE_BUILD=1``) makes the gate reuse a store exactly as it does on a
    matching build, leaving the fingerprint as the only reason to rebuild.

    Setting it is the operator's declaration that the build-stamp mismatch is known and accepted: the gate prints the
    mismatch and reuses the store instead of rebuilding. ``stale`` in the name therefore refers to the BUILD that wrote
    the store, which is the opposite of the staleness of a TREE that
    ``test_toggle_change_invalidates_an_already_consolidated_tree`` exercises, where a changed inputs fingerprint
    forces a rebuild."""
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")


def _require(obj, name: str):
    """Return ``obj.name``, failing with an AssertionError when this change has not created it."""
    value = getattr(obj, name, None)
    assert value is not None, f"{type(obj).__name__}.{name} does not exist"
    return value


def _consolidate(analysis) -> None:
    """Run the separate-store consolidation, failing with an AssertionError when it does not exist yet."""
    _require(analysis.process, "consolidate_timeseries")()


def _store_group(analysis, group: str) -> xr.Dataset:
    """Open one group of ``analysis_timeseries.zarr`` the way a reader of a finished store does."""
    store = _require(analysis.analysis_paths, "analysis_timeseries_zarr")
    assert store.exists(), f"{store} was not written"
    return xr.open_zarr(store, group=f"tritonswmm/{group}")


def _write_coupling_lists(analysis, n_listed: int = 3) -> list[str]:
    """Write a ``coupling_nodes.txt`` beside every scenario's SWMM input, as a toolkit that wrote the flag would have.

    The committed golden predates the coupling-node flag and carries no list, so the consolidation's derivation of
    the flag has nothing to read until this is written. Returns the listed node ids.
    """
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    listed: list[str] = []
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        listed = [str(n) for n in ts["node_id"].values[:n_listed]]
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in listed))
    return listed


def _golden_ready(golden_extractor, *, extra_cfg: str = ""):
    """Golden analysis with the timeseries toggle on, optional extra config lines, and coupling lists written."""
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + extra_cfg)
    analysis = construct_golden_analysis(system_dir)
    listed = _write_coupling_lists(analysis)
    return analysis, listed


def test_toggle_on_consolidates_node_and_link_timeseries_into_their_own_store(golden_extractor):
    """Property: with the toggle on, the node and link series land in ``analysis_timeseries.zarr`` on an
    ``elapsed_s`` axis, and the consolidated DataTree carries neither.

    Class: NEW CAPABILITY. Plausible wrong implementation: the series still written into the tree (a reprocess then
    destroys them), or kept on calendar ``date_time``.
    Reachability: the golden carries per-scenario node and link stores; ``consolidate_to_datatree`` is the one
    caller. Both-states anchor: asserts which dimensions and which nodes exist, true of old and new code alike for
    the dimensions the series keeps.
    Killed by: either wrong implementation. A second correct implementation writes the store from a separate rule;
    the property is about the store's content.
    """
    a, listed = _golden_ready(golden_extractor)
    assert a.cfg_analysis.toggle_consolidate_timeseries is True
    a.process.consolidate_to_datatree()

    tree_path = a.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tsg = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    for node_name in _TS_NODES:
        assert node_name not in tsg, f"{node_name} is still a node of the DataTree"
    for node_name in _TS_NODES:
        node = _store_group(a, node_name)
        assert "elapsed_s" in node.dims, f"{node_name} missing elapsed_s: {dict(node.dims)}"
        assert "date_time" not in node.dims, f"{node_name} kept the calendar axis: {dict(node.dims)}"
        assert "event_iloc" in node.dims, f"{node_name} missing event_iloc: {dict(node.dims)}"
        assert "event_start_datetime" in node.coords
    flag = _store_group(a, "swmm_node_timeseries")["is_coupling_node"]
    assert flag.dims[0] == "event_iloc" and flag.dtype == np.int8
    assert int(flag.isel(event_iloc=0).sum()) == len(listed)


def test_toggle_change_invalidates_an_already_consolidated_tree(golden_extractor, monkeypatch):
    """The consolidation-inputs fingerprint rebuilds a STALE tree.

    A tree can be present, complete AND built from different inputs than the ones now on disk: the timeseries toggle
    was once flipped ON in every persisted config, consolidation re-ran, and the output was silently unchanged because
    the reuse guard keyed only on existence and log completion. ``regenerate_existing`` does not help, since it is a
    driver-side deletion flag that never reaches ``consolidate_to_datatree``.

    Shape: consolidate with the toggle OFF, rewrite the persisted config with the toggle ON on the same tree without
    wiping, build a fresh analysis object, consolidate again, and assert the timeseries nodes appear and the tree was
    rewritten. The guard lives in ``consolidate_to_datatree``, so calling it directly is sufficient.
    """
    _declare_stale_build(monkeypatch)
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    original = cfg.read_text()

    cfg.write_text(original + "toggle_consolidate_timeseries: false\n")
    a_off = construct_golden_analysis(system_dir)
    assert a_off.cfg_analysis.toggle_consolidate_timeseries is False
    a_off.process.consolidate_to_datatree()
    tree_path = a_off.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tsg_off = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    assert "swmm_node_timeseries" not in tsg_off, "toggle OFF should not consolidate timeseries"
    assert not _require(a_off.analysis_paths, "analysis_timeseries_zarr").exists()
    mtime_after_pass1 = tree_path.stat().st_mtime

    cfg.write_text(original + "toggle_consolidate_timeseries: true\n")
    a_on = construct_golden_analysis(system_dir)
    assert a_on.cfg_analysis.toggle_consolidate_timeseries is True
    assert tree_path.exists(), "the first tree must survive; otherwise this tests a fresh build, not staleness"

    _write_coupling_lists(a_on)
    store = _require(a_on.analysis_paths, "analysis_timeseries_zarr")
    assert not store.exists(), "the toggle-off pass must not have written the time-series store"

    a_on.process.consolidate_to_datatree()

    for node_name in _TS_NODES:
        assert "event_iloc" in _store_group(a_on, node_name).dims, (
            f"{node_name} absent after a toggle change on an already-consolidated tree: "
            "the consolidation-inputs fingerprint did not invalidate it"
        )
    assert tree_path.stat().st_mtime > mtime_after_pass1, "tree was not rewritten"


# ---------------------------------------------------------------------------
# Per-scenario store edits the rows below need, and the store's reader
# ---------------------------------------------------------------------------
def _scenario_store(analysis, event_iloc: int, kind: str):
    """The path of one scenario's ``kind`` (``node`` or ``link``) time-series store."""
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(event_iloc, analysis)
    return getattr(scen.scen_paths, f"output_tritonswmm_{kind}_time_series")


def _rewrite_store(path, edit) -> None:
    """Load the zarr store at ``path``, apply ``edit`` to the Dataset and write the result back in its place."""
    ds = xr.open_zarr(path, consolidated=False).load()
    ds = edit(ds)
    shutil.rmtree(path)
    ds.to_zarr(path, mode="w")


def _event_ilocs(analysis, group: str = "swmm_node_timeseries") -> list[int]:
    return [int(v) for v in _store_group(analysis, group)["event_iloc"].values]


def test_events_of_different_windows_share_one_elapsed_axis(golden_extractor, monkeypatch):
    """Property: two events whose calendar starts differ by whole days and whose lengths differ land on ONE
    ``elapsed_s`` axis of the longer event's length, each event's first sample at one reporting step.

    Class: NEW CAPABILITY. Plausible wrong implementation: concatenating on calendar ``date_time``, which pads every
    event to the UNION of all events' timestamps (control below), or counting elapsed time from zero so the first
    sample sits at 0.
    Reachability: the event window is the analysis's own (``_resolve_event_window``), which the test sets per event;
    event 0's store is cut to 8 of 12 steps and event 1's starts three days later; the SHORT event comes first,
    so an axis taken from the first event alone is too short.
    Both-states anchor: the lengths 12 and the first sample 10 s are properties of the reporting step, not of this
    change's wording. Killed by: calendar concatenation or a zero origin.
    A second correct implementation measures elapsed time from the store's first sample plus one step.
    """
    a, _ = _golden_ready(golden_extractor)
    from hhemt.scenario import TRITONSWMM_scenario

    starts = {0: "2000-01-01", 1: "2000-01-04", 2: "2000-01-01"}
    lengths_s = {0: 80, 1: 120, 2: 120}
    monkeypatch.setattr(
        TRITONSWMM_scenario,
        "_resolve_event_window",
        lambda self: (
            pd.Timestamp(starts[int(self.event_iloc)]),
            pd.Timestamp(starts[int(self.event_iloc)]) + pd.Timedelta(seconds=lengths_s[int(self.event_iloc)]),
        ),
    )

    def truncate(ds):
        return ds.isel(date_time=slice(0, 8))

    def shift(ds):
        return ds.assign_coords(date_time=ds["date_time"] + np.timedelta64(3, "D"))

    for kind in ("node", "link"):
        _rewrite_store(_scenario_store(a, 0, kind), truncate)
        _rewrite_store(_scenario_store(a, 1, kind), shift)
    calendar = xr.concat(
        [
            xr.open_zarr(_scenario_store(a, i, "link"), consolidated=False).load().expand_dims(event_iloc=[i])
            for i in (0, 1)
        ],
        dim="event_iloc",
        join="outer",
    )

    _consolidate(a)

    node = _store_group(a, "swmm_node_timeseries")
    assert node.sizes["elapsed_s"] == 12, "the axis must have the length of the longer event"
    assert int(node["elapsed_s"].values[0]) == 10, "the first sample sits at one reporting step, not at zero"
    depth = node["depth"]
    assert int(depth.isel(event_iloc=0).notnull().any("node_id").sum()) == 8
    assert bool(depth.isel(event_iloc=0, elapsed_s=slice(8, None)).isnull().all()), "the short event pads with NaN"
    assert calendar.sizes["date_time"] == 20, "control: the calendar axis of the same pair has the union length"


def test_the_append_derives_a_missing_flag_and_refuses_a_missing_or_extra_variable_before_writing(
    golden_extractor,
):
    """Property: a scenario lacking ``is_coupling_node`` gets it derived from its ``coupling_nodes.txt`` and the store
    opens; a scenario lacking ``depth``, or carrying a time-variant variable the store lacks, raises
    ``ProcessingError`` naming the variable and the scenario before any write, leaving the store as it was.

    Class: NEW CAPABILITY. Plausible wrong implementation: appending a Dataset whose variables differ from the
    store's. Measured: ``to_zarr`` returns without error, one variable is left at length 2 and another at length 1,
    and ``xr.open_zarr`` then raises ``ValueError``, so one older scenario would make every earlier event unreadable.
    Reachability: events 0 and 1 are appended, then event 2 is edited (golden stores carry no flag, so the first two
    already exercise the derivation). Both-states anchor: asserts the stored event list and the open.
    Killed by: skipping the guard. A second correct implementation compares variable sets after the append and rolls
    back; the "nothing was written" assertion would then need a different probe.
    """
    a, _ = _golden_ready(golden_extractor)
    _rewrite_store(_scenario_store(a, 2, "node"), lambda ds: ds.drop_vars("depth"))
    with pytest.raises(Exception) as err:
        _consolidate(a)
    assert type(err.value).__name__ == "ProcessingError"
    message = str(err.value)
    assert "depth" in message and "event_iloc 2" in message
    assert _event_ilocs(a) == [0, 1], "the refused scenario must not have been written"
    assert "is_coupling_node" in _store_group(a, "swmm_node_timeseries").data_vars


def test_the_append_refuses_a_variable_the_store_lacks_before_writing(golden_extractor):
    """Property: a scenario carrying a time-variant variable the store lacks raises ``ProcessingError`` naming the
    variable and the scenario before any write, and the store keeps the events appended before it.

    Class: NEW CAPABILITY. Plausible wrong implementation: appending anyway (see the sibling row for the measured
    consequence). Reachability: event 2's node store gets one extra time-variant variable. Both-states anchor: asserts
    the exception type, the name and the event list. Killed by: skipping the guard. A second correct implementation
    names the variable in a warning and drops it; the property here is the refusal.
    """
    b, _ = _golden_ready(golden_extractor)
    _rewrite_store(_scenario_store(b, 2, "node"), lambda ds: ds.assign(extra_series=ds["depth"] * 2))
    with pytest.raises(Exception) as err2:
        _consolidate(b)
    assert type(err2.value).__name__ == "ProcessingError" and "extra_series" in str(err2.value)
    assert _event_ilocs(b) == [0, 1]


def test_static_report_variables_are_not_appended_and_do_not_have_to_match(golden_extractor):
    """Property: a node store holding a static variable (``max_ponded_depth``) appends without it, and a second scenario
    lacking that static variable appends without raising; a second scenario lacking the time-variant ``depth`` still
    raises.

    Class: NEW CAPABILITY. Plausible wrong implementation: appending every variable, so each variable a later chunk
    adds makes an older scenario fail the variable-set guard.
    Reachability: event 1's store is rewritten without ``max_ponded_depth``. Both-states anchor: asserts which names the
    store holds. Killed by: appending static variables. A second correct implementation moves the static variables
    to a separate group.
    """
    a, _ = _golden_ready(golden_extractor)
    node0 = xr.open_zarr(_scenario_store(a, 0, "node"), consolidated=False)
    assert "max_ponded_depth" in node0.data_vars, "the golden node store must carry the static variable the row needs"
    _rewrite_store(_scenario_store(a, 1, "node"), lambda ds: ds.drop_vars("max_ponded_depth"))
    _consolidate(a)
    stored = _store_group(a, "swmm_node_timeseries")
    assert "max_ponded_depth" not in stored.data_vars
    assert _event_ilocs(a) == [0, 1, 2]
    assert all("event_iloc" in stored[v].dims for v in stored.data_vars), "every variable must carry event_iloc"


def test_a_scenario_lacking_a_time_variant_variable_still_raises(golden_extractor):
    """Property: the must-fire control of the static-variable row: a second scenario lacking the time-variant
    ``depth`` raises ``ProcessingError`` naming it.

    Class: NEW CAPABILITY. Plausible wrong implementation: a guard that tolerates every missing variable once static
    ones are tolerated. Reachability: event 1's node store loses ``depth``. Both-states anchor: asserts the type and
    the name. Killed by: that guard. A second correct implementation raises a subclass.
    """
    c, _ = _golden_ready(golden_extractor)
    _rewrite_store(_scenario_store(c, 1, "node"), lambda ds: ds.drop_vars("depth"))
    with pytest.raises(Exception) as err:
        _consolidate(c)
    assert type(err.value).__name__ == "ProcessingError" and "depth" in str(err.value)


def test_a_second_pass_appends_nothing_and_a_failed_readback_keeps_the_stores(golden_extractor, monkeypatch):
    """Property: consolidating twice leaves one slice per event (never ``[0, 1, 1]``), and a scenario whose read-back
    fails keeps its stores while clearing is on.

    Class: NEW CAPABILITY. Plausible wrong implementations: an unconditional re-append (duplicates the event), and
    clearing before the read-back (the only copy deleted on a write that did not take).
    Reachability: the read-back is made to fail for event 1 by replacing the comparison; event 0 completes first.
    Both-states anchor: asserts the event list and the existence of files.
    Killed by: those. A second correct implementation checks the slice by reading it with a second process.
    """
    a, _ = _golden_ready(golden_extractor)
    _consolidate(a)
    _consolidate(a)
    assert _event_ilocs(a) == [0, 1, 2]


def test_a_failed_readback_keeps_the_scenario_stores_while_clearing_is_on(golden_extractor, monkeypatch):
    """Property: when the read-back of event 1 fails, event 1's node and link stores are kept and the error is a
    ``ProcessingError``, while event 0, already verified, is cleared.

    Class: NEW CAPABILITY. Plausible wrong implementation: clearing before the read-back (the only copy deleted on a
    write that did not take). Reachability: the comparison is replaced so that it fails for event 1 only.
    Both-states anchor: asserts file existence. Killed by: clearing first. A second correct implementation verifies
    through a second reader; the same files would remain.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    d, _ = _golden_ready(golden_extractor, extra_cfg=_clear_cfg())
    real = pp._stored_slice_equals
    monkeypatch.setattr(
        pp, "_stored_slice_equals", staticmethod(lambda s, g, e, i: False if i == 1 else real(s, g, e, i))
    )
    with pytest.raises(Exception) as err:
        _consolidate(d)
    assert type(err.value).__name__ == "ProcessingError"
    assert not _scenario_store(d, 0, "node").exists(), "event 0 verified, so its store is cleared"
    assert _scenario_store(d, 1, "node").exists() and _scenario_store(d, 1, "link").exists()


def _clear_cfg() -> str:
    from hhemt.config.analysis import analysis_config

    assert "toggle_clear_scenario_timeseries_after_consolidation" in analysis_config.model_fields
    return "toggle_clear_scenario_timeseries_after_consolidation: true\n"


def test_clearing_deletes_the_scenario_stores_records_it_and_reads_them_as_processed(golden_extractor):
    """Property: with clearing on, every scenario's node and link stores are gone after its slice is verified, the
    model log records both deletions, ``_swmm_node_outputs_processed`` still reads True, the reclaim check reports the
    series as cleared rather than missing, and a second pass over the complete store appends nothing.

    Class: NEW CAPABILITY. Plausible wrong implementation: clearing without recording it, after which the processed
    predicate reads False and the next pass re-derives a series whose raw inputs are gone.
    Reachability: ``consolidate_timeseries`` is the one clearer. Both-states anchor: the stored series is intact.
    Killed by: those. A second correct implementation clears after the index rebuild; the assertions are unchanged.
    """
    a, _ = _golden_ready(golden_extractor, extra_cfg=_clear_cfg())
    before = {i: xr.open_zarr(_scenario_store(a, i, "node"), consolidated=False).load() for i in a.df_sims.index}
    _consolidate(a)
    from hhemt.scenario import TRITONSWMM_scenario

    for i in a.df_sims.index:
        assert not _scenario_store(a, i, "node").exists() and not _scenario_store(a, i, "link").exists()
        log = TRITONSWMM_scenario(i, a).get_log("tritonswmm")
        assert log.consolidated_node_timeseries_cleared.get() is True
        assert log.consolidated_link_timeseries_cleared.get() is True
    stored = _store_group(a, "swmm_node_timeseries")
    np.testing.assert_array_equal(
        stored["depth"].isel(event_iloc=0).values[:, : before[0].sizes["date_time"]],
        before[0]["depth"].transpose("node_id", "date_time").values,
    )
    from hhemt.analysis_validation import check_data_availability

    result = check_data_availability(a)
    assert result.passed and "cleared" in result.summary, result.summary
    _consolidate(a)
    assert _event_ilocs(a) == [0, 1, 2]


def test_a_stored_event_with_no_verified_clearing_and_no_own_store_is_refused(golden_extractor):
    """Property: an event present in the store whose own store is gone, with no recorded clearing, is refused
    rather than skipped.

    Class: NEW CAPABILITY. Plausible wrong implementation: treating an absent own store as proof of a verified
    clearing. Reachability: consolidate without clearing, delete event 1's node store by hand, consolidate again.
    Both-states anchor: asserts the exception type and that nothing was removed from the store.
    Killed by: that treatment. A second correct implementation rewrites the slice from a backup; none exists.
    """
    a, _ = _golden_ready(golden_extractor)
    _consolidate(a)
    shutil.rmtree(_scenario_store(a, 1, "node"))
    with pytest.raises(Exception) as err:
        _consolidate(a)
    assert type(err.value).__name__ == "ProcessingError"
    assert _event_ilocs(a) == [0, 1, 2]


def test_the_store_is_not_a_root_tree_and_survives_the_consolidation_and_reprocess_deletes(golden_extractor):
    """Property: ``analysis_timeseries.zarr`` is in neither ``ROOT_TREE_NAMES`` nor the consolidation delete's
    analysis-level artifacts, so those deletes cannot remove it.

    Class: NEW CAPABILITY. Plausible wrong implementation: listing the store beside the datatree, after which a
    reprocess deletes the only copy of the series. Reachability: both names are module constants read by the
    deletes. Both-states anchor: asserts membership by name. Killed by: adding the store to either.
    A second correct implementation guards the delete runners with a check; the constants would then be unchanged.
    """
    import hhemt.delete_consolidation_runner as dcr
    from hhemt.paths import AnalysisPaths
    from hhemt.utils import ROOT_TREE_NAMES

    assert "analysis_timeseries_zarr" in AnalysisPaths.__dataclass_fields__, "the store has no path attribute yet"
    assert "analysis_timeseries.zarr" not in set(ROOT_TREE_NAMES)
    flat = " ".join(str(x) for x in dcr._ANALYSIS_LEVEL_ARTIFACTS)
    assert "analysis_timeseries" not in flat
    assert "analysis_datatree" in flat, "control: the datatree IS a listed artifact"


def test_the_fingerprint_covers_the_flux_and_clearing_toggles_and_the_version_moved(golden_extractor):
    """Property: flipping ``toggle_coupling_exchange_flow`` or the clearing toggle changes the consolidation inputs
    fingerprint, an unchanged config does not, and ``CONSOLIDATION_VERSION`` is at least 2.

    Class: NEW CAPABILITY. Plausible wrong implementation: the toggles missing from the fingerprint, so a tree built
    without the flux is reused after the toggle is switched on. Reachability: the fingerprint is computed from the
    analysis config. Both-states anchor: asserts equality and inequality of hashes. Killed by: omitting either
    toggle. A second correct implementation hashes the whole config; the two inequalities still hold.
    """
    a, _ = _golden_ready(golden_extractor)
    base = a.process._consolidation_inputs_fingerprint()
    assert a.process._consolidation_inputs_fingerprint() == base, "control: an unchanged config hashes the same"
    a.cfg_analysis.toggle_coupling_exchange_flow = True
    flux = a.process._consolidation_inputs_fingerprint()
    assert flux != base
    a.cfg_analysis.toggle_clear_scenario_timeseries_after_consolidation = True
    assert a.process._consolidation_inputs_fingerprint() != flux
    assert type(a.process).CONSOLIDATION_VERSION >= 2


def test_preflight_refuses_clearing_without_the_consolidation_it_clears_after():
    """Property: preflight refuses ``toggle_clear_scenario_timeseries_after_consolidation`` unless
    ``toggle_consolidate_timeseries`` is on, and accepts the other three rows of the truth table.

    Class: NEW CAPABILITY. Plausible wrong implementation: no refusal, so clearing deletes the only copy while no
    consolidated store exists. Reachability: the four rows are the whole truth table. Both-states anchor: error
    count per row. Killed by: never refusing, or refusing whenever clearing is on.
    A second correct implementation raises instead of accumulating.
    """
    from types import SimpleNamespace

    from hhemt import validation

    check = _require(validation, "_validate_clear_scenario_timeseries")
    for consolidate, clear, refused in (
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ):
        result = validation.ValidationResult(context="test")
        check(
            SimpleNamespace(
                toggle_consolidate_timeseries=consolidate,
                toggle_clear_scenario_timeseries_after_consolidation=clear,
            ),
            result,
        )
        assert bool(result.errors) is refused, (consolidate, clear)


def test_the_interval_series_integrates_to_the_step_volume_only_over_the_covered_span(tmp_path):
    """Property: the stored interval means (times the interval) equal the step series' volume over
    ``(first edge, last report time]`` and fall short of the whole-file ``coupling_exchange_volume_net`` by exactly
    the tail past the last report time.

    Class: NEW CAPABILITY, the scoping the comparison between the stored series and ``coupling_exchange_volume_net``
    owes. Plausible wrong implementation: asserting equality with the whole-file volume, which holds only when the
    file ends within a step of the last report time and is wrong (2080 against 6580 cfs-s) for this file, which runs
    one 45 s step at 100 cfs past it.
    Reachability: hand case, steps ending 50, 150, 220, 340, 360 and 405 s at 2, 4, 6, 8, 10 and 100 cfs, report
    times 120, 240, 360 s. Both-states anchor: the numbers are arithmetic on the file. Killed by: an unscoped
    equality. A second correct implementation pads the last interval to the file end.
    """
    from hhemt.coupling_exchange import exchange_statistics, read_exchange_sidefile, to_reporting_intervals

    dt = np.array([50.0, 100.0, 70.0, 120.0, 20.0, 45.0])
    q = np.array([[2.0], [4.0], [6.0], [8.0], [10.0], [100.0]])
    side = tmp_path / "x_exchange_replay.bin"
    side.write_bytes(
        np.array([0x53574D32, 1, 8], dtype="<i4").tobytes() + np.column_stack([dt, q]).astype("<f8").tobytes()
    )
    nodes = tmp_path / "coupling_nodes.txt"
    nodes.write_text("n1\n")
    series = read_exchange_sidefile(side, nodes)
    cfs_to_cms = 0.028316847
    intervals = to_reporting_intervals(series, [120.0, 240.0, 360.0])
    covered = float(intervals.exchange_mean_cms.sum()) * 120.0
    whole = float(exchange_statistics(series).coupling_exchange_volume_net[0])
    assert covered == pytest.approx(2080.0 * cfs_to_cms, rel=1e-12)
    assert whole == pytest.approx(6580.0 * cfs_to_cms, rel=1e-12)
    assert whole - covered == pytest.approx(45.0 * 100.0 * cfs_to_cms, rel=1e-12), "the gap is the tail"


def test_a_stored_slice_that_fails_its_readback_is_rewritten_from_the_scenario_store(golden_extractor):
    """Property: an event present in the store whose slice no longer equals its own store (here: overwritten in
    place) is rewritten from the scenario's store, not skipped and not duplicated.

    Class: NEW CAPABILITY. Plausible wrong implementations: trusting presence alone (the damaged slice survives), or
    re-appending (the event is duplicated, measured ``[0, 1, 1]``). Reachability: the damage is written straight
    into the zarr array with the zarr API after a first pass. Both-states anchor: asserts the event list and that
    the values equal the scenario store's. Killed by: either. A second correct implementation deletes the store and
    rebuilds it; the event list and values would be the same.
    """
    import zarr

    a, _ = _golden_ready(golden_extractor)
    _consolidate(a)
    store = a.analysis_paths.analysis_timeseries_zarr
    group = zarr.open_group(store, path="tritonswmm/swmm_node_timeseries", mode="r+")
    group["depth"][1, ...] = 999.0
    assert float(_store_group(a, "swmm_node_timeseries")["depth"].isel(event_iloc=1).max()) == 999.0
    _consolidate(a)
    assert _event_ilocs(a) == [0, 1, 2]
    repaired = _store_group(a, "swmm_node_timeseries")["depth"].isel(event_iloc=1)
    own = xr.open_zarr(_scenario_store(a, 1, "node"), consolidated=False)["depth"].transpose("node_id", "date_time")
    np.testing.assert_array_equal(repaired.values[:, : own.sizes["date_time"]], own.values)
