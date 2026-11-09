"""Reviewer tests for family parity (family parity: SWMM-only and TRITON-only series consolidated and cleared).

No solver and no Snakemake. Every test quantifies over a CLASS the coder's own file does not enumerate: the series
modes (one table row each), the three model logs (one clearing record each), and the kill window between a pending
clearing record and the delete. The committed multi-sim golden supplies the three arms.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_FAMILY_TOGGLES = {
    "tritonswmm": "toggle_tritonswmm_model",
    "swmm": "toggle_swmm_model",
    "triton": "toggle_triton_model",
}
# model -> ((scenario-path attr of the store, log-field stem), ...); the stems are the log names D14 and D18 fix.
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
_GROUPS = {
    "tritonswmm": (
        "tritonswmm/swmm_node/timeseries",
        "tritonswmm/swmm_link/timeseries",
        "tritonswmm/triton/timeseries",
    ),
    "swmm": ("swmm_only/swmm_node/timeseries", "swmm_only/swmm_link/timeseries"),
    "triton": ("triton_only/triton/timeseries",),
}
# scenario-path attr prefix -> (tree family, owning model)
_ATTR_FAMILY = {
    "output_tritonswmm_": ("tritonswmm", "tritonswmm"),
    "output_swmm_only_": ("swmm_only", "swmm"),
    "output_triton_only_": ("triton_only", "triton"),
}


def _post_processing():
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing

    return TRITONSWMM_analysis_post_processing


def _ready(golden_extractor, *, families, clear: bool = True):
    """Golden analysis enabling exactly ``families``, series consolidation on, the clearing switch as given."""
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
        + "toggle_consolidate_timeseries: true\n"
        + f"toggle_clear_scenario_timeseries_after_consolidation: {'true' if clear else 'false'}\n"
    )
    analysis = construct_golden_analysis(system_dir)
    if "tritonswmm" in families:
        # the golden's coupled node store carries no coupling flag; the consolidation derives it from this list
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


def _own(a, event_iloc: int, attr: str) -> Path:
    return Path(getattr(_scen(a, event_iloc).scen_paths, attr))


def _flag(a, event_iloc: int, model: str, name: str):
    """The value of one log field of ``model``'s log, re-read from disk; None when the field is unset."""
    field = getattr(_scen(a, event_iloc).get_log(model), name, None)
    return None if field is None else field.get()


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(Path(a.analysis_paths.analysis_datatree_zarr), group=name, consolidated=False)


def test_every_series_mode_names_one_family_one_kind_and_one_clearing_record_consistently():
    """Property: for every mode of the series mode table, the scenario-path attr, the tree path and the cleared and
    pending log fields name the SAME family and the SAME kind, and every family appears in the table.

    Input class: the rows of ``_TIMESERIES_MODE_CONFIG`` (attr), ``_TIMESERIES_MODE_TO_TREE_PATH`` and
    ``_TIMESERIES_CLEARED_FIELDS`` joined on the mode key. Reachability: the three tables are the ones
    ``consolidate_timeseries`` indexes by one mode key; a row whose family or kind disagrees between tables writes a
    store to one group and records its clearing in another kind's field (D18 closes the domain: three families).
    Both-states anchor: the check reads the join, so it holds before this chunk for the coupled rows and after it for
    all six. Must-fire control: all three families are present (a table that lost a family's rows reds the last
    assertion). Killed by: a SWMM-only row pointing at the coupled tree path, or the TRITON-only row recording its
    clearing in the node field. Passes under a second correct implementation that derives a table from another.
    """
    cls = _post_processing()
    attrs = cls._TIMESERIES_MODE_CONFIG
    paths = cls._TIMESERIES_MODE_TO_TREE_PATH
    fields = cls._TIMESERIES_CLEARED_FIELDS
    assert set(attrs) == set(paths) == set(fields)
    families_seen = set()
    for mode, attr in attrs.items():
        prefix = next(p for p in _ATTR_FAMILY if attr.startswith(p))
        tree_family, _model = _ATTR_FAMILY[prefix]
        families_seen.add(tree_family)
        group = paths[mode]
        assert group.startswith(f"{tree_family}/") and group.endswith("/timeseries"), (mode, group)
        assert mode.startswith(f"{tree_family}_"), (mode, tree_family)
        if "_triton_" in f"_{mode}" and mode.endswith("_triton_ts"):
            kind, expected_group_kind = "triton", "triton"
        elif mode.endswith("_swmm_node_ts"):
            kind, expected_group_kind = "node", "swmm_node"
        elif mode.endswith("_system_ts"):
            kind, expected_group_kind = "system", "system"
        else:
            assert mode.endswith("_swmm_link_ts"), mode
            kind, expected_group_kind = "link", "swmm_link"
        assert group == f"{tree_family}/{expected_group_kind}/timeseries", (mode, group)
        cleared, pending = fields[mode]
        assert cleared == f"consolidated_{kind}_timeseries_cleared", (mode, cleared)
        assert pending == f"consolidated_{kind}_timeseries_clearing_pending", (mode, pending)
    assert families_seen == {"tritonswmm", "swmm_only", "triton_only"}


def test_every_non_performance_series_store_of_the_scenario_paths_has_a_consolidation_mode():
    """Property: the set of per-scenario series stores the scenario paths declare (``output_*_timeseries`` or
    ``output_*_time_series``, performance excluded) equals the set of stores the mode table consolidates.

    Input class: the dataclass fields of ``ScenarioPaths``, read from the class, so a store a later change adds is
    covered without an edit here (D18 names this domain: "the series stores of ``ScenarioPaths``"; the per-rank
    performance series is the stated exception, never consolidated). Reachability: the fields are read from the real
    class; a store with no mode is written by the run and never consolidated or cleared, which is the defect D18 was
    raised for. Both-states anchor: set equality holds for the coupled rows before the chunk and for all six after it.
    Must-fire control: the derived domain names at least one store of each of the three families. Killed by: dropping
    any one of the three new rows from the mode table. Passes under a second correct implementation that adds the
    performance stores' consolidation, if it also removes them from this exception (a decision, not this test's).
    """
    from hhemt.paths import ScenarioPaths

    domain = {
        f.name
        for f in dataclasses.fields(ScenarioPaths)
        if f.name.startswith("output_")
        and f.name.endswith(("_timeseries", "_time_series"))
        and "performance" not in f.name
    }
    assert {name for name in domain if name.startswith(tuple(_ATTR_FAMILY))} == domain
    assert {p for p in _ATTR_FAMILY if any(n.startswith(p) for n in domain)} == set(_ATTR_FAMILY), domain
    assert domain == set(_post_processing()._TIMESERIES_MODE_CONFIG.values())


@pytest.mark.parametrize("family", ["tritonswmm", "swmm", "triton"])
def test_a_single_family_analysis_records_its_clearing_in_its_own_log_and_no_other(golden_extractor, family):
    """Property: with ONE model enabled and the clearing switch on, the consolidation clears that model's stores and
    records the cleared field True in THAT model's log, and no other model's log records any consolidated clearing.

    Input class: the three single-family analyses. Reachability: the real ``consolidate_timeseries`` over the golden
    with the other two models switched off. A coupled-log constant (the pre-change ``scen.get_log("tritonswmm")``) is
    the plausible wrong implementation: in the SWMM-only and TRITON-only arms it records the clearing in the coupled
    log and leaves the owning log unset. Both-states anchor: the assertion turns on which log field reads True, which
    the pre-change tree also answers (for the coupled arm True in the coupled log, for the others unset). Must-fire
    control: the owning log's fields read True, so an implementation that records nothing fails the first loop.
    Killed by: any implementation that writes a family's clearing into another model's log. Passes under a second
    correct implementation that keeps one clearing log of its own, if it is still addressed per model.
    """
    a = _ready(golden_extractor, families=(family,), clear=True)
    a.process.consolidate_timeseries()
    for event_iloc in a.df_sims.index:
        for attr, stem in _STORES[family]:
            assert not _own(a, event_iloc, attr).exists(), (event_iloc, attr)
            assert _flag(a, event_iloc, family, f"{stem}_cleared") is True, (event_iloc, family, stem)
        for other in _STORES:
            if other == family:
                continue
            for _attr, stem in _STORES[other]:
                assert _flag(a, event_iloc, other, f"{stem}_cleared") is not True, (event_iloc, other, stem)
                assert not _flag(a, event_iloc, other, f"{stem}_clearing_pending"), (event_iloc, other, stem)


@pytest.mark.parametrize("model", ["tritonswmm", "swmm", "triton"])
def test_a_clearing_killed_after_its_pending_record_is_finished_from_the_stored_slice_for_every_model(
    golden_extractor, monkeypatch, model
):
    """Property: when the process dies after a model's pending clearing record is written and a store of that
    model is part-deleted, the next consolidation finishes the clearing from the slice already in the group (every
    store of the model gone, cleared True and pending empty in THAT model's log), appends the remaining events, and
    leaves the stored slice of the killed event byte-equal to what it held before the resume.

    Input class: the three models, killed at the first delete of that model's stores. Reachability: the real
    ``consolidate_timeseries`` with ``hhemt.du_sentinels.delete_and_account`` made to die on the first call that names
    one of the model's stores; the pending record is read from the owning model's log at that instant. Both-states
    anchor: the assertions read the end state of the stores, the log fields and the stored slice, not any message.
    Must-fire control: at the kill, the pending record is non-empty and the cleared field is not True for every store
    of the model (a clearing that deletes before it records, the wrong order, fails there), and the killed
    event's own store is still on disk until the test part-deletes it. Killed by: recording the pending state in the
    wrong model's log, recording it after the delete, or a resume that re-verifies against the half-deleted source
    and overwrites the slice. Passes under a second correct implementation that clears all families in one delete,
    provided it records the pending state before that delete in each owning log.
    """
    import shutil

    import hhemt.du_sentinels as du

    class _Killed(RuntimeError):
        pass

    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=True)
    owned = {str(_own(a, 0, attr)) for attr, _stem in _STORES[model]}
    real = du.delete_and_account
    seen_at_kill: dict[str, object] = {}

    def killer(paths, **kwargs):
        if owned & {str(p) for p in paths}:
            for attr, stem in _STORES[model]:
                seen_at_kill[stem] = (
                    _flag(a, 0, model, f"{stem}_clearing_pending"),
                    _flag(a, 0, model, f"{stem}_cleared"),
                    _own(a, 0, attr).exists(),
                )
            raise _Killed("killed after the pending record")
        return real(paths, **kwargs)

    monkeypatch.setattr(du, "delete_and_account", killer)
    with pytest.raises(_Killed):
        a.process.consolidate_timeseries()
    monkeypatch.setattr(du, "delete_and_account", real)

    assert set(seen_at_kill) == {stem for _attr, stem in _STORES[model]}
    for stem, (pending, cleared, still_there) in seen_at_kill.items():
        assert pending, f"{model}: {stem} had no pending record when the delete began"
        assert cleared is not True, f"{model}: {stem} was already recorded cleared before the delete"
        assert still_there, f"{model}: {stem} store was gone before the delete began (control)"

    # Part-delete one of the model's stores of event 0, as a kill part-way through the delete would.
    first_attr = _STORES[model][0][0]
    shutil.rmtree(_own(a, 0, first_attr))
    group_before = {g: _group(a, g).sel(event_iloc=[0]).load() for g in _GROUPS[model]}

    a.process.consolidate_timeseries()

    for event_iloc in a.df_sims.index:
        for family, stores in _STORES.items():
            for attr, stem in stores:
                assert not _own(a, event_iloc, attr).exists(), (event_iloc, attr)
                assert _flag(a, event_iloc, family, f"{stem}_cleared") is True, (event_iloc, family, stem)
                assert not _flag(a, event_iloc, family, f"{stem}_clearing_pending"), (event_iloc, family, stem)
    for groups in _GROUPS.values():
        for g in groups:
            assert [int(v) for v in _group(a, g)["event_iloc"].values] == [0, 1, 2], g
    for g, before in group_before.items():
        after = _group(a, g).sel(event_iloc=[0]).load()
        assert sorted(after.data_vars) == sorted(before.data_vars), g
        for name in before.data_vars:
            assert np.array_equal(after[name].values, before[name].values, equal_nan=True), (g, name)


def test_a_second_consolidation_after_every_family_was_cleared_changes_nothing(golden_extractor):
    """Property: re-running the consolidation over an analysis whose stores of all three families are already cleared
    raises nothing, recreates no store, appends no event, and leaves every group byte-equal.

    Input class: the re-fire of the consolidation step (a Snakemake re-run, or a second ``process`` call) over the
    cleared state of every model. Reachability: the real ``consolidate_timeseries`` twice. Both-states anchor: the
    second call's outcome is read from the stores and the groups, so the same assertions hold on the coupled family
    before this chunk. Must-fire control: the first run really cleared (the six stores are absent before the second
    run), so a run that never cleared would fail there rather than pass vacuously. Killed by: a plan step that, for a
    cleared SWMM-only or TRITON-only scenario, reads the missing store as an error or appends the event a second time
    (the measured duplicate is ``[0, 1, 1]``). Passes under a second correct implementation that skips cleared
    scenarios before planning.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=True)
    a.process.consolidate_timeseries()
    for event_iloc in a.df_sims.index:
        for stores in _STORES.values():
            for attr, _stem in stores:
                assert not _own(a, event_iloc, attr).exists(), "control: the first run cleared every store"
    before = {g: _group(a, g).load() for groups in _GROUPS.values() for g in groups}

    a.process.consolidate_timeseries()

    for event_iloc in a.df_sims.index:
        for family, stores in _STORES.items():
            for attr, stem in stores:
                assert not _own(a, event_iloc, attr).exists(), (event_iloc, attr)
                assert _flag(a, event_iloc, family, f"{stem}_cleared") is True, (event_iloc, family, stem)
    for g, ds_before in before.items():
        ds_after = _group(a, g).load()
        assert [int(v) for v in ds_after["event_iloc"].values] == [0, 1, 2], g
        assert sorted(ds_after.data_vars) == sorted(ds_before.data_vars), g
        for name in ds_before.data_vars:
            assert np.array_equal(ds_after[name].values, ds_before[name].values, equal_nan=True), (g, name)


@pytest.mark.parametrize(
    "summary_mode, ts_mode",
    [
        ("swmm_only_node", "swmm_only_swmm_node_ts"),
        ("swmm_only_link", "swmm_only_swmm_link_ts"),
        ("triton_only", "triton_only_triton_ts"),
    ],
)
def test_a_rebuild_that_drops_a_family_mode_removes_its_summary_and_leaves_its_real_series_untouched(
    golden_extractor, monkeypatch, summary_mode, ts_mode
):
    """Property: when an analysis stops producing a SWMM-only or TRITON-only mode (its model switched off after a first
    consolidation), the rebuild removes that mode's ``summary`` group and leaves the REAL ``timeseries`` branch the
    first consolidation wrote beside it byte-identical, and still a group.

    Input class: the three modes that write a series besides the coupled one. Reachability:
    ``_write_summary_branches`` deletes ``set(_MODE_TO_TREE_PATH.values()) - written`` and then runs
    ``consolidate_timeseries``; a toggled-off model
    reaches it with its summary AND series stores unbound (both unbound here, as the scenario paths report them).
    Both-states anchor: the assertion is on the bytes of the series files and the existence of the group, true of the
    pre-chunk tree for the coupled series. Must-fire control: before the rebuild the series branch exists and holds
    files (the consolidation makes the branch exist; a tree without it fails there, not vacuously).
    Why this test and not the repaired neighbour: ``test_review_1d1_datatree_branch_layout.py::
    test_a_rebuild_removes_the_summary_of_a_mode_the_analysis_no_longer_produces_and_keeps_every_series`` now plants
    nothing, leaves the SWMM-only series bound, and so lets ``consolidate_timeseries`` re-create a series branch the
    removal deleted: measured green under a removal of the stale mode's PARENT group, red under the pre-chunk test and
    tree. This test unbinds the series store as well, so nothing can re-create the branch.
    Killed by: deleting ``Path(store) / family / kind`` instead of the ``summary`` path. Passes under a second correct
    implementation that removes the summary group through any other deletion of that one path.
    """
    import hhemt.processing_analysis as pa
    import hhemt.provenance as prov
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from tests.test_datatree_branch_layout import _hashes, _tree_path
    from tests.test_datatree_branch_layout import _ready as _branch_ready

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)
    a = _branch_ready(golden_extractor)
    a.process.consolidate_to_datatree()
    store = Path(_tree_path(a))
    summary_group = pp._MODE_TO_TREE_PATH[summary_mode]
    stale_parent = Path(summary_group).parent
    series_group = pp._TIMESERIES_MODE_TO_TREE_PATH[ts_mode]
    assert Path(series_group).parent == stale_parent
    assert (store / summary_group).is_dir(), "control: the first consolidation wrote the summary"
    before = _hashes(store, series_group)
    assert before, "control: the first consolidation wrote the real series branch, with files"

    unbound = {pp._MODE_CONFIG[summary_mode][0], pp._TIMESERIES_MODE_CONFIG[ts_mode]}
    real_scenario = pa.TRITONSWMM_scenario

    class _Paths:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return None if name in unbound else getattr(self._real, name)

    class _Scenario:
        def __init__(self, real):
            self._real = real
            self.scen_paths = _Paths(real.scen_paths)

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda *args, **kwargs: _Scenario(real_scenario(*args, **kwargs)))
    monkeypatch.setattr(pp, "_consolidation_inputs_fingerprint", lambda self: "inputs-after-a-toggle")
    a.process.consolidate_to_datatree()

    assert not (store / summary_group).exists(), "the dropped mode's summary is removed"
    assert _hashes(store, series_group) == before, "the real series beside it is untouched"
    assert (store / stale_parent / "zarr.json").is_file(), "the group that still holds a series stays a group"


def _sim_post_processing(a, event_iloc: int, model: str):
    """A post-processing object over the scenario's real log for ``model`` and its real scenario paths."""
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    scen = _scen(a, event_iloc)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.log = scen.get_log(model)
    proc.scen_paths = scen.scen_paths
    proc.log.processing_log.outputs.clear()
    for name in ("TRITON_timeseries_written", "SWMM_node_timeseries_written", "SWMM_link_timeseries_written"):
        field = getattr(proc.log, name, None)
        if field is not None:
            field.set(False)
    return proc


def _record(a, proc, stem: str, value: bool) -> None:
    a.process._log_field(proc.log, f"{stem}_cleared").set(value)


def test_the_processed_state_predicate_of_each_model_reads_exactly_its_own_kind_record(golden_extractor):
    """Property (the rule as a quantified predicate): for every (model, kind) member of the domain, with that member's
    cleared record the ONLY evidence set True in the model's own log, the model's processed-state predicate of that kind
    is True and the predicate of every OTHER kind of the same model is False; with all records False every predicate is
    False.

    D = the six members (tritonswmm: node, link, triton; swmm: node, link; triton: triton), P(member) = "reads
    processed iff its own kind's record is True". The flagged set per member is {that member's predicate}; the match
    set is what the real predicates return. Input class: one record set at a time, in each model's real log, with the
    log's own processed records cleared so the cleared record is the only evidence. Reachability: the real
    ``_swmm_node_outputs_processed``, ``_swmm_link_outputs_processed`` and ``TRITON_outputs_processed`` over the real
    scenario paths. Both-states anchor: the assertion turns on which predicates answer True, true of the coupled
    members before this chunk. Must-fire control: the all-False row reads False for every predicate, so an always-True
    predicate fails there, and each set row requires its own predicate True, so an always-False one fails there.
    Killed by: reinstating a ``model == "tritonswmm"`` conjunct on a cleared read (the SWMM-only members then read
    False), or one predicate reading another kind's record (a second member reads True). Passes under a second correct
    implementation that keeps one combined record per model, if it is set for all kinds together.
    """
    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=False)
    kinds = {
        "tritonswmm": (
            "consolidated_node_timeseries",
            "consolidated_link_timeseries",
            "consolidated_triton_timeseries",
        ),
        "swmm": ("consolidated_node_timeseries", "consolidated_link_timeseries"),
        "triton": ("consolidated_triton_timeseries",),
    }

    def reads(proc, model: str) -> dict[str, bool]:
        out = {}
        if "consolidated_node_timeseries" in kinds[model]:
            out["consolidated_node_timeseries"] = proc._swmm_node_outputs_processed(
                "tritonswmm" if model == "tritonswmm" else "swmm"
            )
            out["consolidated_link_timeseries"] = proc._swmm_link_outputs_processed(
                "tritonswmm" if model == "tritonswmm" else "swmm"
            )
        if "consolidated_triton_timeseries" in kinds[model]:
            out["consolidated_triton_timeseries"] = proc.TRITON_outputs_processed
        return out

    for model, stems in kinds.items():
        proc = _sim_post_processing(a, 0, model)
        for stem in stems:
            _record(a, proc, stem, False)
        proc.log.processing_log.outputs.clear()
        assert not any(reads(proc, model).values()), f"control: {model} with no record reads processed"
        for stem in stems:
            _record(a, proc, stem, True)
            got = reads(proc, model)
            assert got[stem] is True, (model, stem, got)
            assert [s for s, v in got.items() if v and s != stem] == [], (model, stem, got)
            _record(a, proc, stem, False)


@pytest.mark.parametrize(
    "model, kind, group",
    [
        ("tritonswmm", "triton", "tritonswmm/triton/timeseries"),
        ("triton", "triton", "triton_only/triton/timeseries"),
        ("tritonswmm", "swmm", "analysis_datatree.zarr"),
        ("swmm", "swmm", "analysis_datatree.zarr"),
    ],
)
def test_a_summary_rebuild_from_a_cleared_family_store_is_refused_naming_the_consolidated_branch(
    golden_extractor, model, kind, group
):
    """Property: asking a model to rebuild its summary when its series store was cleared after consolidation and no
    summary is recorded refuses with a message that names the family's consolidated branch (the TRITON refusals name
    ``tritonswmm/triton/timeseries`` or ``triton_only/triton/timeseries`` by the model asked; the SWMM refusals name
    ``analysis_datatree.zarr``), while the same call with the cleared record False and the store equally absent refuses
    with a message that names no branch.

    D = the four summary exports the chunk changed (TRITON summary for the coupled and the TRITON-only model, SWMM
    summaries for the coupled and the SWMM-only model); P(member) = "cleared record True and store absent refuses,
    naming where the series went". Input class: the golden with the member's store deleted from disk and the
    scenario's processing log empty. Reachability: the real ``_export_TRITON_summary`` and ``_export_SWMM_summaries``
    over the real scenario paths; the rebuild is requested by a reprocess of a cleared scenario. Both-states anchor:
    the control row is the same absent store with the record False, so the assertion turns on whether the refusal
    names the branch, a property of the cleared record present before the chunk for the coupled members. Must-fire
    control: the cleared-False row must refuse and must not name the branch, so a gate that always names it, or a
    call that never refuses, fails one of the two rows. Killed by: removing the cleared gate from the SWMM-only or
    TRITON-only export (the cleared row then refuses without naming the branch), or naming the coupled branch for the
    TRITON-only model. Passes under a second correct implementation that raises another exception type from a shared
    helper, if the message names the branch. Not established: that the shape REQUIRES these two refusals to be
    family-generic; the chunk brief names the processed-state predicates, and the coder extended the same gate to the
    two summary exports (see the findings table).
    """
    import shutil

    from hhemt.exceptions import ProcessingError

    a = _ready(golden_extractor, families=("tritonswmm", "swmm", "triton"), clear=False)
    stems = {
        "triton": ["consolidated_triton_timeseries"],
        "swmm": ["consolidated_node_timeseries", "consolidated_link_timeseries"],
    }[kind]
    stores = {
        ("tritonswmm", "triton"): ["output_tritonswmm_triton_timeseries"],
        ("triton", "triton"): ["output_triton_only_timeseries"],
        ("tritonswmm", "swmm"): ["output_tritonswmm_node_time_series", "output_tritonswmm_link_time_series"],
        ("swmm", "swmm"): ["output_swmm_only_node_time_series", "output_swmm_only_link_time_series"],
    }[(model, kind)]
    for attr in stores:
        shutil.rmtree(_own(a, 0, attr))

    def attempt(cleared: bool):
        proc = _sim_post_processing(a, 0, model)
        for stem in stems:
            _record(a, proc, stem, cleared)
        if kind == "triton":
            return proc._export_TRITON_summary(model)
        return proc._export_SWMM_summaries("tritonswmm" if model == "tritonswmm" else "swmm")

    with pytest.raises((ProcessingError, FileNotFoundError)) as control:
        attempt(cleared=False)
    assert group not in str(control.value), f"control: the uncleared refusal must not name {group}: {control.value}"
    with pytest.raises((ProcessingError, FileNotFoundError)) as err:
        attempt(cleared=True)
    assert group in str(err.value), str(err.value)
