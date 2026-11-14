"""Analysis-level scenario records and the scenario-directory removal.

Problem this file solves: under the default configuration a scenario's directory is deleted after its results are
verified into the consolidated tree, so everything a later reader needs lives in one analysis-level JSON record. These
tests pin the properties that make that safe: the record precedes the delete, a resume never reads a half-deleted
directory, only the conjunction of three switches removes a directory, a predicate reads the record path-only, and a
removed sensitivity member answers only through a read-only view that fails loudly on any attribute it lacks.

No solver, no Snakemake and no synthetic analysis is built, so every test is laptop-admissible. The rows that need a
generated Snakefile or a constructed sensitivity master are in the compile tier on Rivanna
(``tests/test_synth_04_multisim_with_snakemake.py``) and are NOT covered here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import scenario_records as sr
from hhemt.scenario_records import (
    MemberRemovedError,
    RecordedMemberView,
    ScenarioRemovedError,
    all_members_for_reading,
    build_scenario_record,
    df_status_rows_from_record,
    event_ids_for_plan,
    event_is_removed,
    read_scenario_record,
    record_for,
    removal_switches_on,
    remove_scenario_directory,
    report_event_ids,
    scenario_stores_cleared,
    supersede_scenario_record,
    write_scenario_record,
)

EVENT = "event_index.0"


def _make_scenario_dir(analysis_dir: Path, event_id: str = EVENT, *, cleared: bool = True) -> Path:
    """A scenario directory with one model log, a scenario log, a DU sentinel and two files of payload."""
    sim = analysis_dir / "sims" / event_id
    (sim / "processed").mkdir(parents=True)
    (sim / "_status").mkdir()
    log = {"simulation_completed": True, "marker": "live-log"}
    if cleared:
        log["consolidated_summaries_cleared"] = {"state": "done", "stems": {"s": {"g": ["v"]}}}
    (sim / "log_tritonswmm.json").write_text(json.dumps(log))
    (sim / "scenario_prep_log.json").write_text(json.dumps({"scenario_creation_complete": True}))
    (sim / "_status" / "_du.json").write_text(json.dumps({"disk_utilization_bytes": 4096}))
    (sim / "processed" / "a.bin").write_bytes(b"x" * 10)
    (sim / "hydraulics.inp").write_text("inp")
    (analysis_dir / "_status").mkdir(exist_ok=True)
    return sim


def _fake_analysis(analysis_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir, simulation_directory=analysis_dir / "sims"),
        cfg_analysis=SimpleNamespace(target_processed_output_type="zarr"),
    )


# ---------------------------------------------------------------------------
# D22 order: the record precedes the removal.
# ---------------------------------------------------------------------------
def test_record_is_pending_when_the_directory_is_deleted_and_done_after(tmp_path, monkeypatch):
    """Property: the record exists in state pending at the moment of deletion and is done afterwards.

    Class: NEW CAPABILITY. Plausible wrong implementation: write the record after the delete (the landed clearing's
    order), which this spy sees as no record at delete time. Why required: the directory is the only source the
    record is built from, so a kill between delete and record loses the scenario's facts. Kills: moving the
    write_scenario_record call below delete_and_account. A second correct implementation: write pending, set the
    flag file, then delete, in one helper; the test still passes because it observes only the record at delete time.
    """
    sim = _make_scenario_dir(tmp_path)
    seen: dict = {}
    import hhemt.du_sentinels as du

    real = du.delete_and_account

    def spy(paths, **kw):
        seen["state_at_delete"] = (read_scenario_record(tmp_path, EVENT) or {}).get("state")
        seen["dir_present_at_delete"] = sim.exists()
        return real(paths, **kw)

    monkeypatch.setattr(du, "delete_and_account", spy)
    final = remove_scenario_directory(tmp_path, EVENT, sim, run_completed={"tritonswmm": True})
    assert final == "done"
    assert seen == {"state_at_delete": "pending", "dir_present_at_delete": True}
    assert not sim.exists()
    rec = read_scenario_record(tmp_path, EVENT)
    assert rec["state"] == "done"
    assert rec["du_bytes_before_removal"] == 4096
    assert "hydraulics.inp" in rec["listing"] and "processed/a.bin" in rec["listing"]
    assert rec["model_logs"]["tritonswmm"]["marker"] == "live-log"


def test_resume_of_a_pending_record_never_reads_the_half_deleted_directory(tmp_path):
    """Property: a pending record is finished from itself; the directory is only deleted, never re-read.

    Class: NEW CAPABILITY. Wrong implementation: rebuild the record from whatever is left of the directory, which
    overwrites the verified facts with a truncated listing. Control (must fire): the same half-deleted directory with
    NO record is built into a record that reflects the truncated directory, proving the builder does read it.
    Kills: calling build_scenario_record in the pending branch. A second correct implementation: finish removal in a
    separate function that only unlinks; the test still passes.
    """
    sim = _make_scenario_dir(tmp_path)
    write_scenario_record(tmp_path, EVENT, {**build_scenario_record(sim, EVENT, state="pending"), "sentinel": "kept"})
    (sim / "log_tritonswmm.json").unlink()  # half of the directory is already gone
    remove_scenario_directory(tmp_path, EVENT, sim)
    rec = read_scenario_record(tmp_path, EVENT)
    assert rec["state"] == "done" and rec["sentinel"] == "kept"
    assert "tritonswmm" in rec["model_logs"], "the verified log content must survive a resume over a damaged directory"
    assert not sim.exists()

    # control: no record, same damage -> the record is built from the (damaged) directory
    other = _make_scenario_dir(tmp_path, "event_index.1")
    (other / "log_tritonswmm.json").unlink()
    remove_scenario_directory(tmp_path, "event_index.1", other)
    assert "tritonswmm" not in read_scenario_record(tmp_path, "event_index.1")["model_logs"]


# ---------------------------------------------------------------------------
# D24 (a): the removal keys on the conjunction of three switches.
# ---------------------------------------------------------------------------
def _cfg(**over):
    base = {
        "toggle_clear_scenario_timeseries_after_consolidation": True,
        "clear_raw": "all",
        "remove_after_processing": "all",
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "over",
    [
        {"toggle_clear_scenario_timeseries_after_consolidation": False},
        {"clear_raw": "none"},
        {"clear_raw": ["swmm"]},
        {"remove_after_processing": "none"},
        {"remove_after_processing": ["timeseries"]},
    ],
)
def test_any_one_switch_off_keeps_the_directories(over):
    """Property: removal_switches_on is False when exactly one of the three switches is off.

    Class: NEW CAPABILITY. Wrong implementation: key on the clearing switch alone. Why: a user who wants to reprocess
    sets any one switch off and expects the directories to stay. Kills: dropping any one conjunct. A second correct
    implementation: evaluate the three predicates in another order. The all-on control follows.
    """
    assert removal_switches_on(_cfg(**over), ["tritonswmm"]) is False
    assert removal_switches_on(_cfg(), ["tritonswmm"]) is True  # must-fire control: all on removes


def test_electing_every_class_by_list_is_equivalent_to_all():
    """Property: a list naming all seven classes counts as ``all``; one naming six does not. Class: NEW CAPABILITY."""
    every = sorted(sr.ALL_REMOVE_CLASSES)
    assert removal_switches_on(_cfg(remove_after_processing=every), ["tritonswmm"]) is True
    assert removal_switches_on(_cfg(remove_after_processing=every[1:]), ["tritonswmm"]) is False
    assert removal_switches_on(_cfg(clear_raw=["tritonswmm", "swmm"]), ["tritonswmm", "swmm"]) is True


def test_a_scenario_whose_stores_are_not_all_cleared_keeps_its_directory(tmp_path):
    """Property: scenario_stores_cleared is False until every store is recorded cleared under a done record.

    Class: NEW CAPABILITY. Wrong implementation: treat a missing or pending record as cleared. Kills: returning True
    on state pending. Second correct implementation: read the field through record_for. Control: done -> True.
    """
    sim = _make_scenario_dir(tmp_path, cleared=False)
    assert scenario_stores_cleared(sim, ["tritonswmm"]) is False
    log = json.loads((sim / "log_tritonswmm.json").read_text())
    log["consolidated_summaries_cleared"] = {"state": "pending", "stems": {}}
    (sim / "log_tritonswmm.json").write_text(json.dumps(log))
    assert scenario_stores_cleared(sim, ["tritonswmm"]) is False
    log["consolidated_summaries_cleared"] = {"state": "done", "stems": {}}
    log["consolidated_node_timeseries_cleared"] = False
    (sim / "log_tritonswmm.json").write_text(json.dumps(log))
    assert scenario_stores_cleared(sim, ["tritonswmm"]) is False
    log["consolidated_node_timeseries_cleared"] = True
    (sim / "log_tritonswmm.json").write_text(json.dumps(log))
    assert scenario_stores_cleared(sim, ["tritonswmm"]) is True  # must-fire control
    assert scenario_stores_cleared(tmp_path / "sims" / "no_such", ["tritonswmm"]) is False


# ---------------------------------------------------------------------------
# The resolver and the path-only predicates.
# ---------------------------------------------------------------------------
def test_record_for_reads_the_live_log_then_the_record(tmp_path):
    """Property: record_for answers from the live log before removal and from the record after.

    Class: NEW CAPABILITY. Wrong implementation: always read the directory, which returns None after removal. Kills:
    deleting the record branch. Second correct implementation: copy the log into the record and drop the live read for
    done records only. A pending record also reads the record (the directory is never a source again).
    """
    sim = _make_scenario_dir(tmp_path)
    analysis = _fake_analysis(tmp_path)
    assert record_for(analysis, EVENT, "tritonswmm")["marker"] == "live-log"
    assert record_for(analysis, EVENT, "swmm") is None
    remove_scenario_directory(tmp_path, EVENT, sim)
    assert not sim.exists()
    assert record_for(analysis, EVENT, "tritonswmm")["marker"] == "live-log"


def test_summary_predicate_is_true_for_a_removed_scenario_without_constructing_one(tmp_path, monkeypatch):
    """Property: scenario_summaries_present is True for a removed scenario and builds no TRITONSWMM_scenario.

    Class: NEW CAPABILITY (the predicate read before directory paths, which are absent after removal). Wrong
    implementation: construct the scenario to read its log. The constructor is patched to raise. Control (must fire):
    an event with no record and no files is False. Kills: removing the event_is_removed early return. Second correct
    implementation: resolve through summary_clearing_record, which also reads the record.
    """
    from hhemt import scenario as scenario_module
    from hhemt.summary_paths import scenario_summaries_present, summary_clearing_record

    def boom(*a, **k):
        raise AssertionError("the predicate must be path-only")

    monkeypatch.setattr(scenario_module, "TRITONSWMM_scenario", boom)
    sim = _make_scenario_dir(tmp_path)
    analysis = _fake_analysis(tmp_path)
    remove_scenario_directory(tmp_path, EVENT, sim)
    assert scenario_summaries_present(analysis, EVENT, ["tritonswmm"]) is True
    assert summary_clearing_record(tmp_path / "sims" / EVENT, "tritonswmm")["state"] == "done"
    assert scenario_summaries_present(analysis, "event_index.9", ["tritonswmm"]) is False


def test_the_summary_branch_guard_reads_the_records_of_removed_scenarios(tmp_path):
    """Property: records_summary_clearing is True for an analysis whose only evidence is the record, False with none.

    Class: REGRESSION on the glob-only reader (it globbed sims/*/log_*.json, which finds nothing after removal and
    would let delete_summary_branches delete the only copy of the summaries). Kills: reverting to the log glob.
    Second correct implementation: read the records through record_for per event.
    """
    from hhemt.delete_reprocess_zarr_runner import records_summary_clearing

    sim = _make_scenario_dir(tmp_path)
    assert records_summary_clearing(tmp_path) is True  # live log with a done clearing record
    remove_scenario_directory(tmp_path, EVENT, sim)
    assert not (tmp_path / "sims" / EVENT).exists()
    assert records_summary_clearing(tmp_path) is True
    (tmp_path / "_status" / "scenario_records" / f"{EVENT}.json").unlink()
    assert records_summary_clearing(tmp_path) is False  # must-fire control: no record, no directory


# ---------------------------------------------------------------------------
# D24 (g): the planning rule per event state.
# ---------------------------------------------------------------------------
def test_sim_ids_exclude_removed_and_pending_but_report_ids_keep_every_event(tmp_path):
    """Property: removed (done) and pending events are out of SIM_IDS; superseded, failed and never-run are in.

    Class: NEW CAPABILITY. Wrong implementation: filter on directory presence, which would drop an incomplete event
    whose directory was deleted by hand and keep a pending one. Report ids keep every event so a render-floor force
    re-plans each page. Kills: reading ``pending`` as present. Second correct implementation: read the states from
    one listing of the records directory instead of per-event reads.
    """
    ids = ["e_done", "e_pending", "e_superseded", "e_failed", "e_never"]
    write_scenario_record(tmp_path, "e_done", {"state": "done"})
    write_scenario_record(tmp_path, "e_pending", {"state": "pending"})
    write_scenario_record(tmp_path, "e_superseded", {"state": "superseded"})
    assert event_ids_for_plan(tmp_path, ids) == ["e_superseded", "e_failed", "e_never"]
    assert report_event_ids(ids) == ids
    assert event_is_removed(tmp_path, "e_pending") and not event_is_removed(tmp_path, "e_superseded")


def test_generation_reads_a_pending_record_and_performs_no_filesystem_act(tmp_path):
    """Property: planning over a pending record deletes nothing and writes nothing (D24 (g), row 'removal interrupted').

    Class: NEW CAPABILITY. Wrong implementation: finish the removal at generation, which would run ahead of the lock
    check and the in-flight reconcile. Observed by a before/after walk of the analysis directory.
    """
    sim = _make_scenario_dir(tmp_path)
    write_scenario_record(tmp_path, EVENT, build_scenario_record(sim, EVENT, state="pending"))

    def walk():
        return sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    before = walk()
    assert event_ids_for_plan(tmp_path, [EVENT]) == []
    assert walk() == before and sim.exists()


def test_a_run_stage_force_supersedes_a_removed_record_and_only_a_removed_one(tmp_path):
    """Property: supersede changes done and pending records to superseded, and returns False for none or superseded.

    Class: NEW CAPABILITY. Kills: superseding an absent record (inventing one). Second correct implementation:
    supersede through write_scenario_record with a merged payload. Planning then includes the event.
    """
    write_scenario_record(tmp_path, "e1", {"state": "done"})
    assert supersede_scenario_record(tmp_path, "e1") is True
    assert read_scenario_record(tmp_path, "e1")["state"] == "superseded"
    assert supersede_scenario_record(tmp_path, "e1") is False
    assert supersede_scenario_record(tmp_path, "never") is False
    assert event_ids_for_plan(tmp_path, ["e1"]) == ["e1"]


def test_a_malformed_record_reads_as_absent(tmp_path):
    """Property: a record with an unknown state, bad JSON or a non-mapping reads as no record (never as removed)."""
    d = tmp_path / "_status" / "scenario_records"
    d.mkdir(parents=True)
    (d / "e1.json").write_text("{not json")
    (d / "e2.json").write_text(json.dumps({"state": "weird"}))
    (d / "e3.json").write_text(json.dumps([1, 2]))
    assert [read_scenario_record(tmp_path, e) for e in ("e1", "e2", "e3")] == [None, None, None]
    with pytest.raises(ValueError):
        write_scenario_record(tmp_path, "e4", {"state": "weird"})


# ---------------------------------------------------------------------------
# df_status and the disk-use report read the record.
# ---------------------------------------------------------------------------
def test_df_status_rows_of_a_removed_scenario_are_never_blank(tmp_path):
    """Property: a removed event yields one row per enabled model carrying the captured verdict and bytes.

    Class: NEW CAPABILITY. Wrong implementation: skip removed events (the row vanishes from scenario_status.csv) or
    emit NaN for run_completed. Captured rows win over the minimal row. Kills: dropping the recorded verdict.
    """
    sim = _make_scenario_dir(tmp_path)
    remove_scenario_directory(tmp_path, EVENT, sim, run_completed={"tritonswmm": True, "swmm": False})
    rec = read_scenario_record(tmp_path, EVENT)
    rows = df_status_rows_from_record(rec, {"w": 1}, 0, ["tritonswmm", "swmm"], "d")
    assert [(r["model_type"], r["run_completed"], r["disk_utilization_bytes"]) for r in rows] == [
        ("tritonswmm", True, 4096),
        ("swmm", False, 4096),
    ]
    rec["captured"] = {"df_status_rows": [{"model_type": "tritonswmm", "run_completed": True, "actual_gpus": 2}]}
    rows = df_status_rows_from_record(rec, {"w": 1}, 0, ["tritonswmm", "swmm"], "d")
    assert rows == [{"w": 1, "model_type": "tritonswmm", "run_completed": True, "actual_gpus": 2, "event_iloc": 0}]


def test_removed_scenario_bytes_come_from_the_records(tmp_path):
    """Property: removed_scenario_bytes reports each removed event's bytes at removal; superseded events are absent."""
    from hhemt.du_sentinels import removed_scenario_bytes

    sim = _make_scenario_dir(tmp_path)
    remove_scenario_directory(tmp_path, EVENT, sim)
    write_scenario_record(tmp_path, "e_sup", {"state": "superseded", "du_bytes_before_removal": 7})
    assert removed_scenario_bytes(tmp_path) == {EVENT: 4096}


# ---------------------------------------------------------------------------
# Removed members: the read-only view and the loud gap.
# ---------------------------------------------------------------------------
def _view(**kw) -> RecordedMemberView:
    import pandas as pd

    return RecordedMemberView(
        "3",
        "member_3",
        Path("/master"),
        df_sims=pd.DataFrame({"x": [1]}),
        cfg_analysis=SimpleNamespace(),
        enabled_models=["swmm"],
        weather_indexer=lambda i: {"x": i},
        **kw,
    )


def test_a_view_provides_its_attributes_and_raises_loudly_on_any_other():
    """Property: every provided attribute answers; any other read raises MemberRemovedError naming member and attribute.

    Class: NEW CAPABILITY. Wrong implementation: return None or a default for an unknown attribute, which turns a gap
    into a silent zero. The view is also read-only (assignment raises). Control: provided names do not raise.
    """
    v = _view()
    assert v._get_enabled_model_types() == ["swmm"] and v._retrieve_weather_indexer_using_integer_index(0) == {"x": 0}
    assert v.member_id == "3" and v.analysis_id == "member_3" and len(v.df_sims) == 1
    with pytest.raises(MemberRemovedError, match=r"member '3'|member\s+'3'|'3'.*analysis_paths|analysis_paths"):
        _ = v.analysis_paths
    with pytest.raises(MemberRemovedError):
        v.some_future_attribute = 1
    with pytest.raises(MemberRemovedError, match="df_status"):
        _ = v.df_status  # no builder supplied: loud, not an empty frame


def test_all_members_for_reading_is_active_then_recorded_and_refuses_overlap():
    """Property: the merged mapping lists active members then views; a member in both is a defect.

    Kills: silently letting the view shadow an active member. Control: disjoint keys merge in order.
    """
    merged = all_members_for_reading({"1": "a1"}, {"2": "v2"})
    assert list(merged) == ["1", "2"]
    with pytest.raises(ValueError, match="both active and recorded"):
        all_members_for_reading({"1": "a"}, {"1": "v"})


def test_a_removed_scenario_handle_components_raise_naming_the_event():
    """Property: a removed scenario's directory-backed components raise ScenarioRemovedError naming the event."""
    h = sr._RemovedScenarioHandle("event_index.4")
    with pytest.raises(ScenarioRemovedError, match="event_index.4"):
        h.refresh()
    with pytest.raises(AttributeError):
        _ = h.__deepcopy__  # dunder probes must not be swallowed as removal errors


def test_the_member_directory_removal_order_is_marker_pending_then_delete_then_done(tmp_path):
    """Property: move_member_records writes the marker ``pending`` and copies records, yaml and log before any delete.

    Class: NEW CAPABILITY. Wrong implementation: write the marker ``done`` at move time, which would make a kill
    during the member-directory delete look finished. The member directory is not touched by the move.
    """
    master = tmp_path
    member = master / "members" / "member_3"
    (member / "_status" / "scenario_records").mkdir(parents=True)
    (member / "_status" / "scenario_records" / "e1.json").write_text(json.dumps({"state": "done"}))
    (member / "member_3.yaml").write_text("a: 1")
    (member / "analysis_log.json").write_text("{}")
    sr.move_member_records(
        member,
        master,
        "3",
        analysis_id="member_3",
        yaml_name="member_3.yaml",
        analysis_log=member / "analysis_log.json",
    )
    assert sr.read_member_record(master, "3")["state"] == "pending"
    assert not sr.member_is_recorded(master, "3")
    dest = sr.member_records_dir(master, "3")
    assert (dest / "member_3.yaml").read_text() == "a: 1" and (dest / "analysis_log.json").exists()
    assert sr.recorded_member_event_done(master, "3", "e1") is True
    assert member.exists()
    rec = sr.read_member_record(master, "3")
    rec["state"] = "done"
    sr.write_member_record(master, "3", rec)
    assert sr.member_is_recorded(master, "3")


def test_the_member_reprocess_delete_runner_creates_nothing_for_an_absent_member(tmp_path, capsys):
    """Property: delete_member_reprocess_runner.main exits 0 and creates no path when the member directory is absent.

    Class: REGRESSION (the landed runner wrote a submission sentinel and a completion flag under the member
    directory, re-creating a removed member). Observed by listing the master before and after.
    """
    from hhemt import delete_member_reprocess_runner as runner

    missing = tmp_path / "members" / "member_9"
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert runner.main(["--member-id", "9", "--analysis-dir", str(missing)]) == 0
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    assert not missing.exists()
    assert "absent" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Round 2: the census of what a removed member's view must provide, and the dynamic arm.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent / "src" / "hhemt"
_CENSUS_SITES = (
    "analysis_validation.py",
    "report_renderers/per_analysis_summary.py",
    "report_renderers/sensitivity_benchmarking.py",
    "report_renderers/system_overview.py",
    "report_renderers/_cli.py",
    "workflow.py",
    "sensitivity_analysis.py",
)
#: Names a class-(2) site reads that the view deliberately does NOT provide: each is read only in a function that
#: first tests ``is_recorded_member`` and takes the master-tree arm (asserted below, so an unguarded read fails).
_ADAPTED_NAMES = frozenset({"analysis_paths", "process", "_system"})


def _view_attribute_census() -> dict[str, set[str]]:
    """Attribute names read on the member variable of every class-(2) loop, followed through calls to a fixed point.

    Class-(2) loops iterate ``_all_members_for_reading()`` or ``_iter_members_or_self(...)``. A member passed whole to a
    same-module function contributes the names that function reads on the matching parameter.
    """
    import ast

    def attrs_on(node, var):
        return {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == var
        }

    def passes(node, var):
        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                for i, a in enumerate(n.args):
                    if isinstance(a, ast.Name) and a.id == var:
                        f = n.func
                        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                        if name:
                            out.append((name, i))
        return out

    def closure(fmap, node, var, seen):
        names = attrs_on(node, var)
        for fname, i in passes(node, var):
            if fname in fmap and (fname, i) not in seen:
                seen.add((fname, i))
                params = [a.arg for a in fmap[fname].args.args if a.arg != "self"]
                if i < len(params):
                    names |= closure(fmap, fmap[fname], params[i], seen)
        return names

    union: dict[str, set[str]] = {}
    for rel in _CENSUS_SITES:
        tree = ast.parse((_SRC / rel).read_text())
        fmap = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for n in ast.walk(tree):
            if isinstance(n, ast.For):
                src = ast.unparse(n.iter)
                if "all_members_for_reading" in src or "_iter_members_or_self" in src:
                    tgt = n.target
                    var = tgt.elts[-1].id if isinstance(tgt, ast.Tuple) else tgt.id
                    for stmt in n.body:
                        for a in closure(fmap, stmt, var, set()):
                            union.setdefault(a, set()).add(f"{rel}:{n.lineno}")
    return union


def test_the_view_provides_every_attribute_the_class_two_sites_read():
    """Property: every attribute name read on a member at a class-(2) whole-experiment site is provided by the view or
    is an adapted name whose reading function tests ``is_recorded_member``.

    Class: NEW CAPABILITY. Wrong implementation: a view that omits a name a site reads (the gap this census was built
    for: the first round's view lacked five validation names). Must-fire control: the census is non-empty, contains the
    two names the shape measured at the page-pair site and the master ``df_status`` loop, and the matcher run over a
    source with the guard removed reports the adapted name unguarded. Kills: deleting a name from ``_PROVIDED``.
    Second correct implementation: provide the name through ``__getattr__`` delegation to a record reader; the test
    still passes because it checks ``_PROVIDED``, which that implementation must also list.
    """
    union = _view_attribute_census()
    assert {"_retrieve_weather_indexer_using_integer_index", "df_sims", "cfg_analysis", "df_status"} <= set(union)
    missing = set(union) - RecordedMemberView._PROVIDED - _ADAPTED_NAMES
    assert not missing, f"the view lacks {sorted(missing)}, read at {[sorted(union[m])[:2] for m in sorted(missing)]}"
    validation = (_SRC / "analysis_validation.py").read_text()
    assert "is_recorded_member" in validation, "adapted names are read only behind an is_recorded_member test"


def test_the_members_string_matcher_finds_the_two_getattr_sites():
    """Property: the second matcher (a ``members`` mapping reached through a string) finds exactly the two files the
    shape classifies, so the census of forms found today is reproducible. Control: the matcher returns non-zero.
    """
    import re

    hits = sorted(
        str(p.relative_to(_SRC))
        for p in _SRC.rglob("*.py")
        if re.search(r"""["']members["']""", p.read_text()) and "version_migration" not in p.parts
    )
    # processing_system.py is class (1) by decision (active-only list) and keeps its string reach; system_overview.py
    # was the class-(2) site and now iterates every member, so the matcher no longer finds it there.
    assert "processing_system.py" in hits and "report_renderers/system_overview.py" not in hits


def _fake_master(tmp_path: Path, *, with_view: bool = True):
    """A master whose member 3 is removed: the master's analysis object, its sensitivity and one view."""
    import pandas as pd

    master = tmp_path / "master"
    rec = sr.member_records_dir(master, "3") / "scenario_records"
    rec.mkdir(parents=True)
    (rec / "event_index.0.json").write_text(
        json.dumps(
            {"state": "done", "model_logs": {"swmm": {"simulation_completed": True}}, "run_completed": {"swmm": True}}
        )
    )
    sr.write_member_record(master, "3", {"state": "done"})
    view = RecordedMemberView(
        "3",
        "member_3",
        master,
        df_sims=pd.DataFrame({"e": [0]}),
        cfg_analysis=SimpleNamespace(weather_time_series_timestep_dimension_name="t"),
        enabled_models=["swmm"],
        weather_indexer=lambda i: {"event_index": i},
        sim_run_reader=lambda i: sr.recorded_member_sim_run(master, "3", "member_3", "event_index.0"),
        sim_run_processing_reader=lambda i: sr.recorded_member_sim_run(master, "3", "member_3", "event_index.0"),
    )
    sens = SimpleNamespace(members={}, recorded_members={"3": view}, _all_members_for_reading=lambda: {"3": view})
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        sensitivity=sens,
        analysis_paths=SimpleNamespace(
            analysis_dir=master,
            experiment_datatree_zarr=master / "sensitivity_datatree.zarr",
            simulation_directory=master / "sims",
        ),
    )
    return analysis


@pytest.mark.parametrize("check", ["check_scenarios_setup", "check_scenarios_run", "check_timeseries_processed"])
def test_validation_sites_run_against_a_removed_member_without_member_removed_error(tmp_path, check):
    """DYNAMIC ARM. Property: each class-(2) validation check executes against a master with one removed member and
    raises no ``MemberRemovedError``; a name the view lacks surfaces here whatever the static census found.

    Class: NEW CAPABILITY. Control (must fire): the same check over a view with an attribute deleted from
    ``_PROVIDED`` raises, shown for ``n_scenarios`` below. Kills: removing the view's ``n_scenarios``.
    """
    from hhemt import analysis_validation as av

    analysis = _fake_master(tmp_path)
    result = getattr(av, check)(analysis)
    assert result.name


def test_the_analysis_summary_check_tests_the_masters_group_for_a_removed_member(tmp_path):
    """Property: a removed member counts as present iff the master's ``member_{id}`` group exists; absent -> reported.

    Class: NEW CAPABILITY. Wrong implementation: read ``analysis_paths`` on the view (raises) or skip the member.
    """
    from hhemt import analysis_validation as av

    analysis = _fake_master(tmp_path)
    zarr = analysis.analysis_paths.experiment_datatree_zarr
    zarr.mkdir(parents=True)
    assert [d for d in av.check_analysis_summaries_created(analysis).details if "member_3" in d["detail"]]
    (zarr / "member_3").mkdir()
    assert not [d for d in av.check_analysis_summaries_created(analysis).details if "member_3" in d["detail"]]


def test_a_view_without_a_provided_name_fails_the_dynamic_arm(tmp_path):
    """Control: a view that cannot answer ``n_scenarios`` makes the check raise."""
    from hhemt import analysis_validation as av

    analysis = _fake_master(tmp_path)
    view = analysis.sensitivity.recorded_members["3"]

    class _Gap(RecordedMemberView):
        n_scenarios = property(lambda self: (_ for _ in ()).throw(MemberRemovedError(self.member_id, "n_scenarios")))

    gap = _Gap(
        "3", "member_3", view.master_dir, df_sims=view.df_sims, cfg_analysis=view.cfg_analysis, enabled_models=["swmm"]
    )
    analysis.sensitivity._all_members_for_reading = lambda: {"3": gap}
    with pytest.raises(MemberRemovedError):
        av.check_scenarios_setup(analysis)


def test_a_force_naming_a_recorded_member_refuses_with_configuration_error(tmp_path):
    """Property: scope member naming a recorded member refuses (any stage); scope all refuses at simulate/process and
    passes at consolidate/render; scope event plans active members only. Class: NEW CAPABILITY.
    Wrong implementation: the landed ``RuntimeError("... is not in the members dict")``. Kills: dropping the check.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sa

    refuse = sa._refuse_force_on_recorded_members
    me = SimpleNamespace(recorded_members={"3": object()})
    for stage in ("simulate", "process", "consolidate", "render"):
        with pytest.raises(ConfigurationError, match="member"):
            refuse(me, SimpleNamespace(scope="member", tokens=("3",), stage=stage))
    with pytest.raises(ConfigurationError):
        refuse(me, SimpleNamespace(scope="all", tokens=(), stage="process"))
    refuse(me, SimpleNamespace(scope="all", tokens=(), stage="consolidate"))
    refuse(me, SimpleNamespace(scope="event", tokens=("e",), stage="simulate"))
    refuse(me, SimpleNamespace(scope="member", tokens=("9",), stage="simulate"))  # an active member is not refused
    refuse(SimpleNamespace(recorded_members={}), SimpleNamespace(scope="all", tokens=(), stage="simulate"))


def test_data_availability_reads_a_removed_scenarios_record_and_lists_surviving_directories(tmp_path):
    """Property: check_data_availability reads the model log through the resolver for a removed scenario and names the
    surviving directories with their reasons in its summary. Class: NEW CAPABILITY. Control: a live, uncompleted
    scenario directory is listed as ``not completed``; a removed one is not listed.
    """
    import pandas as pd

    from hhemt import analysis_validation as av

    keep = _make_scenario_dir(tmp_path, "event_index.1", cleared=False)
    (keep / "log_tritonswmm.json").write_text(json.dumps({"simulation_completed": False}))
    gone = _make_scenario_dir(tmp_path, EVENT)
    remove_scenario_directory(tmp_path, EVENT, gone, run_completed={"tritonswmm": True})
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=False,
            toggle_clear_scenario_timeseries_after_consolidation=True,
            clear_raw="all",
            remove_after_processing="all",
            target_processed_output_type="zarr",
            weather_event_indices=["event_index"],
        ),
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path, simulation_directory=tmp_path / "sims"),
        df_sims=pd.DataFrame({"event_index": [0, 1]}),
        _get_enabled_model_types=lambda: ["tritonswmm"],
    )
    analysis._retrieve_weather_indexer_using_integer_index = lambda i: {"event_index": i}
    surviving = av.surviving_directories_with_reasons(analysis)
    assert surviving == {"event_index.1": "not completed"}
    assert "Surviving scenario directories: 1 not completed" in av.check_data_availability(analysis).summary


# ---------------------------------------------------------------------------
# Round 3: the dynamic arm over renderer and workflow sites, and the two-member master.
# ---------------------------------------------------------------------------
def _two_member_master(tmp_path: Path):
    """A master with member 3 removed (a SWMM-only view with a captured elapsed time) and member 4 active."""
    import pandas as pd

    analysis = _fake_master(tmp_path)
    master = analysis.analysis_paths.analysis_dir
    rec_file = sr.member_records_dir(master, "3") / "scenario_records" / "event_index.0.json"
    rec = json.loads(rec_file.read_text())
    rec["captured"] = {"swmm_total_elapsed": 12.5}
    rec_file.write_text(json.dumps(rec))
    removed = analysis.sensitivity.recorded_members["3"]

    def _status_rows():
        import pandas as pd

        record = json.loads(rec_file.read_text())
        return pd.DataFrame(df_status_rows_from_record(record, {"event_index": 0}, 0, ["swmm"], "d"))

    object.__setattr__(removed, "_df_status_builder", _status_rows)

    class _ActiveRun:
        is_recorded = False
        scen_paths = SimpleNamespace(swmm_full_rpt_file=None)

        def model_run_completed(self, mt):
            return True

        _scenario = None

    run = _ActiveRun()
    run._scenario = run
    active = SimpleNamespace(
        df_sims=pd.DataFrame({"event_index": [0, 1]}),
        analysis_id="member_4",
        analysis_paths=SimpleNamespace(
            analysis_dir=master / "members" / "member_4", simulation_directory=master / "members" / "member_4" / "sims"
        ),
        cfg_analysis=SimpleNamespace(weather_time_series_timestep_dimension_name="t"),
        _get_enabled_model_types=lambda: ["swmm"],
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_index": i},
        _retrieve_sim_runs=lambda i: run,
        _retrieve_sim_run_processing_object=lambda i: run,
        n_scenarios=2,
        _all_scenarios_created=True,
        _scenarios_not_created=[],
        _all_sims_run=True,
        _scenarios_not_run=[],
        df_status=pd.DataFrame({"event_iloc": [0, 1], "model_type": ["swmm", "swmm"], "run_completed": [True, True]}),
        process=SimpleNamespace(),
    )
    members = {"4": active}
    recorded = {"3": removed}
    sens = analysis.sensitivity
    sens.members = members
    sens.recorded_members = recorded
    sens._all_members_for_reading = lambda: all_members_for_reading(members, recorded)
    sens.df_setup = pd.DataFrame({"x": [1, 2]}, index=["3", "4"])
    return analysis, removed, active


def test_renderer_sites_run_against_a_removed_member_and_read_both_members(tmp_path):
    """DYNAMIC ARM, renderer sites. Property: the per-analysis summary verdicts, the benchmarking wall label, model arm
    and row collection execute against a master with one removed and one active member, raise no MemberRemovedError,
    and the benchmarking rows carry BOTH members (the removed member's value read from its record).

    Class: NEW CAPABILITY. Must-fire control: the active-only count would be 1 row (member 4 has no readable report in
    this fixture, so the removed member's row is the discriminator); a row set without sa_id '3' is the failure.
    Kills: iterating ``members`` instead of ``_all_members_for_reading()`` at the benchmarking loop.
    """
    from hhemt.report_renderers import per_analysis_summary as pas
    from hhemt.report_renderers import sensitivity_benchmarking as sb

    analysis, removed, active = _two_member_master(tmp_path)
    assert pas._is_scenario_successful(removed, 0) is True and pas._is_scenario_pending(removed, 0) is False
    assert pas._is_scenario_successful(active, 1) is True
    assert sb._wall_label(analysis.sensitivity) == "SWMM elapsed time"
    assert sb._resolve_model_arm(analysis) is None
    rows, sources = sb._collect_rows(analysis, "performance.Total")
    assert {r["sa_id"] for r in rows} == {"3"} and rows[0]["value"] == 12.5
    assert any(str(s).endswith("event_index.0.json") for s in sources), "the declared source is the record file"


def test_workflow_class_two_sites_run_against_a_removed_member(tmp_path):
    """DYNAMIC ARM, workflow site ``_enumerate_member_event_pairs``. Property: both members' (member, event) pairs are
    enumerated (the removed member's from its view); a view that lacks the weather indexer is LOUD (MemberRemovedError),
    not a silent empty pair list. Control: with the indexer removed the generic ``except Exception`` would have
    swallowed it; the test fails if the raise is absent. Kills: deleting ``except MemberRemovedError: raise``.
    """
    from hhemt.workflow import SensitivityAnalysisWorkflowBuilder as B

    analysis, removed, active = _two_member_master(tmp_path)
    fake_self = SimpleNamespace(sensitivity_analysis=analysis.sensitivity)
    members, events = B._enumerate_member_event_pairs(fake_self, context_label="test")
    assert sorted(set(members)) == ["3", "4"] and len(members) == 3 and len(events) == 3

    gap = RecordedMemberView(
        "5", "member_5", Path("/m"), df_sims=removed.df_sims, cfg_analysis=SimpleNamespace(), enabled_models=["swmm"]
    )
    analysis.sensitivity._all_members_for_reading = lambda: {"5": gap}
    with pytest.raises(MemberRemovedError):
        B._enumerate_member_event_pairs(fake_self, context_label="test")


def test_master_status_and_summary_checks_read_both_members(tmp_path):
    """TWO-MEMBER ROWS. Property: the master ``df_status`` has rows for both members (the removed member's from its
    moved records, with the recorded verdict) and ``check_scenarios_setup`` / ``check_scenarios_run`` count both.
    Class: NEW CAPABILITY. Wrong implementation: master status iterating active members only (one member's rows).
    """
    from hhemt import analysis_validation as av
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sa

    analysis, removed, active = _two_member_master(tmp_path)
    sens = analysis.sensitivity
    sens.df_setup = sens.df_setup  # index "3","4"
    for m in (removed,):
        m.cfg_analysis.is_experiment_member = True
    active.cfg_analysis.is_experiment_member = True
    df = sa.df_status.fget(sens)
    assert sorted(df["member_id"].unique()) == ["3", "4"]
    removed_rows = df[df["member_id"] == "3"]
    assert bool(removed_rows["run_completed"].iloc[0]) is True
    assert av.check_scenarios_setup(analysis).passed and av.check_scenarios_run(analysis).passed


# ---------------------------------------------------------------------------
# Round 4: the removal step after the last clearing, the kept INPs, the performance disclosure row.
# ---------------------------------------------------------------------------
def _removal_host(tmp_path: Path, events=("event_index.0", "event_index.1"), *, switches=True):
    """A stand-in ``TRITONSWMM_analysis_post_processing`` host carrying only what the removal step reads."""
    import pandas as pd

    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as PP

    cfg = SimpleNamespace(
        toggle_clear_scenario_timeseries_after_consolidation=switches,
        clear_raw="all",
        remove_after_processing="all",
    )
    analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path, simulation_directory=tmp_path / "sims"),
        cfg_analysis=cfg,
        df_sims=pd.DataFrame({"e": range(len(events))}),
        _get_enabled_model_types=lambda: ["tritonswmm"],
    )
    host = SimpleNamespace(_analysis=analysis)
    host._event_slug = lambda i: events[int(i)]
    host._event_sim_folder = lambda i: PP._event_sim_folder(host, i)
    return host, PP


def _patch_scenario(monkeypatch, completed: dict[str, bool]):
    """Replace the scenario constructor the step uses with a path-only stand-in (no directory is created)."""
    from hhemt import processing_analysis as pa

    class _Scen:
        def __init__(self, event_iloc, analysis):
            self.scen_paths = SimpleNamespace(swmm_full_rpt_file=None)
            self._evt = f"event_index.{event_iloc}"

        def model_run_completed(self, m):
            return completed[self._evt]

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", _Scen)


def test_removal_step_removes_only_fully_cleared_completed_scenarios(tmp_path, monkeypatch):
    """Property: of two scenarios, the one whose stores are cleared and whose run completed is removed (record done,
    directory gone, INPs kept); the one that FAILED keeps its directory and gets no record.

    Class: NEW CAPABILITY. Wrong implementation: remove every scenario once the switches are on. Must-fire control:
    the failed scenario flips to removed when its run is marked completed. Kills: dropping the completion conjunct.
    Second correct implementation: gate on the model log's simulation_completed field instead of model_run_completed.
    """
    host, PP = _removal_host(tmp_path)
    for evt in ("event_index.0", "event_index.1"):
        sim = _make_scenario_dir(tmp_path, evt)
        (sim / "swmm").mkdir()
        (sim / "swmm" / "hydraulics.inp").write_text("[TITLE]")
    _patch_scenario(monkeypatch, {"event_index.0": True, "event_index.1": False})
    assert PP._remove_cleared_scenario_directories(host) == ["event_index.0"]
    assert not (tmp_path / "sims" / "event_index.0").exists() and (tmp_path / "sims" / "event_index.1").is_dir()
    assert read_scenario_record(tmp_path, "event_index.0")["state"] == "done"
    assert read_scenario_record(tmp_path, "event_index.1") is None
    assert sr.kept_inp_paths(tmp_path)["hydraulics"].read_text() == "[TITLE]"
    _patch_scenario(monkeypatch, {"event_index.0": True, "event_index.1": True})
    assert PP._remove_cleared_scenario_directories(host) == ["event_index.1"]  # control


def test_removal_step_waits_for_every_store_and_every_switch(tmp_path, monkeypatch):
    """Property: a scenario with a zarr store still under ``processed/`` (a hydrology or performance store no log names)
    keeps its directory; with a switch off nothing is removed. Control: removing the leftover store removes it.
    Kills: gating only on the log fields.
    """
    host, PP = _removal_host(tmp_path, ("event_index.0",))
    sim = _make_scenario_dir(tmp_path)
    (sim / "processed" / "SWMM_hydro_tseries.zarr").mkdir()
    _patch_scenario(monkeypatch, {"event_index.0": True})
    assert PP._remove_cleared_scenario_directories(host) == [] and sim.is_dir()
    (sim / "processed" / "SWMM_hydro_tseries.zarr").rmdir()
    off, _ = _removal_host(tmp_path, ("event_index.0",), switches=False)
    assert PP._remove_cleared_scenario_directories(off) == [] and sim.is_dir()
    assert PP._remove_cleared_scenario_directories(host) == ["event_index.0"]


def test_removal_step_finishes_a_pending_record_without_reading_the_directory(tmp_path, monkeypatch):
    """Property: a ``pending`` record is finished (directory deleted, ``done``) even if the stores check would now fail.
    Class: NEW CAPABILITY. Kills: re-verifying a pending scenario against its half-deleted directory.
    """
    host, PP = _removal_host(tmp_path, ("event_index.0",))
    sim = _make_scenario_dir(tmp_path, cleared=False)  # would fail scenario_stores_cleared
    write_scenario_record(tmp_path, "event_index.0", build_scenario_record(sim, "event_index.0", state="pending"))
    _patch_scenario(monkeypatch, {"event_index.0": False})
    assert PP._remove_cleared_scenario_directories(host) == ["event_index.0"]
    assert not sim.exists() and read_scenario_record(tmp_path, "event_index.0")["state"] == "done"


def test_system_overview_reads_the_kept_inps_when_every_directory_is_gone(tmp_path):
    """Property: with no scenario directory, ``_resolve_inp_sources`` returns the kept copies; with no kept
    copies it still raises FileNotFoundError. Class: REGRESSION on the all-removed master. Control: the raise.
    """
    from hhemt.report_renderers.system_overview import _resolve_inp_sources

    scen_paths = SimpleNamespace(swmm_full_inp=None, swmm_hydro_inp=None, swmm_hydraulics_inp=tmp_path / "gone.inp")
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        _retrieve_sim_runs=lambda i: SimpleNamespace(_scenario=SimpleNamespace(scen_paths=scen_paths)),
    )
    with pytest.raises(FileNotFoundError):
        _resolve_inp_sources(analysis)
    sim = _make_scenario_dir(tmp_path)
    (sim / "swmm").mkdir()
    (sim / "swmm" / "full.inp").write_text("full")
    sr.capture_system_overview_inps(sim, tmp_path)
    hydro, hydraulics = _resolve_inp_sources(analysis)
    assert hydro == hydraulics and hydro.read_text() == "full"


def test_performance_clearing_field_is_disclosed_by_data_availability():
    """Property: ``consolidated_performance_timeseries_cleared`` is a data-availability disclosure field (E84-F2)."""
    from hhemt import analysis_validation as av

    assert "consolidated_performance_timeseries_cleared" in av._RECLAIM_LOG_FIELDS


# ---------------------------------------------------------------------------
# Round 5: the consolidation skips removed events like events already in the append-only groups.
# ---------------------------------------------------------------------------
_TS_GROUPS = ("swmm_node/timeseries", "swmm_link/timeseries")


def _golden_series_analysis(golden_extractor):
    """Golden multi-sim analysis, series consolidation on, clearing off lists written."""
    from tests.fixtures.golden_trees import construct_golden_analysis
    from tests.test_synth_timeseries_consolidation import _write_coupling_lists

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    analysis = construct_golden_analysis(system_dir)
    _write_coupling_lists(analysis)
    return analysis


def _remove_event_directory(analysis, event_iloc: int) -> str:
    """Record the event's directory as removed (done) and delete it, as the removal step does."""
    import shutil

    from hhemt.scenario import compute_event_id_slug

    evt = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(event_iloc))
    sim = Path(analysis.analysis_paths.simulation_directory) / evt
    write_scenario_record(Path(analysis.analysis_paths.analysis_dir), evt, {"state": "done", "model_logs": {}})
    shutil.rmtree(sim)
    return evt


def test_reconsolidation_appends_the_new_event_and_leaves_a_removed_events_slice_byte_identical(golden_extractor):
    """Property: with event 0 removed (record done, directory gone) and event 2 new, ``consolidate_timeseries`` appends
    event 2 and leaves event 0's (and event 1's) slices byte-identical; the axis comes from the stored group.

    Class: NEW CAPABILITY. Wrong implementation: rebuild every event (raises ScenarioRemovedError / FileNotFoundError on
    the removed one) or rewrite its slice. Reachability: the golden's three events are split into two rounds by
    restricting ``df_sims``. Both-states anchor: the group holds events [0, 1] before and [0, 1, 2] after. Must-fire
    control: event 2 is genuinely new (absent before). Kills: deleting the skip in ``consolidate_timeseries``.
    Second correct implementation: skip the removed event inside ``_plan_timeseries_write`` as a ``verified`` action.
    """
    import xarray as xr

    a = _golden_series_analysis(golden_extractor)
    full = a.df_sims
    a.df_sims = full.iloc[:2]
    a.process.consolidate_timeseries()
    store = a.analysis_paths.analysis_datatree_zarr
    before = {g: xr.open_zarr(store, group=f"tritonswmm/{g}").load() for g in _TS_GROUPS}
    assert [int(v) for v in before[_TS_GROUPS[0]]["event_iloc"].values] == [0, 1]

    a.df_sims = full
    _remove_event_directory(a, 0)
    a.process.consolidate_timeseries()
    for g in _TS_GROUPS:
        after = xr.open_zarr(store, group=f"tritonswmm/{g}").load()
        assert [int(v) for v in after["event_iloc"].values] == [0, 1, 2]
        for name in before[g].data_vars:
            xr.testing.assert_identical(before[g][name], after[name].isel(event_iloc=[0, 1]))


def test_a_removed_event_absent_from_the_group_refuses_naming_it(golden_extractor):
    """Must-fire: a removed event whose slice the group lacks refuses with ProcessingError naming the event, writing
    nothing. Control: the same analysis with the event NOT removed consolidates. Kills: skipping silently.
    """
    from hhemt.exceptions import ProcessingError

    a = _golden_series_analysis(golden_extractor)
    evt = _remove_event_directory(a, 0)
    with pytest.raises(ProcessingError, match=evt):
        a.process.consolidate_timeseries()
    assert not any(Path(a.analysis_paths.analysis_dir).glob("analysis_datatree.zarr/tritonswmm/swmm_node/timeseries"))


def test_a_removed_event_with_no_cleared_summary_cannot_be_read_as_a_summary(golden_extractor):
    """Dynamic arm of ``_event_summary_dataset``: a removed event whose log records no cleared summary raises
    ProcessingError (the summary exists nowhere) instead of a ScenarioRemovedError from deep in a handle.
    """
    from hhemt.exceptions import ProcessingError

    a = _golden_series_analysis(golden_extractor)
    evt = _remove_event_directory(a, 0)
    with pytest.raises(ProcessingError, match=evt):
        a.process._event_summary_dataset("tritonswmm_triton", 0, drop_flag=False)


def test_clear_summary_stores_and_axis_skip_a_removed_event(golden_extractor):
    """Dynamic arm: ``_clear_summary_stores`` and ``_timeseries_axis`` run over an analysis with a removed event without
    touching its handle. The axis extent equals the one computed with the event present (the stored group spans it).
    """
    a = _golden_series_analysis(golden_extractor)
    a.process.consolidate_to_datatree()
    _, _, axis_present = a.process._timeseries_axis()
    _remove_event_directory(a, 0)
    _, _, axis_removed = a.process._timeseries_axis()
    assert list(axis_removed) == list(axis_present)
    a.process._clear_summary_stores(a.analysis_paths.analysis_datatree_zarr, drop_flag=False)


def test_hydrology_consolidation_refuses_a_removed_event_absent_from_its_groups(golden_extractor):
    """Must-fire for ``consolidate_hydrology``: a removed event whose slice the hydrology groups lack refuses naming it
    (the skip is not silent). Not covered: the skip-and-append path of the hydrology groups, which needs a hydrology
    capture per event (the fixtures of ``tests/test_hydrology_groups.py``); that path is exercised only by inspection.
    """
    from hhemt.exceptions import ProcessingError

    a = _golden_series_analysis(golden_extractor)
    evt = _remove_event_directory(a, 0)
    with pytest.raises(ProcessingError, match=evt):
        a.process.consolidate_hydrology()


def test_hydrology_reconsolidation_appends_the_new_event_and_leaves_a_removed_events_slices_untouched(golden_extractor):
    """Property: with the hydrology groups present for events 0 and 1, event 0 recorded removed (directory gone) and
    event 2 new, ``consolidate_hydrology`` appends event 2 and leaves event 0's and 1's slices identical in the series
    and the node summary group.

    Class: NEW CAPABILITY. Wrong implementation: rebuild every event (raises on the removed one's handle or rewrites its
    slice). Both-states anchor: the series group holds events [0, 1] before and [0, 1, 2] after. Must-fire control: the
    sibling test (a removed event absent from the groups refuses naming it) and event 2 being absent before.
    Kills: removing the removed-event filter in ``consolidate_hydrology``. Second correct implementation: skip the
    removed event inside the per-event body.
    """
    import xarray as xr

    from tests.test_hydrology_groups import _NODE, _TS, _group, _ready, _write_capture

    a = _ready(golden_extractor, clear=False)
    full = a.df_sims
    flows = {"J1": [0.0, 1.0, 3.0, 2.0] + [0.5] * 8, "J2": [1.0] * 12}
    a.df_sims = full.iloc[:2]
    _write_capture(a, 0, flows)
    _write_capture(a, 1, {"J2": [2.0] * 11})
    a.process.consolidate_hydrology()
    before = {g: _group(a, g) for g in (_TS, _NODE)}
    assert [int(v) for v in before[_TS]["event_iloc"].values] == [0, 1]

    a.df_sims = full
    _write_capture(a, 2, {"J3": [0.25] * 12})
    _remove_event_directory(a, 0)
    a.process.consolidate_hydrology()
    for g in (_TS, _NODE):
        after = _group(a, g)
        assert [int(v) for v in after["event_iloc"].values] == [0, 1, 2]
        for name in before[g].data_vars:
            xr.testing.assert_identical(before[g][name], after[name].isel(event_iloc=[0, 1]))


def test_removal_captures_the_live_status_rows_when_the_caller_passes_df_status(tmp_path, monkeypatch):
    """Property (R-1, Form 2): rows passed as ``df_status`` are captured into the record verbatim (event column
    # Event 1 has no row in the passed table: an empty capture, which the reader treats as none.
    """
    import pandas as pd

    host, PP = _removal_host(tmp_path, ("event_index.0", "event_index.1"))
    for evt in ("event_index.0", "event_index.1"):
        _make_scenario_dir(tmp_path, evt)
    _patch_scenario(monkeypatch, {"event_index.0": True, "event_index.1": True})
    df = pd.DataFrame({"event_iloc": [0], "model_type": ["tritonswmm"], "actual_gpus": [2]})
    PP._remove_cleared_scenario_directories(host, df_status=df)
    rec0 = read_scenario_record(tmp_path, "event_index.0")["captured"]
    assert rec0["df_status_rows"] == [{"model_type": "tritonswmm", "actual_gpus": 2}]
    # Event 1 has no row in the passed table: an empty capture, which the reader treats as "none" (minimal rows).
    assert read_scenario_record(tmp_path, "event_index.1")["captured"]["df_status_rows"] == []


def test_df_status_for_consolidation_falls_back_only_on_a_missing_snakefile():
    """Property: ``_df_status_for_consolidation`` returns None on FileNotFoundError and propagates any other error."""
    from hhemt.analysis import TRITONSWMM_analysis as A

    class _Missing:
        @property
        def df_status(self):
            raise FileNotFoundError("Snakefile")

    class _Other:
        @property
        def df_status(self):
            raise KeyError("boom")

    assert A._df_status_for_consolidation(_Missing()) is None
    with pytest.raises(KeyError):
        A._df_status_for_consolidation(_Other())


def test_removal_step_runs_on_the_early_reuse_return(golden_extractor, monkeypatch):
    """Property (D22 resume, D24 (e)): when ``consolidate_to_datatree`` takes its EARLY REUSE return (tree present, log
    complete, inputs fingerprint and stamp match), the removal step still runs: a record left ``pending`` is finished
    and a scenario that is eligible but unremoved is removed, with nothing rebuilt.

    Class: REGRESSION on the reuse return (it returned before the removal step). Reachability: a first consolidation
    killed during event 1's delete leaves event 0 done, event 1 pending, event 2 unremoved, and a complete tree.
    Must-fire proof the reuse path is the one taken: ``_write_summary_branches`` is patched to raise, so any rebuild
    fails the test. Kills: calling the removal step only after the rebuild. Second correct implementation: a separate
    resume entry that the reuse return calls.
    """
    from tests.fixtures.golden_trees import construct_golden_analysis

    def switches_on(a):
        c = a.cfg_analysis
        c.toggle_consolidate_timeseries = True
        c.toggle_clear_scenario_timeseries_after_consolidation = True
        c.clear_raw = "all"
        c.remove_after_processing = "all"

    first = construct_golden_analysis(golden_extractor("multisim"))
    switches_on(first)
    system_dir = first.analysis_paths.analysis_dir.parent
    original = sr.finish_scenario_removal

    def killed(analysis_dir, event_id, sim_folder):
        if event_id == "event_index.1":
            raise RuntimeError("simulated kill during the delete")
        return original(analysis_dir, event_id, sim_folder)

    monkeypatch.setattr(sr, "finish_scenario_removal", killed)
    with pytest.raises(RuntimeError, match="simulated kill"):
        first.process.consolidate_to_datatree()
    monkeypatch.setattr(sr, "finish_scenario_removal", original)

    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as PP

    def no_rebuild(self, *a, **k):
        raise AssertionError("the reuse path must not rebuild the tree")

    monkeypatch.setattr(PP, "_write_summary_branches", no_rebuild)
    # A dirty checkout never matches a recorded consolidation build stamp, which would force a rebuild and hide the
    # reuse path; the documented operator declaration makes the gate reuse the store as it does on a matching build.
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")
    rerun = construct_golden_analysis(system_dir)
    switches_on(rerun)
    rerun.process.consolidate_to_datatree()
    events = ("event_index.0", "event_index.1", "event_index.2")
    analysis_dir = Path(rerun.analysis_paths.analysis_dir)
    sims = Path(rerun.analysis_paths.simulation_directory)
    assert {e: (sr.record_state(analysis_dir, e), (sims / e).is_dir()) for e in events} == {
        e: ("done", False) for e in events
    }
