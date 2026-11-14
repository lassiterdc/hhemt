"""The benchmarking renderer's SWMM-only row collection reads a removed event of an ACTIVE member from its record.

Problem this file pins. A SWMM-only sensitivity member keeps its own scenario records under
``{member_dir}/_status/scenario_records/``. Between the member's scenarios being removed (record ``done``, the
``sims/{event}/`` directory gone) and the master removing the member's whole directory, the member is still an
ACTIVE ``TRITONSWMM_analysis``, not a recorded view. That is the state an interrupted master consolidation leaves.
``_collect_rows`` asked such a member for ``_retrieve_sim_run_processing_object(i)``, which builds the processing
object and calls ``get_log`` on a removed scenario, and ``get_log`` raises ``ScenarioRemovedError``. The renderer
already reads ``captured['swmm_total_elapsed']`` for a removed MEMBER; the guard routes this removed EVENT through
the same branch.

The trees are the committed sensitivity golden converted to a SWMM-only arm by two toggles in its system config.
Nothing here compiles or runs a solver, so it runs on the laptop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hhemt.scenario_records import ScenarioRemovedError, scenario_record_path, write_scenario_record

EVENT = "event_index.0"


@pytest.fixture
def swmm_only_master(golden_extractor, monkeypatch):
    """The sensitivity golden with its arm switched to SWMM-only, as an analysis object, plus the module under test.

    A SWMM-only master's tree carries no ``performance/summary`` node, which is what sends ``_collect_rows`` to the
    per-scenario branch; the golden's coupled tree carries one, so ``_find_perf_node`` is told the node is absent.
    """
    from hhemt.report_renderers import sensitivity_benchmarking as sb
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("sensitivity")
    cfg = system_dir / "system_config.yaml"
    cfg.write_text(
        cfg.read_text()
        .replace("toggle_tritonswmm_model: true", "toggle_tritonswmm_model: false")
        .replace("toggle_swmm_model: false", "toggle_swmm_model: true")
    )
    monkeypatch.setattr(sb, "_find_perf_node", lambda tree, member_id: None)
    return construct_golden_analysis(system_dir), sb


def _member_dir(master, member_id: str) -> Path:
    return Path(master.sensitivity.members[member_id].analysis_paths.analysis_dir)


def _remove_event(master, member_id: str, captured: dict) -> Path:
    """Mark the member's only event removed (record ``done``, directory gone) as the removal step leaves it."""
    member_dir = _member_dir(master, member_id)
    write_scenario_record(member_dir, EVENT, {"state": "done", "model_logs": {}, "captured": captured})
    shutil.rmtree(member_dir / "sims" / EVENT)
    return scenario_record_path(member_dir, EVENT)


def _write_live_report(master, member_id: str, hhmmss: str) -> Path:
    rpt = _member_dir(master, member_id) / "sims" / EVENT / "out_swmm" / "full.rpt"
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(f"  Total elapsed time: {hhmmss}\n")
    return rpt


def test_a_removed_event_of_an_active_swmm_only_member_reads_its_recorded_elapsed_time(swmm_only_master):
    """Property: ``_collect_rows`` returns one row for each active SWMM-only member whose elapsed time can be read, the
    value of a removed event coming from its record and the value of a live event from its ``full.rpt``, and it does
    not raise ``ScenarioRemovedError`` for a removed event.

    Class: REGRESSION. Red on the pre-change tree: member 0 is active and its event is removed, so the collector
    raises ``ScenarioRemovedError`` from ``get_log`` (output pasted in the build record).
    Reachability: the state is the one an interrupted master consolidation leaves, member records ``done`` and the
    member directory not yet moved; member 0 has ``is_recorded_member`` false and a ``done`` record.
    Both-states anchor: the assertion turns on rows and declared sources, which both the current and the guarded
    collector produce, and not on any wording the guard introduces.
    Must-fire control: member 1 is live with a readable report, so a collector that returned no rows at all, or only
    the live row, fails on member 0's row; a collector that ignored the record fails on the value 12.5.
    Differently positioned inputs: member 2 is removed with NO captured elapsed time and must contribute no row;
    member 3 is live with no report and must contribute no row.
    Kills: deleting the ``record is not None`` branch of ``_swmm_only_event_handle`` (the call goes back to
    ``_retrieve_sim_run_processing_object`` and raises).
    Second correct implementation: catch ``ScenarioRemovedError`` at the call and read ``record_field`` instead; the
    rows and sources are the same.
    """
    master, sb = swmm_only_master
    record_0 = _remove_event(master, "0", {"swmm_total_elapsed": 12.5})
    rpt_1 = _write_live_report(master, "1", "00:00:07")
    _remove_event(master, "2", {})

    rows, sources = sb._collect_rows(master, "performance.Total")

    assert {r["sa_id"]: r["value"] for r in rows} == {"0": 12.5, "1": 7.0}
    assert len(rows) == 2, "member 2 (removed, nothing captured) and member 3 (live, no report) add no row"
    assert record_0 in sources, "the declared source of a removed event is its record file"
    assert rpt_1 in sources
    assert json.loads(record_0.read_text())["state"] == "done", "the collector reads the record and never rewrites it"


def test_a_removed_event_is_still_the_pre_guard_failure_for_a_caller_that_asks_the_member_directly(swmm_only_master):
    """Control for the property above: the failure the guard removes is real and lives in the member, not in the test.

    Class: REGRESSION control. Asking the active member for the removed event's processing object raises
    ``ScenarioRemovedError`` today and after the guard (the guard changes the renderer, not the member). A removed event
    that this raised for is the state the first test reads from the record.
    Both-states anchor: ``_retrieve_sim_run_processing_object`` is untouched by the guard.
    Kills: making the renderer guard unnecessary by changing the member to stop raising; this test then reds and
    says so.
    Second correct implementation: none needed, the assertion is about the member's existing behaviour.
    """
    master, _ = swmm_only_master
    _remove_event(master, "0", {"swmm_total_elapsed": 12.5})
    member = master.sensitivity.members["0"]

    assert member._event_is_removed(0) is True
    with pytest.raises(ScenarioRemovedError):
        member._retrieve_sim_run_processing_object(0)


@pytest.mark.parametrize(
    ("state", "directory_kept", "expected_value", "source_is_record"),
    [
        ("pending", False, 12.5, True),
        ("pending", True, 12.5, True),
        ("superseded", True, 7.0, False),
    ],
    ids=["pending-directory-gone", "pending-directory-kept", "superseded-directory-kept"],
)
def test_the_record_state_alone_decides_whether_an_active_members_event_reads_from_its_record(
    swmm_only_master, state, directory_kept, expected_value, source_is_record
):
    """Property: for an active SWMM-only member's event with a scenario record, ``_collect_rows`` reads the captured
    elapsed time when the record state is ``pending`` or ``done`` (whether or not ``sims/{event}/`` still exists) and
    reads the live ``full.rpt`` when it is ``superseded``; the declared source follows the value.

    Class: the record states of one event, ``pending`` with the directory gone, ``pending`` with it kept, and
    ``superseded`` with it kept. The ``done`` state is the coder's first test. The two ``pending`` rows are
    REGRESSION (red on the pre-change collector: it raises for the gone directory and returns the live 7.0 for the
    kept one); the ``superseded`` row is held, green before and after.
    Reachability: ``remove_scenario_directory`` writes ``pending``, deletes, then sets ``done``, so a kill between
    any two of those steps leaves exactly these states on an active member; a force re-run supersedes a record.
    Both-states anchor: the assertion turns on the row value and its declared source, which the pre-change and the
    guarded collector both produce, not on any wording the guard introduces.
    Kills: narrowing the guard's state test from ``not in REMOVED_STATES`` to ``!= "done"`` (the gone-directory
    ``pending`` row raises again), and widening it to any record (the ``superseded`` row reads 12.5).
    Second correct implementation: ``event_is_removed(analysis_dir, event_id)`` for the decision plus a record read
    for the handle; the same rows and sources.
    """
    master, sb = swmm_only_master
    member_dir = _member_dir(master, "0")
    write_scenario_record(
        member_dir, EVENT, {"state": state, "model_logs": {}, "captured": {"swmm_total_elapsed": 12.5}}
    )
    live_rpt = _write_live_report(master, "0", "00:00:07")
    if not directory_kept:
        shutil.rmtree(member_dir / "sims" / EVENT)

    rows, sources = sb._collect_rows(master, "performance.Total")

    assert {r["sa_id"]: r["value"] for r in rows} == {"0": expected_value}
    assert (scenario_record_path(member_dir, EVENT) in sources) is source_is_record
    assert (live_rpt in sources) is (not source_is_record)
