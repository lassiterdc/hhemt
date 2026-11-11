"""Reviewer tests for the scenario and member directory removal.

Problem this file solves. The coder's own tests for the removal drive it through stand-in objects (a host namespace, a
patched scenario constructor, a fake master). Under the defaults the removal really runs inside
``consolidate_to_datatree`` and ``consolidate_sensitivity_datatree`` and every later reader then meets a tree whose
scenario and member directories are gone. These tests run the REAL objects: the golden multi-sim and sensitivity
analyses (no solver, no Snakemake, no compile; laptop-admissible), remove their directories through the real pipeline,
and then exercise the consumers a removed analysis must still serve: a second consolidation, a resume after a kill,
the validation checks, the master Snakefile generators and the reprocess entry.

Each test quantifies over a CLASS (every reader of an analysis whose directories are all removed), not over one site.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVENTS = ("event_index.0", "event_index.1", "event_index.2")


def _switches_on(analysis) -> None:
    """Turn the three removal switches on in memory (the golden configs carry the pre-default values)."""
    cfg = analysis.cfg_analysis
    cfg.toggle_consolidate_timeseries = True
    cfg.toggle_clear_scenario_timeseries_after_consolidation = True
    cfg.clear_raw = "all"
    cfg.remove_after_processing = "all"


def _states(analysis) -> dict[str, tuple[str | None, bool]]:
    """Per event: (analysis-level record state or None, whether ``sims/{event}`` exists)."""
    from hhemt import scenario_records as sr

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    sims = Path(analysis.analysis_paths.simulation_directory)
    return {e: (sr.record_state(analysis_dir, e), (sims / e).is_dir()) for e in EVENTS}


def _fresh(system_dir: Path):
    """A new analysis object over the same tree, as a later session or a rerun job sees it."""
    from tests.fixtures.golden_trees import construct_golden_analysis

    analysis = construct_golden_analysis(system_dir)
    _switches_on(analysis)
    return analysis


def test_a_second_consolidation_over_removed_scenarios_neither_raises_nor_recreates_a_directory(golden_extractor):
    """Class: every scenario of the analysis is removed (record done, directory gone) and ``consolidate_to_datatree``
    runs again, which is what ``rule consolidate`` does on a replan and what ``reprocess(start_with="consolidate")``
    asks for (D24 (g) table row REMOVED, D24 (h): "re-plans from the tree ... and re-stamps").

    Why required: the call must succeed and create no ``sims/`` child. Measured on the unmodified tree, the second
    call raises ``ScenarioRemovedError`` from ``provenance._output_ids`` (``TRITONSWMM_scenario(...).get_log``, the
    ``provenance.py:912`` site the coder listed as an erratum), reached through ``emit_provenance`` at
    ``processing_analysis.consolidate_to_datatree``.

    Kills: a removal that leaves any reader of the model log on the constructor path. Control (the discrimination
    run): with ``provenance._output_ids`` patched to return no ids the same call succeeds, so the red is the
    provenance reader and nothing else. Second correct implementation: ``_output_ids`` reading its model-log fields
    through ``scenario_records.record_for`` instead of a handle, or the call skipping provenance emission for a
    removed event.
    """
    first = _fresh(golden_extractor("multisim"))
    first.process.consolidate_to_datatree()
    assert all(state == "done" and not exists for state, exists in _states(first).values()), _states(first)

    again = _fresh(first.analysis_paths.analysis_dir.parent)
    again.process.consolidate_to_datatree()
    assert not any(exists for _s, exists in _states(again).values()), "the second consolidation re-created a directory"


def test_the_provenance_reader_is_the_only_blocker_of_a_second_consolidation(golden_extractor, monkeypatch):
    """HELD control for the two tests around it (green before and after the coder's turn): with the provenance id
    reader neutralised, a second consolidation over fully removed scenarios succeeds and re-creates no directory.

    Class: a fully removed analysis. Why required: it separates "the provenance reader is the defect" from "the
    removal or the consolidation skip is broken". Kills: a fix that edits the consolidation skip instead of the
    reader. Mutation that reds it: re-introducing a constructor call for a removed event inside the consolidation.
    Second correct implementation: any other neutralisation of ``_output_ids`` (it returns a list of ids).
    """
    import hhemt.provenance as prov

    first = _fresh(golden_extractor("multisim"))
    first.process.consolidate_to_datatree()
    monkeypatch.setattr(prov, "_output_ids", lambda *a, **k: [])
    again = _fresh(first.analysis_paths.analysis_dir.parent)
    again.process.consolidate_to_datatree()
    assert not any(exists for _s, exists in _states(again).values())


def test_a_consolidation_rerun_after_a_kill_mid_removal_finishes_every_pending_removal(golden_extractor, monkeypatch):
    """Class: the removal loop was killed after writing event 1's record ``pending`` (event 0 ``done``, event 2
    untouched) and ``consolidate_to_datatree`` runs again (D22: "a resume finding ``pending`` ... finishes the delete
    and sets ``done``"; D24 (g) table row REMOVAL INTERRUPTED: "finished by the first act of rule consolidate").

    Why required: after the rerun no event may stay ``pending`` or keep its directory, otherwise the directory the
    record says is removed is kept forever and the planner has already excluded the event from ``SIM_IDS``, so no
    rule will ever finish it. Measured on the unmodified tree: the rerun raises ``ScenarioRemovedError`` (provenance
    reader, as the sibling test) and leaves event 1 ``pending`` with its directory and event 2 unremoved.

    Kills: any reader of the model log on the constructor path reachable from the consolidation entry. Control: with
    ``provenance._output_ids`` patched the rerun ends with all three events ``done`` and no directory, so the resume
    logic itself is correct. Second correct implementation: finishing pending removals in a step that runs before
    ``emit_provenance`` and the early reuse return.
    """
    from hhemt import scenario_records as sr

    first = _fresh(golden_extractor("multisim"))
    system_dir = first.analysis_paths.analysis_dir.parent
    original = sr.finish_scenario_removal

    def killed(analysis_dir, event_id, sim_folder):
        if event_id == "event_index.1":
            raise RuntimeError("simulated kill during the delete")
        return original(analysis_dir, event_id, sim_folder)

    monkeypatch.setattr(sr, "finish_scenario_removal", killed)
    with pytest.raises(RuntimeError, match="simulated kill"):
        first.process.consolidate_to_datatree()
    assert _states(first) == {
        "event_index.0": ("done", False),
        "event_index.1": ("pending", True),
        "event_index.2": (None, True),
    }, "the interrupted state this test resumes from was not produced"
    monkeypatch.setattr(sr, "finish_scenario_removal", original)

    rerun = _fresh(system_dir)
    rerun.process.consolidate_to_datatree()
    assert _states(rerun) == {e: ("done", False) for e in EVENTS}, _states(rerun)


def _removed_sensitivity_master(golden_extractor):
    """A golden sensitivity master whose four members were all removed through the real consolidation."""
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("sensitivity")
    master = construct_golden_analysis(system_dir)
    for member in master.sensitivity.members.values():
        _switches_on(member)
    master.sensitivity.consolidate_sensitivity_datatree(verbose=False)
    fresh = construct_golden_analysis(system_dir)
    assert not fresh.sensitivity.members and len(fresh.sensitivity.recorded_members) == 4
    return fresh


def test_check_data_availability_runs_over_a_master_whose_members_were_removed(golden_extractor):
    """Class: every removed member of a sensitivity master, under the defaults (each member's model log carries the
    ``consolidated_summaries_cleared`` record, so ``check_data_availability`` reaches ``scenario_summaries_present``
    for it). ``validate_analysis`` runs this check for the report's Errors-and-Warnings sidebar.

    Why required: D24 (e) gives a removed member a read-only view that provides no ``analysis_paths`` by design, so
    every call site that hands a view to a path-reading predicate must branch on ``is_recorded_member`` first (the
    other checks in ``analysis_validation`` do). Measured: ``check_data_availability`` raises ``MemberRemovedError``
    for attribute ``analysis_paths`` out of ``summary_paths.scenario_summaries_present`` at its call in
    ``analysis_validation``. Note the coder's own test of this check uses a fake master that never reaches that call.

    Kills: calling a path predicate with a view. Control: the same master's ``check_scenarios_run`` returns a
    ``CheckResult`` (the sites that already branch). Second correct implementation: ``scenario_summaries_present``
    itself answering ``True`` for a view (a removed member's summaries are in the master tree by construction).
    """
    from hhemt import analysis_validation as av

    master = _removed_sensitivity_master(golden_extractor)
    assert av.check_scenarios_run(master).passed  # control: the branching sites already serve a removed member
    result = av.check_data_availability(master)
    assert result.passed, result.summary


def test_master_snakefile_generation_survives_a_master_with_no_active_member(golden_extractor):
    """Class: a sensitivity master all of whose members are removed (``members`` empty, ``recorded_members`` full) and
    whose Snakefile is generated again, as any resubmission of a completed default analysis does (D24 (g) "A dry run of
    a completed default analysis plans zero jobs").

    Why required: ``members`` now holds the ACTIVE members only (D24 (e) Form 3), so a generator that takes a value
    from a loop variable leaked out of ``for member_id, analysis in ...members.items()`` has none when every member is
    removed. Measured: ``generate_master_snakefile_content`` raises ``UnboundLocalError`` on ``analysis`` at the
    ``experiment_consolidation`` rule's resource block (``analysis.cfg_analysis.hpc_runtime_min_...``).

    Kills: leaking a per-member loop variable into whole-experiment output. Control: the reprocess master generator
    already serves the same master (asserted below). Second correct implementation: sizing the experiment
    consolidation rule from ``self.experiment.cfg_analysis`` or from ``all_members_for_reading()``.
    """
    master = _removed_sensitivity_master(golden_extractor)
    builder = master.sensitivity._workflow_builder
    assert builder.generate_reprocess_master_snakefile_content(start_with="consolidate")  # control
    assert builder.generate_master_snakefile_content()


def test_a_process_stage_reprocess_of_a_master_of_removed_members_refuses(golden_extractor):
    """Class: ``reprocess(start_with="process")`` over a sensitivity master whose members' scenarios were removed
    (D24 (h): "``start_with="process"`` ... of a removed scenario refuse at the reprocess entry with
    ``ConfigurationError`` naming the scenarios and stating the remedy").

    Why required: the process stage needs raw output that no longer exists, so the silent alternative is a "completed
    successfully" report for a reprocess that planned nothing. Measured: ``reprocess(start_with="process",
    dry_run=True)`` on the removed master returns ``success: True, message: Sensitivity reprocess completed
    successfully`` (the single-analysis path refuses; the master dispatches before that refusal).

    Kills: leaving the refusal in the single-analysis branch only. Control: the single-analysis refusal is asserted
    in the sibling arm below. Second correct implementation: refusing inside the master's own ``reprocess`` for any
    recorded member, rather than at the shared entry.
    """
    from hhemt.exceptions import ConfigurationError
    from tests.fixtures.golden_trees import construct_golden_analysis

    single = construct_golden_analysis(_removed_single_analysis_dir(golden_extractor))  # control: the arm that refuses
    _switches_on(single)
    with pytest.raises(ConfigurationError):
        single.reprocess(start_with="process", dry_run=True)

    master = _removed_sensitivity_master(golden_extractor)
    with pytest.raises(ConfigurationError):
        master.reprocess(start_with="process", dry_run=True)


def _removed_single_analysis_dir(golden_extractor) -> Path:
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("multisim")
    analysis = construct_golden_analysis(system_dir)
    _switches_on(analysis)
    analysis.process.consolidate_to_datatree()
    return system_dir


def test_sim_ids_follow_the_d24g_table_through_the_real_generator_and_generation_touches_nothing(golden_extractor):
    """HELD (green before and after the coder's turn): the five event states of the D24 (g) table reach the real
    production generator as the table says, and generation performs no filesystem act for a ``pending`` record.

    Class: the table's rows REMOVED (done), REMOVAL INTERRUPTED (pending), SUPERSEDED, and the three record-less rows
    (FAILED, NEVER RUN, SWITCH-OFF) which the generator cannot tell apart and the table gives one answer. Pass A
    writes done, pending and superseded records on the three events; pass B has none. ``SIM_IDS`` must be exactly the
    non-removed events, ``REPORT_EVENT_IDS`` every event, and ``sims/`` plus the record directory must be byte-for-byte
    the same listing after generation as before it.

    Kills: reading ``pending`` as present (a filesystem act at generation), or excluding ``superseded`` from the plan.
    Second correct implementation: deriving ``SIM_IDS`` from the flags in ``_status/`` instead of the records.
    """
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("multisim")
    analysis = construct_golden_analysis(system_dir)
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    sims = Path(analysis.analysis_paths.simulation_directory)
    records = analysis_dir / "_status" / "scenario_records"
    records.mkdir(parents=True, exist_ok=True)

    def ids(text: str, name: str) -> list[str]:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"{name} = "))
        return eval(line.split(" = ", 1)[1])  # noqa: S307 -- a list literal written by the generator

    def listing() -> list[str]:
        return sorted(str(p.relative_to(analysis_dir)) for p in analysis_dir.rglob("*") if "Snakefile" not in p.name)

    for evt, state in zip(EVENTS, ("done", "pending", "superseded"), strict=True):
        (records / f"{evt}.json").write_text(json.dumps({"event_id": evt, "state": state, "model_logs": {}}))
    before = listing()
    text = analysis._workflow_builder.generate_snakefile_content()
    assert ids(text, "SIM_IDS") == ["event_index.2"]
    assert ids(text, "REPORT_EVENT_IDS") == list(EVENTS)
    assert [p for p in listing() if p.startswith(("sims", "_status/scenario_records"))] == [
        p for p in before if p.startswith(("sims", "_status/scenario_records"))
    ]
    assert (sims / "event_index.1").is_dir(), "generation deleted the directory of a pending record"

    for p in records.glob("*.json"):
        p.unlink()
    text = analysis._workflow_builder.generate_snakefile_content()  # control: no record means the full chain
    assert ids(text, "SIM_IDS") == list(EVENTS)
