"""The status table and its export read a removed scenario from its record, never from its (absent) log.

No solver and no Snakemake process: the committed multi-sim golden is configured so the default removal fires, the
real consolidation removes the scenario directories it is allowed to, one record is then deleted to leave a live
scenario beside the removed ones, and the real `df_status` and the export's CSV writer run over that tree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.golden_trees import construct_golden_analysis, dry_run_plan


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout would rebuild)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _slug(analysis, event_iloc) -> str:
    from hhemt.scenario import compute_event_id_slug

    return compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(event_iloc))


def _removed_tree(golden_extractor):
    """The multi-sim golden with scenario 0 removed through the real recorded-before-delete removal and the other
    scenarios left live, and the Snakefile planned (the status table reads its allocations)."""
    from hhemt.scenario_records import capture_system_overview_inps, remove_scenario_directory

    system_dir = golden_extractor("multisim")
    analysis = construct_golden_analysis(system_dir)
    dry_run_plan(analysis)
    models = list(analysis._get_enabled_model_types())
    evt = _slug(analysis, 0)
    sim = Path(analysis.analysis_paths.simulation_directory) / evt
    capture_system_overview_inps(sim, analysis.analysis_paths.analysis_dir)
    remove_scenario_directory(
        analysis.analysis_paths.analysis_dir, evt, sim, run_completed={m: True for m in models}, captured={}
    )
    return analysis


def test_df_status_returns_the_captured_row_of_a_removed_scenario_and_the_export_completes(golden_extractor, tmp_path):
    """Property: after the removal ran, `df_status` returns one row per event and model, a removed scenario's rows
    carry its recorded directory path and no call touches its log, and the export's CSV writer completes.

    Must-fire control: the removal DID remove at least one scenario (its directory is absent and its record says
    removed), so the assertions are not satisfied by a tree where nothing was removed.
    Kills it: a reader that builds the scenario or reads its log for a removed event
    (`ScenarioRemovedError`). A second correct implementation answers from the record for every event.
    """
    from hhemt.export_scenario_status import export_scenario_status_to_csv

    a = _removed_tree(golden_extractor)
    removed = [i for i in a.df_sims.index if a._event_is_removed(i)]
    assert [int(i) for i in removed] == [0]
    assert len(a.df_sims) > 1, "control: a live scenario sits beside the removed one"
    assert removed, "control: scenario 0 is removed and its directory is absent"
    df = a.df_status
    models = a._get_enabled_model_types()
    assert len(df) == len(a.df_sims) * len(models)
    for i in removed:
        sub = df[df["event_iloc"] == i]
        assert len(sub) == len(models)
        assert all(str(d).endswith(_slug(a, i)) for d in sub["scenario_directory"])
        assert not Path(a.analysis_paths.simulation_directory / _slug(a, i)).exists(), "nothing re-created"
    live = df[df["event_iloc"] != 0]
    assert not live.empty and bool(live["scenario_setup"].all()), "a live scenario keeps its live row"
    csv = export_scenario_status_to_csv(a, tmp_path / "status.csv", df_status=df)
    assert len(pd.read_csv(csv)) == len(df)


def test_the_event_page_reads_a_removed_scenarios_geometry_from_the_kept_inp_and_never_builds_its_run(golden_extractor):
    """Property: the per-sim pages take an event's conduit geometry from `event_hydraulics_inp`; for a removed
    scenario that is the `.inp` kept beside the records, no processing object is built and no directory is
    re-created; a live scenario still answers from its own paths.

    The series and summaries of the page already come from `analysis_datatree.zarr`; the only directory read was
    this `.inp`, so no panel degrades. Must-fire control: the former reader (`_retrieve_sim_run_processing_object`)
    on the same removed event raises or re-creates the removed directory.
    Kills it: a reader that builds the run object of a removed event.
    """
    from hhemt.report_renderers.per_sim_conduit_flow import event_hydraulics_inp

    a = _removed_tree(golden_extractor)
    sim0 = Path(a.analysis_paths.simulation_directory) / _slug(a, 0)
    assert not sim0.exists()
    a._retrieve_sim_run_processing_object = _forbidden
    kept = event_hydraulics_inp(a, 0)
    assert kept.exists() and "scenario_records" in kept.as_posix()
    assert not sim0.exists(), "nothing re-created"
    del a._retrieve_sim_run_processing_object
    live = event_hydraulics_inp(a, 1)
    assert live.exists() and "scenario_records" not in live.as_posix()
    # Control: the former reader raised for a removed scenario (the failure measured on the real tree).
    from hhemt.scenario_records import ScenarioRemovedError

    with pytest.raises(ScenarioRemovedError):
        a._retrieve_sim_run_processing_object(0)


def _forbidden(*_a, **_k):
    raise AssertionError("the run object of a removed scenario must not be built")
