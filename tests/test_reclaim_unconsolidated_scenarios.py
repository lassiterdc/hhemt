"""The A2 fallback must skip an event whose SUMMARIES are absent, not only one whose flag is present.

Two skips, and they are asymmetric. An event in SIM_IDS carries a
f_consolidate_scenario flag and is caught by SKIP 1. An event OUTSIDE SIM_IDS carries no
flag -- _available_event_ids excluded it for having at least one enabled model's summary
missing -- and is caught ONLY by SKIP 2. The pre-amendment single-skip form reclaimed
that second class: a scenario an operator is mid-recovery on, whose prep inputs a re-run
needs. Arm B is that case, and it FAILS against the single-skip form.

Arm C is not decoration. Without a case that MUST reclaim, a function that does nothing
at all passes arms A and B, and the module would certify inertness as correctness.

Fixture-free by construction: both skips `continue` before TRITONSWMM_scenario is
constructed, so the logic under test sits upstream of __init__'s _create_directories().
Every assertion is on which event ids reached the stubs, mirroring
tests/test_prep_inputs_reclaim_waits_for_runs.py.
"""

from __future__ import annotations

import types

_EVENTS = {0: "evt-a", 1: "evt-b", 2: "evt-c"}
_FLAGGED = {"evt-a"}  # in SIM_IDS -> its per-scenario rule ran
_SUMMARISED = {"evt-a", "evt-c"}  # evt-b is the summary-absent divergence


def _probe(tmp_path, monkeypatch):
    """Drive reclaim_unconsolidated_scenarios; return (constructed, reclaimed) event ids."""
    import hhemt.process_simulation as _ps
    import hhemt.scenario as _sc
    import hhemt.summary_paths as _sp
    from hhemt.consolidate_workflow import reclaim_unconsolidated_scenarios

    status = tmp_path / "_status"
    status.mkdir(parents=True, exist_ok=True)
    for _eid in _FLAGGED:
        (status / f"f_consolidate_scenario_evt-{_eid}_complete.flag").touch()

    constructed: list[str] = []
    reclaimed: list[str] = []

    def _fake_scenario(iloc, _analysis):
        constructed.append(_EVENTS[iloc])
        return types.SimpleNamespace(event_iloc=iloc, log=types.SimpleNamespace())

    def _fake_reclaim(scen, scoped, adir, verbose=False):
        reclaimed.append(_EVENTS[scen.event_iloc])
        return {"prep_inputs": True}

    monkeypatch.setattr(_sc, "compute_event_id_slug", lambda indexer: indexer["event_id"])
    monkeypatch.setattr(_sc, "TRITONSWMM_scenario", _fake_scenario)
    monkeypatch.setattr(_sp, "scenario_summaries_present", lambda _a, eid, _m: eid in _SUMMARISED)
    monkeypatch.setattr(_ps, "reclaim_scenario_scoped_classes", _fake_reclaim)

    analysis = types.SimpleNamespace(
        df_sims=types.SimpleNamespace(index=sorted(_EVENTS)),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": _EVENTS[i]},
    )
    reclaim_unconsolidated_scenarios(analysis, ["tritonswmm"], ("prep_inputs",), tmp_path)
    return constructed, reclaimed


def test_arm_a_flagged_event_is_skipped(tmp_path, monkeypatch):
    """In SIM_IDS: the per-scenario barrier already reclaimed it. SKIP 1."""
    constructed, reclaimed = _probe(tmp_path, monkeypatch)
    assert "evt-a" not in constructed
    assert "evt-a" not in reclaimed


def test_arm_b_summary_absent_event_is_skipped(tmp_path, monkeypatch):
    """THE regression. Pre-amendment (SKIP 1 only) evt-b is constructed AND reclaimed."""
    constructed, reclaimed = _probe(tmp_path, monkeypatch)
    assert "evt-b" not in constructed, "a summary-absent event was constructed -- SKIP 2 is missing"
    assert "evt-b" not in reclaimed, "reclaimed a scenario this run excluded for absent summaries"


def test_arm_c_unflagged_but_summarised_event_is_reclaimed(tmp_path, monkeypatch):
    """The control. Without it, a no-op function passes arms A and B."""
    _constructed, reclaimed = _probe(tmp_path, monkeypatch)
    assert reclaimed == ["evt-c"]
