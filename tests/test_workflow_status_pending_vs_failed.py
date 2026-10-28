"""``get_workflow_status`` splits the incomplete scenarios into FAILED and PENDING.

A scenario is incomplete when any enabled model has not completed. It is PENDING when
none of its unfinished models wrote a model log (the runs have not started), and FAILED
when at least one did. ``simulations_pending`` and ``simulations_failed`` carry the two
counts, so a resume of a queued analysis no longer reports every queued scenario as
failed.

No ``TRITONSWMM_analysis`` and no scenario is constructed. The analysis is a stand-in
object carrying the analysis's plain methods and stubbed state, and
``hhemt.analysis.TRITONSWMM_scenario`` is replaced by a table-driven fake that answers
which models are enabled, which completed, and what each unfinished model's log shows.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.analysis as analysis_mod
from hhemt.analysis import TRITONSWMM_analysis

_MODELS = ["triton", "swmm"]

# Per scenario: for each enabled model, None = completed, else the log class of the
# unfinished model ("no_log", "unclassified" or "timeout").
_REGULAR = {
    0: {"triton": None, "swmm": None},  # complete
    1: {"triton": "no_log", "swmm": "no_log"},  # pending: no unfinished model wrote a log
    2: {"triton": "no_log", "swmm": "unclassified"},  # failed: one unfinished model wrote a log
    3: {"triton": None, "swmm": "timeout"},  # failed
}


class _Field:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeScenario:
    """Answers the scenario questions the split asks, from the owning analysis's table."""

    def __init__(self, event_iloc, analysis):
        self._row = analysis.table[event_iloc]
        self.run = SimpleNamespace(
            model_types_enabled=list(self._row),
            _classify_model_log_failure=lambda m: self._row[m],
        )
        self.log = SimpleNamespace(logfile=Path(f"/sims/{analysis.name}/evt-{event_iloc}/log.json"))

    def model_run_completed(self, model_type):
        return self._row[model_type] is None


def _member(name, table):
    return SimpleNamespace(name=name, table=table, df_sims=SimpleNamespace(index=list(table)))


def _stand_in(*, sensitivity, tables):
    """An object carrying TRITONSWMM_analysis's plain methods, with its state stubbed."""
    methods = {
        k: v for k, v in TRITONSWMM_analysis.__dict__.items() if inspect.isfunction(v) and not k.startswith("__")
    }
    obj = type("_StandIn", (), methods)()
    members = {name: _member(name, table) for name, table in tables.items()}
    incomplete = [
        f"/sims/{name}/evt-{e}"
        for name, table in tables.items()
        for e, row in table.items()
        if any(v is not None for v in row.values())
    ]
    n_scenarios = sum(len(t) for t in tables.values())
    obj._system = SimpleNamespace(
        log=SimpleNamespace(
            dem_processed=_Field(True),
            mannings_processed=_Field(True),
            compilation_tritonswmm_cpu_successful=_Field(True),
        ),
        cfg_system=SimpleNamespace(toggle_use_constant_mannings=True),
    )
    obj.cfg_analysis = SimpleNamespace(
        execution_environment="native", analysis_id="an_x", toggle_sensitivity_analysis=sensitivity
    )
    obj.analysis_paths = SimpleNamespace(analysis_dir=Path("/nonexistent/an_x"))
    obj.log = SimpleNamespace(datatree_consolidation_complete=_Field(False))
    obj.n_scenarios = n_scenarios
    obj.n_sims = n_scenarios * len(_MODELS)
    obj._get_enabled_model_types = lambda: list(_MODELS)
    obj._all_scenarios_created = True
    obj._scenarios_not_created = []
    obj._all_sims_run = not incomplete
    obj._scenarios_not_run = incomplete
    obj._TRITON_time_series_not_processed = []
    obj._SWMM_time_series_not_processed = []
    if sensitivity:
        obj.sensitivity = SimpleNamespace(members=members)
    else:
        only = members["base"]
        obj.name, obj.table, obj.df_sims = only.name, only.table, only.df_sims
    return obj


@pytest.fixture(autouse=True)
def _fake_scenarios(monkeypatch):
    monkeypatch.setattr(analysis_mod, "TRITONSWMM_scenario", _FakeScenario)


def test_regular_analysis_splits_pending_from_failed():
    """Property: an incomplete scenario is PENDING iff no unfinished model wrote a log.

    Class: incomplete scenarios on a regular analysis with several enabled models. This
    test is one complete scenario, one with no unfinished model logged, one where only
    some unfinished models logged, and one with a timed-out model.

    Why it is required: ``simulations_pending`` was always 0 and every incomplete scenario
    was reported as failed, so a freshly queued analysis read as all-failed.

    What kills it: counting every incomplete scenario as failed (failed 3, pending 0); or
    calling a scenario pending when ANY unfinished model has no log (failed 1, pending 2).

    A second correct implementation under which it still passes: collecting the set of
    log classes over a scenario's unfinished models and calling it pending iff that set
    is exactly ``{"no_log"}``.
    """
    status = TRITONSWMM_analysis.get_workflow_status(_stand_in(sensitivity=False, tables={"base": _REGULAR}))

    assert (status.simulations_failed, status.simulations_pending) == (2, 1)
    assert status.simulations_failed + status.simulations_pending == len(status.simulation.failed_items)


def test_sensitivity_master_splits_across_members():
    """Property: on a sensitivity master the split runs over every member's scenarios.

    Class: sensitivity masters, whose incomplete scenarios live in their members. Two
    members here: one carrying the regular table, one with a single pending scenario.

    What kills it: splitting only the master's own (empty) scenario list, or only one member.
    """
    tables = {"m0": _REGULAR, "m1": {0: {"triton": "no_log", "swmm": None}}}
    status = TRITONSWMM_analysis.get_workflow_status(_stand_in(sensitivity=True, tables=tables))

    assert (status.simulations_failed, status.simulations_pending) == (2, 2)


def test_differently_positioned_all_complete_reports_zero_of_each():
    """Control: with every model run complete, both counts are 0 -- before and after."""
    complete = {0: {"triton": None, "swmm": None}, 1: {"triton": None, "swmm": None}}
    status = TRITONSWMM_analysis.get_workflow_status(_stand_in(sensitivity=False, tables={"base": complete}))

    assert (status.simulations_failed, status.simulations_pending) == (0, 0)
