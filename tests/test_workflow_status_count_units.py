"""``get_workflow_status`` compares every count against a total of the SAME unit.

The status report's preparation, simulation and processing phases each subtract a list
of incomplete items from a total. Every one of those lists holds one entry per SCENARIO
(one weather event within one member): a scenario is "not run" when any of its enabled
models has not completed, and "not processed" when its TRITON or SWMM outputs are not
all written. The total therefore has to count scenarios too. Counting it in model runs
(scenarios times enabled models) makes the two units agree only when one model is
enabled; with two, an analysis where nothing has happened reports half of it done.

No ``TRITONSWMM_analysis`` is constructed: the method is called unbound on a stand-in
carrying only the attributes it reads, with two models enabled and two scenarios.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hhemt.analysis import TRITONSWMM_analysis


class _Field:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _stand_in(*, not_prepared, not_run, triton_missing, swmm_missing):
    """Two scenarios, two enabled models (TRITON-only and SWMM-only), so 4 model runs."""
    scenarios = 2
    models = ["triton", "swmm"]
    system = SimpleNamespace(
        log=SimpleNamespace(
            dem_processed=_Field(True),
            mannings_processed=_Field(True),
            compilation_tritonswmm_cpu_successful=_Field(True),
        ),
        cfg_system=SimpleNamespace(toggle_use_constant_mannings=True),
    )
    return SimpleNamespace(
        _system=system,
        cfg_analysis=SimpleNamespace(execution_environment="native", analysis_id="an_x"),
        analysis_paths=SimpleNamespace(analysis_dir=Path("/nonexistent/an_x")),
        log=SimpleNamespace(datatree_consolidation_complete=_Field(False)),
        n_scenarios=scenarios,
        n_sims=scenarios * len(models),
        _get_enabled_model_types=lambda: list(models),
        _all_scenarios_created=not not_prepared,
        _scenarios_not_created=list(not_prepared),
        _all_sims_run=not not_run,
        _scenarios_not_run=list(not_run),
        _split_incomplete_scenarios=lambda: (list(not_run), []),
        _TRITON_time_series_not_processed=list(triton_missing),
        _SWMM_time_series_not_processed=list(swmm_missing),
    )


def test_nothing_done_reports_zero_progress_with_two_models():
    """Property: with no scenario prepared, run or processed, every phase reports 0 done.

    Class: any analysis with more than one model enabled whose incomplete lists name
    every scenario. This test is the two-scenario, two-model point of that class.

    Why it is required: the status report is what decides whether a run is finished;
    with the totals in model runs, ``n_total - len(not_prepared)`` reports 2 of 4
    scenarios prepared when none is, and processing reports 2 of 4 outputs written.

    What kills it: computing the phase totals from ``n_sims`` (scenarios times enabled
    models) while the incomplete lists count scenarios.

    A second correct implementation under which it still passes: counting every phase in
    model runs instead (one incomplete entry per scenario and model), which also yields
    zero completed when nothing has run.
    """
    every = ["/sims/evt-1", "/sims/evt-2"]
    status = TRITONSWMM_analysis.get_workflow_status(
        _stand_in(not_prepared=every, not_run=every, triton_missing=every, swmm_missing=every)
    )

    assert status.preparation.progress == 0.0
    assert status.simulation.progress == 0.0
    assert status.processing.progress == 0.0
    assert status.simulations_completed == 0


def test_differently_positioned_half_done_reports_one_half():
    """Satisfying input in a different position: one of two scenarios complete everywhere.

    With one scenario left in every incomplete list, each phase is half done whichever
    unit the totals use, provided the count and its total share one. Before the fix the
    preparation phase read 3 of 4 here.
    """
    one = ["/sims/evt-2"]
    status = TRITONSWMM_analysis.get_workflow_status(
        _stand_in(not_prepared=one, not_run=one, triton_missing=one, swmm_missing=one)
    )

    assert status.preparation.progress == 0.5
    assert status.simulation.progress == 0.5
    assert status.processing.progress == 0.5
    assert status.simulations_completed * 2 == status.total_simulations


def test_all_done_reports_complete():
    """Control: empty incomplete lists give full progress in both units, before and after."""
    status = TRITONSWMM_analysis.get_workflow_status(
        _stand_in(not_prepared=[], not_run=[], triton_missing=[], swmm_missing=[])
    )

    assert status.preparation.progress == 1.0
    assert status.simulation.progress == 1.0
    assert status.processing.progress == 1.0
    assert status.simulations_completed == status.total_simulations
