"""Synth-tier guard for the hydrology-disabled prepare path (spec S8).

PRE-FIX ARM, stated so the two-arm result is checkable rather than asserted:
against unmodified scenario.py this test FAILS, because prepare_scenario calls
write_hydrograph_files() at :848 OUTSIDE the toggle_use_swmm_for_hydrology guard
at :833, so hydrograph_outputs_gate finds the hydrographs absent AND hydro.out
absent (never produced) and raises ProcessingError. POST-FIX it passes.
"""

from __future__ import annotations

import pytest

from hhemt.exceptions import ProcessingError


@pytest.mark.requires_snakemake_subprocess
def test_prepare_scenario_with_hydrology_disabled_does_not_raise(
    synth_case_hydrology_disabled,
):
    analysis = synth_case_hydrology_disabled
    scen = analysis._retrieve_sim_run_processing_object(0)._scenario
    try:
        scen.prepare_scenario()
    except ProcessingError as exc:
        pytest.fail(
            "prepare_scenario raised on a config the validator explicitly permits "
            f"(toggle_use_swmm_for_hydrology=False): {exc}"
        )
    assert scen.log.scenario_creation_complete.get() is True
    assert not scen.scen_paths.hyg_timeseries.exists()
    assert not scen.scen_paths.hyg_locs.exists()
