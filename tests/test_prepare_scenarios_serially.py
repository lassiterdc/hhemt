"""Fixture-free pins for the R12 rider (VMS members b1 and b2).

(b1) the prepare-scenario launcher closure returns the runner subprocess return
code; (b2) run_prepare_scenarios_serially post-checks the scenario logs and raises
RuntimeError naming every scenario whose scenario_creation_complete is not True,
mirroring TRITONSWMM_sensitivity_analysis.prepare_scenarios_in_each_analysis.
No compile, no simulation: a skip_run cached case under a tmp_path override and
monkeypatched collaborators.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def cached_case(tmp_path, monkeypatch):
    """Construct-only (skip_run) cached case under a tmp_path override: no compile,
    no simulation, no shared-root write."""
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="prepare_serially_probe", start_from_scratch=False, skip_run=True
    )


def test_prepare_launcher_returns_subprocess_return_code(cached_case, monkeypatch):
    """(b1) The launcher closure returns the runner's return code instead of None."""
    import hhemt.scenario as scenario_mod
    from hhemt.scenario import TRITONSWMM_scenario

    monkeypatch.setattr(scenario_mod, "run_subprocess_with_tee", lambda **kwargs: SimpleNamespace(returncode=7))
    scen = TRITONSWMM_scenario(0, cached_case.analysis)
    launcher = scen._create_subprocess_prepare_scenario_launcher()
    assert launcher() == 7


def _patch_serial_driver(monkeypatch, analysis, *, all_created: bool):
    monkeypatch.setattr(analysis, "retrieve_prepare_scenario_launchers", lambda **kwargs: [lambda: 0])
    monkeypatch.setattr(analysis, "_update_log", lambda *a, **k: None)
    monkeypatch.setattr(type(analysis), "_all_scenarios_created", property(lambda self: all_created))
    monkeypatch.setattr(type(analysis), "_scenarios_not_created", property(lambda self: ["sims/event_index.0"]))


def test_run_prepare_scenarios_serially_raises_when_a_scenario_is_not_created(cached_case, monkeypatch):
    """(b2) violating arm: a not-True completion flag after the loop raises, naming the scenario."""
    _patch_serial_driver(monkeypatch, cached_case.analysis, all_created=False)
    with pytest.raises(RuntimeError, match="Preparation failed") as excinfo:
        cached_case.analysis.run_prepare_scenarios_serially()
    assert "sims/event_index.0" in str(excinfo.value)


def test_run_prepare_scenarios_serially_returns_when_all_scenarios_created(cached_case, monkeypatch):
    """(b2) satisfying arm: every completion flag True -> normal return."""
    _patch_serial_driver(monkeypatch, cached_case.analysis, all_created=True)
    assert cached_case.analysis.run_prepare_scenarios_serially() is None
