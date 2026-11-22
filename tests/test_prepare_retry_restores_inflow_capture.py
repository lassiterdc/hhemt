"""A prepare retried after a kill restores the per-node inflow capture.

Measured on Rivanna (member S9): a prepare job stopped OUT_OF_MEMORY, Snakemake retried it, and consolidation then
failed for good with ``has no per-node inflow capture and no hydro.out to rebuild it from``. The mechanism is the
order inside ``write_hydrograph_files``: the hydrograph files and their log flags were written BEFORE the capture, and
``hydrograph_outputs_gate`` skips a scenario whose hydrographs are logged and present, so a kill between the flags and
the capture write left a scenario the retry skipped; the default reclaim then removed ``hydro.out``.

No solver and no Snakemake: the committed multi-sim golden carries a real ``hydro.inp`` and ``hydro.out``, and the
kill is simulated by making the capture writer raise.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.fixtures.golden_trees import construct_golden_analysis


def _fresh(a, event_iloc: int = 0):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _capture_dir(scen) -> Path:
    from hhemt.swmm_runoff_modeling import hydrology_capture_path

    return hydrology_capture_path(scen)


def _scenario(golden_extractor):
    system_dir = golden_extractor("multisim")
    a = construct_golden_analysis(system_dir)
    scen = _fresh(a)
    # the state before the first prepare attempt: no capture, no hydrograph files, no flags
    shutil.rmtree(_capture_dir(scen), ignore_errors=True)
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink(missing_ok=True)
    scen.log.hyg_timeseries_created.set(False)
    scen.log.hyg_locs_created.set(False)
    return a, scen


def _capture_present(scen) -> bool:
    from hhemt.swmm_runoff_modeling import _capture_openable

    return _capture_openable(_capture_dir(scen))


def test_a_prepare_killed_during_the_capture_write_and_retried_ends_with_the_capture(golden_extractor, monkeypatch):
    """Property: a first attempt that dies while the capture is being written, then a retry of the same prepare, ends
    with the capture present and the hydrographs flagged.

    Class: REGRESSION (the flags were set before the capture write, so the retry's gate skipped). Reachability: the
    real ``write_hydrograph_files`` and gate on the golden scenario, with the capture writer raising ``MemoryError`` on
    the first attempt only. Must-fire control: the first attempt really died (it raised and left no capture), so the
    retry is what produced the capture rather than the first attempt.
    Killed by: setting the hydrograph flags before the capture write. Passes under a second correct implementation
    that leaves the flags unset, or that detects the missing capture in the gate.
    """
    from hhemt.swmm_runoff_modeling import SWMMRunoffModeler

    a, scen = _scenario(golden_extractor)

    def _killed(self, node_inflow):
        raise MemoryError("simulated memory-limit kill during the capture write")

    with monkeypatch.context() as m:
        m.setattr(SWMMRunoffModeler, "_write_node_inflow_capture", _killed)
        with pytest.raises(MemoryError):
            scen._runoff_modeler.write_hydrograph_files()
    retry = _fresh(a)
    assert not _capture_present(retry), "control: the killed first attempt wrote no capture"

    retry._runoff_modeler.write_hydrograph_files()
    after = _fresh(a)
    assert _capture_present(after)
    assert Path(after.scen_paths.hyg_timeseries).exists() and Path(after.scen_paths.hyg_locs).exists()
    assert after.log.hyg_timeseries_created.get() is True and after.log.hyg_locs_created.get() is True


def test_a_scenario_logged_and_present_without_a_capture_gets_it_while_hydro_out_exists(golden_extractor):
    """Property: the already-written gate does not skip a scenario whose hydrographs are logged and present but whose
    capture is missing while ``hydro.out`` still exists, and the build that follows writes the capture.

    Class: REGRESSION for a scenario left in that state by a toolkit that set the flags first. Both-states anchor:
    with ``hydro.out`` removed the same scenario still skips (the post-reclaim re-prep the fast path exists for), so
    the arm is keyed on ``hydro.out`` rather than firing on every capture-less scenario.
    Killed by: removing the new arm, or making it ignore ``hydro.out``.
    """
    from hhemt.swmm_runoff_modeling import hydrograph_outputs_gate

    a, scen = _scenario(golden_extractor)
    scen._runoff_modeler.write_hydrograph_files()  # hydrographs, flags and capture all written
    shutil.rmtree(_capture_dir(scen))
    scen = _fresh(a)
    assert not _capture_present(scen)

    assert hydrograph_outputs_gate(scen) == "fell_through"
    scen._runoff_modeler.write_hydrograph_files()
    assert _capture_present(_fresh(a))

    # the post-reclaim state: no hydro.out and no capture, hydrographs logged and present -> still the fast path
    shutil.rmtree(_capture_dir(scen))
    Path(str(scen.scen_paths.swmm_hydro_inp).replace(".inp", ".out")).unlink()
    assert hydrograph_outputs_gate(_fresh(a)) == "skipped"
