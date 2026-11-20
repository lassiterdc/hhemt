"""Independent review of the prepare-retry capture fix (``write_hydrograph_files`` order and the gate's new arm).

No solver and no Snakemake: the committed multi-sim golden carries a real ``hydro.inp`` and ``hydro.out``; a kill is
simulated by making one step of ``write_hydrograph_files`` raise.

Invariant under test, stated once: after ANY interrupted ``write_hydrograph_files``, a scenario whose two hydrograph
log flags are both True has an openable per-node capture (equivalently: no kill point leaves flags without the
capture), and a retry of the same prepare ends with the capture present and the flags True.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.golden_trees import construct_golden_analysis


def _fresh(a, event_iloc: int = 0):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _capture_dir(scen) -> Path:
    from hhemt.swmm_runoff_modeling import hydrology_capture_path

    return hydrology_capture_path(scen)


def _capture_present(scen) -> bool:
    from hhemt.swmm_runoff_modeling import _capture_openable

    return _capture_openable(_capture_dir(scen))


def _flags(scen) -> bool:
    return bool(scen.log.hyg_timeseries_created.get() and scen.log.hyg_locs_created.get())


def _hydro_out(scen) -> Path:
    return Path(str(scen.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))


def _scenario(golden_extractor):
    """The state before the first prepare attempt: no capture, no hydrograph files, no flags."""
    system_dir = golden_extractor("multisim")
    a = construct_golden_analysis(system_dir)
    scen = _fresh(a)
    shutil.rmtree(_capture_dir(scen), ignore_errors=True)
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink(missing_ok=True)
    scen.log.hyg_timeseries_created.set(False)
    scen.log.hyg_locs_created.set(False)
    assert _hydro_out(scen).exists(), "control: the golden carries hydro.out, so the capture is writable"
    return a, scen


class _Killed(RuntimeError):
    pass


def _inject(monkeypatch, point: str) -> None:
    """Make one step of ``write_hydrograph_files`` raise. The points span the whole function."""
    import hhemt.swmm_runoff_modeling as srm

    def boom(*args, **kwargs):
        raise _Killed(point)

    if point == "capture_write":
        monkeypatch.setattr(srm.SWMMRunoffModeler, "_write_node_inflow_capture", boom)
    elif point == "capture_publish":
        monkeypatch.setattr(srm, "write_zarr", boom)
    elif point == "hydrograph_frame":
        monkeypatch.setattr(srm, "hydrograph_frame", boom)
    elif point == "after_flags_read_csv":
        # the first pd.read_csv in the function is after BOTH log flags are set
        monkeypatch.setattr(pd, "read_csv", boom)
    else:  # pragma: no cover
        raise AssertionError(point)


KILL_POINTS = ["capture_write", "capture_publish", "hydrograph_frame", "after_flags_read_csv"]


@pytest.mark.parametrize("point", KILL_POINTS)
def test_no_kill_point_leaves_the_flags_without_the_capture_and_the_retry_restores_it(
    golden_extractor, monkeypatch, point
):
    """Property: at every kill point, flags set implies the capture exists, and a retry ends complete.

    Class: REGRESSION over the four kill points that span the function (capture write entry, the zarr publish,
    the hydrograph build, and after both flags). Violating arm: the pre-fix order sets the flags before the capture,
    so ``capture_write``, ``capture_publish`` and ``after_flags_read_csv`` each leave flags True with no capture.
    Satisfying arm in a different correct state: ``hydrograph_frame`` dies after the capture and before any flag
    (capture present, flags False). Both are asserted by the same implication, so the assertion is not keyed on one
    state. Must-fire control: the injected failure really raised.
    """
    a, scen = _scenario(golden_extractor)
    with monkeypatch.context() as m:
        _inject(m, point)
        with pytest.raises(_Killed):
            scen._runoff_modeler.write_hydrograph_files()
    killed = _fresh(a)
    assert (not _flags(killed)) or _capture_present(killed), f"flags set without the capture after a kill at {point}"

    killed._runoff_modeler.write_hydrograph_files()
    after = _fresh(a)
    assert _capture_present(after)
    assert _flags(after)
    assert Path(after.scen_paths.hyg_timeseries).exists() and Path(after.scen_paths.hyg_locs).exists()


def test_a_non_memory_capture_failure_leaves_nothing_stranded(golden_extractor, monkeypatch):
    """Property: a capture write that fails for a non-memory reason (permissions) stops BEFORE any hydrograph file or
    flag, leaves no partial store, and the retry completes.

    Class: any exception from the capture write. Must-fire control: the PermissionError really propagated.
    """
    import hhemt.swmm_runoff_modeling as srm

    a, scen = _scenario(golden_extractor)

    def _denied(*args, **kwargs):
        raise PermissionError("simulated: processed/ not writable")

    with monkeypatch.context() as m:
        m.setattr(srm, "write_zarr", _denied)
        with pytest.raises(PermissionError):
            scen._runoff_modeler.write_hydrograph_files()
    failed = _fresh(a)
    assert not _flags(failed)
    assert not Path(failed.scen_paths.hyg_timeseries).exists()
    assert not Path(failed.scen_paths.hyg_locs).exists()
    assert not _capture_dir(failed).exists()

    failed._runoff_modeler.write_hydrograph_files()
    assert _capture_present(_fresh(a)) and _flags(_fresh(a))


def test_the_gate_does_not_recompute_a_healthy_scenario_and_converges_in_one_pass(golden_extractor, monkeypatch):
    """Property: with the capture openable the gate skips and nothing is rewritten; a capture-less logged scenario is
    rebuilt exactly ONCE and then skips (no loop).

    Class: healthy prepared scenario; capture-less-with-hydro.out scenario. Must-fire control: the spy counts the
    first, real write, so a zero on the healthy re-entry is not a broken counter.
    """
    from hhemt.swmm_runoff_modeling import SWMMRunoffModeler, hydrograph_outputs_gate

    a, scen = _scenario(golden_extractor)
    calls: list[int] = []
    real = SWMMRunoffModeler._write_node_inflow_capture

    def spy(self, node_inflow):
        calls.append(1)
        return real(self, node_inflow)

    monkeypatch.setattr(SWMMRunoffModeler, "_write_node_inflow_capture", spy)

    scen._runoff_modeler.write_hydrograph_files()
    assert calls == [1], "control: the first prepare wrote the capture once"

    healthy = _fresh(a)
    assert hydrograph_outputs_gate(healthy) == "skipped"
    healthy._runoff_modeler.write_hydrograph_files()
    assert calls == [1], "a healthy re-entry must not rewrite the capture"

    shutil.rmtree(_capture_dir(healthy))
    stranded = _fresh(a)
    assert hydrograph_outputs_gate(stranded) == "fell_through"
    stranded._runoff_modeler.write_hydrograph_files()
    assert calls == [1, 1], "the capture-less scenario is rebuilt once"
    again = _fresh(a)
    assert hydrograph_outputs_gate(again) == "skipped"
    again._runoff_modeler.write_hydrograph_files()
    assert calls == [1, 1], "and then skips: no loop"


def test_a_post_reclaim_scenario_without_hydro_out_still_skips_with_and_without_a_capture(golden_extractor):
    """Property: hydro.out gone means the fast path, whether or not a capture exists (the post-reclaim re-prep).

    Class: both capture states under a reclaimed hydro.out. Must-fire control: the files really are logged and present
    and hydro.out really is absent before the gate is read.
    """
    from hhemt.swmm_runoff_modeling import hydrograph_outputs_gate

    a, scen = _scenario(golden_extractor)
    scen._runoff_modeler.write_hydrograph_files()
    _hydro_out(scen).unlink()
    with_capture = _fresh(a)
    assert _flags(with_capture) and not _hydro_out(with_capture).exists() and _capture_present(with_capture)
    assert hydrograph_outputs_gate(with_capture) == "skipped"
    shutil.rmtree(_capture_dir(with_capture))
    assert hydrograph_outputs_gate(_fresh(a)) == "skipped"


def test_capture_openable_rejects_an_empty_dir_a_temp_leftover_and_a_corrupt_store_but_accepts_a_real_one(
    golden_extractor,
):
    """Property: ``_capture_openable`` is True only for a store that opens. Violating inputs: empty directory, a
    ``.tmp`` sibling with the final absent, a store whose root document is truncated. Satisfying input: the real one.
    Must-fire control: the real store returns True first.
    """
    from hhemt.swmm_runoff_modeling import _capture_openable

    a, scen = _scenario(golden_extractor)
    scen._runoff_modeler.write_hydrograph_files()
    final = _capture_dir(scen)
    assert _capture_openable(final) is True

    tmp = final.with_name(final.name + ".tmp")
    shutil.move(str(final), str(tmp))
    assert _capture_openable(final) is False, "final absent with only the temp sibling present"
    shutil.move(str(tmp), str(final))

    root_doc = final / "zarr.json"
    good = root_doc.read_bytes()
    root_doc.write_bytes(good[: len(good) // 2])
    assert _capture_openable(final) is False, "truncated root document"
    root_doc.write_bytes(good)
    assert _capture_openable(final) is True

    shutil.rmtree(final)
    final.mkdir()
    assert _capture_openable(final) is False, "empty directory"
