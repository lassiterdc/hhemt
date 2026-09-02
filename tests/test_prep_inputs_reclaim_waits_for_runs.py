"""`prep_inputs` must not delete inputs a model has not yet consumed.

`dats/`, `extbc/` and the per-scenario weather timeseries are consumed at RUN time by
the solver, but the reclaim branch is selected by which model is PROCESSING. The
existing `model_type == "tritonswmm"` guard stops three models racing to unlink the same
file; it does not stop a model whose simulation has not started from losing its inputs.
`run_swmm` and `process_tritonswmm` hang off `prepare_scenario` independently -- there is
no DAG edge between them -- which is how 123 scenarios lost `dats/` and then aborted with
seven `ERROR 361`s.

Every assertion is on whether the three PATHS still exist. The decline also prints, and
that message is deliberately NOT asserted on: a message assertion is green pre-fix by
construction.
"""

from __future__ import annotations

import types
from pathlib import Path

from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _P


def _probe(tmp_path: Path, *, unrun: tuple[str, ...]):
    """Drive the real `remove_after_processing` with policy == ['prep_inputs'] only."""
    dats = tmp_path / "dats"
    extbc_dir = tmp_path / "extbc"
    weather = tmp_path / "weather.nc"
    dats.mkdir()
    (dats / "grid-ind156.dat").write_text("x\n")
    extbc_dir.mkdir()
    (extbc_dir / "tseries.txt").write_text("x\n")
    weather.write_text("x\n")

    enabled = ("tritonswmm", "swmm")
    fake = types.SimpleNamespace(
        _analysis=types.SimpleNamespace(
            cfg_analysis=types.SimpleNamespace(remove_after_processing=["prep_inputs"]),
            analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path),
        ),
        _run=types.SimpleNamespace(model_types_enabled=enabled),
        _scenario=types.SimpleNamespace(
            event_iloc=0,
            model_run_completed=lambda m: m not in unrun,
        ),
        scen_paths=types.SimpleNamespace(
            dir_weather_datfiles=dats,
            extbc_tseries=extbc_dir / "tseries.txt",
            weather_timeseries=weather,
        ),
        log=types.SimpleNamespace(prep_inputs_reclaimed=None),
    )
    fake._reclaim_classes = _P._reclaim_classes
    fake._resolve_clear_raw = lambda *_a, **_k: "none"
    fake._remove_reclaimed = lambda p, ad, v: (
        __import__("shutil").rmtree(p) if Path(p).is_dir() else Path(p).unlink()
    )
    fake._reclaim_paths = lambda *a, **k: []
    _P.remove_after_processing(fake, model_type="tritonswmm", which="both", verbose=False)
    return dats, extbc_dir, weather


def test_prep_inputs_declines_while_a_model_has_not_run(tmp_path):
    """THE regression. Pre-fix all three paths are deleted while `swmm` has not run."""
    dats, extbc_dir, weather = _probe(tmp_path, unrun=("swmm",))
    assert dats.exists(), "dats/ deleted while the standalone SWMM run had not happened"
    assert extbc_dir.exists(), "extbc/ deleted while a model had not run"
    assert weather.exists(), "weather timeseries deleted while a model had not run"


def test_prep_inputs_proceeds_once_every_enabled_model_has_run(tmp_path):
    """The guard must not disable the feature: with all runs complete it still reclaims."""
    dats, extbc_dir, weather = _probe(tmp_path, unrun=())
    assert not dats.exists()
    assert not extbc_dir.exists()
    assert not weather.exists()
