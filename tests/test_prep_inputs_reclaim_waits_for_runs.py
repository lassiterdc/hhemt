"""`prep_inputs` must not be reclaimed before every enabled model has RUN.

`dats/`, `extbc/` and the per-scenario weather timeseries are consumed at RUN time by
the solver. They used to be reclaimed from the per-MODEL processing path, selected by
which model was PROCESSING, behind a `model_type == "tritonswmm"` fence and an `_unrun`
runtime guard -- and `run_swmm` and `process_tritonswmm` hang off `prepare_scenario`
independently with no DAG edge between them, which is how 123 scenarios lost `dats/`
and then aborted with seven `ERROR 361`s.

The reclaim now lives in `reclaim_scenario_scoped_classes`, called from the `--event-id`
arm of `consolidate_workflow`, which `rule consolidate_scenario` invokes. That rule fans
in on `d_process_{model}` for EVERY enabled model, and both that flag and the `c_run`
it depends on are SUCCESS markers, so arrival entails every enabled model both ran and
processed. The fence and the guard are gone because the fan-in supersedes them.

WHAT EACH ARM CATCHES, stated because the obvious rewrite of this module is vacuous.
An arm that merely asserts the three paths SURVIVE on the per-model path would be green
for the wrong reason -- nothing reclaims them there any more -- and would stay green if
the relocated reclaim fired at the wrong time. So the three arms below assert,
respectively: that the reclaim is ABSENT from the per-model path (fails if anyone
re-adds it, which is the original wrong time); that the fan-in gating the NEW call site
still names every enabled model (fails if that gate narrows); and that the relocated
call site actually reclaims (fails if the feature is disabled by the move). Every
assertion is on paths or on emitted rule text. The decline messages are deliberately
NOT asserted on: a message assertion is green pre-fix by construction.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _P
from hhemt.process_simulation import reclaim_scenario_scoped_classes
from hhemt.workflow import SnakemakeWorkflowBuilder


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
    fake._remove_reclaimed = lambda p, ad, v: __import__("shutil").rmtree(p) if Path(p).is_dir() else Path(p).unlink()
    fake._reclaim_paths = lambda *a, **k: []
    _P.remove_after_processing(fake, model_type="tritonswmm", which="both", verbose=False)
    return dats, extbc_dir, weather


def test_prep_inputs_is_no_longer_reclaimed_on_the_per_model_path(tmp_path):
    """THE regression, inverted by the relocation. `unrun=()` is the MOST permissive
    input the old code had -- every enabled model complete -- so if the class were still
    on this path it would fire here. It must not: the per-model path is precisely the
    wrong time, and this arm goes red the moment anyone re-adds it there."""
    dats, extbc_dir, weather = _probe(tmp_path, unrun=())
    assert dats.exists(), "prep_inputs reclaimed from the per-model path; it belongs to consolidate_scenario"
    assert extbc_dir.exists(), "extbc/ reclaimed from the per-model path"
    assert weather.exists(), "weather timeseries reclaimed from the per-model path"


def test_consolidate_scenario_fans_in_on_every_enabled_model(tmp_path):
    """The gate that REPLACED the `_unrun` guard. The relocated reclaim is safe only
    because its rule waits on every enabled model's `d_process` flag; this arm fails if
    that fan-in ever narrows to a subset."""
    enabled = ["tritonswmm", "triton", "swmm"]
    stub = types.SimpleNamespace(python_executable="python")
    block = SnakemakeWorkflowBuilder._build_consolidate_scenario_rule_block(
        stub,
        enabled_models=enabled,
        config_args="",
        log_dir_str="logs",
        conda_env_path="env",
        consolidate_scenario_resources="        runtime=10",
        compression_level=5,
    )
    inputs = re.search(r"^    input: (.*)$", block, re.M).group(1)
    for model in enabled:
        assert f"d_process_{model}_evt-" in inputs, f"consolidate_scenario does not wait on {model}"
    assert "--event-iloc" in block, "the reclaim call site needs the iloc the emitter holds"


def test_prep_inputs_is_reclaimed_at_the_relocated_call_site(tmp_path):
    """The move must not disable the feature: at the new site the three paths do go."""
    dats = tmp_path / "dats"
    extbc_dir = tmp_path / "extbc"
    weather = tmp_path / "weather.nc"
    dats.mkdir()
    (dats / "grid-ind156.dat").write_text("x\n")
    extbc_dir.mkdir()
    (extbc_dir / "tseries.txt").write_text("x\n")
    weather.write_text("x\n")

    scen = types.SimpleNamespace(
        event_iloc=0,
        scen_paths=types.SimpleNamespace(
            dir_weather_datfiles=dats,
            extbc_tseries=extbc_dir / "tseries.txt",
            weather_timeseries=weather,
        ),
    )
    outcome = reclaim_scenario_scoped_classes(scen, ("prep_inputs",), tmp_path, verbose=False)
    assert outcome["prep_inputs"] is True
    assert not dats.exists()
    assert not extbc_dir.exists()
    assert not weather.exists()
