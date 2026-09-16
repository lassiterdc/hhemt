"""Half-A prevention and Half-B force-reaches-solver coverage for the ragged-frame remedy.

EVIDENTIARY BAR, stated per test. (a), (c) and (g) are REGRESSION tests and fail pre-fix for
the right reason: pre-fix the code SKIPS instead of RAISING ((a): DID NOT RAISE; (c): the
exists() branch carries a Continue and no Raise; (g): four Continue nodes in the chunk loop).
(d), (e) and (f) are NEW-CAPABILITY coverage -- pre-fix the `force_rerun_pending` field does
not exist and they raise AttributeError/TypeError at the first `.set()`/`.get()`/keyword,
which is trivially true and evidentially empty -- so their load-bearing property is failure
under the plausible WRONG implementation, named in each docstring. (b) is the
differently-positioned SATISFYING arm of (a) and is permanently green by design; it exists so
(a) cannot be satisfied by an assertion that fires on every frame.

No test here compiles or executes a solver. (e)/(f) stop at the hotstart picker with a
sentinel exception raised by a monkeypatch, so nothing past the completion gate runs.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from hhemt import process_simulation
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import return_fpath_wlevels
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.workflow import ResolvedForceRerunSpec

_REPORTING_INTERVAL_S = 60.0  # 1 min per reporting step, so tstep index == timestep_min


def _seed_raw(out_dir: Path, counts: dict[str, int]) -> None:
    """Write empty TRITON-named raw files: {var}_{step}_0.bin for step in 1..counts[var]."""
    for var, n in counts.items():
        for step in range(1, n + 1):
            (out_dir / f"{var}_{step}_0.bin").touch()


def _chapter_loop_ast() -> ast.For:
    """The `for chunk_idx, chunk_start in enumerate(...)` loop of the chapter writer, parsed."""
    src = inspect.getsource(process_simulation.TRITONSWMM_sim_post_processing._streaming_chunked_zarr_write)
    tree = ast.parse(textwrap.dedent(src))
    chunk_loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and any(isinstance(e, ast.Name) and e.id == "chunk_idx" for e in n.target.elts)
    ]
    assert len(chunk_loops) == 1
    return chunk_loops[0]


def test_a_ragged_frame_is_refused_and_names_the_short_variable(tmp_path):
    """(a) REGRESSION. Pre-fix: no raise -- the outer join silently produces a NaN cell and
    `pytest.raises` reports DID NOT RAISE. The write-side signature (MH short by one, the
    others equal) is used so the test covers the producer no cleanup fix reaches."""
    _seed_raw(tmp_path, {"MH": 16, "H": 17, "QX": 17, "QY": 17})
    with pytest.raises(ProcessingError) as excinfo:
        return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    msg = str(excinfo.value)
    assert "MH" in msg and "17" in msg  # the short variable and the missing timestep_min
    assert "'H'" not in msg.replace("MH", "")  # the long variables are not named as short


def test_b_equal_index_frame_is_not_refused(tmp_path):
    """(b) SATISFYING ARM of (a), differently positioned (all four equal at 17). Green pre-
    and post-fix by design; it discriminates an over-firing precondition."""
    _seed_raw(tmp_path, {"MH": 17, "H": 17, "QX": 17, "QY": 17})
    df = return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    assert df.shape == (17, 4)
    assert not df.isna().any().any()


def test_c_missing_file_between_listing_and_read_raises_not_skips():
    """(c) REGRESSION, structural. Pre-fix: the FIRST `for tstep_min in chunk_timesteps`
    loop's `if not f.exists()` branch carries a Continue and no Raise, so the two assertions
    below fail on the production loop itself. Post-fix: the branch body is exactly one
    `raise ProcessingError(...)`. Structural rather than end-to-end because driving
    `_streaming_chunked_zarr_write` needs a DEM raster and a zarr target; a re-implemented
    decision loop inside the test would test the test, not the code."""
    chunk_loop = _chapter_loop_ast()
    tstep_loops = sorted(
        (
            n
            for n in ast.walk(chunk_loop)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "tstep_min"
        ),
        key=lambda n: n.lineno,
    )
    assert tstep_loops, "the per-variable tstep_min loop must exist"
    read_loop = tstep_loops[0]
    exists_ifs = [
        n
        for n in ast.walk(read_loop)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Call)
        and isinstance(n.test.operand.func, ast.Attribute)
        and n.test.operand.func.attr == "exists"
    ]
    assert len(exists_ifs) == 1, "exactly one `if not f.exists()` branch in the read loop"
    branch = exists_ifs[0]
    assert not [n for n in ast.walk(branch) if isinstance(n, ast.Continue)], "the exists() branch must not continue"
    raises = [n for n in branch.body if isinstance(n, ast.Raise)]
    assert len(raises) == 1 and isinstance(raises[0].exc, ast.Call), "the exists() branch must raise"
    assert isinstance(raises[0].exc.func, ast.Name) and raises[0].exc.func.id == "ProcessingError"


def test_g_chapter_build_loop_contains_no_continue():
    """(g) STRUCTURAL POST-CONDITION of Component 2 + Deltas 2a/2b. Pre-fix: the `for
    chunk_idx` loop body carries FOUR `continue` statements (the dead index guard, the
    exists() skip, the empty-variable skip, the empty-chunk skip) and this assertion fails
    by count. Post-fix: zero. Guards against a dead-guard survivor."""
    chunk_loop = _chapter_loop_ast()
    continues = [n for n in ast.walk(chunk_loop) if isinstance(n, ast.Continue)]
    assert continues == [], f"chapter-build loop must contain no `continue`; found {len(continues)}"


def _spec(models, stage="simulate"):
    return ResolvedForceRerunSpec(scope="all", tokens=(), stage=stage, models=tuple(models))


def test_d_invalidator_sets_marker_for_named_models_only(synth_all_models_analysis):
    """(d) NEW-CAPABILITY. Wrong-implementation bar: a set keyed on `spec.stage` (or
    ungated) writes the marker for the reuse-site call in the second block and those
    assertions fail; a set that ignores `spec.models` writes it for triton/tritonswmm in
    the first block and those assertions fail."""
    analysis = synth_all_models_analysis
    analysis._invalidate_processing_log_for_force_rerun(_spec(["swmm"], stage="simulate"), set_force_marker=True)
    for event_iloc in range(len(analysis.df_sims)):
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        assert scen.get_log("swmm").force_rerun_pending.get() is True
        assert not scen.get_log("triton").force_rerun_pending.get()
        assert not scen.get_log("tritonswmm").force_rerun_pending.get()
    # Reset and re-run WITHOUT the opt-in, at the literal stage="simulate" the three
    # reuse sites pass: nothing may be set -- this is the reprocess-must-not-re-run-the-
    # solver guarantee, and it is what a stage-keyed implementation gets wrong.
    for event_iloc in range(len(analysis.df_sims)):
        _ml = TRITONSWMM_scenario(event_iloc, analysis).get_log("swmm")
        _ml.force_rerun_pending.set(False)
    analysis._invalidate_processing_log_for_force_rerun(_spec(["swmm", "triton", "tritonswmm"], stage="simulate"))
    for event_iloc in range(len(analysis.df_sims)):
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        for m in ("swmm", "triton", "tritonswmm"):
            assert not scen.get_log(m).force_rerun_pending.get()


class _PastTheGate(Exception):
    """Sentinel raised by the stubbed hotstart picker: reaching it proves the completion gate
    was bypassed, and stops the command build before anything solver-shaped happens."""


def _stub_prune_and_picker(monkeypatch, run, pruned: list) -> None:
    monkeypatch.setattr(
        type(run),
        "prune_hotstart_cfgs_above_step",
        lambda self, mt, *, target_step: pruned.append((mt, target_step)) or 0,
    )
    monkeypatch.setattr(
        type(run),
        "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation",
        lambda self, model_type: (_ for _ in ()).throw(_PastTheGate()),
    )


def test_e_marker_bypasses_completion_gate_and_prunes(synth_all_models_analysis, monkeypatch):
    """(e) NEW-CAPABILITY. Wrong-implementation bar: an implementation that consults the
    marker AFTER the gate returns None here and `_PastTheGate` is never raised; one that
    bypasses without pruning leaves `pruned` empty."""
    analysis = synth_all_models_analysis
    scen = TRITONSWMM_scenario(0, analysis)
    run = scen.run
    monkeypatch.setattr(type(scen), "model_run_completed", lambda self, mt: True)
    pruned: list[tuple[str, int]] = []
    _stub_prune_and_picker(monkeypatch, run, pruned)
    scen.get_log("triton").force_rerun_pending.set(True)
    with pytest.raises(_PastTheGate):
        run.prepare_simulation_command(pickup_where_leftoff=True, verbose=False, model_type="triton")
    assert pruned == [("triton", 0)]


def test_f_without_marker_every_non_force_path_is_unchanged(synth_all_models_analysis, monkeypatch):
    """(f) FIELD-SET INTERSECTION IS EMPTY ON THE NON-FORCE PATH. With the marker unset
    (legacy None) a completed scenario short-circuits exactly as before: returns None, the
    picker is never reached, nothing is pruned. Wrong-implementation bar: a consult that
    treats None as set prunes here and reaches the picker. The precondition assertion holds
    because the fixture is built start_from_scratch (a fresh tree per test)."""
    analysis = synth_all_models_analysis
    scen = TRITONSWMM_scenario(0, analysis)
    run = scen.run
    monkeypatch.setattr(type(scen), "model_run_completed", lambda self, mt: True)
    pruned: list = []
    _stub_prune_and_picker(monkeypatch, run, pruned)
    assert scen.get_log("triton").force_rerun_pending.get() in (None, False)
    result = run.prepare_simulation_command(pickup_where_leftoff=True, verbose=False, model_type="triton")
    assert result is None
    assert pruned == []
