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
from hhemt.process_simulation import return_fpath_wlevels, triton_raw_frame_or_raise
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
    """(a) REGRESSION, RETARGETED. Pre-fix: no raise -- the outer join silently produces a
    NaN cell and `pytest.raises` reports DID NOT RAISE. The write-side signature (MH short
    by one, the others equal) is used so the test covers the producer no cleanup fix
    reaches -- "producer" there means the CAUSE of the ragged state, the interrupted solver
    write as opposed to the interrupted clear, not the producing function. That coverage is
    a property of this FIXTURE and is unchanged.

    WHAT CHANGED, AND UNDER WHOSE AUTHORITY. The refusal moved from the enumerator to
    `triton_raw_frame_or_raise`, so the subject of the assertion moves with it. This test
    was GREEN before that move and asserted that `return_fpath_wlevels` raises; that premise
    is being deliberately overturned, on a two-specialist concurrence and a developer ruling
    of 2026-09-16 that a currently-green test may be amended when it asserts a shape being
    overturned for better architecture. THIS IS NOT, AND MAY NEVER BE CITED AS, permission
    to satisfy a RED test by weakening the instrument: the prohibited act alters a correct
    assertion to hide a defect, whereas this retires an assertion that is obsolete rather
    than wrong. The bar is unchanged -- same fixture, same exception type, same two message
    assertions -- and this test must still FAIL if the refusal is deleted from its new home.
    The enumerator's new postcondition is covered separately by test_a2/test_a3/test_a4,
    which did not exist before and which are the reason the instrument gets LARGER here
    rather than merely relocated."""
    _seed_raw(tmp_path, {"MH": 16, "H": 17, "QX": 17, "QY": 17})
    with pytest.raises(ProcessingError) as excinfo:
        triton_raw_frame_or_raise(tmp_path, _REPORTING_INTERVAL_S, model_label="test")
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


def test_a2_ragged_frame_is_RETURNED_by_the_enumerator_not_refused(tmp_path):
    """The enumerator's NEW postcondition, which has NO coverage before this change.

    EVIDENTIARY BAR: this fails under the PRE-MOVE implementation for the right reason --
    pre-move the enumerator raises ProcessingError here and the call never returns. It is
    the guard against re-introducing a raise in the enumerator, which would break the three
    eda/raw_resume_identity.py consumers and the four estate consumption points with no
    other test noticing. Same fixture as test_a, deliberately: one fixture, two subjects,
    so a reader can see that the condition did not change -- only who refuses it."""
    _seed_raw(tmp_path, {"MH": 16, "H": 17, "QX": 17, "QY": 17})
    df = return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    assert list(df.columns) == ["max_wlevel_m", "wlevel_m", "velocity_x_mps", "velocity_y_mps"]
    assert df.isna().any().any()  # the raggedness is PRESERVED, not repaired
    assert int(df.isna().sum().sum()) == 1  # exactly the one MH cell, not a widened hole


def test_a3_absent_variable_is_dropped_not_fabricated(tmp_path):
    """The DROP, which is the half the developer ruled on and which nothing else asserts.

    EVIDENTIARY BAR: fails under the pre-move implementation (which raises) AND under the
    rejected NAME alternative (which would yield four columns with NaN). The column-list
    equality is what discriminates the two arms -- a NaN-count assertion alone would pass
    under NAME for the wrong reason. The index-name assertion pins the second half of the
    fabrication defect: an unnamed empty Series also drops `timestep_min` from the index."""
    _seed_raw(tmp_path, {"MH": 2, "H": 2})
    df = return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    assert list(df.columns) == ["max_wlevel_m", "wlevel_m"]
    assert not df.isna().any().any()  # an absent variable can produce NO NaN cell
    assert df.index.name == "timestep_min"


def test_a4_directory_with_no_recognised_raw_yields_an_empty_frame(tmp_path):
    """The all-dropped case, which is a `pd.concat([])` ValueError if written naively.

    EVIDENTIARY BAR: fails under the obvious implementation of the drop (filter the list,
    concat it) because `pd.concat([])` raises ValueError: No objects to concatenate. The
    `.empty` assertion is not decoration -- it is the discriminator BOTH
    `triton_raw_frame_or_raise` and `compare_triton_raw_timeseries` key on for
    "no processable output", and a GR_*-only directory is the state that reaches it."""
    (tmp_path / "GR_0_0.bin").touch()
    df = return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    assert df.empty
    assert list(df.columns) == []


def test_a5_consumer_refuses_a_frame_whose_variable_was_dropped(tmp_path):
    """THE DEVELOPER'S FAIL-FAST. This is the one behaviour this VMS adds on top of the
    concurred shape -- "if there are downstream processees that are EXPECTING real values
    for columns that are dropped, that should be a fast fail" -- and it is the branch the
    rest of the suite does not reach: test_a3 proves the PRODUCER drops the column, and
    without this test nothing proves the CONSUMER then refuses.

    EVIDENTIARY BAR, stated as failure under the plausible WRONG implementation rather than
    pre-fix, because pre-fix the callee does not exist and a NameError is evidentially
    empty. Two wrong implementations this fails against, both of which a reviewer would
    otherwise pass: (1) one that omits the _absent check entirely -- the frame is returned,
    no exception, and `pytest.raises` reports DID NOT RAISE; (2) one that reuses the ragged
    message for both conditions -- the final assertion fails, because a configured absence
    must NOT be told to re-run, and re-running an unchanged cfg reproduces it forever.

    WHY THE CENSUS SAYS THIS BRANCH IS THE ONLY FAST FAILURE THERE IS. Nothing downstream
    fails on a dropped column: the chapter writer ITERATES df_outputs.columns rather than
    indexing by name, verify_and_flag_chapter checks only the timestep count, xr.concat
    fills a missing variable with NaN, and the summary RECONSTRUCTS max_wlevel_m from
    wlevel_m. A KeyError exists at exactly one site, four stages later, for three of the
    four variables -- and for max_wlevel_m there is no failure anywhere at all.

    SOLVER-FREE BY DEPENDENCY CHAIN, confirmed at source rather than asserted: this test
    requests only `tmp_path`, a pytest builtin, so it pulls in no project fixture; the sole
    autouse fixture reachable from it is tests/conftest.py's session-scoped
    _pytest_uses_non_interactive_snakemake_lock_clear, which sets one environment variable
    and reaches no compile; and the repo-root conftest.py declares no autouse fixture at
    all. Select it by node id -- this module's test_d/test_e/test_f take a project fixture
    and are not in this test's chain."""
    _seed_raw(tmp_path, {"MH": 2, "H": 2})

    # The producer DROPS what the solver never wrote, and does not fabricate it.
    df = return_fpath_wlevels(tmp_path, _REPORTING_INTERVAL_S)
    assert "velocity_x_mps" not in df.columns
    assert "velocity_y_mps" not in df.columns

    # The consumer then REFUSES it. This is the fail-fast.
    with pytest.raises(ProcessingError) as excinfo:
        triton_raw_frame_or_raise(tmp_path, _REPORTING_INTERVAL_S, model_label="test")
    msg = str(excinfo.value)
    assert "velocity_x_mps" in msg and "velocity_y_mps" in msg  # names WHAT is absent
    assert "print_option" in msg  # and points at the cfg, which is the actual cause
    # ...and must NOT carry the ragged remedy. This assertion is what proves the predicate
    # SPLIT produced two different remedies rather than one message wearing two hats: the
    # ragged reason ends "re-run the simulation via a force at stage='simulate' naming this
    # model arm", which for a configured absence is a trap that burns allocation forever.
    assert "force at stage='simulate'" not in msg


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
