"""Reviewer's tests for the exchange-replay resume fallback (change 1118-tk2), on the real arms' geometry.

THE PROBLEM. A coupled simulation killed at its wall limit leaves a checkpoint ``config_N.cfg`` whose exchange-replay
side-file stops short of it, and every later resume aborts in the solver with ``Exchange-replay side-file is
incomplete for this checkpoint``. ``TRITONSWMM_run.fall_back_from_exchange_replay_refusal`` sets aside the
checkpoints the side-file does not cover so the next resume starts from one it does.

WHAT THESE TESTS ADD TO THE CODER'S. The coder's synthetic trees use a 240 s reporting interval and invented times.
These replay the three real arms that exhausted their failure budget (R10, R11, R2c of ``phase_b_gpu``): the
checkpoint numbering, the 120 s ``sim_start_time`` of every cfg, and the exact refusal text the solver wrote into each
arm's model log were read from the Rivanna files (names, sizes and one ``sim_start_time=`` line per cfg; no simulation
data). Every expectation here is derived from the refusal's own reached time by an independent oracle, never copied
from the code under test. Each property is also run against deliberately wrong implementations (mutants) and must go
red there, so a green result cannot come from a property that would pass for any implementation.

No test compiles or runs a solver. The trees are synthetic and the "solver" is a few lines that refuse a checkpoint
beyond a stated reach.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import hhemt
from hhemt.run_simulation import ExchangeReplayFallback, TRITONSWMM_run, return_the_reporting_step_from_a_cfg

if TYPE_CHECKING:
    from hhemt.scenario import TRITONSWMM_scenario

#: Reporting interval of the real arms: every cfg read on Rivanna carries ``sim_start_time`` = 120 * its step.
_PRINT_S = 120

#: The three real arms: (arm, number of cfgs in its set-aside stale directory, ``reached`` and ``requires`` as the
#: solver printed them in that arm's model log). These are INPUTS read from the Rivanna logs; no expected output of
#: the code under test is stored here.
_REAL_ARMS = [("R10", 693, "82901", "83160"), ("R11", 741, "88774.2", "88920"), ("R2c", 717, "85920", "86040")]

#: ``ERROR`` in the solver is ``RED << "[ERROR] " << RESET``; the real log carries the colour codes.
_ERROR_TAG = "\x1b[31m[ERROR] \x1b[0m"

_CLEAN_START = "         Re-run this coupled simulation from a clean start (checkpoint_id=0).\n"

#: ``(label, text)`` of every OTHER refusal the solver's ``position_exchange_log`` can print, quoted from
#: ``swmm_triton.h`` of the solver checkout the arms were built from (lines 3118 to 3180). Four of the five end
#: with the same "clean start" sentence the incomplete-side-file refusal ends with; none carries a reached time.
_OTHER_SOLVER_REFUSALS = [
    (
        "side-file missing",
        _ERROR_TAG + "Coupled TRITON-SWMM resume requires the exchange-replay side-file,\n"
        "         but it was not found: out_tritonswmm/swmm/hydraulics_exchange_replay.bin\n"
        "         This checkpoint predates the resume fix (no side-file was written).\n" + _CLEAN_START,
    ),
    (
        "bad header",
        _ERROR_TAG + "Exchange-replay side-file has a bad/missing header: out_tritonswmm/swmm/x.bin\n" + _CLEAN_START,
    ),
    (
        "node count mismatch",
        _ERROR_TAG + "Exchange-replay side-file node count (8) does not match this run (9). "
        "The .inp / coupling changed between runs.\n",
    ),
    (
        "value width mismatch",
        _ERROR_TAG + "Exchange-replay side-file stores 4-byte values but this build's value_t is 8 bytes.\n"
        "         The side-file was written by a build of the other precision\n"
        "         (USE_SINGLE_PRECISION differs). Re-run with the writing build's precision, or\n"
        "         from a clean start (checkpoint_id=0).\n",
    ),
    (
        "outfall stage file missing",
        _ERROR_TAG + "Coupled TRITON-SWMM resume with swmm_driven_outfalls requires the outfall stage file,\n"
        "         but it was not found: out_tritonswmm/outfall_stage.bin\n" + _CLEAN_START,
    ),
]


def _stream(x: float) -> str:
    """``x`` as the solver's ``std::cerr << double`` prints it: six significant digits, so ``1.0368e+06``."""
    return f"{x:.6g}"


def _refusal_text(reached: str, required: str) -> str:
    """The incomplete-side-file refusal as it sits in a real model log, line break and colour codes included."""
    return (
        "[..] SWMM initialized\n"
        f"{_ERROR_TAG}Exchange-replay side-file is incomplete for this checkpoint: reached\n"
        f"         t={reached} s but checkpoint_id requires {required} s "
        "(side-file/checkpoint mismatch, e.g. truncated by a hard kill).\n" + _CLEAN_START
    )


def _refusal_text_older_wording(reached: str, required: str) -> str:
    """The same refusal as the solver builds at ``3a832f7`` and ``b3820a44`` print it (``swmm_triton.h`` lines 644-647).

    Same quantity, ``cum`` after the replay loop, under other words: ``replayed`` / ``up to`` where the builds from
    ``faba2232`` on say ``reached`` / ``t=``. ``3a832f7`` is the build that introduced coupled resume and the one the
    toolkit's own comment in ``prepare_simulation_command`` names as its pinned TRITON.
    """
    return (
        "[..] SWMM initialized\n"
        f"{_ERROR_TAG}Exchange-replay side-file is incomplete for this checkpoint: replayed\n"
        f"         up to {reached} s but checkpoint_id requires {required} s "
        "(side-file/checkpoint mismatch, e.g. truncated by a hard kill).\n" + _CLEAN_START
    )


def _cfg_text(step: int) -> str:
    """One checkpoint cfg in the solver's own form: equal line count at every step, ``sim_start_time`` = 120 * step."""
    return (
        "\n".join(
            [
                'output_folder="out_tritonswmm"',
                f"checkpoint_id={step}",
                f"sim_start_time={step * _PRINT_S:.17f}",
                "sim_duration=116040",
            ]
        )
        + "\n"
    )


#: How many of the newest checkpoints a synthetic tree carries. The rule reads only the checkpoints around the
#: reach, and the picker costs one file read per cfg, so a window keeps each test inside the new-test time budget
#: while the numbering and times stay those of the real arm. Measured once on all three full-size real trees: the
#: window and the full tree choose the same checkpoint and set aside the same cfgs.
_WINDOW = 24


def _make_sim(tmp_path: Path, n_cfgs: int) -> tuple[TRITONSWMM_run, Path]:
    """A coupled sim directory with the newest ``_WINDOW`` checkpoints of an arm that reached ``n_cfgs``."""
    sim = tmp_path / "sim"
    out = sim / "out_tritonswmm"
    (out / "cfg").mkdir(parents=True)
    (out / "swmm").mkdir()
    for k in range(max(1, n_cfgs - _WINDOW + 1), n_cfgs + 1):
        (out / "cfg" / f"config_{k}.cfg").write_text(_cfg_text(k))
        (out / "swmm" / f"hydraulics_state_cp{k}.snapshot").write_bytes(f"snapshot-{k}".encode())
    (out / "swmm" / "hydraulics_exchange_replay.bin").write_bytes(b"side-file-bytes")
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(), analysis_paths=SimpleNamespace()),
        scen_paths=SimpleNamespace(out_tritonswmm=out, out_triton=sim / "out_triton", sim_folder=sim),
    )
    return TRITONSWMM_run(cast("TRITONSWMM_scenario", scen)), out


def _write_log(tmp_path: Path, text: str) -> Path:
    logfile = tmp_path / "logs" / "model_tritonswmm_evt0.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.write_text(text)
    return logfile


def _bytes_multiset(root: Path) -> Counter[str]:
    """Multiset of every file's sha256 under ``root``, ignoring names: a rename keeps it and a delete loses it."""
    return Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file())


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _picked_step(run: TRITONSWMM_run) -> int | None:
    """The reporting step the resume picker would launch at next, or None when it would start fresh."""
    cfg = run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation("tritonswmm")
    return None if cfg is None else return_the_reporting_step_from_a_cfg(cfg)


def _oracle_newest_covered_step(reached_s: float) -> int:
    """The newest checkpoint a side-file reaching ``reached_s`` covers, from the reporting interval alone.

    A second implementation of the rule that shares no code with the one under test: checkpoint K is at
    ``K * _PRINT_S`` seconds, so the newest covered K is the whole number of intervals inside the reach.
    """
    return math.floor(reached_s / _PRINT_S)


def _cfg_dir(run: TRITONSWMM_run) -> Path:
    cfg_dir = run._hotstart_cfg_dir("tritonswmm")
    assert cfg_dir is not None
    return cfg_dir


def _out_dir(run: TRITONSWMM_run) -> Path:
    out = run._scenario.scen_paths.out_tritonswmm
    assert out is not None
    return out


#: ``fallback(run, logfile, resumed_step)``: the call a property makes. The real one and every mutant have this shape.
Fallback = Callable[[TRITONSWMM_run, Path, int], ExchangeReplayFallback | None]


def _real_fallback(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
    return run.fall_back_from_exchange_replay_refusal(logfile, resumed_step=resumed_step, stamp="0042")


def _check_steps_back_to_newest_covered(tmp_path: Path, arm: tuple[str, int, str, str], fallback: Fallback) -> None:
    """Assert the whole step-back property for one real arm; raises AssertionError when ``fallback`` violates it."""
    _, n_cfgs, reached, required = arm
    run, out = _make_sim(tmp_path, n_cfgs)
    before = _bytes_multiset(out.parent)
    fallback(run, _write_log(tmp_path, _refusal_text(reached, required)), n_cfgs)
    expected = _oracle_newest_covered_step(float(reached))
    assert _picked_step(run) == expected, f"next resume would start at {_picked_step(run)}, not {expected}"
    assert not before - _bytes_multiset(out.parent), "a file's bytes no longer exist anywhere under the sim folder"


def _set_aside(path: Path) -> None:
    """Rename ``path`` to the entry's aside form, the way the real implementation does and the mutants imitate."""
    path.rename(path.with_name(path.name + ".stale-resume-0042"))


def _stepback_mutants() -> dict[str, Fallback]:
    """Wrong implementations of the fallback, each of which the step-back property must reject."""

    def revert(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
        return None

    def only_the_refused_cfg(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
        _set_aside(_cfg_dir(run) / f"config_{resumed_step}.cfg")
        return None

    def one_checkpoint_too_far(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
        result = _real_fallback(run, logfile, resumed_step)
        assert result is not None and result.chosen_step is not None
        _set_aside(_cfg_dir(run) / f"config_{result.chosen_step}.cfg")
        return result

    def deletes_instead_of_renaming(
        run: TRITONSWMM_run, logfile: Path, resumed_step: int
    ) -> ExchangeReplayFallback | None:
        match = re.search(r"t=(\S+) s but", logfile.read_text())
        assert match is not None
        for step in range(_oracle_newest_covered_step(float(match.group(1))) + 1, resumed_step + 1):
            (_cfg_dir(run) / f"config_{step}.cfg").unlink()
        return None

    def always_starts_fresh(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
        out = _out_dir(run)
        _set_aside(out)
        out.mkdir()
        return None

    return {
        "revert": revert,
        "only_the_refused_cfg": only_the_refused_cfg,
        "one_checkpoint_too_far": one_checkpoint_too_far,
        "deletes_instead_of_renaming": deletes_instead_of_renaming,
        "always_starts_fresh": always_starts_fresh,
    }


# --------------------------------------------------------------------------------------
# (a) The real arms
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", _REAL_ARMS, ids=[a[0] for a in _REAL_ARMS])
def test_each_real_arm_resumes_from_the_newest_checkpoint_its_side_file_covers(tmp_path, arm):
    """Property: after the fallback, the picker's next resume is the newest checkpoint the reached time covers.

    Class: REGRESSION over the three real stale trees (R10 693 cfgs, R11 741, R2c 717): before the change the picker
    keeps returning the refused newest cfg. Reachability: these are the arms whose failure budgets were spent; the
    reached and required times are the ones their model logs hold. Why required: a wrong step either repeats the
    refusal or throws away finished work. Kills it: any mutant in ``_stepback_mutants``. Second correct
    implementation: move the uncovered cfgs into a ``cfg/stale`` subfolder; the picker's non-recursive glob never
    sees them and the assertion, which reads only the picker's choice and the sim folder's bytes, still holds.
    """
    _check_steps_back_to_newest_covered(tmp_path, arm, _real_fallback)


@pytest.mark.parametrize("mutant", sorted(_stepback_mutants()))
def test_the_step_back_property_rejects_every_wrong_implementation(tmp_path, mutant):
    """Property: each wrong implementation turns the step-back property red on the largest real arm (R11).

    Class: NEW CAPABILITY. This is the mutation proof for the property above: reverting the fallback, resuming one
    checkpoint too early, resuming from an uncovered one, deleting data and discarding the whole run each fail it.
    Kills it: a property that passes for every implementation. Second correct implementation: none needed; the
    subject here is the property, and the real implementation passing it is the previous test.
    """
    arm = _REAL_ARMS[1]
    with pytest.raises(AssertionError):
        _check_steps_back_to_newest_covered(tmp_path, arm, _stepback_mutants()[mutant])


@pytest.mark.parametrize("arm", _REAL_ARMS, ids=[a[0] for a in _REAL_ARMS])
def test_the_set_aside_names_follow_the_entrys_suffix_form(tmp_path, arm):
    """Property: every uncovered cfg survives under a name ending ``.stale-resume-`` and four digits, in ``cfg/``.

    Class: REGRESSION against the entry's own wording ("rename with a ``.stale-resume-{HHMM}`` suffix"). Why required:
    the operator greps for that suffix to find what was set aside. Kills it: a different suffix or a move out of
    ``cfg/``. Second correct implementation: the same suffix applied by copy-then-unlink is not admissible (the
    bytes must not move through a deletion), so none differs materially.
    """
    _, n_cfgs, reached, required = arm
    run, out = _make_sim(tmp_path, n_cfgs)
    run.fall_back_from_exchange_replay_refusal(
        _write_log(tmp_path, _refusal_text(reached, required)), resumed_step=n_cfgs
    )
    newest_covered = _oracle_newest_covered_step(float(reached))
    aside = sorted(p.name for p in (out / "cfg").glob("*.stale-resume-*"))
    pattern = re.compile(r"^config_(\d+)\.cfg\.stale-resume-\d{4}$")
    assert [m.group(1) for m in map(pattern.match, aside) if m] == sorted(
        str(k) for k in range(newest_covered + 1, n_cfgs + 1)
    )


_WORDINGS = {
    "reached-t (faba2232 and later)": _refusal_text,
    "replayed-up-to (3a832f7 and b3820a44)": _refusal_text_older_wording,
}


@pytest.mark.parametrize("wording", sorted(_WORDINGS))
def test_the_refusal_is_recognised_in_the_wording_of_every_solver_build_that_prints_it(tmp_path, wording):
    """Property: the same short side-file steps back identically whichever words the solver's build used.

    Class: NEW CAPABILITY. Plausible wrong implementation: a matcher keyed on one wording, which is the tree under
    review (``reached\\s+t=``). Reachability: ``git show b3820a44:src/swmm_triton.h`` and the same at ``3a832f7``
    print "replayed / up to {cum} s but checkpoint_id requires {Y} s" at lines 644-647, and ``3a832f7`` is the build
    ``prepare_simulation_command`` names as the toolkit's pinned TRITON, so an analysis built from it logs this
    wording. Why required: on those builds the failure is structural (the checkpoint cfg is written before the
    side-file flush), so the builds that most need the repair would get none and repeat the refusal until the budget
    is spent. The assertion turns on what the fallback does to the tree and on the picker's choice, both present
    before and after. Kills it: the single-wording regex. Second correct implementation: take the two figures from
    the line carrying ``but checkpoint_id requires``.
    """
    _, n_cfgs, reached, required = _REAL_ARMS[0]
    run, _ = _make_sim(tmp_path, n_cfgs)
    result = _real_fallback(run, _write_log(tmp_path, _WORDINGS[wording](reached, required)), n_cfgs)
    assert result is not None, "the refusal was not recognised, so the refused checkpoint stays the newest"
    assert _picked_step(run) == _oracle_newest_covered_step(float(reached))


# --------------------------------------------------------------------------------------
# (b) A resume that does not hit this refusal is untouched
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("label, text", _OTHER_SOLVER_REFUSALS, ids=[x[0] for x in _OTHER_SOLVER_REFUSALS])
def test_every_other_refusal_the_solver_prints_leaves_the_tree_untouched(tmp_path, label, text):
    """Property: no refusal other than the incomplete-side-file one changes a byte of the sim folder.

    Class: REGRESSION over the solver's refusal family (five other messages, quoted from ``swmm_triton.h``). Four
    end with the same "Re-run this coupled simulation from a clean start" sentence, so a matcher keyed on that
    sentence would rearrange the tree on them: the plausible wrong implementation. Why required: those refusals
    name no time, so there is nothing to decide a checkpoint on, and renaming data away is not a repair for them.
    Kills it: matching the clean-start sentence. Second correct implementation: match the second line's
    ``checkpoint_id requires`` wording alone.
    """
    run, out = _make_sim(tmp_path, 693)
    before = _tree_bytes(out.parent)
    assert _real_fallback(run, _write_log(tmp_path, text), 693) is None
    assert _tree_bytes(out.parent) == before


def test_the_other_refusals_are_a_must_fire_control_for_the_matcher(tmp_path):
    """Property: the same harness, given the incomplete-side-file refusal, DOES change the tree.

    Class: NEW CAPABILITY. The previous test's zero changes would read the same if ``_real_fallback`` could never fire;
    this is the input in the flagged set that proves it can. Kills it: a fallback inert on every input.
    """
    run, out = _make_sim(tmp_path, 693)
    before = _tree_bytes(out.parent)
    assert _real_fallback(run, _write_log(tmp_path, _refusal_text("82901", "83160")), 693) is not None
    assert _tree_bytes(out.parent) != before


@pytest.mark.parametrize(
    "text",
    [
        "SWMM exchange history replayed to t=82800 s (690 steps); resuming live segment\n",
        _refusal_text("82901", "83160").replace("incomplete for this checkpoint", "complete for this checkpoint"),
        "",
    ],
    ids=["successful-resume", "reworded-refusal", "empty-log"],
)
def test_a_resume_that_did_not_hit_the_refusal_is_untouched(tmp_path, text):
    """Property: a successful resume's log, a reworded refusal and an empty log all leave the tree byte-identical.

    Class: REGRESSION. The reworded case guards the day the solver changes its message: the fallback must go quiet
    rather than act on text it no longer recognises. Kills it: a looser matcher. Second correct implementation:
    token-based matching of the two numbers, which this input also leaves alone.
    """
    run, out = _make_sim(tmp_path, 693)
    before = _tree_bytes(out.parent)
    assert _real_fallback(run, _write_log(tmp_path, text), 693) is None
    assert _tree_bytes(out.parent) == before


# --------------------------------------------------------------------------------------
# (c) and (d) Bounded, never deletes, never loops; a wrong choice fails again and ends
# --------------------------------------------------------------------------------------


def _drive_until_the_solver_accepts(
    tmp_path: Path, n_cfgs: int, true_reach_s: float, fallback: Fallback, max_rounds: int
) -> tuple[int, int | None]:
    """Run resume attempts against a model solver until it accepts a checkpoint or starts fresh.

    The model solver refuses any checkpoint later than ``true_reach_s`` and prints the reach to six significant
    digits, as the real one does, so the toolkit sees a rounded figure while the solver judges on the true one.
    Returns ``(refusals, accepted_step)`` where ``accepted_step`` is None for a fresh start. Raises AssertionError
    when ``max_rounds`` attempts pass without an outcome, which is how a loop shows up.
    """
    run, _ = _make_sim(tmp_path, n_cfgs)
    refusals = 0
    for _ in range(max_rounds):
        step = _picked_step(run)
        if step is None:
            return refusals, None
        if step * _PRINT_S <= true_reach_s:
            return refusals, step
        refusals += 1
        required = _stream(step * _PRINT_S)
        fallback(run, _write_log(tmp_path, _refusal_text(_stream(true_reach_s), required)), step)
    raise AssertionError(f"no outcome after {max_rounds} attempts: the resume refuses forever")


@pytest.mark.parametrize(
    "n_cfgs, true_reach_s, max_refusals",
    [
        (693, 82901.0, 1),
        (693, 82919.97, 2),  # prints as 82920, exactly checkpoint 691's time, but the solver judges on 82919.97
        (717, 85920.04, 1),  # prints as 85920 and in fact covers checkpoint 716
        (693, 3.0, 1),  # below the first checkpoint: nothing is covered
    ],
)
def test_a_wrong_choice_by_the_rounded_figure_is_corrected_in_at_most_one_more_attempt(
    tmp_path, n_cfgs, true_reach_s, max_refusals
):
    """Property: when the printed reach is rounded across a checkpoint, the refusal repeats at most once more.

    Class: NEW CAPABILITY (a solver judging on the true figure while the log carries six digits). Reachability: the
    solver prints ``std::cerr << double``; ``82919.97`` and ``82920`` print the same. Why required: each refusal
    spends one unit of the failure budget, and the entry asks for a bounded retry that ends. The accepted step is
    asserted against the oracle on the TRUE reach. Kills it: a fallback that makes no progress (see the mutant
    test) or one that never reaches the true covered step. Second correct implementation: compare with a wider
    tolerance and accept one extra step back; the bound on refusals would still hold.
    """
    refusals, accepted = _drive_until_the_solver_accepts(tmp_path, n_cfgs, true_reach_s, _real_fallback, 12)
    assert refusals <= max_refusals
    if true_reach_s < _PRINT_S:
        assert accepted is None
    else:
        assert accepted == _oracle_newest_covered_step(true_reach_s)


def test_a_fallback_that_makes_no_progress_is_caught_by_the_driver(tmp_path):
    """Property: the closed-loop driver reports a resume that refuses forever as a failure.

    Class: NEW CAPABILITY (mutation proof for the test above). A fallback that logs and renames nothing leaves the
    picker on the refused cfg, which is the original defect. Kills it: a driver that returns after a fixed number of
    rounds without checking an outcome.
    """

    def no_progress(run: TRITONSWMM_run, logfile: Path, resumed_step: int) -> ExchangeReplayFallback | None:
        return None

    with pytest.raises(AssertionError, match="refuses forever"):
        _drive_until_the_solver_accepts(tmp_path, 693, 82901.0, no_progress, 12)


def test_the_fresh_start_branch_keeps_every_byte_and_is_not_repeated(tmp_path):
    """Property: with nothing covered the whole output moves aside intact, and the next attempt cannot repeat it.

    Class: REGRESSION. Reachability: a side-file that reaches only 100 s on a run whose first checkpoint is at 120 s
    (a kill before the first checkpoint's records flushed). Why required: "never refuse forever, never delete data".
    After the move the picker finds nothing and the launch is fresh (``resumed_step`` 0), so the same stale log can
    no longer trigger anything. Kills it: deleting the directory, or leaving a cfg behind. Second correct
    implementation: move only ``cfg/`` aside and keep the side-file; the assertions read bytes anywhere under the sim
    folder and the picker, so it would pass as long as no data is lost.
    """
    run, out = _make_sim(tmp_path, 693)
    before = _bytes_multiset(out.parent)
    log = _write_log(tmp_path, _refusal_text("100", "83160"))
    assert _real_fallback(run, log, 693) is not None
    assert _picked_step(run) is None
    assert out.is_dir() and not any(out.iterdir())
    assert not before - _bytes_multiset(out.parent)
    again = _tree_bytes(out.parent)
    assert _real_fallback(run, log, 0) is None
    assert _tree_bytes(out.parent) == again


# --------------------------------------------------------------------------------------
# (e) The cluster watchdog that renames out_tritonswmm aside and cancels a silent job
# --------------------------------------------------------------------------------------


def test_a_watchdog_rename_leaves_nothing_for_the_fallback_to_do(tmp_path):
    """Property: with ``out_tritonswmm`` already renamed aside by the watchdog, the fallback does nothing.

    Class: REGRESSION. The watchdog (``cd_watchdog.sh``, line 24) runs ``mv "$O" "$O.stale-watchdog-HHMM"`` and then
    ``scancel``. The retry then has no checkpoint and no output directory: the fallback must return None, create no
    ``out_tritonswmm`` of its own and leave the watchdog's directory alone, because the solver makes the output
    directory itself (``mkdir`` in ``output.h``). Kills it: a fallback that raises on the missing directory (the
    runner's ``except Exception`` would hide it, but the log line would be a false failure) or one that recreates it.
    Second correct implementation: return a result recording that nothing was found.
    """
    run, out = _make_sim(tmp_path, 693)
    watchdog_dir = out.with_name(out.name + ".stale-watchdog-0115")
    out.rename(watchdog_dir)
    before = _tree_bytes(out.parent)
    assert _real_fallback(run, _write_log(tmp_path, _refusal_text("82901", "83160")), 693) is None
    assert _tree_bytes(out.parent) == before
    assert not out.exists()


def test_the_toolkits_aside_name_never_collides_with_the_watchdogs(tmp_path):
    """Property: a fresh-start aside lands beside an existing watchdog aside without touching it.

    Class: REGRESSION. ``mv`` onto an existing directory moves INTO it, so a shared suffix would bury one set of
    outputs inside the other. The two use different words (``stale-watchdog`` and ``stale-resume``), and the
    toolkit's helper numbers a taken name. Kills it: reusing the watchdog's suffix. Second correct implementation:
    any name unique among the siblings.
    """
    run, out = _make_sim(tmp_path, 693)
    watchdog_dir = out.with_name(out.name + ".stale-watchdog-0042")
    watchdog_dir.mkdir()
    (watchdog_dir / "earlier.bin").write_bytes(b"earlier run")
    assert _real_fallback(run, _write_log(tmp_path, _refusal_text("100", "83160")), 693) is not None
    assert (watchdog_dir / "earlier.bin").read_bytes() == b"earlier run"
    assert sorted(p.name for p in out.parent.glob("out_tritonswmm.stale-*")) == [
        "out_tritonswmm.stale-resume-0042",
        "out_tritonswmm.stale-watchdog-0042",
    ]


# --------------------------------------------------------------------------------------
# The runner wiring: the fallback is called before the failure is counted
# --------------------------------------------------------------------------------------


def _runner_main_tree() -> ast.FunctionDef:
    tree = ast.parse((Path(hhemt.__file__).parent / "run_simulation_runner.py").read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")


def _call_lines(node: ast.AST, name: str) -> list[int]:
    return [
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and (getattr(sub.func, "attr", None) == name or getattr(sub.func, "id", None) == name)
    ]


def _not_completed_branch(main: ast.FunctionDef) -> ast.If:
    branches = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and _call_lines(n.test, "model_run_completed")
        and _call_lines(n, "_bump_terminal_failures")
    ]
    assert len(branches) == 1
    return branches[0]


def _statement_positions(branch: ast.If, name: str) -> list[int]:
    """Indices of the branch's top-level statements that contain a call to ``name``, in body order."""
    return [i for i, stmt in enumerate(branch.body) if _call_lines(stmt, name)]


def _assert_fallback_runs_before_the_failure_is_counted(main: ast.FunctionDef) -> None:
    branch = _not_completed_branch(main)
    fallback = _statement_positions(branch, "fall_back_from_exchange_replay_refusal")
    counted = _statement_positions(branch, "_bump_terminal_failures")
    assert len(fallback) == 1, "the not-completed branch must call the fallback exactly once"
    assert fallback[0] < min(counted), "the fallback must run before the failure is counted"


def test_the_runner_calls_the_fallback_ahead_of_the_failure_budget():
    """Property: in the not-completed branch the fallback call precedes the failure count and the branch still fails.

    Class: REGRESSION on the wiring. Why required: after the budget write a raise could skip the repair, and a
    branch that returns 0 would mark an unfinished simulation done. Kills it: removing the call, or moving it after
    ``_bump_terminal_failures``. Second correct implementation: the call inside a helper that the branch calls (this
    test would then need the helper named).
    """
    main = _runner_main_tree()
    _assert_fallback_runs_before_the_failure_is_counted(main)
    returns = [n for n in ast.walk(_not_completed_branch(main)) if isinstance(n, ast.Return)]
    assert returns and all(isinstance(r.value, ast.Constant) and r.value.value == 1 for r in returns)


@pytest.mark.parametrize("mutation", ["call_removed", "call_moved_after_the_count"])
def test_the_wiring_property_rejects_a_reverted_or_misplaced_fallback(mutation):
    """Property: the wiring check goes red when the call is removed or placed after the failure is counted.

    Class: NEW CAPABILITY (mutation proof). The tree is edited in memory, so no source file changes.
    """
    main = _runner_main_tree()
    branch = _not_completed_branch(main)
    holder = next(
        n
        for n in ast.walk(branch)
        if isinstance(n, ast.Try) and _call_lines(n, "fall_back_from_exchange_replay_refusal")
    )
    branch.body.remove(holder)
    if mutation == "call_moved_after_the_count":
        branch.body.insert(len(branch.body) - 1, holder)
    with pytest.raises(AssertionError):
        _assert_fallback_runs_before_the_failure_is_counted(main)
