"""A coupled resume the solver refuses over a short exchange-replay side-file falls back to a covered checkpoint.

THE PROBLEM. A coupled TRITON-SWMM simulation killed at its wall limit or by a node loss can leave
a hotstart checkpoint ``config_N.cfg`` whose exchange-replay side-file (``*_exchange_replay.bin``)
stops short of it. Every later resume then launches the solver at that checkpoint and the solver
aborts with ``Exchange-replay side-file is incomplete for this checkpoint: reached t=86408.5 s but
checkpoint_id requires 86640 s``. The picker keeps choosing the same newest cfg, so every retry
repeats the abort until the failure budget is spent.

THE CHANGE UNDER TEST. ``TRITONSWMM_run.fall_back_from_exchange_replay_refusal`` reads the model log
a resume just wrote. On exactly that refusal it renames aside, with a ``.stale-resume-{HHMM}``
suffix, every checkpoint cfg the side-file does not cover, so the next attempt's picker resumes
from the newest checkpoint the side-file does cover. When none is covered it renames the whole
``out_tritonswmm`` directory aside and leaves an empty one, so the next attempt starts fresh.
Nothing is deleted. A resume that did not hit this refusal is not touched.

No test here compiles or runs a solver. The trees are synthetic: ``config_N.cfg`` files written in
the solver's own ``sim_start_time=`` form, plus a side-file stand-in.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import hhemt
from hhemt.run_simulation import TRITONSWMM_run, parse_exchange_replay_refusal

if TYPE_CHECKING:
    from hhemt.scenario import TRITONSWMM_scenario

#: The reporting interval of the synthetic run. Checkpoint K is written at K * _PRINT_S seconds, the
#: way the solver's ``output_cfg`` stamps ``sim_start_time`` for a fixed reporting interval.
_PRINT_S = 240

#: ``ERROR`` in the solver is ``RED << "[ERROR] " << RESET``, so the real line carries ANSI colour
#: codes around the tag. The fixtures keep them: a matcher that only works on the colourless text
#: would pass here and miss the real log.
_ERROR_TAG = "\x1b[31m[ERROR] \x1b[0m"


def _refusal_log(reached: str, required: str) -> str:
    """The model log of a resume the solver refused, with the solver's own line wrapping."""
    return (
        "[..] Reading checkpoint files\n"
        "[OK] Checkpoint files read\n"
        f"{_ERROR_TAG}Exchange-replay side-file is incomplete for this checkpoint: reached\n"
        f"         t={reached} s but checkpoint_id requires {required} s "
        "(side-file/checkpoint mismatch, e.g. truncated by a hard kill).\n"
        "         Re-run this coupled simulation from a clean start (checkpoint_id=0).\n"
    )


def _cfg_text(step: int) -> str:
    """One checkpoint cfg. Same line count for every step, so the picker's modal rule keeps them all."""
    return (
        "\n".join(
            [
                'output_folder="out_tritonswmm"',
                f"print_interval={_PRINT_S}",
                f"checkpoint_id={step}",
                f"sim_start_time={step * _PRINT_S:.17f}",
                "sim_duration=172800",
            ]
        )
        + "\n"
    )


def _sim(tmp_path: Path, steps) -> tuple[TRITONSWMM_run, Path]:
    """A coupled sim directory holding ``config_{k}.cfg`` for each step, a side-file and some outputs."""
    sim = tmp_path / "sim"
    out = sim / "out_tritonswmm"
    (out / "cfg").mkdir(parents=True)
    (out / "swmm").mkdir()
    for k in steps:
        (out / "cfg" / f"config_{k}.cfg").write_text(_cfg_text(k))
    (out / "swmm" / "hydraulics_exchange_replay.bin").write_bytes(b"side-file-bytes")
    (out / "swmm" / "hydraulics_state_cp3.snapshot").write_bytes(b"snapshot-bytes")
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(), analysis_paths=SimpleNamespace())
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=analysis,
        scen_paths=SimpleNamespace(out_tritonswmm=out, out_triton=sim / "out_triton", sim_folder=sim),
    )
    return TRITONSWMM_run(cast("TRITONSWMM_scenario", scen)), out


def _log(tmp_path: Path, text: str) -> Path:
    logfile = tmp_path / "logs" / "model_tritonswmm_evt0.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.write_text(text)
    return logfile


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` keyed by its path relative to ``root``, with its bytes."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _newest_cfg_the_picker_would_choose(run: TRITONSWMM_run) -> str | None:
    cfg = run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation("tritonswmm")
    return None if cfg is None else cfg.name


# --------------------------------------------------------------------------------------
# The refusal parser
# --------------------------------------------------------------------------------------


def test_the_parser_reads_both_times_from_the_solvers_own_two_line_message():
    """Property: both figures come out of the refusal as the solver wraps it, ANSI colour included.

    Class: NEW CAPABILITY. Plausible wrong implementation: a single-line regex, which never matches
    because the solver breaks the line between ``reached`` and ``t=``. Why required: the reached time is
    the only record of how much of the series the side-file holds. Kills it: the real two-line text.
    Second correct implementation: split the text into tokens and find the two ``t=`` and ``requires``
    numbers by position.
    """
    parsed = parse_exchange_replay_refusal(_refusal_log("86408.5", "86640"))
    assert parsed == (86408.5, 86640.0)


@pytest.mark.parametrize(
    "reached, required, expected",
    [("1.0368e+06", "1.0584e+06", (1036800.0, 1058400.0)), ("0", "240", (0.0, 240.0))],
)
def test_the_parser_reads_the_default_stream_forms_the_solver_prints(reached, required, expected):
    """Property: the solver prints with the default stream precision, so large and zero times take other forms.

    Class: NEW CAPABILITY. Wrong implementation: ``\\d+\\.?\\d*``, which stops at ``1.0368`` and reads a
    million-second run's time as one second. Kills it: the exponent form.
    """
    assert parse_exchange_replay_refusal(_refusal_log(reached, required)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Simulation ends\n",
        # The solver's other side-file refusals say "re-run from a clean start" too, but they are not
        # this refusal and carry no times; falling back on them would be a guess.
        f"{_ERROR_TAG}Coupled TRITON-SWMM resume requires the exchange-replay side-file,\n"
        "         but it was not found: /x/hydraulics_exchange_replay.bin\n",
        f"{_ERROR_TAG}Exchange-replay side-file has a bad/missing header: /x/h.bin\n",
        # Marker present, figures absent: refuse to parse rather than invent a coverage time.
        f"{_ERROR_TAG}Exchange-replay side-file is incomplete for this checkpoint: reached\n",
    ],
)
def test_the_parser_ignores_text_that_is_not_this_refusal(text):
    """Property: only the incomplete-side-file refusal, with both figures, parses.

    Class: NEW CAPABILITY. Wrong implementation: matching the shared tail ``Re-run this coupled
    simulation from a clean start``, which the missing-file and bad-header refusals also print.
    """
    assert parse_exchange_replay_refusal(text) is None


# --------------------------------------------------------------------------------------
# The fallback
# --------------------------------------------------------------------------------------


def test_the_newest_uncovered_checkpoint_is_set_aside_and_the_covered_one_is_chosen(tmp_path):
    """Property: with the side-file at 86408.5 s, checkpoint 361 (86640 s) goes aside and 360 (86400 s) is chosen.

    Class: REGRESSION. Red on the pre-change tree: the method does not exist, and the picker keeps
    returning ``config_361.cfg`` for ever. The measured case is R2b/R2c: the side-file at 86408.5 s and the
    checkpoint at 86640 s. Second correct implementation: delete nothing but move the cfgs into an
    ``aside/`` subdirectory the picker's ``*.cfg`` glob does not see.
    """
    run, out = _sim(tmp_path, steps=range(355, 362))
    assert _newest_cfg_the_picker_would_choose(run) == "config_361.cfg"  # the defect, before

    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86640")), resumed_step=361, stamp="0017"
    )

    assert result is not None and result.action == "stepped_back"
    assert result.chosen_step == 360
    assert [p.name for p in result.renamed] == ["config_361.cfg.stale-resume-0017"]
    assert _newest_cfg_the_picker_would_choose(run) == "config_360.cfg"
    assert not (out / "cfg" / "config_361.cfg").exists()
    assert (out / "cfg" / "config_361.cfg.stale-resume-0017").read_text() == _cfg_text(361)


def test_every_uncovered_checkpoint_goes_aside_not_only_the_one_that_was_refused(tmp_path):
    """Property: checkpoints 361 and 362 are both past a side-file that ends at 86408.5 s, so both go.

    Class: NEW CAPABILITY. Plausible wrong implementation: step back one checkpoint (or set aside only
    the cfg the refused launch used). Red under it: after 362 alone goes, the picker chooses 361 and the
    solver refuses again with the same figures. Second correct implementation: compute the cut from the
    refusal's ``requires`` time instead of ``reached`` and rename every cfg at or above that time,
    which is the same set whenever the resumed cfg is the newest.
    """
    run, _ = _sim(tmp_path, steps=range(358, 363))
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86880")), resumed_step=362, stamp="0017"
    )
    assert result is not None
    assert sorted(p.name for p in result.renamed) == [
        "config_361.cfg.stale-resume-0017",
        "config_362.cfg.stale-resume-0017",
    ]
    assert _newest_cfg_the_picker_would_choose(run) == "config_360.cfg"


def test_a_checkpoint_between_the_side_file_end_and_the_refused_one_is_set_aside_too(tmp_path):
    """Property: the side-file ends at 86300 s; checkpoints 360 (86400 s) and 361 (86640 s) are both uncovered.

    Class: NEW CAPABILITY. Plausible wrong implementation: set aside only the resumed cfg (361) and trust
    the next-newest. Red under it: 360 is still past the side-file, so the retry refuses at 86400 s.
    Second correct implementation: read the side-file's records and sum its step lengths directly.
    """
    run, _ = _sim(tmp_path, steps=range(357, 362))
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86300", "86640")), resumed_step=361, stamp="0017"
    )
    assert result is not None
    assert result.chosen_step == 359  # 359 * 240 = 86160 s, covered
    assert _newest_cfg_the_picker_would_choose(run) == "config_359.cfg"


def test_a_checkpoint_exactly_at_the_side_file_end_is_covered(tmp_path):
    """Property: the boundary is inclusive. A checkpoint at 86400 s is covered by a side-file reaching 86400 s.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``t < reached``, which sets aside the checkpoint
    the solver would have accepted (it refuses only when ``reached < required - 1e-6``). Red under it:
    360 would go aside and the run would lose a covered checkpoint.
    """
    run, _ = _sim(tmp_path, steps=range(358, 362))
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86400", "86640")), resumed_step=361, stamp="0017"
    )
    assert result is not None
    assert result.chosen_step == 360


def test_a_figure_the_stream_rounded_does_not_set_aside_a_covered_checkpoint(tmp_path):
    """Property: ``reached`` is printed to six significant digits, so a checkpoint inside the rounding is covered.

    The side-file really reaches 86400.04 s and the solver prints ``86400``. Checkpoint 360 is at
    86400 s, so it is covered by either figure; a checkpoint at 86400.04 s would be too, and the
    tolerance is half a unit of the sixth digit.
    Class: NEW CAPABILITY. Plausible wrong implementation: an exact float comparison against the rounded
    figure that treats a checkpoint at 86400.04 s as uncovered.
    """
    run, out = _sim(tmp_path, steps=range(358, 362))
    # A checkpoint whose time is a few hundredths of a second past the printed figure.
    (out / "cfg" / "config_360.cfg").write_text(
        _cfg_text(360).replace("86400.00000000000000000", "86400.04000000000000000")
    )
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86400", "86640")), resumed_step=361, stamp="0017"
    )
    assert result is not None
    assert result.chosen_step == 360


def test_when_no_checkpoint_is_covered_the_output_directory_goes_aside_and_a_fresh_one_is_left(tmp_path):
    """Property: with the side-file at 100 s, no checkpoint (first at 240 s) is covered, so the run restarts clean.

    Class: REGRESSION. Red on the pre-change tree: nothing sets the directory aside, so the picker keeps
    choosing a checkpoint the solver refuses. Second correct implementation: keep ``out_tritonswmm`` and
    move every child of it into a ``stale-resume`` subdirectory of itself.
    """
    run, out = _sim(tmp_path, steps=range(1, 4))
    before = _snapshot(out)

    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("100", "720")), resumed_step=3, stamp="0017"
    )

    assert result is not None and result.action == "fresh_start"
    aside = out.parent / "out_tritonswmm.stale-resume-0017"
    assert result.aside_dir == aside
    assert _snapshot(aside) == before  # nothing deleted, nothing altered
    assert out.is_dir() and list(out.iterdir()) == []  # an empty directory the solver can write into
    assert _newest_cfg_the_picker_would_choose(run) is None  # the next launch starts at step 0


def test_nothing_is_deleted_by_a_step_back(tmp_path):
    """Property: every byte that existed before the fallback exists after it, under its original or aside name.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``unlink`` the uncovered cfgs. Red under it:
    the aside name does not exist.
    """
    run, out = _sim(tmp_path, steps=range(358, 363))
    before = _snapshot(out)
    run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86880")), resumed_step=362, stamp="0017"
    )
    after = _snapshot(out)
    assert sorted(after.values()) == sorted(before.values())
    assert (out / "swmm" / "hydraulics_exchange_replay.bin").read_bytes() == b"side-file-bytes"


def test_a_second_fallback_in_the_same_minute_does_not_overwrite_the_first(tmp_path):
    """Property: the aside name is unique even when the stamp repeats, so the first set-aside survives.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``Path.rename`` onto a name that already
    exists, which on POSIX silently replaces it. Red under it: the first run's cfg bytes are gone.
    """
    run, out = _sim(tmp_path, steps=range(358, 363))
    run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86880")), resumed_step=362, stamp="0017"
    )
    # The next attempt re-created checkpoints 361 and 362 and was refused again within the minute.
    for k in (361, 362):
        (out / "cfg" / f"config_{k}.cfg").write_text(_cfg_text(k) + "# second run\n")
    run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86880")), resumed_step=362, stamp="0017"
    )
    first = (out / "cfg" / "config_362.cfg.stale-resume-0017").read_text()
    assert first == _cfg_text(362)
    assert len(list((out / "cfg").glob("config_362.cfg.stale-resume-0017*"))) == 2


def test_the_default_stamp_is_the_wall_clock_hour_and_minute(tmp_path):
    """Property: without an explicit stamp the suffix is ``.stale-resume-`` and four digits.

    Class: NEW CAPABILITY. Wrong implementation: a bare ``.stale`` suffix, or a stamp carrying colons
    that a path on a Windows-mounted scratch rejects.
    """
    run, out = _sim(tmp_path, steps=range(358, 362))
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86640")), resumed_step=361
    )
    assert result is not None
    assert re.fullmatch(r"config_361\.cfg\.stale-resume-\d{4}", result.renamed[0].name)


def test_the_returned_message_says_which_checkpoint_was_chosen_and_why(tmp_path):
    """Property: the record a reader of the log finds names the chosen step and the two times that decided it.

    Class: NEW CAPABILITY. Wrong implementation: a silent rename. Why required: a later reader of a resumed
    sim's log must be able to tell a deliberate step-back from a lost checkpoint.
    """
    run, _ = _sim(tmp_path, steps=range(358, 362))
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86640")), resumed_step=361, stamp="0017"
    )
    assert result is not None
    for needle in ("checkpoint 360", "86408.5", "86640", "config_361.cfg"):
        assert needle in result.message, needle


# --------------------------------------------------------------------------------------
# What must not move
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "log_text",
    [
        "Simulation ends\n",
        "[OK] Checkpoint files read\nSWMM exchange history replayed to t=86400 s (360 steps)\n",
        f"{_ERROR_TAG}Coupled TRITON-SWMM resume requires the exchange-replay side-file,\n"
        "         but it was not found: /x/hydraulics_exchange_replay.bin\n",
        f"{_ERROR_TAG}Exchange-replay side-file has a bad/missing header: /x/h.bin\n",
        "",
    ],
)
def test_a_resume_that_did_not_hit_the_refusal_is_untouched(tmp_path, log_text):
    """Property: any other outcome of a resume leaves the tree byte-identical and returns None.

    Class: REGRESSION (behaviour preservation). Plausible wrong implementation: step back on every failed
    resume, or on any "clean start" advice. Red under it: the tree listing changes.
    """
    run, out = _sim(tmp_path, steps=range(358, 362))
    before = _snapshot(out.parent)
    result = run.fall_back_from_exchange_replay_refusal(_log(tmp_path, log_text), resumed_step=361, stamp="0017")
    assert result is None
    assert _snapshot(out.parent) == before


def test_a_fresh_launch_that_logs_the_refusal_text_is_untouched(tmp_path):
    """Property: with ``resumed_step == 0`` (the launch used the base cfg) nothing is set aside.

    Class: NEW CAPABILITY. A clean start cannot be refused for a checkpoint it did not read, so a
    refusal-shaped line in its log is not evidence about the tree. Plausible wrong implementation: act on
    the text alone. Red under it: the output directory moves.
    """
    run, out = _sim(tmp_path, steps=range(358, 362))
    before = _snapshot(out.parent)
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86640")), resumed_step=0, stamp="0017"
    )
    assert result is None
    assert _snapshot(out.parent) == before


def test_only_the_coupled_model_has_a_side_file_to_fall_back_on(tmp_path):
    """Property: a TRITON-only run (no exchange side-file) is never rearranged, whatever its log says."""
    run, out = _sim(tmp_path, steps=range(358, 362))
    before = _snapshot(out.parent)
    result = run.fall_back_from_exchange_replay_refusal(
        _log(tmp_path, _refusal_log("86408.5", "86640")), resumed_step=361, model_type="triton", stamp="0017"
    )
    assert result is None
    assert _snapshot(out.parent) == before


def test_an_unreadable_log_leaves_the_tree_alone(tmp_path):
    """Property: a log that cannot be read is not evidence of the refusal."""
    run, out = _sim(tmp_path, steps=range(358, 362))
    before = _snapshot(out.parent)
    assert run.fall_back_from_exchange_replay_refusal(tmp_path / "no-such.log", resumed_step=361, stamp="0017") is None
    assert _snapshot(out.parent) == before


def test_a_log_with_undecodable_bytes_is_still_read(tmp_path):
    """Property: a stray non-UTF-8 byte in the log does not hide the refusal.

    Class: NEW CAPABILITY. The runner's ledger step already failed once on a log byte the ambient
    encoding could not decode. Wrong implementation: ``read_text()`` with the default error mode, which
    raises ``UnicodeDecodeError`` and so never repairs the tree.
    """
    run, _ = _sim(tmp_path, steps=range(358, 362))
    logfile = tmp_path / "logs" / "model_tritonswmm_evt0.log"
    logfile.parent.mkdir(parents=True)
    logfile.write_bytes(b"\xff\xfe junk\n" + _refusal_log("86408.5", "86640").encode())
    result = run.fall_back_from_exchange_replay_refusal(logfile, resumed_step=361, stamp="0017")
    assert result is not None and result.chosen_step == 360


# --------------------------------------------------------------------------------------
# The runner wires it in, and does not change what counts as a failure
# --------------------------------------------------------------------------------------


def _runner_main() -> ast.FunctionDef:
    src = (Path(hhemt.__file__).parent / "run_simulation_runner.py").read_text()
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")


def _calls_in(node: ast.AST) -> list[tuple[int, str]]:
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            found.append((sub.lineno, name))
    return found


def test_the_runner_repairs_before_it_records_the_failure_and_still_fails_the_attempt():
    """Property: the fallback runs before the failure is counted and the branch still returns 1.

    So the failure budget counts the refused attempt exactly as before.

    Class: NEW CAPABILITY. Plausible wrong implementation: call the fallback after ``_bump_terminal_failures``
    (a raise inside it would then skip the repair) or return 0 after repairing (which would mark a
    simulation that did not finish as done). A source-shape test, because no solver runs on the laptop;
    the behaviour of the method is pinned above. Second correct implementation: the call sited inside a
    helper that the branch calls, which this test would need adjusting for.
    """
    main = _runner_main()
    branches = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and any(name == "model_run_completed" for _, name in _calls_in(n.test))
        and any(name == "_bump_terminal_failures" for _, name in _calls_in(n))
    ]
    assert len(branches) == 1, "expected exactly one not-completed branch that counts a terminal failure"
    branch = branches[0]
    calls = _calls_in(branch)
    fallback_lines = [ln for ln, name in calls if name == "fall_back_from_exchange_replay_refusal"]
    bump_lines = [ln for ln, name in calls if name == "_bump_terminal_failures"]
    assert len(fallback_lines) == 1, "the fallback is called exactly once in the not-completed branch"
    assert fallback_lines[0] < min(bump_lines)
    returns = [n for n in ast.walk(branch) if isinstance(n, ast.Return)]
    assert returns and all(isinstance(r.value, ast.Constant) and r.value.value == 1 for r in returns)


def test_a_failure_of_the_repair_cannot_change_the_attempts_outcome():
    """Property: an exception inside the fallback is caught and logged, so the attempt still fails the way it did.

    Class: NEW CAPABILITY. Wrong implementation: let it propagate to ``main``'s outer handler, which counts
    the failure twice (once there and once in the branch it never reached). Source-shape for the same
    reason as above.
    """
    main = _runner_main()
    fallback_calls = [
        sub
        for sub in ast.walk(main)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "fall_back_from_exchange_replay_refusal"
    ]
    assert len(fallback_calls) == 1
    guarded = [
        t
        for t in ast.walk(main)
        if isinstance(t, ast.Try)
        and any(sub is fallback_calls[0] for stmt in t.body for sub in ast.walk(stmt))
        and any(h.type is not None and getattr(h.type, "id", "") == "Exception" for h in t.handlers)
    ]
    assert guarded, "the fallback call must sit in a try whose handler catches Exception"
