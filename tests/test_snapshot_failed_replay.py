"""The toolkit writes `swmm_snapshot_failed_replay=1` into the cfg a coupled launch hands the solver.

THE PROBLEM. The solver stops a coupled restart at a checkpoint whose SWMM snapshot write
FAILED, and offers an opt-in key that replays the SWMM exchange history there and carries on.
The toolkit's field ``swmm_snapshot_failed_replay`` has to reach that key on a RESUME as well
as on a clean start: TRITON writes every ``config_K.cfg`` as a copy of the cfg it was launched
with, so a line added only when the base cfg is prepared never reaches a resume of a run that
was prepared without it. The write is therefore made at launch, set or removed, on the cfg
the launch passes to the solver.

This module is deliberately FIXTURE-FREE: it imports no compiled-model fixture and constructs
no ``TRITONSWMM_run``, so it starts no solver and runs in any venue. Seven properties:

1. A launch whose flag already matches the file leaves its bytes and modification time alone.
2. ``enabled=True`` leaves exactly one ``=1`` line, and a second call writes nothing.
3. ``enabled=False`` removes the line, and a stale ``=2`` or a duplicated line collapses to one.
4. The only caller is the launch method, and the cfg generator never calls it.
5. ``write-failed`` is a classified replay reason, so a replay row carrying it is not left
   unclassified.
6. A replay row with that reason raises one alarm, and a replay row with another reason none.
7. The picker's message states what the solver will do with the cfg it is given.
"""

import ast
import os
from pathlib import Path

import pytest

from hhemt import resume_events as re_mod
from hhemt import run_simulation as run_simulation_mod
from hhemt import scenario as scenario_mod
from hhemt.run_simulation import failed_write_resume_message
from hhemt.scenario import SWMM_SNAPSHOT_FAILED_REPLAY_KEY, set_swmm_snapshot_failed_replay

_BASE = 'output_folder="out_tritonswmm"\ndem_filename="d"\n'
_LINE = f"{SWMM_SNAPSHOT_FAILED_REPLAY_KEY}=1"


def _count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(f"{SWMM_SNAPSHOT_FAILED_REPLAY_KEY}="))


def test_a_matching_flag_leaves_the_file_untouched(tmp_path):
    """Property: with the flag off and no key in the cfg, nothing is written -- bytes and mtime unchanged.

    Class: a cfg that does not carry the key, with ``enabled=False`` -- the state every
    toolkit-prepared cfg is in when the field is left at its default.

    Why required, checkable against the code: the default must not alter any run that never
    sets the field, and compare-and-write is what keeps the modification time stable for
    tools that watch the cfg. The returned boolean is read as well as the bytes.

    What kills it: a version that always rewrites the file, or one that returns True without
    comparing.

    A second correct implementation under which it still passes: one that parses the cfg into
    keys and values and rewrites only when the parsed mapping differs.
    """
    cfg = tmp_path / "TRITONSWMM.cfg"
    cfg.write_text(_BASE)
    os.utime(cfg, ns=(1_000_000_000, 1_000_000_000))
    before = cfg.stat()
    assert set_swmm_snapshot_failed_replay(cfg, False) is False
    after = cfg.stat()
    assert cfg.read_text() == _BASE
    assert after.st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("starting_text", [_BASE, _BASE.rstrip("\n")])
def test_enabling_leaves_exactly_one_line_and_is_idempotent(tmp_path, starting_text):
    """Property: ``enabled=True`` leaves exactly one ``=1`` line, and calling again writes nothing.

    Class: a cfg without the key, ending with a newline and not ending with one -- two
    differently-positioned inputs for the line-joining step.

    Why required, checkable against the code: every coupled launch calls the function, so a
    resume that is launched twice must not accumulate a second line, and a cfg whose last
    line has no trailing newline must not have the key glued onto it.

    What kills it: appending without removing first, or appending without a separator.

    A second correct implementation under which it still passes: one that builds the new text
    from a list of lines and joins them once.
    """
    cfg = tmp_path / "TRITONSWMM.cfg"
    cfg.write_text(starting_text)
    assert set_swmm_snapshot_failed_replay(cfg, True) is True
    assert _count(cfg.read_text()) == 1
    assert cfg.read_text().splitlines()[-1] == _LINE
    assert cfg.read_text().startswith(starting_text.rstrip("\n"))
    assert set_swmm_snapshot_failed_replay(cfg, True) is False
    assert _count(cfg.read_text()) == 1


@pytest.mark.parametrize(
    ("starting_text", "enabled", "expected_count"),
    [
        (_BASE + _LINE + "\n", False, 0),
        (_BASE + f"{SWMM_SNAPSHOT_FAILED_REPLAY_KEY}=2\n", True, 1),
        (_BASE + _LINE + "\n" + _LINE + "\n", True, 1),
        (_BASE + _LINE + "\n" + _LINE + "\n", False, 0),
    ],
)
def test_the_line_always_matches_the_flag(tmp_path, starting_text, enabled, expected_count):
    """Property: the cfg ends with exactly one key line when enabled and none when not, from any starting line count.

    Class: a stale key left by an earlier opted-in launch, a different value (``=2``), and a
    duplicated key -- each driven with the flag on and with it off where it applies.

    Why required, checkable against the code: the cfg is reused across launches, so a plain
    run after an opted-in run must not inherit the replay, and a hand-added value must not
    survive as something other than ``1``. Every other line of the cfg is read back as well.

    What kills it: a remove that only strips the first occurrence, or an enable that leaves a
    non-``1`` value in place.

    A second correct implementation under which it still passes: one that filters on a
    regular expression over the key rather than on a string prefix.
    """
    cfg = tmp_path / "TRITONSWMM.cfg"
    cfg.write_text(starting_text)
    set_swmm_snapshot_failed_replay(cfg, enabled)
    text = cfg.read_text()
    assert _count(text) == expected_count
    assert text.startswith(_BASE)
    if enabled:
        assert text.splitlines()[-1] == _LINE


def _call_sites(source: Path, func_name: str) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
                callers.add(fn.name)
    return callers


def test_only_the_launch_method_writes_the_key():
    """Property: ``set_swmm_snapshot_failed_replay`` is called from exactly one function, the launch method.

    Class: every function in the run module and the scenario module -- the two files that
    can reach a cfg -- read over live source.

    Why required, checkable against the code: the key must be written on the cfg the launch
    hands the solver, at the launch, so that it reaches a resume. A second caller in the cfg
    generator would put the key into a base cfg at preparation, which a resume does not
    launch, and would make a run prepared under one value look armed under another.

    What kills it: a call from ``_generate_TRITON_SWMM_cfg``, or the launch call being moved
    into a helper that the launch no longer reaches.

    A second correct implementation under which it still passes: one that performs the write
    in a private helper the launch method calls, provided that helper is the single caller.
    """
    run_src = Path(run_simulation_mod.__file__)
    scenario_src = Path(scenario_mod.__file__)
    assert _call_sites(run_src, "set_swmm_snapshot_failed_replay") == {"prepare_simulation_command"}
    assert _call_sites(scenario_src, "set_swmm_snapshot_failed_replay") == set()


def test_write_failed_is_a_classified_replay_reason():
    """Property: a replay row whose reason is ``write-failed`` is classified, with no raw fallback.

    Class: a replay line in the form the solver emits, carrying the new reason.

    Why required, checkable against the code: ``REPLAY_REASONS`` is a closed set, and a token
    outside it leaves ``replay_reason`` None and records the line under ``replay_reason_raw``.
    The opt-in replay is a known reason, so its row must classify.

    What kills it: leaving ``write-failed`` out of ``REPLAY_REASONS``.

    A second correct implementation under which it still passes: a reason table keyed by
    marker text rather than a set, if the same line classifies the same way.
    """
    assert "write-failed" in re_mod.REPLAY_REASONS
    line = (
        "[..] SWMM exchange history replayed to t=3600 s (11435 steps); resuming live segment "
        "[resume-event checkpoint=7 path=replay reason=write-failed]"
    )
    (event,) = re_mod.harvest_resume_events(line)
    assert event["path"] == "replay"
    assert event["replay_reason"] == "write-failed"
    assert event["replay_reason_status"] == "parsed"
    assert event["replay_reason_raw"] is None


def test_a_write_failed_replay_row_raises_exactly_one_alarm():
    """Property: a replay row with reason ``write-failed`` raises one alarm; one with another reason raises none.

    Class: two replay rows differing only in their reason -- the new one and ``absent``, a
    replay that is the ordinary fallback and not an alarm.

    Why required, checkable against the code: the replay continues with correct numbers, but
    the failed write behind it is undiagnosed and may recur, so it is reported beside the
    resume verdict. An alarm keyed on ``path == "replay"`` alone would fire on every ordinary
    replay.

    What kills it: omitting the alarm, or firing it for any replay row.

    A second correct implementation under which it still passes: one that emits the alarm
    from the classifier instead of from the row-level check, if the audit lists one alarm.
    """
    write_failed = {"path": "replay", "replay_reason": "write-failed", "checkpoint_id": "7"}
    ordinary = {"path": "replay", "replay_reason": "absent", "checkpoint_id": "7"}
    alarms = re_mod._row_alarms(write_failed)
    assert [a["kind"] for a in alarms] == ["write-failed"]
    assert re_mod._row_alarms(ordinary) == []
    audit = re_mod.audit_resume_evidence([write_failed], n_resumes=1)
    assert [a["kind"] for a in audit["alarms"]] == ["write-failed"]
    assert audit["verdict"] == "complete"


def test_the_picker_message_states_what_the_solver_will_do():
    """Property: the message names a replay when the launch writes the key and a stop when it does not.

    Class: the message function over its boolean, with the same checkpoints -- the two
    differently-positioned values of the one input that decides the wording.

    Why required, checkable against the code: the picker resumes at a checkpoint whose
    snapshot write failed, and what the solver does there depends on the cfg it is given. A
    message that says "stops" for a launch that replays, or the reverse, misleads whoever
    reads the log to diagnose a resume.

    What kills it: a message fixed to one wording, or one that reads the opposite value.

    A second correct implementation under which it still passes: one that builds the two
    sentences from a lookup keyed on the boolean and joins them with the same prefix.
    """
    stops = failed_write_resume_message(7, 9, False)
    replays = failed_write_resume_message(7, 9, True)
    assert "stops there with its write-failed error" in stops
    assert "replays the SWMM exchange history there" in replays
    assert "replays" not in stops
    assert "stops there" not in replays
    for message in (stops, replays):
        assert "checkpoint 7 failed to write" in message
        assert "unusable checkpoint 9" in message
