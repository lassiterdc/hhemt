"""Ratchet: every flag of the suite runner's parser is reachable from a verb or is named as unreachable.

`hhemt.suite._runner.build_parser()` is the runner's flag surface, and `hhemt test toolkit` is the
verb surface that is meant to replace invoking `python -m hhemt.suite._runner` by hand. The two
drift silently: a flag added to the parser that no verb forwards is usable only by a caller who
knows the module path, and nothing records that the omission was a decision. This file makes the
omission a recorded decision. Each parser flag must be in exactly one of two sets:

* the VERB-REACHABLE set, derived by invoking every `toolkit` verb that forwards to the runner and
  reading the argv it hands over (so a flag a verb forwards under a hard-coded keyword, such as
  `triage=True`, is seen, and a flag the verb merely declares but never forwards is not); or
* `UNREACHABLE`, a mapping from flag to the consumer that still needs it by hand.

The test fails in three directions, and each is a separate test below: a parser flag in neither set
(a new flag nobody classified), an `UNREACHABLE` key that is no longer a parser flag (a stale
entry), and an `UNREACHABLE` key that a verb now forwards (the entry has to be deleted, which is
the point where the flag stops being a hand-only surface).

Deliberately NOT a hand-listed reachable set: a hand list would be a second copy of the verb
definitions and would go stale in exactly the direction this test exists to catch.

Nothing here compiles or runs a solver or a pytest session; the runner's `main` is replaced by a
recorder, so a verb's body executes only as far as the argv it builds.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from hhemt.suite import _runner
from hhemt.suite._cli import _argv, toolkit_app

# flag -> the consumer that still needs it by hand. A flag leaves this mapping when a verb forwards
# it. "rerun.sh" lives in the companion harness directory `harness/pytest_suite/` and is the PRINCIPAL
# consumer of 11 flags, not the only one: for 7 of them a test, a doc or the justfile also names the
# flag, listed after it, so retiring `rerun.sh` does not retire those consumers. The two flags with no
# located caller are recorded as such rather than given an invented consumer.
UNREACHABLE: dict[str, str] = {
    "--python": "tests/test_suite_*.py pass it; harness/pytest_suite/submit_suite_uva.sh omits it on purpose",
    "--cheap-bins": "rerun.sh; harness check_plan_argv.sh",
    "--heavy-split-budget-min": (
        "rerun.sh; check_plan_argv.sh, harness README.md, tests/test_suite_chain_split.py, docs/reference/cli.md"
    ),
    "--durations-from": (
        "rerun.sh; "
        "check_plan_argv.sh, tests/test_suite_chain_split.py, "
        "tests/test_suite_tier_selection.py, docs/reference/cli.md"
    ),
    "--isolate-trees-per-chunk": (
        "rerun.sh; "
        "check_plan_argv.sh, README.md, tests/test_suite_chain_split.py, "
        "tests/test_suite_isolation_root_node_local.py, docs/reference/cli.md"
    ),
    "--select-tier": "rerun.sh; check_plan_argv.sh, tests/test_suite_tier_selection.py, justfile",
    "--chunk-workers": "rerun.sh; check_plan_argv.sh, tests/test_suite_chunk_workers.py",
    "--keep-failed-trees": "rerun.sh; check_plan_argv.sh, tests/test_suite_isolation_root_node_local.py",
    "--declare-non-compile-bearing": "tests/test_suite_tier_selection.py and the other non-compile planning tests",
    "--warm-target": "tests/test_suite_tier_selection.py and the other tests that plan with a stub warm",
    "--allow-dirty": "no caller located: only the refusal message in _runner.py names it",
    "--warm-performed-externally": "rerun.sh",
    "--warm-log": "rerun.sh",
    "--warm-result": "rerun.sh",
    "--expect-triton-pin": "rerun.sh",
    "--n-chunks": "no caller located: a cross-check against the manifest that no script or test passes",
}

_DUMMY_BY_TYPE = {"integer": "1", "float": "1.0"}


def parser_flags() -> set[str]:
    """Return every long option the runner's argparse parser defines, `--help` excluded.

    Read from the parser object rather than from the source text, so a flag added through any
    construction of `add_argument` (a loop, a helper) is seen.
    """
    parser = _runner.build_parser()
    return {opt for action in parser._actions for opt in action.option_strings if opt.startswith("--")} - {"--help"}


def _dummy_argv_for(command: Any) -> list[str]:
    """Return an argv that gives every option of a Click command a value, so each is forwarded."""
    argv: list[str] = []
    for param in command.params:
        opts = list(getattr(param, "opts", []))
        if not opts:
            continue
        if getattr(param, "is_flag", False):
            argv.append(opts[0])
            continue
        choices = getattr(param.type, "choices", None)
        argv += [opts[0], list(choices)[0] if choices else _DUMMY_BY_TYPE.get(param.type.name, "x")]
    return argv


# Verbs that call a different module and would run real work with dummy arguments. A verb named
# here is never invoked; every other verb is invoked and must reach the runner or the derivation
# refuses, so a verb is never skipped in silence.
NON_FORWARDING_VERBS = frozenset({"aggregate", "owed"})


def forwarded_argvs(
    app: typer.Typer,
    monkeypatch: pytest.MonkeyPatch,
    non_forwarding: frozenset[str] | set[str] = NON_FORWARDING_VERBS,
) -> list[list[str]]:
    """Return the argv each verb of `app` hands to `hhemt.suite._runner.main`.

    Every verb not named in `non_forwarding` is invoked with a dummy value for every option it
    declares, with `_runner.main` replaced by a recorder, both as a module attribute (reached by a
    lazy import or a helper) and as any name bound to it in the verb's own module globals (reached
    by a module-level import). The runner's three entry functions are stubbed so a verb that
    escapes the recorder cannot run real work. A verb that records no argv fails the derivation
    with AssertionError: Deliberately NOT a skip, because a skipped forwarding verb makes the
    reachable set silently short.
    """
    recorded: list[list[str]] = []
    original_main = _runner.main

    def _record(argv: list[str] | None = None) -> int:
        recorded.append(list(argv or []))
        return 0

    monkeypatch.setattr(_runner, "main", _record)
    for entry in ("plan", "triage", "run_chunk"):
        monkeypatch.setattr(_runner, entry, lambda *a, **k: 0)
    group = typer.main.get_command(app)
    runner = CliRunner()
    for name, command in group.commands.items():
        if name in non_forwarding:
            continue
        for g_name, g_val in list(_globals_of(command).items()):
            if g_val is original_main:
                monkeypatch.setitem(_globals_of(command), g_name, _record)
        before = len(recorded)
        result = runner.invoke(app, [name, *_dummy_argv_for(command)])
        assert result.exit_code == 0, f"verb {name!r} did not run to its forwarding call: {result.output}"
        assert len(recorded) > before, f"verb {name!r} reached no recorder: neither read nor declared non-forwarding"
    return recorded


def _globals_of(command: Any) -> dict[str, Any]:
    """Return the module globals of a command's underlying callback, looking through wrappers."""
    return inspect.unwrap(command.callback).__globals__


def reachable_flags(
    app: typer.Typer,
    monkeypatch: pytest.MonkeyPatch,
    non_forwarding: frozenset[str] | set[str] = NON_FORWARDING_VERBS,
) -> set[str]:
    """Return the runner flags that some verb of `app` forwards, read from the recorded argv.

    Deliberately NOT the union of the verbs' declared option names: a declared option that the
    body never forwards would then count as reachable, and the hard-coded `triage=True` keyword
    in the `triage` verb would be missed.
    """
    return {tok for argv in forwarded_argvs(app, monkeypatch, non_forwarding) for tok in argv if tok.startswith("--")}


def classify(flags: set[str], reachable: set[str], unreachable: dict[str, str]) -> list[str]:
    """Return one violation line per way the three sets disagree; an empty list means they agree."""
    out = [
        f"{f}: a parser flag that is in neither the reachable set nor UNREACHABLE"
        for f in sorted(flags - reachable - set(unreachable))
    ]
    out += [f"{f}: an UNREACHABLE key that is not a parser flag" for f in sorted(set(unreachable) - flags)]
    out += [
        f"{f}: an UNREACHABLE key that a verb now forwards; delete the entry"
        for f in sorted(set(unreachable) & reachable)
    ]
    return out


# --- the ratchet over the live code -----------------------------------------------------------------


def test_every_parser_flag_is_reachable_or_recorded_as_unreachable(monkeypatch):
    """Property: no runner flag is in neither the verb-reachable set nor `UNREACHABLE`.

    Class: every `--flag` of `build_parser()`. Why it is required: a flag that no verb forwards
    and that no mapping names is a hand-only surface that nobody decided to leave hand-only, and
    the drift this file guards is that each new flag lands there by default. What kills it: adding
    `p.add_argument("--dummy")` to `build_parser()`. A second correct implementation under which
    it still passes: a runner that registers its flags through a table instead of repeated
    `add_argument` calls, since only the parser object's option strings are read.
    """
    flags = parser_flags()
    assert flags, "the parser reported no flags, so every later assertion is vacuous"
    reachable = reachable_flags(toolkit_app, monkeypatch)
    unclassified = flags - reachable - set(UNREACHABLE)
    assert not unclassified, f"runner flags that no verb forwards and UNREACHABLE does not name: {sorted(unclassified)}"


def test_every_unreachable_key_is_still_a_parser_flag():
    """Property: every `UNREACHABLE` key is a flag the parser still defines.

    Class: every key of the mapping. Why it is required: a key that outlives its flag documents a
    consumer of something that no longer exists, and the mapping then misstates the hand-only
    surface. What kills it: renaming `--cheap-bins` in the parser without editing the mapping. A
    second correct implementation under which it still passes: a parser that defines the flag
    through an alias option string, since the check is membership in the option-string set.
    """
    stale = set(UNREACHABLE) - parser_flags()
    assert not stale, f"UNREACHABLE keys that are not parser flags: {sorted(stale)}"


def test_no_unreachable_key_is_forwarded_by_a_verb(monkeypatch):
    """Property: no `UNREACHABLE` key is a flag that a verb now forwards.

    Class: every key of the mapping, against the derived reachable set. Why it is required: a flag
    exposed on a verb stops needing a hand-only consumer, and leaving its entry says the opposite
    of the code. What kills it: forwarding `--cheap-bins` from `plan` without deleting the entry.
    A second correct implementation under which it still passes: a verb that forwards the flag
    through `_argv` keywords rather than a literal token, since the argv is what is read.
    """
    now_reachable = set(UNREACHABLE) & reachable_flags(toolkit_app, monkeypatch)
    assert not now_reachable, f"UNREACHABLE keys a verb forwards, whose entries must go: {sorted(now_reachable)}"


def test_every_unreachable_entry_names_a_consumer():
    """Property: each `UNREACHABLE` value is a non-blank consumer description.

    Class: every value of the mapping. Why it is required: the mapping is a record of who still
    needs the flag, and a blank value records nothing while satisfying the key check. What kills
    it: `"--allow-dirty": ""`. A second correct implementation under which it still passes: values
    that are structured objects with a `str` form, since only non-blankness is read.
    """
    blank = [flag for flag, consumer in UNREACHABLE.items() if not consumer.strip()]
    assert not blank, f"UNREACHABLE entries with no consumer: {blank}"


def test_the_derived_reachable_set_is_nonempty_and_parses(monkeypatch):
    """Property: the derivation finds forwarded flags, and the runner's own parser accepts them.

    Class: the argv every forwarding verb hands over. Why it is required: the other tests treat
    an empty reachable set as 'nothing is reachable', which would pass every `UNREACHABLE` key
    and flag every parser flag identically, so a broken derivation must be loud. A verb that
    forwards a flag the parser lacks would also fail only at the moment someone uses the verb.
    What kills it: a derivation that skips every verb. A second correct implementation under
    which it still passes: a derivation that records argv at the `subprocess` boundary instead.
    """
    recorded = forwarded_argvs(toolkit_app, monkeypatch)
    assert recorded, "no verb reached the runner, so the reachable set is empty by construction"
    for argv in recorded:
        _runner.build_parser().parse_args(argv)


# --- the differential: the ratchet reddens on a violation and stays green on a legitimate move -----


def test_a_dummy_parser_flag_is_a_violation():
    """Property: a flag in the parser and in neither set produces a violation naming it.

    Class: any new parser flag. Why it is required: this is the failure the ratchet exists for.
    The inputs are synthetic and independent of the live mapping, so retiring a live entry never
    changes this test's verdict. What kills it: a `classify` that only checks `UNREACHABLE` keys
    against the parser. A second correct implementation under which it still passes: one that
    returns a structured violation object per flag, since only the flag name in the report is read.
    """
    violations = classify({"--toolkit", "--dummy"}, {"--toolkit"}, {})
    assert [v.split(":")[0] for v in violations] == ["--dummy"], violations


def test_exposing_a_flag_on_a_verb_while_deleting_its_entry_stays_green(monkeypatch):
    """Property: forwarding a flag from a verb and deleting its entry leaves no violation.

    Class: the legitimate move that retires one `UNREACHABLE` entry (the shape's example is
    `--cheap-bins` on `plan`). Why it is required: a ratchet that reddens on its own intended exit
    would be bypassed rather than followed. The verb is built here and the sets are synthetic, so
    neither the live app nor the live mapping is read and retiring a live entry cannot change this
    verdict. What kills it: a derivation that counts only options declared on a hard-coded verb
    list. A second correct implementation under which it still passes: a verb that forwards
    through a literal argv list instead of `_argv`, since the recorded argv is what is read.
    """
    app = typer.Typer(no_args_is_help=True)

    @app.command("plan")
    def _plan(
        toolkit: str = typer.Option(..., "--toolkit"),
        cheap_bins: int = typer.Option(1, "--cheap-bins"),
    ) -> None:
        """Forward to the runner with an extra flag."""
        from hhemt.suite._runner import main

        raise typer.Exit(code=main(_argv(toolkit=toolkit, cheap_bins=cheap_bins)))

    @app.command("other")
    def _other() -> None:
        """A second verb, so the app is a group whatever the number of commands."""
        raise typer.Exit(code=0)

    reachable = reachable_flags(app, monkeypatch, non_forwarding={"other"})
    assert reachable == {"--toolkit", "--cheap-bins"}, reachable
    flags = {"--toolkit", "--cheap-bins", "--python"}
    assert classify(flags, reachable, {"--python": "a consumer"}) == []
    kept = classify(flags, reachable, {"--python": "a consumer", "--cheap-bins": "a consumer"})
    assert [v.split(":")[0] for v in kept] == ["--cheap-bins"], kept


@pytest.mark.parametrize(
    ("flags", "reachable", "unreachable", "named"),
    [
        ({"--a", "--b"}, {"--a"}, {"--b": "x", "--gone": "x"}, ["--gone"]),
        ({"--a", "--b"}, {"--a"}, {"--b": "x"}, []),
        ({"--a", "--b"}, {"--a", "--b"}, {"--b": "x"}, ["--b"]),
    ],
    ids=["stale-key", "all-classified", "key-now-reachable"],
)
def test_classify_reports_each_disagreement_and_is_silent_on_agreement(flags, reachable, unreachable, named):
    """Property: `classify` names a stale key and a now-reachable key, and is silent on agreement.

    Class: the three relations between the parser flags, the reachable set and `UNREACHABLE`.
    Why it is required: the live-code tests above can only show green on today's code, so these
    synthetic inputs are the only place the red directions of the second and third check, and
    the silent arm, are exercised. What kills it: dropping the second or third comprehension from
    `classify`. A second correct implementation under which it still passes: a set-algebra form
    over one combined structure, since only the report lines are read.
    """
    assert [v.split(":")[0] for v in classify(flags, reachable, unreachable)] == named
