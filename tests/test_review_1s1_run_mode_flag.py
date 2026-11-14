"""Reviewer tests for chunk 1s.1: `hhemt run --mode {resume,fresh}` replaces `--from-scratch` and `--resume`.

The problem this file guards: `hhemt run` once spelled the same choice (wipe or continue) as two
booleans plus a hand-written mutual-exclusion check, while `hhemt run-experiment` spelled it as
one closed `--mode` choice over `RunMode`. The chunk makes the two verbs agree. Each test below
quantifies over a CLASS the suite can enumerate (every `RunMode` member, every retired spelling,
every `hhemt run` command a doc prints) rather than over the one input the coder's own tests use.

No test here compiles a solver or runs a simulation. Every invocation of the Typer app carries a
literal `--dry-run` or `--help`, so the build-request census treats these nodes as excused, and
the analysis, the system and the config loader are stubbed exactly as in `test_cli_12`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

import hhemt.analysis
import hhemt.system
import hhemt.validation
from hhemt.cli import app
from hhemt.orchestration import RunMode

runner = CliRunner()

_REPO = Path(__file__).resolve().parents[1]
_PLAIN_ENV = {"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1", "_TYPER_FORCE_DISABLE_TERMINAL": "1"}
_RETIRED = ("--from-scratch", "--resume")


class _Stop(Exception):
    """Raised by the recording stub so the command ends right after run() is called."""


@pytest.fixture
def run_calls(monkeypatch):
    """Stub the heavy classes and return the list that run() keyword arguments land in."""
    calls: list[dict] = []

    class _Validation:
        has_warnings = False
        warnings: list = []
        is_valid = True
        errors: list = []

    class _Analysis:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, **kwargs):
            return _Validation()

        def run(self, **kwargs):
            calls.append(kwargs)
            raise _Stop

    class _System:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)
    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", _System)
    monkeypatch.setattr(hhemt.validation, "assert_both_configs_load", lambda *a, **k: None)
    return calls


@pytest.fixture
def base_argv(tmp_path):
    """The two required config options plus the literal dry-run flag every invocation here carries."""
    system_config = tmp_path / "system.yaml"
    analysis_config = tmp_path / "analysis.yaml"
    system_config.write_text("version: 1\n")
    analysis_config.write_text("version: 1\n")
    return ["run", "--system-config", str(system_config), "--analysis-config", str(analysis_config), "--dry-run"]


def _verb(name: str):
    """Return the Click command Typer built for one verb."""
    group = typer.main.get_group(app)
    return group.commands[name]


def _option_names(command) -> set[str]:
    """Every spelling (long, short, secondary) the command registers for its options."""
    names: set[str] = set()
    for param in command.params:
        names.update(getattr(param, "opts", ()))
        names.update(getattr(param, "secondary_opts", ()))
    return names


def _mode_param(name: str):
    """Return the Click parameter named `mode` on one verb.

    Only the attributes both Typer generations expose are read from it (`default`, `opts`,
    `help`): Typer 0.29 replaced the Click type object with its own type descriptor, so `.type`
    exists on 0.26 and not on 0.29. The choice set and the case rule are read through the
    parser instead (`_closed_choices`, `_is_case_sensitive`), which is the public behaviour.
    """
    return next(p for p in _verb(name).params if p.name == "mode")


_NOT_ONE_OF = re.compile(r"is not one of ((?:'[^']*'(?:, )?)+)")


def _refusal_of(verb: str, value: str) -> str | None:
    """Return the quoted choice list the parser names when it refuses `--mode value`, else None.

    The parser, not an attribute of the option, is what a user meets, so this reads its refusal
    message: `'x' is not one of 'fresh', 'resume'.`, which Typer 0.26 and 0.29 word the same from
    that phrase onward (0.29 prefixes `Value error,`). The invocation carries `--dry-run` so it
    can never start work, and Rich wraps nothing at the 200 columns `_PLAIN_ENV` sets.
    """
    result = runner.invoke(app, [verb, "--mode", value, "--dry-run"], env=_PLAIN_ENV)
    found = _NOT_ONE_OF.search(" ".join(result.output.split()))
    return found.group(1) if found else None


def _closed_choices(verb: str) -> tuple[str, ...]:
    """Return the choices one verb's `--mode` parser accepts, in the order it names them.

    Read from the refusal of a value no choice set can contain; the must-fire control is that the
    refusal exists at all, so a verb whose `--mode` became a free string fails here, loudly,
    instead of reporting an empty set.
    """
    quoted = _refusal_of(verb, "not-a-mode")
    assert quoted is not None, f"{verb} --mode did not refuse a value outside every choice set"
    return tuple(re.findall(r"'([^']*)'", quoted))


def _is_case_sensitive(verb: str) -> bool:
    """Return whether the verb's `--mode` parser refuses the upper-case spelling of every real member."""
    return all(_refusal_of(verb, member.value.upper()) is not None for member in RunMode)


def test_run_mode_is_the_same_closed_choice_run_experiment_declares():
    """Property: `hhemt run --mode` is the closed choice over every `RunMode` member that `run-experiment` declares.

    Class: every `RunMode` member, and both verbs that take `--mode`. Why it is required: the
    shape adopts `--mode` "as `run-experiment` has", and the two verbs share one enum so that a
    script learns one vocabulary; a choice set, a case rule or a default that differs between
    the verbs gives the same word two meanings. What kills it: declaring the option as a free
    string, as a `Literal` missing a member, with a different `case_sensitive`, or with a default
    other than resume. A second correct implementation under which it still passes: a `Literal`
    annotation or a callback-validated option, since only the choices the parser names in its
    refusal, whether it refuses an upper-case member, and the option default are compared, not the
    Python class that produced them nor Typer's internal type object, which differs between
    Typer 0.26 and 0.29.
    """
    run_mode, experiment_mode = _mode_param("run"), _mode_param("run-experiment")

    assert _closed_choices("run") == _closed_choices("run-experiment")
    assert set(_closed_choices("run")) == {member.value for member in RunMode}
    assert _is_case_sensitive("run") == _is_case_sensitive("run-experiment")
    assert str(run_mode.default) == str(experiment_mode.default) == RunMode.resume.value
    assert "--mode" in run_mode.opts


@pytest.mark.parametrize(
    "retired", [*_RETIRED, "--from-scratch=true", "--resume=false", "--no-resume", "--from_scratch"]
)
@pytest.mark.parametrize("position", ["alone", "before-mode", "after-mode", "with-overwrite"])
def test_every_retired_spelling_is_refused_at_parse_in_every_position(run_calls, base_argv, retired, position):
    """Property: no retired flag spelling parses on `hhemt run`, wherever it sits in the argument list.

    Class: the two retired flags and the spellings a script could plausibly still type (an
    `=value` form, a negated form, an underscore form), each placed alone, before `--mode`, after it,
    and beside `--overwrite`. Why it is required: a retired flag that still parsed would be a
    second route to the wipe that no page describes, and the coder's own test places each retired
    flag only once, last. What kills it: re-adding either flag as an alias, or accepting an
    unknown option through a catch-all. A second correct implementation under which it still
    passes: any refusal that exits 2 without reaching `run()`, since the exit code, the absence of
    a `run()` call and the offending token in the message are what is asserted.
    """
    flags = {
        "alone": [retired],
        "before-mode": [retired, "--mode", "resume"],
        "after-mode": ["--mode", "resume", retired],
        "with-overwrite": ["--overwrite", retired, "--mode", "fresh"],
    }[position]

    result = runner.invoke(app, [*base_argv, *flags])

    assert result.exit_code == 2, result.output
    assert not run_calls, f"{retired} reached run(): {run_calls}"
    assert retired.split("=")[0] in result.output, result.output

    control = runner.invoke(app, [*base_argv, "--mode", "resume"])
    assert len(run_calls) == 1, f"control without the retired flag did not reach run(): {control.output}"


@pytest.mark.parametrize("overwrite", [False, True], ids=["alone", "with-overwrite"])
@pytest.mark.parametrize("member", list(RunMode), ids=lambda m: m.value)
def test_each_run_mode_reaches_run_as_the_boolean_and_overwrite_stays_orthogonal(
    run_calls, base_argv, member, overwrite
):
    """Property: `--mode M` forwards `from_scratch == (M is fresh)`; `--overwrite` changes only `override_force_rerun`.

    Class: every `RunMode` member crossed with `--overwrite` present and absent, taken from the
    enum so a member added later is covered. Why it is required: the shape keeps `--overwrite`
    as the orthogonal force-rerun shorthand and keeps `Analysis.run` on its boolean, so the CLI
    owns the one conversion; an `--overwrite` that flipped the wipe boolean, or a mode that lost
    it, destroys work or silently skips it. What kills it: converting `--mode` anywhere but once
    (for example treating `--overwrite` as a third mode), or dropping the conversion for a member.
    A second correct implementation under which it still passes: a `RunMode`-keyed table or a
    Typer callback doing the conversion, since only the keyword arguments `run()` received are
    asserted, never the console wording.
    """
    flags = ["--mode", member.value, *(["--overwrite"] if overwrite else [])]

    result = runner.invoke(app, [*base_argv, *flags])

    assert len(run_calls) == 1, f"run() not reached; exit {result.exit_code}: {result.output}"
    assert run_calls[0]["from_scratch"] is (member is RunMode.fresh)
    assert run_calls[0]["override_force_rerun"] == ("all" if overwrite else None)
    assert re.search(rf"^\s*Mode:\s+{member.value}\s*$", result.output, re.MULTILINE), result.output


@pytest.mark.parametrize("value", ["overwrite", "from-scratch", "resume,fresh", "", "fresh ", "Fresh", "RESUME"])
def test_a_mode_value_outside_the_closed_set_is_refused_at_parse(run_calls, base_argv, value):
    """Property: a `--mode` value outside the `RunMode` set (under the option's own case rule) exits 2 before `run()`.

    Class: values a user would plausibly type for the old behaviour (`overwrite`, `from-scratch`),
    a joined list, an empty and a trailing-space value, and the upper- and title-case spellings of
    real members. Why it is required: the shape calls the option a parse-time closed choice, so a
    wrong word must be refused by the parser rather than reaching the workflow as a mode that
    means nothing. What kills it: a free-string option that converts with `== "fresh"` and treats
    every other word as resume. A second correct implementation under which it still passes: a
    case-insensitive choice, because the expectation is computed from whether the parser itself
    refuses an upper-case member rather than hard-coded.
    """
    members = {member.value for member in RunMode}
    accepted = value in members if _is_case_sensitive("run") else value.lower() in members

    result = runner.invoke(app, [*base_argv, "--mode", value])

    if accepted:
        assert len(run_calls) == 1, result.output
    else:
        assert result.exit_code == 2, result.output
        assert not run_calls, f"--mode {value!r} reached run(): {run_calls}"


def test_help_names_the_same_mode_choices_on_both_verbs_and_no_retired_option():
    """Property: `run --help` shows `--mode` with the choice set `run-experiment --help` shows, and no retired flag.

    Class: the two verbs' rendered help, and every option the `run` command registers. Why it is
    required: the help is what a user reads to learn the new spelling, and the chunk's Definition
    of Done names it; the order of the bracketed choices is NOT asserted, because the enum
    declares `fresh` before `resume` and renders `[fresh|resume]` (Typer 0.26) or `<fresh|resume>`
    (Typer 0.29), not the `[resume|fresh]` the Definition of Done sentence writes. What kills it:
    a help that omits `--mode`, renders a different set than `run-experiment`, or still lists a
    retired flag as an option. A second
    correct implementation under which it still passes: any renderer that prints the same set in
    another order, since the set of words between the brackets is compared, not the string.
    """

    def choices_of(verb: str) -> set[str]:
        result = runner.invoke(app, [verb, "--help"], env=_PLAIN_ENV)
        assert result.exit_code == 0, result.output
        found = re.search(r"--mode\s+[\[<]?([a-z]+(?:\|[a-z]+)+)[\]>]?", result.output)
        assert found, f"{verb} help carries no --mode choice set:\n{result.output}"
        return set(found.group(1).split("|"))

    assert choices_of("run") == choices_of("run-experiment") == {member.value for member in RunMode}
    assert not set(_RETIRED) & _option_names(_verb("run"))
    help_text = runner.invoke(app, ["run", "--help"], env=_PLAIN_ENV).output
    assert not any(re.search(rf"{re.escape(flag)}(?![\w-])", help_text) for flag in _RETIRED), help_text


def test_the_wipe_override_claim_in_the_mode_help_matches_the_options_each_verb_registers():
    """Property: the `--mode` help says `run` has no wipe override exactly when `run` registers none.

    Class: the one claim the chunk's help text makes about the sibling verb, checked against the
    option sets of both verbs. Why it is required: the help tells a user that `run-experiment`
    has the override and `run` does not, and `forcing-reruns.md` repeats it; a user who reads a
    false "no override" on a verb that grew one, or the reverse, has been misled about how to
    delete work on purpose. What kills it: adding `--override-wipe-nonempty` to `run` without
    rewording the help, or removing it from `run-experiment`. A second correct implementation
    under which it still passes: adding the override to `run` and rewording the help to say so,
    because the test couples the sentence to the tree in both directions.
    """
    help_text = " ".join(str(_mode_param("run").help).split())
    says_no_override = "no override for that refusal" in help_text

    assert "--override-wipe-nonempty" in _option_names(_verb("run-experiment"))
    assert ("--override-wipe-nonempty" not in _option_names(_verb("run"))) == says_no_override


def _run_command_flags(text: str) -> list[tuple[str, str]]:
    """Return `(command text, flag)` for each `--flag` an `hhemt run` command in `text` carries.

    A command is the text from `hhemt run` (never `hhemt run-experiment`) to the first backtick or
    unescaped end of line, joined across backslash continuations, so a flag named later in the same
    prose sentence is not attributed to the command.
    """
    pairs: list[tuple[str, str]] = []
    for found in re.finditer(r"hhemt run(?![-\w])((?:[^`\n\\]|\\\n)*)", text):
        command = found.group(0).replace("\\\n", " ")
        pairs.extend((command, flag) for flag in re.findall(r"(?<![\w-])(--[A-Za-z][\w-]*)", found.group(1)))
    return pairs


def _published_pages() -> list[Path]:
    """The pages a reader can reach: `README.md` plus every Markdown page under `docs/`."""
    return [_REPO / "README.md", *sorted((_REPO / "docs").rglob("*.md"))]


def test_every_flag_a_published_hhemt_run_command_carries_is_an_option_the_run_command_registers():
    """Property: no page prints an `hhemt run` command with an option `hhemt run` no longer has.

    Class: every `hhemt run ...` command printed in `README.md` and any Markdown page under
    `docs/`, crossed with every `--flag` it carries. Why it is required: the chunk edits four
    pages by hand, and a retired flag left in a command a reader copies exits 2; the drift is
    invisible to a grep for the two retired names alone, because it also covers a page that
    mistypes the new one. What kills it: leaving `--from-scratch` or `--resume` in a printed
    command, or writing `--mod`. A second correct implementation under which it still passes: any
    wording of the pages, since only the flags inside a printed `hhemt run` command are compared
    with the registered option names. The must-fire controls are in the test body: the scan finds a
    nonzero number of commands, and a planted command carrying a retired flag is reported.
    """
    registered = _option_names(_verb("run")) | {"--help"}

    planted = _run_command_flags("```bash\nhhemt run --system-config a.yaml \\\n    --from-scratch\n```")
    assert [flag for _, flag in planted if flag not in registered] == ["--from-scratch"]

    scanned = [(page, pair) for page in _published_pages() for pair in _run_command_flags(page.read_text())]
    assert len(scanned) >= 10, f"scan found only {len(scanned)} flags; the matcher is broken, not the pages clean"
    assert any(flag == "--mode" for _, (_, flag) in scanned), "no page prints the new --mode flag in a run command"

    unknown = [
        (page.relative_to(_REPO).as_posix(), command, flag)
        for page, (command, flag) in scanned
        if flag not in registered
    ]
    assert not unknown, f"printed hhemt run commands carry flags the run command does not register: {unknown}"
