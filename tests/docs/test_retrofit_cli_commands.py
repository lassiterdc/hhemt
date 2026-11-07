"""Every `hhemt` command a page shows exists in the command line interface, with its flags.

Several corrections in the review of the reference, how-to and tutorial pages replaced
a Python call with a command line (`hhemt publish`, `hhemt reprex emit`, `hhemt reprocess
--start-with render`, `hhemt run --status`, and others), and the pages named the flags of
those commands from memory. A page that names a command or flag the program lacks gives
the reader a line that fails, and no gate compared the two. This test reads the command
tree of the Typer app in `src/hhemt/cli.py` and checks every `hhemt` invocation in a
shell code block, or in an inline code span that starts with `hhemt`, against it.

Rules of the check, each a decision about a page text, not about the program:

* A shell block is a fenced block with no language or the language bash, sh, shell or
  console. Blocks in other languages (yaml, python, text output) are not commands.
* A backslash at the end of a line joins it to the next line, because a flag on a
  continued line belongs to the command above it.
* A command is checked only when it opens a shell segment (the start of the line, or
  after `&&`, `||`, `;` or a pipe), so `conda activate hhemt` and a path that contains
  the name are not commands.
* A token that opens with a brace, an angle bracket or an ellipsis is a placeholder for
  text the reader supplies. A placeholder in the command position ends the check of
  that line; a placeholder anywhere else is a value.
* `--help` is accepted at every level, because the argument parser adds it outside the
  declared options.

The tests that run no page feed `check_text` a page text, so the controls need no edit
to a tracked page. No test here runs a solver.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import typer

from tests.docs.test_docs_style import tracked_pages

_REPO = Path(__file__).resolve().parents[2]
_SHELL_LANGS = {"", "bash", "sh", "shell", "console"}
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+-]*)")
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_SEGMENT_BREAK = {"&&", "||", ";", "|"}
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass
class Node:
    """One command of the tree: its option strings and its subcommands by name."""

    options: set[str]
    subcommands: dict[str, Node] = field(default_factory=dict)


def command_tree() -> Node:
    """The command tree of the `hhemt` Typer app, built by introspection.

    Every node accepts `--help`, because the argument parser adds that option outside
    the declared parameters.
    """
    from hhemt.cli import app

    def build(cmd) -> Node:
        options = {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)} | {"--help"}
        subs = {name: build(sub) for name, sub in getattr(cmd, "commands", {}).items()}
        return Node(options, subs)

    return build(typer.main.get_command(app))


def _is_placeholder(token: str) -> bool:
    return token.startswith(("{", "<")) or "…" in token or "..." in token


def _split_tokens(text: str) -> list[str]:
    try:
        return shlex.split(text, comments=True)
    except ValueError:  # an apostrophe in prose leaves a quote open
        return text.split()


def _commands_in(tokens: list[str]) -> list[list[str]]:
    """The token lists of the `hhemt` commands that open a segment of `tokens`."""
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEGMENT_BREAK:
            segments.append([])
        else:
            segments[-1].append(tok)
    found = []
    for seg in segments:
        while seg and (_ENV_ASSIGNMENT.match(seg[0]) or seg[0] == "$"):
            seg = seg[1:]
        if seg and seg[0] == "hhemt":
            found.append(seg[1:])
    return found


def check_command(args: list[str], root: Node) -> str | None:
    """A problem message for one command (the tokens after `hhemt`), or None when it is valid."""
    node, path = root, ["hhemt"]
    flags: list[tuple[str, Node]] = []
    for tok in args:
        if _is_placeholder(tok) and not tok.startswith("-"):
            if node.subcommands:
                return None  # the command position holds a placeholder
            continue
        if re.match(r"^--?[A-Za-z]", tok):
            flags.append((tok.split("=", 1)[0], node))
            continue
        if node.subcommands:
            if tok not in node.subcommands:
                return f"`{' '.join(path)}` has no command `{tok}`"
            node = node.subcommands[tok]
            path.append(tok)
    for flag, owner in flags:
        if flag not in owner.options and flag != "-h":
            return f"`{' '.join(path)}` has no option `{flag}`"
    return None


def iter_commands(text: str):
    """Yield (line number, command text, tokens after `hhemt`) for each command in a page text."""
    lines = text.splitlines()
    fence: tuple[str, int, str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(1)[0], len(m.group(1)), m.group(2).lower())
            else:
                for span in _INLINE.findall(line):
                    if span.startswith("hhemt "):
                        for args in _commands_in(_split_tokens(span)):
                            yield i + 1, span, args
        elif m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not m.group(2):
            fence = None
        elif fence[2] in _SHELL_LANGS:
            start, logical = i, line.strip()
            while logical.endswith("\\") and i + 1 < len(lines):
                i += 1
                logical = logical[:-1] + " " + lines[i].strip()
            for args in _commands_in(_split_tokens(logical)):
                yield start + 1, logical, args
        i += 1


def check_text(text: str, root: Node) -> list[tuple[int, str, str]]:
    """Problems in one page text, as (line number, command text, message)."""
    problems = []
    for lineno, command, args in iter_commands(text):
        message = check_command(args, root)
        if message:
            problems.append((lineno, command, message))
    return problems


def test_every_command_in_a_page_exists():
    root = command_tree()
    problems = []
    for page, text in sorted(tracked_pages(_REPO).items()):
        for lineno, command, message in check_text(text, root):
            problems.append(f"{page}:{lineno}: {message}: {command}")
    assert problems == [], "\n".join(problems)


@pytest.fixture(scope="module")
def tree() -> Node:
    return command_tree()


def test_tree_holds_the_commands_the_pages_use(tree):
    """The tree is not empty: a failed introspection would pass every page."""
    assert {"run", "reprocess", "publish", "reprex", "status"} <= set(tree.subcommands)
    assert {"emit", "check"} <= set(tree.subcommands["reprex"].subcommands)


def test_control_a_missing_command_is_refused(tree):
    """The command `hhemt cancel` does not exist until the cancel chunk lands; this
    control fails then, and the page text below passes once the command exists."""
    page = "```bash\nhhemt cancel --experiment DIR\n```\n"
    problems = check_text(page, tree)
    if "cancel" in tree.subcommands:
        assert problems == []
    else:
        assert [p[2] for p in problems] == ["`hhemt` has no command `cancel`"]


def test_control_a_missing_flag_on_a_continued_line_is_refused(tree):
    page = "```bash\nhhemt publish --experiment DIR \\\n    --no-such-flag x\n```\n"
    problems = check_text(page, tree)
    assert [(p[0], p[2]) for p in problems] == [(2, "`hhemt publish` has no option `--no-such-flag`")]


def test_control_an_inline_span_is_checked(tree):
    problems = check_text("Run `hhemt run --no-such-flag` first.\n", tree)
    assert [p[2] for p in problems] == ["`hhemt run` has no option `--no-such-flag`"]


def test_valid_commands_pass(tree):
    page = (
        "```bash\n"
        "hhemt run --system-config a.yaml --analysis-config b.yaml --status\n"
        "hhemt reprex emit --experiment DIR --cluster C \\\n    --container-def x.def\n"
        "hhemt --help\n"
        "```\n"
    )
    assert check_text(page, tree) == []


def test_placeholders_and_other_languages_are_not_checked(tree):
    page = "Use `hhemt {command} --help`.\n```yaml\nhhemt: cancel\n```\n```bash\nconda activate hhemt\n```\n"
    assert check_text(page, tree) == []


def test_pages_exercise_the_check(tree):
    """The scan reads commands from the real pages: the verbs below are named in them,
    so an extraction that finds nothing (a broken fence reader) turns this red."""
    verbs = set()
    for text in tracked_pages(_REPO).values():
        for _, _, args in iter_commands(text):
            if args and args[0] in tree.subcommands:
                verbs.add(args[0])
    assert {"run", "reprocess", "publish", "status"} <= verbs


def test_control_a_mutated_real_page_is_refused(tree):
    """Control on a real page: one flag of a command in `diagnosing-a-failed-run.md`
    given another name makes the check fail on that line."""
    page = "docs/how-to/diagnosing-a-failed-run.md"
    text = tracked_pages(_REPO)[page]
    assert "--status" in text
    assert check_text(text, tree) == []
    mutated = text.replace("--status", "--statuz", 1)
    assert [p[2] for p in check_text(mutated, tree)] == ["`hhemt run` has no option `--statuz`"]


def test_control_an_unknown_command_is_refused_in_every_state_of_the_tree(tree):
    """A command that exists in no state of the tree gives exactly one problem, so a
    checker that continues past an unknown command turns this red."""
    problems = check_text("```bash\nhhemt no-such-command --x\n```\n", tree)
    assert [p[2] for p in problems] == ["`hhemt` has no command `no-such-command`"]
