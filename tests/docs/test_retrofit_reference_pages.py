"""Reference-page statements about code, pinned to the code, for the corrections of the reference style pass.

Two corrections replaced statements that disagreed with the code: the renderer table of
`docs/reference/reporting-sets.md` listed five renderers for the `combined` set where the
registry holds six, and the intro of the cleanup section of `docs/reference/cli.md` said the
commands operate on bookkeeping only, where `delete` and `reset` remove a whole analysis tree
and the `cleanup-*` commands delete only with `--apply`. Each test locates its statement by a
stable anchor (a table row's first cell, a command name), derives the matching fact from the
code, and compares sets. No test compares a sentence of prose, writes down a count or runs a
solver; a control test feeds each comparison a mutated input and expects the refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer

_REPO = Path(__file__).resolve().parents[2]
_SPAN = re.compile(r"`([^`\n]+)`")


def _page(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def renderer_table_rows(page_text: str) -> dict[str, set[str]]:
    """The renderer table of `reporting-sets.md` as `{set name: {renderer key}}`.

    A row is a table line whose first cell is a single code span; its second cell holds the
    renderer keys as code spans. Rows of other tables on the page have another second-cell shape
    and are kept out by requiring a registered-looking key (lower-case words joined by underscores).
    """
    rows: dict[str, set[str]] = {}
    for line in page_text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not re.fullmatch(r"`[a-z_]+`", cells[0]):
            continue
        keys = set(_SPAN.findall(cells[1]))
        if keys and all(re.fullmatch(r"[a-z_]+", k) for k in keys):
            rows[cells[0].strip("`")] = keys
    return rows


def registry_keys() -> dict[str, set[str]]:
    """`{set name: {builder_key}}` from the reporting-set registry."""
    from hhemt.report_renderers import _reporting_sets as rs

    return {name: {sel.builder_key for sel in s.renderer_selection} for name, s in rs.REPORTING_SETS.items()}


def check_renderer_table(rows: dict[str, set[str]], registry: dict[str, set[str]]) -> list[str]:
    """Problems: a set whose page row differs from the registry by a renderer key, or a set with no row."""
    problems = []
    for name, keys in registry.items():
        if name not in rows:
            problems.append(f"set {name} has no row on the page")
            continue
        for key in sorted(keys - rows[name]):
            problems.append(f"{name}: {key} is registered and the page row omits it")
        for key in sorted(rows[name] - keys):
            problems.append(f"{name}: {key} is on the page row and not registered")
    return problems


def test_renderer_table_rows_match_the_registry():
    """Property: each reporting set's row lists exactly the renderer keys the registry holds for it.

    Class: every set in `REPORTING_SETS`, `combined` included. Class NEW CAPABILITY; the plausible
    wrong state is the page before this correction, whose `combined` row omitted
    `cross_solver_comparison`; red there (the control below models it).
    Why required: a reader chooses renderers from this table, and a registered renderer missing
    from it is invisible.
    What kills it: a renderer added to or removed from a set in the registry without the row.
    A second correct implementation: parse the table with a Markdown library and compare sets.
    """
    assert check_renderer_table(renderer_table_rows(_page("docs/reference/reporting-sets.md")), registry_keys()) == []


def test_control_a_dropped_renderer_key_is_refused():
    """The comparison names a renderer the page row omits: drop one `combined` key from the parsed row."""
    rows = renderer_table_rows(_page("docs/reference/reporting-sets.md"))
    rows["combined"] = rows["combined"] - {"cross_solver_comparison"}
    problems = check_renderer_table(rows, registry_keys())
    assert problems == ["combined: cross_solver_comparison is registered and the page row omits it"]


def test_control_an_unregistered_key_or_a_missing_row_is_refused():
    """The comparison refuses a page key the registry lacks and a registered set with no row."""
    rows = renderer_table_rows(_page("docs/reference/reporting-sets.md"))
    rows["default"] = rows["default"] | {"not_a_renderer"}
    del rows["combined"]
    problems = check_renderer_table(rows, registry_keys())
    assert "default: not_a_renderer is on the page row and not registered" in problems
    assert "set combined has no row on the page" in problems


def _cleanup_options() -> dict[str, set[str]]:
    from hhemt.cli import app

    cmd = typer.main.get_command(app)
    return {
        name: {o for p in sub.params for o in (*p.opts, *p.secondary_opts)}
        for name, sub in cmd.commands.items()
        if name.startswith("cleanup-") or name in ("delete", "reset")
    }


def check_cleanup_intro(page_text: str, options: dict[str, set[str]]) -> list[str]:
    """Problems: the intro names `--apply` or `--dry-run` for commands that lack the flag."""
    start = page_text.index("## Cleanup")
    intro = page_text[start : page_text.index("| Command |", start)]
    problems = []
    if "`--apply`" in intro:
        problems += [
            f"{n} lacks --apply" for n, o in sorted(options.items()) if n.startswith("cleanup-") and "--apply" not in o
        ]
    if "`--dry-run`" in intro:
        problems += [f"{n} lacks --dry-run" for n in ("delete", "reset") if "--dry-run" not in options.get(n, set())]
    return problems


def test_cleanup_intro_flags_exist_on_the_commands_it_names():
    """Property: the cleanup intro's `--apply` is on each `cleanup-*` command and its `--dry-run` on `delete`, `reset`.

    Class: every `cleanup-*` command, `delete` and `reset`. Class NEW CAPABILITY; the plausible
    wrong state is a command that drops the flag the intro promises; red under the control below.
    Why required: the intro tells a reader every command here inspects before it deletes.
    What kills it: removing `--apply` from a `cleanup-*` command or `--dry-run` from `delete`/`reset`.
    A second correct implementation: invoke each command with `--help` and read its output.
    """
    assert check_cleanup_intro(_page("docs/reference/cli.md"), _cleanup_options()) == []


def test_control_a_command_without_the_flag_is_refused():
    """Remove `--apply` from one `cleanup-*` command and `--dry-run` from `reset`: both are reported."""
    options = _cleanup_options()
    options["cleanup-stale-metadata"] = options["cleanup-stale-metadata"] - {"--apply"}
    options["reset"] = options["reset"] - {"--dry-run"}
    problems = check_cleanup_intro(_page("docs/reference/cli.md"), options)
    assert problems == ["cleanup-stale-metadata lacks --apply", "reset lacks --dry-run"]


@pytest.mark.parametrize("name", ["delete", "reset", "cleanup-orphans"])
def test_the_cleanup_options_include_the_commands_the_intro_covers(name):
    """The introspection finds the commands the intro covers, so the checks above are not vacuous."""
    assert name in _cleanup_options()
