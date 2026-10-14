"""The published CLI reference lists exactly the commands the CLI registers.

The problem this file guards: `docs/reference/cli.md` is a hand-written page, and nothing ties
its verb tables to the Typer app. A verb added to the CLI without a row is invisible to every
reader of the docs site, and a row left behind after a verb is removed sends readers to a
command that does not exist. This test reads both lists and requires them to be the same set.

What counts as the page's verb list: the first backticked word of each row in every table whose
header is `| Command | What it does |` and that sits directly under a `## ` section. Tables under
a `### ` subsection describe a command group's own subcommands (for example `hhemt test
toolkit`'s `plan` and `chunk`) and are not top-level verbs; the exit-code table has a different
header and is not a verb list either.

Nothing here runs a command: the registered set is read from the Click command tree.
"""

from pathlib import Path

import typer

from hhemt.cli import app

_CLI_PAGE = Path(__file__).resolve().parent.parent / "docs" / "reference" / "cli.md"
_VERB_TABLE_HEADER = "| Command | What it does |"


def _documented_verbs(text: str) -> set[str]:
    """Return the first backticked word of every row in the page's top-level verb tables."""
    verbs: set[str] = set()
    under_h2 = False
    in_verb_table = False
    for line in text.splitlines():
        if line.startswith("### "):
            under_h2, in_verb_table = False, False
        elif line.startswith("## "):
            under_h2, in_verb_table = True, False
        elif line.startswith(_VERB_TABLE_HEADER):
            in_verb_table = under_h2
        elif not line.startswith("|"):
            in_verb_table = False
        elif in_verb_table and line.startswith("| `"):
            first_cell = line.split("|")[1]
            verbs.add(first_cell.split("`")[1].split()[0])
    return verbs


def test_reference_page_lists_exactly_the_registered_commands():
    """Property: the verb tables of `docs/reference/cli.md` name exactly the registered commands.

    Class: every top-level command and command group the Typer app registers, against every
    verb the reference page's top-level tables name. Why it is required: the page is the
    published description of the CLI, and the two lists drift silently in both directions --
    a verb with no row is undocumented, and a row with no verb documents a command that fails
    with "No such command". What kills it: registering a verb without adding its row, removing
    a verb and leaving its row, or a row whose first word is not the verb's name. A second
    correct implementation under which it still passes: one that parses the page with a
    markdown library rather than line by line, since only the two resulting sets are compared.
    """
    registered = set(typer.main.get_command(app).commands)
    documented = _documented_verbs(_CLI_PAGE.read_text(encoding="utf-8"))

    assert documented == registered, (
        f"registered but not documented: {sorted(registered - documented)}; "
        f"documented but not registered: {sorted(documented - registered)}"
    )


def test_the_table_reader_finds_rows_and_skips_subsection_and_exit_code_tables():
    """Property: the page reader counts top-level verb rows and nothing else.

    Class: the three table kinds the page carries -- a top-level verb table, a subsection
    table of a group's subcommands, and the exit-code table. Why it is required: the set
    comparison above is only as good as the reader; a reader that found nothing would compare
    an empty set, and one that read the subsection or exit-code tables would demand verbs named
    `plan` or `0`. What kills it: dropping the `## ` restriction, the header check, or the
    first-word extraction. A second correct implementation under which it still passes: any
    reader returning `{"run", "delete"}` for this sample.
    """
    sample = "\n".join(
        [
            "## Running",
            "",
            "| Command | What it does |",
            "|---|---|",
            "| `run` | Runs. |",
            "| `delete` | Deletes. |",
            "",
            "### `hhemt test toolkit`",
            "",
            "| Command | What it does |",
            "|---|---|",
            "| `plan --toolkit {path}` | Plans. |",
            "",
            "## Exit codes",
            "",
            "| Code | Meaning |",
            "|---|---|",
            "| `0` | Success |",
        ]
    )
    assert _documented_verbs(sample) == {"run", "delete"}
