"""The unit-spelling change altered spelling and nothing else, and its layout rows match the files they excuse.

Each expectation is computed from git history and the working files, not from a table written beside the change:
a word-level comparison of every changed line with its earlier text, a syntax-tree comparison of every changed
Python module with spellings normalised, the layout-gate signatures against the files' own digests, and a census of
the tracked tree. The two spellings are built from fragments, so this file holds no literal of the unwanted one.
No solver runs.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
#: The commit the change starts from; the working tree is compared with it.
BASE = "5a5fa54f"

_STEM = "m" + "e" + "tr" + "e"
_OLD = re.compile(_STEM + "(s?)", re.IGNORECASE)

#: Changed lines that are not a spelling swap, by file, with the number of such lines: the field description gains
#: its unit, the comment about that field is rewritten, the caption rule is rephrased, and the layout rows are added.
EXPECTED_NON_SPELLING = {
    "src/hhemt/config/analysis.py": 1,
    "src/hhemt/cf_conventions.py": 1,
    "src/hhemt/figure_caption.py": 1,
    "_layout_relevant_files.yaml": 1,
}
#: Two earlier reviewer test files were repaired afterwards, by agreement of a second specialist. Their edits are
#: allowed EXACTLY: each entry is (number of changed lines, sha256 of those lines), so one more changed line in either
#: file, or a different line, fails the first test below.
REPAIRED_TEST_FILES = {
    "tests/test_review_clean_slate.py": (22, "37fdb195ee8b99b9c81dc95fbce415acb42416049f0d7c3ef1c2e725b8bba086"),
    "tests/test_review_migration_followups.py": (
        11,
        "437932e34f563bb6110fbf5d0a4603315022b77419e5f5278a18a22b477331e2",
    ),
}
LAYOUT_PATHS = (
    "src/hhemt/config/analysis.py",
    "src/hhemt/cf_conventions.py",
    "src/hhemt/process_simulation.py",
    "src/hhemt/version_migration/versions/V0028__flood_depth_and_unit_discharge_names.py",
)

needs_base = pytest.mark.skipif(
    subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", BASE], capture_output=True).returncode != 0,
    reason="the starting commit is not in this clone",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _respell(text: str) -> str:
    """The earlier spelling rewritten to the wanted one, keeping capitalisation and the plural."""
    new = "m" + "e" + "ter"
    return _OLD.sub(lambda m: (new.capitalize() if m.group(0)[0].isupper() else new) + m.group(1), text)


def _hunks(diff_text: str) -> list[tuple[str, list[str], list[str]]]:
    path, out = "", []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[6:]
        elif line.startswith("@@"):
            out.append((path, [], []))
        elif line.startswith("-") and not line.startswith("---") and out:
            out[-1][1].append(line[1:])
        elif line.startswith("+") and not line.startswith("+++") and out:
            out[-1][2].append(line[1:])
    return out


def non_spelling_changes(diff_text: str) -> dict[str, int]:
    """File -> number of changed lines (or unequal hunks) that are not the old line with only the spelling swapped."""
    counts: dict[str, int] = {}
    for path, old, new in _hunks(diff_text):
        if len(old) != len(new):
            counts[path] = counts.get(path, 0) + 1
            continue
        for o, n in zip(old, new, strict=True):
            if _respell(o) != n:
                counts[path] = counts.get(path, 0) + 1
    return counts


def edit_digest(diff_text: str, path: str) -> tuple[int, str]:
    """(number of changed lines, sha256 of those lines) of ``path`` in a ``git diff -U0`` output."""
    current, lines = "", []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current = line[6:]
        elif current == path and line[:1] in "+-" and not line.startswith(("+++", "---")):
            lines.append(line)
    return len(lines), hashlib.sha256("\n".join(lines).encode()).hexdigest()


def unexpected_non_spelling(diff_text: str) -> dict[str, int]:
    """``non_spelling_changes`` without the two repaired test files, whose edits are checked by digest."""
    return {p: n for p, n in non_spelling_changes(diff_text).items() if p not in REPAIRED_TEST_FILES}


@needs_base
def test_every_changed_line_is_the_earlier_line_with_the_spelling_swapped_except_the_listed_ones():
    """Property: against the starting commit, each changed line equals its earlier line with the spelling swapped,
    except for exactly one line (or one unequal hunk) in each of four named files.

    Class: all hunks of ``git diff -U0`` from the starting commit, paired line by line within equal-sized hunks.
    Kills it: a word added or dropped while a spelling was fixed; a number or a name changed on the same line.
    Must-fire control: appending one word to a spelling-only line adds a report.
    """
    diff_text = _git("diff", "-U0", BASE)
    found = unexpected_non_spelling(diff_text)
    assert found == EXPECTED_NON_SPELLING, found
    for path, expected in REPAIRED_TEST_FILES.items():
        assert edit_digest(diff_text, path) == expected, path
    lines = diff_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("+") and "per meter of flow width" in line:
            lines[i] = line + " extra"
            break
    assert sum(unexpected_non_spelling("\n".join(lines)).values()) == sum(EXPECTED_NON_SPELLING.values()) + 1
    for path, expected in REPAIRED_TEST_FILES.items():
        planted = diff_text.splitlines()
        planted.insert(planted.index(f"+++ b/{path}") + 1, "+one more changed line")
        assert edit_digest("\n".join(planted), path) != expected, path


def _dump(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = _respell(node.value)
        if isinstance(node, ast.FunctionDef):
            node.name = _respell(node.name)
    return ast.dump(tree)


@needs_base
def test_no_changed_python_module_moved_a_statement_beyond_the_field_description():
    """Property: for every Python file changed since the starting commit, the syntax tree without docstrings and with
    string constants and function names respelled equals the earlier one, except for ``config/analysis.py``, whose
    difference is the unit added to the ``manhole_diameter`` description and nothing else, and the two repaired
    test files, whose edits the first test pins by digest.

    Class: every modified ``.py`` file in ``git diff --name-only`` from the starting commit.
    Kills it: an executable statement edited together with a comment; a function given another name; a changed default.
    Must-fire control: a planted ``return not`` is reported.
    """
    changed = [p for p in _git("diff", "--name-only", "--diff-filter=M", BASE).split() if p.endswith(".py")]
    assert len(changed) >= 15, changed
    differing = [p for p in changed if _dump(_git("show", f"{BASE}:{p}")) != _dump((REPO / p).read_text())]
    assert differing == sorted(["src/hhemt/config/analysis.py", *REPAIRED_TEST_FILES]), differing
    work = (REPO / "src/hhemt/config/analysis.py").read_text()
    assert work.count("interaction nodes, in meters.") == 1
    earlier = _git("show", f"{BASE}:src/hhemt/config/analysis.py")
    assert _dump(earlier) == _dump(work.replace("interaction nodes, in meters.", "interaction nodes."))
    sample = (REPO / changed[0]).read_text()
    assert _dump(sample.replace("return ", "return not ", 1)) != _dump(sample)


def test_each_layout_row_added_carries_the_digest_of_the_file_it_excuses():
    """Property: the newest row for each of the four layout-relevant files carries that file's current sha256.

    Class: ``_layout_relevant_files.yaml``'s ``non_breaking_allowlist``, last row per path, against the file bytes.
    Why required: a row stamped before the file's last edit does not excuse it, and the gate then reports the file.
    Kills it: an edit to one of the four files after its row was stamped.
    Must-fire control: the digest of the same file with one byte appended differs from the recorded one.
    """
    rows = yaml.safe_load((REPO / "_layout_relevant_files.yaml").read_text())["non_breaking_allowlist"]
    last = {row["path"]: row.get("layout_signature") for row in rows if isinstance(row, dict)}
    for path in LAYOUT_PATHS:
        data = (REPO / path).read_bytes()
        assert last[path] == hashlib.sha256(data).hexdigest(), path
        assert hashlib.sha256(data + b"\n").hexdigest() != last[path]


def test_the_stored_long_name_text_is_the_same_in_the_live_table_and_the_frozen_copy():
    """Property: the three unit-discharge ``long_name`` strings read the same in the live table and in the frozen
    table of the migration, and none holds the unwanted spelling.

    Class: the three variables, read from the two tables at run time.
    Kills it: one table changed and the other left behind.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP
    from hhemt.version_migration.versions import V0028__flood_depth_and_unit_discharge_names as migration

    names = ("max_unit_discharge", "unit_discharge_x", "unit_discharge_y")
    for name in names:
        live = _CF_VARIABLE_MAP[name]["long_name"]
        frozen = migration._ATTRS[name]["long_name"]
        assert live == frozen, name
        assert "per meter of flow width" in live and not _OLD.search(live), name
