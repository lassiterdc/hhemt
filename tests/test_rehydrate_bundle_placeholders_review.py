"""Held tests for ``rehydrate_placeholders``, SOLVER-FREE: every arm writes under ``tmp_path``.

Each test is green on the current tree and has a named mutation that turns it red.
Producer-looking values are built from fragments so this file is not itself a guard hit.
"""

from __future__ import annotations

import getpass
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt
import hhemt.experiment_definition as ed
from hhemt.bundle import _placeholders as P
from hhemt.exceptions import ConfigurationError
from hhemt.report_plot_ids import event_labels_from_status
from hhemt.report_renderers.metadata import _read_scenario_status

LOGIN = "carol" + "ine"
ANALYSIS_DIR = "/scr" + "atch/" + LOGIN + "/runs/a1"


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_every_consume_side_scenario_directory_reader_is_prefix_insensitive(tmp_path):
    """Property: the two consume-side readers of ``scenario_directory`` that a first census of readers missed
    (``report_plot_ids.event_labels_from_status`` and ``metadata._read_scenario_status``) return the same
    result from a prefix-rewritten ``scenario_status.csv`` as from the original, with no rehydration.

    Class: every reader of the ``scenario_directory`` cell outside source-side validation. The census
    named ONE function; the repository holds three read sites in two files, so "no reader needs
    rehydration" is only true if each of the other two also takes the basename.
    Why required: a reader that opened the cell as a path would see a token and need rehydration, which
    would force the rehydrator to be wired into that reader.
    What kills it: a rewrite that swallows the final segment (here: a table whose value is the whole
    scenario directory) changes both results, and a reader that joins a child onto the cell changes them.
    Second correct implementation: a reader that keys on the last two segments would still pass only if
    the rewrite keeps two, so the test constrains the rewrite to a prefix replacement.
    """
    rows = ["scenario_directory,event_iloc,event_label,model_type,sa_id"]
    rows += [f"{ANALYSIS_DIR}/sims/event_{c},{i},storm {i},tritonswmm,0" for i, c in enumerate("ab")]
    original = "\n".join(rows) + "\n"
    table = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR]})
    rewritten = table.apply(original.encode()).decode()
    assert P.TOKEN_ANALYSIS_DIR in rewritten and rewritten != original

    orig_dir = _write(tmp_path / "o", "scenario_status.csv", original).parent
    new_dir = _write(tmp_path / "n", "scenario_status.csv", rewritten).parent

    expected_labels = event_labels_from_status(orig_dir)
    assert expected_labels == {"event_a": "storm 0", "event_b": "storm 1"}
    assert event_labels_from_status(new_dir) == expected_labels

    orig_rows, _ = _read_scenario_status(orig_dir)
    new_rows, _ = _read_scenario_status(new_dir)
    assert {k for k in orig_rows if "event_a" in k} and set(new_rows) == set(orig_rows)

    # Must-fire control (varies the property, a rewrite that swallows the final segment).
    whole = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [f"{ANALYSIS_DIR}/sims/event_a"]})
    swallowed = _write(tmp_path / "s", "scenario_status.csv", whole.apply(original.encode()).decode()).parent
    assert event_labels_from_status(swallowed) != expected_labels


def test_environment_references_and_tilde_are_not_expanded(tmp_path, monkeypatch):
    """Property: ``${VAR}``, ``$VAR`` and a leading ``~`` for a variable that IS set are byte-identical in the copy.

    Class: shell and environment references in a bundled Snakefile. The sibling test in
    ``test_rehydrate_bundle_placeholders.py`` uses ``${CONDA_PREFIX}``, which only discriminates an
    ``os.path.expandvars`` implementation when that variable happens to be set in the run's environment;
    this test sets its own variable, so it discriminates always.
    Why required: ``Snakefile.source`` carries shell references that must reach the consumer unexpanded.
    What kills it: ``os.path.expandvars`` or ``expanduser`` applied to the text.
    Second correct implementation: ``str.replace`` per closed token on decoded text.
    """
    monkeypatch.setenv("HHEMT_REVIEW_VAR", "/leak")
    body = "a=${HHEMT_REVIEW_VAR} b=$HHEMT_REVIEW_VAR c=~/x d={hhemt_home}\n"
    member = _write(tmp_path / "m", "Snakefile.source", body)
    out = ed.rehydrate_placeholders(member, {P.TOKEN_HOME: "/h"}, dest_dir=tmp_path / "out")
    assert out.read_text(encoding="utf-8") == "a=${HHEMT_REVIEW_VAR} b=$HHEMT_REVIEW_VAR c=~/x d=/h\n"


def test_a_value_with_backslash_escapes_is_written_literally(tmp_path):
    """Property: a supplied value holding backslash sequences that ``re.sub`` treats as escapes
    (``\\1``, ``\\g<0>``, ``\\U``) is written byte-for-byte.

    Class: consumer values containing a backslash (a Windows-style home, or any path with one).
    Why required: a replacement STRING passed to ``re.sub`` raises ``re.error`` or expands the escape, so a
    correct rehydrator must use a function replacement or equivalent.
    What kills it: ``re.sub(token, value, text)`` or ``pattern.sub(value, ...)`` with the value as repl.
    Second correct implementation: ``bytes.replace`` per token, or the shipped lambda replacement.
    """
    value = "a\\1\\g<0>b"
    member = _write(tmp_path / "m", "x.txt", "h={hhemt_home} again={hhemt_home}\n")
    out = ed.rehydrate_placeholders(member, {P.TOKEN_HOME: value}, dest_dir=tmp_path / "out")
    assert out.read_text(encoding="utf-8") == f"h={value} again={value}\n"


def _lookalike_src_layout(root: Path, monkeypatch) -> tuple[Path, Path]:
    """Point ``hhemt.__file__`` at ``{base}/mysrc/hhemt``: the parent name ENDS in ``src`` but is not ``src``."""
    base = (root / "layout").resolve()
    package = base / "mysrc" / "hhemt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(hhemt, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    return base, package


def test_a_parent_directory_that_only_ends_in_src_is_not_a_source_checkout(tmp_path, monkeypatch):
    """Property: with the package at ``{base}/mysrc/hhemt`` there is no checkout root, on BOTH sides: the table
    writes no checkout token for ``{base}/tests/x`` (it stays as it was), and the rehydrator refuses
    ``{hhemt_toolkit_dir}`` until a value is supplied while ``{hhemt_package_dir}`` resolves.

    Class: package locations whose parent directory name is near, but not equal to, ``src``. The sibling tests
    use ``site`` as the non-source layout, which a parent-name test of any looseness (``endswith``, ``in``)
    also treats as "not src".
    Why required: a loose test makes ``{base}`` a checkout root, so rehydration would silently return a
    directory that is not a checkout (emit side: ``{base}/tests/x`` is rewritten to a token whose default
    points at the wrong directory).
    What kills it: ``parent.name.endswith("src")`` in ``_own_checkout_root`` (rehydrator half) or in
    ``build_placeholder_table`` (emit half).
    Second correct implementation: checking for a ``pyproject.toml`` beside ``src/`` instead of the name
    would also treat this layout as not-a-checkout, since the temp layout has neither.
    """
    from hhemt.bundle import _placeholders as P

    base, package = _lookalike_src_layout(tmp_path, monkeypatch)
    text = f"{package}/report_templates/x\n{base}/tests/x\n"
    rewritten = P.build_placeholder_table(SimpleNamespace()).apply(text.encode()).decode()
    assert rewritten == f"{P.TOKEN_PACKAGE_DIR}/report_templates/x\n{base}/tests/x\n"

    toolkit_member = _write(tmp_path / "t", "x.txt", f"{P.TOKEN_TOOLKIT_DIR}/a\n")
    with pytest.raises(ConfigurationError, match="hhemt_toolkit_dir"):
        ed.rehydrate_placeholders(toolkit_member, dest_dir=tmp_path / "o1")
    package_member = _write(tmp_path / "p", "x.txt", f"{P.TOKEN_PACKAGE_DIR}/a\n")
    assert ed.rehydrate_placeholders(package_member, dest_dir=tmp_path / "o2").read_text() == f"{package}/a\n"
