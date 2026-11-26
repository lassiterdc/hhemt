"""Reviewer checks for the file-list selection: the verb, the list reading, the tail rule and the scope binding.

These probe what the builder's own tests leave open: the ``plan`` verb's flag reaching the planner, how the list file
is read (line endings, repeats, an empty list, a missing list), the tail opt-out and a tail named by file, a tracked
directory or a path that is not repository-relative, and a manifest that carries no selection keys when the flag is
absent. Nothing here runs a solver.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hhemt.suite._runner as R
from hhemt.suite import policy as POL
from hhemt.suite._cli import toolkit_app
from tests.test_suite_tier_selection import _env, _git, _project


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = _project(tmp_path)
    (repo / "tests" / "sub").mkdir()
    (repo / "tests" / "sub" / "test_nested.py").write_text("def test_n():\n    assert True\n")
    _git(repo, "add", "tests/sub/test_nested.py")
    _git(repo, "commit", "-q", "-m", "nested")
    return repo


def test_the_plan_verb_passes_select_files_to_the_planner_and_omits_it_when_absent(tmp_path, monkeypatch):
    """The verb is the operator's entry; its flag must reach ``main`` as ``--select-files {path}`` and only then."""
    seen: list[list[str]] = []
    monkeypatch.setattr(R, "main", lambda argv=None: seen.append(list(argv)) or 0)
    runner = CliRunner()
    base = ["plan", "--toolkit", str(tmp_path), "--runs-root", str(tmp_path / "runs")]
    given = runner.invoke(toolkit_app, [*base, "--select-files", str(tmp_path / "sel.txt")])
    plain = runner.invoke(toolkit_app, base)
    assert given.exit_code == 0 and plain.exit_code == 0, (given.output, plain.output)
    assert seen[0][seen[0].index("--select-files") + 1] == str(tmp_path / "sel.txt")
    assert "--select-files" not in seen[1]


def test_the_plan_verb_help_documents_select_files():
    out = CliRunner().invoke(toolkit_app, ["plan", "--help"]).output
    assert "--select-files" in out and "selection" in out


def test_reading_a_list_sorts_dedupes_tolerates_crlf_and_digests_the_bytes_as_given(tmp_path):
    raw = b"tests/test_warm.py\r\n\r\ntests/test_stand.py\r\ntests/test_warm.py\r\n  \r\n"
    path = tmp_path / "sel.txt"
    path.write_bytes(raw)
    paths, digest = R._read_select_files(path)
    assert paths == ["tests/test_stand.py", "tests/test_warm.py"]
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("body", ["", "\n\n  \n"])
def test_an_empty_list_is_refused_naming_the_list(tmp_path, body):
    path = tmp_path / "sel.txt"
    path.write_text(body)
    with pytest.raises(SystemExit) as excinfo:
        R._read_select_files(path)
    assert str(path) in str(excinfo.value)


def test_a_missing_list_file_is_refused_naming_the_path(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(SystemExit) as excinfo:
        R._read_select_files(missing)
    assert str(missing) in str(excinfo.value)


@pytest.mark.parametrize(
    "bad",
    ["tests", "tests/", "./tests/test_warm.py", "tests//test_warm.py", "tests/test_warm.pyc", "TESTS/test_warm.py"],
)
def test_a_directory_or_a_path_that_is_not_the_tracked_spelling_is_refused(repo, tmp_path, monkeypatch, bad):
    monkeypatch.setenv(POL.TAIL_NODE_IDS_FILE_ENV, str(_empty(tmp_path)))
    list_path = tmp_path / "sel.txt"
    list_path.write_text(f"tests/test_warm.py\n{bad}\n")
    with pytest.raises(SystemExit) as excinfo:
        R._validate_select_files(repo, list_path, ["tests/test_warm.py", bad], "all")
    assert bad in str(excinfo.value) and "is not a tracked test file" in str(excinfo.value)


def test_a_tracked_test_file_in_a_subdirectory_is_accepted(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(POL.TAIL_NODE_IDS_FILE_ENV, str(_empty(tmp_path)))
    list_path = tmp_path / "sel.txt"
    list_path.write_text("tests/sub/test_nested.py\n")
    R._validate_select_files(repo, list_path, ["tests/sub/test_nested.py"], "all")


def test_an_empty_tail_opt_out_lets_a_list_omit_the_tail_file(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(POL.TAIL_NODE_IDS_FILE_ENV, str(_empty(tmp_path)))
    list_path = tmp_path / "sel.txt"
    list_path.write_text("tests/test_warm.py\n")
    R._validate_select_files(repo, list_path, ["tests/test_warm.py"], "all")


def test_a_tail_named_by_file_requires_the_files_holding_its_ids_and_names_them_all(repo, tmp_path, monkeypatch):
    tail = tmp_path / "tail.txt"
    tail.write_text("# comment\ntests/test_stand.py::test_solver\ntests/sub/test_nested.py::test_n\n")
    monkeypatch.setenv(POL.TAIL_NODE_IDS_FILE_ENV, str(tail))
    list_path = tmp_path / "sel.txt"
    list_path.write_text("tests/test_warm.py\n")
    with pytest.raises(SystemExit) as excinfo:
        R._validate_select_files(repo, list_path, ["tests/test_warm.py"], "all")
    message = str(excinfo.value)
    assert "tests/sub/test_nested.py" in message and "tests/sub/test_nested.py::test_n" in message, message
    # With the first file listed the second is still owed: the refusal is per file, not once for the first.
    with pytest.raises(SystemExit) as excinfo:
        R._validate_select_files(repo, list_path, ["tests/test_warm.py", "tests/sub/test_nested.py"], "all")
    assert "tests/test_stand.py::test_solver" in str(excinfo.value)
    R._validate_select_files(
        repo, list_path, ["tests/test_warm.py", "tests/test_stand.py", "tests/sub/test_nested.py"], "all"
    )


def test_a_plan_without_the_flag_writes_no_selection_keys_and_a_selection_manifest_keeps_the_full_tail_shape(tmp_path):
    from tests.test_suite_tier_selection import _plan

    manifest = _plan(tmp_path)["manifest"]
    assert "select_files" not in manifest and "select_files_sha256" not in manifest


def test_aggregating_a_selection_with_the_default_scope_binds_selection_without_a_scope_problem(tmp_path):
    """The default ``--scope array`` is the verdict job's call; binding must be silent there and still bind."""
    from tests.test_suite_file_list import _list, _plan_selection

    _project(tmp_path)
    run_dir = _plan_selection(tmp_path, _list(tmp_path, "tests/test_warm.py"))["run_dir"]
    cmd = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--allow-not-green"]
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("verdict="))
    assert " scope=selection " in line, done.stdout[-1500:]
    assert not any("select_files" in ln for ln in done.stdout.splitlines() if ln.startswith("  problem:"))
    summary = json.loads((run_dir / "summary.selection.json").read_text())
    assert summary["select_files"] == ["tests/test_warm.py"]


def _empty(tmp_path: Path) -> Path:
    path = tmp_path / "empty_tail.txt"
    path.write_text("")
    return path
