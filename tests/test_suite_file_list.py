"""A plan can cover exactly a list of test files, and such a selection run can never pass for a suite run.

``--select-files {list}`` on the suite planner takes a file of newline-separated repository-relative test-file paths.
The planner refuses a list naming a path that is not a tracked ``test_*.py`` file, refuses the flag beside
``--select-tier``, and refuses a list that leaves out the file holding the tail node ids; it then collects only the
listed files and records ``select_files`` and ``select_files_sha256`` in the manifest. The aggregator binds such a run
to ``scope=selection`` (never ``union``, never ledgered) and writes ``summary.selection.json`` and
``summary.selection.md`` in place of ``summary.json`` and ``summary.md``. The plans below run over a tiny stand-in
project: pytest only collects, no solver runs, and nothing here needs a cluster.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.suite._runner as R
from hhemt.suite import partition as P
from hhemt.suite import policy as POL
from tests.test_suite_tier_selection import _env, _git, _project

_TAIL_FILE = "tests/test_synth_00_compile_models.py"


def _list(tmp_path: Path, *lines: str, name: str = "selection.txt") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _refusal(repo: Path, list_path: Path, *extra: str) -> str:
    """Run the planner's ``main`` in process and return the text of the refusal it raises."""
    argv = ["--toolkit", str(repo), "--runs-root", str(repo.parent / "runs"), "--select-files", str(list_path), *extra]
    with pytest.raises(SystemExit) as excinfo:
        R.main(argv)
    return str(excinfo.value)


def _plan_selection(tmp_path: Path, list_path: Path) -> dict:
    """Run the real planning driver over the stand-in project with ``--select-files``; return the run it minted."""
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    runs.mkdir()
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable]
    cmd += ["--runs-root", str(runs), "--warm-target", "tests/test_warm.py", "--declare-non-compile-bearing"]
    cmd += ["--select-files", str(list_path)]
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    run_dir = next(runs.iterdir())
    return {"run_dir": run_dir, "manifest": json.loads((run_dir / "manifest.json").read_text()), "stdout": done.stdout}


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    """The stand-in project plus an untracked test file and a tracked file whose name is not ``test_*.py``."""
    repo = _project(tmp_path)
    (repo / "tests" / "helper_not_a_test.py").write_text("VALUE = 1\n")
    _git(repo, "add", "tests/helper_not_a_test.py")
    _git(repo, "commit", "-q", "-m", "helper")
    (repo / "tests" / "test_untracked.py").write_text("def test_u():\n    assert True\n")
    return repo


@pytest.mark.parametrize(
    "bad",
    ["tests/test_untracked.py", "tests/helper_not_a_test.py", "tests/test_missing.py"],
    ids=["untracked", "not-test-named", "missing"],
)
def test_a_list_naming_an_untracked_non_test_or_missing_path_is_refused_naming_that_path_and_the_list(
    stand_in, tmp_path, monkeypatch, bad
):
    """(a) Input class: one listed path that is untracked, tracked but not ``test_*.py``, or absent.

    Required so a silently dropped path never becomes a test that never ran. Killed by a planner that skips an
    unknown path (no refusal) or one that refuses without naming the path and the list file. The tail opt-out is set
    so the refusal is the path rule and not the tail rule.
    """
    for key, value in _env(tmp_path).items():
        if key.startswith("HHEMT_"):
            monkeypatch.setenv(key, value)
    list_path = _list(tmp_path, "tests/test_warm.py", bad)
    message = _refusal(stand_in, list_path)
    assert message.startswith("refusing to plan:"), message
    assert f"{bad} in {list_path}" in message and "is not a tracked test file" in message, message


def test_select_files_with_a_tier_selection_is_refused_naming_both_flags(stand_in, tmp_path):
    """(b) Input class: a valid list beside ``--select-tier compile_tier``.

    Required so a plan covers a tier or a file list and never an unstated intersection. Killed by a planner that
    accepts both, and by one that refuses without naming both flags.
    """
    list_path = _list(tmp_path, "tests/test_warm.py")
    message = _refusal(stand_in, list_path, "--select-tier", "compile_tier")
    assert "--select-files" in message and "--select-tier" in message and "compile_tier" in message, message


def test_the_manifest_collects_exactly_the_listed_files_and_records_the_list_and_its_digest(tmp_path):
    """(c) Input class: a stand-in project of two test files plus the warm file; the list names all three.

    Required so the plan covers the listed files and the manifest ties the run to the exact list (blank lines and
    an unsorted order in the file, a sorted list and the bytes' sha256 in the manifest). Killed by a planner that
    drops the manifest keys or that digests the parsed list instead of the file's bytes.
    """
    _project(tmp_path)
    list_path = _list(tmp_path, "tests/test_warm.py", "", "tests/test_stand.py", "")
    # The next test is the control: a one-file list collects that file's ids only.
    out = _plan_selection(tmp_path, list_path)
    manifest = out["manifest"]
    assert sorted(manifest["collected"]) == [
        "tests/test_stand.py::test_plain",
        "tests/test_stand.py::test_solver",
        "tests/test_warm.py::test_warm",
    ]
    assert manifest["select_files"] == ["tests/test_stand.py", "tests/test_warm.py"]
    assert manifest["select_files_sha256"] == hashlib.sha256(list_path.read_bytes()).hexdigest()


def test_a_one_file_list_collects_only_that_files_node_ids(tmp_path):
    """(c, control) Input class: the list names only ``tests/test_warm.py``.

    Required so the collected ids are the listed file's and not the project's: the other test file's ids must be
    absent. This is the must-be-nonzero control for the previous test's all-files list.
    """
    _project(tmp_path)
    list_path = _list(tmp_path, "tests/test_warm.py")
    manifest = _plan_selection(tmp_path, list_path)["manifest"]
    assert manifest["collected"] == ["tests/test_warm.py::test_warm"]
    assert manifest["select_files"] == ["tests/test_warm.py"]


def test_a_selection_aggregates_to_scope_selection_with_its_own_summary_files_and_no_ledger_line(tmp_path):
    """(d) Input class: a run planned with ``--select-files`` and aggregated with the caller asking for ``union``.

    Required so a file-list green never prints a whole-suite label, never counts as a whole-suite ledger entry, and
    never lands in ``summary.json``, which the triage source rule would pick up as the newest suite run. Killed by an
    aggregator that ignores ``select_files`` (``scope=array`` or ``union``, ``summary.json``, a ledger line).
    """
    _project(tmp_path)
    list_path = _list(tmp_path, "tests/test_warm.py")
    run_dir = _plan_selection(tmp_path, list_path)["run_dir"]
    cmd = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--scope", "union"]
    cmd.append("--allow-not-green")
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("verdict="))
    assert " scope=selection " in line, done.stdout[-1500:]
    problems = [ln for ln in done.stdout.splitlines() if ln.startswith("  problem:")]
    assert any("select_files" in ln and "scope=union" in ln for ln in problems), done.stdout[-1500:]
    assert "ledger=skipped" in done.stdout
    assert not list(run_dir.parent.glob("_ledger*")) and not list(run_dir.parent.glob("*.jsonl"))
    assert (run_dir / "summary.selection.json").is_file() and (run_dir / "summary.selection.md").is_file()
    assert not (run_dir / "summary.json").exists() and not (run_dir / "summary.md").exists()
    summary = json.loads((run_dir / "summary.selection.json").read_text())
    assert summary["scope"] == "selection"
    assert summary["select_files_sha256"] == hashlib.sha256(list_path.read_bytes()).hexdigest()
    assert "SELECTION run (NOT a suite result)" in (run_dir / "summary.selection.md").read_text()


def test_a_run_without_select_files_keeps_its_scope_and_its_summary_file_names(tmp_path):
    """(d, control) Input class: the same stand-in project planned with no ``--select-files``.

    Required so the selection binding is conditional on the manifest key: a plain run still writes ``summary.json``
    and no ``summary.selection.*``, and its result carries no ``select_files`` key.
    """
    repo = _project(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable]
    cmd += ["--runs-root", str(runs), "--warm-target", "tests/test_warm.py", "--declare-non-compile-bearing"]
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    run_dir = next(runs.iterdir())
    agg = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--allow-not-green"]
    subprocess.run(agg, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    assert (run_dir / "summary.json").is_file() and not list(run_dir.glob("summary.selection.*"))
    assert "select_files" not in json.loads((run_dir / "summary.json").read_text())


def test_a_list_that_omits_the_tail_file_is_refused_before_collection_naming_that_file_and_the_list(
    stand_in, tmp_path, monkeypatch
):
    """(e) Input class: the built-in ``TAIL_NODE_IDS``, no tail opt-out, and a list without the tail's file.

    Required so a selection keeps the full-suite shape and the planner never reaches a manifest that
    ``build_manifest`` would refuse later. Killed by a planner that validates after collection (the sentinel below
    fails the test if ``collect`` or ``warm`` runs) and by one that refuses without naming both files. The control:
    the same list plus the tail file passes the validation.
    """
    monkeypatch.delenv(POL.TAIL_NODE_IDS_FILE_ENV, raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("the planner collected or warmed before it validated the list")

    monkeypatch.setattr(R, "collect", _boom)
    monkeypatch.setattr(R, "warm", _boom)
    list_path = _list(tmp_path, "tests/test_warm.py")
    message = _refusal(stand_in, list_path)
    assert _TAIL_FILE in message and str(list_path) in message and "tail node ids" in message, message
    (stand_in / "tests" / "test_synth_00_compile_models.py").write_text("def test_compile_swmm():\n    pass\n")
    _git(stand_in, "add", _TAIL_FILE)
    _git(stand_in, "commit", "-q", "-m", "tail file")
    with_tail = _list(tmp_path, "tests/test_warm.py", _TAIL_FILE, name="with_tail.txt")
    R._validate_select_files(stand_in, with_tail, ["tests/test_warm.py", _TAIL_FILE], "all")


def test_build_manifest_plans_the_tail_as_the_last_chunk_over_a_selections_node_ids(tmp_path):
    """(f) Input class: fake node ids of a selection that include the three built-in tail ids (no solver).

    Required so a selection's plan has the full-suite shape: the three compile nodes leave the array and run as the
    last chunk, ``tail_chunk == chunk_count - 1``. Killed by a planner that leaves a tail id in an array chunk.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_synth_00_compile_models.py").write_text(
        "def test_compile_swmm():\n    pass\n\n\ndef test_compile_tritonswmm():\n    pass\n\n\n"
        "def test_compile_triton_only():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_listed.py").write_text("def test_one():\n    pass\n", encoding="utf-8")
    nodes = [*POL.TAIL_NODE_IDS, "tests/test_listed.py::test_one"]
    manifest = P.build_manifest(
        repo_root=tmp_path,
        node_ids=nodes,
        source_sha="deadbeef",
        run_id="r",
        closures={n: [] for n in nodes},
        tail_node_ids=list(POL.TAIL_NODE_IDS),
    )
    assert manifest["tail_chunk"] == manifest["chunk_count"] - 1
    assert manifest["chunks"][-1]["node_ids"] == list(POL.TAIL_NODE_IDS)
    assert all(not set(POL.TAIL_NODE_IDS) & set(c["node_ids"]) for c in manifest["chunks"][:-1])
