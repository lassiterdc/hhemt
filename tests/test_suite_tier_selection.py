"""A plan can cover the whole suite or only the solver tier, and a tier run can never pass for a whole-suite run.

Work package 1d, chunk 1d.5 of the approved shape. ``--select-tier compile_tier`` adds ``-m compile_tier`` to the
plan's collection and records the selection in the manifest; the aggregator then forces the scope of such a run to
``tier`` (never ``union``, never ledgered), prints ``tier={selection}`` on every verdict line, and a plan over a
duration ledger credits a file with the selected node ids' durations only. The plans below run the real driver over a
tiny stand-in project (a ``conftest.py`` hook marks one test ``compile_tier``); pytest only collects, no solver runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.suite._runner as R

_CONFTEST = """\
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name == "test_solver":
            item.add_marker(pytest.mark.compile_tier)
"""
_TESTS = "def test_solver():\n    assert True\n\n\ndef test_plain():\n    assert True\n"


def _git(repo: Path, *args: str) -> None:
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t" + "@" + "t.invalid", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def _project(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "conftest.py").write_text(_CONFTEST)
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    compile_tier: stand-in solver tier\n")
    (repo / "tests" / "test_stand.py").write_text(_TESTS)
    (repo / "tests" / "test_warm.py").write_text("def test_warm():\n    assert True\n")
    # The driver certifies that it resolved hhemt from the --toolkit tree it plans, so the stand-in project carries
    # a copy of the package under test at ``src/hhemt``.
    shutil.copytree(
        Path(R.__file__).resolve().parents[1], repo / "src" / "hhemt", ignore=shutil.ignore_patterns("__pycache__")
    )
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env["PYTHONPATH"] = str(tmp_path / "repo" / "src")
    env["TMPDIR"] = str(tmp_path)
    # A stand-in project has none of the harness tail's tests, and the planner refuses a built-in tail id it cannot
    # collect; an empty list file is the opt-out (no tail).
    no_tail = tmp_path / "no_tail_ids.txt"
    no_tail.write_text("", encoding="utf-8")
    env["HHEMT_SUITE_TAIL_NODE_IDS_FILE"] = str(no_tail)
    return env


def _plan(tmp_path: Path, *extra: str) -> dict:
    """Run the real planning driver over the stand-in project; return the parsed manifest of the run it minted."""
    repo = _project(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable]
    cmd += ["--runs-root", str(runs), "--warm-target", "tests/test_warm.py", "--declare-non-compile-bearing", *extra]
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    run_dir = next(runs.iterdir())
    return {"run_dir": run_dir, "manifest": json.loads((run_dir / "manifest.json").read_text()), "stdout": done.stdout}


def test_plan_collects_only_the_solver_tier_under_compile_tier_and_both_otherwise(tmp_path):
    """The plan covers one test at ``--select-tier compile_tier`` and both at ``all`` and at the default.

    Input class: a project with one test marked ``compile_tier`` by a conftest hook and one unmarked. Required so a
    Rivanna run can be restricted to the solver tier. Killed by a driver that ignores the flag (it plans both under
    ``compile_tier``). The manifest records the selection in each case.
    """
    tier = _plan(tmp_path / "tier", "--select-tier", "compile_tier")
    assert tier["manifest"]["collected"] == ["tests/test_stand.py::test_solver"]
    assert tier["manifest"]["select_tier"] == "compile_tier"
    allv = _plan(tmp_path / "all", "--select-tier", "all")
    default = _plan(tmp_path / "default")
    both = ["tests/test_stand.py::test_plain", "tests/test_stand.py::test_solver", "tests/test_warm.py::test_warm"]
    assert sorted(allv["manifest"]["collected"]) == both == sorted(default["manifest"]["collected"])
    assert allv["manifest"]["select_tier"] == default["manifest"]["select_tier"] == "all"


def test_default_collection_command_is_unchanged():
    """At the default the tier arguments are empty, so the collection command is today's.

    Input class: the two spellings ``all`` and ``compile_tier`` against the pre-chunk command. Required by the
    shape's byte-for-byte clause. Killed by a helper that always adds ``-m``, and by one that accepts an unknown tier.
    """
    assert R._tier_args("all") == []
    assert R._tier_args("compile_tier") == ["-m", "compile_tier"]
    with pytest.raises(SystemExit):
        R._tier_args("slow")


def _aggregate(run_dir: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--scope", "union"]
    cmd.append("--allow-not-green")
    return subprocess.run(
        cmd, env=_env(run_dir.parent.parent), capture_output=True, text=True, timeout=240, check=False
    )


def test_a_tier_manifest_aggregates_to_scope_tier_and_writes_no_ledger_entry(tmp_path):
    """Aggregating a ``compile_tier`` manifest at ``--scope union`` reports scope=tier and ledgers nothing.

    Input class: a run directory planned with ``--select-tier compile_tier``, aggregated with the caller asking for
    union. Required so a solver-tier green never prints the whole-suite label or counts as a whole-suite entry.
    Killed by today's aggregator (``scope=union``, no tier, a ledger line). Control in the test: the ``all`` run
    aggregated the same way keeps a whole-suite scope (``array``: this run has no complement evidence, so the
    aggregator itself downgrades the union request) with ``tier=all`` and is ledgered.
    """
    tier = _plan(tmp_path / "tier", "--select-tier", "compile_tier")
    out = _aggregate(tier["run_dir"])
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("verdict="))
    assert " scope=tier tier=compile_tier " in line, out.stdout[-1500:]
    assert "select_tier=compile_tier" in out.stdout and "Scope forced to tier" in out.stdout
    assert "ledger=skipped" in out.stdout
    assert not list(tier["run_dir"].parent.glob("*.jsonl")) and not list(tier["run_dir"].parent.glob("ledger*"))
    allv = _plan(tmp_path / "all", "--select-tier", "all")
    ctrl = _aggregate(allv["run_dir"])
    ctrl_line = next(ln for ln in ctrl.stdout.splitlines() if ln.startswith("verdict="))
    assert " tier=all " in ctrl_line and " scope=tier" not in ctrl_line, ctrl.stdout[-1500:]
    assert "ledger=skipped" not in ctrl.stdout and (allv["run_dir"].parent / "_ledger.jsonl").is_file()


def test_verdict_line_composers_both_carry_the_tier():
    """``render_verdict_line`` prints ``tier=`` beside the scope, defaulting to ``all`` for an older result.

    Input class: a result with and without a ``tier`` key. Required because two composers build the line. Killed
    by a composer that omits the token.
    """
    from hhemt.suite import aggregate as A

    counts = {
        "covered": 0, "structurally_excluded": 0, "failed": 0, "unevaluated": 0, "absent": 0, "collected": 0,
        "incidental_skips": 0, A._FIELD_SIGNALLED: 0, A._FIELD_UNSIGNALLED: 0, "signal_completeness": "x",
    }  # fmt: skip
    base = {"verdict": "GREEN", "scope": "tier", "run_id": "r", "counts": counts, "advisories": []}
    assert " scope=tier tier=compile_tier run_id=r " in A.render_verdict_line({**base, "tier": "compile_tier"})
    assert " scope=tier tier=all run_id=r " in A.render_verdict_line(base)


def _ledger(run_dir: Path) -> None:
    rows = [("tests/test_f.py::test_sel", 10.0), ("tests/test_f.py::test_unsel", 100.0)]
    lines = [{"kind": "report", "nodeid": n, "when": "call", "outcome": "passed", "duration": d} for n, d in rows]
    run_dir.mkdir(parents=True)
    (run_dir / "chunk-00.reports.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_a_tier_plan_credits_a_file_with_the_selected_durations_only(tmp_path):
    """A file with a selected (10 s) and an unselected (100 s) test gets 10 s under a tier plan, 110 s otherwise.

    Input class: one duration ledger and a selected-id set. Required so a tier-only prediction is not inflated by
    tests the plan does not run. Killed by a planner that credits the file total.
    """
    _ledger(tmp_path / "prior")
    selected = {"tests/test_f.py::test_sel"}
    assert R._file_durations(tmp_path / "prior", selected) == {"tests/test_f.py": pytest.approx(10.0)}
    assert R._file_durations(tmp_path / "prior") == {"tests/test_f.py": pytest.approx(110.0)}
    assert R._file_durations(tmp_path / "prior", None) == {"tests/test_f.py": pytest.approx(110.0)}


def test_the_tier_plan_passes_the_selected_ids_to_the_duration_table(tmp_path, monkeypatch):
    """``plan`` hands ``_file_durations`` the plan's node ids under a tier and nothing at the default.

    Input class: the two selections, through the real ``plan`` with its collection and manifest build stubbed to
    capture the argument. Required because the helper's filter is inert unless the plan supplies the ids. Killed by
    a plan that always passes ``None`` (the file total is credited).
    """
    from hhemt.suite import partition as P

    seen: list[object] = []
    monkeypatch.setattr(R, "_sha_and_dirty", lambda repo: ("a" * 40, False))
    monkeypatch.setattr(R, "warm", lambda *a, **k: None)
    monkeypatch.setattr(R, "_compile_precondition", lambda *a, **k: {})
    monkeypatch.setattr(R, "_file_durations", lambda d, only=None: seen.append(only) or {"x": 1.0})
    monkeypatch.setattr(R, "collect", lambda repo, py, tier="all": (["t.py::a"], {"t.py::a": []}, {}, []))
    monkeypatch.setattr(P, "build_manifest", lambda **k: {"chunk_count": 1, "chunks": [], "run_id": k["run_id"]})
    monkeypatch.setattr(P, "write_manifest", lambda m, d: Path(d).mkdir(parents=True, exist_ok=True))
    import hhemt.suite as S

    monkeypatch.setattr(S, "write_version_expectation", lambda *a, **k: None)
    monkeypatch.setattr(S, "verify_version_conformance", lambda *a, **k: ("MATCH", {}))
    monkeypatch.setattr("hhemt.cache.paths.tier_address_for", lambda repo: "addr")
    for tier in ("compile_tier", "all"):
        args = R.build_parser().parse_args(
            ["--toolkit", str(tmp_path), "--runs-root", str(tmp_path / tier), "--durations-from", str(tmp_path)]
        )
        args.select_tier = tier
        args.warm_performed_externally = False
        R.plan(args)
    assert seen == [{"t.py::a"}, None]
