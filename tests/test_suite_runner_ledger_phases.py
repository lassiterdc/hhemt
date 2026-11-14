"""The chunk ledger records setup, call and teardown reports of passing tests, each naming its xdist worker.

Work package 1d, chunk 1d.1 of the approved shape (``_runner`` ledger records setup and teardown reports and the
xdist ``worker_id``). The ledger is ``chunk-NN.reports.jsonl``; before this chunk a PASSING test left one ``call``
record and nothing else, so the duration table a heavy split balances on (``_file_durations``) saw call time only and
a session fixture's build cost, paid in the setup phase of the first test that requests it, was invisible.

Properties (class: NEW CAPABILITY, no prior behaviour to regress against):

* P1. A passing test leaves exactly one ``setup``, one ``call`` and one ``teardown`` record.
  Wrong implementation: the pre-chunk filter ``if outcome == "passed" and when != "call": return``. Under it P1 is red
  (the ledger holds a ``call`` row only). Satisfying inputs at two positions: an ordinary passing test, and a test
  whose setup FAILS (a fixture error), which must leave a ``setup`` row with outcome ``failed``, a ``teardown`` row
  and no ``call`` row.
* P2. Every report record carries a ``worker`` key: ``None`` with no xdist, ``gwN`` under ``-n 2``.
  Wrong implementation: no ``worker`` key, or the controller's ``None`` stamped on records a worker produced.

Why required: the duration table is the input of the heavy split (shape item 1d.2) and the worker name is the input
of the xdist-aware ledger (shape item 1d.4); a consumer cannot recover either from a ledger that lacks them.

A second correct implementation under which these still pass: recording the three phases in one combined record per
test would fail P1, so P1 pins the per-phase form the shape states ("setup and teardown reports"). The fsync policy
of the writer is deliberately not tested here: every record is fsynced exactly as before this chunk (developer ruling).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import hhemt.suite._runner as runner

_PROJECT = """
import pytest


@pytest.fixture
def broken():
    raise RuntimeError("fixture setup failed")


def test_passes():
    assert True


def test_setup_fails(broken):
    assert True
"""


def _run_fixture_project(tmp_path: Path, *extra: str) -> list[dict]:
    """Run the plugin over the two-test project in a fresh pytest process; return the ledger's report records."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "test_ledger_probe.py").write_text(textwrap.dedent(_PROJECT))
    ledger = tmp_path / "chunk-00.reports.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_SUITE_"))}
    env[runner.LOGREPORT_ENV] = str(ledger)
    env["PYTHONPATH"] = str(Path(runner.__file__).resolve().parents[2])
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "hhemt.suite._runner", "-p", "no:cacheprovider", "-q", *extra],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("kind") == "report"]


def _phases(rows: list[dict], name: str) -> list[tuple[str, str]]:
    return sorted((r["when"], r["outcome"]) for r in rows if r["nodeid"].endswith("::" + name))


def test_passing_test_leaves_setup_call_and_teardown_records(tmp_path: Path) -> None:
    """P1: a passing test leaves one setup, one call and one teardown record.

    Input class: any test that passes. Required because the duration table needs the setup phase. Killed by the
    pre-chunk filter that drops passing non-call reports. A second correct implementation still passes: any writer
    that emits one record per phase."""
    rows = _run_fixture_project(tmp_path)
    assert _phases(rows, "test_passes") == [("call", "passed"), ("setup", "passed"), ("teardown", "passed")]


def test_failed_setup_leaves_setup_and_teardown_records_and_no_call_record(tmp_path: Path) -> None:
    """P1: a test whose fixture fails leaves a failed setup record and a teardown record, and no call record.

    Input class: any test whose setup errors. Required so a failed fixture build keeps its cost and its reason in the
    ledger. Killed by a writer that drops either phase or invents a call record."""
    rows = _run_fixture_project(tmp_path)
    assert _phases(rows, "test_setup_fails") == [("setup", "failed"), ("teardown", "passed")]
    failed = [r for r in rows if r["nodeid"].endswith("::test_setup_fails") and r["when"] == "setup"][0]
    assert "fixture setup failed" in failed["longrepr"]


def test_every_record_carries_a_worker_key_none_without_xdist(tmp_path: Path) -> None:
    """P2: without xdist every report record carries a ``worker`` key whose value is null.

    Input class: every record of a serial session. Required because a consumer reads the key unconditionally. Killed
    by a writer that omits the key or stamps a worker name on a serial run."""
    rows = _run_fixture_project(tmp_path)
    assert rows, "the probe project must have produced report records"
    assert all("worker" in r for r in rows)
    assert {r["worker"] for r in rows} == {None}


def test_records_name_the_xdist_worker_under_n_2(tmp_path: Path) -> None:
    """P2: under ``-n 2`` every report record names its ``gwN`` worker and each phase of a test is present.

    Input class: every record of an xdist session, written by a worker or by the controller. Required because the
    xdist-aware ledger (shape item 1d.4) attributes phases to workers. Killed by a writer that leaves ``worker``
    null under xdist. A count is not asserted: chunk 1d.4 removes the controller duplicate."""
    import xdist  # noqa: F401 - the test extra declares pytest-xdist; absence is a failure, not a skip

    rows = _run_fixture_project(tmp_path, "-n", "2")
    workers = {r["worker"] for r in rows}
    assert workers, "no records written under -n 2"
    assert None not in workers, f"a record carries no worker id under xdist: {workers}"
    assert all(w.startswith("gw") for w in workers), workers
    # The setup/call/teardown triple of the passing test is still present (controller and worker may both write it;
    # chunk 1d.4 removes that duplication, so this arm asserts presence of each phase and not a count).
    assert {w for w, _ in _phases(rows, "test_passes")} == {"setup", "call", "teardown"}


def test_worker_id_is_read_from_the_controller_node_when_the_environment_has_none(monkeypatch) -> None:
    """P2: the worker id comes from the environment in a worker and from ``node.workerinput`` on the controller.

    Input class: a report with no environment variable set, and the same report with the variable set. Required
    because the controller has no ``PYTEST_XDIST_WORKER``. Killed by a helper that reads only one of the two."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    report = SimpleNamespace(node=SimpleNamespace(workerinput={"workerid": "gw3"}))
    assert runner._xdist_worker_id(report) == "gw3"
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    assert runner._xdist_worker_id(report) == "gw1"
