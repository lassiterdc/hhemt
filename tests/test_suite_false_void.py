"""A chunk whose only holder of a session fixture was skipped before setup is not VOID (counted array 1, finding F1).

In counted array 1, chunks 00 and 01 were voided although every test passed: the only holder of
`rendered_synth_multi_sim` in each is a `skipif` test, which pytest skips before fixture setup, so the fixture never
completed and `classify_chunk` read a lost precondition. Work package 3a, chunk 3a.5: `build_manifest` records beside
each chunk's `expected_fixtures` an `expected_fixture_holders` map, and `classify_chunk` reports a dead fixture as an
advisory instead of a VOID reason when EVERY holder is reported skipped by a skip that is not an xfail. pytest reports
an xfail-marked test whose fixture FAILED setup as `<skipped type="pytest.xfail">`, so that kind must not excuse it.

Every test below runs the real planner (`build_manifest`), the real chunk driver (`hhemt.suite._runner --chunk`) and the
real aggregator over a stand-in project; pytest runs only stand-in tests and nothing here reaches a solver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hhemt.suite._runner as R
from hhemt.suite import aggregate as A
from hhemt.suite import partition as P

_FIXTURE = "rendered_synth_multi_sim"

_HEAD = """\
import pytest


@pytest.fixture(scope="session")
def rendered_synth_multi_sim():
    {body}
"""
_OK_BODY = "return 'built'"
_FAIL_BODY = "raise RuntimeError('session fixture failed setup')"


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t" + "@" + "t.invalid", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _scenario(tmp_path: Path, tests: str, *, fails: bool, others: str = "def test_plain():\n    assert True\n"):
    """Build the stand-in project and the planner's manifest for it; return (repo, run_dir, manifest, node ids).

    `tests` is the body of the holder file after the fixture; the closures handed to `build_manifest` name the
    fixture for every test whose signature does."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_holders.py").write_text(_HEAD.format(body=_FAIL_BODY if fails else _OK_BODY) + tests)
    (repo / "tests" / "test_other.py").write_text(others)
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    nodes = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    closures = {n: ([_FIXTURE] if "test_holders.py" in n and "holder" in n else []) for n in nodes}
    manifest = P.build_manifest(
        repo_root=repo, node_ids=nodes, source_sha=_git(repo, "rev-parse", "HEAD"), run_id="r", closures=closures
    )
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    return repo, run_dir, manifest, nodes


def _run_and_aggregate(repo: Path, run_dir: Path, manifest: dict, nodes: list[str], tmp_path: Path, **edit) -> dict:
    """Write the manifest (after `edit` removes keys from chunk 0), run chunk 0 through the real driver, aggregate."""
    for key in edit.get("drop_chunk_keys", ()):
        for c in manifest["chunks"]:
            c.pop(key, None)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    for c in manifest["chunks"]:
        (run_dir / "chunks" / f"{c['chunk_id']:02d}.txt").write_text("\n".join(c["node_ids"]) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env["PYTHONPATH"] = str(Path(R.__file__).resolve().parents[2])
    env["TMPDIR"] = str(tmp_path)
    for c in manifest["chunks"]:
        cmd = [sys.executable, "-m", "hhemt.suite._runner", "--chunk", str(c["chunk_id"]), "--run-dir", str(run_dir)]
        cmd += ["--toolkit", str(repo), "--python", sys.executable]
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, check=False)
    return A.aggregate(run_dir, scope="array")


def _holder_chunk(result: dict) -> dict:
    """The chunk row of the chunk that holds the fixture's holders (the one with an expected fixture)."""
    rows = [r for r in result["chunks"] if r["state"] != "PASS" or r["advisories"]]
    return rows[0] if rows else result["chunks"][0]


_SKIPIF_SOLE = """\

@pytest.mark.skipif(True, reason="gated")
def test_holder(rendered_synth_multi_sim):
    pass
"""
_SKIPIF_PLUS_PLAIN = (
    _SKIPIF_SOLE
    + """

def test_second_holder(rendered_synth_multi_sim):
    pass
"""
)
_XFAIL_SOLE = """\

@pytest.mark.xfail
def test_holder(rendered_synth_multi_sim):
    pass
"""


def test_build_manifest_records_each_expected_fixtures_holders_exactly(tmp_path):
    """Property: `expected_fixture_holders` names, per expected fixture, exactly the chunk's nodes whose closure
    holds it.

    Class: NEW CAPABILITY, over a stand-in project with two holders and a non-holder. The wrong implementation
    records nothing (today's `build_manifest`) or records every node of the chunk.

    Why it is required, checkable against the code: `classify_chunk` can excuse a dead fixture only by comparing its
    holders with the skipped nodes, and the map is built from the same `closures` that derive `expected_fixtures`.

    What kills it: omitting the map, or deriving it from the chunk's node list instead of the closures.

    A second correct implementation under which it passes: one that records the holders as a set rather than a list,
    if the assertion compared sorted lists."""
    _, _, manifest, nodes = _scenario(tmp_path, _SKIPIF_PLUS_PLAIN, fails=False)
    holders = {n for n in nodes if "holder" in n}
    assert len(holders) == 2 and any("test_plain" in n for n in nodes)
    recorded = [c["expected_fixture_holders"] for c in manifest["chunks"] if c["expected_fixtures"]]
    assert recorded == [{_FIXTURE: sorted(holders)}]
    assert all(c["expected_fixture_holders"] == {} for c in manifest["chunks"] if not c["expected_fixtures"])


def test_a_sole_skipif_holder_gives_pass_with_an_advisory_and_no_void_reason(tmp_path):
    """Property: a real chunk whose only holder of a recorded session fixture is a `skipif(True)` test, with every
    other test passing, classifies PASS, names the fixture and its skipped holder in an advisory, and carries no VOID
    reason.

    Class: REGRESSION, red on today's `classify_chunk`, which returns VOID with "expected session fixture(s) did not
    complete setup" (counted array 1, chunks 00 and 01).

    Why it is required, checkable against the code: pytest skips a `skipif` test before it asks for the fixture, so
    the fixture never completes and `session_fixtures_ok` lacks it, while no precondition was lost.

    What kills it: the pre-change `classify_chunk`. Control: the next tests keep VOID where a holder was not skipped.

    A second correct implementation under which it passes: one that excuses the fixture before the sidecar is read,
    if it still checks every holder."""
    repo, run_dir, manifest, nodes = _scenario(tmp_path, _SKIPIF_SOLE, fails=False)
    result = _run_and_aggregate(repo, run_dir, manifest, nodes, tmp_path)
    row = next(r for r in result["chunks"] if r["advisories"] or r["state"] == "VOID")
    assert (row["state"], row["reasons"]) == ("PASS", []), row
    assert len(row["advisories"]) == 1 and _FIXTURE in row["advisories"][0] and "test_holder" in row["advisories"][0]
    assert any(_FIXTURE in a for a in result["advisories"])


def test_a_second_holder_whose_setup_fails_keeps_the_chunk_void(tmp_path):
    """Control: a skipped holder beside a NOT skipped holder whose setup fails stays VOID.

    Class: NEW CAPABILITY, must-fire against a rule that excuses on ANY skipped holder rather than EVERY holder.

    Why it is required, checkable against the code: the excuse is `set(holders) <= skipped`; the second holder is
    reported as an error, not a skip, so the fixture failure is real.

    What kills it: `set(holders) & skipped`, or excusing when any holder is skipped.

    A second correct implementation under which it passes: one that counts non-skipped holders and excuses at zero."""
    repo, run_dir, manifest, nodes = _scenario(tmp_path, _SKIPIF_PLUS_PLAIN, fails=True)
    result = _run_and_aggregate(repo, run_dir, manifest, nodes, tmp_path)
    row = next(r for r in result["chunks"] if r["state"] == "VOID")
    assert any("did not complete setup" in why and _FIXTURE in why for why in row["reasons"]), row


def test_a_sole_xfail_holder_whose_fixture_fails_stays_void(tmp_path):
    """Control (X1): an xfail-marked sole holder whose fixture FAILS setup is reported by pytest as
    `<skipped type="pytest.xfail">` and must not excuse the dead fixture.

    Class: NEW CAPABILITY, must-fire against a rule that counts every junit skip as a skipped holder.

    Why it is required, checkable against the code: `skipped_non_xfail_nodes` keeps a `<skipped>` only when its `type`
    is not `pytest.xfail`; the junit this stand-in project produces is asserted below to carry that type, so the test
    cannot pass because the skip kind changed.

    What kills it: dropping the type check.

    A second correct implementation under which it passes: one that reads the skip kind from the pytest report
    sidecar instead of the junit."""
    repo, run_dir, manifest, nodes = _scenario(tmp_path, _XFAIL_SOLE, fails=True)
    result = _run_and_aggregate(repo, run_dir, manifest, nodes, tmp_path)
    junit = (run_dir / "chunk-00.junit.xml").read_text()
    assert 'type="pytest.xfail"' in junit, "control: the junit must carry the xfail skip type this test is about"
    assert any(r["state"] == "VOID" and any(_FIXTURE in why for why in r["reasons"]) for r in result["chunks"])


def test_a_manifest_without_the_holders_map_aggregates_void_as_before(tmp_path):
    """Control: the same sole-`skipif` run planned before `expected_fixture_holders` existed stays VOID.

    Class: NEW CAPABILITY. A run planned before this chunk is never re-read with the new rule.

    Why it is required, checkable against the code: `classify_chunk` excuses nothing when `expected_fixture_holders` is
    absent or empty.

    What kills it: defaulting a missing map to "all nodes", or to "excuse".

    A second correct implementation under which it passes: one that versions the manifest and refuses the field's
    absence outright, if it still reports VOID."""
    repo, run_dir, manifest, nodes = _scenario(tmp_path, _SKIPIF_SOLE, fails=False)
    result = _run_and_aggregate(repo, run_dir, manifest, nodes, tmp_path, drop_chunk_keys=("expected_fixture_holders",))
    row = next(r for r in result["chunks"] if r["state"] == "VOID")
    assert any("did not complete setup" in why and _FIXTURE in why for why in row["reasons"]), row
    assert row["advisories"] == []


def _skipped_junit(path: Path, names: list[str]) -> Path:
    """A junit holding one `<skipped type="pytest.skip">` testcase per name, all in `tests/test_holders.py`."""
    cases = "".join(
        f'<testcase classname="tests.test_holders" name="{n}" file="tests/test_holders.py" time="0">'
        '<skipped type="pytest.skip" message="gated">tests/test_holders.py:1: gated</skipped></testcase>'
        for n in names
    )
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>')
    return path


def _clean_status(sha: str = "abc") -> dict:
    """A sidecar that would classify PASS but for the expected fixture: sha pinned, clean tree, one resolution, rc 0."""
    return {
        "session_fixtures_ok": [],
        "source_sha_start": sha,
        "source_sha_end": sha,
        "dirty_start": False,
        "dirty_end": False,
        "resolved_hhemt": {"1": "/x/hhemt/__init__.py"},
        "pytest_exit": 0,
        "slurm_state": "COMPLETED",
    }


def test_a_dead_fixture_with_an_empty_holder_list_is_not_excused_by_the_empty_subset(tmp_path):
    """Property: an expected session fixture that did not complete setup and whose recorded holder list is EMPTY keeps
    the chunk VOID, and a control fixture with a non-empty, wholly skipped holder list beside it is excused.

    Class: REGRESSION against a guard-less excuse. The empty set is a subset of every set, so a rule that tests
    `set(holders) <= skipped` without first requiring holders excuses a fixture nobody was found to hold, which is
    the false pass VOID exists to prevent (partition records an empty list for a fixture admitted only by the
    dynamic-lookup route).

    Why it is required, checkable against the code: `classify_chunk` excuses `fx` only when
    `expected_fixture_holders.get(fx)` is truthy; the junit here reports a skipped test, so the subset test alone
    would hold.

    What kills it: replacing that truthiness test by `fx in expected_fixture_holders`. Control: the second fixture
    (`rendered_synth_sensitivity`, one holder, skipped) is excused in the same call, so the instrument does excuse
    when it should.

    A second correct implementation under which it passes: one that refuses a manifest whose holder list is empty."""
    holder = "tests/test_holders.py::test_holder"
    junit = _skipped_junit(tmp_path / "chunk.junit.xml", ["test_holder"])
    advisories: list[str] = []
    state, why = A.classify_chunk(
        _clean_status(),
        junit,
        "abc",
        [_FIXTURE, "rendered_synth_sensitivity"],
        {_FIXTURE: [], "rendered_synth_sensitivity": [holder]},
        advisories,
    )
    assert state == "VOID", (state, why)
    assert any(_FIXTURE in r and "did not complete setup" in r for r in why), why
    assert not any("rendered_synth_sensitivity" in r for r in why), why
    assert len(advisories) == 1 and "rendered_synth_sensitivity" in advisories[0], advisories


def test_a_holder_list_naming_only_a_node_the_junit_never_reports_keeps_the_chunk_void(tmp_path):
    """Control: a holder absent from the junit is not skipped, so the dead fixture stays a VOID reason.

    Class: NEW CAPABILITY, a different position of the same invariant: a non-empty list that the junit does not
    cover (an id that failed to match). Reds under a rule that excuses when the junit has any skip at all.

    Why it is required, checkable against the code: `set(holders) <= skipped` is False when one holder is missing.

    What kills it: `&` for `<=`, or excusing whenever `skipped` is non-empty.

    A second correct implementation under which it passes: one that counts unmatched holders and excuses at zero."""
    junit = _skipped_junit(tmp_path / "chunk.junit.xml", ["test_holder"])
    state, why = A.classify_chunk(
        _clean_status(), junit, "abc", [_FIXTURE], {_FIXTURE: ["tests/test_holders.py::test_missing"]}, []
    )
    assert state == "VOID" and any(_FIXTURE in r for r in why), (state, why)


def test_a_skipped_holder_that_also_carries_an_error_does_not_excuse_the_dead_fixture(tmp_path):
    """Control: a testcase with both `<skipped>` and `<error>` (a skip in the call phase, then a teardown error) is not
    counted skipped, so the dead fixture stays a VOID reason.

    Class: NEW CAPABILITY, must-fire against a reader that looks only for `<skipped>`. Control: the same junit without
    the `<error>` child is excused.

    Why it is required, checkable against the code: `skipped_non_xfail_nodes` skips a testcase whose children include
    `error` or `failure`; a holder that errored may have failed the fixture.

    What kills it: dropping that exclusion.

    A second correct implementation under which it passes: one that classifies per phase from the report sidecar."""
    holder = "tests/test_holders.py::test_holder"
    plain = _skipped_junit(tmp_path / "plain.junit.xml", ["test_holder"])
    errored = tmp_path / "errored.junit.xml"
    errored.write_text(
        plain.read_text().replace("</testcase>", '<error message="teardown failed">boom</error></testcase>')
    )
    holders = {_FIXTURE: [holder]}
    ok_state, _ = A.classify_chunk(_clean_status(), plain, "abc", [_FIXTURE], holders, [])
    assert ok_state == "PASS", "control: without the error child the skipped holder is excused"
    state, why = A.classify_chunk(_clean_status(), errored, "abc", [_FIXTURE], holders, [])
    assert state == "VOID" and any(_FIXTURE in r for r in why), (state, why)
