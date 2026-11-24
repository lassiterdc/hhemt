"""The complement runner (``--complement``) and the re-keyed run-proof gate (design DR-8, AC-1.3).

An array run declines the two run-proof modules because their ``skipif`` reads the suite role ``chunk``. The suite
runner's ``--complement`` mode appends ONE chunk (``covers: "complement"``) carrying the manifest's declared complement
to a finished array run and runs it under the role ``complement``, which the gate does not decline; the aggregator then
reads ``scope=union``. Every test below runs the real driver over a stand-in project whose two stub tests carry the
run-proof gate, the structural skip reasons and nothing else: pytest only collects and runs stubs, and nothing here
reaches a solver, so none of it is simulation-bearing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.suite._runner as R
import tests.utils_for_testing as tst_ut
from hhemt.suite import aggregate as A
from hhemt.suite import chunk_site, missing_conformance_sites
from tests.test_suite_tail_element import (
    _ARRAY_END,
    _ARRAY_OK,
    _TREE,
    _entry,
    _fixture_run,
    _junit,
    _race,
    _status,
    _times,
)
from tests.test_suite_tier_selection import _CONFTEST, _env, _git, _project

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STUB_FILE = "tests/test_stub_gates.py"

# The two stubs carry the run-proof gate as the two run-proof modules do: the structural reason, imported from the
# aggregator, on a predicate that reads the role. The live one also declares an operator gate, evaluated AFTER the role
# gate here (pytest evaluates the mark nearest the function first) as the doi module's declares it AHEAD of its own.
_STUBS = """\
import json
import os
from pathlib import Path

import pytest

from hhemt.suite.aggregate import LIVE_DEPOSIT_SKIP_REASON, LOCAL_RUN_PROOF_SKIP_REASON


def _in_array_chunk():
    return os.environ.get("HHEMT_SUITE_ROLE") == "chunk"


def _record(name):
    line = json.dumps({"test": name, "role": os.environ.get("HHEMT_SUITE_ROLE")})
    with open(Path(os.environ["STUB_ROLE_DIR"]) / "ran.jsonl", "a", encoding="utf-8") as fh:
        fh.write(line + "\\n")


@pytest.mark.skipif(_in_array_chunk(), reason=LOCAL_RUN_PROOF_SKIP_REASON)
def test_gated_local():
    _record("local")


@pytest.mark.skipif(not os.environ.get("STUB_OPERATOR"), reason="operator-gated stub.")
@pytest.mark.skipif(_in_array_chunk(), reason=LIVE_DEPOSIT_SKIP_REASON)
def test_gated_live():
    _record("live")
"""

_SESSION_HOOK = """

def pytest_sessionstart(session):
    import json
    import os
    from pathlib import Path

    where = os.environ.get("STUB_ROLE_DIR")
    if where:
        entry = {"role": os.environ.get("HHEMT_SUITE_ROLE"), "args": [str(a) for a in session.config.args]}
        with open(Path(where) / "sessions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\\n")
"""


def _runner(base: Path, repo: Path, *argv: str, roles: Path | None = None, extra_env: dict[str, str] | None = None):
    """Run the real driver over the stand-in project at ``repo``, as one process, with the stub environment.

    ``roles`` is the directory the stubs and the session hook record into; the default is the array run's own.
    """
    where = roles if roles is not None else base / "roles"
    where.mkdir(exist_ok=True)
    env = {**_env(base), "STUB_ROLE_DIR": str(where), **(extra_env or {})}
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable, *argv]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, check=False)


def _aggregate(base: Path, run_dir: Path, scope: str) -> str:
    cmd = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--scope", scope]
    done = subprocess.run(cmd, env=_env(base), capture_output=True, text=True, timeout=240, check=False)
    return next(ln for ln in done.stdout.splitlines() if ln.startswith("verdict="))


def _lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def array_run(tmp_path_factory):
    """A planned and fully run array: ``(base, repo, run_dir)``, the run's chunks all executed under the role ``chunk``.

    The stand-in project carries the two gated stubs. The plan reads their skipif reasons, so the manifest's
    ``declared_complement`` is both of them; every chunk then runs through the real ``--chunk`` path.
    """
    base = tmp_path_factory.mktemp("complement")
    repo = _project(base)
    (repo / "conftest.py").write_text(_CONFTEST + _SESSION_HOOK, encoding="utf-8")
    (repo / _STUB_FILE).write_text(_STUBS, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "stubs")
    (base / "roles").mkdir()
    runs = base / "runs"
    runs.mkdir()
    plan = [
        sys.executable,
        "-m",
        "hhemt.suite._runner",
        "--toolkit",
        str(repo),
        "--python",
        sys.executable,
        "--runs-root",
        str(runs),
        "--warm-target",
        "tests/test_warm.py",
        "--declare-non-compile-bearing",
    ]
    done = subprocess.run(plan, env=_env(base), capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    run_dir = next(runs.iterdir())
    count = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["chunk_count"]
    for cid in range(count):
        ran = _runner(base, repo, "--run-dir", str(run_dir), "--chunk", str(cid), extra_env={"STUB_OPERATOR": "1"})
        assert ran.returncode == 0, ran.stdout[-1500:] + ran.stderr[-1500:]
    return base, repo, run_dir


def _copy(array_run, tmp_path_factory) -> Path:
    """A private copy of the finished array's run directory, so each test appends its own complement chunk."""
    _, _, run_dir = array_run
    dest = tmp_path_factory.mktemp("copy") / run_dir.name
    shutil.copytree(run_dir, dest)
    return dest


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_the_complement_mode_appends_one_chunk_and_runs_both_stubs_under_the_complement_role(
    array_run, tmp_path_factory
):
    """Input class: a finished array run whose declared complement is two stub tests, each carrying the run-proof gate.

    Required by AC-1.3 (a)-(e) and the union read: exactly one chunk with ``covers: "complement"`` at the old
    ``chunk_count``, the count raised in the manifest AND the ``chunk_count`` file, a MATCH conformance record, an empty
    roster gap, and both stubs passed having seen the role ``complement``. Killed by a mode that runs the chunk under
    the role ``chunk`` (the stubs would skip), that raises the count in only one place, or that appends nothing.
    """
    base, repo, _ = array_run
    run_dir = _copy(array_run, tmp_path_factory)
    before = _manifest(run_dir)
    n = before["chunk_count"]
    declared = before["declared_complement"]
    assert len(declared) == 2 and not any(c.get("covers") == "complement" for c in before["chunks"])
    roles = run_dir.parent / "roles"

    done = _runner(base, repo, "--complement", "--run-dir", str(run_dir), roles=roles, extra_env={"STUB_OPERATOR": "1"})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]

    after = _manifest(run_dir)
    appended = after["chunks"][n:]
    assert len(appended) == 1 and len(after["chunks"]) == n + 1  # (a)
    assert appended[0]["covers"] == "complement" and appended[0]["chunk_id"] == n
    assert appended[0]["node_ids"] == declared
    assert after["chunk_count"] == n + 1  # (b)
    assert (run_dir / "chunk_count").read_text(encoding="utf-8").strip() == str(n + 1)
    assert (run_dir / "chunks" / f"{n:02d}.txt").read_text(encoding="utf-8").split() == declared
    record = json.loads((run_dir / "_conformance" / f"{chunk_site(n)}.json").read_text(encoding="utf-8"))
    assert record["status"] == "MATCH"  # (c)

    ran = _lines(roles / "ran.jsonl")  # (e)
    assert sorted((r["test"], r["role"]) for r in ran) == [("live", "complement"), ("local", "complement")]
    outcomes = A.parse_junit(run_dir / f"chunk-{n:02d}.junit.xml")
    assert sorted(outcomes.values()) == ["PASSED", "PASSED"] and sorted(outcomes) == sorted(declared)

    line = _aggregate(base, run_dir, "union")
    assert "scope=union" in line, line
    assert missing_conformance_sites(run_dir, chunk_count=after["chunk_count"]) == []  # (d)

    # The refusal: a second mode call appends nothing, exits non-zero and names the chunk and the re-run command.
    frozen = (run_dir / "manifest.json").read_bytes()
    again = _runner(
        base, repo, "--complement", "--run-dir", str(run_dir), roles=roles, extra_env={"STUB_OPERATOR": "1"}
    )
    assert again.returncode != 0
    assert (run_dir / "manifest.json").read_bytes() == frozen
    assert (run_dir / "chunk_count").read_text(encoding="utf-8").strip() == str(n + 1)
    text = again.stdout + again.stderr
    assert f"already has its complement chunk {n}" in text
    assert f"rerun.sh uva --run-id {after['run_id']} --chunk {n}" in text


def test_a_complement_stub_that_skips_again_reads_scope_array_and_the_problems_name_the_union_guard(
    array_run, tmp_path_factory
):
    """Input class: the complement chunk runs with the live stub's operator gate closed, so that stub skips.

    Required by the union guard: a declared-complement node that reaches no evaluated outcome keeps
    ``declared_not_attempted`` non-empty, and the aggregator forces the scope back to ``array`` with a problem that
    names the guard. Killed by a mode whose evidence is accepted whatever its tests did.
    """
    base, repo, _ = array_run
    run_dir = _copy(array_run, tmp_path_factory)
    done = _runner(base, repo, "--complement", "--run-dir", str(run_dir), roles=run_dir.parent / "roles")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]

    line = _aggregate(base, run_dir, "union")
    assert "scope=array" in line and "scope=union" not in line, line
    problems = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["problems"]
    assert any("scope=union requires complement evidence" in p for p in problems), problems


def test_an_array_chunk_records_the_role_chunk_and_the_gated_stubs_skip_with_the_structural_reason(array_run):
    """Input class: the array chunk that holds the two stubs, run through ``--chunk`` as the harness runs it.

    Required by the regression clause: the array still declines the run-proof tests. The chunk's pytest session sees
    ``HHEMT_SUITE_ROLE=chunk``, neither stub body runs, and each skips with the imported structural reason. Killed by
    a ``run_chunk`` that sets another role, and by a gate that stopped declining inside the array.
    """
    base, _, run_dir = array_run
    holder = next(c for c in _manifest(run_dir)["chunks"] if any(_STUB_FILE in n for n in c["node_ids"]))
    sessions = [s for s in _lines(base / "roles" / "sessions.jsonl") if any(_STUB_FILE in a for a in s["args"])]
    assert [s["role"] for s in sessions] == ["chunk"]
    assert not (base / "roles" / "ran.jsonl").exists() or not _lines(base / "roles" / "ran.jsonl")
    reasons = A.skip_reasons(run_dir / f"chunk-{holder['chunk_id']:02d}.junit.xml")
    stub_reasons = {n: r for n, r in reasons.items() if _STUB_FILE in n}
    assert len(stub_reasons) == 2
    assert all(msg in A.STRUCTURAL_SKIP_REASONS for msg in stub_reasons.values()), stub_reasons


@pytest.mark.parametrize(
    ("role", "expected"),
    [("chunk", True), ("complement", False), (None, False)],
    ids=["chunk-declines", "complement-runs", "unset-runs"],
)
def test_in_suite_array_chunk_is_true_only_under_the_chunk_role(monkeypatch, role, expected):
    """Input class: the role variable at ``chunk``, at ``complement`` and unset, with a SLURM job id present in each.

    Required so the gate reads the role and not the scheduler: the job id is set in every row and changes nothing.
    Killed by a gate still keyed on a scheduler variable.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    if role is None:
        monkeypatch.delenv("HHEMT_SUITE_ROLE", raising=False)
    else:
        monkeypatch.setenv("HHEMT_SUITE_ROLE", role)
    assert tst_ut.in_suite_array_chunk() is expected


def test_the_structural_reasons_are_the_two_named_constants_and_each_gated_module_uses_one():
    """Input class: the aggregator's tuple and the ``pytestmark`` of the two gated modules.

    Required by the exact-match coupling the aggregator's comment records as having broken once: one site per reason.
    Killed by a tuple that restates a literal, and by a module whose ``skipif`` reason is its own copy.
    """
    import tests.test_doi_roundtrip_e2e as e2e
    import tests.test_synth_09_from_doi_run_proof as run_proof

    assert A.STRUCTURAL_SKIP_REASONS == (A.LOCAL_RUN_PROOF_SKIP_REASON, A.LIVE_DEPOSIT_SKIP_REASON)
    for module, reason in ((run_proof, A.LOCAL_RUN_PROOF_SKIP_REASON), (e2e, A.LIVE_DEPOSIT_SKIP_REASON)):
        reasons = [m.kwargs["reason"] for m in module.pytestmark if m.name == "skipif"]
        assert reason in reasons and reason in A.STRUCTURAL_SKIP_REASONS, (module.__name__, reasons)


def test_the_stale_text_and_the_scheduler_gate_are_gone():
    """Input class: the aggregator source and the test helpers.

    Required by the retirement: the aggregator no longer says the complement producer does not exist, and the helper
    module no longer defines the scheduler gate. Killed by a rewrite that adds the new gate and leaves the old one.
    """
    assert "does not exist yet" not in Path(A.__file__).read_text(encoding="utf-8")
    assert not hasattr(tst_ut, "on_scheduler_node")
    assert "on_scheduler_node" not in (_REPO_ROOT / "tests" / "utils_for_testing.py").read_text(encoding="utf-8")


def test_the_runner_derives_the_role_from_the_chunk_covers_field():
    """Input class: the two role constants the runner exports.

    Required so the runner and the tests-side statement of the variable name agree. Killed by a rename on one side.
    """
    from tests.fixtures._triton_source_cache import _SUITE_ROLE_ENV

    assert R.SUITE_ROLE_ENV == _SUITE_ROLE_ENV == "HHEMT_SUITE_ROLE"
    assert (R.CHUNK_ROLE, R.COMPLEMENT_ROLE, R.COMPLEMENT_COVERS) == ("chunk", "complement", "complement")


def _with_complement(root: Path, *, written: bool) -> None:
    """Append a complement chunk (id 3, run after the tail) to the tail suite's fixture run directory."""
    node = "tests/test_c.py::test_c0"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["collected"].append(node)
    manifest["chunk_count"] = 4
    manifest["chunks"].append(
        {"chunk_id": 3, "kind": "cheap", "node_ids": [node], "workers": 1, "covers": "complement"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    times = _times("2026-11-03T10:40:00.000+00:00", "2026-11-03T10:50:00.000+00:00")
    status = _status(3, times=times, entries={_TREE: _entry(written=written)})
    (root / "chunk-03.status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "chunk-03.junit.xml").write_text(_junit([node]), encoding="utf-8")


@pytest.mark.parametrize("complement_writes", [False, True], ids=["complement-reads", "complement-writes"])
def test_a_complement_chunk_run_after_the_tail_does_not_withdraw_the_tail_exemption(tmp_path, complement_writes):
    """Input class: a run with array chunks, a tail and a complement chunk that started after the tail ended.

    Required: the complement chunk is not an array chunk, so the tail, which ran after every array chunk, keeps its
    exemption from the shared-tree joins (its write is reported as ``tail_writes``, not as a race). The complement chunk
    stays IN the joins: when it writes a tree the array chunks read, that race is still reported, naming chunk 3 alone.
    Killed by a ``_tail_ran_after`` that counts the complement chunk (the tail's own write would read as a race
    ``writers [2]``), and by an exemption that lets the complement chunk's own write through.
    """
    _fixture_run(tmp_path, tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"), array_times=_ARRAY_OK)
    _with_complement(tmp_path, written=complement_writes)
    r = A.aggregate(tmp_path, scope="array")
    assert _TREE in r["tail_writes"], r["problems"]
    races = _race(r["problems"])
    if complement_writes:
        assert len(races) == 1 and "writers [3]" in races[0], races
    else:
        assert races == [], races
