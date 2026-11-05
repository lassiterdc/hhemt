"""A chunk can run its tests under xdist and the plugin's records stay single and complete.

Work package 1d, chunk 1d.4 of the approved shape. ``--chunk-workers K`` makes the chunk's pytest command carry
``-n K --dist loadgroup``; every chunk, at every K, carries the ``faulthandler_timeout`` override; and the ``_runner``
plugin is xdist-aware: each test phase is recorded once (by the worker that ran it, not again by the controller), and
``fixtures.json`` and ``reach.json`` hold the union over the worker processes instead of the controller's empty last
write. The ledger is ``chunk-NN.reports.jsonl``; the real wrapper (``python -m hhemt.suite._runner --chunk``) runs a
two-file stand-in project with real pytest and xdist subprocesses and no solver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.suite._runner as R

pytest.importorskip("xdist", reason="pytest-xdist is the laptop recipe's dependency and the subject of this file")

_FILE = """\
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.xdist_group("{name}")


@pytest.fixture(scope="session")
def sess_{name}(tmp_path_factory):
    root = Path(os.environ["REACH_BASE"]) / "synthetic_test_runs" / "slug" / "tree_{name}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "marker.txt").write_text("x")
    return root


def test_one(sess_{name}):
    assert sess_{name}.is_dir()


def test_two(sess_{name}):
    assert sess_{name}.is_dir()
"""


def _git(repo: Path, *args: str) -> None:
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t" + "@" + "t.invalid", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def _run_chunk(tmp_path: Path, workers: int) -> dict:
    """Run the real wrapper over the two-file project with ``workers`` workers; return paths and parsed outputs."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    for name in ("a", "b"):
        (repo / f"test_{name}.py").write_text(_FILE.format(name=name))
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    ids = [f"test_{n}.py::test_{t}" for n in ("a", "b") for t in ("one", "two")]
    (run_dir / "chunks" / "00.txt").write_text("\n".join(ids) + "\n")
    manifest = {"run_id": "wk", "chunk_count": 1, "chunk_workers": workers}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env["PYTHONPATH"] = str(Path(R.__file__).resolve().parents[2])
    env["REACH_BASE"] = str(tmp_path / "reach_base")
    env["TMPDIR"] = str(tmp_path)
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--chunk", "0", "--run-dir", str(run_dir)]
    cmd += ["--toolkit", str(repo), "--python", sys.executable]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, check=False)
    ledger = [json.loads(x) for x in (run_dir / "chunk-00.reports.jsonl").read_text().splitlines() if x.strip()]
    return {
        "done": done,
        "reports": [r for r in ledger if r.get("kind") == "report"],
        "fixtures": json.loads((run_dir / "chunk-00.fixtures.json").read_text()),
        "reach": json.loads((run_dir / "chunk-00.reach.json").read_text()),
        "status": json.loads((run_dir / "chunk-00.status.json").read_text()),
        "run_dir": run_dir,
    }


def test_two_workers_write_one_call_record_per_test_and_the_union_of_fixtures_and_reach(tmp_path):
    """At ``--chunk-workers 2`` each test has exactly one call record, and both files' fixtures and trees appear.

    Input class: a real chunk of two files, each with its own session fixture touching its own tree. Required
    because the controller's copy of each report duplicated the ledger and its empty fixture list overwrote the
    workers' (the chunk would be voided for fixtures that did complete). Killed by the previous plugin: two call
    records per test and ``fixtures.json`` equal to ``[]``. Must-fire control in the test: the same project at one
    worker gives the same single records and the same union, so the expectations are not an xdist artefact.
    """
    out = _run_chunk(tmp_path / "w2", 2)
    assert out["status"]["pytest_exit"] == 0, out["done"].stdout[-2000:]
    calls = [r["nodeid"] for r in out["reports"] if r["when"] == "call"]
    assert sorted(calls) == sorted(set(calls)) and len(calls) == 4, calls
    assert {r["worker"] for r in out["reports"]} <= {"gw0", "gw1"}
    assert {"sess_a", "sess_b"} <= set(out["fixtures"]), out["fixtures"]
    labels = {e["label"] for e in out["reach"]["entries"].values()}
    assert {"tree_a", "tree_b"} <= labels, labels
    assert {"sess_a", "sess_b"} <= set(out["status"]["session_fixtures_ok"])
    serial = _run_chunk(tmp_path / "w1", 1)
    assert len([r for r in serial["reports"] if r["when"] == "call"]) == 4
    assert serial["fixtures"] == out["fixtures"]
    assert {e["label"] for e in serial["reach"]["entries"].values()} == labels


def test_each_phase_has_exactly_one_record_under_workers(tmp_path):
    """Every (test, phase) pair has one ledger record at ``--chunk-workers 2``, for setup and teardown too.

    Input class: all three phases of every test in a two-worker chunk. Required so the duration table of chunk 1d.2
    does not count a phase twice. Killed by a plugin whose controller also writes each report (two records).
    """
    out = _run_chunk(tmp_path, 2)
    keys = [(r["nodeid"], r["when"]) for r in out["reports"]]
    assert len(keys) == len(set(keys)) == 12, sorted(keys)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_every_chunk_command_carries_the_faulthandler_override(workers):
    """The faulthandler override is present at every worker count.

    Input class: K of 1, 2 and 4. Required by the ruling that harness chunks pass ``faulthandler_timeout=1800``
    independently of K. Killed by a command that adds it only under xdist (or only serially).
    """
    cmd = R._chunk_pytest_cmd("py", Path("j.xml"), Path("bt"), ["t.py::a"], workers)
    assert "faulthandler_timeout=1800" in cmd
    assert cmd[cmd.index("faulthandler_timeout=1800") - 1] == "-o"


def test_four_workers_add_n_4_dist_loadgroup_and_one_worker_adds_nothing_else():
    """``--chunk-workers 4`` yields ``-n 4 --dist loadgroup``; at 1 only the override differs from today's.

    Input class: K = 4 and K = 1. Required so the default stays today's serial session. Killed by a command that
    adds ``-n`` at K = 1, or that omits ``--dist loadgroup`` at K = 4. The K = 1 reference is the pre-chunk command
    reconstructed from its parts, with the override removed.
    """
    four = R._chunk_pytest_cmd("py", Path("j.xml"), Path("bt"), ["t.py::a"], 4)
    assert four[four.index("-n") : four.index("-n") + 4] == ["-n", "4", "--dist", "loadgroup"]
    one = R._chunk_pytest_cmd("py", Path("j.xml"), Path("bt"), ["t.py::a"], 1)
    today = [
        "py", "-m", "pytest", "-p", "no:cacheprovider", "-v", "-o", "junit_family=xunit1", "--junitxml=j.xml",
        "--basetemp=bt", "-p", "_runner", "--durations=0", "t.py::a",
    ]  # fmt: skip
    without_override = one[:]
    i = without_override.index("faulthandler_timeout=1800")
    del without_override[i - 1 : i + 1]
    assert without_override == today
    assert "-n" not in one


def test_chunk_workers_flag_is_recorded_in_the_plan_and_defaults_to_one():
    """``--chunk-workers`` parses to 1 by default and to the given integer otherwise.

    Input class: the two spellings of the drive command line. Required because the plan records the value in the
    manifest the chunk reads. Killed by a parser without the flag or with another default.
    """
    parser = R.build_parser()
    assert parser.parse_args([]).chunk_workers == 1
    assert parser.parse_args(["--chunk-workers", "4"]).chunk_workers == 4


# ---------------------------------------------------------------------------
# The aggregator pairs junit testcases that carry the loadgroup suffix with their plain manifest ids.
# ---------------------------------------------------------------------------
_GROUPS = [
    "shared_runs_root:synth_sensitivity",
    "shared_runs_root:rendered_synth_multi_sim+synth_multi_sim",
    "g_1.2-x",
]


def _junit(tmp_path: Path, names: list[tuple[str, str]]) -> Path:
    """Write a junit file of ``(name, body)`` testcases for ``tests/test_x.py``, in the real attribute shape."""
    cases = "".join(
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{n}" time="0.1">{b}</testcase>'
        for n, b in names
    )
    path = tmp_path / "chunk-00.junit.xml"
    path.write_text(f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>', encoding="utf-8")
    return path


@pytest.mark.parametrize("group", _GROUPS)
def test_a_loadgroup_suffix_is_not_part_of_the_junit_node_id(tmp_path, group):
    """A testcase named ``test_y@{group}`` reports the plain id ``tests/test_x.py::test_y``.

    Input class: group names holding ``:``, ``+``, ``_``, ``-`` and ``.``. Required because ``--dist loadgroup``
    appends the group to the reported name and the manifest holds plain ids (the cause of ``absent=180`` on a run
    in which the tests ran). Killed by an aggregator that strips nothing.
    """
    from hhemt.suite import aggregate as A

    out = A.parse_junit(_junit(tmp_path, [(f"test_y@{group}", "")]))
    assert out == {"tests/test_x.py::test_y": "PASSED"}


@pytest.mark.parametrize(
    ("name", "plain"),
    [
        ("test_x[a@b]", "test_x[a@b]"),
        ("test_x[p]@shared_runs_root:g", "test_x[p]"),
        ("test_x[a@b]@g", "test_x[a@b]"),
        ("test_x[p@q+r]", "test_x[p@q+r]"),
        ("test_x", "test_x"),
    ],
)
def test_only_the_trailing_suffix_is_removed(name, plain):
    """Only an ``@group`` after the last character of the name is stripped; an ``@`` inside brackets stays.

    Input class: parametrised ids with and without a suffix, with ``@`` inside the brackets, and a bare name.
    Required so a parametrised id is never altered. Killed by stripping every ``@`` or by stripping inside brackets.
    """
    from hhemt.suite import aggregate as A

    assert A._strip_group_suffix(name) == plain


@pytest.mark.parametrize(
    ("body", "want"),
    [
        ("", "PASSED"),
        ("<failure message='boom'/>", "FAILED"),
        ("<error message='setup died'/>", "UNEVALUATED"),
        ("<skipped message='not here'/>", "SKIPPED_INCIDENTAL"),
    ],
)
def test_the_outcome_of_a_suffixed_testcase_reaches_the_plain_id(tmp_path, body, want):
    """The outcome kind is carried to the plain id, so a suffixed failure is counted FAILED and not PASSED.

    Input class: the four junit outcome shapes on a suffixed testcase. Required because the verdict counts by
    outcome. Killed by a pairing that drops the suffixed row (the id is absent) or defaults it to PASSED.
    """
    from hhemt.suite import aggregate as A

    out = A.parse_junit(_junit(tmp_path, [("test_y@shared_runs_root:g", body)]))
    assert out == {"tests/test_x.py::test_y": want}


def test_a_test_under_both_a_suffixed_and_a_plain_name_keeps_the_worse_outcome(tmp_path):
    """Two testcases for one test (suffixed and plain) collapse to one id holding the MORE SEVERE outcome.

    Input class: a pass then a failure, and a failure then a pass, in both orders. Required so a retry or a second
    chunk cannot launder a failure by arrival order. Killed by a parser where the last row wins.
    """
    from hhemt.suite import aggregate as A

    fail = "<failure message='boom'/>"
    one = A.parse_junit(_junit(tmp_path, [("test_y@g", ""), ("test_y", fail)]))
    two = A.parse_junit(_junit(tmp_path, [("test_y", fail), ("test_y@g", "")]))
    assert one == two == {"tests/test_x.py::test_y": "FAILED"}


def _aggregate_with_junit(tmp_path, names):
    from hhemt.suite import aggregate as A
    from tests.test_verdict_partition import _build

    _build(tmp_path, chunk_entries=[{}])
    _junit(tmp_path, names)
    return A.aggregate(tmp_path, scope="array")


def test_suffixed_junit_pairs_with_the_manifest_and_raises_no_id_shape_problem(tmp_path):
    """A run whose junit names carry the suffix reports every test covered, none absent, no id-shape problem.

    Input class: a manifest of two plain ids and a junit naming both with a suffix. Required because the
    aggregator reported ``absent=180`` for tests that ran. Killed by an aggregator that does not strip.
    """
    res = _aggregate_with_junit(tmp_path, [("test_0@shared_runs_root:g", ""), ("test_1@h", "")])
    assert res["counts"]["absent"] == 0 and res["counts"]["covered"] == 2
    assert res["unmatched_junit"] == []
    assert not [p for p in res["problems"] if "id-shape mismatch" in p]


def test_a_real_id_shape_mismatch_still_raises_the_problem(tmp_path):
    """A junit id with no manifest counterpart even after stripping still reports ``id-shape mismatch``.

    Input class: a junit naming ``test_other`` (not in the manifest) for the manifest's ``test_0``, and ``test_1``
    present. Required so the repair does not hide a genuine mismatch. Killed by a pairing that suppresses the
    problem whenever a suffix is seen.
    """
    res = _aggregate_with_junit(tmp_path, [("test_other@g", ""), ("test_1", "")])
    assert res["counts"]["absent"] == 1
    assert res["unmatched_junit"] == ["tests/test_x.py::test_other"]
    assert [p for p in res["problems"] if "id-shape mismatch" in p]
