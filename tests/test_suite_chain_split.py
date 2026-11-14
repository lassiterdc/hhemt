"""A file whose tests depend on each other's tree output is kept whole by the heavy split (R-B2).

Array 1 put the four dependent tests of ``tests/test_synth_01_singlesim.py`` in four chunks, so a consumer found no
producer output. ``partition.chain_kept_whole`` keeps such a file whole: two or more of its nodes hold the same
FUNCTION-scoped tree-reusing builder (``retrieve_*(start_from_scratch=False)``), or one node holds a wiping builder
(``start_from_scratch=True``) and another a reusing builder of the same tree. A session-scoped builder is not a chain
builder. Every test runs over a fixture project in ``tmp_path``: no solver, no collection of the real suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hhemt.suite import _runner
from hhemt.suite import partition as P
from tests.test_suite_tier_selection import _env, _git

_CASES = (
    "class Cases:\n"
    "    @staticmethod\n"
    "    def retrieve_demo_test_case(start_from_scratch=False):\n"
    '        analysis_name = "demo_tree"\n'
    "    @staticmethod\n"
    "    def retrieve_other_test_case(start_from_scratch=False):\n"
    '        analysis_name = "other_tree"\n'
)
_PYTEST = "import pytest\n\n\n"


def _fixture(name: str, case: str, scratch: bool, scope: str | None = None) -> str:
    """Source of one fixture that builds ``retrieve_{case}_test_case`` with the given ``start_from_scratch``."""
    deco = "@pytest.fixture" + (f"(scope={scope!r})" if scope else "")
    call = f"Cases.retrieve_{case}_test_case(start_from_scratch={scratch})"
    return f"{deco}\ndef {name}():\n    return {call}\n\n\n"


_REUSING = _PYTEST + _fixture("demo_cached", "demo", False)
_WIPING = _PYTEST + _fixture("demo_wiped", "demo", True)
_SESSION = _PYTEST + _fixture("demo_session", "demo", False, scope="session")


def _project(tmp_path: Path, conftest: str, tests: dict[str, list[tuple[str, list[str]]]], support: dict | None = None):
    """Write a fixture project; return (files, by_file, closures) the way a collection would report them."""
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text(conftest, encoding="utf-8")
    (tmp_path / "tests" / "fixtures" / "cases.py").write_text(_CASES, encoding="utf-8")
    for rel, text in (support or {}).items():
        (tmp_path / rel).write_text(text, encoding="utf-8")
    by_file: dict[str, list[str]] = {}
    closures: dict[str, list[str]] = {}
    for rel, items in tests.items():
        (tmp_path / rel).write_text("".join(f"def {name}():\n    pass\n\n\n" for name, _ in items), encoding="utf-8")
        by_file[rel] = [f"{rel}::{name}" for name, _ in items]
        for name, fixtures in items:
            closures[f"{rel}::{name}"] = fixtures
    return sorted(by_file), by_file, closures


def _splits(tmp_path: Path, conftest: str, tests, support=None) -> tuple[set[str], dict[str, str]]:
    files, by_file, closures = _project(tmp_path, conftest, tests, support)
    return (
        P.node_split_candidates(files, by_file, closures, tmp_path),
        P.chain_kept_whole(files, by_file, closures, tmp_path),
    )


def test_four_tests_sharing_a_function_scoped_reusing_builder_are_kept_whole(tmp_path):
    """Wrong implementation it reds against: today's ``node_split_candidates``, which node-splits the file."""
    tests = {"tests/test_chain.py": [(f"test_{i}", ["demo_cached"]) for i in range(4)]}
    split, kept = _splits(tmp_path, _REUSING, tests)
    assert "tests/test_chain.py" not in split
    assert kept["tests/test_chain.py"].startswith("shared reusing builder demo_cached")


def test_a_wiping_builder_and_a_reusing_builder_of_one_tree_keep_the_file_whole(tmp_path):
    """Reds against a rule that counts only a SHARED reusing builder (each builder is held by one node here)."""
    tests = {"tests/test_chain.py": [("test_a", ["demo_wiped"]), ("test_b", ["demo_cached"])]}
    split, kept = _splits(tmp_path, _WIPING + _fixture("demo_cached", "demo", False), tests)
    assert "tests/test_chain.py" not in split
    assert kept["tests/test_chain.py"] == "wiping + reusing builder of one tree"


def test_a_wiping_and_a_reusing_builder_of_different_trees_do_not_chain(tmp_path):
    """Control for the same-tree clause: reds against a rule that keeps a file whole on any wiper plus any reuser."""
    conftest = _WIPING + _fixture("other_cached", "other", False)
    tests = {"tests/test_apart.py": [("test_a", ["demo_wiped"]), ("test_b", ["other_cached"])]}
    split, kept = _splits(tmp_path, conftest, tests)
    assert "tests/test_apart.py" in split
    assert kept == {}


def test_a_session_scoped_shared_builder_is_still_node_split(tmp_path):
    """Reds against a rule that counts session-scoped builders (testing-specialist F2's closure reading)."""
    tests = {"tests/test_session.py": [(f"test_{i}", ["demo_session"]) for i in range(3)]}
    split, kept = _splits(tmp_path, _SESSION, tests)
    assert "tests/test_session.py" in split
    assert kept == {}


def test_independent_tests_are_still_node_split(tmp_path):
    """Control: the rule keeps whole only a file with the dependency, not every file."""
    tests = {"tests/test_free.py": [("test_a", []), ("test_b", []), ("test_c", [])]}
    split, kept = _splits(tmp_path, _REUSING, tests)
    assert "tests/test_free.py" in split
    assert kept == {}


def test_a_reusing_builder_under_tests_fixtures_is_recognised(tmp_path):
    """Reds against a scan of ``conftest.py`` alone (software-engineering-specialist V1)."""
    tests = {"tests/test_chain.py": [("test_a", ["demo_cached"]), ("test_b", ["demo_cached"])]}
    support = {"tests/fixtures/builders.py": _REUSING}
    split, kept = _splits(tmp_path, "", tests, support)
    assert "tests/test_chain.py" not in split
    assert "tests/test_chain.py" in kept


def test_a_plan_records_and_prints_a_kept_whole_file_over_the_budget(tmp_path):
    """Reds against today's planner, which records no kept-whole file and prints no such line."""
    files, by_file, closures = _project(
        tmp_path, _REUSING, {"tests/test_chain.py": [(f"test_{i}", ["demo_cached"]) for i in range(4)]}
    )
    nodes = [n for f in files for n in by_file[f]]
    manifest = P.build_manifest(
        repo_root=tmp_path,
        node_ids=nodes,
        source_sha="deadbeef",
        run_id="r",
        closures=closures,
        heavy_split_budget_s=60.0,
        durations={"tests/test_chain.py": 500.0},
        tree_isolation_per_chunk=True,
    )
    assert manifest["kept_whole"] == [
        {
            "file": "tests/test_chain.py",
            "nodes": 4,
            "cost_s": 500.0,
            "reason": "shared reusing builder demo_cached",
        }
    ]
    assert len([c for c in manifest["chunks"] if "tests/test_chain.py" in c["files"]]) == 1
    assert sorted(manifest["chunks"][0]["node_ids"]) == sorted(nodes)
    assert _runner.kept_whole_lines(json.loads(json.dumps(manifest))) == [
        "kept_whole_max_s=500.0",
        "kept_whole_over_budget=tests/test_chain.py:500.0",
    ]


def test_a_kept_whole_file_under_the_budget_prints_the_max_line_only(tmp_path):
    """Control: no ``kept_whole_over_budget`` line when the file fits the budget, none at all when nothing is kept."""
    assert _runner.kept_whole_lines(
        {"kept_whole": [{"file": "a.py", "cost_s": 10.0}], "heavy_split_budget_s": 60.0}
    ) == ["kept_whole_max_s=10.0"]
    assert _runner.kept_whole_lines({"kept_whole": [], "heavy_split_budget_s": 60.0}) == []


def test_one_node_holding_a_wiping_and_a_reusing_builder_is_not_a_chain(tmp_path):
    """Reviewer arm. Reds against a rule that lets the wiping node and the reusing node be the SAME node.

    The shape asks for one node holding the wiping builder and ANOTHER holding the reusing builder of the tree: a
    single test that wipes and then reuses its own tree depends on no sibling's output, so its file stays splittable.
    """
    conftest = _WIPING + _fixture("demo_cached", "demo", False)
    both = ("test_both", ["demo_wiped", "demo_cached"])
    tests = {"tests/test_solo.py": [both, ("test_free", []), ("test_free2", [])]}
    split, kept = _splits(tmp_path, conftest, tests)
    assert "tests/test_solo.py" in split
    assert kept == {}


def test_a_kept_whole_file_costing_exactly_the_budget_prints_no_over_budget_line():
    """Reviewer arm. The shape prints ``kept_whole_over_budget`` for a file whose cost EXCEEDS B: equal does not."""
    manifest = {"kept_whole": [{"file": "a.py", "cost_s": 60.0}], "heavy_split_budget_s": 60.0}
    assert _runner.kept_whole_lines(manifest) == ["kept_whole_max_s=60.0"]


def test_the_real_plan_prints_the_kept_whole_lines_and_records_the_file(tmp_path):
    """The REAL planning driver over a stand-in project prints both lines and the manifest records the file.

    Input class: a heavy component (its tests reach one tree through a function-scoped reusing builder defined under
    ``tests/fixtures/``) holding a chain file of 4 tests whose ledger cost, 500 s, exceeds the 60 s budget. Required by
    the shape: the lines are printed by ``plan`` and not only produced by ``kept_whole_lines``. Killed by a ``plan``
    that does not print them, or prints them before the manifest exists (the helper would see no ``kept_whole``).
    """
    repo = tmp_path / "repo"
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "tests" / "conftest.py").write_text(_REUSING, encoding="utf-8")
    (repo / "tests" / "fixtures" / "cases.py").write_text(_CASES, encoding="utf-8")
    (repo / "tests" / "test_chain.py").write_text(
        "".join(f"def test_{i}(demo_cached):\n    pass\n\n\n" for i in range(4)), encoding="utf-8"
    )
    (repo / "tests" / "test_warm.py").write_text("def test_warm():\n    assert True\n", encoding="utf-8")
    shutil.copytree(
        Path(_runner.__file__).resolve().parents[1],
        repo / "src" / "hhemt",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    rows = [
        {
            "kind": "report",
            "nodeid": f"tests/test_chain.py::test_{i}",
            "when": "call",
            "outcome": "passed",
            "duration": 125.0,
        }
        for i in range(4)
    ]
    (ledger / "chunk-00.reports.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable]
    cmd += ["--runs-root", str(runs), "--warm-target", "tests/test_warm.py", "--declare-non-compile-bearing"]
    cmd += ["--isolate-trees-per-chunk", "--heavy-split-budget-min", "1", "--durations-from", str(ledger)]
    done = subprocess.run(cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    lines = done.stdout.splitlines()
    assert "kept_whole_max_s=500.0" in lines, done.stdout[-1500:]
    assert "kept_whole_over_budget=tests/test_chain.py:500.0" in lines, done.stdout[-1500:]
    manifest = json.loads((next(runs.iterdir()) / "manifest.json").read_text())
    assert manifest["kept_whole"] == [
        {"file": "tests/test_chain.py", "nodes": 4, "cost_s": 500.0, "reason": "shared reusing builder demo_cached"}
    ]


_Z = "tests/test_chain.py::test_z_produce"
_A = "tests/test_chain.py::test_a_consume"


def _ordered_manifest(tmp_path: Path, *, budget_s: float):
    """Plan a chain file defining ``test_z_produce`` before ``test_a_consume`` beside a second file on the same tree.

    The chain file is kept whole (two function-scoped reusing builders of one tree), so its two ids always land in one
    part; the second file makes the component splittable. Returns (manifest, written run dir)."""
    files, by_file, closures = _project(
        tmp_path,
        _REUSING,
        {
            "tests/test_chain.py": [("test_z_produce", ["demo_cached"]), ("test_a_consume", ["demo_cached"])],
            "tests/test_other.py": [("test_other", ["demo_cached"])],
        },
    )
    nodes = [n for f in files for n in by_file[f]]
    manifest = P.build_manifest(
        repo_root=tmp_path,
        node_ids=nodes,
        source_sha="deadbeef",
        run_id="r",
        closures=closures,
        heavy_split_budget_s=budget_s,
        durations={"tests/test_chain.py": 100.0, "tests/test_other.py": 100.0},
        tree_isolation_per_chunk=True,
    )
    run_dir = tmp_path / "run"
    P.write_manifest(manifest, run_dir)
    return manifest, run_dir


def _chain_chunk(manifest: dict) -> dict:
    """The one chunk whose ids include the chain file's."""
    holders = [c for c in manifest["chunks"] if _Z in c["node_ids"]]
    assert len(holders) == 1 and _A in holders[0]["node_ids"], "the chain file must stay whole in one part"
    return holders[0]


@pytest.mark.parametrize(("budget_s", "parts"), [(60.0, 2), (1000.0, 1)], ids=["two-parts", "one-part"])
def test_a_part_lists_a_chain_files_ids_in_definition_order_in_the_manifest_and_the_chunk_file(
    tmp_path, budget_s, parts
):
    """Property: the part holding a chain file lists ``test_z_produce`` before ``test_a_consume`` (definition order)
    in its ``node_ids`` and in ``chunks/NN.txt``, whether the component splits into two parts or stays one.

    Class: REGRESSION, red on today's ``split_heavy_component``, whose two returns sort the ids alphabetically, so the
    consumer ran before the producer (counted array 1: ``test_process_sim`` before ``test_run_sim``). Input classes:
    a component that splits (two parts) and one that does not (one part), each with the producer defined first.

    Why it is required, checkable against the code: chunk 3a.1 keeps a chain file whole in one chunk, which helps only
    if the chunk's list keeps the order tests were defined in; pytest runs the ids in the order it is handed them and
    ``write_manifest`` writes ``node_ids`` verbatim.

    What kills it: ``sorted`` in either return of ``split_heavy_component`` (each return is one arm). Control: the part
    count is asserted, so neither arm can pass by landing in the other shape.

    A second correct implementation under which it passes: one that orders each part by the index in a collected
    id list the planner passes, or one that keeps each file's ids in ``by_file`` order and files in ``files`` order."""
    manifest, run_dir = _ordered_manifest(tmp_path, budget_s=budget_s)
    assert len([c for c in manifest["chunks"] if c["kind"] == "heavy"]) == parts
    chunk = _chain_chunk(manifest)
    assert chunk["node_ids"].index(_Z) < chunk["node_ids"].index(_A)
    listed = (run_dir / "chunks" / f"{chunk['chunk_id']:02d}.txt").read_text().split()
    assert listed == chunk["node_ids"]
    assert manifest["collected"] == sorted(manifest["collected"])


def test_a_part_orders_ids_from_several_files_by_their_collected_index(tmp_path):
    """Property: a part holding ids of several files lists them in the order the collected list gave, which here is
    NOT alphabetical across files (``tests/test_other.py`` collected before ``tests/test_chain.py``).

    Class: NEW CAPABILITY, the cross-file choice: a part's rank is the index of each id in the ``node_ids`` list
    ``build_manifest`` received, so the file order is the collection's and not the files' names'.

    Why it is required, checkable against the code: ``build_manifest`` builds ``by_file`` in first-appearance order of
    the received list and ``split_heavy_component`` ranks ids by reading it in that order; ranking by file name or by
    id would put ``test_chain`` first.

    What kills it: a ``by_file`` keyed in sorted file order, or a rank built from names. Control: the same project
    with the alphabetical collected list orders the chain file first.

    A second correct implementation under which it passes: one that stable-sorts the part by the received list."""
    files, by_file, closures = _project(
        tmp_path,
        _REUSING,
        {
            "tests/test_chain.py": [("test_z_produce", ["demo_cached"]), ("test_a_consume", ["demo_cached"])],
            "tests/test_other.py": [("test_other", ["demo_cached"])],
        },
    )
    other = "tests/test_other.py::test_other"

    def part_for(collected: list[str]) -> list[str]:
        """The single heavy part's ids for a project whose collection reported `collected`, in that order."""
        m = P.build_manifest(
            repo_root=tmp_path,
            node_ids=collected,
            source_sha="deadbeef",
            run_id="r",
            closures=closures,
            heavy_split_budget_s=1000.0,
            durations={"tests/test_chain.py": 100.0, "tests/test_other.py": 100.0},
            tree_isolation_per_chunk=True,
        )
        return next(c["node_ids"] for c in m["chunks"] if c["kind"] == "heavy")

    assert part_for([_Z, _A, other]) == [_Z, _A, other], "control: the alphabetical-by-file collection"
    assert part_for([other, _Z, _A]) == [other, _Z, _A]
