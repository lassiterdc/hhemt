"""The tail element (ruling R-A): the destructive compile tests run as ONE last chunk, after the array.

Three tests of ``tests/test_synth_00_compile_models.py`` rebuild the compiled tier that every array chunk reads, so
running them inside an array chunk is a write/read race on shared state (defect A). The planner moves the ids named by
``policy.TAIL_NODE_IDS`` (or by the file ``HHEMT_SUITE_TAIL_NODE_IDS_FILE`` names) into one last chunk with
``covers: "tail"`` and ``workers: 1``; the aggregator merges it like every chunk, reports its state as ``tail=`` on the
verdict line, and leaves it out of the cross-chunk shared-tree joins only when the recorded wall-clock bounds say it
ran after every array chunk. Every test below runs over a fixture project or a fixture run directory: pytest only
collects stand-in tests, and nothing here reaches a solver.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.suite._runner as R
from hhemt.suite import aggregate as A
from hhemt.suite import partition as P
from tests.test_suite_tier_selection import _env, _git, _project

# --------------------------------------------------------------------------------------------------------------
# build_manifest over a fixture project
# --------------------------------------------------------------------------------------------------------------

_TAIL = "tests/test_destroy.py::test_rebuild"


def _corpus(tmp_path: Path):
    """A destructive file (the tail test plus two neighbours) and a reader file, all reaching one tree."""
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text(
        'def retrieve_demo_test_case():\n    analysis_name = "demo_tree"\n', encoding="utf-8"
    )
    body = "def test_rebuild(retrieve_demo_test_case):\n    pass\n\n\ndef test_a(retrieve_demo_test_case):\n    pass\n"
    (tmp_path / "tests" / "test_destroy.py").write_text(
        body + "\n\ndef test_b(retrieve_demo_test_case):\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_reader.py").write_text(
        "def test_read(retrieve_demo_test_case):\n    pass\n", encoding="utf-8"
    )
    nodes = [
        _TAIL,
        "tests/test_destroy.py::test_a",
        "tests/test_destroy.py::test_b",
        "tests/test_reader.py::test_read",
    ]
    closures = {n: ["retrieve_demo_test_case"] for n in nodes}
    closures[_TAIL] = ["retrieve_demo_test_case", "tritonswmm_cpu_compiled"]
    return nodes, closures


def _build(tmp_path: Path, **kw):
    nodes, closures = _corpus(tmp_path)
    return nodes, P.build_manifest(
        repo_root=tmp_path, node_ids=nodes, source_sha="deadbeef", run_id="r", closures=closures, **kw
    )


def test_a_tail_id_leaves_every_array_chunk_and_fills_exactly_one_last_chunk(tmp_path):
    """Input class: a destructive file with a tail id and two neighbours, plus a reader of the same tree.

    Required so the destructive test never shares a chunk with a reader of the tree it rebuilds. Killed by today's
    ``build_manifest`` (no ``tail_node_ids`` parameter at all; a build that ignores the list leaves the id in an array
    chunk). The neighbours of the tail id stay in the array: the tail takes the NODE, not its file.
    """
    nodes, m = _build(tmp_path, tail_node_ids=[_TAIL])
    *array, tail = m["chunks"]
    assert tail["node_ids"] == [_TAIL] and tail["chunk_id"] == len(m["chunks"]) - 1 == m["tail_chunk"]
    assert all(_TAIL not in c["node_ids"] for c in array)
    assert {"tests/test_destroy.py::test_a", "tests/test_destroy.py::test_b"} <= {
        n for c in array for n in c["node_ids"]
    }
    assert (tail["kind"], tail["workers"], tail["covers"], tail["files"]) == (
        "heavy",
        1,
        "tail",
        ["tests/test_destroy.py"],
    )
    assert all("covers" not in c for c in array)


def test_the_tail_chunk_derives_its_expected_fixtures_from_its_nodes_closures(tmp_path):
    """Input class: the tail node's closure holds a recorded session fixture its neighbours do not.

    Required so a tail that loses its precondition reads VOID rather than a bare failure
    (software-engineering-specialist V2). Killed by a tail chunk appended AFTER the enrichment loop, which
    carries no ``expected_fixtures`` at all.
    """
    _, m = _build(tmp_path, tail_node_ids=[_TAIL])
    assert m["chunks"][-1]["expected_fixtures"] == ["tritonswmm_cpu_compiled"]
    assert all("tritonswmm_cpu_compiled" not in c["expected_fixtures"] for c in m["chunks"][:-1])


def test_collected_is_the_full_list_and_the_chunks_cover_it_exactly_once(tmp_path):
    """Input class: a tail of one id out of four collected.

    Required by the exact-cover contract: ``collected`` keeps the tail id and every collected id is in exactly
    one chunk.
    Killed by a ``build_manifest`` that hands the array ids alone to ``_assert_covers`` (it would then not notice a
    neighbouring node dropped from every chunk, and ``collected`` would shrink).
    """
    nodes, m = _build(tmp_path, tail_node_ids=[_TAIL])
    assigned = [n for c in m["chunks"] for n in c["node_ids"]]
    assert m["collected"] == sorted(nodes)
    assert sorted(assigned) == sorted(nodes) and len(assigned) == len(set(assigned))


def _tail_only_corpus(tmp_path: Path):
    """The destructive file holds ONLY the tail test, beside a reader file reaching the same tree."""
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text(
        'def retrieve_demo_test_case():\n    analysis_name = "demo_tree"\n', encoding="utf-8"
    )
    (tmp_path / "tests" / "test_destroy.py").write_text(
        "def test_rebuild(retrieve_demo_test_case):\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_reader.py").write_text(
        "def test_read(retrieve_demo_test_case):\n    pass\n\n\ndef test_read2(retrieve_demo_test_case):\n    pass\n",
        encoding="utf-8",
    )
    nodes = [_TAIL, "tests/test_reader.py::test_read", "tests/test_reader.py::test_read2"]
    closures = {n: ["retrieve_demo_test_case"] for n in nodes}
    closures[_TAIL] = ["retrieve_demo_test_case", "tritonswmm_cpu_compiled"]
    return nodes, closures


@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "heavy-split"])
def test_a_tail_file_made_only_of_tail_ids_plans(tmp_path, split):
    """Input class: every test of ``tests/test_destroy.py`` is a tail id (the file is no key of the array's per-file
    lists), planned unsplit and with a heavy split.

    Required: ``build_manifest`` must not re-derive the tail chunk's ``node_ids`` from per-file lists that no longer
    hold them. Killed by ``c.setdefault("node_ids", [... by_file[f] ...])``, whose default is evaluated before the call
    and raises ``KeyError`` for the tail's file (measured by the 4a.1 smoke run). Also pins: the tail holds exactly the
    ids, the file is in no array chunk, exact cover holds, and exposure and ``expected_fixtures`` still derive.
    """
    nodes, closures = _tail_only_corpus(tmp_path)
    kw = (
        {"heavy_split_budget_s": 60.0, "durations": {"tests/test_reader.py": 90.0}, "tree_isolation_per_chunk": True}
        if split
        else {}
    )
    m = P.build_manifest(
        repo_root=tmp_path,
        node_ids=nodes,
        source_sha="deadbeef",
        run_id="r",
        closures=closures,
        tail_node_ids=[_TAIL],
        **kw,
    )
    *array, tail = m["chunks"]
    assert tail["node_ids"] == [_TAIL] and m["tail_chunk"] == tail["chunk_id"]
    assert all("tests/test_destroy.py" not in c["files"] and _TAIL not in c["node_ids"] for c in array)
    assigned = [n for c in m["chunks"] for n in c["node_ids"]]
    assert sorted(assigned) == sorted(nodes) and m["collected"] == sorted(nodes)
    assert tail["expected_fixtures"] == ["tritonswmm_cpu_compiled"]
    assert all(m["tail_chunk"] not in rec["chunks"] for rec in m["shared_tree_exposure"])


def test_tail_ids_spread_over_several_files_with_one_made_only_of_tail_ids_plan(tmp_path):
    """Input class: two tail files, ``test_destroy.py`` holding only its tail id and ``test_compile.py`` holding one
    tail id beside two array tests (the 4a.1 smoke fixture's shape).

    Required: the tail chunk holds both ids and nothing else, neither the all-tail file nor the partly-tail file's
    tail id is in an array chunk, the partly-tail file's array tests stay in the array, and exact cover holds. Killed
    by the pre-fix ``setdefault`` (``KeyError`` on the all-tail file) and by a fix that special-cases only ONE tail
    file (the all-tail file or the partly-tail file is then mishandled).
    """
    nodes, closures = _tail_only_corpus(tmp_path)
    (tmp_path / "tests" / "test_compile.py").write_text(
        "def test_c1(retrieve_demo_test_case):\n    pass\n\n\ndef test_c2(retrieve_demo_test_case):\n    pass\n\n\n"
        "def test_c3(retrieve_demo_test_case):\n    pass\n",
        encoding="utf-8",
    )
    tail2 = "tests/test_compile.py::test_c1"
    extra = [tail2, "tests/test_compile.py::test_c2", "tests/test_compile.py::test_c3"]
    closures.update({n: ["retrieve_demo_test_case"] for n in extra})
    m = P.build_manifest(
        repo_root=tmp_path,
        node_ids=nodes + extra,
        source_sha="deadbeef",
        run_id="r",
        closures=closures,
        tail_node_ids=[_TAIL, tail2],
    )
    *array, tail = m["chunks"]
    assert sorted(tail["node_ids"]) == sorted([_TAIL, tail2]) and m["tail_chunk"] == tail["chunk_id"]
    array_ids = {n for c in array for n in c["node_ids"]}
    assert not array_ids & {_TAIL, tail2}
    assert {"tests/test_compile.py::test_c2", "tests/test_compile.py::test_c3"} <= array_ids
    assert all("tests/test_destroy.py" not in c["files"] for c in array)
    assigned = [n for c in m["chunks"] for n in c["node_ids"]]
    assert sorted(assigned) == sorted(nodes + extra)


def test_a_tail_that_holds_every_collected_id_refuses(tmp_path):
    """Input class: every collected id is a tail id. Required: nothing is left for the array, so the plan refuses
    with a message saying so. Killed by a ``build_manifest`` without the guard, which plans an array of zero chunks
    (or fails later with an unrelated error) and never says why."""
    nodes, closures = _tail_only_corpus(tmp_path)
    with pytest.raises(P.PartitionDriftError, match="nothing is left for the array"):
        P.build_manifest(
            repo_root=tmp_path,
            node_ids=nodes,
            source_sha="deadbeef",
            run_id="r",
            closures=closures,
            tail_node_ids=list(nodes),
        )


def test_a_tail_file_with_array_neighbours_plans_as_before(tmp_path):
    """Control: the tail file keeps its two array neighbours in the array, the tail takes only its own id."""
    _, m = _build(tmp_path, tail_node_ids=[_TAIL])
    array_ids = {n for c in m["chunks"][:-1] for n in c["node_ids"]}
    assert {"tests/test_destroy.py::test_a", "tests/test_destroy.py::test_b"} <= array_ids
    assert any("tests/test_destroy.py" in c["files"] for c in m["chunks"][:-1])


def test_the_shared_tree_exposure_is_computed_over_the_array_chunks_only(tmp_path):
    """The tail runs after every array chunk, so its chunk id must not appear among the exposed chunks.

    Input class: a corpus whose destructive file and reader file reach one tree, split into several array chunks plus
    a tail. Required by the approved row (the exposure is over the array chunks because the tail is never concurrent
    with one). Killed by an exposure computed over ALL chunks (the tail's file mentions the tree its neighbours' chunk
    reads). The first assertion is the control: the exposure is non-empty among the array chunks.
    """
    _, m = _build(
        tmp_path,
        tail_node_ids=[_TAIL],
        heavy_split_budget_s=60.0,
        durations={"tests/test_destroy.py": 90.0, "tests/test_reader.py": 1.0},
        tree_isolation_per_chunk=True,
    )
    assert m["shared_tree_exposure"], "control: the corpus must expose a cross-chunk tree among the array chunks"
    assert all(m["tail_chunk"] not in rec["chunks"] for rec in m["shared_tree_exposure"]), m["shared_tree_exposure"]


def test_the_exact_cover_check_is_handed_the_full_collected_list_and_every_assigned_id(tmp_path, monkeypatch):
    """Pins what ``_assert_covers`` receives: the FULL collected list, tail id included, and every chunk's ids.

    Input class: a plan with one tail id. Required (testing-specialist F4): exact cover over the array ids alone would
    not notice a tail chunk that lost an id. Killed by a ``build_manifest`` that hands the array ids alone to the check.
    """
    seen: dict = {}
    real = P._assert_covers
    monkeypatch.setattr(P, "_assert_covers", lambda **kw: (seen.update(kw), real(**kw))[1])
    nodes, _ = _build(tmp_path, tail_node_ids=[_TAIL])
    assert sorted(seen["collected"]) == sorted(nodes) and _TAIL in seen["collected"]
    assert sorted(seen["assigned"]) == sorted(nodes)


def test_the_exact_cover_check_sees_a_node_dropped_next_to_the_tail(tmp_path, monkeypatch):
    """Control for the arm above: the cover check is fed the FULL list, so a dropped neighbour is caught.

    Mutation made in the test: the splitter loses one neighbouring node. A ``build_manifest`` that checked cover over
    the array ids alone would see no gap (the dropped node is in the array list's complement of nothing) and plan.
    """
    real = P.split_heavy_component

    def lossy(files, by_file, durations, budget, node_split=None):
        parts = real(files, by_file, durations, budget, node_split=node_split)
        return [[n for n in part if n != "tests/test_destroy.py::test_b"] for part in parts]

    monkeypatch.setattr(P, "split_heavy_component", lossy)
    with pytest.raises(P.PartitionDriftError, match="test_destroy.py::test_b"):
        _build(
            tmp_path,
            tail_node_ids=[_TAIL],
            heavy_split_budget_s=60.0,
            durations={"tests/test_destroy.py": 90.0, "tests/test_reader.py": 1.0},
            tree_isolation_per_chunk=True,
        )


def test_a_tail_id_that_is_not_collected_refuses(tmp_path):
    """Must-fire: a renamed tail test would silently return to an array chunk, so an id not collected refuses."""
    with pytest.raises(P.PartitionDriftError, match="tail node id"):
        _build(tmp_path, tail_node_ids=["tests/test_destroy.py::test_renamed"])


def test_without_a_tail_the_manifest_is_the_manifest_it_always_was(tmp_path):
    """Input class: the same corpus with ``tail_node_ids`` absent, ``None`` and empty.

    Required by the byte-for-byte clause: no ``tail_chunk`` key, no ``covers``/``workers`` on any chunk, and the three
    spellings of "no tail" plan identically. Killed by a build that always adds ``tail_chunk`` (even ``None``).
    """
    _, absent = _build(tmp_path / "a")
    _, none = _build(tmp_path / "b", tail_node_ids=None)
    _, empty = _build(tmp_path / "c", tail_node_ids=[])
    assert json.dumps(absent, sort_keys=True) == json.dumps(none, sort_keys=True) == json.dumps(empty, sort_keys=True)
    assert "tail_chunk" not in absent and all(not {"covers", "workers"} & set(c) for c in absent["chunks"])
    assert sorted(absent) == [
        "cheap_bins",
        "chunk_count",
        "chunks",
        "collected",
        "declared_complement",
        "heavy_split_budget_s",
        "heavy_split_durations_covered",
        "heavy_split_imputed_seconds_per_node",
        "kept_whole",
        "run_id",
        "shared_tree_exposure",
        "signals",
        "source_sha",
        "tree_isolation_per_chunk",
    ]


# --------------------------------------------------------------------------------------------------------------
# the real plan driver
# --------------------------------------------------------------------------------------------------------------


def _plan(
    tmp_path: Path, tail_file: str | None = None, extra_files: dict[str, str] | None = None, builtin: bool = False
):
    """Run the real planning driver over the stand-in project; return (stdout, returncode, run dir or None)."""
    repo = _project(tmp_path)
    for rel, text in (extra_files or {}).items():
        (repo / rel).write_text(text, encoding="utf-8")
    if extra_files:
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "more")
    runs = tmp_path / "runs"
    runs.mkdir()
    env = _env(tmp_path)
    if tail_file is not None:
        env["HHEMT_SUITE_TAIL_NODE_IDS_FILE"] = tail_file
    if builtin:
        env.pop("HHEMT_SUITE_TAIL_NODE_IDS_FILE", None)
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--toolkit", str(repo), "--python", sys.executable]
    cmd += ["--runs-root", str(runs), "--warm-target", "tests/test_warm.py", "--declare-non-compile-bearing"]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, check=False)
    run_dir = next(iter(runs.iterdir()), None)
    return done, run_dir


def _named(tmp_path: Path, *ids: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tail_ids.txt"
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return str(path)


def test_the_plan_moves_a_named_destructive_test_into_a_last_chunk_and_prints_it(tmp_path):
    """Input class: a stand-in project whose ``tests/test_stand.py::test_solver`` is named a tail id through the file.

    Required so ``plan`` passes the list to ``build_manifest``, records the chunk and prints ``tail_chunk={id}`` beside
    ``chunk_count=``. Killed by today's ``plan`` (the id stays in an array chunk and no such line is printed).
    """
    tail = "tests/test_stand.py::test_solver"
    done, run_dir = _plan(tmp_path, _named(tmp_path, tail))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    last = manifest["chunk_count"] - 1
    assert manifest["tail_chunk"] == last and manifest["chunks"][last]["node_ids"] == [tail]
    assert [c for c in manifest["chunks"] if tail in c["node_ids"]] == [manifest["chunks"][last]]
    assert f"tail_chunk={last}" in done.stdout.splitlines()
    assert manifest["tail_source"] == str(tmp_path / "tail_ids.txt")
    assert manifest["collected"] == sorted(manifest["collected"]) and tail in manifest["collected"]


def test_an_empty_tail_list_records_no_tail_chunk_and_prints_none(tmp_path):
    """Input class: the opt-out, ``HHEMT_SUITE_TAIL_NODE_IDS_FILE`` naming an empty list.

    Required so a stand-in project can plan without the harness tail. Killed by a ``plan`` that falls back to the
    built-in list on an empty file.
    """
    done, run_dir = _plan(tmp_path, _named(tmp_path))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "tail_chunk" not in manifest and all("covers" not in c for c in manifest["chunks"])
    assert "tail_chunk=none" in done.stdout.splitlines()
    assert manifest["tail_source"] == str(tmp_path / "tail_ids.txt")


_PLAN_KEYS_AT_0B13FF95 = [
    "chunk_count",
    "chunk_workers",
    "chunks",
    "cheap_bins",
    "collected",
    "compile_precondition",
    "declared_complement",
    "heavy_split_budget_s",
    "heavy_split_durations_covered",
    "heavy_split_imputed_seconds_per_node",
    "keep_failed_trees",
    "kept_whole",
    "run_id",
    "select_tier",
    "shared_tree_exposure",
    "signals",
    "source_sha",
    "tier_address",
    "tree_isolation_per_chunk",
    "warm",
]


def test_the_empty_opt_out_plans_the_manifest_it_always_planned(tmp_path):
    """Input class: ``HHEMT_SUITE_TAIL_NODE_IDS_FILE`` naming an EMPTY file, over the stand-in project.

    Required by the byte-for-byte clause (the manifest as of ``0b13ff95``): the only new key is ``tail_source``, there
    is no ``tail_chunk`` and no chunk carries ``covers`` or ``workers``, and ``tail_chunk=none`` is printed. Killed by a
    plan that falls back to the built-in list on an empty file, or that always records ``tail_chunk``.
    """
    done, run_dir = _plan(tmp_path, _named(tmp_path))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert sorted(manifest) == sorted([*_PLAN_KEYS_AT_0B13FF95, "tail_source"])
    assert all(not {"covers", "workers"} & set(c) for c in manifest["chunks"])
    assert "tail_chunk=none" in done.stdout.splitlines()


_BUILTIN_REMEDY = "HHEMT_SUITE_TAIL_NODE_IDS_FILE"
_COMPILE_FILE = "tests/test_synth_00_compile_models.py"


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        pytest.param("whole-file-absent", None, id="the-whole-tail-file-is-not-collected"),
        pytest.param(
            "id-renamed",
            {_COMPILE_FILE: "def test_compile_swmm():\n    pass\n\n\ndef test_compile_triton_only():\n    pass\n"},
            id="one-id-renamed-inside-the-file",
        ),
    ],
)
def test_the_builtin_tail_is_strict_and_the_refusal_names_the_remedy(tmp_path, label, extra):
    """Must-fire (DR-9 option 1): a built-in tail id that is not collected refuses the plan, whatever the cause.

    Input class: a stand-in project planned with the built-in list, (a) collecting none of the tail's file, (b)
    collecting the file with ``test_compile_tritonswmm`` missing. Required so a rename, move or removal fails loudly
    instead of returning a destructive test to an array chunk. Killed by a plan that plans no tail when the file is
    not collected (the earlier conditional), and by a plan that drops a missing id. The message names
    ``HHEMT_SUITE_TAIL_NODE_IDS_FILE`` so a fixture project knows the remedy.
    """
    done, run_dir = _plan(tmp_path, None, extra, builtin=True)
    out = done.stdout + done.stderr
    assert done.returncode != 0, label
    assert "tail node id" in out and _BUILTIN_REMEDY in out, out[-1500:]


def test_an_id_renamed_in_a_named_list_refuses_the_plan(tmp_path):
    """Must-fire: a list NAMED through the environment is strict too (an id that is not collected refuses)."""
    done, _ = _plan(tmp_path, _named(tmp_path, "tests/test_stand.py::test_renamed"))
    assert done.returncode != 0 and "tail node id" in (done.stdout + done.stderr)


def _aggregate_cli(run_dir: Path) -> str:
    cmd = [sys.executable, "-m", "hhemt.suite.aggregate", "--run-dir", str(run_dir), "--allow-not-green"]
    done = subprocess.run(
        cmd, env=_env(run_dir.parent.parent), capture_output=True, text=True, timeout=240, check=False
    )
    return next(ln for ln in done.stdout.splitlines() if ln.startswith("verdict="))


def test_the_verdict_line_differs_from_todays_only_by_the_tail_field(tmp_path):
    """Input class: a plan with an empty tail aggregated with no chunk records, and one with a tail.

    Required by the verdict-line contract: ``tail=none`` is the only addition when there is no tail, and a planned tail
    that never reported reads ``tail=MISSING`` (never absent from the line). Killed by an aggregator whose line carries
    no ``tail=`` field.
    """
    _, run_dir = _plan(tmp_path / "none", _named(tmp_path / "none"))
    line = _aggregate_cli(run_dir)
    head, _, tail = line.rpartition(" tail=")
    assert tail == "none" and head.rpartition(" mechanism=")[2] in {"readable", "UNREADABLE"}
    assert " tail=" not in head
    _, run_dir2 = _plan(tmp_path / "some", _named(tmp_path / "some", "tests/test_stand.py::test_solver"))
    assert _aggregate_cli(run_dir2).endswith(" tail=MISSING")


# --------------------------------------------------------------------------------------------------------------
# the real chunk driver
# --------------------------------------------------------------------------------------------------------------

_SLEEPER = """\
import os
import time


def test_{name}():
    path = os.environ["TAIL_SENTINEL"]
    with open(path, "a") as fh:
        fh.write("start {name} %f\\n" % time.time())
    time.sleep(2.5)
    with open(path, "a") as fh:
        fh.write("end {name} %f\\n" % time.time())
"""


def _stand_in_repo(tmp_path: Path, names: tuple[str, ...]) -> tuple[Path, list[str]]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    for name in names:
        (repo / f"test_{name}.py").write_text(_SLEEPER.format(name=name))
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo, [f"test_{n}.py::test_{n}" for n in names]


def _run_real_chunk(tmp_path: Path, names: tuple[str, ...], chunk: dict) -> dict:
    """Run the real chunk driver at manifest ``chunk_workers`` 2 over stand-in sleepers; return intervals and status."""
    repo, ids = _stand_in_repo(tmp_path, names)
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    (run_dir / "chunks" / "00.txt").write_text("\n".join(ids) + "\n")
    chunk = {"chunk_id": 0, "kind": "heavy", "files": [i.split("::")[0] for i in ids], "node_ids": ids, **chunk}
    manifest = {"run_id": "tl", "chunk_count": 1, "chunk_workers": 2, "chunks": [chunk]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    sentinel = tmp_path / "sentinel.txt"
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env["PYTHONPATH"] = str(Path(R.__file__).resolve().parents[2])
    env["TAIL_SENTINEL"] = str(sentinel)
    env["TMPDIR"] = str(tmp_path)
    cmd = [sys.executable, "-m", "hhemt.suite._runner", "--chunk", "0", "--run-dir", str(run_dir)]
    cmd += ["--toolkit", str(repo), "--python", sys.executable]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, check=False)
    marks: dict[str, dict[str, float]] = {}
    for line in sentinel.read_text().splitlines():
        kind, name, stamp = line.split()
        marks.setdefault(name, {})[kind] = float(stamp)
    status = json.loads((run_dir / "chunk-00.status.json").read_text())
    assert status["pytest_exit"] == 0, done.stdout[-1500:] + done.stderr[-1500:]
    return {"intervals": marks, "status": status}


def _overlap(intervals: dict[str, dict[str, float]]) -> bool:
    spans = sorted((v["start"], v["end"]) for v in intervals.values())
    return any(b[0] < a[1] for a, b in zip(spans, spans[1:], strict=False))


def test_a_tail_chunk_runs_its_tests_one_at_a_time_even_at_manifest_workers_2(tmp_path):
    """Input class: a tail of two stand-in destructive tests in no shared ``xdist_group``, manifest ``chunk_workers`` 2.

    Required so the tail's tests never run side by side (they rebuild shared state). Killed by today's runner, whose
    command builds ``-n 2 --dist loadgroup`` for every chunk and puts the two tests on two workers at once. Controls:
    the same two tests in a chunk that carries no ``workers`` DO overlap (so the sentinel instrument can see an
    overlap), and a one-test tail trivially cannot.
    """
    tail = _run_real_chunk(tmp_path / "tail", ("a", "b"), {"workers": 1, "covers": "tail"})
    assert len(tail["intervals"]) == 2 and not _overlap(tail["intervals"]), tail["intervals"]
    control = _run_real_chunk(tmp_path / "ctl", ("a", "b"), {})
    assert _overlap(control["intervals"]), control["intervals"]
    solo = _run_real_chunk(tmp_path / "solo", ("a",), {"workers": 1, "covers": "tail"})
    assert len(solo["intervals"]) == 1


def test_a_real_chunks_status_records_started_at_and_ended_at(tmp_path):
    """Input class: any real chunk run. Required so the aggregator has run-order evidence.

    Killed by today's ``run_chunk``, whose status carries neither field. The values are UTC ISO-8601 with milliseconds
    and ``started_at <= ended_at``.
    """
    status = _run_real_chunk(tmp_path, ("a",), {"workers": 1})["status"]
    assert "T" in status["started_at"] and status["started_at"].endswith("+00:00")
    first, last = A._parse_instant(status["started_at"]), A._parse_instant(status["ended_at"])
    assert first is not None and last is not None and first <= last
    assert len(status["started_at"].rsplit(".", 1)[1].split("+")[0]) == 3


@pytest.mark.parametrize(("own", "expected_n"), [(1, False), (None, True)])
def test_the_tail_chunks_command_carries_no_n_and_no_dist_at_manifest_workers_8(tmp_path, monkeypatch, own, expected_n):
    """The command ``run_chunk`` builds for a chunk with ``workers: 1`` has no ``-n`` and no ``--dist`` at K=8.

    Killed by today's runner (``-n 8 --dist loadgroup`` for every chunk). Control: a chunk carrying no ``workers``
    still gets ``-n 8 --dist loadgroup``.
    """
    import argparse

    repo, ids = _stand_in_repo(tmp_path, ("a",))
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    (run_dir / "chunks" / "00.txt").write_text(ids[0] + "\n")
    chunk = {"chunk_id": 0, "kind": "heavy", "files": ["test_a.py"], "node_ids": ids}
    if own is not None:
        chunk["workers"] = own
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "tl", "chunk_count": 1, "chunk_workers": 8, "chunks": [chunk]})
    )
    seen: list[list[str]] = []
    monkeypatch.setattr(R, "_run_pytest_child", lambda cmd, cwd, env, **kw: seen.append(cmd) or 0)
    ns = argparse.Namespace(run_dir=str(run_dir), chunk=0, n_chunks=None, toolkit=str(repo), python=sys.executable)
    R.run_chunk(ns)
    cmd = seen[0]
    assert ("-n" in cmd and "--dist" in cmd) is expected_n
    if expected_n:
        assert cmd[cmd.index("-n") + 1] == "8"


# --------------------------------------------------------------------------------------------------------------
# the aggregator over fixture run directories
# --------------------------------------------------------------------------------------------------------------

_SHA = "b" * 40
_TREE = "/cache/hhemt/synthetic_test_runs/main/synth_multi_sim"
_ARRAY_END = "2026-11-03T10:00:00.000+00:00"
_TAIL_NODES = [f"tests/test_tail.py::test_t{i}" for i in range(3)]


def _entry(*, written: bool) -> dict:
    return {
        "label": "synth_multi_sim",
        "root_shape": "shared",
        "is_dir": True,
        "has_version_stamp": True,
        "existed_at_first_open": True,
        "exists_at_end": True,
        "read": True,
        "written": written,
        "write_samples": ["triton/build_tritonswmm_cpu/compilation.log"] if written else [],
    }


def _status(cid: int, *, exit_code: int = 0, fixtures=(), times=None, entries=None) -> dict:
    out = {
        "chunk_id": cid,
        "run_id": "FAKE",
        "pytest_exit": exit_code,
        "node_count": 2,
        "source_sha_start": _SHA,
        "source_sha_end": _SHA,
        "dirty_start": False,
        "dirty_end": False,
        "resolved_hhemt": {"pytest": "/x/hhemt/__init__.py", "sibling": "/x/hhemt/__init__.py"},
        "session_fixtures_ok": list(fixtures),
        "observed_trees": [],
        "written_trees": [],
        "reach_entries": entries or {},
        "slurm_state": "COMPLETED",
    }
    if times:
        out.update(times)
    return out


def _junit(nodes: list[str], failing: tuple[str, ...] = ()) -> str:
    cases = []
    for node in nodes:
        file, name = node.split("::")
        body = "<failure message='boom'/>" if node in failing else ""
        cases.append(
            f'<testcase classname="{file[:-3].replace("/", ".")}" file="{file}" name="{name}" time="0.1">'
            f"{body}</testcase>"
        )
    return f'<testsuites><testsuite name="pytest" tests="{len(nodes)}">{"".join(cases)}</testsuite></testsuites>'


def _fixture_run(
    root: Path,
    *,
    tail_status="ok",
    tail_fail=False,
    tail_fixtures=("tritonswmm_cpu_compiled",),
    tail_times=None,
    array_times=None,
    tail_entry=None,
    declared_complement=(),
) -> list[str]:
    """Write a run directory: two array chunks (two nodes each) reading one tree, and a tail chunk of three nodes.

    ``tail_status`` is ``"ok"`` (a status and a junit), or ``None`` (the tail never reported). ``array_times`` maps a
    chunk id to its ``{started_at, ended_at}`` record; absent means the status carries no times.
    """
    array_nodes = [f"tests/test_x.py::test_{i}" for i in range(4)]
    nodes = array_nodes + _TAIL_NODES
    chunks = [
        {
            "chunk_id": c,
            "kind": "heavy",
            "files": ["tests/test_x.py"],
            "node_ids": array_nodes[2 * c : 2 * c + 2],
            "expected_fixtures": [],
        }
        for c in range(2)
    ]
    chunks.append(
        {
            "chunk_id": 2,
            "kind": "heavy",
            "files": ["tests/test_tail.py"],
            "node_ids": _TAIL_NODES,
            "workers": 1,
            "covers": "tail",
            "expected_fixtures": ["tritonswmm_cpu_compiled"],
        }
    )
    manifest = {
        "run_id": "FAKE",
        "source_sha": _SHA,
        "collected": nodes,
        "chunk_count": 3,
        "cheap_bins": 1,
        "chunks": chunks,
        "shared_tree_exposure": [],
        "declared_complement": list(declared_complement),
        "tail_chunk": 2,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for c in range(2):
        times = (array_times or {}).get(c)
        (root / f"chunk-{c:02d}.status.json").write_text(
            json.dumps(_status(c, times=times, entries={_TREE: _entry(written=False)})), encoding="utf-8"
        )
        (root / f"chunk-{c:02d}.junit.xml").write_text(_junit(array_nodes[2 * c : 2 * c + 2]), encoding="utf-8")
    if tail_status is not None:
        entries = {_TREE: tail_entry if tail_entry is not None else _entry(written=True)}
        status = _status(2, exit_code=1 if tail_fail else 0, fixtures=tail_fixtures, times=tail_times, entries=entries)
        (root / "chunk-02.status.json").write_text(json.dumps(status), encoding="utf-8")
        failing = (_TAIL_NODES[0],) if tail_fail else ()
        (root / "chunk-02.junit.xml").write_text(_junit(_TAIL_NODES, failing), encoding="utf-8")
    return nodes


def _times(started: str, ended: str) -> dict:
    return {"started_at": started, "ended_at": ended}


_ARRAY_OK = {
    0: _times("2026-11-03T09:50:00.000+00:00", _ARRAY_END),
    1: _times("2026-11-03T09:51:00.000+00:00", _ARRAY_END),
}


def test_the_tail_chunks_tests_are_counted_passed_and_never_absent(tmp_path):
    """Input class: a tail chunk whose junit holds the three tail ids. Killed by an aggregator skipping the tail."""
    _fixture_run(tmp_path, tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"), array_times=_ARRAY_OK)
    r = A.aggregate(tmp_path, scope="array")
    assert r["counts"]["absent"] == 0 and r["counts"]["covered"] == 7
    assert set(_TAIL_NODES) <= set(A.parse_junit(tmp_path / "chunk-02.junit.xml"))
    assert r["verdict"] == "GREEN" and r["tail"] == "PASS" and r["verdict_line"].endswith(" tail=PASS"), r["problems"]


def test_a_missing_tail_is_not_green_and_the_scope_stays_array(tmp_path):
    """Input class: the tail never reported. Killed by a verdict that ignores the tail (the three tests would be ABSENT
    in the count only, with no ``tail=`` state), or that lets a union request through on a tail it never saw."""
    _fixture_run(tmp_path, tail_status=None, array_times=_ARRAY_OK)
    r = A.aggregate(tmp_path, scope="union")
    assert r["verdict"] == "NOT-GREEN" and r["scope"] == "array" and r["tail"] == "MISSING"
    assert " scope=array " in r["verdict_line"] and r["verdict_line"].endswith(" tail=MISSING")
    assert r["counts"]["absent"] == 3


def test_a_failed_tail_is_not_green(tmp_path):
    """Input class: the tail's first test fails. Killed by an aggregator merging the junit but not the state."""
    _fixture_run(
        tmp_path, tail_fail=True, tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"), array_times=_ARRAY_OK
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "NOT-GREEN" and r["tail"] == "FAIL" and r["verdict_line"].endswith(" tail=FAIL")
    assert r["counts"]["failed"] == 1


def test_a_tail_whose_expected_session_fixture_never_set_up_is_void(tmp_path):
    """Input class: the tail chunk records no ``tritonswmm_cpu_compiled`` in ``session_fixtures_ok``.

    Killed by a tail chunk carrying no ``expected_fixtures`` (software-engineering-specialist V2): it would read PASS.
    """
    _fixture_run(
        tmp_path,
        tail_fixtures=(),
        tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"),
        array_times=_ARRAY_OK,
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "NOT-GREEN" and r["tail"] == "VOID" and r["verdict_line"].endswith(" tail=VOID")


def test_a_manifest_with_a_tail_and_no_declared_complement_does_not_reach_union(tmp_path):
    """Input class: an empty declared complement and a tail chunk, aggregated at ``--scope union``.

    Required by DR-10 option 1 (``covers: "tail"`` is never complement evidence). Killed by option 3, where the tail
    carries ``covers: "complement"`` and alone satisfies the union guard.
    """
    _fixture_run(tmp_path, tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"), array_times=_ARRAY_OK)
    r = A.aggregate(tmp_path, scope="union")
    assert r["scope"] == "array" and any("requires complement evidence" in p for p in r["problems"])


def _race(problems: list[str]) -> list[str]:
    return [p for p in problems if p.startswith("shared-tree")]


@pytest.mark.parametrize(
    ("label", "tail_start", "array_times", "exempt"),
    [
        pytest.param("after", "2026-11-03T10:00:05.000+00:00", _ARRAY_OK, True, id="after"),
        pytest.param("equal", _ARRAY_END, _ARRAY_OK, True, id="equal-instants-exempt"),
        pytest.param("one-ms-early", "2026-11-03T09:59:59.999+00:00", _ARRAY_OK, False, id="one-ms-before"),
        pytest.param("before", "2026-11-03T09:55:00.000+00:00", _ARRAY_OK, False, id="before"),
        pytest.param("no-array-times", _ARRAY_END, None, False, id="array-status-without-times"),
        pytest.param(
            "rerun-after-tail",
            _ARRAY_END,
            {0: _times("2026-11-03T10:10:00.000+00:00", "2026-11-03T10:20:00.000+00:00"), 1: _ARRAY_OK[1]},
            False,
            id="array-chunk-rerun-after-the-tail-started",
        ),
    ],
)
def test_the_tail_is_left_out_of_the_shared_tree_joins_only_when_it_started_after_every_array_chunk_ended(
    tmp_path, label, tail_start, array_times, exempt
):
    """Input class: the tail wrote a tree two array chunks read, in six timing shapes.

    Required (software-engineering-specialist W1, snakemake-specialist R24-1, R25-1): the exemption holds only on
    ``tail.started_at >= array.ended_at`` over every array chunk's CURRENT status, with no tolerance. Killed by today's
    aggregator (always reports the write/read race), by an unconditional exemption (the ``before`` and
    ``no-array-times`` shapes), by a strict ``>`` (``equal``), by any tolerance of 1 ms or more (``one-ms-early``) and
    by a comparison against a first-run ``ended_at`` (``rerun-after-tail``).
    """
    tail_times = _times(tail_start, "2026-11-03T10:30:00.000+00:00")
    _fixture_run(tmp_path, tail_times=tail_times, array_times=array_times)
    r = A.aggregate(tmp_path, scope="array")
    if exempt:
        assert _race(r["problems"]) == [] and _TREE in r["tail_writes"], r["problems"]
        assert r["tail_writes"][_TREE]["write_samples"] == ["triton/build_tritonswmm_cpu/compilation.log"]
    else:
        races = _race(r["problems"])
        assert len(races) == 1 and "writers [2]" in races[0] and r["tail_writes"] == {}, r["problems"]


def test_a_tail_status_without_either_time_gets_no_exemption(tmp_path):
    """Input class: the tail's own status lacks ``started_at`` (and, separately, ``ended_at``). Killed by an exemption
    keyed on the array chunks' times alone."""
    for k, missing in enumerate(("started_at", "ended_at")):
        tail_times = _times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00")
        del tail_times[missing]
        root = tmp_path / f"r{k}"
        root.mkdir()
        _fixture_run(root, tail_times=tail_times, array_times=_ARRAY_OK)
        r = A.aggregate(root, scope="array")
        assert len(_race(r["problems"])) == 1 and r["tail_writes"] == {}, missing


def test_two_array_chunks_with_the_same_observations_still_report_the_race(tmp_path):
    """Control: the exemption is the tail's alone. A write by an array chunk of a tree another reads is still a race."""
    _fixture_run(tmp_path, tail_times=_times(_ARRAY_END, "2026-11-03T10:30:00.000+00:00"), array_times=_ARRAY_OK)
    status = json.loads((tmp_path / "chunk-00.status.json").read_text())
    status["reach_entries"] = {_TREE: _entry(written=True)}
    (tmp_path / "chunk-00.status.json").write_text(json.dumps(status))
    r = A.aggregate(tmp_path, scope="array")
    assert len(_race(r["problems"])) == 1 and "writers [0]" in _race(r["problems"])[0]
