"""Pure-function coverage for the heavy-split partition path.

No compile, no solver, no SLURM, no cluster: `build_manifest` touches disk only to
substring-scan test files, and the fixture closure is supplied directly. Runs in
milliseconds under `pytest harness/pytest_suite/test_partition_split.py`.

WHY THIS FILE EXISTS. A wrong split does not fail loudly. It produces two chunks writing
one analysis tree, and the symptom is an intermittent red in an UNRELATED test -- the
failure shape README property 5 was written for. The only detector that catches it,
aggregate.py's arm B, is a runtime observation costing a full array run to consult, and
its own caveat records that it is false-negative-safe. So the guards below are the whole
of the pre-run safety, and an untested guard is an assertion nobody has ever seen fire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.suite import partition as P


def _fixture_file(tmp_path: Path, rel: str, body: str) -> str:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return rel


@pytest.fixture
def corpus(tmp_path: Path):
    """Two files reaching ONE analysis tree, so they form a single connected component."""
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text(
        'def retrieve_demo_test_case():\n    analysis_name = "demo_tree"\n',
        encoding="utf-8",
    )
    heavy = _fixture_file(
        tmp_path,
        "tests/test_heavy.py",
        'import pytest\n\n\n@pytest.mark.usefixtures("tritonswmm_cpu_compiled")\n'
        "def test_a(retrieve_demo_test_case):\n    pass\n\n\n"
        '@pytest.mark.usefixtures("tritonswmm_cpu_compiled")\n'
        "def test_b(retrieve_demo_test_case):\n    pass\n",
    )
    other = _fixture_file(
        tmp_path,
        "tests/test_other.py",
        "def test_c(retrieve_demo_test_case):\n    pass\n",
    )
    nodes = [f"{heavy}::test_a", f"{heavy}::test_b", f"{other}::test_c"]
    closures = {
        nodes[0]: ["retrieve_demo_test_case", "tritonswmm_cpu_compiled"],
        nodes[1]: ["retrieve_demo_test_case", "tritonswmm_cpu_compiled"],
        nodes[2]: ["retrieve_demo_test_case"],
    }
    return tmp_path, nodes, closures


def _build(root, nodes, closures, **kw):
    return P.build_manifest(
        repo_root=root,
        node_ids=nodes,
        source_sha="deadbeef",
        run_id="testrun",
        closures=closures,
        **kw,
    )


def test_split_without_isolation_refuses(corpus):
    """The guard between a split and a raced analysis tree."""
    root, nodes, closures = corpus
    with pytest.raises(P.PartitionDriftError, match="tree_isolation_per_chunk"):
        _build(root, nodes, closures, heavy_split_budget_s=60.0, durations={"tests/test_heavy.py": 100.0})


def test_split_without_durations_refuses(corpus):
    """The guard between a duration-labelled split and a node-count split."""
    root, nodes, closures = corpus
    with pytest.raises(P.PartitionDriftError, match="durations"):
        _build(root, nodes, closures, heavy_split_budget_s=60.0, tree_isolation_per_chunk=True, durations={})


def test_split_covers_the_universe_exactly(corpus):
    """_assert_covers under a split -- and the tripwire for a node_ids re-derivation.

    If the enrichment loop ever re-derives `node_ids` from `files` with `=` instead of
    `setdefault`, each node-split part reclaims its whole file, every split node lands in
    two chunks, and this fails on the duplicate count rather than in a cluster run.
    """
    root, nodes, closures = corpus
    m = _build(
        root,
        nodes,
        closures,
        heavy_split_budget_s=1.0,
        tree_isolation_per_chunk=True,
        durations={"tests/test_heavy.py": 100.0, "tests/test_other.py": 1.0},
    )
    assigned = [n for c in m["chunks"] for n in c["node_ids"]]
    assert sorted(assigned) == sorted(nodes)
    assert len(assigned) == len(set(assigned))


def test_no_chunk_declares_a_fixture_no_member_requests(corpus):
    """The property that makes a node split incapable of manufacturing a VOID.

    aggregate.py::classify_chunk computes `dead = expected_fixtures - session_fixtures_ok`
    and returns VOID when it is non-empty. A part declaring a fixture none of its members
    requests would therefore go VOID for a partitioning reason rather than a code one --
    a suite made faster at the cost of a green it has not earned.
    """
    root, nodes, closures = corpus
    m = _build(
        root,
        nodes,
        closures,
        heavy_split_budget_s=1.0,
        tree_isolation_per_chunk=True,
        durations={"tests/test_heavy.py": 100.0, "tests/test_other.py": 1.0},
    )
    for c in m["chunks"]:
        requested = set()
        for n in c["node_ids"]:
            requested |= set(closures[n])
        assert set(c["expected_fixtures"]) <= requested, (
            f"chunk {c['chunk_id']} declares {sorted(set(c['expected_fixtures']) - requested)} "
            "that no member node requests"
        )


def test_unsplit_manifest_is_unchanged(corpus):
    """The byte-identity claim: no split flag means the pre-change partition."""
    root, nodes, closures = corpus
    m = _build(root, nodes, closures)
    assert m["chunks"][0]["kind"] == "heavy"
    assert sorted(n for c in m["chunks"] for n in c["node_ids"]) == sorted(nodes)
    assert m.get("tree_isolation_per_chunk") in (None, False)


@pytest.fixture
def corpus_with_nonuniform_file(corpus):
    """`corpus`, plus a file that FAILS the uniformity gate.

    Without such a file the corpus cannot tell a gated split from an ungated one, and the
    assertion below is satisfied vacuously. Measured: with only `test_heavy.py` (which
    PASSES the gate) and single-node `test_other.py`, replacing the gate with `set(comp)`
    or deleting its second predicate leaves every test green.

    `test_mixed.py` mentions `tritonswmm_cpu_compiled`, so the whole-file `_fixtures_used`
    scan attributes it to BOTH nodes -- but only `test_d` requests it. Splitting it would
    hand `test_e` a part declaring a fixture it never sets up, which `aggregate.py`
    classifies VOID.
    """
    root, nodes, closures = corpus
    rel = _fixture_file(
        root,
        "tests/test_mixed.py",
        'import pytest\n\n\n@pytest.mark.usefixtures("tritonswmm_cpu_compiled")\n'
        "def test_d(retrieve_demo_test_case):\n    pass\n\n\n"
        "def test_e(retrieve_demo_test_case):\n    pass\n",
    )
    nodes = nodes + [f"{rel}::test_d", f"{rel}::test_e"]
    closures = dict(closures)
    closures[nodes[-2]] = ["retrieve_demo_test_case", "tritonswmm_cpu_compiled"]
    closures[nodes[-1]] = ["retrieve_demo_test_case"]
    return root, nodes, closures


def test_nonuniform_file_is_not_node_split(corpus_with_nonuniform_file):
    """The gate itself, from both sides -- excluded when non-uniform, split when uniform.

    Kills three mutations the five tests above all survive: replacing the gate with
    `set(comp)`, replacing it with `set()`, and deleting its second predicate.
    """
    root, nodes, closures = corpus_with_nonuniform_file
    files = sorted({P.node_file(n) for n in nodes})
    by_file = {f: [n for n in nodes if P.node_file(n) == f] for f in files}
    comp = P.heavy_components(root, files, P.file_trees(root, closures))[0]
    assert "tests/test_mixed.py" in comp
    cand = P.node_split_candidates(comp, by_file, closures, root)
    assert "tests/test_mixed.py" not in cand
    assert "tests/test_heavy.py" in cand
    m = _build(
        root,
        nodes,
        closures,
        heavy_split_budget_s=1.0,
        tree_isolation_per_chunk=True,
        durations={"tests/test_heavy.py": 100.0, "tests/test_other.py": 1.0, "tests/test_mixed.py": 100.0},
    )
    for c in m["chunks"]:
        requested = set()
        for n in c["node_ids"]:
            requested |= set(closures[n])
        assert set(c["expected_fixtures"]) <= requested, (
            f"chunk {c['chunk_id']} declares {sorted(set(c['expected_fixtures']) - requested)} that no member requests"
        )
    heavy_chunks = {
        c["chunk_id"] for c in m["chunks"] if any(P.node_file(n) == "tests/test_heavy.py" for n in c["node_ids"])
    }
    assert len(heavy_chunks) == 2, (
        "the gate-PASSING file must still be node-split; it landed in "
        f"{len(heavy_chunks)} chunk(s), so the split is disabled"
    )
