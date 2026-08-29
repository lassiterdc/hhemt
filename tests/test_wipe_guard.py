"""Regression tests for the run-path full-tree wipe guard.

Each test states the pre-fix behaviour it would have caught. The incident these
exist for: `analysis.run(from_scratch=True)` deleted an analysis tree holding a
COMPLETED simulation (a `c_run` flag, a `d_process` flag, a per-sub datatree)
that carried no live sentinel precisely because it had finished.
"""

import ast
import inspect
from pathlib import Path

import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.wipe_guard import assert_wipe_is_deliberate, summarize_wipe_cost


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "_status").mkdir()
    return tmp_path


def test_pristine_tree_is_not_protected(tree):
    """A genuinely-fresh tree must wipe silently: a guard that refuses everything
    is routed around. PRE-FIX this had no analogue -- nothing was checked."""
    assert summarize_wipe_cost(tree).is_empty
    assert_wipe_is_deliberate(tree)  # must not raise


def test_absent_tree_is_not_protected(tmp_path):
    """First-ever launch: the dir does not exist yet."""
    assert summarize_wipe_cost(tmp_path / "never_created").is_empty


def test_completed_simulation_blocks_the_wipe(tree):
    """THE INCIDENT. A finished sim leaves a c_run flag and NO live sentinel, so
    _pre_delete_guards' in-flight predicate would have permitted this wipe --
    measured over its body: c_run 0, datatree 0, processed 0."""
    (tree / "_status" / "c_run_tritonswmm_sa-50_evt-e0_complete.flag").touch()
    cost = summarize_wipe_cost(tree)
    assert cost.completed_sims == 1 and cost.in_flight == 0
    with pytest.raises(ConfigurationError) as exc:
        assert_wipe_is_deliberate(tree)
    # The refusal must NAME what it found -- that is what would have shown sa_50.
    assert "1 completed simulation" in str(exc.value)


def test_consolidated_tree_alone_is_detected_but_is_not_the_primary_probe(tree):
    """The datatree test ALONE is insufficient and the incident proves it: the
    a100 arm had f_consolidate_master 0, so no master datatree existed while
    three sims were complete. Both terms are carried; neither is load-bearing
    alone."""
    (tree / "analysis_datatree.zarr").mkdir()
    assert summarize_wipe_cost(tree).consolidated_trees == ("analysis_datatree.zarr",)
    with pytest.raises(ConfigurationError):
        assert_wipe_is_deliberate(tree)


@pytest.mark.parametrize("subdir", ["_submitted", "_queued", "_orchestrator"])
def test_sentinel_classes_block_the_wipe(tree, subdir):
    """In-flight work and a (possibly stale) driver sentinel both refuse. Read by
    PRESENCE, not liveness: presence fails CLOSED and needs no workflow-builder
    import in the wipe path."""
    d = tree / "_status" / subdir
    d.mkdir()
    (d / "simulation_sa_46_evt-e0.json").write_text("{}")
    with pytest.raises(ConfigurationError):
        assert_wipe_is_deliberate(tree)


def test_override_permits_and_announces(tree, capsys):
    """The override is the deliberate signal. It must still PRINT the cost -- an
    operator who asked for this is entitled to see what it destroys."""
    (tree / "_status" / "c_run_triton_evt-e0_complete.flag").touch()
    cost = assert_wipe_is_deliberate(tree, override_wipe_nonempty=True)
    assert cost.completed_sims == 1
    assert "DESTROYING" in capsys.readouterr().out


def test_run_calls_the_guard_before_the_wipe():
    """WIRING. Source-order assertion over analysis.run(): inside the
    `if from_scratch and not dry_run:` block, assert_wipe_is_deliberate must be
    called BEFORE fast_rmtree. PRE-FIX the block held the fast_rmtree call alone,
    so this test fails on the pre-fix source at the first assert.

    This is the AST-lint idiom the repo already uses (test_provenance_discipline,
    check_du_sentinel_sites). It proves the ORDER of the two calls; it does not
    prove run() reaches the block -- see the deliberate gap named in the round-12
    scratch entry.
    """
    src = Path(inspect.getsourcefile(__import__("hhemt.analysis", fromlist=["x"]))).read_text()
    tree = ast.parse(src)
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "from_scratch" in ast.dump(node.test)
        and "dry_run" in ast.dump(node.test)
        and any(
            isinstance(c.func, ast.Name) and c.func.id == "fast_rmtree"
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        )
    ]
    assert len(blocks) == 1, f"expected exactly one from_scratch wipe block, found {len(blocks)}"
    names = [
        c.func.id
        for c in ast.walk(blocks[0])
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert "assert_wipe_is_deliberate" in names, "the wipe block does not call the guard"
    assert names.index("assert_wipe_is_deliberate") < names.index("fast_rmtree"), (
        "the guard must run BEFORE the delete"
    )
