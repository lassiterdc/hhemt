"""The dry-run gate on the fused deleter, both arms, at tmp_path.

The reprocess-dry_run stipulation SANCTIONS the deletion of the REPORT ARTIFACTS -- they
are the mtime trigger that makes a `--dry-run` preview meaningful -- and FORBIDS the
sentinel write. It does NOT sanction deleting figures: the developer ruled (D131) that on
a dry run plots survive, so the reprocess site passes only the report shell through this
helper and keeps its figure list inside the guard. The victim below is a report shell,
which is why these assertions are correct under that ruling and unchanged by it. The
unified deletion tool fuses deletion and accounting, so "delete without accounting" is
expressible only through `delete_and_account_unless_dry_run`. For the FIGURE half of the
distinction see the dry-run figure-survival checks in
tests/test_reprocess_rebuild_invalidation.py; the end-to-end guard is solver-bearing.
"""

from __future__ import annotations

from hhemt.du_sentinels import (
    compute_and_write_scope_sentinel,
    delete_and_account_unless_dry_run,
    read_du_sentinel,
)


def _seeded_scope(tmp_path):
    """A scope dir holding one sized artifact and a materialised sentinel."""
    scope = tmp_path / "analysis"
    (scope / "plots").mkdir(parents=True)
    victim = scope / "analysis_report.html"
    victim.write_bytes(b"x" * 4096)
    compute_and_write_scope_sentinel(scope, scope="analysis")
    sentinel = scope / "_status" / "_du.json"
    assert sentinel.exists(), "precondition: the sentinel was materialised"
    return scope, victim, sentinel


def test_real_run_deletes_and_accounts(tmp_path):
    """dry_run=False must behave exactly as the fused tool did: delete AND decrement."""
    scope, victim, sentinel = _seeded_scope(tmp_path)
    before = read_du_sentinel(sentinel)["disk_utilization_bytes"]

    delete_and_account_unless_dry_run([victim], scope_dir=scope, scope="analysis", dry_run=False)

    assert not victim.exists(), "the real-run arm must delete"
    after = read_du_sentinel(sentinel)["disk_utilization_bytes"]
    assert after < before, "the real-run arm must decrement the scope's sentinel"


def test_dry_run_deletes_but_does_not_write_the_sentinel(tmp_path):
    """dry_run=True must delete (the sanctioned trigger) and NOT re-stamp (the violation).

    Both halves are asserted. Asserting only "the sentinel did not change" would also pass
    for an implementation that deleted nothing, which would break the preview the deletion
    exists to produce -- so the deletion is asserted too.
    """
    scope, victim, sentinel = _seeded_scope(tmp_path)
    before_bytes = sentinel.read_bytes()
    before_mtime = sentinel.stat().st_mtime_ns

    delete_and_account_unless_dry_run([victim], scope_dir=scope, scope="analysis", dry_run=True)

    assert not victim.exists(), "the dry-run arm must still delete -- that is the mtime trigger"
    assert sentinel.read_bytes() == before_bytes, "a dry run must not rewrite _du.json"
    assert sentinel.stat().st_mtime_ns == before_mtime, "a dry run must not re-stamp _du.json"
