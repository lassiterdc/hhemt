"""Unit tests for du_sentinels.py pure-function surfaces (Phase 2, R11).

These exercise the compare-and-write mtime contract, the missing-file read
return, and walk_errors threading — all independent of workflow/synth fixtures.
"""

import os

import pytest

from hhemt import du_sentinels as du


def test_compare_and_write_preserves_mtime_on_identical_payload(tmp_path):
    scope_dir = tmp_path / "scen"
    (scope_dir / "_status").mkdir(parents=True)
    (scope_dir / "data.bin").write_bytes(b"x" * 1000)
    du.compute_and_write_scope_sentinel(scope_dir, scope="scenario")
    sentinel = scope_dir / "_status" / "_du.json"
    mtime1 = sentinel.stat().st_mtime_ns
    # Re-stamp with byte-identical content -> compare-and-write must skip the
    # write (computed_at is excluded from the equality check), preserving mtime.
    du.compute_and_write_scope_sentinel(scope_dir, scope="scenario")
    assert sentinel.stat().st_mtime_ns == mtime1, "byte-identical re-stamp bumped mtime"


def test_walk_errors_threaded_on_unreadable_dir(tmp_path):
    # Inject an OSError during the walk: chmod 000 a *subdirectory* so
    # `os.scandir` on it raises PermissionError (which increments walk_errors).
    # chmod 000 on a *file* would NOT work — stat() reads inode metadata via the
    # parent dir's search permission, not the file's own mode. Skip where chmod
    # is a no-op (root, or Windows CI).
    scope_dir = tmp_path / "scen"
    (scope_dir / "_status").mkdir(parents=True)
    (scope_dir / "data.bin").write_bytes(b"x" * 100)
    locked = scope_dir / "locked"
    locked.mkdir()
    (locked / "inner.bin").write_bytes(b"y" * 50)
    os.chmod(locked, 0o000)
    if os.access(locked, os.R_OK):
        # chmod did not take effect (likely running as root) — cannot inject.
        os.chmod(locked, 0o755)
        pytest.skip("chmod is a no-op on this platform/user; cannot inject walk error")
    try:
        du.compute_and_write_scope_sentinel(scope_dir, scope="scenario")
        payload = du.read_du_sentinel(scope_dir / "_status" / "_du.json")
        assert payload is not None
        assert payload["walk_errors"] > 0, "an unreadable subdirectory in the walk did not increment walk_errors"
    finally:
        os.chmod(locked, 0o755)


def test_read_du_sentinel_returns_none_on_missing(tmp_path):
    assert du.read_du_sentinel(tmp_path / "nope" / "_du.json") is None


def test_sum_child_sentinels_absent_child_never_walks(tmp_path, monkeypatch):
    """Clause 5: 12 sentinel-less children -> zero _walk_root_and_breakdown calls, absent_children == 12."""
    root = tmp_path / "an"
    (root / "_status").mkdir(parents=True)
    for i in range(12):
        (root / "sims" / f"evt{i}" / "out").mkdir(parents=True)
        (root / "sims" / f"evt{i}" / "out" / "f.bin").write_bytes(b"x" * 100)
    calls = {"walk": 0}
    real = du._walk_root_and_breakdown

    def spy(*a, **k):
        calls["walk"] += 1
        return real(*a, **k)

    monkeypatch.setattr(du, "_walk_root_and_breakdown", spy)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        du.sum_child_sentinels(root, scope="analysis", child_scope_dirs=["members", "sims"])
    assert calls["walk"] == 0, "absent child triggered a walk"
    payload = du.read_du_sentinel(root / "_status" / "_du.json")
    assert payload["absent_children"] == 12
    assert any("counted as absent" in str(x.message) for x in w)


def test_delete_and_account_touches_only_named_scope(tmp_path, monkeypatch):
    """Clause 1/2: the tool adjusts the named scenario sentinel (total, breakdown child,
    both cleared counters) and never any ancestor."""
    an = tmp_path / "an"
    scen = an / "sims" / "evt0"
    (an / "_status").mkdir(parents=True)
    (scen / "_status").mkdir(parents=True)
    (scen / "raw").mkdir()
    f = scen / "raw" / "a.bin"
    f.write_bytes(b"x" * 1000)
    du.compute_and_write_scope_sentinel(scen, scope="scenario")
    du.write_du_sentinel(an / "_status" / "_du.json", disk_utilization_bytes=1000, scope="analysis")
    an_mtime = (an / "_status" / "_du.json").stat().st_mtime_ns
    monkeypatch.setattr(du, "_walk_root_and_breakdown", lambda *a, **k: (_ for _ in ()).throw(AssertionError("walked")))
    freed = du.delete_and_account([f], scope_dir=scen, scope="scenario")
    assert freed == 1000
    p = du.read_du_sentinel(scen / "_status" / "_du.json")
    assert p["disk_utilization_bytes"] == 0
    assert p.get("sub_path_breakdown") in (None, {})  # the `raw` child went to 0 and was dropped
    assert p["cleared_bytes_own"] == 1000 and p["cleared_bytes_total"] == 1000
    assert (an / "_status" / "_du.json").stat().st_mtime_ns == an_mtime, "ancestor sentinel was touched"


def test_cleared_bytes_inside_compare_set(tmp_path):
    """Clause 8 (amended): same total + different cleared counters -> REWRITTEN, never skipped."""
    s = tmp_path / "_status" / "_du.json"
    du.write_du_sentinel(s, disk_utilization_bytes=5, scope="scenario", cleared_bytes_own=0, cleared_bytes_total=0)
    assert (
        du.write_du_sentinel(
            s, disk_utilization_bytes=5, scope="scenario", cleared_bytes_own=99, cleared_bytes_total=99
        )
        is True
    )
    assert du.read_du_sentinel(s)["cleared_bytes_own"] == 99


def test_scenario_sentinel_created_without_walk_arming(tmp_path, monkeypatch):
    """Clause 4: with a scenario _status/ present, a per-file deletion through the tool
    performs ZERO full walks (the A1 regression would perform one per file), and the
    upward walker no longer exists."""
    an = tmp_path / "an"
    scen = an / "sims" / "evt0"
    (scen / "raw").mkdir(parents=True)
    files = [scen / "raw" / f"{i}.bin" for i in range(5)]
    for f in files:
        f.write_bytes(b"x" * 10)
    du.compute_and_write_scope_sentinel(scen, scope="scenario")  # what S8 does at creation
    calls = {"walk": 0}
    real = du._walk_root_and_breakdown

    def spy(*a, **k):
        calls["walk"] += 1
        return real(*a, **k)

    monkeypatch.setattr(du, "_walk_root_and_breakdown", spy)
    du.delete_and_account(files, scope_dir=scen, scope="scenario")
    assert calls["walk"] == 0
    assert not hasattr(du, "restamp_parent_sentinels")


def test_sum_child_sentinels_carries_own_cleared_credit_forward(tmp_path):
    """A-S1d: an analysis-scope own deletion credit survives the next aggregation and is
    added to the children's summed totals."""
    an = tmp_path / "an"
    (an / "_status").mkdir(parents=True)
    child = an / "sims" / "evt0"
    (child / "_status").mkdir(parents=True)
    (child / "d.bin").write_bytes(b"c" * 50)
    du.compute_and_write_scope_sentinel(child, scope="scenario")
    du.delete_and_account([child / "d.bin"], scope_dir=child, scope="scenario")  # child cleared_total 50
    (an / "own.bin").write_bytes(b"o" * 30)
    du.sum_child_sentinels(an, scope="analysis", child_scope_dirs=["members", "sims"])
    du.delete_and_account([an / "own.bin"], scope_dir=an, scope="analysis")  # own 30
    du.sum_child_sentinels(an, scope="analysis", child_scope_dirs=["members", "sims"])
    p = du.read_du_sentinel(an / "_status" / "_du.json")
    assert p["cleared_bytes_own"] == 30
    assert p["cleared_bytes_total"] == 80
