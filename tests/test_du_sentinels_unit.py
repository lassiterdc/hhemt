"""Unit tests for du_sentinels.py pure-function surfaces (Phase 2, R11).

These exercise the compare-and-write mtime contract, the missing-file read
return, and walk_errors threading — all independent of workflow/synth fixtures.
"""

import os

import pytest

from TRITON_SWMM_toolkit import du_sentinels as du


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
        assert payload["walk_errors"] > 0, (
            "an unreadable subdirectory in the walk did not increment walk_errors"
        )
    finally:
        os.chmod(locked, 0o755)


def test_read_du_sentinel_returns_none_on_missing(tmp_path):
    assert du.read_du_sentinel(tmp_path / "nope" / "_du.json") is None
