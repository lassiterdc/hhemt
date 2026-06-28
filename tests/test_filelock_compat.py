"""Tests for filesystem-aware file locking (``hhemt._filelock_compat``).

Covers the contract that matters for HPC portability: a hard ``FileLock`` is
returned where ``flock`` works (preserving today's behavior off-Lustre) and a
``SoftFileLock`` where it does not (Lustre / flock-less NFS), with both honoring
the same acquire/release interface.
"""

from filelock import FileLock, SoftFileLock

from hhemt import _filelock_compat as flc


def test_flock_supported_on_local_tmp(tmp_path):
    # The pytest tmp filesystem (tmpfs/ext4) supports flock.
    assert flc.flock_supported(tmp_path) is True


def test_resolve_returns_hard_lock_where_flock_works(tmp_path):
    lock = flc.resolve_filelock(tmp_path / "x.lock", timeout=5)
    assert isinstance(lock, FileLock)
    with lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_resolve_falls_back_to_soft_when_flock_unsupported(tmp_path, monkeypatch):
    # Simulate a Lustre/flock-less filesystem.
    monkeypatch.setattr(flc, "flock_supported", lambda _dir: False)
    lock = flc.resolve_filelock(tmp_path / "y.lock", timeout=5)
    assert isinstance(lock, SoftFileLock)
    assert not isinstance(lock, FileLock)
    with lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_resolve_creates_missing_parent(tmp_path):
    nested = tmp_path / "a" / "b" / "z.lock"
    lock = flc.resolve_filelock(nested, timeout=5)
    assert nested.parent.is_dir()
    with lock:
        pass
