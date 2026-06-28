"""Filesystem-aware file locking for hhemt.

hhemt serializes concurrent JSON writes (run logs in ``log.py``, ``_version.json``
in ``version_migration``) and the synthetic-test build cache with ``filelock``.
``filelock.FileLock`` uses ``fcntl.flock``, which several network/parallel
filesystems do NOT support: on Lustre (OLCF Frontier project / ``$MEMBERWORK``
space) ``flock`` raises ``OSError`` — the kernel leaks errno 524 (ENOTSUPP) — and
some NFS mounts raise errno 95 (EOPNOTSUPP). The hard-flock locks then abort every
log/version/cache write, which is what blocked the Frontier container-validation run.

``resolve_filelock`` probes the lock's directory once and returns a hard
``FileLock`` where ``flock`` works (keeping its automatic crash-release) or a
``SoftFileLock`` (atomic ``O_CREAT | O_EXCL``, filesystem-agnostic) where it does
not. On filesystems that support ``flock`` the returned object is the SAME class
hhemt used before, so behavior off-Lustre is unchanged; only on flock-less
filesystems does the soft fallback engage. This is what makes hhemt runnable with
its working tree on Lustre (Frontier container validation, ADR-3).

Detection is by probe, not by an errno allow-list: the kernel's leaked 524 differs
from Python's ``errno.ENOTSUP`` (95), so attempting a real ``flock`` and catching
any ``OSError`` is the robust test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from filelock import BaseFileLock, FileLock, SoftFileLock

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix (Windows uses msvcrt locking)
    fcntl = None  # type: ignore[assignment]


def flock_supported(directory: Path) -> bool:
    """Return True iff ``fcntl.flock`` works on a file in ``directory``.

    Probes by creating a temp file in ``directory`` and taking + releasing a
    non-blocking exclusive ``flock``. Any ``OSError`` (ENOTSUPP/524 on Lustre,
    EOPNOTSUPP/95 on some NFS, EROFS, ...) means ``flock`` is unavailable here.
    On non-Unix platforms (no ``fcntl``) ``filelock`` uses msvcrt locking, so we
    report supported.
    """
    if fcntl is None:
        return True
    try:
        with tempfile.NamedTemporaryFile(dir=str(directory), prefix=".flock_probe_") as probe:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        return True
    except OSError:
        return False


def resolve_filelock(lock_path: str | Path, timeout: float = -1) -> BaseFileLock:
    """Return a ``FileLock`` (where ``flock`` works) or ``SoftFileLock`` (where it
    does not) for ``lock_path``. Creates the parent directory if absent — a lock
    file's directory must exist to host it. Drop-in for ``filelock.FileLock``;
    the result supports the same ``with`` / ``timeout`` interface.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    cls = FileLock if flock_supported(lock_path.parent) else SoftFileLock
    return cls(str(lock_path), timeout=timeout)
