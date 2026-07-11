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

import os
import uuid
from pathlib import Path

from filelock import BaseFileLock, FileLock, SoftFileLock

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix (Windows uses msvcrt locking)
    fcntl = None  # type: ignore[assignment]


# flock support is a property of the FILESYSTEM, so cache the probe result by device
# id (st_dev). This makes the probe fire ONCE per filesystem (early, e.g. at the first
# cache-build lock) instead of on every lock — so render-time locks on a same-filesystem
# dir reuse the cache and create no `.flock_probe_` temp file, which otherwise trips the
# renderer-IO provenance audit (report_renderers/_provenance_audit.py) as an undeclared read.
_flock_support_by_dev: dict[int, bool] = {}


def flock_supported(directory: Path) -> bool:
    """Return True iff ``fcntl.flock`` works on the filesystem hosting ``directory``.

    Probes ONCE per filesystem (keyed by ``st_dev``): creates a temp file in
    ``directory`` and takes + releases a non-blocking exclusive ``flock``. Any
    ``OSError`` (ENOTSUPP/524 on Lustre, EOPNOTSUPP/95 on some NFS, EROFS, ...) means
    ``flock`` is unavailable there. On non-Unix platforms (no ``fcntl``) ``filelock``
    uses msvcrt locking, so we report supported.
    """
    if fcntl is None:
        return True
    directory = Path(directory)
    try:
        dev = directory.stat().st_dev
    except OSError:
        dev = None
    if dev is not None and dev in _flock_support_by_dev:
        return _flock_support_by_dev[dev]
    # Probe via an EXPLICIT named probe file (O_CREAT|O_EXCL), never
    # tempfile.NamedTemporaryFile: on CPython 3.12 the latter calls
    # _io.open(dir, ..., opener=...), which fires an "open" audit event on the
    # BARE DIRECTORY before the opener creates the named temp file. That bare-dir
    # open escapes the renderer-IO provenance audit's ".flock_probe_" substring
    # allowlist (the bare dir carries no such substring) and fails every per-sim
    # plot rule whose render takes a cold-cache lock. The named probe path below
    # carries ".flock_probe_" so the (single, named) open is allowlisted, and no
    # bare-dir open is ever produced.
    probe_path = directory / f".flock_probe_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        fd = os.open(str(probe_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            result = True
        finally:
            os.close(fd)
            try:
                # EXEMPT-DU: transient-intermediate
                probe_path.unlink()
            except OSError:
                pass
    except OSError:
        result = False
    if dev is not None:
        _flock_support_by_dev[dev] = result
    return result


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
