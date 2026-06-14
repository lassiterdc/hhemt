"""Runtime renderer-IO provenance audit.

Phase 2 (this file's initial content): the capture primitive + the empirically-
characterized incidental-read allowlist. A PEP-578 ``sys.addaudithook("open")``
hook records every file a renderer opens during ``render()`` into a
``contextvars``-scoped buffer; the allowlist holds the host-portable incidental
reads the (Phase-3) subset assertion subtracts.

See library/knowledge/triton-swmm-toolkit/renderer io audit surface and dispatch
model.md and library/knowledge/software-engineering/pep578 open audit hook
misses c level opens.md. NOTE: sys.addaudithook("open") MISSES h5netcdf/GDAL
C-level opens; that is a correctness-safe false-NEGATIVE under the
actual-subset-of-declared invariant, not a false positive.
"""

from __future__ import annotations

import contextvars
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Capture buffer: a contextvars ContextVar so a future in-process multi-renderer
# test harness can scope captures without the (process-permanent, non-removable)
# addaudithook accumulating reads across renderers. Today every renderer runs in
# its own `python -m ..._cli` subprocess, so process-global capture is isolated
# per render; the buffer is swapped in/out by the capture CM regardless.
_capture_buffer: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "_provenance_audit_capture", default=None
)
_HOOK_INSTALLED = False

# Tier-2 incidental-read substrings. Every entry is a host-portable substring;
# NO host-specific absolute literal. Empirically verified by the Phase-2
# capture-only dump pass over the synth_multi_sim fixture (2026-06-14): the
# matplotlib + plotly (incl. kaleido, via per_sim_peak_flood_depth's
# fig.write_image) + swmmio read surfaces. NOTE: the swmmio same-stem `.rpt`
# sidecar (Class 3a) is NOT here — it is subtracted by the Phase-3 Tier-1
# same-stem-sibling predicate clause (a `.rpt`/`.out` substring would destroy
# sensitivity_benchmarking's catch-power, which declares `.rpt` as figure data).
# The `.inp` under-declarations (Class 3b) are renderer `source_paths` fixes,
# not allowlist entries.
_INCIDENTAL_READ_SUBSTRINGS: tuple[str, ...] = (
    "site-packages",
    "__pycache__",
    ".pyc",
    "matplotlib",
    "mpl-data",
    "fontconfig",
    "/fonts/",
    "fontTools",
    ".ttf",
    "plotly/package_data",
    "kaleido",
    "proj.db",
    "share/proj",
    "/gdal/",
    "pyproj",
    # --- Class 1: host-portable special files (dask-distributed memory_limit
    #     + psutil virtual_memory fire these at .compute(); NOT under sys.prefix) ---
    "/proc/meminfo",
    "/sys/fs/cgroup/",
    "/dev/null",
    # --- Class 2: toolkit scenario-bookkeeping log basenames (read+write side
    #     effects of the render-time scenario-access cascade; never figure data;
    #     the leading "/" anchors each to a path-segment boundary, and
    #     "/log_triton.json" is a prefix of the "/log_triton.json.<pid>.tmp"
    #     atomic-write temp file so it covers both the read and the write) ---
    "/log_triton.json",
    "/log_tritonswmm.json",
    "/log_swmm.json",
    "/scenario_prep_log.json",
)


def _install_hook() -> None:
    """Install the process-global open-audit hook exactly once (non-removable)."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return

    def _hook(event: str, args: tuple) -> None:
        if event != "open" or not args:
            return
        buf = _capture_buffer.get()
        if buf is None:
            return
        try:
            buf.add(os.path.realpath(os.fsdecode(args[0])))
        except (TypeError, ValueError, OSError):
            pass  # non-path-like (fd int, etc.) -- skip

    sys.addaudithook(_hook)
    _HOOK_INSTALLED = True


def _runtime_incidental_prefixes(output_path: Path) -> tuple[str, ...]:
    """Runtime-derived incidental prefixes (host-portable; no literals)."""
    return (
        os.path.realpath(sys.prefix),  # conda/venv env
        os.path.realpath(tempfile.gettempdir()),  # platform tempdir
        os.path.realpath(matplotlib.get_data_path()),  # mpl-data
        os.path.realpath(str(output_path.parent)),  # self manifest/preview/svg
    )


@contextmanager
def _capture_reads() -> Iterator[set[str]]:
    """Install the hook and swap in a fresh capture buffer for the block.

    Yields the live capture set. The CALLER is responsible for snapshotting the
    set BEFORE doing any post-render reads (e.g. reading the manifest) -- see the
    Phase-3 ``audit_renderer_io`` self-pollution guard.
    """
    _install_hook()
    fresh: set[str] = set()
    token = _capture_buffer.set(fresh)
    try:
        yield fresh
    finally:
        _capture_buffer.reset(token)
