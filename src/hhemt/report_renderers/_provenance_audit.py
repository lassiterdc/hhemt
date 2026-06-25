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
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

from hhemt.exceptions import ProcessingError

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
    # Analysis-scope completion-flag log read by open_datatree's refresh-before-gate
    # (processing_analysis.py R2): render-time renderers call open_datatree(), whose
    # _refresh_log() reads this analysis log.json under the audit capture context.
    # Bookkeeping/completion state, never figure data — sibling to the log basenames
    # above. Leading "/" anchors it to a path-segment boundary (matches both the bare
    # log.json and its atomic-write temp); does not over-match plot-data files.
    "/log.json",
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
        # A venv created over a base/system interpreter (e.g. uv over the system
        # CPython) reads the BASE interpreter's stdlib -- including the frozen
        # `pythonNN.zip` -- during render. That lives under sys.base_prefix /
        # sys.base_exec_prefix, OUTSIDE sys.prefix, so it must be allowlisted too.
        # No-op when base == prefix (a non-venv conda interpreter). Host-portable
        # and version-independent: runtime-derived, no absolute literal.
        os.path.realpath(sys.base_prefix),  # base interpreter (venv parent)
        os.path.realpath(sys.base_exec_prefix),  # base interpreter exec prefix
        os.path.realpath(tempfile.gettempdir()),  # platform tempdir
        os.path.realpath(matplotlib.get_data_path()),  # mpl-data
        os.path.realpath(str(output_path.parent)),  # self manifest/preview/svg
        # Class X: the toolkit's OWN source tree. A lazily-imported toolkit module
        # (e.g. processing_analysis) is read by the import machinery DURING render
        # (inside the capture window); under an editable/worktree install that .py
        # lives OUTSIDE sys.prefix/site-packages, so it escapes the env prefix above.
        # The package root is Path(__file__).parent.parent (this file is in
        # report_renderers/, the package root is its grandparent). Import-machinery
        # reads of toolkit source are never figure data.
        os.path.realpath(str(Path(__file__).resolve().parent.parent)),
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


def _declared_set_from_manifest(output_path: Path, master_analysis_dir: Path) -> set[Path]:
    """Reconstruct the declared source set for ONE rendered figure.

    Mirrors harvest_source_paths' rebasing (_figure_emission.py:639-668) for a
    single <output>.manifest.json: source_paths_relative resolved against the
    emit-time analysis-dir, where a per-sub manifest at
    plots/sensitivity/per_sim/sa-{N}/... rebases onto master/subanalyses/sa_{N}.
    Unions in artists[].channels[].ref.source_path (the provenance-log channel
    carrying renderer-internal sources not in the top-level list).
    """
    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    master_root = master_analysis_dir.resolve()
    # Detect the sensitivity per-sub position to pick the emit-time root.
    emit_dir = master_root
    try:
        rel_parts = output_path.parent.resolve().relative_to((master_root / "plots").resolve()).parts
        if (
            len(rel_parts) >= 3
            and rel_parts[0] == "sensitivity"
            and rel_parts[1] == "per_sim"
            and rel_parts[2].startswith("sa-")
        ):
            sa_id_rule = rel_parts[2][len("sa-") :].replace(".", "_").replace("-", "_")
            emit_dir = master_root / "subanalyses" / f"sa_{sa_id_rule}"
    except ValueError:
        pass  # output not under master/plots (bundle/test layout) -- use master root
    declared: set[Path] = set()
    for rp in manifest.get("source_paths_relative", []):
        declared.add((emit_dir / Path(rp)).resolve())
    for artist in manifest.get("artists", []):
        for channel in artist.get("channels", []):
            ref = channel.get("ref", {}) or {}
            src = ref.get("source_path")
            if src:
                p = Path(src)
                declared.add(p.resolve() if p.is_absolute() else (emit_dir / p).resolve())
    return declared


def assert_reads_subset_declared(
    actual_reads: set[Path],
    declared_sources: set[Path],
    output_path: Path,
) -> None:
    """Tier-1 directory-prefix subset + Tier-2 incidental allowlist.

    A read passes if it is UNDER any declared source treated as a directory
    prefix (dissolves the zarr-metadata / .aux.xml / shapefile-sibling /
    netCDF-internal fan-out), OR is a same-stem sibling of a declared file (the
    swmmio `.rpt`/`.out` sidecar swmmio opens next to a declared `.inp`), OR
    matches an incidental substring/prefix.
    """
    declared_resolved = [p.resolve() for p in declared_sources]
    incidental_prefixes = _runtime_incidental_prefixes(output_path)
    leaked: list[Path] = []
    for raw in actual_reads:
        ap = raw.resolve()
        s = str(ap)
        # Tier 1: under a declared source (prefix match).
        if any(ap == d or ap.is_relative_to(d) for d in declared_resolved):
            continue
        # Tier 1 (same-stem sibling): the swmmio `.rpt`/`.out` sidecar opened
        # next to a declared `.inp`. A file is not a directory, so is_relative_to
        # does NOT cover it. Narrow to same dir + same stem + different suffix so
        # it does NOT subtract unrelated siblings (a parent-dir prefix would).
        if any(ap.parent == d.parent and ap.stem == d.stem and ap.suffix != d.suffix for d in declared_resolved):
            continue
        # Tier 2: incidental allowlist (substrings + runtime prefixes).
        if any(sub in s for sub in _INCIDENTAL_READ_SUBSTRINGS):
            continue
        if any(s.startswith(pre) for pre in incidental_prefixes):
            continue
        leaked.append(ap)
    if leaked:
        raise ProcessingError(
            operation="provenance audit: renderer read an undeclared file",
            filepath=output_path,
            reason=(
                "renderer-IO provenance audit — reads not covered by declared "
                "source_paths nor the incidental allowlist:\n  "
                + "\n  ".join(sorted(str(p) for p in leaked))
                + "\n\nFix: declare the path via the renderer's source_paths "
                "(emit_plot_with_sources) if it is a real data source, OR add a "
                "host-portable substring to _INCIDENTAL_READ_SUBSTRINGS if it is "
                "incidental infrastructure. Set HHEMT_DISABLE_PROVENANCE_AUDIT=1 "
                "to bypass (not recommended)."
            ),
        )


@contextmanager
def audit_renderer_io(output_path, analysis_dir, *, renderer_name: str):
    """Capture the wrapped renderer's file reads and assert they were declared.

    Fail-open kill-switch: when HHEMT_DISABLE_PROVENANCE_AUDIT=1 the CM
    yields WITHOUT installing the hook (zero capture overhead, zero chance the
    audit machinery raises). Self-pollution guard: the captured set is
    SNAPSHOTTED before the manifest read, so reading <output>.manifest.json
    (which itself fires an "open" event) does not contaminate `actual`.
    """
    if os.environ.get("HHEMT_DISABLE_PROVENANCE_AUDIT") == "1":
        yield
        return
    output_path = Path(output_path)
    analysis_dir = Path(analysis_dir)
    with _capture_reads() as captured:
        yield
        # Snapshot INSIDE the capture block, BEFORE any post-render read.
        actual = {Path(p) for p in captured}
    # Capture is now off (CM exited) -- the manifest read below is not captured.
    declared = _declared_set_from_manifest(output_path, analysis_dir)
    # extra-declared (declared but not read): warn only (toggle-skipped panels).
    # EXCLUDE suffixes the capture hook is known to miss (C-level opens: h5netcdf
    # .nc, GDAL/fiona GIS) -- those are read-but-invisible to sys.addaudithook, so
    # warning on them is a guaranteed false positive on every per-sim render (the
    # weather .nc + watershed/boundary GIS are correctly declared yet never appear
    # in `actual`). Restrict the warning to declared sources whose reads the hook
    # CAN observe, so the warning channel stays meaningful for genuine toggle-skip
    # over-declaration. See SE FQ1 C-open measurement + master Assumptions.
    _C_OPEN_SUFFIXES = (".nc", ".gpkg", ".shp", ".geojson")
    extra_declared = {d for d in (declared - {p.resolve() for p in actual}) if d.suffix not in _C_OPEN_SUFFIXES}
    if extra_declared:
        warnings.warn(
            f"renderer {renderer_name!r} declared sources it did not read "
            f"(likely toggle-skipped panels): " + ", ".join(sorted(str(p) for p in extra_declared)),
            stacklevel=2,
        )
    # extra-actual (read but not declared / not incidental): fatal.
    assert_reads_subset_declared(actual, declared, output_path)
