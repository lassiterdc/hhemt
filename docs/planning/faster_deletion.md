# Faster Directory Deletion Plan

**Status**: Implemented
**Owner**: Toolkit maintainers
**Created**: 2026-02-11

## Purpose

Large analysis folders can take a long time to delete with `shutil.rmtree()`
because it is pure‑Python and performs many syscalls. This plan proposes a
fast, cross‑platform delete helper that uses the OS‑native deletion command
(`rm -rf` on POSIX, `rmdir /s /q` on Windows), with a safe Python fallback.

This document also lists **all deletion call sites** migrated to use the helper.

---

## Goals

1. Provide a fast, cross‑platform `fast_rmtree()` helper.
2. Provide subprocess‑safe cleanup patterns (kill process tree before delete).
3. Document all call sites that should migrate.

---

## Key Idea: Use OS‑native Delete

Native delete tools are optimized and generally much faster:

- **Linux/macOS**: `rm -rf <path>`
- **Windows**: `cmd /c rmdir /s /q <path>`

We still keep a Python fallback (using `shutil.rmtree`) in case:

- the OS tool fails,
- permissions prevent deletion,
- the path is a symlink or file.

---

## Implemented Helper (Current Code)

Location: `src/TRITON_SWMM_toolkit/utils.py`.

```python
from __future__ import annotations

from pathlib import Path
import os
import platform
import shutil
import subprocess
from typing import Optional, Callable


def fast_rmtree(
    path: str | Path,
    *,
    missing_ok: bool = True,
    onerror: Optional[Callable] = None,
) -> None:
    """Fast, cross‑platform directory delete.

    Uses OS‑native delete commands for speed; falls back to shutil.rmtree.

    Parameters
    ----------
    path : str | Path
        Directory path to delete.
    missing_ok : bool
        If True, silently return when path does not exist.
    onerror : callable, optional
        Error handler passed to shutil.rmtree (fallback only).
    """
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(path)

    # If path is a file or symlink, remove directly
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    # Try OS‑native delete first
    try:
        if os.name == "nt":
            # Windows (cmd built‑in rmdir)
            subprocess.run(
                ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # POSIX (rm -rf)
            subprocess.run(
                ["rm", "-rf", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    except Exception:
        # Fallback to shutil for robustness
        shutil.rmtree(path, onerror=onerror)
```

---

## Call‑Site Inventory (Updated)

These are the deletion call sites now using `fast_rmtree()`:

### 1) `src/TRITON_SWMM_toolkit/utils.py`
- `write_zarr_then_netcdf`: `fast_rmtree(f"{fname_out}.zarr")`

### 2) `src/TRITON_SWMM_toolkit/process_simulation.py`
- `_clear_raw_TRITON_outputs`: `fast_rmtree(triton_dir)`
- `_clear_full_timeseries_outputs`: `fast_rmtree(triton_ts_path)`
- `_clear_full_timeseries_outputs`: `fast_rmtree(node_ts_path)`
- `_clear_full_timeseries_outputs`: `fast_rmtree(link_ts_path)`

### 3) `src/TRITON_SWMM_toolkit/case_study_catalog.py`
- `CaseStudyBuilder.__init__`: `fast_rmtree(anlysys_dir)`

### 4) `src/TRITON_SWMM_toolkit/examples.py`
- `_download_data_from_hydroshare`: `fast_rmtree(target)`

### 5) `src/TRITON_SWMM_toolkit/scenario.py`
- `_create_strict_dir_symlink`: `fast_rmtree(target_link)`

### 6) `src/TRITON_SWMM_toolkit/system.py`
- `_download_tritonswmm_source`: `fast_rmtree(TRITONSWMM_software_directory)`
- `_compile_backend` / `_compile_triton_only_backend` / `compile_SWMM` bash scripts
  include `rm -rf` (already native; no Python change needed unless we rework
  compilation to run in‑process)

### 7) `src/TRITON_SWMM_toolkit/analysis.py`
- `run(from_scratch=True)`: `fast_rmtree(self.cfg_analysis.analysis_dir)`

---

## Integration Plan (Completed)

1. **Added helper** to `utils.py`.
2. **Replaced all `shutil.rmtree(...)`** usage with `fast_rmtree(...)`.
3. If deletion fails due to files in use, treat it as a timing/ordering issue
   and adjust the workflow to delete only after the work is complete.

---

## Non‑Goals

- Changing the Snakemake/bash‑script `rm -rf` lines (already fastest path).
- Introducing a new mandatory dependency (psutil remains optional).

---

## Notes / Edge Cases

- On Windows, `rmdir` will fail if a file is still open.
- On POSIX, `rm -rf` can delete symlink targets if passed a resolved path — the
  helper uses direct unlink for symlinks to avoid surprises.
- When deleting very large folders, prefer the OS command over `shutil.rmtree`.
