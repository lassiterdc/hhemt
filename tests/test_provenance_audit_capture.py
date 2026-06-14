"""Phase-2 unit test for the renderer-IO provenance-audit capture primitive.

Verifies (a) ``_capture_reads`` records a Python-level ``open()`` performed inside
the block, and (b) the documented ``sys.addaudithook("open")`` C-open blind spot:
an ``xr.open_dataset(engine="h5netcdf")`` opens the file via a C-level ``H5Fopen``
that does NOT fire the PEP-578 ``"open"`` audit event, so the ``.nc`` path does
NOT appear in the capture set. The blind spot is correctness-safe under the
``actual ⊆ declared`` invariant (a missed read is a false NEGATIVE — the path is
already declared — never a false positive). This is a pure capture-primitive test;
it needs no synth fixture.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from TRITON_SWMM_toolkit.report_renderers._provenance_audit import _capture_reads


def test_capture_reads_records_python_open():
    """A Python-level ``open()`` inside the block lands in the captured set."""
    with tempfile.TemporaryDirectory() as td:
        known = Path(td) / "known.txt"
        known.write_text("payload")
        with _capture_reads() as captured:
            with open(known) as fh:
                fh.read()
        assert os.path.realpath(known) in captured


def test_capture_reads_buffer_resets_after_block():
    """Outside the block the buffer is reset — a later open is not captured."""
    with tempfile.TemporaryDirectory() as td:
        inside = Path(td) / "inside.txt"
        outside = Path(td) / "outside.txt"
        inside.write_text("a")
        outside.write_text("b")
        with _capture_reads() as captured:
            with open(inside) as fh:
                fh.read()
        with open(outside) as fh:  # after the CM exits — no live buffer
            fh.read()
        assert os.path.realpath(inside) in captured
        assert os.path.realpath(outside) not in captured


def test_capture_reads_misses_h5netcdf_c_open():
    """Documents the addaudithook C-open blind spot (correctness-safe false-NEG)."""
    with tempfile.TemporaryDirectory() as td:
        nc = Path(td) / "probe.nc"
        # Write OUTSIDE the capture block.
        xr.Dataset({"v": ("x", np.arange(3))}).to_netcdf(nc, engine="h5netcdf")
        with _capture_reads() as captured:
            ds = xr.open_dataset(nc, engine="h5netcdf")
            ds.load()
            ds.close()
        # h5netcdf opens via C-level H5Fopen -> no Python "open" event fired.
        assert os.path.realpath(nc) not in captured
