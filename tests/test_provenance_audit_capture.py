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
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.report_renderers._provenance_audit import (
    _capture_reads,
    assert_reads_subset_declared,
)


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


# A synthetic absolute root that is NOT under sys.prefix / tempfile.gettempdir() /
# mpl-data / the output dir, so the Tier-2 incidental prefixes cannot mask the
# Tier-1 same-stem-sibling clause being exercised here. These paths need not exist
# on disk -- the predicate is pure path arithmetic (resolve()/parent/stem/suffix).
_SYNTH_DATA_DIR = Path("/audit_test_root_xyz/data")
_SYNTH_OUTPUT = Path("/audit_test_root_xyz/out/fig.html")


def test_same_stem_sibling_rpt_is_subtracted():
    """A `<dir>/hydro.rpt` read IS covered by a declared `<dir>/hydro.inp` sibling."""
    declared = {_SYNTH_DATA_DIR / "hydro.inp"}
    actual = {_SYNTH_DATA_DIR / "hydro.rpt"}
    # Must NOT raise: the swmmio .rpt sidecar is a same-stem sibling of the .inp.
    assert_reads_subset_declared(actual, declared, _SYNTH_OUTPUT)


def test_unrelated_sibling_rpt_is_not_subtracted():
    """A `<dir>/unrelated.rpt` is NOT subtracted (proves same-stem, not parent-dir)."""
    declared = {_SYNTH_DATA_DIR / "hydro.inp"}
    actual = {_SYNTH_DATA_DIR / "unrelated.rpt"}
    with pytest.raises(ProcessingError):
        assert_reads_subset_declared(actual, declared, _SYNTH_OUTPUT)
