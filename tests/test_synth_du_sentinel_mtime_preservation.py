"""V-P1.1 — du_sentinels.write_du_sentinel compare-and-write semantics unit tests.

Maps to Phase 1 master plan requirements R1 (sentinel-write helper exists), R2
(compare-and-write mtime preservation), and A7 (the helper's contract is the
load-bearing mechanism that prevents Snakemake's `--rerun-triggers mtime input`
cascade-rerun on idempotent processing re-runs).

Run:
    conda run -n triton_swmm_toolkit python -m pytest tests/test_synth_du_sentinel_mtime_preservation.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit.du_sentinels import write_du_sentinel, read_du_sentinel


def _write(tmp_path: Path, **overrides) -> bool:
    sentinel = tmp_path / "_status" / "_du.json"
    payload = {
        "disk_utilization_bytes": 12345,
        "scope": "scenario",
        "sub_path_breakdown": None,
        "walk_errors": 0,
    }
    payload.update(overrides)
    return write_du_sentinel(sentinel, **payload)


def test_idempotent_write_preserves_mtime(tmp_path: Path) -> None:
    """Two consecutive writes with identical bytes-affecting payload preserve mtime."""
    sentinel = tmp_path / "_status" / "_du.json"

    # First write — fires the create path.
    rewrote_first = _write(tmp_path)
    assert rewrote_first is True
    first_mtime = sentinel.stat().st_mtime

    # Sleep so any inadvertent rewrite would produce a different mtime than the
    # filesystem's mtime granularity supports.
    time.sleep(1.1)

    # Second write — same bytes-affecting fields, must skip and preserve mtime.
    rewrote_second = _write(tmp_path)
    assert rewrote_second is False, "write_du_sentinel returned True on unchanged payload"
    assert sentinel.stat().st_mtime == first_mtime, (
        "mtime advanced on idempotent re-write — compare-and-write contract violated"
    )


def test_payload_change_bumps_mtime(tmp_path: Path) -> None:
    """A change in disk_utilization_bytes overwrites the file and advances mtime."""
    sentinel = tmp_path / "_status" / "_du.json"

    rewrote_first = _write(tmp_path, disk_utilization_bytes=1000)
    assert rewrote_first is True
    first_mtime = sentinel.stat().st_mtime

    time.sleep(1.1)

    rewrote_second = _write(tmp_path, disk_utilization_bytes=2000)
    assert rewrote_second is True, "write_du_sentinel skipped a genuine payload change"
    assert sentinel.stat().st_mtime > first_mtime

    payload_after = read_du_sentinel(sentinel)
    assert payload_after is not None
    assert payload_after["disk_utilization_bytes"] == 2000


def test_corrupt_existing_triggers_rewrite(tmp_path: Path) -> None:
    """Corrupted existing content (zero-byte or invalid JSON) triggers rewrite."""
    sentinel = tmp_path / "_status" / "_du.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)

    # Seed a corrupt existing file.
    sentinel.write_text("{ not valid json")

    rewrote = _write(tmp_path)
    assert rewrote is True, "write_du_sentinel did not overwrite corrupt existing content"

    payload_after = read_du_sentinel(sentinel)
    assert payload_after is not None
    assert payload_after["disk_utilization_bytes"] == 12345


def test_walk_errors_field_in_compare(tmp_path: Path) -> None:
    """walk_errors field is part of the compare-and-write key (per SE F-I Flag 5)."""
    sentinel = tmp_path / "_status" / "_du.json"

    rewrote_first = _write(tmp_path, walk_errors=0)
    assert rewrote_first is True
    first_mtime = sentinel.stat().st_mtime

    time.sleep(1.1)

    rewrote_second = _write(tmp_path, walk_errors=3)
    assert rewrote_second is True, (
        "write_du_sentinel skipped a walk_errors change — precision contract violated"
    )
    assert sentinel.stat().st_mtime > first_mtime

    payload_after = read_du_sentinel(sentinel)
    assert payload_after is not None
    assert payload_after["walk_errors"] == 3
