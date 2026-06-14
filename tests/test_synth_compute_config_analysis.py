"""Unit tests for the byte-identity analysis tool's PURE functions (no ssh).

Validates the analysis logic at code time rather than first-exercising it on a multi-hour
UVA run (SE Flag 1 / Decision 1). The cluster seam (``_ssh``) is never touched here.
"""

from __future__ import annotations

import hashlib

import pytest

from scripts.experiments.analyze_compute_config_byte_identity import (
    compare_clean_vs_resume,
    divergence_onset_index,
    final_mh_md5,
    group_by_hash,
)


def _md5(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()


def test_group_by_hash_groups_identical_blobs():
    groups = group_by_hash({"a": b"x", "b": b"x", "c": b"y"})
    assert groups == {_md5(b"x"): ["a", "b"], _md5(b"y"): ["c"]}


def test_group_by_hash_empty():
    assert group_by_hash({}) == {}


def test_group_by_hash_sorts_sa_ids_within_group():
    groups = group_by_hash({"z": b"x", "a": b"x", "m": b"x"})
    assert groups == {_md5(b"x"): ["a", "m", "z"]}


def test_final_mh_md5_selects_max_index():
    blobs = {0: b"first", 5: b"final", 3: b"middle"}
    idx, digest = final_mh_md5(blobs)
    assert idx == 5
    assert digest == _md5(b"final")


def test_final_mh_md5_empty_raises():
    with pytest.raises(ValueError):
        final_mh_md5({})


def test_compare_clean_vs_resume_classifies_membership():
    clean = {"gpu_0_r1": "aaa", "mpi_9_r1": "bbb", "serial_6_r1": "ccc"}
    resume = {"gpu_0_r1": "aaa", "mpi_9_r1": "DIVERGED", "openmp_7_r1": "ddd"}
    result = compare_clean_vs_resume(clean, resume)
    assert result["matched"] == ["gpu_0_r1"]
    assert result["diverged"] == ["mpi_9_r1"]
    assert result["clean_only"] == ["serial_6_r1"]
    assert result["resume_only"] == ["openmp_7_r1"]


def test_divergence_onset_index_finds_first_mismatch():
    clean = ["h0", "h1", "h2", "h3"]
    resume = ["h0", "h1", "XX", "h3"]
    assert divergence_onset_index(clean, resume) == 2


def test_divergence_onset_index_none_when_all_match():
    assert divergence_onset_index(["h0", "h1"], ["h0", "h1"]) is None
