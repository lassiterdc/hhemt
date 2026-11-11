"""`compare_reference_outputs` compares scientific variables exactly and states what it leaves out.

Synthetic consolidated trees (no solver): a `tritonswmm/triton/summary` group and a `tritonswmm/performance/summary`
group per side. Excluded by rule, with reasons: the build-provenance stamp coordinates and the floating-point
wall-clock timers of the performance summary. Never excluded: any other variable, any integer or text variable of the
performance summary, and anything missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle._reprex import compare_reference_outputs

TREE = "group_0/analysis_datatree.zarr"


def _triton(depth: float = 1.5, sha: str = "aaa") -> xr.Dataset:
    return xr.Dataset(
        {"max_flood_depth": (("event_iloc", "x"), np.array([[0.1, depth]]))},
        coords={
            "event_iloc": [0],
            "x": [0.0, 1.0],
            "hhemt_producing_sha": ("event_iloc", [sha]),
            "hhemt_producing_version": ("event_iloc", ["1.0.0"]),
        },
    )


def _perf(total: float = 10.0, n_ranks: int = 4) -> xr.Dataset:
    return xr.Dataset(
        {"Total": (("event_iloc",), np.array([total])), "n_ranks": (("event_iloc",), np.array([n_ranks]))},
        coords={"event_iloc": [0]},
    )


def _write(root: Path, side: str, triton: xr.Dataset, perf: xr.Dataset | None) -> None:
    store = root / side / TREE
    shutil.rmtree(store, ignore_errors=True)
    store.parent.mkdir(parents=True, exist_ok=True)
    triton.to_zarr(store, group="tritonswmm/triton/summary", mode="w", consolidated=False, zarr_format=3)
    if perf is not None:
        perf.to_zarr(store, group="tritonswmm/performance/summary", mode="a", consolidated=False, zarr_format=3)


def _compare(tmp_path, ref_triton, new_triton, ref_perf=None, new_perf=None):
    bundle, analysis = tmp_path / "bundle", tmp_path / "analysis"
    _write(bundle, "reference_outputs", ref_triton, ref_perf)
    _write(analysis, "_test", new_triton, new_perf)
    return compare_reference_outputs(bundle, analysis)


def test_identical_trees_pass_with_no_exclusion_beyond_the_rules_and_a_changed_value_fails(tmp_path):
    same = _compare(tmp_path / "a", _triton(), _triton(), _perf(), _perf())
    assert same.passed
    changed = _compare(tmp_path / "b", _triton(1.5), _triton(1.6), _perf(), _perf())
    assert not changed.passed
    text = changed.summary()
    assert "tritonswmm/triton/summary" in text and "max_flood_depth" in text


def test_a_changed_build_stamp_alone_passes_and_the_exclusion_is_listed(tmp_path):
    result = _compare(tmp_path, _triton(sha="aaa"), _triton(sha="bbb"))
    assert result.passed
    names = {(v) for _s, v, _r in result.exclusions}
    assert {"hhemt_producing_sha", "hhemt_producing_version"} <= names
    assert "not compared" in result.summary() and "hhemt_producing_sha" in result.summary()
    # control: the same stamp change with a changed scientific variable still fails
    assert not _compare(tmp_path / "c", _triton(1.5, "aaa"), _triton(1.6, "bbb")).passed


def test_a_changed_timer_alone_passes_and_is_listed_but_a_changed_deterministic_variable_fails(tmp_path):
    result = _compare(tmp_path / "t", _triton(), _triton(), _perf(10.0), _perf(11.0))
    assert result.passed
    assert any(v == "Total" for _s, v, _r in result.exclusions)
    failing = _compare(tmp_path / "d", _triton(), _triton(), _perf(n_ranks=4), _perf(n_ranks=8))
    assert not failing.passed
    assert "n_ranks" in failing.summary()


def test_a_float_variable_outside_the_performance_group_is_never_excluded(tmp_path):
    """The timer rule is scoped to the performance summary: the same float change in a triton group fails."""
    assert not _compare(tmp_path, _triton(1.5), _triton(1.5000001)).passed


def test_a_missing_store_or_variable_fails_even_when_it_would_have_been_excluded(tmp_path):
    bundle, analysis = tmp_path / "bundle", tmp_path / "analysis"
    _write(bundle, "reference_outputs", _triton(), _perf())
    _write(analysis, "_test", _triton(), None)  # the re-run has no performance group
    missing_store = compare_reference_outputs(bundle, analysis)
    assert not missing_store.passed and "not produced by the re-run" in missing_store.summary()
    # a timer variable absent from the re-run's performance group is a failure, not an exclusion
    _write(
        analysis,
        "_test",
        _triton(),
        xr.Dataset({"n_ranks": (("event_iloc",), np.array([4]))}, coords={"event_iloc": [0]}),
    )
    missing_var = compare_reference_outputs(bundle, analysis)
    assert not missing_var.passed and "Total" in missing_var.summary() and "absent" in missing_var.summary()
    assert all(v != "Total" for _s, v, _r in missing_var.exclusions)


@pytest.mark.parametrize("flag", ["exclusions"])
def test_the_result_type_exposes_its_exclusions(tmp_path, flag):
    result = _compare(tmp_path, _triton(), _triton())
    assert isinstance(getattr(result, flag), tuple)
