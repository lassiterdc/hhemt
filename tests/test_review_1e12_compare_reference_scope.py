"""Reviewer tests for what `compare_reference_outputs` can no longer miss.

Problem this file solves: the comparison now leaves out build-provenance stamps and the wall-clock timers of the
performance summary. A check that excludes anything is only as good as what it still refuses, so these tests build a
reference tree with the five result stores and a performance store, change ONE thing in the re-run per case, and
require the comparison to fail for every case that is not a stamp or a timer. No solver; zarr trees in `tmp_path`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle._reprex import ReferenceStoreVerdict, compare_reference_outputs

TREE = "group_0/analysis_datatree.zarr"
RESULT_GROUPS = (
    "tritonswmm/triton/summary",
    "tritonswmm/swmm_node/summary",
    "tritonswmm/swmm_link/summary",
    "tritonswmm/swmm_node_hydro/summary",
    "tritonswmm/swmm_subcatchment_hydro/summary",
)
PERF = "tritonswmm/performance/summary"


def _result(group: str, value: float = 1.0, sha: str = "aaa", events=(0, 1)) -> xr.Dataset:
    """One result group: a float result, a text result, an id coordinate and the two stamp coordinates."""
    n = len(events)
    return xr.Dataset(
        {
            "peak": (("event_iloc",), np.full(n, value)),
            "label": (("event_iloc",), np.array([f"{group[-12:]}{i}" for i in range(n)])),
        },
        coords={
            "event_iloc": list(events),
            "hhemt_producing_sha": ("event_iloc", [sha] * n),
            "hhemt_producing_version": ("event_iloc", ["1.0.0"] * n),
        },
    )


def _perf(total: float = 10.0, zero: float = 0.0, n_ranks: int = 4, host: str = "node-a") -> xr.Dataset:
    """A performance group: two float timers (one zero-valued), an integer and a text variable."""
    return xr.Dataset(
        {
            "Total": (("event_iloc",), np.array([total, total])),
            "Compute_min": (("event_iloc",), np.array([zero, zero])),
            "n_ranks": (("event_iloc",), np.array([n_ranks, n_ranks])),
            "host": (("event_iloc",), np.array([host, host])),
        },
        coords={"event_iloc": [0, 1]},
    )


def _build(root: Path, side: str, *, overrides: dict | None = None, perf: xr.Dataset | None = None, drop=()) -> None:
    """Write the reference (or re-run) tree; `overrides` maps a group name to a replacement Dataset."""
    store = root / side / TREE
    shutil.rmtree(store, ignore_errors=True)
    store.parent.mkdir(parents=True, exist_ok=True)
    first = True
    groups = {g: _result(g) for g in RESULT_GROUPS}
    groups.update(overrides or {})
    groups[PERF] = perf if perf is not None else _perf()
    for group, ds in groups.items():
        if group in drop:
            continue
        ds.to_zarr(store, group=group, mode="w" if first else "a", consolidated=False, zarr_format=3)
        first = False


def _run(tmp_path: Path, **rerun):
    _build(tmp_path / "bundle", "reference_outputs")
    _build(tmp_path / "analysis", "_test", **rerun)
    return compare_reference_outputs(tmp_path / "bundle", tmp_path / "analysis")


def test_an_unchanged_rerun_passes_and_lists_only_stamps_and_timers(tmp_path):
    """Property: the control case. Identical trees pass, and the only exclusions are the two stamps in each of the five
    result stores and the two float timers of the performance group, nothing else.

    Must-fire control: the exclusion list is non-empty and exact, so a comparison that skips everything fails here.
    """
    result = _run(tmp_path)
    assert result.passed
    names = sorted((s.rsplit("/", 2)[-2], v) for s, v, _r in result.exclusions)
    assert names.count(("performance", "Total")) == 1 and names.count(("performance", "Compute_min")) == 1
    assert len([n for n in names if n[1].startswith("hhemt_producing_")]) == 2 * len(RESULT_GROUPS)
    assert len(names) == 2 * len(RESULT_GROUPS) + 2


@pytest.mark.parametrize("group", RESULT_GROUPS)
def test_a_changed_scientific_float_in_each_result_store_fails(tmp_path, group):
    """Property: one changed float in any of the five result stores fails the comparison and names that store.

    Class: the five stores that carry the results. Plausible wrong implementation: an exclusion keyed on a group suffix
    or dtype that also covers a result store. Must-fire control: the unchanged case above passes.
    """
    result = _run(tmp_path, overrides={group: _result(group, value=1.0000000001)})
    assert not result.passed
    assert group in result.summary() and "peak" in result.summary()


@pytest.mark.parametrize("group", RESULT_GROUPS)
def test_a_stamp_changed_together_with_a_result_fails_and_the_stamp_is_still_listed(tmp_path, group):
    """Property: a changed stamp does not hide a changed result in the same store."""
    result = _run(tmp_path, overrides={group: _result(group, value=2.0, sha="bbb")})
    assert not result.passed and "peak" in result.summary()


def test_a_changed_coordinate_or_id_in_a_stamp_bearing_store_fails(tmp_path):
    """Property: a changed `event_iloc` coordinate, a changed text result and a different length all fail in a store
    that also carries the stamps."""
    group = RESULT_GROUPS[0]
    assert not _run(tmp_path / "a", overrides={group: _result(group, events=(0, 2))}).passed
    renamed = _result(group).assign(label=("event_iloc", np.array(["x", "y"])))
    assert not _run(tmp_path / "b", overrides={group: renamed}).passed
    assert not _run(tmp_path / "c", overrides={group: _result(group, events=(0,))}).passed


@pytest.mark.parametrize(
    "perf",
    [_perf(n_ranks=8), _perf(host="node-b")],
    ids=["integer variable", "text variable"],
)
def test_a_changed_integer_or_text_performance_variable_fails(tmp_path, perf):
    """Property: only floating-point performance variables are excluded; an integer or text one is compared exactly."""
    assert not _run(tmp_path, perf=perf).passed


def test_a_changed_timer_alone_passes_including_a_zero_valued_one(tmp_path):
    """Property: a changed float timer, and a zero-valued float that is non-zero in the re-run, are excluded and
    listed."""
    result = _run(tmp_path, perf=_perf(total=99.0, zero=0.5))
    assert result.passed
    assert {"Total", "Compute_min"} <= {v for _s, v, _r in result.exclusions}


def test_a_name_that_only_looks_like_a_stamp_or_a_timer_is_compared(tmp_path):
    """Property: the stamp rule matches exact stamp names, and the timer rule applies to the performance group only. A
    variable `hhemt_producing_sha_extra`, and a float named like a timer in a result group, are compared and a change
    in either fails.

    Must-fire control: the exact stamp name changed alone passes (the rule is live, not absent).
    """
    group = RESULT_GROUPS[1]
    base = _result(group)
    look_alike = base.assign_coords(hhemt_producing_sha_extra=("event_iloc", ["a", "a"]))
    changed = base.assign_coords(hhemt_producing_sha_extra=("event_iloc", ["a", "b"]))
    _build(tmp_path / "bundle", "reference_outputs", overrides={group: look_alike})
    _build(tmp_path / "analysis", "_test", overrides={group: changed})
    assert not compare_reference_outputs(tmp_path / "bundle", tmp_path / "analysis").passed
    timer_like = base.assign(Compute_extra=("event_iloc", np.array([1.0, 1.0])))
    timer_changed = base.assign(Compute_extra=("event_iloc", np.array([1.0, 2.0])))
    _build(tmp_path / "b2", "reference_outputs", overrides={group: timer_like})
    _build(tmp_path / "a2", "_test", overrides={group: timer_changed})
    assert not compare_reference_outputs(tmp_path / "b2", tmp_path / "a2").passed
    assert _run(tmp_path / "ctl", overrides={group: _result(group, sha="zzz")}).passed


def test_a_missing_performance_store_or_a_missing_timer_fails(tmp_path):
    """Property: a performance group the re-run did not produce, and a timer the re-run lacks, are failures and are not
    listed as exclusions."""
    missing_store = _run(tmp_path / "s", drop=(PERF,))
    assert not missing_store.passed
    no_timer = _run(tmp_path / "t", perf=_perf().drop_vars("Total"))
    assert not no_timer.passed and "Total" in no_timer.summary()
    assert all(v != "Total" for _s, v, _r in no_timer.exclusions)


def test_the_verdict_type_keeps_its_positional_construction(tmp_path):
    """Property: `ReferenceStoreVerdict(store, status)` still constructs with no exclusions, and an old-style
    comparison built from such verdicts reports no exclusions and no `not compared` line."""
    from hhemt.bundle._reprex import ReferenceComparison

    verdict = ReferenceStoreVerdict("s", "missing_in_rerun")
    assert verdict.excluded == () and verdict.variables == {}
    comparison = ReferenceComparison((verdict,), (), 0.0, 0.0)
    assert comparison.exclusions == () and "not compared" not in comparison.summary() and not comparison.passed
