"""Reviewer tests for chunk 1h.2 (DR-19): the per-rank performance summary beside the maximum-over-ranks one.

Written by the reviewing seat, against the criterion and not against the coder's tests. Each test names the defect it
catches. The coder greens them without weakening, deleting, skipping or rewriting any of them.

What they add to ``tests/test_performance_summary_by_rank.py``:

* the real ``_write_output`` choke point, which the coder's stand-in ``_Proc`` replaces, so the stamped store is read
  back from disk and not inspected in memory;
* the union-of-ranks join stated explicitly, because the group's whole point is the union and xarray's announced
  default change would turn it into an error;
* a per-scenario by-rank store the model log does not record as written is never trusted;
* the NaN semantics the store's own ``notes`` attribute promises.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.process_simulation import (
    TRITONSWMM_sim_post_processing,
    _write_performance_summary_by_rank,
    performance_summary_by_rank,
    performance_summary_by_rank_store_path,
)
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _RealPost
from hhemt.utils import write_zarr

_GROUP = "tritonswmm/performance/summary_by_rank"
_SUMMARY_NAME = "TRITONSWMM_perf_summary.zarr"
_COLS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")


def _series(ranks: tuple[int, ...], n_steps: int = 4, event_iloc: int = 0, base: float = 1.0) -> xr.Dataset:
    """A per-rank series as ``_export_performance_tseries`` hands it on: dims ``(event_iloc, timestep, Rank)``."""
    shape = (1, n_steps, len(ranks))
    data = {}
    for i, col in enumerate(_COLS):
        rate = base * (i + 1)
        grid = np.empty(shape)
        for r, rank in enumerate(ranks):
            grid[0, :, r] = rate * (rank + 1)
        data[col] = (("event_iloc", "timestep", "Rank"), grid)
    return xr.Dataset(
        data,
        coords={"event_iloc": [event_iloc], "timestep": np.arange(1, n_steps + 1), "Rank": list(ranks)},
    )


# --------------------------------------------------------------------------------------
# the reduction itself
# --------------------------------------------------------------------------------------


def test_a_column_emitted_for_only_some_steps_is_nan_at_every_rank_and_never_a_partial_sum():
    """The store's ``notes`` promise: a short column reads NaN, never a finite understated total."""
    ds = _series((0, 1, 2))
    ds["SWMM"] = ds["SWMM"].where(ds["timestep"] <= 2)

    by_rank = performance_summary_by_rank(ds)

    assert by_rank["SWMM"].isnull().all(), "a partial column reduced to a finite number"
    assert by_rank["Total"].notnull().all(), "the NaN must not spread to a column that was complete"


def test_a_rank_absent_from_early_steps_is_nan_for_that_rank_alone():
    """The Rank axis is the union across allocations; a rank that did not exist for a step is NaN for it."""
    ds = _series((0, 1, 2))
    ds["Total"] = ds["Total"].where(~((ds["Rank"] == 2) & (ds["timestep"] <= 2)))

    by_rank = performance_summary_by_rank(ds)

    assert bool(by_rank["Total"].sel(Rank=2).isnull().all())
    assert bool(by_rank["Total"].sel(Rank=[0, 1]).notnull().all())


def test_the_reduction_adds_no_derived_variable_and_collapses_nothing():
    """Only the series' own columns come back: no ``_min``, ``_coherent``, ``coherent_rank`` or ``n_ranks``."""
    ds = _series((0, 1, 2))

    by_rank = performance_summary_by_rank(ds)

    assert set(by_rank.data_vars) == set(ds.data_vars)
    assert "timestep" not in by_rank.dims
    assert [int(r) for r in by_rank["Rank"].values] == [0, 1, 2], "Rank must stay a labelled axis"


# --------------------------------------------------------------------------------------
# the real write choke point
# --------------------------------------------------------------------------------------


def _real_proc(tmp_path: Path, out_type: str = "zarr"):
    """A ``self`` that supplies the real ``_write_output`` with everything it reads, and records the log entry."""
    entries: list[Path] = []
    proc = SimpleNamespace()
    proc._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(target_processed_output_type=out_type, model_dump=lambda: {}),
        _dict_of_all_sim_files=lambda event_iloc: {},
    )
    proc._system = SimpleNamespace(cfg_system=SimpleNamespace(model_dump=lambda: {}))
    proc._scenario = SimpleNamespace(event_iloc=0, latest_sim_date=lambda **kwargs: "2026-01-01")
    proc._current_model_type = "tritonswmm"
    proc._resolve_producing_stamp = lambda: ("abc1234", "9.9.9")
    proc.log = SimpleNamespace(add_sim_processing_entry=lambda path, *rest: entries.append(Path(path)))
    proc._write_output = lambda ds, f_out, compression_level, verbose, mode: (
        TRITONSWMM_sim_post_processing._write_output(proc, ds, f_out, compression_level, verbose, mode)
    )
    proc.entries = entries
    return proc


def test_the_stamped_store_read_back_from_disk_carries_time_sum_dims_and_the_producing_stamp(tmp_path):
    """Through the real ``_write_output``: what a reader of the per-scenario store sees."""
    proc = _real_proc(tmp_path)
    out = performance_summary_by_rank_store_path(tmp_path / "processed" / _SUMMARY_NAME)

    _write_performance_summary_by_rank(proc, _series((0, 1, 2)), out, 5, False, "tritonswmm_performance")

    stored = xr.open_zarr(out, chunks=None, consolidated=False).load()
    assert proc.entries == [out]
    for col in _COLS:
        assert stored[col].dims == ("event_iloc", "Rank"), col
        assert stored[col].attrs["cell_methods"] == "time: sum", col
        assert stored[col].attrs["units"], f"{col} lost its units"
    assert stored.sizes["Rank"] == 3
    assert stored["hhemt_producing_sha"].dims == ("event_iloc",)
    assert str(stored["hhemt_producing_sha"].values[0]) == "abc1234"
    assert stored.attrs["units"] == "seconds" or stored.attrs.get("notes"), "the store carries no description"


def test_every_stored_long_name_states_the_per_rank_form_and_none_keeps_the_slowest_rank_attribution():
    """Over EVERY described variable, not a hand-picked list: the parent's slowest-rank phrase is false here."""
    from hhemt.cf_conventions import _CF_PERFORMANCE_SUMMARY_BY_RANK_VARIABLES, _CF_PERFORMANCE_VARIABLES

    assert set(_CF_PERFORMANCE_SUMMARY_BY_RANK_VARIABLES) == set(_CF_PERFORMANCE_VARIABLES)
    for name, entry in _CF_PERFORMANCE_SUMMARY_BY_RANK_VARIABLES.items():
        text = str(entry["long_name"]).lower()
        assert "slowest" not in text and "maximum" not in text and "max(" not in text, f"{name}: {text}"
        assert "rank" in text, f"{name}: the per-rank form is not stated"
        assert entry["units"] == _CF_PERFORMANCE_VARIABLES[name]["units"], f"{name}: units drifted"
        assert "Rank" not in str(entry["cell_methods"]), f"{name}: a Rank pair names nothing collapsed"


# --------------------------------------------------------------------------------------
# member consolidation
# --------------------------------------------------------------------------------------


class _Post:
    _PERFORMANCE_BY_RANK_GROUPS = _RealPost._PERFORMANCE_BY_RANK_GROUPS
    _mode_model = staticmethod(_RealPost._mode_model)
    _timeseries_group_state = staticmethod(_RealPost._timeseries_group_state)
    _event_performance_by_rank = _RealPost._event_performance_by_rank
    _consolidate_performance_by_rank = _RealPost._consolidate_performance_by_rank

    def __init__(self, analysis_dir: Path, n: int):
        self._analysis = SimpleNamespace(
            df_sims=pd.DataFrame(index=list(range(n))), analysis_paths=SimpleNamespace(analysis_dir=analysis_dir)
        )

    def _open_engine(self):
        return "zarr"

    def _event_slug(self, event_iloc: int) -> str:
        return f"evt_{event_iloc}"


def _scenario(event_iloc: int, sim_dir: Path, outputs: dict):
    processed = sim_dir / f"evt_{event_iloc}" / "processed"
    return SimpleNamespace(
        event_iloc=event_iloc,
        scen_paths=SimpleNamespace(
            output_tritonswmm_performance_summary=processed / _SUMMARY_NAME,
            output_tritonswmm_performance_timeseries=processed / "TRITONSWMM_perf_tseries.zarr",
            output_triton_only_performance_summary=None,
            output_triton_only_performance_timeseries=None,
        ),
        get_log=lambda model: SimpleNamespace(processing_log=SimpleNamespace(outputs=outputs)),
    )


def _install(tmp_path, monkeypatch, n: int):
    sim_dir = tmp_path / "sims"
    records = {ei: {} for ei in range(n)}
    scens = {ei: _scenario(ei, sim_dir, records[ei]) for ei in range(n)}
    monkeypatch.setattr("hhemt.processing_analysis.TRITONSWMM_scenario", lambda ei, analysis: scens[ei])
    return _Post(tmp_path / "analysis", n), records, scens, tmp_path / "analysis_datatree.zarr"


def _write_recorded(path: Path, ds: xr.Dataset, record: dict, *, success: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr(ds, path)
    record[path.name] = SimpleNamespace(success=success)


def test_the_union_of_ranks_join_is_explicit_so_a_future_xarray_default_cannot_turn_it_into_an_error(
    tmp_path, monkeypatch
):
    """Scenarios of 3 and 2 ranks join on the union. xarray warns that its default will become ``exact``."""
    post, records, scens, store = _install(tmp_path, monkeypatch, 2)
    for ei, ranks in ((0, (0, 1, 2)), (1, (0, 1))):
        path = performance_summary_by_rank_store_path(scens[ei].scen_paths.output_tritonswmm_performance_summary)
        _write_recorded(path, performance_summary_by_rank(_series(ranks, event_iloc=ei)), records[ei])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        written = post._consolidate_performance_by_rank(store, compression_level=5)

    assert written == [_GROUP]
    stored = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    assert [int(r) for r in stored["Rank"].values] == [0, 1, 2]


def test_a_per_scenario_by_rank_store_the_log_does_not_record_is_never_read(tmp_path, monkeypatch):
    """A crash between the zarr write and the log entry leaves a store on disk that may be partial.

    The series store is recorded, so the group is built from it; the unrecorded store's sentinel value must not
    reach the group.
    """
    post, records, scens, store = _install(tmp_path, monkeypatch, 1)
    summary = scens[0].scen_paths.output_tritonswmm_performance_summary
    series = _series((0, 1), event_iloc=0)
    sentinel = performance_summary_by_rank(series) * 0 - 1.0
    _write_recorded(performance_summary_by_rank_store_path(summary), sentinel, {}, success=False)
    _write_recorded(scens[0].scen_paths.output_tritonswmm_performance_timeseries, series, records[0])

    assert post._consolidate_performance_by_rank(store, compression_level=5) == [_GROUP]

    stored = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    assert float(stored["Total"].sel(event_iloc=0, Rank=1)) == pytest.approx(
        float(series["Total"].sum("timestep").sel(event_iloc=0, Rank=1))
    )
    assert bool((stored["Total"] > 0).all())


def test_a_store_recorded_as_failed_is_not_a_source_either(tmp_path, monkeypatch):
    """``success=False`` is a record, not a write: with no other source the scenario is named and nothing is written."""
    post, records, scens, store = _install(tmp_path, monkeypatch, 1)
    summary = scens[0].scen_paths.output_tritonswmm_performance_summary
    path = performance_summary_by_rank_store_path(summary)
    _write_recorded(path, performance_summary_by_rank(_series((0, 1))), records[0], success=False)

    with pytest.warns(UserWarning, match=r"evt_0"):
        written = post._consolidate_performance_by_rank(store, compression_level=5)

    assert written == []


def test_the_group_matches_the_per_scenario_store_value_for_value_on_every_event(tmp_path, monkeypatch):
    """Three events, equal rank counts: the consolidated group is the per-scenario stores stacked, nothing else."""
    post, records, scens, store = _install(tmp_path, monkeypatch, 3)
    expected = {}
    for ei in range(3):
        series = _series((0, 1, 2), event_iloc=ei, base=float(ei + 1))
        expected[ei] = performance_summary_by_rank(series)
        path = performance_summary_by_rank_store_path(scens[ei].scen_paths.output_tritonswmm_performance_summary)
        _write_recorded(path, expected[ei], records[ei])

    assert post._consolidate_performance_by_rank(store, compression_level=5) == [_GROUP]

    stored = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    assert [int(v) for v in stored["event_iloc"].values] == [0, 1, 2]
    for ei in range(3):
        for col in _COLS:
            np.testing.assert_array_equal(
                stored[col].sel(event_iloc=ei).values, expected[ei][col].sel(event_iloc=ei).values, err_msg=col
            )
    for col in _COLS:
        assert stored[col].attrs["cell_methods"] == "time: sum", col
    assert set(stored.data_vars) == set(_COLS), "the group holds the series' columns and nothing derived"


def test_a_tree_with_no_triton_family_in_its_paths_writes_no_group_and_warns_of_nothing(tmp_path, monkeypatch):
    """A family whose scenario paths are None is not produced by the analysis: no group, no warning."""
    post, _records, scens, store = _install(tmp_path, monkeypatch, 1)
    scens[0].scen_paths.output_tritonswmm_performance_summary = None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert post._consolidate_performance_by_rank(store, compression_level=5) == []
