"""AC-1.9: every consolidated store carries a per-rank performance summary beside the maximum-over-ranks one (DR-19).

Builds a three-rank ``performance{N}.txt`` directory by hand and runs no solver, no scenario and no analysis.
The per-scenario export is driven through the production function over a duck-typed stand-in for the
post-processing object (the same shape ``tests/test_perf_rank_coherent_family.py`` uses), and the consolidation of
the group is driven through the production ``_consolidate_performance_by_rank`` over stand-in scenarios whose
per-scenario stores are real zarr stores written by the production writer.

THE CLASSES (CO3).  ``test_the_per_rank_summary_keeps_every_rank_and_sums_each_column_over_timestep`` and
``test_the_per_rank_summary_carries_cell_methods_time_sum_and_the_summary_keeps_its_two_pairs`` are NEW CAPABILITY:
the plausible wrong implementation is the maximum-over-ranks reduction the summary already applies (Rank collapsed to
one value), under which the Rank-size and the per-rank-value assertions fail; and the wrong constant is the
summary's own two-pair string, under which the ``cell_methods`` equality fails.
``test_the_maximum_over_ranks_summary_is_unchanged_by_the_per_rank_export`` is REGRESSION-shaped: it holds the
pre-chunk expression and compares the written values to it. Each has a differently-positioned satisfying input: the
three ranks carry distinct rates so no rank is the maximum of every column.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes
from hhemt.process_simulation import (
    PERFORMANCE_BY_RANK_DIRNAME,
    TRITONSWMM_sim_post_processing,
    _aggregate_perf_tseries,
    performance_summary_by_rank,
    performance_summary_by_rank_store_path,
)
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _RealPost
from hhemt.utils import write_zarr

_BASE_COLS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")
_CATEGORIES = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other")
_N_STEPS = 4
_SUMMARY_NAME = "TRITONSWMM_perf_summary.zarr"
_BY_RANK_NAME = "TRITONSWMM_perf_summary_by_rank.zarr"
_GROUP = "tritonswmm/performance/summary_by_rank"

#: Per-rank, per-unit-timestep rates. Distinct on every rank and every column, so rank 2 is the slowest on every
#: column and rank 0 the fastest: a reduction that collapses ``Rank`` to its maximum returns rank 2's value for all
#: three, and a per-rank value check then fails on ranks 0 and 1.
_RATES = {
    0: {"Compute": 2.0, "MPI": 1.0, "IO": 0.5, "Resize": 0.25, "SWMM": 0.75, "Other": 0.125, "Init": 0.5},
    1: {"Compute": 4.0, "MPI": 2.0, "IO": 1.0, "Resize": 0.5, "SWMM": 1.5, "Other": 0.25, "Init": 1.0},
    2: {"Compute": 8.0, "MPI": 4.0, "IO": 2.0, "Resize": 1.0, "SWMM": 3.0, "Other": 0.5, "Init": 2.0},
}


def _per_rank_expected(rank: int) -> dict[str, float]:
    """The time sum the hand-built files imply on ``rank``: the cumulative file values step by ``rate`` each step."""
    rates = _RATES[rank]
    simulation = sum(rates[c] for c in _CATEGORIES)
    full = {**rates, "Simulation": simulation, "Total": simulation + rates["Init"]}
    return {col: full[col] * _N_STEPS for col in _BASE_COLS}


def _write_perf_file(perf_dir: Path, tstep: int, ranks) -> None:
    """Write one ``performance{tstep}.txt`` in TRITON's emitted shape; the values are cumulative, ``rate * tstep``."""
    content = "%" + ", ".join(["Rank", *_BASE_COLS]) + "\n"
    for rank in ranks:
        rates = _RATES[rank]
        simulation = sum(rates[c] for c in _CATEGORIES)
        row = {**rates, "Simulation": simulation, "Total": simulation + rates["Init"]}
        content += f"{rank}, " + ", ".join(f"{row[c] * tstep}" for c in _BASE_COLS) + "\n"
    # parse_performance_file requires the Average row and drops it.
    content += "Average, " + ", ".join(["0"] * len(_BASE_COLS)) + "\n"
    (perf_dir / f"performance{tstep}.txt").write_text(content)


def _perf_dir(root: Path, ranks) -> Path:
    perf_dir = root / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)
    for tstep in range(1, _N_STEPS + 1):
        _write_perf_file(perf_dir, tstep, ranks)
    return perf_dir


def _tseries(perf_dir: Path, event_iloc: int) -> xr.Dataset:
    """The per-rank series exactly as ``_export_performance_tseries`` hands it on: ``event_iloc`` first."""
    ds = _aggregate_perf_tseries(perf_dir, resume_steps=[])
    ds.attrs = {}
    return ds.assign_coords(coords=dict(event_iloc=event_iloc)).expand_dims("event_iloc")


@pytest.fixture
def three_rank_series(tmp_path) -> xr.Dataset:
    return _tseries(_perf_dir(tmp_path / "three", (0, 1, 2)), event_iloc=7)


class _Proc:
    """A duck-typed ``self`` for ``_export_performance_summary``, recording every write it would have made."""

    def __init__(self, already: tuple[str, ...] = ()):
        self.already = set(already)
        self.writes: list[tuple[Path, xr.Dataset, str]] = []
        self.log = self

    # the `_deferred_log_writes` decorator's requirement
    @contextlib.contextmanager
    def deferred_writes(self):
        yield

    def add_sim_processing_entry(self, *args, **kwargs):
        return None

    def set(self, value):  # the `log_field` stand-in
        return None

    def _already_written(self, f_out):
        return Path(f_out).name in self.already

    def _summary_usable(self, f_out):
        return Path(f_out).name in self.already

    def _write_output(self, ds, f_out, compression_level, verbose, mode):
        self.writes.append((Path(f_out), ds, mode))
        # `get_file_size_MiB` rglobs a `.zarr` path, so the directory must exist.
        Path(f_out).mkdir(parents=True, exist_ok=True)
        (Path(f_out) / ".zattrs").write_text("{}")


def _export(proc: _Proc, ds: xr.Dataset, processed: Path) -> None:
    TRITONSWMM_sim_post_processing._export_performance_summary(
        proc,
        ds=ds,
        fname_out=processed / _SUMMARY_NAME,
        compression_level=5,
        verbose=False,
        log_field=proc,
        mode="tritonswmm_performance",
        verify_present=True,
    )


# --------------------------------------------------------------------------------------
# AC-1.9: the group's shape, values and cell_methods
# --------------------------------------------------------------------------------------


def test_the_per_rank_summary_keeps_every_rank_and_sums_each_column_over_timestep(tmp_path, three_rank_series):
    """Dims ``(event_iloc, Rank)`` with three ranks; every column is the series summed over ``timestep`` per rank."""
    proc = _Proc()
    _export(proc, three_rank_series, tmp_path / "processed")

    by_rank_writes = [w for w in proc.writes if w[0].name == _BY_RANK_NAME]
    assert len(by_rank_writes) == 1, f"expected one per-rank write, got {[w[0].name for w in proc.writes]}"
    path, by_rank, mode = by_rank_writes[0]

    assert path == tmp_path / "processed" / PERFORMANCE_BY_RANK_DIRNAME / _BY_RANK_NAME
    assert mode == "tritonswmm_performance_by_rank"
    assert by_rank.sizes["Rank"] == 3, "the maximum-over-ranks reduction would leave one Rank"
    assert by_rank.sizes["event_iloc"] == 1
    assert set(by_rank.data_vars) == set(three_rank_series.data_vars)
    assert set(_BASE_COLS) <= set(by_rank.data_vars)
    for name, da in by_rank.data_vars.items():
        assert da.dims == ("event_iloc", "Rank"), f"{name} has dims {da.dims}"

    series_sum = three_rank_series.sum(dim="timestep", skipna=False)
    for col in _BASE_COLS:
        np.testing.assert_array_equal(by_rank[col].values, series_sum[col].values)
        for rank in (0, 1, 2):
            got = float(by_rank[col].isel(event_iloc=0).sel(Rank=rank))
            assert got == pytest.approx(_per_rank_expected(rank)[col]), f"{col} at rank {rank}"

    # Control: the ranks differ, so the assertions above separate a per-rank value from a collapsed one.
    assert float(by_rank["Compute"].sel(Rank=0)) != float(by_rank["Compute"].sel(Rank=2))


def test_the_per_rank_summary_carries_cell_methods_time_sum_and_the_summary_keeps_its_two_pairs(
    tmp_path, three_rank_series
):
    """The per-rank mode stamps ``time: sum`` and no ``Rank`` pair; the summary's own mode stamps both pairs."""
    proc = _Proc()
    _export(proc, three_rank_series, tmp_path / "processed")
    (_, by_rank, by_rank_mode) = next(w for w in proc.writes if w[0].name == _BY_RANK_NAME)
    (_, summary, summary_mode) = next(w for w in proc.writes if w[0].name == _SUMMARY_NAME)

    by_rank = apply_cf_attributes(by_rank.copy(deep=True), by_rank_mode)
    summary = apply_cf_attributes(summary.copy(deep=True), summary_mode)

    for col in _BASE_COLS:
        assert by_rank[col].attrs["cell_methods"] == "time: sum", col
        assert summary[col].attrs["cell_methods"] == "time: sum Rank: maximum", col
        assert by_rank[col].attrs["units"] == summary[col].attrs["units"], f"{col}: units drifted between the two"
        assert "slowest rank" not in by_rank[col].attrs["long_name"], (
            f"{col}: the per-rank long_name kept the slowest-rank attribution that is true of the maximum only"
        )


def test_the_maximum_over_ranks_summary_is_unchanged_by_the_per_rank_export(tmp_path, three_rank_series):
    """``performance/summary`` holds the pre-chunk reduction's values and shape, byte for byte on the base columns."""
    proc = _Proc()
    _export(proc, three_rank_series, tmp_path / "processed")
    (_, summary, _mode) = next(w for w in proc.writes if w[0].name == _SUMMARY_NAME)

    legacy = three_rank_series.sum(dim="timestep", skipna=False).max(dim="Rank", keepdims=True)
    for col in _BASE_COLS:
        assert summary[col].dims == legacy[col].dims
        assert summary[col].values.tobytes() == legacy[col].values.tobytes(), f"{col} moved"
    assert summary.sizes["Rank"] == 1, "the summary keeps its retained size-one Rank"


def test_the_per_rank_store_is_written_before_the_summary_and_never_next_to_it(tmp_path, three_rank_series):
    proc = _Proc()
    _export(proc, three_rank_series, tmp_path / "processed")

    assert [w[0].name for w in proc.writes] == [_BY_RANK_NAME, _SUMMARY_NAME]
    assert proc.writes[0][0].parent.name == PERFORMANCE_BY_RANK_DIRNAME


def test_a_summary_written_before_the_per_rank_store_existed_gets_it_on_the_skip_pass(tmp_path, three_rank_series):
    """The summary latched in the log, the per-rank store absent: only the per-rank store is written."""
    proc = _Proc(already=(_SUMMARY_NAME,))
    _export(proc, three_rank_series, tmp_path / "processed")

    assert [w[0].name for w in proc.writes] == [_BY_RANK_NAME], "the summary must not be rewritten"
    np.testing.assert_array_equal(
        proc.writes[0][1]["Total"].values, three_rank_series.sum(dim="timestep", skipna=False)["Total"].values
    )


def test_both_stores_latched_in_the_log_write_nothing(tmp_path, three_rank_series):
    """Control for the two tests above: with both stores written, the skip pass writes nothing."""
    proc = _Proc(already=(_SUMMARY_NAME, _BY_RANK_NAME))
    _export(proc, three_rank_series, tmp_path / "processed")

    assert proc.writes == []


def test_the_per_rank_store_path_is_a_subfolder_of_processed_for_each_model_and_suffix(tmp_path):
    """A subfolder, so the scenario-directory removal's non-recursive ``processed/*.zarr`` glob never sees it."""
    processed = tmp_path / "processed"
    coupled = performance_summary_by_rank_store_path(processed / "TRITONSWMM_perf_summary.zarr")
    triton_only = performance_summary_by_rank_store_path(processed / "TRITON_only_perf_summary.zarr")
    netcdf = performance_summary_by_rank_store_path(processed / "TRITONSWMM_perf_summary.nc")

    assert coupled == processed / PERFORMANCE_BY_RANK_DIRNAME / "TRITONSWMM_perf_summary_by_rank.zarr"
    assert triton_only == processed / PERFORMANCE_BY_RANK_DIRNAME / "TRITON_only_perf_summary_by_rank.zarr"
    assert netcdf.name == "TRITONSWMM_perf_summary_by_rank.nc"
    assert coupled != triton_only
    for store in (coupled, triton_only, netcdf):
        assert store.parent.parent == processed


# --------------------------------------------------------------------------------------
# AC-1.9: member consolidation carries the group
# --------------------------------------------------------------------------------------


class _Post:
    """The production per-rank consolidation methods over a stand-in analysis (two scenarios, no solver)."""

    _PERFORMANCE_BY_RANK_GROUPS = _RealPost._PERFORMANCE_BY_RANK_GROUPS
    _mode_model = staticmethod(_RealPost._mode_model)
    _timeseries_group_state = staticmethod(_RealPost._timeseries_group_state)
    _event_performance_by_rank = _RealPost._event_performance_by_rank
    _consolidate_performance_by_rank = _RealPost._consolidate_performance_by_rank

    def __init__(self, analysis_dir: Path):
        self._analysis = SimpleNamespace(
            df_sims=pd.DataFrame(index=[0, 1]), analysis_paths=SimpleNamespace(analysis_dir=analysis_dir)
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


@pytest.fixture
def two_scenarios(tmp_path, monkeypatch):
    """Scenario 0: three ranks, its own per-rank store. Scenario 1: two ranks, only the per-rank SERIES store.

    Scenario 1 stands for a scenario processed before the per-rank store existed, so its slice is derived from the
    series by the same function the export uses. Returns the stand-in, the two output records and the analysis
    directory.
    """
    sim_dir = tmp_path / "sims"
    records: dict[int, dict] = {0: {}, 1: {}}
    scens = {ei: _scenario(ei, sim_dir, records[ei]) for ei in (0, 1)}
    monkeypatch.setattr("hhemt.processing_analysis.TRITONSWMM_scenario", lambda ei, analysis: scens[ei])

    series0 = _tseries(_perf_dir(tmp_path / "s0", (0, 1, 2)), event_iloc=0)
    series1 = _tseries(_perf_dir(tmp_path / "s1", (0, 1)), event_iloc=1)

    by_rank_path = performance_summary_by_rank_store_path(scens[0].scen_paths.output_tritonswmm_performance_summary)
    by_rank_path.parent.mkdir(parents=True)
    write_zarr(performance_summary_by_rank(series0), by_rank_path)
    records[0][by_rank_path.name] = SimpleNamespace(success=True)

    series1_path = scens[1].scen_paths.output_tritonswmm_performance_timeseries
    series1_path.parent.mkdir(parents=True)
    write_zarr(series1, series1_path)
    records[1][series1_path.name] = SimpleNamespace(success=True)

    return _Post(tmp_path / "analysis"), records, tmp_path / "analysis_datatree.zarr", scens


def test_member_consolidation_writes_the_group_with_every_event_and_the_union_of_ranks(two_scenarios):
    post, _records, store, _scens = two_scenarios

    written = post._consolidate_performance_by_rank(store, compression_level=5)

    assert written == [_GROUP]
    stored = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    assert stored["Total"].dims == ("event_iloc", "Rank")
    assert [int(v) for v in stored["event_iloc"].values] == [0, 1]
    assert [str(v) for v in stored["event_id"].values] == ["evt_0", "evt_1"]
    assert [int(v) for v in stored["Rank"].values] == [0, 1, 2]
    assert stored["Total"].attrs["cell_methods"] == "time: sum"
    compute = stored["Compute"]
    for rank in (0, 1, 2):
        assert float(compute.sel(event_iloc=0, Rank=rank)) == pytest.approx(_per_rank_expected(rank)["Compute"])
    # Scenario 1 came from its series store and ran two ranks: rank 2 is the join's NaN.
    for rank in (0, 1):
        assert float(compute.sel(event_iloc=1, Rank=rank)) == pytest.approx(_per_rank_expected(rank)["Compute"])
    assert np.isnan(float(compute.sel(event_iloc=1, Rank=2)))


def test_a_scenario_with_no_source_leaves_the_group_unwritten_and_names_the_scenario(two_scenarios):
    post, records, store, _scens = two_scenarios
    records[1].clear()

    with pytest.warns(UserWarning, match=r"evt_1"):
        written = post._consolidate_performance_by_rank(store, compression_level=5)

    assert written == []
    assert not (store / _GROUP).exists(), "a group with rows for some events only must not be written"


def test_a_removed_scenario_leaves_an_existing_group_untouched(two_scenarios):
    post, records, store, _scens = two_scenarios
    assert post._consolidate_performance_by_rank(store, compression_level=5) == [_GROUP]
    before = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    records[0].clear()
    records[1].clear()

    assert post._consolidate_performance_by_rank(store, compression_level=5) == []

    after = xr.open_zarr(store, group=_GROUP, chunks=None, consolidated=False).load()
    xr.testing.assert_identical(before, after)
