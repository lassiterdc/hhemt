"""Reviewer tests for chunk 1k.2 (open-water outfall readers, ingestion and CF entries). Uncommitted by design.

Tests promoted from the chunk's own test file (their original description follows).

Readers and ingestion for the driven-stage open-water outfall capture (chunk 1k.2).

Chunk test: it guards ``coupling_diagnostics.read_outfall_stage`` and its reporting-interval mean, the two optional
``domain_balance.csv`` columns, the node-store attachment ``_attach_open_water_outfall_outputs``, the summary drop and
the CF and provenance entries. It is promoted into a module test or deleted at the run's final landing.

Every file under test is written here from the byte layout the shape fixes (section 17.3 of the approved shape: header
magic ``0x444F5331`` "DOS1", then per SWMM step ``{f64 simtime, f64 dt, f64[n] stage_m, f64[n] q_out_m3s,
f64[n] q_back_applied_m3s}``), with ``struct`` and hand-chosen values that differ per field and per outfall, so a reader
that permutes two fields or two outfalls returns a different number than the oracle. The expected values are written
down in the tests, never read back from the code under test.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt import timeseries_selection as ts
from hhemt.cf_conventions import _CF_VARIABLE_MAP, _QUANTITY_PROVENANCE, apply_cf_attributes
from hhemt.coupling_diagnostics import (
    MAGIC_EDGE_FLUX,
    MAGIC_OUTFALL_STAGE,
    outfall_stage_interval_means,
    read_domain_balance,
    read_outfall_stage,
)
from hhemt.exceptions import ProcessingError


def _ds(ends, dts, stage, q):
    return xr.Dataset(
        {
            "outfall_stage_m": (("swmm_step", "node_id"), np.asarray(stage, float)),
            "outfall_discharge_m3s": (("swmm_step", "node_id"), np.asarray(q, float)),
        },
        coords={
            "swmm_step": np.arange(len(ends)),
            "node_id": np.asarray(["A", "B"], dtype=object),
            "simtime": ("swmm_step", np.asarray(ends, float)),
            "dt": ("swmm_step", np.asarray(dts, float)),
        },
    )


@pytest.mark.parametrize("seed", range(20))
def test_interval_mean_matches_a_fine_grid_integral(seed):
    """Oracle independent of the cumulative-sum construction: integrate on a 0.25 s grid with integer-quarter steps."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 40))
    dts = rng.integers(1, 13, size=n) * 0.25
    ends = np.cumsum(dts)
    stage = rng.normal(size=(n, 2))
    q = rng.normal(size=(n, 2))
    r = float(rng.integers(2, 9)) * 0.5
    k_max = int(np.floor(ends[-1] / r))
    if k_max < 2:
        pytest.skip("record shorter than two intervals")
    times = r * np.arange(1, k_max + 1)
    means = outfall_stage_interval_means(_ds(ends, dts, stage, q), times)
    grid = 0.25
    mid = (np.arange(int(round(ends[-1] / grid))) + 0.5) * grid
    step_of = np.searchsorted(ends, mid, side="left")
    for i, t in enumerate(times):
        sel = (mid > t - r) & (mid <= t)
        exp_stage = stage[step_of[sel]].sum(axis=0) * grid / r
        exp_q = q[step_of[sel]].sum(axis=0) * grid / r
        np.testing.assert_allclose(means.stage_m[i], exp_stage, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(means.discharge_m3s[i], exp_q, rtol=1e-10, atol=1e-12)


def test_the_selection_domain_names_the_new_time_series_variables():
    """The closed selection table must list every time-variant variable the coupled node store can gain."""
    domain = set(ts.DOMAIN[("swmm_node", "tritonswmm")])
    assert {"outfall_stage_m", "outfall_discharge_m3s"} <= domain


def test_a_keep_list_may_name_the_outfall_series():
    kept = ts.resolve_keep("swmm_node", "tritonswmm", ["depth", "outfall_stage_m"])
    assert "outfall_stage_m" in kept


# The magic is spelled here as a literal, not imported, so a wrong constant in the module fails against the value the
# solver's file format fixes ("DOS1" read as a little-endian int32 is 0x31534F44 and as written is 0x444F5331).
DOS1 = 0x444F5331


BASE_BALANCE_COLUMNS = [
    "t_end",
    "d_storage_m3",
    "runoff_m3",
    "inflow_m3",
    "extbc_m3",
    "clamp_m3",
    "swmm_exchange_m3",
    "edge_west_m3",
    "edge_east_m3",
    "edge_south_m3",
    "edge_north_m3",
    "residual_m3",
]


OUTFALL_BALANCE_COLUMNS = ["outfall_backflow_m3", "outfall_backflow_deficit_m3"]


def _stage_header(n_items: int, *, magic: int = DOS1, version: int = 1, value_width: int = 8) -> bytes:
    return struct.pack("<iiii", magic, version, value_width, n_items)


def _stage_record(simtime, dt, stage, q_out, q_back) -> bytes:
    n = len(stage)
    return struct.pack(f"<dd{n}d{n}d{n}d", simtime, dt, *stage, *q_out, *q_back)


def _write_stage(path: Path, names, records, *, header: bytes | None = None) -> Path:
    """Write ``outfall_stage.bin`` and its ``outfall_stage.nodes`` sidecar beside it; return the binary's path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header if header is not None else _stage_header(len(names))) + b"".join(records))
    path.with_name("outfall_stage.nodes").write_text("".join(f"{n}\n" for n in names))
    return path


def _two_outfall_file(tmp_path: Path) -> Path:
    """Three steps of two outfalls; every number differs, and the second outfall is in backflow in step 2."""
    return _write_stage(
        tmp_path / "outfall_stage.bin",
        ["E143250", "E147007"],
        [
            _stage_record(1.0, 1.0, [0.5, 1.5], [2.0, 3.0], [0.0, 0.0]),
            _stage_record(2.0, 1.0, [0.75, 1.25], [-4.0, -5.0], [4.0, 5.0]),
            _stage_record(4.5, 2.5, [0.875, 1.125], [6.0, 7.0], [0.0, 0.25]),
        ],
    )


def test_the_magic_constant_is_the_one_the_solver_writes():
    """Property: ``MAGIC_OUTFALL_STAGE`` is ``0x444F5331`` ("DOS1"), distinct from every other capture magic.

    Class: the constant the header check compares against. Why: a wrong constant here would make the reader refuse
    every file the solver writes, or accept another capture's file. What kills it: ``0x31534F44`` (the byte-swapped
    spelling) or a copy of another capture's magic. Second correct implementation: building the constant with
    ``struct.unpack("<i", b"1SOD")``.
    """
    assert MAGIC_OUTFALL_STAGE == DOS1
    assert MAGIC_OUTFALL_STAGE != MAGIC_EDGE_FLUX
    assert struct.pack("<i", MAGIC_OUTFALL_STAGE) == b"1SOD"


def test_a_fixture_outfall_stage_file_decodes_every_field_of_every_outfall(tmp_path):
    """Property: each of the five record fields lands in its own variable, per outfall, in sidecar order, at its step.

    Class: any file of whole records for any outfall count. Why: the three per-outfall arrays follow one another in
    each record, so a reader that interleaves them (outfall-major) or swaps the stage and the discharge returns
    plausible numbers of the right shape. What kills it: a record laid out ``[stage, q_out, q_back] * n``, or the
    discharge read from the stage slot. Second correct implementation: a ``struct``-walk reader with no memory map.
    Control: the step length is 2.5 in the third record while the first two are 1.0, so a reader that takes ``dt``
    from the first record or from the stamp difference differs.
    """
    ds = read_outfall_stage(_two_outfall_file(tmp_path), tmp_path / "outfall_stage.nodes")

    assert ds["outfall_stage_m"].dims == ("swmm_step", "node_id")
    assert list(ds["node_id"].values) == ["E143250", "E147007"]
    np.testing.assert_array_equal(ds["simtime"].values, [1.0, 2.0, 4.5])
    np.testing.assert_array_equal(ds["dt"].values, [1.0, 1.0, 2.5])
    np.testing.assert_array_equal(ds["outfall_stage_m"].values, [[0.5, 1.5], [0.75, 1.25], [0.875, 1.125]])
    np.testing.assert_array_equal(ds["outfall_discharge_m3s"].values, [[2.0, 3.0], [-4.0, -5.0], [6.0, 7.0]])
    np.testing.assert_array_equal(ds["outfall_backflow_applied_m3s"].values, [[0.0, 0.0], [4.0, 5.0], [0.0, 0.25]])
    assert ds["outfall_stage_m"].attrs["units"] == "m"
    assert ds["outfall_discharge_m3s"].attrs["units"] == "m3 s-1"


def test_a_header_only_file_is_an_empty_capture_not_a_refusal(tmp_path):
    """Property: a header and no record reads as a dataset with zero steps (the check that flags an empty capture
    reads the step dimension, so the reader must return it rather than refuse)."""
    path = _write_stage(tmp_path / "outfall_stage.bin", ["E1"], [])
    ds = read_outfall_stage(path, tmp_path / "outfall_stage.nodes")
    assert ds.sizes["swmm_step"] == 0 and list(ds["node_id"].values) == ["E1"]


def test_a_wrong_magic_refuses_and_names_both_magics(tmp_path):
    """Property: a file whose header carries another capture's magic is refused, naming the found and expected magic.

    Class: every magic other than ``DOS1``. Must-fire control: the same bytes with the right magic decode (the test
    above), so the refusal is the magic's and not a layout error.
    """
    path = _write_stage(
        tmp_path / "outfall_stage.bin",
        ["E1"],
        [_stage_record(1.0, 1.0, [0.5], [1.0], [0.0])],
        header=_stage_header(1, magic=MAGIC_EDGE_FLUX),
    )
    with pytest.raises(ProcessingError, match=r"magic 0x44454631, expected 0x444f5331"):
        read_outfall_stage(path, tmp_path / "outfall_stage.nodes")


@pytest.mark.parametrize(
    ("header_kwargs", "message"),
    [({"version": 2}, "version 2"), ({"value_width": 4}, "value_width 4")],
)
def test_a_wrong_version_or_value_width_refuses(tmp_path, header_kwargs, message):
    """Property: the version (1) and the value width (8) are part of the header contract of every capture file."""
    path = _write_stage(
        tmp_path / "outfall_stage.bin",
        ["E1"],
        [_stage_record(1.0, 1.0, [0.5], [1.0], [0.0])],
        header=_stage_header(1, **header_kwargs),
    )
    with pytest.raises(ProcessingError, match=message):
        read_outfall_stage(path, tmp_path / "outfall_stage.nodes")


def test_a_header_width_that_differs_from_the_sidecar_refuses(tmp_path):
    """Property: ``n_items`` is the driven-outfall count and must equal the sidecar's name count.

    What kills it: a reader that trusts the sidecar alone (it would mis-slice every record of a file written for
    another count) or the header alone (it would label columns with too few or too many names).
    """
    path = _write_stage(
        tmp_path / "outfall_stage.bin",
        ["E1", "E2"],
        [_stage_record(1.0, 1.0, [0.5, 0.6], [1.0, 1.1], [0.0, 0.0])],
        header=_stage_header(3),
    )
    with pytest.raises(ProcessingError, match="header n_items is 3 but the sidecar lists 2"):
        read_outfall_stage(path, tmp_path / "outfall_stage.nodes")


def test_a_file_that_ends_inside_a_record_refuses(tmp_path):
    """Property: a partial trailing record is refused with its byte count, never truncated away."""
    path = _two_outfall_file(tmp_path)
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ProcessingError, match="not a whole number of 64-byte records"):
        read_outfall_stage(path, tmp_path / "outfall_stage.nodes")


def test_a_file_that_lists_no_outfall_refuses(tmp_path):
    """Property: a capture of zero outfalls is not a capture the solver writes; the reader refuses it."""
    path = _write_stage(tmp_path / "outfall_stage.bin", [], [])
    with pytest.raises(ProcessingError, match="lists no outfall"):
        read_outfall_stage(path, tmp_path / "outfall_stage.nodes")


def _stage_dataset(ends, dts, stage, q_out) -> xr.Dataset:
    """An in-memory result of ``read_outfall_stage`` for the interval-mean tests (one or two outfalls)."""
    stage = np.asarray(stage, dtype=float)
    q_out = np.asarray(q_out, dtype=float)
    return xr.Dataset(
        {
            "outfall_stage_m": (("swmm_step", "node_id"), stage),
            "outfall_discharge_m3s": (("swmm_step", "node_id"), q_out),
        },
        coords={
            "swmm_step": np.arange(len(ends)),
            "node_id": np.asarray(["A", "B"][: stage.shape[1]], dtype=object),
            "simtime": ("swmm_step", np.asarray(ends, dtype=float)),
            "dt": ("swmm_step", np.asarray(dts, dtype=float)),
        },
    )


def test_a_step_that_straddles_a_report_time_is_split_by_the_time_it_spends_in_each_interval():
    """Property: the mean over a reporting interval is the time integral of the stepwise series over it divided by
    ``R``, and the intervals' integrals add up to the series' whole integral.

    Hand arithmetic. Steps end at 1, 2, 4, 6 with lengths 1, 1, 2, 2, so they cover (0, 1], (1, 2], (2, 4], (4, 6].
    Outfall A stage 10, 20, 30, 40 and discharge 1, -1, 2, -2; outfall B stage 5, 5, 8, 2 and discharge 0, 0, 3, 6.
    ``R = 3``: interval (0, 3] holds the whole of steps 1 and 2 and one second of step 3, and (3, 6] holds the other
    second of step 3 and the whole of step 4. A stage: (10 + 20 + 30) / 3 = 20 and (30 + 80) / 3 = 110 / 3. A
    discharge: (1 - 1 + 2) / 3 = 2/3 and (2 - 4) / 3 = -2/3. B stage: (5 + 5 + 8) / 3 = 6 and (8 + 4) / 3 = 4. B
    discharge: (0 + 0 + 3) / 3 = 1 and (3 + 12) / 3 = 5.
    Class: a step may be longer than an interval or straddle a report time. What kills it: sampling the series at the
    report times (stage 30 and 40 at A, asserted below), or giving the straddling step wholly to the interval it ends
    in (A discharge 0 and 0 where 2/3 and -2/3 are expected, and A stage 10 and 140/3 where 20 and 110/3 are).
    Second correct implementation: summing per-step overlaps with an explicit loop. Control: the volumes
    ``mean * R`` summed over intervals equal the series' integral, 0 for A and 18 for B.
    """
    ds = _stage_dataset(
        [1.0, 2.0, 4.0, 6.0],
        [1.0, 1.0, 2.0, 2.0],
        [[10, 5], [20, 5], [30, 8], [40, 2]],
        [[1, 0], [-1, 0], [2, 3], [-2, 6]],
    )
    means = outfall_stage_interval_means(ds, np.array([3.0, 6.0]))

    assert means.node_ids == ("A", "B")
    assert means.report_interval_s == 3.0
    np.testing.assert_allclose(means.stage_m, [[20.0, 6.0], [110.0 / 3.0, 4.0]], rtol=1e-12)
    np.testing.assert_allclose(means.discharge_m3s, [[2.0 / 3.0, 1.0], [-2.0 / 3.0, 5.0]], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(means.discharge_m3s.sum(axis=0) * 3.0, [0.0, 18.0], atol=1e-12)
    assert means.cell_methods == "time: mean (interval: 3 s)"
    # control: the sampled-at-report-time answer is different, so the equality above is not met by sampling
    assert not np.allclose(means.stage_m[:, 0], [30.0, 40.0])


def test_a_step_longer_than_the_report_interval_contributes_its_own_value_to_each_interval_it_covers():
    """Property: a step that spans several intervals contributes its constant value to each, not its total once.

    Hand arithmetic: one step ``(0, 6]`` with stage 3 and discharge 9; ``R = 2`` gives three intervals of mean stage 3
    and mean discharge 9 each (the integral 54 split three ways). What kills it: dividing the step's integral by the
    number of intervals it touches (18 each), or putting all of it in the first interval.
    """
    ds = _stage_dataset([6.0], [6.0], [[3.0]], [[9.0]])
    means = outfall_stage_interval_means(ds, np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(means.stage_m[:, 0], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(means.discharge_m3s[:, 0], [9.0, 9.0, 9.0])


@pytest.mark.parametrize(
    ("ends", "dts", "times", "message"),
    [
        ([1.0, 3.0], [1.0, 1.0], [2.0, 4.0], "back to back"),  # a hole between step 1 and step 2
        ([1.0, 2.0], [1.0, 1.0], [2.0, 4.0], "the record spans"),  # the record stops before the last report time
        ([3.0, 4.0], [1.0, 1.0], [2.0, 4.0], "the record spans"),  # the record starts after the first interval does
        ([1.0, 2.0], [1.0, 0.0], [1.0, 2.0], "not positive"),  # a zero-length step
        ([1.0, 2.0], [1.0, 1.0], [2.0], "at least two times"),  # a single report time has no interval length
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 4.0], "not evenly spaced"),
    ],
)
def test_a_record_that_does_not_cover_the_intervals_is_refused_not_averaged_over_time_it_does_not_hold(
    ends, dts, times, message
):
    """Property: each defect of the record against the report times is a ``ProcessingError`` naming the defect.

    Class: holes, short coverage, late start, zero step length, one report time, uneven report times. Why: an average
    over an interval the record only partly covers reads as a measured value. Must-fire control: the matching
    well-formed records are accepted by the tests above, so each refusal is the parametrised defect's.
    """
    ds = _stage_dataset(ends, dts, [[1.0]] * len(ends), [[1.0]] * len(ends))
    with pytest.raises(ProcessingError, match=message):
        outfall_stage_interval_means(ds, np.array(times))


def _balance_csv(path: Path, columns, rows) -> Path:
    path.write_text("%" + ", ".join(columns) + "\n" + "".join(", ".join(str(v) for v in row) + "\n" for row in rows))
    return path


def test_the_two_outfall_balance_columns_are_read_by_name_and_every_other_term_is_unmoved(tmp_path):
    """Property: with the two columns in the header the dataset gains ``outfall_backflow_m3`` and
    ``outfall_backflow_deficit_m3``, each carrying its own column, and every other term keeps its own column.

    Hand-chosen values differ per column and per row (the backflow -3 / -30, the deficit 0.25 / 0.5, the exchange
    5 / 50), so a reader that reads the two new columns by position from the old header, or swaps the two, or shifts
    the edge terms, returns a different number. Class: any header with both columns between ``swmm_exchange_m3`` and
    ``edge_west_m3``. Second correct implementation: ``pandas.read_csv`` with ``usecols`` by name.
    """
    columns = [*BASE_BALANCE_COLUMNS[:7], *OUTFALL_BALANCE_COLUMNS, *BASE_BALANCE_COLUMNS[7:]]
    #            t_end  dS    runoff inflow extbc clamp swmm_x  backfl  deficit  W   E   S   N   residual
    rows = [
        (120, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, -3.0, 0.25, 7.0, 8.0, 9.0, 10.0, 0.5),
        (240, 11, 12, 13, 14, 16, 50, -30, 0.5, 17, 18, 19, 20, 0.75),
    ]
    ds = read_domain_balance(_balance_csv(tmp_path / "domain_balance.csv", columns, rows))

    np.testing.assert_array_equal(ds["time_s"].values, [120.0, 240.0])
    np.testing.assert_array_equal(ds["outfall_backflow_m3"].values, [-3.0, -30.0])
    np.testing.assert_array_equal(ds["outfall_backflow_deficit_m3"].values, [0.25, 0.5])
    np.testing.assert_array_equal(ds["swmm_exchange_m3"].values, [5.0, 50.0])
    np.testing.assert_array_equal(ds["edge_west_m3"].values, [7.0, 17.0])
    np.testing.assert_array_equal(ds["residual_m3"].values, [0.5, 0.75])
    for name in OUTFALL_BALANCE_COLUMNS:
        assert ds[name].attrs["units"] == "m3" and ds[name].attrs["cell_methods"] == "time: sum"


def test_without_the_two_columns_the_dataset_is_the_one_it_always_was(tmp_path):
    """Property: the header with neither column reads as the eleven balance terms and no outfall variable.

    Class: every balance file of a run with no driven outfall, which is every file written before the columns existed.
    What kills it: a reader that requires the new columns, or that adds them as zeros (a zero is a measured value).
    Must-fire control: the test above, where the same reader does add them.
    """
    rows = [(120, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 9.0, 10.0, 0.5)]
    ds = read_domain_balance(_balance_csv(tmp_path / "domain_balance.csv", BASE_BALANCE_COLUMNS, rows))
    assert sorted(ds.data_vars) == sorted(c for c in BASE_BALANCE_COLUMNS if c != "t_end")
    assert not any("outfall" in name for name in ds.data_vars)


@pytest.mark.parametrize(
    "extra",
    [["outfall_backflow_m3"], ["outfall_backflow_deficit_m3"], ["outfall_backflow_deficit_m3", "outfall_backflow_m3"]],
    ids=["only the backflow", "only the deficit", "the two swapped"],
)
def test_one_outfall_column_or_the_two_in_the_other_order_is_refused(tmp_path, extra):
    """Property: the columns are both present, in the solver's order, or both absent; any other header is refused.

    Why: the residual closes only with both terms accounted, and a half-present pair is a solver or file defect that a
    quiet read would carry into the ledger.
    """
    columns = [*BASE_BALANCE_COLUMNS[:7], *extra, *BASE_BALANCE_COLUMNS[7:]]
    rows = [tuple(float(i) for i in range(len(columns)))]
    with pytest.raises(ProcessingError, match="expected"):
        read_domain_balance(_balance_csv(tmp_path / "domain_balance.csv", columns, rows))


def _scenario(tmp_path: Path, *, cfg: dict, driven: list[str] | None, with_capture: bool = True, capture_names=None):
    """A post-processing object holding what ``_attach_coupling_outputs`` reads, and its input file.

    Six steps of 60 s end at 60 .. 360 s; the report step is 120 s (three intervals ending at 120, 240, 360 s).
    Outfall ``E1`` (in the run's SWMM output as ``e1``) has stage 1..6 and discharge 0.5, -0.5, 1, 1, 2, 0; outfall
    ``E2`` has stage 10, 20, ..., 60 and discharge 7 throughout, so the two differ at every step.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    out = tmp_path / "out_tritonswmm"
    swmm_dir = out / "swmm"
    swmm_dir.mkdir(parents=True)
    rpt = swmm_dir / "hydraulics.rpt"
    rpt.write_text("  Starting Date ............ 01/01/2020 00:00:00\n")
    (swmm_dir / "coupling_nodes.txt").write_text("c1\n")
    if driven is not None:
        (swmm_dir / "driven_outfalls.txt").write_text("".join(f"{n}\n" for n in driven))
    if with_capture:
        stage = [[float(k), 10.0 * k] for k in range(1, 7)]
        q_out = [[q, 7.0] for q in (0.5, -0.5, 1.0, 1.0, 2.0, 0.0)]
        records = [
            _stage_record(60.0 * (k + 1), 60.0, stage[k], q_out[k], [max(0.0, -q_out[k][0]), 0.0]) for k in range(6)
        ]
        _write_stage(swmm_dir / "outfall_stage.bin", capture_names or ["E1", "E2"], records)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(out_tritonswmm=out, swmm_hydraulics_rpt=rpt)
    base = {"toggle_coupling_exchange_flow": False}
    proc._analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(**base, **cfg))
    return proc, swmm_dir / "hydraulics.inp"


def _node_dataset(node_ids=("c1", "e1", "e2", "j1")) -> xr.Dataset:
    times = pd.date_range("2020-01-01 00:02:00", periods=3, freq="2min")
    return xr.Dataset(
        {"depth": (("date_time", "node_id"), np.ones((3, len(node_ids))))},
        coords={"date_time": times, "node_id": list(node_ids)},
    )


def _attach(proc, inp, ds=None):
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    return TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, ds if ds is not None else _node_dataset(), inp)


def test_driven_outfalls_get_their_stage_and_discharge_on_their_own_nodes_and_nan_elsewhere(tmp_path):
    """Property: the flag is 1 exactly at the listed outfalls, and the two series sit on the listed outfalls' columns
    as the mean over each report interval, NaN at every other node.

    Hand arithmetic, ``R = 120 s`` and steps of 60 s. E1 stage 1..6: (1+2)/2 = 1.5, (3+4)/2 = 3.5, (5+6)/2 = 5.5; E1
    discharge 0.5, -0.5, 1, 1, 2, 0: 0, 1, 1. E2 stage 10..60: 15, 35, 55; E2 discharge 7, 7, 7. The SWMM output spells
    the nodes ``e1`` and ``e2`` and the capture's sidecar ``E1`` and ``E2``, so the match is SWMM's ASCII case fold. The
    node order of the dataset (c1, e1, e2, j1) differs from the capture's (E1, E2) by an offset, so a reader that
    assigns by position puts E1 on c1. Class: any driven list in any order against any dataset node order.
    Must-fire controls: NaN at c1 and j1 (a zero fill reads as a measured zero) and a flag sum of 2, not of the
    dataset's node count.
    """
    cfg = {"swmm_open_water_outfalls": ["E1", "E2"], "open_water_outfall_representation": "driven_stage"}
    proc, inp = _scenario(tmp_path, cfg=cfg, driven=["E1", "E2"])
    out = _attach(proc, inp)

    assert out["is_open_water_outfall"].dims == ("node_id",) and out["is_open_water_outfall"].dtype == np.int8
    assert out["is_open_water_outfall"].values.tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(out["outfall_stage_m"].sel(node_id="e1").values, [1.5, 3.5, 5.5])
    np.testing.assert_allclose(out["outfall_stage_m"].sel(node_id="e2").values, [15.0, 35.0, 55.0])
    np.testing.assert_allclose(out["outfall_discharge_m3s"].sel(node_id="e1").values, [0.0, 1.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(out["outfall_discharge_m3s"].sel(node_id="e2").values, [7.0, 7.0, 7.0])
    for name in ("outfall_stage_m", "outfall_discharge_m3s"):
        assert out[name].dims == ("date_time", "node_id")
        assert np.isnan(out[name].sel(node_id=["c1", "j1"]).values).all(), name
        assert out[name].attrs["cell_methods"] == "time: mean (interval: 120 s)"


def test_with_no_listed_outfall_nothing_is_added(tmp_path):
    """Property: an analysis that lists no open-water outfall gets the node dataset it always got: the coupling flag and
    nothing about outfalls, whether the field is absent, empty, or the capture is absent from the run.

    Class: every analysis before the fields existed (absent) and every analysis that leaves the default (empty).
    Must-fire control: the test above lists the same outfalls and does add them, so the equality below is not met by a
    reader that never adds anything.
    """
    for cfg in ({}, {"swmm_open_water_outfalls": []}):
        proc, inp = _scenario(tmp_path / f"case{len(cfg)}", cfg=cfg, driven=None, with_capture=False)
        out = _attach(proc, inp)
        assert sorted(out.data_vars) == ["depth", "is_coupling_node"], cfg


def test_a_crown_rim_list_gets_the_flag_and_no_series(tmp_path):
    """Property: under ``crown_rim`` the listed outfalls are coupled junctions with no capture; the node dataset gains
    the flag at them and neither series, and the run needs no ``outfall_stage.bin``.

    Why: the comparison representation is what the report contrasts the driven stage with, so it must be findable on
    the node axis, and it writes nothing for the reader to decode. What kills it: a reader that demands the capture
    for every listed outfall (a refusal here) or adds an all-NaN series.
    """
    cfg = {"swmm_open_water_outfalls": ["E1"], "open_water_outfall_representation": "crown_rim"}
    proc, inp = _scenario(tmp_path, cfg=cfg, driven=None, with_capture=False)
    out = _attach(proc, inp)
    assert out["is_open_water_outfall"].values.tolist() == [0, 1, 0, 0]
    assert "outfall_stage_m" not in out.data_vars and "outfall_discharge_m3s" not in out.data_vars


@pytest.mark.parametrize(
    ("scenario_kwargs", "message"),
    [
        ({"driven": None}, "holds no driven_outfalls.txt"),
        ({"driven": ["E1", "E2"], "with_capture": False}, "wrote no outfall_stage.bin"),
        ({"driven": ["E2", "E1"]}, "the capture names the outfalls"),  # listed order differs from the capture's
        ({"driven": ["E1", "E2"], "capture_names": ["E1", "E3"]}, "the capture names the outfalls"),
    ],
    ids=["no list file", "no capture", "list order differs", "another outfall"],
)
def test_a_driven_run_that_lacks_or_contradicts_its_inputs_is_refused(tmp_path, scenario_kwargs, message):
    """Property: each missing or contradictory input of a driven run is a ``ProcessingError`` naming it.

    Class: no ``driven_outfalls.txt``, no ``outfall_stage.bin`` (a solver that predates the key writes none and
    ignores the line), a capture whose outfalls are not the scenario's list in its order, and one naming an outfall
    the list does not. Why: a node store written with the series silently absent reads as an outfall that never
    moved. Must-fire control: the first test of this section reads the same fixtures when they agree.
    """
    cfg = {"swmm_open_water_outfalls": ["E1", "E2"], "open_water_outfall_representation": "driven_stage"}
    proc, inp = _scenario(tmp_path, cfg=cfg, **scenario_kwargs)
    with pytest.raises(ProcessingError, match=message):
        _attach(proc, inp)


def test_a_listed_outfall_that_is_not_a_node_of_the_output_is_refused(tmp_path):
    """Property: a listed outfall absent from the SWMM node output is refused and named, like a coupling node is."""
    cfg = {"swmm_open_water_outfalls": ["E1", "E9"], "open_water_outfall_representation": "driven_stage"}
    proc, inp = _scenario(tmp_path, cfg=cfg, driven=["E1", "E9"])
    with pytest.raises(ProcessingError, match=r"absent from the SWMM node output: \['E9'\]"):
        _attach(proc, inp)


def test_the_summary_drops_the_two_interval_series_and_keeps_the_flag():
    """Property: ``summarize_swmm_simulation_results`` makes no ``_max``/``_last`` pair of the two outfall series and
    passes the static flag through, while every other series still gets its pair.

    Why: the cross-simulation identity check compares every ``_max`` that has a ``_last`` partner, so a pair for a
    per-interval series would enter its compared set for every driven run. Must-fire control: ``depth`` beside them
    does get its pair. What kills it: leaving the two names out of the drop list.
    """
    from hhemt.process_simulation import summarize_swmm_simulation_results

    times = pd.date_range("2020-01-01 00:02:00", periods=3, freq="2min")
    ds = xr.Dataset(
        {
            "depth": (("date_time", "node_id"), np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])),
            "outfall_stage_m": (("date_time", "node_id"), np.array([[np.nan, 5.0]] * 3)),
            "outfall_discharge_m3s": (("date_time", "node_id"), np.array([[np.nan, 1.0]] * 3)),
            "is_open_water_outfall": ("node_id", np.array([0, 1], dtype="int8")),
        },
        coords={"date_time": times, "node_id": ["a", "b"]},
    )
    out = summarize_swmm_simulation_results(ds, event_iloc=0)
    assert "depth_max" in out.data_vars and "depth_last" in out.data_vars
    assert not [v for v in out.data_vars if v.startswith("outfall_")]
    assert out["is_open_water_outfall"].values.tolist() == [[0, 1]]


def test_the_three_new_names_state_a_unit_a_computed_quantity_and_the_sign_of_the_flow():
    """Property: each new node-store variable has a unit in the CF map, a provenance descriptor with every required
    key, and is stamped by the coupled node mode; the discharge's comment states the sign.

    Class: the three names ``is_open_water_outfall``, ``outfall_stage_m``, ``outfall_discharge_m3s``. Why: the data
    dictionary of the report publishes the map and the descriptor table whole, so a name in one and not the other is a
    row asserting a unit and saying nothing about how the number was made. Must-fire control: an unlisted name gets no
    unit, so the stamping below is the map's.
    """
    expected = {"is_open_water_outfall": "1", "outfall_stage_m": "m", "outfall_discharge_m3s": "m3 s-1"}
    ds = xr.Dataset(
        {
            "is_open_water_outfall": ("node_id", np.array([0, 1], dtype="int8")),
            "outfall_stage_m": (("date_time", "node_id"), np.zeros((1, 2))),
            "outfall_discharge_m3s": (("date_time", "node_id"), np.zeros((1, 2))),
            "outfall_unlisted_series": (("date_time", "node_id"), np.zeros((1, 2))),
        }
    )
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    for name, unit in expected.items():
        assert _CF_VARIABLE_MAP[name]["units"] == unit, name
        assert ds[name].attrs["units"] == unit, name
        assert {"spatial_representation", "source_variables", "operation", "reduced_coordinate"} <= set(
            _QUANTITY_PROVENANCE[name]
        ), name
        assert _QUANTITY_PROVENANCE[name]["spatial_representation"] == "point (node)", name
    assert "negative when the stage pushes water into the sewer" in ds["outfall_discharge_m3s"].attrs["comment"]
    assert "units" not in ds["outfall_unlisted_series"].attrs
