"""Reviewer tests for the TRITON flood velocity |q| / h.

The solver writes unit discharge q (m2/s) and depth h. At summary time the toolkit derives the maximum over
output prints of sqrt(qx^2 + qy^2) / h with q and h of the SAME print, NaN where h is at or below a configured
threshold or NaN, plus the print at which it peaks and its two signed components qx / h and qy / h there.
The constructed-dataset file `test_flood_velocity_derivation.py` holds the
arithmetic on float64, in-memory, positive-component inputs. These tests reach the input classes it does not:
signed components, float32 storage and its strict boundary, lazy chunked (dask) input across chunk boundaries,
partly missing discharge, ties, the configuration surface and its classifier, the call site, the CF and
provenance entries, the documentation rows, and the committed golden re-derived with this file's own code.

Every test here neither compiles nor executes a solver, so all of them run on the laptop.
"""

from __future__ import annotations

import functools
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[1]

FV_NAMES = (
    "max_flood_velocity",
    "time_of_max_flood_velocity",
    "x_flood_velocity_at_time_of_max_flood_velocity",
    "y_flood_velocity_at_time_of_max_flood_velocity",
)
THRESHOLD = 0.001


def _ds(h, qx, qy, *, dtype="float64", chunk=None) -> xr.Dataset:
    """A (timestep, y, x) TRITON timeseries; prints are 10 minutes apart, cells are 10 m."""
    h, qx, qy = (np.asarray(a, dtype=dtype) for a in (h, qx, qy))
    nt, ny, nx = h.shape
    dims = ("timestep", "y", "x")
    ds = xr.Dataset(
        {
            "max_flood_depth": (dims, np.maximum.accumulate(np.nan_to_num(h), axis=0)),
            "flood_depth": (dims, h),
            "unit_discharge_x": (dims, qx),
            "unit_discharge_y": (dims, qy),
        },
        coords={
            "timestep": (np.arange(nt) + 1) * 10.0,
            "y": 10.0 * (ny - np.arange(ny)) - 5.0,
            "x": 10.0 * np.arange(nx) + 5.0,
        },
    )
    return ds.chunk({"timestep": chunk}) if chunk else ds


#: Leaves 4 timesteps per chunk, so 3 chunks, on the 12-print, 6 x 4 cell golden stores (see the golden test).
GOLDEN_CHUNKSIZE_MB = 0.0045


@pytest.fixture(autouse=True)
def _no_full_gc_in_the_streaming_reduction(monkeypatch):
    """`gc.collect()` runs once per streaming chunk in the production function and releases memory only, so it
    cannot change a value; in a pytest process that has already imported the docs checker each full collection
    costs seconds (measured 2026-11-02: every random-input case of this file took 114 to 120 s after
    `tests/test_check_docs_content.py`, the golden case 969 s). Stubbed for every test of this file."""
    import hhemt.process_simulation as ps

    monkeypatch.setattr(ps.gc, "collect", lambda *a, **k: 0)


def _summary(ds, chunksize_mb=0.0005, **kw):
    from hhemt.process_simulation import summarize_triton_simulation_results

    return summarize_triton_simulation_results(ds, 0, 10.0, chunksize_mb=chunksize_mb, **kw).isel(event_iloc=0)


def _reference(h, qx, qy, threshold=THRESHOLD):
    """Independent numpy reference: per print |q|/h where h > threshold, first maximum over prints.

    Returns (max, time_index, x_component, y_component), NaN where the cell is never defined.
    """
    h, qx, qy = (np.asarray(a, dtype="float64") for a in (h, qx, qy))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(h > threshold, np.hypot(qx, qy) / h, np.nan)
    defined = ~np.isnan(v).all(axis=0)
    idx = np.argmax(np.where(np.isnan(v), -np.inf, v), axis=0)
    take = lambda a: np.take_along_axis(a, idx[None], axis=0)[0]  # noqa: E731
    mx = np.where(defined, take(v), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            mx,
            np.where(defined, idx, np.nan),
            np.where(defined, take(qx) / take(h), np.nan),
            np.where(defined, take(qy) / take(h), np.nan),
        )


def _take_at(k, a):
    """The value of `a` (nt, ny, nx) at the per-cell print index `k` (ny, nx)."""
    return np.take_along_axis(a, k[None], axis=0)[0]


def _random_case(seed=20261104, nt=9, ny=6, nx=7):
    """Depths and signed discharges with scattered dry, sub-threshold and NaN prints."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.0, 2.0, size=(nt, ny, nx))
    qx = rng.uniform(-3.0, 3.0, size=(nt, ny, nx))
    qy = rng.uniform(-3.0, 3.0, size=(nt, ny, nx))
    h[rng.random(h.shape) < 0.15] = 0.0
    h[rng.random(h.shape) < 0.05] = THRESHOLD  # exactly at the threshold: undefined
    h[rng.random(h.shape) < 0.05] = np.nan
    qx[rng.random(h.shape) < 0.05] = np.nan
    h[:, 0, 0] = 0.0  # an always-dry cell
    h[:, 1, 1] = np.nan  # a never-defined depth
    return h, qx, qy


def test_summary_components_are_signed_and_equal_q_over_the_same_print_depth():
    """Property: the two components are signed qx / h and qy / h at the print where |q| / h peaks.

    Class: cells whose discharge components are NEGATIVE at the peak print (flow toward decreasing x and y), and
    one whose peak print is not the deepest print.
    Why required: the components are qx / h and qy / h, and the constructed tests use
    positive components only, so a magnitude (`abs`) or a swapped axis would pass them.
    Kills it: taking `abs` of either component (-6.0 would read +6.0), swapping x and y, dividing by the
    last print's depth, or by `max_flood_depth`.
    Passes under: an implementation that computes the velocity per print eagerly in numpy and reduces with
    `nanargmax`.
    """
    h = np.full((3, 2, 2), 1.0)
    qx = np.zeros((3, 2, 2))
    qy = np.zeros((3, 2, 2))
    h[:, 0, 0] = [2.0, 0.5, 4.0]
    qx[:, 0, 0] = [1.0, -3.0, 2.0]  # print 2: |q| = 5 over 0.5 -> 10, qx/h = -6, qy/h = -8
    qy[:, 0, 0] = [0.0, -4.0, 1.0]
    out = _summary(_ds(h, qx, qy))
    assert float(out["max_flood_velocity"][0, 0]) == pytest.approx(10.0)
    assert float(out["time_of_max_flood_velocity"][0, 0]) == 20.0
    assert float(out["x_flood_velocity_at_time_of_max_flood_velocity"][0, 0]) == pytest.approx(-6.0)
    assert float(out["y_flood_velocity_at_time_of_max_flood_velocity"][0, 0]) == pytest.approx(-8.0)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("chunk", [None, 1, 4])
def test_summary_matches_an_independent_reference_on_random_input_in_every_storage_form(dtype, chunk):
    """Property: all four variables equal the numpy reference cell by cell, whatever the dtype and chunking.

    Class: random signed discharge with scattered dry, exactly-at-threshold, NaN-depth and NaN-discharge
    prints, stored as float64 or float32 (the toolkit's default store is float32), in memory or as lazy dask
    chunks of 1 or 4 prints (what `open_zarr` hands the function) so the peak falls in different chunks.
    Why required: the streaming reduction merges per-chunk maxima, and the strict `h > threshold` is applied to
    stored values; neither is reached by a small in-memory float64 case.
    Kills it: a threshold compared as `>=` (the planted at-threshold prints read as defined), a chunk-merge that
    keeps the later of two equal maxima, an off-by-one in the carried companions across a chunk boundary, and a
    float32 comparison that lets a depth stored as float32(0.001) through.
    Passes under: an implementation that loads the whole series and reduces once.
    """
    h, qx, qy = _random_case()
    if dtype == "float32":
        # compare in the stored precision: round-trip the inputs so the reference sees what the store holds
        h, qx, qy = (np.asarray(a, dtype="float32").astype("float64") for a in (h, qx, qy))
    out = _summary(_ds(h, qx, qy, dtype=dtype, chunk=chunk))
    mx, idx, vx, vy = _reference(
        h, qx, qy, threshold=np.float64(np.float32(THRESHOLD)) if dtype == "float32" else THRESHOLD
    )
    rtol = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(out["max_flood_velocity"].values, mx, rtol=rtol, equal_nan=True)
    t = np.where(np.isnan(idx), np.nan, (idx + 1) * 10.0)
    np.testing.assert_allclose(out["time_of_max_flood_velocity"].values, t, equal_nan=True)
    np.testing.assert_allclose(
        out["x_flood_velocity_at_time_of_max_flood_velocity"].values, vx, rtol=rtol, equal_nan=True
    )
    np.testing.assert_allclose(
        out["y_flood_velocity_at_time_of_max_flood_velocity"].values, vy, rtol=rtol, equal_nan=True
    )
    assert np.isnan(mx).any() and np.isfinite(mx).any()  # control: the planted dry and wet cells are both present


def test_a_nan_discharge_at_the_best_print_falls_back_to_the_next_best_print_and_a_tie_keeps_the_first():
    """Property: a print with NaN discharge is skipped, not propagated; equal maxima keep the earliest print.

    Class: cell (0, 0) whose largest |q| / h is at a NaN-discharge print; cell (0, 1) with two prints of equal
    velocity but different components.
    Why required: the reduction updates on a strict greater-than from minus infinity, so NaN must never win and a
    tie must not move the time or the companions to the later print.
    Kills it: a NaN-propagating `max`, a `>=` update (the later tie), taking the components from the last of two
    equal maxima.
    Passes under: an implementation that masks NaN before reducing and breaks ties by the smallest index.
    """
    h = np.full((3, 2, 2), 1.0)  # two rows: the function derives the cell size from the coordinate spacing
    qx = np.zeros((3, 2, 2))
    qy = np.zeros((3, 2, 2))
    qx[:, 0, 0] = [2.0, np.nan, 1.0]
    qy[:, 0, 0] = [0.0, 0.0, 0.0]
    qx[:, 0, 1] = [3.0, 0.0, 0.0]  # print 1: (3, 0) over h 1 -> 3
    qy[:, 0, 1] = [0.0, 3.0, 0.0]  # print 2: (0, 3) over h 1 -> 3, a tie
    out = _summary(_ds(h, qx, qy))
    assert float(out["max_flood_velocity"][0, 0]) == pytest.approx(2.0)
    assert float(out["time_of_max_flood_velocity"][0, 0]) == 10.0
    assert float(out["time_of_max_flood_velocity"][0, 1]) == 10.0
    assert float(out["x_flood_velocity_at_time_of_max_flood_velocity"][0, 1]) == pytest.approx(3.0)
    assert float(out["y_flood_velocity_at_time_of_max_flood_velocity"][0, 1]) == pytest.approx(0.0)


def test_config_field_is_a_positive_float_with_default_0_001():
    """Property: `analysis_config.flood_velocity_min_depth_m` defaults to 0.001 and refuses zero and negatives.

    Class: the field's default, its constraint, and values 0, -0.001 and 0.0005.
    Why required: the threshold is configurable with default 0.001 and `gt=0`; a zero threshold turns the
    strict comparison into `h > 0` and lets a cell the solver zeroed below `hextra` report 0 / 0.
    Kills it: a default other than 0.001, no constraint, or `ge=0`.
    Passes under: `PositiveFloat`, which carries the same `Gt(0)` metadata.
    """
    from pydantic import TypeAdapter, ValidationError

    from hhemt.config.analysis import analysis_config

    fi = analysis_config.model_fields["flood_velocity_min_depth_m"]
    assert fi.default == 0.001 and fi.annotation is float
    ta = TypeAdapter(__import__("typing").Annotated[float, *fi.metadata])
    assert ta.validate_python(0.0005) == 0.0005
    for bad in (0, 0.0, -0.001):
        with pytest.raises(ValidationError):
            ta.validate_python(bad)
    assert "hextra" in (fi.description or "")


def test_the_new_config_field_has_a_reprex_bucket():
    """Property: every `analysis_config` field has a reproducibility bucket; the new threshold's is `experiment`.

    Class: `flood_velocity_min_depth_m`, and the totality of `all_field_bucket` over the config's fields.
    Why required: the Metadata renderer calls `all_field_bucket` on every field and raises `KeyError` on an
    unbucketed one, so an unbucketed new field crashes the report's metadata page for every analysis. The
    threshold decides which stored values exist, the class the sibling `process_*` knobs are bucketed in.
    Kills it: leaving the field out of `reprex_taxonomy`, or bucketing it as `user` or `hpc`.
    Passes under: a taxonomy that derives the bucket from a field annotation instead of a table.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_taxonomy import all_field_bucket

    assert all_field_bucket("flood_velocity_min_depth_m") == "experiment"
    missing = []
    for name in analysis_config.model_fields:
        try:
            all_field_bucket(name)
        except KeyError:
            missing.append(name)
    assert missing == [], missing


def test_every_production_call_of_the_summary_passes_the_configured_threshold():
    """Property: each call to `summarize_triton_simulation_results` in `src` passes `flood_velocity_min_depth_m`
    read from the analysis configuration.

    Class: all call sites in `src/hhemt` (one at the time of writing).
    Why required: the function's keyword has a default, so a caller that passes nothing runs with 0.001 and the
    configured threshold the `comment` attribute and the documentation promise is silently ignored.
    Kills it: dropping the keyword at the caller, or passing a literal.
    Passes under: making the keyword required, which cannot be called without it.
    """
    import ast

    calls = []
    for p in (REPO / "src" / "hhemt").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name == "summarize_triton_simulation_results":
                    calls.append((p.relative_to(REPO).as_posix(), node))
    assert calls, "no call site found"
    for rel, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert "flood_velocity_min_depth_m" in kw, rel
        assert isinstance(kw["flood_velocity_min_depth_m"], ast.Attribute), rel
        assert kw["flood_velocity_min_depth_m"].attr == "flood_velocity_min_depth_m", rel


def test_cf_and_provenance_entries_for_the_flood_velocity():
    """Property: the four entries carry the CF and provenance columns, and `flood_water_speed` is on the magnitude only.

    Class: the four names in `_CF_VARIABLE_MAP` and `_QUANTITY_PROVENANCE`.
    Why required: the crate advertises from the CF map and the report renders the provenance table; the
    provenance must state that the maximum is over output prints only.
    Kills it: a velocity `standard_name` on a component, `m2 s-1` on any of them, a provenance that omits
    the same-print rule or the output-prints-only caveat, or `sea_water_speed`.
    Passes under: entries generated in a loop from one template.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _QUANTITY_PROVENANCE

    cols = {
        "max_flood_velocity": ("flood_water_speed", "m s-1", "time: maximum"),
        "time_of_max_flood_velocity": (None, "min", None),
        "x_flood_velocity_at_time_of_max_flood_velocity": (None, "m s-1", None),
        "y_flood_velocity_at_time_of_max_flood_velocity": (None, "m s-1", None),
    }
    for name, (std, units, cm) in cols.items():
        e = _CF_VARIABLE_MAP[name]
        assert (e["standard_name"], e["units"], e["cell_methods"]) == (std, units, cm), name
        assert _QUANTITY_PROVENANCE[name]["spatial_representation"] == "grid cell", name
        assert "flood_depth" in _QUANTITY_PROVENANCE[name]["source_variables"], name
    op = _QUANTITY_PROVENANCE["max_flood_velocity"]["operation"]
    assert "same timestep" in op and "output prints only" in op, op
    assert (
        _QUANTITY_PROVENANCE["max_flood_velocity"]["source_variables"]
        == "unit_discharge_x, unit_discharge_y, flood_depth"
    )


def test_docs_table_rows_for_the_flood_velocity():
    """Property: the TRITON table has a row per flood-velocity name with its unit and CF name, and says so in the note.

    Class: the four rows and the note under the table.
    Why required: the page is the published specification the goldens are pinned to; a velocity row missing or
    carrying `m2 s-1` contradicts the stored attributes.
    Kills it: a missing row, a wrong unit or CF name, or a note that does not state the output-prints-only limit.
    Passes under: a table rendered from the CF map by a generator.
    """
    text = (REPO / "docs" / "reference" / "output-data-model.md").read_text()
    start = text.index("2D surface results from TRITON")
    section = text[start : text.index("### Network results from SWMM", start)]
    rows = {
        m.group(1): [c.strip().strip("`") for c in m.group(0).strip("|\n").split("|")]
        for m in re.finditer(r"^\| `([a-z0-9_]+)` .*$", section, re.M)
    }
    want = {
        "max_flood_velocity": ("m s-1", "flood_water_speed"),
        "time_of_max_flood_velocity": ("min", "n/a"),
        "x_flood_velocity_at_time_of_max_flood_velocity": ("m s-1", "n/a"),
        "y_flood_velocity_at_time_of_max_flood_velocity": ("m s-1", "n/a"),
    }
    for name, (units, std) in want.items():
        assert name in rows, name
        assert (rows[name][1], rows[name][2]) == (units, std), (name, rows[name])
    assert re.search(r"output\s+timesteps|output\s+prints|between\s+prints", section), (
        "the output-prints limit is not stated"
    )


@pytest.mark.parametrize("golden", ["multisim"])
def test_golden_flood_velocity_is_rederived_by_this_file_and_by_the_production_function(golden, tmp_path, monkeypatch):
    """Property: for every TRITON store pair in the golden, this file's own numpy derivation agrees with the
    golden's stored flood velocity AND with `summarize_triton_simulation_results` applied to the golden timeseries.

    Class: every `*TRITON*_tseries.zarr` with a sibling `*_summary.zarr` under `sims/` or `members/` in the zip.
    Why required: the golden generator derives its flood velocity with its own code, and the production function
    with another; only a third derivation can show that the committed fixture is right rather than merely equal
    to a generator that shares an assumption (divisor, mask, tie rule) with the code under test.
    Kills it: a golden whose flood velocity divides by another print's depth or by `max_flood_depth`, a golden
    that masks differently from production, and a production function that disagrees with the fixture.
    Passes under: a regenerated golden produced by any generator that stores the same values.

    Cost bound (measured 2026-11-02): the production function runs `gc.collect()` once per streaming chunk and
    five more times per call, and in a pytest process that has already imported the docs checker each full
    collection costs seconds (969 s for this test after `tests/test_check_docs_content.py`, 25.7 s alone, with 38
    collections per call at the old `chunksize_mb=0.0005`, which is one timestep per chunk). `gc.collect` releases
    memory and cannot change a value, so it is stubbed here, and the chunk size leaves three chunks of four of the
    golden's twelve prints (`GOLDEN_CHUNKSIZE_MB`), so a peak that lands in a later chunk than the running maximum
    is still exercised. Both are checked by the control below (the chunk count) and by mutation runs.
    """
    from hhemt.utils import estimate_timesteps_per_chunk

    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    pairs = sorted(
        n.rsplit("/", 1)[0]
        for n in z.namelist()
        if n.endswith(("TRITON_tseries.zarr/zarr.json", "TRITON_only_tseries.zarr/zarr.json"))
    )
    pairs = [p for p in pairs if p.replace("_tseries.zarr", "_summary.zarr") + "/zarr.json" in z.namelist()]
    assert pairs, "no TRITON tseries/summary pair found in the golden"
    z.extractall(
        tmp_path,
        members=[
            n
            for n in z.namelist()
            if any(n.startswith(p + "/") or n.startswith(p.replace("_tseries", "_summary") + "/") for p in pairs)
        ],
    )
    checked = 0
    for p in pairs:
        ts = xr.open_zarr(tmp_path / p, consolidated=False).load()
        sm = (
            xr.open_zarr(tmp_path / p.replace("_tseries.zarr", "_summary.zarr"), consolidated=False)
            .load()
            .isel(event_iloc=0)
        )
        h, qx, qy = (ts[v].values for v in ("flood_depth", "unit_discharge_x", "unit_discharge_y"))
        mx, idx, vx, vy = _reference(h, qx, qy)
        t = np.where(np.isnan(idx), np.nan, ts["timestep"].values[np.nan_to_num(idx).astype(int)])
        np.testing.assert_allclose(sm["max_flood_velocity"].values, mx, rtol=2e-5, equal_nan=True, err_msg=p)
        np.testing.assert_allclose(sm["time_of_max_flood_velocity"].values, t, equal_nan=True, err_msg=p)
        np.testing.assert_allclose(
            sm["x_flood_velocity_at_time_of_max_flood_velocity"].values, vx, rtol=2e-5, equal_nan=True, err_msg=p
        )
        np.testing.assert_allclose(
            sm["y_flood_velocity_at_time_of_max_flood_velocity"].values, vy, rtol=2e-5, equal_nan=True, err_msg=p
        )
        # control: the chunk size splits the store's prints into 3 chunks, so a print can land in a chunk after
        # the one that holds the previous maximum
        step = estimate_timesteps_per_chunk(
            rds_dem=ts["flood_depth"].isel(timestep=0), n_variables=5, memory_budget_MiB=GOLDEN_CHUNKSIZE_MB
        )
        assert -(-ts.sizes["timestep"] // step) == 3, (p, step, ts.sizes["timestep"])
        prod = _summary(ts, chunksize_mb=GOLDEN_CHUNKSIZE_MB)
        for name in FV_NAMES:
            np.testing.assert_allclose(
                prod[name].values, sm[name].values, rtol=2e-5, equal_nan=True, err_msg=f"{p}:{name}"
            )
        assert np.isfinite(mx).any()  # control: the golden has wet cells, so the comparison is not NaN against NaN
        assert (
            np.nanmax(np.abs(np.hypot(qx, qy)).max(axis=0) - np.nan_to_num(mx))
        ) != 0  # q and v are different fields
        checked += 1
    assert checked >= 1
    meta = json.loads(z.read(pairs[0].replace("_tseries.zarr", "_summary.zarr") + "/max_flood_velocity/zarr.json"))
    assert meta["attributes"]["standard_name"] == "flood_water_speed"


def test_golden_inputs_separate_the_flood_velocity_from_its_plausible_wrong_derivations(tmp_path):
    """Property: the golden's raw fields are shaped so a wrong derivation of the flood velocity would differ.

    Class: every TRITON timeseries store of the multi-sim golden, read for six conditions each of which a
    plausible wrong implementation would pass if absent: a cell whose velocity peak is NOT at the last print; a
    cell whose same-print depth differs from the running maximum at that print; a cell whose peak-print depth
    differs from the last print's depth; a cell-print at or below the 0.001 m threshold; a negative x component
    at the peak; a cell whose peak print differs from the print of maximum discharge.
    Why required: the golden is an independent derivation, and an independent derivation certifies
    nothing if its inputs make every candidate derivation agree. Measured on the committed golden: depth never
    decreases, the running maximum equals the depth at every print, every cell peaks at the last print, no
    cell-print is at or below the threshold, and `qx` is never negative, so dividing by `max_flood_depth` or by
    the last print's depth, taking `abs` of the x component, and dropping the dry mask all leave the
    golden-versus-production comparison green (each was run as a mutation and left the golden test passing).
    Kills it: a generator whose depth is monotone and whose discharge is a positive multiple of it.
    Passes under: a generator in which a drying and re-wetting front, a reversing flow and a dry cell are
    present, however it produces them.
    """
    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / "multisim.zip")
    stores = sorted(
        n.rsplit("/", 1)[0]
        for n in z.namelist()
        if n.endswith(("TRITON_tseries.zarr/zarr.json", "TRITON_only_tseries.zarr/zarr.json"))
    )
    assert stores
    z.extractall(tmp_path, members=[n for n in z.namelist() if any(n.startswith(s + "/") for s in stores)])
    seen = dict.fromkeys(
        ("early_peak", "depth_vs_running_max", "depth_vs_last", "at_threshold", "negative_x", "time_split"), False
    )
    for s in stores:
        ts = xr.open_zarr(tmp_path / s, consolidated=False).load()
        h, mh, qx, qy = (
            ts[v].values.astype("float64")
            for v in ("flood_depth", "max_flood_depth", "unit_discharge_x", "unit_discharge_y")
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(h > THRESHOLD, np.hypot(qx, qy) / h, np.nan)
        wet = ~np.isnan(v).all(axis=0)
        k = np.argmax(np.where(np.isnan(v), -np.inf, v), axis=0)
        kq = np.argmax(np.hypot(qx, qy), axis=0)
        pick = functools.partial(_take_at, k)
        seen["early_peak"] |= bool((wet & (k != h.shape[0] - 1)).any())
        seen["depth_vs_running_max"] |= bool((wet & ~np.isclose(pick(h), pick(mh), rtol=1e-6)).any())
        seen["depth_vs_last"] |= bool((wet & ~np.isclose(pick(h), h[-1], rtol=1e-6)).any())
        seen["at_threshold"] |= bool((h <= THRESHOLD).any())
        seen["negative_x"] |= bool((wet & (pick(qx) < 0)).any())
        seen["time_split"] |= bool((wet & (k != kq)).any())
    assert all(seen.values()), [name for name, ok in seen.items() if not ok]
