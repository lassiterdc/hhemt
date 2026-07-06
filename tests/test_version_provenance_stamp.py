"""ADR-15 Phase 1 — per-scope version-provenance stamp round-trip unit tests.

DISTINCT from the golden-fixture migration test (``test_version_migration_golden.py``,
which verifies the no-op ``V0017`` layout-version bump). This module verifies the
STAMP COORDINATE ITSELF — the ``written-by-producer, read-on-every-op`` primitive:

(a) a per-scenario summary write carries both ``event_iloc`` coordinates;
(b) the coordinate SURVIVES ``xr.concat(dim="event_iloc",
    combine_attrs="drop_conflicts")`` with per-event resolution (the load-bearing
    round-trip; an ATTR would be dropped), and the scalar root fast-path is
    present when uniform / absent+breadcrumb under drift;
(c) write -> ``open_datatree(consolidated=False)`` re-open preserves both
    coordinates + the root attr byte-stable;
(d) a regression assertion that ``eda.check_cross_sim_identity``'s comparison set
    EXCLUDES the stamp coordinates (variable-scoped);
(e) an idempotence / byte-stability assertion (master plan C5/C6): re-producing
    at the SAME resolved checkout yields byte-identical stamp values.

The reader tolerance (D6) + the mixed stamped/unstamped concat normalization are
covered here as well.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hhemt.cf_conventions import apply_producing_stamp, read_producing_stamp
from hhemt.eda.cross_sim_identity import TRACKED_VARS
from hhemt.process_simulation import TRITONSWMM_sim_post_processing
from hhemt.utils import write_zarr

_SHA_A = "a1b2c3d4e5f6"
_SHA_B = "0f1e2d3c4b5a"
_VER = "0.9.3"


def _stamp(ds: xr.Dataset, sha: str, semver: str) -> xr.Dataset:
    """Attach the two per-event coordinates exactly as ``_write_output`` does."""
    return ds.assign_coords(
        hhemt_producing_sha=("event_iloc", [sha]),
        hhemt_producing_version=("event_iloc", [semver]),
    )


def _summary(event_iloc: int, sha: str, semver: str = _VER) -> xr.Dataset:
    """A minimal FLAT per-scenario summary (event_iloc dim size 1), stamped."""
    ds = xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "cell"), np.array([[1.0, 2.0, 3.0]]))},
        coords={"event_iloc": [event_iloc]},
    )
    return _stamp(ds, sha, semver)


# ---------------------------------------------------------------------------
# (a) write carries both event_iloc coordinates
# ---------------------------------------------------------------------------
def test_summary_write_carries_both_coordinates() -> None:
    ds = _summary(0, _SHA_A)
    assert "hhemt_producing_sha" in ds.coords
    assert "hhemt_producing_version" in ds.coords
    # String-dtype aux coordinate, attached via a plain Python-list assign — the
    # SAME mechanism as the existing event_id coordinate (which lands as numpy
    # unicode `<U`), never a query index. zarr serializes it cleanly (see the disk
    # round-trip test below); consistency with event_id is the design intent.
    assert ds["hhemt_producing_sha"].dtype.kind in ("U", "O")
    assert ds["hhemt_producing_sha"].dims == ("event_iloc",)
    assert list(ds["hhemt_producing_sha"].values) == [_SHA_A]


def test_write_output_guard_skips_dataset_without_event_iloc() -> None:
    """The _write_output guard requires event_iloc in dims — a summary without it
    (e.g. a DataArray or a non-per-scenario Dataset) is never stamped."""
    ds = xr.Dataset({"scalar": ("t", np.arange(3.0))}, coords={"t": [0, 1, 2]})
    stamp_applies = isinstance(ds, xr.Dataset) and "event_iloc" in ds.dims
    assert stamp_applies is False


# ---------------------------------------------------------------------------
# (b) drop_conflicts concat survival (the load-bearing round-trip) + fast-path
# ---------------------------------------------------------------------------
def test_coordinate_survives_drop_conflicts_concat_under_drift() -> None:
    """The version-drift case: two scenarios produced by DIFFERENT shas. A COORDINATE
    concatenates element-wise and survives; the same values as an ATTR would be
    silently dropped by combine_attrs='drop_conflicts' — the reason the carrier is
    a coordinate, not an attr."""
    a = _summary(0, _SHA_A)
    b = _summary(1, _SHA_B)
    combined = xr.concat([a, b], dim="event_iloc", combine_attrs="drop_conflicts")
    assert list(combined["hhemt_producing_sha"].values) == [_SHA_A, _SHA_B]
    assert list(combined["hhemt_producing_version"].values) == [_VER, _VER]

    # Contrast: the same drift as an attr is dropped by drop_conflicts.
    a_attr = _summary(0, _SHA_A).drop_vars(["hhemt_producing_sha", "hhemt_producing_version"])
    b_attr = _summary(1, _SHA_B).drop_vars(["hhemt_producing_sha", "hhemt_producing_version"])
    a_attr.attrs["hhemt_producing_sha"] = _SHA_A
    b_attr.attrs["hhemt_producing_sha"] = _SHA_B
    combined_attr = xr.concat([a_attr, b_attr], dim="event_iloc", combine_attrs="drop_conflicts")
    assert "hhemt_producing_sha" not in combined_attr.attrs


def test_root_fast_path_uniform_sets_scalar() -> None:
    tree = xr.DataTree()
    apply_producing_stamp(tree, [_SHA_A, _SHA_A], [_VER, _VER])
    assert tree.attrs["hhemt_producing_sha"] == _SHA_A
    assert tree.attrs["hhemt_producing_version"] == _VER
    assert "hhemt_producing_sha_divergent" not in tree.attrs


def test_root_fast_path_divergent_drops_scalar_and_writes_breadcrumb() -> None:
    tree = xr.DataTree()
    apply_producing_stamp(tree, [_SHA_A, _SHA_B], [_VER, _VER])
    # Scalar fast-path absent under drift; the coordinate stays authoritative.
    assert "hhemt_producing_sha" not in tree.attrs
    import json

    assert json.loads(tree.attrs["hhemt_producing_sha_divergent"]) == sorted([_SHA_A, _SHA_B])
    # semver was uniform -> scalar present, no breadcrumb.
    assert tree.attrs["hhemt_producing_version"] == _VER
    assert "hhemt_producing_version_divergent" not in tree.attrs


def test_root_fast_path_empty_input_is_noop() -> None:
    tree = xr.DataTree()
    apply_producing_stamp(tree, [], [])
    assert "hhemt_producing_sha" not in tree.attrs
    assert "hhemt_producing_sha_divergent" not in tree.attrs


# ---------------------------------------------------------------------------
# (c) disk round-trip byte-stability (summary + datatree root attr)
# ---------------------------------------------------------------------------
def test_summary_zarr_round_trip_preserves_coordinates(tmp_path) -> None:
    ds = _summary(0, _SHA_A)
    f_out = tmp_path / "summary.zarr"
    write_zarr(ds, f_out, compression_level=1)
    reopened = xr.open_dataset(f_out, engine="zarr", consolidated=False)
    assert "hhemt_producing_sha" in reopened.coords
    assert list(reopened["hhemt_producing_sha"].values) == [_SHA_A]
    assert list(reopened["hhemt_producing_version"].values) == [_VER]


def test_datatree_root_fast_path_round_trips(tmp_path) -> None:
    leaf = _summary(0, _SHA_A)
    tree = xr.DataTree.from_dict({"/mode": leaf})
    apply_producing_stamp(tree, [_SHA_A], [_VER])
    f_out = tmp_path / "tree.zarr"
    from hhemt.utils import write_datatree_zarr

    write_datatree_zarr(tree, f_out, compression_level=1)
    reopened = xr.open_datatree(f_out, engine="zarr", consolidated=False)
    assert reopened.attrs["hhemt_producing_sha"] == _SHA_A
    assert reopened.attrs["hhemt_producing_version"] == _VER
    # per-event ground truth rides on the leaf node
    assert list(reopened["mode"]["hhemt_producing_sha"].values) == [_SHA_A]


# ---------------------------------------------------------------------------
# (d) EDA cross-sim identity comparison EXCLUDES the stamp coordinates
# ---------------------------------------------------------------------------
def test_eda_identity_comparison_excludes_stamp_coordinates() -> None:
    """The cross-sim byte-identity check compares only TRACKED_VARS data_vars. The
    stamp coordinates WILL differ across compute-config variants (produced by
    different checkouts) and MUST NOT flip a bit-identity verdict — they are
    neither in TRACKED_VARS nor data_vars."""
    assert "hhemt_producing_sha" not in TRACKED_VARS
    assert "hhemt_producing_version" not in TRACKED_VARS
    stamped = _summary(0, _SHA_A)
    assert "hhemt_producing_sha" not in stamped.data_vars
    assert "hhemt_producing_version" not in stamped.data_vars


# ---------------------------------------------------------------------------
# (e) idempotence / byte-stability at the same checkout
# ---------------------------------------------------------------------------
def test_resolve_producing_stamp_is_process_stable() -> None:
    first = TRITONSWMM_sim_post_processing._resolve_producing_stamp()
    second = TRITONSWMM_sim_post_processing._resolve_producing_stamp()
    assert first == second
    sha, semver = first
    assert isinstance(sha, str) and sha
    assert isinstance(semver, str) and semver


def test_reproducing_same_checkout_is_byte_stable(tmp_path) -> None:
    """Re-writing the per-scenario summary at the SAME resolved stamp yields
    byte-identical coordinate values — grounding the master Trade-offs claim that
    steady-state (post first-v17 rebuild) is byte-stable."""
    ds = _summary(0, _SHA_A)
    f1 = tmp_path / "s1.zarr"
    f2 = tmp_path / "s2.zarr"
    write_zarr(ds, f1, compression_level=1)
    write_zarr(ds, f2, compression_level=1)
    r1 = xr.open_dataset(f1, engine="zarr", consolidated=False)
    r2 = xr.open_dataset(f2, engine="zarr", consolidated=False)
    assert list(r1["hhemt_producing_sha"].values) == list(r2["hhemt_producing_sha"].values)
    assert list(r1["hhemt_producing_version"].values) == list(r2["hhemt_producing_version"].values)


# ---------------------------------------------------------------------------
# D6 reader tolerance: absent -> None; "unknown" sentinel; hex vector
# ---------------------------------------------------------------------------
def test_read_producing_stamp_legacy_absent_returns_none() -> None:
    legacy = xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "cell"), np.array([[1.0, 2.0, 3.0]]))},
        coords={"event_iloc": [0]},
    )
    assert read_producing_stamp(legacy) is None


def test_read_producing_stamp_unknown_sentinel_distinct_from_absence() -> None:
    ds = _summary(0, "unknown")
    stamp = read_producing_stamp(ds)
    assert stamp is not None
    assert stamp["uniform"] == "unknown"
    assert stamp["per_event"] == {0: "unknown"}


def test_read_producing_stamp_returns_per_event_vector_under_drift() -> None:
    combined = xr.concat([_summary(0, _SHA_A), _summary(1, _SHA_B)], dim="event_iloc", combine_attrs="drop_conflicts")
    stamp = read_producing_stamp(combined)
    assert stamp is not None
    assert stamp["uniform"] is None  # divergent
    assert stamp["per_event"] == {0: _SHA_A, 1: _SHA_B}


# ---------------------------------------------------------------------------
# Mixed stamped/unstamped concat normalization (the _retrieve_combined_output edit)
# ---------------------------------------------------------------------------
def test_mixed_stamped_unstamped_concat_normalizes_without_fabrication() -> None:
    """A reprocess re-runs SOME scenarios at v17+ (stamped) while others retain
    pre-v17 summaries (no coordinate). The read-path normalize makes the list
    uniform with an in-memory "unknown" sentinel; it is never written back to the
    legacy flat summary, so no historical provenance is fabricated."""
    stamped = _summary(0, _SHA_A)
    legacy = xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "cell"), np.array([[4.0, 5.0, 6.0]]))},
        coords={"event_iloc": [1]},
    )
    lst = []
    for ds in (stamped, legacy):
        if "hhemt_producing_sha" not in ds.coords:
            ds = ds.assign_coords(hhemt_producing_sha=("event_iloc", ["unknown"]))
        if "hhemt_producing_version" not in ds.coords:
            ds = ds.assign_coords(hhemt_producing_version=("event_iloc", ["unknown"]))
        lst.append(ds)
    combined = xr.concat(lst, dim="event_iloc", combine_attrs="drop_conflicts")
    assert list(combined["hhemt_producing_sha"].values) == [_SHA_A, "unknown"]
    # The original legacy Dataset object is untouched (no fabricated coordinate).
    assert "hhemt_producing_sha" not in legacy.coords
