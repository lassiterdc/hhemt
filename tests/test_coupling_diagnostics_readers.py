"""Readers for the solver's optional coupling-diagnostic capture files.

Chunk test: it guards ``hhemt.coupling_diagnostics`` and the three new performance columns, and is
promoted into a module test or deleted at the run's final landing.

Every file under test is written here from the byte layout the shape fixes, with ``struct`` and
hand-chosen values that differ per field and per node, so a reader that permutes two fields, two
nodes or two edges returns a different number than the oracle. The expected values are written
down in the tests, never read back from the code under test.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from hhemt.analysis import PERF_VARS, PERF_VARS_ORDERED
from hhemt.cf_conventions import _CF_PERFORMANCE_VARIABLES, _PERF_QUANTITY_PROVENANCE
from hhemt.coupling_diagnostics import (
    DIAG_HEADER_BYTES,
    MAGIC_DT_LIMITER,
    MAGIC_EDGE_FLUX,
    MAGIC_SWMM_LINK_FLOW_DIAG,
    MAGIC_SWMM_NODE_PATH3,
    MAGIC_SWMM_NODE_PATH3_STEP,
    MAGIC_SWMM_STEP_DIAG,
    nonconverged_steps_per_node,
    read_domain_balance,
    read_dt_limiter,
    read_edge_flux,
    read_swmm_link_flow_diag,
    read_swmm_node_path3,
    read_swmm_node_path3_step,
    read_swmm_step_diag,
    read_wet_cells,
)
from hhemt.coupling_exchange import FT3_TO_M3_FACTOR
from hhemt.exceptions import ProcessingError

# The magics the tests write are spelled here as literals, not imported, so a wrong constant in
# the module fails against the values the solver's file format fixes.
E2_MAGICS = {
    "step": 0x44535031,
    "node": 0x44504E33,
    "node_step": 0x44505333,
    "link": 0x444C4631,
    "edge": 0x44454631,
    "limiter": 0x44444C31,
}


def _header(kind: str, n_items: int, version: int = 1, value_width: int = 8) -> bytes:
    return struct.pack("<iiii", E2_MAGICS[kind], version, value_width, n_items)


def _step_record(simtime, dt, trials, bad_nodes, p3_pos, p3_neg, link_volume) -> bytes:
    fixed = struct.pack("<ddiiddd", simtime, dt, trials, len(bad_nodes), p3_pos, p3_neg, link_volume)
    return fixed + b"".join(struct.pack("<i", n) for n in bad_nodes)


def _write(path: Path, *parts: bytes) -> Path:
    path.write_bytes(b"".join(parts))
    return path


def _names(path: Path, names) -> Path:
    path.write_text("\n".join(names) + "\n")
    return path


# ---------------------------------------------------------------------------------------------
# swmm_step_diag.bin -- the variable-block file
# ---------------------------------------------------------------------------------------------


def _three_step_file(tmp_path: Path) -> Path:
    """Steps whose node blocks hold 0, 3 and 1 entries, so the walk crosses an empty block, a
    long block and a trailing block, each starting at a different offset."""
    return _write(
        tmp_path / "swmm_step_diag.bin",
        _header("step", 0),
        _step_record(10.0, 0.5, 4, [], 100.0, -20.0, 7.0),
        _step_record(10.5, 0.25, 9, [3, 5, 8], 200.0, -40.0, 8.0),
        _step_record(11.0, 0.125, 2, [5], 300.0, -60.0, 9.0),
    )


def test_step_diag_decodes_every_field_of_every_ragged_record(tmp_path):
    """Property: each field of each record lands in its own variable, and the node blocks are
    attributed to the step that wrote them.

    Class: any file of whole records with blocks of any length, including empty ones.
    Why: ``n_nonconv`` is both a field and the length of the block that follows, so a reader that
    advances by a constant stride, or by the block length of the previous record, mis-slices every
    step after the first non-empty block.
    What kills it: a reader that steps by ``48 + 4 * n_items`` or that swaps ``p3_pos`` and
    ``p3_neg``.
    Second correct implementation: a vectorised two-pass reader that first scans the block lengths
    and then slices; it returns the same arrays.
    """
    ds = read_swmm_step_diag(_three_step_file(tmp_path))

    np.testing.assert_array_equal(ds["simtime"].values, [10.0, 10.5, 11.0])
    np.testing.assert_array_equal(ds["swmm_step_dt"].values, [0.5, 0.25, 0.125])
    np.testing.assert_array_equal(ds["swmm_step_trials"].values, [4, 9, 2])
    np.testing.assert_array_equal(ds["swmm_step_nonconverged_nodes"].values, [0, 3, 1])
    np.testing.assert_allclose(
        ds["swmm_step_path3_volume_pos"].values, np.array([100.0, 200.0, 300.0]) * FT3_TO_M3_FACTOR
    )
    np.testing.assert_allclose(
        ds["swmm_step_path3_volume_neg"].values, np.array([-20.0, -40.0, -60.0]) * FT3_TO_M3_FACTOR
    )
    np.testing.assert_allclose(ds["swmm_step_link_volume"].values, np.array([7.0, 8.0, 9.0]) * FT3_TO_M3_FACTOR)
    np.testing.assert_array_equal(ds["nonconverged_node_index"].values, [3, 5, 8, 5])
    assert ds["swmm_step_nonconverged_nodes"].sum().item() == ds["nonconverged_node_index"].size


def test_step_diag_of_a_header_only_file_is_empty_not_an_error(tmp_path):
    """Property: a run that wrote no step records decodes to an empty Dataset.

    Class: the empty file, the differently-positioned arm of the ragged walk (the loop body never
    runs, so ``nonconverged`` must still exist with the right dtype).
    Why: a coupled run killed before its first SWMM step leaves exactly this file, and a reader
    that concatenates an empty list raises ``ValueError`` instead of reporting zero steps.
    What kills it: ``np.concatenate([])``.
    Second correct implementation: returning a pre-built empty Dataset for ``size == header``.
    """
    ds = read_swmm_step_diag(_write(tmp_path / "e.bin", _header("step", 0)))
    assert ds.sizes["swmm_step"] == 0
    assert ds["nonconverged_node_index"].size == 0
    assert ds["nonconverged_node_index"].dtype == np.dtype("int32")


@pytest.mark.parametrize("cut", [20, 48 + 2], ids=["inside-fixed-part", "inside-node-block"])
def test_step_diag_refuses_a_record_cut_off_mid_way(tmp_path, cut):
    """Property: a file whose last record is incomplete is refused, wherever the cut falls.

    Class: truncation inside the 48-byte fixed part and inside the node block (two positions).
    Why: the capture's resume rule truncates to the last complete record, so a partial record that
    reaches a reader means that rule failed; decoding it would give the last step a short block and
    shift the node attribution.
    What kills it: a reader that stops at the first short read and returns what it has.
    Second correct implementation: validating the file size against a pre-scan of block lengths.
    """
    whole = _write(tmp_path / "w.bin", _header("step", 0), _step_record(1.0, 0.1, 3, [1, 2, 3], 1.0, -1.0, 1.0))
    broken = tmp_path / "b.bin"
    broken.write_bytes(whole.read_bytes()[: DIAG_HEADER_BYTES + cut])
    with pytest.raises(ProcessingError, match="cut off"):
        read_swmm_step_diag(broken)


def test_nonconverged_steps_per_node_counts_by_index_and_refuses_an_out_of_range_index(tmp_path):
    """Property: the per-node count is the number of steps each index appears in, and an index the
    model cannot have is refused.

    Class: index multisets over a node count; the file above has node 5 twice and 3 and 8 once.
    Why: the count lands on the node store, so an index at or past ``n_nodes`` would otherwise be
    silently truncated by ``bincount(minlength=n_nodes)`` growth or dropped.
    What kills it: a count of steps with any non-convergence (3 and 5 would both read 1, 2 for step 1).
    Second correct implementation: a Python ``Counter`` over the index array.
    """
    ds = read_swmm_step_diag(_three_step_file(tmp_path))
    counts = nonconverged_steps_per_node(ds, n_nodes=9)
    np.testing.assert_array_equal(counts, [0, 0, 0, 1, 0, 2, 0, 0, 1])
    with pytest.raises(ProcessingError, match="outside"):
        nonconverged_steps_per_node(ds, n_nodes=8)


def test_every_reader_refuses_a_wrong_magic_version_value_width_or_n_items(tmp_path):
    """Property: each binary reader accepts only its own file kind's magic, version 1, value_width 8
    and the fixed n_items of the solver's format, and refuses a header that differs in any one of them.

    Class: one valid header per file kind, and four single-field corruptions of it (magic taken
    from another kind, version 2, value_width 4, n_items off by one) across all six readers.
    Why: the files are opened by kind but named by the solver; a swapped file or a different
    solver generation must fail at the header and not decode as the wrong record layout.
    What kills it: a reader that checks the magic only (the version and width cases then decode),
    or a magic constant that is wrong in the module (the valid case then fails).
    Second correct implementation: one dispatch table of (magic, n_items) per kind consulted by a
    single open function.
    """
    assert E2_MAGICS == {
        "step": MAGIC_SWMM_STEP_DIAG,
        "node": MAGIC_SWMM_NODE_PATH3,
        "node_step": MAGIC_SWMM_NODE_PATH3_STEP,
        "link": MAGIC_SWMM_LINK_FLOW_DIAG,
        "edge": MAGIC_EDGE_FLUX,
        "limiter": MAGIC_DT_LIMITER,
    }
    nodes = _names(tmp_path / "a.nodes", ["A", "B"])
    links = _names(tmp_path / "a.links", ["L"])
    rec_step = _step_record(1.0, 0.1, 1, [], 0.0, 0.0, 0.0)
    rec_node = struct.pack("<d2d2d2i", 1.0, 1.0, 2.0, -1.0, -2.0, 0, 1)
    rec_node_step = struct.pack("<dd2d", 1.0, 0.5, 1.0, 2.0)
    rec_link = struct.pack("<dd", 1.0, 2.0)
    rec_edge = struct.pack("<ddddd", 1.0, 1.0, 2.0, 3.0, 4.0)
    rec_lim = struct.pack("<ddiiiidd", 1.0, 0.1, 0, 1, 2, 0, 1.0, 2.0)
    cases = {
        "step": (0, rec_step, lambda p: read_swmm_step_diag(p)),
        "node": (2, rec_node, lambda p: read_swmm_node_path3(p, nodes)),
        "node_step": (2, rec_node_step, lambda p: read_swmm_node_path3_step(p, nodes)),
        "link": (1, rec_link, lambda p: read_swmm_link_flow_diag(p, links)),
        "edge": (4, rec_edge, lambda p: read_edge_flux(p)),
        "limiter": (0, rec_lim, lambda p: read_dt_limiter(p)),
    }
    other = {k: next(o for o in E2_MAGICS if o != k) for k in cases}
    for kind, (n_items, record, read) in cases.items():
        ok = _write(tmp_path / f"{kind}.bin", _header(kind, n_items), record)
        read(ok)  # the valid header decodes
        bad = {
            "magic": struct.pack("<iiii", E2_MAGICS[other[kind]], 1, 8, n_items),
            "version": _header(kind, n_items, version=2),
            "value_width": _header(kind, n_items, value_width=4),
            "n_items": _header(kind, n_items + 1),
        }
        for field, header in bad.items():
            with pytest.raises(ProcessingError, match="n_items|header has"):
                read(_write(tmp_path / f"{kind}_{field}.bin", header, record))


# ---------------------------------------------------------------------------------------------
# per-node and per-link files, sized by a sidecar
# ---------------------------------------------------------------------------------------------


def _node_path3_file(tmp_path: Path):
    """Two intervals over three nodes; every value differs so a node or field permutation shows."""
    nodes = _names(tmp_path / "swmm_node_diag.nodes", ["J1", "J2", "S3"])
    body = b""
    for t_end, pos, neg, nc in (
        (120.0, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0], [7, 8, 9]),
        (240.0, [10.0, 20.0, 30.0], [-40.0, -50.0, -60.0], [0, 1, 2]),
    ):
        body += struct.pack("<d", t_end)
        body += struct.pack("<3d", *pos) + struct.pack("<3d", *neg) + struct.pack("<3i", *nc)
    return _write(tmp_path / "swmm_node_path3.bin", _header("node", 3), body), nodes


def test_node_path3_maps_each_value_to_its_node_and_interval(tmp_path):
    """Property: ``pos``, ``neg`` and ``n_nonconv`` land on the right node and interval.

    Class: any record count over any node list; the oracle is the value table in the helper.
    Why: the record interleaves three length-n arrays after a scalar, so an off-by-one in the field
    offsets moves a whole array one node over without failing any shape check.
    What kills it: reading ``neg`` and ``pos`` as one ``(2, n)`` block in the wrong order, or
    treating the scalar as part of the first array.
    Second correct implementation: ``np.frombuffer`` over the whole body with explicit offsets.
    """
    path, nodes = _node_path3_file(tmp_path)
    ds = read_swmm_node_path3(path, nodes)
    assert ds["path3_volume_pos"].dims == ("timestep", "node_id")
    assert list(ds["node_id"].values) == ["J1", "J2", "S3"]
    np.testing.assert_array_equal(ds["time_s"].values, [120.0, 240.0])
    np.testing.assert_allclose(ds["path3_volume_pos"].values, np.array([[1, 2, 3], [10, 20, 30]]) * FT3_TO_M3_FACTOR)
    np.testing.assert_allclose(
        ds["path3_volume_neg"].values, np.array([[-4, -5, -6], [-40, -50, -60]]) * FT3_TO_M3_FACTOR
    )
    np.testing.assert_array_equal(ds["nonconverged_steps"].values, [[7, 8, 9], [0, 1, 2]])


def test_node_path3_refuses_a_sidecar_that_disagrees_with_the_header_or_repeats_a_name(tmp_path):
    """Property: a sidecar whose length differs from the header's ``n_items``, or that repeats a
    name, is refused before any record is read.

    Class: sidecar lengths (equal, one short, one long) and a repeated name.
    Why: the record size derives from the node count, so a wrong count silently re-slices every
    record rather than failing, and a repeated name makes two columns answer to one label.
    What kills it: sizing records from the sidecar alone.
    Second correct implementation: sizing records from the header and checking the sidecar against it.
    """
    path, _ = _node_path3_file(tmp_path)
    with pytest.raises(ProcessingError, match="n_items"):
        read_swmm_node_path3(path, _names(tmp_path / "short.nodes", ["J1", "J2"]))
    with pytest.raises(ProcessingError, match="n_items"):
        read_swmm_node_path3(path, _names(tmp_path / "long.nodes", ["J1", "J2", "S3", "S4"]))
    with pytest.raises(ProcessingError, match="repeats"):
        read_swmm_node_path3(path, _names(tmp_path / "dup.nodes", ["J1", "J2", "J1"]))


def test_node_path3_refuses_trailing_bytes(tmp_path):
    """Property: a body that is not a whole number of records is refused.

    Class: body lengths that are and are not multiples of the record size (one trailing byte).
    Why: a partial final record is the signature of a failed truncation on resume, and ``memmap``
    would otherwise round the record count down and drop it without a word.
    What kills it: deriving the record count with floor division and no remainder check.
    Second correct implementation: comparing ``stat().st_size`` to ``header + n * itemsize`` directly.
    """
    path, nodes = _node_path3_file(tmp_path)
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ProcessingError, match="whole number"):
        read_swmm_node_path3(path, nodes)


def test_node_path3_step_opens_lazily_and_keeps_engine_units(tmp_path):
    """Property: the per-step per-node file decodes to the right ``(step, node)`` values, backed by
    the file and not copied, and carries the engine unit and its conversion factor.

    Class: any step count over any node list; two steps by two nodes here.
    Why: the file is gigabytes at the experiment's size, so a reader that copies it defeats the
    reason the capture is lazy; the unit is stated because the values are NOT converted.
    What kills it: ``np.fromfile`` into a fresh array (``.base`` is then not a memory map).
    Second correct implementation: a dask array over the same memory map.
    """
    nodes = _names(tmp_path / "n.nodes", ["A", "B"])
    body = struct.pack("<dddd", 5.0, 0.5, 1.5, 2.5) + struct.pack("<dddd", 5.5, 0.5, -3.5, 4.5)
    ds = read_swmm_node_path3_step(_write(tmp_path / "s.bin", _header("node_step", 2), body), nodes)
    np.testing.assert_array_equal(ds["path3_volume_step"].values, [[1.5, 2.5], [-3.5, 4.5]])
    np.testing.assert_array_equal(ds["simtime"].values, [5.0, 5.5])
    np.testing.assert_array_equal(ds["dt"].values, [0.5, 0.5])
    assert ds["path3_volume_step"].attrs["units"] == "ft3"
    assert ds["path3_volume_step"].attrs["to_m3_factor"] == FT3_TO_M3_FACTOR
    assert isinstance(ds["path3_volume_step"].variable._data, np.memmap) or isinstance(
        getattr(ds["path3_volume_step"].variable._data, "base", None), np.memmap
    )


def test_link_flow_maps_each_flow_to_its_link_and_converts_to_si(tmp_path):
    """Property: per-step link flows land on the right link and are converted from cfs to m3/s.

    Class: any step count over any link list; the oracle is the value table below.
    Why: a flow carried in the engine's unit next to depths in meters is a silent factor of 35.
    What kills it: omitting the conversion, or transposing ``(step, link)``.
    Second correct implementation: converting lazily at access time.
    """
    links = _names(tmp_path / "l.links", ["C1", "C2", "C3"])
    body = struct.pack("<dddd", 1.0, 10.0, 20.0, 30.0) + struct.pack("<dddd", 2.0, 40.0, 50.0, 60.0)
    ds = read_swmm_link_flow_diag(_write(tmp_path / "l.bin", _header("link", 3), body), links)
    assert ds["link_flow"].dims == ("swmm_step", "link_id")
    np.testing.assert_allclose(ds["link_flow"].values, np.array([[10, 20, 30], [40, 50, 60]]) * FT3_TO_M3_FACTOR)
    np.testing.assert_array_equal(ds["simtime"].values, [1.0, 2.0])
    assert list(ds["link_id"].values) == ["C1", "C2", "C3"]


# ---------------------------------------------------------------------------------------------
# surface-domain files
# ---------------------------------------------------------------------------------------------


def test_edge_flux_orders_the_four_edges_west_east_south_north(tmp_path):
    """Property: the four per-record fluxes are labelled west, east, south, north in that order.

    Class: any step count; every flux value is distinct so a permutation of edges is visible.
    Why: the balance line sums the four edges with signs, so two swapped edges change the residual
    without any shape or dtype error.
    What kills it: a column order of ``(north, south, east, west)`` or any other permutation.
    Second correct implementation: a structured dtype read, then stacking by name.
    """
    body = struct.pack("<ddddd", 3.0, 1.0, 2.0, 3.0, 4.0) + struct.pack("<ddddd", 3.5, 5.0, 6.0, 7.0, 8.0)
    ds = read_edge_flux(_write(tmp_path / "edge_flux.bin", _header("edge", 4), body))
    assert list(ds["edge"].values) == ["west", "east", "south", "north"]
    np.testing.assert_array_equal(ds["edge_flux"].sel(edge="west").values, [1.0, 5.0])
    np.testing.assert_array_equal(ds["edge_flux"].sel(edge="north").values, [4.0, 8.0])
    np.testing.assert_array_equal(ds["simtime"].values, [3.0, 3.5])


def test_dt_limiter_decodes_the_mixed_width_record_and_refuses_an_unknown_reason(tmp_path):
    """Property: the 48-byte record's int32 and float64 fields land on their own variables, and a
    reason code outside the documented 0-2 is refused.

    Class: reason codes 0, 1, 2 (accepted) and 3 (refused); one record each.
    Why: ``rank``, ``grow``, ``gcol`` and ``reason`` are four adjacent int32 fields, so a one-field
    offset error relabels every one of them while the floats stay right.
    What kills it: reading the four int32 as two float64, or an offset one field out.
    Second correct implementation: ``struct.iter_unpack('<ddiiiidd')`` over the body.
    """

    def rec(simtime, reason):
        return struct.pack("<ddiiiidd", simtime, 0.25, 6, 11, 22, reason, 1.5, 9.75)

    ds = read_dt_limiter(_write(tmp_path / "d.bin", _header("limiter", 0), rec(1.0, 0), rec(2.0, 1), rec(3.0, 2)))
    np.testing.assert_array_equal(ds["dt_limiter_reason"].values, [0, 1, 2])
    np.testing.assert_array_equal(ds["dt_limiter_rank"].values, [6, 6, 6])
    np.testing.assert_array_equal(ds["dt_limiter_row"].values, [11, 11, 11])
    np.testing.assert_array_equal(ds["dt_limiter_col"].values, [22, 22, 22])
    np.testing.assert_array_equal(ds["dt_limiter_depth"].values, [1.5] * 3)
    np.testing.assert_array_equal(ds["dt_limiter_celerity"].values, [9.75] * 3)
    assert ds["dt_limiter_reason"].attrs["flag_meanings"] == "cell_limited clipped_to_print_instant init_dt_fallback"
    with pytest.raises(ProcessingError, match="reason code 3"):
        read_dt_limiter(_write(tmp_path / "bad.bin", _header("limiter", 0), rec(1.0, 3)))


def test_a_dt_limiter_record_in_the_ghost_ring_decodes_unchanged_with_its_minus_one_indices(tmp_path):
    """Property: a record whose ``grow`` and ``gcol`` are -1 (the solver records the strict ghost minimum as it is)
    decodes to -1 on ``dt_limiter_row`` and ``dt_limiter_col``, beside an ordinary record, with no range refusal.

    Class: NEW CAPABILITY, index values -1 and an ordinary pair. Plausible wrong implementation: validating indices
    as non-negative, which refuses real runs (2 of 393 records on a 4-rank circular dam-break). Why: the indices
    are stored verbatim, so a clamp or a refusal changes real data. Killed by: a range check or ``abs``.
    Second correct implementation: ``struct.iter_unpack`` then ``np.array``; same values. Control: the ordinary
    record keeps its own indices.
    """

    def rec(simtime, row, col):
        return struct.pack("<ddiiiidd", simtime, 0.25, 1, row, col, 0, 1.5, 9.75)

    ds = read_dt_limiter(_write(tmp_path / "g.bin", _header("limiter", 0), rec(1.0, 7, 9), rec(2.0, -1, -1)))
    np.testing.assert_array_equal(ds["dt_limiter_row"].values, [7, -1])
    np.testing.assert_array_equal(ds["dt_limiter_col"].values, [9, -1])


BALANCE_COLUMNS = [
    "t_end", "d_storage_m3", "runoff_m3", "inflow_m3", "extbc_m3", "clamp_m3", "swmm_exchange_m3",
    "edge_west_m3", "edge_east_m3", "edge_south_m3", "edge_north_m3", "residual_m3",
]  # fmt: skip


def test_domain_balance_requires_the_e2_percent_header_and_names_each_term(tmp_path):
    """Property: the file decodes only with the solver's header row (leading ``%``, exact names), each term
    lands on its own named variable in m3, and any other header or none is refused.

    Class: the valid header, a header with a column of another name, a header without the ``%``, and a
    headerless file.
    Why: the term list is a contract with the solver; a column of another name or in another position that decoded
    positionally would attribute one term's volume to another in the balance.
    What kills it: positional decoding, a fallback for headerless files, or accepting ``simtime``
    as the time column.
    Second correct implementation: ``csv.DictReader`` over the line with the ``%`` stripped.
    """
    path = tmp_path / "domain_balance.csv"
    row1 = [120.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    row2 = [240.0] + [x * 10 for x in row1[1:]]
    body = "\n".join(",".join(str(v) for v in row) for row in (row1, row2)) + "\n"
    path.write_text("%" + ", ".join(BALANCE_COLUMNS) + "\n" + body)
    ds = read_domain_balance(path)
    assert set(ds.data_vars) == set(BALANCE_COLUMNS[1:])
    np.testing.assert_array_equal(ds["time_s"].values, [120.0, 240.0])
    np.testing.assert_array_equal(ds["runoff_m3"].values, [2.0, 20.0])
    np.testing.assert_array_equal(ds["residual_m3"].values, [11.0, 110.0])
    assert ds["edge_north_m3"].attrs == {"units": "m3", "cell_methods": "time: sum"}
    other_name = [c if c != "runoff_m3" else "rain_m3" for c in BALANCE_COLUMNS]
    path.write_text("%" + ", ".join(other_name) + "\n" + body)
    with pytest.raises(ProcessingError, match="expected"):
        read_domain_balance(path)
    path.write_text(", ".join(BALANCE_COLUMNS) + "\n" + body)
    with pytest.raises(ProcessingError, match="starting with"):
        read_domain_balance(path)
    path.write_text(body)
    with pytest.raises(ProcessingError, match="starting with"):
        read_domain_balance(path)
    path.write_text("%" + ", ".join(["simtime", *BALANCE_COLUMNS[1:]]) + "\n" + body)
    with pytest.raises(ProcessingError, match="expected"):
        read_domain_balance(path)


def test_wet_cells_pivots_rows_to_a_rank_by_interval_grid_in_any_file_order(tmp_path):
    """Property: rows land on ``(Rank, timestep)`` by their own ``rank`` and ``print_id``, whatever
    order the file holds them in, and a hole in the grid is refused.

    Class: row orders (rank-major, interval-major shuffled), a missing row, and a headerless file.
    Why: a resumed run appends in an order a reader must not rely on, and a hole means the capture
    truncated wrongly, which a fill would hide.
    What kills it: reshaping the rows without sorting, or ``pivot`` with a fill value.
    Second correct implementation: ``DataFrame.pivot`` followed by an explicit completeness check.
    """
    path = tmp_path / "wet_cells.csv"
    path.write_text(
        "%print_id, simtime, rank, wet_mean, wet_at_print\n"
        "2,240.0,1,4.5,5\n"
        "1,120.0,0,1.5,2\n"
        "2,240.0,0,3.5,4\n"
        "1,120.0,1,2.5,3\n"
    )
    ds = read_wet_cells(path)
    assert ds["wet_cells_mean"].dims == ("Rank", "timestep")
    np.testing.assert_array_equal(ds["wet_cells_mean"].values, [[1.5, 3.5], [2.5, 4.5]])
    np.testing.assert_array_equal(ds["wet_cells_at_print"].values, [[2, 4], [3, 5]])
    np.testing.assert_array_equal(ds["time_s"].values, [120.0, 240.0])
    path.write_text(
        "%print_id, simtime, rank, wet_mean, wet_at_print\n1,120.0,0,1.5,2\n2,240.0,0,3.5,4\n1,120.0,1,2.5,3\n"
    )
    with pytest.raises(ProcessingError, match="exactly once"):
        read_wet_cells(path)
    path.write_text("1,120.0,0,1.5,2\n")
    with pytest.raises(ProcessingError, match="starting with"):
        read_wet_cells(path)


# ---------------------------------------------------------------------------------------------
# the three new performance columns
# ---------------------------------------------------------------------------------------------

NEW_PERF_COLUMNS = ("SWMM_SOLVE", "SWMM_LOG", "DIAG")


def test_the_new_timer_columns_are_minted_ordered_and_described_in_every_table():
    """Property: each of ``SWMM_SOLVE``, ``SWMM_LOG`` and ``DIAG`` is a member of every table that
    must track ``PERF_VARS``, and the display order stays a permutation of the mint set.

    Class: a column name over the four tables (mint set, display order, CF attributes, quantity
    provenance); derived from the lists themselves, never a count.
    Why: a name minted but missing from the display order falls into the unordered tail of
    ``scenario_status.csv``, and one missing a CF or provenance entry renders an em-dash in the
    report's provenance columns.
    What kills it: adding the name to ``PERF_VARS`` only.
    Second correct implementation: deriving the display order and the CF table from ``PERF_VARS``.
    """
    for name in NEW_PERF_COLUMNS:
        assert name in PERF_VARS, name
        assert name in PERF_VARS_ORDERED, name
        assert name in _CF_PERFORMANCE_VARIABLES, name
        assert name in _PERF_QUANTITY_PROVENANCE, name
        assert _CF_PERFORMANCE_VARIABLES[name]["units"] == "s", name
    assert sorted(PERF_VARS_ORDERED) == sorted(PERF_VARS)
    assert len(set(PERF_VARS)) == len(PERF_VARS)
    assert set(_CF_PERFORMANCE_VARIABLES) == set(PERF_VARS)
    assert set(_PERF_QUANTITY_PROVENANCE) == set(PERF_VARS)


def test_the_solve_and_log_children_sit_after_the_four_swmm_children_in_the_display_order():
    """Property: ``SWMM_SOLVE`` and ``SWMM_LOG`` render after the existing four SWMM children, so the
    four-child decomposition of ``SWMM`` stays contiguous and the nested pair reads as a refinement
    of ``SWMM_STEP`` rather than as two more children of ``SWMM``.

    Class: positions in ``PERF_VARS_ORDERED``.
    Why: the four children close ``SWMM`` exactly per rank, and the nested pair does not take part
    in that identity, so interleaving them would read as a six-term sum.
    What kills it: inserting the pair between ``SWMM_STEP`` and ``SWMM_OTHER``.
    Second correct implementation: placing the pair at the end of the list.
    """
    order = PERF_VARS_ORDERED
    swmm = order.index("SWMM")
    assert order[swmm : swmm + 5] == ["SWMM", "SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER"]
    assert order.index("SWMM_SOLVE") > order.index("SWMM_OTHER")
    assert order.index("SWMM_LOG") > order.index("SWMM_OTHER")
