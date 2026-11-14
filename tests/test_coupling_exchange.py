"""Tests for the coupled solver's exchange side-file decoder, ``hhemt.coupling_exchange``.

The decoder is a pure function of a binary file, so every test writes a small side-file with known
values and checks the decoded numbers against figures worked out by hand in the test body. No solver
runs here, and nothing in this file needs a compiled model.

Every expected value is written as a literal or computed by a deliberately different route (a
whole-file numpy formula, a per-step overlap loop) rather than by calling the code under test, so
a wrong rule cannot agree with itself.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from hhemt.exceptions import ProcessingError


class _ModuleUnderTest:
    """Resolve ``hhemt.coupling_exchange`` on first attribute access, inside the test that uses it.

    A module-level import would abort collection with ModuleNotFoundError wherever the module is
    absent, hiding every test as a collection error. Resolving lazily lets each test collect and
    fail in its call phase with an AssertionError raised from this file, which says what is missing.
    """

    def __getattr__(self, name):
        try:
            module = importlib.import_module("hhemt.coupling_exchange")
        except ModuleNotFoundError as exc:
            raise AssertionError("src/hhemt/coupling_exchange.py is absent") from exc
        return getattr(module, name)


ce = _ModuleUnderTest()

# Pinned here as a literal, not imported, so a change to the module's constant is a red test.
CFS_TO_CMS = 0.028316847


def _write_sidefile(
    path: Path,
    dt: np.ndarray,
    q: np.ndarray,
    *,
    magic: int | None = None,
    width: int = 8,
    header_num_nodes: int | None = None,
    trailing: bytes = b"",
) -> Path:
    """Write a side-file in the solver's layout: 12-byte header, then (dt, q[0..n-1]) records."""
    n_nodes = q.shape[1]
    magic = 0x53574D32 if magic is None else magic  # the V2 magic, as a literal so fixtures need no module
    header = np.array([magic, n_nodes if header_num_nodes is None else header_num_nodes, width], dtype="<i4")
    records = np.column_stack([dt, q]).astype(f"<f{width}")
    path.write_bytes(header.tobytes() + records.tobytes() + trailing)
    return path


def _write_nodes(path: Path, names: list[str]) -> Path:
    path.write_text("".join(f"{name}\n" for name in names))
    return path


def _concat(series, block_records=None):
    """All blocks of a series joined into (start_s, end_s, dt_s, exchange_cms) arrays."""
    blocks = list(series.iter_blocks(block_records))
    return tuple(
        np.concatenate([getattr(b, field) for b in blocks])
        for field in ("step_start_s", "step_end_s", "dt_s", "exchange_cms")
    )


def _assert_refused_naming(path, nodes):
    """Assert reading ``path`` raises ProcessingError whose ``filepath`` is ``path``; return the error.

    Compares the structured ``filepath`` attribute, not message text: the message's File line
    carries the pytest temporary directory name, so a substring match on words like "trailing"
    can be satisfied by the directory and cannot fail on the reason.
    """
    with pytest.raises(ProcessingError) as err:
        ce.read_exchange_sidefile(path, nodes)
    assert err.value.filepath == Path(path)
    return err.value


@pytest.fixture
def two_node_files(tmp_path):
    """Five steps on two nodes whose statistics are worked out by hand in the tests.

    dt (s) = 10, 20, 30, 40, 50, so steps end at 10, 30, 60, 100, 150 s.
    q (cfs): n1 = 3, 5, 5, -2, -4 ; n2 = 0, 0, 7, 0, 0.
    """
    dt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    q = np.array([[3.0, 0.0], [5.0, 0.0], [5.0, 7.0], [-2.0, 0.0], [-4.0, 0.0]])
    side = _write_sidefile(tmp_path / "x_exchange_replay.bin", dt, q)
    nodes = _write_nodes(tmp_path / "coupling_nodes.txt", ["n1", "n2"])
    return side, nodes


def test_conversion_constant_is_the_value_the_solver_divides_by():
    """Property: the cfs-to-m3/s constant is exactly 0.028316847.

    Class: NEW CAPABILITY. Plausible wrong implementation: the familiar rounded 0.0283168 (or
    0.3048**3), which differs in the eighth significant figure and mis-scales every volume.
    Why required: the solver divides its m3/s value by this constant when it logs, so decoding
    must multiply by the same number to recover the value the solver computed.
    Killed by: any edit of the constant. A second correct implementation: computing the constant
    as 1 / 35.3146667... would pass only if it reproduced these digits, which is the requirement.
    """
    assert ce.FT3_TO_M3_FACTOR == CFS_TO_CMS


@pytest.mark.parametrize("width", [4, 8])
def test_decode_keeps_the_solver_sign_and_converts_cfs_to_cms(tmp_path, width):
    """Property: each decoded value is the logged cfs value times 0.028316847, sign unchanged.

    Class: NEW CAPABILITY. Plausible wrong implementations: negating the value (the opposite sign
    convention), skipping the conversion, or reading float32 files as float64.
    Why required: positive must mean surface to sewer in every series downstream. This test cannot
    establish that the SOLVER writes that sign; it establishes only that the decoder does not flip
    what the file holds.
    Killed by: a sign flip or a missing/rounded constant. Passes under a second correct
    implementation that converts per block or once per file, since the arithmetic is the same.
    Values are chosen exactly representable in float32 so both widths decode to identical numbers.
    """
    dt = np.array([2.0, 4.0, 8.0])
    q = np.array([[1.0, -0.5], [-0.25, 16.0], [0.0, 3.0]])
    side = _write_sidefile(tmp_path / "s.bin", dt, q, width=width)
    nodes = _write_nodes(tmp_path / "n.txt", ["A", "B"])
    series = ce.read_exchange_sidefile(side, nodes)
    start_s, end_s, dt_s, cms = _concat(series)
    np.testing.assert_array_equal(cms, q * 0.028316847)
    assert cms[0, 0] > 0.0 > cms[0, 1]
    np.testing.assert_array_equal(dt_s, dt)
    np.testing.assert_array_equal(end_s, [2.0, 6.0, 14.0])
    np.testing.assert_array_equal(start_s, [0.0, 2.0, 6.0])
    assert series.node_ids == ("A", "B")
    assert series.value_width == width
    assert series.n_steps == 3


def test_blocks_are_bounded_and_step_times_do_not_depend_on_where_blocks_fall(two_node_files):
    """Property: no block exceeds block_records steps, and step times are identical for any block size.

    Class: NEW CAPABILITY. Plausible wrong implementation: restarting the running clock at each
    block (times then restart at zero), or reading the whole file at once.
    Why required: the read is streamed so memory is bounded by a block; a time axis that moved with
    the block size would make every time-of-peak and interval mean depend on the memory budget.
    Killed by: resetting the clock per block, or ignoring block_records.
    Passes under a second correct implementation that builds the clock with one global cumulative
    sum, because the sums agree exactly on these integer steps.
    """
    side, nodes = two_node_files
    series = ce.read_exchange_sidefile(side, nodes)
    sizes = [len(b.dt_s) for b in series.iter_blocks(2)]
    assert sizes == [2, 2, 1]
    one_block = _concat(series, 1000)
    for block_records in (1, 2, 3):
        pieces = _concat(series, block_records)
        for whole, part in zip(one_block, pieces, strict=True):
            np.testing.assert_array_equal(whole, part)
    np.testing.assert_array_equal(one_block[1], [10.0, 30.0, 60.0, 100.0, 150.0])


def _valid_pair(tmp_path):
    dt = np.array([1.0, 2.0])
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    return (
        _write_sidefile(tmp_path / "ok.bin", dt, q),
        _write_nodes(tmp_path / "ok.txt", ["A", "B"]),
    )


def test_a_valid_file_is_accepted(tmp_path):
    """Property: the refusal cases below are refusals of their defect, not of every file.

    Class: NEW CAPABILITY, the must-fire control for the refusal tests. Plausible wrong
    implementation: a reader that raises on every input, which would satisfy every pytest.raises
    below. Killed by: such a reader. Passes under any correct reader.
    """
    side, nodes = _valid_pair(tmp_path)
    assert ce.read_exchange_sidefile(side, nodes).n_steps == 2


def test_the_older_v1_header_is_refused_naming_the_file_and_the_magic(tmp_path):
    """Property: a V1 file (magic 0x53574D4D) is refused with the file and the magic in the message.

    Class: NEW CAPABILITY. Plausible wrong implementation: treating V1 as V2 (the layout differs by
    the missing width field, so records would be misaligned and decode to wrong numbers silently).
    Why required: three solver builds in use write V1; decoding one as V2 yields plausible garbage.
    Killed by: accepting V1, or refusing without naming the file/magic.
    Passes under a second correct implementation that refuses V1 before or after checking the
    magic is unknown, provided the message still names both.
    """
    side, nodes = _valid_pair(tmp_path)
    v1 = _write_sidefile(
        tmp_path / "v1_exchange_replay.bin", np.array([1.0, 2.0]), np.ones((2, 2)), magic=ce.EXCHANGE_LOG_MAGIC_V1
    )
    with pytest.raises(ProcessingError) as err:
        ce.read_exchange_sidefile(v1, nodes)
    assert "v1_exchange_replay.bin" in str(err.value)
    assert "0x53574d4d" in str(err.value)
    assert ce.read_exchange_sidefile(side, nodes).n_steps == 2


def test_an_unknown_magic_is_refused_naming_the_magic(tmp_path):
    """Property: any magic other than V2 is refused and reported.

    Class: NEW CAPABILITY. Plausible wrong implementation: checking only for V1 and accepting the
    rest. Killed by: that check. Passes under a reader that tests V2 equality first;
    the error is compared by its `filepath`, not its wording.
    """
    _, nodes = _valid_pair(tmp_path)
    bad = _write_sidefile(tmp_path / "bad.bin", np.array([1.0, 2.0]), np.ones((2, 2)), magic=0x12345678)
    _assert_refused_naming(bad, nodes)
    # Reason by construction: `bad` differs from the accepted control only in its magic.
    good, _ = _valid_pair(tmp_path)
    assert ce.read_exchange_sidefile(good, nodes).n_steps == 2


@pytest.mark.parametrize("width", [2, 16, 0])
def test_a_value_width_other_than_4_or_8_is_refused(tmp_path, width):
    """Property: only widths 4 and 8 decode.

    Class: NEW CAPABILITY. Plausible wrong implementation: passing the width to a dtype string and
    letting numpy decode widths 2 or 16 as floats. Killed by: removing the width check. Passes
    under a reader that maps widths through a dict and raises on a missing key.
    """
    _, nodes = _valid_pair(tmp_path)
    side = _write_sidefile(tmp_path / "w.bin", np.array([1.0]), np.ones((1, 2)), width=8)
    header = np.fromfile(side, dtype="<i4", count=3)
    header[2] = width
    raw = bytearray(side.read_bytes())
    raw[:12] = header.astype("<i4").tobytes()
    side.write_bytes(bytes(raw))
    _assert_refused_naming(side, nodes)
    # Reason by construction: the file differs from the accepted control only in its width field.
    good, _ = _valid_pair(tmp_path)
    assert ce.read_exchange_sidefile(good, nodes).n_steps == 2


def test_a_header_node_count_differing_from_the_node_list_is_refused(tmp_path):
    """Property: header num_nodes must equal the coupling-node file's line count.

    Class: NEW CAPABILITY. Plausible wrong implementation: trusting the header and labelling
    columns by position, so every node after the first mismatch gets the wrong name.
    Why required: slot k of a record is line k of the list; a disagreement means the two files are
    from different scenarios. Killed by: dropping the comparison. Passes under a reader that
    compares against a count that ignores blank lines, since the files here have none.
    """
    side = _write_sidefile(tmp_path / "s.bin", np.array([1.0]), np.ones((1, 3)))
    nodes = _write_nodes(tmp_path / "n.txt", ["A", "B"])
    _assert_refused_naming(side, nodes)
    # Reason by construction: the same file is accepted once the list has three names.
    three = _write_nodes(tmp_path / "n3.txt", ["A", "B", "C"])
    assert ce.read_exchange_sidefile(side, three).n_steps == 1


def test_a_trailing_partial_record_is_refused(tmp_path):
    """Property: a file whose length is not header plus a whole number of records is refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: flooring the record count, which drops
    the half-written final step without a word (a run killed mid-write looks complete).
    Killed by: flooring. Passes under a reader that refuses on any nonzero remainder, with a
    different message (the error is compared by its `filepath`, not its wording).
    """
    _, nodes = _valid_pair(tmp_path)
    side = _write_sidefile(tmp_path / "t.bin", np.array([1.0, 2.0]), np.ones((2, 2)), trailing=b"\x00" * 5)
    _assert_refused_naming(side, nodes)
    # Reason by construction, and the control that the check can fail: the identical file without
    # the five stray bytes is accepted, so the trailing bytes are the only possible cause.
    clean = _write_sidefile(tmp_path / "t_clean.bin", np.array([1.0, 2.0]), np.ones((2, 2)))
    assert ce.read_exchange_sidefile(clean, nodes).n_steps == 2


def test_a_file_that_ends_mid_record_after_the_last_whole_one_is_refused_for_every_cut(tmp_path):
    """Property: cutting a valid file at ANY byte inside its final record is refused.

    Class: NEW CAPABILITY, a differently-positioned input to the trailing-partial test: the cut
    lands inside dt, inside a node value and one byte short of complete, not after a whole record.
    Plausible wrong implementation: testing only that the size is a multiple of 4. Killed by: it.
    Passes under any reader that tests divisibility by the record length.
    """
    side, nodes = _valid_pair(tmp_path)
    full = side.read_bytes()
    record_bytes = 8 * 3
    for cut in range(1, record_bytes):
        path = tmp_path / f"cut{cut}.bin"
        path.write_bytes(full[: len(full) - cut])
        _assert_refused_naming(path, nodes)


def test_a_file_with_a_header_and_no_records_and_a_short_header_are_refused(tmp_path):
    """Property: no records, and fewer than 12 bytes, are each refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: np.memmap on a zero-length region
    raises a bare ValueError, or a short read decodes a header from garbage. Killed by: removing
    either check. Passes under readers whose messages differ (the error is compared by `filepath`).
    """
    _, nodes = _valid_pair(tmp_path)
    empty = _write_sidefile(tmp_path / "empty.bin", np.zeros(0), np.zeros((0, 2)))
    _assert_refused_naming(empty, nodes)
    short = tmp_path / "short.bin"
    short.write_bytes(b"\x01" * 7)
    _assert_refused_naming(short, nodes)
    # Reason by construction: a one-record file with the same header and node list is accepted.
    one = _write_sidefile(tmp_path / "one.bin", np.array([1.0]), np.ones((1, 2)))
    assert ce.read_exchange_sidefile(one, nodes).n_steps == 1


def test_a_repeated_node_name_is_refused(tmp_path):
    """Property: a coupling-node list that repeats a name is refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: accepting it, so a node-id coordinate
    holds a duplicate and a lookup by name silently returns the first slot's series.
    Killed by: removing the uniqueness check. Passes under a reader that deduplicates then
    compares counts, since the count then mismatches the header.
    """
    side = _write_sidefile(tmp_path / "d.bin", np.array([1.0]), np.ones((1, 2)))
    nodes = _write_nodes(tmp_path / "n.txt", ["A", "A"])
    _assert_refused_naming(side, nodes)
    # Reason by construction: the same file is accepted with two distinct names.
    distinct = _write_nodes(tmp_path / "n2.txt", ["A", "B"])
    assert ce.read_exchange_sidefile(side, distinct).n_steps == 1


@pytest.mark.parametrize(
    "dt, q, record",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([[1.0], [np.nan], [1.0]]), 1),
        (np.array([1.0, 2.0, 3.0]), np.array([[1.0], [1.0], [np.inf]]), 2),
        (np.array([1.0, -2.0, 3.0]), np.array([[1.0], [1.0], [1.0]]), 1),
        (np.array([np.nan, 2.0, 3.0]), np.array([[1.0], [1.0], [1.0]]), 0),
    ],
)
def test_a_non_finite_value_or_negative_step_is_refused_naming_the_record(tmp_path, dt, q, record):
    """Property: NaN, inf or a negative dt anywhere in the file is refused at the first bad record.

    Class: NEW CAPABILITY. Plausible wrong implementation: no check, so a NaN reaches every
    statistic of its node (the peak comparison skips NaN, the sums turn NaN) with nothing pointing
    at the file. Killed by: removing the check. Passes under a reader that refuses at open rather
    than while streaming, provided the record index is reported.
    """
    side = _write_sidefile(tmp_path / "v.bin", dt, q)
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    series = ce.read_exchange_sidefile(side, nodes)
    with pytest.raises(ProcessingError, match=f"record {record}"):
        ce.exchange_statistics(series)


def test_statistics_equal_hand_computed_values(two_node_files):
    """Property: the eight statistics equal figures worked out by hand from the five-step fixture.

    Class: NEW CAPABILITY. Plausible wrong implementations: reporting the surface peak with its
    sign, counting zero-flow steps in the sewer-to-surface duration (`<= 0`), dating a peak at the
    start of its step, or dating a tied peak at its last occurrence.
    Why required: these are the per-node figures the analysis reports.
    Killed by each of those. Passes under a second correct implementation that computes whole-array
    numpy reductions, because the hand values below do not depend on how they are accumulated.

    Hand arithmetic (cfs and seconds; steps end at 10, 30, 60, 100, 150 s):
      n1 to network: 3x10 + 5x20 + 5x30 = 280 ; peak 5, first reached by the step ending at 30 s.
      n1 to surface: 2x40 + 4x50 = 280 ; peak 4 at the step ending at 150 s ; duration 40 + 50 = 90 s.
      n1 net = 280 - 280 = 0.
      n2 to network: 7x30 = 210 ; peak 7 at 60 s ; nothing to surface, so the peak is 0 with no time.
    """
    side, nodes = two_node_files
    stats = ce.exchange_statistics(ce.read_exchange_sidefile(side, nodes))
    assert tuple(stats.as_dict()) == ce.STATISTIC_NAMES
    assert len(ce.STATISTIC_NAMES) == 8
    np.testing.assert_array_equal(stats.coupling_exchange_peak_to_network, np.array([5.0, 7.0]) * CFS_TO_CMS)
    np.testing.assert_array_equal(stats.coupling_exchange_peak_to_surface, np.array([4.0, 0.0]) * CFS_TO_CMS)
    np.testing.assert_array_equal(stats.coupling_exchange_volume_to_network, np.array([280.0, 210.0]) * CFS_TO_CMS)
    np.testing.assert_array_equal(stats.coupling_exchange_volume_to_surface, np.array([280.0, 0.0]) * CFS_TO_CMS)
    np.testing.assert_array_equal(stats.coupling_exchange_volume_net, np.array([0.0, 210.0]) * CFS_TO_CMS)
    np.testing.assert_array_equal(stats.coupling_exchange_time_of_peak_to_network, [0.5, 1.0])
    np.testing.assert_array_equal(stats.coupling_exchange_time_of_peak_to_surface, [2.5, np.nan])
    np.testing.assert_array_equal(stats.coupling_exchange_duration_to_surface, [90.0, 0.0])
    assert stats.node_ids == ("n1", "n2")


@pytest.mark.parametrize("block_records", [1, 2, 3, 4, 7, 1000])
def test_a_tied_peak_is_dated_at_its_first_step_for_every_block_alignment(two_node_files, block_records):
    """Property: n1's tied network peak (steps 2 and 3) is dated at the first, whatever the blocks.

    Class: NEW CAPABILITY, a differently-positioned input to the hand-computed test: block sizes
    place the tie inside one block (3, 4, 7, 1000), across a block boundary (2) and one step per
    block (1). Plausible wrong implementation: replacing the stored peak on `>=`, which dates a tie
    at the later block. Killed by: it. Passes under a second correct implementation that
    concatenates the series and takes argmax once.
    """
    side, nodes = two_node_files
    stats = ce.exchange_statistics(ce.read_exchange_sidefile(side, nodes), block_records=block_records)
    assert stats.coupling_exchange_time_of_peak_to_network[0] == 0.5


def test_statistics_do_not_depend_on_the_block_size_on_integer_data(tmp_path):
    """Property: every statistic is bit-for-bit the single-block result for block sizes 1, a size
    that does not divide the record count, and a size larger than the file.

    Class: NEW CAPABILITY. Plausible wrong implementations: converting each block to m3/s before
    summing (partial sums then round differently by block size), restarting the clock per block,
    or dropping the final short block.
    Why required: the block size is a memory budget; results must not move with it.
    Killed by: those. Passes under a second correct implementation that accumulates in exact
    integer arithmetic, since integer-valued cfs and seconds sum exactly in float64.
    The whole-file numpy formulas below are an independent check on the single-block value itself.
    """
    rng = np.random.default_rng(20261106)
    n_steps, n_nodes = 37, 4
    dt = rng.integers(1, 10, size=n_steps).astype(float)
    q = rng.integers(-6, 7, size=(n_steps, n_nodes)).astype(float)
    side = _write_sidefile(tmp_path / "r.bin", dt, q)
    nodes = _write_nodes(tmp_path / "n.txt", [f"N{i}" for i in range(n_nodes)])
    series = ce.read_exchange_sidefile(side, nodes)
    reference = ce.exchange_statistics(series, block_records=1000).as_dict()
    for block_records in (1, 5, 36, 37, 1000):
        got = ce.exchange_statistics(series, block_records=block_records).as_dict()
        for name in ce.STATISTIC_NAMES:
            np.testing.assert_array_equal(
                got[name], reference[name], err_msg=f"{name} at block_records={block_records}"
            )
    ends = np.cumsum(dt)
    pos, neg = np.maximum(q, 0.0), np.maximum(-q, 0.0)
    np.testing.assert_allclose(
        reference["coupling_exchange_volume_to_network"], (pos * dt[:, None]).sum(0) * CFS_TO_CMS, rtol=1e-12
    )
    np.testing.assert_allclose(
        reference["coupling_exchange_volume_to_surface"], (neg * dt[:, None]).sum(0) * CFS_TO_CMS, rtol=1e-12
    )
    np.testing.assert_allclose(
        reference["coupling_exchange_volume_net"], (q * dt[:, None]).sum(0) * CFS_TO_CMS, rtol=1e-12
    )
    np.testing.assert_allclose(reference["coupling_exchange_duration_to_surface"], ((q < 0) * dt[:, None]).sum(0))
    np.testing.assert_allclose(reference["coupling_exchange_peak_to_surface"], neg.max(0) * CFS_TO_CMS, rtol=1e-12)
    expected_t = np.where(pos.max(0) > 0, ends[pos.argmax(0)] / 60.0, np.nan)
    np.testing.assert_array_equal(reference["coupling_exchange_time_of_peak_to_network"], expected_t)


def test_statistics_use_the_step_series_so_a_short_spike_is_not_averaged_away(tmp_path):
    """Property: the peak is the flow the solver applied, even when it lasts far less than a report interval.

    Class: NEW CAPABILITY. Plausible wrong implementation: computing statistics from the
    reporting-interval means, which would report 10 cfs lasting one second as 10/120 cfs.
    Why required: peaks must be the solver's own, not an averaged-down figure.
    Killed by: that implementation. Passes under one that scans steps in any order.
    """
    dt = np.array([59.0, 1.0, 60.0])
    q = np.array([[0.0], [10.0], [0.0]])
    side = _write_sidefile(tmp_path / "spike.bin", dt, q)
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    series = ce.read_exchange_sidefile(side, nodes)
    stats = ce.exchange_statistics(series)
    assert stats.coupling_exchange_peak_to_network[0] == 10.0 * CFS_TO_CMS
    means = ce.to_reporting_intervals(series, [120.0], report_interval_s=120.0)
    assert means.exchange_mean_cms[0, 0] == pytest.approx(10.0 * CFS_TO_CMS / 120.0, rel=1e-12)


def test_float32_and_float64_files_give_the_same_statistics(tmp_path):
    """Property: the value width changes how the file is read, not what is computed.

    Class: NEW CAPABILITY. Plausible wrong implementation: accumulating in float32 for width-4
    files, so sums of exactly representable values lose bits. Killed by: that. Passes under a
    reader that widens each block to float64 first, which is the point.
    """
    rng = np.random.default_rng(7)
    dt = rng.integers(1, 5, size=20).astype(float)
    # Integers below 2**24 are exact in float32, but their sums pass 2**24, where a float32 accumulator rounds.
    q = rng.integers(-8, 9, size=(20, 3)).astype(float) * 1_000_003
    nodes = _write_nodes(tmp_path / "n.txt", ["A", "B", "C"])
    results = []
    for width in (4, 8):
        side = _write_sidefile(tmp_path / f"w{width}.bin", dt, q, width=width)
        results.append(ce.exchange_statistics(ce.read_exchange_sidefile(side, nodes)).as_dict())
    for name in ce.STATISTIC_NAMES:
        np.testing.assert_array_equal(results[0][name], results[1][name])
    # Both widths could agree with each other and still both be rounded, so compare with float64 sums.
    exact_net_cfs_s = (q * dt[:, None]).sum(axis=0)
    assert np.abs(exact_net_cfs_s).max() > 2**24
    np.testing.assert_array_equal(results[0]["coupling_exchange_volume_net"], exact_net_cfs_s * CFS_TO_CMS)


def test_interval_means_equal_hand_computed_values_for_steps_that_straddle_report_times(tmp_path):
    """Property: each interval value is the volume that crossed in (t - R, t] divided by R.

    Class: NEW CAPABILITY. Plausible wrong implementations: taking the step that is current at the
    report time (instead of overlapping steps with the interval), or assigning a straddling step
    wholly to the interval it ends in.
    Why required: the stored series must integrate to the solver's volume.
    Killed by: those. Passes under a second correct implementation that differences a cumulative
    volume curve, since the hand values below are the same integral.

    Hand arithmetic, R = 120 s, steps end at 50, 150, 220, 340, 360 s with q = 2, 4, 6, 8, 10 cfs:
      (0, 120]   : 50x2 + 70x4                    = 100 + 280        =  380
      (120, 240] : 30x4 + 70x6 + 20x8             = 120 + 420 + 160  =  700
      (240, 360] : 100x8 + 20x10                  = 800 + 200        = 1000
    and the steps' own total is 100 + 400 + 420 + 960 + 200 = 2080 = 380 + 700 + 1000.
    """
    dt = np.array([50.0, 100.0, 70.0, 120.0, 20.0])
    q = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
    side = _write_sidefile(tmp_path / "i.bin", dt, q)
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    series = ce.read_exchange_sidefile(side, nodes)
    for block_records in (1, 2, 5, 1000):
        out = ce.to_reporting_intervals(series, [120.0, 240.0, 360.0], block_records=block_records)
        np.testing.assert_allclose(
            out.exchange_mean_cms[:, 0], np.array([380.0, 700.0, 1000.0]) / 120.0 * CFS_TO_CMS, rtol=1e-12
        )
        np.testing.assert_array_equal(out.report_times_s, [120.0, 240.0, 360.0])
        assert out.report_interval_s == 120.0
        assert out.node_ids == ("A",)


def test_one_step_longer_than_three_intervals_is_split_across_all_of_them(tmp_path):
    """Property: a step spanning several report times contributes to every interval it overlaps.

    Class: NEW CAPABILITY, a differently-positioned input to the straddling test: here one step
    crosses two report times. Plausible wrong implementation: splitting a step at only its first
    edge. Killed by: that. Passes under any exact overlap implementation.
    Hand arithmetic, R = 120 s, step 1: 300 s at 3 cfs, step 2: 60 s at 6 cfs. Intervals: 120x3 = 360;
    120x3 = 360; 60x3 + 60x6 = 540. Means 3, 3, 4.5 cfs; total 1260 = 900 + 360.
    """
    side = _write_sidefile(tmp_path / "long.bin", np.array([300.0, 60.0]), np.array([[3.0], [6.0]]))
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    out = ce.to_reporting_intervals(ce.read_exchange_sidefile(side, nodes), [120.0, 240.0, 360.0])
    np.testing.assert_allclose(out.exchange_mean_cms[:, 0], np.array([3.0, 3.0, 4.5]) * CFS_TO_CMS, rtol=1e-12)


def test_a_constant_flow_gives_that_flow_in_every_interval_for_irregular_steps(tmp_path):
    """Property: with q constant, every interval mean is q, however the steps fall against report times.

    Class: NEW CAPABILITY, a satisfying input differently positioned from the hand-computed cases:
    irregular step lengths, none aligned to a report time. Plausible wrong implementation: dividing
    by the number of steps in an interval, or by the interval end minus the first step start.
    Killed by: those. Passes under any implementation that divides volume by R.
    """
    dt = np.array([17.0, 41.0, 93.0, 7.0, 130.0, 52.0])
    side = _write_sidefile(tmp_path / "c.bin", dt, np.full((6, 1), 5.0))
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    total = dt.sum()  # 340 s
    out = ce.to_reporting_intervals(ce.read_exchange_sidefile(side, nodes), [100.0, 200.0, 300.0])
    assert total >= 300.0
    np.testing.assert_allclose(out.exchange_mean_cms[:, 0], np.full(3, 5.0 * CFS_TO_CMS), rtol=1e-12)


def _naive_interval_volumes(dt, q, edges):
    """Per-interval cfs-seconds by clipping every step against every interval, step by step."""
    ends = np.cumsum(dt)
    starts = ends - dt
    out = np.zeros((len(edges) - 1, q.shape[1]))
    for i in range(len(edges) - 1):
        for k in range(len(dt)):
            overlap = max(0.0, min(ends[k], edges[i + 1]) - max(starts[k], edges[i]))
            out[i] += q[k] * overlap
    return out


@pytest.mark.parametrize("seed", range(40))
def test_interval_volume_equals_step_volume_for_generated_step_lengths(tmp_path, seed):
    """Property: summed interval volume equals the step series' volume, and each interval equals a
    step-by-step overlap computation, for generated step lengths and block sizes.

    Class: NEW CAPABILITY. Plausible wrong implementations: sampling the step series at report
    times, losing a straddling step's tail, or double counting a step that ends exactly on a report
    time. Killed by: those. Passes under a second correct implementation that differences a
    cumulative volume curve.
    The step lengths run from a fraction of a report interval to more than two intervals, so steps
    sit inside one interval, straddle one report time, and span several. The last step is trimmed
    so the series ends at the last report time.
    The `hypothesis` library is not a declared dependency of this repository, so the generated
    inputs here come from seeded numpy draws; the property is the one a hypothesis test would state.
    """
    rng = np.random.default_rng(seed)
    interval = float(rng.choice([30.0, 120.0, 600.0]))
    n_interval = int(rng.integers(1, 7))
    report_times = interval * np.arange(1, n_interval + 1)
    lengths = []
    while sum(lengths) < report_times[-1]:
        lengths.append(float(rng.uniform(0.01, 2.5) * interval))
    lengths[-1] = report_times[-1] - sum(lengths[:-1])
    dt = np.array(lengths)
    q = rng.normal(0.0, 10.0, size=(len(dt), 3))
    side = _write_sidefile(tmp_path / "g.bin", dt, q)
    nodes = _write_nodes(tmp_path / "n.txt", ["A", "B", "C"])
    series = ce.read_exchange_sidefile(side, nodes)
    out = ce.to_reporting_intervals(
        series, report_times, report_interval_s=interval, block_records=int(rng.choice([1, 3, 1000]))
    )
    edges = np.concatenate(([0.0], report_times))
    expected = _naive_interval_volumes(dt, q, edges) * CFS_TO_CMS / interval
    np.testing.assert_allclose(out.exchange_mean_cms, expected, rtol=1e-9, atol=1e-12)
    step_volume = (q * dt[:, None]).sum(0) * CFS_TO_CMS
    np.testing.assert_allclose(out.exchange_mean_cms.sum(0) * interval, step_volume, rtol=1e-9, atol=1e-9)


def test_cell_methods_states_the_mean_and_the_interval(tmp_path):
    """Property: the stamp reads `time: mean (interval: {R} s)` with R printed without a spurious
    decimal for whole seconds.

    Class: NEW CAPABILITY. Plausible wrong implementation: printing 120.0, or scientific notation
    for large intervals. Killed by: those. Passes under any formatter producing these strings.
    """
    side = _write_sidefile(tmp_path / "cm.bin", np.array([1.0]), np.ones((1, 1)))
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    series = ce.read_exchange_sidefile(side, nodes)
    assert ce.to_reporting_intervals(series, [1.0], report_interval_s=1.0).cell_methods == "time: mean (interval: 1 s)"
    assert (
        ce.to_reporting_intervals(series, [0.5], report_interval_s=0.5).cell_methods == "time: mean (interval: 0.5 s)"
    )
    side2 = _write_sidefile(tmp_path / "cm2.bin", np.array([1.0e6]), np.ones((1, 1)))
    assert ce.to_reporting_intervals(
        ce.read_exchange_sidefile(side2, nodes), [1.0e6], report_interval_s=1.0e6
    ).cell_methods == ("time: mean (interval: 1000000 s)")


@pytest.mark.parametrize(
    "times, interval, reason",
    [
        ([100.0, 250.0, 420.0], None, "evenly spaced"),
        ([120.0, 240.0], 100.0, "evenly spaced"),
        ([60.0], 120.0, "before the start"),
        ([120.0], None, "report_interval_s"),
        ([240.0, 120.0], None, "interval"),
        ([120.0, 240.0, 360.0, 480.0], None, "before the last report time"),
    ],
)
def test_report_times_the_series_cannot_support_are_refused(tmp_path, times, interval, reason):
    """Property: uneven spacing, a first interval before the file starts, a single time without R,
    decreasing times, and a series that stops short of the last report time are each refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: averaging an interval the file only
    partly covers over the full R, which silently dilutes the last value by the missing fraction.
    Killed by: removing the corresponding check. Passes under checks that refuse earlier only if
    they refuse for the same reason, which is why the assertion names the reason.
    The series is 360 s long, so [120, 240, 360] is the one accepted case (see the hand-computed test).
    """
    dt = np.array([100.0, 100.0, 160.0])
    side = _write_sidefile(tmp_path / "r.bin", dt, np.ones((3, 1)))
    nodes = _write_nodes(tmp_path / "n.txt", ["A"])
    series = ce.read_exchange_sidefile(side, nodes)
    with pytest.raises(ProcessingError, match=reason):
        ce.to_reporting_intervals(series, times, report_interval_s=interval)
    assert ce.to_reporting_intervals(series, [120.0, 240.0, 360.0]).exchange_mean_cms.shape == (3, 1)
