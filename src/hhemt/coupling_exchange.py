"""Decode the coupled solver's per-step exchange-flow side-file into SI series and per-node statistics.

TRITON-SWMM exchanges water with the sewer model at every surface timestep, at the manholes the
toolkit lists in ``coupling_nodes.txt``. The solver logs the exact array of per-node exchange flows
it hands to SWMM, once per step, to ``<name>_exchange_replay.bin`` beside the SWMM files, so the
side-file is the only place the exchange is recorded in isolation: SWMM's own node inflow mixes it
with runoff, external-inflow series and upstream link flow. This module turns that binary file into
(a) a block-streamed series in m3/s, (b) the series averaged over the model's reporting intervals,
and (c) eight per-node statistics, without ever holding the whole file in memory.

File layout, as the solver writes it (little-endian, no padding)::

    header : int32 magic ; int32 num_nodes ; int32 value_width        (12 bytes)
    record : value_t dt  ; value_t exchange_q[0 .. num_nodes - 1]     (one per solver step)

``value_t`` is ``float32`` or ``float64`` according to ``value_width`` (4 or 8). ``dt`` is the step
length in seconds. ``exchange_q`` is in cubic feet per second, positive when water moves from the
street surface into the sewer, because SWMM's internal flow unit is cfs and a positive lateral
inflow is water entering the node. Slot ``k`` of a record belongs to line ``k`` of the coupling-node
file, so the header's ``num_nodes`` must equal that file's line count.

The signed convention is kept, and flows are converted to m3/s with ``FT3_TO_M3_FACTOR``, the same
constant the solver divides by when it writes the value. A positive value in every series this
module returns therefore means surface to sewer (drainage); a negative value means sewer to surface
(surcharge).

Statistics and interval means are accumulated in the file's own units (cfs) and converted once at
the end. Deliberately NOT converting each block first: a sum of converted blocks depends on the
block size through floating-point rounding, whereas a sum of integer-valued cfs and seconds is exact
in float64 whatever the block size, so on integer-valued data the result is bit-identical for every
block size. With fractional step lengths the result depends on ``block_records`` at rounding level:
the running time sum is re-associated at each block seam, which measured up to about 4e-15 relative
difference between block sizes on 5,000 random steps. The result is bit-identical only when
``block_records`` is held fixed, and the default block size depends on the node count alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exceptions import ProcessingError

__all__ = [
    "EXCHANGE_LOG_MAGIC_V1",
    "EXCHANGE_LOG_MAGIC_V2",
    "FT3_TO_M3_FACTOR",
    "STATISTIC_NAMES",
    "ExchangeBlock",
    "ExchangeSeries",
    "ExchangeStatistics",
    "ReportingIntervalSeries",
    "exchange_statistics",
    "read_exchange_sidefile",
    "to_reporting_intervals",
]

# The solver divides its m3/s exchange by this constant to log cfs, so multiplying by the same
# constant recovers the m3/s value the solver computed; a rounded 0.0283168 would not.
FT3_TO_M3_FACTOR = 0.028316847

# "SWM2": written by solver builds whose log header carries the value width. "SWMM": the older
# header, which has no width field and cannot be decoded safely (see ``read_exchange_sidefile``).
EXCHANGE_LOG_MAGIC_V2 = 0x53574D32
EXCHANGE_LOG_MAGIC_V1 = 0x53574D4D

_HEADER_BYTES = 12
_SUPPORTED_VALUE_WIDTHS = (4, 8)

# Upper bound on the bytes one block copies into float64. A block is the unit of memory use, so
# peak memory scales with this and not with the file.
_DEFAULT_BLOCK_BYTES = 32 * 1024 * 1024

# Each interval mean is accepted when the step series reaches the last reporting time to within
# this fraction of the reporting interval. Not established: the solver's step lengths are
# accumulated in its own floating-point type, so its last step may end a rounding error short of
# the last print time; this tolerance is a guess at how large that error can be.
_COVERAGE_TOLERANCE_FRACTION = 1e-3

STATISTIC_NAMES: tuple[str, ...] = (
    "coupling_exchange_peak_to_network",
    "coupling_exchange_peak_to_surface",
    "coupling_exchange_volume_to_network",
    "coupling_exchange_volume_to_surface",
    "coupling_exchange_volume_net",
    "coupling_exchange_time_of_peak_to_network",
    "coupling_exchange_time_of_peak_to_surface",
    "coupling_exchange_duration_to_surface",
)


@dataclass(frozen=True)
class ExchangeBlock:
    """One block of consecutive solver steps, in SI units.

    Attributes
    ----------
    step_start_s, step_end_s : numpy.ndarray
        Shape ``(n,)``. Seconds since the start of the side-file; step ``k`` spans
        ``(step_start_s[k], step_end_s[k]]`` and ``step_start_s[k]`` equals ``step_end_s[k-1]``
        bit for bit.
    dt_s : numpy.ndarray
        Shape ``(n,)``. Step lengths in seconds.
    exchange_cms : numpy.ndarray
        Shape ``(n, num_nodes)``. Exchange flow in m3/s, positive from surface to sewer, held
        constant over the step.
    """

    step_start_s: np.ndarray
    step_end_s: np.ndarray
    dt_s: np.ndarray
    exchange_cms: np.ndarray


@dataclass(frozen=True)
class ExchangeSeries:
    """A validated handle on an exchange side-file; the records stay on disk until streamed.

    Attributes
    ----------
    path : pathlib.Path
        The side-file.
    node_ids : tuple of str
        Coupling-node names, in record slot order.
    value_width : int
        Bytes per stored value, 4 or 8.
    n_steps : int
        Number of whole records in the file.
    """

    path: Path
    node_ids: tuple[str, ...]
    value_width: int
    n_steps: int

    @property
    def record_bytes(self) -> int:
        """Bytes in one record: ``dt`` plus one value per coupling node."""
        return self.value_width * (1 + len(self.node_ids))

    def default_block_records(self) -> int:
        """Records per block that keep one block's float64 copy near ``_DEFAULT_BLOCK_BYTES``."""
        return max(1, _DEFAULT_BLOCK_BYTES // (8 * (1 + len(self.node_ids))))

    def iter_blocks(self, block_records: int | None = None) -> Iterator[ExchangeBlock]:
        """Yield the series as consecutive blocks of at most ``block_records`` steps, in m3/s.

        Parameters
        ----------
        block_records : int, optional
            Maximum steps per block. Defaults to a block whose float64 copy is about 32 MiB.
        """
        for start_s, end_s, dt_s, q_cfs in self._iter_raw_blocks(block_records):
            yield ExchangeBlock(start_s, end_s, dt_s, q_cfs * FT3_TO_M3_FACTOR)

    def _iter_raw_blocks(self, block_records: int | None) -> Iterator[tuple[np.ndarray, ...]]:
        """Yield ``(start_s, end_s, dt_s, q_cfs)`` float64 blocks, refusing non-finite or negative values.

        The raw cfs values are what statistics and interval means accumulate (see the module
        docstring). Step times are a running float64 sum of ``dt`` carried across blocks; the sum is
        re-associated at each block seam, so times agree across block sizes to rounding (about
        1e-15 relative) and exactly when the steps are integer-valued.
        """
        n_block = self.default_block_records() if block_records is None else int(block_records)
        if n_block < 1:
            raise ValueError(f"block_records must be at least 1, got {block_records!r}")
        records = np.memmap(
            self.path,
            dtype=np.dtype(f"<f{self.value_width}"),
            mode="r",
            offset=_HEADER_BYTES,
            shape=(self.n_steps, 1 + len(self.node_ids)),
        )
        clock_s = 0.0
        for first in range(0, self.n_steps, n_block):
            block = np.asarray(records[first : first + n_block], dtype=np.float64)
            dt_s = block[:, 0]
            q_cfs = block[:, 1:]
            self._refuse_unusable_values(dt_s, q_cfs, first)
            end_s = clock_s + np.cumsum(dt_s)
            start_s = np.concatenate(([clock_s], end_s[:-1]))
            clock_s = float(end_s[-1])
            yield start_s, end_s, dt_s, q_cfs

    def _refuse_unusable_values(self, dt_s: np.ndarray, q_cfs: np.ndarray, first_record: int) -> None:
        """Refuse a block holding a non-finite value or a negative step length.

        One such value would otherwise flow into every statistic of its node as a NaN or a
        negative volume, with nothing in the result pointing back at the file.
        """
        bad_dt = ~np.isfinite(dt_s) | (dt_s < 0.0)
        bad_q = ~np.isfinite(q_cfs).all(axis=1)
        bad = np.flatnonzero(bad_dt | bad_q)
        if bad.size:
            raise ProcessingError(
                "decode exchange side-file",
                self.path,
                f"record {first_record + int(bad[0])} holds a non-finite value or a negative dt "
                f"({int(bad.size)} unusable record(s) in this block)",
            )


@dataclass(frozen=True)
class ExchangeStatistics:
    """The eight per-node exchange statistics, each an array of shape ``(num_nodes,)``.

    Computed from the solver's own step series, not from reporting-interval means, so a peak is the
    flow the solver actually applied. Flows are m3/s, volumes m3 (sums of flow times step length,
    flow held constant over each step), durations s, and times of peak are minutes since the start
    of the side-file, taken at the END of the step that carried the peak. Where several steps tie
    for the peak the earliest is reported. A node that never exchanged in a direction has a peak of
    0 and a time of peak of NaN for that direction.

    Attributes
    ----------
    coupling_exchange_peak_to_network, coupling_exchange_peak_to_surface : numpy.ndarray
        Largest flow in each direction, both non-negative (the surface peak is a magnitude).
    coupling_exchange_volume_to_network, coupling_exchange_volume_to_surface : numpy.ndarray
        Water moved in each direction, both non-negative.
    coupling_exchange_volume_net : numpy.ndarray
        Signed net volume, positive when more water went to the sewer than came out of it.
    coupling_exchange_time_of_peak_to_network, coupling_exchange_time_of_peak_to_surface : numpy.ndarray
        Minutes since the start of the file at which each peak occurred.
    coupling_exchange_duration_to_surface : numpy.ndarray
        Seconds during which water ran from sewer to surface.
    node_ids : tuple of str
        Node names, in array order.
    """

    node_ids: tuple[str, ...]
    coupling_exchange_peak_to_network: np.ndarray
    coupling_exchange_peak_to_surface: np.ndarray
    coupling_exchange_volume_to_network: np.ndarray
    coupling_exchange_volume_to_surface: np.ndarray
    coupling_exchange_volume_net: np.ndarray
    coupling_exchange_time_of_peak_to_network: np.ndarray
    coupling_exchange_time_of_peak_to_surface: np.ndarray
    coupling_exchange_duration_to_surface: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """The eight statistics keyed by their ``STATISTIC_NAMES`` entry."""
        return {name: getattr(self, name) for name in STATISTIC_NAMES}


@dataclass(frozen=True)
class ReportingIntervalSeries:
    """The exchange flow averaged over the model's reporting intervals.

    Interval ``i`` is ``(report_times_s[i] - R, report_times_s[i]]`` and is labelled at its end.
    Each value is the volume that crossed during the interval divided by ``R``, so
    ``exchange_mean_cms * R`` summed over intervals is exactly the step series' volume.

    Attributes
    ----------
    node_ids : tuple of str
        Node names, in column order.
    report_times_s : numpy.ndarray
        Shape ``(n_intervals,)``. Interval end times, seconds since the start of the side-file.
    exchange_mean_cms : numpy.ndarray
        Shape ``(n_intervals, num_nodes)``. Mean exchange flow in m3/s, positive surface to sewer.
    report_interval_s : float
        The interval length ``R`` in seconds.
    """

    node_ids: tuple[str, ...]
    report_times_s: np.ndarray
    exchange_mean_cms: np.ndarray
    report_interval_s: float

    @property
    def cell_methods(self) -> str:
        """CF ``cell_methods`` text saying each value is a mean over the preceding interval."""
        interval = self.report_interval_s
        text = str(int(interval)) if float(interval).is_integer() else repr(float(interval))
        return f"time: mean (interval: {text} s)"


def read_exchange_sidefile(path: str | Path, coupling_nodes_file: str | Path) -> ExchangeSeries:
    """Validate an exchange side-file against its coupling-node list and return a streaming handle.

    Nothing past the 12-byte header is read here; records are read when the handle is streamed.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``*_exchange_replay.bin`` side-file.
    coupling_nodes_file : str or pathlib.Path
        The ``coupling_nodes.txt`` written for the same scenario, one node name per line in the
        order the solver logs them.

    Returns
    -------
    ExchangeSeries

    Raises
    ------
    ProcessingError
        If the file is shorter than its header, carries the older ``SWMM`` header (its value width
        is not recorded, so a record cannot be decoded without guessing it), carries an unknown
        magic, a width other than 4 or 8, a ``num_nodes`` that differs from the node list's length,
        no records, or a trailing partial record; or if the node list repeats a name.
    """
    path = Path(path)
    node_ids = tuple(line.strip() for line in Path(coupling_nodes_file).read_text().splitlines() if line.strip())
    size = path.stat().st_size
    if size < _HEADER_BYTES:
        raise ProcessingError(
            "decode exchange side-file", path, f"{size} bytes is shorter than the {_HEADER_BYTES}-byte header"
        )
    magic, num_nodes, value_width = (int(v) for v in np.fromfile(path, dtype="<i4", count=3))
    if magic == EXCHANGE_LOG_MAGIC_V1:
        raise ProcessingError(
            "decode exchange side-file",
            path,
            f"magic {magic:#010x} is the older SWMM header, which records no value width; "
            f"only magic {EXCHANGE_LOG_MAGIC_V2:#010x} files can be decoded",
        )
    if magic != EXCHANGE_LOG_MAGIC_V2:
        raise ProcessingError(
            "decode exchange side-file",
            path,
            f"found magic {magic:#010x}, expected {EXCHANGE_LOG_MAGIC_V2:#010x}",
        )
    if value_width not in _SUPPORTED_VALUE_WIDTHS:
        raise ProcessingError(
            "decode exchange side-file", path, f"value width {value_width} is not one of {_SUPPORTED_VALUE_WIDTHS}"
        )
    if num_nodes != len(node_ids):
        raise ProcessingError(
            "decode exchange side-file",
            path,
            f"header num_nodes={num_nodes} but {coupling_nodes_file} lists {len(node_ids)} node(s)",
        )
    if num_nodes < 1:
        raise ProcessingError("decode exchange side-file", path, "the file couples no nodes")
    if len(set(node_ids)) != len(node_ids):
        raise ProcessingError("decode exchange side-file", path, f"{coupling_nodes_file} repeats a node name")
    record_bytes = value_width * (1 + num_nodes)
    n_steps, partial = divmod(size - _HEADER_BYTES, record_bytes)
    if partial:
        raise ProcessingError(
            "decode exchange side-file",
            path,
            f"{partial} trailing byte(s) after {n_steps} whole record(s) of {record_bytes} bytes",
        )
    if n_steps == 0:
        raise ProcessingError("decode exchange side-file", path, "the file holds a header and no records")
    return ExchangeSeries(path=path, node_ids=node_ids, value_width=value_width, n_steps=n_steps)


def exchange_statistics(series: ExchangeSeries, *, block_records: int | None = None) -> ExchangeStatistics:
    """Compute the eight per-node statistics in one streamed pass over the step series.

    Parameters
    ----------
    series : ExchangeSeries
    block_records : int, optional
        Steps per block. Results are equal across block sizes to rounding (about 1e-15 relative) and
        bit-identical when it is held fixed or the data are integer-valued (see the module docstring).

    Returns
    -------
    ExchangeStatistics
    """
    n_nodes = len(series.node_ids)
    peak_net = np.zeros(n_nodes)
    peak_sur = np.zeros(n_nodes)
    t_peak_net = np.full(n_nodes, np.nan)
    t_peak_sur = np.full(n_nodes, np.nan)
    vol_net_dir = np.zeros(n_nodes)
    vol_sur_dir = np.zeros(n_nodes)
    vol_signed = np.zeros(n_nodes)
    dur_sur = np.zeros(n_nodes)
    for _, end_s, dt_s, q_cfs in series._iter_raw_blocks(block_records):
        to_net = np.maximum(q_cfs, 0.0)
        to_sur = np.maximum(-q_cfs, 0.0)
        weight = dt_s[:, None]
        vol_net_dir += (to_net * weight).sum(axis=0)
        vol_sur_dir += (to_sur * weight).sum(axis=0)
        vol_signed += (q_cfs * weight).sum(axis=0)
        dur_sur += (weight * (q_cfs < 0.0)).sum(axis=0)
        # argmax returns the first maximum in the block and ">" keeps the earlier block on a tie, so a
        # peak repeated across steps is dated at its first occurrence.
        for peak, t_peak, flow in ((peak_net, t_peak_net, to_net), (peak_sur, t_peak_sur, to_sur)):
            block_peak = flow.max(axis=0)
            step_of_peak = flow.argmax(axis=0)
            improved = block_peak > peak
            peak[improved] = block_peak[improved]
            t_peak[improved] = end_s[step_of_peak[improved]]
    return ExchangeStatistics(
        node_ids=series.node_ids,
        coupling_exchange_peak_to_network=peak_net * FT3_TO_M3_FACTOR,
        coupling_exchange_peak_to_surface=peak_sur * FT3_TO_M3_FACTOR,
        coupling_exchange_volume_to_network=vol_net_dir * FT3_TO_M3_FACTOR,
        coupling_exchange_volume_to_surface=vol_sur_dir * FT3_TO_M3_FACTOR,
        coupling_exchange_volume_net=vol_signed * FT3_TO_M3_FACTOR,
        coupling_exchange_time_of_peak_to_network=t_peak_net / 60.0,
        coupling_exchange_time_of_peak_to_surface=t_peak_sur / 60.0,
        coupling_exchange_duration_to_surface=dur_sur,
    )


def to_reporting_intervals(
    series: ExchangeSeries,
    report_times_s: Sequence[float] | np.ndarray,
    *,
    report_interval_s: float | None = None,
    block_records: int | None = None,
) -> ReportingIntervalSeries:
    """Average the step series over each reporting interval, conserving volume exactly.

    A solver step may be longer than a reporting interval or may straddle a reporting time; a step
    that straddles one contributes to each interval in proportion to the time it spends in it, so
    the intervals' total volume equals the step series' total volume. Sampling the step series at
    the reporting times instead would lose or double-count whatever the sampled steps do not
    represent.

    Parameters
    ----------
    series : ExchangeSeries
    report_times_s : sequence of float
        Reporting times in seconds since the start of the side-file, strictly increasing and evenly
        spaced; each is the END of its interval.
    report_interval_s : float, optional
        The interval length ``R``. Inferred from the spacing of ``report_times_s`` when there are at
        least two; required when there is one.
    block_records : int, optional
        Steps per block. Results are equal across block sizes to rounding (about 1e-15 relative) and
        bit-identical when it is held fixed or the data are integer-valued.

    Returns
    -------
    ReportingIntervalSeries

    Raises
    ------
    ProcessingError
        If the times are not finite, strictly increasing and evenly spaced at ``R``; if the first
        interval would begin before the start of the file; or if the step series ends before the
        last reporting time (an interval the file does not fully cover would be averaged over time
        it holds no data for).
    """
    times = np.asarray(report_times_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 1 or not np.isfinite(times).all():
        raise ProcessingError(
            "average exchange over reporting intervals",
            series.path,
            "report times must be a non-empty finite 1-D sequence",
        )
    if report_interval_s is None:
        if times.size < 2:
            raise ProcessingError(
                "average exchange over reporting intervals", series.path, "one report time needs report_interval_s"
            )
        report_interval_s = float(times[1] - times[0])
    interval = float(report_interval_s)
    if not (interval > 0.0 and math.isfinite(interval)):
        raise ProcessingError(
            "average exchange over reporting intervals", series.path, f"interval {interval} s is not positive"
        )
    if times.size > 1 and not np.allclose(np.diff(times), interval, rtol=1e-9, atol=0.0):
        raise ProcessingError(
            "average exchange over reporting intervals",
            series.path,
            f"report times are not evenly spaced at {interval} s",
        )
    edges = np.concatenate(([times[0] - interval], times))
    if edges[0] < -1e-9 * interval:
        raise ProcessingError(
            "average exchange over reporting intervals",
            series.path,
            f"the first interval begins at {edges[0]} s, before the start of the file",
        )
    n_interval = times.size
    cfs_seconds = np.zeros((n_interval, len(series.node_ids)))
    last_end_s = 0.0
    for start_s, end_s, dt_s, q_cfs in series._iter_raw_blocks(block_records):
        last_end_s = float(end_s[-1])
        first_edge = np.searchsorted(edges, start_s, side="right") - 1
        last_edge = np.searchsorted(edges, end_s, side="left") - 1
        within_one = (first_edge == last_edge) & (first_edge >= 0) & (first_edge < n_interval)
        if within_one.any():
            # Steps are time-ordered, so the interval indices of the steps inside one interval are
            # non-decreasing and each interval's steps form one contiguous run.
            target = first_edge[within_one]
            volume = q_cfs[within_one] * dt_s[within_one, None]
            run_starts = np.flatnonzero(np.concatenate(([True], target[1:] != target[:-1])))
            cfs_seconds[target[run_starts]] += np.add.reduceat(volume, run_starts, axis=0)
        straddling = ~within_one & (dt_s > 0.0) & (end_s > edges[0]) & (start_s < edges[-1])
        for k in np.flatnonzero(straddling):
            for i in range(max(int(first_edge[k]), 0), min(int(last_edge[k]), n_interval - 1) + 1):
                overlap_s = min(end_s[k], edges[i + 1]) - max(start_s[k], edges[i])
                if overlap_s > 0.0:
                    cfs_seconds[i] += q_cfs[k] * overlap_s
    if last_end_s < edges[-1] - _COVERAGE_TOLERANCE_FRACTION * interval:
        raise ProcessingError(
            "average exchange over reporting intervals",
            series.path,
            f"the step series ends at {last_end_s} s, before the last report time {edges[-1]} s",
        )
    return ReportingIntervalSeries(
        node_ids=series.node_ids,
        report_times_s=times,
        exchange_mean_cms=cfs_seconds * FT3_TO_M3_FACTOR / interval,
        report_interval_s=interval,
    )
