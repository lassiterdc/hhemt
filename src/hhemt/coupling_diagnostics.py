"""Decode the coupled solver's optional diagnostic capture files into xarray Datasets.

When the toolkit asks the solver for coupling diagnostics, TRITON writes a handful of extra files
beside its normal outputs: a per-step record of what the SWMM solve did (how many trials it took,
which nodes failed to converge, how much volume the path-3 flooding term booked), a per-node
account of the same over each print interval, per-step link flows, the open-edge flux of the
surface domain and its balance, wet-cell counts per rank, and the cell that limited each time
step. This module turns each of those files into a Dataset the scenario and analysis stores can
carry, without holding more of a file in memory than its format forces.

Binary file layout, as the solver writes it (little-endian, no padding)::

    header : int32 magic ; int32 version ; int32 value_width ; int32 n_items   (16 bytes)
    body   : records, laid out per file (see each reader)

``version`` is 1 and ``value_width`` is 8 in every file, and each file kind has its own magic
(``MAGIC_*`` below); a reader refuses a header that differs in any of the three. ``n_items`` is the
node count of ``swmm_node_path3.bin`` and ``swmm_node_path3_step.bin``, the link count of
``swmm_link_flow_diag.bin`` (both checked against the sidecar name list written beside the file),
4 for ``edge_flux.bin`` and 0 for ``swmm_step_diag.bin`` and ``dt_limiter.bin``. These values are
fixed by the solver's capture-file format.

Text files carry a header row that starts with ``%`` (the ``performance.txt`` style), names
separated by commas. The readers require the exact column names the solver writes and have no headerless form.

Units. The SWMM capture files (path-3 and link volumes, link flows) are stored unconverted in
the engine's ft3 and cfs and are converted on read with ``FT3_TO_M3_FACTOR``, except the lazy
``swmm_node_path3_step.bin`` view, which keeps ft3 and states the factor as an attribute. The
TRITON-side files are SI on disk and are not converted: ``edge_flux.bin`` in m3/s,
``domain_balance.csv`` terms in m3, ``dt_limiter.bin`` depth in m, celerity in m/s and dt in s, and
``outfall_stage.bin`` stage in m and flows in m3/s.

``outfall_stage.bin`` is written only when the analysis lists a driven-stage open-water outfall (the solver key
``swmm_driven_outfalls``). Its ``n_items`` is the number of driven outfalls, checked against the sidecar
``outfall_stage.nodes`` written beside it, which names them in list order.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .coupling_exchange import FT3_TO_M3_FACTOR
from .exceptions import ProcessingError

__all__ = [
    "DIAG_HEADER_BYTES",
    "DT_LIMITER_REASONS",
    "DiagHeader",
    "OutfallIntervalMeans",
    "outfall_stage_interval_means",
    "read_dt_limiter",
    "MAGIC_DT_LIMITER",
    "MAGIC_EDGE_FLUX",
    "MAGIC_OUTFALL_STAGE",
    "MAGIC_SWMM_LINK_FLOW_DIAG",
    "MAGIC_SWMM_NODE_PATH3",
    "MAGIC_SWMM_NODE_PATH3_STEP",
    "MAGIC_SWMM_STEP_DIAG",
    "read_diag_header",
    "read_domain_balance",
    "read_edge_flux",
    "read_node_names",
    "read_outfall_stage",
    "read_swmm_link_flow_diag",
    "read_swmm_node_path3",
    "read_swmm_node_path3_step",
    "read_swmm_step_diag",
    "read_wet_cells",
    "nonconverged_steps_per_node",
]

DIAG_HEADER_BYTES = 16
_DIAG_VERSION = 1
_DIAG_VALUE_WIDTH = 8

# Per-file magics (ASCII "DSP1" and so on), fixed by the solver's capture-file format.
MAGIC_SWMM_STEP_DIAG = 0x44535031
MAGIC_SWMM_NODE_PATH3 = 0x44504E33
MAGIC_SWMM_NODE_PATH3_STEP = 0x44505333
MAGIC_SWMM_LINK_FLOW_DIAG = 0x444C4631
MAGIC_EDGE_FLUX = 0x44454631
MAGIC_DT_LIMITER = 0x44444C31
MAGIC_OUTFALL_STAGE = 0x444F5331  # "DOS1"

# Reason codes of a ``dt_limiter.bin`` record: why the step took the length it took.
DT_LIMITER_REASONS: dict[int, str] = {
    0: "cell_limited",
    1: "clipped_to_print_instant",
    2: "init_dt_fallback",
}

_STEP_RECORD = struct.Struct("<ddiiddd")  # simtime, dt, trials, n_nonconv, p3_pos, p3_neg, link_volume
_STEP_RECORD_BYTES = _STEP_RECORD.size
_NODE_STEP_DTYPE_FIELDS = (("simtime", "<f8"), ("dt", "<f8"))
_EDGE_FLUX_DTYPE = np.dtype([("simtime", "<f8"), ("west", "<f8"), ("east", "<f8"), ("south", "<f8"), ("north", "<f8")])
_DT_LIMITER_DTYPE = np.dtype(
    [
        ("simtime", "<f8"),
        ("dt", "<f8"),
        ("rank", "<i4"),
        ("grow", "<i4"),
        ("gcol", "<i4"),
        ("reason", "<i4"),
        ("h", "<f8"),
        ("celerity", "<f8"),
    ]
)

_WET_CELLS_COLUMNS = ("print_id", "simtime", "rank", "wet_mean", "wet_at_print")
# The two terms the solver adds to ``domain_balance.csv`` when a driven-stage outfall is listed, after
# ``swmm_exchange_m3`` and before the edge terms. With none listed the header has neither.
_OUTFALL_BALANCE_COLUMNS = ("outfall_backflow_m3", "outfall_backflow_deficit_m3")
_DOMAIN_BALANCE_TIME_COLUMN = "t_end"
_DOMAIN_BALANCE_COLUMNS = (
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
)
_DOMAIN_BALANCE_COLUMNS_WITH_OUTFALL = (
    *_DOMAIN_BALANCE_COLUMNS[: _DOMAIN_BALANCE_COLUMNS.index("swmm_exchange_m3") + 1],
    *_OUTFALL_BALANCE_COLUMNS,
    *_DOMAIN_BALANCE_COLUMNS[_DOMAIN_BALANCE_COLUMNS.index("swmm_exchange_m3") + 1 :],
)


@dataclass(frozen=True)
class DiagHeader:
    """The four int32 fields every diagnostic binary file starts with.

    Attributes
    ----------
    magic, version, value_width, n_items : int
        As written by the solver; see the module docstring for what is and is not checked.
    """

    magic: int
    version: int
    value_width: int
    n_items: int


def read_diag_header(path: str | Path, magic: int, n_items: int | None = None) -> DiagHeader:
    """Read the 16-byte header of a diagnostic binary file and refuse one that is not the expected kind.

    Parameters
    ----------
    path : str or pathlib.Path
        The diagnostic file.
    magic : int
        The magic of the file kind being read (one of the ``MAGIC_*`` constants).
    n_items : int, optional
        When given, the header's ``n_items`` must equal it (the fixed widths of the solver's format).

    Raises
    ------
    ProcessingError
        If the file is shorter than its header, or the magic, the version (must be 1), the value
        width (must be 8) or a given ``n_items`` differs.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < DIAG_HEADER_BYTES:
        raise ProcessingError(
            "decode diagnostic file", path, f"{size} bytes is shorter than the {DIAG_HEADER_BYTES}-byte header"
        )
    found = DiagHeader(*(int(v) for v in np.fromfile(path, dtype="<i4", count=4)))
    problems = []
    if found.magic != magic:
        problems.append(f"magic {found.magic:#010x}, expected {magic:#010x}")
    if found.version != _DIAG_VERSION:
        problems.append(f"version {found.version}, expected {_DIAG_VERSION}")
    if found.value_width != _DIAG_VALUE_WIDTH:
        problems.append(f"value_width {found.value_width}, expected {_DIAG_VALUE_WIDTH}")
    if n_items is not None and found.n_items != n_items:
        problems.append(f"n_items {found.n_items}, expected {n_items}")
    if problems:
        raise ProcessingError("decode diagnostic file", path, "header has " + "; ".join(problems))
    return found


def read_node_names(path: str | Path) -> tuple[str, ...]:
    """Read a sidecar name list (``swmm_node_diag.nodes`` or ``swmm_link_flow_diag.links``).

    Names are one per line in the order the matching binary file stores them. A blank line is
    skipped, and a repeated name is refused because a repeat would make two columns answer to one
    label.

    Raises
    ------
    ProcessingError
        If a name repeats.
    """
    path = Path(path)
    names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(set(names)) != len(names):
        raise ProcessingError("decode diagnostic sidecar", path, "the name list repeats a name")
    return names


def _check_regular_body(path: Path, header_bytes: int, record_bytes: int) -> int:
    """Return the whole-record count of a fixed-size-record file, refusing a partial trailing record."""
    body = path.stat().st_size - header_bytes
    if body % record_bytes:
        raise ProcessingError(
            "decode diagnostic file",
            path,
            f"{body} body bytes is not a whole number of {record_bytes}-byte records "
            f"({body % record_bytes} trailing bytes)",
        )
    return body // record_bytes


def _check_sidecar_width(path: Path, header: DiagHeader, names: tuple[str, ...]) -> None:
    """Refuse a file whose header width differs from its sidecar's name count."""
    if header.n_items != len(names):
        raise ProcessingError(
            "decode diagnostic file",
            path,
            f"header n_items is {header.n_items} but the sidecar lists {len(names)} name(s)",
        )


def read_swmm_step_diag(path: str | Path) -> xr.Dataset:
    """Decode ``swmm_step_diag.bin``, the per-SWMM-step record, into a Dataset on dim ``swmm_step``.

    Each step is a 48-byte record ``{f64 simtime, f64 dt, i32 trials, i32 n_nonconv, f64 p3_pos,
    f64 p3_neg, f64 link_volume}`` followed by ``n_nonconv`` int32 node indices. The block length
    varies per step, so the file is walked once from the front and cannot be memory-mapped as an
    array; the walk accumulates each field in a Python list, which measured about 9.6 times the
    file's size in peak memory on a 500,000-step file.

    ``simtime`` is the END time of the step, the stamp the solver's resume positioning reads.
    Volumes are converted to m3 (see the module docstring). The non-converged node indices are
    returned flat on dim ``nonconverged_entry``; step ``k`` owns the ``swmm_step_nonconverged_nodes[k]``
    entries after those of steps ``0..k-1``.

    Raises
    ------
    ProcessingError
        If the file ends inside a record or inside a node block, or a record carries a negative
        node count.
    """
    path = Path(path)
    read_diag_header(path, MAGIC_SWMM_STEP_DIAG, 0)
    raw = np.memmap(path, dtype="u1", mode="r")
    size = raw.size
    view = memoryview(raw)
    simtime: list[float] = []
    dt: list[float] = []
    trials: list[int] = []
    n_nonconv: list[int] = []
    p3_pos: list[float] = []
    p3_neg: list[float] = []
    link_volume: list[float] = []
    node_blocks: list[np.ndarray] = []
    pos = DIAG_HEADER_BYTES
    while pos < size:
        if pos + _STEP_RECORD_BYTES > size:
            raise ProcessingError(
                "decode swmm step diagnostics", path, f"record {len(simtime)} is cut off at byte {size}"
            )
        t, step_dt, n_trials, n_bad, pos_sum, neg_sum, link_sum = _STEP_RECORD.unpack_from(view, pos)
        if n_bad < 0:
            raise ProcessingError(
                "decode swmm step diagnostics", path, f"record {len(simtime)} carries a negative node count {n_bad}"
            )
        block_end = pos + _STEP_RECORD_BYTES + 4 * n_bad
        if block_end > size:
            raise ProcessingError(
                "decode swmm step diagnostics",
                path,
                f"the node block of record {len(simtime)} is cut off at byte {size}",
            )
        if n_bad:
            node_blocks.append(np.frombuffer(view, dtype="<i4", count=n_bad, offset=pos + _STEP_RECORD_BYTES).copy())
        simtime.append(t)
        dt.append(step_dt)
        trials.append(n_trials)
        n_nonconv.append(n_bad)
        p3_pos.append(pos_sum)
        p3_neg.append(neg_sum)
        link_volume.append(link_sum)
        pos = block_end
    del view
    nonconverged = np.concatenate(node_blocks) if node_blocks else np.zeros(0, dtype="<i4")
    n_steps = len(simtime)
    ds = xr.Dataset(
        {
            "swmm_step_dt": ("swmm_step", np.asarray(dt)),
            "swmm_step_trials": ("swmm_step", np.asarray(trials, dtype="int32")),
            "swmm_step_nonconverged_nodes": ("swmm_step", np.asarray(n_nonconv, dtype="int32")),
            "swmm_step_path3_volume_pos": ("swmm_step", np.asarray(p3_pos) * FT3_TO_M3_FACTOR),
            "swmm_step_path3_volume_neg": ("swmm_step", np.asarray(p3_neg) * FT3_TO_M3_FACTOR),
            "swmm_step_link_volume": ("swmm_step", np.asarray(link_volume) * FT3_TO_M3_FACTOR),
            "nonconverged_node_index": ("nonconverged_entry", nonconverged),
        },
        coords={"swmm_step": np.arange(n_steps), "simtime": ("swmm_step", np.asarray(simtime))},
    )
    for name, units, long_name in (
        ("swmm_step_dt", "s", "Length of the coupled SWMM step"),
        ("swmm_step_trials", "1", "Dynamic-wave trials the SWMM step used"),
        ("swmm_step_nonconverged_nodes", "1", "Non-outfall nodes left non-converged after the last trial"),
        ("swmm_step_path3_volume_pos", "m3", "Domain sum of the positive part of the path-3 flooding volume"),
        ("swmm_step_path3_volume_neg", "m3", "Domain sum of the negative part of the path-3 flooding volume"),
        ("swmm_step_link_volume", "m3", "Domain sum of link volume at the end of the step"),
    ):
        ds[name].attrs.update(units=units, long_name=long_name)
    ds["nonconverged_node_index"].attrs.update(
        units="1", long_name="SWMM node index of a node left non-converged, step by step in file order"
    )
    ds["simtime"].attrs.update(units="s", long_name="End time of the SWMM step, seconds since the start of the run")
    return ds


def nonconverged_steps_per_node(ds_step: xr.Dataset, n_nodes: int) -> np.ndarray:
    """Count, per SWMM node index, the steps the node ended non-converged.

    Parameters
    ----------
    ds_step : xarray.Dataset
        The result of :func:`read_swmm_step_diag`.
    n_nodes : int
        The model's node count; an index at or beyond it is refused rather than silently dropped.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes,)``, int64.

    Raises
    ------
    ProcessingError
        If a recorded index is negative or not below ``n_nodes``.
    """
    index = ds_step["nonconverged_node_index"].values
    if index.size and (index.min() < 0 or index.max() >= n_nodes):
        raise ProcessingError(
            "count non-converged steps per node",
            None,
            f"node index range [{int(index.min())}, {int(index.max())}] is outside [0, {n_nodes})",
        )
    return np.bincount(index, minlength=n_nodes).astype("int64")


def read_swmm_node_path3(path: str | Path, nodes_file: str | Path) -> xr.Dataset:
    """Decode ``swmm_node_path3.bin``, the per-node path-3 account per print interval.

    Each record is ``{f64 t_end, f64[n] pos, f64[n] neg, i32[n] n_nonconv}`` for the ``n``
    non-outfall nodes in the order ``nodes_file`` lists them. Returns a Dataset on dims
    ``(timestep, node_id)`` whose ``time_s`` coordinate is the interval end; volumes are m3.

    Raises
    ------
    ProcessingError
        If the header's ``n_items`` differs from the name count, or the file ends inside a record.
    """
    path = Path(path)
    header = read_diag_header(path, MAGIC_SWMM_NODE_PATH3)
    names = read_node_names(nodes_file)
    _check_sidecar_width(path, header, names)
    n = len(names)
    dtype = np.dtype([("t_end", "<f8"), ("pos", "<f8", (n,)), ("neg", "<f8", (n,)), ("n_nonconv", "<i4", (n,))])
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, dtype.itemsize)
    records = np.memmap(path, dtype=dtype, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    ds = xr.Dataset(
        {
            "path3_volume_pos": (("timestep", "node_id"), np.asarray(records["pos"]) * FT3_TO_M3_FACTOR),
            "path3_volume_neg": (("timestep", "node_id"), np.asarray(records["neg"]) * FT3_TO_M3_FACTOR),
            "nonconverged_steps": (("timestep", "node_id"), np.asarray(records["n_nonconv"])),
        },
        coords={
            "timestep": np.arange(n_records),
            "node_id": np.asarray(names, dtype=object),
            "time_s": ("timestep", np.asarray(records["t_end"])),
        },
    )
    ds["path3_volume_pos"].attrs.update(units="m3", long_name="Sum of the positive part of the path-3 volume")
    ds["path3_volume_neg"].attrs.update(units="m3", long_name="Sum of the negative part of the path-3 volume")
    ds["nonconverged_steps"].attrs.update(units="1", long_name="SWMM steps the node ended the interval non-converged")
    ds["time_s"].attrs.update(units="s", long_name="End of the print interval, seconds since the start of the run")
    return ds


def read_swmm_node_path3_step(path: str | Path, nodes_file: str | Path) -> xr.Dataset:
    """Open ``swmm_node_path3_step.bin`` lazily: per step, per node signed path-3 volume.

    Each record is ``{f64 simtime, f64 dt, f64[n] volume}``. The file is large (gigabytes at the
    experiment's size), so the returned ``path3_volume_step`` variable is a view on a read-only
    memory map: nothing is read until a slice is taken, and a slice is converted to m3 by the
    caller's arithmetic on the ``FT3_TO_M3_FACTOR`` attribute rather than copying the file here.

    Raises
    ------
    ProcessingError
        If the header's ``n_items`` differs from the name count, or the file ends inside a record.
    """
    path = Path(path)
    header = read_diag_header(path, MAGIC_SWMM_NODE_PATH3_STEP)
    names = read_node_names(nodes_file)
    _check_sidecar_width(path, header, names)
    n = len(names)
    dtype = np.dtype([*_NODE_STEP_DTYPE_FIELDS, ("volume", "<f8", (n,))])
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, dtype.itemsize)
    records = np.memmap(path, dtype=dtype, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    ds = xr.Dataset(
        {"path3_volume_step": (("swmm_step", "node_id"), records["volume"])},
        coords={
            "swmm_step": np.arange(n_records),
            "node_id": np.asarray(names, dtype=object),
            "simtime": ("swmm_step", np.asarray(records["simtime"])),
            "dt": ("swmm_step", np.asarray(records["dt"])),
        },
    )
    ds["path3_volume_step"].attrs.update(
        units="ft3",
        long_name="Signed path-3 flooding volume of the node in the step, in the SWMM engine's internal unit",
        to_m3_factor=FT3_TO_M3_FACTOR,
    )
    return ds


def read_swmm_link_flow_diag(path: str | Path, links_file: str | Path) -> xr.Dataset:
    """Decode ``swmm_link_flow_diag.bin``: per SWMM step, ``Link[j].newFlow`` of the named links.

    Each record is ``{f64 simtime, f64[L] flow}`` for the ``L`` links ``links_file`` lists. Returns
    a Dataset on ``(swmm_step, link_id)`` with flow in m3/s.

    Raises
    ------
    ProcessingError
        If the header's ``n_items`` differs from the link count, or the file ends inside a record.
    """
    path = Path(path)
    header = read_diag_header(path, MAGIC_SWMM_LINK_FLOW_DIAG)
    names = read_node_names(links_file)
    _check_sidecar_width(path, header, names)
    n = len(names)
    dtype = np.dtype([("simtime", "<f8"), ("flow", "<f8", (n,))])
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, dtype.itemsize)
    records = np.memmap(path, dtype=dtype, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    ds = xr.Dataset(
        {"link_flow": (("swmm_step", "link_id"), np.asarray(records["flow"]) * FT3_TO_M3_FACTOR)},
        coords={
            "swmm_step": np.arange(n_records),
            "link_id": np.asarray(names, dtype=object),
            "simtime": ("swmm_step", np.asarray(records["simtime"])),
        },
    )
    ds["link_flow"].attrs.update(units="m3 s-1", long_name="Link flow at the end of the SWMM step")
    return ds


def read_edge_flux(path: str | Path) -> xr.Dataset:
    """Decode ``edge_flux.bin``: per surface step, the interface flux through each domain edge.

    Each record is ``{f64 simtime, f64 west, f64 east, f64 south, f64 north}``. Returns a Dataset
    on ``(step, edge)``; flux is in m3 s-1 on disk, so flux times step
    length is the volume the interval balance line needs.

    Raises
    ------
    ProcessingError
        If the file ends inside a record.
    """
    path = Path(path)
    read_diag_header(path, MAGIC_EDGE_FLUX, 4)
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, _EDGE_FLUX_DTYPE.itemsize)
    records = np.memmap(path, dtype=_EDGE_FLUX_DTYPE, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    flux = np.stack([np.asarray(records[edge]) for edge in ("west", "east", "south", "north")], axis=1)
    ds = xr.Dataset(
        {"edge_flux": (("step", "edge"), flux)},
        coords={
            "step": np.arange(n_records),
            "edge": np.asarray(["west", "east", "south", "north"], dtype=object),
            "simtime": ("step", np.asarray(records["simtime"])),
        },
    )
    ds["edge_flux"].attrs.update(units="m3 s-1", long_name="Mass flux through the domain edge in the step")
    return ds


def read_dt_limiter(path: str | Path) -> xr.Dataset:
    """Decode ``dt_limiter.bin``: per step, the cell whose time-step bound was the smallest.

    Each 48-byte record is ``{f64 simtime, f64 dt, i32 rank, i32 grow, i32 gcol, i32 reason,
    f64 h, f64 celerity}``; ``grow`` and ``gcol`` are global grid
    indices; a cell of the ghost ring is recorded as it is, with an index of -1 or one outside the rank's rows.
    ``reason`` is ``0`` cell-limited, ``1`` clipped to the print instant, ``2`` the ``init_dt`` fallback
    (:data:`DT_LIMITER_REASONS`). Returns a Dataset on dim ``step``.

    Raises
    ------
    ProcessingError
        If the file ends inside a record or a record carries a reason code outside 0-2.
    """
    path = Path(path)
    read_diag_header(path, MAGIC_DT_LIMITER, 0)
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, _DT_LIMITER_DTYPE.itemsize)
    records = np.memmap(path, dtype=_DT_LIMITER_DTYPE, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    reason = np.asarray(records["reason"])
    unknown = np.flatnonzero(~np.isin(reason, list(DT_LIMITER_REASONS)))
    if unknown.size:
        raise ProcessingError(
            "decode dt limiter", path, f"record {int(unknown[0])} carries reason code {int(reason[unknown[0]])}"
        )
    ds = xr.Dataset(
        {
            "dt_limiter_dt": ("step", np.asarray(records["dt"])),
            "dt_limiter_rank": ("step", np.asarray(records["rank"])),
            "dt_limiter_row": ("step", np.asarray(records["grow"])),
            "dt_limiter_col": ("step", np.asarray(records["gcol"])),
            "dt_limiter_reason": ("step", reason),
            "dt_limiter_depth": ("step", np.asarray(records["h"])),
            "dt_limiter_celerity": ("step", np.asarray(records["celerity"])),
        },
        coords={"step": np.arange(n_records), "simtime": ("step", np.asarray(records["simtime"]))},
    )
    ds["dt_limiter_reason"].attrs.update(
        flag_values=np.array(sorted(DT_LIMITER_REASONS), dtype="int32"),
        flag_meanings=" ".join(DT_LIMITER_REASONS[k] for k in sorted(DT_LIMITER_REASONS)),
    )
    for name, units in (("dt", "s"), ("depth", "m"), ("celerity", "m s-1"), ("rank", "1"), ("row", "1"), ("col", "1")):
        ds[f"dt_limiter_{name}"].attrs["units"] = units
    for name in ("row", "col"):
        ds[f"dt_limiter_{name}"].attrs["long_name"] = (
            f"Global grid {name} of the step-limiting cell; a cell of the ghost ring is recorded as it is, with an "
            "index of -1 or one outside the rank's rows"
        )
    return ds


def read_outfall_stage(path: str | Path, nodes_file: str | Path) -> xr.Dataset:
    """Decode ``outfall_stage.bin``: per SWMM step, the stage TRITON set at each driven outfall and its flows.

    Each record is ``{f64 simtime, f64 dt, f64[n] stage_m, f64[n] q_out_m3s, f64[n] q_back_applied_m3s}`` for the
    ``n`` driven outfalls ``nodes_file`` lists (``outfall_stage.nodes``, in list order). ``simtime`` is the END time
    of the step (the stamp of the exchange side-file) and ``dt`` its length. ``q_out_m3s`` is the outfall conduit's
    flow toward the open water, so a negative value is backflow into the sewer; ``q_back_applied_m3s`` is the part of
    that backflow the surface actually gave up, never negative. Every value is SI on disk and is returned as written.
    Returns a Dataset on ``(swmm_step, node_id)``.

    Raises
    ------
    ProcessingError
        If the magic, version or value width differs, the header's ``n_items`` is zero or differs from the name count,
        or the file ends inside a record.
    """
    path = Path(path)
    header = read_diag_header(path, MAGIC_OUTFALL_STAGE)
    names = read_node_names(nodes_file)
    _check_sidecar_width(path, header, names)
    if not names:
        raise ProcessingError(
            "decode outfall stage", path, "the file lists no outfall; a driven run lists at least one"
        )
    n = len(names)
    dtype = np.dtype(
        [
            ("simtime", "<f8"),
            ("dt", "<f8"),
            ("stage_m", "<f8", (n,)),
            ("q_out_m3s", "<f8", (n,)),
            ("q_back_applied_m3s", "<f8", (n,)),
        ]
    )
    n_records = _check_regular_body(path, DIAG_HEADER_BYTES, dtype.itemsize)
    records = np.memmap(path, dtype=dtype, mode="r", offset=DIAG_HEADER_BYTES, shape=(n_records,))
    ds = xr.Dataset(
        {
            "outfall_stage_m": (("swmm_step", "node_id"), np.asarray(records["stage_m"])),
            "outfall_discharge_m3s": (("swmm_step", "node_id"), np.asarray(records["q_out_m3s"])),
            "outfall_backflow_applied_m3s": (("swmm_step", "node_id"), np.asarray(records["q_back_applied_m3s"])),
        },
        coords={
            "swmm_step": np.arange(n_records),
            "node_id": np.asarray(names, dtype=object),
            "simtime": ("swmm_step", np.asarray(records["simtime"])),
            "dt": ("swmm_step", np.asarray(records["dt"])),
        },
    )
    ds["outfall_stage_m"].attrs.update(
        units="m", long_name="Water-surface elevation TRITON set as the stage of the driven outfall before the step"
    )
    ds["outfall_discharge_m3s"].attrs.update(
        units="m3 s-1",
        long_name="Flow of the driven outfall's conduit toward the open water in the step; negative is backflow",
    )
    ds["outfall_backflow_applied_m3s"].attrs.update(
        units="m3 s-1", long_name="Backflow into the sewer the surface gave up at the driven outfall in the step"
    )
    ds["simtime"].attrs.update(units="s", long_name="End time of the SWMM step, seconds since the start of the run")
    ds["dt"].attrs.update(units="s", long_name="Length of the SWMM step")
    return ds


# A step that does not start where the one before it ended, by more than this many seconds, is a hole or an overlap
# in the record and not rounding of the solver's running time.
_STEP_CONTIGUITY_TOLERANCE_S = 1e-6
# The record must reach the last report time to within this fraction of the interval, the tolerance the exchange
# reader (``coupling_exchange.to_reporting_intervals``) holds its own coverage to.
_INTERVAL_COVERAGE_TOLERANCE_FRACTION = 1e-3


@dataclass(frozen=True)
class OutfallIntervalMeans:
    """The driven-outfall stage and discharge averaged over the model's reporting intervals.

    Interval ``i`` is ``(report_times_s[i] - R, report_times_s[i]]`` and is labelled at its end. Each value is the time
    integral of the stepwise series over the interval divided by ``R``, so ``discharge_m3s * R`` summed over the
    intervals is exactly the volume the step series carries.

    Attributes
    ----------
    node_ids : tuple of str
        Outfall names, in column order.
    stage_m, discharge_m3s : numpy.ndarray
        Shape ``(n_intervals, n_outfalls)``. The mean stage in m and the mean signed flow toward the open water in m3/s.
    report_interval_s : float
        The interval length ``R`` in seconds.
    """

    node_ids: tuple[str, ...]
    stage_m: np.ndarray
    discharge_m3s: np.ndarray
    report_interval_s: float

    @property
    def cell_methods(self) -> str:
        """CF ``cell_methods`` text saying each value is a mean over the preceding interval."""
        interval = self.report_interval_s
        text = str(int(interval)) if float(interval).is_integer() else repr(float(interval))
        return f"time: mean (interval: {text} s)"


def outfall_stage_interval_means(ds_stage: xr.Dataset, report_times_s: np.ndarray) -> OutfallIntervalMeans:
    """Average the stepwise stage and discharge of :func:`read_outfall_stage` over each reporting interval.

    A step may be longer than an interval or straddle a reporting time; it contributes to each interval in proportion
    to the time it spends in it, the rule the exchange flow follows, so no step is sampled and none is counted twice.
    The integral is taken as a running sum over the steps that is read at each interval edge, which makes the means of
    adjacent intervals telescope onto the whole record.

    Parameters
    ----------
    ds_stage : xarray.Dataset
        The result of :func:`read_outfall_stage`.
    report_times_s : numpy.ndarray
        Reporting times in seconds since the start of the run, finite, strictly increasing and evenly spaced, at least
        two; each is the END of its interval.

    Raises
    ------
    ProcessingError
        If the record is empty, a step length is not positive and finite, the steps do not run back to back, the
        report times are not even, the first interval would begin before the run, or the record does not cover the
        first and last interval.
    """
    ends = np.asarray(ds_stage["simtime"].values, dtype="float64")
    dt = np.asarray(ds_stage["dt"].values, dtype="float64")
    node_ids = tuple(str(n) for n in ds_stage["node_id"].values)
    times = np.asarray(report_times_s, dtype="float64")
    operation = "average the driven outfall stage over reporting intervals"
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ProcessingError(operation, None, "report times must be a finite 1-D sequence of at least two times")
    interval = float(times[1] - times[0])
    if not interval > 0.0 or not np.allclose(np.diff(times), interval, rtol=1e-9, atol=0.0):
        raise ProcessingError(operation, None, f"report times are not evenly spaced at a positive {interval} s")
    edges = np.concatenate(([times[0] - interval], times))
    if edges[0] < -1e-9 * interval:
        raise ProcessingError(operation, None, f"the first interval begins at {edges[0]} s, before the run starts")
    if ends.size == 0:
        raise ProcessingError(operation, None, "the record holds no step")
    if not (np.isfinite(ends).all() and np.isfinite(dt).all() and (dt > 0.0).all()):
        raise ProcessingError(operation, None, "a step end time or length is not finite, or a length is not positive")
    starts = ends - dt
    seam = starts[1:] - ends[:-1]
    if seam.size and np.abs(seam).max() > _STEP_CONTIGUITY_TOLERANCE_S:
        first = int(np.argmax(np.abs(seam)))
        raise ProcessingError(
            operation,
            None,
            f"step {first + 1} starts {seam[first]} s after step {first} ends; the steps are not back to back",
        )
    slack = _INTERVAL_COVERAGE_TOLERANCE_FRACTION * interval
    if starts[0] > edges[0] + slack or ends[-1] < edges[-1] - slack:
        raise ProcessingError(
            operation,
            None,
            f"the record spans {starts[0]} s to {ends[-1]} s and the reporting intervals span {edges[0]} s to "
            f"{edges[-1]} s",
        )
    at = np.searchsorted(ends, edges, side="left")
    inside = at < ends.size
    partial_s = np.zeros(edges.size)
    partial_s[inside] = np.clip(edges[inside] - starts[at[inside]], 0.0, dt[at[inside]])

    def mean_over_intervals(values: np.ndarray) -> np.ndarray:
        running = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values * dt[:, None], axis=0)])
        integral = running[at].copy()
        integral[inside] += values[at[inside]] * partial_s[inside, None]
        return np.diff(integral, axis=0) / interval

    return OutfallIntervalMeans(
        node_ids=node_ids,
        stage_m=mean_over_intervals(np.asarray(ds_stage["outfall_stage_m"].values, dtype="float64")),
        discharge_m3s=mean_over_intervals(np.asarray(ds_stage["outfall_discharge_m3s"].values, dtype="float64")),
        report_interval_s=interval,
    )


def _read_percent_header_csv(path: Path, accepted: tuple[tuple[str, ...], ...]) -> pd.DataFrame:
    """Read a capture csv whose first line is ``%name, name, ...`` and require exactly one of the ``accepted`` columns.

    Raises
    ------
    ProcessingError
        If the first line does not start with ``%`` or its names are none of the ``accepted`` lists.
    """
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("%"):
        raise ProcessingError("decode diagnostic csv", path, "the first line is not a header row starting with '%'")
    names = tuple(n.strip() for n in lines[0][1:].split(","))
    if names not in accepted:
        expected = " or ".join(str(list(columns)) for columns in accepted)
        raise ProcessingError("decode diagnostic csv", path, f"header names {list(names)}, expected {expected}")
    return pd.read_csv(path, skiprows=1, header=None, names=list(names), skipinitialspace=True)


def read_domain_balance(path: str | Path) -> xr.Dataset:
    """Decode ``domain_balance.csv``, one closed volume-balance row per print interval.

    The header row is ``%t_end, d_storage_m3, runoff_m3, inflow_m3, extbc_m3, clamp_m3,
    swmm_exchange_m3, edge_west_m3, edge_east_m3, edge_south_m3, edge_north_m3, residual_m3``;
    every term but ``t_end`` is a volume in m3 summed over the interval, signed positive into the
    domain. Returns a Dataset on ``timestep`` with ``time_s`` the interval end.

    When the analysis lists a driven-stage open-water outfall the solver writes two more columns,
    ``outfall_backflow_m3`` (signed positive into the domain, so the negative of the backflow taken from the cells) and
    ``outfall_backflow_deficit_m3`` (the backflow volume the surface could not supply, positive, kept out of the
    residual), between ``swmm_exchange_m3`` and ``edge_west_m3``. They are read by those names and the Dataset gains the
    two variables; without them the header is the list above and the Dataset is the one it always was.

    Raises
    ------
    ProcessingError
        If the header row is missing or its names are neither the list above nor that list with both outfall columns
        (one of the two alone is refused).
    """
    path = Path(path)
    frame = _read_percent_header_csv(path, (_DOMAIN_BALANCE_COLUMNS, _DOMAIN_BALANCE_COLUMNS_WITH_OUTFALL))
    values = frame.drop(columns=[_DOMAIN_BALANCE_TIME_COLUMN])
    ds = xr.Dataset(
        {c: ("timestep", values[c].to_numpy(dtype="float64")) for c in values.columns},
        coords={
            "timestep": np.arange(len(frame)),
            "time_s": ("timestep", frame[_DOMAIN_BALANCE_TIME_COLUMN].to_numpy("float64")),
        },
    )
    for c in values.columns:
        ds[c].attrs.update(units="m3", cell_methods="time: sum")
    if _OUTFALL_BALANCE_COLUMNS[0] in ds.data_vars:
        ds["outfall_backflow_m3"].attrs["long_name"] = (
            "Volume the backflow into the driven open-water outfalls added to the surface domain, signed positive into "
            "the domain, so zero or negative because backflow takes water from the cells"
        )
        ds["outfall_backflow_deficit_m3"].attrs["long_name"] = (
            "Backflow volume the surface could not supply at the driven open-water outfalls, positive and not in the "
            "residual"
        )
    ds["time_s"].attrs.update(units="s", long_name="End of the print interval, seconds since the start of the run")
    return ds


def read_wet_cells(path: str | Path) -> xr.Dataset:
    """Decode ``wet_cells.csv``: per rank, per print interval, interior cells with ``h > hextra``.

    The header row is ``%print_id, simtime, rank, wet_mean, wet_at_print``. Returns a Dataset on
    ``(Rank, timestep)``. The rank grid must be complete: a rank absent from an interval is
    refused rather than filled, because a missing row after a resume is the defect the capture's
    truncation rule exists to prevent and a fill would hide it.

    Raises
    ------
    ProcessingError
        If the header row is missing or wrong, or the ``(rank, print_id)`` grid has a hole or a repeat.
    """
    path = Path(path)
    frame = _read_percent_header_csv(path, (_WET_CELLS_COLUMNS,))
    ranks = np.sort(frame["rank"].unique())
    prints = np.sort(frame["print_id"].unique())
    if len(frame) != len(ranks) * len(prints) or frame.duplicated(["rank", "print_id"]).any():
        raise ProcessingError(
            "decode wet cells",
            path,
            f"{len(frame)} row(s) do not fill a grid of {len(ranks)} rank(s) by {len(prints)} interval(s) exactly once",
        )
    grid = frame.set_index(["rank", "print_id"]).sort_index()
    shape = (len(ranks), len(prints))
    simtime = grid["simtime"].to_numpy("float64").reshape(shape)
    interval_time = simtime[0] if len(ranks) else np.zeros(0)
    if not np.all(simtime == interval_time):
        raise ProcessingError("decode wet cells", path, "ranks disagree on the time of an interval")
    ds = xr.Dataset(
        {
            "wet_cells_mean": (("Rank", "timestep"), grid["wet_mean"].to_numpy("float64").reshape(shape)),
            "wet_cells_at_print": (("Rank", "timestep"), grid["wet_at_print"].to_numpy("float64").reshape(shape)),
        },
        coords={
            "Rank": ranks.astype("int64"),
            "timestep": np.arange(len(prints)),
            "print_id": ("timestep", prints.astype("int64")),
            "time_s": ("timestep", interval_time),
        },
    )
    ds["wet_cells_mean"].attrs.update(units="1", long_name="Mean over the interval's steps of interior wet cells")
    ds["wet_cells_at_print"].attrs.update(units="1", long_name="Interior wet cells at the print instant")
    return ds
