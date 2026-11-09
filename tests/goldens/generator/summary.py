"""Summary-layer writer for the golden analysis trees.

A golden is one committed zip file holding a finished analysis tree and the files hhemt needs to construct an
analysis object from it (``tests/goldens/SPEC.md`` section 1).

Writes the per-scenario processed stores under ``sims/{event_id}/processed/`` and the
consolidated tiers (``analysis_datatree.zarr``, ``experiment_datatree.zarr``) at the reduced
dimensions of ``tests/goldens/SPEC.md`` section 2. Every store carries a consolidated-metadata index at its root,
as a layout 27 toolkit writes it (SPEC.md section 8).

Specification followed (SPEC.md section 8): the published data model
``docs/reference/output-data-model.md`` fixes the documented variables, their units and CF
``standard_name`` values, the tier layout and the root attributes; where that document is silent the
variable inventory is the baseline recorded in SPEC.md section 8, observed (tag ``[OBS]``, SPEC.md
section 0) in a finished tree written at hhemt layout version 22. That baseline is known short of layout
version 26 for the performance stores, which gained ``<Col>_coherent`` columns and the
``perf_column_set_across_allocations`` coordinate after layout 22 (SPEC.md section 8): this module
writes the nine documented-or-observed timing columns and nothing more.

Imports: standard library, ``numpy``, ``xarray`` and the sibling ``raw`` module only, because the generator is
independent of hhemt: that independence is the premise of ``tests/test_golden_conformance.py``, and a generator that
imported hhemt would have hhemt accept trees hhemt itself wrote (``tests/test_golden_regeneration.py``). Everything
is closed-form and deterministic, because a fresh generation must equal the committed golden zips by content (the
same test).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from .raw import NODE_OUTFALL, Dimensions, Network, _triton_field, default_network

# Fixed provenance stand-ins: a golden has no real producing commit (SPEC.md 4.2).
PRODUCING_SHA = "000000000000"
PRODUCING_VERSION = "0.0.0+golden"
CREATED = "2000-01-01T00:00:00+00:00"
SYSTEM_ID = "golden_system"  # the data model names the attribute and fixes no value
# The ``Analysis ended on`` time of the SWMM report that raw.py writes (``_ENDED`` there). A fixed literal, not
# computed from ``_times``, whose last step falls at 00:02:00.
ANALYSIS_END = "2000-01-01 00:00:12"

PERF_COLUMNS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")

# Store file name stems per arm, as the per-scenario tier names them (SPEC.md 4.3).
ARM_PREFIX = {"tritonswmm": "TRITONSWMM", "triton": "TRITON_only", "swmm": "SWMM_only"}
# Group names of the consolidated tree (SPEC.md section 8, baseline inventory).
ARM_GROUP = {"tritonswmm": "tritonswmm", "triton": "triton_only", "swmm": "swmm_only"}

# Documented TRITON variables (output-data-model.md, "2D surface results (TRITON)").
_TRITON_ATTRS = {
    "max_flood_depth": {
        "units": "m",
        "standard_name": "flood_water_thickness",
        "long_name": "Maximum flood depth",
        "cell_methods": "time: maximum",
    },
    "flood_depth": {
        "units": "m",
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth timeseries",
    },
    "flood_depth_last_tstep": {
        "units": "m",
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth at final timestep",
        "cell_methods": "time: point",
    },
    "time_of_max_flood_depth": {"units": "min", "long_name": "Time of maximum flood depth"},
    "max_flood_velocity": {
        "units": "m s-1",
        "standard_name": "flood_water_speed",
        "long_name": "Maximum flood velocity",
        "cell_methods": "time: maximum",
    },
    "time_of_max_flood_velocity": {"units": "min", "long_name": "Time of maximum flood velocity"},
    "x_flood_velocity_at_time_of_max_flood_velocity": {
        "units": "m s-1",
        "long_name": "Flood velocity x at the time of maximum flood velocity",
    },
    "y_flood_velocity_at_time_of_max_flood_velocity": {
        "units": "m s-1",
        "long_name": "Flood velocity y at the time of maximum flood velocity",
    },
    "max_unit_discharge": {
        "units": "m2 s-1",
        "long_name": "Maximum unit discharge",
        "cell_methods": "time: maximum",
    },
    "unit_discharge_x": {"units": "m2 s-1", "long_name": "Unit discharge x"},
    "unit_discharge_y": {"units": "m2 s-1", "long_name": "Unit discharge y"},
    "unit_discharge_x_at_time_of_max_unit_discharge": {
        "units": "m2 s-1",
        "long_name": "Unit discharge x at the time of maximum unit discharge",
    },
    "unit_discharge_y_at_time_of_max_unit_discharge": {
        "units": "m2 s-1",
        "long_name": "Unit discharge y at the time of maximum unit discharge",
    },
    "time_of_max_unit_discharge": {"units": "min", "long_name": "Time of maximum unit discharge"},
    "final_surface_flood_volume": {
        "units": "m3",
        "long_name": "Final surface flood volume",
        "cell_methods": "area: sum",
    },
}
# Documented SWMM network variables (output-data-model.md, "Network results (SWMM)"), plus max_velocity, which
# that document's note on unit discharge defines on a link node as maximum conduit velocity.
_SWMM_ATTRS = {
    "total_inflow_vol": {"units": "10^6 L", "long_name": "Total inflow volume"},
    "max_flow": {"units": "m3 s-1", "long_name": "Maximum flow rate"},
    "max_over_full_flow": {"units": "1", "long_name": "Maximum flow as a fraction of full-flow capacity"},
    "max_over_full_depth": {"units": "1", "long_name": "Maximum depth as a fraction of full depth"},
    "max_velocity": {"units": "m s-1", "standard_name": "sea_water_speed", "long_name": "Maximum conduit velocity"},
}

# The unit of every other stored numeric SWMM variable. SWMM is in its SI system (the toolkit refuses any model whose
# flow units are not CMS, SPEC.md section 8): depths and heads in m, flows in m3 s-1, volumes in 10^6 L, and the
# times of the maxima in minutes. A capacity is a dimensionless fraction (or a control setting); a flow balance
# error is a percent where SWMM printed a number alone and litres where it printed `ltr` (the pair is split).
# Written from the data model, not from hhemt (this module imports nothing from it).
_SWMM_UNITS = {
    "depth": "m",
    "head": "m",
    "inflow_flow": "m3 s-1",
    "flooding": "m3 s-1",
    "flow": "m3 s-1",
    "velocity": "m s-1",
    "link_depth": "m",
    "capacity": "1",
    "capacity_setting": "1",
    "hours_flooded": "h",
    "lateral_inflow_vol": "10^6 L",
    "tot_flooded_vol": "10^6 L",
    "max_flood_rate": "m3 s-1",
    "max_lateral_inflow": "m3 s-1",
    "max_total_inflow": "m3 s-1",
    "max_ponded_depth": "m",
    "time_of_max_flood": "min",
    "time_of_max_total_inflow": "min",
    "time_of_max_flow": "min",
    "flow_balance_error": "percent",
    "flow_balance_error_volume": "L",
}
for _stem in (
    "depth",
    "head",
    "inflow_flow",
    "flooding",
    "flow",
    "velocity",
    "link_depth",
    "capacity",
    "capacity_setting",
):
    for _suffix in ("last", "max"):
        _SWMM_UNITS[f"{_stem}_{_suffix}"] = _SWMM_UNITS[_stem]
_SWMM_ATTRS.update(
    {name: {"units": unit, "long_name": name.replace("_", " ").capitalize()} for name, unit in _SWMM_UNITS.items()}
)

# The nine timing columns of the performance stores: every one is a cumulative time in seconds.
_PERF_ATTRS = {col: {"units": "s", "long_name": f"{col} time"} for col in PERF_COLUMNS}

# The member parameters of the experiment tier: each is a count.
_PARAMETER_ATTRS = {
    name: {"units": "1", "long_name": long_name}
    for name, long_name in (
        ("n_mpi_procs", "Number of MPI tasks"),
        ("n_omp_threads", "Number of OpenMP threads"),
        ("n_gpus", "Number of GPUs"),
        ("n_nodes", "Number of nodes"),
    )
}


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


# A variable the data model documents takes its attributes from ``table``; any other variable gets only a
# ``long_name`` generated from its name, because where the data model is silent SPEC.md section 8 has the generator
# follow a baseline inventory that records names, dimensions and dtypes and no per-variable units.
def _with_attrs(ds: xr.Dataset, table: dict[str, dict[str, str]]) -> xr.Dataset:
    for name in ds.data_vars:
        ds[name].attrs = dict(table.get(str(name), {"long_name": _label(str(name))}))
    return ds


def _provenance(ds: xr.Dataset) -> xr.Dataset:
    """Per-event provenance string coordinates, as the per-scenario summary stores carry them."""
    return ds.assign_coords(
        hhemt_producing_sha=("event_iloc", np.array([PRODUCING_SHA])),
        hhemt_producing_version=("event_iloc", np.array([PRODUCING_VERSION])),
    )


def _coords_xy(dims: Dimensions) -> tuple[np.ndarray, np.ndarray]:
    x = 5.0 + 10.0 * np.arange(dims.cols, dtype=np.float64)
    y = 10.0 * (dims.rows - np.arange(dims.rows, dtype=np.float64)) - 5.0
    return x, y


def _timestep_attrs(ds: xr.Dataset) -> xr.Dataset:
    """The unit of the `timestep` coordinate: minutes elapsed from the simulation start (not `minutes`: that spelling
    decodes to timedelta64 at the oldest xarray hhemt admits)."""
    ds["timestep"].attrs = {"long_name": "Model timestep (minutes since simulation start)", "units": "min", "axis": "T"}
    return ds


def _xy_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds["x"].attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}
    ds["y"].attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}
    return ds


# --------------------------------------------------------------------------------------
# TRITON stores
# --------------------------------------------------------------------------------------


def triton_tseries(dims: Dimensions, *, scale: float) -> xr.Dataset:
    steps = range(1, dims.n_steps + 1)
    x, y = _coords_xy(dims)
    tmin = np.array([k * dims.report_step_s / 60.0 for k in steps])

    def stack(prefix: str) -> np.ndarray:
        return np.stack([_triton_field(prefix, k, dims, scale) for k in steps]).astype("float32")

    ds = xr.Dataset(
        {
            "max_flood_depth": (("timestep", "y", "x"), stack("MH")),
            "flood_depth": (("timestep", "y", "x"), stack("H")),
            "unit_discharge_x": (("timestep", "y", "x"), stack("QX")),
            "unit_discharge_y": (("timestep", "y", "x"), stack("QY")),
        },
        coords={"timestep": tmin, "y": y, "x": x},
        attrs={"sim_date": "", "output_creation_date": CREATED},
    )
    # A reduction method describes a summary, not a series that keeps its time axis: the timeseries
    # variables carry the summary's attributes minus `cell_methods`.
    series_attrs = {k: {a: v for a, v in e.items() if a != "cell_methods"} for k, e in _TRITON_ATTRS.items()}
    return _timestep_attrs(_xy_attrs(_with_attrs(ds, series_attrs)))


# The wet-depth threshold below which the flood velocity is undefined (SPEC.md section 8): the value the data model
# documents for ``flood_velocity_min_depth_m``.
FLOOD_VELOCITY_MIN_DEPTH_M = 0.001


def _summary_over_steps(dims: Dimensions, scale: float) -> dict[str, np.ndarray]:
    """The per-cell reductions over the steps that the TRITON summary stores, derived with this package's own code.

    Written from the data model's definitions (``docs/reference/output-data-model.md``, "2D surface results"),
    not from hhemt: (1) the unit-discharge magnitude |q| = sqrt(qx^2 + qy^2) with its maximum over steps, the step
    of that maximum and the components there; (2) the step at which the printed depth is largest; (3) the flood
    velocity |q| / h with q and h of the SAME step, defined only where h exceeds the threshold, with its maximum,
    step and components q / h there; NaN where never defined. A strict greater-than keeps the FIRST maximum, as the
    reduction it documents does. sqrt of the sum of squares, not np.hypot, for host independence."""
    shape = (dims.rows, dims.cols)
    out = {
        "q_best": np.full(shape, -np.inf),
        "q_step": np.full(shape, np.nan),
        "q_x": np.full(shape, np.nan),
        "q_y": np.full(shape, np.nan),
        "h_best": np.full(shape, -np.inf),
        "h_step": np.full(shape, np.nan),
        "v_best": np.full(shape, -np.inf),
        "v_step": np.full(shape, np.nan),
        "v_x": np.full(shape, np.nan),
        "v_y": np.full(shape, np.nan),
    }
    for k in range(1, dims.n_steps + 1):
        h = _triton_field("H", k, dims, scale)
        qx = _triton_field("QX", k, dims, scale)
        qy = _triton_field("QY", k, dims, scale)
        mag = np.sqrt(qx * qx + qy * qy)
        better = mag > out["q_best"]
        out["q_best"] = np.where(better, mag, out["q_best"])
        out["q_step"] = np.where(better, float(k), out["q_step"])
        out["q_x"] = np.where(better, qx, out["q_x"])
        out["q_y"] = np.where(better, qy, out["q_y"])
        deeper = h > out["h_best"]
        out["h_best"] = np.where(deeper, h, out["h_best"])
        out["h_step"] = np.where(deeper, float(k), out["h_step"])
        wet = h > FLOOD_VELOCITY_MIN_DEPTH_M
        h_safe = np.where(wet, h, 1.0)
        v = np.where(wet, mag / h_safe, np.nan)
        faster = wet & (v > out["v_best"])
        out["v_best"] = np.where(faster, v, out["v_best"])
        out["v_step"] = np.where(faster, float(k), out["v_step"])
        out["v_x"] = np.where(faster, qx / h_safe, out["v_x"])
        out["v_y"] = np.where(faster, qy / h_safe, out["v_y"])
    out["v_best"] = np.where(np.isinf(out["v_best"]), np.nan, out["v_best"])
    return out


def triton_summary(dims: Dimensions, *, scale: float) -> xr.Dataset:
    last = dims.n_steps
    x, y = _coords_xy(dims)
    h_last = _triton_field("H", last, dims, scale)
    mh = _triton_field("MH", last, dims, scale)
    # sqrt of the sum of squares, not np.hypot: hypot's last bit depends on the host libm, sqrt is correctly rounded,
    # so the stored values are the same on every host.
    red = _summary_over_steps(dims, scale)
    minutes = dims.report_step_s / 60.0
    cell = ("event_iloc", "y", "x")
    ds = xr.Dataset(
        {
            "time_of_max_unit_discharge": (cell, (red["q_step"] * minutes)[None]),
            "unit_discharge_x_at_time_of_max_unit_discharge": (cell, red["q_x"][None].astype("float64")),
            "unit_discharge_y_at_time_of_max_unit_discharge": (cell, red["q_y"][None].astype("float64")),
            "time_of_max_flood_depth": (cell, (red["h_step"] * minutes)[None]),
            "max_unit_discharge": (cell, red["q_best"][None].astype("float64")),
            "max_flood_velocity": (cell, red["v_best"][None]),
            "time_of_max_flood_velocity": (cell, (red["v_step"] * minutes)[None]),
            "x_flood_velocity_at_time_of_max_flood_velocity": (cell, red["v_x"][None]),
            "y_flood_velocity_at_time_of_max_flood_velocity": (cell, red["v_y"][None]),
            "final_surface_flood_volume": (("event_iloc",), np.array([h_last.sum() * 100.0], dtype="float32")),
            "max_flood_depth": (cell, mh[None].astype("float32")),
            "flood_depth_last_tstep": (cell, h_last[None].astype("float32")),
        },
        coords={"event_iloc": np.array([0], dtype="int64"), "y": y, "x": x},
        attrs={"sim_date": "", "output_creation_date": CREATED},
    )
    return _xy_attrs(_with_attrs(_provenance(ds), _TRITON_ATTRS))


# --------------------------------------------------------------------------------------
# Performance stores (nine timing columns; SPEC.md 7.1 and the data model's performance caveat)
# --------------------------------------------------------------------------------------


# Cumulative over the run, as the raw performance files are (SPEC.md section 7.1), so the columns that scale with the
# elapsed time ``t`` grow with ``step``.
def _perf_values(step: int, dims: Dimensions, coupled: bool) -> dict[str, float]:
    t = dims.report_step_s * step * 1.0e-3
    compute, io, other = 0.70 * t, 0.25 * t, 0.05 * t
    swmm = 0.20 * t if coupled else 0.0
    simulation = compute + io + swmm + other
    init = 0.01
    return {
        "Compute": compute,
        "MPI": 0.0,
        "IO": io,
        "Resize": 0.0,
        "SWMM": swmm,
        "Other": other,
        "Simulation": simulation,
        "Init": init,
        "Total": simulation + init,
    }


_PERF_NOTES = (
    "Per-column slowest-rank cumulative cost. 'Total' / 'Simulation' / 'Init' carry wallclock semantics; the "
    "category columns are slowest-rank cost."
)


def perf_tseries(dims: Dimensions, *, coupled: bool) -> xr.Dataset:
    steps = range(1, dims.n_steps + 1)
    tmin = np.array([k * dims.report_step_s / 60.0 for k in steps])
    data = {
        col: (
            ("event_iloc", "timestep", "Rank"),
            np.array([[[_perf_values(k, dims, coupled)[col]] for k in steps]]),
        )
        for col in PERF_COLUMNS
    }
    ds = xr.Dataset(
        data,
        coords={"event_iloc": np.array([0], dtype="int64"), "timestep": tmin, "Rank": np.array([0], dtype="int64")},
        attrs={"sim_date": "", "output_creation_date": CREATED},
    )
    return _timestep_attrs(_with_attrs(_provenance(ds), _PERF_ATTRS))


def perf_summary(dims: Dimensions, *, coupled: bool) -> xr.Dataset:
    vals = _perf_values(dims.n_steps, dims, coupled)
    ds = xr.Dataset(
        {col: (("event_iloc",), np.array([vals[col]])) for col in PERF_COLUMNS},
        coords={"event_iloc": np.array([0], dtype="int64")},
        attrs={"sim_date": "", "output_creation_date": CREATED, "units": "seconds", "notes": _PERF_NOTES},
    )
    return _with_attrs(_provenance(ds), _PERF_ATTRS)


# --------------------------------------------------------------------------------------
# SWMM stores
# --------------------------------------------------------------------------------------

_SWMM_ATTRS_DS = {
    "analysis_end_datetime": ANALYSIS_END,
    "flow_units": "cms",
    "runoff_continuity_error": float("nan"),
    "runoff_continuity_error_units": "percent",
    "flow_continuity_error": 0.0,
    "flow_continuity_error_units": "percent",
    "system_flooding": 0.0,
    "system_flooding_units": "10^6 L",
    "sim_date": "",
    "output_creation_date": CREATED,
}


def _times(dims: Dimensions) -> np.ndarray:
    start = np.datetime64("2000-01-01T00:00:00", "ns")
    return np.array([start + np.timedelta64(k * dims.report_step_s, "s") for k in range(1, dims.n_steps + 1)])


def _ramp(dims: Dimensions, scale: float) -> np.ndarray:
    return scale * np.arange(1, dims.n_steps + 1, dtype=np.float64) / dims.n_steps


_NODE_SCALARS = (
    "flow_balance_error",
    "flow_balance_error_volume",
    "hours_flooded",
    "lateral_inflow_vol",
    "max_flood_rate",
    "max_lateral_inflow",
    "max_ponded_depth",
    "max_total_inflow",
    "time_of_max_flood",
    "time_of_max_total_inflow",
    "tot_flooded_vol",
    "total_inflow_vol",
)
_LINK_SCALARS = (
    "max_flow",
    "max_over_full_depth",
    "max_over_full_flow",
    "max_velocity",
    "time_of_max_flow",
)


def _node_series(network: Network, dims: Dimensions, scale: float) -> dict[str, np.ndarray]:
    ramp = _ramp(dims, scale)
    depth = np.zeros((len(network.nodes), dims.n_steps))
    head = np.zeros_like(depth)
    for j, n in enumerate(network.nodes):
        if n.type_code != NODE_OUTFALL:
            depth[j] = 0.02 * ramp * (1 + 0.05 * j)
        head[j] = n.invert_m + depth[j]
    inflow = np.tile(0.001 * ramp, (len(network.nodes), 1))
    return {"depth": depth, "head": head, "inflow_flow": inflow, "flooding": np.zeros_like(depth)}


def _node_scalar_values(series: dict[str, np.ndarray], dims: Dimensions) -> dict[str, np.ndarray]:
    n = series["depth"].shape[0]
    t_last = dims.n_steps * dims.report_step_s / 60.0
    zeros = np.zeros(n)
    # SWMM prints a node's flow balance error as a percent, or in litres followed by `ltr` where the node's
    # outflow volume is under one cubic foot, so each node has exactly one of the two: even-numbered nodes a
    # percent, odd-numbered nodes litres, the other variable NaN (the data model's definition of the pair).
    percent_row = np.arange(n) % 2 == 0
    return {
        "flow_balance_error": np.where(percent_row, 0.0, np.nan),
        "flow_balance_error_volume": np.where(percent_row, np.nan, 0.0),
        "hours_flooded": zeros,
        "lateral_inflow_vol": zeros,
        "max_flood_rate": series["inflow_flow"].max(axis=1),
        "max_lateral_inflow": np.full(n, 0.0005),
        "max_ponded_depth": zeros,
        "max_total_inflow": series["inflow_flow"].max(axis=1),
        "time_of_max_flood": zeros,
        "time_of_max_total_inflow": np.full(n, t_last),
        "tot_flooded_vol": zeros,
        "total_inflow_vol": zeros,
    }


def swmm_node_tseries(network: Network, dims: Dimensions, *, scale: float = 1.0) -> xr.Dataset:
    ser = _node_series(network, dims, scale)
    ids = np.array([n.id for n in network.nodes])
    data = {k: (("node_id", "date_time"), v) for k, v in ser.items()}
    for k, v in _node_scalar_values(ser, dims).items():
        data[k] = (("node_id",), v)
    data["type"] = (("node_id",), np.array([n.type_name for n in network.nodes]))
    ds = xr.Dataset(data, coords={"node_id": ids, "date_time": _times(dims)}, attrs=dict(_SWMM_ATTRS_DS))
    return _with_attrs(ds, _SWMM_ATTRS)


def swmm_node_summary(network: Network, dims: Dimensions, *, scale: float = 1.0) -> xr.Dataset:
    ser = _node_series(network, dims, scale)
    ids = np.array([n.id for n in network.nodes])
    data: dict[str, tuple] = {}
    for name in ("depth", "flooding", "head", "inflow_flow"):
        data[f"{name}_last"] = (("event_iloc", "node_id"), ser[name][:, -1][None])
        data[f"{name}_max"] = (("event_iloc", "node_id"), ser[name].max(axis=1)[None])
    for k, v in _node_scalar_values(ser, dims).items():
        data[k] = (("event_iloc", "node_id"), v[None])
    data["type"] = (("event_iloc", "node_id"), np.array([[n.type_name for n in network.nodes]]))
    ds = xr.Dataset(
        data,
        coords={"event_iloc": np.array([0], dtype="int64"), "node_id": ids},
        attrs=dict(_SWMM_ATTRS_DS),
    )
    return _with_attrs(_provenance(ds), _SWMM_ATTRS)


def _link_series(network: Network, dims: Dimensions, scale: float) -> dict[str, np.ndarray]:
    ramp = _ramp(dims, scale)
    n = len(network.links)
    flow = np.stack([0.001 * ramp * (1 + 0.05 * j) for j in range(n)])
    return {
        "flow": flow,
        "link_depth": np.tile(0.01 * ramp, (n, 1)),
        "velocity": np.tile(0.1 * ramp, (n, 1)),
        "cap": np.tile(0.02 * ramp, (n, 1)),
    }


def _link_scalar_values(ser: dict[str, np.ndarray], dims: Dimensions) -> dict[str, np.ndarray]:
    n = ser["flow"].shape[0]
    return {
        "max_flow": ser["flow"].max(axis=1),
        "max_over_full_depth": ser["link_depth"].max(axis=1) / 0.6,
        "max_over_full_flow": ser["cap"].max(axis=1),
        "max_velocity": ser["velocity"].max(axis=1),
        "time_of_max_flow": np.full(n, dims.n_steps * dims.report_step_s / 60.0),
    }


def swmm_link_tseries(network: Network, dims: Dimensions, *, coupled: bool, scale: float = 1.0) -> xr.Dataset:
    ser = _link_series(network, dims, scale)
    cap = "capacity_setting" if coupled else "capacity"
    ids = np.array([lk.id for lk in network.links])
    data = {
        cap: (("link_id", "date_time"), ser["cap"]),
        "flow": (("link_id", "date_time"), ser["flow"]),
        "link_depth": (("link_id", "date_time"), ser["link_depth"]),
        "velocity": (("link_id", "date_time"), ser["velocity"]),
    }
    for k, v in _link_scalar_values(ser, dims).items():
        data[k] = (("link_id",), v)
    data["type"] = (("link_id",), np.array(["CONDUIT"] * len(ids)))
    ds = xr.Dataset(data, coords={"link_id": ids, "date_time": _times(dims)}, attrs=dict(_SWMM_ATTRS_DS))
    return _with_attrs(ds, _SWMM_ATTRS)


def swmm_link_summary(network: Network, dims: Dimensions, *, coupled: bool, scale: float = 1.0) -> xr.Dataset:
    ser = _link_series(network, dims, scale)
    cap = "capacity_setting" if coupled else "capacity"
    ids = np.array([lk.id for lk in network.links])
    row = ("event_iloc", "link_id")
    data = {
        f"{cap}_last": (row, ser["cap"][:, -1][None]),
        f"{cap}_max": (row, ser["cap"].max(axis=1)[None]),
        "flow_last": (row, ser["flow"][:, -1][None]),
        "flow_max": (row, ser["flow"].max(axis=1)[None]),
        "link_depth_last": (row, ser["link_depth"][:, -1][None]),
        "link_depth_max": (row, ser["link_depth"].max(axis=1)[None]),
        "velocity_last": (row, ser["velocity"][:, -1][None]),
        "velocity_max": (row, ser["velocity"].max(axis=1)[None]),
    }
    for k, v in _link_scalar_values(ser, dims).items():
        data[k] = (row, v[None])
    data["type"] = (row, np.array([["CONDUIT"] * len(ids)]))
    ds = xr.Dataset(
        data,
        coords={"event_iloc": np.array([0], dtype="int64"), "link_id": ids},
        attrs=dict(_SWMM_ATTRS_DS),
    )
    return _with_attrs(_provenance(ds), _SWMM_ATTRS)


# --------------------------------------------------------------------------------------
# Per-event processed directory
# --------------------------------------------------------------------------------------


def event_stores(arm: str, network: Network, dims: Dimensions) -> dict[str, xr.Dataset]:
    """The processed stores of one arm, keyed by store file name without extension.

    Names follow SPEC.md 4.3: ``TRITONSWMM_perf_tseries``, ``TRITONSWMM_TRITON_tseries``,
    ``TRITONSWMM_SWMM_node_tseries`` ... for the coupled arm; ``TRITON_only_*`` and ``SWMM_only_*``
    for the single-model arms."""
    p = ARM_PREFIX[arm]
    out: dict[str, xr.Dataset] = {}
    if arm in ("tritonswmm", "triton"):
        coupled = arm == "tritonswmm"
        scale = 1.15 if coupled else 1.0
        tname = f"{p}_TRITON" if coupled else p
        out[f"{p}_perf_tseries"] = perf_tseries(dims, coupled=coupled)
        out[f"{tname}_tseries"] = triton_tseries(dims, scale=scale)
        out[f"{p}_perf_summary"] = perf_summary(dims, coupled=coupled)
        out[f"{tname}_summary"] = triton_summary(dims, scale=scale)
    if arm in ("tritonswmm", "swmm"):
        coupled = arm == "tritonswmm"
        scale = 1.15 if coupled else 1.0
        sname = f"{p}_SWMM" if coupled else p
        out[f"{sname}_node_tseries"] = swmm_node_tseries(network, dims, scale=scale)
        out[f"{sname}_link_tseries"] = swmm_link_tseries(network, dims, coupled=coupled, scale=scale)
        out[f"{sname}_node_summary"] = swmm_node_summary(network, dims, scale=scale)
        out[f"{sname}_link_summary"] = swmm_link_summary(network, dims, coupled=coupled, scale=scale)
    return out


def write_zarr(ds: xr.Dataset, path: Path) -> None:
    """Write one store with a consolidated-metadata index at its root (layout 27, SPEC.md section 8)."""
    with warnings.catch_warnings():
        # Hides zarr's UnstableSpecificationWarning, which writing a fixed-width string array (the provenance
        # coordinates, the ids and ``type``) raises because that dtype has no Zarr V3 specification. Not established:
        # that this is the only warning the filter hides.
        warnings.simplefilter("ignore")
        ds.to_zarr(path, mode="w", consolidated=True)


def write_event_processed(
    event_dir: Path,
    *,
    event_iloc: int,
    arms: tuple[str, ...],
    network: Network | None = None,
    dims: Dimensions | None = None,
) -> dict[str, Path]:
    """Write every processed store of one event; returns store file name -> path.

    ``event_iloc`` is the event's row position; every store that has an ``event_iloc`` coordinate (the summaries and
    ``perf_tseries``) carries it as the
    ``event_iloc`` coordinate value."""
    net = network or default_network()
    d = dims or Dimensions()
    proc = Path(event_dir) / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for arm in arms:
        for name, ds in event_stores(arm, net, d).items():
            if "event_iloc" in ds.coords:
                ds = ds.assign_coords(event_iloc=np.array([event_iloc], dtype="int64"))
            path = proc / f"{name}.zarr"
            write_zarr(ds, path)
            written[name] = path
    return written


# --------------------------------------------------------------------------------------
# Consolidated tiers (data model: per-analysis and per-experiment tiers)
# --------------------------------------------------------------------------------------

# Per arm, the ``(leaf name, source summary store)`` pairs: ``analysis_leaves`` joins each into the group
# ``/{ARM_GROUP}/{leaf}/summary``, and together they are the group set of the baseline inventory in SPEC.md section 8.
_LEAVES = {
    "tritonswmm": (
        ("triton", "TRITONSWMM_TRITON_summary"),
        ("swmm_node", "TRITONSWMM_SWMM_node_summary"),
        ("swmm_link", "TRITONSWMM_SWMM_link_summary"),
        ("performance", "TRITONSWMM_perf_summary"),
    ),
    "triton": (("triton", "TRITON_only_summary"), ("performance", "TRITON_only_perf_summary")),
    "swmm": (("swmm_node", "SWMM_only_node_summary"), ("swmm_link", "SWMM_only_link_summary")),
}

_RO_CRATE = json.dumps({"@context": "https://w3id.org/ro/crate/1.2/context", "@graph": []}, indent=2)


def _read_summary(processed: Path, store: str) -> xr.Dataset:
    with warnings.catch_warnings():
        # Not established what this hides: reading a summary store raised no warning in a probe, while writing one
        # raises the string-dtype warning that ``write_zarr`` hides.
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(processed / f"{store}.zarr", consolidated=False).load()
    ds.close()
    return ds


def analysis_leaves(
    event_dirs: list[tuple[str, Path]], arms: tuple[str, ...], prefix: str = ""
) -> dict[str, xr.Dataset]:
    """Per-analysis tree leaves: each arm's summaries joined over events along ``event_iloc``.

    ``event_dirs`` is a list of ``(event_id, event_dir)`` in event order (``generate_multisim`` and
    ``generate_sensitivity`` in ``__main__.py`` build it in the loop over ``enumerate(ids)`` that passes each index as
    that event's ``event_iloc``); ``prefix`` is prepended to
    every group path (``"/member_0"`` for a member's node of the experiment tree)."""
    leaves: dict[str, xr.Dataset] = {}
    for arm in arms:
        for leaf, store in _LEAVES[arm]:
            parts = []
            for _event_id, edir in event_dirs:
                parts.append(_read_summary(edir / "processed", store))
            joined = xr.concat(parts, dim="event_iloc")
            ids = np.array([eid for eid, _ in event_dirs])
            joined = joined.assign_coords(event_id=("event_iloc", ids))
            leaves[f"{prefix}/{ARM_GROUP[arm]}/{leaf}/summary"] = joined
    return leaves


def write_analysis_datatree(
    path: Path,
    event_dirs: list[tuple[str, Path]],
    arms: tuple[str, ...],
    *,
    analysis_id: str,
    fingerprint: str,
) -> None:
    """Write ``analysis_datatree.zarr`` (per-analysis tier).

    ``analysis_id`` and ``SYSTEM_ID`` are the root attributes the data model's "Global attributes" require.
    ``fingerprint`` is stored as ``consolidation_inputs_fingerprint``; ``__main__.py`` passes a fixed placeholder,
    because this package cannot compute hhemt's value."""
    tree = xr.DataTree.from_dict(analysis_leaves(event_dirs, arms))
    tree.attrs = {
        "analysis_id": analysis_id,
        "system_id": SYSTEM_ID,
        "output_creation_date": CREATED,
        "consolidation_version": "1",
        "consolidation_inputs_fingerprint": fingerprint,
        "Conventions": "CF-1.13",
        "ro_crate_metadata": _RO_CRATE,
        "hhemt_producing_sha": PRODUCING_SHA,
        "hhemt_producing_version": PRODUCING_VERSION,
    }
    with warnings.catch_warnings():
        # The same string-dtype warning as in ``write_zarr``, here raised by writing a tree.
        warnings.simplefilter("ignore")
        tree.to_zarr(path, mode="w", consolidated=True)


def write_experiment_datatree(
    path: Path,
    members: dict[int, dict],
    member_events: dict[int, list[tuple[str, Path]]],
    arms: tuple[str, ...],
    *,
    analysis_id: str,
) -> None:
    """Write ``experiment_datatree.zarr`` (per-experiment tier): a ``parameters`` dataset with one row for each key
    of ``members`` and one ``member_{id}`` node for each of those same keys, so ``member_events`` must hold an entry
    for every one (a missing key raises ``KeyError``).

    ``tests/goldens/SPEC.md`` section 8 describes the tier as a ``parameters`` dataset "describing every defined
    member" with one node "per completed member"; this writer does not tell the two sets apart, and the sensitivity
    golden has four members (SPEC.md section 1)."""
    ids = sorted(members)
    params = xr.Dataset(
        {
            "run_mode": (("member_id",), np.array([members[m]["run_mode"] for m in ids])),
            "n_mpi_procs": (("member_id",), np.array([members[m]["n_mpi_procs"] for m in ids], dtype="int64")),
            "n_omp_threads": (("member_id",), np.array([members[m]["n_omp_threads"] for m in ids], dtype="int64")),
            "n_gpus": (("member_id",), np.array([members[m]["n_gpus"] for m in ids], dtype="int64")),
            "n_nodes": (("member_id",), np.array([members[m]["n_nodes"] for m in ids], dtype="int64")),
        },
        coords={"member_id": np.array(ids, dtype="int64")},
    )
    for _name, _attrs in _PARAMETER_ATTRS.items():
        params[_name].attrs = dict(_attrs)
    d: dict[str, xr.Dataset] = {"/parameters": params}
    for m in ids:
        d[f"/member_{m}"] = xr.Dataset(attrs={k: members[m][k] for k in members[m]})
        d.update(analysis_leaves(member_events[m], arms, prefix=f"/member_{m}"))
    tree = xr.DataTree.from_dict(d)
    tree.attrs = {
        "Conventions": "CF-1.13",
        "title": "TRITON-SWMM sensitivity analysis results",
        "analysis_id": analysis_id,
        "system_id": SYSTEM_ID,
        "output_creation_date": CREATED,
        "hhemt_producing_sha": PRODUCING_SHA,
        "hhemt_producing_version": PRODUCING_VERSION,
        "ro_crate_metadata": _RO_CRATE,
    }
    with warnings.catch_warnings():
        # The same string-dtype warning as in ``write_zarr``, here raised by writing a tree.
        warnings.simplefilter("ignore")
        tree.to_zarr(path, mode="w", consolidated=True)
