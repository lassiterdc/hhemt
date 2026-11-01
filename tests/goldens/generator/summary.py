"""Summary-layer writer for the golden analysis trees.

A golden is one committed zip file holding a finished analysis tree and the files hhemt needs to construct an
analysis object from it (``tests/goldens/SPEC.md`` section 1).

Writes the per-scenario processed stores under ``sims/{event_id}/processed/`` and the
consolidated tiers (``analysis_datatree.zarr``, ``experiment_datatree.zarr``) at the reduced
dimensions of ``tests/goldens/SPEC.md`` section 2.

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
    "max_wlevel_m": {
        "units": "m",
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Maximum water level over simulation",
        "cell_methods": "timestep_min: maximum",
    },
    "wlevel_m": {
        "units": "m",
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level timeseries",
    },
    "wlevel_m_last_tstep": {
        "units": "m",
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Water level at final timestep",
        "cell_methods": "timestep_min: point",
    },
    "max_velocity_mps": {
        "units": "m s-1",
        "standard_name": "sea_water_speed",
        "long_name": "Maximum flood velocity",
        "cell_methods": "timestep_min: maximum",
    },
    "velocity_x_mps": {"units": "m s-1", "standard_name": "sea_water_x_velocity", "long_name": "Flood velocity x"},
    "velocity_y_mps": {"units": "m s-1", "standard_name": "sea_water_y_velocity", "long_name": "Flood velocity y"},
    "time_of_max_velocity_min": {"units": "minutes", "long_name": "Time of maximum velocity"},
    "final_surface_flood_volume_m3": {
        "units": "m3",
        "long_name": "Final surface flood volume",
        "cell_methods": "area: sum",
    },
}
# Documented SWMM network variables (output-data-model.md, "Network results (SWMM)"), plus max_velocity_mps, which
# that document's warning "max_velocity_mps means two different things" defines on a link node as maximum conduit
# velocity.
_SWMM_ATTRS = {
    "total_inflow_vol_10e6_ltr": {"units": "10^6 L", "long_name": "Total inflow volume"},
    "max_flow_cms": {"units": "m3 s-1", "long_name": "Maximum flow rate"},
    "max_over_full_flow": {"units": "1", "long_name": "Maximum flow as a fraction of full-flow capacity"},
    "max_over_full_depth": {"units": "1", "long_name": "Maximum depth as a fraction of full depth"},
    "max_velocity_mps": {"units": "m s-1", "standard_name": "sea_water_speed", "long_name": "Maximum conduit velocity"},
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
            "max_wlevel_m": (("timestep_min", "y", "x"), stack("MH")),
            "wlevel_m": (("timestep_min", "y", "x"), stack("H")),
            "velocity_x_mps": (("timestep_min", "y", "x"), stack("QX")),
            "velocity_y_mps": (("timestep_min", "y", "x"), stack("QY")),
        },
        coords={"timestep_min": tmin, "y": y, "x": x},
        attrs={"sim_date": "", "output_creation_date": CREATED},
    )
    return _xy_attrs(_with_attrs(ds, _TRITON_ATTRS))


def triton_summary(dims: Dimensions, *, scale: float) -> xr.Dataset:
    last = dims.n_steps
    x, y = _coords_xy(dims)
    h_last = _triton_field("H", last, dims, scale)
    mh = _triton_field("MH", last, dims, scale)
    qx = _triton_field("QX", last, dims, scale)
    qy = _triton_field("QY", last, dims, scale)
    # sqrt of the sum of squares, not np.hypot: hypot's last bit depends on the host libm, sqrt is correctly rounded,
    # so the stored values are the same on every host.
    vmax = np.sqrt(qx * qx + qy * qy)
    t_last = last * dims.report_step_s / 60.0
    cell = ("event_iloc", "y", "x")
    ds = xr.Dataset(
        {
            "time_of_max_velocity_min": (cell, np.full((1, dims.rows, dims.cols), t_last)),
            "velocity_x_mps_at_time_of_max_velocity": (cell, qx[None].astype("float64")),
            "velocity_y_mps_at_time_of_max_velocity": (cell, qy[None].astype("float64")),
            "time_of_max_wlevel_min": (cell, np.full((1, dims.rows, dims.cols), t_last)),
            "max_velocity_mps": (cell, vmax[None].astype("float64")),
            "final_surface_flood_volume_m3": (("event_iloc",), np.array([h_last.sum() * 100.0], dtype="float32")),
            "max_wlevel_m": (cell, mh[None].astype("float32")),
            "wlevel_m_last_tstep": (cell, h_last[None].astype("float32")),
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
            ("event_iloc", "timestep_min", "Rank"),
            np.array([[[_perf_values(k, dims, coupled)[col]] for k in steps]]),
        )
        for col in PERF_COLUMNS
    }
    ds = xr.Dataset(
        data,
        coords={"event_iloc": np.array([0], dtype="int64"), "timestep_min": tmin, "Rank": np.array([0], dtype="int64")},
        attrs={"sim_date": "", "output_creation_date": CREATED},
    )
    return _with_attrs(_provenance(ds), {})


def perf_summary(dims: Dimensions, *, coupled: bool) -> xr.Dataset:
    vals = _perf_values(dims.n_steps, dims, coupled)
    ds = xr.Dataset(
        {col: (("event_iloc",), np.array([vals[col]])) for col in PERF_COLUMNS},
        coords={"event_iloc": np.array([0], dtype="int64")},
        attrs={"sim_date": "", "output_creation_date": CREATED, "units": "seconds", "notes": _PERF_NOTES},
    )
    return _with_attrs(_provenance(ds), {})


# --------------------------------------------------------------------------------------
# SWMM stores
# --------------------------------------------------------------------------------------

_SWMM_ATTRS_DS = {
    "analysis_end_datetime": ANALYSIS_END,
    "flow_units": "cms",
    "runoff_continuity_error_perc": float("nan"),
    "flow_continuity_error_perc": 0.0,
    "system_flooding_10e6_ltr": 0.0,
    "sim_date": "",
    "output_creation_date": CREATED,
}


def _times(dims: Dimensions) -> np.ndarray:
    start = np.datetime64("2000-01-01T00:00:00", "ns")
    return np.array([start + np.timedelta64(k * dims.report_step_s, "s") for k in range(1, dims.n_steps + 1)])


def _ramp(dims: Dimensions, scale: float) -> np.ndarray:
    return scale * np.arange(1, dims.n_steps + 1, dtype=np.float64) / dims.n_steps


_NODE_SCALARS = (
    "flow_balance_error_percent",
    "hours_flooded",
    "lateral_inflow_vol_10e6_ltr",
    "max_flow_cms",
    "max_lateral_inflow_cms",
    "max_ponded_depth_m",
    "max_total_inflow_cms",
    "time_of_max_flood_min",
    "time_of_max_flow_min",
    "tot_flooded_vol_10e6_ltr",
    "total_inflow_vol_10e6_ltr",
)
_LINK_SCALARS = (
    "max_flow_cms",
    "max_over_full_depth",
    "max_over_full_flow",
    "max_velocity_mps",
    "time_of_max_flow_min",
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
    return {"depth_m": depth, "head_m": head, "inflow_flow_cms": inflow, "flooding_cms": np.zeros_like(depth)}


def _node_scalar_values(series: dict[str, np.ndarray], dims: Dimensions) -> dict[str, np.ndarray]:
    n = series["depth_m"].shape[0]
    t_last = dims.n_steps * dims.report_step_s / 60.0
    zeros = np.zeros(n)
    return {
        "flow_balance_error_percent": zeros,
        "hours_flooded": zeros,
        "lateral_inflow_vol_10e6_ltr": zeros,
        "max_flow_cms": series["inflow_flow_cms"].max(axis=1),
        "max_lateral_inflow_cms": np.full(n, 0.0005),
        "max_ponded_depth_m": zeros,
        "max_total_inflow_cms": series["inflow_flow_cms"].max(axis=1),
        "time_of_max_flood_min": zeros,
        "time_of_max_flow_min": np.full(n, t_last),
        "tot_flooded_vol_10e6_ltr": zeros,
        "total_inflow_vol_10e6_ltr": zeros,
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
    for name in ("depth_m", "flooding_cms", "head_m", "inflow_flow_cms"):
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
        "flow_cms": flow,
        "link_depth_m": np.tile(0.01 * ramp, (n, 1)),
        "velocity_mps": np.tile(0.1 * ramp, (n, 1)),
        "cap": np.tile(0.02 * ramp, (n, 1)),
    }


def _link_scalar_values(ser: dict[str, np.ndarray], dims: Dimensions) -> dict[str, np.ndarray]:
    n = ser["flow_cms"].shape[0]
    return {
        "max_flow_cms": ser["flow_cms"].max(axis=1),
        "max_over_full_depth": ser["link_depth_m"].max(axis=1) / 0.6,
        "max_over_full_flow": ser["cap"].max(axis=1),
        "max_velocity_mps": ser["velocity_mps"].max(axis=1),
        "time_of_max_flow_min": np.full(n, dims.n_steps * dims.report_step_s / 60.0),
    }


def swmm_link_tseries(network: Network, dims: Dimensions, *, coupled: bool, scale: float = 1.0) -> xr.Dataset:
    ser = _link_series(network, dims, scale)
    cap = "capacity_setting" if coupled else "capacity"
    ids = np.array([lk.id for lk in network.links])
    data = {
        cap: (("link_id", "date_time"), ser["cap"]),
        "flow_cms": (("link_id", "date_time"), ser["flow_cms"]),
        "link_depth_m": (("link_id", "date_time"), ser["link_depth_m"]),
        "velocity_mps": (("link_id", "date_time"), ser["velocity_mps"]),
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
        "flow_cms_last": (row, ser["flow_cms"][:, -1][None]),
        "flow_cms_max": (row, ser["flow_cms"].max(axis=1)[None]),
        "link_depth_m_last": (row, ser["link_depth_m"][:, -1][None]),
        "link_depth_m_max": (row, ser["link_depth_m"].max(axis=1)[None]),
        "velocity_mps_last": (row, ser["velocity_mps"][:, -1][None]),
        "velocity_mps_max": (row, ser["velocity_mps"].max(axis=1)[None]),
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
    """Write one store without consolidated metadata (the data model requires ``consolidated=False``)."""
    with warnings.catch_warnings():
        # Hides zarr's UnstableSpecificationWarning, which writing a fixed-width string array (the provenance
        # coordinates, the ids and ``type``) raises because that dtype has no Zarr V3 specification. Not established:
        # that this is the only warning the filter hides.
        warnings.simplefilter("ignore")
        ds.to_zarr(path, mode="w", consolidated=False)


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
# ``/{ARM_GROUP}/{leaf}``, and together they are the group set of the baseline inventory in SPEC.md section 8.
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
            leaves[f"{prefix}/{ARM_GROUP[arm]}/{leaf}"] = joined
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
        tree.to_zarr(path, mode="w", consolidated=False)


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
        tree.to_zarr(path, mode="w", consolidated=False)
