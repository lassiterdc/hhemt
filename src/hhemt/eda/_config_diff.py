"""Config-diff-maps EDA figure (redesign of the cross-sim-identity plot).

Answers two questions for a compute-config sensitivity master:
  1. Which compute-configuration GROUPS produce byte-identical flood results?
  2. For each group that differs from the serial-CPU baseline, what is the SIGNED
     spatial difference (per DEM cell + per SWMM conduit) vs serial CPU?

Reads the consolidated ``sensitivity_datatree.zarr`` directly (per-cell
``max_wlevel_m`` (y,x) + per-conduit ``max_flow_cms`` (link_id)) plus one sub's
``hydraulics.inp`` for conduit geometry — all present in a render bundle, so the
figure re-renders locally via ``Bundle.eda(plots_only=True)`` with no HPC re-emit.

Vocabulary (locked, /design-figure iteration 1): three distinct quantities —
  * ``diff``         = signed ``group - serial``            -> maps
  * ``percent diff`` = signed ``100*(group-serial)/serial`` -> maps
  * ``absolute diff``= ``|group - serial|`` (max magnitude)  -> summary table only.

Config labels are DERIVED from each sub node's compute-config attrs
(``run_mode``/``n_gpus``/``n_mpi_procs``/``n_omp_threads``/``n_nodes``) — never the
``sa_id`` name string. Only one representative diff-set is rendered per byte-identical
group; each group's panels carry a caption naming every member config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

#: Diverging colorscale for signed diffs (iter-2 user feedback): RED = NEGATIVE
#: (lower than serial), white = 0, BLUE = POSITIVE. Plotly "RdBu" maps low->red,
#: high->blue, so with zmid=0 negative reads red and positive reads blue.
_DIVERGING = "RdBu"
#: Serial-CPU reference (absolute magnitude) palettes REUSED from the refined report
#: (config/report.py): depth_cmap="YlGnBu", peak_flow_cmap="Reds".
_REF_DEPTH = "YlGnBu"
_REF_FLOW = "Reds"

#: Diff/pct maps auto-scale to their ACTUAL signed range (iter-3: the iter-2 floor is
#: REVERTED — even micrometer-scale diffs are worth seeing, e.g. the MPI-vs-GPU
#: spatial-pattern difference). Symmetric about 0 (zmid=0); this epsilon only guards a
#: degenerate all-zero range from collapsing the diverging colorscale.
_RANGE_EPS = 1e-12

_PANEL_H_PX = 350
_TABLE_H_PX = 40


def _identity_labels(root: Path) -> dict[str, int] | None:
    """sa_id -> byte-identity group label, read from the flat-summary-derived
    partition persisted by cross_sim_identity at eda/eda_cross_sim_identity.zarr.

    This is the ONLY identity source (Gotcha 44 / the `eda bit identity check reads
    flat summaries not consolidated tree` stipulation): the consolidated tree is
    NEVER compared for identity. Returns None when the artifact is absent (a legacy
    bundle) -- the caller then renders an explicit "unknown (identity artifact
    absent)" state and MUST NOT fall back to a positional consolidated compare."""
    store = root / "eda" / "eda_cross_sim_identity.zarr"
    if not store.exists():
        return None
    ds = xr.open_zarr(store, consolidated=False)
    if "identity_group" not in ds:
        return None
    labels = {str(sa): int(v) for sa, v in zip(ds["sa_id"].values, ds["identity_group"].values, strict=False)}
    # The partition is persisted over the NON-reference sa_ids (matching the artifact's
    # other vars' coord, so identity_group's addition is purely additive); the reference's
    # own group rides in the `reference_group` attr. Fold it back in so the reference is
    # grouped with its byte-identical peers rather than rendering "unknown".
    ref = ds.attrs.get("reference_sa_id")
    ref_group = ds.attrs.get("reference_group")
    if ref is not None and ref_group is not None:
        labels[str(ref)] = int(ref_group)
    return labels


def _align_to(ref: xr.DataArray, da: xr.DataArray) -> np.ndarray:
    """Return da's values reindexed to ref's coords/dim-order (exact join), so a
    downstream positional subtraction (ref - da) compares matched cells. Falls back
    to da's own values when coords are incomparable (the identity column, sourced from
    the flat-summary partition artifact, already reports that case as not-identical).

    Identity fast-path: when the dim order and every index already match, the align is a
    no-op -- return the array without a second full materialization. _load_subs opens the
    tree with no chunks=, so each .values access materializes a full (y, x) numpy array
    (~118 MB on a 0.35 m Norfolk DEM); paying that twice per group per variable would
    roughly double the EDA render path's peak RSS for no numerical benefit."""
    if ref.dims == da.dims and all(
        ref.indexes[d].equals(da.indexes[d]) for d in ref.dims if d in ref.indexes and d in da.indexes
    ):
        return np.asarray(da.values)
    try:
        _, da_al = xr.align(ref, da, join="exact")
        return np.asarray(da_al.transpose(*ref.dims).values)
    except (ValueError, KeyError):
        return np.asarray(da.values)


def _within_family(g: dict, serial_grp: dict) -> bool:
    """True when group g shares the serial baseline's HARDWARE family (CPU vs GPU).

    A difference WITHIN the CPU family (serial vs an MPI/OpenMP/Hybrid decomposition) is
    expected floating-point non-associativity; a CROSS-family (GPU-vs-CPU) difference is
    disclosed with its bound instead. Keyed on run_mode: any 'gpu' member => GPU family."""

    def _family(run_modes) -> str:
        return "gpu" if any(str(rm) == "gpu" for rm in run_modes) else "cpu"

    return _family(g["run_modes"]) == _family(serial_grp["run_modes"])


def _identity_cell(identical, g: dict, serial_grp: dict, wad: float, fad: float) -> str:
    """R2's three-state 'identical to serial?' value. A bare 'no' reads as a defect;
    a real cross-decomposition divergence must be DISCLOSED with its bound, and an
    absent identity artifact must read 'unknown', never 'no'."""
    if identical is None:
        return "unknown (identity artifact absent)"
    if identical:
        return "identical"
    if _within_family(g, serial_grp):
        return "differs (within-family expected)"
    return f"differs (bounded, disclosed: max_abs={max(wad, fad):.3e})"


def _to_int(attrs: dict, key: str) -> int:
    try:
        return int(float(attrs.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _gpu_hardware(attrs: dict) -> str:
    """Hardware token derived from the ensemble partition (Gotcha 54: partition IS the
    hardware axis). ``'gpu-a100-80' -> 'a100-80'``, ``'gpu-a6000' -> 'a6000'``. Empty
    when no partition attr is present."""
    part = str(attrs.get("hpc.partition", "") or "")
    return part[len("gpu-") :] if part.startswith("gpu-") else part


def _derive_config_label(attrs: dict) -> str:
    """Deterministic compute-config label from config attrs (never the sa_id name).

    CPU configs use ONE consistent form: ``{Mode} {ranks}r×{threads}t ({total} CPU)`` —
    ``ranks`` = MPI processes, ``threads`` = OpenMP threads PER RANK, ``total`` = ranks ×
    threads. This makes a Hybrid config legible (ranks + threads/rank + total CPUs) while
    keeping Serial/OpenMP/MPI on the same axes: Serial 1r×1t (1 CPU), OpenMP 1r×8t (8 CPU),
    MPI 8r×1t (8 CPU), Hybrid 2r×2t (4 CPU).

    GPU configs are a distinct resource axis (GPUs, not CPUs): ``GPU ×{n} ({hardware})``,
    with hardware from the ensemble partition so an a6000 1-GPU job and an a100 1-GPU job
    are DISTINCT configs. Replicate suffixes (``_r1``/``_r2``) are NOT in the identity, so
    replicates share one label.
    """
    rm = str(attrs.get("run_mode", "?"))
    ng, nm, no, nn = (_to_int(attrs, k) for k in ("n_gpus", "n_mpi_procs", "n_omp_threads", "n_nodes"))
    if rm == "gpu":
        hw = _gpu_hardware(attrs)
        label = f"GPU ×{ng} ({hw})" if hw else f"GPU ×{ng}"
    else:
        name = {"serial": "Serial", "openmp": "OpenMP", "mpi": "MPI", "hybrid": "Hybrid"}.get(rm, rm)
        ranks, threads = max(nm, 1), max(no, 1)
        label = f"{name} {ranks}r×{threads}t ({ranks * threads} CPU)"
    if nn > 1:
        label += f", {nn} nodes"
    return label


def _load_subs(root: Path) -> dict[str, dict]:
    """Load per-sub compute-config + spatial arrays from sensitivity_datatree.zarr."""
    dt = xr.open_datatree(str(root / "sensitivity_datatree.zarr"), engine="zarr", consolidated=False)
    subs: dict[str, dict] = {}
    for g in dt.groups:
        if g.count("/") != 1 or not g.startswith("/sa_"):
            continue
        node = dt[g]
        try:
            tri = dt[g + "/tritonswmm/triton"]
            lnk = dt[g + "/tritonswmm/swmm_link"]
        except KeyError:
            continue  # sub missing coupled outputs (e.g. triton-only) — skip
        sa_id = str(node.attrs.get("sa_id", g[len("/sa_") :]))
        subs[sa_id] = {
            "attrs": dict(node.attrs),
            "label": _derive_config_label(node.attrs),
            "run_mode": str(node.attrs.get("run_mode", "")),
            "wlevel": tri["max_wlevel_m"].isel(event_iloc=0),  # (y, x) with x/y coords
            "flow": lnk["max_flow_cms"].isel(event_iloc=0),  # (link_id,)
        }
    return subs


def _group_by_identity(subs: dict[str, dict], root: Path) -> list[dict]:
    """Cluster subs into byte-identity groups using the PERSISTED flat-summary partition.

    Identity comes ONLY from ``eda/eda_cross_sim_identity.zarr``'s ``identity_group`` label
    (Gotcha 44); the consolidated arrays are NEVER compared for identity here. When the
    partition artifact is absent (``labels is None`` -- a legacy bundle) every sub becomes
    its own singleton group and the summary column renders "unknown"."""
    labels = _identity_labels(root)
    groups: list[dict] = []
    for sa_id, s in subs.items():
        w = np.asarray(s["wlevel"].values)
        f = np.asarray(s["flow"].values)
        for grp in groups:
            # Identity comes ONLY from the flat-summary-derived partition (Gotcha 44).
            # labels is None -> the artifact is absent (legacy bundle): every sub becomes
            # its own singleton group; we never fall back to a positional consolidated compare.
            if (
                labels is not None
                and labels.get(sa_id) is not None
                and labels.get(sa_id) == labels.get(grp["members"][0])
            ):
                grp["members"].append(sa_id)
                grp["labels"].append(s["label"])
                grp["run_modes"].append(s["run_mode"])
                break
        else:
            groups.append(
                {
                    "members": [sa_id],
                    "labels": [s["label"]],
                    "run_modes": [s["run_mode"]],
                    "wlevel": w,
                    "flow": f,
                    "wlevel_da": s["wlevel"],
                    "flow_da": s["flow"],
                }
            )
    return groups


def _signed_pct(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """100*(group-serial)/serial, NaN where the serial baseline is ~0 (undefined)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * delta / base
    pct[np.abs(base) < 1e-12] = np.nan
    return pct


def _load_conduit_geometry(root: Path) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """link_id -> (inlet_xy, outlet_xy) from any sub's hydraulics.inp (geometry is shared)."""
    import swmmio

    inps = sorted(root.glob("subanalyses/*/sims/*/swmm/hydraulics.inp"))
    if not inps:
        return {}
    model = swmmio.Model(str(inps[0]))
    coords = model.inp.coordinates
    conduits = model.inp.conduits
    geom: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for row in conduits.itertuples():
        if row.InletNode in coords.index and row.OutletNode in coords.index:
            p_in = (float(coords.at[row.InletNode, "X"]), float(coords.at[row.InletNode, "Y"]))
            p_out = (float(coords.at[row.OutletNode, "X"]), float(coords.at[row.OutletNode, "Y"]))
            geom[str(row.Index)] = (p_in, p_out)
    return geom


def _watershed_polygon(root: Path):
    """The bundled watershed polygon (``external/watershed.geojson``), or None. It is the
    drainage area north of the sea wall — used both as the display/colorbar mask and as a
    boundary overlay labeled 'watershed'."""
    wpath = root / "external" / "watershed.geojson"
    if not wpath.exists():
        return None
    import geopandas as gpd

    return gpd.read_file(wpath).geometry.union_all()


def _watershed_mask(poly, xd, yd) -> np.ndarray | None:
    """Boolean (ny, nx) mask — True where a DEM cell center falls INSIDE the watershed
    polygon. Restricts the displayed water level AND the depth colorbar range to the
    drainage area (north of the sea wall), matching the report's watershed-masking
    convention (``utils.create_mask_from_shapefile`` / ``per_sim_peak_flood_depth``)."""
    if poly is None:
        return None
    from shapely import contains_xy

    xx, yy = np.meshgrid(np.asarray(xd, dtype=float), np.asarray(yd, dtype=float))
    return np.asarray(contains_xy(poly, xx, yy), dtype=bool)


def _polygon_boundary_rings(poly) -> list[tuple[list[float], list[float]]]:
    """Exterior ring(s) of a (Multi)Polygon as (xs, ys) coordinate lists, for plotting the
    watershed boundary as a line overlay on each map."""
    if poly is None:
        return []
    rings = []
    for gpoly in getattr(poly, "geoms", [poly]):
        if gpoly.geom_type == "Polygon":
            xs, ys = gpoly.exterior.xy
            rings.append((list(xs), list(ys)))
    return rings


def _apply_mask(z, mask):
    """NaN-out cells outside the watershed mask (so they are not displayed / not in range)."""
    if mask is None:
        return z
    return np.where(mask, z, np.nan)


def _heatmap(da_like, z, *, x, y, colorscale, zmid=None, zmin=None, zmax=None, cbar_title, cbar_x, cbar_y, cbar_len):
    return go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(text=cbar_title, side="right", font=dict(size=10)),
            x=cbar_x,
            y=cbar_y,
            len=cbar_len,
            yanchor="middle",
            thickness=10,
            tickfont=dict(size=9),  # match the axis tick-label font size
            exponentformat="e",  # scientific notation, not the SI "p"/"n" prefixes
        ),
        hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>%{z:.4g}<extra></extra>",
    )


def _conduit_traces(geom, values_by_link, *, colorscale, vmin, vmax, cbar_title, cbar_x, cbar_y, cbar_len, diverging):
    """One line per conduit colored by its value; a hidden marker trace carries the colorbar."""
    traces = []
    span = (vmax - vmin) or 1.0
    for link_id, ((x0, y0), (x1, y1)) in geom.items():
        v = values_by_link.get(link_id)
        if v is None or not np.isfinite(v):
            color = "rgba(180,180,180,0.5)"
        else:
            t = (v - vmin) / span
            color = sample_colorscale(colorscale, [float(np.clip(t, 0, 1))])[0]
        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=6),
                hoverinfo="text",
                text=f"{link_id}: {v:.4g}" if v is not None and np.isfinite(v) else f"{link_id}: n/a",
                showlegend=False,
            )
        )
    # colorbar carrier (invisible markers spanning the value range)
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                cmid=0.0 if diverging else None,
                color=[vmin],
                showscale=True,
                colorbar=dict(
                    title=dict(text=cbar_title, side="right", font=dict(size=10)),
                    x=cbar_x,
                    y=cbar_y,
                    len=cbar_len,
                    yanchor="middle",
                    thickness=10,
                    tickfont=dict(size=9),
                    exponentformat="e",
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return traces


def build_config_diff_figure(root: Path) -> go.Figure:
    """Assemble the config-diff-maps figure from the bundle/analysis root.

    Layout (iter-2): 2 columns — DEM rasters | SWMM conduits. Row 1 = summary table.
    Panel A (row 2) = serial-CPU reference (absolute depth + flow, report palettes).
    Panels B, C ... = each byte-identical config group that differs from serial, as
    a diff row + a percent-diff row. Diverging maps (RdBu: red = below serial, blue =
    above) use a symmetric range floored at 3 cm / 0.01 cms / 0.1 %, so micrometer-
    scale diffs read as uniform "no meaningful difference" rather than saturating.
    """
    subs = _load_subs(root)
    if not subs:
        fig = go.Figure()
        fig.update_layout(height=_PANEL_H_PX, title="Config diff maps (no coupled sub-analyses found)")
        return fig

    # Identity labels (flat-summary partition) — shared by the grouping and the summary
    # column's three-state verdict; None on a legacy bundle with no identity artifact.
    labels = _identity_labels(root)
    groups = _group_by_identity(subs, root)
    serial_grp = next((g for g in groups if "serial" in g["run_modes"]), None)
    if serial_grp is None:
        fig = go.Figure()
        fig.update_layout(height=_PANEL_H_PX, title="Config diff maps (no serial-CPU baseline sub found)")
        return fig
    # Align every group's arrays to the serial reference's coords/dim-order BEFORE any
    # positional subtraction or identity flag (artifact-vector 1: a per-sub dim/coord
    # reorder would otherwise make a positional compare/subtract mismatch cells). The
    # consolidated read is value-preserving (attribute-only CF + lossless Blosc, no
    # dtype narrowing), so this alignment is the only correctness gap vs the compliant
    # cross_sim_identity comparison.
    for _g in groups:
        _g["wlevel"] = _align_to(serial_grp["wlevel_da"], _g["wlevel_da"])
        _g["flow"] = _align_to(serial_grp["flow_da"], _g["flow_da"])
    base_w = serial_grp["wlevel"]
    base_f = serial_grp["flow"]
    # Deterministic panel order (B, C, …): sort the differing groups by their sorted config
    # labels so Panel-letter assignment is stable across renders (not dict-discovery order).
    diff_groups = sorted(
        (g for g in groups if g is not serial_grp),
        key=lambda g: sorted(set(g["labels"])),
    )
    geom = _load_conduit_geometry(root)

    def _configs(g):
        # The single source of a group's config identity: sorted DISTINCT labels. The top
        # table (# configs), the per-panel side table, and the panel letters ALL derive
        # from this, so they are deterministically in alignment.
        return sorted(set(g["labels"]))

    def _rng(actual):
        # Data-driven symmetric half-range (iter-3, no floor): the actual magnitude,
        # epsilon-guarded so an all-zero diff does not collapse the diverging scale.
        a = float(actual)
        return a if np.isfinite(a) and a > _RANGE_EPS else _RANGE_EPS

    # ---- summary table (absolute diff, table-only) ----
    # Serial-containing group FIRST (item: first table row is always the serial group).
    # `# configs` counts DISTINCT compute configs (len of the deduped label set) so it
    # equals the hand-countable comma-separated list in the "Compute-config group" cell —
    # NOT len(members), which counts sa_ids incl. r1/r2 replicates.
    ordered_groups = [serial_grp] + [g for g in groups if g is not serial_grp]
    table_rows = []
    for i, g in enumerate(ordered_groups):
        wad = float(np.nanmax(np.abs(g["wlevel"] - base_w)))
        fad = float(np.nanmax(np.abs(g["flow"] - base_f)))
        # Three-state, per R2. `identical` is None when the identity artifact is absent
        # (legacy bundle) -- rendered "unknown", NEVER silently "no". Sourced ONLY from the
        # flat-summary partition label (Gotcha 44), never a positional consolidated compare.
        if labels is None:
            identical = None
        else:
            identical = bool(
                labels.get(g["members"][0]) is not None
                and labels.get(g["members"][0]) == labels.get(serial_grp["members"][0])
            )
        # Top table keyed by Panel letter (A/B/C); the full config list now lives in the
        # per-panel table beside each panel's maps, not in this summary row.
        table_rows.append(
            [
                f"Panel {chr(ord('A') + i)}",
                len(_configs(g)),
                _identity_cell(identical, g, serial_grp, wad, fad),
                f"{fad:.4g}",
                f"{wad:.4g}",
            ]
        )

    # ---- grid: 2 cols (rasters | conduits); row1 table; row2 serial ref;
    #      then per differing group a diff row + a percent-diff row. ----
    map_rows = 1 + 2 * len(diff_groups)  # serial + per-group (diff, pct)
    total_rows = 1 + map_rows
    specs = [[{"type": "table", "colspan": 2}, None]]
    for _ in range(map_rows):
        specs.append([{"type": "xy"}, {"type": "xy"}])

    # Top summary table is now single-line per group (keyed by Panel letter); the config
    # lists live in the per-panel side tables. Height = header + one row per group.
    table_px = int((len(groups) + 2) * 26 + 20)
    table_h = table_px / (table_px + _PANEL_H_PX * map_rows)
    map_h = (1.0 - table_h) / max(map_rows, 1)
    row_heights = [table_h] + [map_h] * map_rows

    # Subplot titles are placed MYSELF (centered over each map's tight x-domain, small
    # font) after the domains are pinned — make_subplots would center them over the full
    # column CELL, mis-aligning them from the narrow maps.
    fig = make_subplots(
        rows=total_rows,
        cols=2,
        specs=specs,
        row_heights=row_heights,
        vertical_spacing=min(0.07, 0.6 / max(total_rows, 1)),
        horizontal_spacing=0.13,
    )

    fig.add_trace(
        go.Table(
            columnwidth=[1.0, 1.2, 1.2, 1.4, 1.4],
            header=dict(
                values=[
                    "Panel",
                    "# configs in group",
                    "identical to serial?",
                    "max_flow_cms abs diff",
                    "max_wlevel_m abs diff",
                ],
                align="left",
                fill_color="#eef2f7",
                font=dict(size=11),
            ),
            cells=dict(
                values=list(zip(*table_rows, strict=False)) if table_rows else [[]],
                align="left",
                font=dict(size=11),
                height=22,
            ),
        ),
        row=1,
        col=1,
    )

    any_sub = next(iter(subs.values()))
    xd = [float(v) for v in any_sub["wlevel"]["x"].values]
    yd = [float(v) for v in any_sub["wlevel"]["y"].values]
    x0, x1 = min(xd), max(xd)
    y0, y1 = min(yd), max(yd)
    x_extent = (x1 - x0) or 1.0
    y_extent = (y1 - y0) or 1.0
    map_aspect = x_extent / y_extent  # one map's width/height in DATA units
    # Plot lims = the DEM extent BUFFERED by the same fraction on x and y (preserves the
    # 1:1 aspect) — a little breathing room between the axes and the watershed (item).
    _BUF = 0.06
    xr0, xr1 = x0 - _BUF * x_extent, x1 + _BUF * x_extent
    yr0, yr1 = y0 - _BUF * y_extent, y1 + _BUF * y_extent
    # Fraction of each map's y-domain that is EMPTY top buffer (data does not reach the domain
    # top). Subplot titles are placed just above the DATA top, inside this band, so they sit
    # near the plot (not floating up by the panel edge).
    _top_buf_frac = _BUF / (1 + 2 * _BUF)

    # ---- MANUAL layout (px budget) — make_subplots' UNIFORM vertical_spacing cannot give
    #      SMALL within-panel row gaps AND a LARGER between-panel gap at the same time. We
    #      assign every map row's y-domain (and the table's) explicitly. ----
    fig_width = 1000
    tbl_x = [0.035, 0.215]  # per-panel config-table region; narrowed so the y-tick labels +
    map_start = 0.30  # "y (m)" title fit in the [0.215, 0.30] gap (item: no table overlap)
    inter_gap = 0.025  # gap between the raster colorbar and the conduit column (item: tighter)

    # Map rows per panel: Panel A = [serial] (row 2, below the row-1 summary table); each
    # diff panel = [diff, pct].
    serial_row = 2
    panel_rows = [[serial_row]] + [[serial_row + 1 + 2 * gi, serial_row + 2 + 2 * gi] for gi in range(len(diff_groups))]
    H_MAP = _PANEL_H_PX  # map height (px)
    G_WITHIN = 24  # within-panel row gap (diff→pct) — SMALL (item)
    G_TOP = 26  # above a panel's first map (subplot title + outline top)
    G_FOOTER = 62  # below a panel's last map (x-ticks + x-title + swatch + gap + outline)
    G_INTER = 30  # between-panel extra gap ≈ watershed-box width (item: more space)
    G_TABLE = 16
    T_MARGIN, B_MARGIN = 80, 45

    plot_h = table_px + G_TABLE + G_INTER * (len(panel_rows) - 1)
    for prows in panel_rows:
        plot_h += G_TOP + len(prows) * H_MAP + (len(prows) - 1) * G_WITHIN + G_FOOTER
    fig_height = plot_h + T_MARGIN + B_MARGIN

    def _f(px):  # px -> paper-height fraction
        return px / plot_h

    # x-domain: 1:1 with H_MAP (map_width_px = H_MAP × data-aspect).
    wfrac = H_MAP * map_aspect / fig_width
    dom1 = [map_start, map_start + wfrac]
    dom2_start = dom1[1] + 0.078 + inter_gap  # col-1 colorbar band + gap (widened: col-1 cbar
    #   labels were touching col-2's left y-ticks — positive space now between them)
    dom2 = [dom2_start, dom2_start + wfrac]

    row_ydom: dict[int, list[float]] = {}
    table_top, table_bot = 1.0, 1.0 - _f(table_px)
    cur = table_bot - _f(G_TABLE)
    for pi, prows in enumerate(panel_rows):
        cur -= _f(G_TOP)
        for ri, r in enumerate(prows):
            row_ydom[r] = [cur - _f(H_MAP), cur]
            cur -= _f(H_MAP)
            if ri < len(prows) - 1:
                cur -= _f(G_WITHIN)
        cur -= _f(G_FOOTER)
        if pi < len(panel_rows) - 1:
            cur -= _f(G_INTER)
    for r, yd_ in row_ydom.items():
        next(fig.select_xaxes(row=r, col=1)).domain = dom1
        next(fig.select_xaxes(row=r, col=2)).domain = dom2
        next(fig.select_yaxes(row=r, col=1)).domain = yd_
        next(fig.select_yaxes(row=r, col=2)).domain = yd_
    fig.data[0].domain = dict(x=[0.03, 0.97], y=[table_bot, table_top])  # summary table

    def _ydom(r, c=1):
        return row_ydom[r]

    def _xdom(r, c):
        return dom1 if c == 1 else dom2

    # Colorbars hug the right edge of their map, centered on the (uniform) map y-domain.
    cb_x = {1: dom1[1] + 0.006, 2: dom2[1] + 0.006}
    cb_len = _f(H_MAP) * 0.75  # 75% of the map's y-domain, centered — fully inside the ylim

    def _cb_y(r):
        d0, d1 = row_ydom[r]
        return (d0 + d1) / 2.0

    # Watershed (north of the sea wall): mask displayed water level + depth colorbar range
    # to the drainage area (excludes the coastal storm-tide extreme), and overlay its
    # boundary on every map labeled 'watershed'.
    wpoly = _watershed_polygon(root)
    wmask = _watershed_mask(wpoly, xd, yd)

    # ---- Global (across-panel) diff ranges, keyed by UNIT — like colorbars share one
    #      vmin/vmax. Depth-diff ("m") and flow-diff ("cms") each share ONE symmetric range
    #      across ALL panels; depth-% and flow-% are BOTH "%", so they share ONE unified range
    #      spanning every panel's depth-% AND flow-%. (Flow diffs are exactly 0.0 in coupled
    #      byte-identical groups, so unifying % is lossless — flow-% renders flat because it
    #      IS flat.) Colorbars with DIFFERENT colorscales (the sequential serial references)
    #      are exempt: a shared unit with a different color signals a different scale.
    def _maxabs(a):
        a = np.asarray(a)
        a = a[np.isfinite(a)]
        return float(np.nanmax(np.abs(a))) if a.size else 0.0

    _g_depth = _g_flow = _g_pct = 0.0
    for g in diff_groups:
        _dw = _apply_mask(g["wlevel"] - base_w, wmask)
        _pw = _apply_mask(_signed_pct(g["wlevel"] - base_w, base_w), wmask)
        _df = g["flow"] - base_f
        _pf = _signed_pct(_df, base_f)
        _g_depth = max(_g_depth, _maxabs(_dw))
        _g_flow = max(_g_flow, _maxabs(_df))
        _g_pct = max(_g_pct, _maxabs(_pw), _maxabs(_pf))
    wsym = _rng(_g_depth)  # depth diff (m) — shared across every panel
    fsym = _rng(_g_flow)  # flow diff (cms) — shared across every panel
    pct_sym = _rng(_g_pct)  # percent difference (%) — depth% ∪ flow%, shared across every panel

    def _links(g):
        return [str(x) for x in g["flow_da"]["link_id"].values]

    annotations = []

    def _panel_label(text, first_row, last_row):
        # Short rotated "Panel X" label at the far left, centered on the panel using the
        # ACTUAL subplot y-domains (accounts for vertical_spacing) so it sits centered
        # inside the panel outline (item: panel labels centered in the outline).
        y_top = _ydom(first_row)[1]
        y_bot = _ydom(last_row)[0]
        annotations.append(
            dict(
                x=0.016,
                y=(y_top + y_bot) / 2.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                textangle=-90,
                showarrow=False,
                font=dict(size=13, color="#111"),
                text=text,
            )
        )

    def _panel_config_table(g, first_row, last_row):
        # Per-panel table (left of the maps) listing the byte-identical configs in the
        # group, so the panel LABEL can stay short ("Panel X"). Positioned by explicit
        # domain spanning the panel's y-range.
        y_top = _ydom(first_row)[1]
        y_bot = _ydom(last_row)[0]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["byte-identical configs"],
                    align="left",
                    fill_color="#eef2f7",
                    font=dict(size=9),
                    height=20,
                ),
                cells=dict(values=[_configs(g)], align="left", font=dict(size=9), height=18),
                domain=dict(x=tbl_x, y=[max(0.0, y_bot), min(1.0, y_top)]),
            )
        )

    # ---- Panel A: serial-CPU reference (absolute magnitudes, report palettes) ----
    fmax = _rng(np.nanmax(np.abs(base_f)))
    # Serial depth reference: displayed water level AND the colorbar vmax are restricted to
    # the watershed mask (north of the sea wall) — the coastal storm-tide extreme south of
    # the wall is excluded, so the inland floodplain signal is visible (item: vmax based on
    # the region NORTH of the sea wall, via the geodataframe polygon mask).
    base_w_disp = _apply_mask(base_w, wmask)
    depth_vmax = float(np.nanmax(base_w_disp)) if np.isfinite(base_w_disp).any() else None
    fig.add_trace(
        _heatmap(
            base_w_disp,
            base_w_disp,
            x=xd,
            y=yd,
            colorscale=_REF_DEPTH,
            zmin=0,
            zmax=depth_vmax if depth_vmax and depth_vmax > 0 else None,
            cbar_title="m",
            cbar_x=cb_x[1],
            cbar_y=_cb_y(serial_row),
            cbar_len=cb_len,
        ),
        row=serial_row,
        col=1,
    )
    serial_links = _links(serial_grp)
    for tr in _conduit_traces(
        geom,
        dict(zip(serial_links, np.asarray(base_f), strict=False)),
        colorscale=_REF_FLOW,
        vmin=0,
        vmax=fmax,
        cbar_title="cms",
        cbar_x=cb_x[2],
        cbar_y=_cb_y(serial_row),
        cbar_len=cb_len,
        diverging=False,
    ):
        fig.add_trace(tr, row=serial_row, col=2)
    _panel_label("<b>Panel A</b> — Serial CPU reference", serial_row, serial_row)
    _panel_config_table(serial_grp, serial_row, serial_row)

    # ---- Panels B, C ... : per differing group (diff row + percent-diff row) ----
    for gi, g in enumerate(diff_groups):
        diff_row = serial_row + 1 + 2 * gi
        pct_row = diff_row + 1
        # Depth diff/pct maps masked to the watershed (display consistency with the serial
        # reference); flow (conduits) are already all upstream, so not masked.
        dw = _apply_mask(g["wlevel"] - base_w, wmask)
        pw = _apply_mask(_signed_pct(g["wlevel"] - base_w, base_w), wmask)
        df = g["flow"] - base_f
        pf = _signed_pct(df, base_f)
        # wsym / fsym / pct_sym are GLOBAL (computed above): like-unit colorbars share one range.
        links = _links(g)

        # diff row: depth diff (raster) | flow diff (conduits)
        fig.add_trace(
            _heatmap(
                dw,
                dw,
                x=xd,
                y=yd,
                colorscale=_DIVERGING,
                zmid=0,
                zmin=-wsym,
                zmax=wsym,
                cbar_title="m",
                cbar_x=cb_x[1],
                cbar_y=_cb_y(diff_row),
                cbar_len=cb_len,
            ),
            row=diff_row,
            col=1,
        )
        for tr in _conduit_traces(
            geom,
            dict(zip(links, np.asarray(df), strict=False)),
            colorscale=_DIVERGING,
            vmin=-fsym,
            vmax=fsym,
            cbar_title="cms",
            cbar_x=cb_x[2],
            cbar_y=_cb_y(diff_row),
            cbar_len=cb_len,
            diverging=True,
        ):
            fig.add_trace(tr, row=diff_row, col=2)

        # percent-diff row: depth % (raster) | flow % (conduits)
        fig.add_trace(
            _heatmap(
                pw,
                pw,
                x=xd,
                y=yd,
                colorscale=_DIVERGING,
                zmid=0,
                zmin=-pct_sym,
                zmax=pct_sym,
                cbar_title="percent difference (%)",
                cbar_x=cb_x[1],
                cbar_y=_cb_y(pct_row),
                cbar_len=cb_len,
            ),
            row=pct_row,
            col=1,
        )
        for tr in _conduit_traces(
            geom,
            dict(zip(links, np.asarray(pf), strict=False)),
            colorscale=_DIVERGING,
            vmin=-pct_sym,
            vmax=pct_sym,
            cbar_title="percent difference (%)",
            cbar_x=cb_x[2],
            cbar_y=_cb_y(pct_row),
            cbar_len=cb_len,
            diverging=True,
        ):
            fig.add_trace(tr, row=pct_row, col=2)

        panel = chr(ord("B") + gi)
        # Left-margin rotated label; group title from the SAME _configs(g) as the table
        # (all hardware variants, never a single a6000 rep), so panel + table stay aligned.
        _panel_label(f"<b>Panel {panel}</b>", diff_row, pct_row)
        _panel_config_table(g, diff_row, pct_row)

    # Bottom map row of EACH panel (Panel A = serial_row; each diff panel = its pct row) —
    # these rows carry the "x (m)" title (item: every panel has an x label).
    panel_bottom_rows = {serial_row + 2 * k for k in range(len(diff_groups) + 1)}
    for r in range(2, total_rows + 1):
        for c in (1, 2):
            # No DEM boundary box (showline/mirror OFF) — the watershed boundary overlay is
            # the only frame now (item). Buffered range gives breathing room. Ticks + labels
            # stay; x-title on each panel's bottom row; y-title on col-1 only (col-2 shares y,
            # and a col-2 y-title collides with the col-1 colorbar's rotated unit label).
            fig.update_xaxes(
                row=r,
                col=c,
                title_text="x (m)" if r in panel_bottom_rows else "",
                title_font=dict(size=10),
                range=[xr0, xr1],
                constrain="domain",
                showgrid=False,
                zeroline=False,
                showline=False,
                mirror=False,
                ticks="outside",
                # x tick labels ONLY on each panel's bottom row (like the x-title): a non-
                # bottom (diff) row shares x with the pct row below it, and its tick labels
                # would collide with the pct row's subplot title in the small within-gap.
                showticklabels=(r in panel_bottom_rows),
                tickfont=dict(size=9),
            )
            yax = next(fig.select_yaxes(row=r, col=c), None)
            anchor = yax.anchor if yax is not None else None
            fig.update_yaxes(
                row=r,
                col=c,
                title_text="y (m)" if c == 1 else "",
                title_font=dict(size=10),
                range=[yr0, yr1],
                constrain="domain",
                showgrid=False,
                zeroline=False,
                showline=False,
                mirror=False,
                ticks="outside",
                # y tick labels on col-1 only (col-2 shares the same y); this also frees the
                # space where the col-1 colorbar's rotated "percent difference (%)" title was
                # colliding with the col-2 y labels.
                showticklabels=(c == 1),
                tickfont=dict(size=9),
                scaleanchor=anchor,
                scaleratio=1,
            )

    # ---- watershed boundary overlay on every map (no figure legend; a per-panel swatch
    #      is drawn below each panel's x-axis instead) ----
    rings = _polygon_boundary_rings(wpoly)
    for r in range(2, total_rows + 1):
        for c in (1, 2):
            for xs, ys in rings:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#111", width=1.3),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=c,
                )

    # ---- my own subplot titles: centered over each map's tight x-domain, small font ----
    _titles = {
        (serial_row, 1): "Serial depth (m)",
        (serial_row, 2): "Serial flow (cms)",
    }
    for gi in range(len(diff_groups)):
        dr = serial_row + 1 + 2 * gi
        pr = dr + 1
        _titles[(dr, 1)] = "Depth diff (m)"
        _titles[(dr, 2)] = "Flow diff (cms)"
        _titles[(pr, 1)] = "Depth % diff"
        _titles[(pr, 2)] = "Flow % diff"
    for (r, c), t in _titles.items():
        xa0, xa1 = _xdom(r, c)
        ya0, ya1 = _ydom(r, c)
        data_top = ya1 - _top_buf_frac * (ya1 - ya0)  # paper-y of the DATA top (below domain top)
        annotations.append(
            dict(
                x=(xa0 + xa1) / 2.0,
                y=data_top + _f(4),  # just above the data, inside the empty top buffer band
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#444"),
                text=t,
            )
        )

    # ---- per panel: a "watershed" swatch (unfilled rectangle + label) below the x-axis,
    #      and a black DASHED outline enclosing the table + maps + colorbars + labels ----
    shapes = []
    panel_spans = [(serial_row, serial_row)] + [
        (serial_row + 1 + 2 * gi, serial_row + 2 + 2 * gi) for gi in range(len(diff_groups))
    ]
    # Watershed swatch sits on the "x (m)" title line, in the gap BETWEEN the two maps'
    # x-titles (col-1 title ~dom1 center, col-2 ~dom2 center) — compact, inside the panel,
    # not down in the inter-panel seam.
    ws_cx = (dom1[1] + dom2[0]) / 2.0
    # px-based swatch dims so they're consistent regardless of the figure height and fit in
    # the G_FOOTER budget: 28 px wide × 16 px tall (WIDER than tall; ~1.3× the 10-pt text).
    _SW_HALF_W = 14 / fig_width  # paper-x
    _SW_HALF_H = _f(8)  # paper-y
    for first_row, last_row in panel_spans:
        y_top = _ydom(first_row, 1)[1]
        y_bot = _ydom(last_row, 1)[0]
        sw_y = y_bot - _f(34)  # below the x-axis "x (m)" title
        # watershed legend swatch = small UNFILLED rectangle, wider than tall (item)
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=ws_cx - _SW_HALF_W,
                x1=ws_cx + _SW_HALF_W,
                y0=sw_y - _SW_HALF_H,
                y1=sw_y + _SW_HALF_H,
                line=dict(color="#111", width=1.2),
                fillcolor="rgba(0,0,0,0)",
            )
        )
        annotations.append(
            dict(
                x=ws_cx + _SW_HALF_W + 0.006,  # AFTER the box's right edge (item: text was overlapping the box)
                y=sw_y,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(size=10, color="#111"),
                text="watershed",
            )
        )
        # dashed panel outline: left of the table -> right of the colorbar labels; bottom
        # a FIXED px gap below the swatch (so the swatch↔outline gap is identical across all
        # panels, WITH space — not the adjacent look A/B had).
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.006,
                x1=cb_x[2] + 0.12,
                y0=sw_y - _SW_HALF_H - _f(12),
                y1=y_top + _f(12),
                line=dict(color="black", width=1, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )
        )

    fig.update_layout(
        height=fig_height,
        width=fig_width,
        margin=dict(t=T_MARGIN, l=30, r=30, b=B_MARGIN),
        title="Config diff maps: spatial difference vs serial-CPU baseline, per byte-identical config group",
        annotations=list(fig.layout.annotations) + annotations,
        shapes=shapes,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def config_diff_source_paths(root: Path) -> list[Path]:
    """Declared source_paths for the harvest chain: the consolidated tree (diff maps) +
    the flat-summary-derived identity artifact (the identity column) + one inp.

    Without the identity artifact declaration, ``Bundle.eda(plots_only=True)`` would
    re-render the identity column from an absent partition and the harvest would SKIP it
    with a warning (Gotcha 50) -- a silently wrong column, the exact failure class this
    plan closes. The bundle file set is EXACTLY the union of manifest-declared source_paths
    (the `bundle file set is computed from manifest harvest` stipulation)."""
    srcs: list[Path] = [
        root / "sensitivity_datatree.zarr",
        root / "eda" / "eda_cross_sim_identity.zarr",
    ]
    inps = sorted(root.glob("subanalyses/*/sims/*/swmm/hydraulics.inp"))
    if inps:
        srcs.append(inps[0])
    return srcs
