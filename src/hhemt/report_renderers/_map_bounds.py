"""Shared map-panel bounds computation for per-figure layout parity.

Consumed by `system_overview` and the per_sim_* renderers so cross-figure
spatial extents stay identical. Pulled out of `system_overview._compute_panel_bounds`
on 2026-05-16 (Phase 3 F-I-5) to enable per_sim_peak_flood_depth to share the
same square+padded-union bounds.
"""

from __future__ import annotations


def compute_padded_square_bounds(
    dem_bounds,
    hydro_model,
    hydraulics_model,
    *,
    padding_frac: float = 0.02,
) -> tuple[float, float, float, float]:
    """Return a padded SQUARE (xmin, ymin, xmax, ymax) that encompasses both
    the TRITON DEM modeled extent AND every SWMM node coordinate from both
    the hydrology and hydraulics models.

    Without this expansion, left-edge nodes (e.g., upstream-most subcatchment
    outlets whose coordinates sit slightly outside the DEM's processed bounds)
    are clipped at the panel boundary. The padding + square-up also satisfy
    the user-requested "lock 1:1 aspect ratio" for any plot with
    easting/northing axes: with Δx == Δy and `scaleanchor` enforced on the
    Plotly y-axes, the data renders square inside the panel envelope.

    Either `hydro_model` or `hydraulics_model` may be `None`; in that case
    only the supplied model's coordinates contribute. Passing both as `None`
    reduces the bounds to a padded square around the DEM extent alone.
    """
    xmins = [float(dem_bounds[0])]
    ymins = [float(dem_bounds[1])]
    xmaxs = [float(dem_bounds[2])]
    ymaxs = [float(dem_bounds[3])]
    for model in (hydro_model, hydraulics_model):
        coords_df = getattr(getattr(model, "inp", None), "coordinates", None)
        if coords_df is None or len(coords_df) == 0:
            continue
        xmins.append(float(coords_df["X"].min()))
        xmaxs.append(float(coords_df["X"].max()))
        ymins.append(float(coords_df["Y"].min()))
        ymaxs.append(float(coords_df["Y"].max()))
    xmin, xmax = min(xmins), max(xmaxs)
    ymin, ymax = min(ymins), max(ymaxs)
    dx, dy = xmax - xmin, ymax - ymin
    if dx < dy:
        delta = (dy - dx) / 2.0
        xmin -= delta
        xmax += delta
    elif dy < dx:
        delta = (dx - dy) / 2.0
        ymin -= delta
        ymax += delta
    side = xmax - xmin
    pad = side * padding_frac
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
