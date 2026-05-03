"""Persistent GeoJSON exporter for SWMM-derived spatial layers.

Writes one GeoJSON per element kind into `<output_dir>/`:

- `subcatchments.geojson` (Polygon) — from hydro `[POLYGONS]` + `[SUBCATCHMENTS]`
- `drainage_lines.geojson` (LineString) — subcatchment centroid → outlet node
- `junctions.geojson` (Point) — from hydraulics `[JUNCTIONS]` + `[COORDINATES]`
- `outfalls.geojson` (Point) — from hydraulics `[OUTFALLS]` + `[COORDINATES]`
- `conduits.geojson` (LineString) — inlet-node coords → outlet-node coords

These are durable, tool-agnostic copies of the SWMM model's spatial structure,
designed for use in QGIS / ArcGIS / other GIS software for manual map building.
The renderer calls this helper as a side effect of `system_overview.render()`
so the layers stay in sync with each new analysis run.

The exporter is idempotent — calling it multiple times with the same inputs
produces byte-identical outputs (modulo geopandas version differences).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from TRITON_SWMM_toolkit import swmm_schema as _ss

if TYPE_CHECKING:
    import swmmio


_LAYER_NAMES = (
    "subcatchments",
    "drainage_lines",
    "junctions",
    "outfalls",
    "conduits",
)


def export_swmm_gis_layers(
    hydro_model: swmmio.Model,
    hydraulics_model: swmmio.Model,
    output_dir: Path,
    target_crs: Any | None = None,
) -> dict[str, Path]:
    """Write one GeoJSON per element kind into `output_dir`.

    Returns a `{layer_name: written_path}` dict for the five layers above.
    Layers with zero features are still written (empty FeatureCollection)
    so downstream tooling can rely on the file's presence.

    `target_crs` is forwarded to each `GeoDataFrame` (it accepts pyproj `CRS`,
    EPSG int, EPSG string, or any other geopandas-compatible CRS spec). When
    `None`, the layer is written with no CRS metadata — caller is responsible
    for assigning one before consuming in CRS-aware GIS software.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["subcatchments"] = _write_subcatchments(
        hydro_model, output_dir / "subcatchments.geojson", target_crs,
    )
    paths["drainage_lines"] = _write_drainage_lines(
        hydro_model, output_dir / "drainage_lines.geojson", target_crs,
    )
    paths["junctions"] = _write_junctions(
        hydraulics_model, output_dir / "junctions.geojson", target_crs,
    )
    paths["outfalls"] = _write_outfalls(
        hydraulics_model, output_dir / "outfalls.geojson", target_crs,
    )
    paths["conduits"] = _write_conduits(
        hydraulics_model, output_dir / "conduits.geojson", target_crs,
    )
    return paths


def _write_subcatchments(model, path: Path, crs) -> Path:
    polygons_df = getattr(model.inp, "polygons", None)
    subcatch_df = getattr(model.inp, "subcatchments", None)
    records: list[dict[str, Any]] = []
    if polygons_df is not None and len(polygons_df) > 0:
        for sc_name in polygons_df.index.unique():
            rows = polygons_df.loc[[sc_name]]
            verts = list(zip(
                rows["X"].astype(float), rows["Y"].astype(float), strict=True,
            ))
            if len(verts) < 3:
                continue
            geom = Polygon(verts)
            props: dict[str, Any] = {"name": str(sc_name)}
            if subcatch_df is not None and sc_name in subcatch_df.index:
                row = subcatch_df.loc[sc_name]
                for col in _ss.SUBCATCHMENT_FEATURE_PROPS:
                    if col in row.index:
                        props[col.lower()] = _coerce(row[col])
            records.append({"geometry": geom, **props})
    _write_geojson(records, path, crs)
    return path


def _write_drainage_lines(model, path: Path, crs) -> Path:
    polygons_df = getattr(model.inp, "polygons", None)
    subcatch_df = getattr(model.inp, "subcatchments", None)
    coords_df = model.inp.coordinates
    records: list[dict[str, Any]] = []
    if (
        polygons_df is not None
        and len(polygons_df) > 0
        and subcatch_df is not None
    ):
        for sc_name in polygons_df.index.unique():
            if sc_name not in subcatch_df.index:
                continue
            outlet_name = subcatch_df.at[sc_name, "Outlet"]
            if outlet_name not in coords_df.index:
                continue
            rows = polygons_df.loc[[sc_name]]
            verts = list(zip(
                rows["X"].astype(float), rows["Y"].astype(float), strict=True,
            ))
            if not verts:
                continue
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            ox = float(coords_df.at[outlet_name, "X"])
            oy = float(coords_df.at[outlet_name, "Y"])
            records.append({
                "geometry": LineString([(cx, cy), (ox, oy)]),
                "subcatchment": str(sc_name),
                "outlet": str(outlet_name),
            })
    _write_geojson(records, path, crs)
    return path


def _write_junctions(model, path: Path, crs) -> Path:
    junctions_df = model.inp.junctions
    coords_df = model.inp.coordinates
    records: list[dict[str, Any]] = []
    for name, row in junctions_df.iterrows():
        if name not in coords_df.index:
            continue
        x = float(coords_df.at[name, "X"])
        y = float(coords_df.at[name, "Y"])
        invert = _coerce(row.get(_ss.JUNC_INVERT_ELEV))
        max_depth = _coerce(row.get(_ss.JUNC_MAX_DEPTH))
        props: dict[str, Any] = {
            "name": str(name),
            "invert_elev": invert,
            "max_depth": max_depth,
        }
        try:
            if invert is not None and max_depth is not None:
                props["rim_elev"] = float(invert) + float(max_depth)
        except (TypeError, ValueError):
            pass
        records.append({"geometry": Point(x, y), **props})
    _write_geojson(records, path, crs)
    return path


def _write_outfalls(model, path: Path, crs) -> Path:
    outfalls_df = model.inp.outfalls
    coords_df = model.inp.coordinates
    records: list[dict[str, Any]] = []
    for name, row in outfalls_df.iterrows():
        if name not in coords_df.index:
            continue
        x = float(coords_df.at[name, "X"])
        y = float(coords_df.at[name, "Y"])
        props: dict[str, Any] = {
            "name": str(name),
            "invert_elev": _coerce(row.get(_ss.OUTFALL_INVERT_ELEV)),
        }
        if _ss.OUTFALL_TYPE in row.index:
            props["outfall_type"] = _coerce(row[_ss.OUTFALL_TYPE])
        records.append({"geometry": Point(x, y), **props})
    _write_geojson(records, path, crs)
    return path


def _write_conduits(model, path: Path, crs) -> Path:
    conduits_df = model.inp.conduits
    junctions_df = model.inp.junctions
    outfalls_df = model.inp.outfalls
    coords_df = model.inp.coordinates
    inverts: dict[str, float] = {}
    for nm, row in junctions_df.iterrows():
        try:
            inverts[nm] = float(row[_ss.JUNC_INVERT_ELEV])
        except (KeyError, TypeError, ValueError):
            pass
    for nm, row in outfalls_df.iterrows():
        try:
            inverts[nm] = float(row[_ss.OUTFALL_INVERT_ELEV])
        except (KeyError, TypeError, ValueError):
            pass
    records: list[dict[str, Any]] = []
    for row in conduits_df.itertuples():
        if row.InletNode not in coords_df.index or row.OutletNode not in coords_df.index:
            continue
        p_in = (
            float(coords_df.at[row.InletNode, "X"]),
            float(coords_df.at[row.InletNode, "Y"]),
        )
        p_out = (
            float(coords_df.at[row.OutletNode, "X"]),
            float(coords_df.at[row.OutletNode, "Y"]),
        )
        length = _coerce(getattr(row, "Length", None))
        props: dict[str, Any] = {
            "name": str(row.Index),
            "inlet_node": str(row.InletNode),
            "outlet_node": str(row.OutletNode),
            "length_m": length,
            "roughness": _coerce(getattr(row, "Roughness", None)),
        }
        inv_in = inverts.get(row.InletNode)
        inv_out = inverts.get(row.OutletNode)
        if length and inv_in is not None and inv_out is not None:
            try:
                props["slope_pct"] = 100.0 * (inv_in - inv_out) / float(length)
            except (TypeError, ZeroDivisionError):
                pass
        records.append({
            "geometry": LineString([p_in, p_out]),
            **props,
        })
    _write_geojson(records, path, crs)
    return path


def _write_geojson(
    records: list[dict[str, Any]], path: Path, crs: Any | None,
) -> None:
    if records:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    else:
        # geopandas raises on empty data without explicit columns; supply a
        # minimal schema so an empty layer file still exists.
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf.to_file(path, driver="GeoJSON")


def _coerce(value: Any) -> Any:
    """Best-effort scalar coercion for GeoJSON property serialization."""
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        # pandas / numpy scalars
        return value.item()  # type: ignore[attr-defined]
    except AttributeError:
        return str(value)
