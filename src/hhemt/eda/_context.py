"""load_eda_context — the single installed-hhemt loader that re-derives every
notebook-bound EDA variable from on-disk artifacts resident at ``root`` (ADR-13).

``root`` is EITHER a live ``analysis_dir`` OR a transported ``bundle.root``. The
loader NEVER pickles emit-time state — it re-derives on EXECUTION, so the seed
notebook survives bundle transport and refreshes deterministically every run.
``is_bundle`` (presence of ``{root}/bundle_manifest.json``) gates the ADR-9
byte-identity calc cells, because a bundle carries no source per-scenario summaries.

Absence contract (mixed, FQ1 Option C): configs RAISE on absence (a root without a
config is not a valid EDA root); experiment-shape-dependent fields
(``sensitivity_datatree``, ``performance``) and data artifacts (``datatree``,
``scenario_status``, ``swmm_features``, ``triton_dem``) are ``None`` on absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    import xarray as xr

    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config


@dataclass(frozen=True)
class EdaContext:
    """Re-derived experiment context bound as named variables in the EDA notebook."""

    datatree: xr.DataTree | None
    sensitivity_datatree: xr.DataTree | None
    cfg_analysis: analysis_config
    cfg_system: system_config
    scenario_status: pd.DataFrame | None
    swmm_features: dict[str, gpd.GeoDataFrame] | None
    triton_dem: xr.DataArray | None
    performance: xr.DataTree | None
    is_bundle: bool


def load_eda_context(root: Path | str) -> EdaContext:
    """Re-derive the 9 EdaContext fields from artifacts resident at ``root``.

    Raises ``FileNotFoundError`` only for the two config files (a root without a
    config is not a valid EDA root). Every data field is ``None`` when its artifact
    is legitimately absent (non-sensitivity → no ``sensitivity_datatree``; a bundle
    or partial tree → no ``datatree``/``scenario_status``/``swmm_features``/``triton_dem``).
    """
    import geopandas as gpd
    import pandas as pd
    import rioxarray as rxr
    import xarray as xr

    from hhemt.config.analysis import analysis_config
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.system import system_config

    root = Path(root)
    is_bundle = (root / "bundle_manifest.json").exists()

    cfg_analysis_path = root / "cfg_analysis.yaml"
    cfg_system_path = root / "cfg_system.yaml"
    if not cfg_analysis_path.exists():
        raise FileNotFoundError(
            f"{root} is not a valid EDA root: cfg_analysis.yaml absent. "
            f"load_eda_context expects an analysis_dir or a render-bundle root."
        )
    if not cfg_system_path.exists():
        raise FileNotFoundError(f"{root} is not a valid EDA root: cfg_system.yaml absent.")
    cfg_analysis = yaml_to_model(cfg_analysis_path, analysis_config)
    cfg_system = yaml_to_model(cfg_system_path, system_config)

    def _open_tree(p: Path) -> xr.DataTree | None:
        return xr.open_datatree(str(p), engine="zarr", consolidated=False) if p.exists() else None

    datatree = _open_tree(root / "analysis_datatree.zarr")
    sensitivity_datatree = (
        _open_tree(root / "sensitivity_datatree.zarr") if cfg_analysis.toggle_sensitivity_analysis else None
    )

    status_csv = root / "scenario_status.csv"
    scenario_status = pd.read_csv(status_csv) if status_csv.exists() else None

    # SWMM GIS layers are exported under system_dir/gis (system_overview.py:138),
    # NOT analysis_dir/gis. On a bundle root the harvested copy is at {root}/gis;
    # prefer the bundle copy, fall back to the system-dir source for a live analysis.
    bundle_gis = root / "gis"
    system_gis = Path(cfg_system.system_directory) / "gis"
    gis_dir = bundle_gis if (bundle_gis.is_dir() and any(bundle_gis.glob("*.gpkg"))) else system_gis
    swmm_features = (
        {p.stem: gpd.read_file(p) for p in sorted(gis_dir.glob("*.gpkg"))}
        if gis_dir.is_dir() and any(gis_dir.glob("*.gpkg"))
        else None
    )

    # The processed DEM is a SysPaths-derived artifact, NOT a cfg_system field:
    # system.py builds it as system_dir / f"elevation_{target_dem_resolution:.2f}m.dem".
    # On a bundle root the harvested copy is root-relative; prefer it when present.
    dem_name = f"elevation_{cfg_system.target_dem_resolution:.2f}m.dem"
    bundle_dem = root / dem_name
    system_dem = Path(cfg_system.system_directory) / dem_name
    dem_path = bundle_dem if bundle_dem.exists() else system_dem
    triton_dem = rxr.open_rasterio(dem_path) if dem_path.exists() else None

    # performance is a derived VIEW of the consolidated datatree (the */performance
    # nodes), never a separate artifact — None exactly when datatree is None or
    # carries no performance node.
    performance: xr.DataTree | None = None
    if datatree is not None:
        perf_nodes = [n for n in datatree.subtree if str(n.name) == "performance"]
        performance = datatree if perf_nodes else None

    return EdaContext(
        datatree=datatree,
        sensitivity_datatree=sensitivity_datatree,
        cfg_analysis=cfg_analysis,
        cfg_system=cfg_system,
        scenario_status=scenario_status,
        swmm_features=swmm_features,
        triton_dem=triton_dem,
        performance=performance,
        is_bundle=is_bundle,
    )
