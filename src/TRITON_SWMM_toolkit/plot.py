import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import xarray as xr
from typing import Optional
from matplotlib.axes import Axes
import json
from typing import Union, Optional


def plot_polygon_boundary_on_ax(ax, shp_path: Path, color="black", linewidth=1):
    gdf = gpd.read_file(shp_path)
    gdf.boundary.plot(ax=ax, color=color, linewidth=linewidth)


def plot_discrete_raster(
    rds,
    cbar_lab,
    cmap: str = "",  # requires either this or colors
    colors: pd.Series = pd.Series(),  # colors indexed by value
    labs: pd.Series = pd.Series(),  # description for tick labels on colorbar indexed by value
    label_axes_ticks=False,  # determines whether or not x and y axis ticks are labeled
    ax: Optional[Axes] = None,
    watershed_shapefile: Path = Path(""),
    watershed_shapefile_color="black",
):
    # filter out no data values
    nodata = rds.rio.nodata
    if nodata is not None:
        mask = rds.values != nodata
    else:
        mask = ~np.isnan(rds.values)
    rds = rds.where(mask)
    unique_vals = np.unique(rds.values[~np.isnan(rds.values)])
    unique_vals.sort()
    if len(colors) > 0:
        cs = colors.loc[unique_vals]
        cmap_obj = ListedColormap(cs)
    else:
        cmap_obj = colormaps[cmap]

    bounds = np.append(unique_vals, unique_vals[-1] + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N)
    tick_positions = (bounds[:-1] + bounds[1:]) / 2
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    img = rds.plot(ax=ax, cmap=cmap_obj, norm=norm, add_colorbar=False)

    cbar = fig.colorbar(img, ax=ax, boundaries=bounds, ticks=tick_positions)

    if len(labs) > 0:
        ticklabels = labs.loc[unique_vals]
    else:
        ticklabels = [str((val)) for val in unique_vals]
    cbar.ax.set_yticklabels(ticklabels)
    cbar.set_label(cbar_lab)
    if not label_axes_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_title("")
    if len(watershed_shapefile.name) > 0:
        plot_polygon_boundary_on_ax(
            ax, watershed_shapefile, color=watershed_shapefile_color
        )
    # plt.show()

    return ax


def plot_landuse_raster(
    landuse_raster,
    landuse_lookup,
    landuse_colname,
    landuse_description_colname,
    landuse_plot_color_colname,
    watershed_shapefile,
    watershed_shapefile_color="black",
    ax: Optional[Axes] = None,
):
    rds = rxr.open_rasterio(landuse_raster)
    df_lu_lookup = pd.read_csv(landuse_lookup).set_index(landuse_colname)
    labs = df_lu_lookup[landuse_description_colname]
    colors = df_lu_lookup[landuse_plot_color_colname]

    ax = plot_discrete_raster(
        rds,
        cbar_lab="",
        colors=colors,
        labs=labs,
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color=watershed_shapefile_color,
        ax=ax,
    )
    return ax


def plot_continuous_raster(
    rds: xr.DataArray,
    cbar_lab: str,
    cmap: str = "viridis",
    label_axes_ticks: bool = False,
    watershed_shapefile: Optional[Path] = None,
    watershed_shapefile_color: str = "black",
    ax: Optional[Axes] = None,
    show_cbar: bool = True,
    vmin=None,
    vmax=None,
    **cbar_kwargs,
):
    # Mask nodata values if present
    nodata = rds.rio.nodata if hasattr(rds, "rio") else None
    mask = ~np.isnan(rds.values) if nodata is None else (rds.values != nodata)
    rds = rds.where(mask)

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    if vmin is None:
        vmin = rds.min()

    if vmax is None:
        vmax = rds.max()

    cmap_obj = colormaps[cmap].copy()
    for key in ["set_under", "set_over", "set_bad"]:
        if key in cbar_kwargs:
            getattr(cmap_obj, key)(cbar_kwargs[key])

    img = rds.plot(  # type: ignore
        ax=ax,
        cmap=cmap_obj,
        add_colorbar=show_cbar,
        vmin=vmin,
        vmax=vmax,
    )
    if show_cbar:
        img.colorbar.set_label(cbar_lab)

    if not label_axes_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    ax.set_title("")

    if watershed_shapefile and watershed_shapefile.exists():
        plot_polygon_boundary_on_ax(
            ax,
            watershed_shapefile,
            color=watershed_shapefile_color,
        )

    # plt.show()
    return ax


def process_dem_for_plotting(
    rds_dem, dem_out_of_watershed=None, dem_building_height=None
):
    """
    Docstring for process_dem_for_plotting

    :param rds_dem: dem dataset
    :param dem_out_of_watershed: height assigned to DEM outside of watershed
    :param dem_building_height: height assigned to DEM where it overlaps buildings

    This assigns n/a to all dem grid cells that are outside of the watershed or represent buildings
    """
    rds_dem_plot_ready = rds_dem.copy()
    if dem_out_of_watershed is not None:
        mask_not_out_of_shed = rds_dem != dem_out_of_watershed
        rds_dem_plot_ready = rds_dem_plot_ready.where(mask_not_out_of_shed)
    if dem_building_height is not None:
        mask_not_building = rds_dem != dem_building_height
        rds_dem_plot_ready = rds_dem_plot_ready.where(mask_not_building)
    return rds_dem_plot_ready


# TODO
# - include the boundary condition shapefile in plots; add an option to include a callout with a description


def print_json_file_tree(
    source: Union[str, Path, dict], base_dir: Optional[Union[str, Path]] = None
) -> None:

    # -------------------------
    # Load JSON or dict
    # -------------------------
    if isinstance(source, (str, Path)):
        with Path(source).open("r") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError("source must be a path to a JSON file or a dict")

    # -------------------------
    # Extract all Path objects
    # -------------------------
    # def is_path_like(s: str) -> bool:
    #     return os.path.isabs(s) or "/" in s or "\\" in s or Path(s).suffix != ""

    def extract_paths(obj) -> list[Path]:
        if isinstance(obj, Path):
            return [obj.expanduser()]
        # if isinstance(obj, str) and is_path_like(obj):
        #     return [Path(obj).expanduser()]
        elif isinstance(obj, dict):
            paths = []
            for v in obj.values():
                if isinstance(v, Path):
                    paths.extend(extract_paths(v))
            return paths
        elif isinstance(obj, list):
            paths = []
            for v in obj:
                if isinstance(v, Path):
                    paths.extend(extract_paths(v))
            return paths
        return []

    all_paths = extract_paths(data)
    if not all_paths:
        print("(no paths found)")
        return

    # -------------------------
    # Auto-detect base directory
    # -------------------------
    if base_dir is None:
        common_parts = list(zip(*(p.parts for p in all_paths if p.parts)))
        root_parts = []
        for parts in common_parts:
            if len(set(parts)) == 1:
                root_parts.append(parts[0])
            else:
                break
        base_dir = Path(*root_parts)
    base_dir = Path(base_dir)

    # -------------------------
    # Build tree with full Path objects
    # -------------------------
    class Node:
        def __init__(self, path: Path, is_file: bool):
            self.path = path
            self.is_file = is_file
            self.children = {}

    root_node = Node(base_dir, is_file=False)

    for p in all_paths:
        try:
            rel = p.relative_to(base_dir)
        except ValueError:
            continue
        current = root_node
        for i, part in enumerate(rel.parts):
            is_leaf = i == len(rel.parts) - 1
            sub_path = current.path / part
            if part not in current.children:
                current.children[part] = Node(
                    sub_path, is_file=is_leaf and sub_path.is_file()
                )
            current = current.children[part]

    # -------------------------
    # Print tree
    # -------------------------
    def print_node(node: Node, prefix=""):
        items = list(node.children.items())
        for i, (name, child) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "

            display_name = name
            if child.is_file:
                if not child.path.exists():
                    display_name += " [MISSING]"
            else:  # folder
                if not child.path.exists():
                    display_name += "/ [MISSING]"
                elif not any(child.path.iterdir()):
                    display_name += "/ [EMPTY]"
                else:
                    display_name += "/"

            print(prefix + connector + display_name)

            if child.children:
                extension = "    " if is_last else "│   "
                print_node(child, prefix + extension)

    print(f"{base_dir}/")
    print_node(root_node)
