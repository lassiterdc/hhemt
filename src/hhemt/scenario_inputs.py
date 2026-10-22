"""
Scenario Input Orchestration and SWMM Hydraulics Module

This module orchestrates scenario preparation and handles creation of the SWMM hydraulics
model, which is a direct input to TRITON-SWMM (specified in the TRITON-SWMM .cfg file).

The ScenarioInputGenerator class coordinates:
1. SWMM runoff modeling (via SWMMRunoffModeler) - generates hydrograph inputs
2. SWMM hydraulics model creation and modification - direct TRITON-SWMM input
3. TRITON boundary condition file generation
4. Full SWMM model creation (optional, via SWMMFullModelBuilder)
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import swmmio

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


class ScenarioInputGenerator:
    """
    Orchestrates scenario input preparation for TRITON-SWMM.

    This class coordinates the creation of all input files needed for a TRITON-SWMM
    scenario, including:
    - SWMM hydraulics model (.inp file) - direct input to TRITON-SWMM
    - TRITON boundary condition files (extbc)
    - Coordination with SWMMRunoffModeler for hydrograph generation
    - Coordination with SWMMFullModelBuilder for optional full SWMM models

    Attributes
    ----------
    scenario : TRITONSWMM_scenario
        Reference to the parent scenario object
    cfg_analysis : AnalysisConfig
        Analysis configuration settings
    system : TRITONSWMM_system
        System configuration and paths
    """

    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        """
        Initialize the ScenarioInputGenerator.

        Parameters
        ----------
        scenario : TRITONSWMM_scenario
            The parent scenario object containing configuration and paths
        """
        self.scenario = scenario
        self.cfg_analysis = scenario._analysis.cfg_analysis
        self.system = scenario._system

    def create_hydraulics_model_from_template(self, swmm_model_template, destination: Path) -> None:
        """
        Create SWMM hydraulics model from template - direct input to TRITON-SWMM.

        This creates the SWMM hydraulics .inp file that is specified in the TRITON-SWMM
        configuration file. This model defines the drainage network structure that
        TRITON-SWMM will use for coupled surface-subsurface flow simulation.

        Fills in template placeholders with scenario-specific values including
        time series data, rain gauges, simulation timing, and reporting intervals.

        Parameters
        ----------
        swmm_model_template : Path
            Path to the SWMM hydraulics template file
        destination : Path
            Path where the filled template should be written (typically hydraulics.inp)
        """
        from .swmm_utils import create_swmm_inp_from_template

        create_swmm_inp_from_template(self.scenario, swmm_model_template, destination)
        return

    def update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(self, verbose: bool = False) -> None:
        """
        Choose which ``[INFLOWS]`` nodes of the hydraulics model TRITON-SWMM couples to the surface.

        TRITON-SWMM couples to the surface exactly the nodes listed in ``[INFLOWS]``, so this
        step decides the coupled set by deleting inflow lines. Every DEM cell holding an
        inflow-carrying node is resolved by ``scenario.resolve_cell_inflow_nodes``: junctions
        stay coupled; outfalls, storage nodes and dividers are dropped, named together in one
        warning; and a cell whose inflow nodes are all of those types stops preparation with a
        named error. With ``one_coupled_inflow_node_per_dem_cell`` set, each cell keeps a single
        junction chosen by pipe size, and every prepared scenario records each cell that had
        more than one inflow node in ``_status/inflow_node_selection.json``.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detailed progress messages (default: False)
        """
        import json
        import warnings

        from .scenario import (
            inp_link_table,
            inp_node_table,
            remove_inflow_lines,
            resolve_cell_inflow_nodes,
            return_df_of_nodes_grouped_by_DEM_gridcell,
        )

        f_inp = self.scenario.scen_paths.swmm_hydraulics_inp
        one_per_cell = bool(self.cfg_analysis.one_coupled_inflow_node_per_dem_cell)
        df_node_locs, _lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            f_inp, self.system.sys_paths.dem_processed, verbose=verbose
        )
        model = swmmio.Model(str(f_inp))
        nodes = inp_node_table(model)
        links = inp_link_table(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inflows = model.inp.inflows
        inflow_node_ids = [] if inflows is None else list(inflows.index.astype(str))

        drop: list[str] = []
        ineligible_dropped: list[str] = []
        selection_cells: list[dict] = []
        for (dem_x, dem_y), group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
            resolution = resolve_cell_inflow_nodes(
                list(group.node_key), nodes, links, inflow_node_ids, one_per_cell=one_per_cell
            )
            drop.extend(resolution.dropped)
            ineligible_dropped.extend(
                f"{n} ({resolution.node_types[n].lower()})"
                for n in resolution.dropped
                if resolution.drop_reasons[n] == "eligibility"
            )
            if one_per_cell and len(resolution.node_types) > 1:
                selection_cells.append(
                    {
                        "dem_x_coord": float(dem_x),
                        "dem_y_coord": float(dem_y),
                        "inflow_nodes": [{"node": n, "type": t} for n, t in resolution.node_types.items()],
                        "kept": resolution.kept[0] if resolution.kept else None,
                        "dropped": [{"node": n, "reason": resolution.drop_reasons[n]} for n in resolution.dropped],
                    }
                )
        if ineligible_dropped:
            warnings.warn(
                f"{f_inp}: [INFLOWS] lines removed for nodes TRITON-SWMM cannot couple, each sharing "
                f"a DEM cell with a junction that stays coupled: {ineligible_dropped}",
                stacklevel=2,
            )
        if verbose:
            print(f"Removing {len(drop)} inflow nodes.")
        remove_inflow_lines(f_inp, drop)
        if one_per_cell:
            # Written for EVERY scenario prepared with the option on, empty when no cell was
            # reduced, so that an ABSENT file means "not prepared by this build" to the report's
            # inflow-node reduction disclosure rather than reading as "nothing reduced".
            record = self.scenario.scen_paths.sim_folder / "_status" / "inflow_node_selection.json"
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"cells": selection_cells}, indent=2) + "\n")
        self.scenario.log.inflow_nodes_in_hydraulic_inp_assigned = True
        return

    def update_swmm_threads_in_inp_file(self, inp_file_path: Path) -> None:
        """
        Update THREADS parameter in SWMM .inp file OPTIONS section.

        Modifies the THREADS option in the [OPTIONS] section to match the
        n_omp_threads configuration parameter. This enables dynamic control
        of SWMM threading for performance tuning and benchmarking studies.

        Parameters
        ----------
        inp_file_path : Path
            Path to the SWMM .inp file to modify

        Notes
        -----
        - Preserves all other OPTIONS parameters unchanged
        - Uses line-by-line replacement to maintain file structure
        - If THREADS parameter not found, no modification occurs (backward compatible)
        """
        n_threads = self.cfg_analysis.n_omp_threads

        with open(inp_file_path) as fp:
            lines = fp.readlines()

        # Find and replace THREADS line in [OPTIONS] section
        in_options_section = False

        for idx, line in enumerate(lines):
            # Track when we enter/exit OPTIONS section
            if "[OPTIONS]" in line:
                in_options_section = True
                continue
            elif line.startswith("[") and in_options_section:
                # Left OPTIONS section
                break

            # Replace THREADS line if found
            if in_options_section and line.strip().startswith("THREADS"):
                # Preserve spacing format: "THREADS              {value}"
                lines[idx] = f"THREADS              {n_threads}\n"
                break

        # Write back modified file
        with open(inp_file_path, "w") as fp:
            fp.writelines(lines)

        # Note: If THREADS not found, file remains unchanged (backward compatible)
        return

    def create_external_boundary_condition_files(self) -> None:
        """
        Create boundary condition files for TRITON.

        Generates two files for TRITON-SWMM external boundary conditions:
        1. Time series file with water level boundary conditions
        2. Location file specifying where boundary conditions apply on the DEM

        Updates the scenario log to indicate files were created successfully.
        """
        import geopandas as gpd
        import rioxarray as rxr

        from .scenario import (
            extract_vertex_coordinates,
            find_closest_dem_coord,
            infer_side,
        )

        weather_time_series_storm_tide_datavar = self.cfg_analysis.weather_time_series_storm_tide_datavar
        storm_tide_units = self.cfg_analysis.storm_tide_units

        dem_processed = self.system.sys_paths.dem_processed
        storm_tide_boundary_line_gis = self.cfg_analysis.storm_tide_boundary_line_gis
        ds_event_ts = self.scenario.ds_event_ts
        df_water_levels = (ds_event_ts[weather_time_series_storm_tide_datavar].reset_coords(drop=True).to_dataframe())[
            weather_time_series_storm_tide_datavar
        ].to_frame()

        # .dt.seconds is the SECONDS COMPONENT (0-86399) and silently wraps on a
        # gap of >= 24 h; .dt.total_seconds() is the non-wrapping form. Contiguous
        # steps are unaffected. The .dropna() that USED to justify this was removed by
        # the [Q85] scrub, but the choice stands on a second ground: a non-uniform
        # weather axis can still open a >= 24 h gap here, and the uniformity assertion
        # in _weather_step_seconds fires from the post-write window-agreement guard,
        # AFTER this writer has already run. Do not narrow this back to .dt.seconds.
        tseries_diff_hrs = pd.Series(df_water_levels.index).diff().dt.total_seconds() / 60 / 60
        tseries_diff_hrs.loc[0] = 0

        df_water_levels["time_hr"] = tseries_diff_hrs.cumsum().values
        df_water_levels.set_index("time_hr", inplace=True)

        sim_id_str = self.scenario.sim_id_str

        wlevel_first_line = f"%{sim_id_str} Water Level Boundary Condition\n"
        wlevel_second_line = f"%Time(hr) water_elevation ({storm_tide_units})\n"

        with open(self.scenario.scen_paths.extbc_tseries, "w") as f:
            f.write(wlevel_first_line + wlevel_second_line)

        df_water_levels.to_csv(self.scenario.scen_paths.extbc_tseries, mode="a", header=False)

        self.scenario.log.extbc_tseries_created.set(True)
        # write external boundary condition location file
        rds_dem = rxr.open_rasterio(dem_processed)
        gdf_bc = gpd.read_file(storm_tide_boundary_line_gis)
        str_line1 = "% BC Type, X1, Y1, X2, Y2, BC"
        gdf_row = gdf_bc.loc[0, :]
        vertices = extract_vertex_coordinates(gdf_row.geometry)
        lst_x = []
        lst_y = []
        for vertex in vertices:
            lst_x.append(vertex[0])
            lst_y.append(vertex[1])
        # find x and ys at edge of DEM representing the boundary condition
        min_x = min(lst_x)
        min_y = min(lst_y)
        max_x = max(lst_x)
        max_y = max(lst_y)
        BC_side = infer_side(rds_dem, min_x, max_x, min_y, max_y)
        x1, y1 = find_closest_dem_coord(min_x, min_y, BC_side, rds_dem)
        x2, y2 = find_closest_dem_coord(max_x, max_y, BC_side, rds_dem)
        BC_with_quotes = f'"{self.scenario.scen_paths.extbc_tseries.name}"'
        str_line2 = "{},{},{},{},{},{}".format("1", x1, y1, x2, y2, BC_with_quotes)
        # write file

        fpath_extbc = self.scenario.scen_paths.extbc_loc
        with open(fpath_extbc, "w") as f:
            f.write(str_line1 + "\n")
            f.write(str_line2 + "\n")
        self.scenario.log.extbc_loc_created.set(True)
        return
