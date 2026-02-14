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

import pandas as pd
import numpy as np
from pathlib import Path
import swmmio
from scipy.stats import rankdata
from typing import TYPE_CHECKING

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

    def create_hydraulics_model_from_template(
        self, swmm_model_template, destination: Path
    ) -> None:
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

    def update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
        self, verbose: bool = False
    ) -> None:
        """
        Update SWMM hydraulics model structure to match DEM grid for TRITON-SWMM coupling.

        Ensures only one inflow node per DEM grid cell by identifying the most
        downstream node in each cell and removing inflow assignments from other nodes.
        This prevents duplicate inflow assignments in TRITON-SWMM coupling where
        surface runoff from each DEM cell is routed to the drainage network.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detailed progress messages (default: False)
        """
        from .scenario import return_df_of_nodes_grouped_by_DEM_gridcell, calc_area

        dem_processed = self.system.sys_paths.dem_processed

        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scenario.scen_paths.swmm_hydraulics_inp, dem_processed, verbose=verbose
        )
        lst_grps_more_than_1_node = []
        count = -1
        for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
            keys = list(group.node_key)
            if len(group) > 1:
                count += 1
                group["grid_id"] = count
                lst_grps_more_than_1_node.append(group)
        if verbose:
            print(
                "There are {} gridcells with more than 1 node.".format(
                    len(lst_grps_more_than_1_node)
                )
            )
        df_overlapping_nodes = pd.concat(lst_grps_more_than_1_node)
        model = swmmio.Model(str(self.scenario.scen_paths.swmm_hydraulics_inp))
        links = model.links.dataframe
        nodes = model.nodes.dataframe
        inflows = model.inp.inflows
        all_upstream_nodes_to_drop = []
        lst_ambiguous_nodes = []
        for grid_id, group in df_overlapping_nodes.groupby(["grid_id"]):
            keys = list(group.node_key)
            downstream_links = links[links.InletNode.isin(keys)]
            upstream_nodes_to_drop = []
            if len(downstream_links) == 0:
                # this means that these two nodes do not have downstream conduits
                node_to_keep = inflows[inflows.index.isin(keys)].index.values
                if len(node_to_keep) == 1:
                    # unambiguous
                    upstream_nodes_to_drop = upstream_nodes_to_drop + list(
                        np.asarray(keys)[np.asarray(keys) != node_to_keep[0]]
                    )
                else:
                    node_to_keep = find_lowest_inv(node_to_keep, nodes)
                    if len(node_to_keep) > 1:
                        if verbose:
                            print("###########################################")
                            print(
                                "Warning: node selection ambiguous for nodes {} even after trying to select based on the lowest invert elevation. Choose the first node in the list...".format(
                                    list(node_to_keep)
                                )
                            )
                            print("###########################################")
                        if node_to_keep not in lst_ambiguous_nodes:
                            lst_ambiguous_nodes.append(node_to_keep)
                    node_to_keep = node_to_keep[0]
            else:
                keys = list(downstream_links.InletNode)
                # compute link area
                lst_areas = []
                for id, row in downstream_links.iterrows():
                    lst_areas.append(calc_area(row))
                # rank pipe area
                ranks = rankdata(lst_areas, method="min")
                # find unique ranks
                df_rank_counts = pd.DataFrame(np.unique(ranks, return_counts=True)).T
                df_rank_counts.columns = ["rank", "count"]
                df_rank_counts = df_rank_counts.set_index("rank")
                # add smaller nodes to the list of nodes to drop
                upstream_nodes_to_drop += list(np.asarray(keys)[ranks != max(ranks)])
                if df_rank_counts.loc[max(ranks), "count"] > 1:
                    downstream_links_tied = downstream_links[ranks == max(ranks)]
                    # if a node appears as an outlet node, drop it
                    downstream_links_w_outlets = downstream_links_tied[
                        downstream_links_tied.OutletNode.isin(keys)
                    ]
                    upstream_nodes_to_drop += list(
                        downstream_links_w_outlets.InletNode.values
                    )
                    # remove nodes that don't have an outlet node
                    remaining_dwnstrm_links = downstream_links_tied[
                        ~downstream_links_tied.InletNode.isin(upstream_nodes_to_drop)
                    ]
                    node_to_keep = remaining_dwnstrm_links.InletNode.values
                    if len(node_to_keep) > 1:
                        node_to_keep = find_lowest_inv(node_to_keep, nodes)
                        if len(node_to_keep) > 1:
                            if verbose:
                                print("###########################################")
                                print(
                                    "Warning: node selection ambiguous for nodes {} even after trying to select based on the lowest invert elevation. Choose the first node in the list...".format(
                                        list(node_to_keep)
                                    )
                                )
                                print("###########################################")
                        node_to_keep = node_to_keep[0]
                        upstream_nodes_to_drop = downstream_links[
                            downstream_links.InletNode != node_to_keep
                        ].InletNode.values
                else:
                    idx_largest_pipe = np.argmax(lst_areas)
                    node_to_keep = keys[idx_largest_pipe]
                    upstream_nodes_to_drop = downstream_links[
                        downstream_links.InletNode != node_to_keep
                    ].InletNode.values
            all_upstream_nodes_to_drop = all_upstream_nodes_to_drop + list(
                upstream_nodes_to_drop
            )
        if verbose:
            print("Removing {} inflow nodes.".format(len(all_upstream_nodes_to_drop)))
        # write new swmm model, removing outflow that won't be used by TRITON-SWMM
        with open(self.scenario.scen_paths.swmm_hydraulics_inp, "r") as fp:
            lines = fp.readlines()
        for idx_line, line in enumerate(lines):
            if "[INFLOWS]" in line:
                inflows_section_first_line_num = idx_line
            if "[CURVES]" in line:
                inflows_section_last_line_num = idx_line
        line_nums_to_remove = []
        lines_to_remove = []
        for inflow_line in np.arange(
            inflows_section_first_line_num, inflows_section_last_line_num + 1
        ):
            line = lines[inflow_line]
            node_id = line.split("    ")[0]
            if node_id in all_upstream_nodes_to_drop:
                line_nums_to_remove.append(inflow_line)
                lines_to_remove.append(line)
        # overwrite hydaulics model
        with open(self.scenario.scen_paths.swmm_hydraulics_inp, "w") as fp:
            for number, line in enumerate(lines):
                if number not in line_nums_to_remove:
                    fp.write(line)
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

        with open(inp_file_path, "r") as fp:
            lines = fp.readlines()

        # Find and replace THREADS line in [OPTIONS] section
        in_options_section = False
        threads_found = False

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
                threads_found = True
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
        import rioxarray as rxr
        import geopandas as gpd
        from .scenario import (
            extract_vertex_coordinates,
            infer_side,
            find_closest_dem_coord,
        )

        weather_timeseries = self.cfg_analysis.weather_timeseries
        weather_event_indexers = self.scenario.weather_event_indexers
        weather_time_series_storm_tide_datavar = (
            self.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        simulation_folders = self.scenario._analysis.analysis_paths.simulation_directory
        storm_tide_units = self.cfg_analysis.storm_tide_units

        dem_processed = self.system.sys_paths.dem_processed
        storm_tide_boundary_line_gis = self.cfg_analysis.storm_tide_boundary_line_gis
        ds_event_ts = self.scenario.ds_event_ts
        df_water_levels = (
            ds_event_ts[weather_time_series_storm_tide_datavar]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )[weather_time_series_storm_tide_datavar].to_frame()

        tseries_diff_hrs = pd.Series(df_water_levels.index).diff().dt.seconds / 60 / 60  # type: ignore
        tseries_diff_hrs.loc[0] = 0

        df_water_levels["time_hr"] = tseries_diff_hrs.cumsum().values
        df_water_levels.set_index("time_hr", inplace=True)

        sim_id_str = self.scenario.sim_id_str

        wlevel_first_line = f"%{sim_id_str} Water Level Boundary Condition\n"
        wlevel_second_line = f"%Time(hr) water_elevation ({storm_tide_units})\n"

        with open(self.scenario.scen_paths.extbc_tseries, "w") as f:
            f.write(wlevel_first_line + wlevel_second_line)

        df_water_levels.to_csv(
            self.scenario.scen_paths.extbc_tseries, mode="a", header=False
        )

        self.scenario.log.extbc_tseries_created.set(True)
        # write external boundary condition location file
        rds_dem = rxr.open_rasterio(dem_processed)
        gdf_bc = gpd.read_file(storm_tide_boundary_line_gis)  # type: ignore
        str_line1 = "% BC Type, X1, Y1, X2, Y2, BC"
        gdf_row = gdf_bc.loc[0, :]
        vertices = extract_vertex_coordinates(gdf_row.geometry)
        lst_x = []
        lst_y = []
        for vertex in vertices:  # type: ignore
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


def find_lowest_inv(node_to_keep, nodes):
    """
    Find nodes with the lowest invert elevation.

    Helper function for identifying the most downstream node when multiple nodes
    fall within the same DEM grid cell.

    Parameters
    ----------
    node_to_keep : array-like
        Node IDs to evaluate
    nodes : DataFrame
        DataFrame containing node information including InvertElev

    Returns
    -------
    list
        Node IDs with the lowest invert elevation
    """
    from .scenario import find_lowest_inv as scenario_find_lowest_inv

    return scenario_find_lowest_inv(node_to_keep, nodes)
