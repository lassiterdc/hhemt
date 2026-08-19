"""
SWMM Runoff Modeling Module

This module handles SWMM hydrology-only modeling to generate runoff hydrographs
that serve as inputs to TRITON-SWMM. This includes:
- Creating rainfall and water level input files for SWMM
- Creating and running SWMM hydrology-only models
- Extracting runoff hydrographs from SWMM output for TRITON-SWMM input
"""

import pandas as pd
import sys
from pathlib import Path
from pyswmm import Simulation, Output
from swmm.toolkit.shared_enum import NodeAttribute
from typing import TYPE_CHECKING

from hhemt.exceptions import ProcessingError

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


def _assert_validated_swmm_stack() -> None:
    """Fail closed before executing SWMM unless the installed pyswmm / swmm-toolkit
    is the conda-forge-validated pairing (pyswmm 2.x + swmm-toolkit 0.15.x).

    The validated SWMM engine is distributable only via conda-forge: swmmio 0.8.5
    caps pyswmm<2.0 (unsatisfiable with the pyswmm 2.x prepare_scenario requires),
    and pip metadata cannot express the ``--no-deps`` install environment.yaml uses,
    so ``pip install hhemt`` resolves an UNVALIDATED SWMM stack whose C extension the
    project cannot certify against the observed free()/SIGABRT teardown crash. Refuse
    to run rather than risk silent heap corruption. Override at your own risk with
    HHEMT_ALLOW_UNVALIDATED_SWMM_STACK=1.
    """
    import os
    from importlib.metadata import PackageNotFoundError, version

    if os.environ.get("HHEMT_ALLOW_UNVALIDATED_SWMM_STACK") == "1":
        return
    problems: list[str] = []
    for pkg, want in (("pyswmm", "2."), ("swmm-toolkit", "0.15.")):
        try:
            got = version(pkg)
        except PackageNotFoundError:
            problems.append(f"{pkg} not installed (need {want}x)")
            continue
        if not got.startswith(want):
            problems.append(f"{pkg}=={got} (need {want}x)")
    if problems:
        raise RuntimeError(
            "Refusing to execute SWMM on an unvalidated Python stack ("
            + "; ".join(problems)
            + "). The validated engine ships only via conda-forge; install per "
            "docs/how-to/installation.md Option A (`conda env create -f "
            "environment.yaml` plus the two --no-deps post-create steps). Set "
            "HHEMT_ALLOW_UNVALIDATED_SWMM_STACK=1 to bypass at your own risk."
        )


class SWMMRunoffModeler:
    """
    Handles SWMM hydrology modeling for generating runoff inputs to TRITON-SWMM.

    This class manages the complete workflow for using SWMM to model rainfall-runoff
    processes and generate hydrograph inputs for TRITON-SWMM:
    1. Creates rainfall and water level .dat files for SWMM input
    2. Creates SWMM hydrology-only model from template
    3. Executes SWMM hydrology model
    4. Extracts runoff hydrographs and formats them for TRITON-SWMM

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
        Initialize the SWMMRunoffModeler.

        Parameters
        ----------
        scenario : TRITONSWMM_scenario
            The parent scenario object containing configuration and paths
        """
        self.scenario = scenario
        self.cfg_analysis = scenario._analysis.cfg_analysis
        self.system = scenario._system

    def write_swmm_rainfall_dat_files(self) -> None:
        """
        Generate rainfall input files from weather data for SWMM.

        Creates .dat files for each rain gauge with time series data formatted
        for SWMM input. Files are written to the scenario's weather data directory.
        Updates the scenario log with paths to created files.
        """
        weather_timeseries = self.cfg_analysis.weather_timeseries
        weather_event_indexers = self.scenario.weather_event_indexers
        subcatchment_raingage_mapping = (
            self.system.cfg_system.subcatchment_raingage_mapping
        )
        subcatchment_raingage_mapping_gage_id_colname = (
            self.system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        rainfall_units = self.cfg_analysis.rainfall_units

        df_sub_raingage_mapping = pd.read_csv(subcatchment_raingage_mapping)  # type: ignore
        sim_id_str = self.scenario.sim_id_str

        # retreieve dataframe of rainfall time series
        df_allrain = (
            self.scenario.ds_event_ts[
                df_sub_raingage_mapping[subcatchment_raingage_mapping_gage_id_colname]
                .unique()
                .astype(str)
            ]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )
        dic_rain_paths = dict()

        for gage in df_allrain:
            s_rain = df_allrain[gage]
            df_rain = pd.DataFrame(
                dict(
                    date=df_allrain.index.strftime("%m/%d/%Y"),  # type: ignore
                    time=df_allrain.index.time,  # type: ignore
                    rain=s_rain,
                )
            )
            # define filepaths and write
            fname_raindat = "grid-ind{}.dat".format(gage)
            f_out_swmm_rainfall = (
                self.scenario.scen_paths.dir_weather_datfiles / fname_raindat
            )
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(f";;rain gage {gage} for sim {sim_id_str} \n")
                file.write(f";;Rainfall ({rainfall_units})\n")
            df_rain.to_csv(
                f_out_swmm_rainfall, sep="\t", index=False, header=False, mode="a"
            )
            dic_rain_paths[str(gage)] = f_out_swmm_rainfall

        self.scenario.log.swmm_rainfall_dat_files.set(dic_rain_paths)
        return

    def write_swmm_waterlevel_dat_files(self) -> None:
        """
        Generate water level input files for SWMM boundary conditions.

        Creates a .dat file with storm tide/water level time series formatted
        for SWMM input. Updates the scenario log with the path to the created file.
        """
        storm_tide_units = self.cfg_analysis.storm_tide_units
        weather_time_series_storm_tide_datavar = (
            self.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        weather_timeseries = self.cfg_analysis.weather_timeseries
        weather_event_indexers = self.scenario.weather_event_indexers

        sim_id_str = self.scenario.sim_id_str

        s_wlevel = (
            self.scenario.ds_event_ts[weather_time_series_storm_tide_datavar]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )[weather_time_series_storm_tide_datavar]

        fname_wleveldat = "waterlevel.dat"

        f_out_swmm_wlevel = (
            self.scenario.scen_paths.dir_weather_datfiles / fname_wleveldat
        )

        # create data frame with proper formatting to be read in SWMM
        df_wlevel = pd.DataFrame(
            dict(
                date=s_wlevel.index.strftime("%m/%d/%Y"),  # type: ignore
                time=s_wlevel.index.time,  # type: ignore
                water_level_m=s_wlevel,
            )
        )

        with open(f_out_swmm_wlevel, "w+") as file:
            file.write(f";;water level for sim {sim_id_str}\n")
            file.write(f";;Water Level ({storm_tide_units})\n")

        df_wlevel.to_csv(
            f_out_swmm_wlevel, sep="\t", index=False, header=False, mode="a"
        )
        self.scenario.log.storm_tide_for_swmm.set(f_out_swmm_wlevel)
        return

    def create_hydrology_model_from_template(
        self, swmm_model_template, destination: Path
    ) -> None:
        """
        Create SWMM hydrology-only model from template file.

        Fills in template placeholders with scenario-specific values including
        time series data, rain gauges, simulation timing, and reporting intervals.

        Parameters
        ----------
        swmm_model_template : Path
            Path to the SWMM hydrology template file
        destination : Path
            Path where the filled template should be written (typically hydro.inp)
        """
        from .swmm_utils import create_swmm_inp_from_template

        create_swmm_inp_from_template(self.scenario, swmm_model_template, destination)
        return

    def run_swmm_hydro_model(
        self, rerun_if_exists: bool = False, verbose: bool = False
    ) -> None:
        """
        Execute SWMM hydrology-only model to generate runoff hydrographs.

        Runs the hydrology-only SWMM model to generate runoff hydrographs.
        Can skip execution if outputs already exist.

        Parameters
        ----------
        rerun_if_exists : bool, optional
            If True, rerun even if outputs exist (default: False)
        verbose : bool, optional
            If True, print status messages (default: False)
        """
        sim_complete = self.scenario.log.hydro_swmm_sim_completed.get() is True
        if (not sim_complete) or rerun_if_exists:
            self.scenario.log.hydro_swmm_sim_completed.set(False)
            _assert_validated_swmm_stack()
            with Simulation(str(self.scenario.scen_paths.swmm_hydro_inp)) as sim:
                sim.execute()
            self.scenario.log.hydro_swmm_sim_completed.set(True)
        else:
            if verbose:
                print("Hydrology-only SWMM model already executed. Not re-running.")
        return

    def _write_node_inflow_capture(self, d_node_capture, d_node_gridcell, times_hr, flow_units) -> None:
        """Persist the PER-NODE inflow series that tseries.hyg destroys.

        WHY PER-NODE. `tseries.hyg` writes one column per DEM GRIDCELL -- the member
        nodes are summed away one line before the write -- so the file cannot answer
        "total runoff volume per node location", which is the quantity that was asked
        for. Capturing pre-sum also makes the float32 storage EXACTLY lossless: every
        value is an unmodified SWMM REAL4 (`#define REAL4 float`, solver/output.c:50),
        whereas the post-sum column is a float64 sum of float32s and float32 storage of
        THAT rounds it.

        Writes nothing when no node carried positive inflow -- legal for a scenario
        whose gridcells are all dry, and the reason this is a guard rather than an
        unconditional write.
        """
        import numpy as np
        import xarray as xr

        from hhemt.du_sentinels import restamp_parent_sentinels

        if not d_node_capture or times_hr is None:
            return
        out_path = Path(self.scenario.scen_paths.sim_folder) / "processed" / "hydrology_inflow_summary.zarr"
        node_ids = sorted(d_node_capture)
        inflow = np.asarray(
            [np.asarray(d_node_capture[n], dtype=_HYDROGRAPH_CAPTURE_DTYPE) for n in node_ids],
            dtype=_HYDROGRAPH_CAPTURE_DTYPE,
        )
        t_hr = np.asarray(times_hr, dtype="float64")
        # Volume integrates cms over HOURS, so the 3600 is a unit conversion, not a
        # fudge. float64 here deliberately: the reduction accumulates over ~2,000
        # timesteps and is not itself a stored SWMM value.
        # `np.trapezoid` is the NumPy>=2.0 spelling and `np.trapz` the <2.0 one;
        # this env pins numpy 1.26.4, where `trapezoid` does not exist, and the
        # project supports py3.11-3.12 across both numpy majors. Resolve by
        # attribute rather than by version test so neither spelling is a hard
        # floor -- `trapz` is deprecated in 2.x but still present, and
        # `trapezoid` is absent in 1.x, so first-available is the only form
        # correct on both.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        volume_m3 = _trapz(inflow.astype("float64"), x=t_hr, axis=1) * 3600.0
        ds = xr.Dataset(
            data_vars={
                "inflow_cms": (("node_id", "time_hr"), inflow),
                "total_inflow_volume_m3": (("node_id",), volume_m3),
            },
            coords={
                "node_id": np.asarray(node_ids, dtype=str),
                "time_hr": t_hr.astype("float32"),
                "dem_x_coord": (("node_id",), np.asarray([d_node_gridcell[n][0] for n in node_ids])),
                "dem_y_coord": (("node_id",), np.asarray([d_node_gridcell[n][1] for n in node_ids])),
            },
            attrs={
                "flow_units": str(flow_units),
                "system_total_inflow_volume_m3": float(volume_m3.sum()),
                "n_gridcells": int(len({tuple(v) for v in d_node_gridcell.values()})),
                "notes": (
                    "Per-NODE TOTAL_INFLOW captured pre-gridcell-sum from swmm/hydro.out. "
                    "float32 is exactly lossless w.r.t. SWMM's REAL4 binary. Supports "
                    "CONTENT-identical regeneration of strmflow/tseries.hyg (values "
                    "byte-recoverable; gridcell grouping re-derivable from the DEM and "
                    "hydro.inp) -- NOT byte-identical TEXT regeneration, which has not "
                    "been round-tripped."
                ),
            },
        )
        ds = ds.assign_coords(event_iloc=self.scenario.event_iloc).expand_dims("event_iloc")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(out_path, mode="w")
        restamp_parent_sentinels(out_path, analysis_dir=self.scenario._analysis.analysis_paths.analysis_dir)  # PATTERN B

    def write_hydrograph_files(self) -> None:
        """
        Extract runoff hydrographs from SWMM output and format for TRITON-SWMM.

        Extracts runoff from SWMM hydrology model output and creates:
        1. Time series file with discharge hydrographs for each DEM grid cell
        2. Location file mapping hydrographs to DEM coordinates

        These files serve as inflow boundary conditions for TRITON-SWMM.
        Updates the scenario log to indicate files were created successfully.
        """
        import rioxarray as rxr
        from .scenario import return_df_of_nodes_grouped_by_DEM_gridcell

        if hydrograph_outputs_gate(self.scenario) == "skipped":
            return

        dem_processed = self.system.sys_paths.dem_processed

        sim_id_str = self.scenario.sim_id_str
        hydro_outfile = str(self.scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out")
        rds_dem = rxr.open_rasterio(dem_processed)
        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scenario.scen_paths.swmm_hydro_inp, dem_processed
        )

        d_time_series = dict()
        # Bound HERE, not described in prose beside the spec that loads them. Round 6
        # halted on exactly this: S5's change block referenced both names without
        # binding either, and because the block is an indented fragment that ast.parse
        # rejects, the free-name gate skipped it silently for five rounds.
        d_node_capture: dict = {}
        d_node_gridcell: dict = {}
        lst_nodes_with_inflow = []
        with Output(hydro_outfile) as out:
            flow_units = out.units["flow"]  # type: ignore
            inflow_first_line = f"%Runoff for sim {sim_id_str}\n"
            inflow_second_line = f"%Time(hr) Discharge ({flow_units})\n"
            need_to_create_time_series = True
            for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
                keys = list(
                    group.node_key
                )  # list of node ids that fall within a single gridcell
                d_flows = {}
                for key in keys:
                    if key not in lst_outfalls:
                        d_inflow = pd.Series(
                            out.node_series(key, NodeAttribute.TOTAL_INFLOW)  # type: ignore
                        )
                        if (
                            need_to_create_time_series
                        ):  # create first column with time in hours
                            tseries = pd.Series(d_inflow.index).diff().dt.seconds / 60 / 60  # type: ignore
                            tseries.iloc[0] = 0
                            d_time_series["time_hr"] = tseries.cumsum().values
                            need_to_create_time_series = False
                        # create dataframe with the flow of all nodes within the gridcell
                        if d_inflow.sum() > 0:  # type: ignore
                            lst_nodes_with_inflow.append(key)
                            d_flows[key] = d_inflow.values
                            # PER-NODE capture, taken here rather than downstream, and the
                            # position is load-bearing. One line below, the member nodes of
                            # a gridcell are SUMMED (df_flows.sum(axis=1)) and the per-node
                            # identity is gone -- tseries.hyg is per-GRIDCELL, so its 821
                            # columns are centroids, not nodes, and it cannot answer "total
                            # runoff volume per node location". These values are unmodified
                            # SWMM REAL4 (`#define REAL4 float`, solver/output.c:50), so
                            # storing them float32 is EXACTLY lossless; the post-sum column
                            # is a float64 sum of float32s and is not.
                            d_node_capture[key] = d_inflow.values
                            d_node_gridcell[key] = coords
                # combine time series into a dataframe
                if len(d_flows) > 0:
                    df_flows = pd.DataFrame(d_flows)
                    d_time_series[coords] = df_flows.sum(axis=1)
            # write hydrograph file
            df_node_inflow = pd.DataFrame(d_time_series)

            with open(self.scenario.scen_paths.hyg_timeseries, "w") as f:
                f.write(inflow_first_line + inflow_second_line)
            df_node_inflow.to_csv(
                self.scenario.scen_paths.hyg_timeseries,
                mode="a",
                index=False,
                header=False,
            )
            self.scenario.log.hyg_timeseries_created.set(True)
            # write hydrograph location file
            str_first_line = "%X-Location,Y-Location"
            with open(self.scenario.scen_paths.hyg_locs, "w") as f:
                f.write(str_first_line + "\n")
                for col in df_node_inflow.columns:
                    if "time" in col:  # skip column named time
                        continue
                    x = col[0]
                    y = col[1]
                    f.write("{},{}\n".format(x, y))
            self.scenario.log.hyg_locs_created.set(True)
            self._write_node_inflow_capture(
                d_node_capture,
                d_node_gridcell,
                d_time_series.get("time_hr"),
                flow_units,
            )
            # verifying that all nodes are within the DEM
            xllcorner = rds_dem.x.values.min()  # type: ignore
            yllcorner = rds_dem.y.values.min()  # type: ignore
            df_xylocs = pd.read_csv(
                self.scenario.scen_paths.hyg_locs, header=0, names=["x", "y"]
            )
            if df_xylocs.x.min() < xllcorner:
                print("problem with x's")
            elif df_xylocs.y.min() < yllcorner:
                print("problem with y's")
            else:
                pass
            # check to make sure dimensions are correct
            df_hyg_loc = pd.read_csv(self.scenario.scen_paths.hyg_locs)
            df_hyg_test = pd.read_csv(
                self.scenario.scen_paths.hyg_timeseries, skiprows=2
            )
            if ((df_hyg_test.shape[1] - 1) - df_hyg_loc.shape[0]) != 0:
                print("ERROR ENCOUNTERED IN SETTING UP INPUTS")
                print(
                    "The shapes of the hydrograph file and the hydrograph location file do not match up."
                )
                print(f"{Path(self.scenario.scen_paths.hyg_locs).parent}")
                print("df_hyg_test.shape")
                print(df_hyg_test.shape)
                print("df_hyg_loc.shape")
                print(df_hyg_loc.shape)
                print("df_hyg_test.head()")
                print(df_hyg_test.head())
                print("df_hyg_loc.head()")
                print(df_hyg_loc.head())
                sys.exit()
        return



#: Storage dtype for the per-node inflow capture. float32 is EXACTLY LOSSLESS here
#: and float64 is pure padding, which is a measurement rather than a preference:
#: SWMM's binary output writer declares `#define REAL4 float` (solver/output.c:50)
#: and stores node results through `REAL4* NodeResults` (:86), writing
#: sizeof(REAL4) records (:167). pyswmm's Output.node_series -- the sole source of
#: every value captured -- therefore yields values carrying 24 mantissa bits,
#: widened to Python float on the way out. float64 would record 29 guaranteed-zero
#: bits per value, costing ~1 MB/scenario (~3.8 GB over a 3,798-scenario ensemble)
#: for no information.
#:
#: SCOPE OF THE REVERSIBILITY CLAIM, precisely. The capture supports
#: CONTENT-identical regeneration of strmflow/tseries.hyg (values byte-recoverable,
#: gridcell grouping re-derivable from the DEM + hydro.inp, both persistent). It does
#: NOT establish BYTE-identical TEXT regeneration: pandas writes repr(float(x)) with
#: float_format=None, which is deterministic but whose round-trip has not been run.
#: Do not upgrade the claim without running it.
#:
#: The dtype is exact only for the PER-NODE series. The per-GRIDCELL column in
#: tseries.hyg is df_flows.sum(axis=1) -- a float64 sum of float32s -- and float32
#: storage of THAT quantity rounds it (<= 1 ulp). Capturing pre-sum is what makes
#: the lossless claim true.
_HYDROGRAPH_CAPTURE_DTYPE = "float32"


def hydrograph_outputs_gate(scenario) -> str:
    """Three-state already-written gate for the TRITON inflow hydrographs.

    Returns ``"skipped"`` when the hydrographs are logged AND present on disk (the fast
    path the hydro.out reclaim needs), or ``"fell_through"`` when the caller must build
    them. Raises ``ProcessingError`` when they are absent AND their rebuild source is
    absent too -- the one state in which scenario preparation cannot proceed at all.

    Module-level rather than inline in ``write_hydrograph_files`` so the three branches are
    testable without a prepared scenario; a gate reachable only through a function that
    needs a full fixture is a gate whose branches get asserted by proxy.
    """
    # WHY THIS GATE EXISTS. write_hydrograph_files is called UNCONDITIONALLY by
    # prepare_scenario (scenario.py:846), while its producer one line above --
    # run_swmm_hydro_model -- SKIPS the simulation when hydro_swmm_sim_completed is True.
    # So before this gate, every re-entry into prepare_scenario re-opened hydro.out, and
    # reclaiming hydro.out armed a latent failure on every subsequent re-prep.
    #
    # TWO LAYERS, NOT ONE, and the reason is specific to the reclaim this enables -- do not
    # simplify this to the log check alone. The log fields say the hydrographs WERE
    # written; they cannot say the files are still there. Before the hydro.out reclaim a
    # log-true/disk-absent divergence was recoverable, because hydro.out was on disk to
    # regenerate from. After it, a log-only gate that skipped on a missing strmflow/ would
    # hand TRITON a sim with no inflow inputs and nothing on disk to rebuild them -- silent
    # and permanent.
    _hyg_logged = bool(
        scenario.log.hyg_timeseries_created.get()
        and scenario.log.hyg_locs_created.get()
    )
    _hyg_present = (
        scenario.scen_paths.hyg_timeseries.exists()
        and scenario.scen_paths.hyg_locs.exists()
    )
    if _hyg_logged and _hyg_present:
        return "skipped"
    if not _hyg_present:
        _hydro_out = Path(str(scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
        if not _hydro_out.exists():
            raise ProcessingError(
                operation="write_hydrograph_files",
                filepath=str(scenario.scen_paths.hyg_timeseries),
                reason=(
                    "the TRITON inflow hydrographs are absent AND the SWMM hydrology "
                    f"output they are built from ({_hydro_out}) is also absent, so there "
                    "is no rebuild source. This scenario cannot be prepared without "
                    "re-running the hydrology simulation: clear hydro_swmm_sim_completed "
                    "on this scenario's log (or re-prepare the scenario from scratch) and "
                    "re-run. If remove_after_processing named 'hydro_out', that reclaim "
                    "is why this file is gone; it is a disclosed reclaim, not a failed run."
                ),
            )
    return "fell_through"
