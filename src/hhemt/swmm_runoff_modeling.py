"""
SWMM Runoff Modeling Module

This module handles SWMM hydrology-only modeling to generate runoff hydrographs
that serve as inputs to TRITON-SWMM. This includes:
- Creating rainfall and water level input files for SWMM
- Creating and running SWMM hydrology-only models
- Extracting runoff hydrographs from SWMM output for TRITON-SWMM input
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
from pyswmm import Output, Simulation
from swmm.toolkit.shared_enum import NodeAttribute

from hhemt._compile_venue import COMPILE_VENUE_ENV, swmm_execution_refused
from hhemt.cf_conventions import apply_cf_attributes
from hhemt.exceptions import ProcessingError
from hhemt.utils import open_store, write_zarr

if TYPE_CHECKING:
    import numpy as np

    from .scenario import TRITONSWMM_scenario


def _assert_validated_swmm_stack() -> None:
    """Fail closed before executing SWMM unless the installed pyswmm / swmm-toolkit
    is the conda-forge-validated pairing (pyswmm 2.x + swmm-toolkit 0.15.x).

    What this checks is a VERSION PREDICATE, not a provenance story: the installed
    ``pyswmm`` must start ``2.`` and ``swmm-toolkit`` must start ``0.15.``. The
    0.16+/0.17 lineage is the abi3 build carrying the observed free()/SIGABRT
    teardown crash, which the project cannot certify, so the guard refuses rather
    than risk silent heap corruption.

    ``pyproject.toml`` pins ``pyswmm>=2,<3`` and ``swmm-toolkit>=0.15.3,<0.16``,
    both strictly inside those prefixes, so a plain ``pip install hhemt`` resolves a
    stack this guard ACCEPTS. Do not restate the retired claim that the engine ships
    only from conda-forge -- it seeded that claim on five public documentation
    surfaces before it was caught. Conda is recommended for the HPC stack
    ``environment.yaml`` pins, not because pip cannot run SWMM. Note the predicate is
    looser than the pairing named above: it admits any ``2.x`` pyswmm, and a pip
    resolve currently lands 2.1.0. Override at your own risk with
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
            + "). Install a validated stack per docs/how-to/installation.md -- "
            "Option A (`conda env create -f environment.yaml` plus the two "
            "--no-deps post-create steps) is the recommended path, and this "
            "project's own dependency pins also resolve an accepted stack. Set "
            "HHEMT_ALLOW_UNVALIDATED_SWMM_STACK=1 to bypass at your own risk."
        )


def _refuse_swmm_in_undeclared_test_venue() -> None:
    """Refuse to execute SWMM from a test session that declared no permitted venue.

    WHY A TEST-FRAMEWORK NAME APPEARS IN SHIPPED CODE, pre-empted here because it
    reads like a layering violation on first sight and is not one. The predicate is
    a CONJUNCTION whose first term is an ambient environment fact -- the same kind of
    fact the guard directly above reads from HHEMT_ALLOW_UNVALIDATED_SWMM_STACK. It
    is false for every process that is not inside a pytest test phase, so no user of
    this package can reach the refusal at all.

    WHY NOT AN EXPLICITLY EXPORTED MARKER INSTEAD. That alternative was considered
    and rejected: a marker the test harness must SET is an arming flag, and an
    arming flag that goes missing DISARMS its guard silently. The venue token here
    only ever RELAXES, so every way it can fail -- absent, misspelt, renamed, deleted
    by an unrelated subsystem -- lands on refusal.

    The predicate itself lives in hhemt._compile_venue and is read there and nowhere
    else, so the venue token has ONE meaning across the compile guard and this one.
    """
    if swmm_execution_refused():
        raise RuntimeError(
            "Refusing to run SWMM from a test session on an undeclared machine. This "
            "test would execute the SWMM engine, which is only intended to run where "
            f"the machine has been declared as a permitted venue. Run `just test-gated`, "
            f"which sets {COMPILE_VENUE_ENV} for you; or run the fast test tier, which does "
            f"not execute the engine. To invoke pytest directly, set "
            f"{COMPILE_VENUE_ENV}=toolchain. See the compile-venue note in docs/contributing.md."
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
        subcatchment_raingage_mapping = self.system.cfg_system.subcatchment_raingage_mapping
        subcatchment_raingage_mapping_gage_id_colname = (
            self.system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        rainfall_units = self.cfg_analysis.rainfall_units

        df_sub_raingage_mapping = pd.read_csv(subcatchment_raingage_mapping)  # type: ignore
        sim_id_str = self.scenario.sim_id_str

        # retreieve dataframe of rainfall time series
        df_allrain = (
            self.scenario.ds_event_ts[
                df_sub_raingage_mapping[subcatchment_raingage_mapping_gage_id_colname].unique().astype(str)
            ]
            .reset_coords(drop=True)
            .to_dataframe()
        )
        dic_rain_paths = dict()

        for gage in df_allrain:
            s_rain = df_allrain[gage]
            df_rain = pd.DataFrame(
                dict(
                    date=df_allrain.index.strftime("%m/%d/%Y"),
                    time=df_allrain.index.time,
                    rain=s_rain,
                )
            )
            # define filepaths and write
            fname_raindat = f"grid-ind{gage}.dat"
            f_out_swmm_rainfall = self.scenario.scen_paths.dir_weather_datfiles / fname_raindat
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(f";;rain gage {gage} for sim {sim_id_str} \n")
                file.write(f";;Rainfall ({rainfall_units})\n")
            df_rain.to_csv(f_out_swmm_rainfall, sep="\t", index=False, header=False, mode="a")
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
        weather_time_series_storm_tide_datavar = self.cfg_analysis.weather_time_series_storm_tide_datavar

        sim_id_str = self.scenario.sim_id_str

        s_wlevel = (
            self.scenario.ds_event_ts[weather_time_series_storm_tide_datavar].reset_coords(drop=True).to_dataframe()
        )[weather_time_series_storm_tide_datavar]

        fname_wleveldat = "waterlevel.dat"

        f_out_swmm_wlevel = self.scenario.scen_paths.dir_weather_datfiles / fname_wleveldat

        # create data frame with proper formatting to be read in SWMM
        df_wlevel = pd.DataFrame(
            dict(
                date=s_wlevel.index.strftime("%m/%d/%Y"),
                time=s_wlevel.index.time,
                water_level_m=s_wlevel,
            )
        )

        with open(f_out_swmm_wlevel, "w+") as file:
            file.write(f";;water level for sim {sim_id_str}\n")
            file.write(f";;Water Level ({storm_tide_units})\n")

        df_wlevel.to_csv(f_out_swmm_wlevel, sep="\t", index=False, header=False, mode="a")
        self.scenario.log.storm_tide_for_swmm.set(f_out_swmm_wlevel)
        return

    def create_hydrology_model_from_template(self, swmm_model_template, destination: Path) -> None:
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

    def run_swmm_hydro_model(self, rerun_if_exists: bool = False, verbose: bool = False) -> None:
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
            _refuse_swmm_in_undeclared_test_venue()
            with Simulation(str(self.scenario.scen_paths.swmm_hydro_inp)) as sim:
                sim.execute()
            self.scenario.log.hydro_swmm_sim_completed.set(True)
        else:
            if verbose:
                print("Hydrology-only SWMM model already executed. Not re-running.")
        return

    def _write_node_inflow_capture(self, node_inflow: "HydrologyNodeInflow") -> None:
        """Persist the PER-NODE inflow series that tseries.hyg destroys, over the full node population.

        WHY PER-NODE. `tseries.hyg` writes one column per DEM GRIDCELL -- the member
        nodes are summed away one line before the write -- so the file cannot answer
        "total runoff volume per node location", which is the quantity that was asked
        for. Capturing pre-sum also makes the float32 storage EXACTLY lossless: every
        value is an unmodified SWMM REAL4 (`#define REAL4 float`, solver/output.c:50),
        whereas the post-sum column is a float64 sum of float32s and float32 storage of
        THAT rounds it.

        WHY THE FULL POPULATION, zeros included. A node whose inflow summed to 0 is 0 at every step
        (TOTAL_INFLOW is non-negative), so storing it is exact, and it makes a dry node explicit in the
        consolidated ``swmm_node_hydro`` group instead of absent from one event and present in another. The
        landed writer kept only nodes with positive inflow, so two events of one analysis carried different
        ``node_id`` sets and no fixed axis could hold both.

        Writes nothing only when the model has no non-outfall node or its output carries no time axis.
        """
        ds = node_inflow_capture_dataset(node_inflow, self.scenario.event_iloc)
        if ds is None:
            return
        out_path = Path(self.scenario.scen_paths.sim_folder) / "processed" / "hydrology_inflow_summary.zarr"
        # Written without `_write_output`, so the unit of `inflow` and `total_inflow_volume` is stamped here.
        apply_cf_attributes(ds, "hydrology_inflow_summary")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_zarr(ds, out_path, chunks=None, encoding={})
        # DN-3 / clause 2: a WRITE during scenario preparation needs no per-mutation
        # accounting; the scenario's own sentinel is re-derived at consolidate_scenario
        # (clause 10). The removed restamp fired once per scenario preparation and
        # re-summed the ANALYSIS scope over every sentinel-less sibling -- this was the
        # prepare_scenario starvation mode on the Norfolk stochastic campaign.

    def write_hydrograph_files(self) -> None:
        """
        Extract runoff hydrographs from SWMM output and format for TRITON-SWMM.

        Extracts runoff from SWMM hydrology model output and creates:
        1. Time series file with discharge hydrographs for each DEM grid cell
        2. Location file mapping hydrographs to DEM coordinates

        These files serve as inflow boundary conditions for TRITON-SWMM.
        Updates the scenario log to indicate files were created successfully.

        The per-node inflow is read from ``hydro.out`` while it exists. With it reclaimed, the same files are
        rebuilt from the per-node capture (``processed/hydrology_inflow_summary.zarr``) and, once that is cleared
        into the consolidated tree, from the event's slice of ``swmm_node_hydro/timeseries``; the
        gridcell sum is then taken over the persistent ``hydro.inp`` and DEM, so the rebuilt files are
        CONTENT-identical to the ones first written, not byte-identical text.
        """
        import rioxarray as rxr

        from .scenario import return_df_of_nodes_grouped_by_DEM_gridcell

        if hydrograph_outputs_gate(self.scenario) == "skipped":
            return

        dem_processed = self.system.sys_paths.dem_processed

        hydro_outfile = str(self.scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out")
        rds_dem = rxr.open_rasterio(dem_processed)
        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scenario.scen_paths.swmm_hydro_inp, dem_processed
        )

        if Path(hydro_outfile).exists():
            node_inflow = read_hydrology_node_inflow(hydro_outfile, df_node_locs, lst_outfalls)
            persist_capture = True
        else:
            node_inflow = read_hydrology_node_inflow_from_capture(self.scenario)
            persist_capture = False

        # write hydrograph file
        df_node_inflow = hydrograph_frame(node_inflow.inflow_by_node, df_node_locs, lst_outfalls, node_inflow.times_hr)
        sim_id_str = self.scenario.sim_id_str
        with open(self.scenario.scen_paths.hyg_timeseries, "w") as f:
            f.write(f"%Runoff for sim {sim_id_str}\n" + "%Time(hr) Discharge (m3/s)\n")
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
                f.write(f"{x},{y}\n")
        self.scenario.log.hyg_locs_created.set(True)
        if persist_capture:
            self._write_node_inflow_capture(node_inflow)
        # verifying that all nodes are within the DEM
        xllcorner = rds_dem.x.values.min()  # type: ignore
        yllcorner = rds_dem.y.values.min()  # type: ignore
        df_xylocs = pd.read_csv(self.scenario.scen_paths.hyg_locs, header=0, names=["x", "y"])
        if df_xylocs.x.min() < xllcorner:
            print("problem with x's")
        elif df_xylocs.y.min() < yllcorner:
            print("problem with y's")
        else:
            pass
        # check to make sure dimensions are correct
        df_hyg_loc = pd.read_csv(self.scenario.scen_paths.hyg_locs)
        df_hyg_test = pd.read_csv(self.scenario.scen_paths.hyg_timeseries, skiprows=2)
        if ((df_hyg_test.shape[1] - 1) - df_hyg_loc.shape[0]) != 0:
            print("ERROR ENCOUNTERED IN SETTING UP INPUTS")
            print("The shapes of the hydrograph file and the hydrograph location file do not match up.")
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
#: That holds for a CMS hydrology model only. Any other flow unit is converted to m3/s
#: before capture (``flow_unit_factor_to_cms``), and float32 then rounds the converted
#: values.
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

#: m3/s per one unit of each SWMM flow unit: 0.3048**3; 0.003785411784/60;
#: 0.003785411784*1e6/86400; 1e3/86400.
_FLOW_UNIT_TO_CMS = {
    "CMS": 1.0,
    "LPS": 0.001,
    "MLD": 0.011574,
    "CFS": 0.028317,
    "GPM": 6.3090e-5,
    "MGD": 0.043813,
}


def flow_unit_factor_to_cms(flow_units: str, source) -> float:
    """The factor converting one SWMM flow unit to m3/s; an unknown unit raises ProcessingError naming it.

    The hydrology model's own unit reaches TRITON unconverted otherwise: an LPS model's
    hydrographs arrive 1000 times too large and an MLD model's 86.4 times, although both pass
    every SI check.
    """
    factor = _FLOW_UNIT_TO_CMS.get(str(flow_units).strip().upper())
    if factor is None:
        raise ProcessingError(
            operation="flow_unit_factor_to_cms",
            filepath=Path(source),
            reason=f"unknown SWMM flow unit {flow_units!r}; expected one of {sorted(_FLOW_UNIT_TO_CMS)}",
        )
    return factor


#: SWMM's US flow units. SWMM treats a model as US when its flow unit is one of these, and an
#: absent FLOW_UNITS line means CFS.
_US_FLOW_UNITS = frozenset({"CFS", "GPM", "MGD"})

#: SWMM's six flow-unit words, in the order SWMM searches them when it resolves a
#: FLOW_UNITS value. No word is a prefix of another, so the order assigns nothing
#: differently; it is kept so the resolution reads as SWMM's own.
_SWMM_FLOW_UNIT_WORDS = ("CFS", "GPM", "MGD", "CMS", "LPS", "MLD")


def inp_flow_units(inp_path) -> str:
    """The flow unit SWMM will run the ``.inp`` at ``inp_path`` in; ``CFS`` when none is declared.

    Reads the file and applies ``flow_units_from_inp_text``, which states the rule.
    """
    return flow_units_from_inp_text(Path(inp_path).read_text())


def flow_units_from_inp_text(text: str, *, default: str | None = "CFS") -> str | None:
    """The flow unit SWMM will run a ``.inp`` in, read from the file's text; ``default`` when none is declared.

    A text-level function so the config-load template check and the preparation check share one
    reading: where two readers of one rule diverge, the toolkit either refuses a valid model at
    config load or passes one that preparation then refuses. ``default`` is SWMM's own default,
    CFS; a caller that must tell an absent line from a declared one passes ``default=None``.

    Read the way SWMM reads it, because a reading that differs lets a template pass this
    check and then run in another unit. SWMM applies every ``FLOW_UNITS`` line of every
    ``[OPTION…`` section in file order, so the LAST one wins; it ignores a line that carries no
    value; and it matches the key and the value by prefix, case-insensitively, so
    ``flow_units cfsx`` is a CFS model. Returns the resolved unit word, upper-cased. A value
    matching none of SWMM's six words is returned as written: SWMM rejects that file itself.

    Deliberately NOT the first matching line, which is what this returned before: a template
    declaring ``CMS`` and later ``CFS`` read as CMS here and ran as US.
    """
    from .scenario import _swmm_fold, _swmm_tokens

    in_options = False
    value = None
    for line in text.splitlines():
        tokens = _swmm_tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith("["):
            in_options = _swmm_fold(tokens[0]).startswith("[OPTION")
            continue
        if in_options and _swmm_fold(tokens[0]).startswith("FLOW_UNITS") and len(tokens) > 1:
            value = _swmm_fold(tokens[1])
    if value is None:
        return default
    for word in _SWMM_FLOW_UNIT_WORDS:
        if value.startswith(word):
            return word
    return value


def refuse_us_flow_units(inp_paths) -> None:
    """Refuse, at scenario preparation, a HYDROLOGY model whose flow unit is a US unit.

    The hydrology model is the one SWMM model whose flows the toolkit converts: its hydrographs
    are scaled to m3/s by ``flow_unit_factor_to_cms``, so CMS, LPS and MLD are all accepted for
    it. US units are refused because the toolkit treats a SWMM run as SI throughout. The
    hydraulics model and the standalone (full) model are checked by
    ``refuse_non_cms_flow_units`` instead, which is stricter because their flows are read
    unconverted.
    """
    for inp_path in inp_paths:
        units = inp_flow_units(inp_path)
        if units in _US_FLOW_UNITS:
            raise ProcessingError(
                operation="refuse_us_flow_units",
                filepath=Path(inp_path),
                reason=(
                    f"SWMM model uses US flow units ({units}); the toolkit supports SI flow "
                    "units only (CMS, LPS, MLD) for the hydrology model. Convert the model to SI units."
                ),
            )


def refuse_non_cms_flow_units(inp_paths) -> None:
    """Refuse, at scenario preparation, a hydraulics or standalone SWMM model not in CMS.

    The toolkit reads every flow of these two models -- the coupled model's report and the
    standalone model's output -- as m3/s and converts none of them. SWMM writes those flows in
    the model's own unit, so an LPS model would be read 1000 times too large and an MLD model
    86.4 times, with no error; a US-unit model would be processed in the wrong units after its
    run had spent its compute. Refusing here, before any SWMM run, is what keeps that from
    happening for a new run; the readers refuse a non-CMS output too, for outputs this check
    never saw.
    """
    for inp_path in inp_paths:
        units = inp_flow_units(inp_path)
        if units != "CMS":
            raise ProcessingError(
                operation="refuse_non_cms_flow_units",
                filepath=Path(inp_path),
                reason=(
                    f"SWMM model uses flow units {units}; the hydraulics and standalone SWMM models "
                    "must use CMS, because the toolkit reads their flows as m3/s without "
                    "conversion. An absent FLOW_UNITS line is SWMM's CFS default. Set "
                    "FLOW_UNITS CMS in the SWMM template named by system_config."
                ),
            )


class HydrologyNodeInflow(NamedTuple):
    """The hydrology model's per-node inflow of one event, as read from ``hydro.out`` or rebuilt from a capture.

    ``inflow_by_node`` holds one m3/s series per NON-OUTFALL node, in the order of the DEM-gridcell groups the
    hydrograph builder sums over (so a rebuild that walks the same groups meets the same gridcell order);
    ``gridcell_of_node`` is each node's ``(dem_x_coord, dem_y_coord)``; ``times_hr`` is the hours since the first
    output period (0 at the first period, which SWMM stamps one report step after the simulation start), or None
    when the model has no non-outfall node; ``flow_units`` is the unit the model reported in (the series are
    already converted to m3/s).
    """

    inflow_by_node: dict
    gridcell_of_node: dict
    times_hr: "np.ndarray | None"
    flow_units: str


def read_hydrology_node_inflow(hydro_outfile, df_node_locs, lst_outfalls) -> HydrologyNodeInflow:
    """Read ``TOTAL_INFLOW`` of EVERY non-outfall node of the hydrology output, converted once to m3/s.

    The population is every node of ``df_node_locs`` that is not an outfall, whether or not it received inflow
    in this event: a node with no inflow is returned as an all-zero series (exact, because ``TOTAL_INFLOW`` is
    non-negative). The landed reader returned only the nodes whose inflow summed above zero, which is the filter
    :func:`hydrograph_frame` still applies to the gridcell sums and no longer applies to the capture. Module
    level, taking the DEM-gridcell table as an argument, so it runs on an older scenario's ``hydro.out`` (the
    consolidation rebuilds a missing capture with it) and in a test that supplies its own node table.
    """
    inflow_by_node: dict = {}
    gridcell_of_node: dict = {}
    times_hr = None
    with Output(str(hydro_outfile)) as out:
        flow_units = out.units["flow"]
        # TRITON applies every hydrograph as m3/s, so each series is converted ONCE, where it is read, before
        # either consumer: tseries.hyg and the per-node capture then both carry m3/s. A CMS model's factor is 1
        # and its values are not touched.
        to_cms = flow_unit_factor_to_cms(flow_units, hydro_outfile)
        for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
            for key in list(group.node_key):  # node ids that fall within a single gridcell
                if key in lst_outfalls:
                    continue
                d_inflow = pd.Series(out.node_series(key, NodeAttribute.TOTAL_INFLOW))
                if to_cms != 1.0:
                    d_inflow = d_inflow * to_cms
                if times_hr is None:  # first column of the hydrograph file: time in hours
                    # .dt.seconds wraps on a >= 24 h gap; use total_seconds().
                    tseries = pd.Series(d_inflow.index).diff().dt.total_seconds() / 60 / 60
                    tseries.iloc[0] = 0
                    times_hr = tseries.cumsum().values
                # PER-NODE capture, taken here rather than downstream, and the position is load-bearing: the
                # member nodes of a gridcell are SUMMED by hydrograph_frame and the per-node identity is gone
                # (tseries.hyg is per-GRIDCELL, so its columns are centroids, not nodes). These values are SWMM
                # REAL4 (`#define REAL4 float`, solver/output.c:50) times the unit factor, so for a CMS model
                # storing them float32 is EXACTLY lossless; the post-sum column is a float64 sum and is not.
                inflow_by_node[key] = d_inflow.values
                gridcell_of_node[key] = coords
    return HydrologyNodeInflow(inflow_by_node, gridcell_of_node, times_hr, str(flow_units))


def hydrograph_frame(inflow_by_node, df_node_locs, lst_outfalls, times_hr) -> pd.DataFrame:
    """The hydrograph table: a ``timestep`` column then one summed column per DEM gridcell that received inflow.

    A gridcell column is the sum over its non-outfall nodes that carry positive inflow; a gridcell none of whose
    nodes did is omitted, exactly as the landed writer left it out. The gridcells are walked in the order of
    ``df_node_locs.groupby``, so the table built from ``hydro.out``, from the per-scenario capture and from the
    consolidated group has one column order. Values are summed as float64.
    """
    import numpy as np

    d_time_series: dict = {}
    if times_hr is not None:
        d_time_series["timestep"] = times_hr
    for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
        d_flows = {}
        for key in list(group.node_key):
            if key in lst_outfalls or key not in inflow_by_node:
                continue
            arr = np.asarray(inflow_by_node[key], dtype="float64")
            if arr.sum() > 0:
                d_flows[key] = arr
        if len(d_flows) > 0:
            d_time_series[coords] = pd.DataFrame(d_flows).sum(axis=1)
    return pd.DataFrame(d_time_series)


def node_inflow_capture_dataset(node_inflow: HydrologyNodeInflow, event_iloc: int):
    """The per-node capture Dataset of one event (``inflow`` on ``node_id`` and ``timestep``), or None if empty.

    ``timestep`` is float64 HOURS since the first output period; ``total_inflow_volume`` is the trapezoid of
    ``inflow`` over those hours times 3600, in m3. The dataset carries ``event_iloc`` as a length-1 dimension.
    """
    import numpy as np
    import xarray as xr

    if not node_inflow.inflow_by_node or node_inflow.times_hr is None:
        return None
    node_ids = sorted(node_inflow.inflow_by_node)
    inflow = np.asarray(
        [np.asarray(node_inflow.inflow_by_node[n], dtype=_HYDROGRAPH_CAPTURE_DTYPE) for n in node_ids],
        dtype=_HYDROGRAPH_CAPTURE_DTYPE,
    )
    t_hr = np.asarray(node_inflow.times_hr, dtype="float64")
    # Volume integrates cms over HOURS, so the 3600 is a unit conversion, not a fudge. float64 here
    # deliberately: the reduction accumulates over ~2,000 timesteps and is not itself a stored SWMM value.
    # `np.trapezoid` is the NumPy>=2.0 spelling and `np.trapz` the <2.0 one; this env pins numpy 1.26.4, where
    # `trapezoid` does not exist, and the project supports py3.11-3.12 across both numpy majors. Resolve by
    # attribute rather than by version test so neither spelling is a hard floor -- `trapz` is deprecated in
    # 2.x but still present, and `trapezoid` is absent in 1.x, so first-available is the only form correct on
    # both.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    volume_m3 = _trapz(inflow.astype("float64"), x=t_hr, axis=1) * 3600.0
    ds = xr.Dataset(
        data_vars={
            "inflow": (("node_id", "timestep"), inflow),
            "total_inflow_volume": (("node_id",), volume_m3),
        },
        coords={
            "node_id": np.asarray(node_ids, dtype=str),
            "timestep": t_hr.astype("float64"),
            "dem_x_coord": (("node_id",), np.asarray([node_inflow.gridcell_of_node[n][0] for n in node_ids])),
            "dem_y_coord": (("node_id",), np.asarray([node_inflow.gridcell_of_node[n][1] for n in node_ids])),
        },
        attrs={
            # The unit the values were converted FROM; inflow is always m3/s.
            "flow_units": str(node_inflow.flow_units),
            "system_total_inflow_volume": float(volume_m3.sum()),
            "system_total_inflow_volume_units": "m3",
            "n_gridcells": int(len({tuple(v) for v in node_inflow.gridcell_of_node.values()})),
            "notes": (
                "Per-NODE TOTAL_INFLOW of every non-outfall node (zeros included) captured pre-gridcell-sum "
                "from swmm/hydro.out, as SWMM REAL4 times the unit factor to m3/s (1 for a CMS model, for "
                "which float32 is exactly lossless w.r.t. SWMM's REAL4 binary). Supports CONTENT-identical "
                "regeneration of strmflow/tseries.hyg (values byte-recoverable; gridcell grouping "
                "re-derivable from the DEM and hydro.inp) -- NOT byte-identical TEXT regeneration, which "
                "has not been round-tripped."
            ),
        },
    )
    return ds.assign_coords(event_iloc=event_iloc).expand_dims("event_iloc")


def rebuild_node_inflow_capture_from_hydro_out(scenario):
    """The per-node capture Dataset of a scenario rebuilt from its ``hydro.out``, or ``ProcessingError`` naming it.

    For an older scenario whose capture was never written, or was written by the landed writer over positive-inflow
    nodes only (the consolidation reindexes that one onto the full population). Writes nothing: the caller
    consolidates the Dataset and the capture store is never created for it.
    """
    from .scenario import return_df_of_nodes_grouped_by_DEM_gridcell

    hydro_out = Path(str(scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
    if not hydro_out.exists():
        raise ProcessingError(
            operation="rebuild_node_inflow_capture_from_hydro_out",
            filepath=str(hydro_out),
            reason=(
                f"scenario {scenario.sim_id_str} (event_iloc {scenario.event_iloc}) has no per-node inflow "
                "capture and no hydro.out to rebuild it from, so its hydrology inflow cannot be consolidated. The "
                "capture is written when the scenario is prepared and the default reclaim removes hydro.out, so a "
                "capture deleted by a process-stage reprocess with regenerate_existing, or by hand, leaves no source. "
                "Prepare the scenario again to restore it"
            ),
        )
    df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
        scenario.scen_paths.swmm_hydro_inp, scenario._system.sys_paths.dem_processed
    )
    node_inflow = read_hydrology_node_inflow(hydro_out, df_node_locs, lst_outfalls)
    ds = node_inflow_capture_dataset(node_inflow, scenario.event_iloc)
    if ds is None:
        raise ProcessingError(
            operation="rebuild_node_inflow_capture_from_hydro_out",
            filepath=str(hydro_out),
            reason=f"scenario {scenario.sim_id_str}: the hydrology model has no non-outfall node to capture",
        )
    return ds


def hydrology_capture_path(scenario) -> Path | None:
    """The scenario's per-node inflow capture store, or None for a scenario that carries no ``sim_folder``.

    The None arm is for the gate's fixture-free tests, which bind only the names the gate reads (their stand-in has
    hydrograph and ``hydro.inp`` paths and no scenario folder); a prepared scenario always has one.
    """
    sim_folder = getattr(getattr(scenario, "scen_paths", None), "sim_folder", None)
    if sim_folder is None:
        return None
    return Path(sim_folder) / "processed" / "hydrology_inflow_summary.zarr"


def processed_outputs_to_delete(sim_folder) -> list[Path]:
    """The children of a scenario's ``processed`` directory that a process-stage rebuild deletes: all but the capture.

    The per-node inflow capture (``hydrology_capture_path``) is written when the scenario is prepared, not when it is
    processed, and the SWMM hydrology output it can be rebuilt from is removed by the default reclaim. A rebuild of
    the processed outputs therefore keeps it, and deletes every other child. A scenario with no capture has its whole
    ``processed`` directory returned.
    """
    from types import SimpleNamespace

    processed = Path(sim_folder) / "processed"
    if not processed.exists():
        return []
    keep = hydrology_capture_path(SimpleNamespace(scen_paths=SimpleNamespace(sim_folder=sim_folder)))
    if not keep.exists():
        return [processed]
    return sorted(child for child in processed.iterdir() if child != keep)


def _capture_openable(path) -> bool:
    """True iff the capture exists AND opens (a zarr store is a directory, so existence alone is not enough)."""

    if path is None or not Path(path).exists():
        return False
    try:
        open_store(str(path), kind="zarr").close()
    except Exception:
        return False
    return True


def hydrograph_rebuild_source(scenario) -> str | None:
    """Which source rebuilds the inflow hydrographs of ``scenario``: ``hydro_out``, ``capture``, ``group`` or None.

    Tried in that order. ``hydro_out`` is the SWMM hydrology output; ``capture`` is the per-node store
    ``processed/hydrology_inflow_summary.zarr`` while it exists; ``group`` is the event's slice of the
    consolidated ``swmm_node_hydro/timeseries`` group, usable only when the scenario's log records the
    capture as consolidated (a recorded consolidation means the slice was read back equal before the capture
    was cleared). None means none of the three is present.
    """
    if Path(str(scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out")).exists():
        return "hydro_out"
    if _capture_openable(hydrology_capture_path(scenario)):
        return "capture"
    analysis = getattr(scenario, "_analysis", None)
    if analysis is not None and analysis.process.hydrology_slice_available(scenario.event_iloc, scenario):
        return "group"
    return None


def read_hydrology_node_inflow_from_capture(scenario) -> HydrologyNodeInflow:
    """The per-node inflow of ``scenario`` from its capture store, else from its slice of the consolidated group.

    The hydrograph rebuild source that stands in for ``hydro.out``. The capture's ``timestep`` is its
    time column; the group's is ``elapsed_s`` shifted to start at 0, over the event's own non-padded length.
    The DEM gridcell of each node is not stored here: the caller sums by gridcell over the persistent
    ``hydro.inp``, which is why ``gridcell_of_node`` is empty and unused on this path.
    """
    import numpy as np

    source = hydrograph_rebuild_source(scenario)
    if source == "capture":
        ds = open_store(hydrology_capture_path(scenario), kind="zarr").load()
        flow_units = str(ds.attrs.get("flow_units", "CMS"))
        inflow = ds["inflow"].isel(event_iloc=0) if "event_iloc" in ds["inflow"].dims else ds["inflow"]
        times_hr = np.asarray(ds["timestep"].values, dtype="float64")
        node_ids = [str(n) for n in ds["node_id"].values]
        by_node = {n: np.asarray(inflow.sel(node_id=n).values, dtype="float64") for n in node_ids}
        return HydrologyNodeInflow(by_node, {}, times_hr, flow_units)
    if source == "group":
        node_ids, inflow_arr, times_hr = scenario._analysis.process.hydrology_event_inflow_from_tree(
            scenario.event_iloc
        )
        by_node = {n: np.asarray(inflow_arr[i], dtype="float64") for i, n in enumerate(node_ids)}
        return HydrologyNodeInflow(by_node, {}, times_hr, "CMS")
    hydro_out = Path(str(scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
    raise ProcessingError(
        operation="write_hydrograph_files",
        filepath=str(scenario.scen_paths.hyg_timeseries),
        reason=(
            "the TRITON inflow hydrographs are absent and there is no rebuild source: none of the three sources "
            f"they are rebuilt from is present: the SWMM hydrology output ({hydro_out}), the per-node capture "
            f"({hydrology_capture_path(scenario)}), and a recorded slice of the consolidated "
            "swmm_node_hydro/timeseries group. This scenario cannot be prepared without re-running the "
            "hydrology simulation: clear hydro_swmm_sim_completed on this scenario's log (or re-prepare the "
            "scenario from scratch) and re-run. If remove_after_processing named 'hydro_out' or "
            "'hydrographs', that reclaim is why these files are gone; it is a disclosed reclaim, not a failed run."
        ),
    )


def inp_report_step_s(inp_path) -> int | None:
    """The ``REPORT_STEP`` of a SWMM ``.inp`` in whole seconds, or None when the file does not declare one.

    Read the way SWMM reads an option line: the key matches by case-insensitive prefix and the value is
    ``hh:mm:ss`` (``hh:mm`` and a bare number of hours are read too); the last declaring line of an ``[OPTIONS]``
    section wins. The hydrology group's ``elapsed_s`` axis is built from this step, never from the
    capture's float32 hours.
    """
    in_options = False
    value = None
    for line in Path(inp_path).read_text(errors="replace").splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("["):
            in_options = stripped.upper().startswith("[OPTION")
            continue
        tokens = stripped.split()
        if in_options and len(tokens) > 1 and tokens[0].upper().startswith("REPORT_STEP"):
            value = tokens[1]
    if value is None:
        return None
    parts = value.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for n in numbers:
        seconds = seconds * 60.0 + n
    if len(numbers) == 1:
        seconds = numbers[0] * 3600.0
    elif len(numbers) == 2:
        seconds = numbers[0] * 3600.0 + numbers[1] * 60.0
    return int(round(seconds))


def hydrograph_outputs_gate(scenario) -> str:
    """Three-state already-written gate for the TRITON inflow hydrographs.

    Returns ``"skipped"`` when the hydrographs are logged AND present on disk (the fast
    path the hydro.out reclaim needs), or ``"fell_through"`` when the caller must build
    them. Raises ``ProcessingError`` when they are absent AND none of their three rebuild
    sources (``hydro.out``, the per-node capture, the consolidated group's slice) is present --
    the one state in which scenario preparation cannot proceed at all.

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
    _hyg_logged = bool(scenario.log.hyg_timeseries_created.get() and scenario.log.hyg_locs_created.get())
    _hyg_present = scenario.scen_paths.hyg_timeseries.exists() and scenario.scen_paths.hyg_locs.exists()
    if _hyg_logged and _hyg_present:
        return "skipped"
    if not _hyg_present:
        # THREE rebuild sources, tried in order: hydro.out, the per-node capture, and the event's slice
        # of the consolidated swmm_node_hydro/timeseries group. Only with none of them present is the scenario
        # unpreparable, and the error names all three.
        if hydrograph_rebuild_source(scenario) is None:
            _hydro_out = Path(str(scenario.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
            raise ProcessingError(
                operation="write_hydrograph_files",
                filepath=str(scenario.scen_paths.hyg_timeseries),
                reason=(
                    "the TRITON inflow hydrographs are absent and there is no rebuild source: none of the three "
                    f"sources they are rebuilt from is present: the SWMM hydrology output ({_hydro_out}), "
                    "the per-node capture "
                    f"({hydrology_capture_path(scenario)}), and a recorded slice of the consolidated "
                    "swmm_node_hydro/timeseries group. This scenario cannot be prepared without "
                    "re-running the hydrology simulation: clear hydro_swmm_sim_completed "
                    "on this scenario's log (or re-prepare the scenario from scratch) and "
                    "re-run. If remove_after_processing named 'hydro_out' or 'hydrographs', that reclaim "
                    "is why these files are gone; it is a disclosed reclaim, not a failed run."
                ),
            )
    return "fell_through"
