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

    def create_hydraulics_model(self, destination: Path) -> None:
        """
        Write the hydraulics SWMM role model of this scenario to ``destination``.

        The role is derived from ``system_config.SWMM_model`` and only the forcing the toolkit owns
        (simulation window, rain gauges, rain and water-level series) is rewritten for this scenario.
        """
        from .swmm_utils import write_swmm_role_model

        write_swmm_role_model(self.scenario, "hydraulics", destination)
        return

    def update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(self, verbose: bool = False) -> None:
        """
        Write the list of SWMM nodes this scenario's TRITON-SWMM run couples to the surface.

        TRITON-SWMM couples exactly the nodes named in the file its cfg's
        ``swmm_coupling_nodes`` key names, so this step decides the coupled set by writing that
        file, ``coupling_nodes.txt`` beside ``hydraulics.inp``, one name per line in SWMM node
        order. ``hydraulics.inp`` itself is left as the template filled it: no ``[INFLOWS]`` line
        is added, removed or rewritten, so every inflow the model declares reaches SWMM.

        The candidates are the ones ``system_config.swmm_coupling_mode`` declares. Every DEM
        cell holding a candidate is resolved by ``scenario.resolve_cell_inflow_nodes``:
        junctions are written; outfalls, storage nodes and dividers are not, and are named
        together in one warning; and a cell whose candidates are all of those types stops
        preparation with a named error. With ``one_coupled_inflow_node_per_dem_cell`` set
        (applied under ``legacy`` only), each cell keeps a single junction chosen by pipe size,
        and every prepared scenario records each cell that had more than one candidate in
        ``_status/inflow_node_selection.json``.

        Parameters
        ----------
        verbose : bool, optional
            If True, print detailed progress messages (default: False)
        """
        import json
        import warnings

        from .scenario import coupling_nodes_file_for, inp_link_table, select_coupling_nodes

        f_inp = self.scenario.scen_paths.swmm_hydraulics_inp
        one_per_cell = bool(self.cfg_analysis.one_coupled_inflow_node_per_dem_cell)
        model = swmmio.Model(str(f_inp))
        links = inp_link_table(model)
        selection = select_coupling_nodes(
            self.system.cfg_system,
            f_inp,
            self.system.sys_paths.dem_processed,
            links,
            one_per_cell=one_per_cell,
        )

        ineligible_dropped: list[str] = []
        selection_cells: list[dict] = []
        for (dem_x, dem_y), resolution in selection.cells:
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
                f"{f_inp}: nodes TRITON-SWMM cannot couple were not written to the coupling-node list, "
                f"each sharing a DEM cell with a junction that stays coupled: {ineligible_dropped}",
                stacklevel=2,
            )
        nodes_file = coupling_nodes_file_for(f_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{name}\n" for name in selection.written))
        if verbose:
            print(f"Wrote {len(selection.written)} coupling nodes to {nodes_file}.")
        self._write_open_water_outfalls(selection.written, f_inp)
        self._write_manhole_table(selection.written, f_inp)
        if one_per_cell:
            # Written for EVERY scenario prepared with the option on, empty when no cell was
            # reduced, so that an ABSENT file means "not prepared by this build" to the report's
            # inflow-node reduction disclosure rather than reading as "nothing reduced".
            record = self.scenario.scen_paths.sim_folder / "_status" / "inflow_node_selection.json"
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"cells": selection_cells}, indent=2) + "\n")
        self.scenario.log.inflow_nodes_in_hydraulic_inp_assigned = True
        return

    def _write_open_water_outfalls(self, coupling_nodes, hydraulics_inp) -> None:
        """Write the open-water outfall files beside ``coupling_nodes.txt`` when ``swmm_open_water_outfalls`` lists any.

        The list is judged first by ``swmm_utils.require_valid_open_water_outfalls``, the rule run entry applies,
        and then against ``coupling_nodes``, the exact set TRITON couples: a ``driven_stage`` outfall must not be a
        coupling node (it has no exchange slot) and a ``crown_rim`` outfall must be one (a converted junction that
        exchanges nothing would sit in the model as a dry pipe end). Under ``driven_stage`` the names are written to
        ``driven_outfalls.txt``, one per line in the model's ``[OUTFALLS]`` order. Under ``crown_rim``, when the
        solver's DEM-rim route is on, they are written to ``dem_rim_exempt_nodes.txt`` in the same form, and a
        ``crown_rim_outfalls.json`` records the sizing each listed node was given. With no listed outfall nothing is
        written.
        """
        import json

        from .exceptions import ConfigurationError
        from .scenario import (
            _swmm_fold,
            coupling_nodes_file_for,
            dem_rim_exempt_nodes_file_for,
            driven_outfalls_file_for,
            swmm_node_rim_key_is_written,
        )
        from .swmm_runnable import _parse
        from .swmm_utils import (
            _read_model_text,
            crown_rim_outfalls,
            crown_rim_sizing,
            driven_stage_outfalls,
            open_water_outfalls,
            require_valid_open_water_outfalls,
        )
        from .water_level_boundary import model_outfalls

        if not open_water_outfalls(self.cfg_analysis):
            return
        cfg_system = self.system.cfg_system
        require_valid_open_water_outfalls(cfg_system, self.cfg_analysis)
        coupled = {_swmm_fold(n) for n in coupling_nodes}
        model_order = model_outfalls(_parse(_read_model_text(Path(cfg_system.SWMM_model))))

        def in_model_order(names) -> list[str]:
            """The ``[OUTFALLS]`` rows' own spellings of ``names``, in the order the model lists them."""
            wanted = {_swmm_fold(n) for n in names}
            return [row.name for key, row in model_order.items() if _swmm_fold(key) in wanted]

        driven = driven_stage_outfalls(self.cfg_analysis)
        both = [n for n in driven if _swmm_fold(n) in coupled]
        if both:
            raise ConfigurationError(
                field="swmm_open_water_outfalls",
                message=(
                    f"driven_stage outfall(s) {both} are among the coupling nodes written for this scenario "
                    f"({coupling_nodes_file_for(hydraulics_inp)}); a driven outfall is not a coupling node"
                ),
                fix_hint="Remove the name from swmm_coupling_nodes_file, or use crown_rim",
            )
        if driven:
            driven_outfalls_file_for(hydraulics_inp).write_text("".join(f"{n}\n" for n in in_model_order(driven)))
        crown = crown_rim_outfalls(self.cfg_analysis)
        if not crown:
            return
        uncoupled = [n for n in crown if _swmm_fold(n) not in coupled]
        if uncoupled:
            raise ConfigurationError(
                field="swmm_open_water_outfalls",
                message=(
                    f"crown_rim outfall(s) {uncoupled} are not among the coupling nodes written for this scenario "
                    f"({coupling_nodes_file_for(hydraulics_inp)}); a converted outfall that is not coupled "
                    "exchanges no water"
                ),
                fix_hint="Set swmm_coupling_mode to all_junctions or name the outfall in swmm_coupling_nodes_file, "
                "or use driven_stage",
            )
        if swmm_node_rim_key_is_written(self.cfg_analysis):
            dem_rim_exempt_nodes_file_for(hydraulics_inp).write_text("".join(f"{n}\n" for n in in_model_order(crown)))
        sized, _reasons = crown_rim_sizing(cfg_system, self.cfg_analysis)
        record = Path(hydraulics_inp).parent / "crown_rim_outfalls.json"
        record.write_text(
            json.dumps(
                {
                    "representation": "crown_rim",
                    "diameter_rule": (
                        "area_equivalent: d = sqrt(4 A / pi), A the flow area of every barrel (scenario.calc_area)"
                    ),
                    "rim": "SWMM depth written as 0, which SWMM raises to the crown of the outlet conduit",
                    "outfalls": [
                        {
                            "name": r.name,
                            "outlet_conduit": r.conduit,
                            "barrels": r.barrels,
                            "shape": r.shape,
                            "area_m2": r.area_m2,
                            "diameter_m": r.diameter_m,
                            "crown_m_above_invert": r.crown_m,
                            "loss_coefficient": r.loss_coefficient,
                        }
                        for r in sized.values()
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    def _write_manhole_table(self, coupling_nodes, hydraulics_inp) -> None:
        """Write ``manhole_table.txt`` beside ``coupling_nodes.txt`` when a table is set or a ``crown_rim`` list is.

        The table is judged against ``coupling_nodes``, the exact set TRITON couples, by the rules the
        run-entry check applies to the candidates, so a table that passed run entry can still be refused
        here when a coupling node it names was dropped from the set. The file written lists every
        coupling node in the order of ``coupling_nodes.txt``, the two constants on each node that has no
        row, so the file TRITON reads is always complete. A ``crown_rim`` open-water outfall takes the diameter
        and loss the toolkit sized from its outlet conduit (``swmm_utils.crown_rim_sizing``), whatever the table
        or the constants say, and the table may not name it. With neither a table nor such an outfall nothing is
        written.
        """
        from .scenario import (
            ManholeTableContext,
            manhole_table_file_for,
            manhole_table_text,
            require_valid_manhole_table,
        )
        from .swmm_utils import crown_rim_outfalls, crown_rim_sizing

        # Read tolerantly: the configuration stand-ins of tests/test_scenario_boundary_guards.py carry
        # only the fields their cases set, and 16 of its tests reach this method without a table.
        table = getattr(self.cfg_analysis, "manhole_table", None)
        crown = crown_rim_outfalls(self.cfg_analysis)
        if table is None and not crown:
            return
        context = ManholeTableContext(
            table=Path(table) if table is not None else Path(""),
            model=str(self.system.cfg_system.SWMM_model),
            diameter_constant=self.cfg_analysis.manhole_diameter,
            loss_constant=self.cfg_analysis.manhole_loss_coefficient,
            crown_rim_nodes=crown,
        )
        rows = (
            require_valid_manhole_table(
                table, coupling_nodes, self.cfg_analysis.manhole_table_fill_with_constants, context
            )
            if table is not None
            else []
        )
        sized, _reasons = crown_rim_sizing(self.system.cfg_system, self.cfg_analysis)
        table_file = manhole_table_file_for(hydraulics_inp)
        table_file.write_text(
            manhole_table_text(
                rows,
                coupling_nodes,
                self.cfg_analysis.manhole_diameter,
                self.cfg_analysis.manhole_loss_coefficient,
                fixed={name: (r.diameter_m, r.loss_coefficient) for name, r in sized.items()},
            )
        )

    def create_external_boundary_condition_files(self) -> None:
        """
        Create the boundary condition files for TRITON.

        Writes the location file ``extbc/loc.extbc``, one row per record of the external-boundary layer in record
        order, and one level-versus-time file ``extbc/tseries_{k}.txt`` for each ``water_level`` record. A record's
        type picks its row: ``water_level`` names its series file, ``free_outflow`` writes 0.0, and ``normal_slope``
        and ``froude_number`` write the number the layer holds. With the external boundaries off the location file
        holds its header only and no level file is written.

        Updates the scenario log to indicate files were created successfully, and records each record's type, its
        reprojected endpoints, the DEM cells they snap to, and the distance of each matched outfall to its line.
        """
        import rioxarray as rxr

        from hhemt.exceptions import ProcessingError
        from hhemt.water_level_boundary import (
            EXTBC_TYPE_CODE,
            boundary_level_series,
            boundary_records_for,
            matched_outfall_problems,
            model_coordinates,
            model_outfalls,
            snap_segments,
        )

        cfg_system = self.system.cfg_system
        paths = self.scenario.scen_paths
        records: tuple = ()
        if self.cfg_analysis.toggle_external_boundaries:
            records, _problems = boundary_records_for(self.cfg_analysis, cfg_system)

        rows: list[str] = []
        segments: list[dict] = []
        if records:
            from hhemt.swmm_runnable import _parse
            from hhemt.swmm_utils import _read_model_text

            rds_dem = rxr.open_rasterio(self.system.sys_paths.dem_processed)
            snapped, problems = snap_segments(records, rds_dem)
            sections = _parse(_read_model_text(Path(cfg_system.SWMM_model)))
            more, distances = matched_outfall_problems(
                records,
                model_outfalls(sections),
                model_coordinates(sections),
                self.cfg_analysis.water_level_boundary_outfall_max_distance_m,
            )
            if problems or more:
                raise ProcessingError(
                    operation="create_external_boundary_condition_files",
                    reason="; ".join(p.message for p in [*problems, *more]),
                )
            sim_id_str = self.scenario.sim_id_str
            for record, (_side, x1, y1, x2, y2) in snapped:
                last = str(0.0)
                if record.kind == "water_level":
                    level = boundary_level_series(self.scenario, record)
                    hours = [(t - level.times[0]).total_seconds() / 3600 for t in level.times]
                    near = ", ".join(f"{n} {d:.1f} m" for n, d in sorted(distances.get(record.index, {}).items()))
                    first_line = (
                        f"%{sim_id_str} Water Level Boundary Condition, record {record.index}, outfalls "
                        f"{' '.join(record.outfalls)}, distance to the line {near}\n"
                    )
                    second_line = f"%Time(hr) water_elevation (m), {level.source}\n"
                    tseries = paths.extbc_tseries(record.index)
                    with open(tseries, "w") as f:
                        f.write(first_line + second_line)
                        f.writelines(f"{h},{v}\n" for h, v in zip(hours, level.meters, strict=True))
                    last = f'"{tseries.name}"'
                elif record.kind in ("normal_slope", "froude_number"):
                    last = str(record.value)
                rows.append(f"{EXTBC_TYPE_CODE[record.kind]},{x1},{y1},{x2},{y2},{last}")
                segments.append(
                    {
                        "record": record.index,
                        "type": record.kind,
                        "endpoints": [list(v) for v in (record.vertices[0], record.vertices[-1])],
                        "snapped_cells": [[x1, y1], [x2, y2]],
                        "outfalls": list(record.outfalls),
                        "outfall_distance_m": {n: round(d, 3) for n, d in distances.get(record.index, {}).items()},
                    }
                )
        if any(r.kind == "water_level" for r in records):
            self.scenario.log.extbc_tseries_created.set(True)

        with open(paths.extbc_loc, "w") as f:
            f.write("% BC Type, X1, Y1, X2, Y2, BC\n")
            f.writelines(row + "\n" for row in rows)
        self.scenario.log.external_boundary_segments.set(segments)
        self.scenario.log.extbc_loc_created.set(True)
        return
