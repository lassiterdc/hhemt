"""V0028: drop the unit tokens from stored names and carry the unit in a ``units`` attribute.

Layouts 26 and 27 stored TRITON's unit discharge under velocity names and its depth under water-level names,
spelled every unit into the SWMM-tier and hydrology names (``depth_m``, ``flow_cms``, ``total_runoff_10e6_ltr``),
named the time axis ``timestep_min`` (``time_hr`` in the runoff-model store), and carried six dataset attribute
keys with a unit token. Layout 28 removes every unit token and states the unit in a ``units`` attribute (a
``{key}_units`` companion attribute for an attribute key).

This migration renames and re-stamps only (D5): it does not derive the flood-velocity variables, which need the
raw rasters. Every rename is keyed on the TIER of the store or tree node, never on the name alone, because one
layout-26 name sits on two tiers with two meanings (``max_velocity_mps`` is the TRITON maximum unit discharge and the
SWMM link maximum conduit velocity; ``max_flow_cms`` and ``time_of_max_flow_min`` differ between the node and link
tiers, D16). A node of a store whose tier cannot be determined is REFUSED when it carries a retired name
(``MigrationBlockedError``, raised while planning, before any write) and left alone when it carries only generic
names (a legacy ``depth``, which is in no retired table). Two names map by store origin:
``flow_balance_error_percent`` held a percentage on some
rows and litres on others and becomes ``flow_balance_error_unsplit`` with no ``units`` (D17); ``max_ponded_depth_m``
keeps its values as ``max_ponded_depth`` in ``m`` with a ``comment`` that a value from a model not routed by Dynamic
Wave is a ponded volume (D18). The migration never writes the parser-only names ``flow_balance_error`` and
``flow_balance_error_volume``.

Values are never read or written for a variable rename (the array directory is renamed), so a renamed variable is
bit-for-bit what it was. The attributes of a renamed or re-stamped variable are replaced from the literal tables
below, which are frozen with this module and are the layout-28 ``cf_conventions`` entries as of its authoring, so
no stale ``standard_name`` or ``cell_methods`` survives. A second run finds no old name and plans nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import zarr

from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError

version_from: int = 27
version_to: int = 28
description: str = "Drop unit tokens from stored variable, coordinate and attribute names; state units in attributes"

#: TRITON's variables, timeseries and summary alike.
_TRITON: dict[str, str] = {
    "max_wlevel_m": "max_flood_depth",
    "wlevel_m": "flood_depth",
    "velocity_x_mps": "unit_discharge_x",
    "velocity_y_mps": "unit_discharge_y",
    "time_of_max_wlevel_min": "time_of_max_flood_depth",
    "wlevel_m_last_tstep": "flood_depth_last_tstep",
    "max_velocity_mps": "max_unit_discharge",
    "time_of_max_velocity_min": "time_of_max_unit_discharge",
    "velocity_x_mps_at_time_of_max_velocity": "unit_discharge_x_at_time_of_max_unit_discharge",
    "velocity_y_mps_at_time_of_max_velocity": "unit_discharge_y_at_time_of_max_unit_discharge",
    "final_surface_flood_volume_m3": "final_surface_flood_volume",
}
#: Names shared by the node and link tiers that mean one thing on both.
_SWMM_COMMON: dict[str, str] = {
    "depth_m": "depth",
    "head_m": "head",
    "flooding_cms": "flooding",
    "inflow_flow_cms": "inflow_flow",
    "depth_m_last": "depth_last",
    "depth_m_max": "depth_max",
    "head_m_last": "head_last",
    "head_m_max": "head_max",
    "inflow_flow_cms_last": "inflow_flow_last",
    "inflow_flow_cms_max": "inflow_flow_max",
    "flooding_cms_last": "flooding_last",
    "flooding_cms_max": "flooding_max",
}
#: Node tier. ``max_flow_cms`` is the Node Flooding Summary's maximum rate and ``time_of_max_flow_min`` the Node Inflow
#: Summary's time of the maximum total inflow (D16); ``flow_balance_error_percent`` and ``max_ponded_depth_m`` map by
#: store origin (D17, D18).
_NODE: dict[str, str] = {
    **_SWMM_COMMON,
    "max_lateral_inflow_cms": "max_lateral_inflow",
    "max_total_inflow_cms": "max_total_inflow",
    "max_flow_cms": "max_flood_rate",
    "time_of_max_flood_min": "time_of_max_flood",
    "time_of_max_flow_min": "time_of_max_total_inflow",
    "tot_flooded_vol_10e6_ltr": "tot_flooded_vol",
    "lateral_inflow_vol_10e6_ltr": "lateral_inflow_vol",
    "total_inflow_vol_10e6_ltr": "total_inflow_vol",
    "max_ponded_depth_m": "max_ponded_depth",
    "flow_balance_error_percent": "flow_balance_error_unsplit",
}
_LINK: dict[str, str] = {
    "flow_cms": "flow",
    "velocity_mps": "velocity",
    "link_depth_m": "link_depth",
    "flow_cms_last": "flow_last",
    "flow_cms_max": "flow_max",
    "velocity_mps_last": "velocity_last",
    "velocity_mps_max": "velocity_max",
    "link_depth_m_last": "link_depth_last",
    "link_depth_m_max": "link_depth_max",
    "max_velocity_mps": "max_velocity",
    "max_flow_cms": "max_flow",
    "time_of_max_flow_min": "time_of_max_flow",
    "total_inflow_vol_10e6_ltr": "total_inflow_vol",
}
#: Names unique to one store or to the water-quality variables, so the tier does not decide them.
_UNIQUE: dict[str, str] = {
    "total_precip_mm": "total_precip",
    "total_runon_mm": "total_runon",
    "total_evap_mm": "total_evap",
    "total_infil_mm": "total_infil",
    "imperv_runoff_mm": "imperv_runoff",
    "perv_runoff_mm": "perv_runoff",
    "total_runoff_mm": "total_runoff_depth",
    "total_runoff_10e6_ltr": "total_runoff_volume",
    "peak_runoff_cms": "peak_runoff",
    "inflow_cms": "inflow",
    "total_inflow_volume_m3": "total_inflow_volume",
    "pollutant_time_of_max_conc_min": "pollutant_time_of_max_conc",
    "outfall_total_vol_10e6_ltr": "outfall_total_vol",
    "qual_routing_continuity_error_perc": "qual_routing_continuity_error",
    "runoff_qual_continuity_error_perc": "runoff_qual_continuity_error",
}
_EDA_COMPUTE: dict[str, str] = {
    "max_abs_diff_m": "max_abs_depth_diff",
    "rmse_wetted_m": "rmse_wetted_depth",
    "pct_diff_p95": "relative_depth_diff_p95",
}
#: ``raw_output_type`` values of the raw byte-identity store, and the cross-sim identity variables. The identity
#: artifact is built from the per-scenario FLAT summaries (its manifest lists the TRITON, SWMM link, SWMM node, perf
#: and hydrology summaries), where the SWMM link velocity is ``velocity_mps_max`` / ``velocity_mps_last``, so a
#: ``max_velocity_mps`` there can only be the TRITON maximum unit discharge, and ``max_flow_cms`` is the link tier's
#: (the artifact's own tracked-variable list). Its bases are the flat-summary names: the TRITON ones, the node and
#: link ``_last`` / ``_max`` reductions, and ``capacity_setting_{last,max}``, which carry no unit token and keep
#: their names.
_RAW_OUTPUT_TYPE_VALUES: dict[str, str] = {
    "max_wlevel_m": "max_flood_depth",
    "wlevel_m": "flood_depth",
    "velocity_x_mps": "unit_discharge_x",
    "velocity_y_mps": "unit_discharge_y",
}
_IDENTITY_VARIABLES: dict[str, str] = {
    **_TRITON,
    **_SWMM_COMMON,
    **{k: v for k, v in _LINK.items() if k.endswith(("_last", "_max")) and k not in _SWMM_COMMON},
    "max_flow_cms": "max_flow",
}

#: Dataset attribute keys that lose their unit token, with the unit the companion ``{key}_units`` states.
_ATTR_KEYS: dict[str, tuple[str, str]] = {
    "reporting_interval_s": ("reporting_interval", "s"),
    "resume_boundaries_min": ("resume_boundaries", "min"),
    "system_total_inflow_volume_m3": ("system_total_inflow_volume", "m3"),
    "runoff_continuity_error_perc": ("runoff_continuity_error", "percent"),
    "flow_continuity_error_perc": ("flow_continuity_error", "percent"),
    "system_flooding_10e6_ltr": ("system_flooding", "10^6 L"),
}

_D18_COMMENT = (
    "In a store migrated from layout 27 or below, a value from a model not routed by Dynamic Wave is a ponded "
    "volume in 1000 m3 and not a depth; the store does not record the routing method."
)
_UNSPLIT_COMMENT = (
    "Each value is a percentage or an absolute error in litres, and this store does not record which: it was "
    "written before the parser kept the two apart."
)
_TIMESTEP_ATTRS = {"long_name": "Model timestep (minutes since simulation start)", "units": "min", "axis": "T"}
_RUNOFF_TIMESTEP_ATTRS = {"long_name": "Model timestep (hours since simulation start)", "units": "h", "axis": "T"}
_CF_KEYS = ("standard_name", "long_name", "units", "cell_methods")
#: The ``long_name`` of an ``identical__{base}`` flag of the cross-sim identity artifact, as the layout-28 writer
#: (``cf_conventions._eda_dynamic_entry``) words it. Frozen with this module; a flag has no unit.
_IDENTICAL_LONG_NAME = "Byte-identity flag for {base} against the reference"

#: Every retired variable name of the tier-keyed tables. A data variable carrying one of these in a store whose tier
#: cannot be determined is refused, because the right new name depends on the tier.
_RETIRED_VARIABLES: frozenset[str] = frozenset({*_TRITON, *_NODE, *_LINK, *_UNIQUE, *_EDA_COMPUTE})
#: Names that two tiers rename differently, so that an artifact which does not fix the tier cannot be renamed.
_AMBIGUOUS_ACROSS_TIERS: frozenset[str] = frozenset({"time_of_max_flow_min", "total_inflow_vol_10e6_ltr"})


def _refuse(store: Path, rel: str, variable: str, why: str) -> MigrationBlockedError:
    where = f"{store}" + (f" (group {rel})" if rel else "")
    return MigrationBlockedError(
        f"V0028: {where} holds the layout-26 variable {variable!r} but {why}, and the new name depends on the tier. "
        "NOTHING HAS BEEN CHANGED by this migration. Move the store aside, or rename the variable by hand to the "
        "layout-28 name of its tier, and re-run."
    )


# --------------------------------------------------------------------------------------------------------------
# Frozen layout-28 attribute tables (the cf_conventions entries as of this migration)
# --------------------------------------------------------------------------------------------------------------

_ATTRS: dict[str, dict[str, str | None]] = {
    "capacity": {
        "standard_name": None,
        "long_name": ("Link capacity at each report step, as SWMM reports it (fraction of the conduit filled)"),
        "units": "1",
        "cell_methods": None,
    },
    "capacity_last": {
        "standard_name": None,
        "long_name": "Link capacity at the final report step, as SWMM reports it",
        "units": "1",
        "cell_methods": None,
    },
    "capacity_max": {
        "standard_name": None,
        "long_name": "Maximum link capacity over the report steps, as SWMM reports it",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "capacity_setting": {
        "standard_name": None,
        "long_name": (
            "Link capacity or control setting at each report step, as SWMM reports it (fraction filled or setting)"
        ),
        "units": "1",
        "cell_methods": None,
    },
    "capacity_setting_last": {
        "standard_name": None,
        "long_name": "Link capacity or control setting at the final report step",
        "units": "1",
        "cell_methods": None,
    },
    "capacity_setting_max": {
        "standard_name": None,
        "long_name": "Maximum link capacity or control setting over the report steps",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "depth": {
        "standard_name": None,
        "long_name": "Water depth at the node at each report step",
        "units": "m",
        "cell_methods": None,
    },
    "depth_last": {
        "standard_name": None,
        "long_name": "Water depth at the node at the final report step",
        "units": "m",
        "cell_methods": None,
    },
    "depth_max": {
        "standard_name": None,
        "long_name": "Maximum water depth at the node over the report steps",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "final_surface_flood_volume": {
        "standard_name": None,
        "long_name": "Final surface flood volume",
        "units": "m3",
        "cell_methods": "area: sum",
    },
    "flood_depth": {
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth timeseries",
        "units": "m",
        "cell_methods": None,
    },
    "flood_depth_last_tstep": {
        "standard_name": "flood_water_thickness",
        "long_name": "Flood depth at final timestep",
        "units": "m",
        "cell_methods": "time: point",
    },
    "flooding": {
        "standard_name": None,
        "long_name": "Flooding rate at the node at each report step",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "flooding_last": {
        "standard_name": None,
        "long_name": "Flooding rate at the node at the final report step",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "flooding_max": {
        "standard_name": None,
        "long_name": "Maximum flooding rate at the node over the report steps",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "flow": {
        "standard_name": None,
        "long_name": "Flow rate in the link at each report step (signed)",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "flow_balance_error": {
        "standard_name": None,
        "long_name": (
            "Flow balance error of the node, as a percentage (Node Inflow Summary, rows SWMM printed as a percentage)"
        ),
        "units": "percent",
        "cell_methods": None,
    },
    "flow_balance_error_unsplit": {
        "standard_name": None,
        "long_name": "Flow balance error of the node, percentage or volume per row (migrated store)",
        "units": None,
        "cell_methods": None,
        "comment": (
            "Each value is a percentage or an absolute error in litres, and this store does not "
            "record which: it was written before the parser kept the two apart."
        ),
    },
    "flow_balance_error_volume": {
        "standard_name": None,
        "long_name": (
            "Flow balance error of the node, as a volume (Node Inflow Summary, rows SWMM printed followed by ltr)"
        ),
        "units": "L",
        "cell_methods": None,
    },
    "flow_last": {
        "standard_name": None,
        "long_name": "Flow rate in the link at the final report step (signed)",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "flow_max": {
        "standard_name": None,
        "long_name": (
            "Maximum signed flow rate in the link over the report steps (largest value, not largest magnitude)"
        ),
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "head": {
        "standard_name": None,
        "long_name": "Hydraulic head at the node at each report step",
        "units": "m",
        "cell_methods": None,
    },
    "head_last": {
        "standard_name": None,
        "long_name": "Hydraulic head at the node at the final report step",
        "units": "m",
        "cell_methods": None,
    },
    "head_max": {
        "standard_name": None,
        "long_name": "Maximum hydraulic head at the node over the report steps",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "hours_flooded": {
        "standard_name": None,
        "long_name": "Hours the node was flooded (Node Flooding Summary)",
        "units": "h",
        "cell_methods": None,
    },
    "imperv_runoff": {
        "standard_name": None,
        "long_name": "Impervious-area runoff depth from the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "inflow": {
        "standard_name": None,
        "long_name": ("Total inflow to the node captured from the hydrology model's output at each report step"),
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "inflow_flow": {
        "standard_name": None,
        "long_name": "Total inflow to the node at each report step",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "inflow_flow_last": {
        "standard_name": None,
        "long_name": "Total inflow to the node at the final report step",
        "units": "m3 s-1",
        "cell_methods": None,
    },
    "inflow_flow_max": {
        "standard_name": None,
        "long_name": "Maximum total inflow to the node over the report steps",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "lateral_inflow_vol": {
        "standard_name": None,
        "long_name": "Total lateral inflow volume to the node (Node Inflow Summary)",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "link_depth": {
        "standard_name": None,
        "long_name": "Water depth in the link at each report step",
        "units": "m",
        "cell_methods": None,
    },
    "link_depth_last": {
        "standard_name": None,
        "long_name": "Water depth in the link at the final report step",
        "units": "m",
        "cell_methods": None,
    },
    "link_depth_max": {
        "standard_name": None,
        "long_name": "Maximum water depth in the link over the report steps",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "link_pollutant_conc": {
        "standard_name": None,
        "long_name": "Pollutant concentration in the link at each report step",
        "units": None,
        "cell_methods": None,
    },
    "link_total_load": {
        "standard_name": None,
        "long_name": "Total pollutant mass carried by the link over the run",
        "units": None,
        "cell_methods": "time: sum",
    },
    "max_abs_depth_diff": {
        "standard_name": None,
        "long_name": ("Maximum absolute difference in peak flood depth between two runs, over the wetted footprint"),
        "units": "m",
        "cell_methods": None,
    },
    "max_flood_depth": {
        "standard_name": "flood_water_thickness",
        "long_name": "Maximum flood depth (solver running maximum of depth)",
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "max_flood_rate": {
        "standard_name": None,
        "long_name": "Maximum flooding rate at the node (Node Flooding Summary, Maximum Rate)",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_flood_velocity": {
        "standard_name": "flood_water_speed",
        "long_name": "Maximum flood velocity over reported timesteps (unit discharge divided by depth)",
        "units": "m s-1",
        "cell_methods": "time: maximum",
    },
    "max_flow": {
        "standard_name": None,
        "long_name": "Maximum flow rate",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_lateral_inflow": {
        "standard_name": None,
        "long_name": "Maximum lateral inflow to the node (Node Inflow Summary)",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_over_full_depth": {
        "standard_name": None,
        "long_name": "Maximum depth-to-full-depth ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "max_over_full_flow": {
        "standard_name": None,
        "long_name": "Maximum flow-to-full-flow ratio",
        "units": "1",
        "cell_methods": "time: maximum",
    },
    "max_ponded_depth": {
        "standard_name": None,
        "long_name": (
            "Maximum ponded depth at the node (Node Flooding Summary; a value only for a Dynamic Wave model)"
        ),
        "units": "m",
        "cell_methods": "time: maximum",
    },
    "max_total_inflow": {
        "standard_name": None,
        "long_name": "Maximum total inflow to the node (Node Inflow Summary)",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "max_unit_discharge": {
        "standard_name": None,
        "long_name": "Maximum unit discharge (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": "time: maximum",
    },
    "max_velocity": {
        "standard_name": "sea_water_speed",
        "long_name": "Maximum conduit velocity",
        "units": "m s-1",
        "cell_methods": "time: maximum",
    },
    "outfall_emc": {
        "standard_name": None,
        "long_name": "Event mean concentration at the outfall (total load over total volume)",
        "units": None,
        "cell_methods": "time: mean",
    },
    "outfall_total_load": {
        "standard_name": None,
        "long_name": "Total pollutant mass discharged at the outfall over the event",
        "units": None,
        "cell_methods": "time: sum",
    },
    "outfall_total_vol": {
        "standard_name": None,
        "long_name": "Total volume discharged at the outfall over the event",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "peak_runoff": {
        "standard_name": None,
        "long_name": "Peak runoff rate from the subcatchment",
        "units": "m3 s-1",
        "cell_methods": "time: maximum",
    },
    "perv_runoff": {
        "standard_name": None,
        "long_name": "Pervious-area runoff depth from the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "pollutant_conc": {
        "standard_name": None,
        "long_name": "Pollutant concentration at the node at each report step",
        "units": None,
        "cell_methods": None,
    },
    "pollutant_max_conc": {
        "standard_name": None,
        "long_name": "Maximum pollutant concentration at the node over the event",
        "units": None,
        "cell_methods": "time: maximum",
    },
    "pollutant_time_of_max_conc": {
        "standard_name": None,
        "long_name": "Time of maximum pollutant concentration (minutes from the report's Starting Date)",
        "units": "min",
        "cell_methods": None,
    },
    "qual_routing_continuity_error": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity error",
        "units": "percent",
        "cell_methods": None,
    },
    "qual_routing_dry_weather_inflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: dry weather inflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_exfiltration_loss": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: exfiltration loss (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_external_inflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: external inflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_external_outflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: external outflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_final_stored_mass": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: final stored mass (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_flooding_loss": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: flooding loss (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_groundwater_inflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: groundwater inflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_initial_stored_mass": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: initial stored mass (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_mass_reacted": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: mass reacted (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_rdii_inflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: rdii inflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "qual_routing_wet_weather_inflow": {
        "standard_name": None,
        "long_name": "Quality Routing Continuity: wet weather inflow (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "relative_depth_diff_p95": {
        "standard_name": None,
        "long_name": (
            "95th percentile of the absolute percentage difference in peak flood depth between "
            "two runs, over cells the reference run wetted to at least tau"
        ),
        "units": "percent",
        "cell_methods": None,
    },
    "rmse_wetted_depth": {
        "standard_name": None,
        "long_name": ("Root-mean-square difference in peak flood depth between two runs, over the wetted footprint"),
        "units": "m",
        "cell_methods": None,
    },
    "runoff_coeff": {
        "standard_name": None,
        "long_name": ("Runoff coefficient of the subcatchment over the run (runoff depth over precipitation)"),
        "units": "1",
        "cell_methods": None,
    },
    "runoff_qual_bmp_removal": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: bmp removal (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_continuity_error": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity error",
        "units": "percent",
        "cell_methods": None,
    },
    "runoff_qual_infiltration_loss": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: infiltration loss (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_initial_buildup": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: initial buildup (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_remaining_buildup": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: remaining buildup (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_surface_buildup": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: surface buildup (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_surface_runoff": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: surface runoff (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_sweeping_removal": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: sweeping removal (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "runoff_qual_wet_deposition": {
        "standard_name": None,
        "long_name": "Runoff Quality Continuity: wet deposition (pollutant mass)",
        "units": None,
        "cell_methods": None,
    },
    "time_of_max_flood": {
        "standard_name": None,
        "long_name": "Time of maximum flooding rate at the node (Node Flooding Summary)",
        "units": "min",
        "cell_methods": None,
    },
    "time_of_max_flood_depth": {
        "standard_name": None,
        "long_name": "Time of maximum flood depth (print at which printed depth is largest)",
        "units": "min",
        "cell_methods": None,
    },
    "time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Time of maximum flood velocity",
        "units": "min",
        "cell_methods": None,
    },
    "time_of_max_flow": {
        "standard_name": None,
        "long_name": "Time of maximum flow in the link (Link Flow Summary)",
        "units": "min",
        "cell_methods": None,
    },
    "time_of_max_total_inflow": {
        "standard_name": None,
        "long_name": "Time of maximum total inflow to the node (Node Inflow Summary)",
        "units": "min",
        "cell_methods": None,
    },
    "time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Time of maximum unit discharge",
        "units": "min",
        "cell_methods": None,
    },
    "tot_flooded_vol": {
        "standard_name": None,
        "long_name": "Total flood volume at the node (Node Flooding Summary)",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "total_evap": {
        "standard_name": None,
        "long_name": "Total evaporation from the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "total_infil": {
        "standard_name": None,
        "long_name": "Total infiltration on the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "total_inflow_vol": {
        "standard_name": None,
        "long_name": "Total inflow volume",
        "units": "10^6 L",
        "cell_methods": "time: sum",
    },
    "total_inflow_volume": {
        "standard_name": None,
        "long_name": "Total inflow volume to the node captured from the hydrology model's output",
        "units": "m3",
        "cell_methods": "time: sum",
    },
    "total_precip": {
        "standard_name": None,
        "long_name": "Total precipitation on the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "total_runoff_depth": {
        "standard_name": None,
        "long_name": "Total runoff depth from the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "total_runoff_volume": {
        "standard_name": None,
        "long_name": "Total runoff volume from the subcatchment over the run",
        "units": "10^6 L",
        "cell_methods": None,
    },
    "total_runon": {
        "standard_name": None,
        "long_name": "Total run-on to the subcatchment over the run",
        "units": "mm",
        "cell_methods": None,
    },
    "unit_discharge_x": {
        "standard_name": None,
        "long_name": "Unit discharge x-component (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_x_at_time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Unit discharge x-component at the time of maximum unit discharge",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_y": {
        "standard_name": None,
        "long_name": "Unit discharge y-component (flow rate per meter of flow width)",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "unit_discharge_y_at_time_of_max_unit_discharge": {
        "standard_name": None,
        "long_name": "Unit discharge y-component at the time of maximum unit discharge",
        "units": "m2 s-1",
        "cell_methods": None,
    },
    "velocity": {
        "standard_name": None,
        "long_name": "Flow velocity in the link at each report step (signed)",
        "units": "m s-1",
        "cell_methods": None,
    },
    "velocity_last": {
        "standard_name": None,
        "long_name": "Flow velocity in the link at the final report step (signed)",
        "units": "m s-1",
        "cell_methods": None,
    },
    "velocity_max": {
        "standard_name": None,
        "long_name": (
            "Maximum signed flow velocity in the link over the report steps (largest value, not largest magnitude)"
        ),
        "units": "m s-1",
        "cell_methods": "time: maximum",
    },
    "washoff_total_load": {
        "standard_name": None,
        "long_name": (
            "Total pollutant mass washed off the subcatchment over the run (in a coupled analysis"
            " generated in the hydrology model and not routed)"
        ),
        "units": None,
        "cell_methods": "time: sum",
    },
    "x_flood_velocity_at_time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Flood velocity x-component at the time of maximum flood velocity",
        "units": "m s-1",
        "cell_methods": None,
    },
    "y_flood_velocity_at_time_of_max_flood_velocity": {
        "standard_name": None,
        "long_name": "Flood velocity y-component at the time of maximum flood velocity",
        "units": "m s-1",
        "cell_methods": None,
    },
}

_EDA_ATTRS: dict[str, dict[str, str | None]] = {
    "identical": {
        "standard_name": None,
        "long_name": "Byte-identity flag against the reference (1 identical, 0 not)",
        "units": "1",
        "cell_methods": None,
    },
    "identity_group": {
        "standard_name": None,
        "long_name": "Byte-identity partition label (members sharing a label are byte-identical)",
        "units": "1",
        "cell_methods": None,
    },
    "max_abs_diff": {
        "standard_name": None,
        "long_name": "Maximum absolute difference against the reference, in the compared variable's unit",
        "units": None,
        "cell_methods": None,
        "comment": ("The compared variable differs along raw_output_type; see the units_by_raw_output_type attribute."),
    },
    "n_extent_disagree": {
        "standard_name": None,
        "long_name": "Count of cells wet in exactly one of the two runs",
        "units": "1",
        "cell_methods": None,
    },
    "n_gpus": {
        "standard_name": None,
        "long_name": "Count of GPUs of the compute configuration",
        "units": "1",
        "cell_methods": None,
    },
    "n_mpi": {
        "standard_name": None,
        "long_name": "Count of MPI tasks of the compute configuration",
        "units": "1",
        "cell_methods": None,
    },
    "n_newly_wet": {
        "standard_name": None,
        "long_name": "Count of cells dry in the reference run and wet in the compared run",
        "units": "1",
        "cell_methods": None,
    },
    "n_nodes": {
        "standard_name": None,
        "long_name": "Count of nodes of the compute configuration",
        "units": "1",
        "cell_methods": None,
    },
    "n_omp": {
        "standard_name": None,
        "long_name": "Count of OpenMP threads of the compute configuration",
        "units": "1",
        "cell_methods": None,
    },
    "n_replicates": {
        "standard_name": None,
        "long_name": "Count of replicate members behind the compute configuration",
        "units": "1",
        "cell_methods": None,
    },
}

_PARAMETER_UNITS: dict[str, str] = {
    "TRITON_reporting_timestep_s": "s",
    "flood_velocity_min_depth_m": "m",
    "hpc_consecutive_failure_budget_simulate": "1",
    "hpc_max_queue_wait_min": "min",
    "hpc_max_wait_for_inflight_min": "min",
    "hpc_mem_allocation_for_analysis_output_consolidation_mb": "MB",
    "hpc_mem_allocation_for_setup_mb": "MB",
    "hpc_mem_allocation_for_sim_output_processing_mb": "MB",
    "hpc_no_progress_timeout_min": "min",
    "hpc_restart_times_other": "1",
    "hpc_restart_times_simulate": "1",
    "hpc_runtime_min_for_analysis_output_consolidation": "min",
    "hpc_runtime_min_for_setup": "min",
    "hpc_runtime_min_for_sim_output_processing": "min",
    "hpc_time_min_per_sim": "min",
    "hpc_total_job_duration_min": "min",
    "hpc_total_nodes": "1",
    "hydraulic_timestep_s": "s",
    "local_cpu_cores_for_workflow": "1",
    "local_gpus_for_workflow": "1",
    "manhole_diameter": "m",
    "manhole_loss_coefficient": "1",
    "mem_gb_per_cpu": "GB",
    "n_gpus": "1",
    "n_mpi_procs": "1",
    "n_nodes": "1",
    "n_omp_threads": "1",
    "open_boundaries": "1",
    "process_append_batch_memory_budget_mb": "MiB",
    "process_append_batch_timesteps": "1",
    "process_output_target_chunksize_mb": "MiB",
    "process_timestep_chunk": "1",
}


# --------------------------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------------------------

_TREE_SEGMENT_TIER = {
    "triton": "triton_summary",
    "swmm_node": "node",
    "swmm_link": "link",
    "swmm_node_timeseries": "node",
    "swmm_link_timeseries": "link",
}


def _store_tier(store: Path) -> str:
    """The tier of a per-scenario store from its file name; ``tree`` for a consolidated tree, else ``unknown``."""
    name = store.stem if store.suffix in (".zarr", ".nc") else store.name
    if store.parent.name == "eda":
        return "eda_raw" if name == "raw_byte_identity" else "eda_other"
    if name.endswith("_datatree") or name == "system_datatree":
        return "tree"
    if name == "hydrology_rpt_summary":
        return "hydrology"
    if name == "hydrology_inflow_summary":
        return "runoff"
    if "_perf_" in name:
        return "perf"
    if "SWMM_node" in name or "SWMM_only_node" in name:
        return "node"
    if "SWMM_link" in name or "SWMM_only_link" in name:
        return "link"
    if name.startswith(("TRITON_only_", "TRITONSWMM_TRITON_")):
        return "triton_tseries" if name.endswith("_tseries") else "triton_summary"
    return "unknown"


def _group_tier(store_tier: str, gpath: str) -> str:
    if store_tier != "tree":
        return store_tier
    last = gpath.rstrip("/").split("/")[-1] if gpath else ""
    if last == "parameters":
        return "parameters"
    if last == "performance":
        return "perf"
    return _TREE_SEGMENT_TIER.get(last, "unknown")


def _rename_map(tier: str) -> dict[str, str]:
    if tier in ("triton_tseries", "triton_summary"):
        return _TRITON
    if tier == "node":
        return {**_NODE, **_UNIQUE}
    if tier == "link":
        return {**_LINK, **_UNIQUE}
    if tier in ("hydrology", "runoff"):
        return _UNIQUE
    if tier == "eda_other":
        return _EDA_COMPUTE
    return {}


def _children(group_dir: Path) -> tuple[list[str], list[str]]:
    """``(arrays, groups)`` directly under a zarr group directory, for zarr format 3 and 2."""
    arrays, groups = [], []
    for child in sorted(group_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / ".zarray").exists():
            arrays.append(child.name)
        elif (child / ".zgroup").exists():
            groups.append(child.name)
        elif (child / "zarr.json").exists():
            node = json.loads((child / "zarr.json").read_text()).get("node_type")
            (arrays if node == "array" else groups).append(child.name)
    return arrays, groups


def _read_attrs(node_dir: Path) -> dict[str, Any]:
    if (node_dir / "zarr.json").exists():
        return dict(json.loads((node_dir / "zarr.json").read_text()).get("attributes", {}))
    if (node_dir / ".zattrs").exists():
        return dict(json.loads((node_dir / ".zattrs").read_text()))
    return {}


def _dimension_names(array_dir: Path) -> list[str]:
    if (array_dir / "zarr.json").exists():
        return list(json.loads((array_dir / "zarr.json").read_text()).get("dimension_names") or [])
    return list(_read_attrs(array_dir).get("_ARRAY_DIMENSIONS", []))


def _walk_groups(root: Path, rel: str = "") -> list[str]:
    out = [rel]
    arrays, groups = _children(root / rel if rel else root)
    for g in groups:
        out.extend(_walk_groups(root, f"{rel}/{g}" if rel else g))
    return out


def _array_edit(name: str, tier: str, is_tseries: bool, current: dict[str, Any]) -> dict[str, Any] | None:
    """The attribute edit for the array that will be called ``name``, or None when it needs none."""
    return _drop_if_applied(_array_edit_wanted(name, tier, is_tseries, current), current)


def _drop_if_applied(edit: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any] | None:
    """None when ``current`` already holds every value ``edit`` sets and none of the keys it removes."""
    if edit is None:
        return None
    done = all(current.get(k) == v for k, v in edit.get("set", {}).items()) and not any(
        k in current for k in edit.get("remove", ())
    )
    return None if done else edit


def _array_edit_wanted(name: str, tier: str, is_tseries: bool, current: dict[str, Any]) -> dict[str, Any] | None:
    """The edit that would put ``name`` at its layout-28 attributes, whether or not it already has them."""
    if tier == "unknown":
        return None
    if tier == "parameters":
        field = name.split(".", 1)[1] if name.startswith(("analysis.", "hpc.")) else name
        unit = _PARAMETER_UNITS.get(field)
        return {"set": {"units": unit}} if unit is not None and "units" not in current else None
    if tier in ("eda_raw", "eda_other"):
        entry = _EDA_ATTRS.get(name)
        if entry is None and name in _EDA_COMPUTE.values():
            entry = _ATTRS.get(name)
        if entry is None and tier == "eda_other" and name.startswith("max_abs_diff__"):
            unit = (_ATTRS.get(name[len("max_abs_diff__") :]) or {}).get("units")
            if unit is not None:
                base = name[len("max_abs_diff__") :]
                entry = {"long_name": f"Maximum absolute difference in {base} against the reference", "units": unit}
        if entry is None and tier == "eda_other" and name.startswith("identical__"):
            # A boolean flag: a long_name and no unit (the layout-28 writer's text, frozen here with the tables).
            entry = {"long_name": _IDENTICAL_LONG_NAME.format(base=name[len("identical__") :])}
    else:
        entry = _ATTRS.get(name)
    if entry is None:
        return None
    remove = ["standard_name", "cell_methods"]
    values = {k: entry.get(k) for k in _CF_KEYS}
    if is_tseries:
        values["cell_methods"] = None
    if entry.get("units") is None and name == "flow_balance_error_unsplit":
        remove.append("units")
    setting = {k: v for k, v in values.items() if v is not None}
    if name == "flow_balance_error_unsplit":
        setting["comment"] = _UNSPLIT_COMMENT
    elif name == "max_ponded_depth":
        setting["comment"] = _D18_COMMENT
    return {"remove": [k for k in remove if k not in setting], "set": setting}


def _group_spec(root: Path, rel: str, store_tier: str, store: Path) -> dict[str, Any] | None:
    gdir = root / rel if rel else root
    tier = _group_tier(store_tier, rel)
    arrays, _ = _children(gdir)
    if tier == "unknown":
        for a in arrays:
            if a in _RETIRED_VARIABLES:
                raise _refuse(
                    store, rel, a, "the tier of this store or group cannot be determined from its name or path"
                )
    renames = {o: n for o, n in _rename_map(tier).items() if o in arrays and n not in arrays}
    if tier == "eda_other":
        for a in arrays:
            for prefix in ("identical__", "max_abs_diff__"):
                if a.startswith(prefix):
                    base = a[len(prefix) :]
                    if base in _IDENTITY_VARIABLES:
                        renames[a] = prefix + _IDENTITY_VARIABLES[base]
                    elif base in _AMBIGUOUS_ACROSS_TIERS:
                        raise _refuse(
                            store, rel, a, "this artifact does not record which tier that variable was read from"
                        )
    dims = {d for a in arrays for d in _dimension_names(gdir / a)}
    rename_dims: dict[str, str] = {}
    if "timestep_min" in dims or "timestep_min" in arrays:
        rename_dims["timestep_min"] = "timestep"
        if "timestep_min" in arrays and "timestep" not in arrays:
            renames["timestep_min"] = "timestep"
    if "time_hr" in dims or "time_hr" in arrays:
        rename_dims["time_hr"] = "timestep"
        if "time_hr" in arrays and "timestep" not in arrays:
            renames["time_hr"] = "timestep"
    is_tseries = tier == "triton_tseries"
    array_attrs: dict[str, dict[str, Any]] = {}
    for old in arrays:
        new = renames.get(old, old)
        if new == "timestep":
            attrs = _RUNOFF_TIMESTEP_ATTRS if tier == "runoff" else _TIMESTEP_ATTRS
            edit = _drop_if_applied({"remove": [], "set": dict(attrs)}, _read_attrs(gdir / old))
            if edit is not None:
                array_attrs[new] = edit
            continue
        edit = _array_edit(new, tier, is_tseries, _read_attrs(gdir / old))
        if edit is not None:
            array_attrs[new] = edit
    group_attrs: dict[str, Any] = {}
    attrs = _read_attrs(gdir)
    key_renames = {k: v[0] for k, v in _ATTR_KEYS.items() if k in attrs and v[0] not in attrs}
    if key_renames:
        group_attrs["rename"] = key_renames
        group_attrs["set"] = {f"{v[0]}_units": v[1] for k, v in _ATTR_KEYS.items() if k in attrs}
    if tier == "eda_raw" and any(
        f'"{k}"' in str(attrs.get("units_by_raw_output_type", "")) for k in _RAW_OUTPUT_TYPE_VALUES
    ):
        group_attrs["json_rename"] = {"units_by_raw_output_type": _RAW_OUTPUT_TYPE_VALUES}
    spec: dict[str, Any] = {}
    if renames:
        spec["rename_arrays"] = renames
    if rename_dims:
        spec["rename_dims"] = rename_dims
    if array_attrs:
        spec["array_attrs"] = array_attrs
    if group_attrs:
        spec["group_attrs"] = group_attrs
    return spec or None


def _plan_zarr_store(ctx: MigrationContext, store: Path) -> None:
    store_tier = _store_tier(store)
    groups: dict[str, Any] = {}
    for rel in _walk_groups(store):
        spec = _group_spec(store, rel, store_tier, store)
        if spec is not None:
            groups[rel] = spec
    if groups:
        ctx.zarr_rewrite(store, groups)
    if store_tier == "eda_raw":
        arrays, _ = _children(store)
        if "raw_output_type" in arrays and any(
            str(v) in _RAW_OUTPUT_TYPE_VALUES for v in zarr.open_group(str(store), mode="r")["raw_output_type"][:]
        ):
            ctx.zarr_replace_string_values(store, "raw_output_type", _RAW_OUTPUT_TYPE_VALUES)


def _plan_netcdf(ctx: MigrationContext, path: Path) -> None:
    """A per-scenario NetCDF file: the same rewrite, planned from the file's own variable and dimension names."""
    import xarray as xr

    tier = _store_tier(path)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        names = list(ds.variables)
        dims = set(ds.dims)
        attrs = dict(ds.attrs)
    if tier == "unknown":
        for a in names:
            if a in _RETIRED_VARIABLES:
                raise _refuse(path, "", a, "the tier of this file cannot be determined from its name")
    renames = {o: n for o, n in _rename_map(tier).items() if o in names and n not in names}
    spec: dict[str, Any] = {}
    rename_dims = {}
    if "timestep_min" in dims or "timestep_min" in names:
        rename_dims["timestep_min"] = "timestep"
        if "timestep_min" in names:
            renames["timestep_min"] = "timestep"
    array_attrs: dict[str, Any] = {}
    for old in names:
        new = renames.get(old, old)
        if new == "timestep":
            array_attrs[new] = {"remove": [], "set": dict(_TIMESTEP_ATTRS)}
            continue
        edit = _array_edit(new, tier, tier == "triton_tseries", {})
        if edit is not None:
            array_attrs[new] = edit
    key_renames = {k: v[0] for k, v in _ATTR_KEYS.items() if k in attrs and v[0] not in attrs}
    if renames:
        spec["rename_arrays"] = renames
    if rename_dims:
        spec["rename_dims"] = rename_dims
    if array_attrs:
        spec["array_attrs"] = array_attrs
    if key_renames:
        spec["group_attrs"] = {
            "rename": key_renames,
            "set": {f"{v[0]}_units": v[1] for k, v in _ATTR_KEYS.items() if k in attrs},
        }
    if renames or rename_dims or key_renames or _nc_needs_attrs(path, array_attrs):
        ctx.netcdf_rewrite(path, spec)


def _nc_needs_attrs(path: Path, array_attrs: dict[str, Any]) -> bool:
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        for name, edit in array_attrs.items():
            if name in ds.variables:
                have = dict(ds[name].attrs)
                if any(have.get(k) != v for k, v in edit.get("set", {}).items()):
                    return True
    return False


# --------------------------------------------------------------------------------------------------------------
# JSON sidecars: plot manifests and EDA verdicts name stored variables
# --------------------------------------------------------------------------------------------------------------

_JSON_VARIABLE_MAP: dict[str, str] = {**_TRITON, **{k: v for k, v in _LINK.items() if k != "max_velocity_mps"}}
_JSON_VARIABLE_MAP["max_velocity_mps"] = "max_velocity"

#: The persisted ``validation_report.json`` names the flat-summary variables the cross-sim identity check compared, so
#: its ``variable`` fields follow the identity-artifact table (``max_velocity_mps`` is the TRITON maximum unit
#: discharge, ``max_flow_cms`` the link tier's maximum flow), plus the names no tier decides.
_REPORT_VARIABLE_MAP: dict[str, str] = {**_UNIQUE, **_IDENTITY_VARIABLES}
#: A retired name the identity-artifact table does not name: the report records no tier, so the new name cannot be
#: chosen and the field is refused.
_REPORT_UNDECIDABLE: frozenset[str] = _RETIRED_VARIABLES - _REPORT_VARIABLE_MAP.keys()


def _refuse_report_field(path: Path, variable: str) -> MigrationBlockedError:
    """The refusal for a ``variable`` field of a ``validation_report.json`` whose new name depends on a tier the
    report does not record; raised while planning, so nothing has been written."""
    return MigrationBlockedError(
        f"V0028: {path} names the layout-26 variable {variable!r} in a 'variable' field but the report does not "
        "record which tier it was read from, and the new name depends on the tier. NOTHING HAS BEEN CHANGED by this "
        "migration. Edit the field to the layout-28 name of its tier, and re-run."
    )


def _renamed_json_text(path: Path) -> str | None:
    """The sidecar's text with every ``variable`` / ``variables`` value renamed, or None when nothing changes.

    Only the ``variable`` / ``variables`` fields are rewritten: a free-text field (a ``summary`` or ``detail``) is
    left as written. A ``validation_report.json`` field naming a retired name that no tier-free table decides is
    refused.
    """
    doc = json.loads(path.read_text())
    undecidable: frozenset[str] = frozenset()
    if path.name.endswith(".verdict.json"):
        mapping = _IDENTITY_VARIABLES  # a verdict names the flat-summary variables it compared
    elif path.name == "validation_report.json":
        mapping = _REPORT_VARIABLE_MAP
        undecidable = _REPORT_UNDECIDABLE
    elif path.name == "peak_flood_depth.manifest.json":
        mapping = _TRITON
    else:
        mapping = _JSON_VARIABLE_MAP

    def walk(node: Any) -> bool:
        changed = False
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "variable" and isinstance(value, str) and value in undecidable:
                    raise _refuse_report_field(path, value)
                if key == "variable" and isinstance(value, str) and value in mapping:
                    node[key] = mapping[value]
                    changed = True
                elif key == "variables" and isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v in undecidable:
                            raise _refuse_report_field(path, v)
                    new = [mapping.get(v, v) if isinstance(v, str) else v for v in value]
                    if new != value:
                        node[key] = new
                        changed = True
                else:
                    changed = walk(value) or changed
        elif isinstance(node, list):
            for item in node:
                changed = walk(item) or changed
        return changed

    return json.dumps(doc, indent=2) if walk(doc) else None


# --------------------------------------------------------------------------------------------------------------
# upgrade
# --------------------------------------------------------------------------------------------------------------

_SKIP_DIRS = {".snakemake", "_software", "render_bundle", "__pycache__", ".git", "weather", "extbc", "dats"}


def _iter_artifacts(root: Path):
    """Yield every zarr store (not descending into it), per-scenario NetCDF file and JSON sidecar under ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if d in _SKIP_DIRS:
                continue
            if d.endswith(".zarr"):
                yield "zarr", here / d
                continue
            if d.endswith(".zarr.done") or ".superseded-" in d:
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            if f.endswith(".nc") and here.name == "processed":
                yield "nc", here / f
            elif f.endswith(".manifest.json") and "plots" in here.parts:
                yield "json", here / f
            elif f.endswith(".verdict.json") and here.name == "eda":
                yield "json", here / f
            elif f == "validation_report.json":
                yield "json", here / f


def upgrade(ctx: MigrationContext) -> None:
    """Plan the renames and re-stamps over every store, NetCDF file and sidecar under ``ctx.target_dir``."""
    root = Path(ctx.target_dir)
    for kind, path in _iter_artifacts(root):
        if kind == "zarr":
            _plan_zarr_store(ctx, path)
        elif kind == "nc":
            _plan_netcdf(ctx, path)
        else:
            text = _renamed_json_text(path)
            if text is not None:
                ctx.rewrite_text_preserving_mtime(path, text)
