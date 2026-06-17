"""Per-field path-rewrite policy table for bundle emission.

Every Pydantic ``Path`` (or ``Optional[Path]``) field declared on
``analysis_config`` and ``system_config`` must appear in
``_PATH_FIELD_POLICY``. The ``test_all_path_fields_have_policy`` test
asserts this exhaustively — a new Path field added to either config
without a corresponding policy entry fails the test loudly rather than
silently leaking an absolute path into the bundle.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PathPolicy(str, Enum):
    """Policy applied to a Pydantic ``Path``-typed cfg field at emit time."""

    BUNDLE_RELATIVE = "bundle_relative"
    """Rewrite the absolute path to its ``analysis_dir``-relative form.

    If the source file is outside ``analysis_dir`` (e.g., DEM, weather
    timeseries, SWMM templates that live elsewhere on HPC), the value
    is set to ``external/{filename}`` mirroring the
    ``_harvest_and_copy_sources`` fallback. No fail-fast — input files
    routinely live outside ``analysis_dir``.
    """

    BUNDLE_RELATIVE_OR_NONE = "bundle_relative_or_None"
    """If the field is ``None``, preserve as ``None``; otherwise apply
    ``BUNDLE_RELATIVE`` semantics."""

    BUNDLE_RELATIVE_LIST = "bundle_relative_list"
    """``list[Path]`` field: apply ``BUNDLE_RELATIVE`` semantics element-wise.

    A ``None`` or empty list serializes to ``[]``. Each element is routed
    through the same absolute-to-relative rewrite as the scalar
    ``BUNDLE_RELATIVE`` policy (``analysis_root`` first, then
    ``system_root``, then ``external/{filename}`` fallback). Required for
    ``static_plot_configs``: the scalar policies only handle a single
    ``str`` value and would pass a non-empty list through unrewritten,
    leaking absolute paths into the bundle.
    """

    FORCED_DOT = "forced_dot"
    """Set to ``"."`` unconditionally. Bundle invariant:
    ``bundle_root == {system_directory|analysis_dir}`` at consume."""

    HELPER_RESOLVED = "helper_resolved"
    """Path is computed by a helper at consume time and is not stored
    statically in the cfg. Not used in the Phase 1 policy table; reserved
    for fields that may move to runtime-derived resolution in later phases."""

    IS_NONE_ACCEPTABLE = "is_None_acceptable"
    """Set to ``None`` at emit time regardless of the original value.

    Host-specific binary/python paths are irrelevant at report-regen
    time (no simulation re-runs); ``None`` is preferred to a stale
    absolute path leaking into the bundle.
    """


# Per-field policy table. Keys must mirror the Pydantic field names on
# system_config (12 entries) and analysis_config (10 entries) verbatim.
_PATH_FIELD_POLICY: dict[str, PathPolicy] = {
    # ---- system_config (12 Path fields) -------------------------------
    "system_directory": PathPolicy.FORCED_DOT,
    "watershed_gis_polygon": PathPolicy.BUNDLE_RELATIVE,
    "DEM_fullres": PathPolicy.BUNDLE_RELATIVE,
    "landuse_lookup_file": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "SWMM_hydraulics": PathPolicy.BUNDLE_RELATIVE,
    "SWMM_hydrology": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "SWMM_full": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "landuse_raster": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "SWMM_software_directory": PathPolicy.IS_NONE_ACCEPTABLE,
    "TRITONSWMM_software_directory": PathPolicy.IS_NONE_ACCEPTABLE,
    "subcatchment_raingage_mapping": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "triton_swmm_configuration_template": PathPolicy.BUNDLE_RELATIVE,
    # ---- analysis_config (Path fields) --------------------------------
    # Phase-4 (4d): python_path retired off analysis_config (no longer a Path field).
    "weather_timeseries": PathPolicy.BUNDLE_RELATIVE,
    "storm_tide_boundary_line_gis": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "weather_event_summary_csv": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "sensitivity_analysis": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "weather_events_to_simulate": PathPolicy.BUNDLE_RELATIVE,
    "analysis_dir": PathPolicy.FORCED_DOT,
    "master_analysis_cfg_yaml": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "brand_theme": PathPolicy.BUNDLE_RELATIVE_OR_NONE,
    "static_plot_configs": PathPolicy.BUNDLE_RELATIVE_LIST,
}


@dataclass
class RewriteResult:
    """Outcome of ``_rewrite_paths_to_relative``.

    Attributes:
        cfg_dict: The rewritten cfg dict (with absolute paths replaced
            per policy).
        invariants: Per-policy bookkeeping consumed by the Phase 3
            ``bundle_manifest.json`` extension. Keys mirror the
            ``PathPolicy`` enum values; values are lists of field names
            that took that policy at this emit.
    """

    cfg_dict: dict
    invariants: dict[str, list[str]] = field(default_factory=dict)


def enumerate_path_fields(cfg_model: type) -> list[str]:
    """Return the names of all ``Path``-typed fields on a Pydantic v2
    model class, including ``Optional[Path]``.

    Uses ``model_fields`` (Pydantic v2 API). The v1 ``__fields__`` API
    is intentionally NOT used — it is removed in Pydantic v2.
    """
    names: list[str] = []
    for name, finfo in cfg_model.model_fields.items():
        annotation = finfo.annotation
        if annotation is Path:
            names.append(name)
            continue
        args = typing.get_args(annotation)
        if args and Path in args:
            names.append(name)
    return names
