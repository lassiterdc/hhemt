"""V0029: remove the nineteen settings that the toolkit stopped reading from a written analysis directory.

Layout 28 wrote sixteen report settings, one EDA setting and two land-use column settings into the files
below, and no module read them. Layout 29 deletes each from its model, and the models refuse an unknown key
(``extra="forbid"``), so a file that still holds one no longer loads. This migration deletes each setting at its
path, and nowhere else, from:

* ``cfg_analysis.yaml``: the sixteen report paths under ``report.`` and ``eda.tabulator_js_mode``;
* every ``report_config_run_entry.*.yaml``: the sixteen report paths with ``report.`` dropped, because that file
  is the report configuration's own dump;
* ``cfg_system.yaml`` and every ``_generated/target_*.yaml``: the two land-use column keys.

The removal is by PATH and never by name. ``tabulator_js_mode`` is also the live setting
``report.interactive.tabulator_js_mode``, which four report renderers read, and a written ``cfg_analysis.yaml``
holds both; a removal of the name at every depth would delete the live one. A file that does not exist, and a
path that is absent from a file, are no change, so a second run plans the same removals and writes nothing.

A content-named transport file (``report_config_run_entry.{sha256 prefix}.yaml``) keeps its name after the rewrite:
the name is a handle a Snakefile reads, and nothing compares it with the content. Bundles are not migrated.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 28
version_to: int = 29
description: str = "Remove nineteen settings that no module read from the written analysis and system configurations"

#: The sixteen report settings, as paths below `report`.
_REPORT_PATHS: tuple[tuple[str, ...], ...] = (
    ("per_analysis_summary", "figure_height_padding_inches"),
    ("per_analysis_summary", "figure_height_per_row_inches"),
    ("per_analysis_summary", "table_scale"),
    ("per_sim", "map", "map_tick_step"),
    ("per_sim", "interactive", "colorbar_range_lock"),
    ("per_sim", "interactive", "time_animation"),
    ("per_sim", "interactive", "visible_layers_default"),
    ("sensitivity", "annotation_fontsize"),
    ("sensitivity", "gpu_marker"),
    ("system_map", "bc_marker"),
    ("system_map", "dem_extent_color"),
    ("system_map", "swmm_link_color"),
    ("system_map", "swmm_link_width"),
    ("system_map", "swmm_node_color"),
    ("system_map", "swmm_node_size"),
    ("system_map", "watershed_color"),
)
#: The analysis configuration's own paths: the report paths below `report`, and the EDA setting.
_CFG_ANALYSIS_PATHS: tuple[tuple[str, ...], ...] = tuple(("report", *p) for p in _REPORT_PATHS) + (
    ("eda", "tabulator_js_mode"),
)
#: The two land-use column settings of the system configuration.
_CFG_SYSTEM_PATHS: tuple[tuple[str, ...], ...] = (
    ("landuse_description_colname",),
    ("landuse_plot_color_colname",),
)


def upgrade(ctx: MigrationContext) -> None:
    """Remove the settings that no module read from the files of a written analysis directory.

    Plans one yaml_remove_key per path: the report paths and the EDA bundling path for cfg_analysis.yaml, the
    report paths without their leading report key for every report_config_run_entry file, and the two land-use
    column keys for cfg_system.yaml and every generated target file. A missing file or an absent path is no
    change, so a second run writes nothing, and every other key of every file is kept, including
    report.interactive.tabulator_js_mode.
    """
    root = ctx.target_dir
    plans = (
        ([root / "cfg_analysis.yaml"], _CFG_ANALYSIS_PATHS),
        (sorted(root.glob("report_config_run_entry.*.yaml")), _REPORT_PATHS),
        ([root / "cfg_system.yaml", *sorted((root / "_generated").glob("target_*.yaml"))], _CFG_SYSTEM_PATHS),
    )
    for files, paths in plans:
        for file in files:
            if not file.is_file():
                continue
            for key_path in paths:
                ctx.yaml_remove_key(file, key_path)
