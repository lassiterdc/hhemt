"""Appendix renderer: emit scenario_status.csv as a Tabulator data grid (HTML).

Migrated from an inline-CSS HTML table to a self-contained Tabulator data grid
at Phase 7 of the interactive_report_renderers feature. The compound per-column
filter UI (with type-aware operators + AND/OR composition), sidebar
column-visibility controls, and localStorage persistence are wired via the
shared ``_tabulator_defaults`` module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hhemt.report_renderers._tabulator_defaults import (
    build_columns_spec,
    build_html_document,
    build_options_dict,
    sanitize_persistence_id,
)

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render scenario_status.csv to a Tabulator data grid at output_path.

    Sources the CSV from ``analysis.analysis_paths.analysis_dir / scenario_status.csv``
    (written by ``export_scenario_status.py`` as a Snakemake onsuccess/onerror
    hook). When the CSV is missing, emits a placeholder HTML noting the absence
    so the appendix entry is never blank.
    """
    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    if static_backend == "plotly":
        from hhemt.report_renderers._static_backend_warning import (
            warn_no_plotly_branch,
        )
        warn_no_plotly_branch("scenario_status_appendix")

    from hhemt.report_renderers._figure_emission import (
        _validate_source_path,
        emit_plot_with_sources,
    )
    from hhemt.report_renderers._provenance import (
        ProvenanceLog,
        ProvenanceRef,
    )

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    csv_path = analysis_dir / "scenario_status.csv"
    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="scenario_status table (Tabulator data grid)",
    ) as a:
        a.add_channel(
            "data",
            ProvenanceRef(source_path="scenario_status.csv"),
        )
        if csv_path.exists():
            _validate_source_path(csv_path)
            df: pd.DataFrame | None = pd.read_csv(csv_path)
            row_count = int(len(df))
            csv_present = True
        else:
            df = None
            row_count = 0
            csv_present = False

    analysis_id = str(getattr(analysis, "analysis_id", "") or "")
    weather_event_indices: list[str] = list(
        getattr(getattr(analysis, "cfg_analysis", None), "weather_event_indices", []) or []
    )
    html_text = _build_tabulator_html(
        df, report_cfg, csv_present, analysis_id, weather_event_indices
    )

    # Declare the expected source unconditionally (ADR-6 D3): scenario_status.csv
    # is the canonical source even when absent — _validate_source_path accepts
    # non-existent paths, so the info-icon names the expected CSV rather than the
    # render-time provenance gate tripping on an empty source_paths.
    source_paths: list[Path] = [csv_path]

    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths,
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "scenario_status_appendix",
            "table_format": "tabulator",
            "row_count": row_count,
            "csv_present": csv_present,
        },
        provenance=prov,
    )


def _build_tabulator_html(
    df: pd.DataFrame | None,
    report_cfg: report_config,
    csv_present: bool,
    analysis_id: str,
    weather_event_indices: list[str],
) -> str:
    """Build a self-contained Tabulator HTML document for the scenario-status grid.

    Delegates options/columns/document construction to ``_tabulator_defaults``.
    The fallback placeholder body is emitted when the CSV is missing — the
    appendix entry is never blank.
    """
    if not csv_present or df is None:
        placeholder = (
            "<h2>Scenario Status</h2>\n"
            "<p><em>scenario_status.csv not yet written — workflow may have "
            "been killed before the onsuccess/onerror Snakemake hook ran.</em></p>"
        )
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            "<title>Scenario Status</title></head><body>"
            f"{placeholder}"
            "</body></html>\n"
        )

    tab_cfg = report_cfg.scenario_status_appendix.interactive

    # Drop constant columns (single unique value, including all-NaN) to cut
    # clutter in single-model / single-event analyses. nunique(dropna=False) <= 1
    # catches both all-equal and all-NaN columns. Co-design (P2): an all-zero
    # n_resumes column (no hotstart resumes occurred) is auto-hidden here —
    # deliberate and correct (the CSV/df_status still carries it).
    if report_cfg.scenario_status_appendix.hide_constant_columns:
        nunique = df.apply(lambda s: s.nunique(dropna=False))
        df = df[[c for c in df.columns if nunique[c] > 1]]

    # iter 9.4 — Compute column_groups FIRST, then reorder df columns to
    # match the group order so the table's left-to-right column display
    # matches the sidebar's top-to-bottom checklist. Without this, the
    # df preserves CSV column order (from analysis.py::_reorder_df_status_columns)
    # which interleaves Scenario ID + Status fields differently than the
    # sidebar's group taxonomy. Any column not in any group (shouldn't
    # happen post-iter-3.1 grouping, but defensive) stays in its original
    # df position, appended after the grouped columns.
    column_groups = _build_column_groups(df, weather_event_indices)
    grouped_fields: list[str] = []
    for _label, fields, _footnote in column_groups:
        grouped_fields.extend(fields)
    ungrouped_fields = [c for c in df.columns if c not in grouped_fields]
    ordered_fields = grouped_fields + ungrouped_fields
    df = df[ordered_fields]

    columns_spec = build_columns_spec(
        df,
        visible_columns_default=tab_cfg.visible_columns_default,
        header_filter=tab_cfg.header_filter,
    )

    effective_pid = tab_cfg.persistence_id or sanitize_persistence_id(analysis_id)
    persistence_key = f"scenario_status_appendix__{effective_pid}"

    # Sorting re-enabled (named-reporting-sets D2). The iter-8 alignment
    # toggle (headerSort click -> fitDataStretch reinitializeWidth remeasure
    # against a sort-dependent visible-row set) is eliminated structurally by
    # assigning every column an explicit `width` in build_columns_spec:
    # Tabulator skips reinitializeWidth for widthFixed columns
    # (fitDataStretch.js:9), and definition.width sets widthFixed=true
    # (Column.js:871-873). The filter-trigger button's click handler already
    # stopPropagation()s, so clicking the filter ▾ does not sort.
    extra_options: dict = {}

    # Replace NaN with None so the Tabulator data records serialize as JSON
    # null rather than the bare `NaN` JS literal (invalid JSON — strict parsers
    # throw, and headerFilter substring match misses NaN-as-NaN cells).
    # The .astype(object) cast is load-bearing: a float64 column cannot hold
    # None, so `df.where(pd.notna(df), None)` on a numeric column silently
    # coerces None back to NaN — casting to object first lets None persist.
    # build_columns_spec above already inspected the original-dtype df, so the
    # type-aware column filters are unaffected by this object-dtype copy.
    df_records = df.astype(object).where(pd.notna(df), None)

    options = build_options_dict(
        df_records,
        columns_spec=columns_spec,
        table_height=tab_cfg.table_height,
        pagination_size=tab_cfg.pagination_size,
        persistence_id=persistence_key,
        extra_options=extra_options,
    )

    js_mode = getattr(
        getattr(report_cfg, "interactive", None), "tabulator_js_mode", "cdn",
    )

    # Deterministic prefix-dispatch column grouping per /design-recommendation
    # Option A (iter 4, agreed at scratch L4150). Algorithm: first-match-wins
    # over (1) prefix dispatch (perf_/actual_/snakemake_), (2) Scenario ID base
    # set ∪ cfg_analysis.weather_event_indices (config-derived), (3) Status set,
    # (4) elimination-fallback "Independent variables / Other".
    #
    # Provenance citations:
    #   - perf_*  → analysis.py:2418 (PERF_VARS_ORDERED loop) + export_scenario_status.py:8-9 docstring
    #   - actual_* → analysis.py:2427-2434 fixed_actual + analysis.py:2531-2553 (parsed from triton log)
    #   - snakemake_* → analysis.py:2435-2440 fixed_snakemake + analysis.py:2475-2480 (snakemake_ prefix sweep)
    #   - Scenario ID base → analysis.py:2409-2417 fixed_identity (row-identifier subset)
    #   - weather indexers → cfg_analysis.weather_event_indices (config/analysis.py:22-25)
    #   - Status → analysis.py:2414-2415 fixed_identity (workflow-status subset)
    #   - Fallback includes backend_used: per provenance trace at analysis.py:171
    #     (`self.backend = "gpu" if cfg_analysis.run_mode == "gpu" else "cpu"`) →
    #     scenario.py:632 (logged at scenario-setup) → analysis.py:2519/2525 (read back
    #     into df_status). backend_used is a REQUESTED-config value despite its
    #     misleading name; it belongs in Independent variables / Other.
    # iter 9.4 — column_groups is computed earlier (before build_columns_spec)
    # so we can reorder df columns to match the group order. It's reused
    # here for the sidebar payload.

    return build_html_document(
        title="Scenario Status",
        container_id="scenario-status-table",
        body_heading_html="<h2>Scenario Status</h2>\n",
        options=options,
        js_mode=js_mode,
        renderer_name="scenario_status_appendix",
        column_groups=column_groups,
    )


# Reserved Scenario ID base (row-identifier subset of analysis.py:2409-2417
# fixed_identity, with the two workflow-status columns lifted out into the
# Status set below).
_SCENARIO_ID_BASE_FIELDS = frozenset({
    "event_iloc",
    "model_type",
    "scenario_directory",
    "sa_id",
    "subanalysis_id",
    "sub_analysis_iloc",
})

# Reserved Status set (workflow-execution-state subset of analysis.py
# fixed_identity).
_STATUS_FIELDS = frozenset({"scenario_setup", "run_completed"})


def _build_column_groups(
    df: pd.DataFrame,
    weather_event_indices: list[str],
) -> list[tuple[str, list[str], str | None]]:
    """Assign each column of df to a sidebar group deterministically.

    Algorithm (first-match-wins):
      1. Prefix dispatch: ``perf_*``, ``actual_*``, ``snakemake_*`` → 3 groups.
      2. Scenario ID = ``_SCENARIO_ID_BASE_FIELDS`` ∪ ``cfg_analysis.weather_event_indices``.
      3. Status = ``_STATUS_FIELDS``.
      4. Elimination fallback → "Independent variables / Other".

    The output preserves declared group order (Scenario ID first, then Status,
    Independent variables / Other, Performance breakdown, Actual resource
    utilization, Snakemake derived resource allocation). Within a group,
    columns appear in df.columns order — so the sidebar reflects the CSV's
    canonical column ordering (from ``analysis.py::_reorder_df_status_columns``).
    Empty groups are omitted from the output (no zero-row sidebar headers).

    See /design-recommendation Option A at scratch L4150 for the full rationale
    + per-group provenance citations.
    """
    weather_indexers: frozenset[str] = frozenset(weather_event_indices)

    perf_cols: list[str] = []
    actual_cols: list[str] = []
    snakemake_cols: list[str] = []
    scenario_id_cols: list[str] = []
    status_cols: list[str] = []
    other_cols: list[str] = []

    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("perf_"):
            perf_cols.append(col_str)
        elif col_str.startswith("actual_"):
            actual_cols.append(col_str)
        elif col_str.startswith("snakemake_"):
            snakemake_cols.append(col_str)
        elif col_str in _SCENARIO_ID_BASE_FIELDS or col_str in weather_indexers:
            scenario_id_cols.append(col_str)
        elif col_str in _STATUS_FIELDS:
            status_cols.append(col_str)
        else:
            other_cols.append(col_str)

    groups: list[tuple[str, list[str], str | None]] = []

    if scenario_id_cols:
        groups.append((
            "ID",
            scenario_id_cols,
            "Scenario identifiers from analysis yaml and sensitivity analysis .xlsx.",
        ))
    if status_cols:
        groups.append((
            "Status",
            status_cols,
            "Workflow execution state per scenario.",
        ))
    if other_cols:
        groups.append((
            "Configuration",
            other_cols,
            "User input fields (analysis config fields, sensitivity analysis .xlsx fields) "
            "or fields derived directly from user input.",
        ))
    if perf_cols:
        groups.append((
            "Performance Breakdown",
            perf_cols,
            "From the simulation's performance.txt. Invalid for hotstart-resumed runs — "
            "reflects only the resumed timestep range.",
        ))
    if actual_cols:
        groups.append((
            "Actual Resource Utilization",
            actual_cols,
            "Parsed from TRITON-SWMM-generated log file.",
        ))
    if snakemake_cols:
        groups.append((
            "Snakemake Assigned Resources",
            snakemake_cols,
            "Parsed from the Snakemake job's SLURM env at runtime.",
        ))

    return groups
