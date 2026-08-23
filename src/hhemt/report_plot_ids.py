"""Canonical report plot-ID minting (ADR-2, single source of truth).

Every report figure carries ONE canonical plot ID minted here. The ID is
threaded three ways with no second source of truth:

  1. the figure-output file STEM (via `plot_output_template`),
  2. the `plot_id` field stamped into the `*.manifest.json` sidecar
     (via `canonical_plot_id`, called by `_figure_emission` at render time),
  3. the `expand()` / `rule all` / `render_report` input lists in all three
     rule-emission generators (workflow.py, bundle/snakefile_generator.py,
     reprocess_snakefile_generator.py) -- all derive from this module.

Grammar (ADR-2): {renderer_kind}[__{descriptor}][__sa.{sa_id}][__evt.{event_id}]
using "." as the within-segment separator. "-" is NOT used: it is absent
from the enforced wildcard charset ^[A-Za-z0-9_.]+$ (C-CHARSET).

This module is the ONLY layout-relevant artifact for figure stems (Decision
D2 option (b)): it is listed in _layout_relevant_files.yaml::layout_relevant.
paths so CI Check B fires on stem-grammar edits WITHOUT putting all of
workflow.py (the toolkit's largest module, overwhelmingly non-layout logic)
in the Check-B blast radius.
"""

from __future__ import annotations

from typing import Literal

#: Per-renderer output extension under each static backend. Part of the
#: on-disk figure filename, hence co-located with the ID grammar (both are
#: stem determinants governed by C-LAYOUT). Lifted verbatim from workflow.py.
_OUTPUT_EXT_BY_RENDERER: dict[str, dict[str, str]] = {
    # Plotly chart-renderer outputs are interactive HTML emitted via pio.to_html;
    # extension must be .html so Snakemake's report engine sets mime_type=text/html
    # and dispatches each figure via <iframe> (which loads HTML correctly under
    # both HTTP and file:// double-click). A .svg extension here triggers
    # mime_type=image/svg+xml and an <img> dispatch that fails to parse HTML.
    "system_overview": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    # The composite per-scenario page (Piece 2). Emits interactive Plotly HTML
    # unconditionally: it composes the plotly builder seams and is rejected at
    # analysis.run() entry under static_backend="matplotlib", so a ".png" entry
    # here would name a file the renderer can never produce.
    "per_sim_event_page": {"matplotlib": ".html", "plotly": ".html"},
    "per_sim_per_sa_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_per_sa_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    "sensitivity_benchmarking": {"matplotlib": ".png", "plotly": ".html"},
    "per_analysis_summary": {"matplotlib": ".html", "plotly": ".html"},
    "scenario_status_appendix": {"matplotlib": ".html", "plotly": ".html"},
    "errors_and_warnings": {"matplotlib": ".html", "plotly": ".html"},
    # Disk utilization is a table renderer -- emits HTML unconditionally
    # (no matplotlib raster branch). Matches per_analysis_summary /
    # scenario_status_appendix / errors_and_warnings.
    "disk_utilization": {"matplotlib": ".html", "plotly": ".html"},
    # Metadata (ADR-14 / C10) is an all-static table/prose page -- emits HTML
    # unconditionally, like the other table renderers above.
    "metadata": {"matplotlib": ".html", "plotly": ".html"},
    # workflow_performance ([Q160](7)): the run timeline + SLURM efficiency page,
    # extracted out of metadata. All-static HTML tables, so both backends resolve to
    # .html exactly as metadata does.
    "workflow_performance": {"matplotlib": ".html", "plotly": ".html"},
    # eda_compute_sensitivity (R11): the in-report adapter for the compute-sensitivity
    # EDA figure family. The EDA free-functions emit interactive Plotly HTML
    # unconditionally (no matplotlib branch), so both backends resolve to .html.
    "eda_compute_sensitivity": {"matplotlib": ".html", "plotly": ".html"},
    # Cross-experiment combined-report figures (PIP-1 Phase 5). HTML tables emitted
    # unconditionally by the two combined renderers (emit_plot_with_sources HTML
    # branch), so both backends resolve to .html — this lets the combined generator's
    # _emit_plot_rule reuse resolve an ext instead of raising KeyError. A NEW renderer
    # key changes NO existing analysis's figure filenames (C-LAYOUT preserved).
    "cross_experiment_compatibility": {"matplotlib": ".html", "plotly": ".html"},
    "cross_experiment_intercomparison": {"matplotlib": ".html", "plotly": ".html"},
    # Cross-experiment spatial diff-maps renderer (b3, Phase 5): emits HTML unconditionally,
    # same as the two cross_experiment renderers above — a key here lets the combined
    # generator's _emit_plot_rule reuse resolve an ext instead of raising KeyError.
    "cross_experiment_intercomparison_maps": {"matplotlib": ".html", "plotly": ".html"},
    # Cross-experiment errors-and-warnings roll-up (F2, v9): emits HTML unconditionally,
    # same as the cross_experiment renderers above — a key here lets the combined
    # generator's _emit_plot_rule reuse resolve an ext instead of raising KeyError.
    "cross_experiment_errors_and_warnings": {"matplotlib": ".html", "plotly": ".html"},
    # Cross-experiment disk-utilization roll-up: emits HTML unconditionally, same as the
    # cross_experiment renderers above — a key here lets the combined generator's
    # _emit_plot_rule reuse resolve an ext instead of raising KeyError.
    "cross_experiment_disk_utilization": {"matplotlib": ".html", "plotly": ".html"},
}


def output_ext_for(static_backend: Literal["matplotlib", "plotly"], renderer_module: str) -> str:
    """Return the output extension for a renderer under the given static backend.

    Three-place output_ext coupling: rule output path, rule report() first arg,
    and rule_all / render_report input lists must all use this same extension.
    """
    return _OUTPUT_EXT_BY_RENDERER[renderer_module][static_backend]


def canonical_plot_id(
    renderer_kind: str,
    *,
    descriptor: str | None = None,
    sa_id: str | None = None,
    event_id: str | None = None,
) -> str:
    """Mint a canonical plot ID per the ADR-2 grammar.

    Segments are joined with "__"; within a segment the separator is ".".
    Order is fixed: renderer_kind, then optional descriptor, then optional
    sa.{sa_id}, then optional evt.{event_id}.

    Callers pass CONCRETE wildcard values (e.g. event_id="year.9_...") to get
    the display/manifest string; the rule-emission generators pass the literal
    brace token (e.g. event_id="{event_id}") to get a Snakefile path template
    via `plot_output_template`.
    """
    segments = [renderer_kind]
    if descriptor is not None:
        segments.append(descriptor)
    if sa_id is not None:
        segments.append(f"sa.{sa_id}")
    if event_id is not None:
        segments.append(f"evt.{event_id}")
    return "__".join(segments)


#: Human display labels per renderer_kind, for humanize_plot_id (K1). Deterministic,
#: single-source, model-agnostic: a pure-TRITON and a coupled figure share the stem
#: grammar, so they resolve to identical labels (G3 fungibility).
_RENDERER_KIND_LABELS: dict[str, str] = {
    # Keys MUST be the renderer_kind literal passed to canonical_plot_id /
    # plot_output_template, NOT the renderer MODULE name — the two differ for the per-sim
    # figures (module per_sim_peak_flood_depth, kind peak_flood_depth). Two dead entries
    # keyed on the module name were retired here; they never matched a minted stem and were
    # invisible only because the generic fallback happened to produce the same words.
    "benchmarking": "Benchmarking",
    "system_overview": "System overview",
    "peak_flood_depth": "Peak flood depth",
    "conduit_flow": "Conduit flow",
    "event_page": "Simulation results",
    "config_diff_maps": "Config-diff maps",
    "b4b_clean_identity": "Cross-hardware raw byte identity over time",
    "eda_cross_sim_identity": "Cross-simulation byte identity",
    "eda_rank_sensitivity": "MPI-rank sensitivity",
    "eda_resume_sensitivity": "Resume sensitivity",
    "eda_cross_hardware_magnitude": "Cross-hardware difference magnitude",
    "eda_compute_sensitivity": "Compute sensitivity",
    "per_analysis_summary": "Analysis summary",
    "scenario_status_appendix": "Scenario status appendix",
    "errors_and_warnings": "Errors and warnings",
    "disk_utilization": "Disk utilization",
    "metadata": "Run metadata",
    "workflow_performance": "Workflow performance",
    "cross_experiment_compatibility": "Cross-experiment compatibility",
    "cross_experiment_intercomparison": "Clean vs resume intercomparison",
    "cross_experiment_intercomparison_maps": "Clean vs resume spatial difference maps",
    "cross_experiment_errors_and_warnings": "Cross-experiment errors and warnings",
    "cross_experiment_disk_utilization": "Cross-experiment disk utilization",
}


def sa_labels_from_status(analysis_dir) -> dict[str, str]:
    """``{sa_id: derived compute-config label}`` from scenario_status.csv, or ``{}``.

    The CSV ships in every bundle AND sits in every live analysis dir, and carries
    every column ``_derive_config_label`` reads (run_mode, n_gpus, n_mpi_procs,
    n_omp_threads, n_nodes, hpc.partition) keyed by sa_id -- so ONE reader serves
    the source-side and bundle-side callers alike.

    The replicate ordinal is appended because ``_derive_config_label`` strips
    ``_rN`` by design ("Replicate suffixes are NOT in the identity, so replicates
    share one label"). Measured on a 28-row status CSV: 14 distinct labels for 28
    sa_ids, so without the ordinal every per-replicate figure would collapse onto
    one card name.

    NEVER raises: an absent or unreadable CSV yields ``{}``, and the caller then
    falls back to today's text, so a bundle without the CSV renders unchanged.

    The ``_config_diff`` import is LOCAL, not module-scope: a module-level edge
    from the plot-id grammar to the EDA package would bind every consumer to it,
    including those that never resolve a label.
    """
    import csv as _csv
    import re as _re
    from pathlib import Path as _Path

    from hhemt.eda._config_diff import _derive_config_label

    path = _Path(analysis_dir) / "scenario_status.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open() as fh:
            for row in _csv.DictReader(fh):
                sa_id = str(row.get("sa_id") or "")
                if not sa_id:
                    continue
                label = _derive_config_label(row)
                m = _re.search(r"_r(\d+)$", sa_id)
                out[sa_id] = f"{label}, replicate {m.group(1)}" if m else label
    except (OSError, ValueError, KeyError):
        return {}
    return out


#: The toolkit-canonical event display-label column. `weather_event_label_column`
#: names the SOURCE column in the user's events CSV; `_project_events_table`
#: renames it to this literal at ingress, so every downstream consumer keys on one
#: name and none of them can disagree about what the column is called.
EVENT_LABEL_COLUMN = "event_label"


def event_display_name(analysis, event_iloc) -> str | None:
    """``event_iloc`` -> configured human display name, or ``None``.

    Reads the label off ``analysis.df_sims``, projected at analysis construction,
    so this performs no file open: a renderer calling it inside ``render()``
    incurs no ``audit_renderer_io`` declared-source obligation.

    NEVER raises. An unset ``weather_event_label_column``, an absent column, an
    out-of-range ``event_iloc`` or an empty cell all yield ``None`` and the caller
    falls back to its current text. Partial coverage is legal by design.
    """
    try:
        df = analysis.df_sims
        if EVENT_LABEL_COLUMN not in df.columns or event_iloc not in df.index:
            return None
        value = df.loc[event_iloc, EVENT_LABEL_COLUMN]
    except Exception:  # noqa: BLE001 -- presentation must never fail a render
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def event_page_reference(analysis, event_iloc) -> str:
    """The reader-facing event reference used on the per-simulation page.

    Carries BOTH identifiers the reader needs and neither one the reader does not:
    the human label for reading the figure, and the `event_iloc` integer for the
    spoken cross-reference an operator makes while doing EDA with the report open
    ("let's look at event iloc 3"). The `event_id` SLUG is deliberately absent --
    it is unspeakable and it already appears on the caption, which is the surface
    that maps a figure to `sims/{event_id}/` and its `_status` flags.

    When no label resolves the return value is `f"event {event_iloc}"`, which is
    byte-identical to the text every current consumer already interpolates -- so an
    unconfigured analysis renders unchanged at every call site with no per-site
    fallback expression.
    """
    label = event_display_name(analysis, event_iloc)
    return f"{label} (event {event_iloc})" if label else f"event {event_iloc}"


def event_labels_from_status(analysis_dir) -> dict[str, str]:
    """``{event_id slug: display label}`` from scenario_status.csv, or ``{}``.

    The event-axis twin of ``sa_labels_from_status``, and deliberately the same
    shape: the CSV ships in every bundle AND sits in every live analysis dir, so
    ONE reader serves the source-side and bundle-side callers alike.

    The slug is the basename of ``scenario_directory`` -- the same string
    ``compute_event_id_slug`` produced when the scenario directory was created --
    which is what makes this map joinable to the ``evt.{event_id}`` segment of a
    canonical plot ID without re-deriving the slug from the indexer columns.

    NEVER raises: an absent or unreadable CSV yields ``{}``, and the caller then
    falls back to today's text, so a bundle without the CSV renders unchanged.
    """
    import csv as _csv
    from pathlib import Path as _Path

    path = _Path(analysis_dir) / "scenario_status.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open() as fh:
            for row in _csv.DictReader(fh):
                scen_dir = str(row.get("scenario_directory") or "")
                label = str(row.get(EVENT_LABEL_COLUMN) or "").strip()
                if not scen_dir or not label:
                    continue
                out[_Path(scen_dir).name] = label
    except (OSError, ValueError, KeyError):
        return {}
    return out


def escape_label_braces(text: str) -> str:
    """Make `text` safe to place in a Snakemake ``report(labels=...)`` VALUE.

    `expand_labels` re-expands every resolved label value through `apply_wildcards`,
    so a `{` or `}` reaching that call is read as a wildcard template and aborts the
    render with `WorkflowError: Failed to resolve wildcards`. Doubling the braces is
    what makes a user-authored label containing one render as itself.

    Declared ONCE and shared, rather than inlined at each call, because there are now
    two independent producers of label values -- `report_label_value` for the
    per-master rules and `humanize_plot_id` for the combined report's `figure` facet
    -- and a second copy of the rule is the drift this module's own single-declaration
    convention exists to prevent.

    ONLY for values that reach a Snakemake label. It is WRONG for the other two
    `humanize_plot_id` consumers: the composed page's `<title>` and the React `name`
    field are never re-expanded, so escaping there would show doubled braces to the
    reader.
    """
    return text.replace("{", "{{").replace("}", "}}")


def report_label_value(mapping, key, fallback_prefix) -> str:
    """One resolved, brace-safe value for a Snakemake ``report(labels=...)`` entry.

    Two jobs, both of which must happen at the same place. It resolves the display
    label for `key`, falling back to `f"{fallback_prefix} {key}"` -- which is the raw
    identifier under a readable prefix, so an unconfigured analysis still reads
    sensibly. And it escapes braces, because `expand_labels` re-expands every
    resolved label VALUE through `apply_wildcards`, so a user-authored label
    containing `{` or `}` would be read as a wildcard template and abort the render
    with `WorkflowError: Failed to resolve wildcards`.

    NEVER raises: a missing key, an empty value, or a non-dict mapping all yield the
    fallback, so a report renders with raw identifiers rather than failing.
    """
    try:
        value = (mapping or {}).get(key)
    except Exception:  # noqa: BLE001 -- presentation must never fail a render
        value = None
    text = str(value).strip() if value is not None else ""
    if not text:
        text = f"{fallback_prefix} {key}"
    return escape_label_braces(text)


#: Snakefile preamble text emitted VERBATIM by every generator that writes rules
#: carrying `report(labels=...)`. Read at Snakefile PARSE time rather than frozen at
#: generation time, because `scenario_status.csv` is written by the
#: `export_scenario_status` RULE -- so at generation time it does not yet exist and a
#: frozen map would be empty on a first run. `workflow.basedir` is the Snakefile's own
#: directory, which is the analysis dir source-side and the bundle root bundle-side, so
#: one block serves every generator with no path threading.
LABEL_GLOBALS_BLOCK = """
from hhemt.report_plot_ids import (
    event_labels_from_status as _event_labels_from_status,
    report_label_value as _report_label_value,
    sa_labels_from_status as _sa_labels_from_status,
)

_EVENT_LABELS = _event_labels_from_status(workflow.basedir)
_SA_LABELS = _sa_labels_from_status(workflow.basedir)
"""


def humanize_plot_id(plot_id: str, sa_labels=None, event_labels=None) -> str:
    """Deterministic plot-id -> human display label (K1).

    Parses the ADR-2 grammar (segments joined by "__"; "." within a segment) and
    produces a reader-facing label. NEVER a per-figure hardcoded string: a new
    renderer/descriptor humanizes via the same rule with no code edit. Strips a
    trailing output extension if present (e.g. "...total.html").
    """
    stem = plot_id
    for _ext in (".html", ".png", ".svg"):
        if stem.endswith(_ext):
            stem = stem[: -len(_ext)]
            break
    segments = stem.split("__")
    kind = segments[0]
    # Fallback sentence-cases the FIRST character only. str.capitalize() lower-cases the
    # remainder, which destroys any correct casing an unlisted kind carries (it is what
    # rendered "eda_cross_sim_identity" as "Eda cross sim identity").
    _generic = kind.replace("_", " ")
    label = _RENDERER_KIND_LABELS.get(kind, _generic[:1].upper() + _generic[1:])
    extras: list[str] = []
    for seg in segments[1:]:
        if seg.startswith("sa."):
            # S4 reframing: an sa_id is admissible as a nominal id, inadmissible where
            # the reader is being shown a compute-config variation -- which is what a
            # card name in a list of sibling figures is. Falls back to the id when no
            # map is supplied, so every non-threading caller renders unchanged.
            _sa = seg[3:]
            extras.append((sa_labels or {}).get(_sa) or f"sub-analysis {_sa}")
        elif seg.startswith("evt."):
            # Mirrors the sa. branch above: a supplied display label wins, the raw
            # slug is the fallback, so every non-threading caller renders unchanged.
            _evt = seg[4:]
            extras.append((event_labels or {}).get(_evt) or f"event {_evt}")
        else:
            # descriptor: "{independent_var}.vs.total" -> "independent var vs total runtime"
            desc = seg.replace(".vs.total", " vs total runtime").replace(".", " ").replace("_", " ")
            extras.append(desc.strip())
    return f"{label}: {', '.join(extras)}" if extras else label


def plot_output_template(
    *,
    renderer_kind: str,
    subdir: str,
    descriptor: str | None = None,
    sa_id: str | None = None,
    event_id: str | None = None,
) -> str:
    """Return the Snakefile-relative output path template for a figure.

    `subdir` is the plots/ subdirectory the figure lives under (e.g.
    "plots", "plots/per_sim/{event_id}", "plots/sensitivity/benchmarking").
    The returned template embeds the canonical plot ID as the file STEM and
    ends in the literal "__OUTPUT_EXT__" token that `_emit_plot_rule`
    substitutes per-backend. Snakemake "{wildcard}" braces in `subdir` and in
    the minted ID survive unescaped (plain-string assembly, no .format()).
    """
    plot_id = canonical_plot_id(
        renderer_kind,
        descriptor=descriptor,
        sa_id=sa_id,
        event_id=event_id,
    )
    return f"{subdir}/{plot_id}__OUTPUT_EXT__"
