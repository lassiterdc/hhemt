"""Cross-experiment clean-vs-resume intercomparison renderer (PIP-1, Phase 5).

Reads the persisted ``combined_intercomparison.json`` read-model (derived CROSS-BUNDLE
by ``bundle/_combine._write_combined_intercomparison``: clean-vs-resume per-compute-config
byte-identity + ``max_abs_diff``, paired via ``compare_variable_exact``) and renders it as
a self-contained **Tabulator** data grid (filterable / sortable, v9/Iteration-2 b1). The
packed 6-field compute-config identity key (``run_mode|n_mpi|n_omp|n_gpus|n_nodes|partition``)
is SPLIT into its six own columns (grouped under "Compute config" in the sidebar), so a
reader can filter/sort by any one field. The rich spatial magnitude panel lives in the
sibling ``cross_experiment_intercomparison_maps`` figure.

Uniform renderer signature per the ``report renderers accept uniform signature`` stipulation;
reads ONLY ``combined_intercomparison.json`` (so ``CombinedBundle.regenerate_report()``
re-renders with no re-merge) and emits via ``emit_plot_with_sources`` (declaring the
read-model as the sole source, satisfying the non-empty-source gate). Same INERT posture as
the compatibility renderer: consumed only by ``_combine.py``'s emit-time direct-render
dispatch, so no Snakefile rule / caption-RST resolution is involved.
"""

from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

import pandas as pd

from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import (
    ProvenanceLog,
    ProvenanceRef,
)
from hhemt.report_renderers._tabulator_defaults import (
    build_columns_spec,
    build_html_document,
    build_options_dict,
)

#: The six compute-config identity fields packed into each pair's ``config`` key by
#: ``_combine._config_identity_from_node_attrs`` (``field=value`` joined on ``|``), split
#: back out into their own Tabulator columns (b1). Integer fields are typed so Tabulator's
#: numeric filters/sorts work.
_CONFIG_FIELDS: tuple[str, ...] = ("run_mode", "n_mpi", "n_omp", "n_gpus", "n_nodes", "partition")
_INT_CONFIG_FIELDS: frozenset[str] = frozenset({"n_mpi", "n_omp", "n_gpus", "n_nodes"})


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    source = analysis_dir / "combined_intercomparison.json"

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment clean-vs-resume intercomparison table (combined_intercomparison.json)",
    ) as artist:
        artist.add_channel(
            "data",
            ProvenanceRef(source_path="combined_intercomparison.json"),
        )
        html = _build_intercomparison_html(source, report_cfg)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=[source],
        analysis_dir=analysis_dir,
        output_format="html",
        provenance=prov,
    )


def _parse_config(config: str) -> dict:
    """Split the packed ``field=value|...`` compute-config key into a per-field dict."""
    out: dict = {f: None for f in _CONFIG_FIELDS}
    for part in str(config).split("|"):
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or key not in out:
            continue
        if key in _INT_CONFIG_FIELDS:
            try:
                out[key] = int(float(value))
            except (TypeError, ValueError):
                out[key] = None
        else:
            out[key] = value
    return out


def _plain_html(heading: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "padding:12px;color:#333;margin:0;}h2{color:#232D4B;}</style></head>"
        "<body><section class='cross-experiment-intercomparison'>" + heading + body + "</section></body></html>"
    )


def _build_intercomparison_html(source: Path, report_cfg) -> str:
    if source.exists():
        payload = _json.loads(source.read_text())
    else:  # combine may not have run; render an honest placeholder
        payload = {"experiments": [], "pairs": []}
    experiments = payload.get("experiments", [])
    pairs = payload.get("pairs", [])

    exp_line = (
        ", ".join(f"{_html.escape(str(e.get('experiment')))} ({_html.escape(str(e.get('role')))})" for e in experiments)
        or "(no experiments recorded)"
    )
    heading = f"<h2>Cross-Experiment Results: clean vs resume</h2>\n<p>Experiments: {exp_line}</p>\n"

    if not pairs:
        body = (
            "<p class='note'>No paired compute-configs found across the two bundles — "
            "the combined report renders the compatibility half only.</p>"
        )
        return _plain_html(heading, body)

    records: list[dict] = []
    for p in pairs:
        row = _parse_config(p.get("config", ""))
        row["variable"] = str(p.get("variable", ""))
        row["event"] = p.get("event_iloc")
        row["clean_vs_resume"] = "identical" if p.get("identical") else "differs"
        row["max_abs_diff"] = p.get("max_abs_diff")
        records.append(row)
    result_fields = ["variable", "event", "clean_vs_resume", "max_abs_diff"]
    df = pd.DataFrame(records, columns=[*_CONFIG_FIELDS, *result_fields])

    columns_spec = build_columns_spec(df, visible_columns_default=None, header_filter=True)
    # NaN -> None so records serialize as JSON null (Tabulator-safe), like scenario_status_appendix.
    df_records = df.astype(object).where(pd.notna(df), None)
    options = build_options_dict(
        df_records,
        columns_spec=columns_spec,
        table_height="540px",
        pagination_size=0,
        persistence_id="cross_experiment_intercomparison",
        extra_options={},
    )
    js_mode = getattr(getattr(report_cfg, "interactive", None), "tabulator_js_mode", "cdn")
    column_groups = [
        (
            "Compute config",
            list(_CONFIG_FIELDS),
            "The six compute-configuration fields, split out of the packed identity key "
            "(run mode, MPI ranks, OpenMP threads, GPUs, nodes, partition).",
        ),
        (
            "Result",
            result_fields,
            "Per key-result variable + event: clean-vs-resume byte-identity and the "
            "magnitude of the largest absolute difference (near-zero = resume reproduced clean).",
        ),
    ]
    return build_html_document(
        title="Cross-Experiment Results: clean vs resume",
        container_id="cross-experiment-intercomparison-table",
        body_heading_html=heading,
        options=options,
        js_mode=js_mode,
        renderer_name="cross_experiment_intercomparison",
        column_groups=column_groups,
    )
