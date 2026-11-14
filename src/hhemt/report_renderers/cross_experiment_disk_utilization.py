"""Cross-experiment disk-utilization renderer (Q7b, iter-2).

Combine-surface variant of disk_utilization: pivots each child bundle's
_status/_du.json into ONE table (rows = scope, one column per model). INERT on
source/bundle generators; consumed only by bundle/_combine.py's emit-time
direct-render dispatch (like the other cross_experiment_* renderers). Reads
child_crates/{eid}/_status/_du.json + each child's sensitivity_datatree.zarr tier
(for the model label) at render time (Option R, CR4-safe) and declares each
_du.json as a source (ADR-6 non-empty-source gate / Gotcha-53 audit).
"""

from __future__ import annotations

from pathlib import Path

from hhemt.du_sentinels import read_du_sentinel
from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef


def _fmt_bytes(size_bytes: int) -> str:
    size: float = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


#: The label of an experiment whose consolidated tree is missing, unreadable or holds no model tier.
MODEL_NOT_RECORDED = "model not recorded"


#: The models an experiment's members can run, in the order the label lists them, each with the
#: tier of the consolidated tree that records it.
_MODEL_TIERS: tuple[tuple[str, str], ...] = (
    ("TRITON", "triton_only/triton"),
    ("TRITON-SWMM", "tritonswmm/triton"),
    ("SWMM", "swmm_only"),
)


def model_from_groups(groups: set[str]) -> str:
    """Every model an experiment's members ran, read from the groups of its consolidated tree.

    The label lists each distinct model once, in the order of `_MODEL_TIERS`, joined by " + "
    (for example "TRITON-SWMM + SWMM" for an experiment with a coupled member and a SWMM-only
    member). A tree whose members hold no model tier returns `MODEL_NOT_RECORDED`, never a guess.
    """
    found: set[str] = set()
    for g in (g for g in groups if g.count("/") == 1 and g.startswith("/member_")):
        for model, tier in _MODEL_TIERS:
            if any(name == f"{g}/{tier}" or name.startswith(f"{g}/{tier}/") for name in groups):
                found.add(model)
    ordered = [model for model, _tier in _MODEL_TIERS if model in found]
    return " + ".join(ordered) if ordered else MODEL_NOT_RECORDED


def model_label_from_models(models: tuple[str, ...]) -> str:
    """The label for the models an `ExperimentTreeSummary` reports: joined by " + ", or `MODEL_NOT_RECORDED`."""
    return " + ".join(models) if models else MODEL_NOT_RECORDED


def _child_model(child: Path) -> str:
    """The experiment's model label (`model_label_from_models`), or `MODEL_NOT_RECORDED` without a readable tree."""
    from hhemt.utils import experiment_tree_summary, resolve_experiment_tree

    store = resolve_experiment_tree(child)
    if not store.exists():
        return MODEL_NOT_RECORDED
    try:
        # Metadata-only summary rather than a full open, which reads every node's zarr.json to
        # answer a question that sits in the member directories. Same label as `model_from_groups`.
        return model_label_from_models(experiment_tree_summary(store).models)
    except Exception:
        return MODEL_NOT_RECORDED


def render(analysis, report_cfg, output_path: Path) -> Path:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    crates = analysis_dir / "child_crates"

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment disk utilization (child_crates/*/_status/_du.json)",
    ) as a:
        cols: list[str] = []  # model labels, in child-sorted order (collision-suffixed)
        per_col: list[dict[str, int]] = []  # scope -> bytes per column
        totals: list[int] = []
        source_paths: list[Path] = []
        if crates.exists():
            for child in sorted(p for p in crates.iterdir() if p.is_dir()):
                sp = child / "_status" / "_du.json"
                source_paths.append(sp)
                a.add_channel("data", ProvenanceRef(source_path=f"child_crates/{child.name}/_status/_du.json"))
                sent = read_du_sentinel(sp)
                label = _child_model(child)
                while label in cols:  # two children same model -> disambiguate by eid
                    label = f"{label} ({child.name})"
                cols.append(label)
                if sent is None:
                    per_col.append({})
                    totals.append(0)
                else:
                    breakdown = sent.get("sub_path_breakdown", {}) or {}
                    per_col.append({str(k): int(v) for k, v in breakdown.items()})
                    totals.append(int(sent.get("disk_utilization_bytes", 0)))
        import pandas as pd

        from hhemt.report_renderers._tabulator_defaults import (
            build_columns_spec,
            build_html_document,
            build_options_dict,
        )

        # Values are carried as MiB FLOATS, never as the pre-formatted byte strings
        # this table used to emit. A Tabulator column of "1.1 GiB" strings sorts
        # lexically (9.9 MiB above 1.1 GiB) and cannot be summed; a single-unit float
        # sorts numerically, sums correctly, and formats through Tabulator's built-in
        # "money" formatter, which needs no JS callable and so survives the JSON
        # options round-trip. MiB is chosen over raw bytes for readability at this
        # corpus's scale (per-scope values span KiB to low GiB).
        _MIB = 1024.0 * 1024.0
        scopes = sorted({s for col in per_col for s in col})
        _unit_cols = [f"{c} (MiB)" for c in cols]
        _records = [
            {"Scope": scope, **{_unit_cols[i]: round(col.get(scope, 0) / _MIB, 1) for i, col in enumerate(per_col)}}
            for scope in scopes
        ]
        df_du = pd.DataFrame(_records, columns=["Scope", *_unit_cols])
        columns_spec = build_columns_spec(
            df_du,
            visible_columns_default=None,
            header_filter=True,
        )
        # The Total row is Tabulator's own column calculation over the displayed
        # values. The shared surface ships no totals feature -- no column spec sets
        # bottomCalc and build_html_document renders no calc row -- but the columns
        # list it returns is the caller's to extend, so this needs no change to
        # _tabulator_defaults. Measured on all four bundles of this generation:
        # sum(sub_path_breakdown) == disk_utilization_bytes exactly, so the summed
        # Total equals the sentinel total this table displayed before.
        for _spec in columns_spec:
            if _spec.get("field") in _unit_cols:
                _spec["bottomCalc"] = "sum"
                _spec["formatter"] = "money"
                _spec["formatterParams"] = {"precision": 1, "symbol": ""}
                _spec["bottomCalcFormatter"] = "money"
                _spec["bottomCalcFormatterParams"] = {"precision": 1, "symbol": ""}
        options = build_options_dict(
            df_du,
            columns_spec=columns_spec,
            table_height="420px",
            pagination_size=0,
            persistence_id="cross_experiment_disk_utilization",
        )
        html = build_html_document(
            title="Cross-Experiment Disk Utilization",
            container_id="cross-experiment-disk-utilization",
            body_heading_html="<h2>Cross-Experiment Disk Utilization</h2>",
            options=options,
            js_mode="cdn",
            renderer_name="cross_experiment_disk_utilization",
        )
        if not source_paths:  # honest empty state + declare the expected dir (ADR-6 D3)
            html = (
                "<section class='cross-experiment-disk-utilization'>"
                "<p class='note'>No experiment is recorded for this combine.</p></section>"
            )
            source_paths = [crates / "_du.json"]

    return emit_plot_with_sources(
        html,
        output_path,
        source_paths=source_paths,
        analysis_dir=analysis_dir,
        provenance=prov,
    )
