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


def _child_model(child: Path) -> str:
    """'TRITON-SWMM' when a coupled tier exists on any /sa_* node, else 'TRITON'."""
    store = child / "sensitivity_datatree.zarr"
    if not store.exists():
        return "TRITON"
    try:
        import xarray as xr

        dt = xr.open_datatree(str(store), engine="zarr", consolidated=False)
        grps = set(dt.groups)
        for g in sorted(g for g in grps if g.count("/") == 1 and g.startswith("/sa_")):
            if f"{g}/tritonswmm/triton" in grps:
                return "TRITON-SWMM"
            if f"{g}/triton_only/triton" in grps:
                return "TRITON"
    except Exception:
        return "TRITON"
    return "TRITON"


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
            {"Scope": scope, **{
                _unit_cols[i]: round(col.get(scope, 0) / _MIB, 1)
                for i, col in enumerate(per_col)
            }}
            for scope in scopes
        ]
        df_du = pd.DataFrame(_records, columns=["Scope", *_unit_cols])
        columns_spec = build_columns_spec(
            df_du, visible_columns_default=None, header_filter=True,
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
            html = "<section class='cross-experiment-disk-utilization'><p class='note'>No child crates recorded.</p></section>"
            source_paths = [crates / "_du.json"]

    return emit_plot_with_sources(
        html,
        output_path,
        source_paths=source_paths,
        analysis_dir=analysis_dir,
        provenance=prov,
    )
