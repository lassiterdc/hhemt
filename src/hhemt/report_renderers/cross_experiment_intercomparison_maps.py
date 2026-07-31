"""Cross-experiment clean-vs-resume SPATIAL intercomparison renderer (b3, Phase 5).

Renders the per-DEM-cell depth + per-conduit flow RESUME-minus-CLEAN diff + %-diff
(RdBu diverging: blue = resume HIGHER than clean) for each compute-config pair that
DIFFERS across the combined bundle's two single-arm child bundles (Q17: only differing
pairs). Reuses the eda._config_diff plot machinery (single presentation-truth source).

Data (Option R — no emit-time artifact, CR4-safe): reads the scalar read-model
combined_intercomparison.json (roles + which (config, variable, event) pairs are
identical:false) and RE-READS each intact child_crates/{eid}/sensitivity_datatree.zarr
+ conduit geometry + watershed polygon at render time. The child bundles are shipped in
place, so this is pure rendering over already-shipped data (Q11) with no re-run.

MANDATORY caveat (master §Risks :218): the coupled runs use TRITON's variable dt, so
SWMM drops the final reporting period (Nperiods=N-1). The truncation is ONE-SIDED, NOT
common-mode: the clean arm drops it on all 28 sub-analyses, while the resume arm
recovers it on 14 — a hotstart restarts the coupling-clock FP accumulation at an
exactly-representable checkpoint time, so emission of the final period is a
deterministic function of the restart time replay_t (predicts 28/28; compute config
has zero explanatory power). SWMM series are therefore compared over the shared
leading periods, which are timestamp-identical across both arms (0 exceptions on 28
configs), so the clean-vs-resume DIFFERENCE (this figure's headline) is sound over
that prefix; any ABSOLUTE-magnitude reference panel carries the truncation. TRITON-side
fields are full-length on both arms and are unaffected. The caveat is annotated on the
figure AND carried in the caption.

FIGURE LAYOUT NOTE: the multi-panel plotly layout is owned by the /eda-spinup design
step (as the sibling cross_experiment_intercomparison renderer's docstring already
states for its rich encoding). build_cross_experiment_diff_figure below is the grounded
CONTRACT + a minimal working first figure; /eda-spinup iterates the pixel layout.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

_TRUNCATION_CAVEAT = (
    "Absolute magnitudes inherit the variable-dt SWMM final-period truncation "
    "(Nperiods=N-1). The truncation is ONE-SIDED, not common-mode: the clean arm "
    "drops the final period on every config, while a resumed run recovers it "
    "whenever the restart time falls on the emitting side of SWMM's report-gate "
    "tolerance (measured: 14 of 28 sub-analyses). SWMM series are therefore "
    "compared over the shared leading periods, which are timestamp-identical "
    "across both arms; the clean-vs-resume DIFFERENCE shown here is taken over "
    "that shared prefix. TRITON-side fields are unaffected (full length on both "
    "arms)."
)


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    read_model = analysis_dir / "combined_intercomparison.json"
    child_stores = sorted((analysis_dir / "child_crates").glob("*/sensitivity_datatree.zarr"))

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="figure",
        note="cross-experiment clean-vs-resume spatial diff maps (child_crates/*/sensitivity_datatree.zarr)",
    ) as artist:
        artist.add_channel("data", ProvenanceRef(source_path="combined_intercomparison.json"))
        for s in child_stores:
            artist.add_channel("data", ProvenanceRef(source_path=f"child_crates/{s.parent.name}/{s.name}"))
        fig = build_cross_experiment_diff_figure(analysis_dir)
        html = fig.to_html(full_html=True, include_plotlyjs="inline")

    # Declare the read-model + every child consolidated store as sources (they are the
    # figure's actual reads; ADR-6 non-empty-source gate + honest provenance).
    sources = [read_model, *child_stores]
    emit_plot_with_sources(
        html,
        output_path,
        source_paths=sources,
        analysis_dir=analysis_dir,
        output_format="html",
        provenance=prov,
    )


#: The compared quantity, named on the figure so the reader cannot mistake it for a
#: final-timestep snapshot. max_wlevel_m is a PEAK summary: the per-DEM-cell maximum
#: water level over the WHOLE simulation.
_VARIABLE_GLOSS = {
    "max_wlevel_m": "max_wlevel_m — per-DEM-cell PEAK water level (maximum over the whole simulation)",
    "max_flow_cms": "max_flow_cms — per-SWMM-link PEAK flow (maximum over the whole simulation)",
}

#: Experiment-id suffix -> model label. Mirrors _combine._MODEL_BY_TOKEN so the renderer's
#: model attribution and the read-model's `model` field cannot drift apart.
_MODEL_BY_TOKEN = (("_tritonswmm", "TRITON-SWMM"), ("_triton", "TRITON"))


def _model_of_experiment(eid: str) -> str:
    for tok, label in _MODEL_BY_TOKEN:
        if eid.endswith(tok):
            return label
    return ""


def build_cross_experiment_diff_figure(combined_root: Path):
    """Assemble the clean-vs-resume spatial diff figure from the combined-bundle root.

    Layout grammar is IMPORTED from eda._config_diff, not re-implemented: the same
    `_heatmap` / `_conduit_traces` primitives, the same `_DIVERGING` colorscale with
    zmid=0, and — load-bearing for G3 fungibility — the same FIXED physically-anchored
    bands `_CONFIG_DIFF_DEPTH_BAND_M` / `_CONFIG_DIFF_FLOW_BAND_CMS`. Binding to the
    fixed bands rather than auto-scaling each render means one hue = one magnitude in
    BOTH model arms and ACROSS the config-diff figure, so a reader may compare them.

    Structure:
      row 1        per-model VERDICT table (always drawn): compared pairs, differing
                   pairs, and the max |resume - clean| per model. When a model is
                   bit-identical this row IS the result -- no empty axes is emitted,
                   and the no-difference finding is stated with its pair count.
      rows 2..N    one 2-column panel per differing (model, config, event):
                   col-1 max_wlevel_m depth diff raster (watershed-masked),
                   col-2 max_flow_cms conduit diff, or an explicit pure-TRITON N/A.

    Diff sign convention: resume MINUS clean; blue = resume HIGHER.
    """
    import numpy as np
    import plotly.graph_objects as go
    import xarray as xr
    from plotly.subplots import make_subplots

    from hhemt.bundle._combine import _config_identity_from_node_attrs
    from hhemt.eda._config_diff import (
        _CONFIG_DIFF_DEPTH_BAND_M,
        _CONFIG_DIFF_FLOW_BAND_CMS,
        _DIVERGING,
        _align_to,
        _apply_mask,
        _conduit_traces,
        _derive_config_label,
        _heatmap,
        _load_conduit_geometry,
        _watershed_boundary_traces,
        _watershed_mask,
        _watershed_polygon,
    )
    from hhemt.figure_caption import add_figure_caption

    read_model = combined_root / "combined_intercomparison.json"
    payload = _json.loads(read_model.read_text()) if read_model.exists() else {"experiments": [], "pairs": []}
    all_pairs = payload.get("pairs", [])
    experiments = payload.get("experiments", [])

    # Coupled-arm predicate: only a combine carrying a TRITON-SWMM arm has SWMM link pairs,
    # so the SWMM-specific truncation caveat is categorically inapplicable to a pure-TRITON
    # combine and MUST NOT render (deterministic-only, F9).
    has_coupled_arm = any(p.get("variable") == "max_flow_cms" for p in all_pairs)

    # ---- per-model roles: with two model arms there are FOUR experiments, so clean/resume
    #      must be resolved WITHIN a model, never globally (the iter-2 defect: one model's
    #      comparison silently stood in for both).
    roles: dict[str, dict[str, str]] = {}
    for ex in experiments:
        eid = str(ex.get("experiment", ""))
        roles.setdefault(_model_of_experiment(eid), {})[str(ex.get("role", ""))] = eid

    models = sorted({str(p.get("model", "")) for p in all_pairs} | set(roles) - {""})
    by_model = {m: [p for p in all_pairs if str(p.get("model", "")) == m] for m in models}
    differing_by_model = {m: [p for p in ps if not p.get("identical", True)] for m, ps in by_model.items()}

    # ---- verdict table (row 1) — ALWAYS drawn; it is the whole figure when nothing differs.
    verdict_rows = []
    for m in models:
        ps, diffs = by_model[m], differing_by_model[m]
        mads = [p["max_abs_diff"] for p in ps if p.get("max_abs_diff") is not None]
        if not ps:
            verdict = "not compared (no clean+resume pair for this model)"
        elif not diffs:
            verdict = f"bit-identical on all {len(ps)} compared pairs — resume reproduces clean"
        else:
            verdict = f"{len(diffs)} of {len(ps)} compared pairs differ"
        verdict_rows.append([m or "(unattributed)", len(ps), verdict, f"{max(mads):.4g}" if mads else "—"])

    # A pair that differs ONLY in max_flow_cms still earns a panel: group by (model, config,
    # event) so the depth raster and the conduit diff share one row rather than each spawning
    # a half-empty panel.
    panel_keys: list[tuple[str, str, int]] = []
    for m in models:
        for p in sorted(differing_by_model[m], key=lambda q: (q["config"], q["event_iloc"])):
            key = (m, str(p["config"]), int(p["event_iloc"]))
            if key not in panel_keys:
                panel_keys.append(key)

    _TABLE_PX = int((len(verdict_rows) + 2) * 26 + 20)
    _MAP_PX = 340
    _FIG_W = 1000
    _T_MARGIN = 70

    caption_text = (
        "Compared quantity: " + _VARIABLE_GLOSS["max_wlevel_m"] + ". "
        + ("Coupled arm additionally compares " + _VARIABLE_GLOSS["max_flow_cms"] + ". " if has_coupled_arm else "")
        + "Maps show resume minus clean; blue = resume higher. Colour bands are the fixed "
        f"{_CONFIG_DIFF_DEPTH_BAND_M} m / {_CONFIG_DIFF_FLOW_BAND_CMS} cms meaningful-difference "
        "thresholds, identical in both model arms, so one hue is one magnitude everywhere."
        + (" " + _TRUNCATION_CAVEAT if has_coupled_arm else "")
    )

    if not panel_keys:
        # Honest no-difference state: the verdict table IS the figure. No empty axes.
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Model", "compared pairs", "clean-vs-resume verdict", "max |resume − clean|"],
                    align="left", fill_color="#eef2f7", font=dict(size=11),
                ),
                cells=dict(
                    values=list(zip(*verdict_rows, strict=False)) if verdict_rows else [[]],
                    align="left", font=dict(size=11), height=22,
                ),
            )
        )
        plot_h = max(_TABLE_PX, 120)
        b_px = add_figure_caption(fig, caption_text, content_w_px=_FIG_W - 60, plot_h_px=plot_h)
        fig.update_layout(
            height=plot_h + _T_MARGIN + b_px,
            width=_FIG_W,
            margin=dict(t=_T_MARGIN, l=30, r=30, b=b_px),
            title="Clean vs resume, spatial: resume reproduces clean on every compared pair",
            paper_bgcolor="white",
        )
        return fig

    # ---- differing pairs present: re-open each model's clean + resume child crates.
    def _subs_by_config(root: Path) -> dict[str, dict]:
        """config-identity key -> {label, attrs, wlevel DataArray, flow DataArray|None}."""
        store = root / "sensitivity_datatree.zarr"
        if not store.exists():
            return {}
        dt = xr.open_datatree(str(store), engine="zarr", consolidated=False)
        out: dict[str, dict] = {}
        for g in dt.groups:
            if g.count("/") != 1 or not g.startswith("/sa_"):
                continue
            attrs = dict(dt[g].attrs)
            key = _config_identity_from_node_attrs(attrs)
            if key in out:
                continue  # first representative per config wins (mirrors _combine)
            tri = None
            for cand in (g + "/tritonswmm/triton", g + "/triton_only/triton"):
                try:
                    tri = dt[cand]
                    break
                except KeyError:
                    continue
            if tri is None:
                continue
            try:
                lnk = dt[g + "/tritonswmm/swmm_link"]
            except KeyError:
                lnk = None
            out[key] = {
                "attrs": attrs,
                "label": _derive_config_label(attrs),
                "wlevel": tri["max_wlevel_m"],
                "flow": (lnk["max_flow_cms"] if lnk is not None else None),
            }
        return out

    crates = combined_root / "child_crates"
    cache: dict[str, dict[str, dict]] = {}

    def _side(model: str, role: str) -> dict[str, dict]:
        eid = roles.get(model, {}).get(role, "")
        if not eid:
            return {}
        if eid not in cache:
            cache[eid] = _subs_by_config(crates / eid)
        return cache[eid]

    n_rows = 1 + len(panel_keys)
    specs = [[{"type": "table", "colspan": 2}, None]] + [[{"type": "xy"}, {"type": "xy"}] for _ in panel_keys]
    plot_h = _TABLE_PX + 24 + len(panel_keys) * (_MAP_PX + 70)
    row_heights = [_TABLE_PX / plot_h] + [(_MAP_PX + 70) / plot_h] * len(panel_keys)
    fig = make_subplots(rows=n_rows, cols=2, specs=specs, row_heights=row_heights, vertical_spacing=0.02)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Model", "compared pairs", "clean-vs-resume verdict", "max |resume − clean|"],
                align="left", fill_color="#eef2f7", font=dict(size=11),
            ),
            cells=dict(
                values=list(zip(*verdict_rows, strict=False)) if verdict_rows else [[]],
                align="left", font=dict(size=11), height=22,
            ),
        ),
        row=1, col=1,
    )

    annotations = []
    for i, (model, cfg_key, evt) in enumerate(panel_keys):
        row = 2 + i
        clean, resume = _side(model, "clean"), _side(model, "resume")
        c, r = clean.get(cfg_key), resume.get(cfg_key)
        label = (c or r or {}).get("label", cfg_key)
        if c is None or r is None:
            annotations.append(dict(
                x=0.5, y=0.5, xref=f"x{row} domain", yref=f"y{row} domain", xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=11, color="#666"),
                text=f"{model} — {label}: child bundle unavailable at render time",
            ))
            continue

        cw = c["wlevel"].isel(event_iloc=evt)
        rw = r["wlevel"].isel(event_iloc=evt)
        xd = [float(v) for v in cw["x"].values]
        yd = [float(v) for v in cw["y"].values]
        wpoly = _watershed_polygon(crates / roles[model]["clean"])
        wmask = _watershed_mask(wpoly, xd, yd)
        dw = _apply_mask(_align_to(cw, rw) - np.asarray(cw.values), wmask)

        fig.add_trace(
            _heatmap(
                dw, dw, x=xd, y=yd, colorscale=_DIVERGING, zmid=0,
                zmin=-_CONFIG_DIFF_DEPTH_BAND_M, zmax=_CONFIG_DIFF_DEPTH_BAND_M,
                cbar_title="m", cbar_x=0.44, cbar_y=1 - (row - 0.5) / n_rows, cbar_len=0.6 / n_rows,
            ),
            row=row, col=1,
        )
        # The watershed boundary overlay belongs on the RASTER column only; the conduit
        # column carries no DEM raster for it to bound. Traces are built by _config_diff
        # so the boundary encoding has one source and no artist is constructed here.
        for _tr in _watershed_boundary_traces(wpoly):
            fig.add_trace(_tr, row=row, col=1)

        if c["flow"] is not None and r["flow"] is not None:
            geom = _load_conduit_geometry(crates / roles[model]["clean"])
            cf = c["flow"].isel(event_iloc=evt)
            df = _align_to(cf, r["flow"].isel(event_iloc=evt)) - np.asarray(cf.values)
            links = [str(v) for v in cf["link_id"].values]
            for tr in _conduit_traces(
                geom, dict(zip(links, np.asarray(df), strict=False)),
                colorscale=_DIVERGING, vmin=-_CONFIG_DIFF_FLOW_BAND_CMS, vmax=_CONFIG_DIFF_FLOW_BAND_CMS,
                cbar_title="cms", cbar_x=0.98, cbar_y=1 - (row - 0.5) / n_rows, cbar_len=0.6 / n_rows,
                diverging=True,
            ):
                fig.add_trace(tr, row=row, col=2)
        else:
            annotations.append(dict(
                x=0.5, y=0.5, xref=f"x{row * 2} domain", yref=f"y{row * 2} domain",
                xanchor="center", yanchor="middle", showarrow=False, font=dict(size=11, color="#666"),
                text="N/A — pure-TRITON<br>(no coupled SWMM conduits)",
            ))

        annotations.append(dict(
            x=0.0, y=1 - (row - 1) / n_rows, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
            showarrow=False, align="left", font=dict(size=11, color="#111"),
            text=f"<b>{model}</b> — {label}, event {evt}: max_wlevel_m peak-water-level difference (resume − clean)",
        ))
        for col in (1, 2):
            fig.update_xaxes(row=row, col=col, title_text="x (m)", title_font=dict(size=10),
                             tickfont=dict(size=9), showgrid=False, zeroline=False)
            fig.update_yaxes(row=row, col=col, title_text="y (m)" if col == 1 else "",
                             title_font=dict(size=10), tickfont=dict(size=9),
                             showgrid=False, zeroline=False, showticklabels=(col == 1))

    b_px = add_figure_caption(fig, caption_text, content_w_px=_FIG_W - 60, plot_h_px=plot_h)
    fig.update_layout(
        height=plot_h + _T_MARGIN + b_px,
        width=_FIG_W,
        margin=dict(t=_T_MARGIN, l=30, r=30, b=b_px),
        title="Clean vs resume, spatial: peak water-level difference per differing compute config",
        annotations=list(fig.layout.annotations) + annotations,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
