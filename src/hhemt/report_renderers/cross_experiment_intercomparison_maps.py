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
    from plotly.subplots import make_subplots

    from hhemt.eda._config_diff import (
        _CONFIG_DIFF_DEPTH_BAND_M,
        _CONFIG_DIFF_FLOW_BAND_CMS,
        _DIVERGING,
        _REF_DEPTH,
        _REF_FLOW,
        _align_to,
        _apply_mask,
        _conduit_traces,
        _derive_config_label,
        _device_count_key,
        _group_by_identity,
        _heatmap,
        _hw_family_key,
        _load_conduit_geometry,
        _load_subs,
        _signed_pct,
        _symmetric_diff_range,
        _watershed_boundary_traces,
        _watershed_mask,
        _watershed_polygon,
    )
    from hhemt.figure_caption import add_figure_caption, content_width_px
    from hhemt.figure_panels import (
        panel_geometry,
        panel_outline_shape,
        side_table_columns,
        side_table_domain,
    )

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

    # Same expression as `_config_diff`'s summary-table height, and now feeding the same
    # `panel_geometry(table_px=...)` parameter in both.
    _TABLE_PX = int((len(verdict_rows) + 2) * 26 + 20)
    _FIG_W = 1000
    _T_MARGIN = 70
    #: Vertical clearance between a panel's top edge and the model subheader above it.
    #: Was an inline `_layout.f(10)`. Measured on the delivered figure the header and its
    #: subtitle sat 0.00343 paper apart, which at the figure's plot_h (~2915 px) is 10 px --
    #: less than the 13 px header's own ~17 px line box, so the two strings overlapped
    #: rather than merely crowding. Derived from the font size so a font change moves it.
    _MODEL_HEADER_FONT_PX = 13
    _MODEL_HEADER_GAP_PX = int(round(_MODEL_HEADER_FONT_PX * 1.35)) + 6

    caption_text = (
        "Compared quantity: "
        + _VARIABLE_GLOSS["max_wlevel_m"]
        + ". "
        + ("Coupled arm additionally compares " + _VARIABLE_GLOSS["max_flow_cms"] + ". " if has_coupled_arm else "")
        + "Maps show resume minus clean; blue = resume higher. Colour bands are the fixed "
        f"{_CONFIG_DIFF_DEPTH_BAND_M} m / {_CONFIG_DIFF_FLOW_BAND_CMS} cms meaningful-difference "
        "thresholds, identical in both model arms, so one hue is one magnitude everywhere."
        + (" " + _TRUNCATION_CAVEAT if has_coupled_arm else "")
    )

    # THREE-STATE CONTRACT (user ruling, Track 4). The figure has three legitimate
    # shapes and the middle one was missing:
    #   FULL           differing pairs exist -> verdict table + reference
    #                  + (diff, pct) per identity group. The family config listing is a
    #                  SIDE table beside each panel, not a row of its own.
    #   REFERENCE-ONLY zero differing pairs but child bundles loadable -> verdict table
    #                  + reference panel per family (with its side table), NO diff panels, and a
    #                  conditional caption stating why they are absent. Measured on
    #                  generation 01655abb60c2 this is the true state: TRITON 14/14 and
    #                  TRITON-SWMM 28/28 bit-identical, max |resume - clean| = 0.
    #   DEGENERATE     child bundles unavailable at render time -> verdict table only
    #                  (the existing `if not row_plan:` return below).
    # Previously the zero-differing case returned the table alone, which is a table and
    # not a plot -- the reference result map is still a result and still belongs.
    # P9: the diff panels are ABSENT rather than placeheld, and the caption says so in
    # words rather than drawing empty axes.
    diffs_present = bool(panel_keys)

    if not diffs_present:
        caption_text += (
            " No difference panels are drawn: every compared pair is bit-identical, so "
            "resume minus clean is exactly zero everywhere and a difference map would "
            "carry no information. The reference panel below is the clean, uninterrupted "
            "run on each hardware family."
        )

    # ---- differing pairs present: per MODEL -> per HARDWARE FAMILY -> per identity group.
    #
    #      HARDWARE is the OUTER axis and identity is the INNER one, and that order is
    #      load-bearing. A byte-identity group is NOT hardware-homogeneous -- on a
    #      pure-TRITON master every config is byte-identical, so `_group_by_identity` returns
    #      ONE group spanning serial, OpenMP, MPI, hybrid AND GPU. Partitioning identity
    #      groups by "the representative's hardware" therefore yields a family key taken from
    #      an arbitrary member and a baseline that is whichever config happened to be first.
    #      Configs are partitioned by family FIRST; identity then collapses byte-identical
    #      configs into one panel WITHIN a family, which is what it can actually express.
    #
    #      The family rule, the device ordering and the identity partition are all IMPORTED
    #      from eda._config_diff rather than re-derived here, so this figure and
    #      config_diff_maps cannot drift in which run they call the reference.
    crates = combined_root / "child_crates"
    _sub_cache: dict[tuple[str, int], dict] = {}

    def _subs_for(eid: str, evt: int) -> dict:
        if not eid:
            return {}
        key = (eid, evt)
        if key not in _sub_cache:
            try:
                _sub_cache[key] = _load_subs(crates / eid, event_iloc=evt)
            except Exception:
                _sub_cache[key] = {}
        return _sub_cache[key]

    def _fallback_groups(subs_: dict) -> list[dict]:
        """Singleton groups for a bundle with no persisted identity artifact.

        Carries `labels` and `n_resumes` even though the pre-restructure fallback did not.
        `figure_panels.side_table_domain` already documents this gap -- "one consumer's
        fallback path fabricates group dicts carrying neither" -- and this renderer is that
        consumer. Once the side table reads those two keys (R6.3) their absence stops being
        a cosmetic difference and becomes a KeyError on legacy bundles.
        """
        return [
            {
                "members": [k],
                "attrs": v.get("attrs", {}),
                "labels": [v.get("label", k)],
                "run_modes": [v.get("run_mode")],
                "n_resumes": [int(v.get("n_resumes", 0))],
                "wlevel_da": v["wlevel"],
                "flow_da": v.get("flow"),
            }
            for k, v in subs_.items()
        ]

    def _sub_family(s: dict) -> str:
        """This ONE sub's hardware family, via the shared rule (cpu | a6000 | a100-80 | ...)."""
        return _hw_family_key({"run_modes": [s.get("run_mode")], "attrs": s.get("attrs", {})})

    def _sub_devkey(s: dict) -> tuple:
        return _device_count_key({"attrs": s.get("attrs", {})})

    def _family_title(fam: str) -> str:
        return "CPU" if fam == "cpu" else f"GPU ({fam})"

    # ---- row plan: model -> family subsection -> [table] + [reference] + (diff, pct) per group.
    #      The model split stays strictly OUTSIDE the family split.
    row_plan: list[dict] = []
    for model in models:
        evts = sorted({int(p["event_iloc"]) for p in differing_by_model[model]}) or [0]
        clean_eid = roles.get(model, {}).get("clean", "")
        resume_eid = roles.get(model, {}).get("resume", "")
        for evt in evts:
            clean_subs = _subs_for(clean_eid, evt)
            resume_subs = _subs_for(resume_eid, evt)
            if not clean_subs or not resume_subs:
                continue
            # ONE PANEL PER BYTE-IDENTITY EQUIVALENCE CLASS (user ruling). The partition is
            # taken over the model's WHOLE sub set, never per hardware family. The previous
            # code split by family FIRST and its comment named the consequence it was built
            # to avoid -- "on a pure-TRITON master every config is byte-identical, so
            # _group_by_identity returns ONE group spanning serial, OpenMP, MPI, hybrid AND
            # GPU" -- which is precisely the figure the user asked for: one TRITON panel
            # covering CPU and GPU together, and two TRITON-SWMM panels because that arm's
            # CPU and GPU classes genuinely differ (measured: 1.19209e-07, one float32 ULP).
            # Panel count therefore follows the DATA. No hardware token participates.
            try:
                model_groups = _group_by_identity(clean_subs, crates / clean_eid)
            except Exception:
                model_groups = _fallback_groups(clean_subs)
            for _gi, _grp in enumerate(model_groups):
                fam_subs = {k: clean_subs[k] for k in _grp["members"] if k in clean_subs}
                if not fam_subs:
                    continue
                # The family's baseline is its MINIMUM-device clean run: serial-CPU for the
                # cpu family, the 1-GPU run for each GPU hardware token.
                base_sa = min(fam_subs, key=lambda k: _sub_devkey(fam_subs[k]))
                # Within a class the sub-partition is the identity of its own members, so
                # this is a no-op grouping that preserves the existing `groups` contract for
                # the diff/pct rows below.
                try:
                    fam_groups = _group_by_identity(fam_subs, crates / clean_eid)
                except Exception:
                    fam_groups = _fallback_groups(fam_subs)
                if not fam_groups:
                    continue
                ctx = dict(
                    model=model,
                    evt=evt,
                    base_sa=base_sa,
                    fam_subs=fam_subs,
                    clean_eid=clean_eid,
                    resume_subs=resume_subs,
                    groups=fam_groups,
                    grp=_grp,
                )
                # No `famtable` row. The family's config listing used to occupy a 110 px
                # FULL-WIDTH table row above each family's maps; it is now a SIDE table
                # beside that family's panel, which is what `_config_diff` already does
                # ("the config lists live in the per-panel side tables") and what
                # `PanelLayout.side_table_x` exists to place. Rows are therefore uniform
                # map rows, which is the precondition for `panel_geometry`.
                row_plan.append(dict(kind="ref", ctx=ctx))
                if diffs_present:
                    for g in fam_groups:
                        row_plan.append(dict(kind="diff", ctx=ctx, g=g))
                        row_plan.append(dict(kind="pct", ctx=ctx, g=g))

    if not row_plan:
        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Model", "compared pairs", "clean-vs-resume verdict", "max |resume - clean|"],
                    align="left",
                    fill_color="#eef2f7",
                    font=dict(size=11),
                ),
                cells=dict(
                    values=list(zip(*verdict_rows, strict=False)) if verdict_rows else [[]],
                    align="left",
                    font=dict(size=11),
                    height=22,
                ),
            )
        )
        plot_h = max(_TABLE_PX, 120)
        # Declare width + side margins BEFORE wrapping, so the caption's content width
        # is READ from the figure (`content_width_px`) rather than re-derived as
        # `_FIG_W - 60` at the call site. Same number, but the figure is now the single
        # source of it, so a margin change cannot silently desynchronise the wrap.
        fig.update_layout(width=_FIG_W, margin=dict(t=_T_MARGIN, l=30, r=30))
        b_px = add_figure_caption(fig, caption_text, content_w_px=content_width_px(fig), plot_h_px=plot_h)
        fig.update_layout(
            height=plot_h + _T_MARGIN + b_px,
            width=_FIG_W,
            margin=dict(t=_T_MARGIN, l=30, r=30, b=b_px),
            title="Clean vs resume, spatial: child bundles unavailable at render time",
            paper_bgcolor="white",
        )
        return fig

    # FQ7 applies here too: no coupled SWMM tier -> a ONE-column grid, no placeholder.
    _ncols = 2 if has_coupled_arm else 1
    n_rows = 1 + len(row_plan)
    # Rows are now uniform: one summary-table row, then only map rows.
    specs = [[{"type": "table", "colspan": 2}, None] if has_coupled_arm else [{"type": "table"}]]
    for _entry in row_plan:
        specs.append([{"type": "xy"}, {"type": "xy"}] if has_coupled_arm else [{"type": "xy"}])

    # No `row_heights`/`vertical_spacing`: every domain is assigned from `panel_geometry`
    # below, once the data aspect is known. See the geometry block after `_grid`.
    fig = make_subplots(rows=n_rows, cols=_ncols, specs=specs)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Model", "compared pairs", "clean-vs-resume verdict", "max |resume - clean|"],
                align="left",
                fill_color="#eef2f7",
                font=dict(size=11),
            ),
            cells=dict(
                values=list(zip(*verdict_rows, strict=False)) if verdict_rows else [[]],
                align="left",
                font=dict(size=11),
                height=22,
            ),
        ),
        row=1,
        col=1,
    )

    _mask_cache: dict[tuple[str, int], tuple] = {}

    def _grid(ctx):
        key = (ctx["clean_eid"], ctx["evt"])
        if key not in _mask_cache:
            base_da = ctx["fam_subs"][ctx["base_sa"]]["wlevel"]
            xd = [float(v) for v in base_da["x"].values]
            yd = [float(v) for v in base_da["y"].values]
            wpoly = _watershed_polygon(crates / ctx["clean_eid"])
            _mask_cache[key] = (xd, yd, wpoly, _watershed_mask(wpoly, xd, yd))
        return _mask_cache[key]

    def _resume_da(ctx, g):
        """The resumed run for this identity group's representative config."""
        for sa_id in g["members"]:
            s = ctx["resume_subs"].get(sa_id)
            if s is not None:
                return s["wlevel"]
        return None

    def _glabel(g) -> str:
        return _derive_config_label(g.get("attrs", {}))

    # ---- Geometry from hhemt.figure_panels. One PANEL per family: its reference map row
    #      followed by that family's (diff, pct) row pair per identity group. The budget is
    #      `PanelBudget`'s defaults, which ARE `_config_diff`'s numbers -- the user asked for
    #      this figure to be styled "practically identically to the config diff maps", so
    #      adopting that budget rather than a bespoke one is the point, not a side effect.
    #      Unlike the `_config_diff` and `_dem_resolution_plots` migrations, this one is NOT
    #      geometry-preserving: dropping the famtable rows and adopting the shared budget
    #      moves the figure deliberately. ----
    _panel_rows: list[list[int]] = []
    for i, entry in enumerate(row_plan):
        r = 2 + i
        if _panel_rows and entry["ctx"] is row_plan[i - 1]["ctx"]:
            _panel_rows[-1].append(r)
        else:
            _panel_rows.append([r])

    _xd0, _yd0, _, _ = _grid(row_plan[0]["ctx"])
    _map_aspect = ((max(_xd0) - min(_xd0)) or 1.0) / ((max(_yd0) - min(_yd0)) or 1.0)

    _layout = panel_geometry(
        _panel_rows,
        table_px=_TABLE_PX,
        map_aspect=_map_aspect,
        fig_width=_FIG_W,
        n_map_cols=_ncols,
    )
    plot_h = _layout.plot_h
    for _r, _yd in _layout.row_ydom.items():
        for _c in range(1, _ncols + 1):
            next(fig.select_xaxes(row=_r, col=_c)).domain = _layout.map_domains[_c]
            next(fig.select_yaxes(row=_r, col=_c)).domain = _yd
    fig.data[0].domain = dict(x=_layout.table_domain["x"], y=_layout.table_domain["y"])

    # Colorbar paper-x DERIVED from each map column's own domain, replacing the
    # hand-placed 0.44 / 0.98 literals. Those two were also inconsistent with each
    # other -- 10 px inside col 1's right edge but 20 px inside col 2's -- so the two
    # colorbars sat at different insets from panels of identical width. One pad, read
    # off the domains, makes them agree and survives a column-layout change.
    _cbar_x = _layout.colorbar_x

    def _cbar_y(r: int) -> float:
        """Centre of row `r`'s ACTUAL y-domain.

        Previously `1 - (row - 0.5) / n_rows`, which assumed every row was the same height
        -- true only while the uniform `row_heights` were in force. With explicit per-row
        domains that expression drifts from the map it labels.
        """
        d0, d1 = _layout.ydom(r)
        return (d0 + d1) / 2.0

    # ONE shared symmetric range across every diff panel, quantised UP to the ladder, so
    # panels stay mutually comparable and co-located arms coincide when their p99s share a
    # bin. Computed on the MASKED arrays, exactly as the panels are drawn.
    _dw, _pw = [], []
    for entry in row_plan:
        if entry["kind"] != "diff":
            continue
        ctx, g = entry["ctx"], entry["g"]
        _, _, _, wmask = _grid(ctx)
        base_da = ctx["fam_subs"][ctx["base_sa"]]["wlevel"]
        rda = _resume_da(ctx, g)
        if rda is None:
            continue
        d = _align_to(base_da, rda) - np.asarray(base_da.values)
        _dw.append(_apply_mask(d, wmask))
        _pw.append(_apply_mask(_signed_pct(d, np.asarray(base_da.values)), wmask))
    wsym = _symmetric_diff_range(_dw, floor=1e-12) if _dw else _CONFIG_DIFF_DEPTH_BAND_M
    psym = _symmetric_diff_range(_pw, floor=1e-12) if _pw else 0.1

    # Single-element lists so the closures below can mutate them without `nonlocal`.
    _last_model_header = [None]
    _panel_ix = [0]
    annotations = []
    for i, entry in enumerate(row_plan):
        row = 2 + i
        ctx, kind = entry["ctx"], entry["kind"]
        model, evt = ctx["model"], ctx["evt"]
        base_s = ctx["fam_subs"][ctx["base_sa"]]
        base_da = base_s["wlevel"]
        base_label = _derive_config_label(base_s.get("attrs", {}))
        xd, yd, wpoly, wmask = _grid(ctx)
        y_top = _layout.ydom(row)[1]

        if kind == "ref":
            # First row of this family's panel: emit the family's config listing as a SIDE
            # table spanning the whole panel, plus the panel title. This content used to
            # occupy its own full-width row above the maps.
            _span = next(p for p in _panel_rows if row in p)
            _p_top, _p_bot = _layout.ydom(_span[0])[1], _layout.ydom(_span[-1])[0]
            # ONE ROW PER CONFIG, mirroring `_config_diff._panel_config_table` (which is the
            # figure the user named as the target). The previous table listed one row per
            # GROUP with a synthetic "group 1" label and a "6 config(s)" count, so the six
            # configs the panel aggregates were never named and their resume counts never
            # shown -- the reported defect. `labels` and `n_resumes` are parallel per-member
            # lists already carried on the group, sourced from the bundle's
            # scenario_status.csv, so this needs no new data path.
            #
            # n_resumes is read from the RESUME arm, never the clean one: `_load_subs` stamps
            # the count from ITS OWN bundle's scenario_status.csv, and `fam_subs`/`groups` are
            # both built from `clean_subs` -- whose n_resumes is 0 for every config by
            # construction, because a clean run never resumed. Reading the reachable copy
            # would render a column of zeros. `resume_subs` is already carried on ctx.
            g_ = ctx["grp"]
            _cfg_col, _nr_col = side_table_columns(
                labels=g_["labels"],
                members=g_["members"],
                attrs_by_sa={_sa: ctx["fam_subs"].get(_sa, {}).get("attrs", {}) for _sa in g_["members"]},
                value_by_sa={
                    _sa: int(ctx["resume_subs"].get(_sa, {}).get("n_resumes", 0)) for _sa in g_["members"]
                },
                order_key=lambda a: _device_count_key({"attrs": a}),
            )
            fig.add_trace(
                go.Table(
                    header=dict(
                        # The arm is NAMED in the header. This panel aggregates the clean and
                        # resume runs of every config it lists, so a bare "n_resumes" would be
                        # ambiguous about which arm it counts -- and the two differ by
                        # construction (clean is always 0).
                        values=["byte-identical configs", "n_resumes (resume arm)"],
                        align="left",
                        fill_color="#eef2f7",
                        font=dict(size=9),
                        height=20,
                    ),
                    cells=dict(values=[_cfg_col, _nr_col], align="left", font=dict(size=9), height=18),
                    domain=side_table_domain(_layout, _span[0], _span[-1]),
                ),
            )
            # PANEL title: family identity ONLY, lifted into the panel's gap-top band. It
            # previously also carried "clean reference: the clean, uninterrupted run on this
            # hardware family", which the per-row Reference label below states verbatim --
            # and a `ref` row IS its panel's first row, so `_p_top == y_top` and the two
            # annotations resolved to the SAME (x, y, xanchor, yanchor). Two strings, one
            # anchor, overprinted. Dividing the content fixes the duplication; the offset
            # keeps them off one line even when a panel's first row moves.
            # MODEL SUBHEADER, emitted once per model rather than once per panel. The panel
            # identity moved to a rotated left-margin label below, matching
            # `_config_diff._panel_label`. The old single annotation carried both -- model
            # arm AND hardware family -- which is why it read as one long string; and the
            # family half is now wrong by construction, because a panel is an identity class
            # that may span CPU and GPU.
            if model != _last_model_header[0]:
                annotations.append(
                    dict(
                        x=0.0,
                        y=_p_top + _layout.f(_MODEL_HEADER_GAP_PX),
                        xref="paper",
                        yref="paper",
                        xanchor="left",
                        yanchor="bottom",
                        showarrow=False,
                        align="left",
                        font=dict(size=13, color="#111"),
                        text=f"<b>{model}</b>",
                    )
                )
                _last_model_header[0] = model
            annotations.append(
                dict(
                    x=0.016,
                    y=(_p_top + _p_bot) / 2.0,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="middle",
                    textangle=-90,
                    showarrow=False,
                    font=dict(size=13, color="#111"),
                    text=f"<b>Panel {chr(ord('A') + _panel_ix[0])}</b>",
                )
            )
            _panel_ix[0] += 1
            fig.add_shape(panel_outline_shape(_layout, _span[0], _span[-1]))

        if kind == "ref":
            zref = _apply_mask(np.asarray(base_da.values), wmask)
            fig.add_trace(
                _heatmap(
                    zref,
                    zref,
                    x=xd,
                    y=yd,
                    colorscale=_REF_DEPTH,
                    cbar_title="m",
                    cbar_x=_cbar_x[1],
                    cbar_y=_cbar_y(row),
                    cbar_len=_layout.colorbar_len,
                ),
                row=row,
                col=1,
            )
            for _tr in _watershed_boundary_traces(wpoly):
                fig.add_trace(_tr, row=row, col=1)
            if has_coupled_arm and base_s.get("flow") is not None:
                geom = _load_conduit_geometry(crates / ctx["clean_eid"])
                bf = base_s["flow"]
                links = [str(v) for v in bf["link_id"].values]
                vmax = float(np.nanmax(np.abs(np.asarray(bf.values))))
                for tr in _conduit_traces(
                    geom,
                    dict(zip(links, np.asarray(bf.values), strict=False)),
                    colorscale=_REF_FLOW,
                    vmin=0,
                    vmax=(vmax if vmax > 0 else 1.0),
                    cbar_title="cms",
                    cbar_x=_cbar_x[2],
                    cbar_y=_cbar_y(row),
                    cbar_len=_layout.colorbar_len,
                    diverging=False,
                ):
                    fig.add_trace(tr, row=row, col=2)
            annotations.append(
                dict(
                    x=0.0,
                    y=y_top,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    align="left",
                    font=dict(size=11, color="#111"),
                    text=(f"Reference - {base_label}: absolute peak water level, event {evt}"),
                )
            )
            continue

        g = entry["g"]
        g_label = _glabel(g)
        rda = _resume_da(ctx, g)
        if rda is None:
            annotations.append(
                dict(
                    x=0.5,
                    y=0.5,
                    xref=f"x{row} domain",
                    yref=f"y{row} domain",
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(size=11, color="#666"),
                    text=f"{model} - {g_label}: no resumed run for this config at render time",
                )
            )
            continue
        d = _align_to(base_da, rda) - np.asarray(base_da.values)
        if kind == "diff":
            z = _apply_mask(d, wmask)
            fig.add_trace(
                _heatmap(
                    z,
                    z,
                    x=xd,
                    y=yd,
                    colorscale=_DIVERGING,
                    zmid=0,
                    zmin=-wsym,
                    zmax=wsym,
                    cbar_title="m",
                    cbar_x=_cbar_x[1],
                    cbar_y=_cbar_y(row),
                    cbar_len=_layout.colorbar_len,
                ),
                row=row,
                col=1,
            )
            for _tr in _watershed_boundary_traces(wpoly):
                fig.add_trace(_tr, row=row, col=1)
            txt = (
                f"{g_label} (resumed alternate) - {base_label} (clean reference), "
                f"same byte-identity class, event {evt}"
            )
        else:
            z = _apply_mask(_signed_pct(d, np.asarray(base_da.values)), wmask)
            fig.add_trace(
                _heatmap(
                    z,
                    z,
                    x=xd,
                    y=yd,
                    colorscale=_DIVERGING,
                    zmid=0,
                    zmin=-psym,
                    zmax=psym,
                    cbar_title="%",
                    cbar_x=_cbar_x[1],
                    cbar_y=_cbar_y(row),
                    cbar_len=_layout.colorbar_len,
                ),
                row=row,
                col=1,
            )
            txt = f"{g_label}: percent difference vs the clean reference within the same byte-identity class"
        annotations.append(
            dict(
                x=0.0,
                y=y_top,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#111"),
                text=txt,
            )
        )

    for ei, _entry in enumerate(row_plan):
        row = 2 + ei
        for col in range(1, _ncols + 1):
            fig.update_xaxes(
                row=row,
                col=col,
                title_text="x (m)",
                title_font=dict(size=10),
                tickfont=dict(size=9),
                showgrid=False,
                zeroline=False,
            )
            fig.update_yaxes(
                row=row,
                col=col,
                title_text="y (m)" if col == 1 else "",
                title_font=dict(size=10),
                tickfont=dict(size=9),
                showgrid=False,
                zeroline=False,
                showticklabels=(col == 1),
            )

    _caption = (
        caption_text + " ONE PANEL PER BYTE-IDENTITY EQUIVALENCE CLASS, derived from the data rather "
        "than from a hardware taxonomy: every config whose outputs are byte-identical shares a "
        "panel, so a class may span CPU and GPU where those agree and split where they do not. "
        "Each panel's side table names every config in its class with that config's resume-arm "
        "resume count, so byte-identity across differing resume counts is visible rather than "
        "asserted. Panels are grouped under their model arm. The reference within a class is its "
        "minimum-device clean, uninterrupted run. Diff panels share one "
        f"symmetric range (+/-{wsym:.4g} m, +/-{psym:.4g} %), quantised to a shared ladder so "
        "co-located model arms coincide whenever their percentiles share a bin."
    )
    # As in the empty-plan branch: declare width + side margins first so the caption's
    # content width is READ from the figure rather than re-derived at the call site.
    fig.update_layout(width=_FIG_W, margin=dict(t=_T_MARGIN, l=30, r=30))
    b_px = add_figure_caption(fig, _caption, content_w_px=content_width_px(fig), plot_h_px=plot_h)
    fig.update_layout(
        height=plot_h + _T_MARGIN + b_px,
        width=_FIG_W,
        margin=dict(t=_T_MARGIN, l=30, r=30, b=b_px),
        title="Clean vs resume, spatial: resumed alternates vs the clean reference within each byte-identity class",
        annotations=list(fig.layout.annotations) + annotations,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
