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
    from hhemt.eda.cross_sim_identity import config_identity_from_node_attrs
    from hhemt.figure_caption import add_figure_caption, content_width_px, text_width_px
    from hhemt.figure_panels import (
        panel_geometry,
        panel_outline_shape,
        side_table_columns,
        side_table_domain,
        watershed_swatch,
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

    # DISPLAY ORDER, declared rather than inherited from string comparison. `sorted()` put
    # TRITON before TRITON-SWMM because "TRITON" < "TRITON-SWMM", which is an accident of
    # alphabet, not a decision. The user asked for the coupled arm first, to match how the
    # config-diff figures are presented.
    #
    # This does NOT share an ordering source with `_config_diff`: that figure renders ONE
    # arm and orders nothing, so there is no source to share. Naming the order here makes it
    # reviewable in one place; a model absent from the list falls to the end, alphabetically,
    # so a new arm appears rather than disappearing.
    _MODEL_DISPLAY_ORDER = ("TRITON-SWMM", "TRITON")
    _seen_models = {str(p.get("model", "")) for p in all_pairs} | set(roles) - {""}
    models = sorted(
        _seen_models,
        key=lambda m: (_MODEL_DISPLAY_ORDER.index(m) if m in _MODEL_DISPLAY_ORDER else len(_MODEL_DISPLAY_ORDER), m),
    )
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

    # PANEL-KEYED SUMMARY, one table per model, mirroring the config-diff figure's shape
    # while substituting the comparison AXIS: that figure compares each group against the
    # minimum-device SERIAL run within one arm, this one compares CLEAN against RESUME for
    # the same config. Copying its "(vs serial)" headers verbatim would label these numbers
    # with a comparison this figure does not perform.
    #
    # The column list is DERIVED from `has_coupled_arm`, never hardcoded: the flow column
    # is meaningful only on the coupled arm, and a fixed five-column header over a
    # pure-TRITON figure is a header-over-cells mismatch -- the defect that made the first
    # attempt at this table undeployable.
    _summary_cols = (
        ["Panel", "# configs in group", "byte-identical clean vs resume?", "max abs depth diff (m)"]
        + (["max abs flow diff (cms)"] if has_coupled_arm else [])
    )

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

    # ---- differing pairs present: per MODEL -> per BYTE-IDENTITY CLASS. The hardware-family
    # level this comment used to name was removed when panels became identity classes: a class
    # may span CPU and GPU where their outputs agree, and split where they do not, so family is
    # a DESCRIPTION of a class rather than a level above it.
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
        """Iter-10 B: GPU families collapse to a bare "GPU".

        The per-model spelling (`GPU (a100-80)`) made a two-family class read
        `GPU (a100-80) + GPU (a6000)`, which is what pushed the rotated panel label past its
        own panel's vertical extent. The user's instruction is the design here: *"I wouldn't
        list the GPUs, i would just say GPU."*

        This is the label's ONLY consumer (`_class_descriptor`), so no other surface loses
        the device model. The panel TABLE still carries each config's full identity — the
        rotated label names the class, the table enumerates it, and duplicating the device
        spelling in both is what made the label the longest string in the figure.
        """
        return "CPU" if fam == "cpu" else "GPU"

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
            # DECLARED panel order. `_group_by_identity` returns its groups in the order it
            # happened to iterate `subs.items()`, i.e. the zarr store's group-traversal
            # order -- reproducible for a fixed store, but not derived from any stated rule,
            # so a store rewritten by different tooling would permute the panels and silently
            # move the "Panel A" label onto a different identity class.
            #
            # Sorted smallest-device-first with the representative label as tiebreak, which
            # is a TOTAL order and mirrors how `_config_diff` orders its own panels
            # ([serial_grp] + sorted non-serial). Both figures now derive their letters from
            # a rule rather than from storage layout.
            # `clean_subs` is bound per-event inside the enclosing loop, so it is captured as
            # a DEFAULT ARGUMENT rather than closed over. A closure would resolve it at call
            # time -- late binding -- and `sorted()` calls the key eagerly, so today it would
            # happen to be correct and would break silently the moment the sort is deferred
            # or the loop advances first. Explicit capture is also what ruff's B023 asks for.
            def _group_order_key(gg, _cs=clean_subs):
                _dev = (
                    _device_count_key({"attrs": _cs[s].get("attrs", {})})
                    for s in gg["members"]
                    if s in _cs
                )
                return (
                    min(_dev, default=()),
                    sorted(gg["labels"])[0] if gg["labels"] else "",
                )

            model_groups = sorted(model_groups, key=_group_order_key)
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
    # EVERY row is a map row. The figure-wide summary table is gone from this path; each
    # model gets its own Panel-keyed table in a band reserved above its first panel, placed
    # by explicit domain rather than occupying a subplot row.
    n_rows = len(row_plan)
    specs = []
    for _entry in row_plan:
        specs.append([{"type": "xy"}, {"type": "xy"}] if has_coupled_arm else [{"type": "xy"}])

    # No `row_heights`/`vertical_spacing`: every domain is assigned from `panel_geometry`
    # below, once the data aspect is known. See the geometry block after `_grid`.
    fig = make_subplots(rows=n_rows, cols=_ncols, specs=specs)

    # The figure-wide verdict table is drawn only on the DEGENERATE path (no panels), where
    # it is the whole figure. Here each model carries its own Panel-keyed table instead.
    # `verdict_rows` is still built above and still consumed by that path.

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

    def _lbl_of(grp, sa) -> str:
        """A member's config label, from the group's parallel members/labels lists.

        The absent rows are built from sa_ids rather than from `side_table_columns`' output,
        so they need this to render a label. Positional over two parallel lists -- no set or
        dict traversal -- so it satisfies the deterministic-labels ruling by construction.
        """
        for _s, _l in zip(grp["members"], grp["labels"], strict=False):
            if _s == sa:
                return _l
        return sa

    def _class_descriptor(ctx_, grp) -> str:
        """Name an identity class by what it holds, not by how it was formed.

        Config-diff can say "Serial CPU reference" because its Panel A IS that run. An
        identity class has no such name, so it is described by its hardware span and size --
        both READ OFF the members the partition produced. The distinction is load-bearing:
        the user ruled panels are formed from byte identity "never from a hardware
        taxonomy", which governs FORMATION. Describing the result is not forming it.

        DETERMINISTIC per the user's ruling: `fams` is sorted before joining and `n` is a
        cardinality, so neither reads set iteration order.
        """
        fams = sorted({_sub_family(ctx_["fam_subs"][s]) for s in grp["members"] if s in ctx_["fam_subs"]})
        n = len({lbl for lbl in grp["labels"]})
        # Dedupe AFTER titling, not before. `fams` is a set of RAW family keys, so two GPU
        # families survive it distinctly and, once both title to "GPU", would join as
        # "GPU + GPU". Sorting the titled set keeps the determinism the docstring promises
        # while collapsing the duplicate: {a100-80, a6000} -> "GPU", {cpu, a100-80} -> "CPU + GPU".
        span = " + ".join(sorted({_family_title(f) for f in fams})) if fams else "unclassified"
        return f"{span}, {n} b4b config{'s' if n != 1 else ''}"

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
    # Row 1 is no longer the summary table -- each model now carries its own table in a
    # reserved band -- so map rows start at 1 rather than 2. Three other sites read this
    # offset and ALL of them move together; renumbering a subset binds domains to rows that
    # exist and hold different content, which renders wrong rather than raising.
    #
    # `_panel_model` is appended in the `else` branch ONLY, the branch that starts a new
    # panel, so it stays index-aligned with `_panel_rows` by construction rather than by a
    # parallel counter that could drift.
    _panel_rows: list[list[int]] = []
    _panel_model: list[str] = []
    for i, entry in enumerate(row_plan):
        r = 1 + i
        if _panel_rows and entry["ctx"] is row_plan[i - 1]["ctx"]:
            _panel_rows[-1].append(r)
        else:
            _panel_rows.append([r])
            _panel_model.append(str(entry["ctx"].get("model", "")))

    _xd0, _yd0, _, _ = _grid(row_plan[0]["ctx"])
    _map_aspect = ((max(_xd0) - min(_xd0)) or 1.0) / ((max(_yd0) - min(_yd0)) or 1.0)

    # `table_px=0`: the figure-wide table row is gone from this path, replaced by one
    # Panel-keyed table per model placed in a reserved band above that model's first panel.
    # Each band is sized from its own row count, so a model with two identity classes gets a
    # taller band than one with a single class.
    _model_first_panel: dict[str, int] = {}
    for _pi, _prows in enumerate(_panel_rows):
        _m = _panel_model[_pi]
        if _m not in _model_first_panel:
            _model_first_panel[_m] = _pi
    _band_px = {
        _pi: int((sum(1 for x in _panel_model if x == _m) + 2) * 26 + 20) + _MODEL_HEADER_GAP_PX
        for _m, _pi in _model_first_panel.items()
    }
    _layout = panel_geometry(
        _panel_rows,
        table_px=0,
        map_aspect=_map_aspect,
        fig_width=_FIG_W,
        n_map_cols=_ncols,
        group_starts=_band_px,
    )
    plot_h = _layout.plot_h
    for _r, _yd in _layout.row_ydom.items():
        for _c in range(1, _ncols + 1):
            next(fig.select_xaxes(row=_r, col=_c)).domain = _layout.map_domains[_c]
            next(fig.select_yaxes(row=_r, col=_c)).domain = _yd
    # `fig.data[0].domain` removed with the table trace: trace 0 is now the first panel's
    # Heatmap, which has no `domain`, so the line raised ValueError on first render. It is
    # the one site in this renumbering that fails loudly rather than silently.

    # Colorbar paper-x DERIVED from each map column's own domain, replacing the
    # hand-placed 0.44 / 0.98 literals. Those two were also inconsistent with each
    # other -- 10 px inside col 1's right edge but 20 px inside col 2's -- so the two
    # colorbars sat at different insets from panels of identical width. One pad, read
    # off the domains, makes them agree and survives a column-layout change.
    # PER-MODEL SUMMARY TABLE, emitted into the band `group_starts` reserved above each
    # model's first panel. Placed by explicit domain with no row/col -- the same actuator
    # the per-panel side table already uses, which is required here because R9.4 removed
    # table cells from `specs` entirely and there is no subplot slot to put this in.
    #
    # Pairs are attributed to a panel through the CONFIG IDENTITY, not the sa_id:
    # `_combine` keys its pair records with `config_identity_from_node_attrs`, which
    # deliberately excludes replicate suffixes so a clean and a resume sub of one config
    # collide. `grp["members"]` are sa_ids. Joining those two directly yields an EMPTY
    # intersection and a table of blank magnitude columns that raises nothing.
    _pairs_by_cfg: dict[str, list[dict]] = {}
    for _p in all_pairs:
        _pairs_by_cfg.setdefault(str(_p.get("config", "")), []).append(_p)

    def _panel_summary_row(_letter: str, _grp, _ctx) -> list:
        """One summary row for one identity-class panel, on the CLEAN-vs-RESUME axis."""
        _ids = {
            config_identity_from_node_attrs(_ctx["fam_subs"][s].get("attrs", {}))
            for s in _grp["members"]
            if s in _ctx["fam_subs"]
        }
        _ps = [q for cid in sorted(_ids) for q in _pairs_by_cfg.get(cid, [])]
        _n_cfg = len({lbl for lbl in _grp["labels"]})
        _depth = [
            q["max_abs_diff"] for q in _ps
            if q.get("variable") == "max_wlevel_m" and q.get("max_abs_diff") is not None
        ]
        _flow = [
            q["max_abs_diff"] for q in _ps
            if q.get("variable") == "max_flow_cms" and q.get("max_abs_diff") is not None
        ]
        _verdict = (
            "no comparable pair" if not _ps
            else ("identical" if all(q.get("identical", True) for q in _ps)
                  else f"{sum(1 for q in _ps if not q.get('identical', True))} of {len(_ps)} differ")
        )
        _row = [f"Panel {_letter}", _n_cfg, _verdict, f"{max(_depth):.4g}" if _depth else "—"]
        # The flow column exists only when the coupled arm does, so the row width tracks
        # `_summary_cols` rather than being fixed at five.
        if has_coupled_arm:
            _row.append(f"{max(_flow):.4g}" if _flow else "—")
        return _row

    _rows_by_model: dict[str, list[list]] = {}
    for _pi, _prows in enumerate(_panel_rows):
        _m = _panel_model[_pi]
        _ctx_pi = row_plan[_prows[0] - 1]["ctx"]
        _letter = chr(ord("A") + len(_rows_by_model.get(_m, [])))
        _rows_by_model.setdefault(_m, []).append(_panel_summary_row(_letter, _ctx_pi["grp"], _ctx_pi))

    for _m, _pi in _model_first_panel.items():
        _dom = _layout.group_domains.get(_pi)
        if _dom is None:
            continue
        _cells = list(zip(*_rows_by_model[_m], strict=False)) if _rows_by_model.get(_m) else [[]]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=_summary_cols, align="left", fill_color="#eef2f7", font=dict(size=10), height=22
                ),
                cells=dict(values=_cells, align="left", font=dict(size=10), height=20),
                domain=dict(x=_layout.table_domain["x"], y=[max(0.0, _dom[0]), min(1.0, _dom[1])]),
            ),
        )

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
        row = 1 + i  # see _panel_rows: row 1 is a map row now, not the summary table
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
            # BOTH ARMS, one row each: the panel aggregates the clean run AND its resumed
            # counterpart for every config in the class, so a table listing only one of them
            # under-reports what the panel displays. Row count doubles, which is the
            # observable the user named -- but the ASSERTION below is the requirement, since
            # a doubled count is equally satisfied by listing one config twice and dropping
            # another.
            g_ = ctx["grp"]
            _clean_members = [s for s in g_["members"] if s in ctx["fam_subs"]]
            _resume_members = [s for s in g_["members"] if s in ctx["resume_subs"]]
            _attrs = {_sa: ctx["fam_subs"].get(_sa, {}).get("attrs", {}) for _sa in g_["members"]}

            def _ordk(a):
                return _device_count_key({"attrs": a})

            _clean_cfg, _clean_nr = side_table_columns(
                labels=g_["labels"], members=_clean_members, attrs_by_sa=_attrs,
                value_by_sa={_sa: int(ctx["fam_subs"].get(_sa, {}).get("n_resumes", 0)) for _sa in _clean_members},
                order_key=_ordk,
            )
            _res_cfg, _res_nr = side_table_columns(
                labels=g_["labels"], members=_resume_members, attrs_by_sa=_attrs,
                value_by_sa={_sa: int(ctx["resume_subs"].get(_sa, {}).get("n_resumes", 0)) for _sa in _resume_members},
                order_key=_ordk,
            )
            # MEMBERSHIP MADE VISIBLE, not a count check and not a hard failure. The table
            # must enumerate exactly the runs the panel's byte-identity claim covers, and a
            # member present on one arm only must SAY so rather than be dropped -- dropping
            # it would let the table imply a comparison the figure did not make, which is the
            # defect this guards. Rendering it as an explicit absence achieves that without
            # destroying the figure, and matches the standing ruling that a check which
            # cannot apply renders N/A rather than failing.
            #
            # BOTH directions are handled. A clean run with no resumed counterpart is the
            # expected asymmetry (`_resume_da` returns None and the panel already draws a
            # "no resumed run" placeholder), but a resumed run with no clean counterpart is
            # equally undifferenceable and would otherwise vanish from the table entirely.
            #
            # Amendment 2: the absent rows are SORTED on the same key as the present rows.
            # They inherit `g_["members"]` order otherwise, which would leave one column
            # ordered two ways -- present rows by (device_count, label), absent rows by
            # storage traversal.
            _clean_only = sorted(
                (s for s in _clean_members if s not in ctx["resume_subs"]),
                key=lambda s: (_ordk(_attrs.get(s, {})), _lbl_of(g_, s)),
            )
            _resume_only = sorted(
                (s for s in _resume_members if s not in ctx["fam_subs"]),
                key=lambda s: (_ordk(_attrs.get(s, {})), _lbl_of(g_, s)),
            )
            _absent_cfg, _absent_role, _absent_nr = [], [], []
            for _sa in _clean_only:
                _absent_cfg.append(_lbl_of(g_, _sa))
                _absent_role.append("compared (absent)")
                _absent_nr.append("—")
            for _sa in _resume_only:
                _absent_cfg.append(_lbl_of(g_, _sa))
                _absent_role.append("reference (absent)")
                _absent_nr.append("—")
            _role_col = ["reference"] * len(_clean_cfg) + ["compared"] * len(_res_cfg) + _absent_role
            _cfg_col = _clean_cfg + _res_cfg + _absent_cfg
            _nr_col = _clean_nr + _res_nr + _absent_nr
            fig.add_trace(
                go.Table(
                    header=dict(
                        # The arm is NAMED in the header. This panel aggregates the clean and
                        # resume runs of every config it lists, so a bare "n_resumes" would be
                        # ambiguous about which arm it counts -- and the two differ by
                        # construction (clean is always 0).
                        # `role` names each row explicitly rather than leaving the reader to
                        # infer it from n_resumes. The user distinguished the two classes by
                        # VALUE (0 vs 3), which works only while the resumed count is
                        # non-zero -- two rows reading `Serial 1r x 1t | 0` would otherwise be
                        # indistinguishable duplicates. The arm qualifier moves out of the
                        # n_resumes header because both arms now appear in the same column.
                        values=["byte-identical configs", "role", "n_resumes"],
                        align="left",
                        fill_color="#eef2f7",
                        font=dict(size=9),
                        height=20,
                    ),
                    cells=dict(
                        values=[_cfg_col, _role_col, _nr_col],
                        align="left",
                        font=dict(size=9),
                        height=18,
                    ),
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
                # Per-MODEL panel letters. `_panel_ix` was a single figure-wide counter, so
                # the second model's first panel read "Panel C" while the user asked for each
                # model section to start at A -- "the TRITON section will have a table and
                # then a panel A". Each config-diff figure is one arm and starts at A; a model
                # section here mirrors one of those figures, so it starts at A too.
                #
                # The reset lives HERE, not beside the increment. This branch is the only
                # place that observes the model CHANGE, and it overwrites `_last_model_header`
                # on the same iteration -- so a comparison made later reads the already-
                # updated value and never fires. The summary table reads these same letters,
                # so a stale reset would put two different names on one panel.
                _panel_ix[0] = 0
                _last_model_header[0] = model
            # Iter-10 B: the rotated label is laid out along the panel's VERTICAL extent, so
            # its overflow axis is the one `textangle=-90` makes counter-intuitive — a label
            # that is "too long" runs past the panel's top and bottom dashed edges, not its
            # sides. The user's success criterion is absolute: "there is no text that overlaps
            # the panel dashed outline at all". Collapsing the GPU spelling above removes the
            # cause in the shapes this dataset produces; this guard is what makes the criterion
            # hold for a dataset that has not been seen, where a class could span more families
            # or carry a longer descriptor.
            #
            # Shrink-to-fit rather than wrap: a wrapped rotated label grows along the
            # PERPENDICULAR axis, which is the 0.016-wide left margin, so wrapping trades an
            # overrun of the horizontal edges for an overrun of the vertical one. Font size is
            # the axis with slack. The floor is 8px — below that the label stops being legible
            # and the honest failure is a visibly small label rather than a silent overlap.
            _label_text = f"<b>Panel {chr(ord('A') + _panel_ix[0])}</b> — {_class_descriptor(ctx, g_)}"
            _panel_px = max(1.0, (_p_top - _p_bot) * _layout.plot_h)
            _label_font = 13
            while _label_font > 8 and text_width_px(_label_text, font_px=_label_font) > _panel_px * 0.92:
                _label_font -= 1
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
                    font=dict(size=_label_font, color="#111"),
                    # `<b>Panel X</b> — {descriptor}`, the format `_config_diff` uses. The
                    # descriptor names what the class CONTAINS, derived from the members the
                    # byte-identity partition actually produced -- so if a future dataset puts
                    # CPU and GPU configs in one class the label says so, rather than
                    # asserting a split that did not happen. The partition stays data-derived;
                    # only the NAME reads hardware, which is description, not classification.
                    text=_label_text,
                )
            )
            _panel_ix[0] += 1
            fig.add_shape(panel_outline_shape(_layout, _span[0], _span[-1]))
            # The watershed ring is drawn on every map here (`_watershed_boundary_traces`)
            # but has never carried a legend key, so the reader saw an unexplained outline.
            # Gated on the polygon actually existing, so the key cannot label a ring that
            # was not drawn. The outline above is deliberately NOT inside this guard --
            # that coupling is the defect figure_panels' module note describes.
            if wpoly:
                _ws_rect, _ws_text = watershed_swatch(_layout, _span[0], _span[-1])
                fig.add_shape(_ws_rect)
                annotations.append(_ws_text)

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
            # Iter-10 A: this subtitle overlapped the dashed panel outline on BOTH axes, and
            # each overlap had its own cause. Horizontally it began at x=0.0 while the outline
            # begins at x0=0.006, so it started OUTSIDE the box and crossed the left edge.
            # Vertically it was anchored `bottom` at exactly `y_top` — the outline's own top
            # edge — so the dash ran through the glyphs, which is the strikethrough the user
            # photographed.
            #
            # Moved INSIDE the panel rather than above it. Above is the tempting placement and
            # it is the wrong one: the model subheader is emitted at `_p_top + MODEL_HEADER_GAP`
            # and a ref row IS its panel's first row, so a subtitle lifted above the top edge
            # collides with that header on the first panel of every model — trading an overlap
            # with a line for an overlap with text. Inside, the band between the top edge and
            # the table/map row is already empty, which is where this text was trying to sit.
            annotations.append(
                dict(
                    x=0.012,
                    y=y_top - _layout.f(3),
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
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
        row = 1 + ei  # see _panel_rows
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
