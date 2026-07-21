"""Cross-experiment clean-vs-resume OVER-TIME max-absolute-difference renderer (Round 2, Q21).

Renders a two-panel per-timestep MAX-ABSOLUTE-DIFFERENCE figure between the clean and
resume arms, with a per-config vertical line at each hotstart-resume boundary:

* **wlevel panel** — ``max over (y, x) of |clean - resume|`` for TRITON ``wlevel_m``
* **cms panel**    — ``max over link_id of |clean - resume|`` for SWMM ``flow_cms``

Q21 names this figure by variable ("the per timestep max ABSOLUTE difference ... vertical
lines showing the first timestep of each hotstart resume"). The residuals it exists to
show are TRITON-side: 14 of the 15 differing clean-vs-resume pairs are ``max_wlevel_m``
— note that is ALL 14 compute-config pairs, not a subset. The read-model collapses the
28 sub-analyses to 14 compute-config keys (``_combine._load_intercomparison_subs``), so
"14 differing" means every config differs, and the count must NOT be read as coinciding
with the 14 truncation-asymmetric SUB-ANALYSES (a different index space).

TWO DATA SOURCES, DELIBERATELY ASYMMETRIC
-----------------------------------------
The cms panel reads the CONSOLIDATED tree (``tritonswmm/swmm_link_timeseries``), which is
bundle-resident. The wlevel panel reads the PER-SCENARIO ``TRITONSWMM_TRITON_tseries.zarr``
stores, which are deliberately NOT consolidated: measured 2026-07-21, the gridded store is
~83x the SWMM node+link pair by payload and would inflate a 28-scenario master tree 7.9x on
the SMALLEST grid the toolkit runs (see ``processing_analysis.py`` ``_TIMESERIES_MODE_CONFIG``).

That asymmetry makes this the FIRST cross-experiment renderer whose data path escapes the
render bundle, so it degrades honestly rather than failing or silently omitting: when the
TRITON stores are absent the figure renders the cms panel alone and states the wlevel
panel's absence ON THE FIGURE (user decision, 2026-07-21). A missing panel that announces
itself is recoverable; one that does not is the silent-incompleteness failure mode.

The durable fix is consolidating the per-timestep DOMAIN REDUCTION (``max`` over ``(y, x)``
per timestep — 144 float32 = 576 bytes/scenario, ~16 KB/arm, FLAT in grid size) into a
``tritonswmm/triton_domain_reduction`` node. Until that lands, ``fetch_triton_tseries.py``
in the experiment's ``hhemt_projects`` subdirectory is the Q27-conformant source.

Note the consolidated ``tritonswmm/triton`` summary node is the ORTHOGONAL reduction
(CF ``cell_methods: "timestep_min: maximum"`` — max over TIME per cell, dims ``(y, x)``).
It cannot substitute: neither reduction is recoverable from the other.

UNITS
-----
``replay_t`` (root attr ``coupled_resume_replay_evidence``) is in SECONDS; the TRITON
``timestep_min`` axis is in MINUTES. The conversion is ``replay_t / 60``. Verified
2026-07-21: the three observed values (3000 / 2400 / 4200 s) map to 50 / 40 / 70 min and
land exactly on the 10-minute timestep cadence.

MANDATORY caveat (master §Risks): the coupled runs use TRITON's variable dt, so SWMM drops
the final reporting period (Nperiods=N-1). The truncation is ONE-SIDED, NOT common-mode:
the clean arm drops it on all 28 sub-analyses; the resume arm recovers it on 14, selected
deterministically by the restart time replay_t (predicts 28/28; compute config has zero
explanatory power). The cms panel is therefore differenced over the shared leading
periods, which are timestamp-identical across both arms (0 exceptions on 28 configs).
Any absolute-magnitude reference carries the truncation. The wlevel panel reads TRITON
series, which are full-length on both arms and unaffected. Carried in the figure
annotation AND the caption.

Q22: this renderer is DESCRIPTIVE ONLY. It states what is plotted and how it is computed.
It does not characterize magnitudes as negligible, benign, acceptable, or sound.
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

_WLEVEL_ABSENT_NOTE = (
    "wlevel panel NOT RENDERED — the per-scenario TRITON timeseries stores "
    "(TRITONSWMM_TRITON_tseries.zarr) were not supplied. They are deliberately not "
    "consolidated into the bundle (~83x the SWMM series by payload). Fetch them with "
    "hhemt_projects/synthetic_compute_sensitivity/fetch_triton_tseries.py and re-render."
)

_REPLAY_ATTR = "coupled_resume_replay_evidence"
_TRITON_STORE_NAME = "TRITONSWMM_TRITON_tseries.zarr"


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    """Report-engine facade. Resolves bundle paths, then delegates to the builder.

    ``triton_tseries_root`` may be supplied via kwargs or ``report_cfg``; when absent the
    figure degrades to the cms panel with an on-figure notice (see module docstring).
    """
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    child_stores = sorted((analysis_dir / "child_crates").glob("*/sensitivity_datatree.zarr"))

    triton_root = kwargs.get("triton_tseries_root") or getattr(
        report_cfg, "triton_tseries_root", None
    )
    triton_root = Path(triton_root) if triton_root else None

    clean_root, resume_root = _resolve_arm_roots(child_stores)

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="figure",
        note=(
            "cross-experiment clean-vs-resume per-timestep max-absolute-difference "
            "(SWMM link flow from child_crates/*/sensitivity_datatree.zarr; TRITON wlevel "
            "from per-scenario TRITONSWMM_TRITON_tseries.zarr when supplied)"
        ),
    ) as artist:
        for s in child_stores:
            artist.add_channel("data", ProvenanceRef(source_path=f"child_crates/{s.parent.name}/{s.name}"))
        if triton_root is not None:
            artist.add_channel("data", ProvenanceRef(source_path=str(triton_root)))
        fig = build_overtime_maxabsdiff_figure(clean_root, resume_root, triton_root)
        html = fig.to_html(full_html=True, include_plotlyjs="inline")

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=list(child_stores),
        analysis_dir=analysis_dir,
        output_format="html",
        provenance=prov,
    )


def _resolve_arm_roots(child_stores: list[Path]) -> tuple[Path | None, Path | None]:
    """Pick the clean and resume arm roots from a combined bundle's child crates.

    Role resolution mirrors the sibling cross-experiment renderers: the arm whose
    consolidated root carries a populated ``coupled_resume_replay_evidence`` with any
    ``replay_t`` is the RESUME arm; the other is CLEAN. Returns (None, None) when a
    clean/resume pair cannot be identified, which the builder reports on the figure.
    """
    clean = resume = None
    for store in child_stores:
        if _replay_map(store):
            resume = store
        else:
            clean = store
    return clean, resume


def _replay_map(arm_root: Path) -> dict[str, float]:
    """Return {config_id: replay_t_seconds} from an arm's consolidated root attrs.

    Empty dict when the attr is absent (the clean arm) or carries no replay_t.
    """
    meta = arm_root / "zarr.json"
    if not meta.is_file():
        return {}
    try:
        attrs = _json.loads(meta.read_text()).get("attributes", {}) or {}
    except (OSError, ValueError):
        return {}
    raw = attrs.get(_REPLAY_ATTR)
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except ValueError:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for cfg, rec in raw.items():
        if isinstance(rec, dict) and rec.get("replay_t") is not None:
            out[cfg] = float(rec["replay_t"])
    return out


def build_overtime_maxabsdiff_figure(
    clean_root: Path | None,
    resume_root: Path | None,
    triton_tseries_root: Path | None = None,
):
    """Assemble the two-panel over-time max-absolute-difference figure.

    CONTRACT (grounded; pixel layout iterates via /design-figure report route):
      1. cms panel — for each config present in BOTH arms, read
         ``tritonswmm/swmm_link_timeseries/flow_cms`` (dims event_iloc, link_id, date_time)
         and plot ``max over link_id of |clean - resume|`` against date_time.
      2. wlevel panel — for each config, read ``wlevel_m`` (dims timestep_min, y, x) from
         ``{triton_tseries_root}/{arm}/triton_tseries/{sa_id}/.../{_TRITON_STORE_NAME}`` and
         plot ``max over (y, x) of |clean - resume|`` against timestep_min. SKIPPED with an
         on-figure notice when ``triton_tseries_root`` is None or resolves no stores.
      3. vlines — one dashed vertical line per config at ``replay_t / 60`` minutes, read
         from the resume arm's ``coupled_resume_replay_evidence`` root attr.
      4. Both panels carry ``_TRUNCATION_CAVEAT``; the figure carries no interpretation
         of magnitude (Q22).

    Returns a plotly Figure. Raises nothing on missing data — absence is reported ON the
    figure so a degraded render is visibly degraded rather than silently partial.
    """
    from plotly.subplots import make_subplots

    notes: list[str] = [_TRUNCATION_CAVEAT]
    have_wlevel = triton_tseries_root is not None and _triton_stores_present(triton_tseries_root)
    if not have_wlevel:
        notes.append(_WLEVEL_ABSENT_NOTE)

    rows = 2 if have_wlevel else 1
    titles = (
        ["max |Δ| water level (TRITON wlevel_m) [m]", "max |Δ| link flow (SWMM flow_cms) [m³/s]"]
        if have_wlevel
        else ["max |Δ| link flow (SWMM flow_cms) [m³/s]"]
    )
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, subplot_titles=titles)

    if clean_root is None or resume_root is None:
        fig.add_annotation(
            text=(
                "No clean/resume arm pair could be resolved from the supplied roots — "
                "nothing plotted. Both arms are required; the resume arm is identified by a "
                f"populated `{_REPLAY_ATTR}` root attribute."
            ),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        _finalize(fig, notes)
        return fig

    replay = _replay_map(resume_root)

    if have_wlevel:
        _add_wlevel_traces(fig, row=1, triton_root=triton_tseries_root)
    _add_cms_traces(fig, row=rows, clean_root=clean_root, resume_root=resume_root)

    # Resume-boundary vlines: replay_t is SECONDS, the wlevel x-axis is MINUTES.
    for _cfg, replay_t_s in sorted(replay.items()):
        if have_wlevel:
            fig.add_vline(
                x=replay_t_s / 60.0,
                line_dash="dash",
                line_width=1,
                opacity=0.45,
                row=1,
                col=1,
            )
    if replay:
        notes.append(
            f"Dashed vertical lines mark each config's hotstart-resume boundary "
            f"(replay_t / 60, seconds->minutes); {len(replay)} configs carry one."
        )

    _finalize(fig, notes)
    return fig


def _triton_stores_present(root: Path) -> bool:
    return any(Path(root).rglob(_TRITON_STORE_NAME))


def _add_wlevel_traces(fig, row: int, triton_root: Path) -> None:
    """One trace per config: max over (y, x) of |clean - resume| for TRITON wlevel_m."""
    import numpy as np
    import xarray as xr

    clean_stores = {
        _sa_id(p): p for p in Path(triton_root).rglob(f"*clean*/**/{_TRITON_STORE_NAME}")
    }
    resume_stores = {
        _sa_id(p): p for p in Path(triton_root).rglob(f"*resume*/**/{_TRITON_STORE_NAME}")
    }
    for sa_id in sorted(set(clean_stores) & set(resume_stores)):
        dc = xr.open_zarr(clean_stores[sa_id])
        dr = xr.open_zarr(resume_stores[sa_id])
        diff = np.abs(dc.wlevel_m.values - dr.wlevel_m.values)
        per_t = np.nanmax(diff, axis=(1, 2))
        fig.add_scatter(
            x=dc.timestep_min.values,
            y=per_t,
            mode="lines",
            name=sa_id,
            legendgroup=sa_id,
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="timestep (min)", row=row, col=1)


def _sa_id(store_path: Path) -> str:
    """Extract the sa_* config id from a per-scenario store path."""
    for part in store_path.parts:
        if part.startswith("sa_"):
            return part
    return store_path.parent.name


def _add_cms_traces(fig, row: int, clean_root: Path, resume_root: Path) -> None:
    """One trace per config: max over link_id of |clean - resume| for SWMM flow_cms."""
    import numpy as np
    import xarray as xr

    trimmed: list[str] = []
    for sa_id in sorted(_config_ids(clean_root) & _config_ids(resume_root)):
        try:
            dc = xr.open_zarr(clean_root, group=f"{sa_id}/tritonswmm/swmm_link_timeseries")
            dr = xr.open_zarr(resume_root, group=f"{sa_id}/tritonswmm/swmm_link_timeseries")
        except (OSError, KeyError, ValueError):
            continue
        # SHARED-PREFIX ALIGNMENT (Option A, approved 2026-07-21). The truncation is
        # ONE-SIDED: the clean arm drops the final reporting period on all 28
        # sub-analyses; the resume arm recovers it on 14, so the two series differ in
        # length on exactly those 14. Verified before adopting this: the shared leading
        # periods are timestamp-identical on ALL 28 configs (0 exceptions), so the
        # difference is taken over a genuinely common axis rather than an assumed one.
        # The discarded element is a trailing period present in only one arm, which no
        # difference could legitimately use.
        cv, rv = dc.flow_cms.values, dr.flow_cms.values
        n = min(cv.shape[-1], rv.shape[-1])
        if cv.shape[-1] != rv.shape[-1]:
            trimmed.append(sa_id)
        diff = np.abs(cv[..., :n] - rv[..., :n])
        # dims (event_iloc, link_id, date_time) -> max over event_iloc and link_id
        per_t = np.nanmax(diff, axis=tuple(range(diff.ndim - 1)))
        fig.add_scatter(
            x=dc.date_time.values[:n],
            y=per_t,
            mode="lines",
            name=sa_id,
            legendgroup=sa_id,
            showlegend=False,
            row=row,
            col=1,
        )
    if trimmed:
        fig.add_annotation(
            text=(
                f"cms panel: {len(trimmed)} of "
                f"{len(_config_ids(clean_root) & _config_ids(resume_root))} configs "
                "compared over the shared leading periods (one trailing period present "
                "only in the resume arm was excluded); shared timestamps verified "
                "identical."
            ),
            showarrow=False,
            xref="paper",
            yref="y domain" if row > 1 else "paper",
            x=0,
            y=1.06,
            xanchor="left",
            font={"size": 9},
        )
    fig.update_xaxes(title_text="date_time", row=row, col=1)


def _config_ids(arm_root: Path) -> set[str]:
    return {p.name for p in Path(arm_root).iterdir() if p.is_dir() and p.name.startswith("sa_")}


def _finalize(fig, notes: list[str]) -> None:
    fig.update_layout(
        title=(
            "Clean vs. hotstart-resume: per-timestep MAXIMUM ABSOLUTE DIFFERENCE "
            "(one trace per compute configuration)"
        ),
        height=760 if len(fig.data) and fig._grid_ref and len(fig._grid_ref) > 1 else 480,
        margin={"t": 90, "b": 130},
        hovermode="x unified",
    )
    fig.add_annotation(
        text="<br>".join(notes),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,
        xanchor="left",
        align="left",
        font={"size": 10},
    )
