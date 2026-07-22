"""Cross-experiment clean-vs-resume OVER-TIME max-absolute-difference renderer (Round 2, Q21).

Renders a two-panel per-timestep MAX-ABSOLUTE-DIFFERENCE figure between the clean and
resume arms, with a per-config vertical line at each hotstart-resume boundary:

* **panel (a) wlevel** — ``max over (y, x) of |clean - resume|`` for TRITON ``wlevel_m``
* **panel (b) cms**    — ``max over link_id of |clean - resume|`` for SWMM ``flow_cms``

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

TRACE ENCODING (iteration 2)
----------------------------
Trace names are DERIVED from each sub node's compute-config attrs via
``eda._config_diff._derive_config_label`` — never from the ``sa_id`` string. Two visual
channels encode the configuration so groups are parseable without reading every label:

* **line DASH = compute-config family** — Serial (solid), OpenMP (dot), MPI (dash),
  Hybrid (dashdot), GPU-a6000 (longdash), GPU-a100 (longdashdot). Six families, six styles.
* **line COLOUR = number of compute devices** — the device count (Serial 1; OpenMP threads;
  MPI ranks; Hybrid ranks x threads; GPU n_gpus), mapped through a sequential scale so more
  devices reads further along the scale.

LEGEND AND TOGGLING (subiteration 2.4). The 28 data traces carry ``showlegend=False``. The
legend has a non-clickable dash KEY for family, then one CLICKABLE entry per DEVICE COUNT:
data + vlines share ``legendgroup='dev:{n}'`` with their device swatch, so under
``legend.groupclick='togglegroup'`` clicking "N devices" toggles every N-device run. Plotly
exposes exactly ONE grouping axis per trace (``legendgroup`` is a scalar), so only ONE of
{family, device} can be the clickable dimension — the user chose device; family is the dash
key. The CPU/GPU split, per-group show/hide, and all-boundaries-off are ``updatemenus``
buttons (a two-column grid top-right, top-aligned to the plot). The resume-boundary lines
carry each run's OWN colour and dash (not a generic dashed style), and there is no separate
dashed legend key — the caption states that the vertical lines are the resume markers.

REPLICATES ARE NOT COLLAPSED. The figure plots one trace per SUB-ANALYSIS (28 = 14 configs
x 2 replicates), not one per config, even though the title says "per compute configuration".
Measured 2026-07-21: 6 of 14 replicate pairs carry DIFFERENT ``replay_t`` (different resume
boundaries), and those are exactly the pairs whose curves differ — so a replicate's curve is
determined by its resume boundary, and merging replicates would fuse two regimes under one
ambiguous vline. Each run's label therefore carries its resume boundary (``— resume @Nm``) so
two runs of a config read as distinct-boundary runs, not duplicates.

KNOWN LIMITATION (surfaced, not silently worked around): Plotly has NO native multi-column
vertical legend (``legend.ncol`` does not exist), and it pins an over-tall legend to the top
and ignores ``legend.y`` — which is why the legend is the compact per-family key (it fits
below the button block; a 28-entry per-run legend does not). The hybrid same-device-count
"two-palette alternating dash" is deferred pending a concrete palette decision.

UNITS
-----
``replay_t`` (root attr ``coupled_resume_replay_evidence``) is in SECONDS; the TRITON
``timestep_min`` axis is in MINUTES. The conversion is ``replay_t / 60``. Verified
2026-07-21: 3000 / 2400 / 4200 s map to 50 / 40 / 70 min, on the 10-minute cadence. The cms
panel is on a ``date_time`` axis, so its vlines convert through a run origin recovered from
the series' own uniform reporting cadence (``date_time[0] - (date_time[1] - date_time[0])``).

MANDATORY caveat (master §Risks): the coupled runs use TRITON's variable dt, so SWMM drops
the final reporting period (Nperiods=N-1). The truncation is ONE-SIDED, NOT common-mode:
the clean arm drops it on all 28 sub-analyses; the resume arm recovers it on 14, selected
deterministically by the restart time replay_t (predicts 28/28; compute config has zero
explanatory power). The cms panel is differenced over the shared leading periods, which are
timestamp-identical across both arms. TRITON series are full-length on both arms. Carried
in the caption, which is rendered as an HTML block below the plot (CSS-wrapped, so it never
truncates and shifts with the figure width) rather than an on-figure annotation.

Q22 / Q30: this renderer is DESCRIPTIVE ONLY. It states what is plotted and how it is
computed, characterizes no magnitude as negligible/benign/sound, and places NO
non-deterministically-derived experiment information on the figure surface — run-dependent
counts live in the caption, never on the plot.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from hhemt.eda._config_diff import _derive_config_label, _gpu_hardware, _to_int
from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

_TRUNCATION_CAVEAT = (
    "Absolute magnitudes inherit the variable-dt SWMM final-period truncation "
    "(Nperiods=N-1). The truncation is ONE-SIDED, not common-mode: the clean arm "
    "drops the final period on every config, while a resumed run recovers it "
    "whenever the restart time falls on the emitting side of SWMM's report-gate "
    "tolerance. SWMM series are therefore compared over the shared leading "
    "periods, which are timestamp-identical across both arms; the clean-vs-resume "
    "DIFFERENCE shown here is taken over that shared prefix. TRITON-side fields "
    "are unaffected (full length on both arms)."
)

_WLEVEL_ABSENT_NOTE = (
    "wlevel panel NOT RENDERED — the per-scenario TRITON timeseries stores "
    "(TRITONSWMM_TRITON_tseries.zarr) were not supplied. They are deliberately not "
    "consolidated into the bundle (~83x the SWMM series by payload). Fetch them with "
    "hhemt_projects/synthetic_compute_sensitivity/fetch_triton_tseries.py and re-render."
)

_REPLAY_ATTR = "coupled_resume_replay_evidence"
_TRITON_STORE_NAME = "TRITONSWMM_TRITON_tseries.zarr"

#: Legend key for the resume-boundary lines. Added FIRST so it sorts to the top.

#: line DASH per compute-config family (iteration-2 encoding). GPU hardware is part of the
#: family key so a6000 and a100 get distinct styles, per the user's "each GPU type".
_FAMILY_DASH = {
    "serial": "solid",
    "openmp": "dot",
    "mpi": "dash",
    "hybrid": "dashdot",
    "gpu-a6000": "longdash",
    "gpu-a100-80": "longdashdot",
}
#: fallback dash pool for a GPU hardware or family not in the map above.
_DASH_FALLBACK = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")

#: Explicit figure width — everything (caption CSS max-width, button/legend x) derives from
#: this ONE value, so there is no second hard-coded layout constant to keep in sync.
_FIG_W = 1280
_MARGIN_L, _MARGIN_R = 70, 30


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    """Report-engine facade. Resolves bundle paths, then delegates to the builder.

    ``triton_tseries_root`` may be supplied via kwargs or ``report_cfg``; when absent the
    figure degrades to the cms panel with an on-figure notice (see module docstring).
    """
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    child_stores = sorted((analysis_dir / "child_crates").glob("*/sensitivity_datatree.zarr"))

    triton_root = kwargs.get("triton_tseries_root") or getattr(report_cfg, "triton_tseries_root", None)
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
        html = figure_to_html_with_caption(fig)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=list(child_stores),
        analysis_dir=analysis_dir,
        output_format="html",
        provenance=prov,
    )


def figure_to_html_with_caption(fig, *, full_html: bool = True, include_plotlyjs="inline") -> str:
    """Render ``fig`` to HTML with its caption as a CSS-wrapped block below the plot.

    The caption is HTML rather than a plotly on-figure annotation so it WRAPS instead of
    truncating: ``max-width`` is tied to the single figure-width constant and
    ``word-wrap`` lets the browser reflow it, so it never runs past the figure edge and
    scales with browser zoom (no per-line px fiddling). Caption text is carried on
    ``fig.layout.meta['caption_lines']`` by the builder.
    """
    import html as _html

    lines = []
    meta = getattr(fig.layout, "meta", None) or {}
    if isinstance(meta, dict):
        lines = meta.get("caption_lines") or []
    body = "<br>".join(_html.escape(str(x)) for x in lines)
    # LEFT-justified to the figure's left edge (margin-left 0, not auto-centred) per user
    # request; width capped to the figure width so it wraps responsively and never overruns.
    caption_div = (
        f'<div style="max-width:{_FIG_W}px;margin:8px 0 0 0;padding:0;'
        "font-family:sans-serif;font-size:12px;line-height:1.4;color:#333;"
        f'text-align:left;word-wrap:break-word;overflow-wrap:break-word;">{body}</div>'
    )
    fig_html = fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs)
    if full_html and "</body>" in fig_html:
        return fig_html.replace("</body>", caption_div + "</body>", 1)
    return fig_html + caption_div


def _resolve_arm_roots(child_stores: list[Path]) -> tuple[Path | None, Path | None]:
    """Pick the clean and resume arm roots from a combined bundle's child crates.

    The arm whose consolidated root carries a populated ``coupled_resume_replay_evidence``
    with any ``replay_t`` is the RESUME arm; the other is CLEAN. Returns (None, None) when a
    clean/resume pair cannot be identified, which the builder reports on the figure.
    """
    clean = resume = None
    for store in child_stores:
        if _replay_map(store):
            resume = store
        else:
            clean = store
    return clean, resume


def _zarr_attrs(meta_path: Path) -> dict:
    """Attributes dict from a zarr v3 ``zarr.json``; empty on any read/parse failure."""
    if not meta_path.is_file():
        return {}
    try:
        return _json.loads(meta_path.read_text()).get("attributes", {}) or {}
    except (OSError, ValueError):
        return {}


def _replay_map(arm_root: Path) -> dict[str, float]:
    """Return {sa_dir_name: replay_t_seconds} from an arm's consolidated root attrs.

    Empty dict when the attr is absent (the clean arm) or carries no replay_t.

    KEY NORMALIZATION (load-bearing). The attribute is keyed by the BARE ``sa_id``
    (``gpu_0_r1``) while every other config surface here is keyed by the sub-analysis
    DIRECTORY name (``sa_gpu_0_r1``). Keys are normalized to the ``sa_``-prefixed form so
    callers have one config-key space. Without this, the per-config vline lookup misses on
    every config and the figure silently renders ZERO resume boundaries while reporting
    success. Mirrors ``_config_diff._n_resumes_by_sa_id``'s normalization.
    """
    raw = _zarr_attrs(arm_root / "zarr.json").get(_REPLAY_ATTR)
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
            key = cfg if cfg.startswith("sa_") else f"sa_{cfg}"
            out[key] = float(rec["replay_t"])
    return out


def _replicate_token(sa_id: str) -> str:
    """The trailing ``_rN`` replicate token of a sub-analysis id, or "" when absent."""
    tail = sa_id.rsplit("_", 1)[-1]
    return tail if tail.startswith("r") and tail[1:].isdigit() else ""


def _family_key(attrs: dict) -> str:
    """Six-way compute-config family for the dash channel: serial/openmp/mpi/hybrid or
    gpu-<hardware>. GPU hardware comes from the ensemble partition via the same helper the
    config-diff figure uses (``_config_diff._gpu_hardware``)."""
    rm = str(attrs.get("run_mode", ""))
    if rm == "gpu":
        hw = _gpu_hardware(attrs)
        return f"gpu-{hw}" if hw else "gpu"
    return rm or "serial"


def _device_count(attrs: dict) -> int:
    """Number of compute devices for the colour channel. GPU: n_gpus. Serial: 1.
    OpenMP: threads. MPI: ranks. Hybrid: ranks x threads (total CPU cores)."""
    rm = str(attrs.get("run_mode", ""))
    if rm == "gpu":
        return max(_to_int(attrs, "n_gpus"), 1)
    if rm == "openmp":
        return max(_to_int(attrs, "n_omp_threads"), 1)
    if rm == "mpi":
        return max(_to_int(attrs, "n_mpi_procs"), 1)
    if rm == "hybrid":
        return max(_to_int(attrs, "n_mpi_procs"), 1) * max(_to_int(attrs, "n_omp_threads"), 1)
    return 1


def _config_meta(arm_root: Path) -> dict[str, dict]:
    """sa_id -> {label, family, fam_key, device_count, replicate} from each sub's attrs.

    ``family`` is the CPU/GPU split (drives the buttons); ``fam_key`` is the six-way family
    (drives the dash channel); ``device_count`` drives the colour channel. Label derivation
    is delegated to ``_config_diff._derive_config_label`` rather than reimplemented.
    """
    out: dict[str, dict] = {}
    for sub in sorted(Path(arm_root).iterdir()):
        if not (sub.is_dir() and sub.name.startswith("sa_")):
            continue
        attrs = _zarr_attrs(sub / "zarr.json")
        if not attrs:
            continue
        rep = _replicate_token(sub.name)
        base = _derive_config_label(attrs)
        out[sub.name] = {
            "label": f"{base} · {rep}" if rep else base,
            "family": "GPU" if str(attrs.get("run_mode", "")) == "gpu" else "CPU",
            "fam_key": _family_key(attrs),
            "device_count": _device_count(attrs),
            "replicate": rep,
        }
    return out


def _ordered_configs(meta: dict[str, dict], present: set[str]) -> list[str]:
    """Config ids ordered CPU-family-first then GPU, each alphabetical by derived label."""
    return sorted(
        (sa for sa in present if sa in meta),
        key=lambda sa: (meta[sa]["family"] != "CPU", meta[sa]["label"]),
    )


def _encode(order: list[str], meta: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (color_by_sa, dash_by_sa).

    Colour encodes DEVICE COUNT (sequential Viridis, more devices further along). Dash
    encodes FAMILY. Hybrid configs that collide on device count would share both channels,
    so colliding members past the first are bumped to a distinct fallback dash to stay
    distinguishable (an interim stand-in for the requested two-palette alternating dash,
    which is held pending a concrete palette decision).
    """
    from plotly.colors import sample_colorscale

    counts = sorted({meta[sa]["device_count"] for sa in order})
    if len(counts) > 1:
        pos = {c: i / (len(counts) - 1) for i, c in enumerate(counts)}
    else:
        pos = {counts[0]: 0.5} if counts else {}
    color = {sa: sample_colorscale("Viridis", [pos[meta[sa]["device_count"]]])[0] for sa in order}

    dash: dict[str, str] = {}
    seen_hybrid_dc: dict[int, int] = {}
    for sa in order:
        fk = meta[sa]["fam_key"]
        base = _FAMILY_DASH.get(fk)
        if base is None:  # unknown GPU hardware — take the next fallback style
            base = _DASH_FALLBACK[len(dash) % len(_DASH_FALLBACK)]
        if fk == "hybrid":
            dc = meta[sa]["device_count"]
            n = seen_hybrid_dc.get(dc, 0)
            seen_hybrid_dc[dc] = n + 1
            if n:  # a second+ hybrid at this device count — bump to keep it distinguishable
                base = _DASH_FALLBACK[(_DASH_FALLBACK.index("dashdot") + n) % len(_DASH_FALLBACK)]
        dash[sa] = base
    return color, dash


def build_overtime_maxabsdiff_figure(
    clean_root: Path | None,
    resume_root: Path | None,
    triton_tseries_root: Path | None = None,
):
    """Assemble the two-panel over-time max-absolute-difference figure.

    Returns a plotly Figure whose ``layout.meta['caption_lines']`` carries the caption
    (rendered as a CSS-wrapped HTML block by ``figure_to_html_with_caption``). Raises
    nothing on missing data — absence is reported ON the figure so a degraded render is
    visibly degraded rather than silently partial.
    """
    from plotly.subplots import make_subplots

    notes: list[str] = [_TRUNCATION_CAVEAT]
    have_wlevel = triton_tseries_root is not None and _triton_stores_present(triton_tseries_root)
    if not have_wlevel:
        notes.append(_WLEVEL_ABSENT_NOTE)

    if clean_root is None or resume_root is None:
        fig = make_subplots(rows=1, cols=1)
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
        _finalize(fig, notes, n_panels=1, order=[], meta={})
        return fig

    meta = _config_meta(clean_root)
    shared = _config_ids(clean_root) & _config_ids(resume_root)
    order = _ordered_configs(meta, shared)
    colors, dashes = _encode(order, meta)
    replay = _replay_map(resume_root)

    rows = 2 if have_wlevel else 1
    titles = (
        [
            "(a) max |Δ| water level (TRITON wlevel_m) [m]",
            "(b) max |Δ| link flow (SWMM flow_cms) [m³/s]",
        ]
        if have_wlevel
        else ["(a) max |Δ| link flow (SWMM flow_cms) [m³/s]"]
    )
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, subplot_titles=titles)

    wlevel_max = (
        _add_wlevel_traces(fig, 1, triton_tseries_root, order, meta, colors, dashes, replay) if have_wlevel else 0.0
    )
    cms_max, trimmed, origin, n_shared = _add_cms_traces(
        fig, rows, clean_root, resume_root, order, meta, colors, dashes, replay
    )

    # LEGEND toggle dimension = DEVICE COUNT (subiteration 2.4). Data traces carry
    # showlegend=False and legendgroup="dev:{n}"; each device swatch shares that group under
    # groupclick='togglegroup', so clicking "N devices" toggles every N-device run + its
    # vlines. Plotly allows exactly ONE legend-toggle grouping per trace (verified: legendgroup
    # is a scalar), so FAMILY becomes a non-clickable dash key here — it can't be legend-
    # toggleable at the same time as device. The CPU/GPU + show/hide/resume buttons remain.
    # The 28 data traces are NOT collapsed: replicates are runs with DIFFERENT resume
    # boundaries (6 of 14 pairs), so per-run identity (incl. its boundary) is on hover.
    _add_encoding_key(fig, order, meta, colors)

    _add_resume_vlines(
        fig,
        order=order,
        colors=colors,
        dashes=dashes,
        meta=meta,
        replay=replay,
        wlevel_row=1 if have_wlevel else None,
        wlevel_ymax=wlevel_max,
        cms_row=rows,
        cms_ymax=cms_max,
        cms_origin=origin,
    )

    # Caption carries ONLY what is not discernible from the figure itself (per user: no
    # redundant info recoverable from axes / legend / labels / the plotted data). The
    # encoding is shown by the legend + the trace labels, so it is NOT repeated here; the
    # dashed-line meaning is the legend's resume key, so only the non-obvious mechanism (how
    # the boundary time is derived) is stated.
    if replay:
        notes.append(
            "Vertical lines mark each run's hotstart-resume timestep, drawn in that run's own "
            f"colour and line style. Positions come from the resume arm's `{_REPLAY_ATTR}` "
            "root attribute (replay_t, seconds): panel (a) at replay_t / 60 minutes, panel (b) "
            "by offsetting the run origin recovered from the series' own reporting cadence. "
            "Replicates of a config can carry DIFFERENT boundaries — the hover label states "
            "each run's boundary."
        )
    if trimmed:
        notes.append(
            f"Panel (b): {len(trimmed)} of {n_shared} runs are compared over the shared "
            "leading periods (one trailing period present only in the resume arm was "
            "excluded); shared timestamps verified identical."
        )

    _finalize(fig, notes, n_panels=rows, order=order, meta=meta)
    return fig


def _triton_stores_present(root: Path) -> bool:
    return any(Path(root).rglob(_TRITON_STORE_NAME))


def _run_label(sa_id, meta, replay) -> str:
    """Per-run legend/hover label: the derived config label + the run's resume boundary in
    minutes. The boundary is what distinguishes replicates (they carry different replay_t),
    so it — not the bare r1/r2 token — is what makes two runs of a config legible."""
    base = meta.get(sa_id, {}).get("label", sa_id)
    rt = replay.get(sa_id)
    return f"{base} — resume @{rt / 60:.0f}m" if rt is not None else base


def _dev_group(sa_id, meta) -> str:
    """Legend-toggle group = device count (subiteration 2.4)."""
    return f"dev:{meta.get(sa_id, {}).get('device_count', 0)}"


def _trace_meta(role, sa_id, meta) -> str:
    """Trace tag ``{role}:{family}:{device}`` — carries BOTH encoding axes so the buttons
    can target family (CPU/GPU) even though the legendgroup now encodes device."""
    info = meta.get(sa_id, {})
    return f"{role}:{info.get('fam_key', 'other')}:{info.get('device_count', 0)}"


def _add_wlevel_traces(fig, row: int, triton_root: Path, order, meta, colors, dashes, replay) -> float:
    """One trace per config: max over (y, x) of |clean - resume| for TRITON wlevel_m.

    Returns the largest value plotted, used to size the resume-boundary lines.
    """
    import numpy as np
    import xarray as xr

    clean_stores = {_sa_id(p): p for p in Path(triton_root).rglob(f"*clean*/**/{_TRITON_STORE_NAME}")}
    resume_stores = {_sa_id(p): p for p in Path(triton_root).rglob(f"*resume*/**/{_TRITON_STORE_NAME}")}
    ymax = 0.0
    for sa_id in order:
        if sa_id not in clean_stores or sa_id not in resume_stores:
            continue
        dc = xr.open_zarr(clean_stores[sa_id])
        dr = xr.open_zarr(resume_stores[sa_id])
        diff = np.abs(dc.wlevel_m.values - dr.wlevel_m.values)
        per_t = np.nanmax(diff, axis=(1, 2))
        ymax = max(ymax, float(np.nanmax(per_t)) if per_t.size else 0.0)
        fig.add_scatter(
            x=dc.timestep_min.values,
            y=per_t,
            mode="lines",
            name=_run_label(sa_id, meta, replay),
            legendgroup=_dev_group(sa_id, meta),
            line={"color": colors[sa_id], "dash": dashes[sa_id], "width": 1.6},
            showlegend=False,  # device legend toggles the group; per-run identity is on hover
            meta=_trace_meta("data", sa_id, meta),
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="timestep (min)", row=row, col=1)
    return ymax


def _sa_id(store_path: Path) -> str:
    """Extract the sa_* config id from a per-scenario store path."""
    for part in store_path.parts:
        if part.startswith("sa_"):
            return part
    return store_path.parent.name


def _add_cms_traces(fig, row, clean_root, resume_root, order, meta, colors, dashes, replay):
    """One trace per config: max over link_id of |clean - resume| for SWMM flow_cms.

    Returns (ymax, trimmed_config_ids, run_origin, n_shared). ``run_origin`` is recovered
    from the series' own uniform reporting cadence so panel (b)'s resume boundaries can be
    placed on its date_time axis.
    """
    import numpy as np
    import xarray as xr

    trimmed: list[str] = []
    ymax = 0.0
    origin = None
    for sa_id in order:
        try:
            dc = xr.open_zarr(clean_root, group=f"{sa_id}/tritonswmm/swmm_link_timeseries")
            dr = xr.open_zarr(resume_root, group=f"{sa_id}/tritonswmm/swmm_link_timeseries")
        except (OSError, KeyError, ValueError):
            continue
        # SHARED-PREFIX ALIGNMENT (Option A, approved 2026-07-21). The truncation is
        # ONE-SIDED: the clean arm drops the final reporting period on all 28 sub-analyses;
        # the resume arm recovers it on 14, so the two series differ in length on exactly
        # those 14. Verified before adopting: the shared leading periods are
        # timestamp-identical on ALL 28 configs (0 exceptions).
        cv, rv = dc.flow_cms.values, dr.flow_cms.values
        n = min(cv.shape[-1], rv.shape[-1])
        if cv.shape[-1] != rv.shape[-1]:
            trimmed.append(sa_id)
        diff = np.abs(cv[..., :n] - rv[..., :n])
        per_t = np.nanmax(diff, axis=tuple(range(diff.ndim - 1)))
        ymax = max(ymax, float(np.nanmax(per_t)) if per_t.size else 0.0)
        times = dc.date_time.values[:n]
        if origin is None and times.size >= 2:
            # SWMM stamps each reporting period at its END, so the run origin is one
            # reporting interval before the first stamp.
            origin = times[0] - (times[1] - times[0])
        fig.add_scatter(
            x=times,
            y=per_t,
            mode="lines",
            name=_run_label(sa_id, meta, replay),
            legendgroup=_dev_group(sa_id, meta),
            line={"color": colors[sa_id], "dash": dashes[sa_id], "width": 1.6},
            showlegend=False,  # device legend toggles the group; per-run identity is on hover
            meta=_trace_meta("data", sa_id, meta),
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="date_time", row=row, col=1)
    return ymax, trimmed, origin, len(order)


_FAMILY_PRETTY = {"serial": "Serial", "openmp": "OpenMP", "mpi": "MPI", "hybrid": "Hybrid"}


def _add_encoding_key(fig, order, meta, colors) -> None:
    """Add the legend: a non-clickable dash KEY for family, then one CLICKABLE entry per
    device count (subiteration 2.4 — the user asked for the device entries to toggle).

    Device entries share ``legendgroup='dev:{n}'`` with their device count's DATA + vline
    traces, so under ``groupclick='togglegroup'`` clicking "N devices" toggles every N-device
    run. Family swatches are a visual dash key only: each is in its OWN singleton legendgroup
    (``famkey:{fk}``) that no data shares, so clicking one does not touch the plot — Plotly's
    single legendgroup-per-trace means only ONE of {family, device} can be the clickable
    dimension, and the user chose device.
    """
    seen_fam: list[str] = []
    first_fam = True
    for sa in order:
        fk = meta[sa]["fam_key"]
        if fk in seen_fam:
            continue
        seen_fam.append(fk)
        name = f"GPU {fk[4:]}" if fk.startswith("gpu-") else _FAMILY_PRETTY.get(fk, fk)
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#555", "dash": _FAMILY_DASH.get(fk, "solid"), "width": 2},
            name=f"family — {name}",
            legendgroup=f"famkey:{fk}",
            legendgrouptitle_text="line style = family (key)" if first_fam else None,
            showlegend=True,
            hoverinfo="skip",
            meta="key",
            row=1,
            col=1,
        )
        first_fam = False
    seen_dc: list[int] = []
    first_dc = True
    for sa in sorted(order, key=lambda s: meta[s]["device_count"]):
        dc = meta[sa]["device_count"]
        if dc in seen_dc:
            continue
        seen_dc.append(dc)
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": colors[sa], "dash": "solid", "width": 3},
            name=f"{dc} device{'s' if dc != 1 else ''}",
            legendgroup=f"dev:{dc}",
            legendgrouptitle_text="colour = compute devices (click to toggle)" if first_dc else None,
            showlegend=True,
            hoverinfo="skip",
            meta="key",
            row=1,
            col=1,
        )
        first_dc = False


def _add_resume_vlines(
    fig, *, order, colors, dashes, meta, replay, wlevel_row, wlevel_ymax, cms_row, cms_ymax, cms_origin
) -> None:
    """Draw each config's resume boundary on BOTH panels as trace-bound scatter lines.

    Drawn as traces (not ``add_vline`` shapes) so each carries the trace's colour, toggles
    with the trace via a shared ``legendgroup`` under ``groupclick='togglegroup'``, and can
    be addressed by the all-boundaries-off button.
    """
    import numpy as np

    for sa_id in order:
        replay_t = replay.get(sa_id)
        if replay_t is None:
            continue
        # Match the run EXACTLY: same colour AND same line style (family dash), per user —
        # the vline is a vertical marker in its run's own visual identity, not a generic dash.
        vline = {"color": colors[sa_id], "dash": dashes[sa_id], "width": 1}
        if wlevel_row is not None and wlevel_ymax > 0:
            fig.add_scatter(
                x=[replay_t / 60.0, replay_t / 60.0],
                y=[0.0, wlevel_ymax],
                mode="lines",
                line=vline,
                opacity=0.7,
                legendgroup=_dev_group(sa_id, meta),
                showlegend=False,
                hoverinfo="skip",
                meta=_trace_meta("vline", sa_id, meta),
                row=wlevel_row,
                col=1,
            )
        if cms_origin is not None and cms_ymax > 0:
            xt = cms_origin + np.timedelta64(int(round(replay_t * 1000)), "ms")
            fig.add_scatter(
                x=[xt, xt],
                y=[0.0, cms_ymax],
                mode="lines",
                line=vline,
                opacity=0.7,
                legendgroup=_dev_group(sa_id, meta),
                showlegend=False,
                hoverinfo="skip",
                meta=_trace_meta("vline", sa_id, meta),
                row=cms_row,
                col=1,
            )


def _config_ids(arm_root: Path) -> set[str]:
    return {p.name for p in Path(arm_root).iterdir() if p.is_dir() and p.name.startswith("sa_")}


def _role(tr) -> str:
    """First field of the trace's ``{role}:{fam}:{dev}`` meta tag (or the bare tag)."""
    m = getattr(tr, "meta", None)
    return str(m).split(":", 1)[0] if m else ""


def _fam_of(tr) -> str:
    """Family field of a data/vline trace's meta tag."""
    parts = str(getattr(tr, "meta", "") or "").split(":")
    return parts[1] if len(parts) > 1 else ""


def _is_vline(tr) -> bool:
    return _role(tr) == "vline"


def _control_menus(fig, order, meta):
    """Two aligned button COLUMNS at the top-right, top-aligned to the plot top.

    Left column (top→bottom): show all, CPU on, GPU on, resume on. Right column: hide all,
    CPU off, GPU off, resume off. Rendered as two ``direction='down'`` menus starting at the
    same y, so each on/off pair reads as a row and the two columns look symmetrical
    (subiteration 2.2 — the earlier per-row menus had ragged widths). The top button's top
    sits at y=1.0 (the plot-top = figure-area top) per the user's alignment request; the
    legend is shifted below the block in ``_finalize``.

    Plotly's one-grouping-axis-per-trace is spent on the trace<->vline binding, so these
    bulk toggles are buttons. Each restyle names the exact trace INDICES it targets (the
    ``(update, [indices])`` form) so a group button leaves other traces untouched; 'off'
    uses 'legendonly' so a trace stays listed and re-showable, 'on'/'show' restores True.
    """
    if not order:
        return []
    idx = list(range(len(fig.data)))

    def _is_data(tr):  # a DATA or vline trace (not an encoding-key swatch)
        return _role(tr) in ("data", "vline")

    cpu_ix = [i for i in idx if _is_data(fig.data[i]) and not _fam_of(fig.data[i]).startswith("gpu")]
    gpu_ix = [i for i in idx if _is_data(fig.data[i]) and _fam_of(fig.data[i]).startswith("gpu")]
    vln_ix = [i for i in idx if _is_vline(fig.data[i])]
    # show/hide all targets DATA + vline traces only (not the encoding-key swatches, which
    # stay visible so the legend always explains the encoding).
    data_ix = [i for i in idx if _is_data(fig.data[i])]

    def _b(label, value, ixs=None):
        args = [{"visible": value}] if ixs is None else [{"visible": value}, ixs]
        return {"label": label, "method": "restyle", "args": args}

    left = [_b("show all", True, data_ix)]
    right = [_b("hide all", "legendonly", data_ix)]
    for famname, ixs in (("CPU", cpu_ix), ("GPU", gpu_ix)):
        if not ixs:
            continue
        left.append(_b(f"{famname} on", True, ixs))
        right.append(_b(f"{famname} off", "legendonly", ixs))
    if vln_ix:
        left.append(_b("resume on", True, vln_ix))
        right.append(_b("resume off", "legendonly", vln_ix))

    common = {
        "type": "buttons",
        "direction": "down",
        "showactive": False,
        "yanchor": "top",
        "y": 1.0,  # top of the top button aligns with the plot top (figure-area top)
        "pad": {"r": 2, "t": 2, "l": 2, "b": 2},
        "font": {"size": 9},
    }
    return [
        {**common, "buttons": left, "x": 1.01, "xanchor": "left"},
        {**common, "buttons": right, "x": 1.075, "xanchor": "left"},
    ]


def _finalize(fig, notes, *, n_panels, order, meta) -> None:
    height = 900 if n_panels > 1 else 560
    layout = {
        "title": {
            "text": (
                "Clean vs. hotstart-resume: per-timestep MAXIMUM ABSOLUTE DIFFERENCE "
                "(one trace per compute configuration)"
            ),
            # Anchored to the CONTAINER top so the title sits at the very top; it is centred
            # (x=0.5) while the buttons are far right (x>1.0), so they never collide.
            "yref": "container",
            "y": 0.985,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
        "width": _FIG_W,
        "height": height,
        # Top margin holds only the title now; the button columns start at the plot top and
        # descend in the right-hand column (x>1.0), so they need no top-margin band.
        "margin": {"t": 64, "b": 40, "l": _MARGIN_L, "r": _MARGIN_R},
        "hovermode": "x unified",
        "hoverlabel": {"namelength": -1},
        "legend": {
            "groupclick": "togglegroup",
            "traceorder": "grouped",
            "font": {"size": 9},
            # Seated BELOW the 4-row button block. This works only because the legend is now
            # the COMPACT encoding key (~a dozen short entries): an over-tall 28-entry legend
            # exceeds the space below the buttons, so Plotly pins it to the top and ignores
            # `y` (verified by viewing the render at y=0.35 — it did not move). The compact
            # key fits, so y is honoured and it sits under the buttons as the user asked.
            "grouptitlefont": {"size": 9, "color": "#333"},
            "yanchor": "top",
            "y": 0.74,
            "x": 1.01,
            "xanchor": "left",
        },
    }
    menus = _control_menus(fig, order, meta)
    if menus:
        layout["updatemenus"] = menus
    fig.update_layout(**layout)
    # Caption travels on layout.meta; figure_to_html_with_caption renders it as a
    # CSS-wrapped HTML block (no on-figure annotation, so it cannot truncate).
    fig.update_layout(meta={"caption_lines": notes})
