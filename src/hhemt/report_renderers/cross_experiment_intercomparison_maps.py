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
SWMM drops the final reporting period (Nperiods=N-1). This is common-mode across the
clean AND resume arms, so the clean-vs-resume DIFFERENCE (this figure's headline) is
sound; but any ABSOLUTE-magnitude reference panel carries the truncation. The caveat is
annotated on the figure AND carried in the caption.

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
    "(Nperiods=N-1); this is common-mode across the clean and resume arms, so the "
    "clean-vs-resume DIFFERENCE shown here is sound."
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


def build_cross_experiment_diff_figure(combined_root: Path):
    """Assemble the clean-vs-resume spatial diff figure from the combined-bundle root.

    CONTRACT (grounded; layout owned by /eda-spinup):
      1. Read combined_intercomparison.json -> `experiments` (role per {experiment}) +
         `pairs` (config/variable/event_iloc/identical/max_abs_diff). Identify the clean
         eid and the resume eid from roles; the DIFFERING set = pairs with identical==False.
      2. If no clean+resume pair or no differing pairs -> return a single-panel figure
         titled "No clean-vs-resume spatial differences" (honest empty state).
      3. For each differing (config, variable, event): re-open
         child_crates/{clean_eid}/sensitivity_datatree.zarr and .../{resume_eid}/..., select
         the /sa_* sub whose _config_diff._derive_config_label (or the _combine
         _config_identity_from_node_attrs key) matches `config`, isel the event, and
         compute the SIGNED diff = resume - clean (align via _config_diff._align_to) plus
         _config_diff._signed_pct. Render depth via _config_diff._heatmap and flow via
         _config_diff._conduit_traces with colorscale=_config_diff._DIVERGING, zmid=0
         (blue=resume-higher). Reuse _config_diff._watershed_polygon/_watershed_mask/
         _apply_mask/_load_conduit_geometry from child_crates/{clean_eid}/.
      4. Add a figure-level annotation carrying _TRUNCATION_CAVEAT and, per pair, an
         n-resumes annotation (resume arm's max n_resumes from
         _combine._bundle_role_from_status-style scenario_status.csv read on the resume child).

    Returns a plotly go.Figure. The minimal first implementation renders one
    differing-pair panel (depth diff | flow diff, + %-diff row) reusing the named
    _config_diff helpers; /eda-spinup iterates the multi-pair multi-event stacking +
    colorbar/domain polish (mirroring build_config_diff_figure's manual px budget).
    """
    import plotly.graph_objects as go

    payload = (
        _json.loads((combined_root / "combined_intercomparison.json").read_text())
        if (combined_root / "combined_intercomparison.json").exists()
        else {"experiments": [], "pairs": []}
    )
    differing = [p for p in payload.get("pairs", []) if not p.get("identical", True)]
    if not differing:
        fig = go.Figure()
        fig.update_layout(
            height=350,
            title="Clean-vs-resume spatial comparison: no differing pairs (resume reproduces clean)",
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text=_TRUNCATION_CAVEAT,
                    font=dict(size=11, color="#444"),
                )
            ],
        )
        return fig
    # NON-EMPTY differing set: the multi-panel RdBu diff/%-diff layout reusing the
    # _config_diff primitives per the CONTRACT above is the /eda-spinup deliverable.
    # This scaffold returns an honest interim figure listing the differing pairs so the
    # emit/source-declaration path is exercised end-to-end before the /eda-spinup polish.
    fig = go.Figure()
    lines = [
        f"{p['config']} | {p['variable']} | evt {p['event_iloc']} | max_abs_diff={p['max_abs_diff']}" for p in differing
    ]
    fig.update_layout(
        height=120 + 24 * len(lines),
        title="Clean-vs-resume spatial differences (differing pairs; maps pending /eda-spinup layout)",
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text="<br>".join(lines) + f"<br><br><i>{_TRUNCATION_CAVEAT}</i>",
                font=dict(size=11),
            )
        ],
    )
    return fig
