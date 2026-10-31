"""Cross-solver comparison renderer: one grouped bar panel per model arm.

WHY THIS FILE EXISTS. A combined report assembled from bundles that ran DIFFERENT
solver commits answers a question no per-experiment report can: did the solver change
move the measurement? Each experiment's own report is true only at its own pin, so the
comparison can exist only where the bundles are merged — here, at combine time.

WHAT IT READS. One persisted read-model at the analysis-dir root,
``cross_solver_comparison.json``, written beside the merged tree. It is a flat cell
list keyed on ``(solver, model_arm, config)`` with one numeric ``value`` per cell and a
per-cell ``status`` of ``ok`` or ``not_run``, plus the producer's per-solver ``solver_info``
(commit date and legend label, see `_solver_info`). Nothing else is read, so re-rendering
this figure from a shipped combined bundle needs no re-merge.

WHAT IT DRAWS. One panel per model arm. Within a panel the x axis is the compute config,
a series (and a colour) is one solver, and the bar height is the read-model's metric. A
solver that did not run a config its peers ran is a ``not_run`` cell: NO bar is drawn at
that position and the top of the panel carries an ``x`` marker instead. That distinction
is the figure's reason to exist — an absent bar and a measured zero are the same shape on
a bar chart, and confusing them is how a wrong-pin arm at a plausible path passes every
structural guard.

INERTNESS IS THE CALLER'S, NOT THIS MODULE'S. Whether this figure renders at all is
decided by the ``combined`` reporting set's ``has_multiple_solvers`` predicate. A
single-solver combine — every clean-vs-resume combine — never dispatches this renderer.
When the read-model is nonetheless absent or empty this module renders an honest
placeholder rather than a vacuous single-solver figure.

Deliberately NOT a table: the read-model's third key becomes an x-axis position here,
which is what makes the across-solver SHAPE visible. The Tabulator surfaces in this
package serve per-row inspection; this figure serves the comparison.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, TypedDict

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from hhemt.figure_caption import add_figure_caption
from hhemt.report_renderers import _plotly_theme  # noqa: F401 -- registers the `triton_journal` template
from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._model_arms import MODEL_DISPLAY_NAMES, PAGE_MODEL_ORDER
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

__all__ = ["render"]

#: Read-model filename at the analysis-dir root, beside ``combined_intercomparison.json``.
#: The combine step writes it; this module is its only reader.
READ_MODEL_FILENAME = "cross_solver_comparison.json"

#: The one per-cell status token this module READS. ``not_run`` must never be mistaken for
#: a measurement, so it is a named constant rather than a literal at each compare. The
#: complementary ``ok`` token is deliberately NOT named here: the reader does not branch on
#: it (a cell with no numeric value is ``not_run`` whatever its status says), and a constant
#: this module never reads would suggest a branch that does not exist.
_STATUS_NOT_RUN = "not_run"

#: Fallback y-axis title when the read-model names neither a metric label nor a metric key.
#: Deliberately says "metric" rather than defaulting to a unit: a figure that invents the
#: unit of a quantity it did not read mislabels every bar it draws.
_UNLABELLED_METRIC = "metric"

#: Fixed figure geometry in pixels. The plot area stays 368 px tall (560 - 96 - 96, today's value) so
#: adding a caption grows the figure downward instead of squeezing the bars.
_MARGIN_L_PX = 84
_MARGIN_R_PX = 150
_MARGIN_T_PX = 96
_MARGIN_B_PX = 96
_PLOT_H_PX = 368
#: Height below the plot taken by the rotated x tick labels and the axis title; the caption starts under it.
_X_AXIS_BAND_PX = 60


class _Cell(TypedDict):
    """One normalised ``(solver, model_arm, config)`` observation."""

    solver: str
    model_arm: str
    config: str
    value: float | None
    not_run: bool


def render(analysis: Any, report_cfg: Any, output_path: Path, **kwargs: Any) -> Path:
    """Render the cross-solver comparison figure to ``output_path``.

    ``analysis`` is duck-typed on ``analysis_paths.analysis_dir`` alone, because the two
    call sites are deliberately minimal contexts rather than full analyses: the combine
    step hands over a context exposing only that path, and the child-report re-render
    path does the same. ``report_cfg`` supplies the Plotly bundling mode and the optional
    preview-raster toggle; an unrecognised ``**kwargs`` entry is ignored so a dispatch
    that grows an argument does not break rendering.

    Returns ``output_path`` unchanged, for chaining.
    """
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    source = analysis_dir / READ_MODEL_FILENAME
    payload, problem = _read_model(analysis_dir)
    cells = _normalize_cells(payload)

    interactive = getattr(report_cfg, "interactive", None)
    js_mode = str(getattr(interactive, "plotly_js_mode", "cdn"))
    preview_enabled = bool(getattr(interactive, "html_preview_rasterization", False))

    prov = ProvenanceLog()
    preview_figure: Any | None = None
    if cells:
        fig, facts = _build_figure(cells, payload, prov)
        html = _figure_to_html(fig, js_mode=js_mode)
        preview_figure = fig if preview_enabled else None
    else:
        # The placeholder page draws no traces, so it owns no per-artist provenance;
        # this one record is what carries the declared source into the manifest, and it
        # is declared even though the file is absent (ADR-6 D3: a presence-gated
        # declaration would make this figure's declared set depend on render timing).
        with prov.artist(
            axes_id="placeholder",
            kind="text",
            note="cross-solver comparison placeholder (no read-model cells)",
        ) as artist:
            artist.add_channel("data", ProvenanceRef(source_path=READ_MODEL_FILENAME))
        html = _placeholder_html(problem)
        facts = {
            "solver_count": 0,
            "solvers": [],
            "undated_solvers": [],
            "model_arm_count": 0,
            "model_arms": [],
            "config_count": 0,
            "cell_count": 0,
            "not_run_cell_count": 0,
        }

    return emit_plot_with_sources(
        html,
        output_path,
        [source],
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "cross_solver_comparison",
            **facts,
            # Recorded on the manifest in BOTH branches (None when the read-model loaded),
            # so a reader of a rendered placeholder bundle can tell an absent read-model
            # from an unparseable one without re-running the combine.
            "read_model_problem": problem,
        },
        provenance=prov,
        preview_figure=preview_figure,
    )


def _read_model(analysis_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Load the read-model and say WHAT went wrong when nothing usable came back.

    Returns ``(payload, problem)``, where ``problem`` is ``None`` on success and a
    human-readable sentence otherwise. Neither state raises: this render runs at the END
    of a long combine, so a JSON typo that raised would cost a whole re-render to
    discover a syntax error the page could have named on the spot.

    ``problem`` is a SEPARATE return rather than a sentinel key inside the payload, and
    the distinction it exists for is the one that used to be invisible: an absent file,
    an unparseable file and a parseable file with no cells all degraded to ``{}``, so a
    reader saw one message — "no read-model was found" — for three different causes, one
    of which is a defect in the file rather than a missing one.

    The message is a bare CAUSE with no filename, because the two consumers want
    different amounts of context: `_placeholder_html` renders the sentence for a human,
    and the manifest records the same token for a machine. Neither re-derives the other.
    """
    source = analysis_dir / READ_MODEL_FILENAME
    if not source.exists():
        return {}, "absent"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, f"unparseable JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, f"not a JSON object (parsed as {type(payload).__name__})"
    return payload, None


def _as_float(value: Any) -> float | None:
    """Coerce a read-model value to a finite float, or ``None`` when it is not one.

    ``bool`` is rejected before the conversion: ``float(True)`` is ``1.0``, which would
    turn a stray flag in the payload into a plausible-looking measurement at value one.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _normalize_cells(payload: dict[str, Any]) -> list[_Cell]:
    """Keep the cells that can be placed on the figure's three axes.

    A cell missing any of ``solver``, ``model_arm`` or ``config`` has no position on the
    figure, so it is dropped rather than guessed onto an axis. Everything else is kept,
    including cells with no value: those become ``not_run``, because a cell that names a
    solver and a config but carries no measurement is exactly the "did not run" case this
    figure exists to make visible.

    A cell whose ``status`` SAYS it ran but which carries no numeric value is also
    ``not_run``, and the ``status`` is deliberately not consulted once the value is
    missing. The alternative — trusting the claim and leaving the cell unmarked — draws
    nothing at that position AND omits it from the not_run count, so a read-model that
    says "ok" and ships a null becomes indistinguishable from a correct one with a solver
    that simply never ran. An unmarked gap is the one thing this figure cannot report.
    """
    out: list[_Cell] = []
    for raw in payload.get("cells") or []:
        if not isinstance(raw, dict):
            continue
        solver = str(raw.get("solver") or "").strip()
        model_arm = str(raw.get("model_arm") or "").strip()
        config = str(raw.get("config") or "").strip()
        if not solver or not model_arm or not config:
            continue
        value = _as_float(raw.get("value"))
        status = str(raw.get("status") or "").strip().lower()
        not_run = status == _STATUS_NOT_RUN or value is None
        out.append(
            {
                "solver": solver,
                "model_arm": model_arm,
                "config": config,
                "value": None if not_run else value,
                "not_run": not_run,
            }
        )
    return out


def _config_label(config: str) -> str:
    """Short human label for a packed compute-config identity key.

    The packing is ``field=value`` joined on ``|``. This parses it through
    ``cross_experiment_intercomparison._parse_config`` rather than re-deriving the split:
    the intercomparison table reads the SAME key format, and a second parser is a second
    source that would drift the moment a field is renamed. Fields absent from a given key
    are dropped from the label instead of printed as ``None``, so a read-model written
    before a field existed still labels its cells.
    """
    from hhemt.report_renderers.cross_experiment_intercomparison import _parse_config

    fields = _parse_config(config)
    parts: list[str] = []
    run_mode = fields.get("run_mode")
    if run_mode:
        parts.append(str(run_mode))
    n_mpi, n_omp = fields.get("n_mpi"), fields.get("n_omp")
    if n_mpi is not None and n_omp is not None:
        parts.append(f"{n_mpi}x{n_omp}")
    elif n_mpi is not None:
        parts.append(str(n_mpi))
    n_gpus = fields.get("n_gpus")
    if n_gpus:
        parts.append(f"{n_gpus} GPU")
    partition = fields.get("partition")
    if partition:
        parts.append(str(partition))
    return " · ".join(parts) or config


def _ordered_arms(cells: list[_Cell], payload: dict[str, Any]) -> list[str]:
    """Model arms in report display order, then any arm the display table does not name.

    Order comes from ``_model_arms.PAGE_MODEL_ORDER`` — the DISPLAY tuple, not the legacy
    tie-break one — so a panel header here and a composed per-scenario page header cannot
    disagree about which arm comes first or what it is called. An arm the read-model names
    but the display table does not is APPENDED and never dropped: ``_model_arms`` records
    that a fallback chain can drop an arm silently, and an arm dropped here would read as a
    solver that never ran.
    """
    declared: list[str] = []
    for raw in payload.get("model_arms") or []:
        token = str(raw or "").strip()
        if token and token not in declared:
            declared.append(token)
    for cell in cells:
        if cell["model_arm"] not in declared:
            declared.append(cell["model_arm"])
    known = [arm for arm in PAGE_MODEL_ORDER if arm in declared]
    unknown = sorted(arm for arm in declared if arm not in PAGE_MODEL_ORDER)
    return known + unknown


def _solver_info(payload: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """The producer's per-solver ``{pin: {"date", "label"}}``, reduced to what the figure uses.

    The combine step measures each solver's commit date from the clone that built it and
    writes it beside the solver identity, so this figure never recovers a date by parsing a
    label. A parse is wrong in exactly the case that matters: the identity is the pinned git
    ref, which carries no date, so every solver would tie on the date and fall through to hash
    order.

    A ``date`` counts only when it is exactly eight digits (``YYYYMMDD``); anything else is
    treated as ABSENT rather than placed, because a dated series' position is an age claim
    and a malformed date cannot back one. A read-model without ``solver_info`` (one written
    before the producer supplied it, or a hand-written payload) therefore yields an empty
    mapping, and every solver is undated. ``label`` is carried only when it is a non-empty
    string. This function reads the payload and nothing else.
    """
    raw_info = payload.get("solver_info")
    out: dict[str, dict[str, str | None]] = {}
    if not isinstance(raw_info, dict):
        return out
    for raw_pin, raw_entry in raw_info.items():
        pin = str(raw_pin or "").strip()
        if not pin or not isinstance(raw_entry, dict):
            continue
        date = raw_entry.get("date")
        label = raw_entry.get("label")
        out[pin] = {
            "date": date if isinstance(date, str) and len(date) == 8 and date.isdigit() else None,
            "label": label.strip() if isinstance(label, str) and label.strip() else None,
        }
    return out


def _ordered_solvers(cells: list[_Cell], payload: dict[str, Any]) -> list[str]:
    """Solver identities in figure order: dated solvers oldest first, then undated by identity.

    The order is computed ONCE for the whole figure and every panel uses it, so a solver's
    colour and legend position mean one thing across panels. Dated solvers come first,
    ascending by the producer-supplied commit date with the identity as the tie-break;
    undated solvers follow, ordered by identity. A dated series' position is an age claim,
    so the dated series stay contiguous from the oldest and no undated series sits among
    them; an undated series is a solver whose date was not captured, so it is neither the
    oldest nor the newest, and placing it last keeps that from being read off its position.

    Deliberately NOT a parse of the identity token: the read-model's ``solver`` is the pinned
    git ref and carries no date segment, so parsing it orders every solver by hash.

    A solver reaching the figure only through a cell, or named in ``solvers`` but missing
    from ``solver_info``, is undated; no solver is dropped, because a dropped solver would
    read as one that never ran.
    """
    info = _solver_info(payload)
    declared = {str(raw or "").strip() for raw in payload.get("solvers") or []}
    declared.update(cell["solver"] for cell in cells)
    identities = {token for token in declared if token}
    dated = sorted(
        (pin for pin in identities if info.get(pin, {}).get("date")), key=lambda pin: (info[pin]["date"], pin)
    )
    undated = sorted(pin for pin in identities if not info.get(pin, {}).get("date"))
    return dated + undated


def _solver_labels(solvers: list[str], payload: dict[str, Any]) -> dict[str, str]:
    """Legend text per solver identity: the producer's label, else the identity itself.

    A label shared by two solvers would draw two series under one name, so on any collision
    the colliding solvers fall back to their identities, which are distinct by construction.
    The label is display only; the series, the colour and the provenance selection all stay
    keyed on the identity.
    """
    info = _solver_info(payload)
    labels = {pin: info.get(pin, {}).get("label") or pin for pin in solvers}
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return {pin: (pin if counts[label] > 1 else label) for pin, label in labels.items()}


def _ordered_configs(cells: list[_Cell], payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Config identity keys and their x-tick labels.

    Declared order first, then any key reaching the figure only through a cell. Two
    distinct packed keys can render the same short label when a field the label drops is
    the only thing that differs; that collision falls back to the packed keys for ALL
    ticks, because a duplicated tick would merge two configs into one x position and a
    reader would see one solver's bar where two runs exist.
    """
    configs: list[str] = []
    for raw in payload.get("configs") or []:
        token = str(raw or "").strip()
        if token and token not in configs:
            configs.append(token)
    for cell in cells:
        if cell["config"] not in configs:
            configs.append(cell["config"])
    labels = [_config_label(config) for config in configs]
    if len(set(labels)) != len(labels):
        labels = list(configs)
    return configs, labels


def _panel_range(values: list[float]) -> tuple[float, float, float]:
    """``(axis_bottom, axis_top, not_run_marker_y)`` for one panel.

    Derived from that panel's own measured values, with the marker placed ABOVE the tallest
    bar so a ``not_run`` mark can never be read as a magnitude. The baseline is included in
    the lower bound so a bar chart always starts at zero rather than at its own minimum,
    which is what would make two bars' heights non-comparable.
    """
    low = min([0.0, *values])
    high = max([0.0, *values])
    if high <= low:
        high = low + 1.0
    pad = (high - low) * 0.12
    return low, high + pad, high + pad * 0.55


def _build_figure(cells: list[_Cell], payload: dict[str, Any], prov: ProvenanceLog) -> tuple[go.Figure, dict[str, Any]]:
    """Build the figure and the structural facts the manifest records.

    One ``prov.artist`` block per ``(model arm, solver)`` panel series, each declaring the
    read-model with a ``selection`` naming the arm and solver it covers. The block is per
    series rather than per figure because the trace calls must sit LEXICALLY inside it
    (``tests/test_provenance_discipline.py`` walks the AST for that enclosure), and a
    single outer block would attribute every bar in every panel to one anonymous artist.

    Trace construction colours by the solver's position in the sorted solver list rather
    than leaving it to Plotly's colorway. A colorway assigns colours by trace order, so
    with one set of traces per panel the same solver would take a different colour in each
    panel — and a figure whose colour does not mean one thing is a figure whose legend is
    wrong.
    """
    solvers = _ordered_solvers(cells, payload)
    labels = _solver_labels(solvers, payload)
    dated = {pin for pin, entry in _solver_info(payload).items() if entry["date"]}
    arms = _ordered_arms(cells, payload)
    configs, tick_text = _ordered_configs(cells, payload)
    metric_title = str(payload.get("metric_label") or payload.get("metric") or _UNLABELLED_METRIC)

    indexed: dict[tuple[str, str, str], _Cell] = {
        (cell["solver"], cell["model_arm"], cell["config"]): cell for cell in cells
    }

    # `cells` is non-empty whenever this function is called, so `arms` is non-empty too;
    # the `or [""]` keeps the subplot count and the title count derived from ONE list, so
    # a future empty-arms caller produces one untitled panel rather than a count mismatch
    # between `cols=` and `subplot_titles=`.
    panels = arms or [""]
    n_cols = len(panels)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[MODEL_DISPLAY_NAMES.get(arm, arm) for arm in panels],
        shared_yaxes=False,
    )

    not_run_count = 0
    for col, arm in enumerate(panels, start=1):
        values = [
            indexed[(solver, arm, config)]["value"]
            for solver in solvers
            for config in configs
            if (solver, arm, config) in indexed
            and indexed[(solver, arm, config)]["value"] is not None
            and not indexed[(solver, arm, config)]["not_run"]
        ]
        bottom, top, marker_y = _panel_range([value for value in values if value is not None])

        for index, solver in enumerate(solvers):
            heights: list[float | None] = []
            for config in configs:
                cell = indexed.get((solver, arm, config))
                # A cell ABSENT from the read-model is not_run exactly as an explicit
                # `not_run` cell is. Drawing it as zero is the one error this figure
                # exists to prevent: "ran and measured nothing" and "never ran" would
                # become indistinguishable, which is how a wrong-pin arm passes review.
                heights.append(None if cell is None or cell["not_run"] else cell["value"])
            colour = _plotly_theme.OKABE_ITO_COLORWAY[index % len(_plotly_theme.OKABE_ITO_COLORWAY)]
            missing = [
                tick_text[position]
                for position, config in enumerate(configs)
                if (solver, arm, config) not in indexed or indexed[(solver, arm, config)]["not_run"]
            ]
            not_run_count += len(missing)
            with prov.artist(
                axes_id=f"panel_{arm or 'default'}_solver_{solver}",
                kind="bar",
                note=f"{MODEL_DISPLAY_NAMES.get(arm, arm)}: one series for solver {labels[solver]}",
            ) as artist:
                artist.add_channel(
                    "data",
                    ProvenanceRef(
                        source_path=READ_MODEL_FILENAME,
                        selection={"solver": solver, "model_arm": arm},
                    ),
                )
                fig.add_trace(
                    go.Bar(
                        x=tick_text,
                        y=heights,
                        name=labels[solver],
                        legendgroup=solver,
                        showlegend=(col == 1),
                        marker_color=colour,
                        hovertemplate="%{x}<br>%{y}<extra>" + labels[solver] + "</extra>",
                    ),
                    row=1,
                    col=col,
                )
                if missing:
                    fig.add_trace(
                        go.Scatter(
                            x=missing,
                            y=[marker_y] * len(missing),
                            mode="markers",
                            name=labels[solver],
                            legendgroup=solver,
                            showlegend=False,
                            marker={"symbol": "x-thin-open", "size": 10, "color": colour},
                            hovertemplate="%{x}<br>not_run<extra>" + labels[solver] + "</extra>",
                            cliponaxis=False,
                        ),
                        row=1,
                        col=col,
                    )

        fig.update_xaxes(title_text="compute config", tickangle=-20, row=1, col=col)
        fig.update_yaxes(
            title_text=metric_title if col == 1 else None,
            range=[bottom, top],
            row=1,
            col=col,
        )

    fig.update_layout(
        template="triton_journal",
        title="Cross-solver comparison",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        showlegend=bool(solvers),
        legend={"title": "solver", "orientation": "v", "yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.01},
        margin={"l": _MARGIN_L_PX, "r": _MARGIN_R_PX, "t": _MARGIN_T_PX, "b": _MARGIN_B_PX},
        width=430 * n_cols,
        height=_MARGIN_T_PX + _PLOT_H_PX + _MARGIN_B_PX,
    )
    if not_run_count:
        bottom_px = add_figure_caption(
            fig,
            "x marks a not_run cell: that solver has no simulation at that config, which is "
            "NOT a measured zero. A config missing from one solver is kept on the x axis so "
            "its absence is visible.",
            content_w_px=430 * n_cols - _MARGIN_L_PX - _MARGIN_R_PX,
            plot_h_px=_PLOT_H_PX,
            axis_band_px=_X_AXIS_BAND_PX,
        )
        fig.update_layout(
            margin={"b": bottom_px},
            height=_MARGIN_T_PX + _PLOT_H_PX + bottom_px,
        )

    facts = {
        "metric": payload.get("metric"),
        "solver_count": len(solvers),
        "solvers": solvers,
        # Which solvers carry no usable commit date: their position is a tie-break, not an age
        # claim, and a reader of the manifest can tell without re-opening the read-model.
        "undated_solvers": [pin for pin in solvers if pin not in dated],
        "model_arm_count": len(arms),
        "model_arms": arms,
        "config_count": len(configs),
        "cell_count": len(cells),
        "not_run_cell_count": not_run_count,
    }
    return fig, facts


def _figure_to_html(fig: go.Figure, *, js_mode: str) -> str:
    """Serialize the figure as a standalone HTML document.

    The mode bar keeps its export buttons but drops the selection tools: on a grouped bar
    chart a lasso or box selection looks like a filter and is not one, so a reader can
    "select" a subset that then affects nothing and reads as broken. The export filename is
    fixed so a saved SVG lands under the figure's own name rather than plotly's default.
    """
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "hoverCompareCartesian"],
        "toImageButtonOptions": {"format": "svg", "filename": "cross_solver_comparison", "scale": 2},
    }
    return pio.to_html(fig, include_plotlyjs=js_mode, full_html=True, config=config)


def _placeholder_html(problem: str | None) -> str:
    """Honest empty state, naming WHICH of the three causes produced it.

    ``problem`` is `_read_model`'s cause token: ``None`` when the read-model parsed but
    carried no placeable cell, otherwise the cause. The two cases get different sentences
    because they call for different actions — an absent read-model means the combine has
    not run, while an unparseable one means it ran and wrote something broken.
    """
    if problem is None:
        diagnosis = f"<code>{READ_MODEL_FILENAME}</code> carries no placeable cells."
    else:
        diagnosis = f"<code>{READ_MODEL_FILENAME}</code> could not be used: {escape(problem)}."
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "padding:12px;color:#333;margin:0;}h2{color:#232D4B;}.note{color:#666;}"
        "</style></head><body><section class='cross-solver-comparison'>"
        "<h2>Cross-solver comparison</h2>" + f"<p class='note'>{diagnosis} "
        "This figure is produced only by a combine over bundles that ran more than one "
        "solver commit; a single-solver combine renders no cross-solver figure at all."
        "</p></section></body></html>"
    )
