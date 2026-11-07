"""eda/_promote.py — EDA->report promotion via config-materialization (ADR-11).

Promote an EDA plot (already rendered by ``eda/_plotting.py`` with a canonical
ADR-2 plot-ID and declared source paths) into either a STANDARD publication-static
plot (emit an ADR-4 ``StaticPlotBaseConfig`` YAML keyed on the plot-ID, which the
user lists in ``cfg_analysis.static_plot_configs``) or a NAMED reporting set
(record the plot-ID's promotion intent against an ADR-5 ``ReportingSet``;
register-only -- the Snakemake ``report()``-routing adapter is owned by
``reporting-system_eda-skill``).

ZERO renderer re-coding (ASR-11): the EDA renderer is reused unchanged; promotion
is pure config-materialization keyed on the plot-ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from hhemt.config.report import _apply_reporting_set_renames
from hhemt.config.static_plots import StaticPlotBaseConfig
from hhemt.exceptions import ConfigurationError
from hhemt.report_renderers._reporting_sets import REPORTING_SETS


@dataclass(frozen=True)
class EdaReportingSetRegistration:
    """Typed record of an EDA plot-ID's promotion intent against a named ReportingSet.

    The handoff artifact that a future ``report()``-routing adapter for EDA plots
    consumes. ``routing`` names the component that owns the deferred routing, as
    ``"deferred:{owner}"``.
    """

    plot_id: str
    set_name: str
    routing: str


# Mirrors the canonical wildcard charset (report_plot_ids.py; the ADR-2
# ^[A-Za-z0-9_.]+$ grammar -- no hyphen). StaticPlotBaseConfig re-validates plot_id
# on the static path; register_* needs an independent check.
_PLOT_ID_CHARSET = re.compile(r"^[A-Za-z0-9_.]+$")

# Vector/raster formats producible by BOTH matplotlib AND Plotly-Kaleido. A
# Plotly-sourced EDA plot must not emit pgf/ps (Kaleido cannot produce them).
_PLOTLY_PORTABLE_FORMATS = frozenset({"pdf", "svg", "png", "eps"})

_PROMOTED_HEADER = (
    "# Promoted EDA plot (source render backend: plotly).\n"
    "# The matplotlib-specific colorbar/colormap fields below have no effect for this\n"
    "# plot ID until the static plots consumer defines render dispatch for EDA plot IDs.\n"
)


def promote_eda_plot_to_static_config(
    plot_id: str,
    *,
    output_path: Path | None = None,
    caption: str | None = None,
) -> Path:
    """Emit a base ``StaticPlotBaseConfig`` YAML for an EDA plot, per D-2/D-3 option a.

    Populates ONLY backend-neutral base fields. The matplotlib-specific colorbar and colormap
    fields stay at schema defaults that are inert and harmless for a Plotly figure. NEVER a
    per-function subclass. ``output_format`` stays at the base default ``"pdf"``, a format
    that is portable across matplotlib AND Plotly-Kaleido. Writes a REAL file, because
    ``config/analysis.py::_check_static_plot_configs_exist`` raises on a missing
    path. ``output_path=None`` defaults to a cwd-relative
    ``promoted_static_configs/{plot_id}.yaml`` -- NEVER under ``analysis_dir/``.
    """
    if not _PLOT_ID_CHARSET.match(plot_id):
        raise ValueError(f"plot_id {plot_id!r} is not charset-safe; must match ^[A-Za-z0-9_.]+$ (ADR-2).")
    # renderer_kind is the ADR-2 plot-ID's leading '__'-segment (the renderer
    # module name), the same convention static_snakefile_generator uses. For a
    # promoted EDA plot it is not (yet) a key in STATIC_PLOT_CONFIG_REGISTRY —
    # that is expected per the header comment (EDA-plot-ID render dispatch is
    # deferred); the field is required so the emitted YAML validates round-trip.
    renderer_kind = plot_id.split("__", 1)[0]
    cfg = StaticPlotBaseConfig(plot_id=plot_id, renderer_kind=renderer_kind, caption=caption)
    if cfg.output_format not in _PLOTLY_PORTABLE_FORMATS:
        raise ValueError(
            f"output_format {cfg.output_format!r} is not producible by Plotly-Kaleido; "
            f"a promoted Plotly-sourced EDA plot must use one of {sorted(_PLOTLY_PORTABLE_FORMATS)}."
        )
    if output_path is None:
        output_path = Path.cwd() / "promoted_static_configs" / f"{plot_id}.yaml"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    output_path.write_text(_PROMOTED_HEADER + body)
    # lean (c): surface the paste-ready cfg_analysis line so the user can wire the
    # emitted static config into their analysis config without hunting for the field.
    print(
        f"[eda-promote] Static config for {plot_id!r} written to {output_path}.\n"
        f"[eda-promote] Add it to your analysis config to render it as a publication static plot:\n"
        f"[eda-promote]   static_plot_configs:\n"
        f"[eda-promote]     - {output_path}",
        flush=True,
    )
    return output_path


def register_eda_plot_in_reporting_set(plot_id: str, set_name: str) -> EdaReportingSetRegistration:
    """Record an EDA plot-ID's promotion intent against a named ``ReportingSet``.

    Register-ONLY. Validates ``plot_id`` against the canonical plot-ID charset and ``set_name`` against
    the keys of ``REPORTING_SETS``, and returns a registration record that a future
    ``report()``-routing adapter consumes. The function leaves a live ``renderer_selection`` tuple
    unmutated, because a live append raises KeyError in the pre-adapter dispatcher, and authors no
    ``report_renderers/`` adapter, ``workflow.py`` builder, or dispatcher-map edit. A set whose selection renders no
    EDA figure raises ``ConfigurationError`` naming the sets that do.
    """
    if not _PLOT_ID_CHARSET.match(plot_id):
        raise ValueError(f"plot_id {plot_id!r} is not charset-safe; must match ^[A-Za-z0-9_.]+$ (ADR-2).")
    # Rename before the membership check: applied at the registry lookup below it, an old key would
    # already have been refused here as unregistered.
    (set_name,) = _apply_reporting_set_renames([set_name])
    if set_name not in REPORTING_SETS:
        raise ValueError(
            f"set_name {set_name!r} is not a registered ReportingSet; valid sets: {sorted(REPORTING_SETS)}."
        )
    # A set whose selection carries the eda_compute_sensitivity adapter renders EDA figures, so
    # a registration against it is routed (config-selectable via report.reporting_set). A set
    # without the adapter renders none, so a registration against it would route nothing: it is
    # refused, naming the sets that do render EDA figures.
    rendering_sets = sorted(
        name
        for name, reporting_set in REPORTING_SETS.items()
        if any(sel.builder_key == "eda_compute_sensitivity" for sel in reporting_set.renderer_selection)
    )
    if set_name not in rendering_sets:
        raise ConfigurationError(
            "reporting_set",
            f"reporting set {set_name!r} renders no EDA figure, so registering {plot_id!r} against it "
            f"would route nothing. Sets that render EDA figures: {', '.join(rendering_sets)}.",
            fix_hint=f"Use one of: {', '.join(rendering_sets)}.",
        )
    print(
        f"[eda-promote] {plot_id!r} registered against reporting set {set_name!r}: "
        f"this set already renders the EDA adapter; select it with "
        f"report.reporting_set={set_name!r} in your analysis config.",
        flush=True,
    )
    return EdaReportingSetRegistration(
        plot_id=plot_id,
        set_name=set_name,
        routing="deferred:reporting-system_eda-skill",
    )
