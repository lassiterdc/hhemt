"""Single frozen plotting-vocabulary source of truth (ADR-1, reporting-system).

Project-owned token enums (vmin/vmax strategy, value->encoding policy, font
targets, panel-scale policy, encoding channels) PLUS thin registry-validator
functions that membership-check dynamically-named matplotlib/plotly tokens
against the live registries. Both layers are exposed as ``Annotated`` Pydantic
type aliases so a model field is just ``cmap: MplColormap``, and the SAME
module is imported by the plotting functions -- legal values live in exactly
one place (user O-f requirement; TO-3).

Enum base is ``(str, Enum)``, NOT ``enum.StrEnum``: the project floor is
``requires-python = ">= 3.10"`` and ``just testall`` collects the suite on
3.10, but ``StrEnum`` is 3.11+. ``(str, Enum)`` is JSON-byte-identical to
``StrEnum`` under Pydantic v2 (members serialize by value; identical
JSON-schema ``enum`` emission) and mirrors ``bundle/_path_policy.py``
(ADR-1 decision D1).

Registry membership is resolved LAZILY inside each validator body, never at
module import, so ``import ...config.viz_vocabulary`` does not eagerly import
matplotlib/plotly on every path that reaches it (notably the D2-rewired
``report_renderers/_provenance.py``).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import AfterValidator

# ---- (1) project-owned tokens (typed, JSON-schema-enumerated) ---------------


class VminVmaxStrategy(str, Enum):
    absolute = "absolute"
    quantile = "quantile"
    per_panel_max = "per_panel_max"
    shared_across_panels = "shared_across_panels"


class ValueEncodingPolicy(str, Enum):
    colorscale = "colorscale"
    one_to_one_dict = "one_to_one_dict"
    value_to_shape = "value_to_shape"


class FontTarget(str, Enum):
    figure_title = "figure_title"
    axis_label = "axis_label"
    tick_label = "tick_label"
    legend_title = "legend_title"
    legend_text = "legend_text"
    callout = "callout"
    caption = "caption"
    subplot_label = "subplot_label"


class PanelScalePolicy(str, Enum):
    shared = "shared"
    independent = "independent"


class EncodingChannel(str, Enum):
    """Visual encoding channels a data variable may drive in a figure.

    Single source for ``report_renderers/_provenance._Channel`` (ADR-1, D2).
    The member set MUST equal the legacy ``_Channel`` Literal, in order
    (verified 2026-06-06 against ``_provenance.py:47-49``);
    ``tests/test_viz_vocabulary.py::test_encoding_channel_value_stability``
    pins it so the single-source rewire can never silently drift.
    """

    x = "x"
    y = "y"
    z = "z"
    color = "color"
    size = "size"
    linewidth = "linewidth"
    alpha = "alpha"
    hatch = "hatch"
    gid = "gid"
    other = "other"


# ---- (2) shared registry-membership validator helper -----------------------

_ENUMERATE_FULL_THRESHOLD = 20


def _registry_membership_validator(
    value: object,
    *,
    legal: list,
    label: str,
    enumerate_hint: str,
) -> object:
    """Membership-check ``value`` against ``legal``; raise a size-gated error.

    Legal sets with <= ``_ENUMERATE_FULL_THRESHOLD`` members are enumerated in
    full; larger sets render a sample + total count + an enumerate hint, so a
    single bad token never dumps 180 colormaps / 162 symbols into a traceback.
    """
    if value in legal:
        return value
    legal_sorted = sorted(str(v) for v in legal)
    if len(legal_sorted) <= _ENUMERATE_FULL_THRESHOLD:
        legal_str = ", ".join(legal_sorted)
    else:
        sample = ", ".join(legal_sorted[:8])
        legal_str = f"{sample}, ... ({len(legal_sorted)} total)"
    raise ValueError(f"{value!r} is not a valid {label}. Legal values: {legal_str}. Enumerate via: {enumerate_hint}.")


# ---- (3) registry-validator functions (lazy registry resolution) -----------


def validate_mpl_colormap(name: str) -> str:
    import matplotlib

    return _registry_membership_validator(
        name,
        legal=list(matplotlib.colormaps),
        label="matplotlib colormap",
        enumerate_hint="list(matplotlib.colormaps)",
    )


def validate_mpl_marker(marker: int | str) -> int | str:
    # MarkerStyle.markers keys include 12 int codes (0-11) AND str codes; accept
    # both -- a str-only alias would make the int markers unreachable
    # (Pydantic would coerce int 0 -> "0", which is not a key).
    from matplotlib.markers import MarkerStyle

    return _registry_membership_validator(
        marker,
        legal=list(MarkerStyle.markers),
        label="matplotlib marker",
        enumerate_hint="matplotlib.markers.MarkerStyle.markers",
    )


def validate_mpl_linestyle(ls: str) -> str:
    from matplotlib.lines import Line2D

    return _registry_membership_validator(
        ls,
        legal=list(Line2D.lineStyles),
        label="matplotlib linestyle",
        enumerate_hint="matplotlib.lines.Line2D.lineStyles",
    )


def validate_mpl_color(color: str) -> str:
    # Non-enumerable space (hex / RGBA / named). Validate by construction.
    from matplotlib.colors import to_rgba

    try:
        to_rgba(color)
    except ValueError as exc:
        raise ValueError(f"{color!r} is not a matplotlib color: {exc}") from exc
    return color


def validate_plotly_colorscale(name: str) -> str:
    # .lower() is LOAD-BEARING: named_colorscales() returns 94 lowercase-only
    # entries, but plotly also accepts capitalized built-ins ("Viridis",
    # "RdBu"); lowering bridges that gap. Do not drop it.
    import plotly.colors as pc

    legal = pc.named_colorscales()
    if name.lower() in legal:
        return name
    return _registry_membership_validator(
        name.lower(),
        legal=list(legal),
        label="plotly named colorscale",
        enumerate_hint="plotly.colors.named_colorscales()",
    )


def validate_plotly_symbol(symbol: str) -> str:
    # SymbolValidator().values str-filtered = 324 (162 names + 162 numeric
    # aliases). Restrict the frozen config vocabulary to the 162 human-readable
    # NAMES -- numeric aliases ("0" == circle) are opaque in a config file
    # (data-visualization-specialist R1, 2026-06-06).
    from plotly.validators.scatter.marker import SymbolValidator

    legal = [v for v in SymbolValidator().values if isinstance(v, str) and not v.lstrip("-").isdigit()]
    return _registry_membership_validator(
        symbol,
        legal=legal,
        label="plotly marker symbol",
        enumerate_hint=("plotly.validators.scatter.marker.SymbolValidator().values"),
    )


def validate_font_family(name: str) -> str:
    # No authoritative font registry -- matplotlib silently falls back to
    # DejaVu Sans for an unavailable family, so membership cannot be validated
    # without over-promising. Validate non-empty only (ADR-1).
    if not name or not name.strip():
        raise ValueError("font family must be a non-empty string")
    return name


# ---- (4) Annotated type aliases for direct Pydantic-field use --------------

MplColormap = Annotated[str, AfterValidator(validate_mpl_colormap)]
MplMarker = Annotated[int | str, AfterValidator(validate_mpl_marker)]
MplLineStyle = Annotated[str, AfterValidator(validate_mpl_linestyle)]
MplColor = Annotated[str, AfterValidator(validate_mpl_color)]
PlotlyColorscale = Annotated[str, AfterValidator(validate_plotly_colorscale)]
PlotlySymbol = Annotated[str, AfterValidator(validate_plotly_symbol)]
FontFamily = Annotated[str, AfterValidator(validate_font_family)]
