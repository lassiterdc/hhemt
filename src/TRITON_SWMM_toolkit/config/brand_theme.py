"""Pydantic v2 model for the brand-theme layer (ADR-7 layer 2, reporting-system).

The brand theme is the institutional-identity layer: the small set of hex colors
and the report upper-left text that distinguish one deploying lab's report from
another's. It is intentionally MINIMAL — it carries ONLY the report.css :root
palette (6 colors), the HTML-table primary/accent defaults those colors feed, and
the navbar upper-left text. The Okabe-Ito categorical DATA palette is NOT here: it
is a perceptually-validated colorblind-safe choice frozen in the plotting theme,
not swappable branding (Decision D-5).

Color fields are typed with the frozen layer-1 ``MplColor`` alias from
``viz_vocabulary`` (ADR-1) so hex/named/RGBA values validate at config-load and the
legal-color vocabulary lives in exactly one place. Field names are semantic
(primary/accent/neutral_*), not institution-specific.

One-way import rule (mirrors config/report.py <- config/analysis.py): this module
imports ONLY viz_vocabulary + base; neither report.py nor analysis.py is imported
here, so both may import this module without a cycle.
"""

from __future__ import annotations

from pydantic import Field

from TRITON_SWMM_toolkit.config.base import cfgBaseModel
from TRITON_SWMM_toolkit.config.viz_vocabulary import MplColor


class brand_theme(cfgBaseModel):
    """Institutional brand-identity layer (ADR-7 layer 2)."""

    primary_color: MplColor = Field(
        "#232D4B",
        description=(
            "Dominant brand color (UVA Blue default). Maps to report.css "
            "--uva-blue (table th background, panel/h2/h3 headings) and defaults "
            "_HtmlTableStyleBase.primary_color."
        ),
    )
    accent_color: MplColor = Field(
        "#E57200",
        description=(
            "Accent brand color (UVA Orange default). Maps to report.css "
            "--uva-orange (link hover, sidebar icon, the L204 chart-bar mask) "
            "and derives _HtmlTableStyleBase.row_hover_bg_color."
        ),
    )
    neutral_light: MplColor = Field(
        "#F1F1EF",
        description=(
            "Light neutral. Maps to report.css --uva-light-gray and defaults _HtmlTableStyleBase.row_alt_bg_color."
        ),
    )
    neutral_medium: MplColor = Field(
        "#DADADA",
        description=(
            "Medium neutral. Maps to report.css --uva-medium-gray and defaults _HtmlTableStyleBase.cell_border_color."
        ),
    )
    text_muted: MplColor = Field(
        "#666666",
        description="Muted text neutral. Maps to report.css --uva-text-gray.",
    )
    link_color: MplColor = Field(
        "#495E9D",
        description=(
            "Hyperlink color. Maps to report.css --uva-link-blue (white-on-link contrast 6.23:1, WCAG AA pass)."
        ),
    )
    upper_left_text: str | None = Field(
        None,
        description=(
            "Report navbar upper-left text. When None (default), the render "
            "facade substitutes cfg_analysis.analysis_id (ADR-7 / Decision D-6)."
        ),
    )


# Code-frozen default theme used when cfg_analysis.brand_theme is None (no YAML
# provided). Mirrors the report_config field's "frozen default" semantics.
DEFAULT_BRAND_THEME = brand_theme()
