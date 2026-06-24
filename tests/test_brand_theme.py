"""Brand-theme + branding-realization unit tests (reporting-system config-layering-and-branding, Phase 1).

Fast, fixture-free unit tests cover the brand_theme model (R-1), the
report.css.j2 render byte-stability + config-drive at the
``_emit_report_artifacts`` surface (R-3/R-8), the placeholder map, and the
D-5 HTML-table overlay pattern (R-4). ``_emit_report_artifacts`` is the EXACT
function the dominant ``render_report_runner`` fresh-instance path calls, so
testing it directly (no ``run()`` warmup, no ``self._brand_theme``) is the
regression cover for plan-review SE Flag 1.

The real-data run→render→bundle path (formerly the ``@pytest.mark.slow`` tests
in this file) is now covered end-to-end by ``test_analysis_test_end_to_end.py``
via ``analysis.test()`` — the retired ``test_PC_*`` tier's replacement.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hhemt.config.brand_theme import DEFAULT_BRAND_THEME, brand_theme
from hhemt.workflow import _brand_theme_css_map, _emit_report_artifacts

_DEFAULT_HEXES = ("#232D4B", "#E57200", "#F1F1EF", "#DADADA", "#666666", "#495E9D")


# ---- R-1: model validation ---------------------------------------------------
def test_brand_theme_defaults_construct():
    t = brand_theme()
    assert t.primary_color == "#232D4B"
    assert t.accent_color == "#E57200"
    assert t.neutral_light == "#F1F1EF"
    assert t.neutral_medium == "#DADADA"
    assert t.text_muted == "#666666"
    assert t.link_color == "#495E9D"
    assert t.upper_left_text is None
    assert DEFAULT_BRAND_THEME == t


def test_brand_theme_validates_hex_via_mplcolor():
    brand_theme(primary_color="#0A7E8C")  # valid hex
    brand_theme(primary_color="rebeccapurple")  # valid named
    with pytest.raises(ValidationError):
        brand_theme(primary_color="not_a_color_xyz")


def test_brand_theme_css_map_maps_semantic_to_placeholder():
    t = brand_theme(primary_color="#0A7E8C", accent_color="#112233")
    m = _brand_theme_css_map(t)
    assert m["uva_blue"] == "#0A7E8C"
    assert m["uva_orange"] == "#112233"
    assert set(m) == {
        "uva_blue",
        "uva_orange",
        "uva_light_gray",
        "uva_medium_gray",
        "uva_text_gray",
        "uva_link_blue",
    }


# ---- R-3 / R-8: report.css.j2 render at the _emit_report_artifacts surface ----
# This is the exact function the dominant render_report_runner fresh-instance
# path calls — no analysis instance / run() warmup needed (SE Flag 1).
def test_report_css_default_is_byte_stable(tmp_path):
    _emit_report_artifacts(tmp_path)  # brand_theme=None -> code-frozen UVA default
    css = (tmp_path / "report" / "report.css").read_text()
    for hexval in _DEFAULT_HEXES:
        assert hexval in css
    assert "${" not in css  # no unrendered placeholder remains
    assert "#1e293b" in css  # Tailwind slate UI color left untouched


def test_report_css_config_driven_primary(tmp_path):
    theme = brand_theme(primary_color="#0A7E8C")
    _emit_report_artifacts(tmp_path, brand_theme=_brand_theme_css_map(theme))
    css = (tmp_path / "report" / "report.css").read_text()
    assert "#0A7E8C" in css  # custom primary rendered into the :root block
    assert "--uva-blue: #0A7E8C;" in css
    # The default primary no longer appears as the rendered :root VALUE (it may
    # still appear in the static descriptive header comment, which is fine).
    assert "--uva-blue: #232D4B;" not in css
    assert "${" not in css


# ---- R-4 / D-5: HTML-table overlay sources brand defaults from the theme ------
def test_table_overlay_sources_brand_defaults_from_theme():
    # Exercises the model_validate overlay pattern run() applies at D-5: the
    # brand-derived HTML-table fields take theme colors; frozen semantic fields
    # (th_text_color etc.) do not.
    from hhemt.config.report import report_config

    theme = brand_theme(
        primary_color="#0A7E8C",
        accent_color="#112233",
        neutral_light="#445566",
        neutral_medium="#778899",
    )
    base = report_config()
    overlay = {
        "primary_color": theme.primary_color,
        "cell_border_color": theme.neutral_medium,
        "row_alt_bg_color": theme.neutral_light,
        "row_hover_bg_color": theme.accent_color,
    }
    merged = type(base).model_validate(
        {
            **base.model_dump(),
            "errors_and_warnings": {
                **base.errors_and_warnings.model_dump(),
                **overlay,
            },
            "scenario_status_appendix": {
                **base.scenario_status_appendix.model_dump(),
                **overlay,
            },
        }
    )
    assert merged.errors_and_warnings.primary_color == "#0A7E8C"
    assert merged.errors_and_warnings.cell_border_color == "#778899"
    assert merged.errors_and_warnings.row_alt_bg_color == "#445566"
    assert merged.errors_and_warnings.row_hover_bg_color == "#112233"
    assert merged.scenario_status_appendix.primary_color == "#0A7E8C"
    # Frozen semantic colors are NOT theme-driven.
    assert merged.errors_and_warnings.th_text_color == base.errors_and_warnings.th_text_color
    assert merged.errors_and_warnings.body_text_color == base.errors_and_warnings.body_text_color
