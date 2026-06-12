"""Brand-theme + branding-realization tests (reporting-system config-layering-and-branding, Phase 1).

Two tiers:

* Fast, fixture-free unit tests cover the brand_theme model (R-1), the
  report.css.j2 render byte-stability + config-drive at the
  ``_emit_report_artifacts`` surface (R-3/R-8), the placeholder map, and the
  D-5 HTML-table overlay pattern (R-4). ``_emit_report_artifacts`` is the EXACT
  function the dominant ``render_report_runner`` fresh-instance path calls, so
  testing it directly (no ``run()`` warmup, no ``self._brand_theme``) is the
  regression cover for plan-review SE Flag 1.
* ``@pytest.mark.slow`` real-data tests use the Norfolk cached fixture to cover
  the ``run()`` resolution ladder (R-2), the resolved HTML-table primary
  (R-4/D-5 through ``run()``), and bundle reachability / regenerate
  reproduction (R-6/V-7). They mirror the verified ``test_PC_04`` /
  ``test_synth_08`` run→render→bundle idioms.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml
from pydantic import ValidationError

from TRITON_SWMM_toolkit.config.brand_theme import DEFAULT_BRAND_THEME, brand_theme
from TRITON_SWMM_toolkit.workflow import _brand_theme_css_map, _emit_report_artifacts

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
    from TRITON_SWMM_toolkit.config.report import report_config

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


# ---- R-2 / D-5 (through run): real-data resolution + branded render -----------
@pytest.mark.slow
def test_run_resolves_branded_css(tmp_path, norfolk_single_sim_analysis_cached):
    theme_yaml = tmp_path / "brand_theme.yaml"
    theme_yaml.write_text(yaml.safe_dump({"primary_color": "#0A7E8C"}))
    analysis = norfolk_single_sim_analysis_cached
    analysis.run(from_scratch=False, override_brand_theme=theme_yaml)
    # Resolution ladder stored the theme; the D-5 overlay re-sourced the table.
    assert analysis._brand_theme.primary_color == "#0A7E8C"
    assert analysis._cfg_report.errors_and_warnings.primary_color == "#0A7E8C"
    analysis.render_report(format="html")
    css = (analysis.analysis_paths.analysis_dir / "report" / "report.css").read_text()
    assert "#0A7E8C" in css
    assert "${" not in css


# ---- R-6 / V-7: bundle reachability — regenerate reproduces the theme ---------
@pytest.mark.slow
def test_bundle_regenerate_reproduces_theme(tmp_path, norfolk_single_sim_analysis_cached):
    theme_yaml = tmp_path / "brand_theme.yaml"
    theme_yaml.write_text(yaml.safe_dump({"primary_color": "#0A7E8C"}))
    analysis = norfolk_single_sim_analysis_cached
    analysis.run(from_scratch=False, override_brand_theme=theme_yaml)
    analysis.render_report(format="html")

    bundle_zip = tmp_path / "bundle.zip"
    bundle_path = analysis.bundle_report_data(bundle_zip)
    assert bundle_path.exists(), "bundle zip not emitted"

    result = subprocess.run(
        [
            "TRITON_SWMM_toolkit",
            "report-from-bundle",
            str(bundle_path),
            "--format",
            "html",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"report-from-bundle failed:\n{result.stdout}\n{result.stderr}"
    unpack_dir = bundle_path.parent / bundle_path.stem
    regen_css = (unpack_dir / "report" / "report.css").read_text()
    assert "#0A7E8C" in regen_css
    assert "${" not in regen_css
