"""Tests for the frozen plotting-vocabulary source of truth (ADR-1).

Covers R-1..R-9 of the reporting-system_frozen-viz-vocabulary plan.
"""

from __future__ import annotations

import sys

import pytest
from pydantic import BaseModel, ValidationError

from hhemt.config import viz_vocabulary as viz

_PROJECT_ENUMS = [
    viz.VminVmaxStrategy,
    viz.ValueEncodingPolicy,
    viz.FontTarget,
    viz.PanelScalePolicy,
    viz.EncodingChannel,
]


# ---- R-1 / R-5-safety: every project enum is a str subclass (3.10 guard) ----
@pytest.mark.parametrize("enum_cls", _PROJECT_ENUMS)
def test_project_enums_subclass_str(enum_cls):
    # Guards the D1 decision: a future refactor to StrEnum would pass on 3.11+
    # but ImportError-fail collection on 3.10. issubclass(str) is the cheap,
    # 3.10-collectable tripwire that keeps the (str, Enum) idiom.
    assert issubclass(enum_cls, str)


# ---- R-1 / R-7 / R-9: EncodingChannel value set pins the legacy Literal -----
def test_encoding_channel_value_stability():
    assert [c.value for c in viz.EncodingChannel] == [
        "x",
        "y",
        "z",
        "color",
        "size",
        "linewidth",
        "alpha",
        "hatch",
        "gid",
        "other",
    ]


def test_provenance_channel_is_encoding_channel_alias():
    from hhemt.report_renderers import _provenance

    assert _provenance._Channel is viz.EncodingChannel


# ---- R-2: membership pass/fail per registry validator -----------------------
@pytest.mark.parametrize(
    "validator, good, bad",
    [
        (viz.validate_mpl_colormap, "viridis", "not_a_colormap_xyz"),
        (viz.validate_mpl_marker, "o", "not_a_marker_xyz"),
        (viz.validate_mpl_linestyle, "--", "not_a_linestyle_xyz"),
        (viz.validate_mpl_color, "red", "not_a_color_xyz"),
        (viz.validate_plotly_colorscale, "Viridis", "not_a_scale_xyz"),
        (viz.validate_plotly_symbol, "square-open", "not_a_symbol_xyz"),
    ],
)
def test_registry_validator_membership(validator, good, bad):
    assert validator(good) == good
    with pytest.raises(ValueError):
        validator(bad)


# ---- R-5: int marker codes are reachable ------------------------------------
def test_mpl_marker_accepts_int_code():
    assert viz.validate_mpl_marker(0) == 0


# ---- R-6: plotly symbol legal set is names-only (no numeric aliases) --------
def test_plotly_symbol_rejects_numeric_alias():
    with pytest.raises(ValueError):
        viz.validate_plotly_symbol("0")


# ---- R-3: size-gated error message ------------------------------------------
def test_colormap_error_is_size_gated():
    with pytest.raises(ValueError) as exc:
        viz.validate_mpl_colormap("not_a_colormap_xyz")
    msg = str(exc.value)
    assert "not_a_colormap_xyz" in msg
    assert "total)" in msg  # large registry -> sample + count, not full dump
    assert "list(matplotlib.colormaps)" in msg


def test_linestyle_error_enumerates_full_small_set():
    with pytest.raises(ValueError) as exc:
        viz.validate_mpl_linestyle("not_a_linestyle_xyz")
    msg = str(exc.value)
    assert "total)" not in msg  # small registry -> full enumeration


# ---- R-4: Annotated aliases wire the validators into Pydantic ----------------
def test_annotated_aliases_validate_in_pydantic_model():
    class _M(BaseModel):
        cmap: viz.MplColormap
        marker: viz.MplMarker
        ls: viz.MplLineStyle
        color: viz.MplColor
        colorscale: viz.PlotlyColorscale
        symbol: viz.PlotlySymbol
        font: viz.FontFamily

    ok = _M(
        cmap="viridis",
        marker="o",
        ls="--",
        color="red",
        colorscale="Viridis",
        symbol="square-open",
        font="DejaVu Sans",
    )
    assert ok.cmap == "viridis"

    with pytest.raises(ValidationError):
        _M(
            cmap="not_a_colormap_xyz",
            marker="o",
            ls="--",
            color="red",
            colorscale="Viridis",
            symbol="square-open",
            font="DejaVu Sans",
        )


# ---- A1: aliases survive the cfgBaseModel "*"-mode=before star-validator -----
def test_annotated_aliases_validate_on_cfgbasemodel():
    # The downstream consumer plans declare these aliases on cfgBaseModel
    # subclasses, where cfgBaseModel's @field_validator("*", mode="before")
    # (_check_paths_exist) co-runs with each alias's AfterValidator. Confirm the
    # star-validator (which returns non-Path values unchanged) does not shadow
    # the registry validation.
    from hhemt.config.base import cfgBaseModel

    class _Cfg(cfgBaseModel):
        cmap: viz.MplColormap = "viridis"

    assert _Cfg().cmap == "viridis"
    with pytest.raises(ValidationError):
        _Cfg(cmap="not_a_colormap_xyz")


def test_font_family_rejects_empty():
    with pytest.raises(ValueError):
        viz.validate_font_family("  ")


# ---- R-8: importing the module BODY is cheap (no matplotlib/plotly at load) --
def test_import_does_not_load_matplotlib_or_plotly():
    # Load ONLY the viz_vocabulary module body via spec_from_file_location, in a
    # subprocess that never imports the hhemt PACKAGE. Importing
    # the dotted path would fire hhemt/__init__.py (-> .toolkit),
    # which eagerly imports matplotlib (empirically confirmed 2026-06-07) — that
    # is a package-init concern, NOT a viz_vocabulary concern. This test isolates
    # the module body so R-8 ("the module itself does not eagerly import
    # matplotlib/plotly") is what is actually asserted.
    import subprocess

    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'viz_vocabulary', r'" + str(viz.__file__) + "')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib eagerly imported'\n"
        "assert 'plotly' not in sys.modules, 'plotly eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
