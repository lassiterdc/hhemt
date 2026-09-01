"""Q4 (iter-2): system_overview is model-parametric — a pure-TRITON master renders the DEM
+ Mannings panels (no SWMM hydrology/hydraulics), a coupled/swmm master adds the SWMM panels
in a 2×2 grid. The gate is _master_renders_swmm over the enabled model types (unioned across
sensitivity subs); Phase-2 adds the Manning's panel to BOTH arms."""

import types

from hhemt.report_renderers.system_overview import (
    _build_manifest_data,
    _master_renders_swmm,
)


def _mk(models, sensitivity=False, subs=None):
    a = types.SimpleNamespace()
    a.cfg_analysis = types.SimpleNamespace(toggle_sensitivity_analysis=sensitivity)
    a._get_enabled_model_types = lambda: list(models)
    if subs is not None:
        a.sensitivity = types.SimpleNamespace(analyses=subs)
    return a


def test_pure_triton_master_renders_no_swmm():
    assert _master_renders_swmm(_mk(["triton"])) is False


def test_coupled_and_standalone_swmm_render_swmm():
    assert _master_renders_swmm(_mk(["tritonswmm"])) is True
    assert _master_renders_swmm(_mk(["swmm"])) is True


def test_sensitivity_master_unions_over_subs():
    sub_t = _mk(["triton"])
    sub_c = _mk(["tritonswmm"])
    # all-pure-TRITON subs -> DEM-only
    assert _master_renders_swmm(_mk(["triton"], sensitivity=True, subs={"a": sub_t})) is False
    # ANY SWMM-bearing sub -> render SWMM
    assert _master_renders_swmm(_mk(["triton"], sensitivity=True, subs={"a": sub_t, "b": sub_c})) is True


def test_manifest_pure_triton_has_dem_and_mannings_only():
    """Q4 Phase-2: a pure-TRITON master's manifest carries exactly the TRITON DEM +
    Mannings panels (no SWMM), and the Mannings panel names its resolution. The
    pure-TRITON path needs no swmmio models, so the manifest is unit-testable directly."""
    md = _build_manifest_data(
        analysis_id="a",
        dem_bounds=(0.0, 0.0, 10.0, 10.0),
        hydro_model=None,
        hydraulics_model=None,
        bc_present=False,
        mannings_res=10.0,
    )
    names = [p["name"] for p in md["panels"]]
    assert names == ["triton_dem", "mannings"], names
    mannings = next(p for p in md["panels"] if p["name"] == "mannings")
    assert mannings["title"] == "Mannings (10.0m)"


def test_manifest_mannings_present_without_resolution_label():
    """The Manning's panel is present even when the resolution is unknown (falls back
    to a bare 'Mannings' title) — it never depends on a SWMM model being loaded."""
    md = _build_manifest_data(
        analysis_id="a",
        dem_bounds=(0.0, 0.0, 1.0, 1.0),
        hydro_model=None,
        hydraulics_model=None,
        bc_present=False,
    )
    mannings = [p for p in md["panels"] if p["name"] == "mannings"]
    assert len(mannings) == 1 and mannings[0]["title"] == "Mannings"
