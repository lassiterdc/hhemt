"""Reader-facing text of the map pages: the code facts their wording states.

Package 3a rewrites the text of the map pages one page per chunk. Each chunk adds the tests
of its page here (shape binding B3): a test asserts the code fact the text states, located by a
stable anchor, never by comparing a whole sentence, writes no typed count, and carries a
must-fire control. Nothing here compiles or runs a solver.

Chunk 3a.1, the P01 System map page (shape item I26, section 7 rows for P01).

The P01 page needs a DEM raster and SWMM models, so it is not a stub-rendered page kind in
`tests/test_caption_page_kinds.py`; the caption/page agreement is checked here against the panel
headings the renderer prints (shape B4), and the reviewer checks the rest against the matrix rows.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pandas as pd
import pytest

from hhemt.report_renderers import system_overview
from hhemt.report_renderers.system_overview import (
    DEM_PANEL_TITLE,
    HYDRAULICS_PANEL_TITLE,
    HYDROLOGY_PANEL_TITLE,
    MANNINGS_PANEL_TITLE,
    _build_manifest_data,
)
from tests.test_caption_page_kinds import CAPTIONS, bold_names

_RETIRED_PHRASES = re.compile(
    r"peak water level|sub-analys[ei]s|sensitivity master|model (?:type|arm)|sentinel|sidecar|compute-config",
    re.IGNORECASE,
)
_ALWAYS_PRESENT = {DEM_PANEL_TITLE, MANNINGS_PANEL_TITLE}


def _caption() -> str:
    return (CAPTIONS / "system_map.rst").read_text("utf-8")


def _coupled_manifest() -> dict:
    inp_h = types.SimpleNamespace(
        polygons=pd.DataFrame({"x": [0.0, 1.0]}, index=["s1", "s1"]),
        subcatchments=pd.DataFrame({"a": [1]}),
    )
    inp_x = types.SimpleNamespace(
        junctions=pd.DataFrame({"a": [1]}), outfalls=pd.DataFrame({"a": [1]}), conduits=pd.DataFrame({"a": [1]})
    )
    return _build_manifest_data(
        analysis_id="a",
        dem_bounds=(0.0, 0.0, 10.0, 10.0),
        hydro_model=types.SimpleNamespace(inp=inp_h),
        hydraulics_model=types.SimpleNamespace(inp=inp_x),
        bc_present=True,
        mannings_res=10.0,
    )


def test_the_caption_bolds_only_panels_every_page_prints_and_the_page_prints_them() -> None:
    """The caption names the two always-present panels, as the page headings print them.

    Input class: the `system_map.rst` source against the headings of a coupled and a pure-TRITON
    manifest. Required because brief rule 9 reads each bold span as a page element name and shape K9
    leaves the conditional SWMM panels to their own headings. Killed by bolding Hydrology or Hydraulics
    in the caption, or by a heading that no longer carries the bolded name ("Mannings" printed against
    a caption that bolds "Manning's n"). A second correct implementation words the panels differently
    while the bold names equal the printed headings.
    """
    names = set(bold_names(_caption()))
    coupled_titles = [p["title"] for p in _coupled_manifest()["panels"]]

    assert names == _ALWAYS_PRESENT
    for name in names:
        assert any(t.startswith(name) for t in coupled_titles)


@pytest.mark.parametrize(
    "old_caption", ["The **Hydrology** panel (x) and a **Hydraulics** panel (y)", "The **Mannings** panel"]
)
def test_the_caption_bold_check_fires_on_the_old_wording(old_caption: str) -> None:
    """Must-fire control for the test above: the old bold spans are not the always-present panel names."""
    assert set(bold_names(old_caption)) != _ALWAYS_PRESENT


def test_the_caption_cuts_the_conditional_panels_the_storm_tide_clause_and_the_deprecated_override() -> None:
    """The caption keeps the opening sentence, the DEM colours, the Manning's n panel and the CRS.

    Input class: the `system_map.rst` source. Required because shape rows P01 (K9, K15) cut the
    conditional panels and the line's presence from the caption and the deprecated override from the
    CRS sentence. Killed by restoring any of them. A second correct implementation words the kept
    sentences differently and passes.
    """
    text = _caption()

    assert text.startswith("Model inputs for **")
    assert "horizontal_epsg" in text
    for cut in ("deprecated", "target_epsg", "Hydrology", "Hydraulics", "storm-tide", "When the analysis includes"):
        assert cut not in text


@pytest.mark.parametrize("cut", ["deprecated", "target_epsg", "Hydrology", "storm-tide"])
def test_the_caption_cut_check_fires_on_the_old_caption(cut: str) -> None:
    """Must-fire control: the old caption text contains each cut phrase."""
    old = (
        "the storm-tide boundary-condition line is drawn when the analysis has one. a Hydrology panel ... "
        "unless the deprecated ``report.system_map.target_epsg`` override is set."
    )
    assert cut in old


def test_a_conditional_panel_heading_carries_one_line_saying_what_the_panel_draws() -> None:
    """Each SWMM panel heading is one line: its name, a colon, and what it draws (shape K9).

    Input class: the manifest of a coupled analysis. Required because the caption no longer describes
    the conditional panels. Killed by the bare heading "Hydrology" or by a heading broken over lines.
    A second correct implementation words the line differently and passes.
    """
    titles = {p["name"]: p["title"] for p in _coupled_manifest()["panels"]}

    for key, base in (("hydrology", "Hydrology"), ("hydraulics", "Hydraulics")):
        title = titles[key]
        assert title.startswith(f"{base}: ") and len(title) > len(base) + 3
        assert "<br>" not in title and "\n" not in title and "—" not in title
        assert _RETIRED_PHRASES.search(title) is None
    assert titles["hydrology"] == HYDROLOGY_PANEL_TITLE and titles["hydraulics"] == HYDRAULICS_PANEL_TITLE


@pytest.mark.parametrize("bare", ["Hydrology", "Hydraulics"])
def test_the_heading_line_check_fires_on_a_bare_heading(bare: str) -> None:
    """Must-fire control: a bare heading does not satisfy the heading-line condition."""
    assert not (bare.startswith(f"{bare}: ") and len(bare) > len(bare) + 3)


def test_the_manning_panel_heading_matches_the_caption_and_keeps_its_resolution() -> None:
    """The Manning's n heading prints the name the caption bolds, with the resolution when known."""
    md = _coupled_manifest()
    mannings = next(p for p in md["panels"] if p["name"] == "mannings")

    assert mannings["title"] == f"{MANNINGS_PANEL_TITLE} (10.0m)"
    assert "Mannings" not in mannings["title"]


@pytest.mark.parametrize("marker", ["<title>System overview", 'f"Easting (m)'])
def test_the_page_title_and_the_dem_axis_title_carry_no_em_dash(marker: str) -> None:
    """The document title and the DEM x-axis title join their parts without an em dash (brief rule 8).

    Input class: the source lines that build the two strings. Killed by restoring the dash. A second
    correct implementation uses any other separator.
    """
    lines = [ln for ln in Path(system_overview.__file__).read_text("utf-8").splitlines() if marker in ln]

    assert lines
    assert all("—" not in ln for ln in lines)


def test_the_em_dash_line_check_fires_on_the_old_line() -> None:
    """Must-fire control: the old title line carries an em dash."""
    assert "—" in "<title>System overview — {_title_panels}</title>"


# ---- 3a.1 round 2: the static route prints the same panel name and colour-bar label ----------


def _static_mannings_axes():
    """A matplotlib axes holding a raster with a colour bar, labelled as the shared plotter labels it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    fig, ax = plt.subplots()
    da = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    da.plot(ax=ax, x="x", y="y", add_colorbar=True)
    ax.set_title("Mannings (10.0m)")
    next(a for a in ax.collections).colorbar.set_label("mannings")
    return fig, ax


def _colorbar_labels(ax) -> list[str]:
    return [a.colorbar.ax.get_ylabel() for a in (*ax.images, *ax.collections) if getattr(a, "colorbar", None)]


def test_the_static_manning_panel_prints_the_heading_the_interactive_route_prints() -> None:
    """The static route's panel title equals the manifest's, with the resolution when known.

    Input class: a static Manning's n axes as the shared plotter leaves it ("Mannings (10.0m)"),
    labelled by the renderer's helper. Required because the caption bolds **Manning's n** under both
    routes (brief rule 9). Killed by leaving the plotter's title in place. A second correct
    implementation sets the same string by any means.
    """
    from hhemt.report_renderers.system_overview import _label_static_mannings_panel

    fig, ax = _static_mannings_axes()
    _label_static_mannings_panel(ax, 10.0)
    manifest_title = next(p["title"] for p in _coupled_manifest()["panels"] if p["name"] == "mannings")

    assert ax.get_title() == manifest_title == "Manning's n (10.0m)"
    fig.clf()


def test_the_static_manning_panel_heading_without_a_resolution_is_the_bare_name() -> None:
    from hhemt.report_renderers.system_overview import _label_static_mannings_panel

    fig, ax = _static_mannings_axes()
    _label_static_mannings_panel(ax, None)

    assert ax.get_title() == MANNINGS_PANEL_TITLE
    fig.clf()


def test_the_static_manning_colour_bar_is_labelled_with_the_panel_name() -> None:
    """The colour-bar label reads "Manning's n", not the plotter's lowercase "mannings" (brief rule 2).

    Input class: the same axes. Killed by leaving the plotter's label. A second correct
    implementation words the label with the unit if one exists and keeps the name.
    """
    from hhemt.report_renderers.system_overview import _label_static_mannings_panel

    fig, ax = _static_mannings_axes()
    _label_static_mannings_panel(ax, 10.0)

    assert _colorbar_labels(ax) == [MANNINGS_PANEL_TITLE]
    fig.clf()


def test_the_static_label_check_fires_on_the_plotters_own_labels() -> None:
    """Must-fire control for the three tests above: the plotter's labels fail them.

    The unlabelled axes carry the title "Mannings (10.0m)" and the colour-bar label "mannings", and
    the check reads both, so a reverted helper is caught.
    """
    fig, ax = _static_mannings_axes()

    assert ax.get_title() != "Manning's n (10.0m)"
    assert _colorbar_labels(ax) != [MANNINGS_PANEL_TITLE]
    fig.clf()


def test_the_static_branch_calls_the_label_helper_after_the_shared_plotter() -> None:
    """The static branch applies the helper after `processed_mannings` returns, or the plotter's labels win.

    Input class: the renderer source. Killed by dropping the call or by moving it before the plotter.
    A second correct implementation relabels in any way that runs after the plotter.
    """
    lines = Path(system_overview.__file__).read_text("utf-8").splitlines()
    plotter = [i for i, ln in enumerate(lines) if ".processed_mannings(ax=ax_mannings)" in ln]
    helper = [i for i, ln in enumerate(lines) if "_label_static_mannings_panel(ax_mannings" in ln]

    assert len(plotter) == 1 and len(helper) == 1
    assert helper[0] > plotter[0]


def test_the_label_helper_labels_the_colour_bar_the_real_static_plotter_creates() -> None:
    """The helper finds and relabels the colour bar that `processed_mannings` really creates.

    Input class: the shared plotter run on a tiny raster (not a stand-in), then the helper. Required
    because the helper silently skips an axes with no colour bar attached to an image or collection, so
    a plotter that changes how it attaches the bar would leave the lowercase label with every other test
    green. Anchored on the printed title and exactly one colour-bar label, both properties of any correct
    plotter and helper. Killed by dropping the helper, or by a plotter whose bar the helper cannot find.
    A second correct implementation relabels by another route.
    """
    import types

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from hhemt.plot_system import TRITONSWMM_system_plotting
    from hhemt.report_renderers.system_overview import _label_static_mannings_panel

    rds = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("y", "x"), coords={"y": [0.0, 1.0], "x": [0.0, 1.0]})
    cfg = types.SimpleNamespace(watershed_gis_polygon=None, target_dem_resolution=10.0)
    plotter = TRITONSWMM_system_plotting(types.SimpleNamespace(mannings_rds=rds, cfg_system=cfg, sys_paths=None))
    fig, ax = plt.subplots()
    plotter.processed_mannings(ax=ax)
    before = _colorbar_labels(ax)
    _label_static_mannings_panel(ax, 10.0)

    assert before and before != [MANNINGS_PANEL_TITLE]
    assert ax.get_title() == "Manning's n (10.0m)"
    assert _colorbar_labels(ax) == [MANNINGS_PANEL_TITLE]
    fig.clf()


# ---- 3a.2: the P02 Event page, P03 Per-member peak flood depth, P04 Per-member conduit flow ----

_REPORT_RENDERERS = Path(system_overview.__file__).parent


def _caption_of(name: str) -> str:
    return (CAPTIONS / f"{name}.rst").read_text("utf-8")


def _renderer_source(module: str) -> str:
    return (_REPORT_RENDERERS / f"{module}.py").read_text("utf-8")


def _depth_definition(text: str) -> str:
    match = re.search(r"\*\*Peak flood depth\*\* is [^.]*\.", text)
    assert match is not None
    return match.group(0)


def test_the_peak_flood_depth_definition_is_one_wording_on_both_pages_and_names_the_solver_maximum() -> None:
    """P02 and P03 define peak flood depth in the same words, and the words are true of the plotted variable.

    Input class: the two captions, the renderer source and the solver-output mapping. Required because
    the per-member caption said the depth was "taken by the toolkit's post-processing" while the
    plotted variable is the TRITON solver's own maximum: `TRITON_RAW_VAR_BY_PREFIX["MH"]` is
    `max_wlevel_m` and the per-member renderer reads `ds["max_wlevel_m"]`. Killed by restoring the
    post-processing wording, by naming a different mapped variable, or by the two pages diverging. A
    second correct implementation words the definition differently while the two captions agree and
    state a solver maximum of the depth.
    """
    from hhemt.process_simulation import TRITON_RAW_VAR_BY_PREFIX

    event, member = (
        _depth_definition(_caption_of("per_sim_event_page")),
        _depth_definition(_caption_of("per_sim_peak_flood_depth")),
    )

    assert TRITON_RAW_VAR_BY_PREFIX["MH"] == "max_wlevel_m"
    assert 'ds["max_wlevel_m"]' in _renderer_source("per_sim_peak_flood_depth")
    assert event == member
    assert "maximum water depth" in member and "TRITON solver" in member
    assert "post-processing" not in member


@pytest.mark.parametrize(
    ("event", "member"),
    [
        (
            "**Peak flood depth** is the per-cell maximum water depth "
            "across all simulation timesteps, computed by the TRITON solver.",
            "**Peak flood depth** is the per-cell maximum of TRITON's water depth "
            "across all output timesteps, taken by the toolkit's post-processing.",
        )
    ],
)
def test_the_definition_check_fires_on_the_two_old_wordings(event: str, member: str) -> None:
    """Must-fire control for the test above: the old pair differs and the old per-member text names post-processing."""
    assert event != member and "post-processing" in member


def test_the_mapping_check_fires_on_a_different_variable() -> None:
    """Must-fire control: a mapping that sends MH elsewhere fails the equality the test asserts."""
    assert {"MH": "max_flow_cms"}["MH"] != "max_wlevel_m"


_CUT_PHRASES = {
    "per_sim_event_page": ["Scroll", "four sections", "three-model", "sea wall", "same forcing", "System Information"],
    "per_sim_peak_flood_depth": ["may be a bounding box", "System Information", "Three-panel"],
    "per_sim_conduit_flow": ["bbox", "SWMM conduit flow panel"],
}
_EMPHASIS_CAPS = re.compile(r"\b(?:NOT|ALL|ONLY|NEVER)\b")
_HEDGE_OR_WHICH = re.compile(r"\b(?:may|might|typically|usually|generally|likely|which)\b", re.IGNORECASE)


@pytest.mark.parametrize("name", sorted(_CUT_PHRASES))
def test_a_per_event_caption_cuts_its_rows_and_carries_no_hedge_or_capital_emphasis(name: str) -> None:
    """Each of the three captions drops the clauses its shape rows cut and keeps the prose rules.

    Input class: the three captions. Required because shape rows P02 to P04 cut the scroll and order
    text, the hand count of sections, the clause true of one system, the "same forcing" clause, the
    sidebar pointer and the hedged bounding-box clause (K9, K15), and brief rules 3 and 8 bar hedges,
    `which` and capitals for emphasis (the style test's four-letter floor misses `NOT` and `ALL`).
    Killed by restoring any of them. A second correct implementation words the kept sentences
    differently and passes.
    """
    text = _caption_of(name)

    for phrase in _CUT_PHRASES[name]:
        assert phrase not in text
    assert _EMPHASIS_CAPS.search(text) is None
    assert _HEDGE_OR_WHICH.search(text) is None
    assert _RETIRED_PHRASES.search(text) is None


@pytest.mark.parametrize("old", ["is NOT masked", "ALL events", "which may be a bounding box", "might differ"])
def test_the_prose_rule_matchers_fire_on_the_old_wording(old: str) -> None:
    """Must-fire control for the test above: each matcher fires on a known-bad input."""
    assert _EMPHASIS_CAPS.search(old) or _HEDGE_OR_WHICH.search(old)


def test_the_member_pages_name_the_member_and_the_event_and_the_event_page_names_the_event() -> None:
    """The first sentences use the term-table words: `member` and `event` (shape rows P02 to P04)."""
    for name, needles in (
        ("per_sim_event_page", ["event **{{ snakemake.wildcards.event_id }}**"]),
        (
            "per_sim_peak_flood_depth",
            ["member **{{ snakemake.wildcards.member_id }}**", "event **{{ snakemake.wildcards.event_id }}**"],
        ),
        (
            "per_sim_conduit_flow",
            ["member **{{ snakemake.wildcards.member_id }}**", "event **{{ snakemake.wildcards.event_id }}**"],
        ),
    ):
        first = _caption_of(name).splitlines()[0]
        assert all(n in first for n in needles)
        assert "scenario" not in first


def test_every_caption_bold_name_is_printed_by_a_renderer_of_the_page() -> None:
    """Each bold span is a panel heading or the legend label that the page prints (brief rule 9, both routes).

    Input class: the three captions against the quoted strings of the renderer sources and
    `WATERSHED_LEGEND_LABEL`. The headings are the same strings on the static and the interactive route
    in these renderers (titles are set once per panel). Killed by a bold span no renderer prints, such as
    the old "overlay only", "clipped to the watershed" or model names. A second correct implementation
    names the panels differently while the captions and the headings agree.
    """
    from hhemt.figure_panels import WATERSHED_LEGEND_LABEL

    sources = {
        "per_sim_event_page": _renderer_source("per_sim_peak_flood_depth") + _renderer_source("per_sim_conduit_flow"),
        "per_sim_peak_flood_depth": _renderer_source("per_sim_peak_flood_depth"),
        "per_sim_conduit_flow": _renderer_source("per_sim_conduit_flow"),
    }
    for name, source in sources.items():
        for bold in bold_names(_caption_of(name)):
            assert f'"{bold}"' in source or bold == WATERSHED_LEGEND_LABEL, (name, bold)


@pytest.mark.parametrize("bold", ["overlay only", "clipped to the watershed", "TRITON-SWMM"])
def test_the_bold_name_check_fires_on_the_old_bold_spans(bold: str) -> None:
    """Must-fire control for the test above: no renderer prints these old bold spans."""
    source = _renderer_source("per_sim_peak_flood_depth") + _renderer_source("per_sim_conduit_flow")
    assert f'"{bold}"' not in source


def test_the_watershed_legend_label_names_the_supplied_polygon_in_the_pages_and_the_captions() -> None:
    """The legend entry reads "watershed polygon" everywhere the pages set it (shape K15).

    Input class: the shared constant, the two renderers' trace names and the captions. Required because
    the entry read "watershed extent (bbox)" whatever geometry the system supplied. Killed by a literal
    old label in a renderer or a caption that names the old one. A second correct implementation picks
    another label that does not say "bbox" while the constant, the traces and the captions agree.
    """
    from hhemt.figure_panels import WATERSHED_LEGEND_LABEL

    assert WATERSHED_LEGEND_LABEL == "watershed polygon"
    for module in ("per_sim_peak_flood_depth", "per_sim_conduit_flow"):
        assert "watershed extent (bbox)" not in _renderer_source(module)
        assert "name=WATERSHED_LEGEND_LABEL," in _renderer_source(module)
    for name in ("per_sim_peak_flood_depth", "per_sim_conduit_flow"):
        assert f"**{WATERSHED_LEGEND_LABEL}**" in _caption_of(name)
        assert "bbox" not in _caption_of(name)


def test_the_legend_label_check_fires_on_the_old_label() -> None:
    """Must-fire control: the old label carries "bbox" and differs from the new constant."""
    assert "bbox" in "watershed extent (bbox)" and "watershed extent (bbox)" != "watershed polygon"


def _sentences_with(text: str, needle: str) -> list[str]:
    """The sentences of a caption (split after a period, a colon or a line break) that contain the needle."""
    return [s.strip() for s in re.split(r"(?<=\.)\s+|\n+", text) if needle in s]


@pytest.mark.parametrize("needle", ["**Max / full flow**", "**Peak flow**", "overlay only"])
def test_the_conduit_sentences_are_one_wording_on_the_event_page_and_the_member_conduit_page(needle: str) -> None:
    """P02 and P04 state each conduit sentence in the same words (shape rows P02 and P04: one wording).

    Input class: the sentence of each caption holding the needle (the ratio definition, the peak flow
    definition, the overlay-only statement). Required because the P02 row keeps these three "one wording
    shared with P04" and a reader meets both pages. Killed by the P02 overlay sentence that drops
    "It is" and the comparison with the depth maps while P04 keeps them. A second correct
    implementation words each sentence differently while both captions carry the same one.
    """
    event = _sentences_with(_caption_of("per_sim_event_page"), needle)
    member = _sentences_with(_caption_of("per_sim_conduit_flow"), needle)

    assert event and member
    assert event == member


def test_the_wording_check_separates_a_shared_sentence_from_a_diverged_one() -> None:
    """Must-fire control for the test above: the extractor returns unequal lists for two diverged texts."""
    a = "The watershed boundary is an overlay only: values are not masked."
    b = "It is an overlay only: unlike the depth maps, values are not masked."

    assert _sentences_with(a, "overlay only") != _sentences_with(b, "overlay only")
    assert _sentences_with(a, "overlay only") == _sentences_with(a + " Other.", "overlay only")
