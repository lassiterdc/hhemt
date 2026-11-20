"""Reviewer probes for the system-overview present-DEM change (1118-tk4).

Invariant: every raster-dependent input the overview passes to its figure builder (the DEM sentinel heights that
mask building and outside-watershed cells, and the vertical CRS) describes the SAME system whose DEM is drawn.

Violating input: master config carries no sentinel heights, the only built resolution belongs to a member system
whose config carries them. Satisfying input in a different correct state: the master's own pair is built, so master
and drawn system coincide.
"""

from hhemt.report_renderers import system_overview
from tests.test_system_overview_present_dem import (  # noqa: F401  (autouse fixture is imported by name)
    _analysis,
    _built,
    _no_kaleido_no_build_stamp,
)


def _capture_plotly_kwargs(monkeypatch) -> dict:
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(system_overview, "_render_plotly_branch", _fake)
    return captured


def test_sentinel_heights_follow_the_drawn_system_when_it_is_not_the_master(tmp_path, monkeypatch):
    """Drawn 3.5 m member system owns building height 80; the master (1.1 m, unbuilt) owns none."""
    _built(tmp_path, 3.5)
    analysis = _analysis(tmp_path, master_resolution=1.1, member_resolutions=(3.5,))
    member = analysis.sensitivity.unique_system_targets[0].system
    member.cfg_system.dem_building_height = 80.0
    member.cfg_system.dem_outside_watershed_height = 9999.0
    captured = _capture_plotly_kwargs(monkeypatch)

    system_overview.render(
        analysis, __import__("hhemt.config.report", fromlist=["report_config"]).report_config(), tmp_path / "o.html"
    )

    assert captured["dem_building_height"] == 80.0, captured["dem_building_height"]
    assert captured["dem_outside_watershed_height"] == 9999.0, captured["dem_outside_watershed_height"]


def test_sentinel_heights_unchanged_when_master_pair_is_built(tmp_path, monkeypatch):
    """Control: master pair built, master heights are passed (the pre-change behaviour)."""
    _built(tmp_path, 1.1, 3.5)
    analysis = _analysis(tmp_path, master_resolution=1.1, member_resolutions=(3.5,))
    analysis._system.cfg_system.dem_building_height = 70.0
    captured = _capture_plotly_kwargs(monkeypatch)

    system_overview.render(
        analysis, __import__("hhemt.config.report", fromlist=["report_config"]).report_config(), tmp_path / "o.html"
    )

    assert captured["dem_building_height"] == 70.0, captured["dem_building_height"]
