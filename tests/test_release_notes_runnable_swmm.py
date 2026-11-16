"""Pins the v0.3.0 release entry: the runnable-SWMM feature is the headline and the three version copies agree.

The entry in HISTORY.md for v0.3.0 highlights ONE feature, a plain runnable SWMM model driving the
coupled model, with the observation points and initial-condition rasters that came with it. The
citation, release-venue and run-directory-reaper side items of the same run are not release notes
and stay out of it (developer ruling, approved shape DR-12 and package 3a). Every test reads text
files only, so each runs well under 0.15 s.
"""

from __future__ import annotations

import re
from pathlib import Path

import scripts.check_release_metadata as gate  # repo root is on sys.path under pytest

_REPO = Path(__file__).resolve().parent.parent
_RELEASE = "0.3.0"


def _entry(version: str) -> str:
    """The HISTORY.md body under `## v{version} (...)`, up to the next release heading."""
    text = (_REPO / "HISTORY.md").read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^## v{re.escape(version)} \(\d{{4}}-\d{{2}}-\d{{2}}\)[ \t]*\n(.*?)(?=^## v|\Z)", text)
    assert match is not None, f"HISTORY.md has no '## v{version} (YYYY-MM-DD)' entry"
    return match.group(1)


def test_newest_history_heading_version_is_the_pyproject_and_citation_version():
    """REGRESSION: a bump that edits one of the three copies leaves the release metadata disagreeing."""
    headings = gate.HISTORY_HEADING.findall((_REPO / "HISTORY.md").read_text(encoding="utf-8"))
    py_version, cff_version, _cff_date, _history_date = gate._read(_REPO)
    assert headings[0][0] == py_version == cff_version


def test_release_metadata_gate_passes_on_the_stamped_date():
    """The pre-tag gate agrees with the stamped entry: one date in CITATION.cff and in the heading."""
    headings = gate.HISTORY_HEADING.findall((_REPO / "HISTORY.md").read_text(encoding="utf-8"))
    assert gate.check(_REPO, headings[0][1]) == []


def test_citation_declares_one_version_and_one_release_date():
    """The gate refuses more than one hit, so a references entry that grew a column-0 key would break the tag."""
    text = (_REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert len(gate.CFF_VERSION.findall(text)) == 1
    assert len(gate.CFF_DATE.findall(text)) == 1


def test_release_entry_names_the_runnable_swmm_feature():
    entry = _entry(_RELEASE)
    for token in ("SWMM_model", "swmm_outfalls_to_couple"):
        assert token in entry, token


def test_release_entry_names_the_retired_templates_and_the_new_required_settings():
    entry = _entry(_RELEASE)
    for token in (
        "SWMM_hydraulics",
        "SWMM_hydrology",
        "SWMM_full",
        "triton_swmm_configuration_template",
        "TRITON_courant_number",
        "TRITON_hextra_m",
    ):
        assert token in entry, token


def test_release_entry_names_the_observation_and_initial_condition_additions():
    entry = _entry(_RELEASE)
    for token in (
        "observation_points",
        "TRITON_initial_depth_raster",
        "TRITON_initial_qx_raster",
        "TRITON_initial_qy_raster",
    ):
        assert token in entry, token


def test_release_entry_keeps_the_side_items_out_of_the_headline():
    """The works-cited, release-venue and reaper items are not release notes (developer ruling)."""
    entry = _entry(_RELEASE).lower()
    for token in ("reaper", "works cited", "cited", "doi", "zenodo", "venue", "citation"):
        assert token not in entry, token


def test_release_entry_states_the_outfall_decisions_the_series_files_and_the_renamed_boundary_keys():
    """The outfall and series paragraph names the decision lists, the thread setting, the new keys and layout 34."""
    entry = _entry(_RELEASE)
    for token in (
        "swmm_outfalls_to_keep",
        "swmm_threads",
        "toggle_external_boundaries",
        "external_boundary_gis",
        "layout version 34",
        "stated decision",
    ):
        assert token in entry, token


def test_release_entry_no_longer_describes_the_rain_and_tide_series_as_rewritten():
    """The first paragraph's phrase 'the rain and tide time series' went with the replaced paragraph."""
    entry = _entry(_RELEASE)
    assert "rain and tide time series" not in entry
    # Control: the matcher fires on the sentence the replaced paragraph carried.
    replaced = "rewrites only the simulation window, the rain gauges and the rain and tide time series"
    assert "rain and tide time series" in replaced
