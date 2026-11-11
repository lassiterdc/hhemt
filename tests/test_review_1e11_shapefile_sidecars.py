"""Reviewer tests for the shapefile sidecar helper: its matching rules and the sites that go through it.

Problem this file solves: a shapefile is several files, and a bundle that carries the `.shp` alone cannot be read.
The coder's tests cover the carried set on one well-formed shapefile; these cover the helper's matching rules (letter
case, a sidecar that is not a file), the two plan sites on a single shapefile, a collision between two shapefiles of
one stem, and the by-reference arm. No solver; shapefiles are written with geopandas into `tmp_path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.bundle import _emit
from hhemt.exceptions import ConfigurationError
from tests.test_bundle_shapefile_sidecars import _write_shapefile
from tests.test_publish_deposit_paths import _world


def test_the_helper_matches_by_stem_case_insensitively_and_counts_only_files(tmp_path):
    """Property: `shapefile_sidecars` finds a sidecar whatever its letter case, and a sidecar that is a directory or a
    broken link is no sidecar (so a `.shp` with only such an entry is refused).

    Class: the required sidecar slot (`.shx`) filled by a file, an upper-case file, a directory, a broken link.
    Plausible wrong implementation: `exists()` in place of `is_file()`, which accepts a directory named `x.shx`.
    Must-fire control: the plain lower-case file is found, so a helper that finds nothing fails here.
    Kills it: matching by exact lower-case name, or by `exists()`. Passes under a second correct implementation that
    resolves the sidecar through `glob` with a case-folded pattern.
    """
    shp = _write_shapefile(tmp_path / "plain")
    assert {p.name for p in _emit.shapefile_sidecars(shp)} >= {"boundary.shx", "boundary.dbf"}

    upper = _write_shapefile(tmp_path / "upper")
    (upper.parent / "boundary.shx").rename(upper.parent / "boundary.SHX")
    assert "boundary.SHX" in {p.name for p in _emit.shapefile_sidecars(upper)}

    as_dir = _write_shapefile(tmp_path / "dir")
    (as_dir.parent / "boundary.shx").unlink()
    (as_dir.parent / "boundary.shx").mkdir()
    with pytest.raises(ConfigurationError, match=r"\.shx"):
        _emit.shapefile_sidecars(as_dir)

    broken = _write_shapefile(tmp_path / "link")
    (broken.parent / "boundary.shx").unlink()
    (broken.parent / "boundary.shx").symlink_to(tmp_path / "nowhere.shx")
    with pytest.raises(ConfigurationError, match=r"\.shx"):
        _emit.shapefile_sidecars(broken)


def test_a_shapefile_declared_twice_is_planned_once_through_both_sites_without_a_collision(tmp_path):
    """Property: one `.shp` that is both a declared input and a renderer source yields the declared-input copy and the
    harvest plan with the same member names for the `.shp` and every sidecar, and no collision refusal.

    Class: the same file reached through `_plan_declared_inputs` and `_harvest_sources` (the two sites that carry it).
    Plausible wrong implementation: a harvest that names the sidecars from a different root than the `.shp`, so the
    two sites disagree on the archive name. Must-fire control: the harvest plan holds at least the `.shp` and four
    sidecars, so an empty plan cannot satisfy the comparison.
    Kills it: a sidecar `rel` computed from the source path instead of the `.shp`'s own `rel`.
    """
    shp = _write_shapefile(tmp_path / "src")
    world = _world(tmp_path, system_over={"watershed_gis_polygon": str(shp)})
    planned, _deposits = _emit._plan_declared_inputs(world.analysis, None)
    declared = {Path(i.rel).name for i in planned if Path(i.rel).stem == "boundary"}
    analysis_dir = Path(world.analysis.analysis_paths.analysis_dir)
    entries, absent, _shallow = _emit._harvest_sources({"plot": [shp, shp]}, analysis_dir)
    harvested = {Path(k).name for k in entries if Path(k).stem == "boundary"}
    assert not absent
    assert harvested >= {"boundary.shp", "boundary.shx", "boundary.dbf", "boundary.prj", "boundary.cpg"}
    assert declared == harvested


def test_two_shapefiles_of_one_stem_in_different_directories_refuse_rather_than_mix_sidecars(tmp_path):
    """Property: two different shapefiles that share a file name and would land under one archive name are refused by
    the harvest plan, and the refusal is not skipped by the sidecars (a `.dbf` of the second is never paired with the
    first's `.shp`).

    Class: two directories, same stem, different content in every file. Plausible wrong implementation: a plan that
    keeps the first `.shp` and silently adds only the sidecars the first lacks. Must-fire control: the same file
    given twice is accepted (one entry), so the refusal is about differing content.
    Not established: two byte-identical `.shp` files with different `.dbf` files are refused at the `.dbf`, which this
    test reads as the intended outcome but does not assert beyond the first collision.
    """
    first = _write_shapefile(tmp_path / "a")
    second = _write_shapefile(tmp_path / "b", name="boundary")
    import geopandas as gpd
    from shapely.geometry import Point

    gpd.GeoDataFrame({"id": [9]}, geometry=[Point(5, 5)], crs=32147).to_file(second)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    _emit._harvest_sources({"plot": [first, first]}, analysis_dir)
    with pytest.raises(ConfigurationError, match="two different files"):
        _emit._harvest_sources({"plot": [first], "other": [second]}, analysis_dir)


def test_a_by_reference_shapefile_pins_the_shp_alone_and_does_not_refuse_the_missing_sidecars(tmp_path):
    """Property: an EXCLUDED (by-reference) `.shp` produces one `input_deposit` block and no carried sidecar, and a
    missing sidecar does not refuse the emit (the producer deposits that input elsewhere).

    Class: the by-reference arm over a `.shp` with no `.shx`. Plausible wrong implementation: running the sidecar
    helper before the exclusion test, which would refuse an input the bundle does not carry. Must-fire control: the
    same `.shp` NOT excluded is refused, so the helper is live on this input.
    Kills it: moving the helper above the `excluded` branch.
    """
    from hhemt.config.bundle_exclude import BundleExcludeConfig

    shp = _write_shapefile(tmp_path / "src")
    (shp.parent / "boundary.shx").unlink()
    world = _world(tmp_path, system_over={"watershed_gis_polygon": str(shp)})
    with pytest.raises(ConfigurationError, match=r"\.shx"):
        _emit._plan_declared_inputs(world.analysis, None)
    exclude = BundleExcludeConfig.model_validate({"exclusions": {"watershed_gis_polygon": {"citation": "c"}}})
    planned, deposits = _emit._plan_declared_inputs(world.analysis, exclude)
    assert len(deposits) == 1 and deposits[0]["relpath"].endswith("boundary.shp")
    assert not [p for p in planned if Path(p.rel).stem == "boundary"]
