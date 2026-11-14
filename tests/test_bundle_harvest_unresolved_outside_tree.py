"""An ABSENT source that lies OUTSIDE the analysis directory is
classified under its own key instead of being reported as a missing ``external/{name}`` file.

A sidecar records each out-of-tree source as a ``../``-relative path from the render-time
analysis directory. Harvested from a tree at another depth the path lands nowhere, and the
harvest used to record ``external/{basename}`` in ``declared_sources_absent`` although the
declared-input copy carries a file under that archive name. With the ``unresolved_outside``
keyword the source is recorded as its ``os.path.relpath`` spelling in its own list, with its
own aggregated warning and its own manifest key.

Arms: classification (keyword given), no-keyword (the must-not-fire twin, the behaviour before the
keyword existed), pre-populated list, and manifest. Every test writes only ``tmp_path``
files; none compiles or runs a solver.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from hhemt.bundle._emit import _harvest_sources, _write_bundle_manifest

_UNRESOLVED_REL = "../moved_away/matrix.csv"


def _tree(tmp_path: Path) -> tuple[Path, dict[str, list[Path]]]:
    """An analysis dir with one in-tree absent, one out-of-tree absent and one out-of-tree present source."""
    analysis_dir = tmp_path / "site" / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    existing = tmp_path / "site" / "inputs" / "events.nc"
    existing.parent.mkdir(parents=True)
    existing.write_text("x", encoding="utf-8")
    in_tree_absent = analysis_dir / "_status" / "_du.json"
    outside_absent = analysis_dir / _UNRESOLVED_REL  # the sidecar's ../ spelling, resolved here
    sources = {
        "r1": [in_tree_absent, outside_absent, existing],
        "r2": [outside_absent],  # a second renderer declares the same unresolved source
    }
    return analysis_dir, sources


def _harvest(analysis_dir: Path, sources: dict[str, list[Path]], **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _harvest_sources(sources, analysis_dir, **kw)
    return result, [w for w in caught if issubclass(w.category, UserWarning)]


def test_classification_arm_records_the_unresolved_source_under_its_own_list(tmp_path: Path) -> None:
    analysis_dir, sources = _tree(tmp_path)
    unresolved: list[str] = []
    (entries, absent, _shallow), caught = _harvest(analysis_dir, sources, unresolved_outside=unresolved)

    assert absent == ["_status/_du.json"], "only the in-tree absent source, under its in-tree name"
    assert unresolved == ["../moved_away/matrix.csv"], "once, as its relpath spelling, though two renderers declare it"
    assert not any("matrix.csv" in a for a in absent)
    assert "external/events.nc" in entries, "the existing out-of-tree source is still planned as external/{name}"
    assert "events.nc" not in " ".join(absent + unresolved)

    assert len(caught) == 2, [str(w.message) for w in caught]
    first, second = caught
    assert all(not str(w.filename).endswith(".py") and w.lineno == 0 for w in caught)
    assert "_status/_du.json" in str(first.message)
    assert "declared_sources_absent" in str(first.message)
    assert "matrix.csv" not in str(first.message)
    assert "declared_sources_unresolved_outside_tree" in str(second.message)
    assert "../moved_away/matrix.csv" in str(second.message)
    assert "_du.json" not in str(second.message)
    assert "declared_sources_absent" not in str(second.message)


def test_classification_arm_with_only_the_unresolved_source_issues_one_warning(tmp_path: Path) -> None:
    analysis_dir, sources = _tree(tmp_path)
    only = {"r": [analysis_dir / _UNRESOLVED_REL]}
    unresolved: list[str] = []
    (_entries, absent, _shallow), caught = _harvest(analysis_dir, only, unresolved_outside=unresolved)

    assert absent == []
    assert unresolved == ["../moved_away/matrix.csv"]
    assert len(caught) == 1
    assert "declared_sources_unresolved_outside_tree" in str(caught[0].message)


def test_given_but_nothing_unresolved_leaves_the_list_empty_and_warns_nothing(tmp_path: Path) -> None:
    """Differently-positioned satisfying input: the keyword is given and every out-of-tree source resolves."""
    analysis_dir, sources = _tree(tmp_path)
    present = {"r": [sources["r1"][2]]}
    unresolved: list[str] = []
    (_entries, absent, _shallow), caught = _harvest(analysis_dir, present, unresolved_outside=unresolved)
    assert absent == [] and unresolved == [] and caught == []


def test_no_keyword_arm_behaves_as_the_parent_commit(tmp_path: Path) -> None:
    """The must-not-fire twin: with no keyword the unresolved source is a declared_absent entry."""
    analysis_dir, sources = _tree(tmp_path)
    (entries, absent, shallow), caught = _harvest(analysis_dir, sources)

    assert absent == ["_status/_du.json", "external/matrix.csv"]
    assert "external/events.nc" in entries
    assert not any("events.nc" in a for a in absent)
    assert shallow == []
    assert len(caught) == 1, "1b.1's single aggregated warning only"
    assert "declared_sources_absent" in str(caught[0].message)
    assert "declared_sources_unresolved_outside_tree" not in str(caught[0].message)


def test_manifest_arm_writes_the_key_only_when_non_empty(tmp_path: Path) -> None:
    def manifest(name: str, **kw) -> dict:
        staging = tmp_path / name
        staging.mkdir()
        _write_bundle_manifest(
            staging,
            sources_by_renderer={"r": [Path("eda/a.zarr")]},
            analysis_id="aid",
            git_sha="deadbeef",
            **kw,
        )
        return json.loads((staging / "bundle_manifest.json").read_text())

    key = "declared_sources_unresolved_outside_tree"
    assert manifest("full", declared_sources_unresolved_outside_tree=["../moved_away/matrix.csv"])[key] == [
        "../moved_away/matrix.csv"
    ]
    assert key not in manifest("empty", declared_sources_unresolved_outside_tree=[])
    assert key not in manifest("none", declared_sources_unresolved_outside_tree=None)
    assert key not in manifest("default")


def test_a_pre_populated_list_keeps_its_entries_and_comes_back_sorted(tmp_path: Path) -> None:
    """The list is appended to, then de-duplicated and sorted; the warning names only what THIS harvest found."""
    analysis_dir, sources = _tree(tmp_path)
    unresolved = ["../prior_entry.csv", "../moved_away/matrix.csv"]  # one distinct, one a duplicate of the find
    (_entries, absent, _shallow), caught = _harvest(analysis_dir, sources, unresolved_outside=unresolved)

    assert unresolved == ["../moved_away/matrix.csv", "../prior_entry.csv"]
    assert absent == ["_status/_du.json"]
    assert len(caught) == 2
    second = str(caught[1].message)
    assert "../moved_away/matrix.csv" in second and "prior_entry" not in second
    assert "1 declared source(s)" in second, "the count is of this harvest's finds, not of the merged list"
