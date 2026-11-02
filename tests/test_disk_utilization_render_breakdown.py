"""The Disk Utilization card must list files written after the sentinel was computed.

The analysis sentinel is stamped when processing finishes; the report plots are
written afterwards. A card built only from the recorded breakdown therefore has
no row for `plots/`, and its Total understates the directory. On a page
regenerated from a bundle the root holds harvested files only, so the recorded
rows stand and nothing is measured.

Rows are read from the table data embedded in the page, not matched as text.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hhemt.du_sentinels import write_du_sentinel
from hhemt.report_renderers import disk_utilization
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

_MIB = 1024 * 1024


def _render(analysis_dir: Path, tmp_path: Path) -> str:
    out = tmp_path / "disk_utilization.html"
    analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=analysis_dir))
    disk_utilization.render(analysis, None, out)
    return out.read_text()


def _rows(html: str) -> dict[str, float]:
    """`{Scope: Size (MiB)}` from the `data` array of the Tabulator options in the page."""
    marker = 'const tableOptions = {"data": '
    data, _end = json.JSONDecoder().raw_decode(html[html.index(marker) + len(marker) :])
    return {row["Scope"]: row["Size (MiB)"] for row in data}


def _write(path: Path, n_mib: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n_mib * _MIB)


def _write_sentinel(analysis_dir: Path, breakdown: dict[str, int]) -> None:
    write_du_sentinel(
        analysis_dir / "_status" / "_du.json",
        disk_utilization_bytes=sum(breakdown.values()),
        scope="analysis",
        sub_path_breakdown=breakdown,
    )


def _live_analysis(tmp_path: Path) -> Path:
    """Sentinel records sims 3, members 3, zarr 1 MiB; the directory now differs."""
    analysis_dir = tmp_path / "analysis"
    _write(analysis_dir / "sims" / "evt0" / "f.bin", 8)  # recorded 3: the recorded total stands
    _write(analysis_dir / "members" / "member_1" / "f.bin", 7)  # recorded 3: the recorded total stands
    _write(analysis_dir / "analysis_datatree.zarr" / "c", 1)
    _write_sentinel(
        analysis_dir,
        {"sims": 3 * _MIB, "members": 3 * _MIB, "analysis_datatree.zarr": _MIB},
    )
    # Written after the sentinel, in the way the report plots are:
    _write(analysis_dir / "plots" / "fig.png", 2)
    _write(analysis_dir / "top.bin", 1)  # a top-level file
    _write(analysis_dir / "_status_old" / "f", 5)  # a `_status*` entry other than `_status`
    _write(analysis_dir / "_test" / "f", 4)  # the ephemeral smoke-test tree
    (analysis_dir / "empty_dir").mkdir()  # holds no data
    (analysis_dir / "empty_file").write_bytes(b"")  # holds no data
    return analysis_dir


def test_live_root_rows_follow_the_rollups_skips(tmp_path: Path) -> None:
    rows = _rows(_render(_live_analysis(tmp_path), tmp_path))

    assert rows == {
        "sims": 3.0,
        "members": 3.0,
        "analysis_datatree.zarr": 1.0,
        "plots": 2.0,
        "top.bin": 1.0,
    }


def test_plots_directory_created_after_sentinel_appears_with_its_measured_size(tmp_path: Path) -> None:
    rows = _rows(_render(_live_analysis(tmp_path), tmp_path))

    assert rows["plots"] == 2.0


def test_child_scope_totals_stay_those_recorded_in_the_sentinel(tmp_path: Path) -> None:
    rows = _rows(_render(_live_analysis(tmp_path), tmp_path))

    assert (rows["sims"], rows["members"]) == (3.0, 3.0)


def test_status_entry_ephemeral_tree_and_top_level_file_each_handled_as_the_rollup_does(tmp_path: Path) -> None:
    rows = _rows(_render(_live_analysis(tmp_path), tmp_path))

    assert "_status_old" not in rows
    assert "_test" not in rows
    assert rows["top.bin"] == 1.0


def test_entries_holding_no_data_get_no_row(tmp_path: Path) -> None:
    rows = _rows(_render(_live_analysis(tmp_path), tmp_path))

    assert "empty_dir" not in rows
    assert "empty_file" not in rows


def test_sentinel_computed_at_is_printed(tmp_path: Path) -> None:
    analysis_dir = _live_analysis(tmp_path)
    computed_at = json.loads((analysis_dir / "_status" / "_du.json").read_text())["computed_at"]

    assert computed_at in _render(analysis_dir, tmp_path)


def _bundle_root(tmp_path: Path) -> Path:
    """A bundle-like root: metadata-only consolidated tree, a manifest, a plots dir."""
    bundle = tmp_path / "bundle"
    (bundle / "analysis_datatree.zarr").mkdir(parents=True)
    (bundle / "analysis_datatree.zarr" / ".zgroup").write_text("{}")
    (bundle / BUNDLE_MANIFEST_FILENAME).write_text("{}")
    _write(bundle / "plots" / "fig.png", 1)
    _write_sentinel(bundle, {"analysis_datatree.zarr": 300 * _MIB, "sims": 600 * _MIB})
    return bundle


def test_bundle_root_shows_the_recorded_rows_and_measures_nothing(tmp_path: Path) -> None:
    rows = _rows(_render(_bundle_root(tmp_path), tmp_path))

    assert rows == {"analysis_datatree.zarr": 300.0, "sims": 600.0}


def test_bundle_page_says_the_rows_were_not_remeasured(tmp_path: Path) -> None:
    html = _render(_bundle_root(tmp_path), tmp_path)

    assert "was not re-measured" in html


def test_live_page_does_not_claim_to_be_unmeasured(tmp_path: Path) -> None:
    assert "not re-measured" not in _render(_live_analysis(tmp_path), tmp_path)
