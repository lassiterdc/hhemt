"""Unit tests for the Workflow performance report renderer (`[Q160]`(7)).

HPC-free / compile-free. This page carries the run timeline and the SLURM
resource-efficiency table, both extracted out of the Metadata page: they describe how
the workflow RAN rather than what it produced.

`test_status_sidecars_are_declared_when_they_exist` MOVED here verbatim from
tests/test_synth_metadata_renderer.py. It was never a metadata assertion in substance --
it asserts the `_status/*.flag.json` declarations, the `status_flag_count` manifest key,
and that the timeline they back is rendered. All three are properties of THIS renderer
now, so the test follows the behaviour rather than being repaired in place.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from hhemt.config.report import report_config
from hhemt.report_renderers import metadata, workflow_performance


def _fake_analysis(analysis_dir: Path):
    """render() reads exactly two attributes off the analysis object."""
    return types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1"),
    )


_EFF_HEADER = (
    ",JobID,JobName,Elapsed,NNodes,NCPUS,MaxRSS,ReqMem,Elapsed_sec,TotalCPU_sec,"
    "MaxRSS_MB,RequestedMem_MB,MainJobID,CPU Efficiency (%),Memory Usage (%)\n"
)
#: One realistic data row: a job step, as the plugin emits it (parent + .batch rows
#: are dropped upstream, which is why JobName reads `python` rather than a rule name).
_EFF_ROW = "2,18573918.0,python,00:00:23,1,1,710788K,,23.0,0.0,694.12,1000.0,18573918,0.0,69.41\n"


def _render(analysis_dir: Path, *, slurm_csv: str | None = None) -> tuple[str, dict]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if slurm_csv is not None:
        # Faithful to snakemake-executor-plugin-slurm: --slurm-efficiency-report-path
        # is treated as a DIRECTORY, so the driver's `.csv`-suffixed path materializes
        # on disk as a directory that CONTAINS the real efficiency_report_{uuid}.csv.
        eff_dir = analysis_dir / "logs" / "slurm_efficiency_report"
        nested = eff_dir / "slurm_efficiency_report_20260101T000000.csv"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "efficiency_report_deadbeef.csv").write_text(slurm_csv)
    output_path = analysis_dir / "plots" / "workflow_performance.html"
    workflow_performance.render(_fake_analysis(analysis_dir), report_config(), output_path)
    manifest = json.loads((analysis_dir / "plots" / "workflow_performance.manifest.json").read_text())
    return output_path.read_text(), manifest


#: `[Q160]`(7): these tables are Tabulator data grids now, so a rendered row is a JSON
#: entry in `tableOptions.data` rather than a server-rendered `<td>`. The whole-cell
#: DISCRIMINATION the original `<td>18573918</td>` assertion existed for is preserved and
#: not weakened: `"Job ID (job record)": "18573918"` cannot match the step grain
#: `18573918.0` any more than the `<td>` form could, because both delimit the full value.
_JOB_GRAIN_CELL = '"Job ID (job record)": "18573918"'


def test_status_sidecars_are_declared_when_they_exist(tmp_path):
    """Every `_status/*.flag.json` the page OPENS is declared (declared ⊆ actual).

    Globbing is audit-invisible (os.scandir), but `read_text()` is not — so a sidecar
    that is read and not declared would break the ADR-6 invariant, and one that is
    declared and not read would put a false claim in the bundle manifest.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    for name, payload in (
        ("a_setup_target_0_complete.flag.json", {"rule_name": "setup_target_0", "slurm_job_id": "1"}),
        ("c_run_complete.flag.json", {"rule_name": "simulation_sa_x_evt_0", "slurm_job_id": "2"}),
    ):
        (analysis_dir / "_status" / name).write_text(json.dumps(payload))

    html, manifest = _render(analysis_dir)
    declared = manifest["source_paths_relative"]

    assert "_status/a_setup_target_0_complete.flag.json" in declared
    assert "_status/c_run_complete.flag.json" in declared
    assert manifest["renderer_data"]["status_flag_count"] == 2

    # And the timeline they back is actually rendered, with derived purposes.
    assert "Run timeline" in html
    assert "compile / setup" in html and "simulate" in html


def test_timeline_heading_and_table_ids_do_not_collide(tmp_path):
    """`[Q160]`(7): promoting the timeline to a top-level section must not duplicate an id.

    `_heading("Run timeline")` mints `id="run-timeline"` via `_anchor`. Before the move
    the table itself also carried `table_id="run-timeline"`, so the promotion would have
    put the attribute on TWO elements — invalid HTML, and the jump-nav's
    `data-jump="run-timeline"` would resolve to whichever the parser reached first. This
    is the same collision the SLURM table already dodges with `slurm-efficiency-table`.

    The assertion is on the INVARIANT (each id occurs once), not on the specific suffix,
    so renaming the table id again does not falsely redden this test.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    (analysis_dir / "_status" / "c_run_complete.flag.json").write_text(
        json.dumps({"rule_name": "simulation_sa_x_evt_0", "slurm_job_id": "2"})
    )
    html, _ = _render(analysis_dir)

    anchor = metadata._anchor("Run timeline")
    assert html.count(f'id="{anchor}"') == 1, "the section anchor must be unique on the page"
    assert f'href="#{anchor}"' in html, "the jump-nav must link the section it anchors"


def test_page_title_and_jump_nav_cover_this_pages_sections(tmp_path):
    """The wrapper is shared with metadata but parameterised — this page gets its own
    title and its own anchor vocabulary, and NONE of the metadata page's sections."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    html, _ = _render(analysis_dir)

    assert "<h2>Workflow performance — A1</h2>" in html
    for title in metadata._WORKFLOW_PERFORMANCE_SECTION_TITLES:
        assert f'href="#{metadata._anchor(title)}"' in html
    for title in metadata._SECTION_TITLES:
        assert f'href="#{metadata._anchor(title)}"' not in html, f"{title} belongs to the Metadata page"


def test_absent_inputs_degrade_to_banners_not_an_error(tmp_path):
    """A fresh analysis has no status sidecars and no efficiency CSV — the efficiency
    report is written at workflow teardown, AFTER this render. Both absences are the
    documented expected state, so the page must render banners rather than raise."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    html, manifest = _render(analysis_dir)

    assert manifest["renderer_data"]["status_flag_count"] == 0
    assert manifest["renderer_data"]["slurm_csv_present"] is False
    assert "Run timeline" in html and "SLURM Efficiency" in html
    assert "No SLURM resource-efficiency data" in html


# --- SLURM efficiency: MOVED verbatim-in-substance from test_synth_metadata_renderer.py.
# Every assertion below is a property of THIS page now. The one assertion that did NOT
# survive the move is `"ro-crate-metadata.json" in declared`: this renderer does not read
# the crate, so declaring it would be a false manifest claim. Its removal is the point of
# the move, not collateral.


def test_slurm_section_renders_table_and_declares_the_csv_file(tmp_path):
    """R5: the globbed CSV is rendered AND declared as a source — the FILE, not the dir."""
    html, manifest = _render(tmp_path / "analysis", slurm_csv=_EFF_HEADER + _EFF_ROW)
    # The table's GRAIN moved from step to job ([Q145]/[Q153]), so the identity it renders
    # is the allocation `18573918`, not the step `18573918.0`. Asserted as a whole cell
    # rather than a substring: a bare `18573918` also matches the old step id, so it could
    # not tell the two grains apart and would pass under either. See _JOB_GRAIN_CELL.
    assert _JOB_GRAIN_CELL in html
    # Memory-used % is now a reduction over steps rather than the plugin's own column, and
    # renders at one decimal (694.12 MB of 1000.0 MB requested).
    assert "69.4" in html
    declared = manifest["source_paths_relative"]
    assert any(p.endswith(".csv") for p in declared), declared
    # Declaring the DIRECTORY would raise in _validate_source_path once it exists.
    assert "logs/slurm_efficiency_report" not in declared


def test_slurm_report_nested_inside_a_csv_named_directory(tmp_path):
    """The REAL production shape: the `.csv` path is a DIRECTORY, report nested inside.

    snakemake-executor-plugin-slurm treats `--slurm-efficiency-report-path` as a
    DIRECTORY and writes `efficiency_report_{run_uuid}.csv` into it, while the toolkit
    passes a file-shaped `slurm_efficiency_report_{ts}.csv` path. So on every SLURM run
    the plugin mkdir's a directory with that `.csv` name and the real report lands one
    level deeper. The pre-2026-07-20 flat glob matched the DIRECTORY and `read_text()`
    raised `IsADirectoryError`, killing the whole rule; on the paths that did not crash
    it silently matched no real report at all, which is why this panel had never
    rendered data on any SLURM run.

    Nothing else in this file constructs this shape with a DISTINCT inner filename, so
    without this test the recursive-descend fix is unexercised against the case that
    motivated it.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # The plugin's actual layout: a DIRECTORY whose name ends in .csv, report inside.
    eff_dir = analysis_dir / "logs" / "slurm_efficiency_report"
    csv_named_dir = eff_dir / "slurm_efficiency_report_20260101T000000.csv"
    csv_named_dir.mkdir(parents=True, exist_ok=True)
    (csv_named_dir / "efficiency_report_abc123.csv").write_text(_EFF_HEADER + _EFF_ROW)

    output_path = analysis_dir / "plots" / "workflow_performance.html"
    # Must not raise IsADirectoryError.
    workflow_performance.render(_fake_analysis(analysis_dir), report_config(), output_path)

    html = output_path.read_text()
    # Job grain per [Q145]/[Q153]; whole-cell form so it cannot pass on the old step id.
    assert _JOB_GRAIN_CELL in html, "nested report was not found by the recursive glob"
    assert "69.4" in html

    manifest = json.loads((analysis_dir / "plots" / "workflow_performance.manifest.json").read_text())
    declared = manifest["source_paths_relative"]
    # The NESTED file must be declared -- never the .csv-named directory, which
    # _validate_source_path rejects as a directory-as-source.
    assert any(p.endswith("efficiency_report_abc123.csv") for p in declared), declared


def test_slurm_csv_with_header_only_degrades(tmp_path):
    """R5: a header-only CSV yields the heading + an info banner, not an empty table."""
    html, _ = _render(tmp_path / "analysis", slurm_csv=_EFF_HEADER)
    assert 'id="slurm-efficiency"' in html
    assert "no job rows" in html


def test_absent_slurm_csv_degrades_gracefully(tmp_path):
    """R7: no efficiency CSV -> heading present, teardown-timing explained."""
    html, manifest = _render(tmp_path / "analysis")
    assert 'id="slurm-efficiency"' in html
    assert "teardown" in html
    # Narrowed from a blanket `endswith(".csv")` to the EFFICIENCY csv specifically,
    # which is what this test's name and docstring have always scoped it to. The
    # blanket form was a safe over-approximation only while the efficiency CSV was the
    # sole csv this renderer could declare; `scenario_status.csv` is now declared
    # UNCONDITIONALLY per ADR-6 D3 (see workflow_performance.render), so the blanket
    # form would forbid a declaration the convention requires. `_render` never writes a
    # scenario_status.csv, so the R7 guarantee this test exists for -- the absent-SLURM
    # path declares no efficiency CSV -- is preserved exactly.
    assert not any(
        "efficiency_report" in p for p in manifest["source_paths_relative"]
    ), manifest["source_paths_relative"]


def test_slurm_report_path_is_a_directory_not_a_file(tmp_path):
    """Regression (Q8): the plugin writes efficiency_report_{uuid}.csv INSIDE a
    `.csv`-named directory; the renderer must descend to the inner file and must
    NOT raise IsADirectoryError on read_text()."""
    html, manifest = _render(tmp_path / "analysis", slurm_csv=_EFF_HEADER + _EFF_ROW)
    # Job grain per [Q145]/[Q153]; whole-cell form so it cannot pass on the old step id.
    assert _JOB_GRAIN_CELL in html and "69.4" in html
    declared = manifest["source_paths_relative"]
    assert any(p.endswith("efficiency_report_deadbeef.csv") for p in declared), declared
