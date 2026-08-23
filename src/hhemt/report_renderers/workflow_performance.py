"""Workflow performance renderer: the run timeline + the SLURM efficiency table.

`[Q160]`(7). Both tables used to live on the Metadata page -- the timeline as an
`<h4>5b.` sub-block of `5. Process`, the SLURM table as a top-level section. Neither is
provenance: they describe how the workflow RAN, not what it produced or how to
reproduce it. They render here instead, under their own top-level report category.

WHY THIS MODULE IMPORTS FROM `metadata` RATHER THAN CARRYING ITS OWN COPIES.
The page chrome (`_wrap_html_doc`, `_jump_nav`) and the eight data helpers below are
metadata-private today and have NO other consumer in `src/` -- every mention of them in
`slurm_job_recovery.py` and `slurm_store.py` is prose in a comment, not an import. Moving
them here would be a ~390-line relocation that changes no behaviour and breaks
`tests/test_job_purpose_map_tiering.py`'s import; importing them costs one line each and
leaves that test untouched. Renderer-to-renderer import is the established pattern in this
package, not a novel reach: `per_sim_event_page` imports from `per_sim_peak_flood_depth`,
and `per_sim_conduit_flow` from `system_overview`. The topology is one-directional
(`workflow_performance -> metadata`) and acyclic.

The anchor vocabulary for THIS page (`_WORKFLOW_PERFORMANCE_SECTION_TITLES`) is declared
in `metadata.py` beside `_SECTION_TITLES`, for the same single-declaration reason the
comment there gives: `_jump_nav` reads it, `_jump_nav` is shared, and a second copy would
drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from hhemt.config.report import report_config
from hhemt.report_renderers.metadata import (
    _SCENARIO_STATUS_FILENAME,
    _SLURM_EFF_RELDIR,
    _WORKFLOW_PERFORMANCE_SECTION_TITLES,
    _absent_banner,
    _apply_config_fallbacks,
    _build_slurm_efficiency_html,
    _job_purpose_map,
    _load_job_recovery,
    _provenance_timeline,
    _read_scenario_status,
    _read_status_flag_payloads,
    _resolve_all_efficiency_csvs,
    _resolve_inline_css,
    _wrap_html_doc,
)

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis

__all__ = ["render"]


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render the Workflow performance page (run timeline + SLURM efficiency)."""
    from hhemt.report_renderers._figure_emission import emit_plot_with_sources
    from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    analysis_id = str(getattr(analysis.cfg_analysis, "analysis_id", "") or "")

    source_paths: list[Path] = []
    prov = ProvenanceLog()

    # (1) Run timeline -- projected from the per-rule `_status/*.flag.json` sidecars.
    # Globbing is audit-invisible (os.scandir), but the per-file read_text() is NOT, so
    # every sidecar actually OPENED is declared here per the declared-subset-of-actual
    # invariant (Gotcha 53). `_status/` is already copytree'd into the render bundle, so
    # this adds manifest rows and no payload bytes.
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="whole-experiment run timeline (per-rule status sidecars)",
    ) as artist:
        status_payloads, status_files = _read_status_flag_payloads(analysis_dir)
        source_paths.extend(status_files)
        for sidecar in status_files:
            artist.add_channel("status", ProvenanceRef(source_path=str(sidecar.relative_to(analysis_dir))))
        timeline = _provenance_timeline(status_payloads) if status_payloads else None
        if timeline is None:
            timeline_html = _absent_banner(
                "Run timeline",
                "No completed-rule records — the per-rule status sidecars "
                "(_status/*.flag.json) were not found. They are written one at a time as "
                "each workflow rule finishes; re-render after the run completes to populate.",
            )
        else:
            timeline_html = timeline.html

    # (2) SLURM efficiency -- glob + descend (os.scandir/os.stat; audit-invisible) to
    # EVERY inner efficiency_report_*.csv FILE, then declare them all. The plugin writes
    # the real CSV INSIDE a `.csv`-NAMED DIRECTORY (see _resolve_all_efficiency_csvs), so
    # read_text() on the glob match itself raises IsADirectoryError, and declaring the
    # DIRECTORY would raise in _validate_source_path (directory-as-source rejected unless
    # zarr). Declaring the whole union is also what carries it into the render bundle:
    # `_harvest_and_copy_sources` copies exactly the declared set and
    # `_copy_supporting_files` never touches logs/, so bundle carriage follows
    # declaration and needs no bundle-side change.
    eff_dir = analysis_dir.joinpath(*_SLURM_EFF_RELDIR)
    eff_csvs = _resolve_all_efficiency_csvs(eff_dir)
    # Stage-A recovery of the job and `.batch` rows the plugin's parsing drops.
    _recovery_map, _recovery_path = _load_job_recovery(analysis_dir)
    scenario_map, scenario_status_path = _read_scenario_status(analysis_dir)
    _apply_config_fallbacks(scenario_map, analysis)
    # ADR-6 D3: declare the expected source UNCONDITIONALLY. `scenario_status.csv` is
    # written by the SEPARATE `export_scenario_status` rule, which can race AFTER this
    # plot rule in the DAG (the same race `per_analysis_summary` documents), so a
    # presence-gated declaration makes this figure's DECLARED SET render-timing-
    # dependent -- two runs of the same analysis can declare different sources. That
    # non-determinism is what made `test_same_named_figures_are_fungible` flaky once
    # the model-keyed disjointness was normalized away. Declaring an expected-but-
    # absent source is safe on all three consuming surfaces: `_validate_source_path`
    # accepts non-existent paths, the renderer-IO audit only WARNS on declared-but-
    # unread (`_provenance_audit.py:287`; it RAISES only on undeclared reads, `:235`),
    # and the bundle harvest skips-with-warning (Gotcha 50). In-tree precedent:
    # `disk_utilization.py` declares `_status/_du.json` the same way.
    source_paths.append(analysis_dir / _SCENARIO_STATUS_FILENAME)

    def _gpu_hardware_for_partition(partition: str) -> str:
        """Partition -> GPU hardware, via the toolkit's own deterministic resolver.

        In-memory config only (no file read): `resolve_gpu_target` returns (None, None)
        for a CPU partition, an undeclared partition, or a missing HPC-system config, so
        this degrades to an empty cell rather than raising.
        """
        try:
            from hhemt.config.hpc_system import resolve_gpu_target

            hardware, _backend = resolve_gpu_target(getattr(analysis, "cfg_hpc_system", None), partition)
        except Exception:  # noqa: BLE001 -- a display column must not break the render
            return ""
        return hardware or ""

    if eff_csvs:
        source_paths.extend(eff_csvs)
        # Stage-A recovery artifact, declared only in the branch that reads it: the
        # absent-SLURM path declares no CSV at all, and an unconditional append here
        # broke that contract (caught by test_absent_slurm_csv_degrades_gracefully).
        if _recovery_path is not None:
            source_paths.append(_recovery_path)
        with prov.artist(
            axes_id="html_section",
            kind="table",
            note="SLURM resource-efficiency reports (union across all workflow submissions)",
        ) as artist:
            for csv_path in eff_csvs:
                artist.add_channel(
                    "data",
                    ProvenanceRef(source_path=str(csv_path.relative_to(analysis_dir))),
                )
            if scenario_status_path is not None:
                artist.add_channel("scenario_status", ProvenanceRef(source_path=_SCENARIO_STATUS_FILENAME))
            # Tier 2 of the purpose join. Declared INSIDE this branch, matching the
            # `_recovery_path` contract directly above: the absent-SLURM path reads
            # nothing here and must declare nothing. Declared even when absent (ADR-6 D3)
            # so the info-icon names the source, and because the renderer-IO audit
            # requires declared >= actual reads. Graceful-absent: an unreadable or
            # missing index yields {} and the join degrades to exactly Tier 1.
            _job_index_path = analysis_dir / "_status" / "_job_index.json"
            source_paths.append(_job_index_path)
            artist.add_channel("job_index", ProvenanceRef(source_path="_status/_job_index.json"))
            _job_index: dict[str, str] = {}
            if _job_index_path.exists():
                try:
                    _loaded = json.loads(_job_index_path.read_text())
                    if isinstance(_loaded, dict):
                        _job_index = {str(k): str(v) for k, v in _loaded.items()}
                except (OSError, ValueError):
                    _job_index = {}
            slurm = _build_slurm_efficiency_html(
                [(str(p.relative_to(analysis_dir)), p.read_text()) for p in eff_csvs],
                _job_purpose_map(status_payloads, _job_index),
                scenario_map,
                _gpu_hardware_for_partition,
                recovery=_recovery_map,
            )
            slurm_html = slurm.html
    else:
        slurm = None
        slurm_html = _absent_banner(
            "SLURM Efficiency",
            "No SLURM resource-efficiency data — this analysis ran in local/native "
            "mode, or the end-of-workflow efficiency report has not yet been written. "
            "It is finalized at workflow teardown, AFTER the report is rendered, so it "
            "is expected to be absent on the run that produces this page; re-render "
            "after the run completes to populate it.",
        )

    # `[Q160]`(7): both tables are Tabulator fragments now, so their styles and scripts
    # are emitted ONCE at document level while the body carries only mount points. A
    # degenerate branch (no status sidecars, no efficiency CSV, or a CSV with no job
    # rows) yields a banner and NO fragment, so a page with nothing to show contributes
    # no Tabulator assets at all -- which is what keeps the absent-SLURM page small.
    fragments = [
        tbl.fragment for tbl in (timeline, slurm) if tbl is not None and tbl.fragment is not None
    ]
    html = _wrap_html_doc(
        analysis_id,
        _resolve_inline_css(report_cfg),
        timeline_html,
        slurm_html,
        fragments=fragments,
        page_title="Workflow performance",
        section_titles=_WORKFLOW_PERFORMANCE_SECTION_TITLES,
    )
    return emit_plot_with_sources(
        html,
        output_path,
        source_paths,
        analysis_dir=analysis_dir,
        output_format="html",
        # ADR-6 Gate-A: this page legitimately declares NOTHING on a fresh analysis --
        # the status sidecars may not exist yet and the efficiency CSV is written at
        # workflow teardown, AFTER this render. That is the documented absent-SLURM
        # state, not a provenance defect, so the empty-sources gate is opted out of
        # rather than papered over with a source the page does not read.
        allow_empty_sources=True,
        manifest_data={
            "renderer": "workflow_performance",
            "status_flag_count": len(status_files),
            "slurm_csv_present": bool(eff_csvs),
            "slurm_csv_count": len(eff_csvs),
            "scenario_status_present": scenario_status_path is not None,
        },
        provenance=prov,
    )
