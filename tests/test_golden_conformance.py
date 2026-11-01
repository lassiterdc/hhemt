"""hhemt must accept the committed goldens and must reject a golden broken in any one layer.

The goldens are written by an independent generator (``tests/goldens/SPEC.md``); this module is the other half of
that independence: it asks hhemt's own readers whether the trees are finished analyses.

* construction: hhemt builds an analysis object from each extracted golden;
* validation: once ``scenario_status.csv`` is exported and a dry run has written the Snakefile (the two files a
  golden does not carry, SPEC section 9), every check ``validate_analysis`` aggregates passes;
* mutation: one corruption per layer (raw, summary, completion records, consolidated) is rejected by the check that
  reads that layer, and the unmutated tree is the control;
* plan: a dry run over a fresh extraction schedules no setup, prepare, simulate, process or consolidate job, only
  report-stage jobs (a golden carries no plots or report), and the same extraction with one run flag older than
  its input schedules that scenario's run, process and consolidation jobs.

No test here compiles or runs a solver; a dry run only plans.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from hhemt.analysis_validation import validate_analysis
from tests.fixtures.golden_trees import (
    BASE_TIME,
    TIER_SECONDS,
    construct_golden_analysis,
    dependency_tier,
    dry_run_plan,
    export_status_csv,
)

ANALYSIS = "synth_multi_sim"
EVENT = "event_index.1"

# Report-stage jobs of a dry run over a fresh extraction. A golden has no plots or report, so these are planned;
# the stages the golden completed (setup, prepare, run, process, consolidate) are not.
MULTISIM_REPORT_PLAN = {
    "all": 1,
    "export_scenario_status": 1,
    "plot_disk_utilization": 1,
    "plot_errors_and_warnings": 1,
    "plot_metadata": 1,
    "plot_per_analysis_summary_table": 1,
    "plot_per_sim_event_page": 3,
    "plot_scenario_status_appendix": 1,
    "plot_system_overview": 1,
    "plot_workflow_performance": 1,
    "render_report": 1,
}
SENSITIVITY_REPORT_PLAN = {
    "all": 1,
    "export_scenario_status": 1,
    "plot_disk_utilization": 1,
    "plot_errors_and_warnings": 1,
    "plot_metadata": 1,
    "plot_per_analysis_summary_table": 1,
    "plot_per_sim_per_member_conduit_flow": 4,
    "plot_per_sim_per_member_peak_flood_depth": 4,
    "plot_scenario_status_appendix": 1,
    "plot_system_overview": 1,
    "plot_workflow_performance": 1,
    "render_report": 1,
}
# Rule-name prefixes of the stages the golden finished. ``run_`` and ``process_`` match the per-model rule families
# (``run_tritonswmm`` and ``process_tritonswmm`` appear in STALE_RUN_ADDS); ``consolidate`` matches both ``consolidate``
# and ``consolidate_scenario``. Not established by this file: why ``setup`` and ``prepare`` are bare prefixes.
COMPLETED_STAGE_RULES = ("setup", "prepare", "run_", "process_", "consolidate")
STALE_RUN_ADDS = {"run_tritonswmm": 1, "process_tritonswmm": 1, "consolidate_scenario": 1, "consolidate": 1}


def _completed_stage_jobs(plan: dict[str, int]) -> dict[str, int]:
    """The planned jobs of a stage the golden finished; ``export_scenario_status`` is report-stage."""
    return {r: n for r, n in plan.items() if r.startswith(COMPLETED_STAGE_RULES)}


def _validate(analysis):
    """Plan (which writes the Snakefile), export the status table, then run every aggregated check.

    Both files exist before any check runs, as the module docstring's validation bullet requires. Not established by
    this file: whether the export must follow the dry run or only both must precede the checks."""
    dry_run_plan(analysis)
    export_status_csv(analysis)
    return validate_analysis(analysis)


def _failed(report) -> list[str]:
    return [c.name for c in report.checks if not c.passed]


# ---- construction and validation ---------------------------------------------------------------------------


def test_multisim_golden_constructs_and_validates(golden_multisim_analysis) -> None:
    assert golden_multisim_analysis.cfg_analysis.analysis_id == ANALYSIS
    assert len(golden_multisim_analysis.df_sims) == 3
    report = _validate(golden_multisim_analysis)
    assert _failed(report) == []
    assert len(report.checks) > 0


def test_sensitivity_golden_constructs_and_validates(golden_sensitivity_analysis) -> None:
    analysis = golden_sensitivity_analysis
    assert analysis.cfg_analysis.toggle_sensitivity_analysis
    report = _validate(analysis)
    assert _failed(report) == []
    assert len(report.checks) > 0


# ---- one mutation per layer ----------------------------------------------------------------------------------


def _mutate_raw(root: Path) -> None:
    """Raw layer: the SWMM report loses its terminal line, so the run is not finalized."""
    rpt = root / ANALYSIS / "sims" / EVENT / "out_tritonswmm" / "swmm" / "hydraulics.rpt"
    text = rpt.read_text()
    assert "Analysis ended on" in text
    rpt.write_text(text.replace("Analysis ended on", "Analysis stopped"))


def _mutate_summary(root: Path) -> None:
    """Summary layer: a per-scenario summary store disappears.

    ``MUTATIONS`` pairs this with the check ``Timeseries processed``: the control test shows that check passes on the
    intact golden and the parametrised test asserts it fails here. Why that check reads this store is not shown in
    this file."""
    shutil.rmtree(root / ANALYSIS / "sims" / EVENT / "processed" / "TRITONSWMM_TRITON_summary.zarr")


def _mutate_records(root: Path) -> None:
    """Completion records: the scenario log says preparation did not complete.

    ``MUTATIONS`` pairs this with the check ``Scenarios setup``: the control test shows that check passes on the
    intact golden and the parametrised test asserts it fails here. Why that check reads
    ``scenario_creation_complete`` is not shown in this file."""
    log = root / ANALYSIS / "sims" / EVENT / "scenario_prep_log.json"
    record = json.loads(log.read_text())
    assert record["scenario_creation_complete"] is True
    record["scenario_creation_complete"] = False
    log.write_text(json.dumps(record))


def _mutate_consolidated(root: Path) -> None:
    """Consolidated tier: the per-analysis DataTree store disappears.

    ``MUTATIONS`` pairs this with the check ``Analysis summaries created``: the control test shows that check passes
    on the intact golden and the parametrised test asserts it fails here. Why that check reads this store is not
    shown in this file."""
    shutil.rmtree(root / ANALYSIS / "analysis_datatree.zarr")


MUTATIONS = {
    "raw": (_mutate_raw, "Scenarios ran"),
    "summary": (_mutate_summary, "Timeseries processed"),
    "records": (_mutate_records, "Scenarios setup"),
    "consolidated": (_mutate_consolidated, "Analysis summaries created"),
}


def test_the_unmutated_golden_passes_every_mutation_check(golden_extractor) -> None:
    """Control for the mutation tests: the checks they name pass on the golden as committed."""
    report = _validate(construct_golden_analysis(golden_extractor("multisim")))
    passed = {c.name for c in report.checks if c.passed}
    assert {name for _, name in MUTATIONS.values()} <= passed


@pytest.mark.parametrize("layer", sorted(MUTATIONS))
def test_a_mutation_in_each_layer_is_rejected(layer: str, golden_extractor) -> None:
    mutate, check_name = MUTATIONS[layer]
    root = golden_extractor("multisim")
    mutate(root)
    failed = _failed(_validate(construct_golden_analysis(root)))
    assert check_name in failed, f"{layer} mutation was not reported by {check_name!r}; failed checks: {failed}"


# ---- the dry-run plan ------------------------------------------------------------------------------------------


def test_fresh_multisim_extraction_plans_report_stage_jobs_only(golden_multisim_analysis) -> None:
    plan = dry_run_plan(golden_multisim_analysis)
    assert _completed_stage_jobs(plan) == {}
    assert plan == MULTISIM_REPORT_PLAN
    # The total of MULTISIM_REPORT_PLAN's counts; the equality above already fixes it, so this adds no check.
    assert sum(plan.values()) == 13


def test_fresh_sensitivity_extraction_plans_report_stage_jobs_only(golden_sensitivity_analysis) -> None:
    plan = dry_run_plan(golden_sensitivity_analysis)
    assert _completed_stage_jobs(plan) == {}
    assert plan == SENSITIVITY_REPORT_PLAN
    # The total of SENSITIVITY_REPORT_PLAN's counts; the equality above already fixes it, so this adds no check.
    assert sum(plan.values()) == 18


def test_a_run_flag_older_than_its_input_plans_that_scenarios_downstream_jobs(golden_extractor) -> None:
    """The control of the two tests above: the same planner schedules jobs when one flag is out of date."""
    root = golden_extractor("multisim")
    status = root / ANALYSIS / "_status"
    prepare = status / f"b_prepare_evt-{EVENT}_complete.flag"
    run = status / f"c_run_tritonswmm_evt-{EVENT}_complete.flag"
    older = prepare.stat().st_mtime - TIER_SECONDS
    # The aged run flag must stay newer than the setup flag, so the only out-of-order pair is prepare then run: the
    # expected plan below adds no setup or prepare job, only STALE_RUN_ADDS.
    assert older > BASE_TIME + dependency_tier("a_setup_complete.flag") * TIER_SECONDS
    os.utime(run, (older, older))
    plan = dry_run_plan(construct_golden_analysis(root))
    assert plan == {**MULTISIM_REPORT_PLAN, **STALE_RUN_ADDS}
    # 17 is the 13 report-stage jobs of MULTISIM_REPORT_PLAN plus the 4 jobs of STALE_RUN_ADDS.
    assert sum(plan.values()) == 17
