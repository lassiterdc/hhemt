"""Synthetic-model report-target-predicate parity tier (Phase 1).

Regression for the predicate-divergence class behind the live
``uva_sensitivity_suite`` reprocess ``render_report`` failure: the per-sim figure
``report()``/``rule all``/render-target enumeration must filter on
**summary-existence** (the predicate consolidation consumes), NOT on the
``c_run`` flag — because ``c_run`` is a strictly weaker signal (Gotcha 34): a sim
can have run (``c_run`` present) with its per-scenario summary absent (e.g. after
a ``regenerate_existing`` deletion), which enumerated an unsatisfiable per-sim
report target and made ``render_report`` raise ``WorkflowError: File ... marked
for report but does not exist``.

Generator-parity scope (Phase 1): the TWO **reprocess** generators —
``SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content``
(whole-sub, ``start_with``-aware) and
``reprocess_snakefile_generator._available_event_ids`` (per-event) — both filter
their report-target enumeration on summary-existence via the shared
``workflow._scenario_summaries_present`` / ``_sub_analysis_summaries_complete``
helpers. The PRODUCTION generator ``generate_master_snakefile_content`` is
intentionally DESCOPED (it generates before sims run, so a generation-time gate
is harmful, and its render failure is unreachable in a single-DAG ``run()`` — the
report branch is transitively gated behind ``master_consolidation``); it is NOT
covered here. See the phase-doc descope note + the v2 research addendum.
"""

import ast
import re
import shutil

import pytest

import tests.utils_for_testing as tst_ut
from hhemt.exceptions import ConfigurationError

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."),
    # Runs the sensitivity workflow to completion (real summaries on disk) before
    # deleting one; needs the compiled binaries. Skips without cmake+mpic++;
    # HARD-FAILS under HHEMT_REQUIRE_COMPILE_TIER=1.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled"),
]

# model_type -> the FULL consolidate-consumed per-enabled-mode summary attrs
# (mirrors analysis.py::_reconcile_stale_process_flags_against_summaries::
# _SUMMARY_ATTRS_BY_MODEL — the canonical set the report-target predicate and
# consolidation's _retrieve_combined_output both key against). Used only to pick
# a present summary to delete.
_SUMMARY_ATTRS_BY_MODEL = {
    "tritonswmm": (
        "output_tritonswmm_triton_summary",
        "output_tritonswmm_node_summary",
        "output_tritonswmm_link_summary",
        "output_tritonswmm_performance_summary",
    ),
    "triton": (
        "output_triton_only_summary",
        "output_triton_only_performance_summary",
    ),
    "swmm": (
        "output_swmm_only_node_summary",
        "output_swmm_only_link_summary",
    ),
}


@pytest.mark.slow
def test_report_target_predicate_excludes_summary_absent_sub(
    synth_sensitivity_analysis,
):
    """The two reprocess generators exclude a c_run-present/summary-absent sub/event.

    Arrange: run the sensitivity workflow to completion (real summaries + c_run
    flags on disk), then delete ONE per-scenario summary for one sub's first
    event while leaving its ``c_run`` flag — the exact Gotcha-34 divergence state.

    Assert:
    - (d) ``generate_reprocess_master_snakefile_content`` RETURNS without raising
      (the ~6785 shared-sub-inclusion EQUALITY invariant holds — generation does
      not abort).
    - (a)/(b) the sensitivity reprocess generator EXCLUDES the incomplete sub from
      both its per-sa consolidate enumeration (``completed_sa_ids`` / ``rule all``)
      and its per-sim plot targets (``SA_EVENT_PAIRS``), while a COMPLETE sub is
      still enumerated.
    - (a)/(b) the non-sensitivity ``_available_event_ids`` EXCLUDES the divergent
      event (per-event predicate) while keeping every complete event.
    - (c) an UNSCOPED ``reprocess(start_with="consolidate")`` over the divergent tree
      FAILS FAST: the R7 consolidate-stage divergence preflight (Gotcha 40) raises
      ``ConfigurationError`` so a partial ``sensitivity_datatree.zarr`` is never
      silently published. (The generator-level exclusion in (a)/(b)/(d) is independent
      — R7 fires on execution, not on content generation.) The fail-fast (Gotcha 40)
      vs graceful-exclusion (Gotcha 37) tension for the unscoped path is flagged for
      main's owner; this test conservatively pins the newer guard's behavior.
    """
    from hhemt.constants import (
        consolidate_subanalysis_flag,
        sim_run_flag_per_sa,
    )
    from hhemt.reprocess_snakefile_generator import (
        _available_event_ids,
        _enabled_models,
    )
    from hhemt.scenario import (
        TRITONSWMM_scenario,
        compute_event_id_slug,
    )

    analysis = synth_sensitivity_analysis

    # ── Arrange: full sensitivity run so real summary zarrs + c_run flags exist. ──
    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="both",
        override_clear_raw="none",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"
    tst_ut.assert_analysis_summaries_created(analysis)

    master_dir = analysis.analysis_paths.analysis_dir

    sa_items = list(analysis.sensitivity.sub_analyses.items())
    assert len(sa_items) >= 2, (
        f"parity test needs >=2 sub-analyses (one to break, one to keep); fixture produced {len(sa_items)}"
    )
    target_sa_key, target_sub = sa_items[0]
    complete_sa_key, _complete_sub = sa_items[1]
    target_sa = str(target_sa_key)
    complete_sa = str(complete_sa_key)

    scen = TRITONSWMM_scenario(0, target_sub)
    evt = compute_event_id_slug(target_sub._retrieve_weather_indexer_using_integer_index(0))
    model_type = scen.run.model_types_enabled[0]

    # The c_run flag is the precondition for the divergence (the predicate the
    # OLD code keyed on — must remain present after we delete the summary).
    c_run_flag = master_dir / sim_run_flag_per_sa(model_type, target_sa, evt)
    assert c_run_flag.exists(), f"expected c_run flag present before delete: {c_run_flag}"

    # Delete one present per-scenario summary for the target sub's first event.
    summary_to_delete = None
    for attr in _SUMMARY_ATTRS_BY_MODEL[model_type]:
        p = getattr(scen.scen_paths, attr, None)
        if p is not None and p.exists():
            summary_to_delete = p
            break
    assert summary_to_delete is not None, f"no present summary output found to delete for model_type={model_type}"
    if summary_to_delete.is_dir():
        shutil.rmtree(summary_to_delete)
    else:
        summary_to_delete.unlink()
    assert not summary_to_delete.exists(), "summary should be deleted"
    assert c_run_flag.exists(), "c_run flag must survive summary deletion (this IS the divergence state)"

    # ── (d) generation must not raise + (a)/(b) sensitivity reprocess generator
    # excludes the summary-absent sub. ──────────────────────────────────────────
    content = analysis.sensitivity._workflow_builder.generate_reprocess_master_snakefile_content(
        start_with="consolidate"
    )
    # Consolidate-flag enumeration (completed_sa_ids -> rule all): the flag names
    # are literal strings in the generated content.
    target_consolidate_flag = consolidate_subanalysis_flag(target_sa)
    complete_consolidate_flag = consolidate_subanalysis_flag(complete_sa)
    assert target_consolidate_flag not in content, (
        "incomplete sub's consolidate flag must NOT be enumerated "
        "(completed_sa_ids / rule all / per-sa consolidate loop) — the ~6785 "
        f"equality holds and the sub is excluded: {target_consolidate_flag}"
    )
    # Sanity: a COMPLETE sub IS still enumerated (the fix is not over-filtering).
    assert complete_consolidate_flag in content, (
        f"complete sub's consolidate flag must remain enumerated: {complete_consolidate_flag}"
    )

    # Per-sim plot targets are enumerated via the SA_EVENT_PAIRS_SA list (the plot
    # rule path is wildcarded `sa-{sa_id}/{event_id}` + `zip` expand over the
    # list), so parse the list rather than matching an expanded literal.
    m = re.search(r"^SA_EVENT_PAIRS_SA = (\[.*?\])\s*$", content, re.MULTILINE)
    assert m is not None, "SA_EVENT_PAIRS_SA assignment not found in generated reprocess content"
    sa_event_pairs_sa = ast.literal_eval(m.group(1))
    assert target_sa not in sa_event_pairs_sa, (
        f"incomplete sub {target_sa!r} must be excluded from SA_EVENT_PAIRS_SA={sa_event_pairs_sa}"
    )
    assert complete_sa in sa_event_pairs_sa, (
        f"complete sub {complete_sa!r} must remain in SA_EVENT_PAIRS_SA={sa_event_pairs_sa}"
    )

    # ── (a)/(b) non-sensitivity per-event predicate excludes ONLY the divergent
    # event. ────────────────────────────────────────────────────────────────────
    sub_builder = target_sub._workflow_builder
    enabled = _enabled_models(sub_builder)
    n_events = len(target_sub.df_sims)
    all_event_ids = [
        compute_event_id_slug(target_sub._retrieve_weather_indexer_using_integer_index(i)) for i in range(n_events)
    ]
    available = _available_event_ids(target_sub, enabled_models=enabled, all_event_ids=all_event_ids)
    assert evt not in available, "non-sensitivity _available_event_ids must exclude the summary-absent event"
    assert len(available) == n_events - 1, "exactly the one summary-absent event is excluded; all complete events kept"

    # ── (c) end-to-end: an UNSCOPED reprocess(start_with="consolidate") over a tree
    # with a c_run-present/summary-absent divergence must FAIL FAST. The R7 consolidate-
    # stage divergence preflight (Gotcha 40, sensitivity_analysis.py) raises
    # ConfigurationError so a partial sensitivity_datatree.zarr is never silently
    # published with the divergent sub dropped. This is the NEWER deliberate guard; it
    # intentionally supersedes the older "proceed over the completed subset" expectation
    # for the UNSCOPED call (the generator-level exclusion in (a)/(b)/(d) above is still
    # proven — R7 only fires on actual execution, not on content generation).
    #
    # NOTE (flagged for main's owner — see follow-up): R7/Gotcha-40 (fail-fast) and the
    # generator's Gotcha-37 (graceful exclusion) are two deliberate features in tension
    # for the unscoped path. This test pins the fail-fast behavior conservatively rather
    # than softening the guard from the worktree; whether the intended end-state is
    # fail-fast or proceed-over-subset is main's design decision.
    report_zip = master_dir / "analysis_report.zip"
    if report_zip.exists():
        report_zip.unlink()
    with pytest.raises(ConfigurationError, match="cannot consolidate"):
        analysis.reprocess(start_with="consolidate", execution_mode="local")
