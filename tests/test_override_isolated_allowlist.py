"""The set of tests that build a private analysis tree is a committed, reviewed list.

The build-request census (``tests/test_build_request_census.py``) classifies every run-reaching test by where its
analysis tree comes from. ``override-isolated`` is the expensive class: a test of it builds its own tree, from a case
factory called with a literal ``start_from_scratch=True`` or from a per-test clone helper, instead of reading a shared
cache or a committed golden. Such a test is a deliberate cost, so it must not appear without a reviewer seeing it.

This test measures the class with the census's own collector (``_collect_in_child``) and its own class constant, so the
classification is not restated here, and compares the measured node ids to ``OVERRIDE_ISOLATED_NODE_IDS`` below.

* A new ``override-isolated`` test turns this red, naming the id: add it to the list in the same change, or build its
  tree from a golden or the shared cache instead.
* A test that stops being ``override-isolated`` (converted, renamed or deleted) turns it red too, naming the stale id:
  remove it from the list, so the list shrinks with the suite instead of rotting.

The measurement is static (it reads call sites, as the census does) and runs one collecting child process over
``tests/``, so it takes about half a minute. The census states what its walk cannot see.
"""

from __future__ import annotations

from tests.test_build_request_census import _TREE_OVERRIDE_ISOLATED, _collect_in_child

# The run-reaching test nodes the census classifies ``override-isolated``, sorted. Unit: pytest node ids.
OVERRIDE_ISOLATED_NODE_IDS: tuple[str, ...] = (
    "tests/test_bundle.py::test_reprex_bundle_carries_runnable_set",
    "tests/test_cleanup_orphans_on_run.py::test_cleanup_all_orphans_dry_run",
    "tests/test_cleanup_orphans_on_run.py::test_cleanup_all_orphans_force_deletes",
    "tests/test_cleanup_orphans_on_run.py::test_cleanup_all_orphans_force_required",
    "tests/test_cleanup_orphans_on_run.py::test_find_orphan_datatree_groups",
    "tests/test_cleanup_orphans_on_run.py::test_find_orphan_input_fingerprints",
    "tests/test_cleanup_orphans_on_run.py::test_find_orphan_status_flags",
    "tests/test_cleanup_orphans_on_run.py::test_run_aborts_when_orphans_and_flag_false",
    "tests/test_cleanup_orphans_on_run.py::test_run_cleans_when_flag_true",
    "tests/test_cleanup_orphans_on_run.py::test_run_dry_run_preserves_all_orphans",
    "tests/test_doi_roundtrip_e2e.py::test_leg1_self_contained_roundtrip_runs",
    "tests/test_doi_roundtrip_e2e.py::test_leg2_exclude_config_roundtrip_fetches_by_reference",
    "tests/test_doi_roundtrip_e2e.py::test_leg3_hydroshare_authenticated_private_retrieval",
    "tests/test_experiments_from_doi.py::test_from_doi_happy_path_is_runnable_and_self_contained",
    "tests/test_experiments_from_doi.py::test_from_doi_missing_input_fails_closed",
    "tests/test_experiments_from_doi.py::test_from_doi_no_main_entity_fails_closed",
    "tests/test_force_rerun_stage_axis.py::test_render_floor_dry_run_preserves_figures",
    "tests/test_force_rerun_stage_axis.py::test_simulate_floor_dry_run_preserves_processing_log",
    "tests/test_force_rerun_stage_axis.py::test_simulate_floor_dry_run_still_deletes_flags",
    "tests/test_node_local_config_guard.py::test_ack_env_bypasses_the_refusal",
    "tests/test_node_local_config_guard.py::test_explicit_slurm_locus_on_local_family_still_refuses",
    "tests/test_node_local_config_guard.py::test_local_reprocess_does_not_refuse",
    "tests/test_node_local_config_guard.py::test_local_submit_does_not_refuse_node_local_configs",
    "tests/test_node_local_config_guard.py::test_reprocess_refuses_node_local_configs",
    "tests/test_node_local_config_guard.py::test_sensitivity_reprocess_refuses_node_local_configs",
    "tests/test_node_local_config_guard.py::test_sensitivity_submit_workflow_refuses_node_local_configs",
    "tests/test_node_local_config_guard.py::test_slurm_submit_allows_shared_filesystem_configs",
    "tests/test_node_local_config_guard.py::test_slurm_submit_refuses_node_local_configs",
    "tests/test_report_config_run_entry.py::test_control_run_without_report_config_records_none",
    "tests/test_report_config_run_entry.py::test_run_records_the_run_entry_config",
    "tests/test_reprocess_delete_r8.py::test_nonsensitivity_explicit_local_execution_mode_is_not_overridden",
    "tests/test_reprocess_delete_r8.py::test_nonsensitivity_failed_scoped_delete_raises_instead_of_consolidating_stale",
    "tests/test_reprocess_rebuild_invalidation.py::test_reprocess_consolidate_inprocess_preserves_processed",
    "tests/test_reprocess_rebuild_invalidation.py::test_reprocess_regenerate_slurm_route_clears_flags_and_logs",
    "tests/test_reprocess_rebuild_invalidation.py::test_reprocess_regenerate_slurm_route_clears_log_nonsensitivity",
    "tests/test_sensitivity_submit_stamps_master.py::test_direct_sensitivity_submit_stamps_the_master",
    "tests/test_synth_02_multisim.py::test_concurrently_process_scenario_timeseries",
    "tests/test_synth_02_multisim.py::test_isolated_fixture_does_not_perturb_session_tree",
    "tests/test_synth_04_multisim_with_snakemake.py::test_snakemake_workflow_concurrency_and_process_monitoring",
    "tests/test_synth_04_multisim_with_snakemake.py::test_snakemake_workflow_dry_run",
    "tests/test_synth_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end",
    "tests/test_synth_05_sensitivity_analysis_with_snakemake.py::test_snakemake_sensitivity_workflow_dry_run",
    "tests/test_synth_05_sensitivity_analysis_with_snakemake.py::test_snakemake_sensitivity_workflow_execution",
    "tests/test_synth_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_swmm_only_execution",
    "tests/test_synth_07_reprocess.py::test_consolidate_to_datatree_rebuilds_when_log_incomplete",
    "tests/test_synth_07_reprocess.py::test_dry_run_submit_workflow_leaves_no_orchestrator_sentinel",
    "tests/test_synth_07_reprocess.py::test_reprocess_dry_run_no_destructive_mutation",
    "tests/test_synth_07_reprocess.py::test_reprocess_never_calls_input_even_with_stale_lock",
    "tests/test_synth_07_reprocess.py::test_reprocess_proceeds_with_submitted_workers_no_orchestrator",
    "tests/test_synth_07_reprocess.py::test_reprocess_refusal_does_not_destroy_the_report",
    "tests/test_synth_07_reprocess.py::test_reprocess_refuses_fast_with_live_orchestrator",
    "tests/test_synth_07_reprocess.py::test_reprocess_render_only",
    "tests/test_synth_07_reprocess_migration.py::test_reprocess_idempotent_when_current",
    "tests/test_synth_07_reprocess_migration.py::test_reprocess_refuses_when_layout_outdated",
    "tests/test_synth_07_validation_report.py::test_errors_and_warnings_renders_html_multi_sim_real",
    "tests/test_synth_07_validation_report.py::test_errors_and_warnings_renders_html_sensitivity_real",
    "tests/test_synth_07_validation_report.py::test_validation_report_multi_sim_failing_disk_mutations",
    "tests/test_synth_07_validation_report.py::test_validation_report_sensitivity_failing_disk_mutations",
    "tests/test_synth_08_sensitivity_reprocess.py::test_failed_scoped_delete_raises_instead_of_consolidating_stale",
    "tests/test_synth_08_sensitivity_reprocess.py::test_sensitivity_reprocess_dry_run_no_destructive_mutation",
    "tests/test_synth_08_sensitivity_reprocess.py::test_sensitivity_reprocess_refuses_fast_with_live_orchestrator",
    "tests/test_synth_09_from_doi_run_proof.py::test_from_doi_reconstituted_bundle_runs_locally",
    "tests/test_synth_09_reprex_roundtrip.py::test_analysis_reprex_bundle_degenerate_dispatch",
    "tests/test_synth_09_reprex_roundtrip.py::test_reprex_roundtrip_native_problem_pairs_and_amendments",
    "tests/test_synth_09_reprex_roundtrip.py::test_reprex_sif_digest_match_and_mismatch",
    "tests/test_synth_analysis_test_end_to_end.py::test_synth_analysis_test_sensitivity",
    "tests/test_synth_eda_promotion.py::test_default_output_path_is_cwd_relative_not_analysis_dir",
    "tests/test_synth_eda_promotion.py::test_emit_populates_only_neutral_fields",
    "tests/test_synth_eda_promotion.py::test_static_config_roundtrips_with_eda_plot_id",
    "tests/test_synth_flag_writes.py::test_override_force_rerun_clears_processing_log_outputs",
    "tests/test_synth_flag_writes.py::test_override_force_rerun_does_not_clear_other_member_processing_log",
    "tests/test_synth_flag_writes.py::test_override_force_rerun_event_iloc_invalidates_only_named_events",
    "tests/test_synth_fungibility_same_named_figures.py::test_same_named_figures_are_fungible",
    "tests/test_synth_per_sim_model_arms.py::test_composite_page_renders_end_to_end",
    "tests/test_synth_sensitivity_csv_row_rerun.py::test_empty_independent_vars_degenerate_case",
    "tests/test_synth_sensitivity_csv_row_rerun.py::test_fingerprint_files_written_on_first_invocation",
    "tests/test_synth_sensitivity_csv_row_rerun.py::test_fingerprint_idempotent_when_csv_unchanged",
    "tests/test_synth_sensitivity_csv_row_rerun.py::test_helper_returns_false_on_unchanged_repeat_call",
    "tests/test_synth_static_plots.py::test_static_plots_generates_conduit_flow_figure",
    "tests/test_synth_static_plots.py::test_static_plots_generates_publication_figure",
    "tests/test_synth_static_plots.py::test_static_plots_generates_sensitivity_benchmarking_figure",
    "tests/test_synth_static_plots.py::test_static_plots_generates_system_overview_figure",
    "tests/test_unread_settings_rulings.py::test_benchmarking_plotly_render_reads_the_aggregation_at_the_scaling_series",
    "tests/test_unread_settings_rulings.py::test_benchmarking_static_render_reaches_the_label_and_both_encoding_reads",
    "tests/test_unread_settings_rulings.py::test_conduit_render_reaches_its_label_and_quantile_reads",
    "tests/test_unread_settings_rulings.py::test_depth_plotly_render_reaches_the_lower_clip",
    "tests/test_unread_settings_rulings.py::test_depth_render_independent_policy_drops_both_cross_event_ranges",
    "tests/test_unread_settings_rulings.py::test_depth_render_quantile_strategy_takes_the_quantile_across_events_when_shared",
    "tests/test_unread_settings_rulings.py::test_depth_render_reaches_its_clip_label_and_shared_range_reads",
    "tests/test_unread_settings_rulings.py::test_system_overview_render_reaches_its_label",
    "tests/test_workflow_rerun_triggers_empirical.py::test_rerun_triggers_multi_sim_add_remove",
)


def test_override_isolated_set_equals_the_committed_list(tmp_path):
    """Property: the nodes the census classifies ``override-isolated`` equal ``OVERRIDE_ISOLATED_NODE_IDS``.

    Class: every collected node in ``tests/`` that reaches a workflow run. Why it is required: the class marks the
    tests that pay for a private tree, and nothing else in the suite reds when one is added. What kills it: a new
    test calling a case factory with a literal ``start_from_scratch=True`` or a clone helper, a test that leaves the
    class without its list entry being removed, or a collection narrowed by ini ``addopts`` or a conftest hook (the
    ``deselected == 0`` assertion guards that; the collecting child strips the parent's ``-m``, ``-k``,
    ``PYTEST_ADDOPTS`` and xdist variables, so those never reach it). A second correct implementation under which it
    still passes: one that reads the census artifact instead of collecting, since only the two id sets are compared.
    """
    report = _collect_in_child(tmp_path / "census.json")
    assert report["deselected"] == 0, (
        f"the collecting child deselected {report['deselected']} item(s); a narrowed collection hides nodes from "
        "the comparison. Look for -m/-k/--deselect in pytest addopts or in conftest hooks."
    )
    measured = {n["id"] for n in report["nodes"] if n["sites"] and n["tree"] == _TREE_OVERRIDE_ISOLATED}
    committed = set(OVERRIDE_ISOLATED_NODE_IDS)
    assert len(committed) == len(OVERRIDE_ISOLATED_NODE_IDS), "OVERRIDE_ISOLATED_NODE_IDS lists a node id twice"
    added = sorted(measured - committed)
    removed = sorted(committed - measured)
    assert not added and not removed, (
        "the override-isolated set changed.\n"
        f"  new (build the tree from a golden or the shared cache, or add the id to the list): {added}\n"
        f"  stale (no longer override-isolated or no longer collected; remove the id from the list): {removed}"
    )
