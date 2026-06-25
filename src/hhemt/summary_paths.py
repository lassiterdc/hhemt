"""Path-only per-scenario summary-existence predicates.

Leaf module (imports only stdlib + ``hhemt.scenario.compute_event_id_slug``)
so both ``workflow.py`` (the heavy workflow hub) and ``analysis_validation.py``
(a leaf consumed by renderers + consolidate) can import these predicates
without an import cycle. PATH-ONLY: MUST NOT instantiate ``TRITONSWMM_scenario``
(its constructor mkdir's ``processed/``, ``swmm/``, ``out_swmm/``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Report-target completeness predicate (report-target-predicate unification).
# ---------------------------------------------------------------------------
# Canonical per-enabled-mode summary FILE STEMS under sims/{event_id}/processed/.
# Keyed IDENTICALLY to
# analysis.py::_reconcile_stale_process_flags_against_summaries::
# _SUMMARY_ATTRS_BY_MODEL — the exact set whose absence makes
# processing_analysis._retrieve_combined_output raise FileNotFoundError (the
# predicate consolidate_*_datatree skips on, Gotcha 36). The set is never
# narrowed (Gotcha 34): a sim's c_run flag can exist with its summary absent,
# so c_run is a STRICTLY WEAKER signal than summary-existence; enumerating a
# report target on c_run produces an unsatisfiable target when the summary is
# missing (the render_report failure this predicate closes). Stems mirror
# scenario.ScenarioPaths' output_*_summary naming (scenario.py:148-224).
_SUMMARY_STEMS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "tritonswmm": (
        "TRITONSWMM_TRITON_summary",
        "TRITONSWMM_SWMM_node_summary",
        "TRITONSWMM_SWMM_link_summary",
        "TRITONSWMM_perf_summary",
    ),
    "triton": (
        "TRITON_only_summary",
        "TRITON_only_perf_summary",
    ),
    "swmm": (
        "SWMM_only_node_summary",
        "SWMM_only_link_summary",
    ),
}


def scenario_summaries_present(analysis, event_id: str, enabled_models: list[str]) -> bool:
    """True iff every enabled model's per-sim summary file exists for ``event_id``.

    Path-only existence probe mirroring
    ``processing_analysis._retrieve_combined_output``'s test (the predicate
    ``consolidate_*_datatree`` skips on). It MUST NOT instantiate
    ``TRITONSWMM_scenario`` — that constructor mkdir's ``processed/``,
    ``swmm/``, and ``out_swmm/`` as a side effect (scenario.py:63/65/82), so a
    generation-time read-only probe would create scenario subdirectories and
    pay the full constructor cost O(subs x events) on every ``analysis.run()``.
    Paths are derived directly from ``ScenarioPaths``' naming convention
    (``sims/{event_id}/processed/{stem}.{out_type}``) using the pure
    ``compute_event_id_slug`` slug the caller already holds. The FULL canonical
    summary set is required (Gotcha 34) — a narrower set marks a sub COMPLETE
    that consolidation SKIPS, re-opening the exact render bug.
    """
    processed = analysis.analysis_paths.simulation_directory / event_id / "processed"
    out_type = analysis.cfg_analysis.target_processed_output_type
    for model in enabled_models:
        stems = _SUMMARY_STEMS_BY_MODEL.get(model, ())
        if not stems:
            return False
        for stem in stems:
            if not (processed / f"{stem}.{out_type}").exists():
                return False
    return True


def sub_analysis_summaries_complete(sub_analysis, enabled_models: list[str]) -> bool:
    """Whole-sub predicate: True iff EVERY scenario in the sub has all summaries.

    Whole-sub (not per-event) because ``consolidate_sensitivity_datatree``'s
    skip is all-or-nothing per sub-analysis (Gotcha 36 stipulation:
    ``_retrieve_combined_output`` concatenates per-scenario summaries along
    ``event_iloc`` and is all-or-nothing per sub). Filtering per-event here
    would be MORE permissive than consolidation and re-introduce a mismatch.
    Path-only via ``compute_event_id_slug`` — no ``TRITONSWMM_scenario``
    instantiation (Note A). A sub-analysis is a full Analysis instance
    (Gotcha 11), so ``sub_analysis`` is passed directly as ``analysis``.
    """
    from hhemt.scenario import compute_event_id_slug

    for event_iloc in sub_analysis.df_sims.index:
        ev = sub_analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        event_id = compute_event_id_slug(ev)
        if not scenario_summaries_present(sub_analysis, event_id, enabled_models):
            return False
    return True
