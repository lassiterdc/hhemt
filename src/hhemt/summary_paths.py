"""Path-only per-scenario summary-existence predicates.

Leaf module (imports only stdlib + ``hhemt.scenario.compute_event_id_slug``)
so both ``workflow.py`` (the heavy workflow hub) and ``analysis_validation.py``
(a leaf consumed by renderers + consolidate) can import these predicates
without an import cycle. PATH-ONLY: MUST NOT instantiate ``TRITONSWMM_scenario``
(its constructor mkdir's ``processed/``, ``swmm/``, ``out_swmm/``).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
# DELIBERATE INDEPENDENT COPY, do not lift: tests/test_synth_remove_after_processing.py:45
# hand-maintains this membership as the guard's independent ground truth. Its own :40-44
# comment states why, and its guards compare it against PRODUCTION _reclaim_attrs.
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

_SUMMARY_ATTRS_BY_MODEL: dict[str, tuple[str, ...]] = {
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


def summary_artifact_present(path: Path, out_type: str) -> bool:
    """True iff ``path`` exists AND is the artifact KIND ``out_type`` implies.

    A zarr store is a DIRECTORY and a NetCDF summary is a FILE, so a bare
    ``.exists()`` is satisfied by either shape under either setting and cannot tell a
    readable artifact from one the reader raises on. ``is_file()`` ALONE is the wrong
    repair and is the one this docstring exists to forbid: it returns False on the zarr
    store that every DEFAULT configuration produces, which would report every healthy
    analysis incomplete at every consumer of the predicates below.

    Keyed on ``out_type`` rather than on the suffix because every caller here CONSTRUCTS
    the path from ``out_type`` and therefore knows the expected kind exactly. The
    suffix-keyed sibling in ``eda/cross_sim_identity._summary_paths`` is the GLOB-side
    form, written for a helper that discovers paths it did not construct; it is strictly
    weaker here, because it accepts a plain FILE at ``{stem}.zarr``.

    EXPORTED so this is the single source of the rule rather than its fourth independent
    statement. Three other sites state it today -- ``process_simulation._open_engine`` as
    an engine choice, ``process_simulation._summary_usable`` as an open probe, and
    ``eda/cross_sim_identity._summary_paths`` as a suffix rule -- and none imports
    another. Adopting this at those three is a one-line import each, deliberately NOT
    done here: two of the three files belong to no workstream in the current partition.
    Deliberately NOT an openability probe (the ``_summary_usable`` form): that would need
    xarray and O(subs x events x stems) I/O, and this module's PATH-ONLY contract exists
    so ``workflow.py`` and ``analysis_validation.py`` can both import it without a cycle.
    """
    return path.is_dir() if out_type == "zarr" else path.is_file()


def scenario_summaries_present(analysis, event_id: str, enabled_models: list[str]) -> bool:
    """True iff every enabled model's per-sim summary is present AND is the KIND
    ``target_processed_output_type`` implies — a zarr STORE (directory) under
    ``zarr``, a NetCDF FILE under ``nc``.

    Path-only KIND probe. It is NARROWER than a bare existence probe and STRICTLY
    WEAKER than ``processing_analysis._retrieve_combined_output``'s test, which it no
    longer mirrors: that test is a CONJUNCTION of existence (``processing_analysis.py``
    ``:501``) and a per-model processing-log record (``:539-546``) that, in its own
    words, "is the only signal in the tree that attests RETURN rather than PRESENCE".
    This predicate closes the KIND axis and leaves the READABILITY axis exactly where it
    was: a fill-filled partial zarr store satisfies ``is_dir()`` and OPENS, and only the
    log record screens it out. That is not a gap to close here — the log hangs off the
    scenario and the PATH-ONLY contract below forbids instantiating one. It remains the
    predicate ``consolidate_*_datatree`` skips on, and it MUST NOT instantiate
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
            candidate = processed / f"{stem}.{out_type}"
            if not summary_artifact_present(candidate, out_type):
                # THE WRONG-KIND CASE IS THE ONLY ONE THAT WARNS, and the condition is
                # exact rather than heuristic: the kind check has just failed, so an
                # EXISTING path is present-but-wrong-kind and nothing else. Absence --
                # the ordinary mid-workflow state this predicate reports on constantly --
                # is silent. Without this line the fix makes its own target population
                # QUIETER: today a zarr store under a '.nc' name reaches
                # processing_analysis._retrieve_combined_output and raises an uncaught
                # IsADirectoryError (measured; it is an OSError, so the four enclosing
                # (FileNotFoundError, ValueError) handlers do not catch it), and after
                # this fix both consumers skip it in silence -- consolidate_workflow.py
                # at its SKIP 2 `continue`, which mis-diagnoses the branch in-comment as
                # "mid-recovery", and workflow.py, which drops the sub. That population
                # is FROZEN: _already_written is log-based and no future run regenerates
                # it, so this predicate is its last detector and a silent skip is not
                # detection.
                if candidate.exists():
                    logger.warning(
                        "Summary artifact present but of the WRONG KIND, so this scenario "
                        "is being skipped as incomplete: %s exists but is not a %s. "
                        "target_processed_output_type is %r, which expects a %s. This is "
                        "the signature of an analysis written by a past 'nc' run on a "
                        "gridded TRITON configuration; no re-run regenerates it, because "
                        "the completion check is log-based and the configuration is now "
                        "refused at preflight.",
                        candidate,
                        "directory (zarr store)" if out_type == "zarr" else "file",
                        out_type,
                        "zarr store directory" if out_type == "zarr" else "NetCDF file",
                    )
                return False
    return True


def analysis_summaries_complete(analysis, enabled_models: list[str]) -> bool:
    """Whole-sub predicate: True iff EVERY scenario in the sub has all summaries.

    Whole-sub (not per-event) because ``consolidate_sensitivity_datatree``'s
    skip is all-or-nothing per member (Gotcha 36 stipulation:
    ``_retrieve_combined_output`` concatenates per-scenario summaries along
    ``event_iloc`` and is all-or-nothing per sub). Filtering per-event here
    would be MORE permissive than consolidation and re-introduce a mismatch.
    Path-only via ``compute_event_id_slug`` — no ``TRITONSWMM_scenario``
    instantiation (Note A). A member is a full Analysis instance
    (Gotcha 11), so ``member`` is passed directly as ``analysis``.
    """
    from hhemt.scenario import compute_event_id_slug

    for event_iloc in analysis.df_sims.index:
        ev = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        event_id = compute_event_id_slug(ev)
        if not scenario_summaries_present(analysis, event_id, enabled_models):
            return False
    return True
