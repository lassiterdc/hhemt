"""Synthetic-model report-target-predicate parity tier.

Regression for the predicate-divergence class: the per-sim figure
``report()``/``rule all``/render-target enumeration must filter on
**summary-existence** (the predicate consolidation consumes), NOT on the
``c_run`` flag — because ``c_run`` is a strictly weaker signal (the comment above
``_SUMMARY_STEMS_BY_MODEL`` in ``hhemt.summary_paths`` states why): a sim
can have run (``c_run`` present) with its per-scenario summary absent (e.g. after
a ``regenerate_existing`` deletion), which enumerated an unsatisfiable per-sim
report target and made ``render_report`` raise ``WorkflowError: File ... marked
for report but does not exist``.

Generator-parity scope: the TWO **reprocess** generators —
``SensitivityAnalysisWorkflowBuilder.generate_reprocess_master_snakefile_content``
(whole-sub, ``start_with``-aware) and
``reprocess_snakefile_generator._available_event_ids`` (per-event) — both filter
their report-target enumeration on summary-existence via the shared
``workflow._scenario_summaries_present`` / ``_analysis_summaries_complete``
helpers. The PRODUCTION generator ``generate_master_snakefile_content`` is
intentionally NOT covered: it generates before sims run, so a generation-time gate
is harmful.
"""

import ast
import re
import shutil

import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.summary_paths import _SUMMARY_ATTRS_BY_MODEL

# The test runs on the committed sensitivity golden (extracted into ``tmp_path`` by
# ``golden_sensitivity_analysis``), so it needs no solver build. The reprocess it ends with is refused by a
# preflight before any rule runs, so none of the prerequisites a reprocess that executes rules needs (a
# dry-run-written Snakefile, the interpreter on ``PATH``, an identity shim for child interpreters) is required.
pytestmark = [pytest.mark.requires_snakemake_subprocess]


# Property: the reprocess generators exclude a c_run-present/summary-absent sub from the per-member
# consolidate enumeration and the per-sim plot targets, and an unscoped reprocess over that
# divergent tree fails fast rather than publishing a partial datatree.
def test_report_target_predicate_excludes_summary_absent_sub(
    golden_sensitivity_analysis,
):
    """The two reprocess generators exclude a c_run-present/summary-absent sub/event.

    Arrange: take the finished sensitivity golden (real summaries + c_run flags on
    disk), then delete ONE per-scenario summary for one sub's first
    event while leaving its ``c_run`` flag — the c_run-present/summary-absent
    divergence state the module docstring describes.

    Assert:
    - (d) ``generate_reprocess_master_snakefile_content`` RETURNS without raising
      (its own check that the per-member consolidate loop and ``completed_member_ids``
      emit the same members, which raises ``RuntimeError`` on a mismatch, holds —
      generation does not abort).
    - (a)/(b) the sensitivity reprocess generator EXCLUDES the incomplete sub from
      both its per-member consolidate enumeration (``completed_member_ids`` / ``rule all``)
      and its per-sim plot targets (``MEMBER_EVENT_PAIRS``), while a COMPLETE sub is
      still enumerated.
    - (a)/(b) the non-sensitivity ``_available_event_ids`` EXCLUDES the divergent
      event (per-event predicate) while keeping every complete event.
    - (c) an UNSCOPED ``reprocess(start_with="consolidate")`` over the divergent tree
      FAILS FAST: the consolidate-stage divergence preflight in
      ``TRITONSWMM_sensitivity_analysis.reprocess`` raises ``ConfigurationError`` so a
      partial ``sensitivity_datatree.zarr`` is never silently published. (The
      generator-level exclusion in (a)/(b)/(d) is independent — the preflight fires
      on execution, not on content generation.) That fail-fast and the generators'
      graceful exclusion are two deliberate behaviors in tension for the unscoped
      path; this test pins the preflight's fail-fast behavior, and which end-state is
      intended is a design question this test does not settle.

    What kills it: a generator that filters on ``c_run`` still enumerates the
    divergent sub, so the (a)/(b) exclusion asserts fail; removing the preflight
    makes ``pytest.raises(ConfigurationError, match="cannot consolidate")`` fail.
    A generator that excluded the sub by a different summary-existence check than
    the shared helpers would still pass, because the asserts read the generated
    content and the returned event ids and never name the helper.
    """
    from hhemt.constants import (
        consolidate_analysis_flag,
        sim_run_flag_per_member,
    )
    from hhemt.reprocess_snakefile_generator import (
        _available_event_ids,
        _enabled_models,
    )
    from hhemt.scenario import (
        TRITONSWMM_scenario,
        compute_event_id_slug,
    )

    analysis = golden_sensitivity_analysis

    master_dir = analysis.analysis_paths.analysis_dir

    member_items = list(analysis.sensitivity.members.items())
    assert len(member_items) >= 2, (
        f"parity test needs >=2 members (one to break, one to keep); fixture produced {len(member_items)}"
    )
    target_member_key, target_sub = member_items[0]
    complete_member_key, _complete_sub = member_items[1]
    target_member = str(target_member_key)
    complete_member = str(complete_member_key)

    scen = TRITONSWMM_scenario(0, target_sub)
    evt = compute_event_id_slug(target_sub._retrieve_weather_indexer_using_integer_index(0))
    model_type = scen.run.model_types_enabled[0]

    # The c_run flag is the precondition for the divergence (the predicate the
    # OLD code keyed on — must remain present after we delete the summary).
    c_run_flag = master_dir / sim_run_flag_per_member(model_type, target_member, evt)
    assert c_run_flag.exists(), f"expected c_run flag present before delete: {c_run_flag}"

    # Delete one present per-scenario summary for the target sub's first event.
    # _SUMMARY_ATTRS_BY_MODEL (declared once in hhemt.summary_paths) maps model_type to
    # the full set of per-enabled-mode summary attributes; its file-stem counterpart
    # _SUMMARY_STEMS_BY_MODEL is "Keyed IDENTICALLY" to it. This test uses it only to
    # find one present summary to delete.
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
    # excludes the summary-absent sub.
    content = analysis.sensitivity._workflow_builder.generate_reprocess_master_snakefile_content(
        start_with="consolidate"
    )
    # Consolidate-flag enumeration (completed_member_ids -> rule all): the flag names
    # are literal strings in the generated content.
    target_consolidate_flag = consolidate_analysis_flag(target_member)
    complete_consolidate_flag = consolidate_analysis_flag(complete_member)
    assert target_consolidate_flag not in content, (
        "incomplete sub's consolidate flag must NOT be enumerated "
        "(completed_member_ids / rule all / per-member consolidate loop) — the "
        f"shared sub-inclusion equality holds and the sub is excluded: {target_consolidate_flag}"
    )
    # Sanity: a COMPLETE sub IS still enumerated (the fix is not over-filtering).
    assert complete_consolidate_flag in content, (
        f"complete sub's consolidate flag must remain enumerated: {complete_consolidate_flag}"
    )

    # Per-sim plot targets are enumerated via the MEMBER_EVENT_PAIRS_MEMBER list (the plot
    # rule path is wildcarded `member-{member_id}/{event_id}` + `zip` expand over the
    # list), so parse the list rather than matching an expanded literal.
    m = re.search(r"^MEMBER_EVENT_PAIRS_MEMBER = (\[.*?\])\s*$", content, re.MULTILINE)
    assert m is not None, "MEMBER_EVENT_PAIRS_MEMBER assignment not found in generated reprocess content"
    member_event_pairs_member = ast.literal_eval(m.group(1))
    assert target_member not in member_event_pairs_member, (
        f"incomplete sub {target_member!r} must be excluded from MEMBER_EVENT_PAIRS_MEMBER={member_event_pairs_member}"
    )
    assert complete_member in member_event_pairs_member, (
        f"complete sub {complete_member!r} must remain in MEMBER_EVENT_PAIRS_MEMBER={member_event_pairs_member}"
    )

    # ── (a)/(b) non-sensitivity per-event predicate excludes ONLY the divergent
    # event.
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
    # with a c_run-present/summary-absent divergence must FAIL FAST. The consolidate-stage
    # divergence preflight in TRITONSWMM_sensitivity_analysis.reprocess raises
    # ConfigurationError so a partial sensitivity_datatree.zarr is never silently
    # published with the divergent sub dropped. For the UNSCOPED call this test asserts
    # that the call raises ConfigurationError matching "cannot consolidate" (the
    # generator-level exclusion in (a)/(b)/(d) above is still proven — the preflight
    # only fires on actual execution, not on content generation).
    #
    # Not established: which end-state is intended. The preflight's fail-fast and the
    # generators' graceful exclusion are two deliberate behaviors in tension for the
    # unscoped path, and this test pins the fail-fast one rather than softening the
    # guard; whether the intended end-state is fail-fast or proceed-over-subset is an
    # open design question that this test's pass or fail does not answer.
    # The golden carries no analysis_report.zip (tests/goldens/SPEC.md section 9).
    report_zip = master_dir / "analysis_report.zip"
    if report_zip.exists():
        report_zip.unlink()
    with pytest.raises(ConfigurationError, match="cannot consolidate"):
        analysis.reprocess(start_with="consolidate", execution_mode="local")
