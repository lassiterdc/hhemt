"""Reviewer tests for the defaults (consolidation, clearing, clear_raw and remove_after_processing).

Three tests, each over a CLASS the coder's own tests sample at one point:

* the three-switch removal conjunction over the PRODUCT of the four keys' settings (the coder pins each key flipped
  alone);
* the real process runner under the default policy, under both arms of D24 (b), on a golden scenario (the coder's
  guard tests read NAME sets and a stubbed runner);
* the cleared-but-kept scenario state read by ``df_status`` for every model family (the coder's regression test
  covers the coupled family only).

All three are laptop-admissible: no solver, no compile, goldens extracted into ``tmp_path``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from hhemt.config.analysis import analysis_config
from hhemt.scenario_records import removal_switches_on
from tests.fixtures.golden_trees import construct_golden_analysis, dry_run_plan
from tests.test_defaults_clear_everything import _cfg_dict
from tests.test_summary_store_clearing import _own, _ready, _record

_ALL_CLASSES = [
    "timeseries",
    "raw_swmm_binaries",
    "coupled_rpt",
    "hydro_out",
    "prep_inputs",
    "hydrographs",
    "standalone_rpt",
]


@pytest.fixture
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout would rebuild)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _independent_expectation(consolidate, clearing, clear_raw, remove, enabled) -> bool:
    """D24 (a)'s conjunction restated from the shape text, not from ``removal_switches_on``.

    Clearing resolved True (unset follows the consolidation toggle), ``clear_raw`` covering every enabled model, and
    every one of the seven classes elected.
    """
    clearing_resolved = consolidate if clearing == "unset" else clearing
    if not clearing_resolved:
        return False
    covers = clear_raw == "all" or (clear_raw != "none" and set(enabled) <= set(clear_raw))
    elects_all = remove == "all" or (remove != "none" and set(_ALL_CLASSES) <= set(remove))
    return bool(covers and elects_all)


_CONSOLIDATE = [True, False]
_CLEARING = ["unset", True, False]
_CLEAR_RAW = ["all", "none", ["tritonswmm"], ["tritonswmm", "triton", "swmm"]]
# "all", "none", the seven classes in two orders, and the seven lists that each omit exactly one class.
_REMOVE = ["all", "none", list(_ALL_CLASSES), list(reversed(_ALL_CLASSES))] + [
    [c for c in _ALL_CLASSES if c != omitted] for omitted in _ALL_CLASSES
]


def test_the_removal_conjunction_holds_over_every_combination_of_the_four_keys(tmp_path: Path) -> None:
    """Property: for every combination of consolidation (2 settings), clearing (3), ``clear_raw`` (4) and
    ``remove_after_processing`` (11), a config that loads has ``removal_switches_on`` equal to the shape's
    conjunction, and a config naming clearing True with consolidation False is the only combination that fails to
    load as accepted by preflight.

    Input class: the 264-point product of the four keys. Plausible wrong implementation: one key's default changed
    but a value form the coder's single-flip tests never name (a list ``clear_raw`` covering only a subset, a
    six-class list, a seven-class list in a different order) mis-resolving the removal in either direction: a
    silent keep of every directory, or a silent removal of directories the user meant to keep for reprocessing.
    REGRESSION-class characterization: green on the tree under review; HELD, see the enumeration in the round file.
    Both-states anchor: the assertion turns on the resolved config values, present before and after the change.
    A second correct implementation moves the resolution into another validator; the test reads the loaded config.
    Killed by: ``removal_switches_on`` treating a list ``remove_after_processing`` or ``clear_raw`` as electing
    everything, or the clearing switch resolving independently of the consolidation toggle.
    """
    enabled = ["tritonswmm"]
    from hhemt import validation

    checked = 0
    on_count = 0
    for consolidate, clearing, clear_raw, remove in itertools.product(_CONSOLIDATE, _CLEARING, _CLEAR_RAW, _REMOVE):
        over = {
            "toggle_consolidate_timeseries": consolidate,
            "clear_raw": clear_raw,
            "remove_after_processing": remove,
        }
        if clearing != "unset":
            over["toggle_clear_scenario_timeseries_after_consolidation"] = clearing
        cfg = analysis_config.model_validate(_cfg_dict(tmp_path, **over))
        result = validation.ValidationResult(context="test")
        validation._validate_clear_scenario_timeseries(cfg, result)
        refused = bool(result.errors)
        assert refused is (clearing is True and consolidate is False), (consolidate, clearing, clear_raw, remove)
        got = removal_switches_on(cfg, enabled)
        want = _independent_expectation(consolidate, clearing, clear_raw, remove, enabled)
        assert got is want, (consolidate, clearing, clear_raw, remove, got, want)
        checked += 1
        on_count += int(got)
    # Must-fire control: the product contains removal-ON points and removal-OFF points, so equality above is not a
    # comparison of two constants.
    assert checked == 264
    assert 0 < on_count < checked, on_count


@pytest.mark.parametrize("consolidation_on", [True, False])
def test_the_runner_under_the_default_policy_deletes_no_summary_and_no_series_store(
    golden_extractor, consolidation_on
) -> None:
    """Property (D24 (b)): the REAL ``remove_after_processing`` of a golden scenario, run under the shipped default
    policy (``"all"``, ``clear_raw`` ``"all"``) with consolidation on and with it off, removes no per-scenario
    ``*_summary`` store and no per-scenario ``*_tseries`` store, and removes only artifacts of the seven classes.

    Input class: the two arms of ``toggle_consolidate_timeseries`` under the default policy, on a scenario that
    holds every kind of store. Plausible wrong implementation: ``_reclaim_attrs`` kept honest (empty) while another
    branch of ``_reclaim_paths`` or the runner body deletes a store of the preserve set; the coder's guards read the
    NAME set of the class and a stubbed runner, so they cannot see a delete that does not go through that name set.
    Must-fire control: the same call removes the raw SWMM binary (``hydraulics.out``) of the scenario, so the
    runner did act on this input and the preserve assertion is not a comparison against an idle call.
    Why required: with consolidation on the series live in the per-scenario store until the consolidation's
    verified clearing, so a runner that deleted them first would leave the consolidation nothing to copy; with it off
    the series are the processed output and exist nowhere else.
    Killed by: restoring the earlier behaviour (the class applied at process time), or any new delete of a store
    named in ``summary_paths._SUMMARY_STEMS_BY_MODEL`` or of a ``*_tseries`` store. Passes under a second correct
    implementation that records the discharge in the model log instead of printing it.
    """
    a = construct_golden_analysis(golden_extractor("multisim"))
    a.cfg_analysis.remove_after_processing = "all"
    a.cfg_analysis.clear_raw = "all"
    a.cfg_analysis.toggle_consolidate_timeseries = consolidation_on
    sims = Path(a.analysis_paths.simulation_directory)

    def stores() -> set[str]:
        return {
            str(p.relative_to(sims))
            for p in sims.rglob("*.zarr")
            if p.is_dir() and (p.stem.endswith("_summary") or p.stem.endswith("_tseries"))
        }

    before = stores()
    assert before, "control: the golden holds per-scenario summary and series stores"
    raw = sims / "event_index.0" / "out_tritonswmm" / "swmm" / "hydraulics.out"
    assert raw.exists(), "control: the raw SWMM binary the default policy reclaims is present before the call"
    a._retrieve_sim_run_processing_object(0).remove_after_processing(model_type="tritonswmm")
    assert not raw.exists(), "control: the runner acted on this scenario"
    assert stores() == before, sorted(before ^ stores())


@pytest.mark.parametrize("family", ["tritonswmm", "triton", "swmm"])
def test_df_status_of_a_kept_scenario_whose_summaries_were_cleared_matches_the_uncleared_reading(
    golden_extractor, _pin_clean_build, family
) -> None:
    """Property: with consolidation and clearing on and the directories kept, ``df_status`` returns the same
    performance cells after the consolidation cleared the per-scenario summaries as it returned before, for each
    model family.

    Input class: the three model families (coupled, TRITON-only, SWMM-only); the coder's regression test pins the
    coupled family. Plausible wrong implementation: the cleared-summary read keyed on the coupled family's model
    name, so the TRITON-only row opens a store the clearing deleted and raises. Must-fire control: the family's
    summary store exists before the consolidation and is gone after it, the clearing record is ``done``, and for the
    families that carry a performance summary the compared cells are not all None before the clearing.
    Why required: a user who sets ``clear_raw: none`` and takes the other defaults is in exactly this state, and
    ``df_status`` feeds the status CSV, the validation report and the report front page.
    Killed by: reverting the ``analysis._get_performance_summary_row`` clearing-record branch, or restricting it to
    one family. Passes under a second correct implementation that reads the tree for every event.
    """
    a = _ready(golden_extractor, clear=True, families=(family,))
    dry_run_plan(a)
    stem = {
        "tritonswmm": "TRITONSWMM_TRITON_summary",
        "triton": "TRITON_only_summary",
        "swmm": "SWMM_only_node_summary",
    }[family]
    assert _own(a, 0, stem).exists(), "control: the family's summary store is present before the consolidation"
    before = a.df_status
    perf = [c for c in before.columns if c.startswith("perf_") and c != "perf_column_set_across_allocations"]
    if family != "swmm":
        assert perf and before[perf].notna().any().any(), "control: the performance cells compared are not all None"
    a.process.consolidate_to_datatree()
    assert _record(a, 0, family) is not None, "the clearing record must exist, or nothing was cleared"
    assert not _own(a, 0, stem).exists(), "the summary store must be cleared"
    after = a.df_status
    assert len(after) == len(before)
    for col in perf:
        assert after[col].tolist() == before[col].tolist(), col
