"""The clearing toggle of the consolidated time series resolves to the consolidation toggle when it is left unset.

Once the consolidated store holds a scenario's verified slice, the scenario's own time-series copies are redundant,
so an analysis that turns consolidation on and says nothing about clearing clears them (approved shape D17, errata
E52). The field is ``bool | None`` and the analysis config's after-validator resolves ``None`` to the value of
``toggle_consolidate_timeseries``, so every reader sees a bool. An explicit ``False`` keeps the copies, and an
explicit ``True`` without consolidation stays refused by preflight.

No solver, no golden tree and no filesystem beyond the three empty input files the minimal config dict touches.
"""

from __future__ import annotations

from pathlib import Path

from hhemt import validation
from hhemt.config.analysis import analysis_config
from tests.test_config_validation import _minimal_analysis_config_dict

_CLEAR = "toggle_clear_scenario_timeseries_after_consolidation"


def _resolved(tmp_path: Path, **toggles) -> analysis_config:
    """The analysis config built from the minimal valid dict with ``toggles`` merged in, through the validators."""
    return analysis_config.model_validate({**_minimal_analysis_config_dict(tmp_path), **toggles})


def _preflight_refusals(cfg: analysis_config) -> list:
    """Errors of the clearing refusal alone, on an already-resolved config."""
    result = validation.ValidationResult(context="test")
    validation._validate_clear_scenario_timeseries(cfg, result)
    return list(result.errors)


def test_unset_clearing_resolves_to_true_when_consolidation_is_on(tmp_path: Path) -> None:
    """Property: with consolidation on and the clearing key absent, the resolved clearing value is True.

    Class: NEW CAPABILITY. Plausible wrong implementation: keeping ``default=False``, which resolves to False
    and leaves every consolidated analysis holding two copies of the series. Reachability: a config dict with
    ``toggle_consolidate_timeseries`` true and no clearing key is the shape of the experiment YAMLs this default
    serves; ``model_validate`` runs the same after-validator a loaded ``cfg_analysis.yaml`` runs. Both-states
    anchor: the assertion turns on the resolved value of a field present in the old and the new code, not on
    wording the change introduces. Why required: the readers in ``processing_analysis`` convert the field with
    ``bool(...)``, so a value left at None would read as False and silently keep the copies. Killed by: removing
    the resolution or resolving to a constant False. A second correct implementation resolves in a field
    validator that reads ``info.data``; the test reads only the resolved attribute.
    """
    cfg = _resolved(tmp_path, toggle_consolidate_timeseries=True)
    assert cfg.toggle_clear_scenario_timeseries_after_consolidation is True


def test_explicit_false_keeps_the_copies_when_consolidation_is_on(tmp_path: Path) -> None:
    """Property: an explicit False survives resolution, so the per-scenario copies are kept.

    Class: REGRESSION against the over-eager resolution (a default that overwrote a written False); on the
    pre-change tree the field already held False, so the row is a held test, mutated by resolving with
    ``or`` instead of ``is None``. Reachability: the four E52 test files write this exact pair of lines.
    Both-states anchor: asserts a value an explicit key carries in both the old and the new code. Why required:
    those tests read the per-scenario stores back after consolidation, and a True there deletes the stores.
    Killed by: ``self.field = self.field or self.toggle_consolidate_timeseries``. A second correct
    implementation resolves only when the key is absent from ``model_fields_set``; the test passes under it.
    """
    cfg = _resolved(tmp_path, toggle_consolidate_timeseries=True, **{_CLEAR: False})
    assert cfg.toggle_clear_scenario_timeseries_after_consolidation is False


def test_unset_clearing_resolves_to_false_when_consolidation_is_off(tmp_path: Path) -> None:
    """Property: with consolidation off and the clearing key absent, the resolved clearing value is False.

    Class: REGRESSION, held: green before and after. Reachability: a default analysis (neither key written) is
    every analysis that never opted in. Both-states anchor: the resolved value of the same field. Why required:
    a True here would be refused by preflight, so every default analysis would fail to launch. Killed by:
    resolving to True unconditionally. A second correct implementation resolves in ``model_post_init`` rather
    than in a model validator; the resolved value is the same and the test passes.
    """
    cfg = _resolved(tmp_path)
    assert cfg.toggle_consolidate_timeseries is True, "consolidation is on by default (D24 (a))"
    assert cfg.toggle_clear_scenario_timeseries_after_consolidation is True
    assert _preflight_refusals(cfg) == [], "an analysis that mentions neither key must clear preflight"
    # The off arm, now explicit: with consolidation off the unset clearing key still resolves to False.
    off = _resolved(tmp_path, toggle_consolidate_timeseries=False)
    assert off.toggle_clear_scenario_timeseries_after_consolidation is False
    assert _preflight_refusals(off) == [], "consolidation off with clearing unset must clear preflight"


def test_the_resolved_value_is_what_a_written_config_carries(tmp_path: Path) -> None:
    """Property: a dumped config carries the resolved bool, never None, for every input row.

    Class: NEW CAPABILITY. Plausible wrong implementation: resolving in the readers, which leaves the field
    None in the persisted ``cfg_analysis.yaml`` and makes a re-read differ from the run that wrote it.
    Reachability: ``model_dump`` is what the toolkit persists. Both-states anchor: the dumped value's type.
    Why required: the shape states that a written resolved config carries the resolved value. Killed by:
    resolution at read time only. A second correct implementation resolves in ``model_post_init``; the test
    reads the dump and passes.
    """
    for toggles, expected in (
        ({"toggle_consolidate_timeseries": True}, True),
        ({"toggle_consolidate_timeseries": True, _CLEAR: False}, False),
        ({"toggle_consolidate_timeseries": False}, False),
        ({}, True),
    ):
        dumped = _resolved(tmp_path, **toggles).model_dump()[_CLEAR]
        assert dumped is expected, (toggles, dumped)


def test_explicit_true_without_consolidation_is_still_refused_and_with_it_is_not(tmp_path: Path) -> None:
    """Property: preflight refuses an explicit True with consolidation off and accepts it with consolidation on.

    Class: REGRESSION, held. The refusal is the must-fire control of the three rows above: it is the one input
    where resolution must NOT rewrite the value, since an explicit True is not None. Reachability: the config
    below validates, because the field accepts any bool, and the refusal is the preflight check's.
    Both-states anchor: the error count per row. Why required: clearing without a consolidated store deletes the
    only copy of the series. Killed by: resolving an explicit True to False (the refusal never fires), or
    refusing whenever clearing is on. A second correct implementation words the refusal differently or reports
    it with a different field label; the test reads only that the error names the field.
    """
    refused = _preflight_refusals(_resolved(tmp_path, toggle_consolidate_timeseries=False, **{_CLEAR: True}))
    assert len(refused) == 1, "explicit True with consolidation off must be refused"
    assert _CLEAR in str(refused[0])
    accepted = _preflight_refusals(_resolved(tmp_path, toggle_consolidate_timeseries=True, **{_CLEAR: True}))
    assert accepted == [], "explicit True with consolidation on must be accepted"
