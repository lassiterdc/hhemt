"""Preflight refuses the series discard election together with ``observation_points`` (developer's gate G18).

With ``toggle_consolidate_timeseries`` False, ``clear_raw`` covering every enabled model and ``remove_after_processing``
a list naming ``timeseries``, the process runner deletes the per-scenario series stores. The observation groups of the
consolidation read the per-scenario gridded store and nothing clears it afterwards, so the pair is refused before any
allocation is spent.

Non-solver and fixture-free: stub configs (preflight validators read configs with ``getattr``), no golden, no compile,
no master run.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import validation
from hhemt.validation import ValidationResult, _validate_timeseries_discard_observation_points

_SYSTEM = {
    "observation_points": Path("points.csv"),
    "toggle_triton_model": False,
    "toggle_tritonswmm_model": True,
    "toggle_swmm_model": False,
}
_ANALYSIS = {
    "toggle_consolidate_timeseries": False,
    "clear_raw": "all",
    "remove_after_processing": ["timeseries", "coupled_rpt"],
}


def _errors(system: dict, analysis: dict) -> list:
    result = ValidationResult(context="test")
    _validate_timeseries_discard_observation_points(SimpleNamespace(**system), SimpleNamespace(**analysis), result)
    return list(result.errors)


def test_the_full_conjunction_is_refused_and_the_message_names_the_settings():
    """Property: the four-leg conjunction yields exactly one error that names every setting and the way out."""
    errors = _errors(_SYSTEM, _ANALYSIS)
    assert len(errors) == 1
    text = f"{errors[0].message} {errors[0].fix_hint}"
    for name in ("observation_points", "toggle_consolidate_timeseries", "clear_raw", "remove_after_processing"):
        assert name in text, name
    assert "timeseries" in errors[0].fix_hint and "unset observation_points" in errors[0].fix_hint


@pytest.mark.parametrize(
    ("system_over", "analysis_over"),
    [
        ({}, {"toggle_consolidate_timeseries": True}),
        ({"observation_points": None}, {}),
        ({}, {"clear_raw": "none"}),
        ({}, {"clear_raw": ["triton"]}),
        ({}, {"remove_after_processing": "all"}),
        ({}, {"remove_after_processing": ["coupled_rpt"]}),
        ({}, {"remove_after_processing": "none"}),
    ],
    ids=[
        "consolidation-on",
        "observation-points-unset",
        "clear_raw-none",
        "clear_raw-not-covering",
        "blanket-all",
        "list-without-timeseries",
        "remove-none",
    ],
)
def test_each_missing_leg_is_not_refused(system_over, analysis_over):
    """Must-fire control: the same configs with that one leg restored are refused (the first test), so a pass here
    is the missing leg's effect and not a validator that never fires."""
    assert _errors({**_SYSTEM, **system_over}, {**_ANALYSIS, **analysis_over}) == []


def test_clear_raw_covering_every_enabled_model_by_list_is_refused():
    """The coverage leg reads the enabled models from the system toggles: a list naming the enabled model covers."""
    assert len(_errors(_SYSTEM, {**_ANALYSIS, "clear_raw": ["tritonswmm"]})) == 1
    assert _errors(_SYSTEM, {**_ANALYSIS, "clear_raw": ["swmm"]}) == []


def test_the_refusal_is_wired_into_preflight_validate():
    """The check runs from ``preflight_validate`` beside the other cross-config checks."""
    assert "_validate_timeseries_discard_observation_points(cfg_system, cfg_analysis, result)" in inspect.getsource(
        validation.preflight_validate
    )


# ---------------------------------------------------------------------------
# Sensitivity members: a member's configs never pass preflight_validate, so the refusal runs once per member row.
# ---------------------------------------------------------------------------
def _members_errors(tmp_path, master_over: dict, rows: dict, *, system_by_member=None, master_system=None) -> list:
    """The per-member refusal over a sensitivity table whose rows are ``rows`` (member id -> analysis.* cells)."""
    import pandas as pd

    from hhemt.config.analysis import analysis_config
    from hhemt.validation import _validate_members_timeseries_discard_observation_points
    from tests.test_defaults_clear_everything import _cfg_dict

    master = analysis_config.model_validate(_cfg_dict(tmp_path, **master_over))
    df = pd.DataFrame.from_dict(rows, orient="index")
    systems = system_by_member or {str(member): SimpleNamespace(**_SYSTEM) for member in rows}
    result = ValidationResult(context="test")
    _validate_members_timeseries_discard_observation_points(
        df, SimpleNamespace(**(master_system or _SYSTEM)), master, systems, result
    )
    return list(result.errors)


# The master states the election but keeps consolidation on, so the master itself is not refused.
_MASTER = {"clear_raw": "all", "remove_after_processing": ["timeseries", "coupled_rpt"]}


def test_a_member_row_that_sweeps_into_the_refused_combination_is_refused_and_named(tmp_path):
    """Property: the member whose swept cell turns consolidation off is refused, the error names that member and still
    names the settings and the fix; the member that does not sweep it is not."""
    errors = _members_errors(
        tmp_path,
        _MASTER,
        {"0": {"analysis.toggle_consolidate_timeseries": True}, "1": {"analysis.toggle_consolidate_timeseries": False}},
    )
    assert len(errors) == 1
    text = f"{errors[0].field} {errors[0].message} {errors[0].fix_hint}"
    assert "member_id=1" in errors[0].message and "row[1]" in errors[0].field
    for name in ("observation_points", "toggle_consolidate_timeseries", "clear_raw", "remove_after_processing"):
        assert name in text, name
    assert "unset observation_points" in errors[0].fix_hint


def test_a_member_row_that_sets_only_some_legs_is_not_refused(tmp_path):
    """Must-fire control: member 1 above is refused with consolidation off; here the same off row has a keeping
    ``clear_raw`` (one leg gone), the system leg gone (no observation points for that member), or no swept cell."""
    no_points = {**_SYSTEM, "observation_points": None}
    errors = _members_errors(
        tmp_path,
        _MASTER,
        {
            "0": {"analysis.toggle_consolidate_timeseries": False, "analysis.clear_raw": "none"},
            "1": {"analysis.toggle_consolidate_timeseries": False},
            "2": {"analysis.toggle_consolidate_timeseries": float("nan")},
        },
        system_by_member={
            "0": SimpleNamespace(**_SYSTEM),
            "1": SimpleNamespace(**no_points),
            "2": SimpleNamespace(**_SYSTEM),
        },
    )
    assert errors == []


def test_a_master_that_is_already_refused_is_not_repeated_per_member(tmp_path):
    """The master's own error is the one report; the members are not repeated."""
    errors = _members_errors(
        tmp_path,
        {**_MASTER, "toggle_consolidate_timeseries": False},
        {"0": {"analysis.clear_raw": "all"}, "1": {"analysis.clear_raw": "all"}},
    )
    assert errors == []


def test_the_member_check_is_wired_into_the_per_member_preflight():
    """The per-member check runs from ``_validate_per_member_system_configs`` with the rows' effective systems."""
    source = inspect.getsource(validation._validate_per_member_system_configs)
    assert (
        "_validate_members_timeseries_discard_observation_points(df, cfg_system, cfg_analysis, system_by_member"
        in source
    )
