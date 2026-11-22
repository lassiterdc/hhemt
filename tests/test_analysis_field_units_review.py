"""Tests for change 1118-units: the unit of ``swmm_coupling_lag`` and the numeric-field census.

The census (``_numeric_analysis_fields`` in ``test_every_stored_variable_has_units.py``) is the only thing that
makes ``cf_conventions._ANALYSIS_FIELD_UNITS`` complete, so these tests grade the census as an instrument. They run
the census test of that module against a table with one entry deleted, against a model with one numeric field added,
and against an independent census built from the pydantic JSON schema. They also run ``stamp_parameter_units`` over
a synthetic ``parameters`` dataset with the Phase B column set. Nothing here compiles, runs a solver or calls SLURM.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Optional

import pandas as pd
import pytest
import xarray as xr
from pydantic import Field
from pydantic.fields import FieldInfo

import hhemt.cf_conventions as cf
from hhemt.cf_conventions import _ANALYSIS_FIELD_UNITS, stamp_parameter_units
from hhemt.config.analysis import analysis_config
from hhemt.sensitivity_analysis import _parameter_column_field


def _census_module():
    """The census module, imported late so pytest does not collect its tests a second time here."""
    return importlib.import_module("tests.test_every_stored_variable_has_units")


def _schema_says_numeric(prop: dict) -> bool:
    """True when a pydantic JSON-schema property can carry a number (an independent reading of the annotation)."""
    if prop.get("type") in ("integer", "number"):
        return True
    for key in ("enum",):
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in prop.get(key, [])):
            return True
    if isinstance(prop.get("const"), (int, float)) and not isinstance(prop.get("const"), bool):
        return True
    return any(_schema_says_numeric(q) for k in ("anyOf", "oneOf", "allOf") for q in prop.get(k, []))


def _model_with(field_name: str, annotation, default):
    """A stand-in for ``analysis_config`` that carries its fields plus one more, as pydantic stores them.

    The census reads only ``model_fields`` (name to ``FieldInfo``), and ``FieldInfo.from_annotated_attribute`` splits
    an ``Annotated`` wrapper into annotation and metadata exactly as a declared class attribute does. A stand-in is used
    instead of ``create_model(..., __base__=analysis_config)`` because building a 128-field subclass per case costs
    about 0.14 s of the 0.1 s budget a new test node has; the subclass form was run once over the same rows and gave
    the same verdicts.
    """
    fields = {**analysis_config.model_fields, field_name: FieldInfo.from_annotated_attribute(annotation, default)}
    return SimpleNamespace(model_fields=fields)


@pytest.mark.parametrize("deleted", sorted(_ANALYSIS_FIELD_UNITS))
def test_the_census_goes_red_when_any_single_unit_entry_is_deleted(deleted, monkeypatch):
    """Property: deleting ANY one entry of ``_ANALYSIS_FIELD_UNITS`` makes the census test fail naming it.

    Class: every key of the live table (44 at this tree), which spans the four annotation kinds the request names:
    ``swmm_coupling_lag`` (``Literal[0, 1]``), ``swmm_report_step_s`` (``int | None``), ``TRITON_courant_number``
    (``float``) and ``n_gpus`` (``int``). REGRESSION in kind: the table and the census agree on the live tree, so each
    row is red only under the named mutant. Why it is required: a table entry the census does not demand is a unit
    nothing keeps. What kills it: a census that read a subset of the model's fields. A second correct implementation:
    a census over the pydantic JSON schema instead of the annotation, as in the independent-census test below.
    """
    kept = {k: v for k, v in _ANALYSIS_FIELD_UNITS.items() if k != deleted}
    assert len(kept) == len(_ANALYSIS_FIELD_UNITS) - 1, "control: the mutant removed exactly one entry"
    monkeypatch.setattr(cf, "_ANALYSIS_FIELD_UNITS", kept)
    with pytest.raises(AssertionError) as caught:
        _census_module().test_every_scalar_numeric_analysis_field_has_a_unit()
    assert deleted in str(caught.value)


def test_the_unmutated_census_test_passes():
    """Control for the mutant tests: on the live table the census test is green, so a red is the mutant."""
    _census_module().test_every_scalar_numeric_analysis_field_has_a_unit()


@pytest.mark.parametrize(
    "annotation, default",
    [
        (Literal[0, 1], 0),
        (Literal[0, 1, 2], 0),
        (Literal.__getitem__((0.5, 1.0)), 0.5),  # a float Literal is outside the typing spec; built at runtime
        (Literal[0, "auto"], "auto"),
        (Optional[int], None),  # noqa: UP045 -- the typing spelling is the property under test
        (int | None, None),
        (float, 1.0),
        (float | None, None),
        (int, 1),
        (Annotated[int, Field(ge=0)], 0),
        (Optional[Annotated[float, Field(gt=0)]], None),  # noqa: UP045 -- the typing spelling is the property under test
    ],
    ids=lambda v: str(v).replace("typing.", ""),
)
def test_a_new_numeric_analysis_field_without_a_unit_is_caught(annotation, default, monkeypatch):
    """Property: a new analysis field whose annotation can hold a number and has no table entry fails the census test.

    Class: the annotation spellings a developer adds a numeric field with, including the ``Literal`` integer this
    change is about (``Literal[0, 1]``). NEW CAPABILITY: the plausible wrong implementation is the earlier census
    (``annotation in (int, float)`` over the outermost generic's arguments), which is red here on every ``Literal`` row.
    Why it is required: the failed consolidation came from exactly such a field. What kills it: a census blind to the
    row's spelling. A second correct implementation: a census over the JSON schema.
    """
    monkeypatch.setattr(
        "hhemt.config.analysis.analysis_config", _model_with("zz_reviewer_new_field", annotation, default)
    )
    with pytest.raises(AssertionError) as caught:
        _census_module().test_every_scalar_numeric_analysis_field_has_a_unit()
    assert "zz_reviewer_new_field" in str(caught.value)


@pytest.mark.parametrize(
    "annotation, default",
    [
        (bool, False),
        (bool | None, None),
        (Literal[True, False], True),
        (Literal["a", "b"], "a"),
        (str, ""),
        (str | None, None),
        (Path, Path(".")),
        (Optional[Path], None),  # noqa: UP045 -- the typing spelling is the property under test
        (list[Path], []),
        (tuple[int, ...] | None, None),
        (dict[str, float] | None, None),
    ],
    ids=lambda v: str(v).replace("typing.", ""),
)
def test_a_new_non_numeric_analysis_field_is_not_forced_to_carry_a_unit(annotation, default, monkeypatch):
    """Property: a bool, string, path or container field added without a unit leaves the census test green.

    Class: the annotation spellings that cannot become a numeric column of the ``parameters`` dataset. REGRESSION
    against over-firing: a census that called any of them numeric would force a meaningless unit into the table (and
    the reverse assertion would then fail). The must-fire control is the test above, which runs this same harness and
    goes red on a numeric row. Why it is required: ``np.issubdtype(bool, np.number)`` is False, so
    ``stamp_parameter_units``
    never reads a bool column. What kills it: counting ``bool`` (an ``int`` subclass) as ``int``. A second correct
    implementation: a census over the JSON schema.
    """
    monkeypatch.setattr(
        "hhemt.config.analysis.analysis_config", _model_with("zz_reviewer_new_field", annotation, default)
    )
    _census_module().test_every_scalar_numeric_analysis_field_has_a_unit()


def test_a_table_entry_for_a_field_that_stores_no_number_is_caught(monkeypatch):
    """Property: the census test's second assertion refuses a unit entry whose field stores no number.

    Class: a table key naming a string field (the opposite drift of a deleted entry). Why it is required: the second
    assertion is the half of the equality the request's wording ("covers") does not mention. What kills it: dropping
    the second assertion. A second correct implementation: a table keyed by a field's metadata.
    """
    monkeypatch.setattr(
        cf, "_ANALYSIS_FIELD_UNITS", {**_ANALYSIS_FIELD_UNITS, "open_water_outfall_representation": "1"}
    )
    with pytest.raises(AssertionError) as caught:
        _census_module().test_every_scalar_numeric_analysis_field_has_a_unit()
    assert "open_water_outfall_representation" in str(caught.value)


def test_the_census_equals_an_independent_json_schema_census():
    """Property: the annotation census names exactly the fields a JSON-schema reading calls numeric.

    Class: all 128 fields of ``analysis_config`` (the whole population, not a sample). A second implementation that
    shares no code with the census's ``typing`` walk is the instrument that can disagree with it. Why it is required:
    the figure the unit table is held to is only as good as the census's notion of numeric; ``bool``, ``str``,
    ``Path | None`` and the ``Literal`` of strings are in the population and must land on the non-numeric side.
    What kills it: a census that counts a bool or a ``Literal`` of strings, or misses a ``Literal`` of numbers. A second
    correct implementation is this test's own schema reader.
    """
    props = analysis_config.model_json_schema()["properties"]
    by_schema = {name for name, prop in props.items() if _schema_says_numeric(prop)}
    by_census = _census_module()._numeric_analysis_fields()
    assert len(props) == len(analysis_config.model_fields), "control: the schema lists every field"
    assert by_schema, "control: the schema reading found numeric fields, so equality is not two empty sets"
    assert by_census == by_schema, (sorted(by_census - by_schema), sorted(by_schema - by_census))
    assert "swmm_coupling_lag" in by_census
    assert not by_census & {n for n, f in analysis_config.model_fields.items() if f.annotation in (bool, str, Path)}


def test_the_hpc_alias_targets_are_not_numeric_so_no_hpc_column_needs_a_unit():
    """Property: every field an ``hpc.{alias}`` overlay column resolves to is outside the numeric census.

    Class: the two aliases of ``_HPC_ALIAS_TO_ANALYSIS_FIELD`` (the partition selectors). Why it is required: the
    ``hpc.`` path of ``_parameter_column_field`` resolves to the field and the table is then asked about it; an alias
    to a numeric field would need a unit. What kills it: adding a numeric alias without a table entry. A second correct
    implementation: asserting the alias target's annotation is ``str``.
    """
    from hhemt.sensitivity_analysis import _HPC_ALIAS_TO_ANALYSIS_FIELD

    numeric = _census_module()._numeric_analysis_fields()
    assert _HPC_ALIAS_TO_ANALYSIS_FIELD, "control: the alias table is not empty"
    for alias, field in _HPC_ALIAS_TO_ANALYSIS_FIELD.items():
        assert field in analysis_config.model_fields, alias
        assert field not in numeric, alias


def _phase_b_parameters() -> xr.Dataset:
    """A ``parameters`` dataset with the Phase B column set, built the way the writer builds it from ``df_setup``."""
    df = pd.DataFrame(
        {
            "analysis.swmm_coupling_lag": [0, 1, 0, 1],
            "analysis.swmm_coupling_interval": [1, 1, 4, 4],
            "analysis.swmm_report_step_s": [60.0, 120.0, float("nan"), 120.0],
            "analysis.n_gpus": [1, 2, 1, 2],
            "analysis.hpc_time_min_per_sim": [30, 30, 60, 60],
            "analysis.open_water_outfall_representation": ["driven_stage", "crown_rim", "driven_stage", "crown_rim"],
        },
        index=pd.Index(["S1", "S2", "S3", "S4"], name="sa_id"),
    )
    return xr.Dataset.from_dataframe(df)


def test_the_phase_b_column_set_stamps_every_numeric_column_and_leaves_the_string_alone():
    """Property: the Phase B ``parameters`` dataset stamps without error, each numeric column gets its own unit, and
    the string column gets none.

    Class: the six columns the Phase B matrices write, with ``swmm_coupling_lag`` taking both of its values (0 and 1)
    and ``swmm_report_step_s`` carrying a missing cell. REGRESSION: before the table entry the call raises the
    ``ValueError`` of the failed consolidation. Why it is required: this is the call ``experiment_consolidation`` makes
    at ``sensitivity_analysis.py:567``. What kills it: dropping the entry, or stamping the string column. A second
    correct implementation: stamping from the field's own metadata.
    """
    ds = stamp_parameter_units(_phase_b_parameters(), _parameter_column_field)
    expected = {
        "analysis.swmm_coupling_lag": "1",
        "analysis.swmm_coupling_interval": "1",
        "analysis.swmm_report_step_s": "s",
        "analysis.n_gpus": "1",
        "analysis.hpc_time_min_per_sim": "min",
    }
    for column, unit in expected.items():
        assert ds[column].attrs.get("units") == unit, column
    assert "units" not in ds["analysis.open_water_outfall_representation"].attrs
    assert str(ds["analysis.swmm_coupling_lag"].dtype).startswith("int"), "control: the lag column is an integer dtype"


def test_the_phase_b_column_set_is_refused_without_the_lag_entry_with_the_consolidation_message(monkeypatch):
    """Property: the same dataset raises the message of the failed consolidation when only the lag entry is missing.

    Class: the single-entry mutant of the entry this change adds. NEW CAPABILITY control for the test above: a stamp
    that never refused would pass it. Why it is required: it ties the green test to the red the Rivanna jobs showed.
    What kills it: a stamp that skips integer columns. A second correct implementation: any stamp that refuses a field
    that declares no unit.
    """
    monkeypatch.setattr(
        cf, "_ANALYSIS_FIELD_UNITS", {k: v for k, v in _ANALYSIS_FIELD_UNITS.items() if k != "swmm_coupling_lag"}
    )
    with pytest.raises(ValueError, match=r"analysis\.swmm_coupling_lag.*swmm_coupling_lag.*numeric and has no unit"):
        stamp_parameter_units(_phase_b_parameters(), _parameter_column_field)


def test_the_unit_of_the_lag_is_the_unit_of_the_interval():
    """Property: the lag and the interval are both a count of TRITON steps and carry the same dimensionless unit.

    Class: the two ``swmm_coupling_*`` fields. Why it is required: the interval's description is "the number of TRITON
    steps per SWMM step" and the lag's is "1 TRITON step behind"; both reach the solver cfg as bare integers
    (``triton_cfg.py`` ``COUPLING_INTERVAL`` and ``COUPLING_LAG``) and neither is a time. What kills it: a time unit
    on either. A second correct implementation: a table that names the unit ``"steps"``, which this table does not
    use for any count (every count here is ``"1"``).
    """
    assert _ANALYSIS_FIELD_UNITS["swmm_coupling_lag"] == "1"
    assert _ANALYSIS_FIELD_UNITS["swmm_coupling_lag"] == _ANALYSIS_FIELD_UNITS["swmm_coupling_interval"]
    assert {u for k, u in _ANALYSIS_FIELD_UNITS.items() if "count" in k or k.startswith("n_")} == {"1"}
