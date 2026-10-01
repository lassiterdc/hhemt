"""The per-pair CF-1.13 section 7.3 oracle in `tests/_cf_cell_methods_grammar.py`, exercised.

WHY THE ORACLE NEEDS ITS OWN TESTS. It is the instrument every `cell_methods` assertion in this
suite now resolves through, so a silently-wrong oracle makes every one of those assertions agree
with it rather than with CF. Its refusal messages are also stated as SCOPE BOUNDARIES rather than
as malformedness verdicts, and a message nothing exercises is a message that drifts.

WHAT IS DELIBERATELY NOT ASSERTED HERE. The census integers in `cell_method_pairs`'s docstring
(36 / 14 / 22) are a property of the CF corpus and of the stated extractor, not of this module, and
re-deriving them needs that corpus on disk. The docstring states the instrument so a reader can
re-derive them; this file asserts the PARSER's behaviour on one exemplar per refused family instead,
which is what would actually move if the parser drifted.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ._cf_cell_methods_grammar import (
    CF_STANDARD_NAMES_ADMITTED_AS_CELL_METHODS_NAMES,
    cell_method_pairs,
    cell_methods_classes,
    cell_methods_name_class,
)


def _summary_like() -> xr.Dataset:
    """The shipped shape: a retained size-one `Rank` DIMENSION carrying NO coordinate.

    Measured on `_aggregate_perf_summary`'s real output: `"Rank" in ds.dims` is True while
    `"Rank" in ds.coords` and `"Rank" in ds.variables` are both False, because
    `max(dim="Rank", keepdims=True)` leaves a dimension without a coordinate.
    """
    return xr.Dataset({"Total": (("Rank",), np.array([1.0]))})


# --------------------------------------------------------------------------------------
# The pair split -- the whole reason a colon split was retired.
# --------------------------------------------------------------------------------------


def test_the_two_pair_value_splits_into_two_pairs():
    assert cell_method_pairs("time: sum Rank: maximum") == [("time", "sum"), ("Rank", "maximum")]


def test_a_colon_split_cannot_express_the_two_pair_value():
    """The measurement the retired predicate rested on, pinned so nobody reinstates it.

    The middle element carries the FIRST pair's method and the SECOND pair's name together, so the
    colon split yields no pair structure at all rather than the wrong element of one.
    """
    assert "time: sum Rank: maximum".split(":") == ["time", " sum Rank", " maximum"]


def test_a_single_pair_value_splits_into_one_pair():
    assert cell_method_pairs("area: sum") == [("area", "sum")]


# --------------------------------------------------------------------------------------
# The FIVE refused families, one exemplar each. Every message is a SCOPE boundary.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "family"),
    [
        ("time: standard_deviation (interval: 1 day)", "parenthetical qualifier (CF 7.3.2)"),
        ("lat: lon: standard_deviation", "multi-name pair (CF 7.3.1)"),
        ("area: mean where sea_ice over sea", "portion clause (CF 7.3.3)"),
        ("time: mean within days time: mean over days", "climatological qualifier (CF 7.4)"),
        ("area: anomaly_wrt areamin", "method with operand (CF 7.5)"),
    ],
)
def test_each_refused_family_raises_rather_than_mis_parsing(value, family):
    """A widening CF admits and this parser does not must FAIL LOUDLY, never mis-parse.

    Each exemplar is a verbatim member of the corpus census. The `ValueError` is the scope
    boundary: a NEW corpus value landing here is a signal to widen the parser, not a signal that
    the value is malformed.
    """
    with pytest.raises(ValueError):
        cell_method_pairs(value)
    assert family  # the family label is the reason this row exists; keep it in the node id


def test_an_empty_value_is_refused():
    with pytest.raises(ValueError):
        cell_method_pairs("")


def test_a_method_with_no_name_before_it_is_refused():
    with pytest.raises(ValueError):
        cell_method_pairs("maximum")


# --------------------------------------------------------------------------------------
# The four name classes, and the one that is unexercised in production.
# --------------------------------------------------------------------------------------


def test_a_retained_dimension_takes_class_1():
    ds = _summary_like()
    assert cell_methods_name_class("Rank", ds["Total"], ds) == "dimension"


def test_a_scalar_coordinate_takes_class_2_even_though_no_corpus_value_does():
    """Class 2 ships for CORRECTNESS, not coverage -- so its arm is exercised HERE and only here.

    No value this repository stamps takes it: the summary's carrier is a retained DIMENSION,
    chosen over a scalar because a differing scalar is promoted to `dims=("event_iloc",)` at the
    consolidation concat. Omitting the arm would make this oracle refuse a legitimate CF form, and
    the arm is the tripwire for a regression back to the scalar carrier.
    """
    ds = xr.Dataset({"Total": ((), 1.0)}).assign_coords(Rank=1)
    assert ds["Rank"].ndim == 0
    assert cell_methods_name_class("Rank", ds["Total"], ds) == "scalar_coordinate"


def test_area_takes_class_4():
    ds = xr.Dataset({"vol": ((), 0.0)})
    assert cell_methods_name_class("area", ds["vol"], ds) == "area"


def test_time_takes_class_3_because_no_axis_is_named_time():
    """CF 7.3.4's precondition is an ABSENCE, and this arm asserts the absence rather than a presence."""
    ds = _summary_like()
    assert "time" not in ds["Total"].dims and "time" not in ds.coords
    assert cell_methods_name_class("time", ds["Total"], ds) == "standard_name"


def test_class_3_is_refused_when_a_dimension_of_that_name_exists():
    """7.3.4: the standard-name form "cannot be used ... if the name of a dimension ... is identical"."""
    ds = xr.Dataset({"Total": (("time",), np.array([1.0, 2.0]))})
    # Class 1 claims it first, which is the mutual exclusion stated positively.
    assert cell_methods_name_class("time", ds["Total"], ds) == "dimension"


def test_class_3_is_refused_when_a_coordinate_of_that_name_exists_on_another_variable():
    """THE GUARD'S EXTRA WIDTH, exercised so the disclosure in its docstring is falsifiable.

    CF's NORMATIVE sentence (corpus line 3164) forbids class 3 only when a DIMENSION or SCALAR
    coordinate variable shares the name; a 1-d coordinate variable not on this variable's dims
    would be PERMITTED by it. This oracle refuses that case anyway -- the extra width is this
    module's conservative choice and not a CF-sanctioned option. Narrowing it would be
    CF-conformant, which is exactly why the width needs a test rather than a comment.
    """
    ds = xr.Dataset(
        {"Total": (("Rank",), np.array([1.0]))},
        coords={"time": ("time", np.array([0.0, 1.0]))},
    )
    assert "time" not in ds["Total"].dims
    assert "time" in ds.coords and ds["time"].ndim == 1
    assert cell_methods_name_class("time", ds["Total"], ds) is None


def test_an_unknown_name_takes_no_class():
    ds = _summary_like()
    assert cell_methods_name_class("timestep_min", ds["Total"], ds) is None


def test_the_class_3_vocabulary_is_an_explicit_enumeration():
    """Not derived from the corpus's declared `standard_name` values, and the reason is a category gap.

    A `standard_name` attribute names a VARIABLE's quantity; a class-3 cell_methods name names an
    AXIS. Deriving one from the other would make the vocabulary move whenever an unrelated variable
    gained a standard_name.
    """
    assert CF_STANDARD_NAMES_ADMITTED_AS_CELL_METHODS_NAMES == frozenset({"time"})


# --------------------------------------------------------------------------------------
# The class SEQUENCE, which is what distinguishes a two-class value from a renamed dimension.
# --------------------------------------------------------------------------------------


def test_the_class_sequence_is_returned_in_pair_order():
    ds = _summary_like()
    assert cell_methods_classes("time: sum Rank: maximum", ds["Total"], ds, where="Total") == [
        "standard_name",
        "dimension",
    ]


def test_the_class_sequence_discriminates_a_renamed_dimension_from_the_shipped_value():
    """A caller asserting only "it resolved" is satisfied by a value that names one axis twice.

    `"Rank: sum Rank: maximum"` resolves on BOTH pairs, so a predicate that only checked
    resolvability would accept it. The SEQUENCE does not.
    """
    ds = _summary_like()
    assert cell_methods_classes("Rank: sum Rank: maximum", ds["Total"], ds, where="Total") == [
        "dimension",
        "dimension",
    ]


def test_an_unresolvable_pair_fails_with_the_offending_name_in_the_message():
    ds = _summary_like()
    with pytest.raises(AssertionError, match="none of CF-1.13"):
        cell_methods_classes("timestep_min: sum Rank: maximum", ds["Total"], ds, where="Total")


def test_the_oracle_imports_nothing_from_the_module_it_audits():
    """The independence rule, asserted mechanically rather than trusted.

    Sourcing the admissibility rule from `cf_conventions` would make a wrong rule agree with itself
    at both ends, and the disagreement every consuming test exists to surface becomes structurally
    unreachable.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).with_name("_cf_cell_methods_grammar.py").read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["__future__"], (
        f"the oracle imports {imported}; it must import nothing but __future__, and in particular "
        "nothing from hhemt.cf_conventions -- the module it audits"
    )
