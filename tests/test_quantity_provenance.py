"""Guards for `cf_conventions._QUANTITY_PROVENANCE` (Iteration 10, item I).

The computed-quantity descriptor table is hand-authored, so the risk it carries is
DRIFT: a variable gains a CF entry and never gains a descriptor, and the metadata
report renders an em-dash where a reader expects the operation. These tests make that
a CI failure instead of a reading error.

They deliberately do NOT assert that any particular operation string is *correct* --
no test can check prose against a dask expression. They assert coverage, vocabulary,
and non-emptiness, which is the part a machine can decide.
"""

from __future__ import annotations

import pytest

from hhemt.cf_conventions import (
    _CF_VARIABLE_MAP,
    _QUANTITY_PROVENANCE,
    quantity_provenance,
)

#: The controlled vocabulary for `spatial_representation`. Kept small on purpose: the
#: column exists to tell a reader what geometry a value describes, and a free-text
#: column would defeat that within two additions.
_SPATIAL_VOCAB = frozenset(
    {
        "grid cell",
        "point (node)",
        "line (conduit)",
        "whole domain (scalar)",
    }
)

_REQUIRED_KEYS = frozenset({"spatial_representation", "source_variables", "operation", "reduced_coordinate"})


def test_quantity_provenance_is_in_bijection_with_the_cf_variable_map():
    """Every CF-mapped variable has a descriptor, and vice versa.

    This is the drift guard the descriptor columns depend on. A variable present in
    one table and absent from the other is exactly the state that publishes a report
    row asserting a long_name and a unit while saying nothing about how the number was
    computed.
    """
    cf_only = sorted(set(_CF_VARIABLE_MAP) - set(_QUANTITY_PROVENANCE))
    prov_only = sorted(set(_QUANTITY_PROVENANCE) - set(_CF_VARIABLE_MAP))
    assert not cf_only, (
        f"CF-mapped variables with no computed-quantity descriptor: {cf_only}. "
        "Add an entry to cf_conventions._QUANTITY_PROVENANCE grounded in the "
        "expression that computes the variable, not in its cell_methods string."
    )
    assert not prov_only, (
        f"Descriptors for variables absent from _CF_VARIABLE_MAP: {prov_only}. "
        "A descriptor for a variable the pipeline does not emit is published as a "
        "false claim about the data (the same failure the 2026-07-21 removal fixed)."
    )


@pytest.mark.parametrize("var_name", sorted(_QUANTITY_PROVENANCE))
def test_each_descriptor_carries_every_required_key_non_empty(var_name):
    entry = _QUANTITY_PROVENANCE[var_name]
    missing = sorted(_REQUIRED_KEYS - set(entry))
    assert not missing, f"{var_name} descriptor is missing {missing}"
    blank = sorted(k for k in _REQUIRED_KEYS if not str(entry[k]).strip())
    assert not blank, f"{var_name} descriptor has blank {blank}"


@pytest.mark.parametrize("var_name", sorted(_QUANTITY_PROVENANCE))
def test_spatial_representation_uses_the_controlled_vocabulary(var_name):
    value = _QUANTITY_PROVENANCE[var_name]["spatial_representation"]
    assert value in _SPATIAL_VOCAB, (
        f"{var_name} declares spatial_representation={value!r}, which is outside the "
        f"controlled vocabulary {sorted(_SPATIAL_VOCAB)}. Widen the vocabulary "
        "deliberately if a new geometry is genuinely needed."
    )


def test_reader_returns_a_copy_and_none_for_unknown():
    """`quantity_provenance` must not hand out the module table itself.

    The renderer reads this per row; a caller that mutated the returned dict would
    corrupt every subsequent row on the page.
    """
    entry = quantity_provenance("max_wlevel_m")
    assert entry is not None
    entry["operation"] = "MUTATED"
    assert _QUANTITY_PROVENANCE["max_wlevel_m"]["operation"] != "MUTATED"
    assert quantity_provenance("no_such_variable") is None


def test_last_timestep_variable_is_described_as_a_selection_not_a_reduction():
    """Regression guard for the defect that motivated this table.

    `wlevel_m_last_tstep` carries `cell_methods="timestep_min: point"`, but `point`
    in CF means the variable RETAINS the time dimension with no method applied. The
    computation (process_simulation.py, `summarize_triton_simulation_results`) is
    `ds["wlevel_m"].sel(timestep_min=tsteps.max())` -- a selection. If the descriptor
    ever drifts back to reduction language, the table has reacquired the error it was
    written to correct.
    """
    entry = _QUANTITY_PROVENANCE["wlevel_m_last_tstep"]
    assert "select" in entry["operation"].lower()
    assert "maximum" not in entry["operation"].lower()
