"""Tests for the native C flags field ``TRITONSWMM_extra_c_flags``: the compile identity it joins.

``TRITONSWMM_extra_c_flags`` is now an element of the compile-target identity key that
``_build_unique_system_targets`` groups per-member system YAMLs by. The preflight
``_validate_per_member_system_configs`` states its own copy of that key and refuses YAMLs that "collapse to the
same compile target but differ" on a field outside it. This file asserts the two copies agree on the new field.

This is a one-time test, to be deleted at the landing: it guards ``_validate_per_member_system_configs`` in
``hhemt.validation`` and the key of ``_build_unique_system_targets`` in ``hhemt.sensitivity_analysis``.

VENUE: laptop-safe; no solver is compiled or run; every root is under ``tmp_path``.
"""

from __future__ import annotations

import pytest

from hhemt.validation import ValidationResult, _validate_per_member_system_configs
from tests.test_native_c_flags import _targets
from tests.test_per_member_system_configs import (
    _cfg_analysis_stub,
    _master_system_for_test,
    _write_system_yaml,
)


def _preflight(tmp_path, common=None, **b_overrides) -> ValidationResult:
    """Run the per-member preflight over two YAMLs that agree except for ``b_overrides`` on the second.

    ``common`` holds overrides applied to BOTH YAMLs, so a control can hold a field equal at a non-default value.
    """
    common = common or {}
    yaml_a = _write_system_yaml(tmp_path / "a.yaml", target_dem_resolution=10.0, **common)
    yaml_b = _write_system_yaml(tmp_path / "b.yaml", target_dem_resolution=10.0, **{**common, **b_overrides})
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(f"member_id,system_config_yaml\n0,{yaml_a}\n1,{yaml_b}\n")
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master_system_for_test(tmp_path), _cfg_analysis_stub(csv_path), result)
    return result


def test_member_yamls_differing_only_in_the_c_flags_are_not_reported_as_collapsing(tmp_path):
    """Property: two per-member system YAMLs that differ ONLY in ``TRITONSWMM_extra_c_flags`` clear the preflight.

    Input class: a pair of member YAMLs, same ``target_dem_resolution``, every other field equal, the C flags unequal.
    Class: NEW CAPABILITY. The plausible wrong implementation is the one under review: the builder's key gained the
    field and the preflight's own copy of the key did not, so the preflight reports "collapse to the same compile
    target" for rows the builder builds as two targets. Reachability: ``_validate_per_member_system_configs`` runs at
    analysis construction for any sensitivity CSV with a ``system_config_yaml`` column; the pair above is such a CSV.
    Both-states anchor: the control below differs in a field OUTSIDE the key and must still be refused, so the
    assertion turns on key membership and not on a validator that never refuses.
    Why it is required: the refusal's own message says the rows "collapse to the same compile target"; for this
    field they do not (``_build_unique_system_targets`` keys on it), so a sweep of per-member YAMLs that varies the
    flags, the one way this field can differ across members by YAML, is refused for a reason that is false.
    Killed by: restating the field in the validator's key (its ``groups`` key and its ``dedup_key_fields``).
    A second correct implementation that derives both copies from one shared key function still passes, because
    the assertion is on the validator's verdict for the pair.
    """
    result = _preflight(tmp_path, TRITONSWMM_extra_c_flags="-O3")

    assert result.is_valid, [str(e) for e in result.errors]


@pytest.mark.parametrize("flags", ["", "-O3"])
def test_member_yamls_differing_outside_the_key_are_still_reported_as_collapsing(tmp_path, flags):
    """Control for the test above: a divergence in a field outside the key is still refused, with or without flags.

    Input class: same pair, ``constant_mannings`` unequal (a field outside the key), the C flags equal at ``flags``
    in both YAMLs. Class: REGRESSION guard against a repair that disables the check (for example by exempting every
    field). Killed by: a repair that exempts every field from the divergence check.
    """
    result = _preflight(
        tmp_path,
        common={"TRITONSWMM_extra_c_flags": flags, "toggle_use_constant_mannings": True, "constant_mannings": 0.035},
        constant_mannings=0.04,
    )

    assert not result.is_valid
    assert any("constant_mannings" in str(e) for e in result.errors)


@pytest.mark.parametrize(
    "first, second",
    [("-O2", "-O3"), ("-O3", "-O3 -march=native")],
    ids=["different_flags", "shared_leading_token"],
)
def test_rows_with_two_different_nonempty_c_flags_are_distinct_build_targets(tmp_path, monkeypatch, first, second):
    """Property: two rows whose C flags are DIFFERENT NON-EMPTY strings resolve to two build targets.

    Input class: any pair of unequal non-empty flag strings, including a pair that shares its first token.
    Class: NEW CAPABILITY. Plausible wrong implementations: a key built from whether the field is set (rows with
    ``-O2`` and ``-O3`` would share one binary), or from a prefix of it (rows ``-O3`` and ``-O3 -march=native`` would).
    Reachability: both arise in an optimisation sweep, which is what the field exists for. Both-states anchor: the
    existing distinct-targets test uses only ``""`` against a non-empty value, which those wrong keys also separate,
    and the existing agreement control keeps equal values together; neither separates the wrong keys from the right one.
    Why it is required: rows that disagree on the flags are different builds, and one binary would be used for both.
    Killed by: keying on ``bool(flags)`` or on the first token. A second correct implementation that keys on the
    whole stripped string through another expression still passes.
    """
    targets = _targets(tmp_path, monkeypatch, [first, second])

    assert sorted(t.analysis_ids for t in targets) == [["m0"], ["m1"]]
