"""Unit tests for the ADR-10 field_bucket classifier.

Totality proof composes two set-equality guards:
  (1) test_all_path_fields_have_policy (tests/test_bundle.py): every config
      Path field is a _PATH_FIELD_POLICY key.
  (2) test_policy_to_bucket_is_total (below): every PathPolicy member is a
      _POLICY_TO_BUCKET key.
(1) composed with (2) proves field_bucket is total over config Path fields.
"""
from __future__ import annotations

import pytest

from hhemt.bundle._path_policy import _PATH_FIELD_POLICY, PathPolicy
from hhemt.config.reprex_taxonomy import _POLICY_TO_BUCKET, field_bucket

_VALID_BUCKETS = {"user", "hpc", "experiment"}

def test_policy_to_bucket_is_total() -> None:
    """Every PathPolicy member maps to a bucket (the enum-coverage guard).

    This is half (2) of the totality proof; half (1) is
    test_all_path_fields_have_policy in tests/test_bundle.py.
    """
    assert set(_POLICY_TO_BUCKET) == set(PathPolicy)
    assert set(_POLICY_TO_BUCKET.values()) <= _VALID_BUCKETS

@pytest.mark.parametrize("field_name", sorted(_PATH_FIELD_POLICY))
def test_field_bucket_returns_valid_bucket(field_name: str) -> None:
    """Behavioral: every config Path field classifies to a valid bucket."""
    assert field_bucket(field_name) in _VALID_BUCKETS

def test_field_bucket_raises_on_non_path_field() -> None:
    """Contract: a non-path / unknown field raises KeyError."""
    with pytest.raises(KeyError):
        field_bucket("case_name")
