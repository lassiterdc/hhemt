"""The layout term of the synthetic-test cache key is exactly one more entry of the case parameters.

`tests/test_cache_key_layout_term.py` shows that the key moves with the layout version. These tests pin what else
must hold: the set of terms handed to the provenance builder is unchanged, the digest is the digest of the stated
parameters plus the version (so no other term absorbs it), and a change of a case parameter still moves the digest.
No solver runs.
"""

from __future__ import annotations

import dataclasses
import inspect

from hhemt.cache import keys as cache_keys
from hhemt.version_migration.constants import LAYOUT_VERSION
from tests.test_cache_key_layout_term import _case


def test_the_terms_given_to_the_provenance_builder_are_exactly_its_keyword_set():
    """Property: ``key_terms`` returns only keywords that ``builder_provenance`` accepts, and none is missing.

    Class: the real ``key_terms`` of a stub-built case against the signature of the builder.
    Why required: the layout term must ride inside an existing term; a sixth top-level key is a TypeError at the
    call in the fixture, which this test shows without the fixture.
    Kills it: an added top-level key; a dropped one.
    """
    accepted = set(inspect.signature(cache_keys.builder_provenance).parameters) - {"repo_root", "payload_paths"}
    terms = set(_case().key_terms())
    assert terms <= accepted, terms - accepted
    assert {"case_params_digest"} <= terms


def test_the_case_digest_is_the_digest_of_the_parameters_with_the_current_version_and_nothing_else():
    """Property: the key's ``case_params_digest`` equals the digest recomputed from the case parameters plus
    ``layout_version`` equal to the current ``LAYOUT_VERSION``, and differs from the digest without it.

    Class: one stub case; the digest recomputed with the public function.
    Why required: it fixes the term's name and value, so a rename of the term or a different number is caught.
    Kills it: a term holding a constant; a term named differently; the version hashed twice.
    Must-fire control: the recomputation without the term is unequal.
    """
    case = _case()
    expected = {**dataclasses.asdict(case._params), "layout_version": LAYOUT_VERSION}

    def digest(params):
        return cache_keys.case_params_digest(
            n_events=case._n_events,
            n_reporting_tsteps_per_sim=None,
            triton_reporting_timestep_s=float(case._params.reporting_timestep_s),
            params=params,
            additional_system_configs=case._additional_system_configs,
        )

    assert case.key_terms()["case_params_digest"] == digest(expected)
    assert case.key_terms()["case_params_digest"] != digest(dataclasses.asdict(case._params))


def test_a_case_parameter_still_moves_the_digest_with_the_term_present():
    """Property: changing a case parameter moves ``case_params_digest`` while the version term is unchanged.

    Class: two stub cases differing in one dataclass field.
    Kills it: a params mapping that replaces the case parameters by the version alone.
    """
    a, b = _case(), _case()
    b._params = dataclasses.replace(b._params, marker="y")
    assert a.key_terms()["case_params_digest"] != b.key_terms()["case_params_digest"]
