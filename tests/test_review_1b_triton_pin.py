"""Reviewer tests for the solver pin: the faba2232 admission test must not pin the size of the registry.

The coder's `test_no_registered_defect_is_indeterminate_at_faba2232` asserts `len(REGISTRY) == 8`.
Admitting a ninth defect that is correctly registered at faba2232 is the ordinary way the registry
grows, and it must not turn that test red.
"""

from __future__ import annotations

import dataclasses

from hhemt import model_defects
from tests import test_triton_pin_faba2232 as pin


def test_a_ninth_defect_registered_absent_at_faba2232_leaves_the_admission_test_green(monkeypatch):
    """The coder's no-indeterminate test stays green when the registry gains a correctly registered defect.

    Property: the admission test asserts what the shape specifies (no registered defect reads
    indeterminate at faba2232) and nothing about how many defects are registered.

    Class: every registry that is the shipped eight plus any further defect whose `known_absent_in`
    carries faba2232 and whose `also_present_in` does not. The input here is the first of the
    shipped entries re-keyed under a new id, which keeps its `fixed_in` so the unregistered-sha
    control still applies to it.

    Why it is required: `REGISTRY` is the file's growth point (a defect is added when a new model
    bug is found), and the test would then red on a registry that satisfies the shape, so the next
    editor either deletes the assertion or hand-bumps the count, and neither is a check on
    faba2232.

    What kills it (the named change to production code): none; the change is to the coder's test,
    which drops the `len(REGISTRY) == 8` assertion (the loop below it already fails on any defect
    that reads indeterminate or present at faba2232).

    A second correct implementation under which it passes: a test that asserts only the loop and
    the `SHA_ORNL_COUPLING_DECLARATION_PIN == faba2232` equality, or one that checks that the eight
    shipped defect ids are all present in the registry rather than that the registry has eight.

    Class: NEW CAPABILITY. Plausible wrong implementation: the coder's `len(REGISTRY) == 8`; red
    under it, output pasted in the reviewer's round file. Reachability: a registry of nine entries
    reaches the failing assertion; artifact: tests/test_triton_pin_faba2232.py line 37. Both-states
    anchor: the assertion turns on the registry's membership in both the current and a corrected
    test, not on wording a change introduces. Invocation isolation (second arm): dropping the
    `len(REGISTRY) == 8` line from a copy of the coder's test turns this test green.
    """
    first = model_defects.REGISTRY[0]
    ninth = dataclasses.replace(
        first,
        defect_id="TRITON-REVIEW-HYPOTHETICAL-NINTH",
        known_absent_in=first.known_absent_in | {pin._FABA2232},
        also_present_in=frozenset(),
    )
    grown = (*model_defects.REGISTRY, ninth)
    monkeypatch.setattr(pin, "REGISTRY", grown)

    pin.test_no_registered_defect_is_indeterminate_at_faba2232()
