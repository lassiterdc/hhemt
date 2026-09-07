"""Direct tests of the compile/execution venue predicate.

WHY THIS FILE EXISTS, and it is not the usual reason. Every test that reaches the
Snakemake-shell SWMM route for real is `compile_tier` or
`requires_snakemake_subprocess`, and Gate A deselects both; Gate B declares a
permitted venue, so the guard correctly permits there. **No green run this project
performs can demonstrate that the refusal works.** These tests are the only
instrument that goes red if the predicate regresses.

They IMPORT the single implementation rather than restating it. A test-side copy of
a production predicate is the defect this repository already carries at
tests/utils_for_testing.py:87-101, where five tests assert a duplicate of the SWMM
version pair and the production body is exercised by nothing.
"""

import hhemt._compile_venue as venue


def test_refuses_in_test_session_with_no_venue_declared(monkeypatch):
    """Arm 1 — the defect this whole shape exists to close."""
    monkeypatch.delenv(venue.COMPILE_VENUE_ENV, raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_compile_venue.py::probe (call)")
    assert venue.swmm_execution_refused() is True


def test_permits_in_test_session_on_a_declared_venue(monkeypatch):
    """Arm 2 — Gate B and Rivanna must keep working."""
    monkeypatch.setenv(venue.COMPILE_VENUE_ENV, "toolchain")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_compile_venue.py::probe (call)")
    assert venue.swmm_execution_refused() is False


def test_permits_outside_a_test_session_whatever_the_flag_says(monkeypatch):
    """Arm 3 — the public user, who must never be able to trip this guard.

    Parameterised over the flag deliberately: the first conjunct alone decides,
    so no value of the venue token may make a non-test process refuse.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    for value in ("toolchain", "", "nonsense-value"):
        monkeypatch.setenv(venue.COMPILE_VENUE_ENV, value)
        assert venue.swmm_execution_refused() is False
    monkeypatch.delenv(venue.COMPILE_VENUE_ENV, raising=False)
    assert venue.swmm_execution_refused() is False


def test_unrecognised_venue_value_does_not_relax(monkeypatch):
    """The token RELAXES; an unrecognised value must relax nothing."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_compile_venue.py::probe (call)")
    for value in ("", "  ", "TOOLCHAIN-typo", "rivanna", "1", "true"):
        monkeypatch.setenv(venue.COMPILE_VENUE_ENV, value)
        assert venue.swmm_execution_refused() is True


def test_declared_venue_is_case_and_whitespace_insensitive(monkeypatch):
    """The reader normalises, so a shell quirk cannot silently arm a permitted venue."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_compile_venue.py::probe (call)")
    for value in ("Toolchain", "  toolchain  ", "TOOLCHAIN"):
        monkeypatch.setenv(venue.COMPILE_VENUE_ENV, value)
        assert venue.swmm_execution_refused() is False
