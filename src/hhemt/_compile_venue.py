"""Single reader for the compile/execution VENUE declaration.

ONE token, ``HHEMT_COMPILE_VENUE``, names the VENUE and governs BOTH compilation
and SWMM execution. It is read HERE and nowhere else: two independent
``os.environ.get`` calls for one fact is how one concept becomes two behaviours
without anyone renaming anything, and this repository already carries that exact
mistake at ``tests/utils_for_testing.py`` versus this package's SWMM version
predicate.

POLARITY, and it is the whole design. The token RELAXES; it never ARMS. An
absent, misspelt, renamed or unrecognised value therefore lands on the SAFE
side at both read sites -- the compile guard arms, and SWMM execution inside a
test session refuses. An arming flag that goes missing disarms its guard; a
relaxing flag that goes missing tightens it.
"""

import os

#: The one environment variable. Named COMPILE for the tier it gates, but it
#: names the VENUE and governs execution too -- see the module docstring.
COMPILE_VENUE_ENV = "HHEMT_COMPILE_VENUE"

#: Venues permitted to build the solver and to execute SWMM from a test session.
#: Members name a CAPABILITY, never a machine: the predicate asks whether this
#: venue may build and execute, and a machine name answers a different question.
#: An UNRECOGNISED value is not an error and is not permitted: it relaxes nothing.
PERMITTED_VENUES = frozenset({"toolchain"})


def venue_is_permitted() -> bool:
    """True iff the declared venue is one this project permits to build and execute."""
    return os.environ.get(COMPILE_VENUE_ENV, "").strip().lower() in PERMITTED_VENUES


def in_test_session() -> bool:
    """True iff we are inside a pytest test PHASE (setup, call, or teardown).

    ``PYTEST_CURRENT_TEST`` is set by ``_pytest/runner.py`` for each of the three
    runtest phases and popped unconditionally afterwards, so this is False at
    module import, at collection, and in every non-pytest process -- including
    every process a user of this package will ever run.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def swmm_execution_refused() -> bool:
    """The CONJUNCTIVE predicate: refuse iff in a test session AND venue not permitted.

    The first conjunct is intrinsically observable and needs no arming, which is
    what makes a public user's session permitted BY CONSTRUCTION rather than by
    configuration. The second only ever relaxes.
    """
    return in_test_session() and not venue_is_permitted()
