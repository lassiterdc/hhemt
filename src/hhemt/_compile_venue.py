"""Single reader for the two tokens that govern the test tier's build and run venue.

TWO environment tokens are read in this module, each HERE and nowhere else.

``HHEMT_COMPILE_VENUE`` names the VENUE and governs BOTH compilation and SWMM
execution. Its polarity is the whole design: the token RELAXES; it never ARMS. An
absent, misspelt, renamed or unrecognised value therefore lands on the SAFE side at
both read sites -- the compile guard arms, and SWMM execution inside a test session
refuses. An arming flag that goes missing disarms its guard; a relaxing flag that
goes missing tightens it.

``HHEMT_REQUIRE_COMPILE_TIER`` names the ANTI-SILENCE requirement: with it declared,
a skip of the compile tier inside a test session is a hard failure instead of a
quiet ``N skipped``. Its polarity is the opposite one, and that is why the two are
not one predicate: this one ARMS, so an absent value leaves the graceful skip in
place -- which is what every toolchain-less environment needs.

WHY BOTH TOKENS LIVE IN ONE MODULE, stated as the failure it prevents rather than as
tidiness. Two independent ``os.environ.get`` calls for one fact is how one concept
becomes two behaviours without anyone renaming anything. Concretely: a suite runner
that decides whether a compile-bearing plan may proceed and a fixture that decides
whether to skip would each hold their own copy, and the moment one copy spells the
test ``== "1"`` while the other uses a truthiness test, the runner accepts a
compile-bearing plan, the fixtures skip every node in the compile tier, and the pair
the runner recorded reads as satisfied. The output of that run is green and green is
exactly wrong: nothing in it says the session could not compile. This repository
already carries the same shape one predicate over, where the pyswmm/swmm-toolkit
version pair is spelled once in ``tests/utils_for_testing.py`` and again in
``src/hhemt/swmm_runoff_modeling.py``.

Deliberately NOT one composed "may this run proceed" predicate over both tokens: a
caller holds the two independently because it must name WHICH conjunct failed, and a
single composed predicate collapses that diagnostic to "refused".
"""

import os

#: The one environment variable naming the venue. Named COMPILE for the tier it
#: gates, but it names the VENUE and governs execution too -- see the module docstring.
COMPILE_VENUE_ENV = "HHEMT_COMPILE_VENUE"

#: Venues permitted to build the solver and to execute SWMM from a test session.
#: Members name a CAPABILITY, never a machine: the predicate asks whether this
#: venue may build and execute, and a machine name answers a different question.
#: An UNRECOGNISED value is not an error and is not permitted: it relaxes nothing.
PERMITTED_VENUES = frozenset({"toolchain"})


def venue_is_permitted() -> bool:
    """True iff the declared venue is one this project permits to build and execute."""
    return os.environ.get(COMPILE_VENUE_ENV, "").strip().lower() in PERMITTED_VENUES


def compile_tier_required() -> bool:
    """True iff the compile tier MUST run, so a compile-tier skip is a hard failure.

    THE SINGLE READER for ``HHEMT_REQUIRE_COMPILE_TIER``. The suite runner (which
    refuses a compile-bearing plan when this is false) and the compile fixture (which
    turns a skip into a failure when it is true) both call THIS function, so they
    cannot disagree about what "required" means.

    The token and its arming value are spelled as LITERALS rather than lifted into
    constants, and the reason is a measurement rather than style: the instrument that
    verifies the single-reader property greps for this exact expression, so a
    constant would satisfy the code and silently empty the check that proves it. The
    same trade is taken deliberately in ``hhemt.cache.paths.cache_root``.

    ``== "1"`` rather than a truthiness test, because ``HHEMT_REQUIRE_COMPILE_TIER=0``
    is a DECLARATION that reads as falsy -- a user exporting 0 to turn the requirement
    off would instead get the loud failure, which is the opposite of what they asked
    for.
    """
    return os.environ.get("HHEMT_REQUIRE_COMPILE_TIER") == "1"


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
    return refused_in_undeclared_test_venue()


def refused_in_undeclared_test_venue() -> bool:
    """THE conjunction, named for what it tests rather than for who asks.

    Refuse iff inside a pytest test phase AND no permitted venue is declared. Both
    guards consume this: SWMM execution (swmm_runoff_modeling.py) and solver
    compilation (system.py). ``swmm_execution_refused`` remains as a delegation so
    its existing consumers are unaffected -- the predicate is composed HERE and
    nowhere else, which is the property that keeps a future third conjunct from
    reaching one guard and not the other.
    """
    return in_test_session() and not venue_is_permitted()
