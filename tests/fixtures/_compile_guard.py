"""The compile-entry-point guard: replace every compile entry point with a refusal.

RELOCATED from tests/conftest.py so the REPO-ROOT conftest can import it. The
arming DECISION lives there (a fail-closed default relaxed only by
HHEMT_COMPILE_VENUE); this module owns only the MECHANISM.
"""

#: The PUBLIC compile entry points, as of LAYOUT_VERSION 22 -- and nothing below them.
#:
#: WHY THIS LAYER AND NOT A DEEPER ONE. The criterion is structural: guard the layer
#: whose members have NO BEHAVIOUR OTHER THAN the guarded operation. `compile_TRITON_SWMM`
#: and its siblings mean "compile a solver" and mean nothing else. Everything below them
#: does something else as well -- `_compile_backend` is a lock wrapper, and
#: `_compile_SWMM_locked` assembles a bash script (system.py:2107), chmods it (:2108) and
#: shells out (:2116), so it is a generator and a delegator rather than a builder.
#:
#: The supporting EVIDENCE, which is a census and therefore decays: measured across the
#: suite, legitimate non-compiling callers rise with depth -- 0 at this layer, 2 at the
#: lock wrappers (tests/test_compile_lock.py drives one deliberately), 3 at the `_locked`
#: compilers (tests/test_setup_target_resources.py stubs subprocess.run and drives one),
#: and unbounded at subprocess.run. Two earlier versions of this tuple included wrappers
#: and each refused a legitimate caller. The `hasattr` loop below catches a RENAMED member
#: and never catches a NEW CALLER, so treat the counts as evidence for the structural
#: criterion rather than as the criterion itself.
#:
#: WHAT THIS DOES NOT COVER, stated because it is a real residual rather than an absence.
#: A test that calls a lock wrapper or a `_locked` compiler AND lets it build is not
#: refused here. That set is EMPIRICALLY EMPTY today -- every such caller in the suite
#: either forces a lock timeout, stubs the inner function, or stubs subprocess.run -- and
#: it is empty by measurement, NOT by construction: `pytest --collect-only -m compile_tier`
#: over those two files returns "no tests collected (13 deselected)", so reaching a real
#: build by that route would NOT imply the compile_tier marker and would NOT be deselected
#: from Gate A. A future test of that shape needs this tuple revisited.
#: Re-derive with: grep -n "    def .*compile" src/hhemt/system.py
COMPILE_ENTRY_POINTS = (
    "compile_TRITON_SWMM",
    "compile_TRITON_only",
    "compile_SWMM",
)


def _forbid_compile(name):
    """Return a stand-in for a compile entry point that refuses instead of building."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"System.{name} was reached while the compile guard was armed. A test that "
            "compiles belongs in the compile tier. If this test DOES build a solver, "
            "either mark it `@pytest.mark.compile_tier`, or make its dependency on "
            "`tritonswmm_cpu_compiled` visible to the collection-time fixture closure -- "
            "a `request.getfixturevalue(...)` request is NOT visible, which is why this "
            "guard exists. The guard arms unless HHEMT_COMPILE_VENUE names a permitted "
            "venue, and additionally whenever HHEMT_FORBID_COMPILE=1 is set; to run "
            "compile-bearing tests, declare the venue AND leave HHEMT_FORBID_COMPILE unset."
        )

    return _raise


def arm_compile_guard():
    """Replace every compile entry point with a refusal.

    Scoped to COMPILE and deliberately NOT to `TRITONSWMM_analysis.run`: a guard on
    `run` over-fires on the legitimate `run(dry_run=True)`-over-a-patched-
    `submit_workflow` pattern (three such tests in tests/test_from_scratch_honesty.py),
    while the compile guard fires on genuine leaks only.

    Fails loudly if an entry point is renamed or removed, because a guard that silently
    patches nothing is indistinguishable from a guard that found nothing.
    """
    from hhemt.system import TRITONSWMM_system

    for name in COMPILE_ENTRY_POINTS:
        if not hasattr(TRITONSWMM_system, name):
            raise RuntimeError(
                f"compile guard: TRITONSWMM_system has no attribute {name!r}. The entry-point "
                "census in tests/fixtures/_compile_guard.py is stale; re-derive it before "
                "running any gate."
            )
        setattr(TRITONSWMM_system, name, _forbid_compile(name))
