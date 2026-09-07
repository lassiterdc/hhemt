"""The compile-entry-point guard: replace every compile entry point with a refusal.

RELOCATED from tests/conftest.py so the REPO-ROOT conftest can import it. The
arming DECISION lives there (a fail-closed default relaxed only by
HHEMT_COMPILE_VENUE); this module owns only the MECHANISM.
"""

#: Every public compile entry point plus every shared backend, as of LAYOUT_VERSION 22.
#: `_compile_backend_locked`, `_compile_SWMM_locked` and `_compile_triton_only_backend_locked`
#: are reached ONLY from the members below (system.py:1665 and siblings), so they are covered
#: transitively. Re-derive with: grep -n "    def .*compile" src/hhemt/system.py
COMPILE_ENTRY_POINTS = (
    "compile_TRITON_SWMM",
    "_compile_backend",
    "compile_TRITON_only",
    "_compile_triton_only_backend",
    "compile_SWMM",
)


def _forbid_compile(name):
    """Return a stand-in for a compile entry point that refuses instead of building."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"System.{name} was reached while the compile guard was armed. A test that "
            "compiles belongs in the compile tier. Either mark it "
            "`@pytest.mark.compile_tier`, or make its dependency on "
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
