"""The first tests `check_known_resume_defects` has ever had, plus the (A) disclosure.

WHY THIS MODULE EXISTS AT ALL. Before it, `grep -rn "check_known_resume_defects"
--include=*.py tests/ src/ scripts/` returned five hits outside the WP-2B review carrier:
one definition, one production call site in `validate_analysis`, and three prose mentions
-- two of them in `test_coupled_resume_validity.py` docstrings that NAME this function and
then exercise `model_defects.resolve` instead. So the premise that was pinned was the
RESOLVER's, and the deciding premise -- what the CHECK reports when the resolver cannot
answer -- was unpinned. That is the condition under which the registry's per-sha remedy
lapsed once (`01e95a76`) and was repaired once without anything going red.

TWO PROPERTIES ARE PINNED HERE AND THEY POINT IN OPPOSITE DIRECTIONS. Both are the
developer's rulings and neither is this module's preference:

  * RULED FOR (2026-10-03): the PASSING SUMMARY must disclose its basis, so a registered
    pin and an unregistered one no longer emit the same row. Pinned by
    `test_registered_and_unregistered_rows_differ`.
  * RULED AGAINST (2026-09-26, restated 2026-10-03 as scoped to categories only): there is
    NO third VERDICT category -- an indeterminate registry lookup still returns
    `passed=True, applicable=True`. Pinned by `test_indeterminate_is_not_a_third_verdict`.

A future round that "completes" the disclosure by adding an indeterminate verdict will go
red on the second test, and one that reverts the wording will go red on the first. The two
together are the whole ruling, which is why they live in one module.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION. Nothing here constructs a
`TRITONSWMM_analysis` and no fixture reaches the `*_compiled` family, so
`analysis.py`'s modified-tracked-file guard cannot fire at setup. Screened by fixture
closure (`pytest --collect-only -q --fixtures-per-test` extracting only the fixture-NAME
lines), never by module text.
"""

from __future__ import annotations

import pandas as pd
import pytest

import hhemt.analysis_validation as av
from hhemt.model_defects import REGISTRY, resolve

#: The sha the synthetic tier actually builds (`tests/fixtures/_triton_source_cache.py`).
#: Restated rather than imported so this module stays free of that module's
#: `platformdirs` / `hhemt.utils` import weight; `scripts/check_live_pin_registered.py`
#: is what keeps the restatement honest, and `test_the_restated_pin_is_the_live_pin`
#: below is the local tripwire.
_SHA_REGISTERED = "320b3313c99c3e929e77a258ff6e4b041bae2927"
#: In NO registry set -- the state any branch tip occupies before someone registers it.
_SHA_UNREGISTERED = "deadbeefcafef00ddeadbeefcafef00ddeadbeef"
#: A build that genuinely carries all three RESUME defects. THE POSITIVE CONTROL: without it a
#: green on the two rows above is indistinguishable from a check that reports nothing.
#: It does NOT carry the snapshot-traversal defect, and that is a property of the sha rather
#: than a hole in the control: `external/swmm/src/solver/snapshot.c` does not exist here
#: (`git ls-tree` -> 0 entries, against a control needle `src/triton.h` -> 1), and
#: `is-ancestor 5fadf167 15eb18a5` -> rc 1. Listing it in that entry's `also_present_in`
#: to make this control symmetric would assert a defect in code that is not there.
_SHA_PRE_FIX = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"
#: THE SECOND POSITIVE CONTROL, and the registry needs two because no single sha carries
#: every registered defect: the three resume fixes are all ancestors of every build that has
#: the snapshot payload at all. This is the outgoing pin -- it carries the snapshot payload
#: and NOT the fix (`is-ancestor 23b75781 031ea42b` -> rc 1), so it resolves the
#: snapshot defect PRESENT while the other three resolve assessed-ABSENT.
_SHA_PRE_SNAPSHOT_FIX = "031ea42b063ce1083e3d965597cdb44da2e748ef"


class _StubCfgSystem:
    def __init__(self, coupled: bool) -> None:
        self.toggle_tritonswmm_model = coupled


class _StubSystem:
    def __init__(self, coupled: bool) -> None:
        self.cfg_system = _StubCfgSystem(coupled)


class _StubAnalysis:
    """The narrowest object the check reads: `_system.cfg_system.toggle_tritonswmm_model`."""

    def __init__(self, coupled: bool = True) -> None:
        self._system = _StubSystem(coupled)


def _df(*, n_resumes: int = 1, model_type: str = "tritonswmm", columns: bool = True) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame({"scenario_directory": ["s0"]})
    return pd.DataFrame({"model_type": [model_type], "n_resumes": [n_resumes], "event_iloc": [0]})


def _run(monkeypatch, sha, *, coupled: bool = True, df=None):
    # Module-GLOBAL patch, never a local import: the check calls `_read_triton_provenance`
    # through the module namespace precisely so this patch lands (see its own comment).
    monkeypatch.setattr(av, "_read_triton_provenance", lambda _a: sha)
    return av.check_known_resume_defects(_StubAnalysis(coupled=coupled), df_status=df if df is not None else _df())


def _masked(result, sha):
    return result.summary.replace(sha[:12], "{PIN}")


# -------------------------------------------------------------------------------------
# The three N/A gates. Each renders a DISTINCT summary; none is a disclosed-denominator
# PASS over an unknown denominator ([Q130]).
# -------------------------------------------------------------------------------------


def test_no_resume_record_is_na():
    res = av.check_known_resume_defects(_StubAnalysis(), df_status=_df(columns=False))
    assert res.applicable is False and res.passed is True
    assert "No resume record available" in res.summary


def test_nothing_resumed_is_na():
    res = av.check_known_resume_defects(_StubAnalysis(), df_status=_df(n_resumes=0))
    assert res.applicable is False and res.passed is True
    assert "No sim was resumed" in res.summary


def test_unknown_producing_sha_is_na(monkeypatch):
    res = _run(monkeypatch, None)
    assert res.applicable is False and res.passed is True
    assert "Producing-TRITON sha unknown" in res.summary


# -------------------------------------------------------------------------------------
# POSITIVE CONTROL. Run first in the file's reading order for the same reason the review
# carrier runs its own first: a green on the discrimination tests below means nothing if
# the check never reports anything.
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("sha", [_SHA_PRE_FIX, _SHA_PRE_SNAPSHOT_FIX])
def test_an_affected_sha_fails_and_its_present_count_matches_its_detail_rows(monkeypatch, sha):
    """Property: on a build carrying >=1 registered defect the check FAILS, names every PRESENT
    defect, and its PRESENT count equals its detail-row count -- so the remedy reaches the
    operator once per present defect rather than once per registry member.

    Class: REGRESSION. The single-sha form of this control asserted
    `len(res.details) == len(REGISTRY)`, which held only while every registered defect was
    present at one sha. The snapshot-traversal entry broke that -- measured 3 == 4 -- because
    its `introduced_in` postdates `_SHA_PRE_FIX`.

    Why it is required: it is the positive control for every discrimination test below. A green
    on those means nothing if the check never reports a PRESENT defect at all, and the two arms
    are what make the control TOTAL over the registry rather than over one sha's subset.

    What kills it: making `check_known_resume_defects` emit a detail row per registry MEMBER
    instead of per PRESENT defect re-reds both arms; so does dropping the FAIL on a present
    defect.

    A second correct implementation under which it still passes: one that renders the detail
    rows in any other order, or that words the remedy differently -- the assertion turns on the
    COUNT agreeing with the summary's own PRESENT figure, not on row order or wording.
    """
    res = _run(monkeypatch, sha)
    assert res.passed is False
    present_ids = [d.defect_id for d in REGISTRY if resolve(d, sha).status == "present"]
    present = len(present_ids)
    assert present >= 1, "this sha is not a positive control for anything"
    # Each PRESENT defect is NAMED, which is what makes the FAIL actionable. The failing arm
    # names its present set and not the whole registry -- the all-members naming belongs to the
    # never-assessed arm, which `test_unregistered_row_names_its_indeterminate_count_and_the_defect_ids`
    # covers. Asserting the whole registry here passed at _SHA_PRE_FIX only because three of its
    # four members happened to be present and the fourth was named as indeterminate.
    for defect_id in present_ids:
        assert defect_id in res.summary
    assert len(res.details) == present
    assert all(r["scenario"] == "(analysis-level)" for r in res.details)
    # The denominator is on the FAILING arm too: a FAIL naming one defect out of four
    # applicable ones must still say what happened to the other three.
    assert f"{len(REGISTRY)} applicable registry defect(s)" in res.summary
    assert f"{present} PRESENT" in res.summary


def test_the_two_positive_controls_together_cover_every_registered_defect():
    """Property: the union of the two control shas' PRESENT sets is the whole registry.

    Class: NEW CAPABILITY. The plausible wrong implementation is adding a registry entry whose
    `also_present_in` names neither control sha -- the entry would then be asserted PRESENT
    nowhere, and every discrimination test below would still pass. Red under it: the union
    falls short of `REGISTRY` and this names the uncovered defect_id.

    Why it is required: the arm above proves each control sha reports SOMETHING. Only this one
    proves no registered defect is left without a positive control, which is the property that
    silently decayed when the registry went from three entries to four.

    What kills it: a new entry with no affected sha among the controls.

    A second correct implementation under which it still passes: any registry whose entries
    distribute differently across the two shas, or a third control sha added alongside -- the
    assertion is on the UNION's coverage, not on which sha carries which defect.
    """
    covered = {
        d.defect_id
        for sha in (_SHA_PRE_FIX, _SHA_PRE_SNAPSHOT_FIX)
        for d in REGISTRY
        if resolve(d, sha).status == "present"
    }
    missing = {d.defect_id for d in REGISTRY} - covered
    assert not missing, (
        f"no positive control asserts these defect(s) PRESENT: {sorted(missing)}. Add an affected "
        "sha to the entry's also_present_in, or add a control sha that carries it."
    )


def test_the_snapshot_defect_is_present_below_the_fix_and_absent_at_and_above_it(monkeypatch):
    """Property: TRITON-SNAPSHOT-TRAVERSAL-GUARD resolves PRESENT on a build pinned below the
    fix and ABSENT at the fix and above it.

    Class: NEW CAPABILITY. The plausible wrong implementation is registering the new sha in
    `known_absent_in` and forgetting the affected pin in `also_present_in` -- the entry then
    resolves `indeterminate` on the build that actually carries the defect, which
    `check_known_resume_defects` handles exactly like `absent` and reports as a PASS. Red under
    it: the `present` assertion below fails while every other test in this file stays green.

    Why it is required: the registry's whole purpose is per-sha discrimination, and an entry that
    cannot separate an affected build from a fixed one certifies nothing. Measured basis for the
    three points: `is-ancestor 23b75781 031ea42b` -> rc 1 (below), and
    `is-ancestor 23b75781 c7b2fd1d` -> rc 0 (above).

    What kills it: removing 031ea42b from `also_present_in`, or removing either fixed sha from
    `known_absent_in`.

    A second correct implementation under which it still passes: one that reaches the same three
    verdicts through ancestry rather than through the explicit sets -- the assertion names the
    STATUS at each point, and the rule only where the read path's clone-lessness makes the
    explicit set the only reachable one.
    """
    from hhemt.model_defects import (
        SHA_SNAPSHOT_REFUSAL_PIN,
        SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX,
        SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
    )

    entry = next(d for d in REGISTRY if d.defect_id == "TRITON-SNAPSHOT-TRAVERSAL-GUARD")

    below = resolve(entry, _SHA_PRE_SNAPSHOT_FIX)
    assert below.status == "present", below
    assert below.rule == "also_present_set", below

    for sha in (SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS, SHA_SNAPSHOT_REFUSAL_PIN):
        at_or_above = resolve(entry, sha)
        assert at_or_above.status == "absent", (sha, at_or_above)
        assert at_or_above.rule == "known_absent_set", (sha, at_or_above)

    # And the check itself reports the discrimination, not only the resolver: the affected pin
    # FAILS and the live pin PASSES. Without this half the test would pass over a resolver whose
    # verdict no consumer reads.
    assert _run(monkeypatch, _SHA_PRE_SNAPSHOT_FIX).passed is False
    assert _run(monkeypatch, _SHA_LIVE := SHA_SNAPSHOT_REFUSAL_PIN).passed is True
    assert _SHA_LIVE == _SHA_REGISTERED, "the live pin constant drifted from the registry's"


# -------------------------------------------------------------------------------------
# ITEM (A): the ruled disclosure. THE ASSERTION IS A DIFFERENTIAL, not a substring match
# on new wording -- a wording change that leaves the two states identical would satisfy a
# substring assertion and would not have fixed anything.
# -------------------------------------------------------------------------------------


def test_registered_and_unregistered_rows_differ(monkeypatch):
    """The two registry states must emit DISTINGUISHABLE rows once the pin is masked.

    This is the inverse of the WP-2B review carrier's characterization, which recorded the
    pre-ruling behaviour and whose own docstring named inversion as the response if the
    developer ruled for disclosure. He did, on 2026-10-03: "agreed on both fronts."
    """
    # The resolver already discriminated; only the REPORT did not.
    assert {resolve(d, _SHA_REGISTERED).status for d in REGISTRY} == {"absent"}
    assert {resolve(d, _SHA_UNREGISTERED).status for d in REGISTRY} == {"indeterminate"}

    reg = _run(monkeypatch, _SHA_REGISTERED)
    unreg = _run(monkeypatch, _SHA_UNREGISTERED)
    assert _masked(reg, _SHA_REGISTERED) != _masked(unreg, _SHA_UNREGISTERED), (
        "the registered and unregistered rows are byte-identical once the pin substring is "
        "masked -- a reader cannot tell assessed-and-clean from never-assessed, which is "
        "the finding the 2026-10-03 ruling closed"
    )


def test_registered_row_names_its_assessed_count(monkeypatch):
    res = _run(monkeypatch, _SHA_REGISTERED)
    assert res.passed is True
    assert f"{len(REGISTRY)} applicable registry defect(s)" in res.summary
    assert f"{len(REGISTRY)} assessed ABSENT" in res.summary
    assert "INDETERMINATE" not in res.summary


def test_unregistered_row_names_its_indeterminate_count_and_the_defect_ids(monkeypatch):
    res = _run(monkeypatch, _SHA_UNREGISTERED)
    assert res.passed is True
    assert f"{len(REGISTRY)} INDETERMINATE" in res.summary
    assert "NOT assessed at this pin" in res.summary
    # NAMING the unassessed defects is what makes the row actionable rather than merely
    # honest: the remedy is per-defect-set membership.
    for defect in REGISTRY:
        assert defect.defect_id in res.summary
    assert "assessed ABSENT" not in res.summary


def test_disclosed_counts_sum_to_the_denominator(monkeypatch):
    """PRESENT + assessed-ABSENT + INDETERMINATE == the applicable denominator, every arm.

    Checked by ARITHMETIC over the rendered sentence rather than by trusting it, because a
    denominator a reader cannot add up is no better than no denominator.
    """
    import re

    for sha in (_SHA_REGISTERED, _SHA_UNREGISTERED, _SHA_PRE_FIX, _SHA_PRE_SNAPSHOT_FIX):
        res = _run(monkeypatch, sha)
        denom = int(re.search(r"(\d+) applicable registry defect\(s\)", res.summary).group(1))
        parts = 0
        for pat in (r"(\d+) PRESENT", r"(\d+) assessed ABSENT", r"(\d+) INDETERMINATE"):
            m = re.search(pat, res.summary)
            parts += int(m.group(1)) if m else 0
        assert parts == denom, f"{sha[:12]}: parts={parts} denom={denom} summary={res.summary!r}"


# -------------------------------------------------------------------------------------
# THE RULED-AGAINST HALF. Pinned so a future round cannot "complete" the disclosure by
# adding the verdict category the developer declined.
# -------------------------------------------------------------------------------------


def test_indeterminate_is_not_a_third_verdict(monkeypatch):
    """An unresolvable registry lookup stays `passed=True, applicable=True`.

    Developer, 2026-09-26: "reporting no known defects is fine, because that phrase already
    implies what it needs to apply, we dont need a third category." Restated 2026-10-03 as
    scoped to verdict CATEGORIES, which is what left the WORDING open for the sibling test
    above. This test is the other half of that ruling and must not be relaxed to admit an
    `applicable=False` or a `passed=False` indeterminate arm.
    """
    reg = _run(monkeypatch, _SHA_REGISTERED)
    unreg = _run(monkeypatch, _SHA_UNREGISTERED)
    assert (reg.passed, reg.applicable) == (True, True)
    assert (unreg.passed, unreg.applicable) == (True, True)
    assert unreg.details == []
    # And the phrase whose built-in hedge is the developer's stated reason for declining a
    # third category must survive on BOTH passing arms. Two assertions in
    # `test_coupled_resume_validity.py` also grep this substring.
    for res in (reg, unreg):
        assert "no known resume defect" in res.summary


# -------------------------------------------------------------------------------------
# TRIGGER SCOPING -- the selection half of the version x selection cross. The denominator
# is a property of the ARM, not of the registry's length.
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("coupled,expected", [(True, 5), (False, 1)])
def test_trigger_scoping_sets_the_denominator(monkeypatch, coupled, expected):
    """A `resumed_coupled` defect is out of scope for a pure-TRITON arm.

    Four of the five registered defects carry `trigger="resumed_coupled"` and one carries
    `resumed_any`, so a pure-TRITON arm has ONE applicable defect. The expected values are
    DERIVED from the registry below as well as stated, so a registry addition fails here
    loudly instead of making the literal silently wrong -- which is what it did when the
    snapshot-traversal entry landed, and the literal moved 3 -> 4 on that red, and again
    4 -> 5 when the clean-start stale-snapshot entry landed.
    """
    derived = sum(1 for d in REGISTRY if not (d.trigger == "resumed_coupled" and not coupled))
    assert derived == expected, f"registry membership changed: derived={derived} literal={expected}"
    res = _run(monkeypatch, _SHA_REGISTERED, coupled=coupled)
    assert f"{expected} applicable registry defect(s)" in res.summary
    assert f"{expected} assessed ABSENT" in res.summary


def test_the_restated_pin_is_the_live_pin():
    """`_SHA_REGISTERED` must still be the pin the synthetic tier builds.

    The module-level constant is restated rather than imported (see its comment), so this
    is the local tripwire for that restatement. `scripts/check_live_pin_registered.py` is
    the independent, commit-time one; this one keeps THIS module from certifying a stale
    sha as "registered" after a pin bump.
    """
    from tests.fixtures._triton_source_cache import TRITON_PIN

    assert _SHA_REGISTERED == TRITON_PIN
