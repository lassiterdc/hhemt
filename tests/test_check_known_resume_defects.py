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
_SHA_REGISTERED = "031ea42b063ce1083e3d965597cdb44da2e748ef"
#: In NO registry set -- the state any branch tip occupies before someone registers it.
_SHA_UNREGISTERED = "deadbeefcafef00ddeadbeefcafef00ddeadbeef"
#: A build that genuinely carries all three defects. THE POSITIVE CONTROL: without it a
#: green on the two rows above is indistinguishable from a check that reports nothing.
_SHA_PRE_FIX = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"


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


def test_pre_fix_sha_fails_and_names_every_defect(monkeypatch):
    res = _run(monkeypatch, _SHA_PRE_FIX)
    assert res.passed is False
    ids = [d.defect_id for d in REGISTRY]
    for defect_id in ids:
        assert defect_id in res.summary
    # The remedy reaches the operator through `details`, one row per defect.
    assert len(res.details) == len(ids)
    assert all(r["scenario"] == "(analysis-level)" for r in res.details)
    # And the denominator is on the FAILING arm too: a FAIL naming one defect out of three
    # applicable ones must still say what happened to the other two.
    assert f"{len(ids)} applicable registry defect(s)" in res.summary
    assert f"{len(ids)} PRESENT" in res.summary


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

    for sha in (_SHA_REGISTERED, _SHA_UNREGISTERED, _SHA_PRE_FIX):
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


@pytest.mark.parametrize("coupled,expected", [(True, 3), (False, 1)])
def test_trigger_scoping_sets_the_denominator(monkeypatch, coupled, expected):
    """A `resumed_coupled` defect is out of scope for a pure-TRITON arm.

    Two of the three registered defects carry `trigger="resumed_coupled"` and one carries
    `resumed_any`, so a pure-TRITON arm has ONE applicable defect. The expected values are
    DERIVED from the registry below as well as stated, so a registry addition fails here
    loudly instead of making the literal silently wrong.
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
