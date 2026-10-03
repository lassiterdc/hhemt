r"""The residual disagreement among the THREE in-tree resume-time parses, characterized.

WHAT THIS MODULE IS FOR. The 2026-10-03 repair to
`eda/raw_resume_identity.parse_resume_timestep` closed the EMPTY-TAIL class across all
three implementations OVER THIS MODULE'S FIXTURE FAMILY -- measured: the other two were
already total over it, the regex because no match is reachable on an empty tail and the
leaf because of its explicit `if not parts: return None`, so the repair closed the only
member that was open. It did NOT close "the numeric parse" as a class. Measured over this
module's 25 tails, 4 still disagree, and all 4 are the TOKEN rule against the REGEX rule.
This module records those 4 so the next change to any of the three goes red here instead
of silently widening the set.

THE FRAME OF BOTH COUNTS ABOVE IS NAMED, NOT ELIDED, and it is why they carry "over this
module's fixture family" rather than standing as class-wide closure. Every fixture here is
SINGLE-LINE and every tail is a FINITE token, and the implementations hold neither constant:
`tests/test_resume_parse_residual_frame.py` measures the two dimensions this family cannot
express -- TAIL SCOPE (the two token rules split to END OF FILE versus END OF LINE, so on a
multi-line log they return different answers) and THE NON-FINITE FAMILY (`nan`, `-nan`,
`NaN`, `-inf`, `Infinity` diverge by the same mechanism as the `inf s` row below and are
absent from `_RESIDUAL`). That module states in its own docstring that both claims "ARE
TRUE OVER THAT MODULE'S FIXTURE FAMILY AND NEITHER IS TRUE OF THE CLASS". Read the two
together; neither count above is a statement about the class.

IT IS A CHARACTERIZATION, NOT A FINDING, and the distinction matters for how a reader
should treat a green. A green here does not say the three parsers agree; it says they
disagree in exactly the four ways measured at `9e8a2abd`. Narrowing the set is a
legitimate change that must come here and delete rows; WIDENING it silently is what this
module exists to prevent.

THE MECHANISM, because the two interesting rows are not "one of them is stricter".
`processing_analysis._parse_replay_t` matches `([-+0-9.eE]+)` and so consumes a PREFIX of
the whitespace token, then calls `float()` on that prefix -- which SUCCEEDS. The token
rule takes the WHOLE token and lets `float()` refuse it. So on `1_000` the regex returns
1.0 and the token rule returns 1000.0: not None-versus-number, but two different NUMBERS,
silently. That is the shape worth knowing about, and it is a property of prefix matching
rather than of strictness.

REACHABILITY IS BOUNDED AND DISCLOSED. `up_to_time` is a `value_t` streamed with the
default `operator<<`, which never emits `_` or `0x`, so the `1_000` and `0x10` rows are
UNREACHABLE from the solver and are probes of the mechanism rather than live hazards. The
`3600s` row (no space) is likewise unreachable at the live pin, which emits `" s ("` --
the WP-2B review carrier records that and deliberately declines to assert it. `inf` IS
reachable in principle from a diverged sim, and it is the only row of the four for which
that is so.

IT REACHES NO DURABLE ARTIFACT TODAY, and an earlier version of this paragraph said
otherwise. It claimed `inf` was "the one row whose divergence could reach a durable
artifact: the stamp written by `processing_analysis` would carry `resume_t: null` while the
figure's parser reads `inf`". The second clause is FALSE: the figure does not call that
parser. Measured at 24be6058 --
`grep -rnE 'parse_resume_timestep[[:space:]]*\(' --include=*.py src/ scripts/` returns ONE
hit and it is the `def` line itself (`raw_resume_identity.py:181`), and no module under
`src/` or `scripts/` imports the symbol at all. So `parse_resume_timestep` has ZERO
production call sites, and its callers are six test modules. The two things that actually
produce a resume boundary in production are `resume_boundaries_from_schedule`
(`raw_resume_identity.py:227`, called at `:706` from `check_raw_b4b` -- the bit-for-bit
figure) and `processing_analysis._parse_replay_t` (`:950`, called at `:1024-1025` -- the
durable stamp, and the REGEX rule). There is therefore no second reader for the stamp to
disagree with, and the divergence this table records is a property of the FUNCTIONS rather
than a live wrong answer. It matters because `parse_resume_timestep` is the natural thing to
wire in when the figure needs a MEASURED rather than a REQUESTED boundary -- which is
exactly when these rows would start reaching an artifact.
`tests/test_resume_parse_residual_frame.py` carries the same measurement.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION: no `*_compiled` fixture, no
`TRITONSWMM_analysis`.
"""

from __future__ import annotations

import pytest

import hhemt.analysis_validation as av
from hhemt.eda.raw_resume_identity import parse_resume_timestep
from hhemt.processing_analysis import _parse_replay_t
from hhemt.resume_events import _parse_leading_float

_MARKER = av._TRITON_REPLAY_MARKER

#: Every tail on which the three implementations are NOT all equal, measured at 9e8a2abd.
#: Columns: (tail, token-rule value, regex-rule value, reachable-from-the-solver).
_RESIDUAL = [
    # The regex consumes a PREFIX and float() succeeds on it; the token rule takes the
    # whole token and float() refuses it. Unreachable: the emitter writes " s (".
    ("3600s", None, 3600.0, False),
    # float() accepts "inf"; the regex char class cannot match an alphabetic token.
    # REACHABLE in principle from a diverged sim.
    ("inf s", float("inf"), None, True),
    # TWO DIFFERENT NUMBERS, silently: float() accepts PEP-515 underscores, the regex
    # stops at the underscore and parses the prefix "1".
    ("1_000 s", 1000.0, 1.0, False),
    # Same prefix mechanism: the regex parses "0", the token rule refuses "0x10".
    ("0x10 s", None, 0.0, False),
]


def _three(tail, tmp_path):
    text = f"[..] {_MARKER}{tail}\n"
    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_text(text)
    return (
        parse_resume_timestep(log),
        _parse_replay_t(text, _MARKER),
        _parse_leading_float(text.split(_MARKER, 1)[1]),
    )


@pytest.mark.parametrize("tail,expect_token,expect_regex,_reachable", _RESIDUAL)
def test_the_residual_disagreement_is_exactly_as_measured(tmp_path, tail, expect_token, expect_regex, _reachable):
    token, regex, leaf = _three(tail, tmp_path)
    assert repr(token) == repr(expect_token), f"token rule moved on {tail!r}"
    assert repr(regex) == repr(expect_regex), f"regex rule moved on {tail!r}"
    # The two TOKEN-rule implementations must stay extensionally equal: that equality is
    # what the (D) repair bought, and it is the half of the class that IS closed.
    assert repr(leaf) == repr(token), (
        f"the two token-rule implementations have diverged on {tail!r} -- "
        "eda/raw_resume_identity and resume_events declare the SAME rule by intent"
    )


def test_no_implementation_raises_on_the_empty_tail_class(tmp_path):
    """The class the (D) repair closed, pinned across ALL THREE rather than one.

    Before the repair the token rule raised `IndexError` here while the other two already
    returned None, so this is the only row in the file that was ever a FINDING.
    """
    for tail in ("", "   ", "\t", "\n", "  \t \n"):
        token, regex, leaf = _three(tail, tmp_path)
        assert (token, regex, leaf) == (None, None, None), f"{tail!r} -> {(token, regex, leaf)}"


def test_the_disagreement_set_is_token_versus_regex_only(tmp_path):
    """Structural claim, SCOPED TO `_RESIDUAL`: every row here is token-versus-regex.

    If a future row is ADDED TO `_RESIDUAL` in which the two TOKEN implementations differ
    from each other, the class has regressed in a different and worse direction, and this
    test names that rather than letting a new parametrize row absorb it.

    THE SCOPE IS NOT A HEDGE -- the two token rules ALREADY differ off this family, and the
    promise would be false if it were read as class-wide. `_RESIDUAL` is single-line
    throughout, where a file-scoped tail and a line-scoped tail coincide;
    `tests/test_resume_parse_residual_frame.py::test_the_two_token_rules_are_extensionally_UNEQUAL_off_the_single_line_family`
    exhibits the multi-line case where they do not. So this test pins a property of the
    declared list, which is the honest form of a claim no finite corpus can make
    unconditionally -- it does not certify that no such row exists.
    """
    for tail, _t, _r, _reach in _RESIDUAL:
        token, regex, leaf = _three(tail, tmp_path)
        assert repr(token) == repr(leaf), tail
        assert repr(token) != repr(regex), f"{tail!r} no longer disagrees -- delete this row"
