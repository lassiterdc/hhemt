"""The residual disagreement among the THREE in-tree resume-time parses, characterized.

WHAT THIS MODULE IS FOR. The 2026-10-03 repair to
`eda/raw_resume_identity.parse_resume_timestep` closed the EMPTY-TAIL class across all
three implementations -- measured: the other two were already total over it, the regex
because no match is reachable on an empty tail and the leaf because of its explicit
`if not parts: return None`, so the repair closed the only member that was open. It did
NOT close "the numeric parse" as a class. Measured over 25 tails, 4 still disagree, and
all 4 are the TOKEN rule against the REGEX rule. This module records those 4 so the next
change to any of the three goes red here instead of silently widening the set.

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
reachable in principle from a diverged sim, and it is the one row whose divergence could
reach a durable artifact: the stamp written by `processing_analysis` would carry
`resume_t: null` while the figure's parser reads `inf`.

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
    """Structural claim: every residual row is the token rule against the regex rule.

    If a future row appears in which the two TOKEN implementations differ from each other,
    the class has regressed in a different and worse direction, and this test names that
    rather than letting a new parametrize row absorb it.
    """
    for tail, _t, _r, _reach in _RESIDUAL:
        token, regex, leaf = _three(tail, tmp_path)
        assert repr(token) == repr(leaf), tail
        assert repr(token) != repr(regex), f"{tail!r} no longer disagrees -- delete this row"
