"""Migration guard: partial adoption of the caption-geometry system must be visible.

Condition attached to the Iteration-6 ruling that deferred migrating the remaining
`add_figure_caption` callers. The precedent is on the record: `figure_caption.py`
retired eight hand-tuned bottom margins, and `sensitivity_benchmarking.py` then
never adopted it -- silently, for several iterations, until a user complaint
surfaced it.

Scope is deliberately narrow. A module with NO caption may hardcode `b=` freely --
there is nothing to derive. The two flagged patterns are the two that actually
produced this track's defects:

  V1  a module CALLS add_figure_caption and ALSO hardcodes an integer bottom
      margin, i.e. it computed a derived margin and then threw it away;
  V2  a module places a bottom-anchored paper annotation by hand
      (`fig.add_annotation(..., yref="paper", y=<negative literal>)`) instead of
      going through the module -- the S4 defect exactly.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"

#: The modules owning caption/layout geometry are exempt from both rules.
_OWNER = {"figure_caption.py", "figure_layout.py"}

_CALLS_CAPTION = re.compile(r"\badd_figure_caption\s*\(")
_HARDCODED_B = re.compile(r"margin\s*=\s*dict\([^)]*\bb\s*=\s*\d+")
_RAW_BOTTOM_ANNOTATION = re.compile(
    r"add_annotation\((?:[^()]|\([^()]*\))*?y\s*=\s*-\s*0\.\d+", re.S
)


def find_violations(source: str) -> list[str]:
    """Return violation codes for one module's source text."""
    out: list[str] = []
    body = _strip_comments_and_docstrings(source)
    if _CALLS_CAPTION.search(body) and _HARDCODED_B.search(body):
        out.append("V1-hardcoded-bottom-margin-beside-derived-caption")
    if _RAW_BOTTOM_ANNOTATION.search(body):
        out.append("V2-hand-placed-bottom-annotation")
    return out


def _strip_comments_and_docstrings(source: str) -> str:
    """Drop `#` comments and triple-quoted blocks.

    Load-bearing, not defensive: this corpus documents its retired constants IN
    COMMENTS. Matching raw source would flag the very comments that record the
    fix, which is the vacuous-failure mirror of a vacuous pass.
    """
    source = re.sub(r'(?s)""".*?"""', "", source)
    source = re.sub(r"(?s)'''.*?'''", "", source)
    return re.sub(r"#[^\n]*", "", source)


#: Modules knowingly left unmigrated, with the reason. An EXACT-match assertion,
#: not a floor: a new violation fails, and so does silently fixing one without
#: updating this map. That is what makes the deferral non-silent.
KNOWN_DEFERRED = {
    "eda/_dem_resolution_plots.py": (
        "Its `_add_caption` shim calls add_figure_caption and discards the returned "
        "margin (its own docstring says callers that can take the return value should "
        "call the real function). Deferred from Iteration 6: renders no figure in the "
        "Iteration-6 report set."
    ),
}


def _scan() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.name in _OWNER:
            continue
        v = find_violations(path.read_text(encoding="utf-8"))
        if v:
            found[str(path.relative_to(SRC))] = v
    return found


def test_no_unrecorded_partial_adoption():
    """Every module either derives its caption geometry or is a recorded deferral."""
    found = _scan()
    assert set(found) == set(KNOWN_DEFERRED), (
        f"caption-geometry adoption drifted.\n"
        f"  unrecorded violations: {sorted(set(found) - set(KNOWN_DEFERRED))}\n"
        f"  recorded but now clean (remove from KNOWN_DEFERRED): "
        f"{sorted(set(KNOWN_DEFERRED) - set(found))}"
    )


def test_every_deferral_carries_a_reason():
    for mod, reason in KNOWN_DEFERRED.items():
        assert len(reason) > 40, f"{mod} deferral needs a real reason, not a placeholder"


#--- negative controls: the guard must be able to fail ----


@pytest.mark.parametrize(
    "snippet, expected",
    [
        pytest.param(
            "b_px = add_figure_caption(fig, t, content_w_px=680, plot_h_px=800)\n"
            "fig.update_layout(margin=dict(l=30, r=30, t=80, b=140))\n",
            "V1-hardcoded-bottom-margin-beside-derived-caption",
            id="regressed-derived-margin-thrown-away",
        ),
        pytest.param(
            'fig.add_annotation(text="note", xref="paper", yref="paper",\n'
            "                   x=0.5, y=-0.10, showarrow=False)\n",
            "V2-hand-placed-bottom-annotation",
            id="regressed-hand-placed-footnote",
        ),
    ],
)
def test_guard_fires_on_a_deliberately_regressed_module(snippet, expected):
    """A guard that cannot fail is a vacuous pass. These are the regressions."""
    assert expected in find_violations(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param(
            "b_px = add_figure_caption(fig, t, content_w_px=680, plot_h_px=800)\n"
            "fig.update_layout(margin=dict(l=30, r=30, t=80, b=b_px))\n",
            id="clean-derived-margin",
        ),
        pytest.param(
            "fig.update_layout(margin=dict(l=10, r=10, t=40, b=90))\n",
            id="clean-no-caption-so-any-b-is-fine",
        ),
        pytest.param(
            "# The old `y=-0.02` was a fraction of plot_h, so the caption sank.\n"
            "b_px = add_figure_caption(fig, t, content_w_px=680, plot_h_px=800)\n",
            id="clean-retired-constant-mentioned-only-in-a-comment",
        ),
    ],
)
def test_guard_stays_quiet_on_clean_modules(snippet):
    assert find_violations(snippet) == []
