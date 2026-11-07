"""Reviewer cases for the measure in `test_retrofit_explanation_pass.py`.

Property enforced: no `baseline` allowlist entry on a page of the explanation and root subset except the keys
the measure exempts, and no `permitted` entry except the licence-grant entry.

Class: NEW CAPABILITY. Input class: a `baseline` entry on the exempted page that is not the exempted key, and a
`permitted` entry on a subset page that is not the stated licence-grant key.
Plausible wrong implementations, red here and green under the author's tests: (1) exempting every entry of the
exempted page instead of the one exempted key; (2) dropping the `permitted` filter so any permitted entry passes.
Why required: the author's tests run the measure on the live allowlist only, where both wrong forms return nothing.
Second correct implementation: a measure that compares (page, rule, line) with a stated-key set loaded from data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "test_retrofit_explanation_pass.py"
_SPEC = importlib.util.spec_from_file_location("retrofit_explanation_pass_under_review", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_a_second_baseline_entry_on_the_n16_page_is_reported():
    """A baseline entry on the exempted page with another rule or line is a finding; only the exempted key is exempt."""
    extra = {
        "page": "docs/explanation/exploratory-analysis.md",
        "rule": "it",
        "line": "x",
        "status": "baseline",
        "reason": "r",
    }
    assert _MOD.subset_problems([extra]) == [
        "baseline entry left on a subset page: docs/explanation/exploratory-analysis.md rule it: x"
    ]


def test_a_permitted_entry_other_than_the_stated_grant_is_reported():
    """A permitted entry on a subset page that the pass did not state is a finding."""
    extra = {"page": "docs/index.md", "rule": "it", "line": "x", "status": "permitted", "reason": "r"}
    assert _MOD.subset_problems([extra]) == ["permitted entry the pass did not state: docs/index.md rule it: x"]


def test_the_n16_entry_and_the_stated_grant_alone_are_not_reported():
    """Both-states anchor: the two stated keys pass, so the two cases above turn on the key and not on wording."""
    n16 = {"page": _MOD._N16[0], "rule": _MOD._N16[1], "line": _MOD._N16[2], "status": "baseline", "reason": "r"}
    grant = {
        "page": _MOD._PERMITTED[0],
        "rule": _MOD._PERMITTED[1],
        "line": _MOD._PERMITTED[2],
        "status": "permitted",
        "reason": "r",
    }
    assert _MOD.subset_problems([n16, grant]) == []
