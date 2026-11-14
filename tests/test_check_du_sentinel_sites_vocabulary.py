"""Vocabulary-consistency and behavioural probes for the DU-sentinel checker.

The SELF-TEST is the load-bearing one: FS_MUTATORS must equal the union of the
names _check_stmt branches on and the names DELIBERATELY_UNHANDLED declares. It is
what makes a future vocabulary addition impossible to leave unwired -- the defect
class this checker was audited for. The behavioural probes pin the two directions a
widening can go wrong: a shape that must newly FAIL, and shapes that must stay clean.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_du_sentinel_sites.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_du_checker_vocab", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_du_checker_vocab"] = module
    spec.loader.exec_module(module)
    return module


def _rules_for(source: str) -> set[str]:
    tmp = Path(tempfile.mkdtemp())
    src_root = tmp / "src" / "hhemt"
    src_root.mkdir(parents=True, exist_ok=True)
    probe = src_root / "probe.py"
    probe.write_text(source, encoding="utf-8")
    checker = _load_checker()
    checker.REPO_ROOT = tmp
    checker.SRC_ROOT = src_root
    return {v.rule_id for v in checker._check_file(probe)}


def test_vocabulary_is_fully_wired() -> None:
    """THE SELF-TEST. Every FS_MUTATORS member is either branched on or explicitly
    declared unhandled, and nothing is branched on that the vocabulary omits. An
    addition to FS_MUTATORS with no branch and no declared reason fails here rather
    than shipping as a name the checker resolves and then silently ignores."""
    checker = _load_checker()
    declared = checker._BRANCHED_NAMES | set(checker.DELIBERATELY_UNHANDLED)
    assert checker.FS_MUTATORS - declared == set(), "vocabulary member neither branched nor declared unhandled"
    assert declared - checker.FS_MUTATORS == set(), "branched/declared name absent from FS_MUTATORS"


def test_raw_shutil_rmtree_is_flagged() -> None:
    """The blind spot this set closes: a raw shutil.rmtree passed silently before."""
    assert "RAW_RMTREE_UNMAINTAINED" in _rules_for("import shutil\ndef f(p):\n    shutil.rmtree(p)\n")


def test_str_replace_is_not_claimed() -> None:
    """INVARIANT. os.replace is in the vocabulary; str.replace shares its attribute
    name and occurs 112 times in src/hhemt. This fails if qualified-name resolution
    ever regresses to bare-attribute matching."""
    assert _rules_for("def f(s):\n    s.replace('a', 'b')\n") == set()


def test_from_import_rmtree_resolves_qualified() -> None:
    """`from shutil import rmtree` must reach the ENFORCING branch, not merely the
    census -- the alias map qualifies it to shutil.rmtree."""
    assert "RAW_RMTREE_UNMAINTAINED" in _rules_for("from shutil import rmtree\ndef f(p):\n    rmtree(p)\n")


def test_attribute_move_on_a_plain_object_is_not_claimed() -> None:
    """INVARIANT. `move` is a common method name; only shutil.move is in scope."""
    assert _rules_for("def f(o):\n    o.move(1, 2)\n") == set()


def test_unlink_pattern_b_baseline_is_unchanged() -> None:
    """REGRESSION GUARD. The .unlink branch emits DELETION_NOT_ROUTED_THROUGH_TOOL, which
    is NOT warn-listed; this fails if the route rule is ever downgraded to warn-tier."""
    checker = _load_checker()
    assert "DELETION_NOT_ROUTED_THROUGH_TOOL" not in checker.WARN_ONLY_RULES
    assert "DELETION_NOT_ROUTED_THROUGH_TOOL" in _rules_for("def f(p):\n    p.unlink()\n")


def test_direct_fast_rmtree_with_analysis_dir_kwarg_is_a_violation() -> None:
    """Clause 9: the route-based predicate rejects a direct fast_rmtree even WITH the old
    analysis_dir= keyword (the retired presence test passed it)."""
    rules = _rules_for("from hhemt.utils import fast_rmtree\ndef f(p, root):\n    fast_rmtree(p, analysis_dir=root)\n")
    assert rules == {"DELETION_NOT_ROUTED_THROUGH_TOOL"}


def test_rmtree_with_adjacent_restamp_is_no_longer_clean() -> None:
    """Clause 9: PATTERN B no longer exists; an un-annotated raw rmtree is WARN-tier regardless
    of what follows it."""
    assert _rules_for(
        "import shutil\ndef f(p, ad):\n    shutil.rmtree(p)\n    restamp_parent_sentinels(p, analysis_dir=ad)\n"
    ) == {"RAW_RMTREE_UNMAINTAINED"}


def test_deletion_routed_through_tool_is_clean() -> None:
    """The compliant site under the route predicate."""
    assert (
        _rules_for(
            "from hhemt.du_sentinels import delete_and_account\n"
            'def f(p, sd):\n    delete_and_account([p], scope_dir=sd, scope="scenario")\n'
        )
        == set()
    )
