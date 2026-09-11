#!/usr/bin/env python3
"""The tolerated-citation LEDGER: a content-keyed, token-granular, RATCHETING
record of the bare ``path:line`` citations that are published today.

WHY A LEDGER AND NOT A WARNING WINDOW. The gate this feeds is an ERROR from the
commit that lands it, so a NEW citation fails immediately and the class is closed
against growth on day one. A warn-only window gives that up for nothing: during
it a new instance is indistinguishable from the pre-existing ones, so the
interval in which the class is supposed to be closing is the interval in which it
can grow unobserved.

THE KEY IS CONTENT, NEVER A POSITION, and that is not a style preference -- a
ledger of decayed LINE NUMBERS keyed BY line number decays for the same reason
its contents do. Measured over the published population at HEAD ``ec7907a2``: 58
citation lines, of which 48 (83%) have at least one sibling ABOVE them in the same
file, and 35 sit in ``src/hhemt/analysis.py`` across a 2385-line span. Replacing a
bare number with a symbol name routinely rewraps a line, so a positional ledger
would churn on nearly every repair commit and each churn is indistinguishable
from new debt. ``scripts/published_surface_baseline.txt`` is keyed by qualname for
the same reason; content keying is the local precedent, not an invention.

THE UNIT IS A TOKEN, NOT A FINDING, and the two counts differ permanently.
``check_docs_content._binary_findings`` runs ``LINE_CITATION.search(line)`` and
stores one excerpt, so a line carrying two citations produces ONE finding holding
the FIRST token. Four published lines are of that shape. A ledger built from the
gate's findings therefore cannot see the second token of those four, which is why
``check_docs_content`` re-scans each line with ``finditer`` before keying.

THREE CARDINALITIES, ONE PER ROLE, AND NO TWO ARE INTERCHANGEABLE. Measured at
``ec7907a2``: 58 FINDINGS is what a red run prints; 62 SITES is the repair
workload; 53 KEYS is this file's size. The key is (file, token, qualname) and is
NOT injective over sites -- 9 sites re-cite a referent already cited from the same
member -- so a repair retires an entry only when every one of its sites is fixed.
That is the correct verdict: the referent is published until the last one goes.

THE RATCHET IS ASYMMETRIC, and this is the line a reader copying
``check_published_surface.baseline_findings`` must change. That function emits a
finding for BOTH ``ADDED`` and ``REMOVED``, because it watches a set expected to
move either way. A debt ledger has inverted semantics: REMOVED is the goal. A
symmetric comparison here fails the gate every time somebody repairs a citation,
which is the most demoralizing available failure mode for a burn-down.

BUT ``REMOVED`` IS NOT ONE CLASS, and collapsing it is the error this module
exists to avoid. A pinned entry absent at HEAD is absent for one of two reasons,
and the key already distinguishes them:

  * RETIRED -- the citing FILE still exists and the enclosing QUALNAME still
    renders, so only the TOKEN went away. That is a repair. Silent, auto-accepted,
    counted on the success line.
  * ORPHAN -- the entry's ADDRESS dissolved: the citing file is gone, or the
    enclosing member no longer renders, or the token form no longer parses under
    the current ``LINE_CITATION``. The debt was not paid; it stopped being
    observable. That is a FINDING of its own class, because silently dropping it
    is how a ledger absorbs unexplained state.

Pure stdlib, and imports nothing from either gate -- this is a LEAF. The gate
imports it; it imports no gate.
"""

from __future__ import annotations

import re
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "published_citation_ledger.tsv"

#: The root every ledger key is relative to, and the root the gate's
#: root-agreement guard compares published files against.
#:
#: THIS IS A PROPERTY OF WHERE THE LEDGER LIVES, not of where a caller pointed
#: ``--docs-dir``. The gate also computes ``_repo_root(args.docs_dir)`` -- the git
#: root of the DOCS tree -- and the two coincide only when ``--docs-dir`` is left
#: at its default. That argument exists so it need not be, and the repository's
#: own suite relocates it on every run. Keying against the docs-derived root
#: yields traversal paths under any relocated docs tree, so a ledger seeded that
#: way is machine-specific: every entry's ``(root / relpath).is_file()`` fails on
#: another checkout, every entry classifies ORPHAN, and the gate is red for
#: everyone except whoever seeded it.
KEY_ROOT = LEDGER.parent.parent

#: Collapse whitespace around the colon so ``workflow.py : 6684`` and
#: ``workflow.py:6684`` are ONE entry. ``LINE_CITATION`` tolerates the spaces and
#: keeps them in ``m.group(0)``, so without this a reformat of a cited line would
#: read as one retirement plus one addition. Measured at HEAD ``ec7907a2``: zero
#: published tokens carry internal whitespace, so this normalizes nothing today
#: and exists because the tolerance is in the pattern rather than in the corpus.
_WS_AROUND_COLON = re.compile(r"\s*:\s*")


def normalize_token(token: str) -> str:
    """The token text as a ledger key."""
    return _WS_AROUND_COLON.sub(":", token.strip())


def key(relpath: str, token: str, qualname: str) -> str:
    """One ledger line: citing file, normalized token, enclosing rendered member.

    TAB-separated because a citation token contains ``:`` and ``/`` and a qualname
    contains ``.``; a tab is the one separator none of the three fields can carry.
    """
    return f"{relpath}\t{normalize_token(token)}\t{qualname}"


def parse(path: Path = LEDGER) -> set[str]:
    """Pinned entries. A MISSING ledger is an empty one, deliberately.

    An absent file must not be a silent skip: with no ledger every live citation
    is an ADDITION and the gate goes red on all of them, which is the loud state.
    A skip branch here would make an unseeded ledger byte-identical to a clean
    one, which is the failure ``check_published_surface``'s REQUIRED ``--site-dir``
    already refuses for the same reason.
    """
    if not path.is_file():
        return set()
    return {line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write(entries: set[str], path: Path = LEDGER) -> int:
    """Pin the current population. Returns the entry count."""
    path.write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
    return len(entries)


def classify(live: set[str], pinned: set[str], *, repo_root: Path, live_qualnames: set[str]):
    """(added, retired, orphaned) -- the ASYMMETRIC ratchet.

    ``added`` FAILS the gate. ``retired`` is silent and is reported only as a
    count on the success line. ``orphaned`` FAILS as its own class.

    The retired/orphaned split reads the KEY and needs no stored state: an entry
    whose file and qualname both still resolve lost only its token, which is a
    repair; an entry that lost its address did not.
    """
    added = live - pinned
    gone = pinned - live
    retired, orphaned = set(), set()
    for entry in gone:
        parts = entry.split("\t")
        if len(parts) != 3:
            orphaned.add(entry)
            continue
        relpath, _token, qualname = parts
        if (repo_root / relpath).is_file() and qualname in live_qualnames:
            retired.add(entry)
        else:
            orphaned.add(entry)
    return added, retired, orphaned
