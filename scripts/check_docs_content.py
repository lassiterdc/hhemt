#!/usr/bin/env python3
"""CI check: the published docs carry no self-declared placeholder, no bare
``path:line`` citation into a live source file, and no banned vocabulary.

FIVE defect classes across three tiers, and the TIERS are the design. A rule's
tier is decided by what a finding COSTS a reader and by whether resolving it
needs judgment — not by how confident the pattern is:

  * FAILING, fence-skipping  — classes 1 and 2 below (placeholder, line citation)
  * FAILING, fence-INCLUSIVE — class 3 (banned vocabulary); see WORD_BAN_PATTERNS
  * FAILING, unfenced        — class 4 (punctuation); see PUNCTUATION_PATTERNS
  * ADVISORY, never failing  — class 5; see ADVISORY_PATTERNS

Classes 1 and 2, both binary, both cheap, and both observed in this tree:

1. PLACEHOLDER LEAKAGE — a page that tells its own reader its content is
   unfinished. Measured 2026-08-26 at 2 sites: `docs/tutorials/index.md` said the
   tutorials "is authored in a later release-content task" while the nav linked
   to two that existed, and `docs/reference/example-report.md` shipped a
   `**Placeholder.**` note above an empty 600px iframe.

2. DECAYED LINE CITATION — a ``file.py:NNN`` reference into a live repository.
   These decay SILENTLY, precisely because the values were correct when written,
   so nothing about the page's history signals staleness. Measured at 1 site:
   `installation.md` cited `workflow.py:2326` as a SLURM-executor call site; that
   line had become a docstring about the report page.

WHY THE PATTERNS ARE NARROW. A naive ``grep -i placeholder`` over this corpus
returns 7 hits of which 5 are legitimate domain content — ``${VAR}`` templating
and ``{your-allocation}`` substitution instructions. A gate that fires on those
gets routed around, which is strictly worse than no gate. So the placeholder
patterns match SELF-DECLARATIONS about the page's own completeness, never
mentions of placeholder syntax. Content inside fenced code blocks is skipped for
the same reason: a fence is where a legitimate ``TODO`` example lives.

Exit 0 = clean. 1 = findings (enumerated with path:line). 2 = usage error, which
now INCLUDES every population this gate cannot derive. 3 = an unanticipated
internal error, printed with its traceback -- a distinct code because exit 1 is
the FINDINGS code, and a crash reported as findings is a false statement about
the docs. Advisory findings NEVER affect the exit code; pass ``--advisory``.

NOT pure stdlib, and any job that imports this module must install the ``docs``
extra. ``griffe`` and ``yaml`` are imported at module scope, and
``mkdocstrings_handlers.python`` lazily, so a bare checkout cannot even import
this file. This sentence previously read "Pure stdlib", which is what a reader
consults to decide what a CI job must install -- and ``test.yml`` was written
against it and could not collect the test module that imports this one.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

# Self-declarations that a page's own content is unfinished. Deliberately not a
# bare word list: each pattern names a CLAIM about the page, not a topic.
PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("self-declared-placeholder", re.compile(r"\*\*Placeholder\.?\*\*", re.I)),
    ("deferred-to-later-task", re.compile(r"\b(?:is |are |will be )?authored in a later\b", re.I)),
    ("deferred-release-content", re.compile(r"\blater release-content task\b", re.I)),
    ("coming-soon", re.compile(r"\bcoming soon\b", re.I)),
    ("bare-todo-marker", re.compile(r"(?:^|\s)(?:TODO|TBD|FIXME):", re.I)),
    ("stub-self-declaration", re.compile(r"\bis a stub\b", re.I)),
    # An unfilled DATE, which is the placeholder form a release file takes.
    # `2026-XX-XX` declares the entry unfinished exactly as `**Placeholder.**`
    # does, and no pattern above matches it: measured 2026-09-07, all six
    # returned zero on `## v0.1.0 (2026-XX-XX)`. Anchored on a 4-digit year so
    # an `XX-XX` elsewhere in prose is not matched. The trailing guard is a
    # NEGATIVE LOOKAHEAD rather than `\b`: `?` is a non-word character, so a
    # trailing `\b` can never match after `??` and silenced that whole
    # alternative in final position (`2026-??-XX` fired, `2026-XX-??` did not).
    # The `-` inside the class is load-bearing too -- it stops `2026-XX-XX-1`,
    # which a bare `(?!\w)` would still match.
    ("unfilled-date-placeholder", re.compile(r"\b\d{4}-(?:XX|\?\?)-(?:XX|\?\?)(?![\w-])", re.I)),
)

# A source-file reference carrying a line number. Anchored on a real source
# extension so a version string or a time is not matched.
LINE_CITATION = re.compile(r"\b[\w./-]+\.(?:py|yaml|yml|toml|cfg|sh)\s*:\s*\d+\b")

FENCE = re.compile(r"^\s*(?:```|~~~)")

# ---- Vocabulary rules -------------------------------------------------------
#
# THESE SCAN EVERY LINE, FENCES INCLUDED, and that asymmetry with the rules
# above is the whole design decision — do not "fix" it by routing them through
# `_unfenced_lines`.
#
# The rules above skip fences because a fence is where a legitimate `TODO:`
# example belongs. A banned WORD is different in kind: the sites that matter are
# COMMENTS inside fences, which are our own prose and are read by the user
# exactly as body text is. Both measured instances are of that shape:
#   docs/how-to/synthetic-compute-sensitivity-experiment.md:57
#       `# Scaffold: validate + build matrix + write the matrix CSV ...`
#   docs/how-to/running-an-experiment-bundle.md:32
#       `uva: hpc/... # estate-relative (resolved against ... or the estate root)`
# A fence-skipping word ban reports 3 of 4 `scaffold` sites and 1 of 3 `estate`
# occurrences while reading as complete, which is the failure mode this comment
# exists to prevent.
WORD_BAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # User ruling 2026-08-27: "I dont want to use the word 'Scaffold' anywhere in
    # the docs." Prefer set up / prepare / create / generate.
    ("banned-word-scaffold", re.compile(r"\bscaffold\w*\b", re.I)),
    # User ruling: `estate` is undefined and names a private deployment concept a
    # public reader cannot resolve.
    ("banned-word-estate", re.compile(r"\bestate\b", re.I)),
    # Development provenance. A public page must not date itself against the
    # project's internal history: "under v2 graceful-rerun ..." tells a reader
    # there was a v1 they cannot see and cannot need.
    ("development-provenance", re.compile(r"\bunder v\d+\b|\bas of v\d+\b|\bsince v\d+\b", re.I)),
)

# ---- Punctuation rules ------------------------------------------------------
#
# OUR PROSE ONLY: every unfenced line, PLUS comment lines inside fences. The
# discriminator is authorship, not fencing, and the first version of this rule
# got that wrong.
#
# The reason to spare fenced content is that a fence may reproduce external text
# VERBATIM — real command output, a config file, a log line — where normalizing
# punctuation would make the page misquote its own source. That reason is sound
# and still holds. But it does not reach a COMMENT inside a fence, which is our
# own annotation and is read exactly as body text.
#
# Measured 2026-08-28, and the measurement is why this rule changed: the corpus
# held 7 fenced em dashes and ALL SEVEN were in comments we wrote —
# `# Plan only — build the DAG, write nothing:`, `# PyPI version — the durable,
# installable identifier`, and five more of the same shape. Zero were reproduced
# output. So the unfenced-only rule was justified by a case this corpus does not
# contain, and it exempted seven sites it should have caught.
#
# COMMENT_LINE is deliberately narrow — a leading marker only. A `#` mid-line
# inside a shell command is an argument or a fragment, not our prose, and a
# looser pattern would start editing code.
PUNCTUATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("em-dash", re.compile(r"—")),)

# ---- Advisory rules ---------------------------------------------------------
#
# Reported, never failing. The user's framing is explicitly probabilistic —
# "any `, not` is a CANDIDATE for identifying a clause that could be deleted" —
# so each hit needs a human judgment and a hard gate on 23 sites would be a gate
# that gets routed around. Surfaced by `--advisory`, excluded from the exit code.
ADVISORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (("deletable-clause-candidate", re.compile(r",\s+not\b")),)


COMMENT_LINE = re.compile(r"^\s*(?:#|//|--|;)\s")


# ---- Authored-prose population ---------------------------------------------
#
# `rglob("*.md")` has never expressed this gate's actual population. It
# expressed WHATEVER IS ON DISK, which was accidentally identical to AUTHORED
# PROSE until a build-time generator began writing a page into `docs/`. That
# page did not break an invariant; it revealed that the invariant was never
# stated. It is stated here: the gate scans authored prose, and it says what it
# skipped.
#
# A generated page's cells are Pydantic `description=` strings from `src/`, and
# D22b rules `src/` prose out of GATE scope. It does not rule it out of
# VISIBILITY, and the distinction is the whole design: a marker-carrying file is
# skipped by `scan()` and INCLUDED by `scan_advisory()`, so the findings are
# routed to the tier that prints and never gates, rather than dropped. Anyone
# who wants the worklist runs `--advisory` and still gets it.
#
# The marker is a property of the FILE ("this is generated"), not of the lint
# ("this is excused"), which is why it lives in the page rather than in a
# skip-list here. A path-keyed skip would not generalize, and a second
# generated page would silently re-open the gap.
#: The class GROUPS a marker may declare exempt. `prose` is the punctuation and
#: vocabulary pair `### D22b` rules out of gate scope; `binary` is the
#: file-type-independent pair (placeholder leakage, bare line citations).
#: BOTH are spellable on purpose. Hard-coding which group a marker may name
#: would move the class split into this module, which is the position `### A16`
#: rejected when it chose a marker-declared exemption over a central one. What
#: keeps the binary classes enforced is that no live marker declares them, and
#: `tests/test_check_docs_content.py` pins that BY NAME rather than by a count.
EXEMPTABLE_CLASS_GROUPS: frozenset[str] = frozenset({"prose", "binary"})

#: The NAME half of each marker, used for detection. The full declared string a
#: page carries is `GENERATED_MARKER` / `PERSONAL_VOICE_MARKER` below. Detection
#: must key on the NAME rather than on the full string: a page carrying the name
#: with a malformed declaration has to RAISE, and a full-string match would
#: silently fail to detect it and scan the page as if unmarked.
GENERATED_MARKER_NAME = "hhemt:generated-file"

_MARKER_DECL = re.compile(r"(hhemt:[a-z-]+)\s+exempt=([a-z,]*)")


class MarkerDeclarationError(RuntimeError):
    """A marker is present but its exemption declaration is absent or invalid."""


def _declared_exemptions(text: str, marker_name: str) -> frozenset[str] | None:
    r"""The class groups `marker_name` declares exempt, or None when it is absent.

    FAILS LOUDLY rather than degrading. A marker whose declaration is missing,
    empty, or names an unknown group raises. A silent fall back to "exempt
    everything" would rebuild, inside this fix, the whole-file skip the fix
    exists to remove -- which is the failure shape this gate has produced
    repeatedly and is the one thing this parser must not do.

    FENCE-MASKED, PER LINE, and the per-line part is not a stylistic choice. A
    page DOCUMENTING this mechanism carries a fenced example of a correct
    declaration; over raw text that page silently ACQUIRES the exemption it is
    describing, and the same masking stops a fenced example whose declaration is
    malformed from aborting the scan. But `_MARKER_DECL` spans `\s+`, which
    matches a newline, so JOINING the surviving lines would make non-adjacent
    lines ADJACENT and match a marker name before a fence against an `exempt=`
    after it -- turning a loud raise into a silent grant on a page that declared
    nothing. Iterating keeps the legitimate intra-line span and removes every
    cross-line one, including a splice the raw-text form already carried.

    THE NARROWING PER-LINE INTRODUCES, stated because it is a real behaviour
    change and not only a repair. A declaration SPLIT ACROSS LINES -- a
    multi-line HTML comment naming the marker on one line and spelling `exempt=`
    on the next -- was HONOURED before, because `\s+` spans the newline, and now
    RAISES. Measured: it returned `['prose']` and now refuses. That is the same
    span this function removes to close the fence splice, so the two cannot be
    separated: a form that accepts the multi-line declaration accepts the splice.
    The direction is fail-closed, none of the nine live declarations is
    multi-line, and a page wanting the exemption writes it on one line.

    Two further reaches are deliberately NOT closed. A marker name in UNFENCED
    prose still raises, which is why the contributor page states the names
    through a build-time substitution rather than in its own bytes. And an
    UNTERMINATED fence leaves the toggle open, so a declaration below it is not
    seen and the page is scanned rather than skipped -- also fail-closed, and
    stated because it couples this parser to `_unfenced_lines`' toggle semantics.
    """
    seen = False
    for _lineno, line in _unfenced_lines(text):
        if marker_name not in line:
            continue
        seen = True
        for name, raw in _MARKER_DECL.findall(line):
            if name != marker_name:
                continue
            groups = frozenset(part for part in raw.split(",") if part)
            if not groups:
                raise MarkerDeclarationError(f"{marker_name}: `exempt=` declares no class group.")
            unknown = sorted(groups - EXEMPTABLE_CLASS_GROUPS)
            if unknown:
                raise MarkerDeclarationError(
                    f"{marker_name}: unknown class group(s) {unknown}; "
                    f"known groups are {sorted(EXEMPTABLE_CLASS_GROUPS)}."
                )
            return groups
    if not seen:
        return None
    raise MarkerDeclarationError(
        f"{marker_name} is present but declares no `exempt=` class list. "
        f"A marker states what it exempts; known groups are "
        f"{sorted(EXEMPTABLE_CLASS_GROUPS)}."
    )


def _exempt_groups(text: str) -> frozenset[str]:
    r"""Every class group any marker on this page declares exempt.

    REFUSES an `exempt=` declaration on any `hhemt:` name that is not an
    exemption marker. `_MARKER_DECL` matches the whole namespace while this
    consultation is keyed on three names, so before this branch a declaration
    written on a neighbouring marker PARSED, was never consulted, and returned
    silently -- granting nothing while looking exactly like the working form.

    PER LINE for the same reason `_declared_exemptions` is: joining the unfenced
    lines would let `\s+` span the elided fence and refuse a page whose marker
    name and `exempt=` merely sit on either side of one.
    """
    known = (GENERATED_MARKER_NAME, PERSONAL_VOICE_MARKER_NAME, REPO_INTERNAL_MARKER_NAME)
    for _lineno, line in _unfenced_lines(text):
        for name, _raw in _MARKER_DECL.findall(line):
            if name not in known:
                raise MarkerDeclarationError(
                    f"{name} carries `exempt=` but is not an exemption marker. "
                    f"The exemption markers are {sorted(known)}; known class groups "
                    f"are {sorted(EXEMPTABLE_CLASS_GROUPS)}. Drop the `exempt=` "
                    f"declaration, or move it to one of those markers."
                )
    groups: set[str] = set()
    for name in known:
        declared = _declared_exemptions(text, name)
        if declared:
            groups |= declared
    return frozenset(groups)


#: What a generated page CARRIES. `hooks/config_reference.py` interpolates this
#: constant and never authors the string, so widening it here reaches that page
#: with no hook edit. A second generated page inherits this declaration until
#: someone needs otherwise, at which point the hook passes its own list.
GENERATED_MARKER = "hhemt:generated-file exempt=prose"


def _is_generated(text: str) -> bool:
    """True when a page carries the generated-file marker."""
    return _declared_exemptions(text, GENERATED_MARKER_NAME) is not None


def generated_files(docs_dir: Path) -> list[Path]:
    """Every marker-carrying page under `docs_dir`, sorted.

    Printed by `main()` in BOTH output branches. The count is what makes a
    future generated page visible: it goes 1 to 2 on a line somebody reads,
    with nobody having decided to grant an exemption. A source comment cannot
    do that -- it is read once, by whoever writes it.
    """
    return [md for md in _scanned_markdown(docs_dir) if _is_generated(md.read_text(encoding="utf-8", errors="ignore"))]


# A SECOND population that is authored prose but is not PRODUCT prose. The
# contributor guide is the maintainer's own writing in the maintainer's own
# voice; its punctuation answers to personal preference, not to the vocabulary
# and punctuation contract this gate enforces on pages written for a reader of
# the software.
#
# It gets its own marker rather than reusing GENERATED_MARKER for one reason:
# that marker asserts "this file is generated", and the contributor guide is
# hand-written. Reusing it would put a false statement in the page to buy a
# skip, which is the thing a marker-in-the-page design exists to prevent. The
# rule the existing marker states holds for this one too -- it is a property of
# the FILE ("this is personal-voice prose"), not of the lint ("this file is
# excused") -- so it lives in the page and NOT in a path-keyed skip-list here,
# and a second personal-voice page becomes visible as a count going 1 to 2.
#
# Routing is identical to the generated case: skipped by `scan()`, INCLUDED by
# `scan_advisory()`, and named with a count in BOTH of `main()`'s branches.
# Nothing is dropped; the findings move to the tier that prints and never gates.
PERSONAL_VOICE_MARKER_NAME = "hhemt:personal-voice"
PERSONAL_VOICE_MARKER = "hhemt:personal-voice exempt=prose"


def _is_personal_voice(text: str) -> bool:
    """True when a page carries the personal-voice marker."""
    return _declared_exemptions(text, PERSONAL_VOICE_MARKER_NAME) is not None


def personal_voice_files(docs_dir: Path) -> list[Path]:
    """Every personal-voice page under `docs_dir`, sorted.

    Printed by `main()` in BOTH output branches, for the same reason
    `generated_files` is: a skip absent from the output under-reports the
    gate's scope, which is the defect this script exists to catch.
    """
    return [
        md for md in _scanned_markdown(docs_dir) if _is_personal_voice(md.read_text(encoding="utf-8", errors="ignore"))
    ]


def _prose_lines(text: str):
    """Yield (lineno, line) for lines we AUTHORED: unfenced prose, plus comment
    lines inside fences.

    Excludes non-comment fenced content, which may reproduce external text that
    must not be normalized. See the PUNCTUATION_PATTERNS rationale.
    """
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence or COMMENT_LINE.match(line):
            yield i, line


def _all_lines(text: str):
    """Yield (lineno, line) for EVERY line, fences included.

    The counterpart to `_unfenced_lines`, for rules whose defect class lives
    inside fenced comments. See the WORD_BAN_PATTERNS rationale.
    """
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            continue
        yield i, line


def _unfenced_lines(text: str):
    """Yield (lineno, line) for lines OUTSIDE fenced code blocks.

    A fence is exactly where a legitimate `TODO:` example or an illustrative
    `path:line` belongs, so scanning inside one manufactures false positives on
    documentation that is doing its job.
    """
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, line


def _binary_findings(md: Path, text: str) -> list[tuple[str, Path, int, str]]:
    """The two FILE-TYPE-INDEPENDENT tiers: placeholder leakage and line citations.

    Split out of `_gate_findings` so a second population -- shipped package
    metadata, which is not product prose -- can run THESE classes without also
    inheriting the vocabulary and punctuation contracts, which are contracts
    about prose written for a reader of the software and do not bind a TOML
    comment. Extraction rather than duplication is the point: a pattern added
    to `PLACEHOLDER_PATTERNS` reaches both populations with no second edit.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for lineno, line in _unfenced_lines(text):
        for code, pat in PLACEHOLDER_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
        # Backticked spans are NOT excluded, and that is load-bearing rather
        # than an oversight. A live line citation is normally WRITTEN in
        # backticks — the one measured instance in this corpus was
        # `workflow.py:2326` inside a sentence — so excluding inline code
        # makes this check blind to the exact form the defect takes.
        # Measured: with backtick-stripping the gate found 0 of 1 real
        # citations while still reporting the placeholder findings, so it
        # read as working. Fenced blocks are still skipped (see
        # `_unfenced_lines`), which is where an illustrative citation lives.
        m = LINE_CITATION.search(line)
        if m:
            findings.append(("bare-line-citation", md, lineno, m.group(0)))
    return findings


def _gate_findings(md: Path, text: str) -> list[tuple[str, Path, int, str]]:
    """The four FAILING tiers, for one file.

    Factored out so `scan()` and `scan_advisory()` apply the SAME patterns to
    the same bytes. Routing a generated page to the advisory tier only means
    anything if the tier reports the findings the gate would have reported;
    running a different pattern set there would silently drop them while
    looking like routing.
    """
    return _binary_findings(md, text) + _prose_findings(md, text)


def _prose_findings(md: Path, text: str) -> list[tuple[str, Path, int, str]]:
    """The two PROSE tiers: punctuation and vocabulary.

    The complement of `_binary_findings` within `_gate_findings`, extracted so a
    marker can exempt one group without the other. `### D22b` names exactly this
    pair, which is why the split is here and not somewhere finer.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for lineno, line in _prose_lines(text):
        for code, pat in PUNCTUATION_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
    for lineno, line in _all_lines(text):
        for code, pat in WORD_BAN_PATTERNS:
            if pat.search(line):
                findings.append((code, md, lineno, line.strip()))
    return findings


class PopulationDerivationError(RuntimeError):
    """A population this gate scans cannot be derived faithfully.

    THREE causes, and naming only the first is what made three sibling failures
    raise a bare `ValueError` instead: git could not answer for the tracked-file
    population, OR `mkdocs.yml` declares a renderer option this derivation does
    not model, OR the manifested modules yield nothing to check. All three are
    the same statement -- the population is not the page's -- so all three raise
    this type and exit 2 with a clean message and no traceback.

    A SIBLING of `MarkerDeclarationError` rather than a use of it: a repository
    git cannot locate is not a marker-declaration problem, and reusing that type
    to buy a catch would put a false statement in the code -- the ground `A16`
    used to give the personal-voice marker its own name.
    """


def _repo_root(start: Path) -> Path:
    """The repository root git reports for `start`.

    ONE derivation, used by every caller that needs a root. `start.parent` was
    the earlier form and it is an inference: it is right only when `start` is
    exactly the repository's `docs` directory, and silently wrong for anything
    deeper -- which made `SHIPPED_METADATA` resolve nothing and the gate exit 0
    over an empty shipped population.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PopulationDerivationError(
            f"cannot locate the repository root from {start}: `git rev-parse` failed. "
            f"This gate's populations are derived from git and do not fall back to a "
            f"path assumption, because a silent narrowing is what they exist to remove."
        ) from exc
    return Path(out)


def _scanned_markdown(docs_dir: Path) -> list[Path]:
    """Every markdown file this repository SHIPS or BUILDS, sorted and ABSOLUTE.

    A UNION of two halves, and both are load-bearing:

    * every `.md` git tracks in the repository. `rglob` over a directory
      expressed "what is on disk under `docs/`", which was never this gate's
      population: thirteen tracked `.md` files sit outside it, and published
      pages name two of them.
    * every `.md` on disk under `docs_dir`. The generated config-schema page is
      BUILD OUTPUT and this project gitignores it, so a tracked-only population
      drops it -- which would pin `main()`'s generated-file count at 0 forever
      and delete the advisory worklist the marker design deliberately kept.

    `.resolve()` on the second half is NOT cosmetic. `git rev-parse` answers in
    absolute paths and `rglob` inherits the caller's spelling, so a relative
    `--docs-dir` would put BOTH spellings of every docs page in the union -- they
    do not compare equal, so the set does not merge them. Measured: 88 members
    instead of 51, every docs page counted twice, and the first
    `relative_to(repo_root)` in `main()` raising an uncaught ValueError on an
    invocation this gate previously served at exit 0.

    The rglob half is scoped to `docs_dir` and never to the repository root, so
    it does not sweep `site/`, `.venv/`, `test_data/` or `.pytest_cache/`.

    FAIL-CLOSED, deliberately, and the git half runs FIRST. A tree with no git
    raises rather than degrading to the on-disk half. A fallback that silently
    narrows the population is the defect this derivation exists to remove.
    """
    import subprocess

    repo_root = _repo_root(docs_dir)
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "*.md"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PopulationDerivationError(
            f"cannot derive the markdown population: `git ls-files` failed in {repo_root}. "
            f"This gate scans what git tracks plus what the docs build writes, and does "
            f"not fall back to a directory walk, because a silent narrowing is the defect "
            f"the derivation exists to remove."
        ) from exc
    tracked = {repo_root / rel for rel in out.split("\0") if rel}
    return sorted(tracked | set(docs_dir.resolve().rglob("*.md")))


#: A THIRD marker kind, and it needs its own name for the reason the second one
#: got its own. `generated-file` asserts a page is machine-written;
#: `personal-voice` asserts it is the maintainer's own writing. Neither is true
#: of a repo-internal note like `containers/README.md`, and reusing either would
#: put a false statement in the page to buy a skip -- which is the thing a
#: marker-in-the-page design exists to prevent.
REPO_INTERNAL_MARKER_NAME = "hhemt:repo-internal"
REPO_INTERNAL_MARKER = "hhemt:repo-internal exempt=prose"


def _is_repo_internal(text: str) -> bool:
    """True when a page carries the repo-internal marker."""
    return _declared_exemptions(text, REPO_INTERNAL_MARKER_NAME) is not None


def repo_internal_files(docs_dir: Path) -> list[Path]:
    """Every repo-internal page, sorted.

    Printed by `main()` in BOTH output branches, for the same reason
    `generated_files` is: the count is what makes a future marked page visible,
    and that virtue exists only inside the population the derivation walks.
    """
    return [
        md for md in _scanned_markdown(docs_dir) if _is_repo_internal(md.read_text(encoding="utf-8", errors="ignore"))
    ]


def scan(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    findings: list[tuple[str, Path, int, str]] = []
    for md in _scanned_markdown(docs_dir):
        text = md.read_text(encoding="utf-8", errors="ignore")
        # A marker exempts the class groups it DECLARES and nothing else. What
        # it declares is skipped here and INCLUDED by `scan_advisory()`, so the
        # two tiers partition the finding set by construction rather than by a
        # second edit keeping them disjoint. `main()` prints what was skipped
        # either way, and the two populations are counted separately.
        exempt = _exempt_groups(text)
        if "binary" not in exempt:
            findings.extend(_binary_findings(md, text))
        if "prose" not in exempt:
            findings.extend(_prose_findings(md, text))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


def scan_advisory(docs_dir: Path) -> list[tuple[str, Path, int, str]]:
    """Advisory findings — surfaced, never failing. See ADVISORY_PATTERNS.

    Marker-carrying pages -- generated OR personal-voice -- are deliberately NOT
    skipped here. `scan()` skips them so nothing gates on `src/` prose (D22b) or
    on the maintainer's own voice; this tier keeps them visible so the findings
    stay enumerable. Dropping them from both would delete the worklist that a
    later prose sweep would otherwise start from.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for md in _scanned_markdown(docs_dir):
        text = md.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in _unfenced_lines(text):
            for code, pat in ADVISORY_PATTERNS:
                if pat.search(line):
                    findings.append((code, md, lineno, line.strip()))
        # Exactly the groups `scan()` skipped, reported here instead of
        # nowhere. A declaration of what is exempt is, in the same breath, a
        # declaration of what this tier reports.
        exempt = _exempt_groups(text)
        if "binary" in exempt:
            findings.extend(_binary_findings(md, text))
        if "prose" in exempt:
            findings.extend(_prose_findings(md, text))
    # The rendered-docstring population's PROSE classes. `### D22b` rules `src/`
    # prose out of GATE scope and does not rule it out of VISIBILITY, and this is
    # the visibility half -- the anchors are live `src/...py:line` positions a
    # fixer can open, not offsets into a built page nobody edits.
    for _qualname, home, doc_line, doc in rendered_docstrings(SRC_ROOT, API_PAGE):
        for lineno, line in _prose_lines(doc):
            for code, pat in PUNCTUATION_PATTERNS:
                if pat.search(line):
                    findings.append((code, home, doc_line + lineno - 1, line.strip()))
        for lineno, line in _all_lines(doc):
            for code, pat in WORD_BAN_PATTERNS:
                if pat.search(line):
                    findings.append((code, home, doc_line + lineno - 1, line.strip()))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


#: Files that ship to PyPI or are rendered on the project page, and are therefore
#: PUBLIC artifacts, but are not `docs/` prose. `README.md` and `CONTRIBUTING.md`
#: are rendered by PyPI and GitHub; `HISTORY.md` is what `[project.urls]
#: changelog` resolves to; `pyproject.toml` and `CITATION.cff` ARE the published
#: metadata. Only the two file-type-independent classes run here -- see
#: `_binary_findings` for why the prose contracts do not.
#:
#: A LIST rather than a glob, deliberately. A repo-root `rglob("*.md")` would
#: sweep `site/`, `.venv/`, `.claude/` and every planning artifact, and the
#: resulting false-positive volume is how a gate gets routed around. The cost of
#: the list is that a NEW shipped file is not covered until someone adds it, and
#: `main()` prints the population in both branches so that omission is visible
#: rather than silent -- the same design `generated_files` already uses.
SHIPPED_METADATA: tuple[str, ...] = (
    "README.md",
    "CONTRIBUTING.md",
    "HISTORY.md",
    "pyproject.toml",
    "CITATION.cff",
)


def scan_shipped_metadata(repo_root: Path) -> list[tuple[str, Path, int, str]]:
    """Binary-class findings over `SHIPPED_METADATA`, sorted.

    A named entry that does not exist is SKIPPED rather than raising: this list
    is a superset claim about what a project of this shape ships, and a repo
    without a `CITATION.cff` is not a failing repo.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for name in SHIPPED_METADATA:
        path = repo_root / name
        if not path.is_file():
            continue
        findings.extend(_binary_findings(path, path.read_text(encoding="utf-8", errors="ignore")))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


import dataclasses  # noqa: E402
import warnings  # noqa: E402

import griffe  # noqa: E402
import yaml  # noqa: E402

# ---- The RENDERED-DOCSTRING population ------------------------------------
#
# Every member `mkdocstrings` renders on `docs/reference/api.md`, DERIVED FROM
# `griffe` -- the library `mkdocstrings` itself uses -- under the options this
# repository's own `mkdocs.yml` declares. Nothing here models the renderer.
#
# WHY THERE IS NO `ast` TRAVERSAL HERE ANY MORE. There was one, and it was wrong
# in three ways at once, each measured against a site built from a tree in which
# every docstring carried a unique sentinel token:
#
#   * It never read a MODULE's own docstring. `mkdocstrings` renders one at the
#     head of every `:::` block; seven of the nine manifested modules carry one,
#     and all seven were published and unscanned.
#   * It applied `__all__` to a module's OWN members. `mkdocstrings` applies
#     `filters` to those and consults `__all__` only for IMPORTED names, so four
#     `analysis.py` classes rendered unscanned while `experiment_bundle.py`'s
#     imported `ExperimentConfig` was scanned and rendered nowhere.
#   * It keyed members by their ORIGIN module. The page keys them by the module
#     `api.md` declares them through, so 52 of 173 names -- every re-exported
#     symbol, which is to say every name a public API page exists to present --
#     were names no reader or link ever sees.
#
# Patching those three would leave the class that produced them: a derivation at
# the SOURCE-SYNTAX altitude has to answer which module a name resolves to, how
# an alias is followed, and what qualname the page will emit, and those three
# answers are the three defects. `griffe` answers them because it is what the
# renderer asks. Measured at HEAD `9ab08064`: this derivation returns 183 of the
# 183 docstrings the built page renders, with no miss and no over-reach, and all
# 183 of its names are anchors the page emits.
#
# THE MEMBER RULE IS A CONJUNCTION, NOT A CHOICE OF PREDICATE, and dropping
# either half is measurable against the built page:
#
#   * `filters` (from `mkdocs.yml`) governs EVERY name, own-definition or alias.
#     `griffe`'s `is_public` must not stand in for it over a module's OWN
#     members, because `is_public` honours `__all__` there -- defect 2 re-entering
#     through a convenience predicate. It drops `TestRepresentative`,
#     `TestRepresentative.axes`, `TestSubResult` and `TestResult`, one of which
#     carries a rendered em dash, and misses 4 sites the page renders.
#   * An ALIAS must ALSO be exported by the module importing it -- named in that
#     module's `__all__`, which is what `is_public` means FOR AN ALIAS; the two
#     are measured identical on this corpus. Dropping this half is the larger
#     error, because `filters` is a NAME test and says nothing about whether an
#     imported name is re-exported: `filters` alone yields 438 names, of which
#     only 183 are page anchors and 255 are names the page never emits, and it
#     raises TWO `bare-line-citation` findings that would redden this gate on
#     landing.
#
# Both halves, and only both, reach 183 names / 183 page anchors / 0 over-reach.
#
# Report over-reach on the NAME key, not the docstring-site key. The same 438
# names collapse to 271 distinct sites, because a re-exported symbol is reached
# under every module that imports it -- `ConfigurationError` under four. A
# site-keyed count therefore under-reports this failure by 40% and is the reason
# an earlier measurement of it read as smaller than it is.
#
# `filters` is READ from `mkdocs.yml` rather than written here, because the
# population is a property of that file plus `api.md`, and a second copy of a
# setting is a second thing to drift. An option this cannot model raises rather
# than being ignored: a silent no-op on a setting somebody wrote deliberately is
# the same failure that produced the three defects above, wearing a config file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_autodoc_coverage import public_modules  # noqa: E402

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
API_PAGE = _SCRIPT_ROOT / "docs" / "reference" / "api.md"
SRC_ROOT = _SCRIPT_ROOT / "src"
MKDOCS_YML = _SCRIPT_ROOT / "mkdocs.yml"


class _TolerantLoader(yaml.SafeLoader):
    """`mkdocs.yml` carries `!!python/name:` tags that `SafeLoader` refuses."""


_TolerantLoader.add_multi_constructor("", lambda loader, suffix, node: None)
_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda loader, suffix, node: None)


def _module_file(module: str, src: Path) -> Path | None:
    direct = src / (module.replace(".", "/") + ".py")
    if direct.is_file():
        return direct
    pkg = src / module.replace(".", "/") / "__init__.py"
    return pkg if pkg.is_file() else None


# Every python-handler option is CLASSIFIED, and an UNCLASSIFIED one written to a
# NON-DEFAULT value is a loud ValueError.
#
# WHAT "FAILS CLOSED" MEANS HERE, because the phrase invites the wrong reading.
# The check iterates the options actually WRITTEN INTO `mkdocs.yml`, never the
# handler's available fields, so a `mkdocstrings` release adding twenty options
# changes nothing until somebody writes one into the config. **It fires on a
# configuration edit, never on an upgrade.** A denylist was tried first and is
# not durable in the other direction: it fails OPEN on any option a release adds,
# which is the silent-no-op failure this derivation exists to remove -- and that
# failure was demonstrated here, by an earlier draft of this file that read
# `show_submodules`, threaded it through, and discarded it two lines later
# without a sound.
#
# An option written at its OWN DEFAULT is a no-op and passes whatever its class,
# because this derivation already behaves as that default prescribes.
#
# MODELLED: read and acted on.
_MODELLED_OPTIONS = ("filters",)

# NEUTRAL: classified as unable to move WHICH DOCSTRINGS EXIST, so ignored.
# `show_if_no_docstring` is the interesting member -- it changes which members
# get a heading, but this population is docstring-bearing by construction, so it
# cannot move it. The other three are presentation and docstring parsing.
_MEMBERSHIP_NEUTRAL_OPTIONS = ("docstring_style", "members_order", "summary", "show_if_no_docstring")

# MEMBERSHIP-MOVING, RECORDED AS PROSE RATHER THAN AS A TUPLE:
#
#   members, inherited_members, show_submodules, preload_modules, extensions,
#   allow_inspection, force_inspection, merge_init_into_class
#
# These eight were adjudicated individually and the ADJUDICATION is what has to
# survive. The tuple that used to hold them was an ORACLE and an incomplete one:
# it enumerated 8 of the 62 options this derivation refuses, so its only runtime
# effect was to decorate the refusal message for those 8 and stay silent for the
# other 54. Deleting it changes no refusal -- anything unclassified is refused
# anyway. Deleting the RECORD would let a later author reclassify one by
# inspection, which is what these paragraphs prevent.
#
# `merge_init_into_class` is the one that most needs to be written down, because
# it READS presentational and is not. It folds `__init__`'s docstring into the
# class rendering, and `__init__` is excluded by `filters: ["!^_"]` -- so with it
# on, that prose is published and unscanned, which is defect 1's exact shape in a
# second place.
#
# The last three are the sharpest for a different reason: they are not merely
# `PythonOptions` fields, they are `GriffeLoader.__init__` parameters UNDER THE
# SAME NAMES, so `mkdocs.yml` setting one makes the renderer collect with it while
# the loader below collects without it. Measured here: `force_inspection` does not
# shift this population, it makes `hhemt` fail to load at all, which the `except`
# in the load loop would swallow into a silently narrowed scan.
_UNSET = object()


_PYTHON_OPTION_FIELDS: tuple | None = None


def _python_option_fields() -> tuple:
    """`PythonOptions`' dataclass fields -- imported LATE, QUIETLY, and once.

    Importing `mkdocstrings_handlers.python` at module scope emits 350 pydantic
    deprecation warnings from the handler's own option models. Measured: `griffe`
    itself imports clean at 0, the handler at 350, and that is enough to make this
    gate exit 1 under `-W error::DeprecationWarning`, which it did not before.
    They are upstream's warnings to fix and not this gate's to broadcast, so the
    import is deferred to the one function that needs it and silenced for the
    duration of the import alone.
    """
    global _PYTHON_OPTION_FIELDS
    if _PYTHON_OPTION_FIELDS is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mkdocstrings_handlers.python import PythonOptions
        _PYTHON_OPTION_FIELDS = dataclasses.fields(PythonOptions)
    return _PYTHON_OPTION_FIELDS


def _option_default(name: str) -> object:
    """The option's own default, READ from `PythonOptions`, never transcribed.

    A transcribed default is a second copy of a value the library owns, and a
    second copy is the divergence class this whole derivation exists to remove --
    it has no business reappearing in its own constant table. Two of these are
    why: `extensions` and `preload_modules` carry `default_factory=list`, so
    their `.default` is `MISSING` and the real default is `[]`. Transcribed as
    `None`, a `mkdocs.yml` writing the documented default explicitly would RAISE
    -- the harmless value treated as dangerous, which is the truthiness bug's own
    shape surviving inside the fix for it.
    """
    for field in _python_option_fields():
        if field.name != name:
            continue
        if field.default is not dataclasses.MISSING:
            return field.default
        if field.default_factory is not dataclasses.MISSING:
            return field.default_factory()
    return _UNSET


def _handler_options(mkdocs_yml: Path = MKDOCS_YML) -> dict:
    """`filters` as mkdocs.yml declares it, refusing any option not classified.

    Raises ValueError naming every declared python-handler option that is neither
    modelled nor classified membership-neutral. An option set to its own default
    is a no-op and is allowed through whatever its class.
    """
    opts: dict = {"filters": []}
    if not mkdocs_yml.is_file():
        return opts
    cfg = yaml.load(mkdocs_yml.read_text(encoding="utf-8"), Loader=_TolerantLoader) or {}
    for plugin in cfg.get("plugins") or []:
        if not isinstance(plugin, dict) or "mkdocstrings" not in plugin:
            continue
        python = ((plugin["mkdocstrings"] or {}).get("handlers") or {}).get("python") or {}
        declared = python.get("options") or {}
        unclassified = [
            k
            for k, v in declared.items()
            if k not in _MODELLED_OPTIONS and k not in _MEMBERSHIP_NEUTRAL_OPTIONS and v != _option_default(k)
        ]
        if unclassified:
            raise PopulationDerivationError(
                f"{mkdocs_yml} declares python-handler option(s) this population derivation does "
                f"not model: {', '.join(sorted(unclassified))}. "
                "Classify each as modelled or membership-neutral rather than letting the gate "
                "scan a population the page no longer has."
            )
        if "filters" in declared:
            opts["filters"] = list(declared["filters"] or [])
    return opts


def _handler_paths(mkdocs_yml: Path = MKDOCS_YML) -> list[Path]:
    """`handlers.python.paths`, resolved against `mkdocs.yml`'s own directory.

    Returns every declared root, in declaration order; `search_paths` order is
    what breaks a tie when two roots hold the same module name, so the order is
    load-bearing and is preserved rather than sorted.

    This is NOT an `options` key -- it sits BESIDE `options` on the handler -- so
    it belongs here, read, rather than in the classification above, where a check
    keyed on it could never fire. It is what tells `mkdocstrings` where the
    package is; a derivation that hardcodes `src` while the page reads this key is
    the same silent divergence in a second place.

    WHEN `paths` IS ABSENT the fallback is the config directory, NOT `src`.
    `PythonConfig.paths` carries `default_factory` returning `['.']`, resolved
    against `mkdocs.yml`, so a `src` fallback would search somewhere the renderer
    does not -- reintroducing, on the one branch that fires when configuration is
    absent, exactly the hardcode this function exists to remove.

    This function only READS. It refuses nothing, and it says so because a
    docstring describing behaviour its function does not contain is the defect
    class this round exists to close.
    """
    if not mkdocs_yml.is_file():
        return []
    cfg = yaml.load(mkdocs_yml.read_text(encoding="utf-8"), Loader=_TolerantLoader) or {}
    for plugin in cfg.get("plugins") or []:
        if not isinstance(plugin, dict) or "mkdocstrings" not in plugin:
            continue
        python = ((plugin["mkdocstrings"] or {}).get("handlers") or {}).get("python") or {}
        declared = python.get("paths") or ["."]
        return [(mkdocs_yml.parent / entry).resolve() for entry in declared]
    return []


def _resolve_module_file(module: str, roots: list[Path]) -> Path | None:
    """First root holding the module, mirroring `search_paths` precedence.

    Multi-root is legal `mkdocstrings` configuration and is SUPPORTED rather than
    refused. Refusing it would turn a valid config into a hard failure to save
    two small changes -- this loop, and passing the whole list to
    `GriffeLoader(search_paths=...)`, which already takes a sequence. The cost of
    supporting it is that a module name present under two roots resolves to the
    first, which is what `griffe` does with the same list in the same order.
    """
    for root in roots:
        found = _module_file(module, root)
        if found is not None:
            return found
    return None


def _passes_filters(name: str, filters: list[str]) -> bool:
    """`mkdocstrings` filter semantics: last matching rule wins; `!` negates."""
    keep = True
    matched = False
    for rule in filters:
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        if re.search(pattern, name):
            keep = not negate
            matched = True
    if not matched and any(not f.startswith("!") for f in filters):
        return False
    return keep


def _exported(owner) -> set[str]:
    """Names the module lists in `__all__` -- the alias-rendering rule."""
    out: set[str] = set()
    for entry in getattr(owner, "exports", None) or []:
        out.add(entry if isinstance(entry, str) else getattr(entry, "name", ""))
    return {n for n in out if n}


def rendered_docstrings(src: Path = None, api_page: Path = None) -> list[tuple[str, Path, int, str]]:
    """(page qualname, defining file, docstring start line, docstring).

    The qualname is the anchor the PAGE emits -- `hhemt.Toolkit`, not
    `hhemt.toolkit.Toolkit` -- because every downstream assertion over this
    population is a claim about the page and must be keyed the way the page is.

    Raises ValueError if no manifested module resolves, or if a manifested module
    contributes zero members -- an empty contribution is the STRICT signature and
    an empty population would make this gate pass vacuously.
    """
    # No explicit root: take the ones `mkdocs.yml` gives the renderer, so the gate
    # and the page look in the same places by construction rather than by two
    # copies of `src` agreeing. An explicit `src` still wins, which is what lets a
    # test point this at a fixture tree.
    roots = [src] if src is not None else (_handler_paths() or [SRC_ROOT])
    api_page = API_PAGE if api_page is None else api_page
    filters = _handler_options()["filters"]

    modules = public_modules(api_page)
    loader = griffe.GriffeLoader(search_paths=[str(root) for root in roots])
    resolved: list[str] = []
    for module in modules:
        if _resolve_module_file(module, roots) is None:
            continue
        try:
            loader.load(module)
        except Exception:
            continue
        resolved.append(module)
    if not resolved:
        roots_text = ", ".join(str(root) for root in roots)
        raise PopulationDerivationError(
            f"no module named on {api_page} resolved under {roots_text} -- nothing to check."
        )
    loader.resolve_aliases(external=False)

    out: list[tuple[str, Path, int, str]] = []
    seen: set[str] = set()
    per_module: dict[str, int] = {}

    def emit(qualname: str, obj) -> bool:
        doc = getattr(obj, "docstring", None)
        if doc is None or not doc.value or qualname in seen:
            return False
        seen.add(qualname)
        out.append((qualname, Path(obj.filepath), doc.lineno, doc.value))
        return True

    def visit(owner, page_path: str, depth: int) -> int:
        count = 0
        if depth > 2:
            return count
        exported = _exported(owner)
        for name, member in list(owner.members.items()):
            if not _passes_filters(name, filters):
                continue
            target = member
            if member.is_alias:
                if name not in exported:
                    continue
                try:
                    target = member.final_target
                except Exception:
                    continue
            kind = target.kind.value
            if kind not in ("class", "function", "attribute"):
                continue  # `show_submodules` is False and is refused if set
            qualname = f"{page_path}.{name}"
            count += emit(qualname, target)
            if kind == "class":
                count += visit(target, qualname, depth + 1)
        return count

    for module in resolved:
        mod = loader.modules_collection[module]
        count = int(emit(module, mod))
        count += visit(mod, module, 0)
        per_module[module] = count

    empty = [m for m, c in per_module.items() if c == 0]
    if empty:
        raise PopulationDerivationError(
            f"manifested module(s) contributed zero documented members: {', '.join(empty)}. "
            f"That is the STRICT signature -- check the population derivation, not the modules."
        )
    return out


def scan_rendered_docstrings(src: Path = SRC_ROOT, api_page: Path = API_PAGE) -> list[tuple[str, Path, int, str]]:
    """Binary-class findings over the rendered-docstring population, sorted.

    Only `_binary_findings`. The vocabulary and punctuation contracts are `### D22b`
    out-of-scope for `src/` prose and are routed to `scan_advisory()` instead, so
    this population is gated on exactly the two file-type-independent classes that
    `hooks/config_reference.py` already fails the build on for the sibling
    `src/`-derived page.
    """
    findings: list[tuple[str, Path, int, str]] = []
    for _qualname, home, doc_line, doc in rendered_docstrings(src, api_page):
        for code, _p, lineno, excerpt in _binary_findings(home, doc):
            findings.append((code, home, doc_line + lineno - 1, excerpt))
    return sorted(findings, key=lambda f: (str(f[1]), f[2], f[0]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs",
        help=(
            "the documentation root (default: ./docs) -- the docs root, NOT the "
            "repository root. The scanned population is every markdown git tracks "
            "in the repository, plus every markdown on disk under this directory"
        ),
    )
    ap.add_argument(
        "--advisory",
        action="store_true",
        help="also print advisory findings (never affects the exit code)",
    )
    args = ap.parse_args(argv)
    if not args.docs_dir.is_dir():
        print(f"ERROR: docs dir not found: {args.docs_dir}", file=sys.stderr)
        return 2
    try:
        return _run(args)
    except (MarkerDeclarationError, PopulationDerivationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, MarkerDeclarationError):
            print(
                "A generated page is rewritten only when `mkdocs build` runs and is "
                "gitignored, so a stale or absent one is the normal state of a fresh "
                "clone. Run `mkdocs build` first.",
                file=sys.stderr,
            )
        return 2
    except Exception:
        traceback.print_exc()
        print(
            "ERROR: unanticipated failure inside this gate. Exit 3, not 1: 1 is the "
            "FINDINGS code, and a crash reported as findings is a false statement "
            "about the docs.",
            file=sys.stderr,
        )
        return 3


def _run(args: argparse.Namespace) -> int:
    """The body of `main()`, separated so a `MarkerDeclarationError` raised anywhere
    inside it returns the documented exit 2 rather than a traceback.

    Four call sites below read markers -- `scan`, `scan_advisory`, `generated_files`
    and `personal_voice_files` -- so a guard around any ONE of them would leave the
    other three uncovered. This module's docstring contracts three outcomes (0, 1, 2)
    and a traceback is none of them; `check_autodoc_coverage.py` handles the same
    build-artifact case the same way.
    """
    # The repository root, derived ONCE and bound before every consumer. Two of
    # the three `relative_to` sites in this prologue run ABOVE the old binding
    # site, so repairing them without moving the binding is an UnboundLocalError
    # rather than a fix.
    repo_root = _repo_root(args.docs_dir)
    if args.advisory:
        advisory = scan_advisory(args.docs_dir)
        print(f"advisory: {len(advisory)} candidate(s) — judgment required, not a gate.")
        for code, path, lineno, excerpt in advisory:
            rel = path.relative_to(repo_root)
            print(f"  {rel}:{lineno} [{code}] {excerpt[:110]}")

    # Name every class checked, AND every file not checked. A success line that
    # under-reports its own scope is the same defect this gate exists to catch,
    # one level up -- and a skip absent from the output is exactly that
    # under-reporting. Printed in BOTH branches: a red run that hides its
    # population is worse than a green one, not better.
    def _skip_line(label: str, paths: list[Path]) -> str:
        if not paths:
            return f"skipped 0 {label} file(s)"
        rels = ", ".join(
            f"{m.relative_to(repo_root)} (exempt="
            f"{','.join(sorted(_exempt_groups(m.read_text(encoding='utf-8', errors='ignore'))))})"
            for m in paths
        )
        return f"skipped {len(paths)} {label} file(s), routed to --advisory: {rels}"

    shipped = [name for name in SHIPPED_METADATA if (repo_root / name).is_file()]
    rendered = rendered_docstrings(SRC_ROOT, API_PAGE)
    skip_lines = [
        _skip_line("generated", generated_files(args.docs_dir)),
        _skip_line("personal-voice", personal_voice_files(args.docs_dir)),
        _skip_line("repo-internal", repo_internal_files(args.docs_dir)),
        f"scanned {len(shipped)} shipped-metadata file(s) for placeholders and line "
        f"citations only: {', '.join(shipped) if shipped else '(none found)'}",
        f"scanned {len(rendered)} rendered docstring(s) from {API_PAGE.name} for placeholders "
        f"and line citations only; their vocabulary and punctuation are D22b out-of-scope "
        f"and are reported under --advisory, never gated",
    ]

    # `scan()` walks every markdown this repository ships or builds, which
    # INCLUDES the markdown members of SHIPPED_METADATA. Those are already
    # scanned by `scan_shipped_metadata` for the two file-type-independent
    # classes ONLY -- see the SHIPPED_METADATA comment for why the prose
    # contracts do not run on them. Routed out here rather than out of the
    # population, because narrowing the population would also drop them from
    # `scan_advisory`, and full paths rather than names because a nested
    # `README.md` is a different file with a different ruling.
    shipped_paths = {repo_root / name for name in SHIPPED_METADATA}
    findings = (
        [f for f in scan(args.docs_dir) if f[1] not in shipped_paths]
        + scan_shipped_metadata(repo_root)
        + scan_rendered_docstrings()
    )
    if findings:
        print("docs content check FAILED:", file=sys.stderr)
        for code, path, lineno, excerpt in findings:
            rel = path.relative_to(repo_root)
            print(f"  {rel}:{lineno} [{code}] {excerpt[:110]}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). A placeholder tells a reader the page is "
            f"unfinished; a bare path:line citation decays silently as the source "
            f"moves. Replace a line citation with a symbol name, which does not decay.",
            file=sys.stderr,
        )
        for line in skip_lines:
            print(line, file=sys.stderr)
        return 1

    print(
        f"docs content OK under {args.docs_dir}, and shipped metadata clean — no "
        f"placeholder leakage, bare line citation, banned vocabulary, development "
        f"provenance, or em dash."
    )
    for line in skip_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
