"""Each item the published pages state lives on one page; a repeat on a second page is a hit.

The published pages held items stated on more
than one page: the same instructions on three pages, the same explanation on the
README and the home page. A second statement drifts from the first, so a reader who
follows the stale copy acts on a fact that is no longer true. This test finds
repeated passages; `single_source_allowlist.yaml` lists the pairs the audit has not
yet resolved.

Pages read: the same set as `test_docs_style.py` (every tracked `.md` file under
`docs/` plus `README.md`, except `docs/reference/bundles/`), through that module's
`tracked_pages` and `mask_code`, so both tests mask fenced blocks and inline code
alike. A command, a value or a quoted tool output is not prose and is not compared.

A page is split into paragraphs. A paragraph is a run of non-blank masked lines, cut
again at each list item, table row and heading. HTML comments and tags are removed
before the split, because a comment is not published and a tag is markup. A paragraph
is compared as lowercase words, with the target of a link removed. Two paragraphs on different pages are a
pair when they share a run of at least `MIN_RUN` consecutive words. One pair is
reported per two pages and shared run, taking the longest run of each paragraph pair;
a second paragraph pair of the same two pages with the same run is not listed.

`MIN_RUN` is 8. It was chosen by measurement against the census of
items that more than one page states: 8 words is the largest run length that still
finds the repeat of the concept-versus-version DOI explanation, whose two copies share
exactly 8 words in a row, and every shorter length adds pairs that no census row
explains (link text, stock clauses). At a length above 8 the DOI repeat is gone. The
census records the full table with the counting command.

Link text that names its target is not a repeat. A markdown link whose text, lowercased
to words, equals the heading of its target (the first heading of a linked page, the
anchored heading of a link with a fragment) is a name and carries no claim. A shared run
that lies wholly inside such link text, on either side, is dropped from the pair. The
target is read from the tracked pages; a link to an external URL, to a page that is not
tracked or to a heading that does not exist keeps its run in the comparison, and a run
that extends beyond the link text also stays. The anchor half of this rule rests on
`mkdocs build --strict` reporting a fragment that names no heading
(`validation.links.anchors` in `mkdocs.yml`), so a heading whose text changes breaks the build
before it can turn a link text into a repeat.

An entry of the allowlist names the two pages, the shared run and a status
(`baseline` for a pair present when this test was added, `permitted` for a pair a
reviewer concurred with: a summary sentence of `README.md` or `docs/index.md` that
links to the page that owns the item, or a pair that the census and the review class
as not a duplicate of an item). The test fails on a pair with no entry and on an entry that
matches no pair. No code in the repository writes the allowlist.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pytest
import yaml

from tests.docs.test_docs_style import mask_code, tracked_pages

_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).resolve().parent / "single_source_allowlist.yaml"
_STATUSES = ("baseline", "permitted")

MIN_RUN = 8

_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#{1,6}\s)")


@dataclass(frozen=True)
class Para:
    """One paragraph of one page, as lowercase words, with its first line number."""

    page: str
    lineno: int
    words: tuple[str, ...]
    names: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Pair:
    """Two paragraphs on different pages that share a run of words."""

    page_a: str
    lineno_a: int
    page_b: str
    lineno_b: int
    run: str


def _mask_markup(text: str) -> list[str]:
    """Lines of `text` with code masked as `test_docs_style.mask_code` does, then HTML removed.

    An HTML comment is not published and a tag (an `<iframe>` element, an autolink) is
    markup, so none of them is prose. A comment keeps its line breaks so that a
    line number still names the line of the page.
    """
    joined = "\n".join(mask_code(text))
    blank = lambda m: "\n" * m.group(0).count("\n")  # noqa: E731
    joined = _HTML_TAG.sub(blank, _HTML_COMMENT.sub(blank, joined))
    return joined.split("\n")


def _slug(heading: str) -> str:
    """The anchor MkDocs derives from a heading: the Python-Markdown `toc` slug."""
    text = _LINK.sub(lambda m: m.group(1), heading).replace("`", "").replace("*", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text).strip().lower())


def _headings(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """(anchor, words) of each heading of a page, fenced blocks skipped, repeats suffixed `_1`, `_2`."""
    out: list[tuple[str, tuple[str, ...]]] = []
    seen: dict[str, int] = {}
    fence = False
    for line in text.splitlines():
        if re.match(r"^\s*(`{3,}|~{3,})", line):
            fence = not fence
            continue
        m = None if fence else _HEADING.match(line)
        if not m:
            continue
        slug = _slug(m.group(1))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        words = tuple(_WORD.findall(_LINK.sub(lambda k: k.group(1), re.sub(r"`[^`]*`", "", m.group(1))).lower()))
        out.append((slug if not count else f"{slug}_{count}", words))
    return out


def _target_heading(page: str, target: str, pages: dict[str, str]) -> tuple[str, ...] | None:
    """The words of the heading a link names, or None when the pages do not say.

    A link to a page names the page's first heading, and a link with a fragment names the
    heading with that anchor. An external URL, an untracked page and a missing anchor
    give None.
    """
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
        return None
    path, _, fragment = target.partition("#")
    dest = page if not path else posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if dest not in pages:
        return None
    heads = _headings(pages[dest])
    if not fragment:
        return heads[0][1] if heads else None
    return next((w for slug, w in heads if slug == fragment), None)


def paragraphs(page: str, text: str, pages: dict[str, str] | None = None) -> list[Para]:
    """The paragraphs of `text`, from its masked lines.

    A blank masked line ends a paragraph, and a fenced block is blank after masking.
    A list item, a table row or a heading starts a new paragraph, because each states
    one thing and a bullet repeated on another page is a repeat of that thing.

    With `pages`, each paragraph also records the word spans of its links whose text
    equals the heading they name (`Para.names`); without it no span is recorded.
    """
    out: list[Para] = []
    current: list[str] = []
    start = 0

    def flush() -> None:
        joined = " ".join(current)
        words: list[str] = []
        names: list[tuple[int, int]] = []
        pos = 0
        for m in _LINK.finditer(joined):
            words += _WORD.findall(joined[pos : m.start()].lower())
            text_words = _WORD.findall(m.group(1).lower())
            if pages is not None and text_words and _target_heading(page, m.group(2), pages) == tuple(text_words):
                names.append((len(words), len(words) + len(text_words)))
            words += text_words
            pos = m.end()
        words += _WORD.findall(joined[pos:].lower())
        if words:
            out.append(Para(page, start, tuple(words), tuple(names)))
        current.clear()

    for idx, line in enumerate(_mask_markup(text), start=1):
        if not line.strip():
            flush()
            continue
        if current and _BLOCK_START.match(line):
            flush()
        if not current:
            start = idx
        current.append(line)
    flush()
    return out


def _live_match(pa: Para, pb: Para, min_run: int) -> tuple[int, int, int] | None:
    """The longest shared run of two paragraphs that is not wholly a link name, or None.

    A longest run that lies inside a name span on either side is set aside: the words of
    that span are replaced by distinct placeholders and the search repeats, so a second,
    longer-than-threshold run elsewhere in the same two paragraphs is still found.
    """
    wa, wb = list(pa.words), list(pb.words)
    while True:
        m = SequenceMatcher(None, wa, wb, autojunk=False).find_longest_match(0, len(wa), 0, len(wb))
        if m.size < min_run:
            return None
        hit = [
            (para, lo)
            for para, lo in ((pa, m.a), (pb, m.b))
            if any(s <= lo and lo + m.size <= e for s, e in para.names)
        ]
        if not hit:
            return m.a, m.b, m.size
        for para, lo in hit:
            words = wa if para is pa else wb
            for i in range(lo, lo + m.size):
                words[i] = f"\0{para.page}{i}"


def single_source_pairs(pages: dict[str, str], min_run: int = MIN_RUN) -> list[Pair]:
    """Every pair of paragraphs on different pages that share `min_run` words in a row.

    `pages` maps a repository-relative path to the page text. The pairs are sorted
    by the two pages, then the two line numbers. One pair is kept per two pages and
    shared run, with the longest run of each paragraph pair and the line numbers of one
    of the paragraph pairs that share it. A paragraph is found through an index of
    its `min_run`-word windows, so the cost follows the number of shared windows and
    not the square of the paragraph count.
    """
    paras = [p for page in sorted(pages) for p in paragraphs(page, pages[page], pages)]
    index: dict[tuple[str, ...], set[int]] = {}
    for i, para in enumerate(paras):
        for j in range(len(para.words) - min_run + 1):
            index.setdefault(para.words[j : j + min_run], set()).add(i)
    candidates: set[tuple[int, int]] = set()
    for members in index.values():
        ordered = sorted(members)
        for x, a in enumerate(ordered):
            for b in ordered[x + 1 :]:
                if paras[a].page != paras[b].page:
                    candidates.add((a, b))
    pairs: dict[tuple[str, str, str], Pair] = {}
    for a, b in candidates:
        pa, pb = paras[a], paras[b]
        match = _live_match(pa, pb, min_run)
        if match is None:
            continue
        run = " ".join(pa.words[match[0] : match[0] + match[2]])
        key = (pa.page, pb.page, run)
        pairs.setdefault(key, Pair(pa.page, pa.lineno, pb.page, pb.lineno, run))
    return sorted(pairs.values(), key=lambda p: (p.page_a, p.page_b, p.lineno_a, p.lineno_b, p.run))


def load_allowlist(path: Path = _ALLOWLIST) -> list[dict[str, str]]:
    """The allowlist entries, each a mapping with page_a, page_b, run, status and reason."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data["entries"]) if data else []


def check_single_source(pages: dict[str, str], entries: list[dict[str, str]]) -> list[str]:
    """Problems found: an unlisted pair, a stale entry, or a malformed entry.

    Each message names both pages and the shared run, so a failing run points at the
    passage to link to its owner or the entry to remove.
    """
    problems: list[str] = []
    for i, entry in enumerate(entries):
        missing = [k for k in ("page_a", "page_b", "run", "status", "reason") if not str(entry.get(k, "")).strip()]
        if missing:
            problems.append(f"entry {i} lacks {missing}: {entry}")
        elif entry["status"] not in _STATUSES:
            problems.append(f"entry {i} has status {entry['status']!r}; allowed: {_STATUSES}")
        elif entry["page_a"] >= entry["page_b"]:
            problems.append(f"entry {i} lists page_a {entry['page_a']!r} at or after page_b {entry['page_b']!r}")
    allowed = {(e.get("page_a"), e.get("page_b"), e.get("run")) for e in entries}
    seen = set()
    for pair in single_source_pairs(pages):
        seen.add((pair.page_a, pair.page_b, pair.run))
        if (pair.page_a, pair.page_b, pair.run) not in allowed:
            problems.append(
                f"unlisted pair: {pair.page_a}:{pair.lineno_a} and {pair.page_b}:{pair.lineno_b} share: {pair.run}"
            )
    for page_a, page_b, run in sorted(allowed - seen, key=str):
        problems.append(f"stale entry (matches no pair): {page_a} and {page_b}: {run}")
    return problems


def test_each_item_lives_on_one_page():
    """Every repeated run of words across two pages has an allowlist entry, and no entry is stale.

    Class: REGRESSION against the pages as they stand. It is red on a page set with an
    unlisted repeat and on an allowlist with an entry that matches no pair; the two
    controls below build both inputs. A second correct implementation compares
    sentences instead of word windows and agrees on the baseline pairs.
    """
    problems = check_single_source(tracked_pages(_REPO), load_allowlist())
    assert problems == [], "\n".join(problems[:50]) + f"\n({len(problems)} problems)"


_SHARED = "run the check before the merge and read the exit status from the process itself"
_FILLER_A = "The first page opens with its own setup sentence."
_FILLER_B = "The second page opens with different setup text entirely."


def test_a_shared_run_on_two_pages_is_a_pair_and_one_page_is_not():
    """A run of MIN_RUN words on two pages is one pair; the same run twice on one page is none.

    Class: NEW CAPABILITY. The plausible wrong implementation compares every paragraph
    with every other paragraph, including those of its own page; it reports the second
    case. A second correct implementation compares word sets per paragraph pair with
    no index, and passes both cases.
    """
    two = {"a.md": f"{_FILLER_A}\n\n{_SHARED}.\n", "b.md": f"{_FILLER_B}\n\n{_SHARED}.\n"}
    pairs = single_source_pairs(two)
    assert [(p.page_a, p.page_b, p.run) for p in pairs] == [("a.md", "b.md", _SHARED)]
    one = {"a.md": f"{_SHARED}.\n\n{_SHARED}.\n"}
    assert single_source_pairs(one) == []


def test_a_run_one_word_short_is_not_a_pair():
    """The threshold is inclusive at MIN_RUN: MIN_RUN words pair, one fewer does not.

    Class: NEW CAPABILITY. The plausible wrong implementation reads `>` for `>=` and
    needs MIN_RUN + 1 words. A second correct implementation counts shared windows
    through difflib on the word lists and passes both sides of the boundary.
    """
    words = _SHARED.split()
    at = " ".join(words[:MIN_RUN])
    short = " ".join(words[: MIN_RUN - 1])
    assert len(single_source_pairs({"a.md": f"{at}.\n", "b.md": f"{at}.\n"})) == 1
    assert single_source_pairs({"a.md": f"{short}.\n", "b.md": f"{short}.\n"}) == []


def test_code_and_link_targets_are_not_compared():
    """A shared fenced block, a shared inline code span and a shared link target are not pairs.

    Class: NEW CAPABILITY. The plausible wrong implementation tokenizes the raw page
    and reports all three. A second correct implementation masks with its own regexes
    in place of `mask_code`, and passes.
    """
    fenced = f"```bash\n{_SHARED}\n```\n"
    inline = f"Use `{_SHARED}` here.\n"
    link = "See [one](docs/how-to/some/long/shared/path/target/that/repeats/on/two/pages.md).\n"
    pages = {"a.md": fenced + "\n" + inline + "\n" + link, "b.md": fenced + "\n" + inline + "\n" + link}
    assert single_source_pairs(pages) == []


def test_list_items_are_compared_one_by_one():
    """A bullet repeated on another page is a pair, and each bullet of a list is its own paragraph.

    Class: NEW CAPABILITY. The plausible wrong implementation treats a list as one
    block; the page then has one paragraph and the pair reports the first line of the
    list, not line 2. A second correct implementation splits lists with a markdown
    parser, and passes.
    """
    a = f"- first item of page a\n- {_SHARED}\n- last item of page a\n"
    b = f"- other item of page b\n- {_SHARED}\n"
    assert len(paragraphs("a.md", a)) == 3
    pairs = single_source_pairs({"a.md": a, "b.md": b})
    assert [(p.lineno_a, p.lineno_b) for p in pairs] == [(2, 2)]


def test_html_comments_and_tags_are_not_compared():
    """A shared HTML comment or `<iframe>` element is not a pair; the prose around it still is.

    Class: NEW CAPABILITY. The plausible wrong implementation tokenizes the tag
    attributes and the comment text, and reports both. A second correct implementation
    strips markup with an HTML parser instead of two regexes, and passes. The line
    number of the paragraph after a multi-line comment must stay the page's line.
    """
    markup = f'<!-- {_SHARED}\n     {_SHARED} -->\n<iframe\n  src="x"\n  title="{_SHARED}">\n</iframe>\n'
    assert single_source_pairs({"a.md": markup, "b.md": markup}) == []
    after = markup + f"\n{_SHARED}.\n"
    pairs = single_source_pairs({"a.md": after, "b.md": after})
    assert [(p.lineno_a, p.lineno_b) for p in pairs] == [(8, 8)]


_NAME = "the compile venue and why the token relaxes rather than arms"


def _linked(target: str, text: str = _NAME) -> dict[str, str]:
    """Pages for the link-name tests: `a.md` links to `t/owner.md` and `b.md` repeats the link text as prose."""
    return {
        "a.md": f"Opening words of the first page.\n\nSee [{text}]({target}) for the details.\n",
        "b.md": f"Other opening words here.\n\nSomething else entirely, then {text}, ends.\n",
        "t/owner.md": f"# Owner\n\n## {_NAME.capitalize()}\n\nBody.\n",
    }


def test_link_text_equal_to_its_target_heading_is_not_a_repeat():
    """A run wholly inside link text that equals the heading the link names is dropped, with and without a fragment.

    Class: NEW CAPABILITY. The plausible wrong implementation drops every run inside
    link text (form a of the shape), which the controls below catch. A second correct
    implementation resolves the heading by parsing the target with a markdown library
    and passes.
    """
    for target in ("t/owner.md#the-compile-venue-and-why-the-token-relaxes-rather-than-arms",):
        pages = _linked(target)
        assert [p for p in single_source_pairs(pages) if "a.md" in (p.page_a, p.page_b)] == []
    page_title = _linked("t/owner.md", "Owner of the shared material described in this very long link text")
    page_title["t/owner.md"] = "# Owner of the shared material described in this very long link text\n"
    page_title["b.md"] = "Other.\n\nXx Owner of the shared material described in this very long link text yy.\n"
    assert [p for p in single_source_pairs(page_title) if "a.md" in (p.page_a, p.page_b)] == []


def test_control_the_same_run_as_plain_prose_on_another_page_still_fails():
    """Control 1: link text equal to the heading still pairs with the same run written as prose on a third page.

    Class: NEW CAPABILITY. The plausible wrong implementation clears the run for the
    whole page set once any link names it. A second correct implementation clears per
    paragraph pair and passes.
    """
    pages = _linked("t/owner.md#the-compile-venue-and-why-the-token-relaxes-rather-than-arms")
    pages["c.md"] = pages["b.md"]
    keys = {(p.page_a, p.page_b) for p in single_source_pairs(pages)}
    assert ("b.md", "c.md") in keys and not {k for k in keys if "a.md" in k}, keys


def test_control_link_text_one_word_off_the_heading_still_fails():
    """Control 2: link text that differs from the target heading by one word is a repeat.

    Class: NEW CAPABILITY. The plausible wrong implementation compares by prefix or by
    containment. A second correct implementation compares word tuples and passes.
    """
    pages = _linked("t/owner.md#the-compile-venue-and-why-the-token-relaxes-rather-than-arms", _NAME + " today")
    keys = {(p.page_a, p.page_b) for p in single_source_pairs(pages)}
    assert ("a.md", "b.md") in keys, keys


def test_control_an_external_link_whose_text_repeats_a_paragraph_still_fails():
    """Control 3: link text whose target cannot be read from the tracked pages keeps its run.

    Class: NEW CAPABILITY. The plausible wrong implementation treats any link as a name.
    A second correct implementation looks the URL up in a set of tracked paths and passes.
    """
    pages = _linked("https://example.org/page")
    assert ("a.md", "b.md") in {(p.page_a, p.page_b) for p in single_source_pairs(pages)}
    missing = _linked("t/owner.md#no-such-heading")
    assert ("a.md", "b.md") in {(p.page_a, p.page_b) for p in single_source_pairs(missing)}


def test_a_run_longer_than_the_link_text_stays():
    """A run that extends past the link text on one side is not wholly inside it.

    Class: NEW CAPABILITY. The plausible wrong implementation clears any run that
    overlaps link text. A second correct implementation compares span ends and passes.
    """
    pages = _linked("t/owner.md#the-compile-venue-and-why-the-token-relaxes-rather-than-arms")
    pages["b.md"] = f"Other opening.\n\nSee {_NAME} for the details.\n"
    assert ("a.md", "b.md") in {(p.page_a, p.page_b) for p in single_source_pairs(pages)}


_SLUG = _NAME.replace(" ", "-")


def _pairs(a_text, owner, extra=None):
    pages = {"a.md": f"Open.\n\n{a_text}\n", "b.md": f"Other.\n\nXx {_NAME} yy.\n", "t/owner.md": owner}
    pages.update(extra or {})
    return {(p.page_a, p.page_b) for p in single_source_pairs(pages)}


def test_duplicate_headings_get_suffixed_anchors():
    owner = f"# O\n\n## {_NAME}\n\nx\n\n## {_NAME}\n\ny\n"
    assert ("a.md", "b.md") not in _pairs(f"See [{_NAME}](t/owner.md#{_SLUG}_1) now.", owner)  # M4 keeps it
    assert ("a.md", "b.md") in _pairs(f"See [{_NAME}](t/owner.md#{_SLUG}_2) now.", owner)


def test_fragment_must_match_exactly():
    owner = f"# O\n\n## {_NAME}\n"
    assert ("a.md", "b.md") in _pairs(f"See [{_NAME}](t/owner.md#{_SLUG.upper()}) now.", owner)  # M5 drops it
    assert ("a.md", "b.md") in _pairs(f"See [{_NAME}](t/owner.md#{_SLUG}-) now.", owner)


def test_heading_of_another_page_does_not_name_the_target():
    pages = {"t/owner2.md": "# Owner2\n\n## Different\n", "t/other.md": f"# X\n\n## {_NAME}\n"}
    assert ("a.md", "b.md") in _pairs(f"See [{_NAME}](t/owner2.md#{_SLUG}) now.", "# Owner\n", pages)  # M6 drops it


def test_a_page_link_names_the_first_heading_only():
    owner = f"# Owner\n\n## {_NAME}\n"
    assert ("a.md", "b.md") in _pairs(f"See [{_NAME}](t/owner.md) now.", owner)  # M8 (last heading) drops it


def test_a_second_long_run_beside_a_link_name_is_found():
    long_run = "a separate sentence that both paragraphs repeat verbatim"
    pages = {
        "a.md": f"Open.\n\nSee [{_NAME}](t/owner.md#{_SLUG}) then {long_run}.\n",
        "b.md": f"Other.\n\nXx {_NAME} yy {long_run}.\n",
        "t/owner.md": f"# O\n\n## {_NAME}\n",
    }
    assert [(p.page_a, p.page_b, p.run) for p in single_source_pairs(pages) if p.page_a == "a.md"] == [
        ("a.md", "b.md", " ".join(long_run.split()))
    ]  # M10 returns none


def test_a_run_straddling_two_link_names_stays():
    h1, h2 = " ".join(_NAME.split()[:5]), " ".join(_NAME.split()[5:])
    owner = f"# O\n\n## {h1}\n\n## {h2}\n"
    a = f"See [{h1}](t/owner.md#{h1.replace(' ', '-')}) [{h2}](t/owner.md#{h2.replace(' ', '-')}) now."
    assert ("a.md", "b.md") in _pairs(a, owner)


def test_malformed_entries_are_each_reported():
    """An entry with an empty reason, a status outside the allowed set, or pages out of order
    yields one problem that names that entry; removing the validation leaves the three unreported.

    Class: NEW CAPABILITY. The plausible wrong implementation checks only the stale
    and unlisted cases. A second correct implementation validates with a schema
    library, and passes.
    """
    base = {"page_a": "a.md", "page_b": "b.md", "run": "x", "status": "baseline", "reason": "r"}
    entries = [{**base, "reason": ""}, {**base, "status": "ok"}, {**base, "page_a": "c.md"}]
    problems = check_single_source({"a.md": "", "b.md": ""}, entries)
    for index, subject in enumerate(("reason", "'ok'", "'c.md'")):
        assert any(f"entry {index} " in p and subject in p for p in problems), (index, problems)


def test_control_a_pasted_paragraph_fails_naming_both_pages():
    """Control 1: a scratch copy of one page with a paragraph of another page pasted into it fails.

    The pasted paragraph is a real paragraph of the last page (the first one of at
    least `2 * MIN_RUN` words), pasted into the first page, as lowercase words. The
    pair is unlisted whatever the allowlist holds, because no entry carries that run
    for those two pages, and the message must name both pages.
    Class: NEW CAPABILITY. The plausible wrong implementation reports a pair only
    when the two paragraphs are line-for-line equal, and the paste is equal only in
    words. A second correct implementation compares the pasted paragraph through the
    full page-set check, and passes.
    """
    pages = tracked_pages(_REPO)
    target, source = sorted(pages)[0], sorted(pages)[-1]
    paste = next(p for p in paragraphs(source, pages[source]) if len(p.words) >= 2 * MIN_RUN)
    mutated = {**pages, target: pages[target] + "\n\n" + " ".join(paste.words) + "\n"}
    problems = check_single_source(mutated, load_allowlist())
    expected = " ".join(paste.words)
    assert any(p.startswith("unlisted pair") and source in p and target in p and expected in p for p in problems), (
        problems
    )


def test_control_a_changed_entry_is_reported_as_stale():
    """Control 2: one character changed in an entry's run makes it stale.

    The pages and the entry are built here, so the control reads no live allowlist
    entry and holds when the allowlist is empty.
    """
    pages = {"a.md": f"{_SHARED}.\n", "b.md": f"{_SHARED}.\n"}
    entry = {"page_a": "a.md", "page_b": "b.md", "run": _SHARED, "status": "permitted", "reason": "control"}
    assert check_single_source(pages, [entry]) == []
    problems = check_single_source(pages, [{**entry, "run": _SHARED + "x"}])
    assert any(p.startswith("stale entry") and _SHARED + "x" in p for p in problems), problems
    assert any(p.startswith("unlisted pair") and _SHARED in p for p in problems), problems


@pytest.mark.parametrize("page", ["README.md", "docs/index.md", "docs/contributing.md"])
def test_the_page_set_includes_the_pages_that_summarize(page):
    """The pages that summarize and link are read: a page set without them is refused.

    The summaries on `README.md` and `docs/index.md` are the pairs the owner rule
    permits; a page set that dropped them would hide every such pair.
    """
    assert page in tracked_pages(_REPO)
