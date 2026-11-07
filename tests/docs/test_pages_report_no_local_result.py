"""The published pages record no local test result and name no personal machine.

The pages state project rules and the behaviour of the toolkit. A measured timing, a run identifier of the test
harness, or the name of the machine a contributor works on is a local result or a local note, and the pages carry
none of them. Two rules follow, each checked over every line of the population below:

- no line carries a run identifier: `{YYYYMMDD}T{HHMMSS}Z`, a hyphen or an underscore and a hexadecimal digest, with or
  without a host name before it (`laptop-20261104T172458Z-11ac40f5`, `20260904T074157Z_cd4d8f6d9a91`);
- no line carries the word `laptop` at all, and no line carries `Rivanna` or `Frontier` except the lines of
  `KEPT_LINES`, each listed by page and exact text with the ground on which the page keeps the name of a cluster as
  an example.

Population: every tracked page under `docs/` outside `docs/reference/bundles/`, `README.md`, and the two published
surfaces that the page tests read through `tests/docs/test_published_surfaces_style.py` (the configuration-schema
page rendered through its hook, and the docstrings the API pages render). A line inside a fenced block counts,
because a reader copies it.

Class: REGRESSION against the pages as they stood after the merge of `develop` into the run, which held a page of
timings, four run identifiers and 29 lines naming `rivanna` or `laptop`. The controls below plant a run identifier,
`laptop` and `Rivanna` on pages that `KEPT_LINES` does not cover and expect the refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.docs.test_docs_style import tracked_pages
from tests.docs.test_published_surfaces_style import surface_pages

_REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_ID = re.compile(r"\b20\d{6}T\d{6}Z[-_][0-9a-f]{7,40}\b", re.IGNORECASE)
LAPTOP = re.compile(r"laptop", re.IGNORECASE)
CLUSTER = re.compile(r"rivanna|frontier", re.IGNORECASE)

_PROFILE = "docs/how-to/hpc-profile-setup.md"
_FILE_NAME_GROUND = (
    "The line shows the name of a shipped file under `test_data/` that a reader copies or types into a command; "
    "the name is a file name, not prose."
)
_SYSTEM_KEY_GROUND = (
    "The line is a docstring example whose `system` argument takes the registered endpoint key `frontier`; "
    "the key is a code value a reader types, not prose."
)

#: Lines that keep the name of a cluster, by page and exact line (trailing whitespace removed), each with its ground
#: (a named cluster stays only as an example, and a placeholder is preferred where the page does not need the
#: name).
KEPT_LINES: dict[tuple[str, str], str] = {
    (_PROFILE, "hpc_name: uva_rivanna"): (
        "The line is the content of the shipped profile file that a reader copies; the value is a file name, not prose."
    ),
    (
        "docs/tutorials/norfolk-compute-config-sweep.md",
        "    --hpc-system-config hpc_system_config_frontier.yaml",
    ): _FILE_NAME_GROUND,
    ("docs/reference/api-analysis.md", '    system="frontier",'): _SYSTEM_KEY_GROUND,
    ("docs/reference/api-analysis.md", '        system="frontier",'): _SYSTEM_KEY_GROUND,
    (_PROFILE, "    | Field | UVA Rivanna | OLCF Frontier |"): (
        "The table compares the two shipped example profiles and the column headers name the clusters they describe."
    ),
}


def local_result_problems(pages: dict[str, str], kept: dict[tuple[str, str], str] | None = None) -> list[str]:
    """Problems found in `pages` (page path -> text): a run id, `laptop`, an unkept `Rivanna`, or a stale kept line."""
    kept = KEPT_LINES if kept is None else kept
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for page, text in sorted(pages.items()):
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.rstrip()
            if RUN_ID.search(line):
                problems.append(f"run identifier: {page}:{lineno}: {line[:120]}")
            if LAPTOP.search(line):
                problems.append(f"laptop: {page}:{lineno}: {line[:120]}")
            if CLUSTER.search(line):
                if (page, line) in kept:
                    seen.add((page, line))
                else:
                    problems.append(f"unkept cluster name: {page}:{lineno}: {line[:120]}")
    problems += [f"stale kept line: {page}: {line[:120]}" for (page, line) in sorted(set(kept) - seen)]
    problems += [f"kept line carries laptop: {page}: {line[:120]}" for (page, line) in kept if LAPTOP.search(line)]
    return problems


def _population() -> dict[str, str]:
    """The tracked pages plus the two published surfaces, keyed by the page or file the surface names."""
    pages = dict(tracked_pages(_REPO_ROOT))
    for surface in ("config-schema", "api-docstring"):
        pages.update(surface_pages(surface))
    return pages


def test_no_published_line_carries_a_run_id_a_laptop_or_an_unkept_cluster_name():
    """Property: the population holds no run id, no `laptop`, and `Rivanna` only on the kept lines.

    Class: every line of every tracked page, of `README.md`, of the rendered configuration schema and of the rendered
    API docstrings. Why it is required: a page that records a run id, a timing or the name of the machine a
    contributor works on is a local result that goes stale and reads as a project rule. What kills it: a page that
    gains such a line, a `Field` description that names a cluster, or a kept line edited so that it no longer
    matches. A second correct implementation: a matcher over sentences instead of lines, with the same kept list.
    """
    problems = local_result_problems(_population())
    assert problems == [], "\n".join(problems[:50]) + f"\n({len(problems)} problems)"


def test_the_population_reaches_the_surfaces_and_the_kept_lines():
    """Reachability: the population holds the schema page, API pages, `README.md` and every kept line's page."""
    pages = _population()
    assert "docs/reference/config-schema.md" in pages
    assert any(p.startswith("docs/reference/api-") for p in pages)
    assert "README.md" in pages
    assert {page for page, _ in KEPT_LINES} <= set(pages)


def test_control_a_planted_run_id_fails():
    """Must-fire control: a run id planted on a page that `KEPT_LINES` does not cover is a problem."""
    pages = {"docs/index.md": "A run named host-20261104T172458Z-11ac40f5 took a while.\n"}
    assert local_result_problems(pages, kept={}) == [
        "run identifier: docs/index.md:1: A run named host-20261104T172458Z-11ac40f5 took a while."
    ]


def test_control_other_spellings_of_a_run_id_fail():
    """Must-fire control: a run id in another spelling than the first planted one fails, which varies the property.

    Class: the spellings the harness and a reader can produce: the harness's own form (an underscore before the digest),
    an upper-case host, an underscore in the host, an upper-case digest, a lower-case `t` and `z`, and no host.
    """
    spellings = (
        "20260904T074157Z_cd4d8f6d9a91",
        "Node7-20261104T172458Z-11ac40f5",
        "dev_box-20261104T172458Z-11ac40f5",
        "host-20261104T172458Z-11AC40F5",
        "host-20261104t172458z-11ac40f5",
        "20261104T172458Z-11ac40f5",
    )
    for spelling in spellings:
        problems = local_result_problems({"docs/index.md": f"See {spelling} for the run.\n"}, kept={})
        assert [p.split(":")[0] for p in problems] == ["run identifier"], (spelling, problems)


def test_control_a_planted_laptop_fails():
    """Must-fire control: `laptop` fails on a page `KEPT_LINES` does not cover, in any case."""
    for word in ("laptop", "Laptop", "LAPTOP"):
        pages = {"docs/index.md": f"Run the fast tier on a {word}.\n"}
        problems = local_result_problems(pages, kept={})
        assert [p.split(":")[0] for p in problems] == ["laptop"], (word, problems)


def test_control_a_planted_frontier_fails_unless_the_line_is_kept():
    """Must-fire control: `Frontier` fails like `Rivanna` on a page `KEPT_LINES` does not cover, in any case."""
    for word in ("Frontier", "frontier", "FRONTIER"):
        line = f"Build the images on {word} scratch."
        problems = local_result_problems({"docs/index.md": line + "\n"}, kept={})
        assert [p.split(":")[0] for p in problems] == ["unkept cluster name"], (word, problems)
        assert local_result_problems({"docs/index.md": line + "\n"}, kept={("docs/index.md", line): "ground"}) == []


def test_control_a_planted_rivanna_fails_unless_the_line_is_kept():
    """Must-fire control: `Rivanna` fails on a page that `KEPT_LINES` does not cover, and passes on a kept line."""
    line = "Compile on Rivanna scratch."
    problems = local_result_problems({"docs/index.md": line + "\n"}, kept={})
    assert [p.split(":")[0] for p in problems] == ["unkept cluster name"], problems
    assert local_result_problems({"docs/index.md": line + "\n"}, kept={("docs/index.md", line): "ground"}) == []


def test_control_a_kept_line_that_no_page_holds_is_stale_and_a_kept_laptop_is_refused():
    """Must-fire control: the kept list is graded both ways, so a deleted line and a kept `laptop` each fail."""
    stale = local_result_problems({"docs/index.md": "Plain.\n"}, kept={("docs/index.md", "Rivanna gone"): "ground"})
    assert [p.split(":")[0] for p in stale] == ["stale kept line"], stale
    both = local_result_problems(
        {"docs/index.md": "A laptop on Rivanna.\n"}, kept={("docs/index.md", "A laptop on Rivanna."): "ground"}
    )
    assert sorted(p.split(":")[0] for p in both) == ["kept line carries laptop", "laptop"], both
