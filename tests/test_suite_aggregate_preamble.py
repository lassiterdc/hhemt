"""The suite summary's advisories preamble explains every advisory tag the aggregator can emit.

The problem this file guards: `aggregate()` puts some findings into `advisories` rather than
`problems`, each led by a `[name]` tag, and `render_summary_md` prints a standing preamble that
explains how to read each kind and says how many kinds there are. The preamble is hand-written
while the tags live at their emitting sites, so a new advisory site can ship with no reading
rule and a stale count, and a reader then has an unexplained finding under a heading that says
it did not affect the verdict. This test collects the tags from the module's own source and
requires the rendered preamble to explain each and to count them correctly.

Nothing here runs a suite: the tags are read from source with `ast`, and the summary is
rendered from a minimal hand-built result.
"""

import ast
import inspect
import re

import hhemt.suite.aggregate as aggregate_module
from tests.test_verdict_partition import SOFTWARE, _build, _entry

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_ADVISORY_LISTS = {"advisories", "entry_advisories"}


def _emitted_advisory_tags() -> set[str]:
    """Return every leading `[name]` of a string appended to an advisory list in the module."""
    tags: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(aggregate_module))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _ADVISORY_LISTS
            and node.args
        ):
            continue
        first = node.args[0]
        head = first.values[0] if isinstance(first, ast.JoinedStr) else first
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            match = re.match(r"\[([a-z-]+)\]", head.value)
            if match:
                tags.add(match.group(1))
    return tags


class _LenientResult(dict):
    """A result dict whose unset keys render as a placeholder, so only advisories matter."""

    def __missing__(self, key):
        return "-"


def _rendered_preamble() -> str:
    result = _LenientResult(
        scope="array",
        run_id="r",
        verdict_line="v",
        problems=[],
        advisories=["[placeholder] an advisory, so the section renders"],
        unevaluated_by_cause={},
        chunks=[],
        slug_entry_classification={},
        ignored_transient_entries=[],
    )
    text = aggregate_module.render_summary_md(result)
    start = text.index("## Advisories")
    return text[start : text.index("- [placeholder]", start)]


def test_preamble_explains_and_counts_every_emitted_advisory_tag():
    """Property: every advisory tag `aggregate()` can emit has a reading-rule line in the
    rendered preamble, and the preamble's stated count of kinds equals the lines it lists.

    Class: every `[name]` led string appended to `advisories` or `entry_advisories` anywhere in
    the module, against the preamble `render_summary_md` prints whenever advisories exist. Why it
    is required: the advisories section says its findings did not affect the verdict, so a kind
    with no reading rule is the one a reader cannot tell is safe to ignore, and a count that
    disagrees with the list tells the reader the list is incomplete. What kills it: an advisory
    site whose tag has no preamble line (as `[complement-derivation]` had), or a stated count that
    is not the number of tag lines. A second correct implementation under which it still passes:
    a preamble generated from a tag registry rather than written by hand, since only the
    rendered lines and the stated count are read.
    """
    emitted = _emitted_advisory_tags()
    assert {"classifier-self-check", "shared-by-design-write", "derivation-under-connected"} <= emitted, (
        f"tag collection is broken: found only {sorted(emitted)}"
    )

    preamble = _rendered_preamble()
    listed = set(re.findall(r"^- `\[([a-z-]+)\]`", preamble, flags=re.MULTILINE))
    assert emitted - listed == set(), f"emitted but unexplained in the preamble: {sorted(emitted - listed)}"

    counts = re.findall(r"\b(" + "|".join(_NUMBER_WORDS) + r")\s+kinds\b", preamble.lower())
    assert counts, f"the preamble states no count of kinds:\n{preamble}"
    assert {_NUMBER_WORDS[c] for c in counts} == {len(listed)}, (
        f"stated count(s) {counts} disagree with the {len(listed)} kinds listed:\n{preamble}"
    )


def test_shared_by_design_reading_rule_states_the_rule_the_verdict_applies(tmp_path):
    """Property: the preamble's `[shared-by-design-write]` line states the rule `aggregate()`
    applies when it sorts a write to the shared compiled tier into ADVISORY or BLOCKING.

    Class: every write pattern on a shared-by-construction entry, represented by four inputs
    that separate the rule's conditions -- two chunks taking the same lock, two chunks writing
    different files, two chunks writing the same build file, and a chunk recorded as a writer
    with no record of what it wrote. The verdict for each is read from `aggregate()` itself, so
    the line is checked against what the code does rather than against a remembered rule. Why
    it is required: the line tells a reader how to read an advisory that did not affect the
    verdict, and a line that keys BLOCKING on how many chunks wrote the entry tells the reader
    that two chunks taking a lock is a failure, when `aggregate()` reports it as advisory; the
    reader then distrusts a correct GREEN. What kills it: a line that does not say which files
    count as coordination (the two name tests in `_is_coordination_path`), or whose sentence
    stating the BLOCKING condition does not confine it to two or more chunks writing a FILE
    other than a coordination file, or omits the writer that left no record. So each of these
    is refused: the entry-count form ("TWO or more writing it"), the "same entry" form, the
    sentence with "that is not a coordination file" deleted, and the same sentence with the
    negation dropped ("that is a coordination file"). Each of those reads the two-writers-one-
    lock control above as BLOCKING. A second correct implementation under which it still
    passes: a line that states the BLOCKING condition BEFORE the word ("Two or more chunks
    recording a write to one path other than a coordination file ... makes it BLOCKING, as does
    a writer that left no record of what it wrote"), since the check reads the whole sentence
    that contains the word rather than the text after it.

    Not established: the negation is matched from a closed vocabulary (not, non-, other than,
    except, excluding, apart from) placed directly before the coordination class, so a correct
    line using some other construction is refused, and a wrong line that keeps that phrase
    while adding a false condition elsewhere in the sentence passes.
    """

    def verdict(samples_per_chunk: list[list[str]]) -> str:
        """Run `aggregate()` on a fresh run in which each chunk wrote `_software` with the given
        samples, and return whether it filed the entry as ADVISORY or BLOCKING."""
        run = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        run.mkdir()
        _build(
            run,
            chunk_entries=[{SOFTWARE: _entry("_software", written=True, samples=s)} for s in samples_per_chunk],
        )
        result = aggregate_module.aggregate(run, scope="array")
        advisory = any(a.startswith("[shared-by-design-write]") for a in result["advisories"])
        blocking = any(p.startswith("[concurrent-build]") for p in result["problems"])
        assert advisory != blocking, f"expected exactly one tag: {result['advisories']} {result['problems']}"
        return "ADVISORY" if advisory else "BLOCKING"

    # Controls on the code, so the line below is compared with the rule the code applies. The
    # first two have two writers and are ADVISORY, which is why the writer count is not the rule.
    assert verdict([[".triton.provision.lock"]] * 2) == "ADVISORY"
    assert verdict([["a.csv"], ["b.csv"]]) == "ADVISORY"
    assert verdict([["triton/build_tritonswmm_cpu/compilation.log"]] * 2) == "BLOCKING"
    assert verdict([[".triton.provision.lock"], []]) == "BLOCKING"
    assert aggregate_module._is_coordination_path(".hidden")
    assert aggregate_module._is_coordination_path("provision.lock")
    assert not aggregate_module._is_coordination_path("compilation.log")

    lines = [ln for ln in _rendered_preamble().splitlines() if ln.startswith("- `[shared-by-design-write]`")]
    assert len(lines) == 1, f"expected one reading-rule line for the tag, found {len(lines)}"
    line = lines[0]
    assert ".lock" in line, f"the line does not name the `.lock` coordination form:\n{line}"
    assert "`.`" in line or "dot" in line, f"the line does not name the leading-dot coordination form:\n{line}"
    # Read the whole SENTENCE that states the BLOCKING condition, wherever the word sits in it.
    # Deliberately NOT the text after the word: a correct line may state the condition first.
    # And NOT the whole line: the ADVISORY sentence already says "except coordination files",
    # so a whole-line search passes a BLOCKING sentence that lost its own exclusion.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
    blocking = [s.lower() for s in sentences if "BLOCKING" in s]
    assert len(blocking) == 1, f"expected one sentence stating the BLOCKING condition, found {len(blocking)}:\n{line}"
    sentence = blocking[0]
    assert re.search(r"\b(two or more|more than one|several|multiple)\b", sentence), (
        f"the BLOCKING sentence does not require more than one chunk:\n{line}"
    )
    assert re.search(r"\b(file|path)s?\b", sentence), (
        f"the BLOCKING sentence names no file or path, so it keys on the entry:\n{line}"
    )
    negated_coordination = (
        r"(\bnot|\bnon-|\bother than|\bexcept|\bexcluding|\bapart from)\s*(an?\s+|any\s+)?(coordination|lock)"
    )
    assert re.search(negated_coordination, sentence), (
        f"the BLOCKING sentence does not exclude coordination files from the collision:\n{line}"
    )
    assert re.search(r"\bno (write )?(record|sample)", sentence), (
        f"the BLOCKING sentence does not name the writer with no record:\n{line}"
    )
