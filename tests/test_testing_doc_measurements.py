"""Every measured figure on ``docs/explanation/testing-the-toolkit.md`` names its run and matches that run's record.

The page quotes laptop timings and memory figures. A figure with no run id cannot be traced to a measurement, a
figure that differs from its run's record is a typing slip, and a concurrency recommendation whose arithmetic does not
follow from its stated inputs is wrong whatever the inputs say. Four checks, each with a must-fire arm built on a
constructed or mutated copy of the page text and a control arm on the recorded page:

1. every figure with a unit (seconds, minutes, GiB, MiB) sits in a block that names a run id, the pre-existing timing
   figures included (a block is a paragraph, or one list item with its continuation lines);
2. the one block that states a concurrency recommendation (it contains ``floor(``) names a run id and carries its four
   inputs, each with a unit;
3. the recommendation is recomputed as floor((MemAvailable at start - R) / peak use) from those inputs in ONE unit,
   the three inputs must share that unit, R must equal the larger of 2 GiB and 10% of MemTotal, and any other
   recommended number reds;
4. each figure on the page is matched by a LABEL pattern (``LABELS`` below: wall seconds, MemAvailable at start, peak
   session PSS, and so on), attributed to the run id that last precedes it in its block (the first id of the block when
   none precedes it), and compared with the value ``RECORDED`` holds for that label of that run. A figure with a unit
   that no label pattern matches reds as unlabelled, so a swap of two figures of one run, a figure of another run
   under this run's label, and a one-digit change all red. The recorded values are embedded here, read from the
   evidence files named in the comment above ``RECORDED``; the check proves the page agrees with those values, not
   that the values are the measurements.

Nothing here runs a solver or reads the repository beyond the page.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "explanation" / "testing-the-toolkit.md"

_RUN_ID = re.compile(r"laptop-\d{8}T\d{6}Z-[0-9a-f]{8}")
#: A figure with a unit: a number, optional space, then a time or memory unit. ``min`` is not matched alone.
_FIGURE = re.compile(r"(?<![\w.,-])(\d[\d,]*(?:\.\d+)?)\s*(seconds|minutes|GiB|MiB|kB|s)\b")
_TIME_UNITS = {"s", "seconds", "minutes"}

#: The values each run's record holds, per LABEL, read from the evidence files named per run. Values are strings with
#: commas removed; a label that captures several numbers joins them with ``:`` in the order the pattern lists them.
#:
#: laptop-20261104T172458Z-11ac40f5 and laptop-20261104T173100Z-11ac40f5 (invocations 1 and 2 of DR-11):
#:   evidence directory 5a/laptop_dr11: inv1.log / inv2.log (last pytest line: wall seconds and counts),
#:   inv1_state.txt / inv2_state.txt (HEAD, ``free -m`` MemAvailable in MiB and swap, load average),
#:   samples.jsonl (peak ``pss_kB``, minimum MemAvailable, peak SwapTotal - SwapFree, MemTotal, the difference of
#:   ``psi_memory.some.total`` first to last sample of the phase), control_check.txt (2.004 vs 2.000).
#: laptop-20261104T064202Z-47e1074a (5a.2's earlier pooled run):
#:   evidence directory 5a/laptop_5a2: tier.log (last pytest line), start_state.txt, samples.jsonl (first loadavg).
#: laptop-20261105T033340Z-cfd68f2a (the developer-authorised re-measure of the pool; the id is the POOL START time):
#:   evidence directory 5a/laptop_remeasure: pool.log (last pytest line), pool_start.txt, pool_state.txt (HEAD,
#:   ``free -m`` MemAvailable in MiB and swap, load average, the ``git grep`` line), samples.jsonl (phase ``pool``: peak
#:   ``pss_kB``, minimum MemAvailable, peak SwapTotal - SwapFree, the ``psi_memory.some.total`` difference, the maximum
#:   of the first loadavg field), control_check.txt (2.004 vs 2.000), ledger_pool.jsonl (summed ``call`` durations
#:   3508.2 s against ledger_inv1.jsonl's 3026.1 s of laptop_dr11). Differences: 428.16 - 354.57 = 73.59 s;
#:   73.59 / 354.57 = 20.8 %. The cited figures 10.47 and 4.93 are DR-11's peak PSS and peak swap.
#: Derived values: 20.25 = 30.72 - 10.47; 20.78 = 21,279 / 1,024; 3.07 = max(2, 0.1 * 30.72); 1,852.85 = 354.57 +
#: 1,498.28 (recorded under the pool, the run id that precedes it on the page); 300 is the target the shape sets.
RECORDED: dict[str, dict[str, str]] = {
    "laptop-20261104T172458Z-11ac40f5": {
        "wall": "354.57",
        "target": "300",
        "mem_available_start_mib": "21279",
        "swap_start": "4777:8191",
        "load": "4.14",
        "peak_pss": "10.47",
        "min_mem_available": "11.08",
        "peak_swap": "4.93:8.00",
        "headroom": "30.72:10.47:20.25",
        "stall": "2.19",
        "control": "2:20:2.004:2.000",
        "combined_wall": "1852.85",
        "mem_available_start_gib": "20.78",
        "mem_total": "30.72",
        "reserve_floor": "2",
        "reserve": "3.07",
        "peak_use": "10.47",
        "floor_expr": "20.78:3.07:10.47",
        "wall_cited": "354.57",
    },
    "laptop-20261104T173100Z-11ac40f5": {
        "wall": "1498.28",
        "mem_available_start_mib": "22026",
        "swap_start": "5037:8191",
        "load": "18.68",
        "peak_pss": "2.00",
        "peak_swap": "4.92",
    },
    "laptop-20261104T064202Z-47e1074a": {
        "wall": "843.49",
        "mem_available_start_mib": "15803",
        "swap_start": "5630:8191",
        "load": "2.88",
    },
    "laptop-20261105T033340Z-cfd68f2a": {
        "wall": "428.16",
        "target": "300",
        "mem_available_start_mib": "17920",
        "swap_start": "5512:8191",
        "load": "3.98",
        "load_max": "24.99",
        "peak_pss": "8.11",
        "min_mem_available": "10.96",
        "peak_swap": "7.99:8.00",
        "stall": "13.88",
        "control": "2:20:2.004:2.000",
        "delta": "73.59:20.8",
        "summed_call": "3508.2:3026.1",
        "pss_vs": "8.11:10.47",
        "swap_vs": "7.99:4.93",
    },
}

_N = r"(\d[\d,]*(?:\.\d+)?)"  # one number, commas allowed
#: label -> pattern. Single spaces match any run of whitespace, so the page may wrap lines anywhere. Capture groups
#: are the recorded numbers, joined with ``:``. Patterns are anchored on the words around the figure, so the same
#: number under a different label (a swap) does not match this label's record.
LABELS: dict[str, str] = {
    "wall": rf"{_N} s, `\d+ (?:failed|passed)",
    "target": rf"{_N} s target",
    "mem_available_start_mib": rf"(?<!minimum\x20)(?<!at\x20start\x20)MemAvailable {_N} MiB",
    "swap_start": rf"swap {_N} of {_N} MiB used",
    "load": rf"load average {_N}",
    "peak_pss": rf"(?i:peak session PSS) {_N} GiB",
    "min_mem_available": rf"minimum MemAvailable {_N} GiB",
    "peak_swap": rf"peak swap used {_N} GiB(?: of {_N} GiB)?",
    "headroom": rf"MemTotal {_N} - {_N} = {_N} GiB",
    "stall": rf"stall {_N} s",
    "control": rf"known allocation of {_N} GiB held for {_N} s .*? peak of {_N} GiB against {_N} GiB",
    "combined_wall": rf"the two take {_N} s",
    "mem_available_start_gib": rf"MemAvailable at start {_N} GiB",
    "mem_total": rf"MemTotal {_N} GiB",
    "reserve_floor": rf"larger of {_N} GiB and",
    "reserve": rf"\bR {_N} GiB",
    "peak_use": rf"peak use {_N} GiB",
    "floor_expr": rf"floor\(\({_N} - {_N}\) / {_N}\)",
    "load_max": rf"maximum of {_N}",
    "delta": rf"took {_N} s \({_N} %\) longer",
    "summed_call": rf"summed call time was {_N} s against {_N} s",
    "pss_vs": rf"peak session PSS {_N} GiB against {_N} GiB",
    "swap_vs": rf"swap use peaked at {_N} GiB against {_N} GiB",
    "wall_cited": rf"\({_N} s\)",
}
_LABEL_RE = {k: re.compile(v.replace(" ", r"\s+"), re.S) for k, v in LABELS.items()}

#: Per-run values the floor check is also held to: (MemAvailable at start GiB, MemTotal GiB, peak use GiB).
RECOMMENDATION_INPUTS = {"laptop-20261104T172458Z-11ac40f5": ("20.78", "30.72", "10.47")}

_TO_GIB = {"kB": 1 / (1024 * 1024), "MiB": 1 / 1024, "GiB": 1.0}


def _normalise(figure: str) -> str:
    return figure.replace(",", "")


def blocks(text: str) -> list[str]:
    """Paragraphs, with each list item (a line starting ``- ``) a block of its own with its continuation lines."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in para.split("\n"):
            if line.lstrip().startswith("- ") and current:
                out.append("\n".join(current))
                current = []
            current.append(line)
        out.append("\n".join(current))
    return out


def figures(block: str) -> list[tuple[str, str]]:
    """(normalised number, unit) for every figure with a unit in the block."""
    return [(_normalise(m.group(1)), m.group(2)) for m in _FIGURE.finditer(block)]


def figures_without_run_id(text: str) -> list[str]:
    """Every figure with a unit that sits in a block naming no run id (timing and memory figures alike)."""
    bad = []
    for block in blocks(text):
        if _RUN_ID.search(block):
            continue
        bad += [f"{num} {unit}" for num, unit in figures(block)]
    return bad


def timing_figures_without_run_id(text: str) -> list[str]:
    """The subset of ``figures_without_run_id`` that is a timing figure (seconds or minutes)."""
    return [f for f in figures_without_run_id(text) if f.rsplit(" ", 1)[1] in _TIME_UNITS]


def source_findings(text: str) -> list[str]:
    """Figures that disagree with their run's record, are unlabelled, or cite an unrecorded run id.

    Per block: every label pattern is searched; a match is attributed to the run id that last precedes it (the first
    id of the block when none does) and its captured numbers are compared with ``RECORDED[run][label]``. Then every
    figure with a unit must lie inside some label match, or it reds as unlabelled.
    """
    bad: list[str] = []
    for block in blocks(text):
        ids = [(m.start(), m.group(0)) for m in _RUN_ID.finditer(block)]
        if not ids:
            continue
        unknown = {i for _, i in ids} - set(RECORDED)
        if unknown:
            bad.append(f"run id(s) with no recorded values: {sorted(unknown)}")
            continue
        spans: list[tuple[int, int]] = []
        for label, pattern in _LABEL_RE.items():
            for m in pattern.finditer(block):
                spans.append(m.span())
                run = next((i for pos, i in reversed(ids) if pos < m.start()), ids[0][1])
                got = ":".join(_normalise(g) for g in m.groups() if g is not None)
                want = RECORDED[run].get(label)
                if want != got:
                    bad.append(f"{label} {got!r} on {run} (recorded {want!r})")
        for m in _FIGURE.finditer(block):
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                bad.append(f"unlabelled figure {m.group(0)!r}")
    return bad


def _recommendation_block(text: str) -> str | None:
    found = [b for b in blocks(text) if "floor(" in b]
    return found[0] if len(found) == 1 else None


_LABELLED = {
    "mem_available": re.compile(r"MemAvailable at start ([\d,.]+) (kB|MiB|GiB)"),
    "mem_total": re.compile(r"MemTotal ([\d,.]+) (kB|MiB|GiB)(?!\s*-)"),
    "reserve": re.compile(r"\bR ([\d,.]+) (kB|MiB|GiB)"),
    "peak": re.compile(r"peak use ([\d,.]+) (kB|MiB|GiB)"),
}
_STATED_FLOOR = re.compile(r"floor\([^=]*\)\s*=\s*(\d+)")


def recommendation_findings(text: str) -> list[str]:
    """Problems with the concurrency recommendation: missing run id or inputs, mixed units, wrong floor or reserve."""
    block = _recommendation_block(text)
    if block is None:
        return ["no single block states a concurrency recommendation (a block containing 'floor(')"]
    bad: list[str] = []
    ids = set(_RUN_ID.findall(block))
    if not ids:
        bad.append("the recommendation names no run id")
    if "estimate from one run" not in block:
        bad.append("the recommendation lacks the label 'estimate from one run'")
    found = {name: pattern.search(block) for name, pattern in _LABELLED.items()}
    missing = [name for name, m in found.items() if m is None]
    if missing:
        return [*bad, f"the recommendation lacks its input(s): {missing}"]
    values = {name: (float(_normalise(m.group(1))), m.group(2)) for name, m in found.items()}
    units = {unit for _, unit in values.values()}
    if len(units) != 1:
        return [*bad, f"the recommendation's inputs are in more than one unit: {sorted(units)}"]
    stated = _STATED_FLOOR.search(block)
    if stated is None:
        return [*bad, "the recommendation states no 'floor(...) = N'"]
    (avail, _), (total, _), (reserve, _), (peak, _) = (
        values[k] for k in ("mem_available", "mem_total", "reserve", "peak")
    )
    gib = _TO_GIB[next(iter(units))]
    expected_reserve = max(2.0, 0.10 * total * gib) / gib
    if abs(reserve - expected_reserve) > 0.01 * expected_reserve + 1e-9:
        bad.append(f"R {reserve} is not the larger of 2 GiB and 10% of MemTotal ({expected_reserve:.4g})")
    floor = math.floor((avail - reserve) / peak)
    if int(stated.group(1)) != floor:
        bad.append(f"the stated floor {stated.group(1)} differs from floor(({avail} - {reserve}) / {peak}) = {floor}")
    for run_id in ids & set(RECOMMENDATION_INPUTS):
        want = RECOMMENDATION_INPUTS[run_id]
        got = tuple(f"{values[k][0]:.2f}" for k in ("mem_available", "mem_total", "peak"))
        if got != want:
            bad.append(f"inputs {got} differ from the recorded {want} for {run_id}")
    return bad


# --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


_STALE = (
    "The tier is green. Its latest recorded timing, measured on the laptop, is 389.5 s for the\n"
    "xdist invocation and 1032.3 s for the serial one, about 23.7 minutes in total.\n"
)


def test_the_recorded_page_passes_every_check(doc_text):
    """Control arm: the page as recorded has no unsourced figure, a sound recommendation and no source mismatch."""
    assert figures_without_run_id(doc_text) == []
    assert recommendation_findings(doc_text) == []
    assert source_findings(doc_text) == []
    assert {"354.57", "1498.28", "843.49"} <= {n for b in blocks(doc_text) for n, _ in figures(b)}, "no vacuous pass"


def test_a_timing_figure_without_a_run_id_reds_including_the_pre_existing_ones():
    """Must-fire: the stale paragraph (389.5 s, 1032.3 s, 23.7 minutes, no run id) is reported figure by figure.

    Killed by a scan that only checks the new figures, or one that accepts a run id anywhere on the page.
    """
    found = timing_figures_without_run_id(_STALE)
    assert found == ["389.5 s", "1032.3 s", "23.7 minutes"]


def test_the_scan_covers_a_figure_whose_block_lacks_the_run_id_even_when_another_block_has_one(doc_text):
    """Stripping every run id from one bullet reds that bullet's figures only; other blocks' ids are no cover."""
    victim = next(b for b in blocks(doc_text) if "1,498.28 s" in b)
    mutated = doc_text.replace(victim, _RUN_ID.sub("a run", victim))
    assert mutated != doc_text
    found = figures_without_run_id(mutated)
    assert "1498.28 s" in found and "354.57 s" not in found and "843.49 s" not in found


def test_a_recommendation_without_its_run_id_or_its_inputs_reds(doc_text):
    """Must-fire: dropping the run id, one input, or the label from the recommendation block each reds."""
    block = _recommendation_block(doc_text)
    assert block is not None
    no_id = doc_text.replace(block, _RUN_ID.sub("a run", block))
    assert any("no run id" in f for f in recommendation_findings(no_id))
    no_input = doc_text.replace(block, block.replace("peak use 10.47 GiB", "peak use"))
    assert any("lacks its input" in f for f in recommendation_findings(no_input))
    no_label = doc_text.replace(block, block.replace("estimate from one run", "estimate"))
    assert any("label" in f for f in recommendation_findings(no_label))


def test_a_floor_that_differs_by_one_reds(doc_text):
    """Must-fire: the right inputs with floor = 2 (and = 0) are rejected; the recomputation uses the stated inputs."""
    for wrong in ("2", "0"):
        mutated = doc_text.replace("/ 10.47) = 1", f"/ 10.47) = {wrong}")
        assert mutated != doc_text
        assert any("stated floor" in f for f in recommendation_findings(mutated))


def test_inputs_in_kb_with_the_reserve_in_gib_red(doc_text):
    """Must-fire: MemAvailable, MemTotal and peak use in kB beside a reserve in GiB mix units, so the page reds.

    Killed by a check that converts every input to a common unit silently (it would pass: the floor is still 1).
    """
    block = _recommendation_block(doc_text)
    assert block is not None
    kb = block.replace("MemAvailable at start 20.78 GiB", "MemAvailable at start 21,790,000 kB")
    kb = kb.replace("MemTotal 30.72 GiB", "MemTotal 32,210,528 kB").replace(
        "peak use 10.47 GiB", "peak use 10,977,000 kB"
    )
    findings = recommendation_findings(doc_text.replace(block, kb))
    assert any("more than one unit" in f for f in findings), findings


def test_a_reserve_that_is_not_the_larger_of_two_gib_and_ten_percent_reds(doc_text):
    """Must-fire: R = 2.00 GiB (the floor 2 GiB alone) is not max(2 GiB, 10% of 30.72 GiB) = 3.07 GiB."""
    mutated = doc_text.replace("R 3.07 GiB", "R 2.00 GiB").replace("- 3.07)", "- 2.00)")
    assert mutated != doc_text
    assert any("not the larger of 2 GiB" in f for f in recommendation_findings(mutated))


def test_a_changed_input_that_keeps_the_floor_still_reds_against_the_record(doc_text):
    """The recomputation takes its inputs from the page, so the record check is what catches a drifted input."""
    mutated = doc_text.replace("peak use 10.47 GiB", "peak use 10.37 GiB").replace("/ 10.47)", "/ 10.37)")
    assert mutated != doc_text
    assert recommendation_findings(mutated) != []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("354.57 s", "354.58 s"),
        ("1,498.28 s", "1,498.29 s"),
        ("843.49 s", "843.48 s"),
        ("peak session PSS\n  10.47 GiB", "peak session PSS\n  10.57 GiB"),
        ("2.19 s", "2.18 s"),
        ("MemAvailable 21,279 MiB", "MemAvailable 21,278 MiB"),
        ("428.16 s", "428.17 s"),
        ("memory-pressure stall 13.88 s", "memory-pressure stall 13.89 s"),
        ("3,508.2 s against", "3,508.3 s against"),
    ],
)
def test_a_quoted_figure_changed_by_one_digit_reds_against_its_runs_record(doc_text, old, new):
    """Must-fire (source check): one digit of a quoted figure changed, with the run id still present, reds.

    Killed by a check that only tests for the presence of a run id. The control is the unmutated page (empty findings).
    """
    assert source_findings(doc_text) == []
    mutated = doc_text.replace(old, new)
    assert mutated != doc_text, old
    assert source_findings(mutated) != []


def test_a_run_id_with_no_recorded_values_reds(doc_text):
    """A block citing a run id the record does not hold reds, so a new run cannot be cited before it is recorded."""
    mutated = doc_text.replace("laptop-20261104T064202Z-47e1074a", "laptop-20261104T999999Z-47e1074a")
    assert any("no recorded values" in f for f in source_findings(mutated))


#: A claim needs a verb ("is", "currently", "now") between "tier" and "green", or "tier ... passes", so the page's own
#: disclaimer ("does not call the tier green") is not a claim; matched case-insensitively across line wraps.
_GREEN_CLAIM = re.compile(
    r"\btier\s+(?:"
    r"is\s+(?:currently\s+|now\s+|all\s+)?green"
    r"|(?:currently\s+|now\s+)green"
    r"|(?:currently\s+|now\s+)?passes"
    r")\b",
    re.I,
)


def green_claim_findings(text: str) -> list[str]:
    """Red only when the page claims the tier is green while a run of its newest HEAD carries failures or errors.

    The newest HEAD is the id suffix of the run with the greatest timestamp among the ids on the page; a run's failures
    are the ``N failed`` and ``N errors`` counts in its block. A page that makes no claim never reds, and a truthful
    claim after a clean run at a newer HEAD is allowed, so the check does not outlive the failures it guards against.

    This is a SPELLING-BASED TRIPWIRE, not a reader of meaning. It matches the spellings ``_GREEN_CLAIM`` names (for
    example 'The tier is green', 'the tier is currently green', 'the tier passes'). It is silent on other phrasings
    measured against it ('The suite is green on the laptop.', 'Everything passes on the laptop tier.', 'The tier
    passed.', 'No test fails in the laptop tier.', 'The laptop tier is passing.'), and it has one known false positive,
    in the conservative direction ('Whether the tier passes is not recorded here.'). It guards the page against the
    claims this run saw, not against every way of saying it.
    """
    if not _GREEN_CLAIM.search(text):
        return []
    runs = {}
    for block in blocks(text):
        for run_id in _RUN_ID.findall(block):
            counts = [int(n) for n in re.findall(r"(\d+) (?:failed|errors?)\b", block)]
            runs[run_id] = (
                runs.get(run_id, 0) + sum(counts)
                if block.lstrip().startswith(("- `" + run_id,))
                else runs.get(run_id, 0)
            )
    if not runs:
        return ["the page claims the tier is green and records no run"]
    newest_head = max(runs, key=lambda r: r.split("-")[1]).rsplit("-", 1)[1]
    failing = sorted(r for r, n in runs.items() if r.endswith(newest_head) and n)
    return [f"claims the tier is green but {r} (HEAD {newest_head}) carries failures" for r in failing]


def _page_before_the_remeasure(text: str) -> str:
    """The page without the bullet of the re-measure run ``laptop-20261105T033340Z-cfd68f2a`` (the passing one).

    The green-claim arms were written when the newest recorded HEAD carried failures. The re-measure passed all its
    tests, so on the recorded page the newest HEAD is clean and a claim is no longer forbidden by this guard. These arms
    keep their meaning by removing that one bullet, which restores the premise they test: the newest HEAD has failures.
    """
    victim = next(b for b in blocks(text) if b.lstrip().startswith("- `laptop-20261105T033340Z-cfd68f2a`"))
    return text.replace(victim, "")


def test_the_page_does_not_call_the_tier_green_while_its_newest_head_has_failures(doc_text):
    """Control: today's page makes no green claim. Must-fire: adding one reds, because the newest HEAD's pool failed."""
    assert green_claim_findings(doc_text) == []
    claimed = _page_before_the_remeasure(doc_text) + "\nThe tier is green.\n"
    assert any("laptop-20261104T172458Z-11ac40f5" in f for f in green_claim_findings(claimed))


def test_a_green_claim_after_a_clean_run_at_a_newer_head_is_not_forbidden(doc_text):
    """The guard is conditioned on the recorded failures: a later clean run at a new HEAD makes the claim truthful."""
    clean = "\n- `laptop-20261105T090000Z-abcdef12`: the pool, 12 workers, 290.00 s, `6000 passed`.\n"
    assert green_claim_findings(doc_text + clean + "\nThe tier is green.\n") == []
    dirty = "\n- `laptop-20261105T090000Z-abcdef12`: the pool, 12 workers, 290.00 s, `1 failed, 5999 passed`.\n"
    assert green_claim_findings(doc_text + dirty + "\nThe tier is green.\n") != []


def test_two_figures_swapped_within_one_run_red(doc_text):
    """Must-fire (blind spot 1): peak session PSS 10.47 GiB and minimum MemAvailable 11.08 GiB exchanged.

    Both are recorded for the same run id, so a set-membership check passes; the record must map metric to value.
    """
    mutated = doc_text.replace("10.47 GiB, minimum MemAvailable 11.08 GiB", "11.08 GiB, minimum MemAvailable 10.47 GiB")
    if mutated == doc_text:
        mutated = re.sub(
            r"peak session PSS\s+10\.47 GiB, minimum MemAvailable 11\.08 GiB",
            "peak session PSS 11.08 GiB, minimum MemAvailable 10.47 GiB",
            doc_text,
        )
    assert mutated != doc_text
    assert source_findings(mutated) != []


def test_another_runs_timing_in_a_block_citing_both_runs_reds(doc_text):
    """Must-fire (blind spot 2): the serial 1,498.28 s replaced by the pool's 354.57 s in the bullet citing both."""
    victim = next(b for b in blocks(doc_text) if "1,498.28 s" in b)
    assert len(set(_RUN_ID.findall(victim))) == 2, "the bullet must cite both runs for this arm to mean anything"
    mutated = doc_text.replace(victim, victim.replace("1,498.28 s", "354.57 s", 1))
    assert mutated != doc_text
    assert source_findings(mutated) != []


@pytest.mark.parametrize(
    "sentence",
    [
        "The laptop tier is currently green on this machine.",
        "On the laptop the tier is green.",
        "All of the laptop tier passes.",
    ],
)
def test_other_spellings_of_the_green_claim_red_while_the_newest_head_has_failures(doc_text, sentence):
    """Must-fire (reviewer): the guard must not protect one spelling only. Control: the page with no claim is clean."""
    assert green_claim_findings(doc_text) == []
    assert green_claim_findings(_page_before_the_remeasure(doc_text) + f"\n{sentence}\n") != []


def test_deleting_the_new_runs_id_from_its_bullet_reds(doc_text):
    """Must-fire: the re-measure bullet without its run id leaves its figures attributed to another run or to none."""
    mutated = doc_text.replace("- `laptop-20261105T033340Z-cfd68f2a`:", "- `a run`:")
    assert mutated != doc_text
    assert source_findings(mutated) != [] or figures_without_run_id(mutated) != []


def test_the_counting_sentence_names_the_two_separate_selections(doc_text):
    """Reviewer arm. The two runs made as separate selections are the FIRST (pool without the 96) and the SECOND (the 96
    alone); the third run (``laptop-20261104T064202Z-47e1074a``) is a whole pool with the 96 inside and is not one of
    them. The page's opening must not name the third as a separate selection."""
    opening = " ".join(next(b for b in blocks(doc_text) if "What has been measured on the laptop" in b).split())
    assert "The first and second were made as separate selections" in opening, opening
    assert "The second and third were made as separate selections" not in opening
