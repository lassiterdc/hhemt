"""Host-independent pin of the committed observed-events figure's spec (chunk 1p.1).

The committed PNG `docs/tutorials/assets/observed-events-irene-result.png` is a render of one figure of the committed
report `docs/reference/bundles/norfolk-observed-events-report.html`. The byte comparison of that PNG against a fresh
render lives in `tests/docs/test_review_10b2_pass.py` and is skipped on any host whose Arial faces differ from the host
that drew the PNG, so on such a host nothing there reads the report at all. This file reads the report on every host:
it extracts the figure spec (data and layout) with `figures_of` and pins its sha256, with no kaleido and no fonts.

What the pin does and does not show. It fails when the figure of the committed report changes, which is the event the
PNG then no longer depicts. It cannot see a PNG that was re-rendered from a different figure of the same page: that
stays with the Arial-gated comparison and with a human re-read of the PNG. Not established: that the PNG on disk was
drawn from this spec, which only a render on the pinned toolchain shows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prepare_example_report_1p1", _REPO / "scripts" / "prepare_example_report.py"
)
per = importlib.util.module_from_spec(_spec)
sys.modules["prepare_example_report_1p1"] = per
_spec.loader.exec_module(per)

_REPORT = _REPO / "docs/reference/bundles/norfolk-observed-events-report.html"
_IRENE = "year.9_event_type.compound_event_id.1"
_TITLE = "TRITON-SWMM: Peak flood depth"

# The digest of the figure titled `_TITLE` on the page of event `_IRENE`, taken over the compact-separator JSON of
# [data, layout] with sorted keys. The compact separators matter: the default separators give a different digest
# (first 16 hex digits 342c6a0adb680294 for this figure).
#
# RE-PIN RULE. Changing this value without re-rendering the PNG defeats the test, because the pin then certifies a
# spec that the committed PNG does not depict. A re-pin is complete only when a human has re-read
# docs/tutorials/assets/observed-events-irene-result.png beside the new figure and found them the same figure.
_FIGURE_SPEC_SHA256 = "2a0a7ea94637103a0f5e3bb7a19b1daf241c761fe11bc9f2a1d32bf0705748ec"


def _spec_digest(data: list, layout: dict) -> str:
    payload = json.dumps([data, layout], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def committed_figure() -> tuple[list, dict]:
    """The (data, layout) of the one figure titled `_TITLE` on the committed report's page for event `_IRENE`."""
    pages = per.decode_pages(_REPORT.read_text(encoding="utf-8"))
    suffix = f"event_page__evt.{_IRENE}.html"
    keys = [k for k in pages if k.endswith("/" + suffix) or k == suffix]
    assert len(keys) == 1, f"{len(keys)} pages match the event {_IRENE}"
    matches = [(d, lay) for title, d, lay in per.figures_of(pages[keys[0]]) if title == _TITLE]
    assert len(matches) == 1, f"{len(matches)} figures carry the title {_TITLE!r} on that page"
    return matches[0]


def test_the_committed_figure_spec_has_the_pinned_digest(committed_figure):
    """Class: the figure of the committed report that the committed PNG depicts, on any host.

    NEW CAPABILITY. Property: sha256 of the compact sorted-key JSON of [data, layout] equals all 64 hex digits of
    `_FIGURE_SPEC_SHA256`. Why required: the Arial-gated byte comparison is skipped on a host with other faces, so
    without this test a changed figure in the report goes unseen there. Plausible wrong implementation: pinning a
    prefix of the digest, or hashing with the default JSON separators; red under the second, since the control below
    shows the default form differs. What kills it: any change to the figure's traces or layout in the committed
    report. A second correct implementation: hashing the same payload through `json.dumps` on a copy built with
    `copy.deepcopy`, which passes, since the digest is of content.
    """
    data, layout = committed_figure
    assert len(_FIGURE_SPEC_SHA256) == 64
    assert _spec_digest(data, layout) == _FIGURE_SPEC_SHA256


def test_a_changed_spec_changes_the_digest(committed_figure):
    """Control: the digest moves under a change to the layout and under a change to the data, and under the default
    separators, so the pin is a function of the spec and of the stated formula and not a constant.

    Class: the property under suspicion is sensitivity of the digest to the spec, so the arms vary the spec, not the
    token. A second correct implementation of the pin, hashing a parsed copy, passes all three arms.
    """
    data, layout = committed_figure
    base = _spec_digest(data, layout)
    assert base == _FIGURE_SPEC_SHA256
    assert _spec_digest(data, {**layout, "title": {"text": "a different title"}}) != base
    assert _spec_digest(data[:-1], layout) != base
    default_form = hashlib.sha256(json.dumps([data, layout], sort_keys=True).encode()).hexdigest()
    assert default_form != base


# ---------------------------------------------------------------------------------------------------------------------
# Reviewer tests (chunk 1p.1). They vary the committed REPORT before extraction, which the two tests above never do:
# those mutate the already-extracted spec in memory, so they cannot see a report edit that `decode_pages`, `figures_of`
# or the page selection fails to carry through to the digest.
# ---------------------------------------------------------------------------------------------------------------------

_OTHER_TITLE = "TRITON: Peak flood depth"


def _page_with_spans(report_text: str) -> tuple[str, str, dict[str, tuple[int, int]]]:
    """(page key, decoded page text, {heading: (end of data JSON, end of layout JSON)}) for the event `_IRENE`."""
    pages = per.decode_pages(report_text)
    key = next(k for k in pages if k.endswith(f"event_page__evt.{_IRENE}.html"))
    page = pages[key]
    heads = list(per._H2.finditer(page))
    decoder = json.JSONDecoder()
    spans = {}
    for n, head in enumerate(heads):
        stop = heads[n + 1].start() if n + 1 < len(heads) else len(page)
        j = page.find(per._NEW_PLOT, head.end(), stop)
        if j < 0:
            continue
        k = j + len(per._NEW_PLOT)
        _div, k = per._value(page, k, decoder)
        _data, data_end = per._value(page, per._comma(page, k), decoder)
        _layout, layout_end = per._value(page, per._comma(page, data_end), decoder)
        spans[head.group(1).strip()] = (data_end, layout_end)
    return key, page, spans


def _report_with_page_edit(tmp_path: Path, edit) -> Path:
    """A copy of the committed report whose event page for `_IRENE` is `edit(page, spans)`, re-encoded in place."""
    import base64

    text = _REPORT.read_text(encoding="utf-8")
    key, page, spans = _page_with_spans(text)
    old_b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    assert text.count(old_b64) == 1, "the page's base64 does not occur exactly once in the outer HTML"
    new_b64 = base64.b64encode(edit(page, spans).encode("utf-8")).decode("ascii")
    out = tmp_path / "mutated-report.html"
    out.write_text(text.replace(old_b64, new_b64), encoding="utf-8")
    return out


def _digest_of_target(report: Path) -> str:
    pages = per.decode_pages(report.read_text(encoding="utf-8"))
    key = next(k for k in pages if k.endswith(f"event_page__evt.{_IRENE}.html"))
    ((data, layout),) = [(d, lay) for title, d, lay in per.figures_of(pages[key]) if title == _TITLE]
    return _spec_digest(data, layout)


def _add_layout_key(page: str, spans: dict, title: str = _TITLE) -> str:
    end = spans[title][1]
    assert page[end - 1] == "}"
    return page[: end - 1] + ',"reviewer_probe":1' + page[end - 1 :]


def _add_trace(page: str, spans: dict) -> str:
    end = spans[_TITLE][0]
    assert page[end - 1] == "]"
    return page[: end - 1] + ',{"type":"scatter","x":[1],"y":[1]}' + page[end - 1 :]


def test_unedited_copy_reproduces_the_pin(tmp_path):
    """Control for the two arms below: the copy-and-reencode path with an identity edit yields the committed digest.

    Class: the harness itself. Without this, a harness that corrupts the page would make every mutation arm red for
    the wrong reason (reachability: B4 condition 2).
    """
    report = _report_with_page_edit(tmp_path, lambda page, spans: page)
    assert _digest_of_target(report) == _FIGURE_SPEC_SHA256


@pytest.mark.parametrize("edit", [_add_layout_key, _add_trace], ids=["layout-edit", "data-edit"])
def test_an_edit_to_the_target_figure_inside_the_report_moves_the_digest(tmp_path, edit):
    """Class: every edit to the target figure's layout or data made in the committed report before extraction.

    NEW CAPABILITY, row: Q1 form D, DoD 12 "fails when the extracted spec changes". Plausible wrong implementation:
    a digest taken of the page's other figure, of figure 0, or of a value cached outside the report; each leaves the
    digest at the pin under this edit. Reachability: the edit is made to the report text and goes through
    `decode_pages` and `figures_of`, the same path the committed report takes. Both-states anchor: the assertion turns
    on the figure's content, which is present in the committed and the edited report. Invocation isolation: the red is
    the digest, a property of the input report, and does not depend on which tree the run resolves from. Second correct
    implementation that passes: hashing `[data, layout]` through any other deterministic JSON form, with the pin
    recomputed.
    """
    report = _report_with_page_edit(tmp_path, edit)
    assert _digest_of_target(report) != _FIGURE_SPEC_SHA256


def test_an_edit_to_another_figure_of_the_page_leaves_the_digest_at_the_pin(tmp_path):
    """The other side: the pin is the one figure's digest, not the page's, so an edit to a neighbour does not move it.

    Class: every figure of the page other than the one titled `_TITLE`. Plausible wrong implementation: a hash of the
    whole page or of every figure of it, which would redden the Arial-independent test on an unrelated figure edit
    and leave a developer re-pinning without a re-read. Second correct implementation that passes: any per-figure
    extraction. Row: the 1p.1 bullet's "the committed figure (title ... on page ...)".
    """
    report = _report_with_page_edit(tmp_path, lambda page, spans: _add_layout_key(page, spans, _OTHER_TITLE))
    assert _digest_of_target(report) == _FIGURE_SPEC_SHA256
