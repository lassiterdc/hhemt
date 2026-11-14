"""The cross-solver figure is inert on a single-solver combine and orders its series by date.

Two behaviours of work chunk 1b.4 live here. First, the combined Snakefile generator drops the
``cross_solver_comparison`` selection (its rule, its output and its sidebar category) unless the
combine's read-model reports two or more distinct solvers. Second, the renderer orders its
series by the commit date the producer supplied and labels them with the producer's label, and
never by parsing the solver identity.

Nothing here compiles a solver or runs a simulation. The generator tests build a bundle root
that holds only ``cross_solver_comparison.json`` (no children) and read the Snakefile text and
the expected-figure paths; the renderer tests call the module's own helpers on in-memory payloads.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import hhemt.bundle.combined_snakefile_generator as generator
from hhemt.report_renderers import cross_solver_comparison as csc
from hhemt.report_renderers._provenance import ProvenanceLog

_RULE = "plot_cross_solver_comparison"
_CATEGORY = "Cross-Solver Comparison"
#: A rule every combine emits whatever the solver count; the positive control that the
#: generator ran and that "cross-solver absent" is not "nothing emitted".
_CONTROL_RULE = "plot_cross_experiment_intercomparison"

# This campaign's five pins with their measured committer dates. Hash order and date order
# disagree at positions 2 and 3, which is the case the date ordering exists for.
_PINS_WITH_DATES = {
    "a38338b09e62e57c936f51516bbdbe495d89a546": "20260828",
    "e53c2fa0" + "0" * 32: "20260928",
    "b1ff2689" + "0" * 32: "20261002",
    "f9c28daa93f1a4e11bf3eae83086b7d804a9cef0": "20261004",
    "faba223212408f04b7a3dbf6580adedf4a4e6b13": "20261030",
}
_CONFIG = "run_mode=clean|n_mpi=1|n_omp=1|n_gpus=0"


def _bundle_with_read_model(tmp_path: Path, read_model_text: str | None) -> Path:
    """A childless bundle root, holding the read-model only when text is given."""
    root = tmp_path / "bundle"
    root.mkdir()
    if read_model_text is not None:
        (root / "cross_solver_comparison.json").write_text(read_model_text, encoding="utf-8")
    return root


def _generate(root: Path) -> tuple[str, tuple[Path, ...]]:
    """The combined Snakefile body and its expected-figure paths for ``root``."""
    return generator.generate_combined_snakefile(root)


@pytest.mark.parametrize("count", [0, 1])
def test_a_single_solver_combine_renders_no_cross_solver_rule_output_or_category(tmp_path: Path, count: int) -> None:
    """Property: with fewer than two distinct solvers the cross-solver selection is absent.

    Class: every read-model whose ``distinct_solver_count`` is below two. Differently
    positioned satisfying inputs are 0 (an empty combine) and 1 (the clean-vs-resume combine
    at one pin), which sit on either side of the "no solver recorded" boundary.

    Why required: DoD 1b says the renderer is inert, with no category and no figure, when the
    bundles carry fewer than two distinct TRITONSWMM_branch_key values; otherwise every
    clean-vs-resume combine renders a one-bar-per-panel figure that compares nothing.

    What kills it: a generator that harvests every selection unconditionally (today's
    behaviour before this chunk), or one that gates on the renderer name instead of the count.

    Second correct implementation that still passes: filtering the rule list after harvest
    instead of before it, which drops the same rule.
    """
    root = _bundle_with_read_model(tmp_path, json.dumps({"distinct_solver_count": count}))
    body, expected = _generate(root)

    assert _CONTROL_RULE in body, "positive control: the unconditional combined rules are still emitted"
    assert _RULE not in body
    assert _CATEGORY not in body
    assert not [p for p in expected if "cross_solver_comparison" in p.name]


@pytest.mark.parametrize("count", [2, 5])
def test_a_multi_solver_combine_emits_the_cross_solver_rule_output_and_category(tmp_path: Path, count: int) -> None:
    """Property: with two or more distinct solvers the cross-solver selection is present.

    Class: every read-model whose ``distinct_solver_count`` is at least two. Inputs 2 (the
    boundary) and 5 (this campaign's Part B) differ in position.

    Why required: the inert test above would also pass a generator that dropped the selection
    always; this is the arm that fails such a generator.

    What kills it: a gate that tests ``> 2``, or one that never admits the selection.

    Second correct implementation that still passes: evaluating the predicate once per
    template rather than once per selection, which keeps the same single template.
    """
    root = _bundle_with_read_model(tmp_path, json.dumps({"distinct_solver_count": count}))
    body, expected = _generate(root)

    assert _CONTROL_RULE in body
    assert _RULE in body
    assert _CATEGORY in body
    assert [p for p in expected if p.name == "cross_solver_comparison.html"]


@pytest.mark.parametrize(
    "read_model_text",
    [
        None,
        "{not json",
        json.dumps([2]),
        json.dumps({}),
        json.dumps({"distinct_solver_count": "2"}),
        json.dumps({"distinct_solver_count": True}),
    ],
    ids=["absent", "unparseable", "not-an-object", "no-count", "string-count", "bool-count"],
)
def test_a_read_model_with_no_usable_count_leaves_the_figure_inert(tmp_path: Path, read_model_text: str | None) -> None:
    """Property: a read-model that does not state an integer count never admits the figure.

    Class: every absent, unparseable or count-less read-model. Includes ``"2"`` and ``True``,
    which a lax ``>= 2`` comparison would admit (a string compares by type error, and
    ``True`` is the integer one).

    Why required: the figure is a claim about several solvers; a count the combine did not
    write cannot back it, and a bundle produced before the count existed has no cross-solver
    data to draw.

    What kills it: reading a missing count as two, or coercing a bool or string to an integer.

    Second correct implementation that still passes: treating a damaged read-model as an
    error, provided the absent case stays inert.
    """
    root = _bundle_with_read_model(tmp_path, read_model_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        body, _expected = _generate(root)

    assert _CONTROL_RULE in body
    assert _RULE not in body


def test_a_selection_gated_on_a_key_the_combined_path_cannot_evaluate_is_refused(tmp_path: Path, monkeypatch) -> None:
    """Property: an unevaluable predicate key is a loud error, never a silent keep or drop.

    Class: every ``predicate_key`` absent from the combined predicate table. The key used is
    the one the combined set really carries, with the table emptied.

    Why required: skipping an unknown key drops a figure silently, and keeping it ignores the
    gate; either leaves a registry entry that claims to be conditional and is not.

    What kills it: ``dict.get`` with a default, or a loop that ignores ``predicate_key``.

    Second correct implementation that still passes: raising ``ConfigurationError`` instead of
    ``KeyError``, with the key named in the message.
    """
    monkeypatch.setattr(generator, "_COMBINED_PREDICATES", {})
    root = _bundle_with_read_model(tmp_path, json.dumps({"distinct_solver_count": 2}))

    with pytest.raises(KeyError, match="has_multiple_solvers"):
        generator._active_combined_selections(root)


def _payload(solver_info: dict | None, pins: list[str]) -> dict:
    """One arm, one config, one measured cell per pin; ``solver_info`` omitted when None."""
    payload: dict = {
        "metric": "wallclock_seconds",
        "model_arms": ["triton"],
        "solvers": list(pins),
        "configs": [_CONFIG],
        "cells": [
            {"solver": pin, "model_arm": "triton", "config": _CONFIG, "value": float(i + 1), "status": "ok"}
            for i, pin in enumerate(pins)
        ],
    }
    if solver_info is not None:
        payload["solver_info"] = solver_info
    return payload


def _info(pins_with_dates: dict[str, str | None]) -> dict:
    """The producer's ``solver_info`` shape: date (or None) and the pin-derived label."""
    return {
        pin: {"date": date, "label": f"{pin[:8]}_{date}" if date else pin[:8]} for pin, date in pins_with_dates.items()
    }


def test_series_are_ordered_by_the_supplied_date_not_by_the_identity() -> None:
    """Property: dated solvers run oldest first by the producer's date, whatever the pins sort as.

    Class: every payload whose ``solver_info`` dates disagree with the pins' own order. This
    campaign's five pins are the instance (hash order differs from date order at positions 2
    and 3); the pins are full git refs with no date segment, as the producer emits them.

    Why required: the developer ruled "date is more meaningful so lets do date as the sort
    order", and the identity carries no date, so a renderer that sorts the identity yields hash
    order.

    What kills it: sorting the identity strings (hash order), or reading ``payload["solvers"]``
    position as the order (the producer happens to write it sorted, so a payload written in the
    wrong order is the discriminating input and is used here).

    Second correct implementation that still passes: ordering by ``(date, pin)`` from an
    already-built list, which gives the same sequence.
    """
    pins = list(_PINS_WITH_DATES)
    scrambled = sorted(pins)  # hash order, deliberately not the date order
    payload = _payload(_info(_PINS_WITH_DATES), scrambled)

    ordered = csc._ordered_solvers(csc._normalize_cells(payload), payload)

    by_date = sorted(pins, key=lambda pin: _PINS_WITH_DATES[pin])
    assert ordered == by_date
    assert ordered != sorted(pins), "control: hash order differs from date order on these pins"


def test_undated_solvers_follow_every_dated_one_ordered_by_identity() -> None:
    """Property: an undated or badly dated solver sits after all dated solvers, by identity.

    Class: every mix of dated and undated solvers; the undated members here are one with no
    date, one with a malformed date, and one absent from ``solver_info``. The undated pins sort
    BEFORE every dated pin by identity, so a renderer that merged the two groups by identity
    would put them first.

    Why required: a dated series' position is an age claim; an undated series must not sit
    among them, and a malformed date cannot back an age claim either.

    What kills it: placing undated solvers first (the empty-date-sorts-first behaviour of the
    parse this replaces), or accepting a malformed date as a date.

    Second correct implementation that still passes: giving undated solvers a sentinel date
    greater than every real one and sorting once.
    """
    dated = {"f9c28daa" + "0" * 32: "20261004", "faba2232" + "0" * 32: "20261030"}
    info = _info(dated)
    info["00000001" + "0" * 32] = {"date": None, "label": "00000001"}
    info["00000002" + "0" * 32] = {"date": "2026-10-04", "label": "00000002"}  # malformed: not YYYYMMDD
    absent = "00000003" + "0" * 32  # in the cells and the solver list, missing from solver_info
    pins = [*info, absent]
    payload = _payload(info, pins)

    ordered = csc._ordered_solvers(csc._normalize_cells(payload), payload)

    assert ordered == [*sorted(dated, key=dated.get), "00000001" + "0" * 32, "00000002" + "0" * 32, absent]


def test_a_read_model_without_solver_info_is_all_undated_and_still_draws_every_solver() -> None:
    """Property: a payload with no per-solver fields orders by identity and drops no solver.

    Class: every read-model with no ``solver_info`` (written before the producer supplied it,
    or by hand, like the reviewer's fixture).

    Why required: the held reviewer test's payload has no per-solver fields and must stay
    valid; a renderer that raised or dropped solvers there would break it.

    What kills it: indexing ``solver_info`` unguarded, or deriving the solver list only from it.

    Second correct implementation that still passes: treating a missing ``solver_info`` as an
    empty mapping at the call site rather than inside ``_solver_info``.
    """
    pins = ["b1ff2689_x", "a38338b0_y"]
    payload = _payload(None, pins)

    fig, facts = csc._build_figure(csc._normalize_cells(payload), payload, ProvenanceLog())

    assert facts["solvers"] == sorted(pins)
    assert facts["undated_solvers"] == sorted(pins)
    assert [t.name for t in fig.data if t.type == "bar"] == sorted(pins)


def test_the_legend_shows_the_producer_label_while_the_series_stay_keyed_on_the_identity() -> None:
    """Property: legend text is the supplied label; legend groups and series stay per identity.

    Class: every payload whose labels differ from the identities. A colliding pair (two pins
    sharing an eight-character prefix and no date) is the second input: it must not draw two
    series under one name.

    Why required: the label is display only. Keying series on it would merge two builds that
    share a label, which is the defect the identity-keyed cells exist to prevent.

    What kills it: naming traces by the identity (legend shows 40-character refs), or keying
    ``legendgroup`` on the label (the colliding pair merges).

    Second correct implementation that still passes: composing the label in the renderer from
    ``date`` and the pin, provided the colliding pair still falls back to the identities.
    """
    solo = "faba223212408f04b7a3dbf6580adedf4a4e6b13"
    twin_a, twin_b = "12345678" + "a" * 32, "12345678" + "b" * 32
    info = _info({solo: "20261030", twin_a: None, twin_b: None})
    payload = _payload(info, [solo, twin_a, twin_b])

    fig, _facts = csc._build_figure(csc._normalize_cells(payload), payload, ProvenanceLog())

    bars = [t for t in fig.data if t.type == "bar"]
    assert [t.name for t in bars] == ["faba2232_20261030", twin_a, twin_b]
    assert [t.legendgroup for t in bars] == [solo, twin_a, twin_b]
    assert len({t.name for t in bars}) == len(bars), "no two series share a legend name"
