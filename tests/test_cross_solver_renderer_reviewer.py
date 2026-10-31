"""Reviewer tests for chunk 1b.4: producer-to-figure contract, selection invariance, and the not_run note.

The coder's own cases drive the renderer with HAND-WRITTEN read-models and the gate with a
childless bundle. These tests take the differently-positioned inputs of each class: the REAL
producer's output feeding the figure, the whole selection set under the gate, and the one
sentence the figure uses to keep ``not_run`` from reading as a measured zero. Every test is
green on the tree as built and must remain green; each docstring names the production change
that reds it and a second correct implementation that stays green.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hhemt.bundle.combined_snakefile_generator as generator
from hhemt.bundle import _combine as C
from hhemt.report_renderers import cross_solver_comparison as R
from hhemt.report_renderers._provenance import ProvenanceLog
from hhemt.report_renderers._reporting_sets import get_reporting_set

_CPU = {
    "run_mode": "clean",
    "n_mpi_procs": 1,
    "n_omp_threads": 1,
    "n_gpus": 0,
    "n_nodes": 1,
    "hpc.partition": "standard",
}
_CPU_B = {**_CPU, "n_mpi_procs": 4}
# Hash order a38338b0, b1ff2689, e53c2fa0; date order a38338b0, e53c2fa0, b1ff2689.
_PIN_A = "a38338b09e62e57c936f51516bbdbe495d89a546"
_PIN_B = "b1ff2689" + "0" * 32
_PIN_E = "e53c2fa0" + "0" * 32


def _bundle(root: Path, pin: str, arm_group: str, *, date: str | None, member_attrs: list[dict]) -> Path:
    """One single-arm bundle with one member per config; the root carries the commit date when given."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {pin}\n")
    attrs = {"triton_producing_sha": pin}
    if date is not None:
        attrs["triton_producing_commit_date"] = date
    groups = {"/": xr.Dataset(attrs=attrs)}
    for i, ma in enumerate(member_attrs):
        groups[f"/member_{i}"] = xr.Dataset(attrs=ma)
        groups[f"/member_{i}/{arm_group}"] = xr.Dataset()
        groups[f"/member_{i}/{arm_group}/performance"] = xr.Dataset({"Total": ("Rank", np.asarray([10.0 + i]))})
    xr.DataTree.from_dict(groups).to_zarr(root / "sensitivity_datatree.zarr", mode="w", consolidated=False)
    return root


def _produced_payload(tmp_path: Path, specs: list[tuple[str, str | None, str | None]]) -> dict:
    """The read-model the REAL producer writes for solvers given as ``(pin, coupled date, triton date)``."""
    roots = []
    for i, (pin, d1, d2) in enumerate(specs):
        roots.append(_bundle(tmp_path / f"s{i}_c", pin, "tritonswmm", date=d1, member_attrs=[_CPU, _CPU_B]))
        roots.append(_bundle(tmp_path / f"s{i}_t", pin, "triton_only", date=d2, member_attrs=[_CPU, _CPU_B]))
    out = tmp_path / "combined"
    out.mkdir()
    C._write_cross_solver_comparison(out, roots)
    return json.loads((out / R.READ_MODEL_FILENAME).read_text())


def _figure(payload: dict):
    return R._build_figure(R._normalize_cells(payload), payload, ProvenanceLog())


def _bars_by_panel(fig) -> dict[str, list]:
    panels: dict[str, list] = {}
    for trace in fig.data:
        if trace.type == "bar":
            panels.setdefault(trace.xaxis or "x", []).append(trace)
    return panels


@pytest.mark.parametrize("undated_position", ["none", "first_arm_only", "oldest_solver_wholly"])
def test_the_figure_orders_and_labels_what_the_real_producer_writes(tmp_path, undated_position):
    """Property: fed the real producer's read-model, dated solvers come oldest first, labelled by the producer.

    Class: producer outputs over capture states: all dated, one arm of one solver undated, and a
    solver wholly undated. Hash order and date order DISAGREE on the three pins. Shape row:
    Settlement 4 (ordering, labels and degradation) read against the landed producer, and DoD 1b
    ("the figure orders dated solvers ascending by the supplied date and then undated solvers by
    identity, never by parsing a label"). Why required: the coder's cases write ``solver_info``
    by hand, so a renderer and a producer that disagree on a key name or a label form would pass
    every one of them and still order by hash in production. Killed by: ordering on the cells'
    ``solver`` text, or reading a field name the producer does not write. Second correct
    implementation: ordering from ``payload["solvers"]`` position, since the producer writes it
    already ordered; the assertion compares against the producer's own ``solver_info`` dates.
    Class: REGRESSION guard (a held test).
    """
    first_a = None if undated_position == "oldest_solver_wholly" else "20260828"
    second_a = None if undated_position in ("first_arm_only", "oldest_solver_wholly") else "20260828"
    payload = _produced_payload(
        tmp_path, [(_PIN_A, first_a, second_a), (_PIN_B, "20261002", "20261002"), (_PIN_E, "20260928", "20260928")]
    )
    _fig, facts = _figure(payload)
    expected_dated = sorted(
        (pin for pin, v in payload["solver_info"].items() if v["date"]),
        key=lambda p: (payload["solver_info"][p]["date"], p),
    )
    expected_undated = sorted(pin for pin, v in payload["solver_info"].items() if not v["date"])
    assert facts["solvers"] == expected_dated + expected_undated
    assert facts["undated_solvers"] == expected_undated
    if undated_position == "oldest_solver_wholly":
        assert facts["solvers"] == [_PIN_E, _PIN_B, _PIN_A]  # undated last even though it is the oldest by hash
    else:
        assert facts["solvers"] == [_PIN_A, _PIN_E, _PIN_B]
    fig, _ = _figure(payload)
    shown = {t.legendgroup: t.name for t in fig.data if t.type == "bar"}
    assert shown == {pin: payload["solver_info"][pin]["label"] for pin in payload["solver_info"]}


def test_every_panel_draws_the_solvers_in_one_shared_order(tmp_path):
    """Property: with two model arms, each panel lists the same solvers in the same order.

    Class: figures with more than one panel. Shape row: Settlement 4 ("dated solvers ascending by
    date, then undated by identity") applied to a figure, which has one legend and one colour per
    solver. Why required: an order computed per panel would give a solver two colours or two legend
    positions, and a reader comparing panels would compare different solvers. Killed by: sorting
    inside the panel loop on that panel's own cells. Second correct implementation: computing the
    order once and passing it down, which is what the code does today and also stays green.
    Class: REGRESSION guard (a held test).
    """
    payload = _produced_payload(
        tmp_path, [(_PIN_A, "20260828", "20260828"), (_PIN_B, "20261002", "20261002"), (_PIN_E, "20260928", "20260928")]
    )
    fig, _ = _figure(payload)
    panels = _bars_by_panel(fig)
    assert len(panels) == 2
    orders = [[t.legendgroup for t in bars] for bars in panels.values()]
    assert orders[0] == orders[1] == [_PIN_A, _PIN_E, _PIN_B]
    colours = [{t.legendgroup: t.marker.color for t in bars} for bars in panels.values()]
    assert colours[0] == colours[1]


@pytest.mark.parametrize("dates", [("20261004", "20261004"), ("20261004", None)], ids=["same-date", "one-undated"])
def test_equal_or_missing_dates_fall_to_the_identity_not_to_input_order(dates):
    """Property: solvers with the same date (or no date) order by identity whatever order they arrive in.

    Class: ties on the date key. Shape row: Settlement 4 ("dated solvers ascending by date, then
    undated by identity") and Settlement 2 ("ties on date fall to the pin"). Why required: a
    stable sort on the date alone keeps INPUT order for ties, so two renders of one bundle could
    differ. Killed by: sorting on the date only. Second correct implementation: sorting on
    ``(date or sentinel, identity)`` in one pass. Both input orders are run, which is what makes the
    tie observable.
    Class: NEW CAPABILITY (the plausible wrong implementation is a date-only stable sort).
    """
    low, high = "11111111" + "0" * 32, "99999999" + "0" * 32
    info = {
        low: {"date": dates[0], "label": low[:8]},
        high: {"date": dates[1], "label": high[:8]},
    }
    results = []
    for order in ([low, high], [high, low]):
        payload = {
            "metric": "wallclock_seconds",
            "model_arms": ["triton"],
            "solvers": order,
            "configs": ["run_mode=clean|n_mpi=1|n_omp=1|n_gpus=0"],
            "solver_info": info,
            "cells": [
                {
                    "solver": p,
                    "model_arm": "triton",
                    "config": "run_mode=clean|n_mpi=1|n_omp=1|n_gpus=0",
                    "value": 1.0,
                    "status": "ok",
                }
                for p in order
            ],
        }
        results.append(R._ordered_solvers(R._normalize_cells(payload), payload))
    assert results[0] == results[1] == [low, high]


def test_the_gate_removes_exactly_the_conditional_selection_and_nothing_else(tmp_path):
    """Property: below two solvers the combined set loses the cross-solver selection only; from two up it loses none.

    Class: every selection in the registered ``combined`` set, not just the cross-solver one.
    Shape row: the ``has_multiple_solvers`` consumer bullet and DoD 1b ("on False neither the
    generator's report rules nor the direct-render loop puts the figure or its manifest on disk").
    Why required: a gate that also drops a neighbouring selection would hide a different figure from
    every single-solver combine, which is the common case, and the coder's cases look only for the
    cross-solver names. Killed by: gating on the whole set, or filtering by position. Second correct
    implementation: building the active tuple from the registry minus the selections whose predicate
    is false. The expected tuples come from the registry itself, not from a literal.
    Class: NEW CAPABILITY (the plausible wrong implementation drops a sibling).
    """
    registry = get_reporting_set("combined").renderer_selection
    conditional = [s for s in registry if s.predicate_key == "has_multiple_solvers"]
    assert len(conditional) == 1  # exactly one conditional selection is registered

    def active(count):
        root = tmp_path / f"b{count}"
        root.mkdir(exist_ok=True)
        (root / "cross_solver_comparison.json").write_text(json.dumps({"distinct_solver_count": count}))
        return generator._active_combined_selections(root)

    assert active(1) == tuple(s for s in registry if s is not conditional[0])
    assert active(2) == tuple(registry)
    assert active(0) == active(1)


def test_a_not_run_figure_keeps_its_sentence_and_the_same_plot_area():
    """Property: a not_run figure keeps the sentence saying it is not a measured zero, and its plot area.

    Class: figures with and without a not_run cell. Shape row: DoD 1b ("an explicit ``not_run``
    sentinel") and 1b.1's reviewed degradation decision. Why required: the marker is the only mark
    distinguishing "did not run" from a short bar, and the sentence is what teaches a reader that;
    any re-placement of the note (for instance onto the shared caption helper) must keep the words and must
    not squeeze the bars. Killed by: dropping or reducing the note, or shrinking the plot area to make
    room for it. Second correct implementation: placing the sentence in a wrapped caption block that
    grows the figure downward, which keeps the 368 px plot area and also stays green. The assertion
    turns on the words and the plot-area height, both present before and after such a change.
    Class: REGRESSION guard (a held test; it exists to catch a weakening of the figure).
    """
    ok = {
        "solver": "a" * 8 + "0" * 32,
        "model_arm": "triton",
        "config": "run_mode=clean|n_mpi=1|n_omp=1|n_gpus=0",
        "value": 3.0,
        "status": "ok",
    }
    cfg2 = "run_mode=clean|n_mpi=4|n_omp=1|n_gpus=0"
    other = "b" * 8 + "0" * 32
    cells = [
        ok,
        {**ok, "config": cfg2, "value": 2.0},
        {**ok, "solver": other, "value": 4.0},
        {**ok, "solver": other, "config": cfg2, "status": "not_run", "value": None},
    ]
    payload = {
        "metric": "wallclock_seconds",
        "model_arms": ["triton"],
        "solvers": [ok["solver"], other],
        "configs": [ok["config"], cfg2],
        "cells": cells,
    }
    fig, facts = _figure(payload)
    assert facts["not_run_cell_count"] == 1
    notes = " ".join(re.sub(r"<br>", " ", a.text or "") for a in fig.layout.annotations)
    assert re.search(r"x marks a not_run cell", notes)
    assert re.search(r"NOT a measured zero", notes)
    assert fig.layout.height - fig.layout.margin.t - fig.layout.margin.b == 368

    clean_cells = [c if c["status"] == "ok" else {**c, "status": "ok", "value": 5.0} for c in cells]
    clean = {**payload, "cells": clean_cells}
    fig2, facts2 = _figure(clean)
    assert facts2["not_run_cell_count"] == 0
    assert "not_run" not in " ".join(a.text or "" for a in fig2.layout.annotations)
    assert fig2.layout.height - fig2.layout.margin.t - fig2.layout.margin.b == 368
