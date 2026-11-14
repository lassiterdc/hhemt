"""Reviewer test for work package 1b.1: Decision 3's `not_run` guarantee.

Scope, stated so the next author does not over-read this module: it pins ONE
property of the approved shape's Decision 3 — that a ``(solver, config)`` cell the
solver did not run is never drawn as a measured zero and never removes its config
from the figure. It deliberately asserts NOTHING about which variable sits on the x
axis, because Decision 3's own text is inconsistent on that point (see the round
file for work package 1b.1, finding F1); a test that pinned either reading would
fail a correct implementation of the other.
"""

from __future__ import annotations

from hhemt.report_renderers import cross_solver_comparison as csc
from hhemt.report_renderers._provenance import ProvenanceLog

_SOLVER_A = "a38338b0_20250101"
_SOLVER_B = "faba2232_20240601"
_CONFIG_CPU = "run_mode=clean|n_mpi=1|n_omp=1|n_gpus=0"
_CONFIG_GPU = "run_mode=clean|n_mpi=1|n_omp=1|n_gpus=1"


def _payload() -> dict:
    """One arm, two solvers, two configs, with solver B's GPU cell explicitly not_run."""
    return {
        "metric": "walltime_s",
        "model_arms": ["triton"],
        "solvers": [_SOLVER_A, _SOLVER_B],
        "configs": [_CONFIG_CPU, _CONFIG_GPU],
        "cells": [
            {"solver": _SOLVER_A, "model_arm": "triton", "config": _CONFIG_CPU, "value": 10.0, "status": "ok"},
            {"solver": _SOLVER_A, "model_arm": "triton", "config": _CONFIG_GPU, "value": 4.0, "status": "ok"},
            {"solver": _SOLVER_B, "model_arm": "triton", "config": _CONFIG_CPU, "value": 12.0, "status": "ok"},
            {"solver": _SOLVER_B, "model_arm": "triton", "config": _CONFIG_GPU, "status": "not_run"},
        ],
    }


def test_not_run_cell_is_never_drawn_as_zero_and_never_drops_its_config() -> None:
    """Decision 3: a not_run cell renders absent, never as zero and never as a dropped row.

    Input class: any read-model whose cells carry one explicit ``not_run`` cell for a
    ``(solver, config)`` pair a peer solver DID run. The assertion quantifies over every
    such read-model, not over this fixture.

    Why this property is required, checkable against the code: the figure's stated
    purpose is to make "did not run" distinguishable from "ran and produced nothing".
    A synthesized ``0.0`` draws a bar whose height is a measurement the solver never
    made, and a dropped config removes the peer's bar too, so in both cases a reader
    comparing two solvers reads a difference the data does not contain.

    What kills it: making ``_normalize_cells`` hand back ``value=0.0`` for not_run
    cells (a zero appears and the drawn count rises), or making ``_ordered_configs``
    drop a config any solver failed to measure (the peer's measured bar disappears and
    the drawn count falls).
    """
    payload = _payload()
    cells = csc._normalize_cells(payload)
    fig, _facts = csc._build_figure(cells, payload, ProvenanceLog())

    bars = [trace for trace in fig.data if trace.type == "bar"]
    assert bars, "positive control: cells that DID run must reach the figure as bars"
    drawn = [float(y) for trace in bars for y in (trace.y or []) if y is not None]
    measured = [10.0, 4.0, 12.0]

    assert 12.0 in drawn, "positive control: the peer solver's measured value must be drawn"
    assert sorted(drawn) == sorted(measured), (
        "every measured cell must be drawn exactly once and no cell may be invented: "
        f"drawn={sorted(drawn)} measured={sorted(measured)}"
    )
    assert all(value != 0.0 for value in drawn), (
        f"a not_run cell must never be drawn as a measured zero: drawn={sorted(drawn)}"
    )
