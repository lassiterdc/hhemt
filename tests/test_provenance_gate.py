"""Behavioral tests for the ADR-6 Gate-A provenance render-time gate.

Gate-A lives in `_figure_emission.emit_plot_with_sources`: an empty
`source_paths` raises `ProcessingError` unless the caller passes
`allow_empty_sources=True`. The check runs BEFORE the matplotlib/HTML
branch dispatch, so BOTH branches are gated. Complements the static
Gate-B AST-lint in `test_provenance_discipline.py`.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from TRITON_SWMM_toolkit.exceptions import ProcessingError
from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
    emit_plot_with_sources,
)


def _fig() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_empty_sources_raises_matplotlib_branch(tmp_path) -> None:
    out = tmp_path / "fig.png"
    with pytest.raises(ProcessingError):
        emit_plot_with_sources(_fig(), out, [], analysis_dir=tmp_path)


def test_empty_sources_raises_html_branch(tmp_path) -> None:
    out = tmp_path / "fig.html"
    with pytest.raises(ProcessingError):
        emit_plot_with_sources("<div>x</div>", out, [], analysis_dir=tmp_path, output_format="html")


def test_allow_empty_sources_suppresses_matplotlib(tmp_path) -> None:
    out = tmp_path / "fig.png"
    returned = emit_plot_with_sources(_fig(), out, [], analysis_dir=tmp_path, allow_empty_sources=True)
    assert returned == out
    assert out.exists()


def test_allow_empty_sources_suppresses_html(tmp_path) -> None:
    out = tmp_path / "fig.html"
    returned = emit_plot_with_sources(
        "<div>x</div>",
        out,
        [],
        analysis_dir=tmp_path,
        output_format="html",
        allow_empty_sources=True,
    )
    assert returned == out
    assert out.read_text(encoding="utf-8") == "<div>x</div>"


def test_nonempty_sources_pass(tmp_path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp_path / "fig.png"
    returned = emit_plot_with_sources(_fig(), out, [src], analysis_dir=tmp_path)
    assert returned == out


def test_generator_source_paths_not_exhausted_by_gate(tmp_path) -> None:
    """R5: a one-shot generator must be materialized once and survive to the
    branch (not mis-evaluated as falsy, not exhausted before consumption)."""
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp_path / "fig.png"
    gen = (p for p in [src])  # one-shot generator
    returned = emit_plot_with_sources(_fig(), out, gen, analysis_dir=tmp_path)
    assert returned == out
    manifest = out.parent / f"{out.stem}.manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_paths_relative"] == ["src.csv"]
