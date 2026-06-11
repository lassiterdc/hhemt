"""ADR-2 (reporting-system_canonical-plot-id): the canonical plot ID is minted
once in ``report_plot_ids`` and IS both the figure-output file stem and the
``plot_id`` field in the ``*.manifest.json`` sidecar.

These tests pin the two correctness conditions that keep the single source of
truth honest:

1. Grammar (``canonical_plot_id``) -- ``__`` between segments, ``.`` within a
   segment, fixed renderer_kind/descriptor/sa/evt order, charset-safe (no ``-``).
2. Stem == manifest ``plot_id`` (OE-3) -- ``_emit_manifest_sidecar`` stamps the
   field as ``output_path.stem`` so ``harvest_source_paths``' stem-keying and the
   manifest field can never drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit.report_plot_ids import (
    canonical_plot_id,
    plot_output_template,
)
from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
    _emit_manifest_sidecar,
)


def test_canonical_plot_id_grammar() -> None:
    """`__` joins segments, `.` separates within a segment; fixed segment order."""
    assert canonical_plot_id("system_overview") == "system_overview"
    assert canonical_plot_id("peak_flood_depth", event_id="evt001") == "peak_flood_depth__evt.evt001"
    assert canonical_plot_id("conduit_flow", sa_id="3", event_id="evt001") == "conduit_flow__sa.3__evt.evt001"
    assert canonical_plot_id("benchmarking", descriptor="n_gpus.vs.total") == "benchmarking__n_gpus.vs.total"


def test_canonical_plot_id_is_charset_safe() -> None:
    """The grammar never introduces a `-` (absent from the C-CHARSET ^[A-Za-z0-9_.]+$)."""
    pid = canonical_plot_id("peak_flood_depth", sa_id="3", event_id="year.9_event_type.compound")
    assert "-" not in pid
    assert set(pid) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")


def test_plot_output_template_preserves_wildcard_braces() -> None:
    """The Snakefile template keeps literal `{wildcard}` braces and the
    `__OUTPUT_EXT__` token for `_emit_plot_rule` to substitute per backend."""
    tmpl = plot_output_template(
        renderer_kind="peak_flood_depth",
        subdir="plots/per_sim/{event_id}",
        event_id="{event_id}",
    )
    assert tmpl == "plots/per_sim/{event_id}/peak_flood_depth__evt.{event_id}__OUTPUT_EXT__"


@pytest.mark.parametrize(
    "stem",
    [
        "system_overview",  # singleton
        "peak_flood_depth__evt.year.9_event_type.compound",  # per-sim
        "conduit_flow__sa.3__evt.year.9_event_type.compound",  # per-sa
        "benchmarking__n_gpus.vs.total",  # benchmarking descriptor
    ],
)
def test_manifest_plot_id_equals_figure_stem(tmp_path: Path, stem: str) -> None:
    """OE-3: the stamped manifest `plot_id` equals the figure-output stem by
    construction, across every renderer-kind stem shape."""
    output_path = tmp_path / f"{stem}.png"
    _emit_manifest_sidecar(output_path, {"renderer_module": "x"})
    manifest = json.loads((tmp_path / f"{stem}.manifest.json").read_text())
    assert manifest["plot_id"] == stem
    assert manifest["plot_id"] == output_path.stem


def test_manifest_preserves_explicit_plot_id(tmp_path: Path) -> None:
    """`setdefault` semantics: an already-present plot_id is not overwritten by
    the stem (forward-compat for a future renderer that mints it explicitly)."""
    output_path = tmp_path / "peak_flood_depth__evt.evt001.png"
    _emit_manifest_sidecar(output_path, {"plot_id": "explicit_override"})
    manifest = json.loads((output_path.parent / "peak_flood_depth__evt.evt001.manifest.json").read_text())
    assert manifest["plot_id"] == "explicit_override"
