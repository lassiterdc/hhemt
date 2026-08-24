"""Fungibility guarantee (iter-2): a same-named TRITON and TRITON-SWMM figure is always a
valid comparison -- same renderer, same declared data sources modulo SWMM-specific artifacts.

Completes the iter-2 skeleton. Grounded in the ``emit_plot_with_sources`` source-declaration
contract + the ``*.manifest.json`` sidecar (``source_paths_relative`` is the harvest single
source of truth, ``_figure_emission.py``) + the Gotcha-53 renderer-IO audit. Compile-tier
(``@slow``): consumes the two rendered single-model sensitivity masters, so it inherits the
``tritonswmm_cpu_compiled`` skip/hard-fail gate (skips on a bare CI runner; hard-fails under
``HHEMT_REQUIRE_COMPILE_TIER=1``)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# SWMM-specific source markers a pure-TRITON figure legitimately drops (and NOTHING else).
_SWMM_MARKERS = ("swmm", "hydraulics.inp", "hydro.inp", "swmm_link", "max_flow_cms", ".rpt", ".out")

# Per-model status-flag sidecars, whose FILENAME embeds the model type by design:
# `{c_run|d_process}_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag[.json]`
# (the builders in `src/hhemt/constants.py`; that grammar is called a "persistent
# contract" there). `workflow_performance` declares every `_status/*.flag.json` it
# read_text()s, as the Gotcha-53 declared-superset-of-actual invariant REQUIRES --
# so the coupled master declares `*_tritonswmm_*` and the pure-TRITON master
# declares `*_triton_*`, and the two sets are DISJOINT rather than nested.
#
# That is NOT a fungibility violation and NOT a renderer defect: it is provenance
# faithfully naming the model-specific artifact that backs the same-named page.
# Anonymizing it renderer-side would falsify the provenance record. So the model
# axis is normalized HERE, in the comparison key -- exactly as `_normalize_source`
# already normalizes the build-location and analysis-identity axes, which differ
# across the two separately-built masters for reasons that are likewise not the
# fungibility under test.
#
# Anchored to the flag grammar (prefix + model token) so it CANNOT touch a genuine
# SWMM data source: `swmm_hydro.inp` / `swmm_hydraulics.inp` carry no `c_run_` /
# `d_process_` prefix and pass through untouched -- which is what keeps
# `system_overview` available as the non-vacuity SWMM-drop witness below.
#
# On alternation order, corrected after measuring rather than asserting: the
# longest-first order below is DEFENSIVE, not load-bearing for the current token
# set. The trailing `_` is what disambiguates -- matching "triton" inside
# "tritonswmm" leaves `s`, not `_`, so the branch fails and Python backtracks to
# "tritonswmm" regardless of order. Both orders were measured equal here. Order
# WOULD decide in two cases: if the trailing `_` were ever dropped (shortest-first
# then yields `c_run_{MODEL}swmm_...`, verified), or if a future model token were
# an UNDERSCORE-separated extension of another (`triton` vs `triton_gpu`, where
# shortest-first yields `c_run_{MODEL}_gpu_...`, also verified). Longest-first is
# kept because it is correct under all three shapes and costs nothing.
#
# A change to the flag grammar makes this a no-op and the test goes RED with the
# disjoint sets named -- it fails loud, never silently.
_MODEL_KEYED_STATUS_FLAG = re.compile(r"^(c_run|d_process)_(tritonswmm|triton|swmm)_")


def _normalize_source(src: str, analysis_name: str) -> str:
    """Collapse a declared source to a build-location- and analysis-identity-independent key.

    Three axes are collapsed: build location, analysis identity, and MODEL TYPE (see
    ``_MODEL_KEYED_STATUS_FLAG``). ``source_paths_relative`` are relative to each master's own
    ``analysis_dir``. A source OUTSIDE
    that dir (e.g. the ``_sensitivity_configs/{analysis_name}.csv`` compute-config manifest) carries
    a ``../`` prefix whose depth reflects where the master was BUILT and embeds the master's own
    ``analysis_name`` in its basename -- both differ across two SEPARATELY-built, differently-named
    masters (``synth_sensitivity`` vs ``synth_sensitivity_triton_only``) for reasons that are NOT the
    visual/data fungibility under test. Reduce to the basename (drops the build-location ``../``) with
    the per-master ``analysis_name`` -> a fixed placeholder (drops analysis identity), so a genuine
    cross-arm DATA-source divergence still differs by basename and is still caught, while the
    per-analysis provenance CSV and the in-dir data zarr both normalize equal across arms."""
    key = Path(src).name.replace(analysis_name, "{ANALYSIS}")
    return _MODEL_KEYED_STATUS_FLAG.sub(r"\1_{MODEL}_", key)


def _sources_by_plot(plots_dir: Path, analysis_name: str) -> dict[str, set[str]]:
    """{plot_id -> set(normalized source keys)} over every ``*.manifest.json`` under ``plots/``.

    The manifest ``plot_id`` field (defaulted to the output stem by ``_emit_manifest_sidecar``)
    is the harvest key; ``source_paths_relative`` are declared relative to each master's own
    ``analysis_dir`` and normalized via ``_normalize_source`` so the cross-arm comparison is
    apples-to-apples across two separately-built, differently-named masters."""
    out: dict[str, set[str]] = {}
    for man in plots_dir.rglob("*.manifest.json"):
        payload = json.loads(man.read_text())
        plot_id = str(payload.get("plot_id") or man.stem.removesuffix(".manifest"))
        out.setdefault(plot_id, set()).update(
            _normalize_source(s, analysis_name) for s in (payload.get("source_paths_relative", []) or [])
        )
    return out


@pytest.mark.slow
def test_same_named_figures_are_fungible(
    rendered_synth_sensitivity,
    rendered_synth_sensitivity_triton_only,
):
    """For every registered same-named renderer present in BOTH masters:
    (a) both produce a figure (a manifest sidecar exists), and
    (b) the pure-TRITON declared sources are a SUBSET of the coupled figure's sources,
        differing ONLY by SWMM-specific artifacts (adapts by DROPPING only SWMM), and
    (c) the pure-TRITON figure introduces NO source the coupled figure lacks.
    Non-vacuity: at least one shared figure must actually drop a SWMM source, else the
    subset relation holds trivially and proves nothing."""
    tri_dir = Path(rendered_synth_sensitivity_triton_only.analysis_paths.analysis_dir)
    cpl_dir = Path(rendered_synth_sensitivity.analysis_paths.analysis_dir)
    tri = _sources_by_plot(tri_dir / "plots", tri_dir.name)
    cpl = _sources_by_plot(cpl_dir / "plots", cpl_dir.name)

    shared = sorted(set(tri) & set(cpl))
    assert shared, "no same-named figures across the two model masters -- the fixtures rendered nothing comparable"

    swmm_drop_witnesses: list[str] = []
    for plot_id in shared:
        coupled_only = cpl[plot_id] - tri[plot_id]
        non_swmm_extra = {s for s in coupled_only if not any(m in s.lower() for m in _SWMM_MARKERS)}
        assert not non_swmm_extra, (
            f"{plot_id}: coupled figure declares non-SWMM sources absent from the pure-TRITON "
            f"figure -> the two are NOT fungible: {sorted(non_swmm_extra)}"
        )
        triton_only = tri[plot_id] - cpl[plot_id]
        assert not triton_only, (
            f"{plot_id}: pure-TRITON figure declares sources the coupled figure lacks "
            f"(a same-named figure must adapt by DROPPING only): {sorted(triton_only)}"
        )
        if coupled_only:  # the coupled arm dropped >=1 SWMM-specific source in pure-TRITON
            swmm_drop_witnesses.append(plot_id)

    assert swmm_drop_witnesses, (
        "no shared figure dropped a SWMM-specific source in the pure-TRITON arm -- the fungibility "
        "assertion is VACUOUS. A SWMM-source figure (e.g. system_overview, which declares "
        "hydro.inp/hydraulics.inp in the coupled arm and drops them under the pure-TRITON model "
        "gate) MUST be in the rendered set. Do not delete this guard -- fix the renderer/report "
        "state it exposes."
    )
