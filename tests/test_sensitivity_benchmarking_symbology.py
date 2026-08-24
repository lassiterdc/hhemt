"""Symbology invariants for the sensitivity-benchmarking figure, asserted on the BUILT FIGURE.

Why this module exists, and why it asserts on `fig` rather than on the lookup tables it is
easy to assert on instead: both defects it pins were invisible to every other gate. Each
passed the 44-test math module, a 134-test renderer selection, `ruff`, and two agents'
source reads, and each was visible ONLY in a render.

- The `[SPLIT]`: the legend key `N ranks x 1 thread (MPI)` drew a CIRCLE in the legend while
  the GPU columns drew TRIANGLES under that same key. Cause: the legend name came from
  `_decomposition_label` (many-to-one over `group_value`) while the symbol came from
  `is_gpu_group` (keyed on `group_value`), so two co-labelled groups took different marks
  and plotly drew the swatch from whichever claimed the label first.
- The phantom ticks: `dtick=1` on a CPU sweep that ran `{1, 2, 4, 8}` drew labelled ticks at
  3, 5, 6 and 7 -- four resource levels the experiment never visited, which a reader takes
  for missing measurements.

BOTH assertions are anchored on properties that are computable and meaningful in the PRE-FIX
and POST-FIX worlds alike -- one-to-one-ness of a key's mark set, and a subset relation
between labelled tick positions and plotted positions. Neither mentions a symbol name or a
tickmode value introduced by the fix, so neither can go green pre-fix for the wrong reason.
Measured against `867a76e2^`: `test_every_legend_key_maps_to_one_marker_symbol` fails with
`['circle', 'triangle-up']`, and `test_labelled_ticks_are_a_subset_of_plotted_positions` fails
on the CPU column with extra ticks `[3.0, 5.0, 6.0, 7.0]`.

The public helpers below (`build_figure`, `legend_symbol_sets`, `panel_marker_xs`,
`panel_labelled_ticks`, `tick_violations`) take the renderer MODULE as a parameter so the
identical assertion code can be pointed at a historical revision of the renderer. That is what
makes the pre-fix failure a MEASUREMENT rather than a prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hhemt.config.report import SensitivityReportConfig, report_config
from hhemt.report_renderers import sensitivity_benchmarking as sb
from hhemt.report_renderers._provenance import ProvenanceLog

# The real compute-config matrix shape, transcribed from
# `hhemt.synthetic_experiment`'s config list, with `n_devices` derived exactly as
# `_ensure_n_devices_column` derives it (GPUs when present, else mpi x omp).
#
# TWO properties of this fixture are load-bearing and must survive any edit:
#
# 1. The CPU column's device counts are NON-CONTIGUOUS -- {1, 2, 4, 8}. A contiguous
#    sweep cannot exhibit the phantom-tick defect at all, because `dtick=1` and the
#    plotted set coincide. A fixture that quietly became {1, 2, 3, 4} would make
#    `test_labelled_ticks_are_a_subset_of_plotted_positions` pass in both worlds.
# 2. At least two DISTINCT `group_value`s share ONE decomposition label. Here `mpi`,
#    `gpu (a6000)` and `gpu (a100-80)` all label as `N ranks x 1 thread (MPI)`. A fixture
#    with a bijection between group_value and label cannot exhibit the `[SPLIT]` defect.
_MATRIX: tuple[tuple[str, int, int, int, str], ...] = (
    ("gpu", 1, 1, 1, "gpu-a6000"),
    ("gpu", 2, 1, 2, "gpu-a6000"),
    ("gpu", 3, 1, 3, "gpu-a6000"),
    ("gpu", 1, 1, 1, "gpu-a100-80"),
    ("gpu", 2, 1, 2, "gpu-a100-80"),
    ("gpu", 3, 1, 3, "gpu-a100-80"),
    ("serial", 1, 1, 0, "standard"),
    ("openmp", 1, 2, 0, "standard"),
    ("openmp", 1, 8, 0, "standard"),
    ("mpi", 2, 1, 0, "standard"),
    ("mpi", 4, 1, 0, "standard"),
    ("mpi", 8, 1, 0, "standard"),
    ("hybrid", 2, 2, 0, "standard"),
    ("hybrid", 4, 2, 0, "standard"),
)

_CPU_BASE_S = 400.0
_GPU_BASE_S = {"gpu-a6000": 95.0, "gpu-a100-80": 60.0}
_N_REPLICATES = 2


def _matrix_frame() -> pd.DataFrame:
    """The renderer's input frame. Wall-clock values are SYNTHETIC layout fixtures.

    Nothing in this module asserts on a wall-clock number; the invariants are structural
    (which marks a key carries, which positions an axis labels), so the values need only be
    positive and monotone enough to produce a drawable figure. The RNG is seeded so a
    failure is reproducible.
    """
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for run_mode, n_mpi, n_omp, n_gpus, partition in _MATRIX:
        is_gpu = n_gpus > 0
        n_dev = n_gpus if is_gpu else n_mpi * n_omp
        group_value = f"gpu ({partition.removeprefix('gpu-')})" if is_gpu else run_mode
        base = _GPU_BASE_S[partition] if is_gpu else _CPU_BASE_S
        wall = base / (n_dev**0.72) if n_dev > 1 else base
        config_id = f"{group_value}@{n_dev}"
        for replicate in range(_N_REPLICATES):
            w = wall * (1 + rng.normal(0, 0.03))
            rows.append(
                {
                    "sa_id": f"{config_id}#{replicate}",
                    "config_id": config_id,
                    "group_value": group_value,
                    "indep_value": n_dev,
                    "n_devices": n_dev,
                    "wallclock_s": w,
                    "wallclock_hr": w / 3600,
                    "compute_hr": w * n_dev / 3600,
                    "wallclock_disp": w / 60,
                    "compute_disp": w * n_dev / 60,
                    "n_replicates": _N_REPLICATES,
                    "n_mpi_procs": n_mpi,
                    "n_omp_threads": n_omp,
                    "n_gpus": n_gpus,
                    "n_nodes": 1,
                }
            )
    return pd.DataFrame(rows)


def build_figure(module=sb, tmp_path: Path | None = None):
    """Build the real four-row figure through `module`'s own builder.

    `module` is a parameter, not a hard import, so the pre-fix revision of the renderer can
    be loaded from git and driven through this identical code path.

    The per-family baseline wiring below MIRRORS `render()`'s own: resolve one anchor per
    hardware family via `_resolve_family_baselines`, then compute the metrics against that
    anchor. Calling `_compute_speedup_per_group` bare instead would use
    `baseline_mode='per_group'`, which drops every group lacking an N=1 row -- and in this
    matrix openmp, mpi and hybrid all lack one, so the scaling panels would come back empty
    and the tick assertion would silently have nothing to check on two of four rows.
    """
    tmp_path = tmp_path or Path("/tmp")
    df = _matrix_frame()
    sens_cfg = report_config().sensitivity or SensitivityReportConfig(independent_vars=["n_devices"])

    df_avg = df.groupby(["group_value", "n_devices", "config_id"], as_index=False).agg(
        wallclock_s=("wallclock_s", "mean"), sa_id=("sa_id", "first")
    )
    family_baselines = module._resolve_family_baselines(
        df, t_col="wallclock_s", indep_col="n_devices", group_col="group_value"
    )
    speedup, efficiency, speedup_all, efficiency_all = {}, {}, {}, {}
    for group_value, anchor in family_baselines.items():
        sub_avg = df_avg[df_avg["group_value"].astype(str) == group_value]
        sub_raw = df[df["group_value"].astype(str) == group_value]
        if sub_avg.empty:
            continue
        line_sub = sub_avg.loc[sub_avg.groupby("n_devices")["wallclock_s"].idxmin()]
        for target, kind, src in (
            (speedup, "speedup", line_sub),
            (efficiency, "efficiency", line_sub),
            (speedup_all, "speedup", sub_raw),
            (efficiency_all, "efficiency", sub_raw),
        ):
            if src.empty:
                continue
            target.update(
                module._compute_metric_all_rows_per_group(
                    src,
                    t_col="wallclock_s",
                    indep_col="n_devices",
                    group_col="group_value",
                    kind=kind,
                    anchor=anchor,
                )
            )

    fig, _cfg = module._build_sensitivity_benchmarking_figure(
        df,
        speedup,
        efficiency,
        wall_unit="min",
        cost_unit="min",
        independent_var="n_devices",
        group_by_var="run_mode",
        sens_cfg=sens_cfg,
        output_path=tmp_path / "benchmarking.html",
        source_paths=[],
        analysis_dir=tmp_path,
        plotly_js_mode="cdn",
        prov=ProvenanceLog(),
        gpu_legend_suffix="",
        speedup_all_rows=speedup_all,
        efficiency_all_rows=efficiency_all,
        model_arm="coupled",
    )
    return fig


# ── Invariant 1: one legend key, one mark ──────────────────────────────────


def legend_symbol_sets(fig) -> dict[str, set[str]]:
    """Per legend key, the SET of marker symbols drawn under it across every trace.

    A key whose set has more than one member is the divergence: plotly draws the legend
    swatch from the FIRST trace claiming a name and suppresses the rest, so the swatch then
    shows one of the marks and stands for all of them.
    """
    out: dict[str, set[str]] = {}
    for trace in fig.data:
        if "markers" not in (getattr(trace, "mode", "") or ""):
            continue
        symbol = getattr(getattr(trace, "marker", None), "symbol", None)
        if symbol is None:
            continue
        out.setdefault(str(trace.name), set()).add(str(symbol))
    return out


def test_every_legend_key_maps_to_one_marker_symbol():
    """No legend key may be drawn with two different marks.

    This is a property of the EMISSION SITES, not of the symbol lookup table -- asserting
    that `_DECOMPOSITION_SYMBOLS` has one value per key would be true by construction and
    would not have caught the shipped defect, which lived in how the sites CALLED the
    lookup.
    """
    sets = legend_symbol_sets(build_figure())
    assert sets, "no marker traces were emitted -- the fixture built an empty figure"
    split = {name: sorted(symbols) for name, symbols in sets.items() if len(symbols) > 1}
    assert not split, (
        "legend key(s) drawn with more than one marker symbol, so the legend swatch "
        f"misrepresents the series it stands for: {split}"
    )


# ── Invariant 2: an axis labels only positions that were run ───────────────


def panel_marker_xs(fig, row: int, col: int) -> set[float]:
    """The x positions MARKERS occupy in one panel.

    Marker-bearing traces only. The ideal-reference line is drawn from a hardcoded origin
    (`x = [1.0, x_max]`), so including line traces would admit x=1 on any column whose
    smallest run is 2 -- a phantom position arriving from the opposite direction to the one
    this invariant guards.
    """
    ref = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")
    vals: set[float] = set()
    for trace in fig.data:
        if (getattr(trace, "xaxis", None) or "x") != ref:
            continue
        if "markers" not in (getattr(trace, "mode", "") or ""):
            continue
        xs = getattr(trace, "x", None)
        if xs is None:
            continue
        for v in xs:
            try:
                vals.add(float(v))
            except (TypeError, ValueError):
                continue
    return vals


def panel_labelled_ticks(fig, row: int, col: int, plotted: set[float]) -> set[float] | None:
    """The x positions this panel's axis puts a tick at, or None if it does not declare any.

    Computed from whatever tick scheme the axis carries, so the answer exists in the pre-fix
    world (`tickmode='linear'` + `dtick`) and the post-fix world (`tickmode='array'` +
    `tickvals`) alike. Anchoring on the SCHEME NAME instead would make the assertion
    tautologically green before the fix.

    For the linear scheme the enumeration is windowed by the PLOTTED range rather than by the
    axis range, because the axis range is unresolved until render time. That window is a
    lower bound on what is drawn -- autorange always covers the data -- so every position
    this returns is genuinely ticked. Under-counting outside the data range is deliberate and
    is the conservative direction: it can only weaken the assertion, never fabricate a
    violation.

    Returns None when the axis leaves ticks to plotly's autotick, which is not a violation
    here but is not verifiable either; the caller counts those separately so a figure that
    became entirely auto-ticked cannot pass vacuously.
    """
    axis = fig.get_subplot(row, col).xaxis
    mode = getattr(axis, "tickmode", None)
    if mode == "array":
        tickvals = getattr(axis, "tickvals", None)
        if tickvals is None:
            return None
        return {float(v) for v in tickvals}
    if mode == "linear":
        try:
            step = float(getattr(axis, "dtick", None) or 1.0)
            start = float(getattr(axis, "tick0", None) or 0.0)
        except (TypeError, ValueError):
            return None
        if step <= 0 or not plotted:
            return None
        lo, hi = min(plotted), max(plotted)
        first = start + step * np.ceil((lo - start) / step)
        count = int(np.floor((hi - first) / step)) + 1
        return {float(first + step * k) for k in range(max(count, 0))}
    return None


def tick_violations(fig) -> tuple[list[str], int]:
    """Panels whose labelled ticks are NOT a subset of their plotted positions, and how many
    panels were actually checkable."""
    violations: list[str] = []
    checked = 0
    for row in (1, 2, 3, 4):
        for col in (1, 2, 3):
            plotted = panel_marker_xs(fig, row, col)
            if not plotted:
                continue
            ticks = panel_labelled_ticks(fig, row, col, plotted)
            if ticks is None:
                continue
            checked += 1
            extra = sorted(ticks - plotted)
            if extra:
                violations.append(
                    f"row {row} col {col}: ticks at {extra} where no run exists (plotted {sorted(plotted)})"
                )
    return violations, checked


def test_labelled_ticks_are_a_subset_of_plotted_positions():
    """Every tick an axis draws must sit on a device count the experiment actually ran.

    The subset relation is the invariant, not the tick MECHANISM. A tick between two runs is
    not a smaller reading on a continuum -- the x axis enumerates a designed matrix -- so a
    reader takes it for a measurement that is missing.
    """
    fig = build_figure()
    violations, checked = tick_violations(fig)
    assert checked > 0, (
        "no panel declared a tick scheme, so this assertion checked nothing -- a vacuous pass, not a passing figure"
    )
    assert not violations, "axis ticks at positions no run occupies:\n  " + "\n  ".join(violations)


def test_cpu_column_sweep_is_non_contiguous():
    """Guard on the fixture itself: the CPU column must skip device counts.

    Both assertions above are satisfiable by a degenerate fixture. If the CPU sweep ever
    became contiguous, `test_labelled_ticks_are_a_subset_of_plotted_positions` would pass
    against a `dtick=1` renderer and pin nothing at all. This test fails loudly at the
    fixture rather than letting the real assertion rot into a tautology.
    """
    plotted = panel_marker_xs(build_figure(), 4, 1)
    assert plotted, "CPU column carried no marker traces"
    span = set(range(int(min(plotted)), int(max(plotted)) + 1))
    assert span - plotted, (
        f"CPU column device counts {sorted(plotted)} are contiguous, so the phantom-tick "
        "invariant is unfalsifiable against this fixture"
    )


def test_co_labelled_group_values_exist_in_fixture():
    """Guard on the fixture itself: at least one legend key must stand for 2+ group values.

    Same rot risk as above, on the other invariant. With a bijection between `group_value`
    and decomposition label, `test_every_legend_key_maps_to_one_marker_symbol` cannot fail
    however the symbol is keyed.
    """
    labels: dict[str, set[str]] = {}
    for group_value in _matrix_frame()["group_value"].unique():
        gv = str(group_value)
        label = sb._decomposition_label(
            gv,
            is_gpu_group=gv.lower().startswith("gpu"),
            is_hybrid_group=gv.lower() == "hybrid",
        )
        labels.setdefault(label, set()).add(gv)
    shared = {label: sorted(gvs) for label, gvs in labels.items() if len(gvs) > 1}
    assert shared, (
        "no decomposition label is shared by two group_values, so the legend-key invariant "
        f"is unfalsifiable against this fixture: {labels}"
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
