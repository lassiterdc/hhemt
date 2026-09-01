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
# `_ensure_n_devices_column` derives it (GPUs when present, else mpi x omp -- n_nodes
# is deliberately NOT a factor; `n_mpi_procs` is TOTAL ranks per simulation).
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

#: Configs the fixture runs exactly ONCE. Load-bearing, and it is the third fixture
#: property this module guards rather than assumes.
#:
#: Marker fill USED to encode replicate count -- hollow meant "run more than once". With
#: every config replicated, a pre-ruling renderer draws every marker hollow too, so an
#: "every marker is hollow" assertion would pass in BOTH worlds and pin nothing. One
#: single-run config makes the pre-ruling renderer draw exactly one SOLID marker, which is
#: what gives the assertion something to catch. `serial` is chosen because it is the lone
#: baseline point: changing its replicate count moves no device count, no symbol and no
#: colour, so the other invariants in this module are undisturbed.
_SINGLE_RUN_RUN_MODES = frozenset({"serial"})


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
        n_reps = 1 if run_mode in _SINGLE_RUN_RUN_MODES else _N_REPLICATES
        for replicate in range(n_reps):
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
                    "n_replicates": n_reps,
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
        # MIRRORS render()'s aggregation, and that mirroring is why this suite could not
        # detect the rename that broke it: the same edit landed on both sides, so the test
        # stayed consistent with the code it mirrors while both diverged from the consumer.
        # The literal is deliberate here where production uses `_MEMBER_KEY`: `module` is a
        # parameter so a PRE-fix revision can be driven through this same path, and that
        # revision has no such constant. What must match is the FRAME, not the source.
        wallclock_s=("wallclock_s", "mean"),
        sa_id=("sa_id", "first"),
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


# ── Invariant 1b: colour and symbol are locked, and colour is the line colour ──


def legend_style_pairs(fig) -> dict[str, set[tuple[str, str]]]:
    """Per legend key, the SET of (marker symbol, marker outline colour) pairs drawn.

    The outline is the marker's colour identity: the FILL is the replicate channel
    (hollow == this config ran more than once), so a filled and a hollow marker of the
    same series legitimately differ in `marker.color` while sharing `marker.line.color`.
    """
    out: dict[str, set[tuple[str, str]]] = {}
    for trace in fig.data:
        if "markers" not in (getattr(trace, "mode", "") or ""):
            continue
        marker = getattr(trace, "marker", None)
        symbol = getattr(marker, "symbol", None)
        outline = getattr(getattr(marker, "line", None), "color", None)
        if symbol is None or outline is None:
            continue
        out.setdefault(str(trace.name), set()).add((str(symbol), str(outline)))
    return out


def test_colour_and_symbol_are_locked_one_to_one_on_the_built_figure():
    """Each symbol has its own colour, and each colour its own symbol.

    The user's ruling: "each SYMBOL should have its own COLOR ... so color-symbol is a
    locked relationship". The map-level half is asserted in
    `test_sensitivity_benchmarking_math.py`; THIS is the emitted half, and it is not
    redundant with it -- a map can be a perfect bijection while a CALL SITE resolves one
    channel through it and the other through something else, which is exactly the class of
    defect that shipped when symbol read `is_gpu_group` and colour read a run-mode
    resolver.

    Both directions are checked. A figure with four symbols over three colours and one
    with three symbols over four colours are different defects, and a single cardinality
    comparison catches only one of them.
    """
    pairs_by_key = legend_style_pairs(build_figure())
    assert pairs_by_key, "no marker traces were emitted -- the fixture built an empty figure"

    unstable = {k: sorted(v) for k, v in pairs_by_key.items() if len(v) > 1}
    assert not unstable, f"legend key(s) drawn with more than one (symbol, colour) pair: {unstable}"

    pairs = {p for v in pairs_by_key.values() for p in v}
    symbols = {s for s, _c in pairs}
    colours = {c for _s, c in pairs}
    assert len(symbols) == len(
        pairs
    ), f"a symbol is drawn in more than one colour, so the lock is broken: {sorted(pairs)}"
    assert len(colours) == len(
        pairs
    ), f"a colour is drawn with more than one symbol, so the lock is broken: {sorted(pairs)}"


def test_marker_colour_equals_line_colour_for_every_connected_series():
    """ "point color = line color", asserted on the drawn traces.

    Checked on the figure rather than at the assignment because the risk is two SOURCES
    that happen to agree: the marker outline and the connector are set in different
    `go.Scatter` calls, and nothing in either call names the other. A single local feeding
    both is the current implementation and this is what would catch it drifting.

    Series with no connector are skipped by construction, not by exception: the serial
    baseline is a lone point and draws no line, so it has no line colour to match. The
    `checked` guard keeps that skip from swallowing every series.
    """
    fig = build_figure()
    marker_colour = {k: {c for _s, c in v} for k, v in legend_style_pairs(fig).items()}
    line_colour: dict[str, set[str]] = {}
    for trace in fig.data:
        if (getattr(trace, "mode", "") or "") != "lines":
            continue
        colour = getattr(getattr(trace, "line", None), "color", None)
        if colour is None or str(trace.name) not in marker_colour:
            continue
        line_colour.setdefault(str(trace.name), set()).add(str(colour))

    mismatches = {k: (sorted(marker_colour[k]), sorted(v)) for k, v in line_colour.items() if marker_colour[k] != v}
    assert not mismatches, f"marker outline and connector colour disagree: {mismatches}"
    assert len(line_colour) >= 2, (
        f"only {len(line_colour)} connected series were checked -- too few for this "
        "assertion to discriminate; the fixture must carry at least two multi-point series"
    )


def test_a_gpu_series_is_drawn_exactly_like_cpu_mpi_on_the_built_figure():
    """Hardware is the COLUMN and nothing else, asserted where it is visible.

    The retired doctrine gave each GPU hardware token a colour of its own. Under the
    ruling a GPU MPI run and a CPU MPI run share one decomposition, so they share one
    legend key, one symbol and one colour, and are told apart by their column alone. This
    checks the emitted figure rather than the map, because the GPU path reaches the marker
    through `is_gpu_group`, which is precisely the flag the retired scheme keyed on.
    """
    fig = build_figure()
    mpi_key = "MPI ranks"
    pairs = legend_style_pairs(fig).get(mpi_key)
    assert pairs, f"the MPI legend key is absent from the figure; keys={sorted(legend_style_pairs(fig))}"
    assert len(pairs) == 1, (
        f"the MPI key -- which the CPU-MPI series and both GPU series all carry -- is drawn "
        f"with more than one (symbol, colour) pair: {sorted(pairs)}"
    )
    # And the key really is shared, or the assertion above is vacuous.
    gpu_cols = [c for c in (2, 3) if panel_marker_xs(fig, 1, c)]
    assert gpu_cols, "no GPU column carried marker traces"
    for col in gpu_cols:
        ref = fig.get_subplot(1, col).xaxis.plotly_name.replace("axis", "")
        names = {
            str(t.name)
            for t in fig.data
            if (getattr(t, "xaxis", None) or "x") == ref and "markers" in (getattr(t, "mode", "") or "")
        }
        assert names == {mpi_key}, f"GPU column {col} drew unexpected legend keys: {sorted(names)}"


# ── Invariant 1c: every marker is hollow ───────────────────────────────────


def marker_fills(fig) -> dict[str, set]:
    """Per legend key, the SET of `marker.color` values drawn under it.

    Values are normalised to hashables: the retired encoding produced a per-point LIST
    (one fill per marker), so a scalar and a list are both possible readings of this
    property and the test must be able to see either.
    """
    out: dict[str, set] = {}
    for trace in fig.data:
        if "markers" not in (getattr(trace, "mode", "") or ""):
            continue
        fill = getattr(getattr(trace, "marker", None), "color", None)
        if fill is None:
            continue
        # noqa on UP038: the PINNED ruff-pre-commit v0.6.0 hook demands `isinstance(fill, X | Y)`,
        # but ruff REMOVED UP038 (it is slower, and it wrongly implies other typing syntaxes work
        # in isinstance). The installed ruff 0.15.17 does not know the rule, so this directive is
        # inert under the project's E,W,F,I,B,UP select list. DELETE IT when .pre-commit-config.yaml
        # bumps past the removal -- RUF100 will flag it as unused if RUF is ever selected.
        key = tuple(str(f) for f in fill) if isinstance(fill, (list, tuple)) else str(fill)  # noqa: UP038
        out.setdefault(str(trace.name), set()).add(key)
    return out


def solid_marker_traces(fig) -> dict[str, set]:
    """Legend keys carrying any fill other than the transparent one."""
    return {k: v for k, v in marker_fills(fig).items() if v != {"rgba(0,0,0,0)"}}


def test_every_marker_is_hollow():
    """No marker carries a fill, on any trace, under any data.

    The user ruled this for OCCLUSION -- "id rather just go to all the hollow fill since
    sometimes multiple configs kinda overlap" -- and it RETIRES the replicate-count fill
    encoding, under which fill meant "this config was run more than once".

    Asserted on the built figure rather than on the constant, because the defect this
    replaces was never in the constant: `rgba(0,0,0,0)` was already the hollow value, and
    what varied was the per-point BRANCH each call site wrapped it in. A test reading
    `_HOLLOW_FILL` would have been green throughout.

    Fill is checked as a whole-trace property including the LIST form, since the retired
    encoding emitted one fill per point.
    """
    solid = solid_marker_traces(build_figure())
    assert not solid, f"marker traces carrying a fill: {solid}"


def test_both_hollow_spellings_parse_in_their_own_renderer():
    """The two hollow constants are NOT interchangeable, and the wrong one raises.

    `rgba(0,0,0,0)` is a plotly/CSS string; matplotlib's colour parser rejects it with
    `ValueError: 'c' argument must be a color ... not 'rgba(0,0,0,0)'`. The first pass at
    the all-hollow change used the plotly spelling at all six sites: the plotly report
    rendered perfectly and the matplotlib PUBLICATION path raised, so the defect was
    invisible in the figure everyone was looking at.

    This is the fast guard. `tests/test_synth_static_plots.py` catches it too, end-to-end,
    but only after a full static-plot render -- too slow to be the first thing that tells
    you which spelling you used.
    """
    import matplotlib.colors as mcolors

    from hhemt.report_renderers.sensitivity_benchmarking import _HOLLOW_FILL, _HOLLOW_FILL_MPL

    assert (
        mcolors.to_rgba(_HOLLOW_FILL_MPL)[3] == 0.0
    ), f"_HOLLOW_FILL_MPL={_HOLLOW_FILL_MPL!r} is not a transparent matplotlib colour"
    with pytest.raises(ValueError):
        mcolors.to_rgba(_HOLLOW_FILL)
    assert _HOLLOW_FILL.startswith("rgba("), (
        f"_HOLLOW_FILL={_HOLLOW_FILL!r} is no longer the plotly spelling; if the two "
        "renderers now share one vocabulary, collapse the constants rather than leaving "
        "a pair whose distinction no longer holds"
    )


def test_the_fixture_contains_a_config_that_would_have_rendered_SOLID():
    """Guard on the fixture: without a single-run config the hollow test pins nothing.

    Fill used to be hollow for replicated configs, so a fixture in which EVERY config is
    replicated renders all-hollow under the retired encoding too -- and
    `test_every_marker_is_hollow` would pass identically before and after the ruling. The
    same rot risk the non-contiguous-sweep and co-labelled-groups guards cover, on the
    third invariant.
    """
    reps = _matrix_frame().groupby("config_id")["n_replicates"].max()
    single = sorted(reps[reps <= 1].index)
    assert single, (
        "every config in the fixture is replicated, so a pre-ruling renderer would draw "
        f"all markers hollow and the hollow invariant is unfalsifiable; replicate counts: "
        f"{reps.to_dict()}"
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
    assert (
        checked > 0
    ), "no panel declared a tick scheme, so this assertion checked nothing -- a vacuous pass, not a passing figure"
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
