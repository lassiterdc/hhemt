"""Unit tests for the sensitivity_benchmarking renderer's speedup + efficiency math.

Per user-locked design constraint (Phase 6 iter-5): "design tests to make sure you
get your math right" before wiring the helpers into the renderer.

Formulas under test (from HPC benchmarking literature; user-supplied references):

- Strong scaling speedup: S(N) = t(1) / t(N) where t(N) is wallclock at N devices on
  a fixed problem. Ideal: S(N) = N.
- Strong scaling efficiency: E_s(N) = S(N) / N = t(1) / (N × t(N)). Ideal: 1.0.
- Weak scaling efficiency: E_w(N) = t(1) / t(N) where the per-device problem is
  fixed (total problem grows with N). Ideal: 1.0. Numerically equal to speedup for
  the same (t, N) inputs but interpreted differently (weak benchmark holds per-device
  workload constant; strong holds total workload constant).

Edge cases under test:

- N=1 baseline missing for a group → group is excluded from speedup/efficiency
  output (no normalization anchor).
- Multiple sa rows at the same N within a group → MIN-y (best wallclock) wins,
  matching the line-drawing rule from iter-1.
- Single sa per group → renders the one available point (S=1, E=1 if it's the
  N=1 baseline; otherwise excluded for lack of baseline).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from hhemt.report_renderers.sensitivity_benchmarking import (
    _collect_rows,
    _compute_efficiency_per_group,
    _compute_speedup_per_group,
    _find_perf_node,
)


def _df(rows):
    """Helper to build the wallclock-input dataframe shape the helpers expect."""
    return pd.DataFrame(rows, columns=["sa_id", "group", "n_devices", "wallclock_s"])


def _xy(points):
    """Strip production 3-tuple (n, value, sa_id) provenance triples to (n, value).

    The speedup/efficiency helpers return (n, value, sa_id); the sa_id third element
    is load-bearing (hover customdata + hybrid annotations, F2/F3 in
    sensitivity_benchmarking.py). These tests assert the numeric (n, value) contract;
    the sa_id provenance is asserted separately (test_min_y_at_duplicate_n and
    test_global_anchor_picks_min_t_when_multiple_at_min_N).
    """
    return [(p[0], p[1]) for p in points]


# ── Strong speedup: S(N) = t(1) / t(N) ─────────────────────────────────────


class TestComputeSpeedup:
    def test_perfect_speedup_doubling(self):
        """t halves each time N doubles → S = N exactly."""
        df = _df([
            ("a", "mpi", 1, 100.0),
            ("b", "mpi", 2, 50.0),
            ("c", "mpi", 4, 25.0),
            ("d", "mpi", 8, 12.5),
        ])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        assert "mpi" in result
        pts = sorted(result["mpi"], key=lambda r: r[0])
        ns = [p[0] for p in pts]
        speedups = [p[1] for p in pts]
        assert ns == [1, 2, 4, 8]
        assert speedups == pytest.approx([1.0, 2.0, 4.0, 8.0])

    def test_imperfect_speedup(self):
        """Realistic numbers: S(2)=1.8, S(4)=3.5 (sub-linear)."""
        df = _df([
            ("a", "mpi", 1, 10.0),
            ("b", "mpi", 2, 10.0 / 1.8),  # ≈5.555
            ("c", "mpi", 4, 10.0 / 3.5),  # ≈2.857
        ])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        ns = [p[0] for p in sorted(result["mpi"], key=lambda r: r[0])]
        ss = [p[1] for p in sorted(result["mpi"], key=lambda r: r[0])]
        assert ns == [1, 2, 4]
        assert ss == pytest.approx([1.0, 1.8, 3.5])

    def test_missing_n1_baseline_excludes_group(self):
        """Group without N=1 has no anchor; entire group dropped from speedup output."""
        df = _df([
            ("a", "openmp", 2, 5.0),
            ("b", "openmp", 4, 2.5),
        ])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        assert "openmp" not in result

    def test_min_y_at_duplicate_n(self):
        """Multiple sa rows at same N within a group → use MIN wallclock (fastest config wins)."""
        df = _df([
            ("a", "hybrid", 1, 10.0),
            ("b", "hybrid", 4, 4.0),  # slower
            ("c", "hybrid", 4, 2.0),  # faster — should win
        ])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        pts = sorted(result["hybrid"], key=lambda r: r[0])
        # Anchor: t(1)=10. At N=4 use min wallclock 2.0 → S = 10/2 = 5.0.
        assert _xy(pts) == [(1, pytest.approx(1.0)), (4, pytest.approx(5.0))]
        # D2 provenance lock: the N=4 winner is the min-wallclock row, sa_id "c" (t=2.0 < 4.0).
        assert pts[1][2] == "c"

    def test_multiple_groups_independent(self):
        """Each run_mode is anchored to its own t(1); cross-group leakage is forbidden."""
        df = _df([
            ("a", "mpi", 1, 10.0),
            ("b", "mpi", 4, 2.5),
            ("c", "openmp", 1, 20.0),  # different baseline
            ("d", "openmp", 4, 5.0),
        ])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        # mpi: S(4) = 10/2.5 = 4
        # openmp: S(4) = 20/5 = 4
        # The fact that both end up at 4 is coincidence; the load-bearing assertion is
        # that openmp uses ITS OWN t(1)=20, not mpi's t(1)=10.
        mpi_pts = dict(_xy(result["mpi"]))
        openmp_pts = dict(_xy(result["openmp"]))
        assert mpi_pts[1] == pytest.approx(1.0)
        assert mpi_pts[4] == pytest.approx(4.0)
        assert openmp_pts[1] == pytest.approx(1.0)
        assert openmp_pts[4] == pytest.approx(4.0)

    def test_only_n1_point_speedup_is_unity(self):
        """Group with only the N=1 baseline → S(1) = 1.0."""
        df = _df([("a", "serial", 1, 5.0)])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        assert _xy(result["serial"]) == [(1, pytest.approx(1.0))]

    def test_empty_dataframe(self):
        df = _df([])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        assert result == {}


# ── Strong/weak efficiency ─────────────────────────────────────────────────


class TestComputeEfficiency:
    def test_strong_efficiency_perfect(self):
        """Perfect speedup S=N → E_strong = S/N = 1.0 at every N."""
        df = _df([
            ("a", "mpi", 1, 8.0),
            ("b", "mpi", 2, 4.0),
            ("c", "mpi", 4, 2.0),
            ("d", "mpi", 8, 1.0),
        ])
        result = _compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="strong"
        )
        for n, e in _xy(result["mpi"]):
            assert e == pytest.approx(1.0), f"strong E({n}) expected 1.0, got {e}"

    def test_strong_efficiency_imperfect(self):
        """S(N)=1.8 at N=2 → E = 0.9. S(N)=3.5 at N=4 → E = 0.875."""
        df = _df([
            ("a", "mpi", 1, 10.0),
            ("b", "mpi", 2, 10.0 / 1.8),
            ("c", "mpi", 4, 10.0 / 3.5),
        ])
        result = _compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="strong"
        )
        eff = dict(_xy(result["mpi"]))
        assert eff[1] == pytest.approx(1.0)
        assert eff[2] == pytest.approx(1.8 / 2.0)  # 0.9
        assert eff[4] == pytest.approx(3.5 / 4.0)  # 0.875

    def test_weak_efficiency_equals_speedup_numerically(self):
        """E_weak(N) = t(1)/t(N), same number as speedup but different interpretation."""
        df = _df([
            ("a", "mpi", 1, 10.0),
            ("b", "mpi", 2, 11.0),  # weak: per-device problem fixed; t grew slightly with N
            ("c", "mpi", 4, 12.5),
        ])
        weak = dict(_xy(_compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="weak"
        )["mpi"]))
        assert weak[1] == pytest.approx(1.0)
        assert weak[2] == pytest.approx(10.0 / 11.0)
        assert weak[4] == pytest.approx(10.0 / 12.5)  # 0.8

    def test_weak_efficiency_min_y_at_duplicate_n(self):
        df = _df([
            ("a", "hybrid", 1, 10.0),
            ("b", "hybrid", 4, 14.0),
            ("c", "hybrid", 4, 12.0),  # min wins
        ])
        result = dict(_xy(_compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="weak"
        )["hybrid"]))
        assert result[4] == pytest.approx(10.0 / 12.0)

    def test_efficiency_invalid_mode_raises(self):
        df = _df([("a", "mpi", 1, 10.0)])
        with pytest.raises(ValueError, match="mode must be 'strong' or 'weak'"):
            _compute_efficiency_per_group(
                df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="invalid"
            )

    def test_efficiency_missing_n1_excludes_group(self):
        df = _df([
            ("a", "openmp", 2, 5.0),
            ("b", "openmp", 4, 3.0),
        ])
        for mode in ("strong", "weak"):
            result = _compute_efficiency_per_group(
                df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode=mode
            )
            assert "openmp" not in result, f"mode={mode}: group without N=1 baseline must be excluded"


# ── Global-baseline anchor (cross-group t_min at smallest N) ───────────────


class TestGlobalBaselineSpeedup:
    def test_global_anchor_uses_min_t_at_min_N_across_groups(self):
        """Global baseline = min t at smallest N across all groups (typically the
        serial baseline). Each group's points normalize against that anchor.
        """
        df = _df([
            ("a", "serial", 1, 4.0),  # global min t at N=1 → anchor
            ("b", "mpi",    2, 2.0),
            ("c", "openmp", 2, 3.0),
            ("d", "hybrid", 4, 1.0),
        ])
        result = _compute_speedup_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
            baseline_mode="global",
        )
        # All four groups should appear (no per-group N=1 anchor required).
        assert set(result.keys()) == {"serial", "mpi", "openmp", "hybrid"}
        assert dict(_xy(result["serial"])) == {1: pytest.approx(1.0)}
        assert dict(_xy(result["mpi"])) == {2: pytest.approx(4.0 / 2.0)}    # 2.0
        assert dict(_xy(result["openmp"])) == {2: pytest.approx(4.0 / 3.0)}  # ≈1.333
        assert dict(_xy(result["hybrid"])) == {4: pytest.approx(4.0 / 1.0)}  # 4.0

    def test_global_anchor_picks_min_t_when_multiple_at_min_N(self):
        """If multiple groups have N=1 entries, the smallest wallclock among them
        is the global anchor.
        """
        df = _df([
            ("a", "serial",  1, 5.0),
            ("b", "openmp1", 1, 4.0),  # min at N=1 → global anchor
            ("c", "openmp1", 2, 2.0),
        ])
        result = _compute_speedup_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
            baseline_mode="global",
        )
        # Anchor = 4.0. serial @ N=1: 4.0/5.0 = 0.8 (slower than baseline).
        # openmp1 @ N=2: 4.0/2.0 = 2.0.
        assert dict(_xy(result["serial"])) == {1: pytest.approx(0.8)}
        assert dict(_xy(result["openmp1"])) == {1: pytest.approx(1.0), 2: pytest.approx(2.0)}
        # D2 provenance lock: openmp1's N=1 point is the global-anchor row, sa_id "b" (t=4.0).
        assert {p[0]: p[2] for p in result["openmp1"]}[1] == "b"

    def test_global_anchor_includes_groups_without_n1(self):
        """Per-group mode would exclude these; global mode includes them."""
        df = _df([
            ("a", "serial", 1, 10.0),
            ("b", "mpi",    2, 5.0),    # no N=1 entry in mpi group
            ("c", "openmp", 4, 4.0),    # no N=1 entry in openmp group
        ])
        result = _compute_speedup_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
            baseline_mode="global",
        )
        assert "mpi" in result
        assert "openmp" in result
        assert dict(_xy(result["mpi"])) == {2: pytest.approx(2.0)}
        assert dict(_xy(result["openmp"])) == {4: pytest.approx(2.5)}

    def test_global_efficiency_strong_normalizes_by_N(self):
        """Strong efficiency: anchor / (N × t(N))."""
        df = _df([
            ("a", "serial", 1, 4.0),
            ("b", "mpi",    2, 2.0),
        ])
        result = _compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
            mode="strong", baseline_mode="global",
        )
        # Anchor = 4.0. mpi @ N=2: E = 4.0 / (2 × 2.0) = 1.0 (perfect efficiency).
        assert dict(_xy(result["mpi"])) == {2: pytest.approx(1.0)}

    def test_global_efficiency_weak_does_not_normalize_by_N(self):
        df = _df([
            ("a", "serial", 1, 4.0),
            ("b", "mpi",    2, 2.0),
        ])
        result = _compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
            mode="weak", baseline_mode="global",
        )
        # Anchor = 4.0. mpi @ N=2: E_weak = 4.0 / 2.0 = 2.0.
        assert dict(_xy(result["mpi"])) == {2: pytest.approx(2.0)}

    def test_global_baseline_invalid_baseline_mode_raises(self):
        df = _df([("a", "serial", 1, 5.0)])
        with pytest.raises(ValueError, match="baseline_mode"):
            _compute_speedup_per_group(
                df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
                baseline_mode="invalid",
            )


# ── _find_perf_node datatree-group resolution ──────────────────────────────


class TestFindPerfNode:
    """A pure-TRITON sub-analysis consolidates under ``triton_only/performance``;
    the pre-fix reader only tried ``triton/performance`` (a spelling nothing
    writes), so it returned None and the sub contributed zero rows -> the
    ``RuntimeError('No data for benchmarking ...')`` smoke crash (2026-07-26).
    """

    def _tree(self, group: str):
        import xarray as xr

        perf = xr.Dataset(
            {"Total": ("event_iloc", [12.5])}, coords={"event_iloc": [0]}
        )
        return xr.DataTree.from_dict({f"/sa_serial_0_r1/{group}": perf})

    def test_triton_only_performance_node_is_found(self):
        """Pre-fix: returns None (looked in triton/performance)."""
        tree = self._tree("triton_only/performance")
        node = _find_perf_node(tree, "serial_0_r1")
        assert node is not None
        assert "Total" in node.data_vars

    def test_coupled_tritonswmm_performance_still_found(self):
        """Regression: the coupled path is unchanged."""
        tree = self._tree("tritonswmm/performance")
        node = _find_perf_node(tree, "serial_0_r1")
        assert node is not None
        assert "Total" in node.data_vars

    def test_absent_node_returns_none(self):
        """A sub with no performance node still returns None (no raise)."""
        tree = self._tree("swmm_only/swmm_link")
        assert _find_perf_node(tree, "serial_0_r1") is None


# ── Phase 6 change (2): model-arm marker-fill / line-dash encoding ──────────


class TestModelArmEncoding:
    """Assert the single-arm model encoding is actually emitted onto the traces.

    Phase 6 change (2): coupled (tritonswmm) => FILLED markers + SOLID connector;
    uncoupled (triton_only) => OPEN markers (``-open`` symbol variant layered on
    the group-type base symbol) + DASHED connector; model_arm=None => the
    pre-change filled/dashed default (protects swmm-only masters). The math suite
    above asserts numeric correctness; this class closes the silent-visual-defect
    gap by inspecting the emitted ``go.Figure`` traces. Passing ``model_arm=`` is
    itself the fail-against-pre-change reachability guard: the pre-change panel
    builders take no ``model_arm`` kwarg, so these calls raise ``TypeError`` there.
    """

    @staticmethod
    def _sens_cfg():
        from hhemt.config.report import SensitivityReportConfig

        return SensitivityReportConfig(independent_vars=["n_devices"])

    @staticmethod
    def _connector_lines(fig):
        # Data-connector lines only; the ideal-reference line (legendgroup="ideal")
        # is not an arm connector and is excluded.
        return [
            t for t in fig.data
            if t.mode == "lines" and getattr(t, "legendgroup", None) != "ideal"
        ]

    @staticmethod
    def _marker_traces(fig):
        return [t for t in fig.data if t.mode and "markers" in t.mode]

    def _panel12_fig(self, model_arm):
        from plotly.subplots import make_subplots

        from hhemt.report_renderers._provenance import ProvenanceLog
        from hhemt.report_renderers.sensitivity_benchmarking import (
            _plotly_metric_panel,
        )

        # gpu group, 2 points at distinct indep_value => multi-point, non-serial,
        # so the connector LINE trace is emitted and the base symbol is triangle-up.
        df = pd.DataFrame(
            {
                "group_value": ["gpu", "gpu"],
                "indep_value": [1, 2],
                "wallclock_disp": [10.0, 5.0],
            }
        )
        fig = make_subplots(rows=1, cols=1)
        _plotly_metric_panel(
            fig, df, y_col="wallclock_disp", row=1, panel_id="p",
            group_by_var="run_mode", sens_cfg=self._sens_cfg(),
            prov=ProvenanceLog(), show_in_legend=True, model_arm=model_arm,
        )
        return fig

    def _panel34_fig(self, model_arm):
        from plotly.subplots import make_subplots

        from hhemt.report_renderers._provenance import ProvenanceLog
        from hhemt.report_renderers.sensitivity_benchmarking import (
            _plotly_metric_panel_precomputed,
        )

        per_group = {"gpu": [(1.0, 1.0, "sa_gpu_0"), (2.0, 2.0, "sa_gpu_1")]}
        df_for_groups = pd.DataFrame(
            {"group_value": ["gpu", "gpu"], "sa_id": ["sa_gpu_0", "sa_gpu_1"]}
        )
        fig = make_subplots(rows=1, cols=1)
        _plotly_metric_panel_precomputed(
            fig, per_group, df_for_groups=df_for_groups, row=1, panel_id="p",
            ideal_kind="linear", x_max=2.0, ideal_label="ideal",
            sens_cfg=self._sens_cfg(), prov=ProvenanceLog(),
            show_in_legend=True, model_arm=model_arm,
        )
        return fig

    # Iteration 4: these tests previously asserted the Phase-6 arm-conditioned encoding
    # (marker FILL and connector DASH keyed on model_arm). That encoding made the two
    # arms' symbology DISJOINT by construction, which violates the standing requirement
    # that benchmarking symbology be IDENTICAL across models. The requirement was
    # verified once, then silently regressed when the encoding landed, and nothing in
    # the suite caught it -- because the suite asserted the encoding rather than the
    # requirement. These tests now assert the REQUIREMENT, so the regression class
    # cannot recur silently.

    _ARMS = ("coupled", "uncoupled", None)

    @staticmethod
    def _encoding(fig):
        """The figure's symbology as a comparable value: sorted marker symbols + dashes."""
        markers = TestModelArmEncoding._marker_traces(fig)
        lines = TestModelArmEncoding._connector_lines(fig)
        return (
            tuple(sorted(str(t.marker.symbol) for t in markers)),
            tuple(sorted(str(t.line.dash) for t in lines)),
        )

    def test_panel12_symbology_identical_across_every_arm(self):
        # THE requirement: same figure, different arm -> byte-identical symbology.
        encodings = {arm: self._encoding(self._panel12_fig(arm)) for arm in self._ARMS}
        distinct = {enc for enc in encodings.values()}
        assert len(distinct) == 1, (
            "benchmarking symbology must be identical across model arms; got "
            + repr(encodings)
        )

    def test_panel34_symbology_identical_across_every_arm(self):
        encodings = {arm: self._encoding(self._panel34_fig(arm)) for arm in self._ARMS}
        assert len({enc for enc in encodings.values()}) == 1, (
            "benchmarking symbology must be identical across model arms; got "
            + repr(encodings)
        )

    def test_no_arm_conditioned_fill_or_dash_survives(self):
        # The specific mechanism that regressed the requirement: an `-open` fill variant
        # keyed on the arm, and a non-constant connector dash. Assert on the MECHANISM as
        # well as the parity, so a future change that breaks parity some OTHER way still
        # fails the test above while this one names the known cause.
        for arm in self._ARMS:
            for fig in (self._panel12_fig(arm), self._panel34_fig(arm)):
                markers = self._marker_traces(fig)
                assert markers, "expected at least one marker trace"
                assert not any(
                    str(t.marker.symbol).endswith("-open") for t in markers
                ), f"arm-conditioned open-fill survived for model_arm={arm!r}"
                lines = self._connector_lines(fig)
                assert lines, "expected a connector line for the multi-point group"
                assert all(
                    t.line.dash == "solid" for t in lines
                ), f"connector dash is not the constant single-arm style for {arm!r}"


def _stub_analysis(tmp_path, *, write_csv=True, ledger_value=900.0, perf_total=200.0):
    tree_path = tmp_path / "sensitivity_datatree.zarr"
    ds = xr.Dataset(
        {"Total": ("event_iloc", [perf_total])},
        coords={"event_iloc": [0]},
    )
    xr.DataTree.from_dict({"/sa_serial_6_r1/tritonswmm/performance": ds}).to_zarr(
        tree_path, consolidated=False
    )
    if write_csv:
        pd.DataFrame(
            {
                "sa_id": ["serial_6_r1"],
                "event_iloc": [0],
                "wall_clock_ledger_s": [ledger_value],
                "perf_Total": [perf_total],
            }
        ).to_csv(tmp_path / "scenario_status.csv", index=False)
    sub = SimpleNamespace(
        df_sims=pd.DataFrame(index=[0]),
        _get_enabled_model_types=lambda: {"tritonswmm"},
    )
    return SimpleNamespace(
        sensitivity=SimpleNamespace(sub_analyses={"serial_6_r1": sub}),
        analysis_paths=SimpleNamespace(
            sensitivity_datatree_zarr=tree_path,
            analysis_dir=tmp_path,
        ),
    )


def test_collect_rows_prefers_wall_clock_ledger_for_total(tmp_path):
    """SUBSTITUTION arm. A resumed sim's plotted wallclock must be the
    kill-survivable ledger sum, not performance.Total (which on a resumed sim
    covers only the checkpoints present on disk at process time)."""
    analysis = _stub_analysis(tmp_path)
    rows, source_paths = _collect_rows(analysis, "performance.Total")
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(900.0), (
        f"expected wall_clock_ledger_s (900.0), got {rows[0]['value']!r} "
        "-- the renderer is still sourcing performance.Total"
    )
    assert any(p.name == "scenario_status.csv" for p in source_paths), (
        "scenario_status.csv must be declared as a source path when the ledger is used"
    )


def test_collect_rows_falls_back_to_datatree_when_csv_absent(tmp_path):
    """NON-OVER-FIRE arm 1. With no scenario_status.csv the renderer degrades
    to performance.Total rather than raising."""
    analysis = _stub_analysis(tmp_path, write_csv=False)
    rows, _ = _collect_rows(analysis, "performance.Total")
    assert rows[0]["value"] == pytest.approx(200.0)


def test_collect_rows_does_not_substitute_ledger_for_simulation(tmp_path):
    """NON-OVER-FIRE arm 2. The ledger is a whole-process wall and must NOT be
    substituted for the Init/Simulation decompositions."""
    tree_path = tmp_path / "sensitivity_datatree.zarr"
    ds = xr.Dataset(
        {"Total": ("event_iloc", [200.0]), "Simulation": ("event_iloc", [150.0])},
        coords={"event_iloc": [0]},
    )
    xr.DataTree.from_dict({"/sa_serial_6_r1/tritonswmm/performance": ds}).to_zarr(
        tree_path, consolidated=False
    )
    pd.DataFrame(
        {"sa_id": ["serial_6_r1"], "event_iloc": [0], "wall_clock_ledger_s": [900.0]}
    ).to_csv(tmp_path / "scenario_status.csv", index=False)
    sub = SimpleNamespace(
        df_sims=pd.DataFrame(index=[0]),
        _get_enabled_model_types=lambda: {"tritonswmm"},
    )
    analysis = SimpleNamespace(
        sensitivity=SimpleNamespace(sub_analyses={"serial_6_r1": sub}),
        analysis_paths=SimpleNamespace(
            sensitivity_datatree_zarr=tree_path, analysis_dir=tmp_path
        ),
    )
    rows, _ = _collect_rows(analysis, "performance.Simulation")
    assert rows[0]["value"] == pytest.approx(150.0)
