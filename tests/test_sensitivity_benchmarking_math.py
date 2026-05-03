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

import pandas as pd
import pytest

from TRITON_SWMM_toolkit.report_renderers.sensitivity_benchmarking import (
    _compute_efficiency_per_group,
    _compute_speedup_per_group,
)


def _df(rows):
    """Helper to build the wallclock-input dataframe shape the helpers expect."""
    return pd.DataFrame(rows, columns=["sa_id", "group", "n_devices", "wallclock_s"])


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
        assert pts == [(1, pytest.approx(1.0)), (4, pytest.approx(5.0))]

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
        mpi_pts = dict(result["mpi"])
        openmp_pts = dict(result["openmp"])
        assert mpi_pts[1] == pytest.approx(1.0)
        assert mpi_pts[4] == pytest.approx(4.0)
        assert openmp_pts[1] == pytest.approx(1.0)
        assert openmp_pts[4] == pytest.approx(4.0)

    def test_only_n1_point_speedup_is_unity(self):
        """Group with only the N=1 baseline → S(1) = 1.0."""
        df = _df([("a", "serial", 1, 5.0)])
        result = _compute_speedup_per_group(df, t_col="wallclock_s", indep_col="n_devices", group_col="group")
        assert result["serial"] == [(1, pytest.approx(1.0))]

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
        for n, e in result["mpi"]:
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
        eff = dict(result["mpi"])
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
        weak = dict(_compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="weak"
        )["mpi"])
        assert weak[1] == pytest.approx(1.0)
        assert weak[2] == pytest.approx(10.0 / 11.0)
        assert weak[4] == pytest.approx(10.0 / 12.5)  # 0.8

    def test_weak_efficiency_min_y_at_duplicate_n(self):
        df = _df([
            ("a", "hybrid", 1, 10.0),
            ("b", "hybrid", 4, 14.0),
            ("c", "hybrid", 4, 12.0),  # min wins
        ])
        result = dict(_compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group", mode="weak"
        )["hybrid"])
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
        assert dict(result["serial"]) == {1: pytest.approx(1.0)}
        assert dict(result["mpi"]) == {2: pytest.approx(4.0 / 2.0)}    # 2.0
        assert dict(result["openmp"]) == {2: pytest.approx(4.0 / 3.0)}  # ≈1.333
        assert dict(result["hybrid"]) == {4: pytest.approx(4.0 / 1.0)}  # 4.0

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
        assert dict(result["serial"]) == {1: pytest.approx(0.8)}
        assert dict(result["openmp1"]) == {1: pytest.approx(1.0), 2: pytest.approx(2.0)}

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
        assert dict(result["mpi"]) == {2: pytest.approx(2.0)}
        assert dict(result["openmp"]) == {4: pytest.approx(2.5)}

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
        assert dict(result["mpi"]) == {2: pytest.approx(1.0)}

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
        assert dict(result["mpi"]) == {2: pytest.approx(2.0)}

    def test_global_baseline_invalid_baseline_mode_raises(self):
        df = _df([("a", "serial", 1, 5.0)])
        with pytest.raises(ValueError, match="baseline_mode"):
            _compute_speedup_per_group(
                df, t_col="wallclock_s", indep_col="n_devices", group_col="group",
                baseline_mode="invalid",
            )
