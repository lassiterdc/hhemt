"""F11 durable per-attempt wall-time ledger — reader helper.

Tests `read_walltime_ledger_total_s`: it sums the append-only per-attempt ledger the runner
writes at each sim-finalize, and returns None when the ledger is absent (so non-resumed +
legacy trees fall back to the perf-summary total, byte-unchanged). Grounded by the confirmed
Rivanna under-count (sa_serial_6_r1: perf_Total 372.3 s vs Σ per-attempt 489 s, ~24%).
"""

from __future__ import annotations

import json

from hhemt.run_simulation import read_walltime_ledger_total_s


def _write_ledger(model_logfile, records):
    wl_dir = model_logfile.parent / "_walltime"
    wl_dir.mkdir(parents=True, exist_ok=True)
    p = wl_dir / f"{model_logfile.stem}.jsonl"
    with open(p, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_walltime_ledger_total_sums_all_attempts(tmp_path):
    mlf = tmp_path / "logs" / "sims" / "model_triton_sa_serial_6_r1_evt0.log"
    mlf.parent.mkdir(parents=True, exist_ok=True)
    # Three kill-truncated attempts + a completing attempt (the sa_serial_6_r1 shape).
    _write_ledger(
        mlf,
        [
            {"attempt": 0, "wall_s": 134.0, "completed": False},
            {"attempt": 1, "wall_s": 112.0, "completed": False},
            {"attempt": 2, "wall_s": 139.0, "completed": False},
            {"attempt": 3, "wall_s": 104.0, "completed": True},
        ],
    )
    total = read_walltime_ledger_total_s(mlf)
    assert total == 134.0 + 112.0 + 139.0 + 104.0  # 489.0 — the cumulative wall the perf total under-counts


def test_walltime_ledger_absent_returns_none(tmp_path):
    mlf = tmp_path / "logs" / "sims" / "model_triton_sa_serial_6_r1_evt0.log"
    mlf.parent.mkdir(parents=True, exist_ok=True)
    # No _walltime dir written -> None (fallback signal; non-resumed + legacy byte-unchanged).
    assert read_walltime_ledger_total_s(mlf) is None


def test_walltime_ledger_single_attempt(tmp_path):
    mlf = tmp_path / "logs" / "sims" / "model_triton_sa_serial_0_r1_evt0.log"
    mlf.parent.mkdir(parents=True, exist_ok=True)
    _write_ledger(mlf, [{"attempt": 0, "wall_s": 61.5, "completed": True}])
    assert read_walltime_ledger_total_s(mlf) == 61.5


def test_walltime_ledger_tolerates_blank_lines(tmp_path):
    mlf = tmp_path / "logs" / "sims" / "model_triton_x_evt0.log"
    mlf.parent.mkdir(parents=True, exist_ok=True)
    p = _write_ledger(mlf, [{"attempt": 0, "wall_s": 10.0}])
    with open(p, "a") as f:
        f.write("\n")  # trailing blank line
    assert read_walltime_ledger_total_s(mlf) == 10.0
