"""Clause 1: clear_raw_for_timesteps performs ONE accounting call per chapter, not one per file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hhemt import du_sentinels
from hhemt.utils import clear_raw_for_timesteps


def test_clear_raw_for_timesteps_one_accounting_call(tmp_path: Path, monkeypatch):
    scen = tmp_path / "sims" / "evt0"
    raw = scen / "out_triton" / "H"
    raw.mkdir(parents=True)
    paths = [raw / f"H_{i:04d}.bin" for i in range(12)]
    for p in paths:
        p.write_bytes(b"x" * 8)
    df = pd.DataFrame({"H": [str(p) for p in paths]}, index=list(range(12)))
    calls = {"n": 0}
    real = du_sentinels.delete_and_account

    def spy(ps, **kw):
        calls["n"] += 1
        return real(ps, **kw)

    monkeypatch.setattr(du_sentinels, "delete_and_account", spy)
    freed = clear_raw_for_timesteps(df, list(range(12)), scenario_dir=scen)
    assert calls["n"] == 1
    assert freed == 96
    assert not any(p.exists() for p in paths)
