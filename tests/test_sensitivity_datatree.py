"""Unit tests for sensitivity DataTree assembly (Phase 3).

Validates that ``build_sensitivity_datatree()`` produces an ``xr.DataTree`` with:
- ``parameters`` Dataset at the root, indexed by ``sa_id``
- Per-sub-analysis child nodes named ``sa_{sa_id}`` carrying sensitivity
  parameters as ``.attrs``
- Each sub-analysis subtree reproduces the per-analysis tree structure
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr


class _FakeProcess:
    def __init__(self, tree: xr.DataTree):
        self._tree = tree

    def open_datatree(self) -> xr.DataTree:
        return self._tree


def _leaf_ds() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"max_wlevel_m": (("event_iloc", "y", "x"), np.zeros((1, 2, 2)))},
        coords={
            "event_iloc": [0],
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        },
    )


def _build_fake_sub_tree() -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"analysis_id": "sub"}),
            "tritonswmm/triton": _leaf_ds(),
        }
    )


def test_build_sensitivity_datatree_structure(tmp_path):
    from TRITON_SWMM_toolkit.sensitivity_analysis import (
        TRITONSWMM_sensitivity_analysis,
    )

    df_setup = pd.DataFrame(
        {"run_mode": ["gpu", "mpi"], "n_mpi_procs": [1, 2]},
        index=pd.Index(["0", "1"], name="sa_id"),
    )

    # `_refresh_log` is a read-only observer call the cross-analysis datatree read
    # site makes on each sub (added by the log-write-race-fix compute-on-read change);
    # the stub provides a no-op so the structural test does not depend on a real log.
    sub_analyses = {
        "0": SimpleNamespace(
            process=_FakeProcess(_build_fake_sub_tree()), _refresh_log=lambda: None
        ),
        "1": SimpleNamespace(
            process=_FakeProcess(_build_fake_sub_tree()), _refresh_log=lambda: None
        ),
    }

    master = SimpleNamespace(
        cfg_analysis=SimpleNamespace(analysis_id="master_a"),
    )

    sens = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    sens.master_analysis = master
    sens.sub_analyses = sub_analyses
    sens.df_setup = df_setup
    sens.sub_analyses_prefix = "sa_"

    tree = sens.build_sensitivity_datatree()

    assert "parameters" in tree
    assert "sa_0" in tree
    assert "sa_1" in tree
    assert tree["sa_0/tritonswmm/triton"].dataset["max_wlevel_m"].shape == (1, 2, 2)
    assert tree["sa_0"].attrs["run_mode"] == "gpu"
    assert tree["sa_1"].attrs["n_mpi_procs"] == 2
    df = tree["parameters"].dataset.to_dataframe()
    assert list(df.index) == ["0", "1"]
    assert df.loc["0", "run_mode"] == "gpu"
