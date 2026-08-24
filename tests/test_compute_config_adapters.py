"""Adapter-boundary tests for the shared compute-config core.

The core (`hhemt.compute_config.n_devices_from`) owns the ARITHMETIC. Each adapter
owns its own coercion, missing-data policy, and GPU predicate. These tests pin the
per-adapter halves, because that is what a well-meaning future unification would
collapse -- and before this file, neither adapter had a malformed-input test at all,
which is how the two implementations diverged by an `n_nodes` factor unnoticed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hhemt.compute_config import n_devices_from


def test_core_is_arithmetic_only_and_carries_no_node_factor():
    """ranks x threads for CPU, n_gpus for GPU -- no n_nodes anywhere."""
    assert n_devices_from(is_gpu=False, n_gpus=0, n_mpi_procs=4, n_omp_threads=2) == 8
    assert n_devices_from(is_gpu=True, n_gpus=2, n_mpi_procs=4, n_omp_threads=2) == 2


def test_config_diff_adapter_never_raises_on_malformed_attrs():
    """It is an ORDERING key: a panel must sort, never abort.

    `_to_int` swallows a non-numeric to 0 and `max(x, 1)` floors ranks/threads, so a
    junk attrs dict yields 1 rather than an exception or a 0 that would sort first.
    """
    from hhemt.eda._config_diff import _device_count

    assert _device_count({"n_mpi_procs": "not-a-number", "n_omp_threads": None}) == 1
    assert _device_count({}) == 1
    assert _device_count({"run_mode": "gpu", "n_gpus": 3}) == 3


def test_renderer_adapter_raises_only_when_the_x_axis_depends_on_it():
    """Missing columns raise iff independent_var == 'n_devices'; else pass through.

    Anchored on the RETURNED OBJECT in the non-raising arm, not on a message string,
    so the assertion discriminates behaviour rather than wording.
    """
    from hhemt.report_renderers.sensitivity_benchmarking import _ensure_n_devices_column

    incomplete = pd.DataFrame([{"sa_id": "a", "n_mpi_procs": 2}])
    with pytest.raises(ValueError, match="Cannot derive n_devices"):
        _ensure_n_devices_column(incomplete, "n_devices")
    out = _ensure_n_devices_column(incomplete, "n_mpi_procs")
    assert "n_devices" not in out.columns
    assert out.equals(incomplete)


def test_the_two_adapters_keep_DIFFERENT_gpu_predicates():
    """The divergence a unification would silently erase.

    A CPU-mode row that nonetheless requests a GPU: the renderer's broader predicate
    counts DEVICES (1), while `_device_count`'s narrow `run_mode == "gpu"` counts
    CORES (2 x 2 = 4). Both are correct for their own consumer. If this test ever
    reddens because the two agree, the predicate was unified -- read
    `hhemt/compute_config.py`'s docstring before "fixing" it.
    """
    from hhemt.eda._config_diff import _device_count
    from hhemt.report_renderers.sensitivity_benchmarking import _ensure_n_devices_column

    attrs = {"run_mode": "hybrid", "n_gpus": 1, "n_mpi_procs": 2, "n_omp_threads": 2}
    assert _device_count(attrs) == 4

    df = pd.DataFrame([{"sa_id": "h", **attrs, "n_nodes": 1}])
    assert int(_ensure_n_devices_column(df, "n_devices")["n_devices"].iloc[0]) == 1
