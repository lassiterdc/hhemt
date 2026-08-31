"""N1/G3: the config_diff grouping RULE must be identical across model arms.

The grouping AXIS is byte-identity: configs that are b4b-identical to each other form a
group, and the prescribed serial -> CPU -> GPU ordering is applied WITHIN each identity
group. Identity partitions are model-DEPENDENT — measured, the coupled arm's partition
splits along CPU-vs-GPU while the pure-TRITON arm holds one group straddling Serial,
OpenMP, MPI, Hybrid and GPU — so the two arms may legitimately show a different panel
COUNT for the same figure.

That count difference is an honest measured property of each arm's data, NOT a G3
violation. G3 binds the RULE, not the count: both arms must use the same grouping
function, the same ordering key, the same columns and the same palettes, and every
compute config must appear in exactly one panel. These tests assert the rule-level
invariant and deliberately permit the count to differ; the rendered caption carries the
count disclosure so a reader cannot mistake a data-driven difference for a renderer one.

Supersedes the earlier compute-attrs grouping axis, which asserted panel-set identity
across arms and was retired when identity-first grouping was adopted.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import hhemt.eda._config_diff as cd

_CONFIGS = [
    ("member_serial_1", "serial", {"run_mode": "serial", "n_mpi_procs": 1, "n_omp_threads": 1, "n_gpus": 0}),
    ("member_openmp_2", "openmp", {"run_mode": "openmp", "n_mpi_procs": 1, "n_omp_threads": 2, "n_gpus": 0}),
    ("member_openmp_8", "openmp", {"run_mode": "openmp", "n_mpi_procs": 1, "n_omp_threads": 8, "n_gpus": 0}),
    ("member_mpi_2", "mpi", {"run_mode": "mpi", "n_mpi_procs": 2, "n_omp_threads": 1, "n_gpus": 0}),
    ("member_mpi_8", "mpi", {"run_mode": "mpi", "n_mpi_procs": 8, "n_omp_threads": 1, "n_gpus": 0}),
    ("member_hybrid_4", "hybrid", {"run_mode": "hybrid", "n_mpi_procs": 2, "n_omp_threads": 2, "n_gpus": 0}),
    ("member_gpu_a6000_1", "gpu", {"run_mode": "gpu", "n_gpus": 1, "hpc.partition": "gpu-a6000"}),
    ("member_gpu_a6000_2", "gpu", {"run_mode": "gpu", "n_gpus": 2, "hpc.partition": "gpu-a6000"}),
    ("member_gpu_a100_1", "gpu", {"run_mode": "gpu", "n_gpus": 1, "hpc.partition": "gpu-a100-80"}),
]

# The two measured arm shapes. Coupled: GPU configs mutually identical, each CPU config
# distinct. Pure-TRITON: one group straddling every run mode.
_COUPLED = {member: ("G" if "gpu" in member else member) for member, _, _ in _CONFIGS}
_PURE = dict.fromkeys((member for member, _, _ in _CONFIGS), "ALL")


def _subs():
    out = {}
    for member_id, run_mode, attrs in _CONFIGS:
        da = xr.DataArray(np.zeros((4, 4)), dims=("y", "x"))
        out[member_id] = {
            "attrs": attrs,
            "label": cd._derive_config_label(attrs),
            "run_mode": run_mode,
            "n_resumes": 0,
            "wlevel": da,
            "flow": None,
        }
    return out


def _ordered(groups):
    """Panels in rendered order — the ordering key is what G3 binds, not the count."""
    return sorted(groups, key=cd._panel_order_key)


def test_grouping_rule_is_identical_across_model_arms(monkeypatch):
    """G3 at the RULE level: same partition function, same ordering, complete cover.

    Both arms are grouped by the SAME callable and ordered by the SAME key, every compute
    config lands in exactly one panel, and each arm's first panel is the one holding
    serial CPU. The panel COUNT differs — that is the measured property this test exists
    to permit, and asserting it here is what keeps a future reader from "restoring" the
    retired count-level invariant.
    """
    all_labels = {cd._derive_config_label(a) for _, _, a in _CONFIGS}
    seen = {}
    for arm, partition in (("coupled", _COUPLED), ("pure", _PURE)):
        monkeypatch.setattr(cd, "_identity_labels", lambda root, _p=partition: _p)
        groups = _ordered(cd._group_by_identity(_subs(), None))
        # Complete, non-overlapping cover of the same config set.
        flat = [lab for g in groups for lab in g["labels"]]
        assert set(flat) == all_labels, f"{arm}: panels do not cover every compute config"
        assert len(flat) == len(_CONFIGS), f"{arm}: a config appears in more than one panel"
        # The serial-CPU identity group renders first in BOTH arms.
        assert "serial" in groups[0]["run_modes"], f"{arm}: serial group does not sort first"
        seen[arm] = len(groups)

    assert seen["coupled"] != seen["pure"], (
        "the two arms' identity partitions are constructed to differ, so their panel "
        "counts must differ — if they match, this fixture no longer exercises the "
        "data-determined-count property the caption discloses"
    )


def test_serial_identity_group_sorts_first(monkeypatch):
    """_panel_order_key's leading term puts the group CONTAINING serial CPU at the top.

    Under identity grouping the serial group is not necessarily the smallest or the
    CPU-only one, so 'serial first' has to be expressed in the ordering key rather than
    by prepending a separately-selected group.
    """
    monkeypatch.setattr(cd, "_identity_labels", lambda root: _COUPLED)
    groups = _ordered(cd._group_by_identity(_subs(), None))
    assert "serial" in groups[0]["run_modes"]
    # And the term is genuinely leading: a larger, non-serial group still sorts after it.
    assert any(len(set(g["labels"])) > len(set(groups[0]["labels"])) for g in groups[1:]), (
        "fixture no longer contains a larger non-serial group, so the leading serial "
        "term is not being discriminated from the size term"
    )


def test_group_sizes_are_non_vacuous(monkeypatch):
    """'larger groups towards the top' is only meaningful if groups vary in membership."""
    monkeypatch.setattr(cd, "_identity_labels", lambda root: _COUPLED)
    sizes = sorted(len(set(g["labels"])) for g in cd._group_by_identity(_subs(), None))
    assert max(sizes) > 1, "every group holds one config -> the size-ordering clause is vacuous"


def test_uniform_grid_guard_survives(monkeypatch):
    """The cell-wise-subtraction precondition is independent of the grouping axis."""
    monkeypatch.setattr(cd, "_identity_labels", lambda root: _COUPLED)
    subs = _subs()
    subs["member_mpi_8"]["wlevel"] = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))
    with pytest.raises(Exception, match="UNIFORM grid"):
        cd._group_by_identity(subs, None)
