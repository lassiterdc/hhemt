"""Every bundle-harvest source declaration must name a store that EXISTS.

`_harvest_and_copy_sources` skips a declared-but-absent source with a warning and a
non-fatal `continue`, and the manifest harvest is the ONLY path into a render bundle
(`grep -c datatree src/hhemt/bundle/_emit.py` returns 0). So a declaration naming a
retired store silently drops that store from every bundle -- and a contract census
finds these four functions are the only symbols this repair touches that NO test
observes, directly or through any caller.

Anchored on EXISTENCE of the declared path, which is meaningful in both the pre-fix
and post-fix worlds, never on the store's name.

Fixture-free and solver-free: `tmp_path` only, no `conftest` fixture request, no
`compile_TRITON_SWMM` call, no zarr write.
"""

from pathlib import Path

import pytest

from hhemt.eda._config_diff import config_diff_source_paths
from hhemt.eda._dem_resolution_plots import (
    dem_resolution_coupling_source_paths,
    dem_resolution_diff_source_paths,
    dem_resolution_source_paths,
)
from hhemt.utils import ROOT_TREE_NAMES

_DECLARERS = (
    config_diff_source_paths,
    dem_resolution_source_paths,
    dem_resolution_diff_source_paths,
    dem_resolution_coupling_source_paths,
)


def _root(tmp_path: Path, *stores: str) -> Path:
    root = tmp_path / "analysis"
    root.mkdir()
    for s in stores:
        (root / s).mkdir()
    return root


@pytest.mark.parametrize("declare", _DECLARERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("store", ROOT_TREE_NAMES)
def test_declared_root_store_exists_under_the_root(tmp_path: Path, declare, store: str) -> None:
    """The declared consolidated store must be the one actually on disk.

    Parametrized over every accepted name so the unified store is the VIOLATING arm
    (pre-fix each declarer names the retired store, which is absent) and the two
    retired stores are differently-positioned SATISFYING arms a legacy tree still hits.
    """
    root = _root(tmp_path, store)
    declared = declare(root)
    assert declared, f"{declare.__name__} declared no sources at all"
    assert declared[0].exists(), (
        f"{declare.__name__} declared {declared[0].name!r} under a root carrying "
        f"{store!r}; the harvest will SKIP it and the bundle will ship no root tree"
    )
    assert declared[0].name == store


@pytest.mark.parametrize("declare", _DECLARERS, ids=lambda f: f.__name__)
def test_both_retired_names_resolve_to_the_master_aggregate(tmp_path: Path, declare) -> None:
    """A root carrying BOTH retired names must resolve to the MASTER aggregate.

    THIS IS THE CASE THE SINGLE-STORE PARAMETRIZATION ABOVE CANNOT REACH. Every case
    there creates exactly one store, so the both-present shape is unreachable by
    construction and the whole module can pass while resolution order is wrong --
    which it was. `analysis_datatree.zarr` at an experiment root is that experiment's
    OWN single-analysis tree; the aggregate a declarer must publish is the master's.
    """
    root = _root(tmp_path, "analysis_datatree.zarr", "sensitivity_datatree.zarr")
    declared = declare(root)
    assert declared[0].exists()
    assert declared[0].name == "sensitivity_datatree.zarr", (
        f"{declare.__name__} resolved {declared[0].name!r} on a root carrying both "
        f"retired names; ROOT_TREE_NAMES must order the master store before the "
        f"single-analysis one"
    )


def test_declarations_are_absent_tolerant(tmp_path: Path) -> None:
    """A root with no store still yields a non-empty declaration.

    ADR-6 Gate-C hard-fails a figure declaring zero sources, so the declarers must not
    become empty when the store is missing -- the harvest's skip-with-warning is the
    sanctioned degradation, not an empty declaration.
    """
    root = _root(tmp_path)
    for declare in _DECLARERS:
        assert declare(root), f"{declare.__name__} returned an empty declaration"
