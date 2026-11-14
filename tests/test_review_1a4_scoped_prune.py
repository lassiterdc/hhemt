"""After-the-fact checks of the scoped zarr prune (commit d3b866a4). Lives outside the repo."""

import threading
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _store(path):
    xr.Dataset(
        {
            k: (("node_id", "date_time"), np.ones((2, 4)) * i)
            for i, k in enumerate(("depth", "head", "flooding", "inflow_flow"))
        },
        coords={"node_id": ["a", "b"], "date_time": np.arange(4)},
    ).to_zarr(path, mode="w", consolidated=True)
    return path


def _vars(path, consolidated):
    return sorted(xr.open_zarr(path, consolidated=consolidated, chunks=None).data_vars)


def test_scoped_prune_equals_unscoped_prune_and_index_is_current(tmp_path):
    import hhemt.timeseries_selection as sel

    a = _store(tmp_path / "a.zarr")
    b = _store(tmp_path / "b.zarr")
    scen = tmp_path / "scen"
    scen.mkdir()
    sel.prune_timeseries_store(a, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    sel.prune_timeseries_store(b, "swmm_node", "tritonswmm", ["depth"])
    assert _vars(a, True) == _vars(b, True) == ["depth"]
    xr.testing.assert_identical(xr.open_zarr(a, consolidated=True).load(), xr.open_zarr(b, consolidated=True).load())


def test_interrupted_scoped_prune_leaves_a_readable_store_and_rerun_completes(tmp_path, monkeypatch):
    import hhemt.du_sentinels as du
    import hhemt.timeseries_selection as sel

    store = _store(tmp_path / "a.zarr")
    scen = tmp_path / "scen"
    scen.mkdir()
    real = du.delete_and_account
    n = []

    def failing(paths, **kw):
        n.append(1)
        if len(n) == 2:
            raise OSError("killed")
        return real(paths, **kw)

    monkeypatch.setattr(du, "delete_and_account", failing)
    with pytest.raises(OSError):
        sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    monkeypatch.undo()
    # the state an interrupted prune leaves, read the way a reader reads an indexed store
    try:
        listed = _vars(store, True)
        idx = "ok:" + ",".join(listed)
    except Exception as e:
        idx = "RAISES " + type(e).__name__
    print("INTERRUPTED indexed open ->", idx)
    assert not idx.startswith("RAISES"), idx
    sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    assert _vars(store, True) == ["depth"]
    assert set(sel.pruned_variables(store)) == {"head", "flooding", "inflow_flow"}


def test_scope_does_not_leak_after_a_scoped_call_or_a_failure(tmp_path, monkeypatch):
    import hhemt.du_sentinels as du
    import hhemt.timeseries_selection as sel

    scen = tmp_path / "scen"
    scen.mkdir()
    calls = []
    real = du.delete_and_account
    monkeypatch.setattr(du, "delete_and_account", lambda p, **k: (calls.append(1), real(p, **k))[1])
    s1 = _store(tmp_path / "s1.zarr")
    sel.prune_timeseries_store(s1, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    scoped = len(calls)
    assert scoped == 3 and sel._ACCOUNT_SCOPE.get() is None
    s2 = _store(tmp_path / "s2.zarr")
    sel.prune_timeseries_store(s2, "swmm_node", "tritonswmm", ["depth"])
    assert len(calls) == scoped, "an unscoped prune went through the accounting tool"
    s3 = _store(tmp_path / "s3.zarr")
    monkeypatch.setattr(du, "delete_and_account", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        sel.prune_timeseries_store(s3, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    assert sel._ACCOUNT_SCOPE.get() is None


def test_scope_is_not_shared_across_threads(tmp_path, monkeypatch):
    import hhemt.timeseries_selection as sel

    scen = tmp_path / "scen"
    scen.mkdir()
    seen = {}
    gate = threading.Event()
    done = threading.Event()
    real_delete = sel._delete_zarr_array

    def spy(group, name):
        seen.setdefault(threading.current_thread().name, []).append(sel._ACCOUNT_SCOPE.get())
        if threading.current_thread().name == "A":
            gate.set()
            done.wait(5)
        real_delete(group, name)

    monkeypatch.setattr(sel, "_delete_zarr_array", spy)
    sa, sb = _store(tmp_path / "a.zarr"), _store(tmp_path / "b.zarr")
    ta = threading.Thread(
        name="A", target=lambda: sel.prune_timeseries_store(sa, "swmm_node", "tritonswmm", ["depth"], scope_dir=scen)
    )
    tb = threading.Thread(
        name="B",
        target=lambda: (gate.wait(5), sel.prune_timeseries_store(sb, "swmm_node", "tritonswmm", ["depth"]), done.set()),
    )
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    assert all(s == scen for s in seen["A"]) and all(s is None for s in seen["B"])


def test_an_interrupted_scoped_prune_does_not_leave_an_index_listing_a_deleted_array(tmp_path, monkeypatch):
    """Property: after a prune interrupted between two array deletes, an indexed read of the whole store loads without
    error (the index lists no array that is gone), as it does for the unscoped zarr-API route."""
    import hhemt.du_sentinels as du
    import hhemt.timeseries_selection as sel

    scen = tmp_path / "scen"
    scen.mkdir()

    def interrupted(scope):
        store = _store(tmp_path / f"s_{scope is not None}.zarr")
        real = du.delete_and_account
        n = []
        real_del = sel._delete_zarr_array

        def failing_scoped(paths, **kw):
            n.append(1)
            if len(n) == 2:
                raise OSError("killed")
            return real(paths, **kw)

        def failing_unscoped(group, name):
            n.append(1)
            if len(n) == 2:
                raise OSError("killed")
            real_del(group, name)

        with monkeypatch.context() as m:
            if scope is not None:
                m.setattr(du, "delete_and_account", failing_scoped)
            else:
                m.setattr(sel, "_delete_zarr_array", failing_unscoped)
            with pytest.raises(OSError):
                sel.prune_timeseries_store(store, "swmm_node", "tritonswmm", ["depth"], scope_dir=scope)
        return store

    def listed_but_absent(store):
        listed = set(xr.open_zarr(store, consolidated=True, chunks=None).data_vars)
        on_disk = {d.name for d in Path(store).iterdir() if d.is_dir() and (d / "zarr.json").is_file()}
        return sorted(listed - on_disk - {"node_id", "date_time"})

    unscoped = interrupted(None)
    assert listed_but_absent(unscoped) == [], "control: the zarr-API route keeps the index current"
    scoped = interrupted(scen)
    assert listed_but_absent(scoped) == [], listed_but_absent(scoped)
