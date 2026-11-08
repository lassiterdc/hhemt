"""Which time-series variables a run keeps, and the prune that removes the rest from a per-scenario store.

A coupled run writes per-scenario time-series stores for the SWMM nodes, the SWMM links and the gridded TRITON
surface. Each is large, and a downstream user rarely wants every variable. ``analysis_config`` carries one optional
keep-field per kind (``timeseries_keep_swmm_node``, ``timeseries_keep_swmm_link``,
``timeseries_keep_triton_gridded``): ``None`` keeps everything, a string names a toolkit set (``"all"`` or
``"default"``), and a list names variables. This module holds the single table of names those fields select from,
the two toolkit sets, ``resolve_keep``, and the prune.

The domain is a CLOSED table, evaluated from the writers. Regenerate it with::

    grep -n "_OUT_NODE_SERIES\\|_OUT_LINK_SERIES" src/hhemt/swmm_output_parser.py     # SWMM node and link series
    grep -n "TRITON_RAW_VAR_BY_PREFIX" src/hhemt/process_simulation.py                # gridded series

plus ``coupling_exchange_flow``, which the coupled node store gains when ``toggle_coupling_exchange_flow`` is on, and
the two pollutant series a water-quality model adds.

``"default"`` is family-conditional. Gridded: every variable except ``max_flood_depth``, ``unit_discharge_x`` and
``unit_discharge_y`` (their peaks live in the summary). SWMM node of the coupled family: every variable except
``flooding``, which in the coupled build is the growth rate of volume stored above a node's full volume and not water
lost from the network. SWMM node of the standalone family, and SWMM link of every family: ``"default"`` equals
``"all"``. The series ``volume``, ``lateral_inflow`` and ``link_volume`` are in ``"default"`` for every family.

Variables WITHOUT the kind's time dimension are static (``is_coupling_node``, the per-node statistics, the run-level
variables) and are outside the selection: never pruned, whatever a keep-list says. The property, not a list, defines
them, so a static variable a later change adds is covered without an edit here.

The prune is applied after the scenario's summary store is written, because the summaries need every series, and it
never removes a variable outside this table. It is restartable: the pruned names are recorded in the store's
``timeseries_pruned_variables`` attribute BEFORE any array is deleted, so an interrupted prune is recognisable and a
re-run completes it, and a summary rebuilt from a pruned store is refused (``ProcessingError``) instead of silently
losing the statistics of the removed series.
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.utils import consolidate_zarr_store, write_netcdf, zarr_store_has_index

__all__ = [
    "DOMAIN",
    "FAMILIES_BY_KIND",
    "KINDS",
    "PRUNED_ATTR",
    "SET_NAMES",
    "TIME_DIM_BY_KIND",
    "keep_setting",
    "prune_timeseries_store",
    "pruned_variables",
    "resolve_keep",
]

Kind = Literal["swmm_node", "swmm_link", "triton_gridded"]
Family = Literal["tritonswmm", "swmm_only", "triton_only"]

KINDS: tuple[str, ...] = ("swmm_node", "swmm_link", "triton_gridded")
SET_NAMES: tuple[str, ...] = ("all", "default")

#: The store attribute that records which variables a prune removed.
PRUNED_ATTR = "timeseries_pruned_variables"

#: The dimension whose presence makes a variable time-variant, per kind.
TIME_DIM_BY_KIND: dict[str, str] = {"swmm_node": "date_time", "swmm_link": "date_time", "triton_gridded": "timestep"}

#: The model families that write a store of each kind.
FAMILIES_BY_KIND: dict[str, tuple[str, ...]] = {
    "swmm_node": ("tritonswmm", "swmm_only"),
    "swmm_link": ("tritonswmm", "swmm_only"),
    "triton_gridded": ("tritonswmm", "triton_only"),
}

_NODE_COMMON = ("depth", "head", "inflow_flow", "flooding", "volume", "lateral_inflow", "pollutant_conc")
_LINK_COMMON = ("flow", "velocity", "link_depth", "link_volume", "link_pollutant_conc")
_GRIDDED = ("flood_depth", "max_flood_depth", "unit_discharge_x", "unit_discharge_y")

#: (kind, family) -> every selectable variable, in a fixed order.
DOMAIN: dict[tuple[str, str], tuple[str, ...]] = {
    ("swmm_node", "tritonswmm"): (*_NODE_COMMON, "coupling_exchange_flow"),
    ("swmm_node", "swmm_only"): _NODE_COMMON,
    ("swmm_link", "tritonswmm"): (*_LINK_COMMON, "capacity_setting"),
    ("swmm_link", "swmm_only"): (*_LINK_COMMON, "capacity"),
    ("triton_gridded", "tritonswmm"): _GRIDDED,
    ("triton_gridded", "triton_only"): _GRIDDED,
}

_DEFAULT_EXCLUDED: dict[tuple[str, str], tuple[str, ...]] = {
    ("swmm_node", "tritonswmm"): ("flooding",),
    ("triton_gridded", "tritonswmm"): ("max_flood_depth", "unit_discharge_x", "unit_discharge_y"),
    ("triton_gridded", "triton_only"): ("max_flood_depth", "unit_discharge_x", "unit_discharge_y"),
}


def _domain(kind: str, family: str) -> tuple[str, ...]:
    """The domain of ``(kind, family)``, refusing a pair no model writes."""
    try:
        return DOMAIN[(kind, family)]
    except KeyError:
        raise ValueError(
            f"no time-series store of kind {kind!r} exists for model family {family!r}; "
            f"families by kind: {FAMILIES_BY_KIND}"
        ) from None


def keep_setting(cfg, kind: str):
    """The keep-field of one kind read from an analysis config, or None when the config has no such field.

    The three fields are named literally here so that a search for a field name finds the reads. The None default
    serves config stand-ins that do not carry the fields.
    """
    if kind == "swmm_node":
        return getattr(cfg, "timeseries_keep_swmm_node", None)
    if kind == "swmm_link":
        return getattr(cfg, "timeseries_keep_swmm_link", None)
    if kind == "triton_gridded":
        return getattr(cfg, "timeseries_keep_triton_gridded", None)
    raise KeyError(f"unknown time-series kind {kind!r}; the kinds are {sorted(FAMILIES_BY_KIND)}")


def _blank_to_none(value):
    """Map a blank keep-field to None: an empty or whitespace-only string, or an empty list, means keep everything."""
    if isinstance(value, str) and not value.strip():
        return None
    if value is not None and not isinstance(value, str) and len(list(value)) == 0:
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def resolve_keep(kind: str, family: str, value: str | Sequence[str] | None) -> tuple[str, ...]:
    """The variable names a keep-field selects for one kind and model family.

    Parameters
    ----------
    kind : {"swmm_node", "swmm_link", "triton_gridded"}
    family : {"tritonswmm", "swmm_only", "triton_only"}
    value : None, str or sequence of str
        ``None``, a blank string, an empty list and ``"all"`` give the whole domain (a blank keep-field
        keeps everything, as the config description and scope S6 state); ``"default"`` gives the
        family-conditional set; a sequence gives its names, in the order written, repeats dropped, each family
        keeping the names it writes.

    Returns
    -------
    tuple of str

    Raises
    ------
    ValueError
        For a set name that is neither ``"all"`` nor ``"default"``, for a listed name outside the domain, or for a
        (kind, family) pair no model writes. Each message names the domain.
    """
    domain = _domain(kind, family)
    value = _blank_to_none(value)
    if value is None or (isinstance(value, str) and value == "all"):
        return domain
    if isinstance(value, str):
        if value == "default":
            excluded = set(_DEFAULT_EXCLUDED.get((kind, family), ()))
            return tuple(v for v in domain if v not in excluded)
        raise ValueError(
            f"{value!r} is not a toolkit set for {kind} ({family}); the sets are {SET_NAMES}, or list variables "
            f"from the domain {domain}"
        )
    names = list(dict.fromkeys(str(v) for v in value))
    # A list is shared by every family that writes the kind, so it is admitted over their UNION (as preflight
    # does) and each family keeps the part of it that family writes.
    union = {v for f in FAMILIES_BY_KIND[kind] for v in DOMAIN[(kind, f)]}
    outside = [v for v in names if v not in union]
    if outside:
        raise ValueError(f"{outside} are outside the {kind} domain for {family}: {domain}")
    return tuple(v for v in names if v in domain)


def pruned_variables(path: Path) -> tuple[str, ...]:
    """The variables recorded as pruned from the store at ``path``; empty when none, or when the store is absent."""
    path = Path(path)
    if not path.exists():
        return ()
    if path.suffix == ".zarr":
        import zarr

        value = dict(zarr.open_group(str(path), mode="r").attrs).get(PRUNED_ATTR)
    else:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            value = ds.attrs.get(PRUNED_ATTR)
    if value is None:
        return ()
    return tuple(str(v) for v in np.atleast_1d(value))


def _to_prune(ds: xr.Dataset, kind: str, family: str, keep: Sequence[str]) -> list[str]:
    """Time-variant variables of ``ds`` that are in the domain and not in ``keep``."""
    time_dim = TIME_DIM_BY_KIND[kind]
    domain = set(_domain(kind, family))
    return [v for v in ds.data_vars if time_dim in ds[v].dims and v in domain and v not in set(keep)]


#: The scope directory (a scenario folder) whose disk-use sentinel a prune's deletions are accounted against, set
#: for the duration of one ``prune_timeseries_store`` call. A context variable rather than a parameter so that
#: ``_delete_zarr_array`` keeps its two-argument form, which callers patch to interrupt a prune.
_ACCOUNT_SCOPE: contextvars.ContextVar[Path | None] = contextvars.ContextVar("timeseries_prune_scope", default=None)


def _delete_zarr_array(group, name: str) -> None:
    """Delete one array from an open zarr group (its directory and chunk files go with it).

    With an accounting scope set, the array directory is removed through ``du_sentinels.delete_and_account``, the
    toolkit's one deletion tool, so the scenario's disk-use sentinel loses exactly the bytes removed; without a scope
    (a store outside any scenario) the zarr API deletes it. The consolidated index is rebuilt by the caller.
    """
    scope = _ACCOUNT_SCOPE.get()
    if scope is None:
        del group[name]
        return
    from hhemt.du_sentinels import delete_and_account

    delete_and_account([Path(group.store.root) / group.path / name], scope_dir=scope, scope="scenario")


def prune_timeseries_store(
    path: Path, kind: str, family: str, keep: Sequence[str], *, scope_dir: Path | None = None
) -> tuple[str, ...]:
    """Remove from the store at ``path`` the time-variant domain variables that ``keep`` does not name.

    Parameters
    ----------
    path : pathlib.Path
        A per-scenario time-series store, ``*.zarr`` or NetCDF.
    kind, family : str
        As for ``resolve_keep``.
    keep : sequence of str
        The resolved keep-list (``resolve_keep`` output).
    scope_dir : pathlib.Path, optional
        The scenario folder that owns the store. When given, the zarr array deletions are accounted against its
        ``_status/_du.json`` through ``du_sentinels.delete_and_account``. A NetCDF prune is a replace that shrinks the
        file, which no deletion tool accounts; the scenario's next ``consolidate_scenario`` re-derives the total, so
        until then the sentinel reads high, the safe direction.

    Returns
    -------
    tuple of str
        The names removed by THIS call (empty when nothing remained to remove, which is also how a completed prune
        reads on a second call).

    Notes
    -----
    Zarr: ``timeseries_pruned_variables`` is written FIRST (the union with any earlier value), then the arrays are
    deleted, then the consolidated index is rebuilt, so an interrupted prune is recognisable from the attribute and
    re-running it finishes the job and ends in the same state as an uninterrupted one. NetCDF: the kept variables are
    written to a temporary sibling file that replaces the original with ``os.replace``, because ``write_netcdf`` has
    no temp-and-rename and a crash mid-write would leave a partial file where the only copy of the series was.
    """
    path = Path(path)
    if not path.exists():
        raise ProcessingError("prune a time-series store", path, "the store does not exist")
    if path.suffix == ".zarr":
        token = _ACCOUNT_SCOPE.set(Path(scope_dir) if scope_dir is not None else None)
        try:
            return _prune_zarr(path, kind, family, keep)
        finally:
            _ACCOUNT_SCOPE.reset(token)
    return _prune_netcdf(path, kind, family, keep)


def _prune_zarr(path: Path, kind: str, family: str, keep: Sequence[str]) -> tuple[str, ...]:
    import zarr

    indexed = zarr_store_has_index(path)
    with xr.open_zarr(path, consolidated=indexed, chunks=None) as ds:
        doomed = _to_prune(ds, kind, family, keep)
    if not doomed:
        return ()
    group = zarr.open_group(str(path), mode="r+")
    # An earlier interrupted prune may have recorded a name this call's list keeps; a name the store still holds
    # and is not about to delete must not be recorded as pruned.
    still_present = set(group.array_keys()) - set(doomed)
    recorded = sorted((set(pruned_variables(path)) | set(doomed)) - still_present)
    group.attrs[PRUNED_ATTR] = recorded
    try:
        for name in doomed:
            _delete_zarr_array(group, name)
    finally:
        if indexed:
            consolidate_zarr_store(path)
    return tuple(doomed)


def _prune_netcdf(path: Path, kind: str, family: str, keep: Sequence[str]) -> tuple[str, ...]:
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        doomed = _to_prune(opened, kind, family, keep)
        if not doomed:
            return ()
        kept = opened.drop_vars(doomed).load()
    kept.attrs[PRUNED_ATTR] = sorted(set(pruned_variables(path)) | set(doomed))
    temporary = path.with_name(path.name + ".prune.tmp")
    try:
        write_netcdf(kept, temporary)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            # EXEMPT-DU: transient-intermediate
            temporary.unlink()
        raise
    return tuple(doomed)
