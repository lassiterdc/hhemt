#!/usr/bin/env python3
"""Trim the observed-event result of the Norfolk case study to the Hurricane Irene event.

The tutorial notebook `docs/tutorials/interactive-python.ipynb` loads the store this script writes, and the
notebook is the only reader of the store.

Usage::

    python scripts/trim_irene_example_result.py SOURCE DESTINATION

`SOURCE` is the consolidated output store of the observed-event analysis (`analysis_datatree.zarr`).
`DESTINATION` is the store to write, `docs/tutorials/assets/irene-example-result.zarr` in the repository.

The script selects the event by its `event_id`, keeps only the variables that the notebook reads in the four
groups that the notebook reads, writes a Zarr DataTree with a provenance attribute on its root, and prints the
size of the store in bytes. The source store carries no consolidated metadata, so every group is opened with
`consolidated=False`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import xarray as xr

IRENE_EVENT_ID = "year.9_event_type.compound_event_id.1"
SOURCE_ANALYSIS_ID = "norfolk_doc_observed_events"
SHA_VARIABLE = "hhemt_producing_sha"

# Group path -> the variables the notebook reads from it. The two difference plots read `max_flow` of
# both SWMM link groups and `max_flood_depth` of both TRITON groups.
KEPT_VARIABLES: dict[str, tuple[str, ...]] = {
    "tritonswmm/swmm_link": ("max_flow",),
    "swmm_only/swmm_link": ("max_flow",),
    "tritonswmm/triton": ("max_flood_depth",),
    "triton_only/triton": ("max_flood_depth",),
}
# Store layouts differ between group types in which dimension names a variable carries; the notebook reads
# these dimensions only.
EVENT_DIM = "event_iloc"
CAP_BYTES = 5_000_000


def store_bytes(path: Path) -> int:
    """Sum of the sizes in bytes of the files under `path`."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def trim_group(source: Path, group: str, variables: tuple[str, ...]) -> tuple[xr.Dataset, set[str]]:
    """The Irene event of one group, reduced to `variables`, and the producing shas the group records.

    A group that records no producing sha returns an empty set; the caller requires one sha across the groups.
    """
    ds = xr.open_zarr(source, group=group, consolidated=False)
    ids = [str(v) for v in ds["event_id"].values]
    if IRENE_EVENT_ID not in ids:
        raise SystemExit(f"{group} holds no event {IRENE_EVENT_ID}; it holds {ids}")
    position = ids.index(IRENE_EVENT_ID)
    shas: set[str] = set()
    if SHA_VARIABLE in ds.variables:
        shas = {str(v) for v in ds[SHA_VARIABLE].isel({EVENT_DIM: [position]}).values}
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise SystemExit(f"{group} lacks the variables {missing}")
    keep = ds[list(variables)].isel({EVENT_DIM: [position]})
    # The notebook labels by position along these dimensions, so each one needs the coordinate that names its members.
    unlabelled = [d for d in keep.dims if d != EVENT_DIM and d not in keep.coords]
    if unlabelled:
        raise SystemExit(f"{group} has no coordinate for the dimensions {unlabelled}; the source store is incomplete")
    # Keep the labels the notebook reads and drop the other coordinates, including the producing version string.
    keep = keep.drop_vars([c for c in keep.coords if c not in {EVENT_DIM, "event_id", *keep.dims}])
    # Load the values so the source store can close, then drop the encoding of the source so the destination
    # chooses its own chunks and compressors.
    keep = keep.load()
    for name in list(keep.variables):
        keep[name].encoding = {}
        # Fixed-length unicode has no Zarr v3 specification; a variable-length string has one.
        if keep[name].dtype.kind == "U":
            keep[name] = keep[name].astype(object)
    return keep, shas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the observed-event analysis_datatree.zarr store")
    parser.add_argument("destination", type=Path, help="the trimmed store to write")
    args = parser.parse_args(argv)

    groups: dict[str, xr.Dataset] = {}
    shas: set[str] = set()
    for group, variables in KEPT_VARIABLES.items():
        groups[group], group_shas = trim_group(args.source, group, variables)
        shas |= group_shas
    if len(shas) != 1:
        raise SystemExit(
            f"the groups record {len(shas)} producing shas for {IRENE_EVENT_ID}, expected 1: {sorted(shas)}"
        )

    provenance = {
        "source_analysis_id": SOURCE_ANALYSIS_ID,
        "event_id": IRENE_EVENT_ID,
        "variables": {group: list(variables) for group, variables in KEPT_VARIABLES.items()},
        "hhemt_producing_sha": shas.pop(),
    }
    tree = xr.DataTree.from_dict({f"/{group}": ds for group, ds in groups.items()})
    tree.attrs["provenance"] = json.dumps(provenance, sort_keys=True)

    if args.destination.exists():
        shutil.rmtree(args.destination)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    tree.to_zarr(args.destination, mode="w", consolidated=False)

    size = store_bytes(args.destination)
    print(f"wrote {args.destination.name}: {size} bytes")
    if size >= CAP_BYTES:
        print(f"the store is {size} bytes, not under the cap of {CAP_BYTES} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
