#!/usr/bin/env python
"""Byte-identity verification for a compute-config sensitivity master (D5 ground truth).

Compares each sub-analysis's FLAT per-scenario summaries (max_wlevel_m over (y,x);
max_flow_cms over (link_id,)) against a reference sub, for one shared event, using:
  * COMPLIANT  = xr.align(join="exact") + transpose + dtype gate + np.array_equal(equal_nan=True)
  * POSITIONAL = bare np.array_equal on raw .values (mimics eda/_config_diff.py)
If COMPLIANT says identical but POSITIONAL says differs -> reporting-calc ARTIFACT
(dim/coord misalignment). If both say differs -> the divergence is REAL (see FQ3).

Read-only; never constructs TRITONSWMM_scenario. Usage:
  python verify_byte_identity.py /path/to/sensitivity_master \
      [--serial-sa sa_0] [--event-id <sims-subdir-name>] [--out-type zarr|nc]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Full tracked-variable set (mirrors cross_sim_identity.TRACKED_VARS, so the harness's
# AGREE/DISAGREE verdict is comparable to the persisted compliant verdict it is being
# cross-checked against -- a divergence confined to a ratio variable would otherwise be
# invisible to the branch decision, and the SUMMARY line IS the branch selector).
_VARS = {
    "TRITONSWMM_TRITON_summary": ("max_wlevel_m",),
    "TRITONSWMM_SWMM_link_summary": ("max_flow_cms", "max_full_flow_ratio", "max_full_depth_ratio"),
}


def _engine(out_type: str) -> str:
    return "zarr" if out_type == "zarr" else "h5netcdf"


def _open(path: Path, out_type: str) -> xr.Dataset:
    kw = {"engine": _engine(out_type), "chunks": None, "decode_timedelta": False}
    if out_type == "zarr":
        kw["consolidated"] = False
    return xr.open_dataset(path, **kw)


def _sel_event(da: xr.DataArray) -> xr.DataArray:
    return da.isel(event_iloc=0) if "event_iloc" in da.dims else da


def _events(da: xr.DataArray) -> list:
    """Every event_iloc coordinate value present on the DataArray, or [None] when the
    variable carries no event dimension (a scalar-per-event summary)."""
    return list(da["event_iloc"].values) if "event_iloc" in da.dims else [None]


def _at_event(da: xr.DataArray, e) -> xr.DataArray:
    return da.sel(event_iloc=e) if e is not None and "event_iloc" in da.dims else da


def _label(sub_dir: Path) -> str:
    """Best-effort compute-config label from a materialized cfg_analysis.yaml; else sa_id."""
    sa = sub_dir.name
    try:
        import yaml

        for cand in (sub_dir / "cfg_analysis.yaml", sub_dir / "config" / "cfg_analysis.yaml"):
            if cand.exists():
                c = yaml.safe_load(cand.read_text()) or {}
                rm, nm, no = c.get("run_mode", "?"), c.get("n_mpi_procs", 1), c.get("n_omp_threads", 1)
                return f"{sa} [{rm} {nm}r x {no}t]"
    except Exception:
        pass
    return sa


def _detect_out_type(master: Path) -> str:
    if next(master.glob("subanalyses/*/sims/*/processed/TRITONSWMM_TRITON_summary.zarr"), None):
        return "zarr"
    return "nc"


def _summary_path(sub_dir: Path, event_id: str, stem: str, out_type: str) -> Path | None:
    p = sub_dir / "sims" / event_id / "processed" / f"{stem}.{out_type}"
    return p if p.exists() else None


def _three_guard(a: xr.DataArray, b: xr.DataArray) -> tuple[bool, bool, bool, float]:
    """(identical, dtype_match, coord_ok, max_abs_diff) — the compliant predicate."""
    try:
        a_al, b_al = xr.align(a, b, join="exact")
    except (ValueError, KeyError):
        return (False, a.dtype == b.dtype, False, float("nan"))
    b_al = b_al.transpose(*a_al.dims)
    av, bv = a_al.values, b_al.values
    dtype_match = av.dtype == bv.dtype
    identical = bool(np.array_equal(av, bv, equal_nan=True)) and dtype_match
    with np.errstate(invalid="ignore"):
        d = np.abs(av.astype("float64") - bv.astype("float64"))
    fin = d[np.isfinite(d)]
    return (identical, dtype_match, True, float(fin.max()) if fin.size else 0.0)


def _positional(a: xr.DataArray, b: xr.DataArray) -> bool:
    """The eda/_config_diff.py comparison: bare np.array_equal on raw .values, no align."""
    av, bv = np.asarray(a.values), np.asarray(b.values)
    if av.shape != bv.shape:
        return False
    return bool(np.array_equal(av, bv, equal_nan=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("--serial-sa", default=None, help="reference sa_id dir name (default: first)")
    ap.add_argument("--event-id", default=None, help="sims/ subdir name (default: first shared)")
    ap.add_argument("--out-type", default=None, choices=["zarr", "nc"])
    args = ap.parse_args()

    master = args.master
    out_type = args.out_type or _detect_out_type(master)
    subs = sorted(d for d in (master / "subanalyses").glob("sa_*") if d.is_dir())
    if not subs:
        print(f"No subanalyses/sa_* under {master}", file=sys.stderr)
        return 2

    ref_dir = next((d for d in subs if d.name == args.serial_sa), subs[0])
    if args.event_id:
        event_id = args.event_id
    else:
        ev_dirs = sorted(p.name for p in (ref_dir / "sims").glob("*") if p.is_dir())
        if not ev_dirs:
            print(f"No sims/ dirs under {ref_dir}", file=sys.stderr)
            return 2
        event_id = ev_dirs[0]

    print(f"master     : {master}")
    print(f"out_type   : {out_type}")
    print(f"reference  : {_label(ref_dir)}   event: {event_id}\n")

    any_disagreement = False
    # Iterate EACH tracked variable within each summary stem (the DoD requires the FULL
    # cross_sim_identity.TRACKED_VARS surface; _VARS keys a stem to a TUPLE of variables,
    # so a single ds[stem-tuple] index would KeyError and under-cover the branch selector).
    for stem, varnames in _VARS.items():
        ref_path = _summary_path(ref_dir, event_id, stem, out_type)
        if ref_path is None:
            print(f"[{stem}] reference summary absent under {ref_dir.name} — skipped\n")
            continue
        ref_ds = _open(ref_path, out_type)
        for var in varnames:
            if var not in ref_ds.data_vars:
                print(f"[{stem}:{var}] absent in reference — skipped")
                continue
            ref_var = ref_ds[var]
            # Cover EVERY event present in the reference (DoD): the branch selector must
            # arbitrate over the same surface as the persisted verdict, not just event 0.
            for e in _events(ref_var):
                ref_da = _at_event(ref_var, e)
                elabel = f"{var}[event_iloc={e}]" if e is not None else var
                print(f"=== {elabel}  (ref={ref_dir.name}) ===")
                print(
                    f"{'sub':<28}{'compliant':<12}{'positional':<12}{'dtype':<8}{'coord':<8}{'max_abs_diff':<14}verdict"
                )
                for sub in subs:
                    if sub == ref_dir:
                        continue
                    sp = _summary_path(sub, event_id, stem, out_type)
                    if sp is None:
                        print(f"{_label(sub):<28}{'absent':<12}")
                        continue
                    sub_ds = _open(sp, out_type)
                    if var not in sub_ds.data_vars:
                        print(f"{_label(sub):<28}{'absent-var':<12}")
                        continue
                    sub_var = sub_ds[var]
                    if e is not None and e not in _events(sub_var):
                        print(f"{_label(sub):<28}{'absent-event':<12}")
                        continue
                    cmp_da = _at_event(sub_var, e)
                    ident, dtok, coordok, mad = _three_guard(ref_da, cmp_da)
                    pos = _positional(ref_da, cmp_da)
                    if ident != pos:
                        any_disagreement = True
                        verdict = "ARTIFACT (align/dtype) -> fix _config_diff.py"
                    elif ident:
                        verdict = "identical"
                    else:
                        verdict = "REAL divergence -> FQ3"
                    print(
                        f"{_label(sub):<28}{str(ident):<12}{str(pos):<12}{str(dtok):<8}"
                        f"{str(coordok):<8}{mad:<14.6g}{verdict}"
                    )
                print()

    print(
        "SUMMARY:",
        "compliant vs positional DISAGREE on >=1 sub -> _config_diff.py ARTIFACT confirmed"
        if any_disagreement
        else "compliant and positional AGREE on every sub -> reported divergence is REAL (see FQ3)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
