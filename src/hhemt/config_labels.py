"""Deterministic labels for a compute configuration.

LEAF MODULE BY DESIGN: it imports nothing from hhemt, so `report_renderers/`,
`eda/` and `report_plot_ids.py` can all read one labeller rather than growing
three. `report_plot_ids.py` is layout-relevant, and before this module existed it
reached these helpers through `eda/_config_diff.py` -- a 1,654-line module with
plotly and xarray at import time.

NAMED FOR WHAT IT LABELS, not for where the hardware token currently comes from.
`_gpu_hardware` reads `hpc.partition` today, and the user has ruled that the
partition must never be DISPLAYED in a figure -- hardware for display is observed
per-simulation and normalized through `hardware_labels.py`. That substitution lands
here, in one file, and this module's name must survive it.
"""

from __future__ import annotations


def _to_int(attrs: dict, key: str) -> int:
    try:
        return int(float(attrs.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _gpu_hardware(attrs: dict) -> str:
    """Hardware token derived from the ensemble partition. ``'gpu-a100-80' ->
    'a100-80'``, ``'gpu-a6000' -> 'a6000'``. Empty when no partition attr is present.

    SUPERSEDED SOURCE: the partition is not the display hardware axis. The observed
    device is parsed per-simulation from the TRITON log into ``df_status.actual_gpu`` /
    ``actual_cpu`` and normalized by ``hardware_labels.hardware_label``; this function
    is the pre-substitution reader and is retained only until that wiring lands."""
    part = str(attrs.get("hpc.partition", "") or "")
    return part[len("gpu-") :] if part.startswith("gpu-") else part


def _derive_config_label(attrs: dict) -> str:
    """Deterministic compute-config label from config attrs (never the member_id name).

    CPU configs use ONE consistent form: ``{Mode} {ranks}r×{threads}t ({total} CPU)`` —
    ``ranks`` = MPI processes, ``threads`` = OpenMP threads PER RANK, ``total`` = ranks ×
    threads. This makes a Hybrid config legible (ranks + threads/rank + total CPUs) while
    keeping Serial/OpenMP/MPI on the same axes: Serial 1r×1t (1 CPU), OpenMP 1r×8t (8 CPU),
    MPI 8r×1t (8 CPU), Hybrid 2r×2t (4 CPU).

    GPU configs are a distinct resource axis (GPUs, not CPUs): ``GPU ×{n} ({hardware})``,
    with hardware from the ensemble partition so an a6000 1-GPU job and an a100 1-GPU job
    are DISTINCT configs. Replicate suffixes (``_r1``/``_r2``) are NOT in the identity, so
    replicates share one label.
    """
    rm = str(attrs.get("run_mode", "?"))
    ng, nm, no, nn = (_to_int(attrs, k) for k in ("n_gpus", "n_mpi_procs", "n_omp_threads", "n_nodes"))
    if rm == "gpu":
        hw = _gpu_hardware(attrs)
        label = f"GPU ×{ng} ({hw})" if hw else f"GPU ×{ng}"
    else:
        name = {"serial": "Serial", "openmp": "OpenMP", "mpi": "MPI", "hybrid": "Hybrid"}.get(rm, rm)
        ranks, threads = max(nm, 1), max(no, 1)
        label = f"{name} {ranks}r×{threads}t ({ranks * threads} CPU)"
    if nn > 1:
        label += f", {nn} nodes"
    return label
