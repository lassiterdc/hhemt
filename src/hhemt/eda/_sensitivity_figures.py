"""Figure builders for the three compute-sensitivity EDA plots (ADR-9 members 2-4).

Sibling of ``_config_diff.py``: each ``build_*_figure(root)`` returns a plotly
``go.Figure`` for one sensitivity plot, and ``_plotting.py``'s ``_render_*`` route
through them exactly as ``_render_config_diff_maps`` routes through
``_config_diff.build_config_diff_figure``.

THE FIGURE BODIES ARE STUBS. A subsequent interactive ``/eda-spinup`` pass designs the
real panels (reading ``{root}/eda/{plot_id}.zarr``, guaranteed present by
``render_eda_plots``' callee-side absence gate); until then each returns a titled empty
figure so the emit/source-declaration path is exercised end-to-end WITHOUT authoring
figure content. ``sensitivity_source_paths`` declares BOTH the ``{plot_id}.zarr`` artifact
AND the ``{plot_id}.verdict.json`` so the render bundle carries the verdict Phase 5 reads
(SOURCE-DECLARATION contract: the bundle file set is EXACTLY the union of manifest-declared
source_paths -- the ``bundle file set is computed from manifest harvest`` stipulation;
declaring only the zarr leaves the verdict out of every bundle and Phase 5's
``_write_combined_intercomparison`` renders empty while every live-analysis_dir test passes).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def sensitivity_source_paths(root: Path, plot_id: str) -> list[Path]:
    """Declared source_paths for a sensitivity EDA plot: BOTH the calc member's
    ``eda/{plot_id}.zarr`` artifact AND its ``eda/{plot_id}.verdict.json``. A ``.json``
    file passes ``_validate_source_path`` (precedent: disk_utilization declares
    ``_status/_du.json``); the ``.zarr`` passes as a zarr store."""
    return [
        root / "eda" / f"{plot_id}.zarr",
        root / "eda" / f"{plot_id}.verdict.json",
    ]


def _pending_figure(plot_id: str) -> go.Figure:
    """A titled empty figure -- the non-crashing ``/eda-spinup`` seam. Replaced per-plot
    by the interactive design pass; until then it produces valid Plotly HTML with no
    panels and reads no artifact."""
    fig = go.Figure()
    fig.update_layout(title=f"{plot_id} (figure pending /eda-spinup design)")
    return fig


def build_rank_sensitivity_figure(root: Path) -> go.Figure:
    """STUB (/eda-spinup seam): within-family rank-N vs rank-1 byte-identity + magnitude
    panels. The design pass fills this to read ``{root}/eda/eda_rank_sensitivity.zarr``."""
    return _pending_figure("eda_rank_sensitivity")


def build_resume_sensitivity_figure(root: Path) -> go.Figure:
    """STUB (/eda-spinup seam): clean-vs-resume byte-identity + magnitude panels. The
    design pass fills this to read ``{root}/eda/eda_resume_sensitivity.zarr``."""
    return _pending_figure("eda_resume_sensitivity")


def build_cross_hardware_magnitude_figure(root: Path) -> go.Figure:
    """STUB (/eda-spinup seam): ADR-4 1-GPU vs 1-rank serial-CPU characterized-divergence
    panels. The design pass fills this to read
    ``{root}/eda/eda_cross_hardware_magnitude.zarr``."""
    return _pending_figure("eda_cross_hardware_magnitude")
