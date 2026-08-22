"""Figure builders for the three compute-sensitivity EDA plots (ADR-9 members 2-4).

Sibling of ``_config_diff.py``: each ``build_*_figure(root)`` returns a plotly
``go.Figure`` for one sensitivity plot, and ``_plotting.py``'s ``_render_*`` route
through them exactly as ``_render_config_diff_maps`` routes through
``_config_diff.build_config_diff_figure``.

THE FIGURE BODIES ARE STUBS. A subsequent interactive ``/eda-spinup`` pass designs the
real panels (reading ``{root}/eda/{plot_id}.zarr``, guaranteed present by
``render_eda_plots``' callee-side absence gate); until then each returns a titled empty
figure so the emit/source-declaration path is exercised end-to-end WITHOUT authoring
figure content. ``sensitivity_source_paths`` declares BOTH the ``{plot_id}.zarr`` artifact AND the
``{plot_id}.verdict.json`` (SOURCE-DECLARATION contract: the bundle file set is EXACTLY the
union of manifest-declared source_paths -- the ``bundle file set is computed from manifest
harvest`` stipulation), so a render bundle carries the verdict as a general robustness backstop.
Combine-first note: the per-arm ``eda_resume_sensitivity`` figure is a general OPT-IN per-master
capability (removed from the default ``enabled_plots``) that renders only for a single master
carrying BOTH a clean and a resume arm; on a single-arm master it SKIPS and writes no
verdict/zarr. The combine-level clean-vs-resume comparison (``cross_experiment_intercomparison``)
does NOT read this member's verdict -- it derives its data CROSS-BUNDLE from the two bundles'
paired per-config key-result summaries via ``compare_variable_exact`` (see
``bundle/_combine._write_combined_intercomparison``).
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


#: Figure stems whose builder above still returns ``_pending_figure`` — a titled empty
#: figure with no panels. The combine catch-all glob
#: (bundle/combined_snakefile_generator.py::_figures_of) skips these so a ~10 MB
#: Plotly-bundle stub never ships in the combined report as an "Other"-category figure.
#:
#: DELETE A STEM FROM THIS SET IN THE SAME EDIT that replaces its ``_pending_figure``
#: return above. A stale entry here is a SILENT reverse-direction failure: the designed
#: figure renders correctly in the per-arm eda_report.html and is suppressed only in the
#: combined report, where nothing signals the omission.
#:
#: This is deliberately NOT ``config.eda._EDA_DROPS`` and NOT
#: ``config.eda._RETIRED_EDA_FIGURE_STEMS``. These figures are NOT-YET-DESIGNED, not
#: RETIRED. Measured: adding a stem to ``_EDA_DROPS`` STRIPS it from ``enabled_plots`` at
#: config load and warns that it was "dropped (F8)", destroying the opt-in that
#: config/eda.py documents; adding it to ``_RETIRED_EDA_FIGURE_STEMS`` alone leaves the
#: opt-in intact but emits a DeprecationWarning whose renamed and dropped lists are BOTH
#: empty, on every config load that opts in.
_PENDING_EDA_FIGURE_STEMS: frozenset[str] = frozenset(
    {
        "eda_rank_sensitivity",
        "eda_resume_sensitivity",
        "eda_cross_hardware_magnitude",
    }
)
