"""In-report EDA adapter (R11): renders the compute-sensitivity EDA figure
family into ``analysis_report.html`` as config-selectable tabs.

This is the in-report routing adapter for the Phase-3 EDA free-functions. It
exposes the uniform ``render(analysis, report_cfg, output_path, **kwargs)``
signature (per the renderer-uniform-signature stipulation) and DELEGATES the
actual figure emission to ``eda/_plotting.render_eda_plots``. Those free
functions emit under master-rooted ``plots/eda/<plot_id>.html`` and declare each
figure's data source (its ``eda/<plot_id>.zarr`` provenance artifact, or the
consolidated ``sensitivity_datatree.zarr`` it reads directly) via
``emit_plot_with_sources`` — so this adapter passes the renderer-IO provenance
audit (Gotcha 53) WITHOUT re-declaring sources here.

The Snakemake rule that shells out to ``_cli eda_compute_sensitivity`` is
per-figure: ``output_path`` is the ONE figure this invocation must produce. We
therefore scope the render to exactly the ``enabled_plots`` entry whose canonical
plot ID equals ``output_path``'s stem, so a multi-member EDA family (rank /
resume / magnitude, future) yields one rule per figure rather than re-rendering
the whole family per rule.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.config.eda import eda_config
from hhemt.eda._plotting import render_eda_plots
from hhemt.report_plot_ids import canonical_plot_id


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    root = Path(analysis.analysis_paths.analysis_dir)
    eda_cfg = analysis.cfg_analysis.eda

    target_stem = Path(output_path).stem
    wanted = [kind for kind in eda_cfg.enabled_plots if canonical_plot_id(kind) == target_stem]
    if not wanted:
        raise ValueError(
            f"eda_compute_sensitivity: no enabled EDA plot maps to output stem "
            f"{target_stem!r}; enabled_plots={list(eda_cfg.enabled_plots)!r}"
        )

    # Scope the render to just this rule's figure (model_validate re-fires the
    # eda_config validators per the per-row-overlay stipulation; model_copy is
    # forbidden for config models).
    scoped_cfg = eda_config.model_validate({**eda_cfg.model_dump(), "enabled_plots": wanted})
    render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=scoped_cfg)
