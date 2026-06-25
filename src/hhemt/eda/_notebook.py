"""EDA notebook emitter (ADR-13, D-NB1 non-clobbering create-new).

Builds a fresh ``.ipynb`` via ``nbformat`` programmatic cell construction (zero new
dependency; byte-deterministic — no execution-count, no kernel spawn at emit). The
emitter NEVER overwrites an existing notebook (D-NB1): it resolves the first-available
numeric-suffixed sibling. Seed cells carry the ``hhemt-generated`` metadata tag for
provenance ONLY (D-NB1 superseded the two-zone merge — no existing-notebook parse).

Seed-figure cells are SOURCE-CODE cells that call the installed-hhemt
``load_eda_context`` + ``render_eda_plots`` free functions at EXECUTION (ADR-14: the
interactivity gain is computational — re-plot against live re-derived variables), so
there is exactly ONE figure-emit path. The calc cell is gated on ``ctx.is_bundle``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

if TYPE_CHECKING:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.eda import eda_config

#: Cell-metadata tag stamped on every toolkit-seeded cell (provenance only under
#: D-NB1 — drives no merge). A future PR-flow tool can distinguish seeded vs
#: user-authored cells by this tag.
_SEED_TAG = "hhemt-generated"


def _tagged(cell):
    """Stamp the hhemt-generated provenance tag onto a seed cell and return it."""
    cell.metadata.setdefault("tags", []).append(_SEED_TAG)
    return cell


def _resolve_notebook_path(root: Path, notebook_filename: str | None) -> Path:
    """Resolve the first-available non-clobbering notebook path under ``root``.

    ``None`` -> base name ``eda``. A user-passed value is normalized: ``.ipynb`` is
    appended iff absent (case-insensitive). If the resolved target exists, the first
    free numeric-suffixed sibling is returned (``eda.ipynb`` -> ``eda_1.ipynb`` ->
    ``eda_2.ipynb`` ...). Deterministic; no timestamp.
    """
    base = notebook_filename if notebook_filename is not None else "eda"
    if not base.lower().endswith(".ipynb"):
        base = f"{base}.ipynb"
    stem = base[: -len(".ipynb")]
    candidate = root / f"{stem}.ipynb"
    n = 1
    while candidate.exists():
        candidate = root / f"{stem}_{n}.ipynb"
        n += 1
    return candidate


def emit_eda_notebook(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
    is_bundle: bool,
    notebook_filename: str | None = None,
) -> Path:
    """Emit a fresh seeded EDA notebook under ``root`` (non-clobbering). Returns the path.

    ``is_bundle`` gates the ADR-9 byte-identity calc seed cell (a bundle carries no
    source per-scenario summaries). ``eda_cfg.enabled_plots`` drives the seed-figure
    cells (no parallel notebook-config surface).
    """
    cells = []

    cells.append(
        _tagged(
            new_markdown_cell(
                f"# EDA — {cfg_analysis.analysis_id}\n\n"
                "This notebook is **toolkit-seeded** and explore-ready. Seed cells are "
                "regenerated on every `eda()` run; cells you add are yours to keep. "
                "Run the loader cell first to bind the experiment variables, then "
                "edit/add cells freely — your EDA never modifies git-tracked `hhemt` source."
            )
        )
    )

    cells.append(
        _tagged(
            new_code_cell(
                "from pathlib import Path\n"
                "from hhemt.eda import load_eda_context, render_eda_plots\n"
                "root = Path.cwd()\n"
                "ctx = load_eda_context(root)\n"
                "ctx.is_bundle, ctx.cfg_analysis.analysis_id"
            )
        )
    )

    cells.append(
        _tagged(
            new_markdown_cell(
                "## Variable inventory\n\n"
                "`ctx` exposes: `datatree`, `sensitivity_datatree`, `cfg_analysis`, "
                "`cfg_system`, `scenario_status`, `swmm_features`, `triton_dem`, "
                "`performance`, `is_bundle`. Fields are `None` when the experiment "
                "shape or this root does not carry that artifact."
            )
        )
    )

    cells.append(
        _tagged(new_code_cell("{k: (type(v).__name__ if v is not None else None) for k, v in vars(ctx).items()}"))
    )

    # Seed-figure cells: one per enabled plot. Source-code cells calling the SAME
    # render_eda_plots free function the facade uses (one figure-emit path). Figures
    # render at EXECUTION against the loader's re-derived variables (ADR-14).
    cells.append(
        _tagged(
            new_code_cell(
                "# Seed figures: re-rendered on execution from the carried EDA datasets and\n"
                "# DISPLAYED INLINE (the notebook is the primary explore surface, ADR-14).\n"
                "# Guarded on the EDA data-prep zarr(s) being present at root (absent on a\n"
                "# figureless non-sensitivity analysis or a bundle whose eda/ zarr was not\n"
                "# harvested) so this cell never errors.\n"
                "from hhemt.eda import cross_sim_identity_figure_from_root\n"
                "_eda_dir = root / 'eda'\n"
                "_figs = (\n"
                "    [cross_sim_identity_figure_from_root(root)]\n"
                "    if _eda_dir.is_dir() and any(_eda_dir.glob('*.zarr'))\n"
                "    else []\n"
                ")\n"
                "# also persist the standalone plots/eda/*.html artifacts (re-render to disk)\n"
                "plot_paths = (\n"
                "    render_eda_plots(root, cfg_analysis=ctx.cfg_analysis, eda_cfg=ctx.cfg_analysis.eda)\n"
                "    if _figs\n"
                "    else []\n"
                ")\n"
                "for _f in _figs:\n"
                "    _f.show()\n"
                "_figs[0] if _figs else 'no EDA figures for this experiment shape'"
            )
        )
    )

    # ADR-9 byte-identity calc cell — gated on a source root (a bundle carries no
    # source per-scenario summaries, so the calc cannot run).
    if not is_bundle:
        cells.append(
            _tagged(
                new_markdown_cell(
                    "## Cross-sim byte-identity (source-analysis only)\n\n"
                    "Available only on a live analysis directory (a transported bundle "
                    "carries no source per-scenario summaries)."
                )
            )
        )

    nb = new_notebook(cells=cells)
    nbformat.validate(nb)

    out_path = _resolve_notebook_path(root, notebook_filename)
    nbformat.write(nb, out_path)
    return out_path
