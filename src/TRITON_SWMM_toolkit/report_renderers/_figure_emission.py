"""Shared helpers for report_renderers/*. See R15 in master plan.

Name: `_figure_emission.py` (not `_helpers.py`) per SE F-I Flag 10 —
the module name names what it does rather than what it is.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog


def emit_plot_with_sources(
    fig: plt.Figure,
    output_path: Path,
    source_paths: Iterable[Path],
    analysis_dir: Path,
    dpi: int,
    output_format: Literal["png", "svg"] = "png",
    preview_dpi: int = 100,
    manifest_data: dict[str, Any] | None = None,
    provenance: ProvenanceLog | None = None,
) -> Path:
    """Save fig to output_path with source paths embedded as figure metadata.

    `source_paths` are converted to `analysis_dir`-relative paths before embedding
    in the figure file's metadata (PNG `tEXt` chunks or SVG `<metadata>` element).
    The caller must also populate `params.source_paths` in the calling Snakefile
    rule (with the same list) so caption RST templates can interpolate the sources
    at report-render time via `{{ snakemake.params.source_paths }}`.

    Side effects beyond the primary save:
    - Writes a preview PNG sibling at `<stem>.preview.png` at `preview_dpi` for
      cheap subagent visual review per the v1.4 data-viz review algorithm.
      Emitted for ALL full-res formats (PNG and SVG alike) — the preview is
      always PNG so /design-figure's subagent reads work uniformly regardless
      of full-res format. PNG-permissive metadata ("Source", "Software") is
      always used on the preview.
    - Writes a `<stem>.manifest.json` sibling capturing structural facts
      (paths, file sizes, figure dimensions, source paths, optional
      caller-supplied `manifest_data` describing per-panel content).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_path : Path
        Where to write the figure.
    source_paths : Iterable[Path]
        Absolute paths of data files the renderer read. Converted to relative paths
        against `analysis_dir` before metadata embedding. Duplicates are preserved
        (the caller is responsible for deduplication if desired).
    analysis_dir : Path
        Root of the analysis directory; source paths are emitted relative to this.
    dpi : int
        Figure DPI for the full-resolution raster output.
    output_format : {"png", "svg"}
        Format hint for matplotlib's `savefig`. Determines whether PNG `tEXt`
        metadata or SVG `<metadata>` is used for the full-res file. Preview +
        manifest siblings are emitted regardless of full-res format.
    preview_dpi : int, default 100
        DPI for the preview PNG sibling. The v1.4 data-viz review algorithm
        defaults to 100 dpi as a labels-legible / token-cheap target for the
        Improvement and QC subagents.
    manifest_data : dict, optional
        Caller-supplied structural data for the manifest sibling — typically
        per-panel info (title, axis extents, element counts, legend labels).
        Auto-merged with file-system facts under the `renderer_data` key.
    provenance : ProvenanceLog, optional
        Per-artist provenance log built up during render via
        `with prov.artist(...)` context blocks. When non-None, the serialized
        payload is embedded under `manifest["artists"]` (top-level — system-
        defined, distinct from caller-supplied `renderer_data`). The PNG
        `tEXt` chunks and all other manifest schema fields are unchanged.

    Returns
    -------
    Path
        `output_path` unchanged, for chaining.
    """
    analysis_root = str(analysis_dir.resolve())
    rel_sources = [os.path.relpath(str(Path(p).resolve()), analysis_root) for p in source_paths]
    # PNG accepts arbitrary tEXt keys; SVG metadata is restricted to the
    # Dublin Core element set (matplotlib backend_svg validates against it
    # and ValueErrors on unknown keys). "Source" is in Dublin Core; "Software"
    # is not — for SVG, fold the "Software" intent into "Creator", which is.
    # The preview PNG is always PNG-permissive regardless of full-res format.
    preview_metadata: dict[str, str] = {
        "Source": "; ".join(rel_sources),
        "Software": "TRITON-SWMM_toolkit",
    }
    full_res_metadata: dict[str, str]
    if output_format == "svg":
        full_res_metadata = {
            "Source": "; ".join(rel_sources),
            "Creator": "TRITON-SWMM_toolkit",
        }
    else:
        full_res_metadata = preview_metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        format=output_format if output_path.suffix.lstrip(".") != output_format else None,
        metadata=full_res_metadata,
    )

    # Preview + manifest siblings emit for ALL output formats so /design-figure's
    # subagent-read pathway works uniformly regardless of full-res format
    # (matplotlib's fig is reusable across savefig calls; no rasterization needed).
    preview_path = output_path.parent / f"{output_path.stem}.preview.png"
    fig.savefig(
        preview_path,
        dpi=preview_dpi,
        bbox_inches="tight",
        format="png",
        metadata=preview_metadata,
    )
    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest = {
        "full_res_path": str(output_path),
        "preview_path": str(preview_path),
        "full_res_format": output_format,
        "full_res_dpi": dpi,
        "preview_dpi": preview_dpi,
        "figure_size_inches": list(fig.get_size_inches()),
        "full_res_size_bytes": output_path.stat().st_size,
        "preview_size_bytes": preview_path.stat().st_size,
        "source_paths_relative": rel_sources,
        "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_data:
        manifest["renderer_data"] = manifest_data
    if provenance is not None:
        manifest["artists"] = provenance.serialize()
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    plt.close(fig)
    return output_path


def collect_per_sim_source_paths(renderer_kind: str, event_id: str) -> list[str]:
    """Build `source_paths` for a per-sim plot rule at wildcards-resolution time.

    Called from within the generated master Snakefile via a function-based
    `params:` (see `workflow.py:_build_plot_rule_block_per_sim`). Reads
    relative paths so the figure metadata + caption interpolation stay
    portable across analysis-dir relocations.

    Parameters
    ----------
    renderer_kind : {"peak_flood_depth", "conduit_flow"}
        Which per-sim renderer the caller is rendering for.
    event_id : str
        Snakemake wildcards.event_id — typically `event_index.<n>` for the
        synth fixtures.

    Returns
    -------
    list[str]
        Source-path strings expressed relative to the analysis_dir, suitable
        for embedding in the figure's PNG metadata at render time.
    """
    base = f"sims/{event_id}/processed"
    swmm_inp = f"sims/{event_id}/swmm/hydraulics.inp"
    if renderer_kind == "peak_flood_depth":
        return [f"{base}/TRITONSWMM_TRITON_summary.zarr"]
    if renderer_kind == "conduit_flow":
        return [f"{base}/TRITONSWMM_SWMM_link_summary.zarr", swmm_inp]
    raise ValueError(
        f"unknown renderer_kind {renderer_kind!r}; expected 'peak_flood_depth' or 'conduit_flow'"
    )


def collect_sensitivity_source_paths(independent_var: str) -> list[str]:
    """Build source_paths list for the sensitivity benchmarking rule at wildcards-resolution time.

    Called from within the generated master Snakefile via a function-based `params:`
    (see workflow.py `_build_plot_rule_block_sensitivity_benchmarking`). Kept here
    rather than inlined in the Snakefile to preserve readability of generated
    Snakefiles and to make the source-path construction testable in isolation.

    The returned list is relative to the master analysis_dir (per R13 — sensitivity
    reports live in the master dir). Relative paths include the
    `subanalyses/sa_{id}/sims/{event_id}/...` prefix for per-sub-analysis artifacts.
    """
    raise NotImplementedError(
        "Spec only — implementation wired up in Phase 6 via the "
        "SensitivityAnalysisWorkflowBuilder._collect_sensitivity_source_paths() "
        "bound helper."
    )
