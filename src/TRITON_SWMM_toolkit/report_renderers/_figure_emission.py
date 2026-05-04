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


def format_sources_rst(source_paths: list[dict | str]) -> str:
    """Render a list of source-path dicts to a multi-line RST bullet block.

    Produces the text that goes under ``**Sources:**`` in caption RSTs:

    .. code-block:: text

        - ``path/to/source1.zarr``

          - ``data array name 1``
          - ``data array name 2``

        - ``path/to/source2.csv``

    Each main bullet is a data source path; sub-bullets enumerate the
    array / column / section names the renderer reads from that source.
    Pre-rendering in Python (at Snakefile rule-emit time) sidesteps Jinja
    whitespace-control concerns that arise when nested bullets are
    constructed in Jinja templates: Snakemake's caption template engine
    strips Jinja blank lines, which collapses RST nested-list structure
    into a single ``<li>``. Embedding the fully-rendered RST as a single
    string param keeps the bullet structure intact through to docutils.
    """
    lines: list[str] = []
    for src in source_paths:
        if isinstance(src, dict):
            path = src["path"]
            variables = src.get("variables") or []
        else:
            path = str(src)
            variables = []
        lines.append(f"- ``{path}``")
        if variables:
            lines.append("")
            for v in variables:
                lines.append(f"  - ``{v}``")
        lines.append("")
    return "\n".join(lines)


def _validate_source_path(path: str | Path, *, analysis_dir: str | Path | None = None) -> str:
    """Reject directory-as-source (Iter 8 agenda item 4) — except for zarr stores.

    All plottable data lives in files, with one exception: zarr stores
    (``.zarr`` directories with ``.zattrs`` / ``.zgroup`` / ``.zarray``
    markers, or any path whose suffix is ``.zarr``). Zarr is opened via
    xarray's ``open_zarr`` / ``open_dataset`` as a single logical dataset,
    so the path-as-directory is an implementation detail rather than an
    enumerable file collection. All other directory-as-source emissions
    raise ``ValueError`` so future regressions are caught at render time.
    Non-existent paths are accepted (the collector may emit a path on a
    different filesystem or a path the renderer will create).
    """
    p = Path(path)
    if not p.is_absolute() and analysis_dir is not None:
        p = (Path(analysis_dir) / p).resolve()
    if not p.is_dir():
        return str(path)
    # Allow zarr directories (suffix or marker file present).
    if p.suffix == ".zarr" or (p / ".zattrs").exists() or (p / ".zgroup").exists() or (p / ".zarray").exists():
        return str(path)
    raise ValueError(
        f"Directory-as-source rejected (Iter 8 agenda item 4): {path!r} resolves "
        f"to a directory ({p}) that is NOT a zarr store. Source paths must be "
        f"files (or zarr stores); arbitrary directories are not plottable. "
        f"Either fix the collector to point at the specific file inside this "
        f"directory, or expand the directory to its enclosed files."
    )


def per_sim_map_ticks(bounds: tuple[float, float, float, float]) -> tuple[list[float], list[float]]:
    """Return identical (xticks, yticks) for per-sim map panels at a 50-unit
    step rounded to the nearest 50 below/above bounds.

    Forces lim/tick parity between per_sim_peak_flood_depth and
    per_sim_conduit_flow regardless of where the underlying data sits within
    the DEM extent (matplotlib's auto-locator otherwise picks different
    starting/ending ticks based on data content, producing visible tick
    drift between toggles).
    """
    import math

    def _ticks(lo, hi, step=50.0):
        # Constrain ticks WITHIN bounds (round lo UP, hi DOWN) so set_xticks
        # does not extend xlim past the actual data range.
        start = math.ceil(lo / step) * step
        end = math.floor(hi / step) * step
        if end < start:
            return []
        n = int(round((end - start) / step)) + 1
        return [start + i * step for i in range(n)]

    return _ticks(bounds[0], bounds[2]), _ticks(bounds[1], bounds[3])


def add_panel_label(ax, label: str) -> None:
    """Add a regular-weight lowercase parenthesized panel letter ABOVE-LEFT of the axes.

    Convention (Subiteration 9.4): publication-standard `(a)`, `(b)`, `(c)`
    overlays placed in axes-fraction coordinates OUTSIDE the data area —
    `x = -0.02` (slightly left of the y-axis spine), `y = 1.05` (above the
    axes title). Regular weight (no bold), fontsize 11. Placement avoids
    overlapping plot data, panel titles, or tick labels regardless of the
    figure's specific data-extent.
    """
    ax.text(
        -0.02, 1.05, label,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom", ha="right",
        zorder=10,
        clip_on=False,
    )


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
    # Reject directory-as-source per Iter 8 agenda item 4 — all plottable data
    # lives in files. Catches future regressions where a collector emits a
    # `.parent` or directory path by mistake.
    for _p in source_paths:
        _validate_source_path(_p, analysis_dir=analysis_dir)
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


def collect_per_sim_source_paths(
    renderer_kind: str,
    event_id: str,
    *,
    rainfall_datavar: str,
    storm_tide_datavar: str | None,
    dem_rel_path: str | None = None,
    watershed_rel_path: str | None = None,
    sa_id: str | None = None,
) -> list[dict]:
    """Build `source_paths` for a per-sim plot rule at wildcards-resolution time.

    Called from within the generated master Snakefile via a function-based
    `params:` (see `workflow.py:_build_plot_rule_block_per_sim` and
    `_build_plot_rule_block_per_sim_per_sa`). Reads relative paths so the
    figure metadata + caption interpolation stay portable across analysis-dir
    relocations.

    Each returned dict has the schema ``{"path": str, "variables": list[str]}``
    where ``variables`` enumerates the dataset variables the renderer reads
    from that source. Caption RSTs render the dict as a path bullet with
    variable sub-bullets (with a backward-compat shim for callers that still
    return ``list[str]`` — see the caption RST templates).

    Parameters
    ----------
    renderer_kind : {"peak_flood_depth", "conduit_flow"}
        Which per-sim renderer the caller is rendering for.
    event_id : str
        Snakemake wildcards.event_id — typically `event_index.<n>` for the
        synth fixtures.
    dem_rel_path : str, optional
        Analysis-dir-relative path to the processed DEM raster (read by
        peak_flood_depth as the ground-elevation underlay). Baked into the
        Snakefile rule's params closure at emit time by the workflow.py rule
        builder.
    watershed_rel_path : str, optional
        Analysis-dir-relative path to the watershed boundary GIS polygon
        (read by peak_flood_depth as the masking shape).
    sa_id : str, optional
        Sub-analysis id when called at sensitivity-master scope. When present,
        the per-event source paths are prefixed with ``subanalyses/{sa_id}/``
        so the caption-rendered paths are master-analysis-dir-relative (the
        master Snakefile renders captions; per-sa scenarios live under
        ``master_dir/subanalyses/{sa_id}/sims/{event_id}/...``).

    Returns
    -------
    list[dict]
        Source descriptors expressed relative to the analysis_dir.
    """
    sa_prefix = f"subanalyses/{sa_id}/" if sa_id else ""
    base = f"{sa_prefix}sims/{event_id}/processed"
    swmm_inp = f"{sa_prefix}sims/{event_id}/swmm/hydraulics.inp"
    weather_nc = f"{sa_prefix}sims/{event_id}/sim_weather.nc"
    if renderer_kind == "peak_flood_depth":
        sources: list[dict] = [
            {
                "path": f"{base}/TRITONSWMM_TRITON_summary.zarr",
                "variables": ["max_wlevel_m"],
            },
            {
                "path": weather_nc,
                "variables": ["time", rainfall_datavar] + (
                    [storm_tide_datavar] if storm_tide_datavar else []
                ),
            },
        ]
        if dem_rel_path:
            # No sub-bullets: the DEM is read as a single raster (no indexer
            # enumeration); descriptive prose belongs in the caption body, not
            # under the source bullet.
            sources.append({
                "path": dem_rel_path,
                "variables": [],
            })
        if watershed_rel_path:
            # No sub-bullets: the polygon is a single shape used for masking +
            # boundary overlay; no enumerable indexers.
            sources.append({
                "path": watershed_rel_path,
                "variables": [],
            })
        return sources
    if renderer_kind == "conduit_flow":
        sources = [
            {
                "path": f"{base}/TRITONSWMM_SWMM_link_summary.zarr",
                "variables": ["max_over_full_flow", "max_flow_cms", "link_id"],
            },
            {
                "path": swmm_inp,
                "variables": ["[CONDUITS]", "[COORDINATES]"],
            },
            {
                "path": weather_nc,
                "variables": ["time", rainfall_datavar] + (
                    [storm_tide_datavar] if storm_tide_datavar else []
                ),
            },
        ]
        if dem_rel_path:
            sources.append({"path": dem_rel_path, "variables": []})
        if watershed_rel_path:
            sources.append({"path": watershed_rel_path, "variables": []})
        return sources
    raise ValueError(
        f"unknown renderer_kind {renderer_kind!r}; expected 'peak_flood_depth' or 'conduit_flow'"
    )


def collect_sensitivity_source_paths(
    independent_var: str,
    *,
    swmm_only_rpt_rel_paths: list[str] | None = None,
) -> list[dict]:
    """Build source_paths list for the sensitivity benchmarking rule at wildcards-resolution time.

    Called from within the generated master Snakefile via a function-based `params:`
    (see workflow.py `_build_plot_rule_block_sensitivity_benchmarking`). Kept here
    rather than inlined in the Snakefile to preserve readability of generated
    Snakefiles and to make the source-path construction testable in isolation.

    Each returned dict has the schema ``{"path": str, "variables": list[str]}`` —
    see :func:`collect_per_sim_source_paths` for caption-RST rendering details.

    The renderer reads the master `sensitivity_datatree.zarr` for TRITON-coupled /
    TRITON-only sub-analyses (per Phase 6 verification: `/sa_{id}/tritonswmm/performance.Total`
    is present on the master DataTree, dimensioned by event_iloc). The `independent_var`
    columns live in the sensitivity CSV. SWMM-only sub-analyses fall back to per-scenario
    `.rpt` parsing inside the renderer; those `.rpt` paths are not enumerated here
    because they are SWMM-only-conditional and only the renderer knows the enabled
    model types.
    """
    del independent_var  # currently unused; the same source set serves all wildcards
    sources: list[dict] = [
        {
            "path": "sensitivity_datatree.zarr",
            "variables": [
                "/sa_{id}/tritonswmm/performance.Total (per sub-analysis)",
            ],
        },
        {
            "path": "sensitivity_analysis_definition.csv",
            "variables": [
                "sa_id, run_mode, n_mpi_procs, n_omp_threads, n_gpus, n_nodes",
            ],
        },
    ]
    # SWMM-only sub-analyses' .rpt paths (baked into closure at Snakefile-emit
    # time by `workflow.py:_build_plot_rule_block_sensitivity_benchmarking`).
    # Each .rpt is parsed via `swmm_output_parser.parse_total_elapsed` to
    # produce that sub-analysis's wallclock value when the DataTree path
    # `/sa_{id}/tritonswmm/performance.Total` is unavailable (SWMM-only mode
    # has no TRITON-SWMM coupled tree branch).
    for rpt_rel in swmm_only_rpt_rel_paths or []:
        sources.append({
            "path": rpt_rel,
            "variables": ["Total elapsed time (parsed via parse_total_elapsed)"],
        })
    return sources


def harvest_source_paths(
    plots_dir: Path, analysis_dir: Path
) -> dict[str, list[Path]]:
    """Walk ``*.manifest.json`` sidecars under ``plots_dir``; return source
    paths keyed by output figure stem.

    Source-of-truth: the top-level ``source_paths_relative`` field written by
    :func:`emit_plot_with_sources` (paths are relative to ``analysis_dir``;
    resolved here back to absolute paths). Per-artist
    ``channels[].ref.source_path`` entries from the provenance log are unioned
    in as a secondary channel so renderer-internal data sources declared via
    ``prov.artist().add_channel(...)`` are not silently dropped.

    Downstream consumers (the bundle-emit step) union and deduplicate paths
    across all keys before bundle copy, so figure-stem keying is sufficient
    even when multiple per-sim subdirectories emit the same bare filename
    (e.g., ``peak_flood_depth.manifest.json``).
    """
    sources_by_renderer: dict[str, list[Path]] = {}
    for manifest_path in sorted(plots_dir.rglob("*.manifest.json")):
        figure_stem = manifest_path.stem.removesuffix(".manifest")
        manifest = json.loads(manifest_path.read_text())
        rel_paths = manifest.get("source_paths_relative", [])
        paths = [(analysis_dir / Path(rp)).resolve() for rp in rel_paths]
        for artist in manifest.get("artists", []):
            for channel in artist.get("channels", []):
                ref = channel.get("ref", {}) or {}
                src = ref.get("source_path")
                if src:
                    p = Path(src)
                    if not p.is_absolute():
                        p = (analysis_dir / p).resolve()
                    paths.append(p)
        sources_by_renderer.setdefault(figure_stem, []).extend(paths)
    return {
        name: list(dict.fromkeys(paths))
        for name, paths in sources_by_renderer.items()
    }
