"""V0010: backfill canonical `plot_id` into figure manifests; orphan old-stem figures.

reporting-system_canonical-plot-id (ADR-2) changes the figure-output STEM from
renderer-kind-only (`peak_flood_depth`, with event_id in the dir path) to the
canonical grammar `{renderer_kind}[__{descriptor}][__sa.{sa_id}][__evt.{event_id}]`,
and adds a first-class `plot_id` field to every `*.manifest.json` sidecar.

The manifest `plot_id` is the load-bearing migration primitive: manifests are the
harvested-provenance source of truth (`_figure_emission.harvest_source_paths`).
Figure files (.png/.html + .preview.png) are regenerable; this migration deletes
the old-stem figure set so `harvest_source_paths` does not double-key old+new
stems during the transition, and the established first-run rerun cascade
(CLAUDE.md Gotcha 17/29/38) regenerates them at canonical stems on the next
`run`/`reprocess`. The report is re-rendered on every reprocess regardless of
toggle, so the re-render is not extra work the operator would not otherwise do.

The canonical `plot_id` for an already-rendered figure is reconstructed from the
manifest's own on-disk location (renderer_kind = old stem; event_id / sa_id parsed
from the `plots/...` path segments) -- NOT by re-importing the workflow.py minting
logic -- so this migration carries no drift surface with the renderer's grammar.

Known transient caveat: the benchmarking figure's old stem is descriptor-bearing
(`{independent_var}_vs_total`), so `_derive_plot_id` reconstructs its plot_id as
that old stem verbatim, not the freshly-minted `benchmarking__{independent_var}.vs.total`.
This is accepted, not special-cased: the figure is orphaned-and-regenerated, so the
fresh canonical id supersedes the transient one on the next render.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

logger = logging.getLogger(__name__)

version_from: int = 9
version_to: int = 10
description: str = (
    "Backfill canonical plot_id into figure manifests; orphan old-stem figure "
    "files (ADR-2 reporting-system_canonical-plot-id)"
)

# Old-stem figure siblings to orphan once plot_id is backfilled. The manifest
# itself is NOT deleted -- it carries the durable plot_id and is the regenerated
# figure's provenance anchor on re-render.
_OLD_FIGURE_SUFFIXES = (".png", ".html", ".svg", ".preview.png")


def _derive_plot_id(manifest_path: Path, plots_root: Path) -> str:
    """Reconstruct the canonical plot_id from the manifest's on-disk location.

    renderer_kind = the old figure stem (renderer-kind-only pre-V0010).
    event_id / sa_id are parsed from the path segments under plots/, mirroring
    `_figure_emission.harvest_source_paths` (which already parses
    `plots/sensitivity/per_sim/sa-{N}/{event_id}/...`).

    Grammar: {renderer_kind}[__sa.{sa_id}][__evt.{event_id}]
    (descriptor-bearing renderers -- sensitivity benchmarking -- carry their
    descriptor in the old stem already, so renderer_kind captures it.)
    """
    renderer_kind = manifest_path.stem.removesuffix(".manifest")
    rel_parts = manifest_path.parent.resolve().relative_to(plots_root.resolve()).parts
    sa_id: str | None = None
    event_id: str | None = None
    # Sensitivity per-sub: plots/sensitivity/per_sim/sa-{N}/{event_id}/<stem>.manifest.json
    if (
        len(rel_parts) >= 4
        and rel_parts[0] == "sensitivity"
        and rel_parts[1] == "per_sim"
        and rel_parts[2].startswith("sa-")
    ):
        sa_id = rel_parts[2][len("sa-") :]
        event_id = rel_parts[3]
    # Multisim per-sim: plots/per_sim/{event_id}/<stem>.manifest.json
    elif len(rel_parts) >= 2 and rel_parts[0] == "per_sim":
        event_id = rel_parts[1]
    plot_id = renderer_kind
    if sa_id is not None:
        plot_id += f"__sa.{sa_id}"
    if event_id is not None:
        plot_id += f"__evt.{event_id}"
    return plot_id


def upgrade(ctx: MigrationContext) -> None:
    plots_root = ctx.target_dir / "plots"
    if not plots_root.is_dir():
        return  # nothing rendered yet (e.g. system_directory pass) -- no-op

    for manifest_path in sorted(plots_root.rglob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        plot_id = _derive_plot_id(manifest_path, plots_root)
        # Idempotency: skip a manifest already carrying its canonical plot_id.
        # On a re-run the old-stem figures are already gone, so the orphan sweep
        # below would no-op anyway; skipping here keeps the walk tight.
        if manifest.get("plot_id") == plot_id:
            continue
        ctx.log_add_field(manifest_path, "plot_id", plot_id)
        # Orphan the old-stem figure siblings now that plot_id is recorded.
        # Guarded against the manifest (verified replacement) and idempotent:
        # guarded_remove no-ops when the figure is already gone. The figure
        # regenerates at its canonical stem on the next render.
        old_stem = manifest_path.stem.removesuffix(".manifest")
        for suffix in _OLD_FIGURE_SUFFIXES:
            old_fig = manifest_path.parent / f"{old_stem}{suffix}"
            if old_fig.exists():
                ctx.guarded_remove(
                    old_fig,
                    verify_replacement_at=manifest_path,
                    force=True,
                )
