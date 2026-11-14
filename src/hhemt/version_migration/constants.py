"""Version-migration constants.

`LAYOUT_VERSION` is the canonical current layout version. Bump when an
on-disk breaking change is introduced — the CI Check A enforces that a
bump implies a matching `versions/V{N:04d}__*.py` and matching golden
fixtures.

`MINIMUM_SUPPORTED_VERSION` is the floor below which `migrate` refuses to
run. Raised manually on major toolkit releases per the resolved decision
in the primary plan.
"""

from __future__ import annotations

LAYOUT_VERSION: int = 31
MINIMUM_SUPPORTED_VERSION: int = 0

#: Render-bundle manifest schema version. Stamped into bundle_manifest.json
#: at emit time and validated at consume time (report-from-bundle CLI).
#: v3: round-trippable Workflow-Run-Crate (mainEntity) + reprex carriage.
#: The consume-side guard in bundle/__init__.py rejects both >3 and <3, so this
#: 2->3 bump hard-breaks pre-existing v2 bundles (re-emit from source).
#: v4: analysis_config field rename reclaim_after_processing -> remove_after_processing.
#: A bundle carries cfg_analysis.yaml (bundle/_emit.py) and loads it back through
#: analysis_config (bundle/__init__.py), and the model is extra="forbid" -- so a v3
#: bundle emitted before the rename fails Pydantic validation on a post-rename toolkit.
#: The bump does not CREATE that break; it makes it legible, because the exact-match
#: consume-side guard then raises BundleSchemaError (CLI exit 6, naming the version
#: mismatch) instead of an opaque extra_forbidden traceback from deep in config load.
#: This constant versions MEMBER bundles, the single-analysis bundles `Bundle.from_directory`
#: reads and `emit_bundle` stamps. It is 7. The combined bundle's layout change (commit 930c37eb)
#: did not change a member bundle, so it did not move this constant; the combined bundle has its
#: own, `COMBINED_BUNDLE_SCHEMA_VERSION` below. Sharing one constant made `hhemt combine` refuse
#: every schema-7 member bundle, because the member reader tests this value exactly.
BUNDLE_SCHEMA_VERSION: int = 7

#: Combined-bundle manifest schema version, stamped by `_write_combined_bundle_manifest` and
#: tested exactly by `CombinedBundle.from_directory`.
#: v8: the combined bundle's layout. A combined bundle is by default one zip whose members sit
#: as `child_crates/{experiment_id}.zip` (each experiment's own bundle zip, not its entries spread
#: out) with the two re-rendered child figures as `child_crates/{experiment_id}.refreshed/{path}`
#: entries; its manifest and crate name the `.zip` children. `CombinedBundle.from_directory`
#: refuses any other version, so a combined bundle made at v7 must be re-combined to be read.
COMBINED_BUNDLE_SCHEMA_VERSION: int = 8

#: Default _version.json filename (used by both analysis and system stamps).
VERSION_FILE_NAME: str = "_version.json"

#: Render-bundle output subdirectory name (under {analysis_dir}/).
#: The bundle tar lands at {analysis_dir}/{BUNDLE_OUTPUT_SUBDIR}/{tar_name}.
BUNDLE_OUTPUT_SUBDIR: str = "render_bundle"

#: Bundle-internal subdir for the HPC-baseline analysis_report copies.
#: Local re-render of the bundle preserves this dir untouched while
#: replacing the bundle-root analysis_report.{html,zip}.
BUNDLE_BASELINE_SUBDIR: str = "bundle_baseline"

#: Manifest filename emitted at the bundle root by bundle.emit_bundle.
#: Contains schema version, toolkit git sha, source-paths-by-renderer.
BUNDLE_MANIFEST_FILENAME: str = "bundle_manifest.json"

#: Status subdir copied verbatim from analysis_dir into the bundle.
#: Mirrors the Snakemake _status flag layout.
BUNDLE_STATUS_SUBDIR: str = "_status"

#: Plots subdir copied verbatim from analysis_dir into the bundle.
#: Contains the manifest sidecars that drive bundle harvest.
BUNDLE_PLOTS_SUBDIR: str = "plots"

#: Lock timeout for filelock-guarded _version.json writes (seconds).
LOCK_TIMEOUT_SECONDS: float = 30.0
