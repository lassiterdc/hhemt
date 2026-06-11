"""Version-migration constants.

`LAYOUT_VERSION` is the canonical current layout version. Bump when an
on-disk breaking change is introduced — the CI Check A enforces that a
bump implies a matching `versions/V{N:04d}__*.py` and matching golden
fixtures.

`MINIMUM_SUPPORTED_VERSION` is the floor below which `migrate` refuses to
run. Raised manually on major toolkit releases per the resolved decision
in the master plan.
"""

from __future__ import annotations

LAYOUT_VERSION: int = 10
MINIMUM_SUPPORTED_VERSION: int = 0

#: Render-bundle manifest schema version. Stamped into bundle_manifest.json
#: at emit time and validated at consume time (report-from-bundle CLI).
BUNDLE_SCHEMA_VERSION: int = 2

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
