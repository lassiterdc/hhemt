"""Bundle subsystem — render-bundle emission and the consume-side
``Bundle`` class.

Public surface
--------------
- ``Bundle`` — the consume-side entry point for a render bundle. Used by
  callers that already have a bundle on disk and want to regenerate the
  analysis report locally without re-running ``Analysis.run()``.
- ``emit_bundle`` — the emit-side helper. Invoked by
  ``Analysis.bundle_report_data()`` and
  ``TRITONSWMM_sensitivity_analysis.bundle_report_data()`` to produce a
  portable render bundle from a completed HPC analysis.

``Bundle`` is deliberately NOT a subclass of ``TRITONSWMM_analysis``.
Bundle outputs are pre-computed; ``Analysis.run()`` is not callable
against a bundle. The class boundary is the user's binding constraint
per the bundle-portable-report-regeneration plan's Friction 5 design
recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from TRITON_SWMM_toolkit.bundle._emit import (
    _get_toolkit_git_sha,
    emit_bundle,
)
from TRITON_SWMM_toolkit.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)

__all__ = [
    "Bundle",
    "emit_bundle",
    "_get_toolkit_git_sha",
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_SCHEMA_VERSION",
]


class Bundle:
    """A portable render bundle, ready for local report regeneration."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self._root = root.resolve()
        self._manifest = manifest

    @classmethod
    def from_directory(cls, path: Path | str) -> "Bundle":
        """Construct a ``Bundle`` from a bundle directory on disk.

        The directory must contain ``bundle_manifest.json``. All paths
        resolve via ``bundle.root`` at call time — no ``os.chdir``, no
        persisted absolute paths.
        """
        root = Path(path).resolve()
        manifest_path = root / BUNDLE_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No {BUNDLE_MANIFEST_FILENAME} under {root}. "
                f"Is this a render bundle?"
            )
        manifest = json.loads(manifest_path.read_text())
        return cls(root=root, manifest=manifest)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> dict:
        return self._manifest

    def absolute(self, rel: str | Path) -> Path:
        """Resolve a bundle-relative path to an absolute path under
        ``bundle.root``."""
        return (self._root / rel).resolve()

    def regenerate_report(
        self, format: Literal["html", "zip"] = "html"
    ) -> Path:
        """Regenerate the analysis report from bundled data.

        Stubbed in Phase 1. Phase 2 wires the regeneration-scoped
        Snakefile generator; Phase 3 wires the actual subprocess call
        and CLI integration.
        """
        raise NotImplementedError(
            "Bundle.regenerate_report() will be implemented in Phase 3 "
            "after the regeneration-scoped Snakefile generator (Phase 2) "
            "and the CLI rewire (Phase 3) land."
        )
