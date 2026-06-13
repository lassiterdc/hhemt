"""EdaResult — the return contract for every eda/ data-prep function (ADR-9).

An EdaResult carries (a) an optional ``analysis_validation.CheckResult`` verdict
(surfaced in the report's Errors-and-Warnings section via ``validate_analysis()``)
and (b) the path to a derived artifact written under ``{analysis_dir}/eda/`` with a
``<stem>.manifest.json`` sidecar in the ``_figure_emission`` schema, so the artifact
is a first-class ``harvest_source_paths`` provenance source. ``EdaResult.verdict`` IS
a ``CheckResult`` (imported, not re-defined) so the persist+merge in
``validate_analysis()`` round-trips through one dataclass schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from TRITON_SWMM_toolkit.analysis_validation import CheckResult


@dataclass
class EdaResult:
    """Return value of an eda/ data-prep function."""

    verdict: CheckResult | None = None
    artifact_path: Path | None = None
    plot_id: str | None = None
    skipped: bool = False
