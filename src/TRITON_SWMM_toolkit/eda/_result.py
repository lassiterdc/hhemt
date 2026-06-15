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

from dataclasses import dataclass, field
from pathlib import Path

from TRITON_SWMM_toolkit.analysis_validation import CheckResult


@dataclass
class EdaResult:
    """Return value of an eda/ data-prep function."""

    verdict: CheckResult | None = None
    artifact_path: Path | None = None
    plot_id: str | None = None
    skipped: bool = False


@dataclass(frozen=True)
class EdaReportResult:
    """Return value of analysis.eda() / Bundle.eda() (the facade layer).

    Carries the three artifact classes the facade produces: the assembled doc,
    the rendered EDA plots, and the calc-stage verdicts (empty on Bundle.eda,
    which skips calc).
    """

    report_path: Path
    plot_paths: list[Path] = field(default_factory=list)
    verdicts: list[CheckResult] = field(default_factory=list)
