"""V0005: inline source-side report_config.yaml into cfg_analysis.yaml::report.

F2 schema canonicalization (per the bundle-cfg-report-canonicalization plan)
makes `analysis_config.report` the canonical source of truth for renderer
parameters. Legacy analyses on disk have a separate report_config.yaml
referenced by absolute path in their Snakefile's shell lines. This migration
grep-extracts the `--report-config <path>` substring from `{target_dir}/Snakefile`,
loads the referenced YAML, and inlines it into `cfg_analysis.yaml::report`.

When the source-side report_config is unrecoverable (no Snakefile, no
`--report-config` substring, or referenced path missing), the migration
raises MigrationBlockedError naming the analysis-dir path and the minimum
`report:` block the operator must add to cfg_analysis.yaml before re-running
the migration. No fallback, no stub, no heuristic — the operator's manual
edit reflects ground truth about which backend the analysis was actually
rendered with.
"""
from __future__ import annotations

import re
from pathlib import Path

from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError

version_from: int = 4
version_to: int = 5
description: str = (
    "Inline source-side report_config.yaml into cfg_analysis.yaml::report per F2 canonicalization"
)


def _recover_source_report_cfg_path(
    target_dir: Path,
) -> tuple[Path | None, str | None]:
    """Grep `{target_dir}/Snakefile` for the `--report-config <path>` substring.

    Returns `(Path, None)` on success or `(None, reason)` naming the failure
    mode. Single Snakefile read; deterministic failure classification.
    """
    snakefile = target_dir / "Snakefile"
    if not snakefile.exists():
        return None, "no Snakefile present"
    text = snakefile.read_text()
    # Match a path token after `--report-config`. The token excludes
    # whitespace and shell-quoting/grouping characters (`"`, `'`, `` ` ``,
    # `,`, `;`, `)`) so Snakefile shell-line wrappers like `shell: "...
    # --report-config /path/to.yaml"` capture only the path, not the
    # trailing closing quote.
    match = re.search(r"--report-config\s+([^\s\"'`,;)]+)", text)
    if not match:
        return None, "Snakefile present but contains no `--report-config <path>` substring"
    candidate = Path(match.group(1))
    if not candidate.exists():
        return None, (
            f"Snakefile references `--report-config {match.group(1)}` "
            f"but that path is missing"
        )
    return candidate, None


def _blocked_error(target_dir: Path, reason: str) -> MigrationBlockedError:
    cfg_path = target_dir / "cfg_analysis.yaml"
    plots_dir = target_dir / "plots"
    return MigrationBlockedError(
        f"V0005: cannot recover report_config for {target_dir}. Reason: {reason}. "
        f"Add a `report:` block to {cfg_path} (minimum: "
        f"`report: {{interactive: {{static_backend: <matplotlib|plotly>}}}}`) "
        f"and re-run the migration. Choose the backend that matches how this "
        f"analysis was rendered — inspect {plots_dir} for file extensions "
        f"(.png ⇒ matplotlib, .svg ⇒ plotly) as a sanity check, but the "
        f"operator's knowledge of the source-side cfg is authoritative."
    )


def upgrade(ctx: MigrationContext) -> None:
    import yaml as _yaml  # noqa: F401  (kept for parity with sibling migrations)

    cfg_analysis_path = ctx.target_dir / "cfg_analysis.yaml"
    if not cfg_analysis_path.exists():
        # system_directory pass or partial fixture — nothing to inline.
        return

    # Idempotency: rely on yaml_add_field's canonical guard in
    # context._apply_yaml_add_field (single source of truth).
    source_cfg_path, reason = _recover_source_report_cfg_path(ctx.target_dir)
    if source_cfg_path is None:
        raise _blocked_error(ctx.target_dir, reason or "unknown")

    # Read + validate the source-side YAML against the report_config model.
    # A validation failure here is fail-fast: the operator must repair the
    # source cfg or hand-write the `report:` block.
    from hhemt.config.report import report_config
    from hhemt.config.loaders import yaml_to_model

    try:
        cfg = yaml_to_model(source_cfg_path, report_config)
    except Exception as exc:
        raise _blocked_error(
            ctx.target_dir,
            f"source-side report_config at {source_cfg_path} failed to "
            f"load/validate ({exc})",
        ) from exc
    payload = cfg.model_dump(mode="json")

    # `in_model_cls=analysis_config` is accepted for API symmetry with the
    # sibling primitives (yaml_rename_field, etc.); `_apply_yaml_add_field`
    # does not currently consume it for validation. See AW v2 F-FU Flag 10
    # (software-engineering-specialist): unused-noise status confirmed.
    from hhemt.config.analysis import analysis_config

    ctx.yaml_add_field(
        cfg_analysis_path,
        "report",
        payload,
        in_model_cls=analysis_config,
    )
