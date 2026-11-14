"""Standalone CLI to render the analysis report.

Invoked from ``rule render_report``'s shell command (auto-render-as-rule) so
that the report is produced automatically as part of the workflow without
requiring a separate ``render_report()`` call. Also usable directly from the shell.

Usage:
    python -m hhemt.render_report_runner \\
        --system-config /path/to/cfg_system.yaml \\
        --analysis-config /path/to/cfg_analysis.yaml \\
        --format zip

Without ``--format`` the render reads ``report.interactive.report_html_mode``, writes the
format chosen into ``analysis_report.format`` beside the report (the file the workflow's
render rule declares as its output), and the runner deletes the other format's report file
from that directory.

Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Render TRITON-SWMM analysis report.")
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument(
        "--hpc-system-config",
        type=Path,
        default=None,
        help="Optional HPC system config YAML (partition→hardware, container spec). "
        "Threaded by the orchestrator so render matches the run's execution environment.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "zip"],
        default=None,
        help="Output format. Default: the report_html_mode setting of the analysis config.",
    )
    parser.add_argument(
        "--report-config",
        type=Path,
        default=None,
        help="Report config YAML given at run entry. Replaces the analysis config's report: block "
        "for this render. Emitted by the workflow generator only when the run was given a report config.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Render against Snakefile.reprocess instead of the production Snakefile "
        "(set by the reprocess-generator render_report rule shells).",
    )
    args = parser.parse_args()

    try:
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.system import TRITONSWMM_system

        system = TRITONSWMM_system(
            args.system_config,
            # OBSERVER: rendering reads completed outputs and writes no log field, so this
            # subprocess must not author the SHARED system_log.json write that a default
            # construction performs (change 1028-2152, same shape as 1028-1828).
            skip_log_update=True,
        )
        analysis = TRITONSWMM_analysis(
            args.analysis_config,
            system,
            hpc_system_config_yaml=args.hpc_system_config,
            is_main_orchestrator=False,
            # OBSERVER, and the reason this is not cosmetic: a sensitivity-master render
            # materializes every member analysis, each of which forwards this flag, so
            # leaving it False authors one master plus N member analysis-log writes from a
            # job that only reads (change 1028-2152).
            skip_log_update=True,
        )

        if args.report_config is not None:
            from hhemt.config.loaders import yaml_to_model
            from hhemt.config.report import report_config as ReportConfigModel

            # The same attribute analysis.run() sets from its report_config argument. The
            # sensitivity facade holds its own copy that the format resolution reads.
            run_entry_cfg = yaml_to_model(args.report_config, ReportConfigModel)
            analysis._cfg_report = run_entry_cfg
            analysis._run_entry_report_cfg = run_entry_cfg  # marks the record's source as run_entry
            if analysis.cfg_analysis.toggle_sensitivity_analysis:
                analysis.sensitivity._cfg_report = run_entry_cfg
                analysis.sensitivity._run_entry_report_cfg = run_entry_cfg

        if analysis.cfg_analysis.toggle_sensitivity_analysis:
            out = analysis.sensitivity.render_report(format=args.format, reprocess=args.reprocess)
        else:
            out = analysis.render_report(format=args.format, reprocess=args.reprocess)

        if args.format is None:
            # The setting chose the format, so the other format's report file in the same
            # directory is a leftover of an earlier render. An explicit --format leaves that file:
            # the caller asked for that one format and may hold both.
            # The deletion goes through the DU tool, as the reprocess arms that delete the
            # same report files do: the files count toward the analysis-scope total.
            from hhemt import du_sentinels

            other = "zip" if Path(out).suffix == ".html" else "html"
            du_sentinels.delete_and_account(
                [Path(out).parent / f"analysis_report.{other}"], scope_dir=Path(out).parent, scope="analysis"
            )

        logger.info("Rendered report: %s", out)
        print(str(out))
        return 0
    except Exception:
        logger.error("render_report_runner failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
