"""Standalone CLI to render the analysis report.

Invoked from ``rule render_report``'s shell command (auto-render-as-rule) so
that the report is produced automatically as part of the workflow without
requiring a separate ``render_report()`` call. Also usable directly from the shell.

Usage:
    python -m TRITON_SWMM_toolkit.render_report_runner \\
        --system-config /path/to/cfg_system.yaml \\
        --analysis-config /path/to/cfg_analysis.yaml \\
        --format zip

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
        "--format",
        choices=["html", "zip"],
        default="zip",
        help="Output format (default: zip).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Render against Snakefile.reprocess instead of the production Snakefile "
        "(set by the reprocess-generator render_report rule shells).",
    )
    args = parser.parse_args()

    try:
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

        system = TRITONSWMM_system(args.system_config)
        analysis = TRITONSWMM_analysis(args.analysis_config, system, is_main_orchestrator=False)

        if analysis.cfg_analysis.toggle_sensitivity_analysis:
            out = analysis.sensitivity.render_report(format=args.format, reprocess=args.reprocess)
        else:
            out = analysis.render_report(format=args.format, reprocess=args.reprocess)

        logger.info("Rendered report: %s", out)
        print(str(out))
        return 0
    except Exception:
        logger.error("render_report_runner failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
