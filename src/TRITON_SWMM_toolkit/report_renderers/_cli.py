"""Argparse entry point for renderer modules — Snakemake rules call this via `python -m`.

Bootstrap follows the toolkit's canonical loader pattern:
  load_system_config(yaml) -> system_config Pydantic model
  load_analysis_config(yaml) -> analysis_config Pydantic model
  TRITONSWMM_system(cfg_system)
  TRITONSWMM_analysis(cfg_analysis, system)

Do not use TRITONSWMM_system.from_config() / TRITONSWMM_analysis.from_configs() —
those classmethods do not exist on those classes (only Toolkit.from_configs() exists,
at toolkit.py:97). The bootstrap below constructs instances directly.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.config.loaders import yaml_to_model
from TRITON_SWMM_toolkit.config.report import DEFAULT_REPORT_CONFIG, report_config
from TRITON_SWMM_toolkit.system import TRITONSWMM_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("renderer", help="renderer module name under report_renderers/")
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--system-config", required=True, type=Path)
    parser.add_argument("--report-config", type=Path, default=None,
                        help="report_config.yaml path; falls back to DEFAULT_REPORT_CONFIG when omitted "
                             "(matches Phase 1 analysis.run() default behavior).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--event-iloc", type=int, default=None, help="for per-sim renderers")
    parser.add_argument("--independent-var", type=str, default=None, help="for sensitivity renderers")
    parser.add_argument("--sa-id", type=str, default=None,
                        help="sub-analysis id (sensitivity master scope only); when present, the renderer "
                             "receives the resolved sub-analysis instead of the master analysis, so per-sim "
                             "renderers can operate on per-sa-scoped scenario data.")
    args = parser.parse_args()

    # TRITONSWMM_system and TRITONSWMM_analysis take YAML Paths and load internally
    # (system.py:25-27, analysis.py:108-109) — pass Paths, not pre-loaded models.
    system = TRITONSWMM_system(args.system_config)
    analysis = TRITONSWMM_analysis(args.analysis_config, system)
    if args.report_config is not None:
        report_cfg = yaml_to_model(args.report_config, report_config)
    else:
        report_cfg = DEFAULT_REPORT_CONFIG

    # Sub-analysis routing: when --sa-id is present, resolve the sub-analysis from
    # the master and pass it as the `analysis` argument to the renderer. Per-sim
    # renderers then read per-sa-scoped scenario data (sims/, processed/, etc.)
    # without needing to know they were dispatched from the master scope.
    target_analysis = analysis
    if args.sa_id is not None:
        if not getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
            raise ValueError(
                f"--sa-id={args.sa_id} requires the analysis to be a sensitivity master, "
                f"but toggle_sensitivity_analysis is False on {args.analysis_config}."
            )
        sub_analyses = analysis.sensitivity.sub_analyses
        if args.sa_id not in sub_analyses:
            raise ValueError(
                f"--sa-id={args.sa_id!r} not found in master's sub_analyses; "
                f"available: {sorted(sub_analyses.keys())}"
            )
        target_analysis = sub_analyses[args.sa_id]

    module = importlib.import_module(f"TRITON_SWMM_toolkit.report_renderers.{args.renderer}")
    kwargs: dict = {}
    if args.event_iloc is not None:
        kwargs["event_iloc"] = args.event_iloc
    if args.independent_var is not None:
        kwargs["independent_var"] = args.independent_var
    module.render(target_analysis, report_cfg, args.output, **kwargs)


if __name__ == "__main__":
    main()
