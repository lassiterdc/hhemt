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

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.system import TRITONSWMM_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("renderer", help="renderer module name under report_renderers/")
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--system-config", required=True, type=Path)
    parser.add_argument(
        "--hpc-system-config",
        type=Path,
        default=None,
        help="Optional path to the per-HPC-system configuration YAML. Emitted by the "
        "Snakemake rule generator (_get_config_args) for every rule; threaded into "
        "TRITONSWMM_analysis so the renderer builds the analysis consistently with the "
        "sibling runners (setup_workflow/prepare_scenario_runner/export_scenario_status).",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--event-iloc", type=int, default=None, help="for per-sim renderers")
    parser.add_argument("--independent-var", type=str, default=None, help="for sensitivity renderers")
    parser.add_argument(
        "--sa-id",
        type=str,
        default=None,
        help="sub-analysis id (sensitivity master scope only); when present, the renderer "
        "receives the resolved sub-analysis instead of the master analysis, so per-sim "
        "renderers can operate on per-sa-scoped scenario data.",
    )
    parser.add_argument(
        "--static-config-id",
        type=str,
        default=None,
        help="Publication static-plot ID; loads the matching StaticPlotBaseConfig "
        "from cfg_analysis.static_plot_configs and renders in publication mode.",
    )
    parser.add_argument(
        "--static-config-path",
        type=Path,
        default=None,
        help="Absolute path to the per-plot static-config YAML. The static-plots "
        "generator emits this so the render is self-contained — the config is loaded "
        "directly from this path, NOT re-searched in the (possibly override-supplied, "
        "non-persisted) cfg_analysis.static_plot_configs. Falls back to an id-search "
        "when only --static-config-id is given (persisted-config usage).",
    )
    args = parser.parse_args()

    # TRITONSWMM_system and TRITONSWMM_analysis take YAML Paths and load internally
    # (system.py:25-27, analysis.py:108-109) — pass Paths, not pre-loaded models.
    system = TRITONSWMM_system(args.system_config)
    analysis = TRITONSWMM_analysis(args.analysis_config, system, hpc_system_config_yaml=args.hpc_system_config, is_main_orchestrator=False, skip_log_update=True)
    # Post-F2: report cfg lives inline on cfg_analysis (R1, load-time-required).
    report_cfg = analysis.cfg_analysis.report

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
                f"--sa-id={args.sa_id!r} not found in master's sub_analyses; available: {sorted(sub_analyses.keys())}"
            )
        target_analysis = sub_analyses[args.sa_id]

    module = importlib.import_module(f"hhemt.report_renderers.{args.renderer}")
    kwargs: dict = {}
    if args.event_iloc is not None:
        kwargs["event_iloc"] = args.event_iloc
    if args.independent_var is not None:
        kwargs["independent_var"] = args.independent_var

    # Publication static-plot dispatch (Decision B -> B1): a single dispatcher.
    # Load the StaticPlotBaseConfig and pass it into render() as static_cfg.
    # Preferred path: --static-config-path loads the config directly (the
    # static-plots generator always emits it), so the render is self-contained
    # and works even when the configs were supplied via the facade's
    # override_static_plot_configs (which is NOT persisted into the analysis
    # config the rule subprocess re-reads). Fallback: an id-search over
    # cfg_analysis.static_plot_configs for persisted-config usage.
    # An unknown id/path raises ConfigurationError -> CLI exit 2 via the
    # pre-existing cli_utils.EXIT_CODE_MAP entry (no cli_utils edit needed).
    if args.static_config_path is not None or args.static_config_id is not None:
        from hhemt.static_snakefile_generator import _load_static_config

        match = None
        if args.static_config_path is not None:
            scfg = _load_static_config(args.static_config_path)
            if args.static_config_id is None or scfg.plot_id == args.static_config_id:
                match = scfg
            else:
                from hhemt.exceptions import ConfigurationError

                raise ConfigurationError(
                    field="static_config_id",
                    message=(
                        f"--static-config-path={args.static_config_path} carries plot_id="
                        f"{scfg.plot_id!r}, which does not match --static-config-id="
                        f"{args.static_config_id!r}."
                    ),
                    config_path=args.static_config_path,
                )
        else:
            for cfg_path in analysis.cfg_analysis.static_plot_configs:
                scfg = _load_static_config(cfg_path)
                if scfg.plot_id == args.static_config_id:
                    match = scfg
                    break
            if match is None:
                from hhemt.exceptions import ConfigurationError

                raise ConfigurationError(
                    field="static_config_id",
                    message=(
                        f"--static-config-id={args.static_config_id!r} not found in "
                        "cfg_analysis.static_plot_configs (and no --static-config-path given)."
                    ),
                    config_path=args.analysis_config,
                )
        kwargs["static_cfg"] = match

    from hhemt.report_renderers._provenance_audit import audit_renderer_io

    with audit_renderer_io(
        args.output,
        target_analysis.analysis_paths.analysis_dir,
        renderer_name=args.renderer,
    ):
        module.render(target_analysis, report_cfg, args.output, **kwargs)


if __name__ == "__main__":
    import sys

    from hhemt.cli_utils import map_exception_to_exit_code
    from hhemt.exceptions import TRITONSWMMError

    try:
        main()
    except TRITONSWMMError as exc:
        # Map known toolkit errors to the CLI exit-code contract
        # (cli_utils.EXIT_CODE_MAP) so callers get a stable code — e.g. an
        # unknown --static-config-id raises ConfigurationError -> exit 2 (R12).
        # Unexpected (non-toolkit) exceptions are intentionally NOT caught here
        # so their full traceback still surfaces for debugging.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(map_exception_to_exit_code(exc))
