"""End-to-end + structural tests of analysis.static_plots() on the synth model (ADR-8).

Tiering:
- The loader + Snakefile-structure tests are fast and need only a configured
  (not run) synth analysis — they prove the bare-output distribution machinery
  (no report() wrapper, no rule render_report), the --static-config-id /
  --event-iloc threading, and the unknown-config ConfigurationError, without a
  compiled TRITON build.
- The full publication-figure render is marked slow + requires_snakemake_subprocess
  (it launches a Snakemake subprocess against a compiled, consolidated synth tree).
"""

from __future__ import annotations

import pytest
import yaml

from hhemt.exceptions import ConfigurationError
from hhemt.scenario import compute_event_id_slug
from hhemt.static_snakefile_generator import (
    _load_static_config,
    generate_static_snakefile,
)

_RENDERER_KIND = "per_sim_peak_flood_depth"


def _write_peak_flood_depth_config(path, plot_id, *, output_format="pdf", **overrides):
    body = {"plot_id": plot_id, "renderer_kind": _RENDERER_KIND, "output_format": output_format}
    body.update(overrides)
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def _first_event_plot_id(analysis) -> str:
    """Canonical ADR-2 peak-flood-depth plot_id for the analysis's first event."""
    event_id = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(0))
    return f"{_RENDERER_KIND}__evt.{event_id}"


_CONDUIT_RENDERER_KIND = "per_sim_conduit_flow"


def _first_event_conduit_flow_plot_id(analysis) -> str:
    """Canonical ADR-2 conduit-flow plot_id for the analysis's first event (per-sim)."""
    event_id = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(0))
    return f"{_CONDUIT_RENDERER_KIND}__evt.{event_id}"


# R7/R12 — unknown renderer_kind raises ConfigurationError at load time.
def test_load_static_config_unknown_renderer_kind_raises(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"plot_id": "bogus__evt.x.0", "renderer_kind": "not_a_real_kind"}))
    with pytest.raises(ConfigurationError):
        _load_static_config(cfg)


# R3/R4/R5/R6 — generated Snakefile.static is bare-output: no report() wrapper,
# no rule render_report; one plot rule + rule all; the --static-config-id and the
# resolved --event-iloc are threaded onto the renderer shell.
def test_static_snakefile_is_bare_output_and_threads_selectors(synth_multi_sim_analysis, tmp_path):
    analysis = synth_multi_sim_analysis
    plot_id = _first_event_plot_id(analysis)
    cfg = _write_peak_flood_depth_config(tmp_path / f"{plot_id}.yaml", plot_id)

    text = generate_static_snakefile(
        analysis,
        static_plot_configs=[cfg],
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        static_backend="matplotlib",
    )

    # No report() wrapper and no render_report rule (publication standalone files).
    assert "report(" not in text
    assert "rule render_report" not in text
    # rule all + the per-plot rule (rule name sanitized: '.' -> '_').
    assert "rule all:" in text
    assert "rule static_plot_" + plot_id.replace(".", "_") + ":" in text
    # Literal publication output path (extension from output_format), no __OUTPUT_EXT__ token.
    assert f"static_plots/{plot_id}.pdf" in text
    assert "__OUTPUT_EXT__" not in text
    # Selector threading: the static-config-id + the generation-time-resolved event iloc.
    assert f"--static-config-id {plot_id}" in text
    assert "--event-iloc 0" in text


# Phase 2 — a system_overview (system-level, not per-sim) config harvests a
# bare-output rule with NO --event-iloc/--sa-id (its plot_id carries no
# __evt./__sa. segment), proving the registry + generator handle the new kind.
def test_system_overview_static_snakefile_no_per_sim_selectors(synth_multi_sim_analysis, tmp_path):
    analysis = synth_multi_sim_analysis
    plot_id = "system_overview"
    cfg = tmp_path / f"{plot_id}.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"plot_id": plot_id, "renderer_kind": "system_overview", "output_format": "pdf"},
            sort_keys=False,
        )
    )

    text = generate_static_snakefile(
        analysis,
        static_plot_configs=[cfg],
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        static_backend="matplotlib",
    )

    assert "report(" not in text
    assert "rule render_report" not in text
    assert f"rule static_plot_{plot_id}:" in text
    assert f"static_plots/{plot_id}.pdf" in text
    assert f"--static-config-id {plot_id}" in text
    # System-level plot: no per-sim / sensitivity selector threading.
    assert "--event-iloc" not in text
    assert "--sa-id" not in text


# Phase 4 — a sensitivity_benchmarking config carries a `var.{independent_var}`
# selector on its plot_id; the generator threads it as `--independent-var` (the
# renderer's required x-axis arg, which report-mode supplies via a Snakefile
# wildcard the bare-output static path lacks). Pure generation-time string parse —
# uses the light (configured, not-run) multi-sim fixture; no sensitivity data needed.
def test_sensitivity_benchmarking_static_snakefile_threads_independent_var(synth_multi_sim_analysis, tmp_path):
    analysis = synth_multi_sim_analysis
    plot_id = "sensitivity_benchmarking__var.n_devices"
    cfg = tmp_path / "sensitivity_benchmarking__var.n_devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"plot_id": plot_id, "renderer_kind": "sensitivity_benchmarking", "output_format": "pdf"},
            sort_keys=False,
        )
    )

    text = generate_static_snakefile(
        analysis,
        static_plot_configs=[cfg],
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        static_backend="matplotlib",
    )

    assert "report(" not in text
    assert "rule render_report" not in text
    assert f"rule static_plot_{plot_id.replace('.', '_')}:" in text
    assert f"static_plots/{plot_id}.pdf" in text
    assert f"--static-config-id {plot_id}" in text
    # The var.{name} selector is threaded as --independent-var; no per-sim/sa selector.
    assert "--independent-var n_devices" in text
    assert "--event-iloc" not in text
    assert "--sa-id" not in text


# R5 — static_config_ids filters the harvested rule set to the named subset.
def test_static_snakefile_static_config_ids_filter(synth_multi_sim_analysis, tmp_path):
    analysis = synth_multi_sim_analysis
    plot_id = _first_event_plot_id(analysis)
    cfg = _write_peak_flood_depth_config(tmp_path / f"{plot_id}.yaml", plot_id)

    # An id set that excludes the only config yields no plot rule.
    text = generate_static_snakefile(
        analysis,
        static_plot_configs=[cfg],
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        static_backend="matplotlib",
        static_config_ids=["some_other_id"],
    )
    assert "rule static_plot_" not in text
    assert "rule all:" in text


# R1/R10 — facade raises when no static-plot configs are resolvable.
def test_static_plots_facade_empty_configs_raises(synth_multi_sim_analysis):
    analysis = synth_multi_sim_analysis
    with pytest.raises(ConfigurationError):
        analysis.static_plots(override_static_plot_configs=[])


@pytest.mark.slow
@pytest.mark.requires_snakemake_subprocess
def test_static_plots_generates_publication_figure(synthetic_multisim_completed_isolated, tmp_path):
    """static_plots() produces static_plots/{plot_id}.pdf for the peak-flood-depth
    exemplar, distributed via the local executor, with a bare-output Snakefile."""
    analysis = synthetic_multisim_completed_isolated
    plot_id = _first_event_plot_id(analysis)
    cfg = _write_peak_flood_depth_config(tmp_path / f"{plot_id}.yaml", plot_id)

    result = analysis.static_plots(execution_mode="local", override_static_plot_configs=[cfg])

    assert result["success"] is True
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert (analysis_dir / "static_plots" / f"{plot_id}.pdf").exists()
    snakefile_static = (analysis_dir / "Snakefile.static").read_text()
    assert "report(" not in snakefile_static
    assert "rule render_report" not in snakefile_static


@pytest.mark.slow
@pytest.mark.requires_snakemake_subprocess
def test_static_plots_generates_conduit_flow_figure(synthetic_multisim_completed_isolated, tmp_path):
    """Phase 3: static_plots() produces static_plots/{plot_id}.pdf via the
    publication static_cfg branch on the per-sim conduit-flow renderer (per-sim —
    carries an --event-iloc selector), distributed via the local executor,
    bare-output Snakefile."""
    analysis = synthetic_multisim_completed_isolated
    plot_id = _first_event_conduit_flow_plot_id(analysis)
    cfg = tmp_path / f"{plot_id}.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"plot_id": plot_id, "renderer_kind": _CONDUIT_RENDERER_KIND, "output_format": "pdf"},
            sort_keys=False,
        )
    )

    result = analysis.static_plots(execution_mode="local", override_static_plot_configs=[cfg])

    assert result["success"] is True
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert (analysis_dir / "static_plots" / f"{plot_id}.pdf").exists()
    snakefile_static = (analysis_dir / "Snakefile.static").read_text()
    assert "report(" not in snakefile_static
    assert "rule render_report" not in snakefile_static


@pytest.mark.slow
@pytest.mark.requires_snakemake_subprocess
def test_static_plots_generates_system_overview_figure(synthetic_multisim_completed_isolated, tmp_path):
    """Phase 2: static_plots() produces static_plots/system_overview.pdf via the
    publication static_cfg branch on the system-overview renderer (system-level —
    no event selector), distributed via the local executor, bare-output Snakefile."""
    analysis = synthetic_multisim_completed_isolated
    plot_id = "system_overview"
    cfg = tmp_path / f"{plot_id}.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"plot_id": plot_id, "renderer_kind": "system_overview", "output_format": "pdf"},
            sort_keys=False,
        )
    )

    result = analysis.static_plots(execution_mode="local", override_static_plot_configs=[cfg])

    assert result["success"] is True
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert (analysis_dir / "static_plots" / f"{plot_id}.pdf").exists()
    snakefile_static = (analysis_dir / "Snakefile.static").read_text()
    assert "report(" not in snakefile_static
    assert "rule render_report" not in snakefile_static


@pytest.mark.slow
@pytest.mark.requires_snakemake_subprocess
def test_static_plots_generates_sensitivity_benchmarking_figure(synthetic_sensitivity_completed_isolated, tmp_path):
    """Phase 4: static_plots() produces static_plots/{plot_id}.pdf via the publication
    static_cfg branch on the sensitivity-benchmarking renderer (the chart-shaped,
    no-colorbar case; KEEP-no-hybrid — Plotly branch untouched). The benchmarking
    x-axis variable rides the plot_id's var.{name} selector → --independent-var.
    Uses the sensitivity synth fixture (master analysis carries .sensitivity)."""
    analysis = synthetic_sensitivity_completed_isolated.master_analysis
    # n_devices is the synth benchmarking x-axis with data (see
    # test_synth_05_sensitivity_analysis_with_snakemake.py: benchmarking__n_devices.vs.total).
    plot_id = "sensitivity_benchmarking__var.n_devices"
    cfg = tmp_path / "sensitivity_benchmarking__var.n_devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"plot_id": plot_id, "renderer_kind": "sensitivity_benchmarking", "output_format": "pdf"},
            sort_keys=False,
        )
    )

    result = analysis.static_plots(execution_mode="local", override_static_plot_configs=[cfg])

    assert result["success"] is True
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert (analysis_dir / "static_plots" / f"{plot_id}.pdf").exists()
    snakefile_static = (analysis_dir / "Snakefile.static").read_text()
    assert "report(" not in snakefile_static
    assert "rule render_report" not in snakefile_static
