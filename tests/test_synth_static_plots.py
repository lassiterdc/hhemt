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
def test_static_plots_generates_publication_figure(synthetic_multisim_completed, tmp_path):
    """static_plots() produces static_plots/{plot_id}.pdf for the peak-flood-depth
    exemplar, distributed via the local executor, with a bare-output Snakefile."""
    analysis = synthetic_multisim_completed
    plot_id = _first_event_plot_id(analysis)
    cfg = _write_peak_flood_depth_config(tmp_path / f"{plot_id}.yaml", plot_id)

    result = analysis.static_plots(execution_mode="local", override_static_plot_configs=[cfg])

    assert result["success"] is True
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert (analysis_dir / "static_plots" / f"{plot_id}.pdf").exists()
    snakefile_static = (analysis_dir / "Snakefile.static").read_text()
    assert "report(" not in snakefile_static
    assert "rule render_report" not in snakefile_static
