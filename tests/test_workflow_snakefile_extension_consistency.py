"""Snakefile rule_all / render_report ↔ rule output extension symmetry test.

Locks in the invariant that every path declared as input to `rule all` and
`rule render_report` is produced by exactly one rule's `output:` declaration
in the same Snakefile, for both `static_backend` values.

Authored after the plotly-default extension-mismatch bug. See the originating
plan ``plotly default snakefile extension fix.md`` under
``library/docs/planning/projects/TRITON-SWMM_toolkit/bugs/`` (active) or
``.../bugs/completed/`` (after closeout).
"""

from __future__ import annotations

import re
from typing import Literal

import pytest

_RULE_BLOCK_RE = re.compile(r"^rule\s+([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE)
_INPUT_PATH_LINE_RE = re.compile(r'"([^"]+\.(?:png|svg|html))"')


def _parse_rule_outputs(snakefile_text: str) -> dict[str, str]:
    """Return {rule_name: output_path_with_wildcards} for every rule in the file.

    Reads each `rule <name>:` block, finds its `output:` block, extracts the
    first quoted path (either via `report("path", ...)` wrapper or bare quoted
    literal). Returns the rule_all / render_report rules along with every plot
    rule. Caller filters as needed.
    """
    rule_outputs: dict[str, str] = {}
    rule_starts = [(m.start(), m.group(1)) for m in _RULE_BLOCK_RE.finditer(snakefile_text)]
    rule_starts.append((len(snakefile_text), "__END__"))
    for i in range(len(rule_starts) - 1):
        start, name = rule_starts[i]
        end, _ = rule_starts[i + 1]
        block = snakefile_text[start:end]
        out_match = re.search(
            r"^\s*output:\s*\n((?:\s+.*\n)+?)(?=\s*(?:input:|params:|log:|conda:|resources:|shell:|run:|threads:|priority:|retries:|message:|benchmark:|cache:|wildcard_constraints:|group:|envmodules:|container:|notebook:|script:|onsuccess:|onerror:|rule\s|$))",
            block,
            re.MULTILINE,
        )
        if not out_match:
            continue
        output_block = out_match.group(1)
        # Prefer `report(...)` wrapper's first arg, else bare quoted literal.
        rpt = re.search(r'report\(\s*"([^"]+)"', output_block)
        if rpt:
            rule_outputs[name] = rpt.group(1)
            continue
        bare = re.search(r'"([^"]+\.(?:png|svg|html|zip))"', output_block)
        if bare:
            rule_outputs[name] = bare.group(1)
    return rule_outputs


def _parse_rule_inputs(snakefile_text: str, rule_name: str) -> list[str]:
    """Return list of input plot/report paths declared by `rule <name>:`.

    Extracts every quoted path ending in .png/.svg/.html from the rule's
    `input:` block, including paths inside `expand(...)` calls (the first
    quoted string of each expand is the pattern).
    """
    rule_match = re.search(
        rf"^rule\s+{re.escape(rule_name)}:\s*\n(?P<body>(?:\s+.*\n)+?)(?=^rule\s|^onsuccess:|^onerror:|\Z)",
        snakefile_text,
        re.MULTILINE,
    )
    if not rule_match:
        return []
    body = rule_match.group("body")
    in_match = re.search(
        r"^\s*input:\s*\n((?:\s+.*\n)+?)(?=\s*(?:output:|params:|log:|conda:|resources:|shell:|run:|threads:|priority:|retries:|message:|benchmark:|cache:|wildcard_constraints:|group:))",
        body,
        re.MULTILINE,
    )
    if not in_match:
        return []
    input_block = in_match.group(1)
    return _INPUT_PATH_LINE_RE.findall(input_block)


def _output_pattern_to_regex(path: str) -> re.Pattern[str]:
    """Convert a Snakemake output template `{wildcard}` into a regex that
    matches any literal substitution. Used so an output template like
    `analysis_report.{format}` matches concrete inputs like
    `analysis_report.zip` and `analysis_report.html`. Wildcard pattern
    inputs (`expand("plots/per_sim/{event_id}/...")`) reduce to the same
    template form, so the regex also matches identically-wildcarded inputs."""
    escaped = re.escape(path)
    pattern = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", escaped)
    return re.compile(f"^{pattern}$")


def _assert_symmetry(snakefile_text: str, consumer_rule: str) -> None:
    """Every input path of `consumer_rule` must be producible by at least
    one rule's output template (wildcards in the output template match any
    literal substitution; literal-in-input + wildcard-in-output is a
    legal Snakemake DAG edge)."""
    outputs = _parse_rule_outputs(snakefile_text)
    output_patterns = [
        (name, _output_pattern_to_regex(p)) for name, p in outputs.items() if name not in (consumer_rule, "all")
    ]
    consumer_inputs = _parse_rule_inputs(snakefile_text, consumer_rule)
    unmatched: list[str] = []
    for inp in consumer_inputs:
        if not any(pat.match(inp) for _name, pat in output_patterns):
            unmatched.append(inp)
    assert not unmatched, (
        f"`rule {consumer_rule}` declares {len(unmatched)} input path(s) "
        f"with no producing rule output template:\n  " + "\n  ".join(unmatched)
    )


def _generate_multisim_snakefile_text(analysis, static_backend: Literal["matplotlib", "plotly"], monkeypatch) -> str:
    builder = analysis._workflow_builder  # type: ignore[attr-defined]
    monkeypatch.setattr(builder, "_get_report_cfg_static_backend", lambda: static_backend)
    return builder.generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=False,
        prepare_scenarios=True,
        process_timeseries=True,
    )


def _generate_sensitivity_master_snakefile_text(
    sa, static_backend: Literal["matplotlib", "plotly"], monkeypatch
) -> str:
    builder = sa.sensitivity._workflow_builder  # type: ignore[attr-defined]
    base = builder._base_builder
    monkeypatch.setattr(base, "_get_report_cfg_static_backend", lambda: static_backend)
    return builder.generate_master_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=False,
        prepare_scenarios=True,
        process_timeseries=True,
    )


@pytest.mark.parametrize("static_backend", ["matplotlib", "plotly"])
def test_multisim_rule_all_input_symmetry(synth_multi_sim_analysis, monkeypatch, static_backend):
    """rule all inputs in the multisim Snakefile must be produced by some rule output."""
    text = _generate_multisim_snakefile_text(synth_multi_sim_analysis, static_backend, monkeypatch)
    _assert_symmetry(text, consumer_rule="all")


@pytest.mark.parametrize("static_backend", ["matplotlib", "plotly"])
def test_multisim_render_report_input_symmetry(synth_multi_sim_analysis, monkeypatch, static_backend):
    """render_report inputs in the multisim Snakefile must be produced by some rule output."""
    text = _generate_multisim_snakefile_text(synth_multi_sim_analysis, static_backend, monkeypatch)
    _assert_symmetry(text, consumer_rule="render_report")


@pytest.mark.parametrize("static_backend", ["matplotlib", "plotly"])
def test_sensitivity_master_rule_all_input_symmetry(synth_sensitivity_analysis, monkeypatch, static_backend):
    """rule all inputs in the sensitivity-master Snakefile must be produced by some rule output."""
    text = _generate_sensitivity_master_snakefile_text(synth_sensitivity_analysis, static_backend, monkeypatch)
    _assert_symmetry(text, consumer_rule="all")


@pytest.mark.parametrize("static_backend", ["matplotlib", "plotly"])
def test_sensitivity_master_render_report_input_symmetry(synth_sensitivity_analysis, monkeypatch, static_backend):
    """render_report inputs in the sensitivity-master Snakefile must be produced by some rule output."""
    text = _generate_sensitivity_master_snakefile_text(synth_sensitivity_analysis, static_backend, monkeypatch)
    _assert_symmetry(text, consumer_rule="render_report")


def test_plotly_chart_renderers_emit_html_extension():
    # Every renderer whose Plotly column differs from its matplotlib column
    # is a chart figure emitted via pio.to_html and must resolve to .html so
    # Snakemake's report engine dispatches via <iframe> (text/html). A .svg
    # extension here would dispatch via <img> (image/svg+xml) and fail to
    # parse the Plotly HTML content as SVG XML.
    from TRITON_SWMM_toolkit.workflow import _OUTPUT_EXT_BY_RENDERER
    for renderer, exts in _OUTPUT_EXT_BY_RENDERER.items():
        mpl_ext = exts["matplotlib"]
        plotly_ext = exts["plotly"]
        if mpl_ext == plotly_ext:
            continue
        assert plotly_ext == ".html", (
            f"chart renderer {renderer!r} maps plotly -> {plotly_ext!r}; "
            f"expected '.html' so Snakemake's mime detection dispatches to "
            f"<iframe> via text/html."
        )
