"""Snakefile rule_all / render_report ↔ rule output extension symmetry test.

Locks in the invariant that every path declared as input to `rule all` and
`rule render_report` is produced by exactly one rule's `output:` declaration
in the same Snakefile, for both `static_backend` values.

Authored after the plotly-default extension-mismatch bug. See the originating
plan ``plotly default snakefile extension fix.md`` under
``library/docs/planning/projects/hhemt/bugs/`` (active) or
``.../bugs/completed/`` (after closeout).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pytest

_RULE_BLOCK_RE = re.compile(r"^rule\s+([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE)
_INPUT_PATH_LINE_RE = re.compile(r'"([^"]+\.(?:png|svg|html))"')
_PLOT_EXT_RE = re.compile(r"\.(png|svg|html|zip)(?=$|\b)")


@dataclass(frozen=True)
class StructuralDiff:
    rule_name: str
    kind: str
    detail: str


def _strip_plot_ext(path: str) -> str:
    return _PLOT_EXT_RE.sub(".{ext}", path)


def _rule_structure(snakefile_text: str) -> dict[str, str]:
    """Per-rule output template with plot extensions stripped to '.{ext}'.

    Structural skeleton used to compare two backends' Snakefiles: rule-name set
    and extension-agnostic output template must match across backends; only the
    concrete extension is permitted to differ.
    """
    outputs = _parse_rule_outputs(snakefile_text)
    return {name: _strip_plot_ext(path) for name, path in outputs.items()}


def _structural_diff(text_a: str, text_b: str) -> list[StructuralDiff]:
    """Per-rule differences between two Snakefiles classified by kind.

    Kinds: `missing_rule` (rule absent in one side), `structure_mismatch`
    (rule present in both but extension-stripped template differs),
    `extension_swap` (same extension-stripped template, different concrete
    extension — the legal cross-backend difference).
    """
    struct_a = _rule_structure(text_a)
    struct_b = _rule_structure(text_b)
    outputs_a = _parse_rule_outputs(text_a)
    outputs_b = _parse_rule_outputs(text_b)
    diffs: list[StructuralDiff] = []
    for rule in sorted(set(struct_a) | set(struct_b)):
        if rule not in struct_a:
            diffs.append(StructuralDiff(rule, "missing_rule", f"absent in A; B={struct_b[rule]}"))
        elif rule not in struct_b:
            diffs.append(StructuralDiff(rule, "missing_rule", f"absent in B; A={struct_a[rule]}"))
        elif struct_a[rule] != struct_b[rule]:
            diffs.append(StructuralDiff(rule, "structure_mismatch", f"A={struct_a[rule]}, B={struct_b[rule]}"))
        elif outputs_a[rule] != outputs_b[rule]:
            diffs.append(StructuralDiff(rule, "extension_swap", f"A={outputs_a[rule]}, B={outputs_b[rule]}"))
    return diffs


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


_SECTION_KEYWORDS = frozenset({
    "input", "output", "params", "log", "conda", "resources", "shell", "run",
    "threads", "priority", "retries", "message", "benchmark", "cache",
    "wildcard_constraints", "group", "envmodules", "container", "notebook", "script",
})


def _parse_rule_inputs(snakefile_text: str, rule_name: str) -> list[str]:
    """Return list of input plot/report paths declared by `rule <name>:`.

    Line-based parser (Phase 2 fix): the prior implementation used a regex with
    `(?:\\s+.*\\n)+?` + multi-alternation lookahead, which catastrophically
    backtracked (~26 s per `re.search`, 2990 s per call) on Snakefiles emitted by
    the `skip_run=True` builder fixture. This implementation scans line-by-line
    with no backtracking and returns the same items the old regex did.
    """
    lines = snakefile_text.splitlines()
    # 1. Locate the rule header (top-of-line, exact name match).
    header_idx: int | None = None
    for i, line in enumerate(lines):
        m = _RULE_BLOCK_RE.match(line)
        if m and m.group(1) == rule_name:
            header_idx = i
            break
    if header_idx is None:
        return []
    # 2. Walk forward from the header until the next top-level boundary.
    in_input = False
    inputs: list[str] = []
    for line in lines[header_idx + 1:]:
        if line.startswith("rule ") or line.startswith("onsuccess:") or line.startswith("onerror:"):
            break  # next rule / hook — end of this rule's body
        stripped = line.strip()
        if not in_input:
            if stripped.startswith("input:"):
                in_input = True
                after = stripped[len("input:"):].strip()
                if after:
                    inputs.extend(_INPUT_PATH_LINE_RE.findall(after))
            continue
        first_token = stripped.split(":", 1)[0] if ":" in stripped else ""
        if first_token in _SECTION_KEYWORDS - {"input"}:
            break  # next sibling section — end of input block
        inputs.extend(_INPUT_PATH_LINE_RE.findall(line))
    return inputs


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
    assert consumer_inputs, (
        f"`rule {consumer_rule}` parsed to 0 input paths — parser likely degenerate; "
        f"refusing to vacuously pass symmetry check."
    )
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


_BACKENDS: tuple[Literal["matplotlib", "plotly"], ...] = ("matplotlib", "plotly")


def _assert_multisim_symmetry(target: str, builder, monkeypatch) -> None:
    """Phase 2 shared helper: generate the multisim Snakefile for both static
    backends in-process via the session-scope builder fixture, then assert
    (a) per-Snakefile producibility for `target` consumer rule, (b) cross-backend
    extension-agnostic rule structure equality, (c) only `extension_swap`
    diffs between the two backends."""
    snakefiles = {b: _generate_multisim_snakefile_text(builder, b, monkeypatch) for b in _BACKENDS}
    for backend, text in snakefiles.items():
        _assert_symmetry(text, consumer_rule=target)
    assert _rule_structure(snakefiles["matplotlib"]) == _rule_structure(snakefiles["plotly"]), (
        "multisim Snakefile rule_structure (extension-stripped) differs across "
        "static backends"
    )
    non_ext_swap = [d for d in _structural_diff(snakefiles["matplotlib"], snakefiles["plotly"]) if d.kind != "extension_swap"]
    assert not non_ext_swap, (
        "multisim Snakefile structural diffs include non-extension-swap kinds:\n  "
        + "\n  ".join(f"{d.rule_name}: {d.kind} ({d.detail})" for d in non_ext_swap)
    )


def _assert_sensitivity_master_symmetry(target: str, builder, monkeypatch) -> None:
    """As `_assert_multisim_symmetry` but for the sensitivity-master Snakefile."""
    snakefiles = {b: _generate_sensitivity_master_snakefile_text(builder, b, monkeypatch) for b in _BACKENDS}
    for backend, text in snakefiles.items():
        _assert_symmetry(text, consumer_rule=target)
    assert _rule_structure(snakefiles["matplotlib"]) == _rule_structure(snakefiles["plotly"]), (
        "sensitivity-master Snakefile rule_structure (extension-stripped) differs "
        "across static backends"
    )
    non_ext_swap = [d for d in _structural_diff(snakefiles["matplotlib"], snakefiles["plotly"]) if d.kind != "extension_swap"]
    assert not non_ext_swap, (
        "sensitivity-master Snakefile structural diffs include non-extension-swap kinds:\n  "
        + "\n  ".join(f"{d.rule_name}: {d.kind} ({d.detail})" for d in non_ext_swap)
    )


def test_multisim_rule_all_input_symmetry(synth_multi_sim_builder, monkeypatch):
    _assert_multisim_symmetry("all", synth_multi_sim_builder, monkeypatch)


def test_multisim_render_report_input_symmetry(synth_multi_sim_builder, monkeypatch):
    _assert_multisim_symmetry("render_report", synth_multi_sim_builder, monkeypatch)


def test_sensitivity_master_rule_all_input_symmetry(synth_sensitivity_builder, monkeypatch):
    _assert_sensitivity_master_symmetry("all", synth_sensitivity_builder, monkeypatch)


def test_sensitivity_master_render_report_input_symmetry(synth_sensitivity_builder, monkeypatch):
    _assert_sensitivity_master_symmetry("render_report", synth_sensitivity_builder, monkeypatch)


def test_plotly_chart_renderers_emit_html_extension():
    # Every renderer whose Plotly column differs from its matplotlib column
    # is a chart figure emitted via pio.to_html and must resolve to .html so
    # Snakemake's report engine dispatches via <iframe> (text/html). A .svg
    # extension here would dispatch via <img> (image/svg+xml) and fail to
    # parse the Plotly HTML content as SVG XML.
    from hhemt.workflow import _OUTPUT_EXT_BY_RENDERER
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
