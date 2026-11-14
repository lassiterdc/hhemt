"""Reviewer tests for chunk 2a.12: the regeneration Snakefile carries the manifest callable; the combined one does not.

Develop's ``fa98781e`` made ``bundle.snakefile_generator`` pass ``source_paths=()`` and
``sources_from_manifest=True`` for every rule, so ``_emit_plot_rule`` writes one
``source_paths_rst = lambda w, output:`` line per plot rule and no ``source_paths`` param. The
combined generator (``bundle.combined_snakefile_generator``) still passes a harvested tuple and no
``sources_from_manifest``, so it keeps the literal-or-brace form. The coder's restated held test
reads the regenerated text of both checked-in bundle fixtures for the manifest line, the absence of
``source_paths =`` and the absence of the brace callable. These tests position what that reading
does not: an extra source param beside the manifest line, what the written line DOES when
evaluated, a sidecar holding a token, and the combined generator's own output.

Every test writes on ``tmp_path`` only. No solver, compile or ``synth_*`` fixture is involved. Tokens
are built from fragments so the anonymization guard has nothing to match in a tracked test.
"""

from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path

import pytest

from hhemt.bundle import combined_snakefile_generator as combined_generator
from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile
from hhemt.report_renderers._figure_emission import format_sources_rst

_FIXTURES = Path(__file__).parent / "fixtures" / "bundles"
_HOME = "{" + "hhemt_home" + "}"

_SOURCE_PARAM = re.compile(r"^        source_paths\w* = ", re.M)
_MANIFEST_LINE = re.compile(
    r"^        source_paths_rst = lambda w, output: .*_sources_rst_from_manifest\(output\[0\]\),$", re.M
)


def _rule_blocks(text: str) -> list[str]:
    """The Snakefile split into one string per ``rule`` statement (the preamble is dropped)."""
    return [b for b in re.split(r"(?m)^(?=rule )", text) if b.startswith("rule ")]


def _regenerate(fixture: str, tmp_path: Path) -> tuple[Path, str]:
    """Copy a checked-in bundle fixture to ``tmp_path`` and regenerate its Snakefile."""
    bundle = tmp_path / fixture
    shutil.copytree(_FIXTURES / fixture, bundle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bundle, generate_regeneration_snakefile(bundle, static_backend="matplotlib")


@pytest.mark.parametrize("fixture", ["multi_sim", "sensitivity_master"])
def test_each_regenerated_rule_that_carries_a_source_param_carries_exactly_one_and_it_is_the_manifest_callable(
    fixture: str, tmp_path: Path
) -> None:
    """Property: per rule of the regenerated Snakefile, the ``source_paths*`` params are exactly the one manifest line.

    Input class: every rule block of the regenerated Snakefile of both checked-in bundle fixtures,
    selected by the params it carries and not by its name.

    Why required: the shape pins ONE line per rule and no ``source_paths`` param. The coder's held
    test checks that the manifest line is present on each ``plot_`` rule and that no ``source_paths =``
    or brace-callable line exists, so a second ``source_paths_rst`` line, or one named
    ``source_paths_extra``, beside the manifest line passes it. Snakemake refuses a repeated
    keyword at parse time, so such a Snakefile fails the report build.

    Kill: an emitter that writes the manifest line and a second ``source_paths_rst = "..."`` literal
    line (mutation F of the reviewer's file).

    Second correct implementation: an emitter that writes the manifest callable with a different
    ``__import__`` spelling, or on a different line of the params block, passes.
    """
    _, text = _regenerate(fixture, tmp_path)
    blocks = _rule_blocks(text)
    carrying = [b for b in blocks if _SOURCE_PARAM.search(b)]
    plot_rules = [b for b in blocks if b.startswith("rule plot_")]
    assert plot_rules, "the regenerated Snakefile holds no plot rule, so the checks below would be vacuous"
    assert len(carrying) == len(plot_rules), "a plot rule lacks a source param, or a non-plot rule carries one"
    for block in carrying:
        lines = _SOURCE_PARAM.findall(block)
        assert len(lines) == 1, (block.split("\n", 1)[0], lines)
        assert _MANIFEST_LINE.search(block), block.split("\n", 1)[0]


def _manifest_callable(text: str):
    """The ``source_paths_rst`` callable written on the first plot rule, evaluated from the Snakefile text."""
    block = next(b for b in _rule_blocks(text) if b.startswith("rule plot_"))
    line = _MANIFEST_LINE.search(block)
    assert line is not None
    right_hand_side = line.group(0).split(" = ", 1)[1].rstrip(",")
    return eval(right_hand_side)  # noqa: S307 - the text is the toolkit's own emission, read back to run it


@pytest.mark.parametrize("fixture", ["multi_sim", "sensitivity_master"])
def test_the_written_manifest_callable_returns_the_sources_block_of_the_manifest_beside_the_figure(
    fixture: str, tmp_path: Path
) -> None:
    """Property: the written callable returns ``format_sources_rst`` of the sidecar's recorded sources.

    Input class: the callable of the first plot rule of each checked-in fixture, called with the
    output path of a figure whose ``.manifest.json`` sits beside it in the copied bundle.

    Why required: the coder's held test reads the callable's text, and the text names
    ``_sources_rst_from_manifest(output[0])`` whatever that helper returns. The shape pins the form
    because the report's Sources block is read from the manifest at report time; a helper that
    reads another key, or a line whose callee is not that helper, yields an empty block with every
    text assertion still true.

    Kill: ``_sources_rst_from_manifest`` reading ``source_paths`` in place of ``source_paths_relative``
    (mutation P of the reviewer's file).

    Second correct implementation: a callable in another module that reads the same sidecar key and
    formats it with ``format_sources_rst`` passes.
    """
    bundle, text = _regenerate(fixture, tmp_path)
    sidecar = bundle / "plots" / "system_overview.manifest.json"
    recorded = json.loads(sidecar.read_text())["source_paths_relative"]
    expected = format_sources_rst(list(recorded))
    assert recorded and expected, "the fixture sidecar records no source, so the comparison below would be vacuous"
    got = _manifest_callable(text)({}, [str(bundle / "plots" / "system_overview.png")])
    assert got == expected


def test_a_sidecar_holding_a_token_reaches_the_report_through_the_callable_and_never_through_the_snakefile_text(
    tmp_path: Path,
) -> None:
    """Property: a recorded brace-token source is absent from the Snakefile text and present in the callable's result.

    Input class: a copy of the multi-sim fixture whose ``system_overview`` sidecar has a token entry
    prepended to its recorded sources, as a bundle emitted with the placeholder table holds.

    Why required: the shape's reason for the manifest callable is that nothing the sidecar holds
    is written into the Snakefile, where Snakemake would read a brace token as a wildcard. The
    form is only as good as both halves: no token text in the written Snakefile, and the token text
    delivered literally when the callable runs.

    Kill: a regenerator that passes the sidecar's sources through as ``source_paths`` with
    ``sources_from_manifest`` off (mutation K of the reviewer's file); a helper that drops or
    rewrites a source holding a brace.

    Second correct implementation: a callable that reads the sidecar and returns the formatted block
    from a different helper passes.
    """
    bundle = tmp_path / "multi_sim"
    shutil.copytree(_FIXTURES / "multi_sim", bundle)
    sidecar = bundle / "plots" / "system_overview.manifest.json"
    data = json.loads(sidecar.read_text())
    data["source_paths_relative"] = [_HOME + "/.local/x", *data["source_paths_relative"]]
    sidecar.write_text(json.dumps(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    assert _HOME not in text
    got = _manifest_callable(text)({}, [str(bundle / "plots" / "system_overview.png")])
    assert _HOME + "/.local/x" in got


def test_the_combined_generator_keeps_the_literal_or_brace_form_and_never_the_manifest_callable(tmp_path: Path) -> None:
    """Property: the combined generator writes a callable for a token source, a literal otherwise, no manifest line.

    Input class: ``generate_combined_snakefile`` on a bundle root that holds one combined-set
    figure's sidecar with a token entry and a plain entry; every other combined figure has no
    sidecar and so an empty source list. No monkeypatch: the real harvest builds the specs.

    Why required: the shape says the combined generator keeps 2a.7's literal-or-brace form and is
    untouched, and that a combined bundle whose sidecar holds a token still depends on the brace
    callable. The coder's docstring says the same of the tests above it in the file; those tests
    build their specs by hand or replace the harvest, so none of them reaches the combined
    generator's own ``RuleSpec`` construction.

    Kill: the brace branch of ``_emit_plot_rule`` never writing a callable (mutation G); the combined
    generator marking its specs ``sources_from_manifest=True`` (mutation N).

    Second correct implementation: a combined generator that delivers the token text through a
    callable of another spelling (``lambda wildcards:``) passes, as does a different literal repr.
    """
    first = combined_generator._harvest_combined_rule_specs(tmp_path)[0]
    sidecar = tmp_path / first.output_path_template.replace("__OUTPUT_EXT__", ".manifest.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"source_paths_relative": [_HOME + "/x", "plain/a.csv"]}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        body, _ = combined_generator.generate_combined_snakefile(tmp_path)
    assert "_sources_rst_from_manifest" not in body
    blocks = {b.split(":", 1)[0].removeprefix("rule "): b for b in _rule_blocks(body)}
    token_rule = blocks[first.rule_name]
    assert re.search(r"^        source_paths = lambda \w+: \[", token_rule, re.M)
    assert re.search(r"^        source_paths_rst = lambda \w+: ", token_rule, re.M)
    plain = [b for name, b in blocks.items() if name.startswith("plot_") and name != first.rule_name]
    assert plain, "the combined set has no second plot rule, so the literal arm below would be vacuous"
    for block in plain:
        assert re.search(r"^        source_paths = \[", block, re.M), block.split("\n", 1)[0]
        assert not re.search(r"^        source_paths\w* = lambda", block, re.M), block.split("\n", 1)[0]
