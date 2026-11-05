"""Reviewer tests: a recorded source path holding a brace token must not reach Snakemake as a plain string.

Every test writes on ``tmp_path`` and runs only the pinned Snakemake (a dry run, or one stub-interpreter
run). No solver, compile or ``synth_*`` fixture is involved.

The reviewer's tests differ from the coder's in what they position. The coder's inputs are three
one-entry tuples, a dict entry and one three-entry tuple whose token is last; these tests add the
class the bundle readers actually feed the emitter (a JSON list of strings, with the token in the
first, the middle and the last place, repeated, and beside a quote, a backslash or a non-ASCII
character), the two real callers (``bundle.snakefile_generator`` on a copied bundle fixture, and
``bundle.combined_snakefile_generator`` through its public entry point), and the brace-free
population at bundle scale.

The tokens are built from fragments so the anonymization guard has nothing to match in a tracked
test.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import hhemt.bundle.combined_snakefile_generator as combined_generator
from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile
from hhemt.report_renderers._figure_emission import format_sources_rst
from hhemt.workflow import RuleEmissionContext, RuleSpec, _emit_plot_rule

_FIXTURES = Path(__file__).parent / "fixtures" / "bundles"
_HOME = "{" + "hhemt_home" + "}"
_USER = "{" + "hhemt_user" + "}"


def _ctx(python_executable: str = "python") -> RuleEmissionContext:
    return RuleEmissionContext(
        python_executable=python_executable,
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str="--cfg none",
        is_sensitivity=False,
        static_backend="matplotlib",
    )


def _spec(source_paths: tuple, extra_cli_flags: tuple[str, ...] = ()) -> RuleSpec:
    return RuleSpec(
        rule_name="plot_x",
        renderer_module="system_overview",
        input_flags=(),
        output_path_template="plots/x__OUTPUT_EXT__",
        source_paths=source_paths,
        wildcards=(),
        extra_cli_flags=extra_cli_flags,
        extra_params=(),
        report_kwargs=None,
        resources_yaml="mem_mb=100",
        log_path_template="_logs/x.log",
    )


def _snakemake(directory: Path, snakefile: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--directory",
            str(directory),
            "--cores",
            "1",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _plan_text(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


# --------------------------------------------------------------------------------------
# Class: the JSON list of strings a sidecar's ``source_paths_relative`` carries.
# --------------------------------------------------------------------------------------

_SIDECAR_LISTS = [
    pytest.param((_HOME + "/a", "plain/b.csv", "plain/c.csv"), id="token-first-of-three"),
    pytest.param(("plain/a.csv", _HOME + "/b", "plain/c.csv"), id="token-middle-of-three"),
    pytest.param(
        ("plain/a.csv", "plain/b.csv", "../../sfs/weka/scratch/" + _USER + "/c.csv"), id="token-last-of-three"
    ),
    pytest.param((_HOME + "/a", "plain/b", _USER + "/c"), id="two-tokens-and-a-plain"),
    pytest.param(("plain/a.csv", "it's/" + _HOME + '/"q"'), id="token-beside-a-quote-pair"),
    pytest.param(("plain/a.csv", "back\\slash/" + _HOME), id="token-beside-a-backslash"),
    pytest.param(("café/" + _HOME,), id="token-beside-a-non-ascii-character"),
]


@pytest.mark.parametrize("source_paths", _SIDECAR_LISTS)
def test_dry_run_plans_a_sidecar_list_with_a_token_at_any_position(tmp_path: Path, source_paths: tuple) -> None:
    """Property: ``snakemake -n`` plans a plot rule whose recorded list holds a brace token wherever it sits.

    Input class: a flat list of strings (the only shape ``_load_source_paths`` returns), the token
    first, middle or last, repeated, or beside a quote, a backslash or a non-ASCII character.

    Why required: Snakemake reads each plain string of a ``params:`` list as a wildcard reference,
    so on the parent every one of these raises WildcardError while the DAG is built.

    Kill: leaving either source param as a literal; looking at the first entry only; looking at
    the last entry only.

    Second correct implementation: resolving or escaping the tokens before emission also plans.
    """
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(_emit_plot_rule(_spec(source_paths), _ctx()))
    proc = _snakemake(tmp_path, snakefile, "-n", "plots/x.png")
    assert proc.returncode == 0, _plan_text(proc)
    assert "WildcardError" not in _plan_text(proc)


_STUB = """#!/bin/sh
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--output" ]; then out="$a"; fi
  prev="$a"
done
mkdir -p "$(dirname "$out")"
printf '%s\\n' "$@" > "$out"
"""

_SINGLE_ENTRY_TEXT = [
    pytest.param(_HOME + "/.local/x", id="plain-token"),
    pytest.param("a'b\"c" + _HOME, id="single-and-double-quote"),
    pytest.param("a\\b" + _HOME, id="backslash"),
    pytest.param("café/" + _USER, id="non-ascii"),
]


@pytest.mark.parametrize("entry", _SINGLE_ENTRY_TEXT)
def test_a_real_run_delivers_the_recorded_text_of_both_params_unchanged(tmp_path: Path, entry: str) -> None:
    """Property: a rule receives the recorded text in ``params.source_paths`` and ``params.source_paths_rst``.

    Input class: one recorded string holding a brace token together with a character that
    ``repr`` must escape (a quote, a backslash) or that is non-ASCII; the rule is executed through
    a stub interpreter that writes the arguments it received.

    Why required: the emitter writes the value through ``repr`` into Snakefile source, so the
    delivered text equals the recorded text only if the repr round trip holds for these characters.

    Kill: a callable returning a rewritten, unescaped or truncated value; emitting the
    value through a format that drops a backslash.

    Second correct implementation: resolving the callable from a module-level table delivers the
    same text.
    """
    stub = tmp_path / "stub.sh"
    stub.write_text(_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    flags = ("--sources {params.source_paths:q}", "--sources-rst {params.source_paths_rst:q}")
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(_emit_plot_rule(_spec((entry,), flags), _ctx(str(stub))))
    proc = _snakemake(tmp_path, snakefile, "plots/x.png")
    assert proc.returncode == 0, _plan_text(proc)
    received = (tmp_path / "plots" / "x.png").read_text().split("\n")
    assert received[received.index("--sources") + 1] == entry
    assert f"- ``{entry}``" in "\n".join(received[received.index("--sources-rst") + 1 :])


# --------------------------------------------------------------------------------------
# Class: the two real callers of the emitter.
# --------------------------------------------------------------------------------------


def test_the_regeneration_snakefile_of_a_bundle_whose_sidecar_holds_tokens_plans(tmp_path: Path) -> None:
    """Property: ``generate_regeneration_snakefile`` on a bundle with token sidecars writes a plannable Snakefile.

    Input class: a copy of the checked-in multi-sim bundle fixture whose ``system_overview``
    sidecar has two token entries prepended to its recorded sources, as a bundle emitted with the
    placeholder table would hold; the Snakefile is planned with ``--forceall`` so every job is built.

    Why required: the bundle's regeneration Snakefile is the one consumer that reads sidecar
    strings back; if its emitter path bypassed the changed branch the bundle would still fail to
    plan.

    Kill: a fix placed anywhere other than the one emitter this caller calls; a detector that
    never fires for a list read from JSON.

    Second correct implementation: substituting the tokens before ``_emit_plot_rule`` is called.
    """
    bundle = tmp_path / "multi_sim"
    shutil.copytree(_FIXTURES / "multi_sim", bundle)
    sidecar = bundle / "plots" / "system_overview.manifest.json"
    data = json.loads(sidecar.read_text())
    data["source_paths_relative"] = [
        _HOME + "/.local/x",
        "../../sfs/weka/scratch/" + _USER + "/y.csv",
        *data["source_paths_relative"],
    ]
    sidecar.write_text(json.dumps(data))
    snakefile = bundle / "Snakefile"
    snakefile.write_text(generate_regeneration_snakefile(bundle, static_backend="matplotlib"))
    proc = _snakemake(bundle, snakefile, "-n", "--forceall", "plots/system_overview.png")
    assert proc.returncode == 0, _plan_text(proc)
    assert "WildcardError" not in _plan_text(proc)


def test_the_combined_generator_writes_a_snakefile_that_plans_for_a_token_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Property: ``generate_combined_snakefile`` plans when one harvested rule holds a token source.

    Input class: the public entry point of the combined generator with its two harvest functions
    replaced by one rule whose recorded list holds a token (a combined bundle needs child crates
    that this test does not build; the replaced step is the sidecar read, not the emission).

    Why required: the combined generator is the second caller; the brief asks whether it reaches
    the changed branch or has another path that still writes a plain string param.

    Kill: a combined-side emission that bypasses ``_emit_plot_rule``.

    Second correct implementation: the same fix in a shared helper both generators call.
    """
    spec = _spec((_HOME + "/.local/x", "plain/b.csv"))
    monkeypatch.setattr(combined_generator, "_harvest_combined_rule_specs", lambda bundle_root: (spec,))
    monkeypatch.setattr(combined_generator, "_harvest_per_experiment_rule_specs", lambda bundle_root: ())
    body, _ = combined_generator.generate_combined_snakefile(tmp_path)
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(body)
    proc = _snakemake(tmp_path, snakefile, "-n")
    assert proc.returncode == 0, _plan_text(proc)
    assert "WildcardError" not in _plan_text(proc)


def test_snakemake_report_renders_a_token_source_into_a_caption(tmp_path: Path) -> None:
    """Property: ``snakemake --report`` builds a report from a figure rule whose recorded sources hold a token.

    Input class: a figure rule with ``report()`` kwargs and a caption that renders
    ``snakemake.params.source_paths_rst``, one recorded entry holding a token; the figure file
    exists (the bundle's rules are pre-rendered), so the report engine, not a rule shell, is
    what reads the params.

    Why required: the regeneration and combined Snakefiles exist to feed ``--report``; a param
    that plans under ``-n`` but is unreadable by the report engine would still lose the report.

    Kill: any param form the report engine cannot evaluate; a callable that returns a value the
    caption template cannot render.

    Second correct implementation: resolving the tokens before emission also builds the report.
    """
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    spec = RuleSpec(
        rule_name="plot_x",
        renderer_module="system_overview",
        input_flags=(),
        output_path_template="plots/x__OUTPUT_EXT__",
        source_paths=(_HOME + "/.local/x", "plain/y.csv"),
        wildcards=(),
        extra_cli_flags=(),
        extra_params=(),
        report_kwargs={"caption": "cap.rst", "category": "C", "labels": '{"figure": "x"}'},
        resources_yaml="mem_mb=100",
        log_path_template="_logs/x.log",
    )
    (tmp_path / "Snakefile").write_text('rule all:\n    input: "plots/x.png"\n' + _emit_plot_rule(spec, _ctx()))
    (tmp_path / "cap.rst").write_text("Sources:\n\n{{ snakemake.params.source_paths_rst }}\n")
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "x.png").write_bytes(png)
    proc = _snakemake(tmp_path, tmp_path / "Snakefile", "--report", "r.html")
    assert proc.returncode == 0, _plan_text(proc)
    assert _HOME[1:-1] in (tmp_path / "r.html").read_text()


# --------------------------------------------------------------------------------------
# Held tests: green on the parent and green on the change; each must stay green.
# --------------------------------------------------------------------------------------

_BRACE_FREE = [
    pytest.param(("plain/a.csv",), id="one-plain"),
    pytest.param((), id="empty"),
    pytest.param(("it's/a\"b.csv",), id="quotes"),
    pytest.param(("back\\slash/a.csv",), id="backslash"),
    pytest.param(("café/a.csv", "plain/b.csv"), id="non-ascii"),
    pytest.param(({"path": "x.zarr", "variables": ["depth", "speed"]}, "plain/b.csv"), id="dict-entry"),
    pytest.param(({"path": "x.zarr", "variables": []},), id="dict-entry-no-variables"),
]


@pytest.mark.parametrize("source_paths", _BRACE_FREE)
def test_held_a_brace_free_spec_writes_both_params_as_the_plain_repr(source_paths: tuple) -> None:
    """Held. Property: with no brace in any recorded string the two params are the bare ``repr`` values.

    Input class: brace-free specs whose values need ``repr`` escaping (quotes, backslash), are
    non-ASCII, are empty, or are dict entries whose own ``repr`` holds ``{`` (Python syntax, not a
    recorded token).

    Why required: every Snakefile the toolkit writes for itself falls in this class; a callable
    here changes their bytes, and a detector that scanned the emitted repr would take a dict
    entry's own brace for a token.

    Kill (the named mutation that reds it): emit the callable form unconditionally; detect on the
    emitted repr instead of on the recorded strings.

    Second correct implementation: a detector walking the recorded strings by regular expression
    writes the same two lines.
    """
    text = _emit_plot_rule(_spec(source_paths), _ctx())
    assert f"        source_paths = {list(source_paths)!r},\n" in text
    assert f"        source_paths_rst = {format_sources_rst(list(source_paths))!r},\n" in text
    assert "lambda w:" not in text


@pytest.mark.parametrize("fixture", ["multi_sim", "sensitivity_master"])
def test_held_an_unmodified_bundle_fixture_writes_the_manifest_callable_form(fixture: str, tmp_path: Path) -> None:
    """Held. Property: regenerating the unmodified checked-in bundles writes the manifest-callable form, and no other.

    The form pinned follows develop's ``fa98781e``: ``bundle.snakefile_generator`` passes
    ``source_paths=()`` and ``sources_from_manifest=True`` for every rule, so ``_emit_plot_rule``
    writes one ``source_paths_rst = lambda w, output:`` line naming ``_sources_rst_from_manifest``
    and no ``source_paths`` param. The combined generator is a different caller that still
    reaches the brace-token branch and is covered by the tests above.

    Input class: both checked-in bundle fixtures, every plot rule of the regenerated Snakefile.

    Why required: the sources block now comes from the manifest beside the figure, so a plot rule
    that carried a literal or a recorded-string callable instead would put the sidecar's text
    (with any brace token) back into the Snakefile.

    Kill (the named mutations that red it): emit the recorded-string callable form
    (``source_paths = lambda w: ...``) unconditionally; emit the old literal
    ``source_paths = [...]`` form.

    Second correct implementation: any emitter that writes the manifest callable on every plot
    rule and no ``source_paths`` param passes.
    """
    bundle = tmp_path / fixture
    shutil.copytree(_FIXTURES / fixture, bundle)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    plot_rules = [block for block in re.split(r"(?m)^(?=rule )", text) if block.startswith("rule plot_")]
    assert plot_rules, "the regenerated Snakefile holds no plot rule, so the per-rule check below would be vacuous"
    manifest_line = re.compile(
        r"^        source_paths_rst = lambda w, output: .*_sources_rst_from_manifest\(output\[0\]\),$", re.M
    )
    for block in plot_rules:
        assert manifest_line.search(block), block.split("\n", 1)[0]
    assert not re.search(r"^        source_paths = ", text, re.M)
    assert not re.search(r"^        source_paths(_rst)? = lambda w:", text, re.M)
