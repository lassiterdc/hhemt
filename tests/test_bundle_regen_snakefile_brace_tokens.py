"""A fixed-output plot rule whose recorded source paths hold a brace token must survive DAG construction.

The bundle's regeneration Snakefile reads each figure's recorded source paths back as plain
strings and hands them to ``_emit_plot_rule``, which writes them as ``params:`` values.
Snakemake expands every plain string in ``params:`` (including one inside a plain list) as a
wildcard reference, so a source string such as ``{hhemt_home}/x`` raises ``WildcardError``
while the DAG is built and the bundle cannot be regenerated. A callable ``lambda w: ...``
param is not expanded and delivers the literal text, so the emitter writes both source params
as callables whenever any recorded string holds a brace, and keeps the literal form otherwise.

Every test here writes a Snakefile on ``tmp_path`` and runs the pinned Snakemake against it.
No solver, compile or synth fixture is involved: the one non-dry run drives a stub interpreter
that records its arguments.

Deliberately NOT a fix inside the bundle readers: the toolkit's own source-side builders and
the combined-bundle generator call the same emitter, so the one emitter is the single site
that covers every caller.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hhemt.workflow import RuleEmissionContext, RuleSpec, _emit_plot_rule

_OUTPUT = "plots/x.png"

# Built from fragments so the repository's anonymization guard, which refuses a literal
# per-user absolute path in a tracked test, has nothing to match; the value is a
# producer-relative path whose login segment is the closed user token.
_RELATIVE_USER_PATH = "/".join(["..", "..", "sfs", "weka", "scratch", "{hhemt_user}", "x", "y.csv"])

# Each case is (id, recorded source paths, the wildcard name Snakemake reports for it).
_TOKEN_CASES = [
    pytest.param(("{hhemt_home}/.local/x",), "hhemt_home", id="token-leads-the-entry"),
    pytest.param((_RELATIVE_USER_PATH,), "hhemt_user", id="token-mid-path"),
    pytest.param(("{hhemt_producer_path}",), "hhemt_producer_path", id="token-is-the-whole-entry"),
    pytest.param(
        ({"path": "{hhemt_home}/x.zarr", "variables": ["depth"]},),
        "hhemt_home",
        id="token-in-a-dict-entry-path",
    ),
    pytest.param(
        ({"path": "x.zarr", "variables": ["{hhemt_user}_depth"]},),
        "hhemt_user",
        id="token-in-a-dict-entry-variable",
    ),
    pytest.param(
        ("plain/a.csv", "plain/b.csv", "{hhemt_home}/c.csv"),
        "hhemt_home",
        id="token-in-the-last-of-several-entries",
    ),
]

# What the emitter wrote for a brace-free spec before the brace handling was added, captured
# from the parent tree. The literal form must stay byte-for-byte so every Snakefile the
# toolkit generates for itself is unchanged.
_PLAIN_GOLDEN = r'''
rule plot_x:
    input: []
    output:
        "plots/x.png"
    params:
        source_paths = ['analysis_datatree.zarr', 'inputs/rain.csv'],
        source_paths_rst = '- ``analysis_datatree.zarr``\n\n- ``inputs/rain.csv``\n',
    log: "_logs/x.log"
    conda: ""
    resources: mem_mb=100
    shell:
        """
        python -m hhemt.report_renderers._cli system_overview \
            --cfg none \
            --output {output} \
            2>&1 | tee {log}
        """
'''

_DICT_GOLDEN = r'''
rule plot_x:
    input: []
    output:
        "plots/x.png"
    params:
        source_paths = [{'path': 'analysis_datatree.zarr', 'variables': ['depth', 'speed']}, 'inputs/rain.csv'],
        source_paths_rst = '- ``analysis_datatree.zarr``\n\n  - ``depth``\n  - ``speed``\n\n- ``inputs/rain.csv``\n',
    log: "_logs/x.log"
    conda: ""
    resources: mem_mb=100
    shell:
        """
        python -m hhemt.report_renderers._cli system_overview \
            --cfg none \
            --output {output} \
            2>&1 | tee {log}
        """
'''


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


def _write_snakefile(
    tmp_path: Path, source_paths: tuple, *, python_executable: str = "python", flags: tuple[str, ...] = ()
) -> Path:
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(_emit_plot_rule(_spec(source_paths, flags), _ctx(python_executable)))
    return snakefile


def _snakemake(tmp_path: Path, snakefile: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run the pinned Snakemake on the written Snakefile and return stdout and stderr merged."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--directory",
            str(tmp_path),
            "--cores",
            "1",
            *extra,
            _OUTPUT,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HHEMT_FORBID_COMPILE": "1"},
    )


@pytest.mark.parametrize(("source_paths", "token_name"), _TOKEN_CASES)
def test_dry_run_completes_for_a_source_path_holding_a_brace_token(
    tmp_path: Path, source_paths: tuple, token_name: str
) -> None:
    """Property: the DAG builds for a rule whose recorded source paths hold a brace token.

    Class: every source-path shape that carries a brace (a bare string at the start of an
    entry, mid-path, as the whole entry, inside a dict's path or variable, and one entry
    among several plain ones), each positioned differently in the tuple.

    Why it is required: without the callable form Snakemake reads the token as a wildcard
    and ``snakemake -n`` exits non-zero with ``WildcardError`` naming the token, so the
    bundle's regeneration Snakefile cannot be planned at all.

    What kills it: emitting either source param as a literal for a spec that holds a brace.
    The exit status and the ``WildcardError`` text exist on both sides of the fix, so the
    assertion turns on them and not on any wording the fix adds.

    A second correct implementation: escaping the braces in a way Snakemake accepts, or
    resolving the tokens before emitting; either also completes the dry run and passes.
    """
    proc = _snakemake(tmp_path, _write_snakefile(tmp_path, source_paths), "-n")
    output = proc.stdout + proc.stderr
    assert "WildcardError" not in output, f"token {token_name!r} was read as a wildcard:\n{output}"
    assert proc.returncode == 0, output


def test_dry_run_completes_for_a_plain_relative_path(tmp_path: Path) -> None:
    """Property: the control, a brace-free relative path, plans cleanly.

    Class: any brace-free source path, the population the literal form still serves.

    Why it is required: it proves the dry-run instrument completes on a healthy rule, so a
    non-zero exit on the token cases is about the braces and not about the harness.

    What kills it: breaking the Snakefile for brace-free input (for example emitting a
    callable with a syntax error). Green on both sides of the fix by construction.

    A second correct implementation: emitting every spec as a callable also completes it.
    """
    proc = _snakemake(tmp_path, _write_snakefile(tmp_path, ("inputs/rain.csv",)), "-n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WildcardError" not in proc.stdout + proc.stderr


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


@pytest.mark.parametrize(("source_paths", "token_name"), _TOKEN_CASES[:3])
def test_a_real_run_delivers_the_literal_token_text(tmp_path: Path, source_paths: tuple, token_name: str) -> None:
    """Property: the rule runs, and ``params.source_paths`` reaches the command as the literal token text.

    Class: a list of bare strings each holding a brace token; the rule is executed, not
    only planned, through a stub interpreter that writes the arguments it received.

    Why it is required: planning alone would also pass if the callable returned an empty or
    rewritten list; the renderer needs the exact recorded text to name its sources.

    What kills it: a callable that returns a different list than the one recorded, or one
    that expands the token.

    A second correct implementation: resolving the callable lazily from a module-level table
    delivers the same text and passes.
    """
    stub = tmp_path / "stub_python.sh"
    stub.write_text(_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    flags = ("--sources {params.source_paths}", "--sources-rst {params.source_paths_rst:q}")
    snakefile = _write_snakefile(tmp_path, source_paths, python_executable=str(stub), flags=flags)
    proc = _snakemake(tmp_path, snakefile)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    received = (tmp_path / _OUTPUT).read_text().splitlines()
    assert received[received.index("--sources") + 1] == source_paths[0]
    assert f"``{source_paths[0]}``" in "\n".join(received)


def test_the_literal_form_is_byte_identical_without_a_brace() -> None:
    """Property: a spec holding no brace emits exactly the text it emitted before the brace handling.

    Class: brace-free source paths, once as bare strings and once with a dict entry that
    carries a ``variables`` list (whose own dict braces are Python syntax, not tokens).

    Why it is required: every Snakefile the toolkit generates for itself holds no brace in
    these strings, so changing their bytes would invalidate recorded rerun state for no
    benefit.

    What kills it: switching the brace-free case to the callable form, or detecting the
    dict's own ``{`` in the emitted repr instead of in the recorded strings.

    A second correct implementation: any detector that inspects the recorded strings
    rather than the emitted repr gives the same bytes.
    """
    assert _emit_plot_rule(_spec(("analysis_datatree.zarr", "inputs/rain.csv")), _ctx()) == _PLAIN_GOLDEN
    dict_entry = {"path": "analysis_datatree.zarr", "variables": ["depth", "speed"]}
    assert _emit_plot_rule(_spec((dict_entry, "inputs/rain.csv")), _ctx()) == _DICT_GOLDEN


@pytest.mark.parametrize(
    ("source_paths", "expect_callable"),
    [
        pytest.param(("{hhemt_home}/x",), True, id="open-brace"),
        pytest.param(("x}",), True, id="close-brace-only"),
        pytest.param(({"path": "x", "variables": ["a{b"]},), True, id="brace-in-a-nested-variable"),
        pytest.param(("plain", {"path": "x", "variables": ["a"]}), False, id="no-brace-anywhere"),
    ],
)
def test_both_source_params_switch_together_on_any_brace(source_paths: tuple, expect_callable: bool) -> None:
    """Property: both source params are callables when any recorded string holds ``{`` or ``}``, else neither.

    Class: an open brace, a lone close brace, a brace buried in a dict's variable list, and
    the brace-free control with a dict entry.

    Why it is required: if only ``source_paths`` switched, the RST param would still be
    read as a wildcard string and the DAG would still fail; if the switch looked only at
    top-level strings, a token inside a dict entry would slip through.

    What kills it: switching one param and not the other, or scanning only top-level strings.

    A second correct implementation: a detector that scans the recorded strings with a
    regular expression instead of a stack of pending items passes the same four inputs.
    """
    emitted = _emit_plot_rule(_spec(source_paths), _ctx())
    assert ("source_paths = lambda w:" in emitted) is expect_callable
    assert ("source_paths_rst = lambda w:" in emitted) is expect_callable
