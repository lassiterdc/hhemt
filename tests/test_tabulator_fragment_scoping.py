"""Every table fragment's script must be SELF-SCOPED, so a multi-table page still parses.

WHY THIS EXISTS, stated as the failure it pins rather than as the rule it asserts.

`TableFragment.script` bodies are concatenated into ONE `<script>` by every multi-table
page (`metadata.py`'s Reproduction Guide joins three; `workflow_performance.py` joins the
run-timeline and SLURM-efficiency tables). Each body opened with a top-level
``const tableOptions`` / ``__trfTable`` / ``__trfColumnGroups``. Two bodies in one scope is
a redeclaration, which is a PARSE-time SyntaxError -- and a classic script that fails to
parse executes NONE of its statements, including the `_TRF_MOUNT_JS` at the head of that
same block that clones the `<template>` markup into the DOM.

So the page did not render "an unstyled table". It rendered NOTHING, on every report, for
two consecutive iterations, while every byte-level acceptance probe passed: the bundle was
present, the controls were present, the data rows were present, and none of it ever ran.
A grep reads the TEXT of code; a parse error is a property of the code AS A PROGRAM.

Two layers, deliberately:

* `test_fragment_script_is_self_scoped` is structural and ALWAYS runs. It needs no
  JavaScript engine, so it holds on any CI runner.
* `test_concatenated_fragments_parse` is the real check and needs `node`. It compiles the
  concatenation with ``new Function``, which parses without executing -- no browser, no
  DOM, no install. It SKIPS where node is absent rather than passing vacuously, because a
  check that cannot run must not report success.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from hhemt.report_renderers._tabulator_defaults import (
    build_table_fragment,
    tabulator_shared_js,
)

_IIFE_OPEN = "(function () {\n"
_IIFE_CLOSE = "})();\n"


def _fragments(n: int = 3):
    """Build `n` fragments the way a multi-table page does, alternating the sidebar path.

    `column_panel` is alternated because the emitter takes a DIFFERENT branch for each
    value -- the `column_panel=False` path binds its controls document-scoped instead of
    sidebar-scoped -- and both branches declare bindings the wrap has to enclose.
    """
    options = {
        "data": [{"a": 1, "b": "x"}],
        "columns": [{"title": "A", "field": "a"}, {"title": "B", "field": "b"}],
        "persistence": {"sort": True},
    }
    return [
        build_table_fragment(
            container_id=f"tbl-{i}",
            options=dict(options, persistenceID=f"tbl-{i}"),
            renderer_name="test_tabulator_fragment_scoping",
            column_panel=(i % 2 == 0),
        )
        for i in range(n)
    ]


def test_fragment_script_is_self_scoped() -> None:
    """Each body opens and closes its own scope, on BOTH column_panel branches."""
    for frag in _fragments():
        assert frag.script.startswith(_IIFE_OPEN), (
            "fragment script does not open its own scope; concatenating two of these "
            "redeclares their top-level consts and blanks the page"
        )
        assert frag.script.endswith(_IIFE_CLOSE), "fragment script does not close its own scope"


def test_each_fragment_declares_tableoptions_exactly_once() -> None:
    """The binding whose collision caused the outage is still per-fragment, not shared."""
    for frag in _fragments():
        assert frag.script.count("const tableOptions =") == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to parse JS")
def test_concatenated_fragments_parse() -> None:
    """THE REGRESSION. Three fragments plus the shared JS must compile as one script.

    Fails on the pre-fix emitter with `SyntaxError: Identifier 'tableOptions' has already
    been declared` -- verified against the delivered artifact, where 16 of 841 emitted
    scripts carried exactly that error and every one of them was a multi-table page.
    """
    combined = tabulator_shared_js() + "".join(f.script for f in _fragments())
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "concat.js"
        js.write_text(combined)
        probe = Path(tmp) / "probe.js"
        # argv[2], NOT argv[1]. Under `node probe.js target.js` argv[1] is probe.js
        # itself, which parses fine -- so an argv[1] probe reads ITSELF and reports
        # success for any target. That made both node tests here vacuous until
        # `test_the_probe_itself_can_fail` caught it.
        probe.write_text(
            "const fs=require('fs');\n"
            "try { new Function(fs.readFileSync(process.argv[2],'utf8')); }\n"
            "catch (e) { console.error(e.message); process.exit(1); }\n"
        )
        result = subprocess.run(["node", str(probe), str(js)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"concatenated fragment scripts do not parse: {result.stderr.strip()}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to parse JS")
def test_the_probe_itself_can_fail() -> None:
    """A parse check that cannot fail proves nothing; this pins that it can.

    Without this, `test_concatenated_fragments_parse` is indistinguishable from a probe
    whose node invocation silently succeeds on any input.
    """
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "bad.js"
        js.write_text("const dup = 1;\nconst dup = 2;\n")
        probe = Path(tmp) / "probe.js"
        # argv[2], NOT argv[1]. Under `node probe.js target.js` argv[1] is probe.js
        # itself, which parses fine -- so an argv[1] probe reads ITSELF and reports
        # success for any target. That made both node tests here vacuous until
        # `test_the_probe_itself_can_fail` caught it.
        probe.write_text(
            "const fs=require('fs');\n"
            "try { new Function(fs.readFileSync(process.argv[2],'utf8')); }\n"
            "catch (e) { console.error(e.message); process.exit(1); }\n"
        )
        result = subprocess.run(["node", str(probe), str(js)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 1
    assert "already been declared" in result.stderr
