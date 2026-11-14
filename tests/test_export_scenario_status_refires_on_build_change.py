"""A toolkit-build change re-fires the read-model rule and the pages that read its products.

PROBLEM. A re-run that only re-renders the report (a render-stage reprocess, a forced
render floor, a ``run()`` resume after a toolkit upgrade) moves no consolidation flag, so
Snakemake finds ``export_scenario_status`` up to date and the read-model
(``validation_report.json``) keeps the output of the OLD toolkit's validation code. The
fix declares the toolkit-build stamp (``_status/toolkit_build.stamp``) as a second input of
that rule, writes it compare-and-write from ``_write_snakefile_atomic`` whenever the
Snakefile being written declares it, and declares the rule's products as inputs of the two
pages that read them (``plot_metadata``, ``plot_workflow_performance``), because no
re-fire mechanism reaches an undeclared reader.

THE STATE THESE TESTS BUILD IS HAND-BUILT: a ``tmp_path`` tree with NO
``.snakemake/metadata`` record. So they assert only on triggers that are independent of the
record, run with ``--rerun-triggers mtime``: an input newer than an output. They say
nothing about a tree that has a record (``code``/``input``/``params`` triggers).

The rule blocks come from the REAL emitters (``_build_export_scenario_status_rule``,
``_build_plot_rule_block_metadata``, ``_build_plot_rule_block_workflow_performance``),
called on a duck-typed builder stand-in, so a change to what those emit is what these tests
read. No test here requests a ``synth_*`` fixture, compiles or runs a solver: they write
``tmp_path`` files and run ``snakemake -n`` only.

Tests authored here: each carries B4's four conditions and B5's two sides. The shared
facts: REACHABILITY -- every dry run reads files the test itself writes under ``tmp_path``
with a Snakefile the real emitters produced; BOTH-STATES -- every assertion turns on which
rules Snakemake schedules (a property of the emitted DAG edges), never on wording the change
introduces; INVOCATION ISOLATION -- the run resolves its Snakefile and ``--directory`` from
``tmp_path`` alone, so the tree the test runs from does not enter it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.workflow as wf

#: Literal duplicate of the production constant, on purpose: the behaviour arms must run on a
#: tree that has no ``_TOOLKIT_BUILD_STAMP_REL`` (the parent commit) so that they fail for the
#: reason claimed, not on an ImportError. ``test_constant_matches_the_literal_used_here``
#: ties the two together.
STAMP_REL = "_status/toolkit_build.stamp"
FLAG_REL = "_status/e_consolidate_complete.flag"

#: Files the export rule writes and the two pages read or sit downstream of.
_PRODUCTS = ("scenario_status.csv", "workflow_summary.md", "validation_report.json")
_PAGES = ("plots/metadata.html", "plots/workflow_performance.html")
_SCHEDULED = {"export_scenario_status", "plot_metadata", "plot_workflow_performance"}

_STAMP_A = {"hhemt_sha": "a" * 40, "hhemt_version": "1.0.0", "hhemt_dirty": "False"}
_STAMP_B = {"hhemt_sha": "b" * 40, "hhemt_version": "1.0.1", "hhemt_dirty": "False"}


class _BuilderStandIn:
    """The smallest object the three real emitters can run on.

    The emitters are plain functions of ``self``, so calling them unbound on this object
    exercises the shipped emission code and not a copy. ``_build_resource_block`` is the one
    collaborator stubbed with literal text, because its real body needs a full hpc config
    that has no bearing on the DAG edges under test.
    """

    python_executable = "python"
    cfg_analysis = SimpleNamespace(hpc_setup_and_analysis_processing_partition="standard")

    def __init__(self, analysis_dir: Path) -> None:
        self.analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=analysis_dir))

    def _get_config_args(self) -> str:
        return "--system-config s.yaml --analysis-config a.yaml"

    def _get_conda_env_path(self) -> Path:
        return Path("env.yaml")

    def _build_resource_block(self, **_kwargs) -> str:
        return '        slurm_partition="standard",\n        runtime=10,\n        mem_mb=1000'


def _rule_ctx() -> wf.RuleEmissionContext:
    return wf.RuleEmissionContext(
        python_executable="python",
        log_dir_rel="logs",
        conda_env_path="env.yaml",
        config_args_str="--system-config s.yaml --analysis-config a.yaml",
        is_sensitivity=False,
        static_backend="matplotlib",
    )


def _emitted_snakefile(tmp_path: Path) -> str:
    """The three real rule blocks plus a minimal consumer, as one Snakefile text."""
    builder = _BuilderStandIn(tmp_path)
    ctx = _rule_ctx()
    targets = ",\n        ".join(f'"{p}"' for p in ("scenario_status.csv", "workflow_summary.md", *_PAGES))
    rule_all = f"rule all:\n    input:\n        {targets}\n"
    return (
        rule_all
        + wf.SnakemakeWorkflowBuilder._build_plot_rule_block_metadata(builder, ctx=ctx)
        + wf.SnakemakeWorkflowBuilder._build_plot_rule_block_workflow_performance(builder, ctx=ctx)
        + wf.SnakemakeWorkflowBuilder._build_export_scenario_status_rule(builder)
    )


def _dry_run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """``snakemake -n`` against ``tmp_path``'s Snakefile, resolving everything from ``tmp_path``."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-n",
            "--cores",
            "1",
            "--snakefile",
            str(tmp_path / "Snakefile"),
            "--directory",
            str(tmp_path),
            "--rerun-triggers",
            "mtime",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )


def _scheduled_rules(proc: subprocess.CompletedProcess[str]) -> set[str]:
    """Rule names Snakemake planned to run (``rule X:`` headings of the dry-run job listing)."""
    return set(re.findall(r"^rule (\w+):", proc.stdout, re.M)) - {"all"}


def _age(path: Path, seconds_ago: float) -> None:
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


@pytest.fixture
def fixed_stamp(monkeypatch):
    """Pin ``producing_stamp`` so build identity is a test input, not a git read."""
    current = dict(_STAMP_A)
    monkeypatch.setattr("hhemt.provenance.producing_stamp", lambda: dict(current))
    return current


def _built_tree(tmp_path: Path) -> None:
    """A settled tree: consolidation flag and stamp OLD, every product and page NEWER.

    The Snakefile is written through the real ``_write_snakefile_atomic`` (which, on a tree
    that has the change, also writes the stamp). The stamp is then (re)written by hand so the
    same state is reachable on a tree that predates the writer, and aged together with the
    flag so nothing is stale by mtime.
    """
    wf._write_snakefile_atomic(tmp_path / "Snakefile", _emitted_snakefile(tmp_path))
    (tmp_path / "_status").mkdir(exist_ok=True)
    (tmp_path / FLAG_REL).write_text("")
    (tmp_path / STAMP_REL).write_text(json.dumps(_STAMP_A, sort_keys=True, separators=(",", ":")))
    for rel in (*_PRODUCTS, *_PAGES):
        out = tmp_path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("x")
        _age(out, 1000)
    for rel in (FLAG_REL, STAMP_REL):
        _age(tmp_path / rel, 2000)


def test_settled_tree_with_unchanged_stamp_schedules_nothing(tmp_path, fixed_stamp):
    """Property: a tree whose outputs are newer than the flag and the stamp plans no jobs.

    Class: REGRESSION on the over-fire side -- input class "build unchanged, outputs newer",
    which the design must keep a no-op so a resume of an unchanged build schedules nothing.
    Why required: a stamp that were rewritten (or declared in a way that always reads
    newer) on every submit would re-fire the read-model rule and re-render the report on
    every resume. What kills it: making ``_write_snakefile_atomic`` write the stamp
    unconditionally (new mtime each submit) -- see the unit arm below, which pins that
    mechanism -- or declaring the stamp as a ``params``/``always`` input.
    Second correct implementation under which it still passes: a stamp that lives outside
    ``_status/`` (e.g. beside the Snakefile) with the same compare-and-write rule.
    """
    _built_tree(tmp_path)
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _scheduled_rules(proc) == set(), proc.stdout


def test_stamp_content_change_schedules_export_and_both_readers(tmp_path, fixed_stamp):
    """Property: a changed build stamp re-fires the export rule AND the two pages that read it.

    Class: REGRESSION -- red on the parent commit, where the export rule has one input (the
    consolidation flag) and the two pages declare only the flag, so a newer stamp reaches
    nothing. Input class: any build change that rewrites the stamp's content; this input
    is the position "stamp newer than every output, flag older".
    Reachability: the stamp, flag and outputs are real files under ``tmp_path`` and the
    Snakefile is the real emitters' text; Snakemake's mtime trigger reads exactly them.
    Both-states anchor: the assertion turns on the DAG edges (which rule is scheduled), a
    property of the old and the new emission alike.
    Why required: the read-model must be recomputed whenever the code that computes it moved;
    the readers must follow because they open the products and have no other edge to them.
    What kills it: dropping the stamp from the export rule's inputs, or dropping either
    reader's ``additional_inputs`` -- the second is a change that leaves ``export_scenario_status``
    scheduled and a reader stale, which is why the readers are asserted by name.
    Second correct implementation: the stamp declared as a ``named`` input
    (``build = "_status/toolkit_build.stamp"``) instead of a positional one.
    """
    _built_tree(tmp_path)
    (tmp_path / STAMP_REL).write_text(json.dumps(_STAMP_B, sort_keys=True, separators=(",", ":")))
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _scheduled_rules(proc) == _SCHEDULED, proc.stdout


def test_a_newer_product_alone_schedules_both_readers(tmp_path, fixed_stamp):
    """Property: each reader is downstream of the export rule's products by a declared edge.

    Class: NEW CAPABILITY -- the plausible wrong implementation is "stamp declared on the
    export rule only": it passes the previous test's export assertion and leaves the readers
    unscheduled. The input here removes the stamp from the picture (stamp and flag old, only
    ``scenario_status.csv`` and ``validation_report.json`` made newer than the pages) so the
    only path to a re-render is the reader declarations.
    Reachability: both products are real files the readers now declare as inputs.
    Both-states anchor: asserts which rules are scheduled, not any new wording.
    Why required: with no declared edge Snakemake classifies a page up to date no matter how
    the read-model moved. What kills it: removing ``additional_inputs`` from either page.
    Second correct implementation: the pages declare ``rules.export_scenario_status.output``
    by reference instead of by file name.
    """
    _built_tree(tmp_path)
    # Pages stay NEWER than the flag and the stamp (so nothing but a product edge can
    # schedule them); the two products the pages read become the newest files in the tree.
    for rel in ("scenario_status.csv", "validation_report.json"):
        _age(tmp_path / rel, 100)
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {"plot_metadata", "plot_workflow_performance"} <= _scheduled_rules(proc), proc.stdout


def test_export_rule_declares_the_stamp_and_the_pages_declare_the_products():
    """Property: the emitted rule text names the stamp and the product inputs, per rule.

    Class: NEW CAPABILITY -- plausible wrong implementation: the stamp written but never
    declared (the file exists, nothing depends on it); red under it because the rule text
    would not name the path. Reachability: the strings come from the real emitters. Both-states
    anchor: the check is on the rule's ``input:`` block of the rule named, found by rule name,
    so it holds for any wording around the block. Why required: a written-but-undeclared stamp
    is inert. What kills it: removing the declaration from the export emitter.
    Second correct implementation: the same inputs emitted as labelled keyword inputs.
    """
    text = _emitted_snakefile(Path("."))
    blocks = {m.group(1): m.group(2) for m in re.finditer(r"^rule (\w+):\n(.*?)(?=^rule |\Z)", text, re.M | re.S)}
    inputs = {name: block.split("output:")[0] for name, block in blocks.items()}
    assert STAMP_REL in inputs["export_scenario_status"]
    assert "scenario_status.csv" in inputs["plot_metadata"]
    assert "validation_report.json" in inputs["plot_metadata"]
    assert "scenario_status.csv" in inputs["plot_workflow_performance"]
    assert STAMP_REL not in inputs["plot_metadata"], "the stamp is the export rule's input, not the reader's"


def test_constant_matches_the_literal_used_here():
    """Property: the production constant equals the path the behaviour arms hard-code.

    Class: NEW CAPABILITY -- plausible wrong implementation: a constant renamed away from
    ``_status/toolkit_build.stamp`` while the tests above keep passing on a literal.
    Reachability: reads the module attribute. Both-states: compares two strings. Why required:
    the file name is part of the on-disk layout a resumed tree already holds. What kills it:
    renaming the constant's value.
    Second correct implementation: the constant derived from a ``Path`` join with the same result.
    """
    assert wf._TOOLKIT_BUILD_STAMP_REL == STAMP_REL


def test_compare_and_write_through_the_atomic_writer(tmp_path, monkeypatch):
    """Property: the atomic Snakefile writer creates the stamp, leaves it alone when the build
    is unchanged, rewrites it when the build changes, and writes none for a Snakefile that
    does not declare it.

    Class: NEW CAPABILITY, four positions of the input "which build, which content". The
    plausible wrong implementations and the arm each is red under: no writer (arm 1);
    an unconditional write (arm 2 -- the mtime is aged first so a rewrite is visible whatever
    the clock granularity); a stamp that is written once and never refreshed (arm 3); a
    stamp written for every Snakefile whatever it declares (arm 4).
    Reachability: the writer is called directly with real content and a real directory.
    Both-states anchor: asserts file existence, bytes and ``st_mtime_ns`` -- observable on the
    parent as well (where arm 1 reds). Why required: an unconditional write bumps the mtime
    on every submit and re-fires the read-model rule on every resume of an unchanged build.
    Second correct implementation: an ``os.link``-free write that compares a content hash
    stored in the stamp's own first line.
    """
    current = dict(_STAMP_A)
    monkeypatch.setattr("hhemt.provenance.producing_stamp", lambda: dict(current))
    declaring = f'rule x:\n    input:\n        "{STAMP_REL}",\n    output: "y"\n'
    stamp = tmp_path / STAMP_REL

    # Arm 1: a Snakefile that declares the stamp gets a stamp, holding a JSON document that round-trips to the build
    # (the serialization layout is not specified, so none is asserted).
    wf._write_snakefile_atomic(tmp_path / "Snakefile", declaring)
    assert json.loads(stamp.read_text()) == current

    # Arm 2: same build, second write -> bytes and mtime untouched (mtime aged so a rewrite shows).
    _age(stamp, 5000)
    before = (stamp.read_bytes(), stamp.stat().st_mtime_ns)
    wf._write_snakefile_atomic(tmp_path / "Snakefile", declaring)
    assert (stamp.read_bytes(), stamp.stat().st_mtime_ns) == before

    # Arm 3: different build -> rewritten (new bytes, newer mtime).
    current.update(_STAMP_B)
    wf._write_snakefile_atomic(tmp_path / "Snakefile", declaring)
    assert json.loads(stamp.read_text()) == _STAMP_B
    assert stamp.stat().st_mtime_ns > before[1]

    # Arm 4: a Snakefile that does not declare it (the form a delete Snakefile has) writes none.
    other = tmp_path / "other"
    wf._write_snakefile_atomic(other / "Snakefile.delete", 'rule d:\n    output: "z"\n')
    assert not (other / STAMP_REL).exists()


def test_launch_without_the_toolkit_guard_resolves_the_stamp(tmp_path, fixed_stamp):
    """Property: a Snakefile written by the real writer dry-runs with a bare ``snakemake -n``,
    with no ``_pre_snakemake_invocation_guards`` in front of it.

    Class: NEW CAPABILITY -- this is the slurm-mode and validation-dry-run shape the guard
    never reaches. The plausible wrong implementation is the stamp written by the pre-launch
    guard instead of the Snakefile writer: this arm is red under it because an unguarded
    launch finds the declared stamp absent. The consolidation flag is seeded because the export rule and both
    pages declare it and no rule here produces it.
    Reachability: the dry run reads the Snakefile and the files the writer produced.
    Both-states anchor: asserts the exit status and, in the twin below, the NAME of the
    missing input. Why required: a writer-side stamp is what makes every launch path resolve it.
    What kills it: moving the stamp write out of ``_write_snakefile_atomic``.
    Second correct implementation: the stamp written by a helper the writer calls, in any module.
    """
    (tmp_path / "_status").mkdir()
    (tmp_path / FLAG_REL).write_text("")
    wf._write_snakefile_atomic(tmp_path / "Snakefile", _emitted_snakefile(tmp_path))
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_launch_without_a_stamp_fails_loudly_naming_it(tmp_path, fixed_stamp):
    """Property: with the stamp genuinely absent at planning, a bare dry run exits non-zero
    and NAMES the stamp.

    Class: NEW CAPABILITY, the must-fire twin of the arm above. It varies the property under
    suspicion (is the stamp present when Snakemake plans) and not any implementation token:
    the Snakefile is written by the real writer, then the stamp FILE is deleted from disk,
    so the arm goes red only when the stamp is absent and is indifferent to which helper,
    in which module, wrote it. That also makes it consistent with the sibling arm's second
    correct implementation (a helper of any name the writer calls). Asserting the
    missing-input line names ``_status/toolkit_build.stamp`` -- not merely a
    MissingInputException -- keeps the red from coming from the flag seed, which is present.
    Reachability: the dry run reads the real emitted Snakefile. Both-states anchor: the
    message names a path that the rule declaration carries under any emission that declares
    it. Why required: a missing stamp must be loud, never a silent no-op. What kills it: an
    implementation that makes the stamp optional (e.g. ``ancient()``) -- then this exits 0.
    Second correct implementation: a different loud failure that still names the path.
    """
    (tmp_path / "_status").mkdir()
    (tmp_path / FLAG_REL).write_text("")
    wf._write_snakefile_atomic(tmp_path / "Snakefile", _emitted_snakefile(tmp_path))
    (tmp_path / STAMP_REL).unlink()
    proc = _dry_run(tmp_path)
    assert proc.returncode != 0
    assert STAMP_REL in proc.stdout + proc.stderr
