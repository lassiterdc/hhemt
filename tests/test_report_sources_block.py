"""A caption's Sources block is read from the figure's manifest, the same way on every route.

The Sources parameter of a plot rule is a callable of the rule's output that Snakemake
evaluates while it builds the report, after the figure and its manifest exist. The
workflow route and the bundle route emit the same callable, for rules with and without
wildcards, and no rerun trigger names ``params``, because Snakemake records the value it
computed when it planned the run (no manifest yet) and would otherwise rerun every
manifest-reading plot rule.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest

from hhemt import workflow as wf
from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile

TESTS = Path(__file__).resolve().parent
BUNDLES = TESTS / "fixtures" / "bundles"
WORKFLOW_PY = Path(wf.__file__)
_LAMBDA = re.compile(r"(?m)^\s*source_paths_rst = (lambda w, output: .+),$")


def _ctx() -> wf.RuleEmissionContext:
    return wf.RuleEmissionContext(
        python_executable="python",
        log_dir_rel="logs",
        conda_env_path="env.yaml",
        config_args_str="",
        is_sensitivity=False,
        static_backend="matplotlib",
    )


def _rule_block(text: str, rule_name: str) -> str:
    start = text.index(f"rule {rule_name}:")
    nxt = text.find("\nrule ", start + 1)
    return text[start : nxt if nxt != -1 else len(text)]


def _callable(rule_text: str):
    found = _LAMBDA.search(rule_text)
    assert found, "the rule has no manifest-reading Sources parameter"
    return found.group(1), eval(found.group(1))  # noqa: S307 -- the emitted lambda under test


def _paths(rst: str) -> set[str]:
    return set(re.findall(r"^- ``([^`]+)``$", rst, re.M))


def _source_side_rule(rule_name: str, renderer: str, template: str, wildcards: tuple[str, ...]) -> str:
    spec = wf.RuleSpec(
        rule_name=rule_name,
        renderer_module=renderer,
        input_flags=(),
        output_path_template=template,
        source_paths=(),
        wildcards=wildcards,
        extra_cli_flags=(),
        extra_params=(),
        report_kwargs={"caption": "report/captions/x.rst", "category": "C", "labels": '{"figure": "F"}'},
        resources_yaml="mem_mb=1",
        log_path_template="logs/x.log",
        sources_from_manifest=True,
    )
    return wf._emit_plot_rule(spec, _ctx())


# (fixture, rule, renderer, output template, wildcards, a figure file whose manifest the fixture holds)
_CASES = [
    pytest.param(
        "multi_sim",
        "plot_system_overview",
        "system_overview",
        "plots/system_overview__OUTPUT_EXT__",
        (),
        "plots/system_overview.png",
        id="no_wildcards",
    ),
    pytest.param(
        "multi_sim",
        "plot_per_sim_event_page",
        "per_sim_event_page",
        "plots/per_sim/{event_id}/event_page__evt.{event_id}__OUTPUT_EXT__",
        ("event_id",),
        "plots/per_sim/event_index.0/peak_flood_depth.png",
        id="wildcards",
    ),
]


@pytest.mark.parametrize(("fixture", "rule", "renderer", "template", "wildcards", "figure"), _CASES)
def test_both_routes_give_the_same_sources_block(
    tmp_path: Path, fixture, rule, renderer, template, wildcards, figure
) -> None:
    bundle = tmp_path / fixture
    shutil.copytree(BUNDLES / fixture, bundle)
    bundle_text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    bundle_expr, bundle_fn = _callable(_rule_block(bundle_text, rule))
    source_expr, source_fn = _callable(_source_side_rule(rule, renderer, template, wildcards))

    output = [str(bundle / figure)]
    declared = set(
        json.loads((bundle / figure).with_suffix("").with_suffix(".manifest.json").read_text())["source_paths_relative"]
    )
    assert declared, "the fixture manifest declares no source, so the comparison would be vacuous"
    assert bundle_expr == source_expr
    assert _paths(bundle_fn(None, output)) == _paths(source_fn(None, output)) == declared


def test_the_block_is_empty_while_the_manifest_is_absent(tmp_path: Path) -> None:
    """Snakemake evaluates the parameter when it plans the run, before the figure exists."""
    assert wf._sources_rst_from_manifest(str(tmp_path / "plots" / "missing.png")) == ""


def test_a_manifest_holding_no_list_gives_an_empty_block(tmp_path: Path) -> None:
    (tmp_path / "f.manifest.json").write_text("[]")
    assert wf._sources_rst_from_manifest(str(tmp_path / "f.png")) == ""


def test_the_manifest_is_found_for_a_plot_id_containing_dots(tmp_path: Path) -> None:
    (tmp_path / "peak.member.1__evt.2.manifest.json").write_text(json.dumps({"source_paths_relative": ["a.zarr"]}))
    assert _paths(wf._sources_rst_from_manifest(str(tmp_path / "peak.member.1__evt.2.png"))) == {"a.zarr"}


def test_no_source_side_rule_declares_a_hand_written_source_list() -> None:
    """Every rule spec in workflow.py reads its Sources block from the manifest."""
    tree = ast.parse(WORKFLOW_PY.read_text(encoding="utf-8"))
    specs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "RuleSpec"]
    assert specs
    bad = []
    for call in specs:
        kw = {k.arg: k.value for k in call.keywords}
        empty = isinstance(kw.get("source_paths"), ast.Tuple) and not kw["source_paths"].elts
        manifest = (
            isinstance(kw.get("sources_from_manifest"), ast.Constant) and kw["sources_from_manifest"].value is True
        )
        if "source_paths" in kw and not (empty and manifest):
            bad.append(call.lineno)
    assert bad == []


# ---- No rerun trigger may name `params` -----------------------------------------------------


def _trigger_lists(source: str) -> list[list[str]]:
    """Every rerun-trigger list in a module: the profile's ``rerun-triggers`` entry and each
    ``"--rerun-triggers"`` option followed by its values in a command list."""
    tree = ast.parse(source)
    found: list[list[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and key.value == "rerun-triggers" and isinstance(value, ast.List):
                    found.append([e.value for e in value.elts if isinstance(e, ast.Constant)])
        if isinstance(node, ast.List):
            items = node.elts
            for i, item in enumerate(items):
                if isinstance(item, ast.Constant) and item.value == "--rerun-triggers":
                    values = []
                    for nxt in items[i + 1 :]:
                        if (
                            isinstance(nxt, ast.Constant)
                            and isinstance(nxt.value, str)
                            and not nxt.value.startswith("-")
                        ):
                            values.append(nxt.value)
                        else:
                            break
                    found.append(values)
    return found


def test_the_generated_profile_and_the_command_lists_name_four_trigger_lists() -> None:
    assert len(_trigger_lists(WORKFLOW_PY.read_text(encoding="utf-8"))) == 4


def test_no_rerun_trigger_list_names_params() -> None:
    lists = _trigger_lists(WORKFLOW_PY.read_text(encoding="utf-8"))
    assert [t for t in lists if "params" in t] == []


def test_a_rerun_trigger_list_naming_params_is_caught() -> None:
    source = WORKFLOW_PY.read_text(encoding="utf-8")
    mutated = source.replace('"rerun-triggers": ["mtime"],', '"rerun-triggers": ["mtime", "params"],', 1)
    assert mutated != source
    assert [t for t in _trigger_lists(mutated) if "params" in t] == [["mtime", "params"]]
