"""Page statements about code, pinned to the code, for the corrections that review rounds landed.

Review of the reference, how-to, tutorial and explanation pages replaced statements that
disagreed with the code. Each test below locates its statement in a page by a stable
anchor (a heading, a code span, a command name), derives the matching fact from the code,
and compares the two as sets or values. No test compares a sentence of prose, writes down
a count, or runs a solver. Every comparison is a function of its inputs, and a control
test feeds that function a mutated input and expects the refusal, so a comparison that
stops comparing turns a control red.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import types
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from tests.docs.test_docs_style import tracked_pages

_REPO = Path(__file__).resolve().parents[2]
_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return tracked_pages(_REPO)


def section(text: str, heading: str) -> str:
    """The text from the heading line that contains `heading` to the next heading of any level."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("#") and heading in ln)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return "\n".join(lines[start:end])


def section_of(text: str, heading: str) -> str:
    """The text from the heading line that is exactly `heading` to the next heading of any level."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.fullmatch(r"#{1,6}\s+" + re.escape(heading), ln.strip()))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return "\n".join(lines[start:end])


def block_from(text: str, phrase: str) -> str:
    """The text from the line that contains `phrase` to the next heading line."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if phrase in ln)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return "\n".join(lines[start:end])


def spans(text: str) -> set[str]:
    """The inline code spans of `text`."""
    return set(_SPAN.findall(text))


# --- Rerun trigger: reference/rerun-triggers.md against workflow.py -------

_ALL_TRIGGERS = {"mtime", "code", "input", "params", "software-env"}


def code_rerun_triggers(workflow_src: str) -> set[str]:
    """The values of the `"rerun-triggers"` list in the profile that `analysis.run()` writes."""
    m = re.search(r'"rerun-triggers":\s*\[([^\]]*)\]', workflow_src)
    assert m is not None, "workflow.py no longer sets a rerun-triggers list"
    return set(re.findall(r'"([\w-]+)"', m.group(1)))


def check_rerun_page(page_text: str, triggers: set[str]) -> list[str]:
    """Problems: a trigger the code sets and the page omits, or a trigger the page names and the code omits."""
    named = spans(section_of(page_text, "Re-run triggers"))
    problems = [f"trigger {t} is set in code and not named on the page" for t in sorted(triggers - named)]
    problems += [
        f"{t} is named on the page and is not a trigger in code" for t in sorted((_ALL_TRIGGERS - triggers) & named)
    ]
    return problems


def test_rerun_page_names_the_triggers_the_code_sets(pages):
    triggers = code_rerun_triggers(_read("src/hhemt/workflow.py"))
    assert triggers, "the code sets no trigger"
    assert check_rerun_page(pages["docs/reference/rerun-triggers.md"], triggers) == []


def test_control_rerun_trigger_added_in_code_is_refused(pages):
    mutated = _read("src/hhemt/workflow.py").replace(
        '"rerun-triggers": ["mtime"]', '"rerun-triggers": ["mtime", "code"]', 1
    )
    problems = check_rerun_page(pages["docs/reference/rerun-triggers.md"], code_rerun_triggers(mutated))
    assert any("code" in p and "not named on the page" in p for p in problems), problems


def test_control_rerun_trigger_named_on_page_is_refused(pages):
    page = pages["docs/reference/rerun-triggers.md"].replace(
        "only the `mtime` trigger", "only the `mtime` and `code` trigger", 1
    )
    problems = check_rerun_page(page, code_rerun_triggers(_read("src/hhemt/workflow.py")))
    assert any("code" in p and "not a trigger" in p for p in problems), problems


# --- Layout-check hooks: explanation/branching-and-release-model.md ------------------


def hook_stage_pairs(config_text: str) -> set[tuple[str, str]]:
    """(stage, check letter) pairs for the `check_layout_version.py` hooks of a pre-commit config."""
    pairs = set()
    for repo in yaml.safe_load(config_text)["repos"]:
        for hook in repo.get("hooks", []):
            m = re.search(r"check_layout_version\.py (check-[ab])", hook.get("entry", ""))
            if m:
                pairs |= {(stage, m.group(1)[-1].upper()) for stage in hook.get("stages", [])}
    return pairs


def page_stage_pairs(page_text: str) -> set[tuple[str, str]]:
    """(stage, check letter) pairs from the sentences that open `At the {stage} stage`."""
    para = next(p for p in page_text.split("\n\n") if "At the pre-commit stage" in p)
    pairs = set()
    for sentence in re.split(r"(?<=\.)\s+(?=[A-Z])", para):
        m = re.match(r"At the ([\w-]+) stage", sentence)
        if m:
            pairs |= {(m.group(1), letter) for letter in re.findall(r"Check ([AB])\b", sentence)}
    return pairs


def workflows_naming(script: str, workflows: dict[str, str]) -> list[str]:
    return sorted(name for name, text in workflows.items() if script in text)


def _workflow_texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((_REPO / ".github/workflows").glob("*.yml"))}


def test_layout_check_stages_match_the_hooks(pages):
    page = pages["docs/explanation/branching-and-release-model.md"]
    code = hook_stage_pairs(_read(".pre-commit-config.yaml"))
    assert code, "no layout-check hook found"
    assert page_stage_pairs(page) == code


def check_layout_page(page_text: str, hook_pairs: set[tuple[str, str]], workflows: dict[str, str]) -> list[str]:
    """Problems: page stages that differ from the hooks, or a workflow that runs the layout check.

    The page says the check runs in no workflow. The code fact is the absence of the script
    name from every workflow file, so the page wording is not compared.
    """
    problems = []
    if page_stage_pairs(page_text) != hook_pairs:
        problems.append("the stages on the page differ from the hooks")
    problems += [f"workflow {n} names the layout check" for n in workflows_naming("check_layout_version", workflows)]
    return problems


def test_layout_check_is_in_no_workflow(pages):
    page = pages["docs/explanation/branching-and-release-model.md"]
    assert check_layout_page(page, hook_stage_pairs(_read(".pre-commit-config.yaml")), _workflow_texts()) == []


def _layout_text(wording: str) -> str:
    """Synthetic paragraph with the stage sentences and one sentence about workflows, so
    the test reads no wording of the real page."""
    return (
        "At the pre-commit stage, Check A grades the commit. "
        "At the pre-push stage, Check A grades the range, and Check B grades the pushed commits. "
        f"The layout check {wording}."
    )


@pytest.mark.parametrize(
    "wording", ["is not wired into any GitHub Actions workflow", "runs in no GitHub Actions workflow"]
)
def test_layout_check_passes_under_either_wording_and_a_workflow_naming_the_script_fails(wording):
    pairs, workflows = hook_stage_pairs(_read(".pre-commit-config.yaml")), _workflow_texts()
    assert check_layout_page(_layout_text(wording), pairs, workflows) == []
    workflows = {**workflows, "x.yml": "run: python scripts/check_layout_version.py check-a HEAD"}
    assert check_layout_page(_layout_text(wording), pairs, workflows) == ["workflow x.yml names the layout check"]


def test_layout_check_page_stages_changed_fail():
    pairs = hook_stage_pairs(_read(".pre-commit-config.yaml"))
    text = _layout_text("runs in no workflow").replace("Check B grades", "Check C grades")
    assert check_layout_page(text, pairs, {}) == ["the stages on the page differ from the hooks"]


def test_control_layout_hook_stage_changed_is_refused(pages):
    cfg = _read(".pre-commit-config.yaml")
    start = cfg.index("id: check-layout-version-b")
    mutated = cfg[:start] + cfg[start:].replace("stages: [pre-push]", "stages: [pre-commit, pre-push]", 1)
    assert hook_stage_pairs(mutated) != page_stage_pairs(pages["docs/explanation/branching-and-release-model.md"])


def test_control_page_stage_changed_is_refused(pages):
    page = pages["docs/explanation/branching-and-release-model.md"].replace(
        "At the pre-push stage, Check A", "At the pre-push stage, Check C", 1
    )
    assert page_stage_pairs(page) != hook_stage_pairs(_read(".pre-commit-config.yaml"))


def test_control_workflow_naming_the_script_is_found():
    assert workflows_naming("check_layout_version", {"x.yml": "run: python scripts/check_layout_version.py"}) == [
        "x.yml"
    ]


# --- Anonymization guard: contributing.md ---------------------------------------------


def anonymization_facts(config_text: str, workflows: dict[str, str]) -> tuple[set[str], set[str]]:
    """(stages of the `anonymization-guard` hook, names of the workflow jobs that run the script)."""
    stages: set[str] = set()
    for repo in yaml.safe_load(config_text)["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "anonymization-guard":
                stages |= set(hook.get("stages", []))
    jobs = {
        job
        for text in workflows.values()
        for job, body in (yaml.safe_load(text).get("jobs") or {}).items()
        if "check_anonymization.py" in yaml.safe_dump(body)
    }
    return stages, jobs


def check_anonymization_page(page_text: str, stages: set[str], jobs: set[str]) -> list[str]:
    para = next(p for p in page_text.split("\n\n") if "check_anonymization.py" in p and "blocklist" in p)
    named_jobs = set(re.findall(r"`([\w-]+)` job", para))
    m = re.search(r"at both the (\w+) and (\w+) stages", para)
    problems = []
    if named_jobs != jobs:
        problems.append(f"page names jobs {sorted(named_jobs)}, workflows run it in {sorted(jobs)}")
    page_stages = {f"pre-{w}" for w in m.groups()} if m else set()
    if page_stages != stages:
        problems.append(f"page names stages {sorted(page_stages)}, the hook has {sorted(stages)}")
    return problems


def test_anonymization_job_and_stages_match(pages):
    stages, jobs = anonymization_facts(_read(".pre-commit-config.yaml"), _workflow_texts())
    assert stages and jobs
    assert check_anonymization_page(pages["docs/contributing.md"], stages, jobs) == []


def test_control_anonymization_stage_and_job_changes_are_refused(pages):
    stages, jobs = anonymization_facts(_read(".pre-commit-config.yaml"), _workflow_texts())
    page = pages["docs/contributing.md"]
    assert check_anonymization_page(page, stages | {"commit-msg"}, jobs)
    assert check_anonymization_page(page, stages, jobs | {"another-job"})


# --- Documentation gates: contributing.md against scripts/docs_gates.sh ---------------

_GATE_LABELS = {
    "mkdocs build --strict": "strict build",
    "python scripts/check_autodoc_coverage.py": "autodoc coverage",
    "python scripts/check_docs_content.py": "docs content hygiene",
    "python scripts/check_published_surface.py": "published-surface fidelity",
}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def script_gates(script_text: str) -> set[str]:
    """The command of each `gate ...` invocation line of `scripts/docs_gates.sh`, without its arguments."""
    gates = set()
    for line in script_text.splitlines():
        m = re.match(r"^gate (.+)$", line)
        if m:
            words = m.group(1).split()
            gates.add(" ".join(w for w in words if not w.startswith("--site-dir") and w != "site"))
    return gates


def check_gate_page(page_text: str, gates: set[str]) -> list[str]:
    """Problems between the page's gate list and the script's gate lines, compared as sets.

    The labels the page names are the set of known labels found in its gates section. A
    count word after `runs` is checked against the gate lines only when the page states one.
    """
    body = " ".join(section(page_text, "Documentation gates").split())
    problems = []
    unlabelled = gates - set(_GATE_LABELS)
    if unlabelled:
        problems.append(f"gate lines with no page label in this test: {sorted(unlabelled)}")
    expected = {_GATE_LABELS[g] for g in gates & set(_GATE_LABELS)}
    named = {label for label in _GATE_LABELS.values() if label in body}
    problems += [f"the page does not name the gate '{label}'" for label in sorted(expected - named)]
    problems += [
        f"the page names the gate '{label}' and the script has no such gate" for label in sorted(named - expected)
    ]
    m = re.search(r"runs (\w+) gates", body)
    if m and _NUMBER_WORDS.get(m.group(1)) != len(gates):
        problems.append(f"the number word after 'runs' ({m.group(1)}) does not equal the gate lines")
    return problems


def test_gate_list_matches_the_script(pages):
    gates = script_gates(_read("scripts/docs_gates.sh"))
    assert gates
    assert check_gate_page(pages["docs/contributing.md"], gates) == []


def _gates_page(labels: list[str], count_word: str | None) -> str:
    """Synthetic gates section, so the tests read no wording of the real page."""
    runs = f"CI runs {count_word} gates" if count_word else "CI runs gates"
    return "## Documentation gates\n\n" + runs + ": " + ", ".join(labels) + ".\n"


def _count_word(n: int) -> str:
    return next(w for w, v in _NUMBER_WORDS.items() if v == n)


def _script_labels() -> tuple[set[str], list[str]]:
    gates = script_gates(_read("scripts/docs_gates.sh"))
    return gates, sorted(_GATE_LABELS[g] for g in gates)


def test_gate_page_passes_with_a_right_count_word_or_none():
    gates, labels = _script_labels()
    assert check_gate_page(_gates_page(labels, None), gates) == []
    assert check_gate_page(_gates_page(labels, _count_word(len(gates))), gates) == []


def test_control_a_wrong_count_word_is_refused():
    gates, labels = _script_labels()
    problems = check_gate_page(_gates_page(labels, _count_word(len(gates) + 1)), gates)
    assert any("number word" in p for p in problems), problems


def test_control_a_new_gate_in_the_script_is_refused():
    gates, labels = _script_labels()
    problems = check_gate_page(_gates_page(labels, _count_word(len(gates))), gates | {"python scripts/check_new.py"})
    assert any("check_new.py" in p for p in problems) and any("number word" in p for p in problems), problems


def test_control_a_gate_the_script_lacks_is_refused():
    gates, labels = _script_labels()
    problems = check_gate_page(_gates_page(labels, None), gates - {"mkdocs build --strict"})
    assert any("strict build" in p and "no such gate" in p for p in problems), problems


def test_control_a_gate_label_missing_or_changed_is_refused():
    gates, labels = _script_labels()
    dropped = [lbl for lbl in labels if lbl != "autodoc coverage"]
    assert any("autodoc coverage" in p for p in check_gate_page(_gates_page(dropped, None), gates))
    changed = [("documentation coverage" if lbl == "autodoc coverage" else lbl) for lbl in labels]
    assert any("autodoc coverage" in p for p in check_gate_page(_gates_page(changed, None), gates))


# --- Reporting sets: reference/example-report.md --------------------------------------


def check_sidebar_orders(page_text: str, nonstandard: set[str]) -> list[str]:
    named = set(re.findall(r"Only `([\w-]+)`", page_text))
    return [] if named == nonstandard else [f"page names {sorted(named)}, code has {sorted(nonstandard)}"]


def nonstandard_sets() -> set[str]:
    from hhemt.report_renderers import _reporting_sets as rs

    return {n for n, s in rs.REPORTING_SETS.items() if s.category_order != rs._STANDARD_CATEGORY_ORDER}


def test_only_the_named_sets_have_their_own_sidebar_order(pages):
    assert check_sidebar_orders(pages["docs/reference/example-report.md"], nonstandard_sets()) == []


def test_control_another_set_with_its_own_order_is_refused(pages):
    problems = check_sidebar_orders(pages["docs/reference/example-report.md"], nonstandard_sets() | {"benchmarking"})
    assert problems


# --- Output data model: reference/output-data-model.md --------------------------------


def check_node_page(
    page_text: str, tree_paths: set[str], timeseries_paths: set[str], timeseries_default: bool
) -> list[str]:
    problems = []
    body = block_from(page_text, "The groups of an analysis tree are written")
    table = {m for m in re.findall(r"^\| `([\w/]+)` \|", body, re.M)}
    if table != tree_paths:
        problems.append(f"node table {sorted(table)} differs from the code {sorted(tree_paths)}")
    sentence = next(ln for ln in body.splitlines() if "toggle_consolidate_timeseries" in ln)
    ts = set(re.findall(r"`(tritonswmm/\w+_timeseries)`", " ".join(body.split())))
    if ts != timeseries_paths:
        problems.append(f"timeseries nodes {sorted(ts)} differ from the code {sorted(timeseries_paths)}")
    m = re.search(r"defaults to `(true|false)`", sentence)
    if m is None or (m.group(1) == "true") != timeseries_default:
        problems.append("the default of toggle_consolidate_timeseries on the page differs from the code")
    unknown = {
        s
        for s in spans(page_text)
        if re.match(r"(tritonswmm|triton_only|swmm_only)/\w+$", s) and s not in tree_paths | timeseries_paths
    }
    if unknown:
        problems.append(f"node paths on the page that code does not write: {sorted(unknown)}")
    return problems


def node_facts() -> tuple[set[str], set[str], bool]:
    from hhemt.config.analysis import analysis_config
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    field = analysis_config.model_fields["toggle_consolidate_timeseries"]
    ts = set(re.findall(r"tritonswmm/\w+_timeseries", field.description or ""))
    return set(pp._MODE_TO_TREE_PATH.values()), ts, bool(field.default)


def test_node_table_matches_the_tree_modes(pages):
    tree, ts, default = node_facts()
    assert tree and ts
    assert check_node_page(pages["docs/reference/output-data-model.md"], tree, ts, default) == []


def test_control_a_node_added_in_code_or_a_default_flipped_is_refused(pages):
    tree, ts, default = node_facts()
    page = pages["docs/reference/output-data-model.md"]
    assert check_node_page(page, tree | {"swmm_only/extra"}, ts, default)
    assert check_node_page(page, tree, ts, not default)
    assert check_node_page(page.replace("`tritonswmm/swmm_link`", "`tritonswmm_swmm_link`", 1), tree, ts, default)


def test_internal_anchor_of_the_per_system_tier_resolves(pages):
    page = pages["docs/reference/output-data-model.md"]
    assert "(#the-per-system-tier)" in page
    assert any(re.match(r"#+\s+The per-system tier\s*$", ln) for ln in page.splitlines())


# --- YAML blocks that show an analysis-config field ----------------------------------


def yaml_values(page_text: str, key: str) -> list[object]:
    """Values of `key` in the top-level mapping of each fenced yaml block of a page."""
    values = []
    for m in re.finditer(r"^```yaml\n(.*?)^```", page_text, re.S | re.M):
        data = yaml.safe_load(m.group(1))
        if isinstance(data, dict) and key in data:
            values.append(data[key])
    return values


def check_config_values(page_text: str, key: str) -> list[str]:
    from hhemt.config.analysis import analysis_config

    field = analysis_config.model_fields.get(key)
    if field is None:
        return [f"analysis_config has no field {key}"]
    problems = []
    values = yaml_values(page_text, key)
    if not values:
        problems.append(f"the page shows no yaml block with {key}")
    for value in values:
        try:
            TypeAdapter(field.annotation).validate_python(value)
        except ValidationError:
            problems.append(f"{key}: {value!r} is not an accepted value")
    return problems


@pytest.mark.parametrize(
    ("page", "key"),
    [
        ("docs/reference/fair-scope-table.md", "dataset_license"),
        ("docs/reference/output-data-model.md", "target_processed_output_type"),
    ],
)
def test_yaml_blocks_show_accepted_values(pages, page, key):
    assert check_config_values(pages[page], key) == []


def test_control_a_rejected_value_is_refused(pages):
    page = pages["docs/reference/fair-scope-table.md"].replace(
        "dataset_license: CC-BY-4.0", "dataset_license: CC-BY-9.9", 1
    )
    assert check_config_values(page, "dataset_license")
    page = pages["docs/reference/output-data-model.md"].replace(
        "target_processed_output_type: nc", "target_processed_output_type: hdf"
    )
    assert check_config_values(page, "target_processed_output_type")
    assert check_config_values("```yaml\nno_such_key: 1\n```\n", "no_such_key")


# --- EDA configuration: how-to/synthetic-compute-sensitivity-experiment.md ------------


def check_enabled_plots(page_text: str, registered: set[str]) -> list[str]:
    from hhemt.config.analysis import analysis_config

    problems = []
    if "enabled_plots" not in analysis_config.model_fields["eda"].annotation.model_fields:
        problems.append("analysis_config.eda has no enabled_plots field")
    names: set[str] = set()
    for span in spans(page_text):
        m = re.match(r"eda\.enabled_plots:\s*\[(.*)\]$", span)
        if m:
            names |= {n.strip() for n in m.group(1).split(",") if n.strip()}
    if not names:
        problems.append("the page names no eda.enabled_plots list")
    problems += [f"plot kind {n} is not registered" for n in sorted(names - registered)]
    return problems


def test_enabled_plots_names_registered_kinds(pages):
    from hhemt.eda._plotting import _EDA_RENDERERS

    assert (
        check_enabled_plots(pages["docs/how-to/synthetic-compute-sensitivity-experiment.md"], set(_EDA_RENDERERS)) == []
    )


def test_control_an_unregistered_kind_or_the_old_key_is_refused(pages):
    from hhemt.eda._plotting import _EDA_RENDERERS

    page = pages["docs/how-to/synthetic-compute-sensitivity-experiment.md"]
    assert check_enabled_plots(page.replace("dem_resolution_diff_maps", "dem_resolution_nope", 1), set(_EDA_RENDERERS))
    assert check_enabled_plots(page.replace("eda.enabled_plots", "eda_config.enabled_plots"), set(_EDA_RENDERERS))


def test_unregistered_plot_kind_raises_value_error_listing_the_registered_names(pages):
    from hhemt.eda._plotting import _EDA_RENDERERS, render_eda_plots

    assert "`ValueError`" in pages["docs/how-to/running-eda.md"]
    cfg = types.SimpleNamespace(enabled_plots=["no_such_kind"])
    with pytest.raises(ValueError) as exc:
        render_eda_plots(_REPO, cfg_analysis=None, eda_cfg=cfg)
    assert all(name in str(exc.value) for name in _EDA_RENDERERS)


# --- Quickstart: baseline configuration ------------------------------


def check_baseline(page_text: str, workflow_src: str) -> list[str]:
    from hhemt.config.analysis import analysis_config

    problems = []
    block = yaml_block_after(page_text, "Written out, the same compute configuration in the analysis config is")
    for key in ("run_mode", "multi_sim_run_method", "local_cpu_cores_for_workflow"):
        field = analysis_config.model_fields.get(key)
        if field is None or key not in block:
            problems.append(f"{key} missing from the config or the block")
            continue
        try:
            TypeAdapter(field.annotation).validate_python(block[key])
        except ValidationError:
            problems.append(f"{key}: {block[key]!r} is not an accepted value")
    cores = block.get("local_cpu_cores_for_workflow")
    claimed = set(re.findall(r"`local_cpu_cores_for_workflow: (\d+)`", page_text)) | set(
        re.findall(r"\((\d+) in the config above\)", page_text)
    )
    if claimed != {str(cores)}:
        problems.append(f"the prose names core counts {sorted(claimed)}, the block has {cores}")
    if not re.search(r'"cores":\s*self\.cfg_analysis\.local_cpu_cores_for_workflow', workflow_src):
        problems.append("workflow.py no longer passes local_cpu_cores_for_workflow as the local core count")
    named = set(re.findall(r"`(batch_job|1_job_many_srun_tasks|local)`", page_text))
    allowed = set(analysis_config.model_fields["multi_sim_run_method"].annotation.__args__)
    problems += [f"multi_sim_run_method {v} is not accepted" for v in sorted(named - allowed)]
    return problems


def yaml_block_after(page_text: str, anchor: str) -> dict:
    tail = page_text[page_text.index(anchor) :]
    return yaml.safe_load(re.search(r"```yaml\n(.*?)```", tail, re.S).group(1))


def test_norfolk_baseline_matches_the_config_and_the_workflow(pages):
    assert check_baseline(pages["docs/quickstart.md"], _read("src/hhemt/workflow.py")) == []


def test_control_a_core_count_or_run_mode_changed_is_refused(pages):
    page = pages["docs/quickstart.md"]
    src = _read("src/hhemt/workflow.py")
    assert check_baseline(
        page.replace("local_cpu_cores_for_workflow: 4\n```", "local_cpu_cores_for_workflow: 8\n```", 1), src
    )
    assert check_baseline(
        page.replace("run_mode: serial\nmulti_sim_run_method", "run_mode: threaded\nmulti_sim_run_method", 1), src
    )
    assert check_baseline(page, src.replace('"cores": self.cfg_analysis.local_cpu_cores_for_workflow', '"cores": 1'))


# --- Command-line helpers named in the how-to pages -----------------------------------


def parser_flags(source: str) -> set[str]:
    """Long option strings passed to `add_argument` calls in `source`."""
    flags = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flags |= {a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
    return flags


def check_setup_flags(page_text: str, setup_flags: set[str], run_options: set[str]) -> list[str]:
    problems = []
    text = page_text.replace("\\\n", " ")
    for line in text.splitlines():
        if "python -m hhemt.setup_workflow" in line:
            problems += [
                f"{f} is not a setup_workflow option" for f in re.findall(r"--[\w-]+", line) if f not in setup_flags
            ]
    for span in spans(page_text):
        if re.fullmatch(r"--[\w-]+", span) and span not in setup_flags | run_options:
            problems.append(f"{span} is neither a setup_workflow nor a run option")
    return problems


def test_compile_page_flags_exist(pages):
    from tests.docs.test_retrofit_cli_commands import command_tree

    setup_flags = parser_flags(_read("src/hhemt/setup_workflow.py"))
    run = command_tree().subcommands["run"].options
    page = pages["docs/how-to/compiling-the-solver.md"]
    assert {"--compile-triton-swmm", "--compile-triton-only", "--compile-swmm", "--recompile-if-already-done"} <= set(
        re.findall(r"--[\w-]+", page)
    )
    assert check_setup_flags(page, setup_flags, run) == []


def test_control_a_flag_the_parser_lacks_is_refused(pages):
    from tests.docs.test_retrofit_cli_commands import command_tree

    setup_flags = parser_flags(_read("src/hhemt/setup_workflow.py"))
    run = command_tree().subcommands["run"].options
    page = pages["docs/how-to/compiling-the-solver.md"].replace("--compile-triton-swmm", "--compile-triton-swm", 1)
    assert check_setup_flags(page, setup_flags, run)
    assert check_setup_flags(page.replace("`--from-scratch`", "`--from-scrach`", 1), setup_flags, run)


def test_compile_page_toggles_are_system_config_fields(pages):
    from hhemt.config.system import system_config

    toggles = {s for s in spans(pages["docs/how-to/compiling-the-solver.md"]) if re.fullmatch(r"toggle_\w+", s)}
    assert {"toggle_tritonswmm_model", "toggle_triton_model", "toggle_swmm_model"} <= toggles
    assert toggles <= set(system_config.model_fields)


def allowed_start_with(cli_src: str) -> set[str]:
    m = re.search(r"start_with not in \(([^)]*)\)", cli_src)
    assert m is not None, "cli.py no longer validates start_with against a literal tuple"
    return set(re.findall(r"['\"](\w+)['\"]", m.group(1)))


def page_start_with_values(pages: dict[str, str]) -> set[str]:
    return {v for text in pages.values() for v in re.findall(r"--start-with[ =](\w+)", text)}


def test_start_with_values_on_pages_are_accepted(pages):
    shown = page_start_with_values(pages)
    assert {"consolidate", "render"} <= shown
    assert shown <= allowed_start_with(_read("src/hhemt/cli.py"))


def test_control_a_start_with_value_the_code_rejects_is_refused(pages):
    shown = page_start_with_values({"p.md": "hhemt reprocess --start-with plot"})
    assert not shown <= allowed_start_with(_read("src/hhemt/cli.py"))


def option_repeats(cli_path: tuple[str, ...], flag: str) -> bool:
    import typer

    from hhemt.cli import app

    cmd = typer.main.get_command(app)
    for name in cli_path:
        cmd = cmd.commands[name]
    return any(flag in p.opts and getattr(p, "multiple", False) for p in cmd.params)


def check_container_def_page(page_text: str, repeatable: bool) -> list[str]:
    """Problems: the page omits the option, or the option cannot repeat as the page's per-file use needs."""
    problems = [] if "`--container-def`" in page_text else ["the page does not name `--container-def`"]
    if not repeatable:
        problems.append("the option does not repeat")
    return problems


def test_container_def_can_repeat_as_the_page_says(pages):
    page = pages["docs/how-to/reprex-roundtrip.md"]
    assert check_container_def_page(page, option_repeats(("reprex", "emit"), "--container-def")) == []


@pytest.mark.parametrize(
    "wording", ["Pass `--container-def` once for each file.", "Use `--container-def`, and repeat it for each file."]
)
def test_container_def_passes_under_either_wording_and_a_single_use_option_fails(wording):
    assert check_container_def_page(wording, option_repeats(("reprex", "emit"), "--container-def")) == []
    assert check_container_def_page(wording, option_repeats(("reprex", "emit"), "--cluster")) == [
        "the option does not repeat"
    ]
    assert check_container_def_page("Pass the definition file.", True) == ["the page does not name `--container-def`"]


def test_control_a_single_use_option_is_not_repeatable():
    assert not option_repeats(("reprex", "emit"), "--cluster")


def test_override_force_rerun_is_a_run_option_as_the_diagnosis_page_says(pages):
    from tests.docs.test_retrofit_cli_commands import command_tree

    page = pages["docs/how-to/diagnosing-a-failed-run.md"]
    assert "`--override-force-rerun`" in page
    assert "--override-force-rerun" in command_tree().subcommands["run"].options
    assert "--override-force-rerun" not in command_tree().subcommands["status"].options


def test_placeholder_verb_lines_use_options_of_the_run_command(pages):
    from tests.docs.test_retrofit_cli_commands import command_tree

    run = command_tree().subcommands["run"].options
    flags = set()
    for page in ("docs/reference/cli.md", "docs/contributing.md"):
        for span in spans(pages[page]):
            if span.startswith("hhemt {verb}"):
                flags |= set(re.findall(r"--[\w-]+", span))
    assert {"--system-config", "--analysis-config"} <= flags
    assert flags <= run


# --- Publication pages: record keys, result fields and environment variables ----------


def publish_result_keys(source: str) -> set[str]:
    """String keys of the dict literals that carry a `data_doi` key in publishing.py."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            names = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if "data_doi" in names:
                keys |= names
    return keys


def check_record_keys(page_text: str, keys: set[str]) -> list[str]:
    sentences = re.findall(r"prints the deposit record(?::| with) ([^\n]*?)\. ", " ".join(page_text.split()) + " ")
    named = {s for sentence in sentences for s in re.findall(r"`(\w+)`", sentence)}
    problems = [] if named else ["the page names no deposit record keys"]
    return problems + [f"{n} is not a key of the publish result" for n in sorted(named - keys)]


@pytest.mark.parametrize("page", ["docs/how-to/publishing.md", "docs/how-to/doi-roundtrip-e2e.md"])
def test_deposit_record_keys_exist(pages, page):
    assert check_record_keys(pages[page], publish_result_keys(_read("src/hhemt/publishing.py"))) == []


def test_control_a_renamed_record_key_is_refused(pages):
    page = pages["docs/how-to/publishing.md"].replace("`record_url`", "`record_link`", 1)
    assert check_record_keys(page, publish_result_keys(_read("src/hhemt/publishing.py")))


def check_reprex_fields(page_text: str) -> list[str]:
    from hhemt.bundle._reprex import Amendment, ReprexResult

    para = next(p for p in page_text.split("\n\n") if p.startswith("Read `runnable` first"))
    named = set(re.findall(r"`(\w+)`", para))
    result_fields = {f.name for f in dataclasses.fields(ReprexResult)}
    amendment_fields = {f.name for f in dataclasses.fields(Amendment)}
    problems = [f"{n} is not a ReprexResult field" for n in sorted(named - result_fields)]
    if not {"status", "reason"} <= amendment_fields:
        problems.append("Amendment lacks status or reason")
    return problems


def test_reprex_result_fields_exist(pages):
    assert check_reprex_fields(pages["docs/how-to/reprex-roundtrip.md"]) == []


def test_control_a_renamed_result_field_is_refused(pages):
    assert check_reprex_fields(pages["docs/how-to/reprex-roundtrip.md"].replace("`problem_pairs`", "`problems`"))


def zenodo_variables(page_texts: list[str], source: str) -> list[str]:
    named = {v for t in page_texts for v in re.findall(r"HHEMT_(?:ZENODO|HYDROSHARE)_[A-Z_]+", t)}
    return [f"{v} is not read by the code" for v in sorted(named) if v not in source]


def _src_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((_REPO / "src/hhemt").rglob("*.py")))


def test_publication_environment_variables_are_read_by_the_code(pages):
    texts = [
        pages[p]
        for p in ("docs/how-to/publishing.md", "docs/how-to/doi-roundtrip-e2e.md", "docs/how-to/reprex-roundtrip.md")
    ]
    assert any("HHEMT_ZENODO_BASE_URL" in t for t in texts)
    assert zenodo_variables(texts, _src_text()) == []


def test_control_an_unread_variable_is_refused():
    assert zenodo_variables(["Set HHEMT_ZENODO_NOPE=1."], _src_text())


# --- Docstring facts in src modules ----------------------------------------------------


def docstring_rank_sweep(source: str) -> tuple[int, ...]:
    doc = ast.get_docstring(ast.parse(source)) or ""
    m = re.search(r"default ``\(([\d, ]+)\)``", doc)
    assert m is not None, "the module docstring states no rank_sweep default"
    return tuple(int(x) for x in m.group(1).split(","))


def imported_top_modules(source: str) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_synthetic_experiment_docstring_default_matches_the_config():
    from hhemt.config.synthetic_experiment import synthetic_experiment_config

    default = synthetic_experiment_config.model_fields["rank_sweep"].default
    assert docstring_rank_sweep(_read("src/hhemt/synthetic_experiment.py")) == tuple(default)


def test_control_a_docstring_default_changed_is_refused():
    mutated = _read("src/hhemt/synthetic_experiment.py").replace("default ``(2, 4, 8)``", "default ``(2, 4)``", 1)
    assert docstring_rank_sweep(mutated) != (2, 4, 8)


@pytest.mark.parametrize("module", ["src/hhemt/synthetic_experiment.py", "src/hhemt/experiment_definition.py"])
def test_framework_modules_import_neither_tests_nor_scripts(module):
    assert imported_top_modules(_read(module)).isdisjoint({"tests", "scripts"})


def test_control_an_import_from_scripts_is_found():
    assert "scripts" in imported_top_modules("from scripts.experiments import x\n")


def check_stub_title(page_text: str, source: str) -> list[str]:
    """Problems: a `(figure pending ...)` span on the page that the stub-figure source lacks."""
    return [
        f"{s} is not in the stub title"
        for s in sorted(spans(page_text))
        if s.startswith("(figure pending") and s not in source
    ]


def test_stub_figure_title_on_the_page_is_the_title_in_code(pages):
    """The page quotes the stub title as a code span until the title is changed. After the
    change the page holds no such span and the comparison covers nothing."""
    page = pages["docs/tutorials/exploratory-analysis.md"]
    assert check_stub_title(page, _read("src/hhemt/eda/_sensitivity_figures.py")) == []


def test_control_a_changed_title_span_is_refused():
    assert check_stub_title("The title ends `(figure pending other)`.", 'f"{plot_id} (figure pending x)"')
    assert check_stub_title("The title ends `(figure pending x)`.", 'f"{plot_id} (figure pending x)"') == []
