"""Generate `reference/config-schema.md` at build time from the live config models.

WHY A BUILD-TIME HOOK RATHER THAN A COMMITTED FILE. There is no artifact between
builds, so the page cannot drift from the models it describes. A committed
generated file would need a regeneration ritual and a CI diff to reach a weaker
guarantee, and it would invite hand-edits that the next regeneration silently
reverts.

WHY NOTHING HERE IS WRAPPED IN try/except, AND WHY THAT IS DELIBERATE. mkdocs
fails loudly by design: `hooks:` validates the path through `File(exists=True)`,
`_load_hook` calls `exec_module` unguarded, and `Plugins.run_event` calls each
event method with no try/except. That loudness is mkdocs' property, not this
module's. A generator that caught its own exception and emitted a partial page
would degrade silently while looking like defensive programming. So the
introspection runs unguarded and any failure aborts the build. If a partial page
is ever wanted, it must be an explicit and named decision, not a swallowed
exception.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

GENERATED_URI = "reference/config-schema.md"
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Bind the generator to THIS checkout's source tree.
#
# The editable install writes a `.pth` pointing at the MAIN clone's `src`, so a
# plain `import hhemt` from a worktree silently imports the main tree and this
# page would describe code that is not the code under review. Measured this
# round: `hhemt.__file__` resolved to the main clone while building the
# worktree. Prepending keeps the worktree ahead of the `.pth` entry.
_SRC = _REPO_ROOT / "src"


def _bind_local_src() -> None:
    """Make `import hhemt` resolve to THIS checkout, deterministically.

    Prepending to `sys.path` is not sufficient on its own: by the time a hook
    runs, `hhemt` may already be in `sys.modules` from the editable install's
    `.pth`, and an already-imported package wins regardless of path order. So
    the stale modules are dropped before the import.
    """
    if sys.path and sys.path[0] != str(_SRC):
        sys.path.insert(0, str(_SRC))
    stale = [
        name
        for name, mod in list(sys.modules.items())
        if name == "hhemt" or name.startswith("hhemt.")
        if not str(getattr(mod, "__file__", "") or "").startswith(str(_SRC))
    ]
    for name in stale:
        del sys.modules[name]


# --- Step 1: the derivation primitive ---------------------------------------
#
# This returns STRUCTURED DATA, never HTML. It deliberately does NOT reuse
# `report_renderers.metadata._config_field_rows`: that function is a
# reprex-guide row builder, its rows are pre-rendered HTML, and its fourth
# column is an amendment hint rather than a default. Its zero-user-info
# property holds by construction because its signature takes no analysis
# argument, and the strongest way to preserve that is not to edit it.


def field_records(model: type) -> list[dict[str, Any]]:
    """Structured records for every field of a Pydantic config model.

    One dict per field: `name`, `type`, `required_when` (clause list or None),
    `default`, `description`, `options` (glossary dict or None).
    """
    from pydantic_core import PydanticUndefined

    from hhemt.config.base import declared

    records: list[dict[str, Any]] = []
    for name, info in model.model_fields.items():
        default: Any = info.default
        if default is PydanticUndefined:
            default = None if info.default_factory is None else "(computed)"
        records.append(
            {
                "name": name,
                "type": _type_name(info.annotation),
                "required_when": declared(info, "required_when"),
                "applies_when": declared(info, "applies_when"),
                "options": declared(info, "options"),
                "default": default,
                "description": info.description or "",
                "required": info.is_required(),
            }
        )
    return records


def _type_name(annotation: Any) -> str:
    """A readable type name for a table cell."""
    text = str(annotation)
    text = re.sub(r"<class '([^']+)'>", r"\1", text)
    text = text.replace("typing.", "").replace("pathlib.", "")
    text = re.sub(r"\bhhemt\.config\.[a-z_]+\.", "", text)
    return text


# --- Step 2b: the lint mitigation -------------------------------------------
#
# `check_docs_content.py` walks the docs directory with `rglob("*.md")`, so a
# page that never exists on disk is invisible to it. Importing its own pattern
# tuples and running them here converts that blind spot into a build-time gate
# that is STRICTER than the lint: a violation aborts the build rather than
# waiting for someone to run the script.


def _load_lint():
    path = _REPO_ROOT / "scripts" / "check_docs_content.py"
    spec = importlib.util.spec_from_file_location("_hhemt_docs_lint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_lint_clean(markdown: str) -> None:
    lint = _load_lint()
    findings: list[str] = []
    for lineno, line in lint._unfenced_lines(markdown):
        for code, pat in lint.PLACEHOLDER_PATTERNS:
            if pat.search(line):
                findings.append(f"{code} at generated line {lineno}: {line.strip()}")
        hit = lint.LINE_CITATION.search(line)
        if hit:
            findings.append(f"bare-line-citation at generated line {lineno}: {hit.group(0)}")
    for lineno, line in lint._prose_lines(markdown):
        for code, pat in lint.PUNCTUATION_PATTERNS:
            if pat.search(line):
                findings.append(f"{code} at generated line {lineno}: {line.strip()}")
    for lineno, line in lint._all_lines(markdown):
        for code, pat in lint.WORD_BAN_PATTERNS:
            if pat.search(line):
                findings.append(f"{code} at generated line {lineno}: {line.strip()}")
    if findings:
        raise RuntimeError("generated config reference violates docs-content rules:\n  " + "\n  ".join(findings))


# --- Step 3: rendering ------------------------------------------------------


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell.

    A pipe becomes the HTML entity rather than a backslash escape: inside a
    `<code>` span markdown does not unescape `\\|`, so the backslash leaks to the
    reader. Measured on the first build of this page.
    """
    return str(text).replace("|", "&#124;").replace("\n", " ")


def _type_cell(record: dict[str, Any]) -> str:
    """Type, with any options glossary nested inside the same cell.

    A glossary is a nested list here rather than a sixth column: only seven
    fields carry one, their cardinality varies, and a column sized for the
    widest would waste width on every other row. Nothing is truncated.
    """
    parts = [f"<code>{_cell(record['type'])}</code>"]
    options = record["options"]
    if options:
        rendered = "<br>".join(f"`{_cell(k)}` {_cell(v)}" for k, v in options.items())
        parts.append(f"<br>{rendered}")
    return "".join(parts)


def _required_cell(record: dict[str, Any]) -> str:
    """Render requiredness from the predicate DATA, not as hand-written prose."""
    from hhemt.config.base import render_clauses

    if record["required_when"]:
        return f"When {_cell(render_clauses(record['required_when']))}"
    if record["required"]:
        return "Always"
    if record["applies_when"]:
        return f"Optional; applies when {_cell(render_clauses(record['applies_when']))}"
    return "Optional"


def _default_cell(record: dict[str, Any]) -> str:
    default = record["default"]
    if default is None:
        return ""
    return f"`{_cell(default)}`"


def _table(model: type) -> str:
    lines = [
        "| Field | Type | Required | Default | Description |",
        "|---|---|---|---|---|",
    ]
    for record in field_records(model):
        lines.append(
            "| `{name}` | {type} | {req} | {default} | {desc} |".format(
                name=_cell(record["name"]),
                type=_type_cell(record),
                req=_required_cell(record),
                default=_default_cell(record),
                desc=_cell(record["description"]),
            )
        )
    return "\n".join(lines)


def _authored_prose() -> list[str]:
    """Every line of prose THIS MODULE writes, with no model-derived content.

    The lint gate runs over exactly this and not over the rendered tables. The
    design's mitigation targets "the generator's own authored text - headings,
    column labels, the page's intro paragraph", which is prose the lint checks
    today. The description cells are `src/`-side prose, which D22b already ruled
    out of lint scope; linting them here would re-open a decision D22b closed,
    and would fail the build on a string this round is not authorized to edit.
    """
    return [line for line in _render().split("\n") if not line.startswith("|")]


def _render() -> str:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
    from hhemt.config.system import system_config

    marker = _load_lint().GENERATED_MARKER
    sections = [
        f"<!-- {marker} -->",
        "<!-- DO NOT EDIT THIS FILE. It is rewritten from the config models on",
        "     every documentation build by hooks/config_reference.py, so any",
        "     hand-edit is silently discarded at the next build. To change what",
        "     appears here, edit the Pydantic field or this hook. -->",
        "",
        "# Configuration schema",
        "",
        "Every field of every user-facing config, derived from the models themselves",
        "at documentation build time. Nothing on this page is hand-maintained, so it",
        "cannot fall behind the code.",
        "",
        'All config models set `extra="forbid"`, so an unrecognised key is an error and',
        "a mistyped one is caught at load rather than at dispatch.",
        "",
        "Start from the in-repo templates:",
        "`test_data/norfolk_coastal_flooding/template_system_config.yaml` and",
        "`template_analysis_config.yaml`. For the task-oriented path through these",
        "fields see [Fill in your configuration](../how-to/config-filling.md); for the",
        "cluster profile see [HPC-profile setup](../how-to/hpc-profile-setup.md).",
        "",
        "## System config",
        "",
        "Describes the modelled area: the DEM, the SWMM model, and the toggles that",
        "decide which of the three model types are built and run.",
        "",
        _table(system_config),
        "",
        "## Analysis config",
        "",
        "Describes the study: which events to simulate, how each simulation is run,",
        "and what is produced from the results.",
        "",
        _table(analysis_config),
        "",
        "## HPC-system config",
        "",
        '--8<-- "hpc-system-config-role.md"',
        "",
        _table(hpc_system_config),
        "",
        "### PartitionSpec",
        "",
        "One entry per partition under the HPC-system config's `partitions` mapping.",
        "",
        _table(PartitionSpec),
        "",
        _conditional_section(),
        "",
        "## See also",
        "",
        "- [Reporting sets](reporting-sets.md): choosing which figures and tables a report contains.",
        "- [Fill in your configuration](../how-to/config-filling.md): the task-oriented path through these fields.",
        "- [HPC-profile setup](../how-to/hpc-profile-setup.md): the third config, describing your cluster.",
        "- [When and why re-runs happen](../explanation/rerun-faq.md): why editing the derived sensitivity CSV does "
        "not work.",
        "",
    ]
    return "\n".join(sections)


def _conditional_section() -> str:
    """The toggle-dependency section, rendered from the same predicate data.

    This carries the `#toggle-dependent-required-fields` anchor that
    `how-to/config-filling.md` links to. The old hand-maintained page truncated
    its list to "+ 3 colname fields"; every one of those is machine-readable, so
    nothing here is elided.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.base import declared, render_clauses
    from hhemt.config.system import system_config

    by_trigger: dict[str, list[str]] = {}
    for model in (system_config, analysis_config):
        for name, info in model.model_fields.items():
            clauses = declared(info, "required_when")
            if not clauses:
                continue
            by_trigger.setdefault(render_clauses(clauses), []).append(name)

    lines = [
        "## Toggle-dependent required fields",
        "",
        "Setting one field can make others required. Each row below is the same",
        "predicate the validator enforces, so this table cannot disagree with what",
        "the code does.",
        "",
        "| When | These become required |",
        "|---|---|",
    ]
    for trigger in sorted(by_trigger):
        fields = ", ".join(f"`{n}`" for n in sorted(by_trigger[trigger]))
        lines.append(f"| {_cell(trigger)} | {fields} |")
    return "\n".join(lines)


def on_config(config):
    """Write the generated page into the docs tree before file collection.

    WHY THIS WRITES TO DISK RATHER THAN EMITTING A CONTENT-ONLY `File`. Measured
    this round: `htmlproofer` resolves an internal link by locating the target
    page's SOURCE file under `docs_dir`, so a content-only generated File makes
    every link into AND out of this page 404 under `--strict`. Passing
    `abs_src_path` to a path outside `docs_dir` does not satisfy it either, and
    mkdocs refuses `content` and `abs_src_path` together. The design note
    predicting no htmlproofer interaction is falsified.

    The staleness property the design was protecting is unchanged: this file is
    rewritten from the live models on every build and is never hand-maintained,
    so it cannot drift. It is build output and belongs in `.gitignore`.
    """
    _bind_local_src()
    markdown = _render()
    _assert_lint_clean("\n".join(_authored_prose()))
    target = Path(config.docs_dir) / GENERATED_URI
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return config
