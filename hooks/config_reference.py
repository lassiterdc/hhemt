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
from pathlib import Path
from typing import Any

GENERATED_URI = "reference/config-schema.md"

#: The two long tables that also get a standalone full-width page: (config key, page title, file name beside
#: the schema page). Every other table stays on the schema page only.
_STANDALONE_PAGES = (
    ("system", "System config", "config-schema-system.html"),
    ("analysis", "Analysis config", "config-schema-analysis.html"),
)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Bind the generator to THIS checkout's source tree.
#
# The editable install writes a `.pth` pointing at the MAIN clone's `src`, so a
# plain `import hhemt` from a worktree silently imports the main tree and this
# page would describe code that is not the code under review. Measured this
# round: `hhemt.__file__` resolved to the main clone while building the
# worktree. The binder that keeps the worktree ahead of the `.pth` entry and
# evicts an already-imported main-tree package lives in `scripts/local_src.py`,
# is loaded by path in `_load_local_src` below, and `on_config` runs it first.
_SRC = _REPO_ROOT / "src"


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


def _load_local_src():
    """The checkout binder, loaded by path for the reason `_load_lint` is: `scripts/` is not a package."""
    path = _REPO_ROOT / "scripts" / "local_src.py"
    spec = importlib.util.spec_from_file_location("_hhemt_local_src", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_lint_clean(markdown: str, *, binary_only: bool = False) -> None:
    """Fail the build on lint violations in `markdown`.

    `binary_only` runs just the two file-type-independent classes. It exists so
    `on_config` can hold the rendered TABLES to the classes `### D22b` never
    excluded, without holding `src/`-derived prose to the vocabulary and
    punctuation contracts `### D22b` does exclude.
    """
    lint = _load_lint()
    findings: list[str] = []
    for lineno, line in lint._unfenced_lines(markdown):
        for code, pat in lint.PLACEHOLDER_PATTERNS:
            if pat.search(line):
                findings.append(f"{code} at generated line {lineno}: {line.strip()}")
        hit = lint.LINE_CITATION.search(line)
        if hit:
            findings.append(f"bare-line-citation at generated line {lineno}: {hit.group(0)}")
    if binary_only:
        if findings:
            raise RuntimeError("generated config reference violates docs-content rules:\n  " + "\n  ".join(findings))
        return
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
        return f"Optional, applies when {_cell(render_clauses(record['applies_when']))}"
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


#: Every config a HUMAN FILLS IN (`### D127`). That is the population CRITERION, and it
#: partitions on who completes the CONTENT rather than on who creates the file or on
#: whether a CLI flag names its path -- `reprex_config.yaml` is written by the toolkit
#: into every emitted bundle and filled in by the reproducer, so it qualifies. Test a
#: new config against that clause rather than arguing the case again.
#:
#: Membership is ENUMERATED below rather than derived from the criterion, because
#: nothing machine-readable records who fills a config in. Two consequences, and both
#: are deliberate: a newly qualifying config must be added here by hand, and
#: `CaseManifest` is ABSENT as an OPEN SCOPING QUESTION -- a case-study contributor is
#: a different reader from a run operator -- rather than as a criterion-based
#: exclusion. Do not read its absence as settled.
#:
#: Every other model this page tables is reached FROM one of these by following an
#: annotated field type -- `### D99`. The population BELOW the roots is therefore
#: DERIVED, not listed: a nested config joins the page by being referenced, and
#: nothing about it has to be remembered. `ContainerRef` arrives exactly that way, as
#: a field of `ExperimentConfig`, and is deliberately NOT a root of its own.
ENTRY_CONFIGS = (
    "hhemt.config.system:system_config",
    "hhemt.config.analysis:analysis_config",
    "hhemt.config.hpc_system:hpc_system_config",
    "hhemt.config.sif_build:sif_build_config",
    "hhemt.config.experiment_definition:ExperimentConfig",
    "hhemt.config.reprex_config:reprex_config",
)


def _closure_models() -> list[type]:
    """Every config model reachable from `ENTRY_CONFIGS` by annotated field type.

    FORWARD REFERENCES ARE RESOLVED BEFORE THE WALK, and that is not a detail.
    ``hpc_system_config.container`` is annotated ``ForwardRef('ContainerSpec |
    None')``; ``typing.get_args`` returns ``()`` on an unresolved ForwardRef, so a
    walk without ``model_rebuild()`` follows no edge and drops the model without
    saying anything -- measured at 24 models unresolved against 25 resolved, the
    difference being ``ContainerSpec``. A dropped edge UNDER-reports and is
    invisible in the output, so the guard below refuses to render rather than
    stepping over one.

    WHAT THE GUARD ACTUALLY DEFENDS AGAINST, stated precisely because its first
    wording was wrong. It does NOT catch an UNRESOLVABLE forward reference: on one
    of those ``model_rebuild()`` itself raises ``PydanticUndefinedAnnotation`` and
    control never reaches the guard -- verified by construction. What it catches is
    a ForwardRef that survives a SUCCESSFUL rebuild, which is the state this walk
    is in whenever the ``model_rebuild()`` call above is removed or reordered after
    the field iteration. That is the regression it exists for, and it is reachable.
    """
    import importlib
    import typing

    import pydantic

    seen: set[type] = set()
    frontier: list[type] = []
    for spec in ENTRY_CONFIGS:
        modname, _, attr = spec.partition(":")
        model = getattr(importlib.import_module(modname), attr)
        if model not in seen:
            seen.add(model)
            frontier.append(model)
    while frontier:
        model = frontier.pop()
        model.model_rebuild()
        for fieldname, info in model.model_fields.items():
            stack = [info.annotation]
            while stack:
                node = stack.pop()
                if isinstance(node, typing.ForwardRef):
                    raise RuntimeError(
                        f"{model.__name__}.{fieldname} still carries an unresolved forward "
                        f"reference {node!r} after model_rebuild(); the closure would omit "
                        f"it silently, so this page refuses to render instead."
                    )
                if isinstance(node, type) and issubclass(node, pydantic.BaseModel) and node not in seen:
                    seen.add(node)
                    frontier.append(node)
                stack.extend(typing.get_args(node) or ())
    return sorted(seen, key=lambda m: (m.__module__, m.__name__))


def _nested_section(already: tuple[type, ...]) -> str:
    """Every closure member the hand-written sections above do not already table.

    SPLIT ON ROOT-NESS, NOT ON WHAT THE HAND-WRITTEN SECTIONS HAPPEN TO COVER.
    `already` names only the models `_render` tables by hand, so every OTHER
    `ENTRY_CONFIGS` root lands here too -- and a root is a file its reader fills
    in, not a block nested inside one. Labelling it "nested" was true while
    `ENTRY_CONFIGS` and `already` held the same three models and became false the
    moment a root was added without a hand-written section, which is a state no
    post-condition on the tuple can see. The membership test below reads
    `ENTRY_CONFIGS` so the two cannot drift again.

    The nested bucket is mostly report style sub-models -- panel styles, figure
    defaults, interactive backends. They are on this page because `### D99` selects
    the population by REACHABILITY rather than by a hand-list, and a reader who
    never writes a ``report:`` block will not meet them in practice.
    """
    import importlib

    roots = set()
    for spec in ENTRY_CONFIGS:
        modname, _, attr = spec.partition(":")
        roots.add(getattr(importlib.import_module(modname), attr))
    rest = [m for m in _closure_models() if m not in already]
    own_file = [m for m in rest if m in roots]
    nested = [m for m in rest if m not in roots]

    out: list[str] = []
    if own_file:
        out += [
            "## Other configs you fill in",
            "",
            "Each of these is a file of its own, named on a command line rather than",
            "nested inside another config, and each is a config you fill in that no",
            "section above introduces yet.",
            "",
        ]
        for model in own_file:
            out += [f"### {model.__name__}", "", f"From `{model.__module__}`.", "", _table(model), ""]

    out += [
        "## Nested config blocks",
        "",
        "A field's type in a config above reaches each block below. You write each block",
        "nested inside the configs above and never as a separate file. Many blocks are",
        "report styling that you can leave at the defaults. The page tables them because the",
        "population of this page is every config an entry config can reach, so every",
        "referenced config is documented.",
        "",
    ]
    for model in nested:
        out += [f"### {model.__name__}", "", f"From `{model.__module__}`.", "", _table(model), ""]
    return "\n".join(out)


#: Style of the standalone configuration pages. The page follows the visitor's colour scheme through the
#: `prefers-color-scheme` media query (the one signal a static page has) and takes its colours from the two palettes
#: of `mkdocs.yml` `theme.palette`: scheme `slate` (primary blue grey, accent light blue) and scheme `default`
#: (primary blue grey, accent blue). Material's stylesheet (palette.*.min.css) defines blue grey as #546d78 (dark
#: variant #455a63, light variant #607c8a), accent light blue as #0091eb and accent blue as #4287ff; the header,
#: borders and links use those. The two background greys are not Material's. Dark is the
#: base because the site opens in `slate`; the light rules sit inside the media query. The selectors start at `body`
#: so that they win over the Tabulator style that the page inlines, whatever order the two arrive in.
_STANDALONE_STYLE = (
    "body{font-family:system-ui,sans-serif;margin:1rem 1.5rem;background:#2e303e;color:#ffffffde}"
    "h1{font-size:1.4rem;margin:0 0 .4rem}"
    "a{color:#0091eb}"
    "body .tabulator{background-color:#2e303e;border:1px solid #455a63;color:#ffffffde}"
    "body .tabulator .tabulator-header{background-color:#455a63;color:#fff;border-bottom:1px solid #455a63}"
    "body .tabulator .tabulator-header .tabulator-col{background-color:#455a63;border-right:1px solid #607c8a}"
    "body .tabulator .tabulator-header .tabulator-col input"
    "{background:#2e303e;color:#ffffffde;border:1px solid #607c8a}"
    "body .tabulator .tabulator-tableholder .tabulator-table{background-color:#2e303e;color:#ffffffde}"
    "body .tabulator .tabulator-row{background-color:#2e303e;color:#ffffffde}"
    "body .tabulator .tabulator-row.tabulator-row-even{background-color:#363948}"
    "body .tabulator .tabulator-row .tabulator-cell{border-right:1px solid #455a63}"
    "@media (prefers-color-scheme: light){"
    "body{background:#fff;color:#000000de}"
    "a{color:#4287ff}"
    "body .tabulator{background-color:#fff;border:1px solid #cfd8dc;color:#000000de}"
    "body .tabulator .tabulator-header .tabulator-col input{background:#fff;color:#000000de;border:1px solid #b0bec5}"
    "body .tabulator .tabulator-tableholder .tabulator-table{background-color:#fff;color:#000000de}"
    "body .tabulator .tabulator-row{background-color:#fff;color:#000000de}"
    "body .tabulator .tabulator-row.tabulator-row-even{background-color:#f5f7f8}"
    "body .tabulator .tabulator-row .tabulator-cell{border-right:1px solid #cfd8dc}"
    "}"
)


def _standalone_table_page(model: type, title: str) -> str:
    """A full-width HTML page with the sortable, filterable Tabulator table of one config model.

    The page uses the table builder of the reports, with the Tabulator assets inlined, so it needs no network
    at view time. It is plain text per cell: the markdown escapes of `_table` belong to the schema page only.
    Deliberately NOT a markdown page: the theme caps the width of a page, and the width is the reason this
    page exists, because the Description column forces sideways scrolling on the schema page.
    """
    import pandas as pd

    from hhemt.report_renderers._tabulator_defaults import (
        build_columns_spec,
        build_html_document,
        build_options_dict,
    )

    rows = []
    for record in field_records(model):
        options = record["options"] or {}
        type_text = "\n".join([str(record["type"]), *(f"{k} {v}" for k, v in options.items())])
        rows.append(
            {
                "Field": str(record["name"]),
                "Type": type_text,
                "Required": _required_cell(record),
                "Default": "" if record["default"] is None else str(record["default"]),
                "Description": str(record["description"]),
            }
        )
    frame = pd.DataFrame(rows)
    columns = build_columns_spec(frame, visible_columns_default=None, header_filter=True)
    widths = {"Field": 260, "Type": 220, "Required": 220, "Default": 160, "Description": 900}
    for spec in columns:
        spec["width"] = widths[spec["field"]]
        if spec["field"] in ("Type", "Required", "Description"):
            spec["formatter"] = "textarea"
    options = build_options_dict(
        frame,
        columns_spec=columns,
        table_height="calc(100vh - 7rem)",
        pagination_size=0,
        persistence_id=None,
        extra_options={"layout": "fitDataStretch"},
    )
    heading = (
        f"<style>{_STANDALONE_STYLE}</style>\n"
        f"<h1>{title}: configuration schema</h1>\n"
        '<p><a href="config-schema/">Back to the configuration schema</a></p>\n'
    )
    return build_html_document(
        title=f"{title}: configuration schema",
        container_id="config_schema_table",
        body_heading_html=heading,
        options=options,
        js_mode="inline",
        renderer_name="config_reference",
        column_panel=False,
    )


def _render() -> str:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.experiment_definition import ExperimentConfig
    from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
    from hhemt.config.reprex_config import reprex_config
    from hhemt.config.sif_build import sif_build_config
    from hhemt.config.system import system_config

    # Resolve every forward reference before ANY table is rendered: the walk calls model_rebuild(), and a table
    # rendered earlier prints an unresolved annotation as ForwardRef('...') (hpc_system_config.container).
    _closure_models()
    marker = _load_lint().GENERATED_MARKER
    sections = [
        f"<!-- {marker} -->",
        "<!-- DO NOT EDIT THIS FILE. hooks/config_reference.py rewrites this file from the config models on",
        "     every documentation build, so any",
        "     hand-edit is silently discarded at the next build. To change the page,",
        "     edit the Pydantic field or this hook. -->",
        "",
        "# Configuration schema",
        "",
        "The tables below hold every field of every config you fill in. The configs are the system,",
        "analysis, and HPC-system configs, the SIF build config, the experiment descriptor, and",
        "the reprex config. The tables also hold every config reachable from them by following",
        "a field's type. The models supply every row at documentation build time. No row is",
        "hand-maintained or hand-selected. The population is that transitive closure, so a",
        "nested config appears when a field's type references the config.",
        "Config models no entry config reaches are not tabled here.",
        "",
        'All config models set `extra="forbid"`, so an unrecognised key is an error and the',
        "load step catches a mistyped key before dispatch.",
        "",
        "Start from the in-repo templates",
        "`test_data/norfolk_coastal_flooding/template_system_config.yaml` and",
        "`template_analysis_config.yaml`. The page [Fill in your configuration](../how-to/config-filling.md)",
        "gives a step-by-step path through the fields. For the",
        "cluster profile see [HPC-profile setup](../how-to/hpc-profile-setup.md).",
        "",
        "## System config",
        "",
        "Describes the modelled area. The config holds the DEM, the SWMM model, and the toggles that",
        "decide the model types to build and run. The types are TRITON, TRITON-SWMM, and SWMM.",
        "",
        _table(system_config),
        "",
        f"The [full-width system table]({_STANDALONE_PAGES[0][2]}) sorts and filters the fields.",
        "",
        "## Analysis config",
        "",
        "Describes the analysis. The config holds the events to simulate, the way each simulation runs,",
        "and the outputs produced from the results.",
        "",
        _table(analysis_config),
        "",
        f"The [full-width analysis table]({_STANDALONE_PAGES[1][2]}) sorts and filters the fields.",
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
        "## SIF build config",
        "",
        "Describes the machine that BUILDS container images. The config holds the place the images are written,",
        "the partition that builds them, and the checkout of the toolkit that is baked in.",
        "You write this config only in container mode, and `hhemt build-sifs` takes the",
        "config path on the command line.",
        "",
        _table(sif_build_config),
        "",
        "## Experiment descriptor",
        "",
        "The `experiment.yaml` at the root of an experiment definition. The file names the configs",
        "the run uses, the datasets the run consumes, and the toolkit release that the run is pinned to,",
        "so a third party can obtain everything the run needed.",
        "",
        _table(ExperimentConfig),
        "",
        "## Reprex config",
        "",
        "A REPRODUCER fills in this file to run someone else's bundle on their own system.",
        "The file holds their account, their partitions, and the location of their copy of the image. The file ships",
        "in the bundle as a template with the producer's values removed.",
        "",
        _table(reprex_config),
        "",
        _nested_section(
            (
                system_config,
                analysis_config,
                hpc_system_config,
                PartitionSpec,
                sif_build_config,
                ExperimentConfig,
                reprex_config,
            )
        ),
        "",
        _conditional_section(),
        "",
        "## See also",
        "",
        "- [Reporting sets](reporting-sets.md) covers the choice of the figures and tables in a report.",
        "- [Fill in your configuration](../how-to/config-filling.md) gives the step-by-step path through the fields.",
        "- [HPC-profile setup](../how-to/hpc-profile-setup.md) covers the third config, describing your cluster.",
        "- [Rerun triggers](rerun-triggers.md) covers the limits of editing the derived sensitivity CSV.",
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
        "predicate the validator enforces, so the table always agrees with the",
        "code.",
        "",
        "| Trigger | Required fields |",
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
    _load_local_src().bind_local_src(_SRC, {"hhemt"})
    markdown = _render()
    # Two passes over two different populations. The generator's OWN prose --
    # headings, intro, section labels -- answers to all four classes. The
    # rendered table cells are `src/`-side `Field(description=...)` strings and
    # answer to the binary classes only. `_authored_prose()` is unchanged.
    _assert_lint_clean("\n".join(_authored_prose()))
    _assert_lint_clean(markdown, binary_only=True)
    target = Path(config.docs_dir) / GENERATED_URI
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    models = {"system": system_config, "analysis": analysis_config}
    for key, title, name in _STANDALONE_PAGES:
        (target.parent / name).write_text(_standalone_table_page(models[key], title), encoding="utf-8")
    return config
