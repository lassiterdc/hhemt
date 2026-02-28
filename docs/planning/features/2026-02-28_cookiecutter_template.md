# Copier Template System: Implementation Plan

**Written**: 2026-02-28
**Last edited**: 2026-02-28 — added Dependencies section documenting three touch points with the in-progress subagent refactor (2026-02-28_system-level-subagents-in-git-repo.md)

---

## Overview

This is a four-phase plan for building and adopting a Copier-based Python project template system.

| Phase | Goal | Scope |
|-------|------|-------|
| **Phase 1** | Build the `cookiecutter-python-claude` template repo | New repo; no changes to existing projects |
| **Phase 2** | Spin up `multidriver-swg` from the template; verify ReadTheDocs; walk through the update tutorial | New repo; template repo may receive minor fixes |
| **Phase 3** | Make a real template improvement and propagate it to `multidriver-swg` via `copier update`; confirm understanding | Exercises the full update workflow end-to-end |
| **Phase 4** | Retroactively adopt Copier in the TRITON-SWMM toolkit | Modifies existing toolkit repo |

Each phase has its own Definition of Done. Phases must be completed in order — each depends on the previous.

---

## Dependencies on Other In-Progress Work

### Subagent Refactor (`2026-02-28_system-level-subagents-in-git-repo.md`)

An in-progress refactor is promoting project-level agents to user-level (`~/.claude/agents/`) tracked in `~/dev/claude-workspace/`. This affects three specific points in this plan. **Do not implement the affected items until the subagent refactor is complete.**

| Touch point | Current plan | Required update after refactor |
|-------------|-------------|-------------------------------|
| **Phase 1 — `template/.claude/agents/_example-agent.md`** | Stub shows basic frontmatter pattern (`name`, `description`, `model`) | Update stub to show the new pattern: `skills: [<repo>-context]` in frontmatter; no startup-reads block in body. Also update the stub's accompanying comments to explain that real agents live in `~/dev/claude-workspace/agents/`, not `.claude/agents/`. |
| **Phase 1 — `template/.claude/agents/` directory** | Template generates `.claude/agents/` with one stub file | After refactor, `.claude/agents/` in downstream projects should contain only a `README.md` redirect (same pattern as the toolkit). Update the template to generate a `README.md` instead of a flat agent stub. The stub moves to `template/claude-workspace-example/` or is documented in the template `README.md` instead. |
| **Phase 4 — pre-migration audit table** | Row for `template/.claude/agents/_example-agent.md` vs. toolkit's three agent files | After refactor, the toolkit's `.claude/agents/` contains only a `README.md`. The conflict row changes: template generates a `README.md` redirect; toolkit already has one; resolution is trivially "accept template's" (they'll be identical in spirit). |

**Also relevant for future phases**: once `multidriver-swg` has meaningful source code, it will need a `multidriver-swg-context` skill in `claude-workspace`. This is out of scope for all four phases here — it's a natural next step documented as a follow-on task, not a gate.

---

# Phase 1: Build the Template Repository

### Task Understanding

### Requirements

Create a Copier template repository based on the current TRITON-SWMM toolkit repo that:

1. Extracts all **project-agnostic** infrastructure from the current repo into a reusable template
2. Includes a project-agnostic version of the Claude Code infrastructure:
   - `CLAUDE.md` (with instructions pointing to `.prompts/` docs)
   - `.prompts/architecture.md` (generic starter scaffold)
   - `.prompts/conventions.md` (generic project conventions — Part I verbatim + portable Part II items)
   - `.prompts/implementation_plan.md` (fully portable as-is)
   - `.prompts/qaqc_and_commit.md` (fully portable as-is)
   - `.prompts/proceed_with_implementation.md` (fully portable as-is)
3. All new projects should have a `.prompts/architecture.md` and `.prompts/conventions.md` at minimum
4. Include a working documentation framework wired to ReadTheDocs (MkDocs + Material theme; see Documentation Framework decision below)
5. The example agent stub must be clearly marked so Claude won't accidentally invoke it as a real agent

### Assumptions

- The template will live in a **new, separate repository** named `cookiecutter-python-claude` under `~/dev/`
- The template targets Python projects with a similar structure (src layout, pytest, ruff, pre-commit, Conda/uv environments)
- **Copier** is the templating engine (replaces Cookiecutter; see System Overview and comparison table in Implementation Strategy for rationale)
- The `.claude/` infrastructure (agent files, settings) is project-agnostic enough to include with project-specific content stripped
- Project-specific content (TRITON-SWMM source code, SWMM configs, HPC workflow) is excluded
- Template variables cover minimal essential customization: project name, author, description, GitHub username
- This is a **planning-only session** — the repo will be created locally and pushed to GitHub as private in a subsequent session

### Success Criteria

1. A self-contained Copier template repo exists and generates a new project with `copier copy gh:username/cookiecutter-python-claude path/to/new-project`
2. Generated project includes all project-agnostic Claude Code infrastructure
3. Generated `CLAUDE.md` and `.prompts/` files are ready to use out-of-the-box
4. Generated project contains `.copier-answers.yml` enabling future `copier update` runs
5. The template's `README.md` clearly explains how to use it, what it provides, and how to receive template updates
6. Generated project has a working docs framework (MkDocs + Material) that ReadTheDocs can build
7. The example agent stub is clearly annotated so Claude won't accidentally use it

---

### System Overview

This section describes how the template system works end-to-end: initial project generation, local customization, and propagating future template improvements back into existing projects.

### Generating a New Project

```bash
# Install Copier once
pip install copier

# Generate a new project from the template
copier copy gh:lassiterdc/cookiecutter-python-claude ~/dev/my-new-project

# Copier prompts for: project_name, project_slug, package_name, author_name, etc.
# Generates the full project directory, then writes .copier-answers.yml
```

After generation, the developer:
1. Fills in `.prompts/architecture.md` with their project's structure
2. Adds project-specific terminology and patterns to Part II of `.prompts/conventions.md`
3. Replaces `_example-agent.md` with real agent files as the project grows
4. Populates `docs/` stubs with actual content

### What Copier Stores (`.copier-answers.yml`)

Copier automatically writes this file to every generated project:

```yaml
# This file was auto-generated by Copier. Do NOT edit manually.
_src_path: https://github.com/lassiterdc/cookiecutter-python-claude.git
_commit: v1.0.0   # exact template tag/commit used
project_name: My New Project
project_slug: my-new-project
package_name: my_new_project
author_name: Daniel Lassiter
author_email: daniel@example.com
github_username: lassiterdc
description: A short description.
python_version: "3.12"
```

This file is the anchor for all future updates. **Do not edit it manually.**

### Propagating Template Changes to Existing Projects

This is the primary advantage of Copier over Cookiecutter.

**Scenario**: You improve the Universal Principles section of `conventions.md` in the template, or add a new `.prompts/` file. You want those improvements in `project-a`, `project-b`, and `project-c`.

```bash
# In each downstream project:
cd ~/dev/project-a
copier update
```

**What happens internally (three-way merge):**

```
BASE  = what the template produced at the last generation/update (v1.0.0)
OURS  = your current project file (with local customizations)
THEIRS = what the updated template now produces (v1.1.0)
```

Copier regenerates the template at the new version, diffs `THEIRS` vs `BASE` to find what changed in the template, then applies only those changes to `OURS`. Sections you added locally (your Part II project-specific rules, your architecture description) are untouched as long as the template didn't modify the same lines.

**Concrete example for `.prompts/conventions.md`:**

| Situation | Outcome |
|-----------|---------|
| Template improves Part I (Universal Principles); you only edited Part II | ✅ Automatic merge — your Part II survives, template's Part I improvement applied |
| Template adds a new section at the end; you didn't touch that area | ✅ Automatic merge |
| Template rewrites Part I; you also edited Part I | ⚠️ Conflict markers inserted — you resolve manually |
| Template adds a new file (e.g., `.prompts/new_prompt.md`); you didn't have it | ✅ File added automatically |

**Conflict resolution** produces standard-format markers you resolve in your editor:

```
<<<<<<< LOCAL (your version)
Your customized content here
=======
Template's updated content here
>>>>>>> REMOTE
```

### Update Strategy Recommendation

To maximize clean propagation and minimize conflicts:

1. **Keep template sections intact in downstream projects.** Don't rewrite the Part I Universal Principles text — the template "owns" that section. Add your project-specific content only in Part II and III stubs.
2. **Tag template releases.** Use semantic versioning tags (e.g., `v1.0.0`, `v1.1.0`) in the template repo. Copier records the exact tag in `.copier-answers.yml`, making it easy to see which version each project is on.
3. **Update periodically, not constantly.** Run `copier update` in downstream projects when a meaningful template improvement lands — not on every commit.
4. **Skip tasks during update.** Some Copier post-generation hooks (e.g., `git init`, `pre-commit install`) are not idempotent. Use `copier update --skip-tasks` to avoid re-running them.

### What Copier Cannot Propagate

- **Files you deleted locally**: Copier respects deletions and won't re-introduce them during `copier update`. Use `copier recopy` to regenerate from scratch if needed.
- **Completely restructured files**: If the template fundamentally reorganizes a file that you've heavily customized, expect manual conflict resolution.
- **Changes to `.copier-answers.yml` itself**: The answers file is managed by Copier; if template variables change names between versions, manual migration may be needed.

---

### Evidence from Codebase

**Files inspected and key findings:**

- `.prompts/implementation_plan.md` — fully project-agnostic. Can be copied verbatim.
- `.prompts/qaqc_and_commit.md` — fully project-agnostic. Can be copied verbatim.
- `.prompts/proceed_with_implementation.md` — fully project-agnostic. Can be copied verbatim.
- `.prompts/conventions.md` — **mixed**. Part I (Universal Principles, ~lines 7–80) is project-agnostic. Part II (Project-Specific Rules, lines 83–210) is mostly TRITON-SWMM-specific, but the following subsections are portable: "Planning document lifecycle", "Recording out-of-scope observations", "Code style". Part III (AI Working Norms, lines 239–274) is mostly portable with minor toolkit-specific references.
- `.prompts/architecture.md` — fully project-specific. Template version is a minimal stub with headings to fill in.
- `.prompts/debugging_hpc_analysis.md` — fully project-specific HPC debugging protocol. Not included in template.
- `CLAUDE.md` — minimal (just instructs reading two docs). Fully portable pattern.
- `.claude/agents/` — three project-specific agents. Template includes one example stub, clearly annotated as non-functional.
- `.claude/settings.local.json` — local machine permissions. Template includes a generic version.
- `.pre-commit-config.yaml` — ruff hooks (generic) + project-specific `check-claude-docs` hook. Template includes ruff only; check-claude-docs included as commented-out example.
- `pyproject.toml` — project metadata, ruff config, pytest config. Template replaces project-specific fields with Copier `[[var]]` variables.
- `scripts/check_doc_freshness.py` — project-specific file mappings. Template includes a generic stub.
- `.gitignore` — Standard Python gitignore from github/gitignore. Fully portable.
- `CONTRIBUTING.md` — partially generic. Template strips TRITON-SWMM references.
- `docs/conf.py` + `docs/index.rst` — current toolkit uses **Sphinx + RST + `sphinx_rtd_theme`** with nbsphinx and mermaid extensions. Replaced with MkDocs + Material in template (see Documentation Framework decision).
- `.readthedocs.yaml` — wires ReadTheDocs to Sphinx config. Template replaces with MkDocs equivalent.
- `docs/planning/` tree — the planning doc subdirectory structure is a project-agnostic convention worth scaffolding.

---

### Implementation Strategy

### Cookiecutter vs. Copier (Decision: Copier selected)

| Feature | Cookiecutter | Copier |
|---------|-------------|--------|
| **Maturity** | ~10 years, very widely used | ~5 years, rapidly growing |
| **Template update mechanism** | None — generated projects diverge forever | `copier update` syncs changes from template into existing projects via 3-way merge |
| **Syntax** | Jinja2 `{{cookiecutter.var}}` | Jinja2 `[[var]]` (customizable delimiters; no namespace prefix needed) |
| **Configuration** | `cookiecutter.json` (JSON) | `copier.yml` (YAML, richer question types) |
| **Python packaging** | `pip install cookiecutter` | `pip install copier` |
| **ReadTheDocs support** | Full | Full |
| **Ecosystem / plugins** | Larger (PyScaffold, Cruft extend it) | Smaller but growing |
| **Hooks** | `pre_gen_project.py` / `post_gen_project.py` | Same pattern |
| **Private GitHub repos** | `cookiecutter gh:user/repo` | `copier copy gh:user/repo path/` |
| **Conflict resolution** | N/A (no update mechanism) | 3-way merge with inline conflict markers |
| **Key advantage** | Battle-tested, maximum compatibility | Template updates propagate to all derived projects |
| **Key disadvantage** | No update path — once generated, diverges forever | Slightly less familiar; IDE merge UI has known display quirks during conflict resolution |

**Rationale for choosing Copier**: The update mechanism is the decisive factor for this use case. As `conventions.md` Part I evolves (new universal principles, refined AI working norms), those improvements can be propagated to all downstream projects with `copier update` rather than manually copying changes. The cleaner variable syntax (`[[project_name]]` vs `{{cookiecutter.project_name}}`) is a secondary benefit. See the System Overview section for the full update workflow.

### Chosen Documentation Framework: MkDocs + Material

The current toolkit uses Sphinx + RST. For the template, **MkDocs + Material theme** is recommended instead:

| | Sphinx + RST | MkDocs + Material |
|-|-------------|------------------|
| **Format** | reStructuredText (verbose, niche syntax) | Markdown (consistent with `.prompts/` files) |
| **Configuration** | `conf.py` (Python) | `mkdocs.yml` (simple YAML) |
| **Theme** | `sphinx_rtd_theme` (dated look) | Material for MkDocs (modern, widely admired) |
| **ReadTheDocs support** | Full | Full (native MkDocs support since 2022) |
| **Notebook support** | `nbsphinx` (complex setup) | `mkdocs-jupyter` (simpler) |
| **Autodoc (API docs)** | `sphinx.ext.autodoc` (powerful) | `mkdocstrings` (comparable) |
| **Mermaid diagrams** | `sphinxcontrib-mermaid` | Built-in with Material theme |
| **Setup complexity** | High (conf.py, RST syntax, extensions) | Low (one YAML file, Markdown) |
| **Rationale for template** | Already used in TRITON-SWMM toolkit | Better fit for a general Python template; Markdown consistency; lower barrier |

The toolkit can keep Sphinx if desired. The template uses MkDocs, and new projects can switch to Sphinx if they have notebook-heavy docs.

### Chosen Approach

Create a **new git repository** named `cookiecutter-python-claude` at `~/dev/cookiecutter-python-claude/` with the Copier layout:

```
cookiecutter-python-claude/
├── copier.yml                                    # Template variables and questions (replaces cookiecutter.json)
├── README.md                                     # How to use the template
├── .copier-tasks.py                              # Post-generation hook: validates package_name
└── template/                                     # All generated files live here (Copier convention)
    ├── CLAUDE.md
    ├── .prompts/
    │   ├── architecture.md                       # Generic stub (to populate)
    │   ├── conventions.md                        # Part I verbatim + portable Part II + generic Part III
    │   ├── implementation_plan.md                # Copied verbatim
    │   ├── qaqc_and_commit.md                    # Copied verbatim
    │   └── proceed_with_implementation.md        # Copied verbatim
    ├── .claude/
    │   ├── settings.local.json                   # Generic permissions stub
    │   └── agents/
    │       └── _example-agent.md                 # Non-functional pattern stub (prefixed _ + annotated)
    ├── .pre-commit-config.yaml                   # ruff hooks + commented check-claude-docs example
    ├── .gitignore                                 # Standard Python gitignore
    ├── .readthedocs.yaml                         # MkDocs config for ReadTheDocs
    ├── pyproject.toml                             # With Copier vars ([[project_slug]] syntax)
    ├── mkdocs.yml                                 # MkDocs + Material configuration
    ├── CONTRIBUTING.md                            # Generic version
    ├── README.md                                  # Project README stub
    ├── HISTORY.md                                 # Changelog stub
    ├── scripts/
    │   └── check_doc_freshness.py                 # Generic stub with empty mappings
    ├── docs/
    │   ├── index.md                               # Home page + Mermaid flowchart (copier update workflow)
    │   ├── installation.md                        # Install steps + Material admonition block
    │   ├── usage.md                               # Usage guide + Mermaid sequence diagram
    │   ├── api.md                                 # mkdocstrings autodoc block
    │   └── planning/
    │       ├── README.md                          # Planning doc conventions
    │       ├── bugs/completed/.gitkeep
    │       ├── features/completed/.gitkeep
    │       └── refactors/completed/.gitkeep
    ├── src/
    │   └── [[package_name]]/
    │       └── __init__.py
    └── tests/
        └── __init__.py
```

Note: Copier's default template directory is the repo root (files alongside `copier.yml`), but using a `template/` subdirectory is cleaner for separating template files from template metadata. The `_subdirectory: template` key in `copier.yml` activates this.

### Alternatives Considered

- **In-repo template subfolder**: Keep template inside TRITON-SWMM toolkit. Rejected — separate repo allows independent evolution.
- **Sphinx instead of MkDocs**: Keep same docs stack. Rejected for the template — RST syntax and `conf.py` complexity are higher barriers for a general template.
- **Cookiecutter instead of Copier**: Rejected — Copier's update propagation is the decisive advantage for this use case. See comparison table.

---

### File-by-File Change Plan

This plan creates a **new repository** — no files in the TRITON-SWMM toolkit are modified.

### `copier.yml`
Declares template variables with Copier's YAML question format:
```yaml
_subdirectory: template   # Generated files live in template/ subdirectory
_tasks:
  - python .copier-tasks.py validate  # Post-generation validation

project_name:
  type: str
  help: Human-readable project name (e.g. "My Python Project")
  default: My Python Project

project_slug:
  type: str
  help: Repository/directory name — use hyphens (e.g. "my-python-project")
  default: "[[project_name|lower|replace(' ', '-')]]"

package_name:
  type: str
  help: Python import name — use underscores (e.g. "my_python_project")
  default: "[[project_slug|replace('-', '_')]]"

author_name:
  type: str
  help: Your full name
  default: Daniel Lassiter

author_email:
  type: str
  help: Your email address
  default: daniel.lassiter@outlook.com

github_username:
  type: str
  help: Your GitHub username
  default: lassiterdc

description:
  type: str
  help: One-sentence project description

python_version:
  type: str
  help: Minimum Python version
  default: "3.12"
```
Note: `[[...]]` is Copier's default variable delimiter (vs Cookiecutter's `{{...}}`). Copier supports Jinja2 filters inline in defaults (e.g., `lower`, `replace`), which auto-derives `project_slug` from `project_name`. Personal defaults (`author_name`, `author_email`, `github_username`) are pre-populated — just press Enter to accept at generation time.

### `.copier-tasks.py`
Short validation script run after generation. Checks that `package_name` matches `^[a-zA-Z_][a-zA-Z0-9_]*$`. If not, prints a clear error and exits non-zero. This catches cases where the user enters `my-package` (hyphen) which would break `import my-package`.

### `README.md` (template root)
- What this template provides
- Usage: `copier copy gh:lassiterdc/cookiecutter-python-claude path/to/new-project`
- Update usage: `copier update` (run from inside any derived project)
- List of generated files and their purpose
- Customization guide (what to fill in after generation: `architecture.md`, `conventions.md` Part II)

### `template/CLAUDE.md`
Identical in pattern to the current repo's `CLAUDE.md`:
```markdown
# CLAUDE.md

Read these files before beginning any task:
- `.prompts/conventions.md`
- `.prompts/architecture.md`
```

### `template/.prompts/architecture.md`
Generic stub. Includes required section headings with fill-in-the-blank placeholders and a brief note explaining how to use the file. Sections: Project Overview, Key Modules, Workflow Phases (if applicable), Configuration System (if applicable), Gotchas.

### `template/.prompts/conventions.md`
Structure:
- **Part I: Universal Principles** — copied verbatim from current `conventions.md` (lines 7–80): Raise questions, Plan then implement, Do things right, Backward compatibility, Default arguments, Avoid aliases, No cruft, Docstrings/type hints, Fail-fast, Preserve context in exceptions, Prefer log-based checks, System-agnostic, Track utility candidates
- **Part II: Project-Specific Rules** — three portable subsections ported from current Part II:
  - "Planning document lifecycle" (the `docs/planning/` tree structure and bug/feature/refactor organization)
  - "Recording out-of-scope observations" (the `tech_debt_<topic>.md` pattern)
  - "Code style" (ruff, Python ≥3.10 target; remove Pydantic-specific content as that's TRITON-specific)
  - Stub subsection: "Terminology" — placeholder with note to populate with project-specific terms
  - Stub subsection: "Architecture patterns" — placeholder for project-specific patterns
- **Part III: AI Working Norms** — adapted from current Part III with toolkit-specific filenames generalized (e.g., "plan-then-implement workflow" section generalized; "spawning subagents" section kept verbatim as it's fully generic)

**Jinja2 escaping**: Verified — this file contains no `{{` or `}}` literals. No escaping needed.

### `template/.prompts/implementation_plan.md`
Copied verbatim from current repo. **Jinja2 escaping**: Verified — file contains no `{{` or `}}` literals. No escaping needed.

### `template/.prompts/qaqc_and_commit.md`
Copied verbatim. No Jinja2 conflicts.

### `template/.prompts/proceed_with_implementation.md`
Copied verbatim. No Jinja2 conflicts.

### `template/.claude/settings.local.json`
Generic permissions list:
```json
{
  "permissions": {
    "allow": [
      "Bash(python -m pytest:*)",
      "Bash(pytest:*)",
      "Bash(ruff check:*)",
      "Bash(ruff format:*)",
      "Bash(git log:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git mv:*)",
      "Bash(ls:*)",
      "Bash(find:*)",
      "Bash(grep:*)",
      "Bash(echo:*)",
      "Bash(wc:*)"
    ]
  }
}
```

### `template/.claude/agents/_example-agent.md`

The filename is prefixed with `_` and the file begins with a `<!--` HTML comment block clearly marking it as non-functional. Claude's agent discovery reads the YAML frontmatter `name:` and `description:` fields to decide whether to invoke an agent. By setting `name: _example-agent` and adding a note in `description:` that it is a non-functional stub, it will not be auto-invoked. Additionally, the top of the file will have a bold warning:

```markdown
<!--
NON-FUNCTIONAL STUB — This file is a template/pattern example only.
Do NOT rename this file or populate the description field — Claude
will not invoke agents whose name starts with '_' and whose description
explicitly marks them as stubs.
-->
```

The body of the file shows the agent frontmatter pattern and startup read pattern with comments explaining each section. The `description:` field in the frontmatter will read: `"NON-FUNCTIONAL STUB — replace this with your agent description to activate"`.

### `template/.pre-commit-config.yaml`
Ruff hooks only (active). The `check-claude-docs` local hook is included as a commented-out example with instructions on how to enable it once `scripts/check_doc_freshness.py` is populated.

### `template/.gitignore`
Copied verbatim from current repo (standard Python gitignore).

### `template/.readthedocs.yaml`
MkDocs variant:
```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "[[python_version]]"
mkdocs:
  configuration: mkdocs.yml
python:
  install:
    - method: pip
      path: .
    - requirements: requirements.txt
```

### `template/mkdocs.yml`
MkDocs + Material configuration:
```yaml
site_name: "[[project_name]]"
site_description: "[[description]]"
site_author: "[[author_name]]"
repo_url: "https://github.com/[[github_username]]/[[project_slug]]"
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
  palette:
    - scheme: default
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
nav:
  - Home: index.md
  - Installation: installation.md
  - Usage: usage.md
  - API Reference: api.md
```

### `template/pyproject.toml`
Adapted from current repo with Copier variables (`[[var]]` syntax):
- `name = "[[project_slug]]"`
- `description = "[[description]]"`
- `authors = [{name = "[[author_name]]", email = "[[author_email]]"}]`
- `requires-python = ">= [[python_version]]"`
- Keep ruff config (line-length = 120, same lint rules)
- Keep pytest config (testpaths, norecursedirs, markers)
- No TRITON-SWMM dependencies
- Add `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` as optional docs dependencies

### `template/CONTRIBUTING.md`
Generic version: strip TRITON-SWMM-specific references. Keep: fork/branch/PR workflow, AI context documentation section (generalized), pre-commit setup, testing instructions. Update to reference MkDocs instead of Sphinx.

### `template/README.md`
Stub with: project name (from var), description (from var), badges placeholder, installation instructions, quick start, link to docs.

### `template/HISTORY.md`
One-line stub: `# Changelog`.

### `template/scripts/check_doc_freshness.py`
Generic stub: preserve the script structure, pre-commit hook invocation, and the `AGENT_MAPPING` / `CLAUDE_MD_TRIGGERS` dict pattern with empty values and `# TODO: populate` comments.

### `template/docs/index.md`
MkDocs home page. Contains:
- Project name header and one-line description (using `[[project_name]]` and `[[description]]`)
- Quick-links to Installation, Usage, and API Reference
- A **Mermaid flowchart** showing the Copier template update workflow (template repo → `copier copy` → new project → local customization → `copier update` ← template improvements). This verifies Mermaid rendering is wired correctly in Material theme.

### `template/docs/installation.md`
Sample installation page with:
- A conda environment creation command (placeholder env name)
- A `pip install -e .` development install step
- An admonition block (`!!! note`) to verify Material admonition rendering

### `template/docs/usage.md`
Sample usage page with:
- A brief "getting started" narrative
- A fenced code block with syntax highlighting (verifies Pygments integration)
- A second **Mermaid diagram** — a simple sequence diagram showing the plan-then-implement workflow from `.prompts/implementation_plan.md`. Keeps docs self-referential and demonstrates a second diagram type.

### `template/docs/api.md`
API reference page with:
- A `mkdocstrings` autodoc block pointing at `[[package_name]].__init__`
- This requires `[[package_name]]/__init__.py` to have a module-level docstring and a sample function with a Google-style docstring and type hints — verifies `mkdocstrings` renders correctly

The `__init__.py` (updated from the earlier spec) will therefore contain:
```python
"""[[project_name]]: [[description]]"""

__version__ = "0.1.0"


def hello(name: str) -> str:
    """Return a greeting string.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
```

### `template/docs/planning/README.md`
The planning document convention text, extracted from `conventions.md` (the lifecycle section).

### `template/docs/planning/{bugs,features,refactors}/completed/.gitkeep`
Empty files to scaffold the directory structure in git.

### `template/src/[[package_name]]/__init__.py`
Contains a module docstring, version, and a sample typed function so `mkdocstrings` has something real to render:
```python
"""[[project_name]]: [[description]]"""

__version__ = "0.1.0"


def hello(name: str) -> str:
    """Return a greeting string.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
```

### `template/tests/__init__.py`
Empty.

---

### Risks and Edge Cases

| Risk | Mitigation |
|------|-----------|
| Jinja2 syntax in `.prompts/` files conflicts with Copier rendering | **Not a risk** — verified no `{{` / `}}` literals exist in any of the five `.prompts/` source files. No escaping needed. |
| `package_name` entered with hyphens | `.copier-tasks.py` validation hook aborts generation with a clear error |
| `_example-agent.md` description field accidentally triggers Claude invocation | Frontmatter `description` explicitly says "NON-FUNCTIONAL STUB"; HTML comment at top of file; `_` prefix in filename as additional signal |
| MkDocs `mkdocstrings` requires source to be importable during docs build | Include `pip install -e .` in ReadTheDocs build config (already handled by `method: pip, path: .` in `.readthedocs.yaml`) |
| Template diverges from toolkit over time | Acceptable — template is a starting point, not a synchronized fork |

---

### Validation Plan

Since this creates a **new repository** with no changes to TRITON-SWMM toolkit source:

1. **Copier dry-run**: `copier copy --defaults path/to/cookiecutter-python-claude /tmp/test-project` — verify generated structure matches plan
2. **Manual inspection**: Confirm no unresolved `[[...]]` variables remain in generated output
3. **Jinja2 conflict check**: ~~Verify `.prompts/` files render without template errors~~ — pre-verified during planning: no `{{`/`}}` literals in any `.prompts/` source file. No escaping needed.
4. **Agent stub safety check**: Confirm `_example-agent.md` frontmatter won't trigger Claude invocation
5. **Generated project smoke test**: `ruff check .` and `pytest` pass in generated project
6. **MkDocs build**: `mkdocs build` succeeds in generated project (requires `pip install mkdocs mkdocs-material mkdocstrings[python]`)
7. **No TRITON-SWMM smoke tests** — this plan does not modify any existing toolkit files

---

### Documentation and Tracker Updates

- No updates needed to TRITON-SWMM toolkit docs — new repo only
- The template `README.md` is the primary user-facing documentation
- Optional: add a link to the template repo in the toolkit `README.md` once it exists

---

### Decisions Needed from User

All previous decisions are resolved:

1. ✅ **Template repo name/location**: `cookiecutter-python-claude` at `~/dev/`
2. ✅ **Portable Part II items**: Port "Planning document lifecycle", "Recording out-of-scope observations", and "Code style" into the template's `conventions.md`
3. ✅ **Agent stub**: Include `_example-agent.md` — make it clear Claude won't accidentally use it (prefixed `_`, annotated description, HTML comment)
4. ✅ **Copier vs Cookiecutter**: **Copier selected.** The update propagation mechanism is the decisive advantage for this use case. See System Overview and comparison table in Implementation Strategy.
5. ✅ **Session scope**: Planning only — repo will be created locally and pushed to GitHub as private in a subsequent session

All decisions fully resolved. Ready to implement.

---

### Definition of Done

- [ ] New template repository directory created at `~/dev/cookiecutter-python-claude/`
- [ ] `copier.yml` defines all required variables with defaults and help text
- [ ] `template/` subdirectory contains all generated files (`_subdirectory: template` set in `copier.yml`)
- [ ] `.copier-tasks.py` validates `package_name` is a valid Python identifier
- [ ] Template generates a valid project with `copier copy --defaults path/to/template /tmp/test-project`
- [ ] Generated project contains `.copier-answers.yml` (enables future `copier update`)
- [ ] Generated project contains all five `.prompts/` files (architecture stub, conventions generic, implementation_plan, qaqc_and_commit, proceed_with_implementation)
- [ ] Generated `CLAUDE.md` correctly references `.prompts/` docs
- [ ] Generated `.claude/` includes `settings.local.json` and `_example-agent.md` stub
- [ ] `_example-agent.md` has `_` prefix, annotated `description`, and HTML comment block
- [ ] Generated `pyproject.toml` uses Copier `[[var]]` variables for project-specific fields
- [ ] Generated project includes working MkDocs config (`mkdocs.yml`) and `.readthedocs.yaml`
- [ ] `docs/index.md` includes a Mermaid flowchart of the `copier update` workflow
- [ ] `docs/installation.md` includes a Material admonition block
- [ ] `docs/usage.md` includes a syntax-highlighted code block and a Mermaid sequence diagram
- [ ] `docs/api.md` includes a `mkdocstrings` autodoc block pointing at `[[package_name]]`
- [ ] `src/[[package_name]]/__init__.py` includes module docstring and a sample typed function for mkdocstrings to render
- [ ] Template `README.md` documents both `copier copy` (new project) and `copier update` (propagating changes)
- [ ] Planning directory tree scaffolded (`bugs/`, `features/`, `refactors/` each with `completed/`)
- [ ] `ruff check .` passes on the generated project
- [ ] `mkdocs build` succeeds on the generated project

---

### Self-Check Results

1. **Header/body alignment**: All section headings match content. ✓
2. **Section necessity**: All sections are necessary. All decisions resolved. ✓
3. **Conventions alignment**: No changes to existing toolkit source. Plan follows plan-then-implement. ✓
4. **Task-relevance**: Plan is scoped to the template extraction task; no TRITON-SWMM implementation details. ✓

---

# Phase 2: Spin Up `multidriver-swg` and Verify the Template

## Goal

Use the newly built template to create the `multidriver-swg` repo from scratch, confirm that ReadTheDocs builds and renders correctly, and walk through the full update workflow so you understand how to propagate future template changes.

**`multidriver-swg`**: Stochastic weather generator producing correlated compound forcing (rainfall fields + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.

---

### Step 2.1 — Generate the repo

```bash
cd ~/dev
copier copy ~/dev/cookiecutter-python-claude multidriver-swg
```

Copier will prompt through each question. Expected answers:
- `project_name`: `multidriver-swg`
- `project_slug`: `multidriver-swg` (accept default)
- `package_name`: `multidriver_swg` (accept default)
- `author_name`: `Daniel Lassiter` (accept default)
- `author_email`: `daniel.lassiter@outlook.com` (accept default)
- `github_username`: `lassiterdc` (accept default)
- `description`: `Stochastic weather generator producing correlated compound forcing (rainfall fields + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.`
- `python_version`: `3.12` (accept default)

After generation, verify that `.copier-answers.yml` exists at the project root and contains the correct values.

---

### Step 2.2 — Initialize git and push to GitHub

```bash
cd ~/dev/multidriver-swg
git init
git add .
git commit -m "chore: initial project scaffold from cookiecutter-python-claude template"
gh repo create multidriver-swg --private --source=. --remote=origin --push
```

Confirm the repo appears at `https://github.com/lassiterdc/multidriver-swg`.

---

### Step 2.3 — Wire ReadTheDocs

1. Log in to [readthedocs.org](https://readthedocs.org)
2. Click **Import a Project** → find `multidriver-swg`
3. Trigger a build and confirm it succeeds
4. Verify the following render correctly in the built docs:
   - **Mermaid flowchart** on `index.md` (the `copier update` workflow diagram)
   - **Admonition block** on `installation.md` (the `!!! note` block)
   - **Mermaid sequence diagram** on `usage.md`
   - **mkdocstrings API block** on `api.md` — confirm `hello()` function appears with its docstring and type signature

If any of the four fail, diagnose and fix in the template repo before proceeding to Phase 3.

---

### Step 2.4 — Update tutorial walkthrough

This is an interactive walkthrough to confirm you understand the update mechanism before relying on it. Claude will guide you through each step.

**The scenario**: You make a small deliberate improvement to the template — adding a new universal principle to Part I of `conventions.md` — then propagate it to `multidriver-swg` using `copier update`.

**Steps** (Claude will walk you through each one):

1. In the template repo, open `template/.prompts/conventions.md`
2. Add one new sentence to Part I under "Fail-fast": *"Include the offending value in the exception message wherever possible — abstract errors like 'invalid input' are harder to act on than 'expected int, got str for field X'."*
3. Commit and tag the template: `git tag v1.0.1`
4. In `multidriver-swg`, run `copier update --skip-tasks`
5. Observe the diff — the new sentence should appear in your `conventions.md` automatically with no conflict
6. Confirm to Claude that the update applied correctly

**Completion signal**: You describe what happened (what changed, where, how you resolved it) and Claude confirms your understanding is correct before marking this phase done.

---

### Definition of Done — Phase 2

- [ ] `~/dev/multidriver-swg` exists and was generated from the template
- [ ] `.copier-answers.yml` present with correct values
- [ ] `multidriver-swg` repo pushed to GitHub as private
- [ ] ReadTheDocs build succeeds and all four verification points pass (Mermaid, admonition, sequence diagram, mkdocstrings)
- [ ] Update tutorial completed: template improvement propagated to `multidriver-swg` via `copier update`
- [ ] You have described the update outcome and Claude has confirmed your understanding

---

# Phase 3: `copier update` Workflow Tutorial (Reference Guide)

## Goal

Produce a concise written reference you can follow independently in future sessions when you want to propagate template changes to one or more downstream projects. This is a living reference — update it as you encounter edge cases.

This phase is completed **during** Phase 2 (Step 2.4 produces the experience; this phase formalizes it as a reusable document).

---

### The Reference Document

After completing Phase 2 Step 2.4, Claude will write a `docs/planning/reference/copier_update_workflow.md` file **in the `cookiecutter-python-claude` template repo** covering:

#### Part 1: Making a Template Change

```
1. Edit the relevant file(s) in template/
2. Test locally: copier copy --defaults . /tmp/test-update-project
3. Verify the change looks correct in the generated output
4. Commit to the template repo
5. Tag the release: git tag vX.Y.Z && git push --tags
```

#### Part 2: Propagating to a Downstream Project

```
1. cd into the downstream project
2. Run: copier update --skip-tasks
   - Copier fetches the latest template tag
   - Performs 3-way merge (BASE=last template version, OURS=your file, THEIRS=new template)
   - Auto-merges non-conflicting changes
   - Inserts conflict markers for conflicting sections
3. Review all changed files (git diff)
4. Resolve any conflict markers manually
5. Commit: git add . && git commit -m "chore: apply template update vX.Y.Z"
```

#### Part 3: Conflict Scenarios and Resolutions

| Scenario | What you see | What to do |
|----------|-------------|------------|
| Template changed Part I; you only touched Part II | File updated automatically, no markers | Just commit |
| Template changed Part I; you also changed Part I | Conflict markers in the file | Keep the version you want; remove markers; commit |
| Template added a new file you don't have | New file appears in your project | Review and commit |
| Template changed a file you deleted | File reappears | Delete it again; commit |
| Template renamed a variable in `copier.yml` | `.copier-answers.yml` may have stale key | Run `copier update --defaults` and re-answer changed questions |

#### Part 4: Keeping Track of Which Version Each Project Is On

```bash
# Check which template version a project is pinned to:
cat .copier-answers.yml | grep _commit

# Check all downstream projects at once (if organized under ~/dev/):
grep -r "_commit:" ~/dev/*/\.copier-answers.yml 2>/dev/null
```

---

### Definition of Done — Phase 3

- [ ] `docs/planning/reference/copier_update_workflow.md` written in `cookiecutter-python-claude`
- [ ] Document covers all four parts: making a change, propagating it, conflict scenarios, version tracking
- [ ] Document committed and pushed to the template repo

---

# Phase 4: Retroactively Adopt Copier in TRITON-SWMM Toolkit

## Goal

Bring the existing TRITON-SWMM toolkit under the Copier template system so that future `conventions.md` and `.prompts/` improvements can be propagated to it alongside `multidriver-swg` and any other downstream projects.

**Important**: This phase modifies the existing `TRITON-SWMM_toolkit` repo. It does not change any source code — only the Claude/prompts infrastructure and project scaffolding files.

---

### What "Adopting Copier" Means for an Existing Repo

The toolkit was not generated from the template — it predates it. Copier supports this via `copier copy` run into an existing directory. Copier will:
1. Ask all the `copier.yml` questions (pre-filled with defaults)
2. Generate the template files into the existing directory
3. Show conflicts where template files differ from existing files
4. You resolve conflicts, keeping the TRITON-SWMM-specific content and accepting the template's structure where appropriate
5. After accepting, `.copier-answers.yml` is written — from this point forward `copier update` works normally

---

### Step 4.1 — Pre-migration audit

Before running Copier, identify every file the template will touch and note which version (template vs. toolkit-specific) should win each conflict:

| Template file | Toolkit equivalent | Expected resolution |
|--------------|-------------------|---------------------|
| `template/.prompts/conventions.md` | `.prompts/conventions.md` | Keep toolkit's — it has the full TRITON-SWMM Part II. Accept template's Part I/III updates only. |
| `template/.prompts/architecture.md` | `.prompts/architecture.md` | Keep toolkit's entirely — it's fully project-specific. |
| `template/.prompts/implementation_plan.md` | `.prompts/implementation_plan.md` | Accept template's — they should be identical (verbatim copy). |
| `template/.prompts/qaqc_and_commit.md` | `.prompts/qaqc_and_commit.md` | Accept template's — verbatim copy. |
| `template/.prompts/proceed_with_implementation.md` | `.prompts/proceed_with_implementation.md` | Accept template's — verbatim copy. |
| `template/CLAUDE.md` | `CLAUDE.md` | Accept template's — pattern is identical. |
| `template/.claude/settings.local.json` | `.claude/settings.local.json` | Keep toolkit's — has TRITON-SWMM-specific allow list entries. |
| `template/.claude/agents/_example-agent.md` | *(does not exist)* | Accept template's — adds the stub. |
| `template/pyproject.toml` | `pyproject.toml` | Keep toolkit's — has TRITON-SWMM deps, ruff config, pytest config. |
| `template/mkdocs.yml` | *(does not exist — toolkit uses Sphinx)* | Accept template's — adds MkDocs config alongside existing Sphinx. Toolkit can keep both. |
| `template/.readthedocs.yaml` | `.readthedocs.yaml` | Conflict — toolkit uses Sphinx; template uses MkDocs. Decision needed at migration time: switch toolkit to MkDocs, or keep Sphinx and exclude `.readthedocs.yaml` from Copier management. |
| `template/.gitignore` | `.gitignore` | Accept template's (standard Python gitignore) — toolkit's custom additions will need to be re-added. |
| `template/.pre-commit-config.yaml` | `.pre-commit-config.yaml` | Keep toolkit's — has the project-specific `check-claude-docs` hook. |
| `template/CONTRIBUTING.md` | `CONTRIBUTING.md` | Merge — toolkit has TRITON-SWMM-specific content; template has generic structure. Keep toolkit's. |

---

### Step 4.2 — Run Copier into the existing repo

```bash
cd ~/dev/TRITON-SWMM_toolkit
copier copy ~/dev/cookiecutter-python-claude . --overwrite
```

The `--overwrite` flag tells Copier to write files even when they already exist (conflicts will still be shown). Work through each conflict using the resolution guide from Step 4.1.

---

### Step 4.3 — Verify and commit

After resolving all conflicts:

1. Run `ruff check .` — confirm no new linting errors introduced
2. Run `pytest tests/test_PC_01_singlesim.py -v` — confirm single-sim smoke test still passes
3. Confirm `.copier-answers.yml` is present and correct
4. Commit: `chore: adopt Copier template system (cookiecutter-python-claude v1.0.0)`

---

### Open Decision for Phase 4

**`.readthedocs.yaml` conflict** — two options:

| Option | Description | Recommendation |
|--------|-------------|----------------|
| Switch toolkit to MkDocs | Replace Sphinx with MkDocs + Material; update docs content from RST to Markdown | Consistent with template; better long-term; significant docs migration effort |
| Keep Sphinx; exclude from Copier | Add `.readthedocs.yaml` to `copier.yml` `_exclude` list for this project | Avoids docs migration now; toolkit docs stay in RST; Copier won't touch this file |

**Recommended**: Exclude `.readthedocs.yaml` from Copier management for now (add to the project's `.copierignore`). Migrate to MkDocs as a separate task when the toolkit docs need a refresh. This keeps Phase 4 narrowly scoped.

---

### Definition of Done — Phase 4

- [ ] `copier copy` completed into `TRITON-SWMM_toolkit` without errors
- [ ] All file conflicts resolved using the pre-migration audit table
- [ ] `.copier-answers.yml` present in toolkit root
- [ ] `.readthedocs.yaml` conflict resolved (either excluded or migrated)
- [ ] `ruff check .` passes
- [ ] PC_01 smoke test passes
- [ ] Changes committed with chore commit message referencing template version
- [ ] `copier update --skip-tasks` runs successfully (no errors, even if no changes to apply yet)
- [ ] Toolkit appears alongside `multidriver-swg` when running the version-check grep from Phase 3

#user: I would also like to push a change to a doc, like conventions.md, to get first hand experience pushing changes from copier to the associated repos