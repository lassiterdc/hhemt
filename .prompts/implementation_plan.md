# Implementation Plan Prompt

Design a complete implementation plan **before coding**. The goal is to reduce ambiguity, prevent mid-stream rework, and make execution straightforward.

## When to Use

This prompt will be used when called by the user. The expectation is that the user will pass this file as context followed by a filename for the planning document before explaining the task. If a filename is not explicitly provided, recommend one to the user before writing the plan to a file.

## Purpose

Produce a clear, repo-aware plan that:
- captures requirements and assumptions,
- identifies all impacted files and dependencies,
- includes validation and documentation updates,
- surfaces user decisions early,
- and is ready to execute with minimal back-and-forth.

## Required Discovery (Before Planning)

Review enough context to produce a grounded plan:

1. `.prompts/conventions.md` for project rules, development philosophy, and terminology
2. `.prompts/architecture.md` for architecture, key modules, and runner script patterns
3. Existing implementation patterns in `src/` and related tests in `tests/`
4. Any directly referenced files from the user request

## Plan Output Location (Default)

By default, write the implementation plan to the appropriate subdirectory:
- `docs/planning/features/`
- `docs/planning/refactors/`
- `docs/planning/bugs/`

When complete, move the plan to the `completed/` subdirectory within the same type directory (e.g., `docs/planning/bugs/completed/`).

Use a `YYYY-MM-DD_descriptive_snake_case_name.md` filename, where the date is today's date.

If the user does **not** request a different destination, this default is required.

## Planning Workflow

1. Restate the task and success criteria in your own words
2. Identify constraints, dependencies, and likely edge cases
3. Propose approach options (briefly), then select one with rationale
4. Build a phased implementation sequence (prep → code changes → validation)
5. List open decisions that require user confirmation before implementation
6. **Assess plan scope**: Determine whether this plan is atomic (single commit or tightly related set of commits implementable in one session) or multi-phase (phases implemented and committed separately, possibly across sessions). Recommend the appropriate structure:
   - **Atomic**: single planning doc, no subdirectory needed
   - **Multi-phase**: recommend the subdirectory structure described in the "Multi-phase plans" section of `.prompts/conventions.md` but do not create it yet — wait for scope agreement

## Required Output Format

Use **exactly** these headings, in this order:

**Header** with datetime of writing and datetime of last edit with a short summary of the edit.

1. `## Task Understanding`
   - Requirements
   - Assumptions
   - Success criteria

2. `## Evidence from Codebase`
   - Bullet list of files inspected and key findings

3. `## Implementation Strategy`
   - Chosen approach
   - Alternatives considered (1-3 bullets)
   - Trade-offs

4. `## File-by-File Change Plan`
   - For each file: purpose of change + expected impact
   - Include new files, modified files, and any deletions
   - Call out import sites that must be updated

5. `## Risks and Edge Cases`
   - Technical risks and mitigations
   - Edge cases to explicitly validate

6. `## Validation Plan`
   - Specific commands/tests to run
   - For significant changes affecting local execution, include smoke tests (PC_01, PC_02, PC_04, PC_05)
   - Note: smoke tests are only relevant for local runs — SLURM-specific changes require HPC testing coordinated with the developer

7. `## Documentation and Tracker Updates`
   - Docs that may need updates (e.g., `.prompts/conventions.md`, `.prompts/architecture.md`, agent docs, planning trackers)
   - Conditions that trigger those updates

8. `## Decisions Needed from User`
   - Questions that block implementation
   - If proceeding with assumptions, label each assumption with risk level (low/medium/high)

9. `## Definition of Done`
   - Concrete completion checklist

## Project Guardrails

- Do **not** add backward-compatibility shims unless explicitly requested
- Update all import sites immediately when moving/renaming modules
- Prefer consistency with existing repository patterns over introducing novel structure UNLESS novel structure is a significant improvement and is more canonical to Python or the relevant libraries
- Keep plan actionable: avoid vague steps like "refactor as needed"
- Do not assume the developer is an expert in every library or pattern used — explain non-obvious choices and trade-offs clearly

## `#user:` Comments Are Blocking

In planning documents, comments prefixed with `#user:` are developer feedback that must ALL be addressed before any implementation can take place. Remove each comment only after written confirmation from the developer. Implications for the entire planning document should be considered when addressing these comments.

## Plan Quality Self-Check (Required)

After drafting the plan, perform and report this check:

1. **Header/body alignment check**
   - Do all section headers accurately match the content in their section body?
   - If not, rename headers or revise content until aligned.

2. **Section necessity check**
   - Are all sections necessary?
   - If any section can be removed without losing important context or actionable guidance, remove it.

3. **Alignment with `.prompts/conventions.md`**
   - Does the approach align with the design philosophy of this project?
   - Are there adjustments needed to abide by good software development practices?

4. **Task-relevance check**
   - During plan refinement, documents often get bloated with information not relevant to implementation
   - Remove any irrelevant or redundant information

Report a short "Self-Check Results" summary at the end.

## Scope Gate (Required Before Closing This Planning Session)

After the developer has agreed to the full plan scope, re-evaluate:

**Is this plan atomic or multi-phase?**

- **Atomic**: single planning doc is correct. No further action needed.
- **Multi-phase** (phases committed separately, or future agents will need to read phases independently): restructure now into the subdirectory layout described in the "Multi-phase plans" section of `.prompts/conventions.md`. Move and rename the file, create phase docs, and confirm the restructure with the developer before closing.

---

## Approval Gate

Return the plan only.

Do **not** implement code changes yet. Wait for user approval before editing files or running destructive commands.
