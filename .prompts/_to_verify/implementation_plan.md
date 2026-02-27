# Implementation Plan (Consolidated Prompt)

Design a complete implementation plan **before coding**. The goal is to reduce ambiguity, prevent mid-stream rework, and make execution straightforward.

## When to Use

This prompt will be used when called by the user. The expectation is that the user will pass this file as context followed by a filename for the planning document before explaining the task. If a filename is not explicitely provided, recommend one to the user before writing the plan to a file.

## Purpose

Produce a clear, repo-aware plan that:
- captures requirements and assumptions,
- identifies all impacted files and dependencies,
- includes validation and documentation updates,
- surfaces user decisions early,
- and is ready to execute with minimal back-and-forth.

## Required Discovery (Before Planning)

Review enough context to produce a grounded plan:

1. `CLAUDE.md` for project rules and development philosophy
2. Existing implementation patterns in `src/` and related tests in `tests/`
3. Any directly referenced files from the user request

## Planning Workflow

1. Restate the task and success criteria in your own words
2. Identify constraints, dependencies, and likely edge cases
3. Propose approach options (briefly), then select one with rationale
4. Build a phased implementation sequence (prep → code changes → validation)
5. List open decisions that require user confirmation before implementation

## Required Output Format

Use **exactly** these headings, in this order:

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
   - Include smoke tests (PC_01, PC_02, PC_04, PC_05) when changes are significant (note these smoke tests are only relevant for code that effects local runs - anything that will specifically effect slurm runs needs to be tested in coordination with the user)

7. `## Documentation and Tracker Updates`
   - Docs that may need updates (e.g., `CLAUDE.md`, agent docs, planning trackers)
   - Conditions that trigger those updates

8. `## Decisions Needed from User`
   - Questions that block implementation
   - If proceeding with assumptions, label each assumption with risk level (low/medium/high)

9. `## Definition of Done`
   - Concrete completion checklist

## Project Guardrails

- Do **not** add backward-compatibility shims unless explicitly requested
- Update all import sites immediately when moving/renaming modules
- Prefer consistency with existing repository patterns over introducing novel structure UNLESS novel structure is a significant improvement and is more canonical to Python programming, software development, or any related packages and libraries. 
- Keep plan actionable: avoid vague steps like “refactor as needed”

## Plan Quality Self-Check (Required)

After drafting the plan, perform and report this check:

1. **Header/body alignment check**
   - Do all section headers accurately match the content in their section body?
   - If not, rename headers or revise content until aligned.

2. **Section necessity check**
   - Are all sections necessary?
   - If any section can be removed without losing important context or actionable guidance, remove it.

Report a short “Self-Check Results” summary at the end.

## Approval Gate

Return the plan only.

Do **not** implement code changes yet. Wait for user approval before editing files or running destructive commands.

## Write the plan to a markdown file

By default, write the implementation plan to:

`docs/planning/active/`

Use a suitable subdirectory when applicable:
- `docs/planning/active/features/`
- `docs/planning/active/refactors/`
- `docs/planning/active/bugs/`

Use a descriptive snake_case filename ending in `.md`.

If the user does **not** request a different destination, this default is required.

## Do NOT proceed with implementation until the plan has been verified by the developer and explicit permission to proceed with implementation is granted. 