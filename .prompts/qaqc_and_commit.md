# QA, Review, and Commit

A post-implementation review prompt. Work through all four steps before requesting a commit.

## Step 1: Review Planning Documents

Review any related planning documents and verify that success criteria were met and that the documents reflect the changes made — particularly if deviations from the plan were required.

- Note discrepancies, and include exact quotes and/or code chunks along with rationale for the differences
- If a master plan document is associated with this implementation, verify its freshness as it relates to changes made here
- Summarize issues in a table

## Step 2: Check Against Conventions

Read `.prompts/conventions.md` and consider whether the changes align with the project conventions.

- Consider implications for the entirety of each script you touched — this is an opportunity to ensure the entire script, not just recently implemented work, abides by the project conventions
- If there are discrepancies, explain them with direct quotes from `.prompts/conventions.md`
- Include a numerically indexed table covering each item across all sections of conventions.md, with columns for: whether it was honored, exceptions, issues, and recommendations

## Step 2b: Check Architecture Doc Freshness

Review `.prompts/architecture.md` and verify that any architecture, module, or pattern described there still matches the current codebase as changed by this implementation.

- Check: do module names, class names, file paths, and config field descriptions still match the code?
- Check: are any "Gotchas" now resolved or newly relevant?
- Check: should any new runner scripts, modules, or execution modes be documented?
- If updates are needed, include them in the commit (or as a separate chore commit if substantial)

## Step 3: Report Findings

Report back with findings organized into four sections:

**`implementation summary`** — what was done

**`implementation vs plan`** — how the implementation compares to the planning document(s)

**`implementation vs conventions`** — how the implementation compares to `.prompts/conventions.md`

**`input needed`** — a bulleted list of all decisions or questions needing input, with options and recommendations for each

**`implementation plan status`** — Describe the completion status of the primary planning document that was implemented. Recommend if the planning document needs to be refreshed and/or is ready to be closed (see `.prompts/conventions.md` for guidance on planning document organization). Plans should accurately reflect the work done prior to being marked for completion since other agents may need to reference that plan as a dependency.

Note: scope creep may occur as tangential issues are discovered during implementation. Consider whether multiple commits are recommended. If so, report under **`proposed commits`** with subheaders that would become the top commit message, followed by a table of files included in each commit and a summary of the changes in each file.

## Step 4: Coordinate and Commit

If there are no discrepancies, open decisions, or judgment calls that should be reviewed, request to commit changes.

Upon explicit approval, proceed with the commit using the format below. If there are open items, coordinate with the developer to reach resolution first. Once resolved, propose to commit and proceed only with explicit approval.

If there are unrelated changes from the developer, offer to commit them too. Recommend whether to include them in the main commit or as a separate commit, and explain your reasoning.

**Never commit without explicit permission from the developer.**

---

## Commit Format

```
<type>: <short summary> (<reference to plan/phase if applicable>)

<detailed description of what changed, 2-4 paragraphs>

<specific technical details or notable file changes>

<test results summary if applicable>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Commit types**: `feat` · `refactor` · `fix` · `test` · `docs` · `chore`

**Guidelines**:
- Reference the planning document and phase when applicable
- Include file statistics if significant (e.g., "305 lines deleted, 13 inserted")
- State smoke test results when tests were run
- Use HEREDOC format for multi-line messages
- Review `git status` before staging — ensure only intended changes are included
