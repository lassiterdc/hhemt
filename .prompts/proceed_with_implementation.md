# Proceed With Implementation

> **This is a prompt.** Read these instructions carefully and follow them in order. Do not proceed to implementation until each gate condition is met.

## If this is the first call this session to this prompt document, start here

Expected call pattern: `@.prompts/proceed_with_implementation.md docs/planning/path/to/planningdoc.md`

First, ask the user whether to delegate the task to an Opus subagent via the Task tool with `subagent_type: "general-purpose"` and `model: "opus"` or handle it as Sonnet in the current conversation. If Opus is chosen, pass the planning doc path and all relevant context in the subagent prompt and do not perform this work yourself. **Do not proceed with this prompt until you get confirmation from the user.**

The requested work (whether by subagent or current conversation) should:

1. Read the planning doc passed as context. Make note of significant uncertainties. If this is a phase doc inside a multi-phase subdirectory, also read `master.md` in the same directory for cross-phase context and dependencies. Verify this is the correct next phase to implement (check the phase status table in `master.md`).
2. Make sure all decisions requiring developer input have been made.
3. Evaluate the freshness of the plan. Evaluation should include:
   - Review related scripts that will be modified, and any scripts that depend on or are depended on by those scripts
   - If the implementation relies on existing test functions, review them
   - If part of a multi-phase plan, review the preceding phase doc (check the `implemented/` subdirectory within the same planning directory)
4. Read `.prompts/conventions.md` and check for discrepancies between the plan and philosophy. If discrepancies exist, explain them with direct quotes from `.prompts/conventions.md` and provide recommendations on how to handle each.
5. If the plan is associated with a master plan, review the master planning document. If there are discrepancies, report them with direct quotes from both documents and judge which is more likely to be stale.
6. Return a **preflight report** with findings from steps 1–5. The report must include:
   - Model and/or subagents used
   - Decisions needing input — with all relevant context, options, and a recommendation
   - Uncertainties requiring clarification — posed as questions with all relevant context
   - Other changes planned after review that are not already in the planning document
   - Findings from steps 1–5

Once the report is complete, present it to the developer and coordinate until given the go-ahead to proceed with implementation. **Do not proceed without explicit approval.**

---

## If this is a subsequent call to this prompt

Expected call pattern: `@.prompts/proceed_with_implementation.md`

Handle this phase in the current conversation (Sonnet is appropriate here).

This is a final check. Do not proceed until all steps below are complete.

1. If you are unclear about a decision or need clarification, raise questions now. Otherwise proceed to the next step.
2. If decisions and clarifications have made planning documents stale, update them.
3. Report to the developer:
   - Summarize updates to planning documents. List and explain each change with relevant snippets from the revised docs.
   - Consider whether the edits uncovered additional decisions or uncertainties. If so, present them.
   - Make a recommendation whether or not to proceed with implementation.
4. Upon explicit approval from the developer, proceed with implementation.
5. Implement `@.prompts/qaqc_and_commit.md`
