---
prompt_doc_type: sidecar_scratch
sidecar_kind: wrapup_handoff_atomic
main_scratch: /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-15_1303_durable-profile-routine-tests.md
last_sync_hashes:
  '# Follow-up Ideas': 1e0f3648a29f3966e756b16ddb618c58b95591393fc02c13e55a662c3d5576c6
harness: claude-code
plan_name: durable profile routine tests
plan_completion_commit: d6fb4c9a9b413f3307646deb56f1c990ca20cc4e
worktree_path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests
created: '2026-05-16T17:07:31'
smoke_test: false
---
- [x] in progress
## Harness Discipline — Claude Code

- Treat the first post-compaction message as a rehydration trigger, not a
  fresh-session invocation. Do NOT run `/proceed-with-implementation`
  preflight. Do NOT re-enter the phase loop. Plan Completion (phase-doc
  moves to `implemented/`, master to `completed/`, planning-table
  regeneration, commit) already ran in the pre-compaction session.
- Your task is `/wrapup-session` in full, per `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`.
- Do NOT read the main session scratch doc until after wrapup-session Step 6
  (`--inject-into-scratch`) has run. Immediately after that injection — and
  before populating any wrapup sections — read the just-injected wrapup
  section AND the `# scratch doc communication protocols` section of main
  scratch in full. Only then proceed to Step 7 of the skill.
- Sync integrity is verified via
  `scripts/generate/write_compaction_sidecar.py --mode verify`. Do not
  compute hashes yourself.

## Wrapup Entry Point

- Kind: wrapup_handoff_atomic
- Plan name: durable profile routine tests
- Plan completion commit: `d6fb4c9a9b413f3307646deb56f1c990ca20cc4e`
- Worktree: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests`
- Worktree branch: `worktree-toolkit_05-15_1303_durable-profile-routine-tests`
- Task: `/wrapup-session` in full

## Post-Compaction Wrapup — Preamble

You are resuming an atomic plan implementation after an in-session /compact
that was fired at the end of the plan — after the final implementation commit
landed and after plan closeout (plan closeout protocol) ran in the
pre-compaction session. Your single task is `/wrapup-session` in full. Do NOT
re-enter the implementation loop. Do NOT re-run plan closeout. Do NOT issue an
implementation `/commit` — the implementation commit and the plan-closeout
commit are both already in git history.

The first message you receive after compaction looks structurally like a fresh
session invocation, and RLHF training creates a pull to treat it as one —
running preflight, re-reading the full session scratch doc, re-dispatching
specialist reviews. Recognize this pull. Your actual task is to rehydrate from
the wrapup-handoff sidecar, verify sync integrity via the script, run the
First Action GO gate, and then invoke `/wrapup-session` directly.

### Mandatory reads (in order, each in full)

1. Wrapup-handoff sidecar:
   `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests/sidecars/sidecar_wrapup_2026-05-16_1707.md`

2. Wrapup-session skill body (the skill you are about to execute):
   [wrapup session SKILL](../../instructions/skills/wrapup-session/wrapup%20session%20SKILL.md) #inject_path

%% inject-path-start: wrapup-session-skill %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`
    - Session wrap-up checklist — audits for uncommitted changes, outstanding todos, unresolved decisions, and next steps before exiting
%% inject-path-end: wrapup-session-skill %%

### Wrapup entry point

- **Plan name**: `durable profile routine tests`
- **Plan completion commit**: `d6fb4c9a9b413f3307646deb56f1c990ca20cc4e`
- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests`
- **Worktree branch**: `worktree-toolkit_05-15_1303_durable-profile-routine-tests`
- **Task**: run `/wrapup-session` in full, per the skill body above.

Plan closeout already moved the plan file to `completed/`, regenerated planning
tables, and committed those changes. Both the implementation commit and the
plan-closeout commit are in git history; `d6fb4c9a9b413f3307646deb56f1c990ca20cc4e` points at
the plan-closeout commit (HEAD at the time this sidecar was written). If you
find yourself tempted to re-run `$AGENTIC_WORKSPACE/scripts/generate/complete_plan.py`, touch the planning tree,
or invoke `/commit` for the implementation, stop — that work is done.

Do NOT read the main session scratch doc yet. Main scratch is intentionally
out of scope for the First Action gate. You will read it at wrapup-session
Step 6 — see the "Step 6 injection-then-reload contract" below.

### Step 6 injection-then-reload contract (read this carefully)

When `/wrapup-session` reaches Step 6 (Write Report to Session Scratch Doc),
the skill runs `create_from_template --compose wrapup-session
--inject-into-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-15_1303_durable-profile-routine-tests.md` to append the wrapup template
body into main scratch under a new `# Wrapup Session` heading. Once that
injection command returns success and **before you begin populating any
section or proceeding to Step 7**:

1. **Read the just-injected wrapup section in main scratch in full.** This is
   the wrapup-session template that Steps 0-5 findings will be written into.
2. **Read the `# scratch doc communication protocols` section of main scratch
   in full.** This loads the agent reply conventions (callout color semantics,
   agreement-checkbox format, datetime stamps, quoted-user-comment format) and
   the user comment resolution protocol into your context. These are the
   conventions for every user-facing callout you will write during wrapup
   Step 7 (follow-up review) and Step 8 (Verdict).

Only after BOTH reads are complete, continue with wrapup-session by populating
the injected sections from your Step 0-5 findings and proceeding to Step 7.
This reload step is not optional — the communication protocols are the
delivery contract between you and the user for the remainder of the session.

### Worktree context

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests`
- **Worktree branch**: `worktree-toolkit_05-15_1303_durable-profile-routine-tests`
- **Main scratch path**: `/home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-15_1303_durable-profile-routine-tests.md`

All wrapup-session writes go to the worktree (or to main scratch via the
Step 6 injection). Worktree closeout runs as part of `/wrapup-session` Step
7b — push the branch, run merge_review, merge to main, run `$AGENTIC_WORKSPACE/setup.sh` from
the main-tree root, commit regenerated files, and remove the worktree.

### Downstream-skill fidelity framing

Any skill invoked from within this wrapup session (for example, `/review-followup-ideas`, `/plan-implementation`, or `/capture-idea` for follow-up items) runs against approximately fresh context — compaction just freed it. Treat those skills as first-invocation fresh-session runs for fidelity purposes. Do not cite the pre-compact session length, the conversation summary size, or "how much has already happened today" as a reason to compress a skill's gates. If the developer surfaces a concrete context-budget constraint, respond to it; otherwise perform at full fidelity.

## Post-Compaction Wrapup — Halt Triggers

### Halt triggers

1. **Tier 1 wrapup gate**: any of the wrapup-session hard gates fail and
   cannot be remediated within the skill — specifically Step 1 (Unresolved
   User Comments) cannot be cleared, Step 7 (Follow-up Idea Review) cannot
   reach terminal disposition on every idea, or the Verdict step reports
   unresolved items. Halt and surface the specific gate to the user.
2. **Worktree closeout failure**: any of the gated steps in worktree closeout
   (clean-tree check, merge_review, merge conflict resolution, `$AGENTIC_WORKSPACE/setup.sh`
   run, worktree removal) fails. Do not attempt destructive recovery — halt
   and surface the failure mode.
3. **Sync-back integrity failure**: `write_compaction_sidecar.py --mode verify`
   exits with code 2 (hash mismatch) or code 3 (worktree dirty). Never
   overwrite main on a mismatch.
4. **Sidecar consistency failure** — see 4-condition checklist below.
5. **Explicit user interrupt**.

#### Halt trigger #4 — sidecar consistency failure

Trigger if ANY of the following four conditions fail when you read the
wrapup-handoff sidecar:

1. The sidecar frontmatter's `sidecar_kind` field is exactly
   `wrapup_handoff_atomic`.
2. The sidecar body contains a `## Wrapup Entry Point` section.
3. The sidecar's `## Worktree Status` section's branch line matches the
   sidecar frontmatter's `worktree_path` field's derivable branch name
   (`worktree-{slug}`).
4. The sidecar's `last_sync_hashes:` frontmatter contains exactly one entry:
   `# Follow-up Ideas`. (Atomic wrapup does not carry `### Phase Audit
   History` — atomic plans have no phase loop.)

Write each condition's result explicitly as
`Condition N: PASS|FAIL — [observed vs. expected]` before producing the GO
signal. Any FAIL halts.

### Post-halt response protocol

When any halt trigger fires:

1. **Write the halt state** to the sidecar's `# Follow-up Ideas` section with:
   - The halt trigger number that fired
   - The specific condition or evidence that triggered it
   - The wrapup step at which the halt was detected (if applicable)

2. **Do NOT run `--mode sync-back`** on a wrapup-handoff sidecar. Sync-back
   is a phase-loop operation; the wrapup sidecar is not an in-flight sidecar
   and sync-back will refuse or misbehave. If the halt state needs to reach
   main, update the sidecar's `# Follow-up Ideas` section in place — the
   wrapup sidecar's `last_sync_hashes` already reflect main, and follow-up
   ideas written to the sidecar will be visible in the next main-scratch
   read during wrapup-session Step 7.

3. **Wait for user direction**. Do not attempt to diagnose, work around, or
   retry the halt condition on your own. Produce a single-message halt
   report: "Halted: trigger #N — {one-line description}. Awaiting direction."

4. **Route on user direction**: either the user resolves the blocker and
   authorizes continuation with a specific next step, or the user instructs
   a fallback path (manual worktree closeout, deferred wrapup, etc.).

## Post-Compaction Wrapup — First Action

### Part 1 — Read-coverage report (presumption-STOP gate)

**Presumption**: you have not read any files. The default state is "reads not
yet performed." Proceeding to Part 2 requires affirmative evidence of each
mandatory read. RLHF completion-bias creates a pull to produce a
plausible-looking table without having actually done the reads — recognize
this pull and resist it.

For each mandatory read (the wrapup sidecar and the wrapup-session skill
body), run `wc -l {absolute-path}` in a Bash tool call and paste the literal
stdout output into the assertion block. Prose descriptions of "I read this
file" are not acceptable — only `wc -l` stdout counts.

After each `wc -l` receipt, write a one-line self-assertion of the form
`Read coverage: all N lines read, understanding consistent with length.`
substituting the observed N. Both the `wc -l` output and the self-assertion
are required. If you cannot truthfully write the self-assertion — because
you did not actually read the file in full, or because your understanding is
inconsistent with the observed length — HALT. Do not fabricate the
self-assertion to satisfy the gate.

Main scratch is intentionally NOT in this read-coverage list. It is read
during wrapup-session Step 6 per the injection-then-reload contract in the
preamble — do not read main scratch here.

Format:

```
$ wc -l /home/.../sidecars/sidecar_wrapup_2026-04-15_1500.md
87 /home/.../sidecars/sidecar_wrapup_2026-04-15_1500.md
Read coverage: all 87 lines read, understanding consistent with length.
$ wc -l /home/.../library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md
410 /home/.../library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md
Read coverage: all 410 lines read, understanding consistent with length.
```

If any `wc -l` count does not match your understanding of the file's length —
or if you cannot produce `wc -l` output for a file you claim to have read —
or if you cannot truthfully write the per-file self-assertion — HALT. That
is the presumption-STOP gate firing. Do not fabricate counts or
self-assertions; run the tool and read the files.

### Part 2 — Wrapup-entry assertion

Read the sidecar frontmatter and the sidecar's `## Wrapup Entry Point`
section. Write:

`Wrapup-entry assertion: Sidecar: {sidecar_filename}. sidecar_kind: wrapup_handoff_atomic. Plan name: durable profile routine tests. Plan completion commit: d6fb4c9a9b413f3307646deb56f1c990ca20cc4e. Wrapup entry block present in body: YES|NO.`

If any of these do not match the sidecar's actual state, or if the body does
not contain `## Wrapup Entry Point`, HALT per the post-halt response protocol
and surface the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke:

    scripts/generate/write_compaction_sidecar.py --mode verify \
      --main-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-15_1303_durable-profile-routine-tests.md \
      --worktree /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests

Assert the exit code:

- Exit 0: write `Sync integrity (from --mode verify exit code): OK. Worktree: clean.`
- Exit 2: script detected a hash mismatch. HALT per post-halt response
  protocol. Do NOT attempt recovery.
- Exit 3: script detected uncommitted changes in worktree. HALT — the
  pre-compaction session did not finish committing cleanly.

### Conjunctive GO gate (required written assertion before /wrapup-session)

Produce exactly the following assertion text verbatim, substituting GO or
STOP for each Part:

```
Part 1: GO|STOP
Part 2: GO|STOP
Part 3: GO|STOP
Conjunction: (Part 1 = GO) AND (Part 2 = GO) AND (Part 3 = GO) = GO|STOP
Proceeding to /wrapup-session: YES|NO
```

The `Proceeding to /wrapup-session: YES` line is the single gate event. Any
STOP in any Part means NO. Writing the conjunction is required — it is the
forcing function against RLHF task-completion pressure that would otherwise
push the agent to transition silently.

Then invoke `/wrapup-session` directly and follow it in full. Do not
summarize what you learned — the sidecar is the record. Do not ask the user
for confirmation before starting the skill — the GO gate IS the
confirmation.

Because `sidecar_kind: wrapup_handoff_atomic`, `/wrapup-session` Step 8b
(implementation commit) is SKIPPED — the atomic implementation commit and
the plan-closeout commit both ran in the pre-compaction session. Step 8a
(Verdict, final scratch_gate check) runs as usual.

## Follow-up Ideas



%%
"#" + "followup" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.

Specialist findings routed here by /invoke-specialist Step 6 appear under:
  #### Specialist-identified follow-up items
  ##### {specialist-name}
These are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.

Agents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.
%%

## Worktree Status

- branch: worktree-toolkit_05-15_1303_durable-profile-routine-tests
- path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-15_1303_durable-profile-routine-tests

