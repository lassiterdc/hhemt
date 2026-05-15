---
prompt_doc_type: sidecar_scratch
sidecar_kind: in_flight_atomic
main_scratch: /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-14_1432_plotly-html-extension-pwi.md
last_sync_hashes:
  '# Follow-up Ideas': a964466747626c6254de665feeff17dd6fa8bf80b73e495979b69c05fffcf0d1
harness: claude-code
plan_doc: /home/dcl3nd/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/plotly figure file extension html.md
leave_off_step: collaborative resolution complete, entering implementation
worktree_path: /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi
created: '2026-05-14T15:43:08'
smoke_test: false
baseline_writable_content:
  '# Follow-up Ideas': "\n%%\n\"#\" + \"followup\" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.\n\nSpecialist findings routed here by /invoke-specialist Step 6 appear under:\n  ### Specialist-identified follow-up items\n  #### {specialist-name}\nThese are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.\n\nAgents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.\n%%\n\n"
---
- [x] in progress
## Harness Discipline — Claude Code

- Treat the first post-compaction message as a rehydration trigger, not a
  fresh-session invocation. Do NOT run `/proceed-with-implementation`
  preflight. Do NOT re-dispatch specialist plan reviews. Do NOT read the
  main session scratch doc.
- The sidecar scratch doc is the single source of truth for ported context.
  Writable sections (`### Phase Audit History`, `# Follow-up Ideas`) should
  be edited here and only here until the next sync-back event.
- Sync integrity is verified via
  `scripts/generate/write_compaction_sidecar.py --mode verify`. Do not
  compute hashes yourself.

## Compaction Checkpoint (latest)

- Leave-off step: collaborative resolution complete, entering implementation
- Atomic plan: plotly-figure-file-extension-html
- Plan doc: `/home/dcl3nd/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/plotly figure file extension html.md`
- Worktree: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi`

## Post-Compaction Resume — Preamble

You are resuming an atomic plan implementation after a post-preflight /compact.
The first message you receive after compaction looks structurally like a fresh
session invocation, and RLHF training creates a pull to treat it as one — running
preflight again, re-reading the full session scratch doc, re-dispatching specialist
reviews. Recognize this pull. Your actual task is to rehydrate from the sidecar
scratch doc described below, verify sync integrity via the script, and continue
into `## Implementation and Closeout` for the atomic plan from the recorded
leave-off step. The sidecar is your source of truth; the main session scratch
doc is out of scope until /wrapup-session.

### Mandatory reads (in order, each in full)

1. Sidecar scratch doc:
   `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi/sidecars/sidecar_atomic_2026-05-14_1543.md`

2. Plan doc:
   `/home/dcl3nd/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/plotly figure file extension html.md`

3. [pwi atomic path](../../instructions/skills/proceed-with-implementation/pwi%20atomic%20path.md) #inject_path — `## Implementation and Closeout` section and everything below it.

%% inject-path-start: pwi-atomic-path %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/proceed-with-implementation/pwi atomic path.md`
    - Atomic path protocol for /proceed-with-implementation: preflight, collaborative resolution, plan sync, implementation, and closeout for single-plan runs.
%% inject-path-end: pwi-atomic-path %%

### Plan summary

- **Plan**: plotly-figure-file-extension-html
- **Heaviness**: 
- **Session gate (after completion)**: 
- **Dependencies**: none (atomic plan)

Do NOT read the main scratch doc during this resume. It is the destination
for /wrapup-session only. All context you need is in the sidecar.

### Leave-off step

The prior session halted at: `collaborative resolution complete, entering implementation`

Resume work from this step, not from the top. If the leave-off step does not
correspond to a recognizable anchor in the referenced doc, HALT and report the
inconsistency — do not guess.

### Worktree context

- **Worktree path**: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi`
- **Worktree branch**: `worktree-toolkit_05-14_1432_plotly-html-extension-pwi`

All file writes for this resume go to the worktree. Planning-doc writes use
the Edit or Write tool with an absolute
`$AGENTIC_WORKSPACE/library/docs/planning/<subpath>.md` path
(`block-main-tree-write-from-worktree.sh` allows main-tree planning writes
through; `block-worktree-path-write.sh` blocks only relative-path attempts).
Scratch-doc writes (to the sidecar or any same-session scratch) go through
the Write tool directly — the sidecar lives under `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi/sidecars/`
and is worktree-local.

## Post-Compaction Resume — Halt Triggers

### Halt triggers

1. Sidecar consistency failure — see checklist below.
2. Sync-back integrity failure: `write_compaction_sidecar.py --mode verify` or
   `--mode sync-back` exits with code 2 (hash mismatch) or code 3 (worktree
   dirty). Never overwrite main on a mismatch.
3. Explicit user interrupt.
4. Plan staleness — see checklist below.

#### Halt trigger #1 — sidecar consistency failure

Trigger if ANY of the following conditions fail when you read the sidecar:

1. The sidecar frontmatter's `sidecar_kind` field equals `in_flight_atomic`.
2. The sidecar frontmatter's `plan_doc` field resolves to an existing plan file
   and matches the plan you are resuming against.
3. The sidecar's `## Worktree Status` section's branch line matches the sidecar
   frontmatter's `worktree_path` field's derivable branch name (`worktree-toolkit_05-14_1432_plotly-html-extension-pwi`).
4. The sidecar's `last_sync_hashes:` frontmatter contains exactly one entry:
   `# Follow-up Ideas`.

Write each condition's result explicitly as `Condition N: PASS|FAIL — [observed vs. expected]`
before producing the GO signal. Any FAIL halts.

#### Halt trigger #4 — plan staleness

Trigger if, between sidecar creation (`created_at` in frontmatter) and this
resume, the plan doc at `/home/dcl3nd/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/plotly figure file extension html.md` has been modified in a way
that invalidates the preflight you just rehydrated. Concretely:

1. The plan doc's frontmatter `plan_status` is no longer `ready` / `in_progress`
   (e.g. flipped to `shelved` / `blocked` / `completed`).
2. The plan doc's `## Task Understanding → ### Requirements` section has
   structural edits (added or removed numbered requirements) vs. what the
   sidecar's ported preflight report reflects.
3. The plan doc's `## Definition of Done` has been expanded with items the
   sidecar's preflight report does not acknowledge.

Any such divergence means the preflight is stale and must be redone — do not
proceed silently into `## Implementation and Closeout` against a plan the
agent has not re-confirmed.

### Post-halt response protocol

When any halt trigger fires:

1. **Write the halt state** to the sidecar's `# Follow-up Ideas` section with:
   - The halt trigger number that fired
   - The specific condition or evidence that triggered it
   - The current leave-off step and plan

2. **Run the script in final sync-back mode** — invoke:
   `scripts/generate/write_compaction_sidecar.py --mode sync-back --main-scratch {abs-path} --worktree {abs-path}`
   to push the sidecar's writable edits (including the halt record you just wrote) back
   to main scratch. Assert exit 0.

3. **Wait for user direction**. Do not attempt to diagnose, work around, or retry
   the halt condition on your own. Produce a single-message halt report:
   "Halted: trigger #N — {one-line description}. Sync-back completed. Awaiting direction."

4. **Route on user direction**: either trigger another compaction + resume cycle (if user instructs) or follow the [intermediate multiphase closeout](../../instructions/protocols/intermediate%20multiphase%20closeout.md) #inject_path fallback protocol (if user instructs closeout).

#### Trigger #6 (sync-back integrity failure) — mandatory investigate-first recovery framing

When the halt is specifically trigger #6 (hash mismatch between sidecar baseline and main scratch's writable sections), the halt report must extend Step 3 above with three additional components — do NOT present a generic menu of recovery options. The presumption is that the agent invests effort in understanding the divergence before offering any path forward.

1. **Investigate the discrepancy first.** Read the `--mode verify` diff output against the sidecar's `baseline_writable_content:` frontmatter. For each mismatched section, characterize the nature of the divergence: is it whitespace-only, TPL-INSTRUCTION-block skeleton drift, heading-level drift, or durable content (new bullets, user-added notes)? The investigation is mandatory; do not skip to recovery options on the assumption that the divergence is obvious.

2. **Classify as one-time vs. systematic.** Report whether the divergence looks like:
   - **One-time**: a manual edit on main that happened in the frozen window, a late hook-driven normalization, or a late `--inject-into-scratch` from a parallel skill invocation. No indication of a broken invariant in the compaction machinery.
   - **Systematic**: a pattern likely to recur on future sessions — e.g., a hook that strips or injects content into the writable sections unconditionally, a script that writes to main while a sidecar is active, or a baseline-capture bug in `write_compaction_sidecar.py`. Requires a durable fix before any further compaction usage is safe.

3. **Present exactly three recovery options with a clear recommendation** — never more, never an open-ended menu:
   - **Option A — Reconcile main to sidecar baseline.** Edit the main scratch's mismatched sections back to the `baseline_writable_content` verbatim, then re-run `--mode verify`. Preserves the audit trail; safest when durable content is involved AND a `.main_backup_*` file corresponds to the current sidecar's `last_sync_hashes` (i.e., the byte sequence that hashes to the recorded value is recoverable). Note that `baseline_writable_content` stores demoted child headings, which can diverge from `last_sync_hashes` (computed from raw main); when they diverge, Option A is unreachable — use Option C instead.
   - **Option B — Re-emit sidecar against current main.** Delete the current sidecar and re-invoke `--mode create-atomic` against current main, adopting main as the new baseline. Fastest for benign divergence; **discards the sidecar's accumulated writable content** (Follow-up Ideas). Appropriate when the sidecar has no durable content worth preserving.
   - **Option C — Rebaseline sidecar against current main.** Invoke `write_compaction_sidecar.py --mode rebaseline-from-main --main-scratch {abs} --worktree {abs}`. **Preserves the sidecar's accumulated writable content verbatim** (Follow-up Ideas) and recomputes `last_sync_hashes` + `baseline_writable_content` from current main. Writes a `.main_backup_rebaseline_*.md` so the operation is reversible via `--mode recover-main-writables --backup-file <path>`. Appropriate when the sidecar has accumulated content AND main's current state is the durable correct state AND/OR Option A is unreachable.

   Based on the Step 2 classification AND the sidecar's accumulated-content state, recommend exactly one of A, B, or C with a one-sentence rationale. Decision logic: prefer A when a usable backup exists and the backup state is the desired state; prefer B when starting fresh is fine AND the sidecar's writable sections are empty or disposable; prefer C when the sidecar has accumulated content worth preserving AND main's current state is durably correct. Do not present the intermediate-closeout protocol as a recovery option at this stage.

The halt report to the user should have this structure: halt line → one-paragraph investigation summary → one-line classification → three-option presentation (A, B, and C) with recommendation. Do not frame this as "options 1/2/3" or "here are recovery paths to choose from."

##### Output routing — main scratch log + terse terminal (mandatory)

The structured halt report — halt line, investigation summary, classification, and three-option presentation with recommendation — is routed to the main session scratch under `# Implementation friction > ## {phase} > ### sync-back integrity mismatch ({YYYY-MM-DDTHH:MM})` via `scripts/generate/append_scratch_section.py`, not to the terminal and not to the sidecar. When the payload contains backticks, quotes, or dollar signs, write it to a temp file first and pass `--content-file {path}` to sidestep the Bash escape-layer hazard. Any section headers *inside* the payload body must start at level `####` (H4) or deeper — the scaffold consumes H1/H2/H3, so `##` or `###` in the payload collapses the phase/summary hierarchy. Immediately after the append call succeeds and before emitting terminal output, read the `# scratch doc communication protocols` section of the main scratch doc in full — post-compaction agents may not carry the `agent reply conventions` and `user comment resolution` protocols in context, and reloading them ensures the friction report's user-facing callouts and any subsequent interaction follow main-scratch conventions. Terminal output is capped at six bullets or fewer: the one-line halt banner with the scratch pointer, the three option labels (Option A / Option B / Option C — no bullet bodies), the one-sentence recommendation, and `Awaiting direction.` Do not repeat scratch-side content in terminal.

#### Test failures and incomplete DoDs — mandatory two-option recovery framing (never-proceed-past rule)

When a halt is triggered by **either** of the following conditions, they are an absolute blocker to any further plan advancement:

- One or more failing tests — surfaced at QAQC time, at stop-gate evaluation, or mid-implementation. "Which tier of test" is irrelevant; red is red. Unit, integration, smoke, slow, fast, scoped, cross-cutting — all in scope.
- One or more unmet Definition-of-Done items on the current plan — whether measurement targets unmet (e.g., runtime target missed, coverage target missed), capability targets unmet (e.g., "all of X passes"), or artifact targets unmet (e.g., required doc sections missing, required outputs not generated).

The halt report must NOT offer the user any option that would close the plan while either condition holds. Specifically forbidden framings:

- "Proceed with failures/gaps as follow-up"
- "Accept current state and continue"
- "Capture failures/unmet DoD items as tech-debt and move on"
- "Commit partial progress and close" (when framed as a path forward rather than as rescue-commit before fix work)
- "Proceed to `/wrapup-session`" (when failures or unmet DoD items remain)
- "Close this plan; revisit later" (the later revisit is an open-ended deferral without a fix commitment)

Silently closing a plan with red tests or unmet DoD converts a known, localized gap into a future archaeological dig. It is a category error — not a trade-off to weigh.

The halt report must present exactly two options:

- **Option A — Recommend specific fixes for user confirmation.** When the failures or unmet DoD items trace to identifiable root causes that you have enough context to propose concrete fixes for, enumerate each fix as a specific proposal — naming the file, the defect or gap, and the exact change — and ask the user to confirm before applying. Vague proposals like "investigate X" or "look into Y" are not Option A; they are Option B.

- **Option B — Recommend plan mode for thorough investigation.** When the failures or unmet DoD items trace to root causes you do not yet have enough context to fix, or when multiple competing fix approaches need to be weighed, or when the fix would require cross-cutting changes, recommend the user enter plan mode. Plan mode produces an investigation plan that the user then approves before any code changes happen.

Recommend exactly one of A or B with a one-sentence rationale grounded in the specific failure evidence or the specific DoD item that is unmet. Do not present a third "proceed past" option — not as the third bullet, not as an `AskUserQuestion` choice, not as "Other" escape-hatch framing. The never-proceed-past rule is not a default that users can override with a single message; it is a standing norm. If the user explicitly directs you to advance past red tests or unmet DoD, state that this violates the standing norm and ask them to confirm they want to override it before proceeding.

##### Output routing — main scratch log + terse terminal (mandatory)

The structured halt report — halt banner, evidence quotes, investigation summary, classification, full Option A enumeration with file paths and code blocks, full Option B description, and recommendation — is routed to the main session scratch under a `# Implementation friction` header. Terminal output is capped at a BLUF summary of five bullets or fewer. A single canonical location (the main session scratch, never the sidecar) is required so the user can parse and respond without hunting across files. During PWI resume, writing friction to main scratch is a permitted exception to the "do not read main scratch" rule — the friction write is append-only and does not depend on main's current content.

**Scratch routing**: append the full halt report via `scripts/generate/append_scratch_section.py`. If the payload contains backticks, double quotes, dollar signs, or backslashes, write the payload to a temp file via the `Write` tool first and pass its path via `--content-file {path}` (or via stdin if only `--content-stdin` is available). This sidesteps the JSON + bash + script escape-layer hazard that corrupts backtick-heavy markdown when inlined through `--content "..."`. The invocation shape:

    conda run -n agentic python -m scripts.generate.append_scratch_section \
      --scratch-doc {main-session-scratch-path} \
      --header "Implementation friction" \
      --subheader "{phase-name}" \
      --subsubheader "{brief-friction-summary} ({YYYY-MM-DDTHH:MM})" \
      --content-file {temp-payload-path}

- `{main-session-scratch-path}` is always the main session scratch file, never the sidecar.
- `{phase-name}` is the current plan phase — e.g., `Stage A — Implementation + QAQC + Commit`, `Stage B — Plan Closeout`, `Preflight`, `Plan-sync`. Use the stage name the protocol was executing when the halt fired.
- `{brief-friction-summary}` is a 3–8 word title naming what went wrong.
- `{YYYY-MM-DDTHH:MM}` is the current local datetime, suffixed to the subsubheader so multiple frictions in the same phase remain distinct.
- **Payload header depth**: the scaffold consumes H1/H2/H3 (`# Implementation friction > ## {phase} > ### {summary}`). Any section headers *inside* the payload body must start at level `####` (H4) or deeper to nest correctly under the scaffold. Using `##` or `###` inside the payload collapses the phase/summary hierarchy — the payload's `##` becomes a sibling of the scaffold's `## {phase}` and the payload's `###` becomes a sibling of the scaffold's `### {summary}`, flattening the friction report into the top-level section navigation.

**Re-read communication conventions**: immediately after the append call succeeds and before emitting terminal output, read the `# scratch doc communication protocols` section of the main scratch doc in full. Post-compaction agents may not carry the `agent reply conventions` and `user comment resolution` protocols in context (the sidecar's harness discipline block tells them not to read main scratch during resume, so these conventions only reach context via the targeted re-read here). Reloading them ensures the friction report's user-facing callouts and any subsequent interaction follow main-scratch conventions.

**Terminal output**: after the append call succeeds, emit to terminal:

1. One-line halt banner of the form: `Halted: {one-line description}. Friction appended to {main-scratch-path} under # Implementation friction > ## {phase} > ### {summary} ({datetime}).`
2. The two option labels (Option A name, Option B name) — no bullet bodies.
3. The one-sentence recommendation (A or B).
4. `Awaiting direction.`

Do not repeat the scratch-side content in the terminal. The user reads the structured report in scratch; the terminal is only a pointer.
%% inject-path-start: intermediate-multiphase-closeout %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/protocols/intermediate multiphase closeout.md`
    - Intermediate multiphase closeout protocol — parks a worktree mid-implementation without merging to main
    - Read in full
%% inject-path-end: intermediate-multiphase-closeout %%

## Post-Compaction Resume — First Action

### Part 1 — Read-coverage report (presumption-STOP gate)

**Presumption**: you have not read any files. The default state is "reads not
yet performed." Proceeding to Part 2 requires affirmative evidence of each
mandatory read. RLHF completion-bias creates a pull to produce a plausible-looking
table without having actually done the reads — recognize this pull and resist it.

For each mandatory read (sidecar, plan doc, pwi atomic path), run
`wc -l {absolute-path}` in a Bash tool call and paste the literal stdout output
into the assertion block. Prose descriptions of "I read this file" are not
acceptable — only `wc -l` stdout counts.

After each `wc -l` receipt, write a one-line self-assertion of the form:

- **Full-file read**: `Read coverage: all N lines read, understanding consistent with length.`
- **Scoped read** (when the resume prompt or sidecar explicitly specifies a scoped read, e.g. "read § Section X and everything below"): `Read coverage: scoped to <section name>, K of N lines covered, understanding consistent with scope.`

Substitute the observed N (and K when scoped). The self-assertion is the substantive evidence; the `wc -l` output alone is the ritual. Both are required. If you cannot truthfully write the self-assertion — because you did not actually read the file in the required scope, or because your understanding is inconsistent with the observed coverage — HALT. Do not fabricate the self-assertion to satisfy the gate.

A scoped-read assertion is only valid when the resume prompt or sidecar **explicitly** instructs a scoped read. The default posture is full-file coverage; "scoped" is the affirmative-opt-in exception.

Format:

```
$ wc -l /home/.../sidecar_atomic_2026-04-16_2000.md
142 /home/.../sidecar_atomic_2026-04-16_2000.md
Read coverage: all 142 lines read, understanding consistent with length.
$ wc -l /home/.../library/docs/planning/.../my atomic plan.md
188 /home/.../library/docs/planning/.../my atomic plan.md
Read coverage: all 188 lines read, understanding consistent with length.
$ wc -l /home/.../library/prompts/instructions/skills/proceed-with-implementation/pwi atomic path.md
305 /home/.../library/prompts/instructions/skills/proceed-with-implementation/pwi atomic path.md
Read coverage: scoped to ## Implementation and Closeout, 120 of 305 lines covered, understanding consistent with scope.
```

If any `wc -l` count does not match your understanding of the file's length —
or if you cannot produce `wc -l` output for a file you claim to have read —
or if you cannot truthfully write the per-file self-assertion — HALT.
That is the presumption-STOP gate firing. Do not fabricate counts or
self-assertions; run the tool and read the files.

### Part 2 — Checkpoint-match assertion

Read the sidecar's `## Compaction Checkpoint (latest)` section and write:

`Checkpoint-match assertion: Sidecar: sidecar_atomic_2026-05-14_1543.md. Leave-off step (from
checkpoint): collaborative resolution complete, entering implementation. Plan: plotly-figure-file-extension-html.
Checkpoint matches sidecar's ported-context sections: YES|NO.`

If NO, HALT and report the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke the script via `conda run` with the `-m` form:

    conda run -n agentic python -m scripts.generate.write_compaction_sidecar --mode verify \
      --main-scratch /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-14_1432_plotly-html-extension-pwi.md \
      --worktree /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi

**Invocation gotcha**: do NOT use `conda run --cwd <dir> python -m scripts.generate.write_compaction_sidecar ...`. The `--cwd` flag to `conda run` is unreliable for `-m` module invocation in this environment — it produces `ModuleNotFoundError: No module named 'scripts.generate.write_compaction_sidecar'`. Instead, run from the intended `cwd` directly (e.g., `cd` into the worktree first, then invoke without `--cwd`), or omit `--cwd` entirely since the script does not depend on a specific working directory for its own logic.

Assert the exit code:

- Exit 0: write `Sync integrity (from --mode verify exit code): OK. Worktree: clean.`
- Exit 2: script detected a hash mismatch. The script's stderr contains unified
  diffs. HALT per post-halt response protocol. Do NOT attempt recovery.
- Exit 3: script detected uncommitted changes in worktree. HALT — the previous
  round did not complete QAQC+commit cleanly.

### Conjunctive GO gate (required written assertion before Implementation and Closeout)

Produce exactly the following assertion text verbatim, substituting GO or STOP
for each Part:

```
Part 1: GO|STOP
Part 2: GO|STOP
Part 3: GO|STOP
Conjunction: (Part 1 = GO) AND (Part 2 = GO) AND (Part 3 = GO) = GO|STOP
Proceeding to Implementation and Closeout: YES|NO
```

The `Proceeding to Implementation and Closeout: YES` line is the single gate event. Any
STOP in any Part means NO. Writing the conjunction is required — it is the
forcing function against RLHF task-completion pressure that would otherwise push
the agent to transition silently.

Then proceed directly to `## Implementation and Closeout` for the atomic plan. Do not summarize
what you learned — the sidecar is the record.

### Phase Audit History

*(no phases completed yet — first sidecar of this session)*

## Follow-up Ideas

- **Flaky synth-tier end-to-end test (`tests/test_synth_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end`)**: Confirmed pre-existing on `main` (fails on baseline with my changes stashed and a freshly-cleared cache, 3:11s); independent of the plotly-html-extension plan. Failure mode: `submit_workflow` returns `success=False` because Snakemake exits non-zero; Snakemake master log shows execution proceeds ~10–11 of 35 steps then truncates without a visible `RuleException` or `Error in rule` — just stops mid-execution. Sometimes a `LockException` appears (stale `.snakemake/locks/` from prior killed runs). Sometimes the test passes (one isolated run was 151s GREEN; another baseline isolated run was 190s GREEN). User explicitly overrode the PWI never-proceed-past-failing-tests norm at plan closeout to commit the plotly-html-extension work; this flake is the durable follow-up. Worth its own bug-fix plan — diagnose Snakemake state-management interaction with `submit_workflow`'s lock-clear logic and the `start_from_scratch=True` fixture path. **Capture via `/capture-idea` at wrap-up.**

- **Halt trigger #3 fired at resume (2026-05-14)**: `--mode verify` exited 3 — dirty worktree from sidecar archival, not from unfinished implementation work. Affected paths:
  - `sidecars/sidecar_phase5_2026-05-12_2302.md` (deleted/tracked → moved)
  - `sidecars/done/sidecar_phase5_2026-05-12_2302.md` (untracked, archival destination)
  - `sidecars/sidecar_atomic_2026-05-14_1543.md` (untracked — the current in-flight sidecar)

  This is the exact `write_compaction_sidecar.py --mode create` archive-without-commit pattern already captured as a follow-up in the archived `sidecar_phase5_2026-05-12_2302.md`. Leave-off step at halt: `collaborative resolution complete, entering implementation`. Plan: plotly-figure-file-extension-html.

- **PWI worktree-setup should `pip install -e .` from the new worktree** (workspace: TRITON-SWMM_toolkit; affected file: `$AGENTIC_WORKSPACE/scripts/generate/setup_pwi_worktree.py`). Problem: the conda env `triton_swmm_toolkit` has exactly one editable install, pointing at whichever worktree last ran `pip install -e .`. When a fresh PWI worktree is created via `setup_pwi_worktree.py`, the env still binds to the previous worktree's `src/`. Failure mode is silent miscompile: pytest collects tests from the new worktree's `tests/` but `import TRITON_SWMM_toolkit` resolves to the *previous* worktree's `src/`, so source-side changes are invisible to pytest until the developer (or agent) runs `pip install -e .` from the new worktree. Diagnosed this session: my first synth-tier test run generated a `.svg`-extension Snakefile despite my `_OUTPUT_EXT_BY_RENDERER` edit to `.html`, because pytest was importing the prior worktree's pre-edit code. Subprocess Snakemake rules use `/home/dcl3nd/miniconda3/envs/triton_swmm_toolkit/bin/python` which inherits the same editable install — so the bug propagates to every subprocess too. The `test_snakemake_workflow_end_to_end` test (`tests/test_synth_04_multisim_with_snakemake.py:208–220`) already has a `subprocess` probe asserting `str(resolved).startswith(str(repo_src))` for this exact reason — but pytest's own collection-time imports aren't checked. Direction: extend `setup_pwi_worktree.py` to run `conda run -n {env_name} pip install -e .` (or a `uv`-equivalent) from the new worktree as part of its bootstrap, with `env_name` resolved from workspace-architecture frontmatter or a project-level config. Cross-workspace risk: this rebinds the global env, so any *other* open session in a different worktree will silently start running the new worktree's code — surface a one-line warning. Worth its own atomic plan in workspace: agentic-workspace.

- **`subagent-stop-scratch-gate.py` hook fires spuriously on suggestion-mode subagents** (workspace: agentic-workspace; affected file: `$AGENTIC_WORKSPACE/library/prompts/runtimes/claude_code/hooks/subagent-stop-scratch-gate.py`). Problem: the Stop hook requires every subagent's `last_assistant_message` to contain a `SCRATCH_REPORT:` line, but suggestion-mode subagents (the ones spawned by Claude Code's `[SUGGESTION MODE: ...]` prompt to predict the next user input) produce 2-12-word predictions and are not session work — they have no scratch-doc obligation. Observed this session: hook fired with `L1 FAIL: SCRATCH_REPORT line absent` after a suggestion-mode invocation produced "yes push it", with transcript at `/home/dcl3nd/.claude/projects/-home-dcl3nd-dev-TRITON-SWMM-toolkit--claude-worktrees-toolkit-05-14-1432-plotly-html-extension-pwi/7dc9710e-817b-467e-8f2c-772930774243/subagents/agent-a12840f84c052bb51.jsonl`. Direction: detect suggestion-mode subagents (the system prompt starts with `[SUGGESTION MODE:`) and skip the SCRATCH_REPORT gate for them; alternatively, add a subagent-purpose classification step at hook entry. Worth an atomic plan in workspace: agentic-workspace.

- **Pre-existing `tests/test_synth_08_bundle_round_trip.py` failure — stale `bundle.tar` references after tar→zip switch** (workspace: TRITON-SWMM_toolkit; affected file: `tests/test_synth_08_bundle_round_trip.py` lines 84 + 146). Problem: commit `58e8221` ("feat(bundle): Phase 4 — zip emit supersedes tar") switched the bundle output format from `.tar` to `.zip` and updated `tests/test_bundle.py` accordingly, but did not update `tests/test_synth_08_bundle_round_trip.py`. The test still constructs `bundle_tar = tmp_path / "bundle.tar"` and passes that path to `TRITON_SWMM_toolkit report-from-bundle ...` — which now raises `CLIValidationError: ... is neither a .zip file nor a directory` in `cli.py:878`. 4 parametrized tests fail: `test_bundle_round_trip[rendered_synth_multi_sim]`, `test_bundle_round_trip[rendered_synth_sensitivity]`, `test_bundle_baseline_wrapper_section_matches[rendered_synth_multi_sim]`, `test_bundle_baseline_wrapper_section_matches[rendered_synth_sensitivity]`. Verified pre-existing on baseline (my plotly-html plan changes stashed, clean cache → identical failure mode). Direction: find-and-replace `bundle.tar` → `bundle.zip` at the two construction sites in `tests/test_synth_08_bundle_round_trip.py`, then inspect any `tarfile.open(...)` / `subprocess` calls downstream and convert the `tar`-API to `zipfile`-API. Worth a tiny atomic plan or a direct fix.

- **Per-worktree test cache for synth-tier tests** (workspace: TRITON-SWMM_toolkit; affected files: `tests/fixtures/test_case_catalog.py`, `tests/conftest.py`, `src/TRITON_SWMM_toolkit/analysis.py:1685, 2174` (LAYOUT_VERSION stamping hooks); also relevant: `tests/utils_for_testing.py`). Problem: `Local_TestCases.retrieve_synth_multi_sim_test_case(...)` and sibling test-case retrievers stage their workflow into a single global path `/home/dcl3nd/.cache/TRITON_SWMM_toolkit/synthetic_test_runs/{case_name}/`. Multiple worktrees (and main) all read/write the same `_status/*.flag`, `.snakemake/locks/`, `plots/`, `analysis_config.yaml`. Consequence: stale Snakemake locks from a killed run in one worktree fail-fast subsequent runs in another worktree; partial `.svg`/`.html` plot outputs from a different extension regime leak across sessions; the `start_from_scratch=True` semantics don't fully clean state (e.g., LAYOUT_VERSION stamps and V0005 migration side-effects persist). Failure surface diagnosed this session: `test_snakemake_workflow_end_to_end` flaked between PASS (151–190s) and FAIL (14s fail-fast on stale lock; 191s truncate-mid-execution) depending on cache residue at run time. Direction: route `synthetic_test_runs/` under `.claude/worktrees/{slug}/cache/synthetic_test_runs/` (or use pytest's `tmp_path`-based per-session caching), keyed on the worktree path. Trade-off: per-worktree caches cost more disk and don't share compiled TRITON/SWMM binaries across worktrees — mitigate by symlinking the `_software/` subdir (compiled binaries) into the per-worktree cache while keeping per-analysis state isolated. Related upstream issue: commit `4c3be71` (V0005 inline report_config) writes `report: {}` into `analysis_config.yaml` but the `analysis_config` pydantic schema in `src/TRITON_SWMM_toolkit/config/analysis.py` does NOT have a `report` field with `extra=forbid` — schema-load fails with `Extra inputs are not permitted`. This is a *separate* pre-existing bug on main that compounded the cache-corruption flakiness during diagnosis; capture as its own follow-up if not already tracked. Worth its own atomic plan.

%%
"#" + "followup" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.

Specialist findings routed here by /invoke-specialist Step 6 appear under:
  ### Specialist-identified follow-up items
  #### {specialist-name}
These are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.

Agents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.
%%

## Worktree Status


- Worktree path: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-14_1432_plotly-html-extension-pwi`
- Branch: `worktree-toolkit_05-14_1432_plotly-html-extension-pwi`
- Plan frontmatter: `plan_status: in_progress`, `worktree_branch: worktree-toolkit_05-14_1432_plotly-html-extension-pwi`
- Isolation: confirmed

### Phases Remaining

*(section not present in main scratch at create time)*

