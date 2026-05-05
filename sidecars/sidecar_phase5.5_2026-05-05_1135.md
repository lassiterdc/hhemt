---
prompt_doc_type: sidecar_scratch
main_scratch: /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md
last_sync_hashes:
  '### Phase Audit History': ca148f98cf69f8a8465d664c90de057aae281e039cc8fcbe0e6b7b95bca30e4a
  '# Follow-up Ideas': 1e37c6c057de06ae08c94342cf470e48915be6437c3bf9240c98bfde8deb03d1
harness: claude-code
next_phase: '5.5'
next_phase_title: baseline failure remediation (Option B consolidation collapse + Phase 4 audit)
next_phase_doc: /home/***REMOVED***/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/render_bundle/5.5 baseline failure remediation.md
leave_off_step: Phases 4 + 5 complete and pushed; Phase 5.5 doc landed on main (commit 6a02d0397). Compaction firing before Phase 5.5 implementation begins.
worktree_path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi
created: '2026-05-05T11:35:23'
smoke_test: false
sidecar_kind: in_flight_multiphase
baseline_writable_content:
  '### Phase Audit History': "\n\n\n\n\n| Phase | Phase-Local Diff Files | Audit-on-Diff Violations | Resolution | Commit |\n|-------|------------------------|--------------------------|------------|--------|\n| 1 — renderer refactor: eliminate per-sim file dependencies | `src/TRITON_SWMM_toolkit/report_renderers/{per_sim_peak_flood_depth,per_sim_conduit_flow,_hydrology_panel,per_analysis_summary}.py` | None applicable (agentic-workspace `scripts.audit` audits prompt docs, not toolkit Python code). Ruff: zero new violations introduced (10 pre-existing preserved). | N/A — no toolkit-side audit framework. Ruff baseline preserved. | 234681d (toolkit) |\n| 2 — bundle spec discovery via curated example | aw: `library/docs/decisions/TRITON-SWMM_toolkit/bundle layout and contents.md` (new); plan-closeout move of `2 bundle spec...md` to `implemented/`. No toolkit code touched. | N/A (markdown only; no audit framework targets decision docs). | N/A. | 08e6099a6 (aw) |\n| 3 — manifest-harvest infrastructure + renderer audit + stipulation | toolkit: `src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py` (added `harvest_source_paths`); `src/TRITON_SWMM_toolkit/scenario.py` (orthogonal b1 fix to TRITON-only compile gate). aw: `library/docs/stipulations/TRITON-SWMM_toolkit/bundle file set is computed from manifest harvest.md` (new). | Ruff: clean. pytest: 13 pre-existing baseline failures verified orthogonal via `git stash` against worktree tip; Phase 3 + b1 introduce zero regressions. b1 fix reduced baseline failure count 14→13. | 13 baseline failures captured as test-suite-baseline-repair follow-up below. | toolkit: f3aaa90 (Phase 3 helper) + 8b154aa (b1 fix); aw: e012a387d (stipulation) |\n\n"
  '# Follow-up Ideas': "\n\n\n\n\n%%\n\"#\" + \"followup\" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.\n\nSpecialist findings routed here by /invoke-specialist Step 6 appear under:\n  ###### Specialist-identified follow-up items\n  ###### {specialist-name}\nThese are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.\n\nAgents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.\n%%\n\n\n###### Specialist-identified follow-up items\n\n###### triton-swmm-toolkit-specialist\n\n- **Planning-instruction improvement proposal (from Phase 1 review)**: Empirical-schema errors (Flags 1, 7, 8) reveal that plan authors sometimes write pseudocode against a remembered API surface rather than the current source. Targeted protocol amendment: \"before quoting an API in plan pseudocode, verify the API by reading the current source file and citing line numbers.\" Likely belongs as a `/plan-implementation` skill body addition rather than a project-specific rule.\n\n###### software-engineering-specialist\n\n- **Flag 9 — CLI binary naming inconsistency**: Codebase has two competing CLI binary names — documented `triton-swmm` (in CLAUDE.md Common Commands) and the actually-installed `TRITON_SWMM_toolkit` (per pyproject.toml [project.scripts]). Bundle plan correctly uses the installed name but exposes the pre-existing inconsistency: users following CLAUDE.md will type `triton-swmm bundle` and get \"command not found.\" Resolve by either renaming the binary in pyproject.toml or updating CLAUDE.md, orthogonal to bundle workflow.\n- **Flag 10 — Manifest-schema forward-compat narrative drift**: Phase 2 stipulation's \"Forward Compatibility\" clause references a `{\"figure\": {\"sources\": ...}}` → `{\"figure\": {\"panels\": ...}}` migration path, but the harvest helper actually reads `source_paths_relative` (top-level) and `artists[].channels[].ref.source_path` (nested) — neither shape matches. Documentation drift, not implementation defect; surfaces a gap for the panel-grouped provenance plan author to resolve before that plan starts.\n\n###### Smell follow-ups\n\n- **Porter silent-truncation defense (smell at line 2455+ of this scratch)** — agentic-workspace target. Four interventions documented inline as Verbatim Modification Specs: (1) porter exit-4 STRUCTURAL_VIOLATION check + content-floor warning; (2) body-plan-implementation atomic adds explicit nested-H1/H2 prohibition inside Phase H3; (3) invoke-specialist SKILL Plan-Specific Post-Completion adds pre-Step-5 fidelity gate; (4) porter `known_risks` metadata reflects new defense. Promote via `/plan-implementation` after this PWI session closes — atomic plan in agentic-workspace, title candidate: 'porter silent-truncation defense: nested-H1 detection + post-port fidelity gate'.\n\n###### Test-suite baseline repair (Phase 3 QAQC finding)\n\n- **13 pre-existing baseline failures in `pytest tests/ -m \"not slow\"`** (verified pre-Phase-3 via `git stash` against worktree tip `2ae3e19`). Phase 3 + b1 do not introduce any of these; b1 actually reduced the count from 14 → 13 (fixed `test_triton_only_cfg_generation`). Failure clusters:\n  - **Provenance discipline (5 failures)** — `tests/test_provenance_discipline.py`: `per_sim_conduit_flow.py:252` and `:256` have `.plot(...)` calls outside `with prov.artist(...)` blocks (Phase 1 renderer-refactor defect, committed in `234681d`); `sensitivity_benchmarking.py`, `errors_and_warnings.py`, `scenario_status_appendix.py` lack a provenance block entirely (pre-Phase-1).\n  - **TRITONSWMM CPU compile-gate (4 failures)** — `tests/test_synth_00_compile_models.py::test_create_dem_for_TRITON`,
    `::test_create_mannings_file_for_TRITON`, `tests/test_swmm_threads_implementation.py::test_swmm_threads_updated_in_inp_files`, `::test_swmm_threads_different_values`. All call `prepare_scenario` in coupled mode (`toggle_tritonswmm_model = True`) without first compiling TRITONSWMM CPU. Either fixture-side (auto-compile in setup) or test-side (explicit compile call) repair needed.\n  - **Snakemake-driven multisim (4 failures)** — `tests/test_synth_04_multisim_with_snakemake.py`: `test_snakemake_workflow_end_to_end`, `test_run_and_render_report`, `test_render_report_idempotent`, `test_plot_sources_attribution`. Failures not investigated in detail (structurally unreachable from Phase 3 / b1 changes).\n  - Promote as a separate atomic plan: title candidate \"test suite baseline repair: provenance-discipline gaps + TRITONSWMM CPU compile fixture + synth_04 Snakemake regressions\". Recommended scope: 3 commits (one per cluster), with stash-test evidence in each commit message.\n\n%%\nAgents: fill in this section as follows: Record the worktree path and branch.\n%%\n\n- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`\n- **Branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`\n- **Isolation**: confirmed (sparse-checkout configured by `setup_pwi_worktree`; `library/docs/planning/` and `library/_scratch/` excluded — agentic-workspace planning/scratch reads use `$AGENTIC_WORKSPACE/...` absolute paths)\n- **Plan frontmatter**: `plan_status: in_progress`, `worktree_branch: worktree-toolkit_05-03_1359_render-bundle-pwi` recorded by `setup_pwi_worktree`.\n\n**Note**: Initial `EnterWorktree` call used a wrong slug (`05-03_1359_...`); that worktree was removed via `ExitWorktree(action: remove)` before any work landed. Canonical slug per the protocol's \"strip first `_`-delimited segment from scratch doc stem\" rule on `TRITON-SWMM_toolkit_05-03_1359_...` is `toolkit_05-03_1359_...` — agreeing with `setup_pwi_worktree`'s derivation.\n\n##### Phases Remaining\n\n*(section not present in main scratch at create time)*\n\n"
---
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

- Leave-off step: Phases 4 + 5 complete and pushed; Phase 5.5 doc landed on main (commit 6a02d0397). Compaction firing before Phase 5.5 implementation begins.
- Next phase: Phase 5.5 — baseline failure remediation (Option B consolidation collapse + Phase 4 audit)
- Next phase doc: `/home/***REMOVED***/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/render_bundle/5.5 baseline failure remediation.md`
- Worktree: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`

## Post-Compaction Resume — Preamble

You are resuming a multiphase plan implementation after an in-session /compact.
The first message you receive after compaction looks structurally like a fresh
session invocation, and RLHF training creates a pull to treat it as one — running
preflight, re-reading the full session scratch doc, re-dispatching specialist
reviews. Recognize this pull. Your actual task is to rehydrate from the sidecar
scratch doc described below, verify sync integrity via the script, and continue
the autonomous phase loop from the recorded leave-off step. The sidecar is your
source of truth; the main session scratch doc is out of scope until
/wrapup-session.

### Mandatory reads (in order, each in full)

1. Sidecar scratch doc:
   `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi/sidecars/sidecar_phase5.5_2026-05-05_1135.md`

2. Next phase doc:
   `/home/***REMOVED***/dev/agentic-workspace/library/docs/planning/projects/TRITON-SWMM_toolkit/features/render_bundle/5.5 baseline failure remediation.md`

3. [pwi multiphase path](../../instructions/skills/proceed-with-implementation/pwi%20multiphase%20path.md) #inject_path — `## Autonomous Phase Loop` section and everything below it.

%% inject-path-start: pwi-multiphase-path %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/proceed-with-implementation/pwi multiphase path.md`
    - Multiphase path protocol for /proceed-with-implementation: specialist review, master preflight, collaborative resolution, autonomous phase loop with stop-gate evaluation, and wrap-up.
%% inject-path-end: pwi-multiphase-path %%

### Next phase summary

- **Phase**: 5.5 — baseline failure remediation (Option B consolidation collapse + Phase 4 audit)
- **Heaviness**: heavy
- **Session gate (after completion)**: true
- **Dependencies**: library/docs/planning/projects/TRITON-SWMM_toolkit/features/render_bundle/implemented/5 cli commands pathportability rewrite bundleschemaversion.md

Do NOT read the main scratch doc during this resume. It is the destination
for /wrapup-session only. All context you need is in the sidecar.

### Leave-off step

The prior session halted at: `Phases 4 + 5 complete and pushed; Phase 5.5 doc landed on main (commit 6a02d0397). Compaction firing before Phase 5.5 implementation begins.`

Resume work from this step, not from the top. If the leave-off step does not
correspond to a recognizable anchor in the referenced doc, HALT and report the
inconsistency — do not guess.

### Worktree context

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- **Worktree branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`

All file writes for this resume go to the worktree. Planning-doc writes use
the Edit or Write tool with an absolute
`$AGENTIC_WORKSPACE/library/docs/planning/<subpath>.md` path
(`block-main-tree-write-from-worktree.sh` allows main-tree planning writes
through; `block-worktree-path-write.sh` blocks only relative-path attempts).
Scratch-doc writes (to the sidecar or any same-session scratch) go through
the Write tool directly — the sidecar lives under `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi/sidecars/`
and is worktree-local.

## Post-Compaction Resume — Halt Triggers

### Halt triggers

1. Tier 1 stop gate: QAQC `input needed` section non-empty → immediate halt.
2. Tier 2 stop gate: any of the four assertion-based conditions fails.
3. Session capacity gate: heavy phase after ≥3 phases completed in session.
4. Explicit user interrupt.
5. Sidecar consistency failure — see 4-condition checklist below.
6. Sync-back integrity failure: `write_compaction_sidecar.py --mode verify` or
   `--mode sync-back` exits with code 2 (hash mismatch) or code 3 (worktree
   dirty). Never overwrite main on a mismatch.

#### Halt trigger #5 — sidecar consistency failure

Trigger if ANY of the following four conditions fail when you read the sidecar:

1. The sidecar frontmatter's `next_phase` field numerically matches the single
   phase number that appears in the `## Compaction Checkpoint (latest) → Next phase` line.
2. The sidecar's `## Multiphase Preflight Report → ### Phases Remaining` section's
   first numbered item matches the sidecar's `next_phase_title` frontmatter field.
3. The sidecar's `## Worktree Status` section's branch line matches the sidecar
   frontmatter's `worktree_path` field's derivable branch name (`worktree-toolkit_05-03_1359_render-bundle-pwi`).
4. The sidecar's `last_sync_hashes:` frontmatter contains exactly two entries:
   `### Phase Audit History` and `# Follow-up Ideas`.

Write each condition's result explicitly as `Condition N: PASS|FAIL — [observed vs. expected]`
before producing the GO signal. Any FAIL halts.

### Post-halt response protocol

When any halt trigger fires:

1. **Write the halt state** to the sidecar's `# Follow-up Ideas` section with:
   - The halt trigger number that fired
   - The specific condition or evidence that triggered it
   - The current leave-off step and next phase

2. **Run the script in final sync-back mode** — invoke:
   `scripts/generate/write_compaction_sidecar.py --mode sync-back --main-scratch {abs-path} --worktree {abs-path}`
   to push the sidecar's writable edits (including the halt record you just wrote) back
   to main scratch. Assert exit 0.

3. **Wait for user direction**. Do not attempt to diagnose, work around, or retry
   the halt condition on your own. Produce a single-message halt report:
   "Halted: trigger #N — {one-line description}. Sync-back completed. Awaiting direction."

4. **Route on user direction**: either trigger another compaction + resume cycle (if user instructs) or follow the [intermediate multiphase closeout](../../instructions/protocols/intermediate%20multiphase%20closeout.md) #inject_path fallback protocol (if user instructs closeout).

#### Trigger #6 (sync-back integrity failure) — mandatory investigate-first recovery framing

When the halt is specifically trigger #6 (hash mismatch between sidecar baseline and main scratch's `### Phase Audit History` / `# Follow-up Ideas`), the halt report must extend Step 3 above with three additional components — do NOT present a generic menu of recovery options. The presumption is that the agent invests effort in understanding the divergence before offering any path forward.

1. **Investigate the discrepancy first.** Read the `--mode verify` diff output against the sidecar's `baseline_writable_content:` frontmatter. For each mismatched section, characterize the nature of the divergence: is it whitespace-only, TPL-INSTRUCTION-block skeleton drift, heading-level drift, or durable content (new bullets, audit rows, user-added notes)? The investigation is mandatory; do not skip to recovery options on the assumption that the divergence is obvious.

2. **Classify as one-time vs. systematic.** Report whether the divergence looks like:
   - **One-time**: a manual edit on main that happened in the frozen window, a late hook-driven normalization, or a late `--inject-into-scratch` from a parallel skill invocation. No indication of a broken invariant in the compaction machinery.
   - **Systematic**: a pattern likely to recur on future sessions — e.g., a hook that strips or injects content into the writable sections unconditionally, a script that writes to main while a sidecar is active, or a baseline-capture bug in `write_compaction_sidecar.py`. Requires a durable fix before any further compaction usage is safe.

3. **Present exactly two recovery options with a clear recommendation** — never more, never an open-ended menu:
   - **Option A — Reconcile main to sidecar baseline.** Edit the main scratch's mismatched sections back to the `baseline_writable_content` verbatim, then re-run `--mode verify`. Preserves the audit trail; safest when durable content is involved.
   - **Option B — Re-emit sidecar against current main.** Delete the current sidecar and re-invoke `--mode create` against current main, adopting main as the new baseline. Fastest for benign divergence; discards the baseline mismatch evidence.

   Based on the Step 2 classification, recommend exactly one of A or B with a one-sentence rationale. Do not present the intermediate-closeout protocol as a recovery option at this stage — it is only appropriate if the user rejects both A and B, or if the Step 2 classification is "systematic" and the user elects to park the worktree while the systemic issue is fixed out-of-band.

The halt report to the user should have this structure: halt line → one-paragraph investigation summary → one-line classification → two-option presentation (A and B only) with recommendation. Do not frame this as "options 1/2/3" or "here are recovery paths to choose from."

##### Output routing — main scratch log + terse terminal (mandatory)

The structured halt report — halt line, investigation summary, classification, and two-option presentation with recommendation — is routed to the main session scratch under `# Implementation friction > ## {phase} > ### sync-back integrity mismatch ({YYYY-MM-DDTHH:MM})` via `scripts/generate/append_scratch_section.py`, not to the terminal and not to the sidecar. When the payload contains backticks, quotes, or dollar signs, write it to a temp file first and pass `--content-file {path}` to sidestep the Bash escape-layer hazard. Any section headers *inside* the payload body must start at level `####` (H4) or deeper — the scaffold consumes H1/H2/H3, so `##` or `###` in the payload collapses the phase/summary hierarchy. Immediately after the append call succeeds and before emitting terminal output, read the `# scratch doc communication protocols` section of the main scratch doc in full — post-compaction agents may not carry the `agent reply conventions` and `user comment resolution` protocols in context, and reloading them ensures the friction report's user-facing callouts and any subsequent interaction follow main-scratch conventions. Terminal output is capped at five bullets or fewer: the one-line halt banner with the scratch pointer, the two option labels (Option A / Option B — no bullet bodies), the one-sentence recommendation, and `Awaiting direction.` Do not repeat scratch-side content in terminal.

#### Test failures and incomplete DoDs — mandatory two-option recovery framing (never-proceed-past rule)

When a halt is triggered by **either** of the following conditions, they are an absolute blocker to any further plan advancement:

- One or more failing tests — surfaced at QAQC time (Tier 1 `input needed` non-empty), at stop-gate evaluation (Tier 2 rubric), or mid-implementation. "Which tier of test" is irrelevant; red is red. Unit, integration, smoke, slow, fast, scoped, cross-cutting — all in scope.
- One or more unmet Definition-of-Done items on the current phase or plan — whether measurement targets unmet (e.g., runtime target missed, coverage target missed), capability targets unmet (e.g., "all of X passes"), or artifact targets unmet (e.g., required doc sections missing, required outputs not generated).

The halt report must NOT offer the user any option that would close the current phase or advance to the next one while either condition holds. Specifically forbidden framings:

- "Proceed to next phase with failures/gaps as follow-up"
- "Accept current state and continue"
- "Capture failures/unmet DoD items as tech-debt and move on"
- "Commit partial progress and advance" (when framed as a path forward rather than as rescue-commit before fix work)
- "Proceed to `/wrapup-session`" (when failures or unmet DoD items remain)
- "Close this phase; revisit later" (the later revisit is an open-ended deferral without a fix commitment)

Silently closing a phase with red tests or unmet DoD converts a known, localized gap into a future archaeological dig. It is a category error — not a trade-off to weigh.

The halt report must present exactly two options:

- **Option A — Recommend specific fixes for user confirmation.** When the failures or unmet DoD items trace to identifiable root causes that you have enough context to propose concrete fixes for (e.g., a missing config field, an incorrect template ordering, a known toolkit defect with a narrow workaround, a missing assertion needed to meet a DoD checkbox), enumerate each fix as a specific proposal — naming the file, the defect or gap, and the exact change — and ask the user to confirm before applying. Vague proposals like "investigate X" or "look into Y" are not Option A; they are Option B.

- **Option B — Recommend plan mode for thorough investigation.** When the failures or unmet DoD items trace to root causes you do not yet have enough context to fix, or when multiple competing fix approaches need to be weighed, or when the fix would require cross-cutting changes (toolkit internals, multiple components, design trade-offs that affect the broader plan), recommend the user enter plan mode. Plan mode produces an investigation plan that the user then approves before any code changes happen.

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
- `{phase-name}` is the current plan phase — e.g., `Phase 3 — Implementation`, `Phase Closeout`, `Preflight`, `Plan-sync`. Use the stage or phase name the protocol was executing when the halt fired.
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

For each mandatory read (sidecar, next phase doc, pwi multiphase path), run
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
$ wc -l /home/.../sidecar_phase2_2026-04-14_2000.md
147 /home/.../sidecar_phase2_2026-04-14_2000.md
Read coverage: all 147 lines read, understanding consistent with length.
$ wc -l /home/.../library/docs/planning/.../2 plan completeness assessment atomic.md
193 /home/.../library/docs/planning/.../2 plan completeness assessment atomic.md
Read coverage: all 193 lines read, understanding consistent with length.
$ wc -l /home/.../library/prompts/instructions/skills/proceed-with-implementation/pwi multiphase path.md
412 /home/.../library/prompts/instructions/skills/proceed-with-implementation/pwi multiphase path.md
Read coverage: all 412 lines read, understanding consistent with length.
```

If any `wc -l` count does not match your understanding of the file's length —
or if you cannot produce `wc -l` output for a file you claim to have read —
or if you cannot truthfully write the per-file self-assertion — HALT.
That is the presumption-STOP gate firing. Do not fabricate counts or
self-assertions; run the tool and read the files.

### Part 2 — Checkpoint-match assertion

Read the sidecar's `## Compaction Checkpoint (latest)` section and write:

`Checkpoint-match assertion: Sidecar: sidecar_phase5.5_2026-05-05_1135.md. Leave-off step (from
checkpoint): Phases 4 + 5 complete and pushed; Phase 5.5 doc landed on main (commit 6a02d0397). Compaction firing before Phase 5.5 implementation begins.. Next phase: Phase 5.5 — baseline failure remediation (Option B consolidation collapse + Phase 4 audit).
Checkpoint matches sidecar's ported-context sections: YES|NO.`

If NO, HALT and report the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke the script via `conda run` with the `-m` form:

    conda run -n agentic python -m scripts.generate.write_compaction_sidecar --mode verify \
      --main-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md \
      --worktree /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi

**Invocation gotcha**: do NOT use `conda run --cwd <dir> python -m scripts.generate.write_compaction_sidecar ...`. The `--cwd` flag to `conda run` is unreliable for `-m` module invocation in this environment — it produces `ModuleNotFoundError: No module named 'scripts.generate.write_compaction_sidecar'`. Instead, run from the intended `cwd` directly (e.g., `cd` into the worktree first, then invoke without `--cwd`), or omit `--cwd` entirely since the script does not depend on a specific working directory for its own logic.

Assert the exit code:

- Exit 0: write `Sync integrity (from --mode verify exit code): OK. Worktree: clean.`
- Exit 2: script detected a hash mismatch. The script's stderr contains unified
  diffs. HALT per post-halt response protocol. Do NOT attempt recovery.
- Exit 3: script detected uncommitted changes in worktree. HALT — the previous
  round did not complete QAQC+commit cleanly.

### Conjunctive GO gate (required written assertion before Per-Phase Preflight)

Produce exactly the following assertion text verbatim, substituting GO or STOP
for each Part:

```
Part 1: GO|STOP
Part 2: GO|STOP
Part 3: GO|STOP
Conjunction: (Part 1 = GO) AND (Part 2 = GO) AND (Part 3 = GO) = GO|STOP
Proceeding to Per-Phase Preflight: YES|NO
```

The `Proceeding to Per-Phase Preflight: YES` line is the single gate event. Any
STOP in any Part means NO. Writing the conjunction is required — it is the
forcing function against RLHF task-completion pressure that would otherwise push
the agent to transition silently.

Then proceed directly to Per-Phase Preflight for the next phase. Do not summarize
what you learned — the sidecar is the record.

### Phase Audit History






| Phase | Phase-Local Diff Files | Audit-on-Diff Violations | Resolution | Commit |
|-------|------------------------|--------------------------|------------|--------|
| 1 — renderer refactor: eliminate per-sim file dependencies | `src/TRITON_SWMM_toolkit/report_renderers/{per_sim_peak_flood_depth,per_sim_conduit_flow,_hydrology_panel,per_analysis_summary}.py` | None applicable (agentic-workspace `scripts.audit` audits prompt docs, not toolkit Python code). Ruff: zero new violations introduced (10 pre-existing preserved). | N/A — no toolkit-side audit framework. Ruff baseline preserved. | 234681d (toolkit) |
| 2 — bundle spec discovery via curated example | aw: `library/docs/decisions/TRITON-SWMM_toolkit/bundle layout and contents.md` (new); plan-closeout move of `2 bundle spec...md` to `implemented/`. No toolkit code touched. | N/A (markdown only; no audit framework targets decision docs). | N/A. | 08e6099a6 (aw) |
| 3 — manifest-harvest infrastructure + renderer audit + stipulation | toolkit: `src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py` (added `harvest_source_paths`); `src/TRITON_SWMM_toolkit/scenario.py` (orthogonal b1 fix to TRITON-only compile gate). aw: `library/docs/stipulations/TRITON-SWMM_toolkit/bundle file set is computed from manifest harvest.md` (new). | Ruff: clean. pytest: 13 pre-existing baseline failures verified orthogonal via `git stash` against worktree tip; Phase 3 + b1 introduce zero regressions. b1 fix reduced baseline failure count 14→13. | 13 baseline failures captured as test-suite-baseline-repair follow-up below. | toolkit: f3aaa90 (Phase 3 helper) + 8b154aa (b1 fix); aw: e012a387d (stipulation) |
| 4 — bundle_report_data() method + opt-in invocation contract | toolkit: `src/TRITON_SWMM_toolkit/bundle.py` (new); `src/TRITON_SWMM_toolkit/analysis.py` (added `bundle_report_data`); `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` (added `cfg_analysis` assignment + `bundle_report_data`); `src/TRITON_SWMM_toolkit/version_migration/constants.py` (added `BUNDLE_SCHEMA_VERSION = 1` — Phase 5 sequencing deviation). aw: phase 3 + phase 4 doc moves to `implemented/`. | Ruff: bundle.py clean; pre-existing baseline (22 errors) preserved on modified files via stash-test, zero new violations introduced. pytest: 13 pre-existing baseline failures (same set as Phase 3 baseline), zero regressions. Opt-in invariant grep: exit 1 (no matches). Smoke-emit against ~/Downloads/2026-05-02_bundle: 5.8 MB tar with 112 entries. | Phase 5 plan body adjusted in spirit — its "introduce BUNDLE_SCHEMA_VERSION" step is now "verify exists" (DoD assertion `BUNDLE_SCHEMA_VERSION == 1` still passes). | toolkit: b04b312 (Phase 4); aw: 179ce03ae (phase doc moves) |
| 5 — CLI commands + path-portability rewrite + BUNDLE_SCHEMA_VERSION | toolkit: `src/TRITON_SWMM_toolkit/cli.py` (added `bundle` and `report-from-bundle` Typer commands). aw: phase 5 doc to `implemented/`; phase 5 plan body amended (deferred steps 3a/4/5/7); phase 6 plan body amended (absorbed deferred validations + added `test_bundle_baseline_wrapper_section_matches`). | Ruff: bundle.py + constants.py clean; cli.py preserves pre-existing 17-error baseline (4 new B008 typer.Option default warnings match established Typer idiom in `run` / `cleanup-orphans`). `bundle --help` / `report-from-bundle --help`: exit 0. `BUNDLE_SCHEMA_VERSION == 1` assert: exit 0. | Round-trip smoke + Snakefile-leak grep + parse-only check + wrapper-divergence diff deferred to Phase 6 (synth-tier fixtures are local-resolvable; deferral elevates from one-shot to permanent CI regression). User-approved deferral after surfacing as halt trigger #2. | toolkit: 10b4361 (Phase 5 CLI); aw: e4e8586e1 (phase 5 closeout + phase 6 amendment) |

## Follow-up Ideas






%%
"#" + "followup" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.

Specialist findings routed here by /invoke-specialist Step 6 appear under:
  ###### Specialist-identified follow-up items
  ###### {specialist-name}
These are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.

Agents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.
%%


###### Specialist-identified follow-up items

###### triton-swmm-toolkit-specialist

- **Planning-instruction improvement proposal (from Phase 1 review)**: Empirical-schema errors (Flags 1, 7, 8) reveal that plan authors sometimes write pseudocode against a remembered API surface rather than the current source. Targeted protocol amendment: "before quoting an API in plan pseudocode, verify the API by reading the current source file and citing line numbers." Likely belongs as a `/plan-implementation` skill body addition rather than a project-specific rule.

###### software-engineering-specialist

- **Flag 9 — CLI binary naming inconsistency**: Codebase has two competing CLI binary names — documented `triton-swmm` (in CLAUDE.md Common Commands) and the actually-installed `TRITON_SWMM_toolkit` (per pyproject.toml [project.scripts]). Bundle plan correctly uses the installed name but exposes the pre-existing inconsistency: users following CLAUDE.md will type `triton-swmm bundle` and get "command not found." Resolve by either renaming the binary in pyproject.toml or updating CLAUDE.md, orthogonal to bundle workflow.
- **Flag 10 — Manifest-schema forward-compat narrative drift**: Phase 2 stipulation's "Forward Compatibility" clause references a `{"figure": {"sources": ...}}` → `{"figure": {"panels": ...}}` migration path, but the harvest helper actually reads `source_paths_relative` (top-level) and `artists[].channels[].ref.source_path` (nested) — neither shape matches. Documentation drift, not implementation defect; surfaces a gap for the panel-grouped provenance plan author to resolve before that plan starts.

###### Plan-review-surfaced follow-ups (Phase 6 baseline-repair plan, partitioned-cuddling-aho.md / Phase 5.5)

- **HTML-renderer manifest-sidecar emission (SE F-FU Flag 7)** — `errors_and_warnings.py` and `scenario_status_appendix.py` are pure HTML/text renderers that end with `output_path.write_text(html)` and emit no `*.manifest.json` sidecar. Phase 5.5's Cluster A.3+A.4 adds `with prov.artist(...)` AST blocks to satisfy provenance discipline tests but does not wire them through any emission helper. Follow-up: extend `emit_plot_with_sources` to accept HTML payloads, OR factor out a parallel `emit_html_with_sources` so HTML renderers emit manifest sidecars matching the matplotlib renderers' pattern. Would let the bundle harvest declare HTML renderer outputs uniformly.
- **`compilation_cpu_successful` property root-cause investigation (SE F-FU Flag 8 + TS toolkit specialist refinement)** — Phase 5.5's Cluster B session-scoped fixture `tritonswmm_cpu_compiled` is a workaround for `prepare_scenario`'s compile gate failing despite a valid on-disk build. The plan's stated root cause (fresh init = empty log = property false) may not be accurate: per `system.py:1285-1294`, the property already calls `retrieve_compilation_log("cpu")` which reads from disk. The actual cause may be a path-resolution bug in `retrieve_compilation_log` for the synth-fixture's cache layout, OR an interaction with the synth fixture's `_software_root`-vs-`system_directory` split. Fixture works regardless of root cause. Investigate whether the fixture is necessary or whether a path-resolution fix would obviate it. If a fix is found, the fixture can be removed.

###### Known high-risk hardcoded paths (user-flagged 2026-05-05)

- **`"scenario_status.csv"`** — hardcoded as a string literal in multiple places across the toolkit (e.g., `bundle.py::_copy_supporting_files`, `report_renderers/scenario_status_appendix.py`, `report_renderers/errors_and_warnings.py` via `validate_analysis()` chain, `_failing_fixture_helpers.py`). Not centralized in any single source-of-truth (no `analysis_paths.scenario_status_csv` attribute and no `version_migration/constants.py` constant). Each appearance is a drift risk. Promote to the centralized-constants architecture plan below; specifically Phase 2 (B2) of that plan should add `SCENARIO_STATUS_CSV: str = "scenario_status.csv"` to `version_migration/constants.py` and migrate all callers.

###### Centralized-constants architecture (user-driven, 2026-05-05; promotable to a master plan after PWI closes)

- **Goal**: collapse all hardcoded string literals across the toolkit codebase to a single (or very small number of) source-of-truth surfaces; all other references become class-object reads. Phased approach:
  - **Phase 1 — B1 (in-flight, this PWI's Phase 5.5)**: fix the 2 HIGH-priority literals (`"weather_timeseries.nc"` → `cfg_analysis.weather_timeseries`; `"_version.json"` → `VERSION_FILE_NAME`).
  - **Phase 2 — B2**: capture all MEDIUM-priority literals where a path attribute already exists (`"sims/{event_id}/swmm/hydraulics.inp"` → `scenario_paths.swmm_hydraulics_inp` after attribute-name verification; `"scenario_status.csv"` references → `analysis.analysis_paths.scenario_status_csv` if present).
  - **Phase 3 — B3**: refactor AnalysisPaths from dataclass to property-based class so leaf filenames (`"analysis_datatree.zarr"`, `"sensitivity_datatree.zarr"`) are config-derivable.
  - **Phase 4 — Snakemake template chunks**: centralize the Snakemake rule-shell-command templates currently embedded in `src/TRITON_SWMM_toolkit/workflow.py` (multiple multi-hundred-line embedded strings). Extracting these to a templates module would make `workflow.py` MASSIVELY more readable AND abstract patterns potentially reusable by other tool sets.
  - **Phase 5 — `.value` + `.definition` attribute pattern**: introduce a self-documenting-constant idiom — e.g., `class StatusFlag: e_consolidate_complete = SelfDocConstant(value="e_consolidate_complete", definition="Sentinel flag written by Snakemake's consolidate rule on successful per-mode flat-zarr emission.")`. Aligns with self-documenting-enum patterns in larger systems (Pydantic `Field(..., description=...)`, Python `enum.Enum` with docstrings per value). Novel for this codebase.
- **Recommended promotion**: `/plan-implementation` after this PWI closes; multi-phase plan structure aligned with the 5-phase rollout above.

###### Smell follow-ups

- **Porter silent-truncation defense (smell at line 2455+ of this scratch)** — agentic-workspace target. Four interventions documented inline as Verbatim Modification Specs: (1) porter exit-4 STRUCTURAL_VIOLATION check + content-floor warning; (2) body-plan-implementation atomic adds explicit nested-H1/H2 prohibition inside Phase H3; (3) invoke-specialist SKILL Plan-Specific Post-Completion adds pre-Step-5 fidelity gate; (4) porter `known_risks` metadata reflects new defense. Promote via `/plan-implementation` after this PWI session closes — atomic plan in agentic-workspace, title candidate: 'porter silent-truncation defense: nested-H1 detection + post-port fidelity gate'.

###### Test-suite baseline repair (Phase 3 QAQC finding)

- **13 pre-existing baseline failures in `pytest tests/ -m "not slow"`** (verified pre-Phase-3 via `git stash` against worktree tip `2ae3e19`). Phase 3 + b1 do not introduce any of these; b1 actually reduced the count from 14 → 13 (fixed `test_triton_only_cfg_generation`). Failure clusters:
  - **Provenance discipline (5 failures)** — `tests/test_provenance_discipline.py`: `per_sim_conduit_flow.py:252` and `:256` have `.plot(...)` calls outside `with prov.artist(...)` blocks (Phase 1 renderer-refactor defect, committed in `234681d`); `sensitivity_benchmarking.py`, `errors_and_warnings.py`, `scenario_status_appendix.py` lack a provenance block entirely (pre-Phase-1).
  - **TRITONSWMM CPU compile-gate (4 failures)** — `tests/test_synth_00_compile_models.py::test_create_dem_for_TRITON`, `::test_create_mannings_file_for_TRITON`, `tests/test_swmm_threads_implementation.py::test_swmm_threads_updated_in_inp_files`, `::test_swmm_threads_different_values`. All call `prepare_scenario` in coupled mode (`toggle_tritonswmm_model = True`) without first compiling TRITONSWMM CPU. Either fixture-side (auto-compile in setup) or test-side (explicit compile call) repair needed.
  - **Snakemake-driven multisim (4 failures)** — `tests/test_synth_04_multisim_with_snakemake.py`: `test_snakemake_workflow_end_to_end`, `test_run_and_render_report`, `test_render_report_idempotent`, `test_plot_sources_attribution`. Failures not investigated in detail (structurally unreachable from Phase 3 / b1 changes).
  - Promote as a separate atomic plan: title candidate "test suite baseline repair: provenance-discipline gaps + TRITONSWMM CPU compile fixture + synth_04 Snakemake regressions". Recommended scope: 3 commits (one per cluster), with stash-test evidence in each commit message.

%%
Agents: fill in this section as follows: Record the worktree path and branch.
%%

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- **Branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`
- **Isolation**: confirmed (sparse-checkout configured by `setup_pwi_worktree`; `library/docs/planning/` and `library/_scratch/` excluded — agentic-workspace planning/scratch reads use `$AGENTIC_WORKSPACE/...` absolute paths)
- **Plan frontmatter**: `plan_status: in_progress`, `worktree_branch: worktree-toolkit_05-03_1359_render-bundle-pwi` recorded by `setup_pwi_worktree`.

**Note**: Initial `EnterWorktree` call used a wrong slug (`05-03_1359_...`); that worktree was removed via `ExitWorktree(action: remove)` before any work landed. Canonical slug per the protocol's "strip first `_`-delimited segment from scratch doc stem" rule on `TRITON-SWMM_toolkit_05-03_1359_...` is `toolkit_05-03_1359_...` — agreeing with `setup_pwi_worktree`'s derivation.

##### Phases Remaining

*(section not present in main scratch at create time)*

## Worktree Status

%%
Agents: fill in this section as follows: Record the worktree path and branch.
%%

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- **Branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`
- **Isolation**: confirmed (sparse-checkout configured by `setup_pwi_worktree`; `library/docs/planning/` and `library/_scratch/` excluded — agentic-workspace planning/scratch reads use `$AGENTIC_WORKSPACE/...` absolute paths)
- **Plan frontmatter**: `plan_status: in_progress`, `worktree_branch: worktree-toolkit_05-03_1359_render-bundle-pwi` recorded by `setup_pwi_worktree`.

**Note**: Initial `EnterWorktree` call used a wrong slug (`05-03_1359_...`); that worktree was removed via `ExitWorktree(action: remove)` before any work landed. Canonical slug per the protocol's "strip first `_`-delimited segment from scratch doc stem" rule on `TRITON-SWMM_toolkit_05-03_1359_...` is `toolkit_05-03_1359_...` — agreeing with `setup_pwi_worktree`'s derivation.

### Phases Remaining


*(section not present in main scratch at create time)*

