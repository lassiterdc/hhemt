---
prompt_doc_type: sidecar_scratch
sidecar_kind: wrapup_handoff
main_scratch: /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md
last_sync_hashes:
  '### Phase Audit History': cf1deb6820280c692c4284d0e20e2adf5f5efd9096232b21b6a77382c85f88c3
  '# Follow-up Ideas': fb322574739f39ba3473309a8b039278646d8db87f88042d04c1c8cda5098100
harness: claude-code
plan_name: render_bundle
plan_completion_commit: c75f3e3e96ea57ba3ba5765f132b3ac29479f808
worktree_path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi
created: '2026-05-06T14:51:55'
custom_instructions:
  - "Since the sidecar files and planning documents are sufficient for a fresh agent to resume implementation, the workspace is moving towards a system of clearing instead of compacting at PWI heaviness gates. Updating instructional content is happening in a concurrent session, but we are applying clear instead of compact here even though the instructions say compact, and we wanted you to be aware to avoid any confusion about your state after being cleared and resuming implementation."
smoke_test: false
---
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

## Session Custom Instructions

These instructions are specific to this session and were supplied by the user (or agent-recommended and user-approved) at a prior compaction gate. They ride along with every sidecar until session close and must be honored in addition to — not instead of — the standard mandatory reads and the composed atomic content below.

- Since the sidecar files and planning documents are sufficient for a fresh agent to resume implementation, the workspace is moving towards a system of clearing instead of compacting at PWI heaviness gates. Updating instructional content is happening in a concurrent session, but we are applying clear instead of compact here even though the instructions say compact, and we wanted you to be aware to avoid any confusion about your state after being cleared and resuming implementation.

## Wrapup Entry Point

- Kind: wrapup_handoff
- Plan name: render_bundle
- Plan completion commit: `c75f3e3e96ea57ba3ba5765f132b3ac29479f808`
- Worktree: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- Worktree branch: `worktree-toolkit_05-03_1359_render-bundle-pwi`
- Task: `/wrapup-session` in full

## Post-Compaction Wrapup — Preamble

You are resuming a multiphase plan implementation after an in-session /compact
that was fired at the end of the plan — after the final phase committed and
after Plan Completion (plan closeout protocol) ran in the pre-compaction
session. Your single task is `/wrapup-session` in full. Do NOT re-enter the
phase loop. Do NOT re-run Plan Completion. It is already done.

The first message you receive after compaction looks structurally like a fresh
session invocation, and RLHF training creates a pull to treat it as one —
running preflight, re-reading the full session scratch doc, re-dispatching
specialist reviews. Recognize this pull. Your actual task is to rehydrate from
the wrapup-handoff sidecar, verify sync integrity via the script, run the
First Action GO gate, and then invoke `/wrapup-session` directly.

### Mandatory reads (in order, each in full)

1. Wrapup-handoff sidecar:
   `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi/sidecars/sidecar_wrapup_2026-05-06_1451.md`

2. Wrapup-session skill body (the skill you are about to execute):
   [wrapup session SKILL](../../instructions/skills/wrapup-session/wrapup%20session%20SKILL.md) #inject_path

%% inject-path-start: wrapup-session-skill %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`
    - Session wrap-up checklist — audits for uncommitted changes, outstanding todos, unresolved decisions, and next steps before exiting
%% inject-path-end: wrapup-session-skill %%

### Wrapup entry point

- **Plan name**: `render_bundle`
- **Plan completion commit**: `c75f3e3e96ea57ba3ba5765f132b3ac29479f808`
- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- **Worktree branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`
- **Task**: run `/wrapup-session` in full, per the skill body above.

Plan Completion already moved all phase docs to `implemented/` and the master
plan subdirectory to `completed/`. Regeneration of planning tables and the
plan closeout commit are in git history at `c75f3e3e96ea57ba3ba5765f132b3ac29479f808`. If you
find yourself tempted to re-run `$AGENTIC_WORKSPACE/scripts/generate/complete_plan.py` or touch the planning
tree, stop — that work is done.

Do NOT read the main session scratch doc yet. Main scratch is intentionally
out of scope for the First Action gate. You will read it at wrapup-session
Step 6 — see the "Step 6 injection-then-reload contract" below.

### Step 6 injection-then-reload contract (read this carefully)

When `/wrapup-session` reaches Step 6 (Write Report to Session Scratch Doc),
the skill runs `create_from_template --compose wrapup-session
--inject-into-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md` to append the wrapup template
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

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi`
- **Worktree branch**: `worktree-toolkit_05-03_1359_render-bundle-pwi`
- **Main scratch path**: `/home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md`

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

1. The sidecar frontmatter's `sidecar_kind` field is exactly `wrapup_handoff`.
2. The sidecar body contains a `## Wrapup Entry Point` section.
3. The sidecar's `## Worktree Status` section's branch line matches the
   sidecar frontmatter's `worktree_path` field's derivable branch name
   (`worktree-{slug}`).
4. The sidecar's `last_sync_hashes:` frontmatter contains exactly two entries:
   `### Phase Audit History` and `# Follow-up Ideas`.

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

`Wrapup-entry assertion: Sidecar: {sidecar_filename}. sidecar_kind: wrapup_handoff. Plan name: render_bundle. Plan completion commit: c75f3e3e96ea57ba3ba5765f132b3ac29479f808. Wrapup entry block present in body: YES|NO.`

If any of these do not match the sidecar's actual state, or if the body does
not contain `## Wrapup Entry Point`, HALT per the post-halt response protocol
and surface the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke:

    scripts/generate/write_compaction_sidecar.py --mode verify \
      --main-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-03_1359_render-bundle-pwi.md \
      --worktree /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-03_1359_render-bundle-pwi

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

### Phase Audit History








| Phase | Phase-Local Diff Files | Audit-on-Diff Violations | Resolution | Commit |
|-------|------------------------|--------------------------|------------|--------|
| 1 — renderer refactor: eliminate per-sim file dependencies | `src/TRITON_SWMM_toolkit/report_renderers/{per_sim_peak_flood_depth,per_sim_conduit_flow,_hydrology_panel,per_analysis_summary}.py` | None applicable (agentic-workspace `scripts.audit` audits prompt docs, not toolkit Python code). Ruff: zero new violations introduced (10 pre-existing preserved). | N/A — no toolkit-side audit framework. Ruff baseline preserved. | 234681d (toolkit) |
| 2 — bundle spec discovery via curated example | aw: `library/docs/decisions/TRITON-SWMM_toolkit/bundle layout and contents.md` (new); plan-closeout move of `2 bundle spec...md` to `implemented/`. No toolkit code touched. | N/A (markdown only; no audit framework targets decision docs). | N/A. | 08e6099a6 (aw) |
| 3 — manifest-harvest infrastructure + renderer audit + stipulation | toolkit: `src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py` (added `harvest_source_paths`); `src/TRITON_SWMM_toolkit/scenario.py` (orthogonal b1 fix to TRITON-only compile gate). aw: `library/docs/stipulations/TRITON-SWMM_toolkit/bundle file set is computed from manifest harvest.md` (new). | Ruff: clean. pytest: 13 pre-existing baseline failures verified orthogonal via `git stash` against worktree tip; Phase 3 + b1 introduce zero regressions. b1 fix reduced baseline failure count 14→13. | 13 baseline failures captured as test-suite-baseline-repair follow-up below. | toolkit: f3aaa90 (Phase 3 helper) + 8b154aa (b1 fix); aw: e012a387d (stipulation) |
| 4 — bundle_report_data() method + opt-in invocation contract | toolkit: `src/TRITON_SWMM_toolkit/bundle.py` (new); `src/TRITON_SWMM_toolkit/analysis.py` (added `bundle_report_data`); `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` (added `cfg_analysis` assignment + `bundle_report_data`); `src/TRITON_SWMM_toolkit/version_migration/constants.py` (added `BUNDLE_SCHEMA_VERSION = 1` — Phase 5 sequencing deviation). aw: phase 3 + phase 4 doc moves to `implemented/`. | Ruff: bundle.py clean; pre-existing baseline (22 errors) preserved on modified files via stash-test, zero new violations introduced. pytest: 13 pre-existing baseline failures (same set as Phase 3 baseline), zero regressions. Opt-in invariant grep: exit 1 (no matches). Smoke-emit against ~/Downloads/2026-05-02_bundle: 5.8 MB tar with 112 entries. | Phase 5 plan body adjusted in spirit — its "introduce BUNDLE_SCHEMA_VERSION" step is now "verify exists" (DoD assertion `BUNDLE_SCHEMA_VERSION == 1` still passes). | toolkit: b04b312 (Phase 4); aw: 179ce03ae (phase doc moves) |
| 5 — CLI commands + path-portability rewrite + BUNDLE_SCHEMA_VERSION | toolkit: `src/TRITON_SWMM_toolkit/cli.py` (added `bundle` and `report-from-bundle` Typer commands). aw: phase 5 doc to `implemented/`; phase 5 plan body amended (deferred steps 3a/4/5/7); phase 6 plan body amended (absorbed deferred validations + added `test_bundle_baseline_wrapper_section_matches`). | Ruff: bundle.py + constants.py clean; cli.py preserves pre-existing 17-error baseline (4 new B008 typer.Option default warnings match established Typer idiom in `run` / `cleanup-orphans`). `bundle --help` / `report-from-bundle --help`: exit 0. `BUNDLE_SCHEMA_VERSION == 1` assert: exit 0. | Round-trip smoke + Snakefile-leak grep + parse-only check + wrapper-divergence diff deferred to Phase 6 (synth-tier fixtures are local-resolvable; deferral elevates from one-shot to permanent CI regression). User-approved deferral after surfacing as halt trigger #2. | toolkit: 10b4361 (Phase 5 CLI); aw: e4e8586e1 (phase 5 closeout + phase 6 amendment) |
| 5.5 — baseline failure remediation (Cluster A/B/C/D + Spec E/F) + preemptive Phase 6 test file | toolkit: provenance-discipline fixes across 4 renderers (Cluster A); session-scoped `tritonswmm_cpu_compiled` fixture (Cluster B); `consolidate_to_datatree()` Option B refactor + flat-zarr chain removal + `--consolidate-datatree` flag drop (Cluster C); `synth_sensitivity` fixture report_config write (Cluster D); log-based existence-check audit (Spec E); `bundle.py` `VERSION_FILE_NAME` substitution (Spec F); `tests/test_synth_08_bundle_round_trip.py` (preemptively added per Phase 6 spec — filename `08` due to pre-existing `test_synth_07_validation_report.py`). aw: 5.5 plan body. | All 14 pre-existing baseline failures cleared (13 prior + 1 newly-surfaced sensitivity). Stop-gate Tier 1 + Tier 2 cleared with documented user-approved deviations: `cli.py` os.chdir, Option B analysis_dir override, sub-scope harvest detection. | Deviations recorded as user-approved in commit message + plan body; no follow-up debt. | toolkit: c75f3e3 (Phase 5.5 + bundle directory-model fix); aw: 8fd296bcb (5.5 closeout chore) |
| 6 — end-to-end smoke test + architecture doc update + refresh_injections | toolkit: zero source/test changes (test file landed early in Phase 5.5). aw: `library/prompts/workspaces/projects/TRITON-SWMM_toolkit/TRITON SWMM toolkit architecture.md` (added `## Local Renderer Iteration via Bundles` H2 section before `## Cross-Domain References` per phase Bypass spec). | Audit-on-diff stash-test on arch doc: 2 BARE_BACKTICK_MARKDOWN_PATH errors pre-stash, 2 post-stash → zero new violations introduced (pre-existing on line 171, orthogonal to bundle additions). pytest `test_synth_08_bundle_round_trip.py`: 5/5 passed in 896s. `refresh_injections` exit 0. `scripts.audit` exit 0. Ruff: zero toolkit-side changes → baseline trivially preserved. | N/A — no new violations, all DoD items met. | toolkit: (no-op); aw: b51199bd1 (Phase 6 + closeout) |

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

###### Phases Remaining

*(section not present in main scratch at create time)*

###### Wrapup halt — 2026-05-06 (post-clear resume)

- **Halt trigger**: #3 — Sync-back integrity failure (verify exit code 3, worktree dirty).
- **Detected at**: First Action Part 3 (`write_compaction_sidecar.py --mode verify`), before invoking `/wrapup-session`.
- **Evidence**: `git -C <worktree> diff -- sidecars/sidecar_phase5.5_2026-05-05_1135.md` shows a 438-line deletion (file removed, not staged). An untracked copy exists at `sidecars/done/sidecar_phase5.5_2026-05-05_1135.md` and a second `sidecars/done/sidecar_phase6_2026-05-06_1336.md`. Pattern is consistent with `create-wrapup-handoff` archiving prior in-flight sidecars to `done/` without staging the corresponding delete + rename.
- **Other untracked files** (advisory, not blocking per verify): `_version.json`, `system_log.json`, `sidecars/.main_backup_2026-05-06_133648.md`, `sidecars/.main_backup_2026-05-06_145155.md`, `sidecars/sidecar_wrapup_2026-05-06_1451.md` (this sidecar itself).
- **Awaiting user direction.** Per protocol, did not attempt recovery — `git restore`, `git mv`, or `git add` the deleted/renamed sidecar pair would all be unauthorized destructive/state-modifying actions on what may be cross-session bookkeeping.

## Worktree Status

*(section not present in main scratch)*

