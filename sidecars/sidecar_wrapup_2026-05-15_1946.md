---
prompt_doc_type: sidecar_scratch
sidecar_kind: wrapup_handoff
main_scratch: /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-11_1503_bundle-portable-report-regen-pwi.md
last_sync_hashes:
  '### Phase Audit History': 73f4a40adff4d8806dea088b193bc68db34272de5a4159f6eec4ca5d43ffb6e5
  '# Follow-up Ideas': cf1851461002c404ee112e7d509197fc602c53937c3d4ba6e3bec7a84c237c07
harness: claude-code
plan_name: bundle_portable_report_regeneration
plan_completion_commit: 0463143ff61db6dbd3ae7f2802ab7de4f84808d7
worktree_path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi
created: '2026-05-15T19:46:01'
custom_instructions:
  - "PHASE 6 STATUS — Resuming at Step 7 (acceptance) with figure-quality investigation. The MECHANICAL pipeline empirically works end-to-end: Rivanna cfg -> bundle emit -> Globus transfer -> unzip+move (R11 proof) -> regenerate_report() -> analysis_report.html. But the user opened the regenerated HTML and identified SERIOUS PROBLEMS with the plotted figures; requested a fresh session for figure-quality investigation. Artifacts produced this session: (a) ~/scratch_bundles/uva_postF2_v2_smoketest/analysis_report.html (184M) -- the regenerated report with the figure issues; (b) /home/***REMOVED***/Downloads/2026-05-15_uva_sensitivity_suite_bundle/render_bundle_uva_sensitivity_suite_postF2_v2.zip -- the v2 bundle on local disk; (c) the v1 buggy bundle ~/Downloads/2026-05-14_uva_sensitivity_suite_bundle/ + the rename-bandaided ~/scratch_bundles/uva_postF2_smoketest/ -- keep for comparison. The new session should: (1) read this sidecar in full; (2) read the Phase 6 phase doc Freshness section; (3) open analysis_report.html in a browser AND inspect specific renderers' rendered output; (4) categorize issues per renderer (system_overview, sensitivity_benchmarking, per_sim_peak_flood_depth, per_sim_conduit_flow, per_analysis_summary, scenario_status_appendix, errors_and_warnings); (5) determine whether issues are renderer-side bugs (output content wrong), bundle-side bugs (consume path misrendering correct content), or substrate issues (analysis data quality)."
  - "OUTSTANDING BANDAIDS to be retired by future work -- do NOT proceed assuming they hold across re-instantiation: (a) Rivanna's $ANALYSIS_DIR/cfg_analysis.yaml::report.sensitivity is sed-patched in place; ANY re-instantiation of CaseStudyBuilder via cat.UVACaseStudies.benchmarking_norfolk_irene(...) overwrites the patch with sensitivity: null. (b) The plot_sensitivity_benchmarking output was re-rendered via direct renderer CLI invocation (not via analysis.run()) because the in-memory monkey-patch did not reach the snakemake subprocess. (c) Both bandaids exist because of the durable bug captured in the promoted plan at library/docs/planning/projects/TRITON-SWMM_toolkit/features/sensitivity_block_population_in_cfg_write_sites/ (master + 2 phases). That plan was committed locally as 1be327c13 on agentic-workspace but NOT pushed at sidecar-emit time. The plan is the canonical fix path; implement via /proceed-with-implementation when figure-quality work is complete."
  - "OPEN COMMITS + PUSH APPROVALS at sidecar-emit time: (1) agentic-workspace local commit 1be327c13 (plan promote: sensitivity-block-population-in-cfg-write-sites + 2 phases) -- committed locally, NOT pushed to remote. (2) record-process SKILL body has uncommitted local edits applying three INT-1s from this session's smell investigations: side-effect classification tag taxonomy with (informational/diagnostic/generative/mutating/destructive/network) tags, ISO datetime (YYYY-MM-DD HH:MM) timestamp promotion, and Timestamp tripwire verbatim-assertion gate. Audit was clean at edit time. Both should land before any new specialist invocation depends on the post-merge skill body shape."
  - 'FOLLOW-UP IDEAS accumulated -- route via /capture-idea at /wrapup-session in the resuming session: (1) port_h1_sections.py script + /cleanup-scratch SKILL Step 4 dispatch rewrite; (2) V0005 + CaseStudyBuilder Optional[dict] field drop -- DURABLE FIX HAS PROMOTED PLAN (sensitivity-block-population-in-cfg-write-sites); (3) Snakemake tmux-mode swallowing per-rule failures and returning success=True to orchestrator despite missing outputs; (4) cross-repo write-block hook gap (decisions/knowledge/prompts) -- has smell entry with intervention specs; (5) project-wide Plotly-template audit; (6) kaleido failure-mode unit test; (7) sidecar-machinery dirty-worktree pattern at --mode create time. Full bullets at scratch # Follow-up Ideas. Strongly recommend /wrapup-session as the first action of the resuming session to triage these before going deeper into figure-quality work.'
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

## Session Custom Instructions

These instructions are specific to this session and were supplied by the user (or agent-recommended and user-approved) at a prior compaction gate. They ride along with every sidecar until session close and must be honored in addition to — not instead of — the standard mandatory reads and the composed atomic content below.

- PHASE 6 STATUS — Resuming at Step 7 (acceptance) with figure-quality investigation. The MECHANICAL pipeline empirically works end-to-end: Rivanna cfg -> bundle emit -> Globus transfer -> unzip+move (R11 proof) -> regenerate_report() -> analysis_report.html. But the user opened the regenerated HTML and identified SERIOUS PROBLEMS with the plotted figures; requested a fresh session for figure-quality investigation. Artifacts produced this session: (a) ~/scratch_bundles/uva_postF2_v2_smoketest/analysis_report.html (184M) -- the regenerated report with the figure issues; (b) /home/***REMOVED***/Downloads/2026-05-15_uva_sensitivity_suite_bundle/render_bundle_uva_sensitivity_suite_postF2_v2.zip -- the v2 bundle on local disk; (c) the v1 buggy bundle ~/Downloads/2026-05-14_uva_sensitivity_suite_bundle/ + the rename-bandaided ~/scratch_bundles/uva_postF2_smoketest/ -- keep for comparison. The new session should: (1) read this sidecar in full; (2) read the Phase 6 phase doc Freshness section; (3) open analysis_report.html in a browser AND inspect specific renderers' rendered output; (4) categorize issues per renderer (system_overview, sensitivity_benchmarking, per_sim_peak_flood_depth, per_sim_conduit_flow, per_analysis_summary, scenario_status_appendix, errors_and_warnings); (5) determine whether issues are renderer-side bugs (output content wrong), bundle-side bugs (consume path misrendering correct content), or substrate issues (analysis data quality).
- OUTSTANDING BANDAIDS to be retired by future work -- do NOT proceed assuming they hold across re-instantiation: (a) Rivanna's $ANALYSIS_DIR/cfg_analysis.yaml::report.sensitivity is sed-patched in place; ANY re-instantiation of CaseStudyBuilder via cat.UVACaseStudies.benchmarking_norfolk_irene(...) overwrites the patch with sensitivity: null. (b) The plot_sensitivity_benchmarking output was re-rendered via direct renderer CLI invocation (not via analysis.run()) because the in-memory monkey-patch did not reach the snakemake subprocess. (c) Both bandaids exist because of the durable bug captured in the promoted plan at library/docs/planning/projects/TRITON-SWMM_toolkit/features/sensitivity_block_population_in_cfg_write_sites/ (master + 2 phases). That plan was committed locally as 1be327c13 on agentic-workspace but NOT pushed at sidecar-emit time. The plan is the canonical fix path; implement via /proceed-with-implementation when figure-quality work is complete.
- OPEN COMMITS + PUSH APPROVALS at sidecar-emit time: (1) agentic-workspace local commit 1be327c13 (plan promote: sensitivity-block-population-in-cfg-write-sites + 2 phases) -- committed locally, NOT pushed to remote. (2) record-process SKILL body has uncommitted local edits applying three INT-1s from this session's smell investigations: side-effect classification tag taxonomy with (informational/diagnostic/generative/mutating/destructive/network) tags, ISO datetime (YYYY-MM-DD HH:MM) timestamp promotion, and Timestamp tripwire verbatim-assertion gate. Audit was clean at edit time. Both should land before any new specialist invocation depends on the post-merge skill body shape.
- FOLLOW-UP IDEAS accumulated -- route via /capture-idea at /wrapup-session in the resuming session: (1) port_h1_sections.py script + /cleanup-scratch SKILL Step 4 dispatch rewrite; (2) V0005 + CaseStudyBuilder Optional[dict] field drop -- DURABLE FIX HAS PROMOTED PLAN (sensitivity-block-population-in-cfg-write-sites); (3) Snakemake tmux-mode swallowing per-rule failures and returning success=True to orchestrator despite missing outputs; (4) cross-repo write-block hook gap (decisions/knowledge/prompts) -- has smell entry with intervention specs; (5) project-wide Plotly-template audit; (6) kaleido failure-mode unit test; (7) sidecar-machinery dirty-worktree pattern at --mode create time. Full bullets at scratch # Follow-up Ideas. Strongly recommend /wrapup-session as the first action of the resuming session to triage these before going deeper into figure-quality work.

## Wrapup Entry Point

- Kind: wrapup_handoff
- Plan name: bundle_portable_report_regeneration
- Plan completion commit: `0463143ff61db6dbd3ae7f2802ab7de4f84808d7`
- Worktree: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi`
- Worktree branch: `worktree-toolkit_05-11_1503_bundle-portable-report-regen-pwi`
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
   `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi/sidecars/sidecar_wrapup_2026-05-15_1946.md`

2. Wrapup-session skill body (the skill you are about to execute):
   [wrapup session SKILL](../../instructions/skills/wrapup-session/wrapup%20session%20SKILL.md) #inject_path

%% inject-path-start: wrapup-session-skill %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`
    - Session wrap-up checklist — audits for uncommitted changes, outstanding todos, unresolved decisions, and next steps before exiting
%% inject-path-end: wrapup-session-skill %%

### Wrapup entry point

- **Plan name**: `bundle_portable_report_regeneration`
- **Plan completion commit**: `0463143ff61db6dbd3ae7f2802ab7de4f84808d7`
- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi`
- **Worktree branch**: `worktree-toolkit_05-11_1503_bundle-portable-report-regen-pwi`
- **Task**: run `/wrapup-session` in full, per the skill body above.

Plan Completion already moved all phase docs to `implemented/` and the master
plan subdirectory to `completed/`. Regeneration of planning tables and the
plan closeout commit are in git history at `0463143ff61db6dbd3ae7f2802ab7de4f84808d7`. If you
find yourself tempted to re-run `$AGENTIC_WORKSPACE/scripts/generate/complete_plan.py` or touch the planning
tree, stop — that work is done.

Do NOT read the main session scratch doc yet. Main scratch is intentionally
out of scope for the First Action gate. You will read it at wrapup-session
Step 6 — see the "Step 6 injection-then-reload contract" below.

### Step 6 injection-then-reload contract (read this carefully)

When `/wrapup-session` reaches Step 6 (Write Report to Session Scratch Doc),
the skill runs `create_from_template --compose wrapup-session
--inject-into-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-11_1503_bundle-portable-report-regen-pwi.md` to append the wrapup template
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

- **Worktree path**: `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi`
- **Worktree branch**: `worktree-toolkit_05-11_1503_bundle-portable-report-regen-pwi`
- **Main scratch path**: `/home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-11_1503_bundle-portable-report-regen-pwi.md`

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

`Wrapup-entry assertion: Sidecar: {sidecar_filename}. sidecar_kind: wrapup_handoff. Plan name: bundle_portable_report_regeneration. Plan completion commit: 0463143ff61db6dbd3ae7f2802ab7de4f84808d7. Wrapup entry block present in body: YES|NO.`

If any of these do not match the sidecar's actual state, or if the body does
not contain `## Wrapup Entry Point`, HALT per the post-halt response protocol
and surface the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke:

    scripts/generate/write_compaction_sidecar.py --mode verify \
      --main-scratch /home/***REMOVED***/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-11_1503_bundle-portable-report-regen-pwi.md \
      --worktree /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi

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



%%

<!-- TPL-INSTRUCTION: Populated after each phase completes. Append one row per phase to the table below. The QAQC audit-on-diff gate uses phase-local diff (files changed since previous phase commit), not branch-cumulative. -->

| Phase | Phase-Local Diff Files | Audit-on-Diff Violations | Resolution |
|-------|------------------------|--------------------------|------------|
| 1 | `src/TRITON_SWMM_toolkit/bundle/__init__.py`, `src/TRITON_SWMM_toolkit/bundle/_emit.py` (renamed from `bundle.py`), `src/TRITON_SWMM_toolkit/bundle/_path_policy.py`, `src/TRITON_SWMM_toolkit/bundle/_protocol.py`, `src/TRITON_SWMM_toolkit/analysis.py`, `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`, `tests/test_bundle.py`, `tests/fixtures/bundles/build_fixtures.py`, `tests/fixtures/bundles/multi_sim/*`, `tests/fixtures/bundles/sensitivity_master/*` | 0 | clean — `scripts.audit --diff-only --scope file_local,subset` exited 0 |
| 2 | `src/TRITON_SWMM_toolkit/workflow.py` (VMS-1/2 lift, VMS-4 dataclasses, VMS-5 helpers, VMS-6 plot-rule wrapper refactor), `src/TRITON_SWMM_toolkit/bundle/__init__.py` (VMS-8 regenerate_report body, VMS-11 _read_static_backend stub), `src/TRITON_SWMM_toolkit/bundle/_emit.py` (VMS-7 Snakefile.source rename), `src/TRITON_SWMM_toolkit/bundle/snakefile_generator.py` (VMS-9 new file), `src/TRITON_SWMM_toolkit/analysis.py` + `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` (VMS-1 fan-out), `tests/test_bundle_snakefile_generator.py` (VMS-10), `tests/test_workflow_snakefile_byte_identity.py` (Validation Plan item 4), `tests/fixtures/golden_snakefiles/{multi_sim,sensitivity_master}.Snakefile.golden` (pre-refactor byte-identity goldens) | 0 (project-side) | clean — `scripts.audit` is agentic-workspace-scoped; toolkit project audit not in scope. Manual smoke (`pytest`, byte-identity, `snakemake -n`) green. See QAQC scratch entry 2026-05-12 21:21. |
| 3 | `src/TRITON_SWMM_toolkit/bundle/__init__.py` (VMS-1/2/3 — subprocess wiring + `_zip_html`, real `_read_static_backend`, `from_directory` schema-version + invariants check; module-level import of `run_subprocess_with_tee` per Option A resolution to VMS-1/VMS-9 spec contradiction), `src/TRITON_SWMM_toolkit/bundle/_emit.py` (VMS-8 — `_write_bundle_manifest` `bundle_root_invariants` kwarg, `_copy_configs_with_relative_paths` return type → dict, `emit_bundle` callsite capture), `src/TRITON_SWMM_toolkit/config/report.py` (VMS-4 — `static_backend` field), `src/TRITON_SWMM_toolkit/validation.py` (VMS-6 — `_check_static_backend_kaleido_available`), `src/TRITON_SWMM_toolkit/cli.py` (VMS-7 — `report-from-bundle` thin Bundle wrapper, `os.chdir` deleted), `pyproject.toml` (VMS-5 — `viz-export` optional extra), `tests/test_bundle.py` (VMS-9 — 7 new tests; removed obsolete Phase 1 `test_regenerate_report_stub_raises` since VMS-1 replaces the stub), `tests/fixtures/bundles/build_fixtures.py` + `tests/fixtures/bundles/{multi_sim,sensitivity_master}/bundle_manifest.json` (fixture refresh to per-cfg invariants shape) | 0 (project-side) | clean — `scripts.audit` is agentic-workspace-scoped; toolkit project audit not in scope. `pytest tests/test_bundle.py -v` → 15 passed in 0.15s. Resolved one spec gap mid-implementation (VMS-1 local import vs VMS-9 binding-site patch — applied Option A: module-level import). |
| 4 | `src/TRITON_SWMM_toolkit/bundle/_emit.py` (VMS-1/2/3 — `_emit_bundle_tar` → `_emit_bundle_zip` with deterministic fixed-mtime `(1980, 1, 1, 0, 0, 0)` + sorted rglob + `ZIP_STORED`; `import tarfile` → `import zipfile`; output-path default `.tar` → `.zip`; callsite update; module-docstring helper-list refresh), `src/TRITON_SWMM_toolkit/cli.py` (VMS-6 — `report_from_bundle_command` tar-unpack branch swapped to zip-unpack via `zipfile.ZipFile.extractall`; `import tarfile` → `import zipfile`; help/error text, bundle-emit command help/docstring all reference zip), `tests/test_bundle.py` (VMS-7 — `test_zip_determinism` SHA-256 byte-identity check + `test_zip_emit_no_tar_artifact` inspect-source guard), `library/docs/decisions/TRITON-SWMM_toolkit/render bundle is a zip archive.md` (VMS-4 — new decision doc via `/manage-decision`; `superseded_by` → not set; supersedes the prior layout doc's tar-emit subdecision), `library/docs/decisions/TRITON-SWMM_toolkit/bundle layout and contents.md` (VMS-5 — `superseded_by:` set; `decision`/`description` rephrased to layout-only scope; `flashcard_last_validate` + `anticipated_challenges` added; tar.gz alternative bullet removed; body lines for tar→zip emit + supersession H2 added; cross-ref converted to dual-syntax markdown link + `$AGENTIC_WORKSPACE/` backtick path) | 0 (decision-doc audit) | clean — `scripts.audit --files {render bundle is a zip archive.md} {bundle layout and contents.md}` → 0 errors, 0 warnings, 2 clean. `pytest tests/test_bundle.py -v` → 17 passed in 0.24s. DoD greps clean: `grep -rn _emit_bundle_tar src/` → 0 matches; `grep -rn tarfile src/TRITON_SWMM_toolkit/bundle/_emit.py` → 0 matches. Risk #3 ordering held: VMS-4 applied before VMS-5. Cross-tree commit: code on worktree branch, decision docs on main. Mid-phase recovery: parallel main-tree session's merge/reset operations clobbered in-flight unstaged decision-doc edits twice; second-pass re-application persisted after the user signaled the parallel work had landed. |
| 5 | `src/TRITON_SWMM_toolkit/report_renderers/_static_backend_warning.py` (VMS-1 — new one-time-warning helper), `system_overview.py` (VMS-2 — dispatch swap; existing kaleido SVG emit preserved), `per_sim_peak_flood_depth.py` / `per_sim_conduit_flow.py` / `sensitivity_benchmarking.py` (VMS-3/4/5 — dispatch swap + kaleido SVG emit), `per_analysis_summary.py` / `errors_and_warnings.py` / `scenario_status_appendix.py` (VMS-6 — `warn_no_plotly_branch` calls), `tests/test_static_backend_dispatch.py` (VMS-7 — 4 dispatch tests), `src/TRITON_SWMM_toolkit/workflow.py` (VMS-8 — new `_get_report_cfg_static_backend()` helper + 7 `_make_rule_emission_context` callsite swaps; phase doc's `self.analysis.cfg_analysis.cfg_report.static_backend` spec corrected mid-implementation since `cfg_report` is not a child of `analysis_config`), `tests/fixtures/golden_snakefiles/multi_sim.Snakefile.golden` + `sensitivity_master.Snakefile.golden` (re-baselined to reflect plotly cfg-default per Decision 4), `tests/test_workflow_snakefile_byte_identity.py` (docstring updated to Phase 5 basis) | 0 (project-side) | clean — `pytest test_static_backend_dispatch.py test_bundle.py test_bundle_snakefile_generator.py test_workflow_snakefile_byte_identity.py` → 38 passed; `test_synth_07_validation_report.py` → 12 passed. DoD greps: `static_backend="matplotlib"` in workflow.py → 0; `fig.write_image` in renderers → 4; `warn_no_plotly_branch` def+calls → 4. Mid-phase recovery: phase doc's VMS-8 spec referenced a non-existent attribute path (`cfg_analysis.cfg_report`) AND falsely claimed `generate_snakefile_content` accepts `static_backend` kwarg; user approved Option A two-part fix (helper method + golden re-baseline) per the never-proceed-past rule. Goldens were re-baselined in-process via pytest invocation to match the test fixture's `sim_folder`-populated state. Committed at `db1d23b`. |
| 6 | `src/TRITON_SWMM_toolkit/report_renderers/_react_surgery.py` (kwarg renamed `hide_workflow_statistics`→`hide_statistics`→`bundle_mode`; surgery re-shuffled — added unconditional `this.content = "metadata"` initial-view forcing and `bundle_mode`-gated drops for Workflow + Statistics + "General" `ListHeading`; removed the briefly-introduced Workflow→Overview rename per user feedback that the rename pointed at the wrong content view), `src/TRITON_SWMM_toolkit/bundle/__init__.py` (FIX-1a `--report-stylesheet` flag, FIX-1b surgery call with `bundle_mode=True`, FIX-5 zip-mode switched from `_zip_html` wrap-style to snakemake-native multi-file `--report path.zip` via `apply_post_process_surgery_to_zip`, A2 `format` default flipped `html`→`zip`), `src/TRITON_SWMM_toolkit/cli.py` (A2 `report-from-bundle --format` default flipped `html`→`zip`; help text refreshed) | 0 (project-side) | clean — `scripts.audit` is agentic-workspace-scoped; toolkit project audit not in scope (same as rows 2 / 3 / 5). `pytest tests/test_bundle.py tests/test_static_backend_dispatch.py -q` → 24 passed after every fix-round edit; final pass at 17:34 post-FIX-3c+FIX-5. Phase doc claimed "no code changes" — implementation deviated because the smoke test surfaced real bundle-side defects (branding regression, useless menu panels, default-view, wrap-style zip shape) which had to ship in-phase under the never-proceed-past rule. User signed off on visual at 17:34 ("looks great; i think this smoke test is completed"). Mid-implementation friction: worktree-vs-main install gap caused first regenerate to run pre-edit code (resolved via `sys.path.insert(0, $WORKTREE_SRC)` for `python -c` invocations — routed as follow-up); FIX-2a's initial `apply_post_process_surgery_to_zip` was being called on an HTML path (silently swallowed by try/except), fixed by collapsing surgery to a single HTML-path branch before realizing FIX-5 wanted native multi-file zip anyway. |

%%

## Follow-up Ideas



%%

"#" + "followup" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.

Specialist findings routed here by /invoke-specialist Step 6 appear under:

  ###### Specialist-identified follow-up items
  ###### {specialist-name}

These are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.

Agents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.

%%

- Recurring sidecar-machinery dirty-worktree pattern (observed across multiple sessions): `write_compaction_sidecar.py --mode create` archives the prior wrapup sidecar to `sidecars/done/` and emits a new in-flight phase sidecar, but does **not** commit those filesystem moves. The next `/resume-sidecar` → `--mode verify` therefore halts with exit 3 (worktree dirty) until the user manually authorizes a `chore: sidecar archive + Phase 1 in-flight emission` commit. Durable fix candidates worth weighing in a real planning doc:
    - (a) have `write_compaction_sidecar.py` perform an atomic `git add + git commit` for the sidecar archive + new sidecar at emission time, with a deterministic chore message;
    - (b) add a `--mode verify` allowance / exemption for paths under `sidecars/` so resume can proceed without a manual commit step;
    - (c) add a `/resume-sidecar` preflight that detects this exact sidecar-only dirty pattern and auto-commits before invoking verify.

  Should be captured to a permanent idea file via `/capture-idea` at wrap-up.

- Create-time recovery substrate gap in `write_compaction_sidecar.py`: `.main_backup_*.md` files are written only inside `_flush_writable_sections` (called at sync-back time), **never at `--mode create` time**. A freshly-created sidecar that hits a hash mismatch at the first `--mode verify` therefore has no recovery substrate keyed to its own `last_sync_hashes`. This contributed to the trigger-#6 halt encountered in this session before the new `--mode rebaseline-from-main` escape hatch landed (AW commit `954fec390`). Durable fix: extend `mode_create` (and `mode_create_atomic`, `mode_create_wrapup_handoff`) to call `_write_main_backup` after baseline capture, writing `.main_backup_create_{ts}.md`. Related follow-up: add post-validation to `mode_recover_main_writables` — recompute hashes after restore and fail loudly if they don't match `last_sync_hashes`, with a hint to use `--mode rebaseline-from-main`. Should be captured to a permanent idea file via `/capture-idea` at wrap-up.
- **F1 root-cause investigation** (from 11:47 AM smell entry, Intervention I3): the original 'specialist did the work, routed findings to terminal' failure remains under-investigated. The 11:31 intervention took it at face value but its example-template flaw caused F2 and obscured whether the intent would have worked. After I1 + I2 are applied and a clean re-invocation succeeds (or fails in a third distinct way), decide whether to dispatch a deeper investigation — possibly checking transcripts, the project-specialist-on-phase-doc edge case in rubric content, or token-budget pressure on long-running plan-reviews of heavy phase docs.

###### Specialist-identified follow-up items

###### triton-swmm-toolkit-specialist (2026-05-12 plan-review)

- **Flag 6 — workflow.py size growth** (orthogonal to Phase 2 objective): Investigate splitting workflow.py into workflow/builders.py + workflow/rule_emission.py + workflow/snakefile_generation.py after Phase 2 lands. workflow.py is currently 4614 lines per wc -l; the architecture doc pins it at 3451 — already drifted by ~1.2k lines. VMS-4 + VMS-5 add ~80 LOC of module-level dataclasses + helpers. The agent's editorial-stance section in the specialist identity flags this exact pattern as a publication-readiness risk. Specialist source: tritonswmm-toolkit-specialist plan-review 2026-05-12_1221.
- **Flag 7 — `_emit_report_artifacts` fallback-branch coverage** (orthogonal to Phase 2 objective): Add unit test exercising the `_emit_report_artifacts` (ImportError, ModuleNotFoundError) fallback branch (workflow.py:399-400) via monkeypatching importlib.resources.files to raise. With VMS-1's rename to dest_root and the new bundle-side invocation (module-level import per F-I3 resolution), the fallback branch's behavior under bundle invocation is untested. Specialist source: tritonswmm-toolkit-specialist plan-review 2026-05-12_1221.

###### software-engineering-specialist (2026-05-12 plan-review)

- **Flag 6 — `...` ellipsis-stub idiom invites silent partial implementations** (orthogonal to Phase 2 objective; FU per SE rubric Criterion 10): Across future plans, prefer `raise NotImplementedError(...)` over bare `...` for body placeholders in non-stub-file specs. PEP 484 sanctions ellipsis only inside .pyi stub files; in implementation .py files, a body of `...` is legal Python but its semantic is 'return None' — silently passes type-check on most signatures and produces invalid Snakefile fragments at runtime. The current plan's byte-identity golden test covers source-side helpers but not the bundle-side `_make_rule_emission_context` / `_harvest_rule_specs`; a `NotImplementedError` placeholder would fail loudly on first call. Specialist source: software-engineering-specialist plan-review 2026-05-12_1328.

###### triton-swmm-toolkit-specialist (2026-05-12 re-spec session plan-review)

- **F-FU Flag 6 — workflow.py module-level lift audit** (orthogonal to bundle-portable-report-regeneration): Audit `workflow.py` for other `SnakemakeWorkflowBuilder` methods that read no `self` state and could similarly be lifted to module-level for testability and to reduce builder-class concentration. Plan Phase 2 lifts one such method (`_emit_report_artifacts`); whether others exist is unknown without a systematic pass.
- **F-FU Flag 7 — bundle internal layout stipulation crystallization** (orthogonal): Author the deferred stipulation "bundle cfg yamls contain only bundle-root-relative or None Path fields" via `/manage-stipulation` once Plan Phase 1's per-field policy table has accumulated production runtime usage that confirms it is load-bearing. Not blocking for any of Plan Phases 2-6.
- **Planning-Instruction Improvement Proposal — Pass-when assertion atomic** (orthogonal — agentic-workspace meta-improvement): The plan review surfaced all 5 F-I findings as Pass-when-assertion conformance gaps (each finding's remediation was a one-bullet rationale addition forcing the criterion's Pass-when clause to become an explicit auditable assertion). Specialist proposed a new conditional atomic `pass-when-assertion-checklist.md` that would force Pass-when assertion wording at plan-write time, eliminating this finding class. Candidate for a separate /amend-prompt session.

###### software-engineering-specialist (2026-05-12 re-spec session Phase 2 plan-review)

- **F-FU — HPC kaleido env-readiness coordination**: Plan Phase 5 VMS-8 swaps source-side workflow.py call sites to read static_backend from cfg (defaulting to plotly per Decision 4); HPC compute nodes without kaleido installed will hit Plan Phase 3's preflight check. Plan Phase 6 Step 9 documents the resolution (pip install -e '.[viz-export]' OR set static_backend: matplotlib in HPC cfg_analysis.yaml). Coordinate with UVA/Frontier env-setup docs to ensure the viz-export extra is named in conda env bootstrap instructions if HPC users routinely run analysis.run().
- **F-FU — workflow.py module-level extraction follow-up**: post-bundle-plan, extract workflow.py's rule-emission helpers (`_emit_plot_rule`, `_emit_render_report_rule`, `_emit_rule_all`, `_output_ext_for`, `RuleEmissionContext`, `RuleSpec`) into a submodule (e.g., `workflow/_rule_emission.py`) once Plan Phase 2's refactor settles. The Phase 2 lift to module-level is the right shape but workflow.py is large and the helpers form a cohesive sub-surface that could live in a sibling module for testability and reduced builder-class concentration.

###### snakemake-specialist (2026-05-12 re-spec session Phase 2 plan-review)

- **F-FU — Snakemake `--report` semantics knowledge-doc note** (deferred from plan): capture in a knowledge doc the semantics of `snakemake --report` on a fresh working directory with pre-existing outputs. The bundle-portable-report-regeneration plan exposed that `--report` requires recorded metadata (which Plan Phase 3 VMS-1 now supplies via the `--touch` pre-step), but the broader question "when does `--report` re-execute rules vs. only re-render reports" is worth a dedicated knowledge doc entry. Useful for future plans that touch snakemake's reporting machinery.

###### data-visualization-specialist (2026-05-12 re-spec session Phase 2 plan-review — terminal-summary roll-in)

- **F-FU — Project-wide Plotly-template audit**: audit all Plotly-emitting renderers (current + future) for explicit `template="plotly_white"` (or equivalent journal-friendly template) assignment. Plan Phase 5's per-renderer template-assertion grep covers the 4 Plotly-branch renderers in scope; a project-wide pattern check would surface any future renderer additions that miss the template-assertion step.
- **F-FU — Kaleido failure-mode unit test**: add a unit test that simulates kaleido import failure inside a Plotly-branch renderer's `fig.write_image(...)` call (currently wrapped in try/except per VMS-2/3/4/5) and asserts the fallback path (matplotlib branch or no SVG) is reachable. Plan Phase 5 ships with try/except wrappers as defense-in-depth; testing the exception path verifies the fallback doesn't silently emit a broken SVG.

###### Smell findings (Phase 4 — 2026-05-12)

- **Cross-repo write-block hook gap (decisions/knowledge/prompts)** — `block-main-tree-write-from-worktree.sh` is per-repo scoped: from a non-agentic-workspace worktree it derives `main_tree` from the current shell's git context (e.g., `/home/***REMOVED***/dev/TRITON-SWMM_toolkit` for this worktree), and silently approves writes to `$AGENTIC_WORKSPACE/library/docs/decisions/`, `$AGENTIC_WORKSPACE/library/docs/knowledge/`, `$AGENTIC_WORKSPACE/library/prompts/`, etc. The hook docstring at line 13 says "outside $AGENTIC_WORKSPACE entirely" is the open allow set, but the implementation does not enforce this. Surfaced during Phase 4 VMS-4/VMS-5 (which directed `/manage-decision`) when a parallel agentic-workspace session's merge + reset + setup.sh clobbered the in-flight unstaged decision-doc edits twice. Full smell entry with intervention specs in main scratch under `# Smells > ## Smell: Cross-repo write-block hook gap (decisions/knowledge/prompts) — 2026-05-12 22:50`.

  Proposed interventions (recommendation: I1 + I2 only; defer I3):
    - **I1 — Extend hook for cross-repo enforcement** (~+25 LOC in `block-main-tree-write-from-worktree.sh`): add a Stage-2 classification block scoped to `$AGENTIC_WORKSPACE` (resolved from env var, not git context). Allow-list: `${AGENTIC_WORKSPACE}/library/docs/planning/**` and `${AGENTIC_WORKSPACE}/library/_scratch/**`. Block-list: anything else under `${AGENTIC_WORKSPACE}/` when the current worktree is not $AGENTIC_WORKSPACE itself.
    - **I2 — Reconcile hook docstring with implementation** (~+15 LOC, bundled in I1's commit): rewrite the Allow/Block list comments at the head of the hook to describe the actual two-stage logic so a docstring-only reader can predict behavior.
    - **I3 — Refresh PWI multiphase-path doc's worktree-context paragraph** (~+90 words, deferred): surface the actual two-allow-list contract at the place where session-resume agents read it. Defer until I1's block message proves insufficient in practice.

  Should be promoted to a permanent idea file via `/capture-idea` at wrap-up.

- **V0005 migration AND CaseStudyBuilder both drop populated `Optional[dict]` fields when writing `cfg_analysis.yaml::report.sensitivity`** — empirically observed 2026-05-15 11:12 + 11:29 on Rivanna. Two write-sites with the same null-sensitivity outcome.
    - **Write-site 1 — V0005 migration** at `src/TRITON_SWMM_toolkit/version_migration/versions/V0005__inline_report_config.py:97-117`: loaded the git-tracked template `test_data/norfolk_coastal_flooding/report_config_uva_benchmarking_norfolk_irene.yaml` (which has a populated `sensitivity:` block at template lines 31-40 with `mode: benchmarking`, `independent_vars: [n_devices]`, etc.); `cfg.model_dump(mode="json")` produced a populated dict per the V0005 verbose log; but `yaml_add_field` wrote `sensitivity: null` to disk.
    - **Write-site 2 — `CaseStudyBuilder.__init__`** at `src/TRITON_SWMM_toolkit/case_study_catalog.py:95-113`: `cfg_analysis = self.example.analysis.cfg_analysis.model_copy()` (example default has `report.sensitivity=None`), `for key, val in final_analysis_configs.items(): setattr(cfg_analysis, key, val)` (flat setattr — no nested field support, and `analysis_overrides` for benchmarking_norfolk_irene doesn't include `report.sensitivity`), then `yaml.safe_dump(cfg_analysis.model_dump(mode="json"))` writes the buggy cfg to disk. This rewrite happens on EVERY `cat.UVACaseStudies.benchmarking_norfolk_irene(...)` call — the YAML-level sed patch cannot survive a builder re-run.
    - Root-cause hypothesis: both write paths involve `model_dump(mode="json")` which, when fed back into Pydantic via `yaml_add_field`'s `in_model_cls=...` or `model_validate`, may collapse populated `Optional[X]=...` fields back to None during round-trip. Or `analysis_overrides`'s flat setattr is structurally unable to populate nested fields.
    - The cfg fails Pydantic validation at `analysis.run()` time with `ConfigurationError: report.sensitivity must be set` and an indirect error message that points operators at the legacy `report_config.yaml` peer-file path (stale post-F2).
    - Bandaid applied this session: in-memory monkey-patch — set `analysis.cfg_analysis.report.sensitivity` directly in the live Python session before calling `analysis.run()`. The on-disk YAML stays buggy; the in-memory cfg is correct for one run.
    - Durable fixes: (a) modify `CaseStudyBuilder.__init__` to populate `report.sensitivity` from a per-case default when `toggle_sensitivity_analysis=True` and `cfg_analysis.report.sensitivity is None` after override application; (b) accept a nested `report_overrides` parameter on CaseStudyBuilder and populate it for sensitivity cases; (c) separately fix V0005's null-write so future migrations of sensitivity analyses get the right shape from the start (lower priority since write-site 2 is the dominant source). Promote to a permanent idea file via `/capture-idea` at wrap-up.


- **`scripts/edit/port_h1_sections.py` script + `/cleanup-scratch` SKILL Step 4 dispatch rewrite** — `/cleanup-scratch` currently inlines a Python heredoc that the agent transcribes into a Bash invocation each session. The heredoc captures line ranges at Step 1's H1 scan and uses them at Step 4's port; any edit between those two steps silently invalidates the captured ranges. Plus the fence-aware H1 detection is naive and miscounts on unbalanced fences. Empirically caused a 10-minute structural cleanup on 2026-05-14 where the first port left the file in a broken state. Repo has the section-port script idiom established (`port_specialist_plan_to_scratch.py`, `port_captured_ideas_to_scratch.py`). Durable fix: author `scripts/edit/port_h1_sections.py` (~250 LOC) that takes `--active <path> --archive <path> --port-h1 <title>` (repeatable) and `--restore-h1 <title>` (for the `# Scratch Cleanup` parent), rescans at invocation time, detects unbalanced fences with a deterministic exit code + offending-line pointer, and writes atomically. Rewrite `/cleanup-scratch` SKILL Step 4-5 to dispatch to it. Full VMS in `# Smells` under `## Smell: /cleanup-scratch inlines Python heredoc instead of calling a script — 2026-05-14 22:15` → `### INT-1`. Promote to a permanent idea file via `/capture-idea` at `/wrapup-session`.
- **Snakemake tmux-mode swallows per-rule failures, returns `success=True` to orchestrator despite missing outputs** — empirically observed 2026-05-15 11:44–12:09 on Rivanna. The live `analysis.run(dry_run=False, wait_for_job_completion=True)` returned `WorkflowResult(success=True, mode='tmux', message='Workflow completed successfully')`. But the per-rule log at `logs/plots/sensitivity_benchmarking_n_devices.log` showed `ValueError: report_cfg.sensitivity must be set for benchmarking rendering` and the expected output `plots/sensitivity/benchmarking/n_devices_vs_total.html` was never produced. Snakemake's rule_all expects this file; the rule was in the dry-run plan; the rule's subprocess errored; but Snakemake-via-tmux's exit code or status propagation did not surface this failure to the orchestrator. The orchestrator's `success=True` was the misleading signal that initially made me think we were done. Durable fix: investigate how `mode='tmux'` reads the snakemake exit status and either propagate per-rule failures into `WorkflowResult.success=False` OR surface a `partial_failures: list[str]` field on the result so callers can branch. Probably lives in `src/TRITON_SWMM_toolkit/analysis.py` near where the WorkflowResult is constructed, plus the orchestration shim that bridges the tmux session and the parent process. Bandaid this session: operator manually invokes the failing renderer CLI directly with a patched cfg. Promote to a permanent idea via `/capture-idea` at wrap-up.

## Worktree Status

- branch: worktree-toolkit_05-11_1503_bundle-portable-report-regen-pwi
- path: /home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-11_1503_bundle-portable-report-regen-pwi

