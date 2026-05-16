---
prompt_doc_type: sidecar_scratch
sidecar_kind: wrapup_handoff
main_scratch: /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi.md
last_sync_hashes:
  '### Phase Audit History': 128dd829ae24d76fc1ad9986a052feaabfc08cc96216dc046016c0b60cb3cdb5
  '# Follow-up Ideas': 2d4b23c20e70ad532dc5012742d9ab5d61ee50724f975be910ba6101b0a410f1
harness: claude-code
plan_name: bundle_cfg_report_canonicalization
plan_completion_commit: 6aa7a3582a35b6a212e62ba14e4362c4a8fa8fd3
worktree_path: /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi
created: '2026-05-14T17:14:32'
custom_instructions:
  - When running DoD validation commands outside pytest (e.g., python -c 'from TRITON_SWMM_toolkit.version_migration.constants import ...'), prefix with PYTHONPATH=<worktree>/src — the pip-installed package resolves to a different worktree (toolkit_05-11_1503_...) and silent-imports stale constants. PYTHONPATH-prefixing reproduces the pytest conftest's sys.path-prefix behavior.
  - Phase 3 plan body cites specific line numbers (e.g., _read_static_backend lines 130-147, _copy_report_config lines 189-218, --report-config typer Option lines 806-816 / 65-68, callsite arguments lines 836 / 840, report_config_path kwarg lines 477 / 501 / 2113 / 2202 / 2226). Re-verify each with grep -n against the worktree before editing — line numbers may have decayed across Phase 1+2 commits.
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

- When running DoD validation commands outside pytest (e.g., python -c 'from TRITON_SWMM_toolkit.version_migration.constants import ...'), prefix with PYTHONPATH=<worktree>/src — the pip-installed package resolves to a different worktree (toolkit_05-11_1503_...) and silent-imports stale constants. PYTHONPATH-prefixing reproduces the pytest conftest's sys.path-prefix behavior.
- Phase 3 plan body cites specific line numbers (e.g., _read_static_backend lines 130-147, _copy_report_config lines 189-218, --report-config typer Option lines 806-816 / 65-68, callsite arguments lines 836 / 840, report_config_path kwarg lines 477 / 501 / 2113 / 2202 / 2226). Re-verify each with grep -n against the worktree before editing — line numbers may have decayed across Phase 1+2 commits.

## Wrapup Entry Point

- Kind: wrapup_handoff
- Plan name: bundle_cfg_report_canonicalization
- Plan completion commit: `6aa7a3582a35b6a212e62ba14e4362c4a8fa8fd3`
- Worktree: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
- Worktree branch: `worktree-toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
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
   `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi/sidecars/sidecar_wrapup_2026-05-14_1714.md`

2. Wrapup-session skill body (the skill you are about to execute):
   [wrapup session SKILL](../../instructions/skills/wrapup-session/wrapup%20session%20SKILL.md) #inject_path

%% inject-path-start: wrapup-session-skill %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`
    - Session wrap-up checklist — audits for uncommitted changes, outstanding todos, unresolved decisions, and next steps before exiting
%% inject-path-end: wrapup-session-skill %%

### Wrapup entry point

- **Plan name**: `bundle_cfg_report_canonicalization`
- **Plan completion commit**: `6aa7a3582a35b6a212e62ba14e4362c4a8fa8fd3`
- **Worktree path**: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
- **Worktree branch**: `worktree-toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
- **Task**: run `/wrapup-session` in full, per the skill body above.

Plan Completion already moved all phase docs to `implemented/` and the master
plan subdirectory to `completed/`. Regeneration of planning tables and the
plan closeout commit are in git history at `6aa7a3582a35b6a212e62ba14e4362c4a8fa8fd3`. If you
find yourself tempted to re-run `$AGENTIC_WORKSPACE/scripts/generate/complete_plan.py` or touch the planning
tree, stop — that work is done.

Do NOT read the main session scratch doc yet. Main scratch is intentionally
out of scope for the First Action gate. You will read it at wrapup-session
Step 6 — see the "Step 6 injection-then-reload contract" below.

### Step 6 injection-then-reload contract (read this carefully)

When `/wrapup-session` reaches Step 6 (Write Report to Session Scratch Doc),
the skill runs `create_from_template --compose wrapup-session
--inject-into-scratch /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi.md` to append the wrapup template
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

- **Worktree path**: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
- **Worktree branch**: `worktree-toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi`
- **Main scratch path**: `/home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi.md`

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

`Wrapup-entry assertion: Sidecar: {sidecar_filename}. sidecar_kind: wrapup_handoff. Plan name: bundle_cfg_report_canonicalization. Plan completion commit: 6aa7a3582a35b6a212e62ba14e4362c4a8fa8fd3. Wrapup entry block present in body: YES|NO.`

If any of these do not match the sidecar's actual state, or if the body does
not contain `## Wrapup Entry Point`, HALT per the post-halt response protocol
and surface the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke:

    scripts/generate/write_compaction_sidecar.py --mode verify \
      --main-scratch /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi.md \
      --worktree /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_05-13_2337_bundle-cfg-report-canonicalization-pwi

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
| 1 | `src/TRITON_SWMM_toolkit/config/analysis.py`, `src/TRITON_SWMM_toolkit/analysis.py`, `tests/test_config_analysis_report_field.py` (new), `tests/test_config_validation.py`, `tests/fixtures/test_case_builder.py`, `tests/fixtures/bundles/multi_sim/cfg_analysis.yaml`, `tests/fixtures/bundles/sensitivity_master/cfg_analysis.yaml`, `test_data/norfolk_coastal_flooding/template_analysis_config.yaml` | 0 (scope `file_local,subset` returned exit 0; Phase 1 changes are TRITON-SWMM_toolkit code, not agentic-workspace instructional/planning content where the audit operates) | N/A |
| 2 | `src/TRITON_SWMM_toolkit/version_migration/constants.py`, `src/TRITON_SWMM_toolkit/version_migration/exceptions.py`, `src/TRITON_SWMM_toolkit/version_migration/__main__.py`, `src/TRITON_SWMM_toolkit/version_migration/versions/V0005__inline_report_config.py` (new), `tests/test_version_migration_V0005.py` (new), `tests/fixtures/legacy_layouts/v0005_unit_test/{recoverable,no_snakefile,missing_source_cfg,v5_recoverable}/` (new sub-fixtures), `tests/fixtures/legacy_layouts/v5/` (new — deep-copy of v4 to satisfy `_discover_fixture_pairs` golden-test auto-discovery without altering tree) | 0 (scope `file_local,subset` raised `ValueError: path not in agentic-workspace subpath`; Phase 2 changes are entirely TRITON-SWMM_toolkit code/fixtures, outside the audit's instructional/planning operational range — same boundary as Phase 1) | N/A. dry_run_counts: RECOVERABLE=0, UNRECOVERABLE_NO_FLAG=1, UNRECOVERABLE_FILE_MISSING=3 (against `tests/fixtures/legacy_layouts/v0005_unit_test/` unit fixtures; the /tmp/v0005_fixture_source_cfg.yaml that would make the two recoverable fixtures classify as RECOVERABLE is created at test-runtime by the test setup, not at dry-run time). Operator dry-run against actual UVA/Frontier corpus deferred — no cluster access available in this session; flag as Phase 2 follow-up for cluster-attached operator. |
| 3 | `src/TRITON_SWMM_toolkit/bundle/__init__.py`, `src/TRITON_SWMM_toolkit/bundle/_emit.py`, `src/TRITON_SWMM_toolkit/cli.py`, `src/TRITON_SWMM_toolkit/analysis.py`, `src/TRITON_SWMM_toolkit/workflow.py`, `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`, `src/TRITON_SWMM_toolkit/case_study_catalog.py`, `src/TRITON_SWMM_toolkit/report_renderers/_cli.py`, `tests/test_bundle.py`, `tests/test_workflow_snakefile_extension_consistency.py`, `tests/test_synth_05_sensitivity_analysis_with_snakemake.py`, `tests/fixtures/test_case_catalog.py`, `tests/fixtures/bundles/multi_sim/bundle_manifest.json`, `tests/fixtures/bundles/sensitivity_master/bundle_manifest.json` | 0 (scope `file_local,subset` returned exit 0; Phase 3 changes are entirely TRITON-SWMM_toolkit code/tests/fixtures, outside the audit's instructional/planning operational range — same boundary as Phases 1 and 2) | N/A. DoD-grep #1 (scoped `--exclude-dir=version_migration`): zero matches. DoD-grep #2 (`DEFAULT_REPORT_CONFIG` under `src/TRITON_SWMM_toolkit/bundle/`): zero matches. Test results: 20/20 test_bundle, 8/8 extension_consistency (matplotlib+plotly), 4/4 synth_05 fast, 10/10 synth_04 fast (after user-approved `pip install -e .` from this worktree to resolve the cross-worktree pip-install collision documented in custom_instructions[1]). Three plan deviations recorded in QAQC report: (a) DoD-grep #1 scoped to exclude `version_migration/` per Option A preflight refinement; (b) analysis.py deletion scope extended to include orphaned `self._cfg_report_path` write at line 1638; (c) `_OUTPUT_EXT_BY_RENDERER["sensitivity_benchmarking"]` matplotlib→.svg fix for rule_all/render_report symmetry (latent bug Phase 3 exposed). Six sidecar-documented pre-existing failures (`test_synth_08_bundle_round_trip` × 4, `test_workflow_snakefile_byte_identity` × 2) remain untouched by Phase 3. |
%%

## Follow-up Ideas

### Halt — First Action gate Part 3 (sync integrity) — 2026-05-14 post-compact resume

- **Trigger fired**: Halt trigger #3 — `write_compaction_sidecar.py --mode verify` exited 3 (worktree dirty).
- **Wrapup step at which halt detected**: First Action gate Part 3, before /wrapup-session was invoked.
- **Specific dirty content** (one tracked-file deletion, four untracked artifacts):
  - ` D sidecars/sidecar_phase3_2026-05-14_1457.md` — the prior in-flight phase-3 sidecar was deleted from the tracked tree as part of `write_compaction_sidecar.py --mode create-wrapup-handoff`'s archive operation, but the deletion was not committed.
  - `?? sidecars/done/sidecar_phase3_2026-05-14_1457.md` — the archived counterpart of the above (untracked at the new location).
  - `?? sidecars/.main_backup_2026-05-14_145745.md`, `?? sidecars/.main_backup_2026-05-14_171432.md` — sidecar-mode main-scratch backups.
  - `?? sidecars/sidecar_wrapup_2026-05-14_1714.md` — the wrapup-handoff sidecar created by `--mode create-wrapup-handoff` (untracked).
- **Diagnosis**: `write_compaction_sidecar.py --mode create-wrapup-handoff` performs a mv-to-`done/` of every prior in-flight sidecar (per `wrapup session SKILL.md` line 900: "Prior in-flight sidecars have already been archived to `<worktree-path>/sidecars/done/` by `write_compaction_sidecar.py --mode create-wrapup-handoff`"). The pre-compaction session ran the create-wrapup-handoff but did not commit the resulting archive-move + new-sidecar set before /compact fired. On post-compact resume, --mode verify sees the unstaged `D` and exits 3.
- **Why this is structurally similar to** the previously-recorded `/resume-sidecar` claim-toggle vs. Part-3 sync-integrity gate seam (line 356 of this sidecar): both are cases where a script-side write to the sidecar tree leaves the worktree dirty, and the next gate that runs `--mode verify` halts. Here the writer is `--mode create-wrapup-handoff` rather than `sidecar_pick --pick`, but the seam shape is the same. Candidate fix: have `--mode create-wrapup-handoff` stage+commit its own mv+create as a deterministic `chore: archive in-flight sidecars and create wrapup handoff` commit, so the post-compact resume sees a clean tree.
- **Awaiting user direction**. Two obvious paths:
  1. Commit the archive-move + new-sidecar set as a single `chore: archive phase-3 sidecar and stage wrapup-handoff sidecar` commit (mirrors the per-resume-sidecar workaround already documented), then re-run `--mode verify` and re-enter the First Action gate.
  2. Some other resolution the user wants to apply.


%%
"#" + "followup" tag: When the user tags an inline comment with `#` + `followup`, cut the tagged text from its original location, paste it here as a bullet, and add context sub-bullets if the original text does not stand alone without its surrounding context. The agent judges whether sub-bullets are needed — if the comment makes sense on its own, no sub-bullets are required. Change the tag in the original location to `#followup-moved`.

Specialist findings routed here by /invoke-specialist Step 6 appear under:
  ###### Specialist-identified follow-up items
  ###### {specialist-name}
These are appended via append_scratch_section.py and are automatically picked up by /review-followup-ideas.

Agents must never recommend action items for followup unless they are orthogonal to the objectives and target outcomes of a session.
%%

- **Pre-existing test failures unrelated to canonicalization** (6 fast + 2 slow, confirmed on parent commit via stash + baseline `pytest -m "not slow" --tb=no -q`). Grouped:
  1. **`tests/test_synth_08_bundle_round_trip.py` — 4 failures**: `test_bundle_round_trip[rendered_synth_multi_sim]`, `[rendered_synth_sensitivity]`, `test_bundle_baseline_wrapper_section_matches[rendered_synth_multi_sim]`, `[rendered_synth_sensitivity]`. Root cause for the `test_bundle_round_trip` pair: test invokes `TRITON_SWMM_toolkit report-from-bundle <bundle>.tar --format html` (test line ~90), but `cli.py:878` (`report_from_bundle_command`) only accepts `.zip` or unpacked dirs — not `.tar`. The `test_bundle_baseline_wrapper_section_matches` pair likely fails for a related-but-not-yet-inspected reason. **Fix surface**: either patch test to use `.zip` / pre-unpacked dir, or extend CLI to accept `.tar`. **LoE**: Low-Medium (~1–2 h). **Risk**: Low (test or CLI delta).
  2. **`tests/test_workflow_snakefile_byte_identity.py` — 2 failures**: `test_multi_sim_snakefile_byte_identity`, `test_master_snakefile_byte_identity`. Byte-for-byte golden-Snakefile asserts. Without diffing, two possibilities: (a) workflow.py Snakefile-emission drifted from the recorded golden, or (b) the golden artifact is stale. **LoE**: Low (~30–60 min). **Risk**: Medium if (a) is true (real workflow regression); Low if (b).
  3. **`tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py` — 2 `@pytest.mark.slow` failures** (sensitivity-block follow-up, detailed below).
  **Why all deferred from Phase 1**: each failure mode predates this plan and is in a code area Phase 1 did not touch (bundle CLI, workflow emission, sensitivity-validation fallback). Phase 1 introduces zero new failures beyond what was already red on the parent commit. The Phase 1 DoD's "`pytest tests/ -x` runs clean" bullet was scoped to "zero `ValidationError` regressions from fixtures lacking `report:`" — that scope is met.

- **Norfolk sensitivity slow tests fail with `ConfigurationError: report_config.sensitivity must be set`** (orthogonal to Phase 1; verified pre-existing on parent commit via stash + isolated re-run). Failing tests: `tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_swmm_only_execution` (and the `_triton_only` sibling at line 33); likely the broader Norfolk/UVA `retrieve_*_sensitivity_*` factories share the bug. **Diagnosis**: when `toggle_sensitivity_analysis=True` and a sensitivity CSV is attached to the analysis, `validate_sensitivity_independent_vars` at `src/TRITON_SWMM_toolkit/config/report.py:706` raises if `cfg_report.sensitivity is None`. The norfolk PC_06 tests call `analysis.run()` with no `report_config=` arg, so they fall back to the empty default (`DEFAULT_REPORT_CONFIG` pre-Phase-1; `cfg_analysis.report = {}` post-Phase-1). Both have `sensitivity=None`. The synth equivalents (`tests/test_synth_05_sensitivity_analysis_with_snakemake.py:67`, `test_synth_06_*`) work because they pass `report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG` explicitly. **Fix surface**: add a `report` override (with a populated `sensitivity:` block carrying the right `mode` + `independent_vars` for each CSV) to each sensitivity-toggled factory in `tests/fixtures/test_case_catalog.py` — specifically `retrieve_norfolk_cpu_config_sensitivity_case`, `*_triton_only`, `*_swmm_only`, `UVA_TestCases.retrieve_norfolk_UVA_sensitivity_*` (3 variants), and any Frontier sensitivity factories. Independent_vars must match the column names in each `.xlsx` file (`cpu_benchmarking_analysis_swmm.xlsx`, `full_benchmarking_experiment_uva_*.xlsx`, etc.). **LoE**: Low-Medium (~1.5–3 h) — header inspection + factory patches + slow-test re-run; UVA/Frontier variants validated only when run on cluster. **Risk**: Low (test-fixture-only delta) with a Medium open question — was this code path ever exercised cleanly in production, or is the empty-fallback path simply unreachable in real usage? Worth verifying during the fix. **Why deferred from Phase 1**: failure predates this plan; pre-Phase-1 fallback (`DEFAULT_REPORT_CONFIG`) was equally empty, so Phase 1's 2-step resolution did not cause the failure.

###### Specialist-identified follow-up items

###### triton-swmm-toolkit-specialist

- **F-FU Flag 9 (B2 per-model-type renderer backend overrides)** — Per-scenario renderers (`per_sim_peak_flood_depth`, `per_sim_conduit_flow`) read each model-type's processed output. If `static_backend` ever needs to differ per model-type (matplotlib for TRITON-only, plotly for tritonswmm), the current single-`static_backend` setting cannot express it. Capture as architectural follow-up; not in scope for the current canonicalization plan.

- **F-FU Flag 10 (D1 yaml_add_field idempotency invariant)** — V0005 relies on `yaml_add_field`'s internal `if name in data: return` guard for idempotency. Future migrations will depend on the same invariant. Propose authoring a stipulation (`library/docs/stipulations/TRITON-SWMM_toolkit/yaml_add_field is idempotent.md`) once V0005 ships so future migration authors know the invariant is load-bearing. User-gated via `/manage-stipulation` per the F-FU phrasing constraint.

- **Planning-Instruction Improvement Proposal 1**: `plan-implementation` skill body should require a `grep -n` re-verification step against the worktree as part of File-by-File completeness check when the plan cites specific line numbers, since line numbers decay between plan authoring and implementation. Would have caught Flag 2.

- **Planning-Instruction Improvement Proposal 2**: Plan-implementation guidance should include a 'check whether file already exists before specifying create-new' forcing check when a File-by-File entry uses the `body` tag for a `*.py` module in a known existing-file location. Would have caught Flag 4 (which is HIGH-risk — would have silently deleted the existing 71-LOC version_migration/exceptions.py).

###### plan-closeout (this session)

- **Orphan untracked idea file** observed at Step A plan-closeout commit time: `library/docs/planning/projects/TRITON-SWMM_toolkit/ideas/unify multisim and sensitivity master render report emission via the shared emit render report rule helper.md`. Not session-attributable (untracked, no `M` in git status before this session opened). Excluded from the closeout commit per the planning-table cleanup attribution rule. **Follow-up**: triage during /wrapup-session — either accept (commit) the idea standalone, or delete if it's a stale draft from a different session. Tables would re-generate consistently because the file is on disk and will be picked up by future `generate_planning_tables --all`.

###### software-engineering-specialist

- **F-FU Flag 9 (case_study_catalog.py legacy peer-file migration)** — After Phase 3 lands, audit each `report_config_*.yaml` peer file referenced by case study factory functions in `case_study_catalog.py`, migrate the content inline into each corresponding `cfg_analysis.yaml::report` block, delete the peer YAMLs, and update factory functions to pass the new schema. Phase 3 doc currently defers this with 'implementer to pick the migration path' (forbidden-deferral language); make it concrete after Phase 3 ships.

- **F-FU Flag 10 (MigrationContext.execute() behavior unverified)** — Pre-Phase-2 implementation, read `src/TRITON_SWMM_toolkit/version_migration/context.py::MigrationContext.execute()` end-to-end and confirm whether it validates the post-op state against `in_model_cls`. If YES: document the fixture-content requirements in Phase 2 doc (the v4 cfg_analysis.yaml files would need to satisfy `cfgBaseModel._check_paths_exist` on the test machine). If NO: drop the `in_model_cls=analysis_config` argument from V0005's `yaml_add_field` call as unused noise. Tied to Flag 6 fixture-construction strategy.

- **setup_pwi_worktree.py sidecar-init friction** (session-observed 2026-05-14): When a new worktree is created via setup_pwi_worktree.py and the worktree branch's main-tree history contains committed sidecar files under `.claude/worktrees/<slug>/sidecars/`, those sidecars appear in the fresh worktree and block write_compaction_sidecar.py --mode create with 'sidecars already exist' error. Workaround applied this session: moved pre-existing sidecars to `sidecars/done/pre-session-archived-YYYY-MM-DD/` before retrying create. Investigate whether setup_pwi_worktree.py should auto-clear or auto-archive any pre-existing sidecars when creating a new worktree, OR whether the write_compaction_sidecar.py --mode create logic should accept a non-empty sidecars dir when no current-session sidecars are present.

- **`/resume-sidecar` claim-toggle vs. Part-3 sync-integrity gate seam** (session-observed 2026-05-14): When `/resume-sidecar`'s `sidecar_pick --pick N` atomically toggles a sidecar's `- [ ] in progress` line to `- [x] in progress`, the worktree is left dirty by exactly that one-line edit. The post-compaction First Action gate's Part 3 (`write_compaction_sidecar.py --mode verify`) then exits 3 (worktree dirty) and halts the resume. Resolution this session: commit the claim toggle as a `chore: claim phase N sidecar via /resume-sidecar` commit, then re-verify. The seam is structural — every multiphase post-compact resume will hit it. Two candidate fixes: (a) make `sidecar_pick --pick` stage+commit the toggle itself with a deterministic chore message, or (b) loosen `--mode verify`'s exit-3 condition to ignore single-line `- [ ]` → `- [x]` edits on the sidecar's "in progress" line specifically. Option (a) is cleaner because the audit trail of "who claimed this sidecar when" is preserved as a git commit; option (b) is more permissive but harder to reason about.

- **Phase 2 plan-implementation deviation: V0005 unit-test fixture relocation** (session-observed 2026-05-14): Plan specified `tests/fixtures/legacy_layouts/v4/{recoverable,no_snakefile,missing_source_cfg}/` and `v5/{recoverable}/` sub-fixtures. Implementation relocated all four to `tests/fixtures/legacy_layouts/v0005_unit_test/{recoverable,no_snakefile,missing_source_cfg,v5_recoverable}/` because `test_version_migration_golden.py::_discover_fixture_pairs` auto-globs every `v{N}/` directory and parametrizes round-trip equality tests on the full v{from}/ → v{to}/ file tree (minus `_version.json`); sub-fixtures inside v4/ or v5/ would break v3→v4, v0→v5, etc. pairs. Plus `legacy_layouts/v5/` was created as a deep-copy of v4/ so the golden auto-discovery's v0→v5 / v3→v5 / v4→v5 cases trivially pass (V0005 is a no-op on the existing v4 corpus, which has no top-level `cfg_analysis.yaml`). Follow-up: when Phase 3 lands and the report.py / bundle codepaths are deleted, audit whether any other test file hardcodes `tests/fixtures/legacy_layouts/v4/analysis_with_recoverable_snakefile/` style paths from the plan text — none currently do, but the plan's literal path strings might leak into Phase 3 docs.

- **Phase 2 plan-implementation deviation: `MigrationBlockedError` parent class** (session-observed 2026-05-14): Plan body specified `class MigrationBlockedError(Exception)`. Implementation made it `class MigrationBlockedError(MigrationError)` to integrate with the existing 71-LOC exceptions.py hierarchy (rooted in `TRITONSWMMError`) and the `__main__.py` exit-code map (now mapped to validation exit code 2). Captured here per F-FU Flag 4 prior-specialist guidance.

- **Phase 2 plan-implementation deviation: `dry-run-report` subcommand vs. top-level `--dry-run --roots` flag** (session-observed 2026-05-14): Plan's validation command used `python -m TRITON_SWMM_toolkit.version_migration --dry-run --roots tests/fixtures/legacy_layouts/v4/`. Implementation added a `dry-run-report --roots <dir>` subcommand instead because the existing __main__.py is a Typer subcommand surface (`migrate`, `status`, `baseline`, `verify`) with no top-level options. The sidecar's phase summary correctly anticipated this shape as "`dry-run-report` Typer sub-command".

- **Phase 2 plan-implementation refinement: V0005 regex hardening** (session-observed 2026-05-14): The plan's `r"--report-config\s+(\S+)"` regex captures the trailing closing-quote when the Snakefile shell line is wrapped in `shell: "... --report-config /path/to.yaml"`. Implementation hardened to `r"--report-config\s+([^\s\"'`,;)]+)"` so the path token excludes shell-quoting/grouping characters. The hardening is defensive against snakemake shell-line quoting variations and does not change behavior on workflow.py's actual emitter (which produces unquoted paths per `workflow.py:518`).

- **V0005 operator dry-run on UVA/Frontier corpus not performed this session** (deferred from Phase 2 DoD): The Phase 2 DoD `Operator dry-run on actual UVA/Frontier corpus completed; UNRECOVERABLE count documented; count < 20` requires cluster access to run against real analysis dirs. Not available in this development session. The `dry-run-report` CLI surface is in place and exits cleanly against the unit-test fixtures (proving the recovery-classification logic is correct); the operator step is recommended as a separate follow-up before Phase 3 commits, because Phase 3 deletes the F1 codepaths and a high-UNRECOVERABLE result on the real corpus would reopen D3 retroactively.

## Worktree Status

*(section not present in main scratch)*

