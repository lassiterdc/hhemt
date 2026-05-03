---
prompt_doc_type: sidecar_scratch
sidecar_kind: wrapup_handoff
main_scratch: /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi.md
last_sync_hashes:
  '### Phase Audit History': ebaaf7b3c8ab0d6bc78eec05f8e31c658c504a0b76687ac705e545c1419f312b
  '# Follow-up Ideas': b501a59a4bb73ad82f819eb5264ae8c41bf4060bd837c276aecde64bb8194cab
harness: claude-code
plan_name: visualization_and_reporting
plan_completion_commit: 42c1c949b1c6566926126d8195d8faf2c9047a5f
worktree_path: /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi
created: '2026-05-03T00:30:33'
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

## Wrapup Entry Point

- Kind: wrapup_handoff
- Plan name: visualization_and_reporting
- Plan completion commit: `42c1c949b1c6566926126d8195d8faf2c9047a5f`
- Worktree: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
- Worktree branch: `worktree-toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
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
   `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi/sidecars/sidecar_wrapup_2026-05-03_0030.md`

2. Wrapup-session skill body (the skill you are about to execute):
   [wrapup session SKILL](../../instructions/skills/wrapup-session/wrapup%20session%20SKILL.md) #inject_path

%% inject-path-start: wrapup-session-skill %%
  - `$AGENTIC_WORKSPACE/library/prompts/instructions/skills/wrapup-session/wrapup session SKILL.md`
    - Session wrap-up checklist — audits for uncommitted changes, outstanding todos, unresolved decisions, and next steps before exiting
%% inject-path-end: wrapup-session-skill %%

### Wrapup entry point

- **Plan name**: `visualization_and_reporting`
- **Plan completion commit**: `42c1c949b1c6566926126d8195d8faf2c9047a5f`
- **Worktree path**: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
- **Worktree branch**: `worktree-toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
- **Task**: run `/wrapup-session` in full, per the skill body above.

Plan Completion already moved all phase docs to `implemented/` and the master
plan subdirectory to `completed/`. Regeneration of planning tables and the
plan closeout commit are in git history at `42c1c949b1c6566926126d8195d8faf2c9047a5f`. If you
find yourself tempted to re-run `$AGENTIC_WORKSPACE/scripts/generate/complete_plan.py` or touch the planning
tree, stop — that work is done.

Do NOT read the main session scratch doc yet. Main scratch is intentionally
out of scope for the First Action gate. You will read it at wrapup-session
Step 6 — see the "Step 6 injection-then-reload contract" below.

### Step 6 injection-then-reload contract (read this carefully)

When `/wrapup-session` reaches Step 6 (Write Report to Session Scratch Doc),
the skill runs `create_from_template --compose wrapup-session
--inject-into-scratch /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi.md` to append the wrapup template
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

- **Worktree path**: `/home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
- **Worktree branch**: `worktree-toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi`
- **Main scratch path**: `/home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi.md`

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

`Wrapup-entry assertion: Sidecar: {sidecar_filename}. sidecar_kind: wrapup_handoff. Plan name: visualization_and_reporting. Plan completion commit: 42c1c949b1c6566926126d8195d8faf2c9047a5f. Wrapup entry block present in body: YES|NO.`

If any of these do not match the sidecar's actual state, or if the body does
not contain `## Wrapup Entry Point`, HALT per the post-halt response protocol
and surface the specific inconsistency.

### Part 3 — Sync integrity assertion (via script, NOT by reading main)

Do NOT read the main scratch doc. Do NOT compute hashes yourself. Invoke:

    scripts/generate/write_compaction_sidecar.py --mode verify \
      --main-scratch /home/dcl3nd/dev/agentic-workspace/library/_scratch/workspaces/TRITON-SWMM_toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi.md \
      --worktree /home/dcl3nd/dev/TRITON-SWMM_toolkit/.claude/worktrees/toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi

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
| 1 — report_config schema and analysis.run() hook | configs/reports/README.md, configs/reports/default_report_config.yaml, src/TRITON_SWMM_toolkit/analysis.py, src/TRITON_SWMM_toolkit/cli.py, src/TRITON_SWMM_toolkit/config/loaders.py, src/TRITON_SWMM_toolkit/config/report.py, src/TRITON_SWMM_toolkit/sensitivity_analysis.py, src/TRITON_SWMM_toolkit/toolkit.py, src/TRITON_SWMM_toolkit/workflow.py, tests/test_config_validation.py | 0 new ruff errors on touched code (1 new Optional[Path] flagged by UP045 fixed in same session; 7 pre-existing I001 in loaders.py left alone per scope rule). 14/14 tests pass in tests/test_config_validation.py including 8 new Phase-1 validators (F-I-6, F-I-7, Flag 7, Flag 14, Flag 17, cross-field CSV validation). | All DoD items satisfied: schema + default config + README authored; facade threading verified by grep across analysis.py / sensitivity_analysis.py / workflow.py (19 hit sites); Decision + Stipulation docs authored and committed on main (agentic-workspace b0c46c160). | 09399d7 |
| 3 — per-sim renderers stop gate | src/TRITON_SWMM_toolkit/report_renderers/{per_sim_conduit_flow.py, per_sim_peak_flood_depth.py}, src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py (`collect_per_sim_source_paths` helper for the Snakemake function-based `params:`), src/TRITON_SWMM_toolkit/workflow.py (`_build_plot_rule_block_per_sim` method + `rule all` extension with `expand("plots/per_sim/{event_id}/{conduit_flow,peak_flood_depth}.png", event_id=SIM_IDS)`), tests/fixtures/synthetic_model/cache.py (`rainfall_peak_mm_per_hr 100→500`), tests/fixtures/synthetic_model/swmm_template.py (`perc_imperv_map → 100/100/100`, `_CONDUIT_DIAMETER_M 0.2→0.1`) | Phase 3 doc instructed `proc.scen_paths.swmm_hydro_inp` for conduit geometry, but `hydro.inp` is the hydrology-only SWMM variant with no `[CONDUITS]` section — produced a blank figure (iter-1 user feedback). Fixed: read `swmm_hydraulics_inp` (with `swmm_full_inp` fallback). Iter-1 also surfaced 4 visual feedback items (no conduits displayed, missing black outline, perceptually-uniform colormap on both panels, conduits not all surcharging); iter-2 applied the renderer fixes (`Blues`/`Reds` non-overlapping cmaps, two line artists per conduit for black underline + colored fill, draw all conduits regardless of value); iter-3 applied scenario tuning to push every conduit ≥ max-over-full. TRITON summary zarr has no `rio.crs` on the synth fixture — patched the watershed-reprojection branch to skip when raster has no CRS. | All Phase 3 DoD items satisfied: `per_sim_conduit_flow.py` and `per_sim_peak_flood_depth.py` produced with uniform `render(analysis, report_cfg, output_path, *, event_iloc)` signature; both renderers dispatch on `_get_enabled_model_types()` with model-type-skip placeholders; Snakemake per-sim rules wired into `rule all` via `expand()`; pytest tests/test_synth_04_multisim_with_snakemake passes 6/1 (skipped scheduler-context test) in 68s; pytest tests/test_synth_02 + test_synth_05 pass 7/7 in 86s after fixture rebuild. Per-sim figures iter-1 STOP-gate (peak_flood_depth) + iter-3 STOP-gate (conduit_flow) approved by user (2026-04-27). Surcharge confirmed empirically: max_over_full_flow = [1.00, 1.00, 1.07, 1.03] for C1/C2/C3/C4. Pre-existing stipulation `report renderers accept uniform signature` covers the per_sim renderers without amendment. | toolkit `5fe677e` (committed + pushed to `worktree-toolkit_04-23_1737_visualization-and-reporting-multiphase-pwi` 2026-04-27 12:43 EDT, 6 files, 388 insertions, 4 deletions; QAQC: 0 new ruff errors on touched code after 2 NEW E501 in peak_flood_depth.py:112-113 fixed inline by extracting `da_masked.max().compute()` and `da_masked.notnull().sum().compute()` to locals; pytest synth_02 + synth_05 7/7 in 81.86s; audit-on-diff N/A for toolkit-only diff scope); agentic-workspace commit deferred — aw repo carries Phase 2 closeout work (still uncommitted from 2026-04-26) + Phase 3 closeout in flight + Phase 4 substrate just authored; will land as one combined aw commit after Phase 4 implementation session |
| 2 — system map renderer stop gate | src/TRITON_SWMM_toolkit/report_renderers/{__init__.py, _cli.py, _figure_emission.py, system_overview.py}, src/TRITON_SWMM_toolkit/workflow.py (`_build_plot_rule_block_system_overview` + `rule all` extension), src/TRITON_SWMM_toolkit/scenario_inputs.py (empty-list guard in `update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell`), tests/fixtures/synthetic_model/{cache.py, geometry.py, swmm_template.py, vectors.py} (iter-2 topology + sea-wall row + identical 5×5 subcatchments + smaller pipes), agentic-workspace `scripts/png_to_vault_webp.py` (workspace tooling, not toolkit) | swmmio's case-duplicated `[POLYGONS]` section bug — fixed by removing the empty placeholder from STARTER_INP. GDAL `to_raster` crash via swmm.toolkit library-load-order — worked around by driving fixture rebuilds through pytest. Iter-2 Improvement-agent false-negative on the sea-wall objective — overruled after main-agent direct verification of the DEM raster (row 26 all 50m) and QC's independent visual confirmation that the grey stripe is visible. Long-running orphaned `python -c` background process from a prior tool call kept rebuilding the synth fixture cache mid-iteration — diagnosed and killed (PID 2343598). | All DoD items satisfied: `system_overview.py` renders 2-panel (SWMM elements + TRITON DEM) figure with full-res + preview + manifest emission; CLI dispatcher present and exercises uniform signature; Snakemake `plot_system_overview` rule wired into `rule all`; pytest tests/test_synth_02 + test_synth_05 pass 7/7 in 88s with iter-2 fixture topology (4 junctions J1-J4 + sewer_outflow + dummy_outfall, 4 conduits, 3 identical 5×5 subcatchments draining to J1/J2/J3, sea-wall row at DEM matrix_row 26); system_overview iter-1 + iter-2 STOP-gate approved by user (iter-2 closed 2026-04-26 10:15 EDT). Stipulation `report renderers accept uniform signature` authored at agentic-workspace `library/docs/stipulations/TRITON-SWMM_toolkit/`. v1.5 amendment block (11 lessons) added to the in-flight Data Viz Review Algorithm in the session scratch doc. | toolkit `dddc0e0` (committed 2026-04-26 11:30 EDT, 10 files, 1008 insertions, 91 deletions); agentic-workspace commit deferred per user (2026-04-26 11:35 EDT — "fine to skip AW commit right now") — 4 AW-side files (stipulation, master-plan File-by-File update, moved phase doc, `scripts/png_to_vault_webp.py`) remain in working tree pending resolution of pre-existing AW repo state (76 stale staged files + 6 unmerged files; details in scratch § "Commit results") |
| 5 — per-analysis summary renderer stop gate | `src/TRITON_SWMM_toolkit/report_renderers/per_analysis_summary.py` (NEW — workflow-health placeholder table, iter-2 row set: n sims, successful, pending, failed, enabled model types, conditional sensitivity_mode), `src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py` (Option B helper fix per /design-recommendation 2026-04-30 — preview PNG + manifest siblings now emit for ALL output formats; dual full_res_metadata vs preview_metadata; `full_res_format` manifest field), `src/TRITON_SWMM_toolkit/workflow.py` (`_build_plot_rule_block_per_analysis_summary` + `_collect_per_analysis_summary_source_paths` + `rule all` extension), `src/TRITON_SWMM_toolkit/config/report.py` (iter-2 metrics list update: `enabled_model_types` + `sensitivity_mode` replaced `avg_triton_continuity_error` + `avg_swmm_continuity_error`), `tests/test_synth_04_multisim_with_snakemake.py` (≤1-step cross-model timestep tolerance with `_BC_TRUNCATION_KNOWN_BUG` error message documenting upstream TRITON-SWMM coupled-mode FP-drift truncation under active BC) | 0 new ruff errors on Phase-5-added code (per_analysis_summary.py 0/0, _figure_emission.py 0/0, config/report.py 0/0, test_synth_04 0/0; workflow.py 22 pre-existing errors all in lines outside Phase-5-added range ~535/721-808). Audit-on-diff N/A — toolkit-only diff scope; `scripts.audit` is agentic-workspace-internal. 23 tests pass (synth_04 1/1 in 192s; synth_02 4/4 + synth_05 5/5 + parser regression 13/13 in 208s combined). | All DoD items satisfied: workflow-health placeholder table renders 5 rows for the synth_multi_sim fixture (sensitivity row correctly suppressed; no sensitivity_mode in this analysis); user verbal "table approved" 2026-05-01 ~12:35 PM cleared the Hard STOP gate; Phase B + Phase D specialist consults SKIPPED per user directive (placeholder framing — comprehensive diagnostics live in the v2 plan). Phase doc refreshed to "What Was Built" framing per `plan completion and closure.md`. Phase 5 closeout absorbed two prereq files surfaced during validation: helper fix (Option B per /design-recommendation 2026-04-30) and test_synth_04 BC-truncation tolerance (≤1-step with explicit known-bug error message, investigation captured at `~/.claude/plans/async-cooking-gadget.md`). Iter-2 surfaced an algorithm-improvement candidate ("placeholder-figure short-circuit" in /design-figure) — user-rejected ("do not update algorithm" 2026-05-01); not folded into `# Data Viz Review Algorithm`. | toolkit `2dabb18` (committed + pushed 2026-05-01); agentic-workspace `8bf60c454` (phase doc refresh + move to implemented/) |
| 6 — sensitivity benchmarking renderer stop gate | `src/TRITON_SWMM_toolkit/report_renderers/sensitivity_benchmarking.py` (NEW — 4-panel renderer: wall-clock + compute-cost + strong-scaling speedup + strong-scaling efficiency; DataTree-aware reads with SWMM-only `parse_total_elapsed` fallback; Okabe-Ito CVD-safe palette; LaTeX y-labels via mathtext; uniform marker/annotation params across panels; axes-anchored title/footnote), `src/TRITON_SWMM_toolkit/swmm_output_parser.py` (NEW `parse_total_elapsed(rpt_path) -> float \| None` helper for SWMM 5.2.x canonical "Total elapsed time" forms), `src/TRITON_SWMM_toolkit/config/report.py` (extended `SensitivityReportConfig` with `group_by_var: str \| None` + `show_gridlines: bool = True`; removed iter-experimental `show_ideal_weak_scaling_line` per user feedback "misleading"), `src/TRITON_SWMM_toolkit/report_renderers/_figure_emission.py` (implemented `collect_sensitivity_source_paths` body — was NotImplementedError placeholder), `src/TRITON_SWMM_toolkit/workflow.py` (`SensitivityAnalysisWorkflowBuilder._build_plot_rule_block_sensitivity_benchmarking` + `_collect_sensitivity_source_paths` lambda + master `rule all` `expand` extension; `_get_config_args` gained `include_report_config: bool = False` flag so non-renderer rules don't emit `--report-config`), `tests/test_sensitivity_benchmarking_math.py` (NEW — 19 unit tests: per-group + global-baseline anchoring, perfect/imperfect speedup, strong/weak efficiency, MIN-y at duplicate N, missing-N=1 exclusion), `tests/test_swmm_output_parser_refactoring.py` (NEW `TestParseTotalElapsed` class — 5 tests for `parse_total_elapsed`), `tests/fixtures/synthetic_model/report_config_synth_sensitivity.yaml` (NEW — synth-tier report config), `tests/test_synth_05_sensitivity_analysis_with_snakemake.py` (wired `report_config_path=_SYNTH_SENSITIVITY_REPORT_CONFIG` into all three test paths + benchmarking-rule presence assertions), `tests/test_synth_06_sensitivity_analysis_triton_and_swmm_only.py` (wired `report_config=...` into `analysis.run()`; dropped stale "consolidation bug" NOTE — verified stale via TRITON-only execution test pass) | 0 new ruff errors on Phase-6-added code after 2 fixable auto-fixed (1 import-sort I001 in test_sensitivity_benchmarking_math.py + 1 zip-strict B905 in sensitivity_benchmarking.py). Audit-on-diff N/A — toolkit-only diff scope. 42 tests pass (math 19/19 in 0.07s; parser 5/5 in 0.09s; synth_02 4/4 + synth_05 4/4 non-slow in 186s; synth_05 slow execution 1/1 in 215s with full sensitivity workflow + benchmarking SVG existence assertion; synth_06 TRITON-only 1/1 in 240s). Algorithm-improvement candidate surfaced in iter-9: "anchor figure-level decoration to plot-area axes via `ax.set_title()` / `ax.text(transform=ax.transAxes)` rather than `fig.suptitle()` / `fig.text()` whenever y-label widths offset plot area" — captured in iter-9 round block; not yet folded (no `# Data Viz Review Algorithm` section in this scratch). | All DoD items satisfied: 4-panel publication-grade benchmarking figure renders against `synth_sensitivity_analysis_cached`; user verbal "approved" 2026-05-01 19:08 EDT cleared the Hard STOP gate at iter-9. 9 Phase C iterations: bar→scatter→dual-panel→4-panel→test-driven math (19 unit tests) → global-baseline anchoring → strong-scaling efficiency math correction → uniform marker/annotation params → axes-anchored title+footnote (fix plot-vs-figure-center misalignment from y-label widths). Phase B kickoff specialist consult applied verbatim (4 In-Scope Action Drafts: Okabe-Ito palette, vertical stacking, suptitle+footnote, x-axis label mapping). Phase D specialist consult deferred per user; Phase E closeout completed (Figure spec frozen 2026-05-01 19:30 -0400). Routed pre-existing `/write-to-scratch` "marker-only" prompt smell via `/address-smell` → 4 verbatim modification specs applied (default flipped to write body content; navigation-only marker becomes opt-in exception). 5 specialist follow-ups + 1 algorithm-improvement candidate routed to scratch `# Follow-up Ideas`. v2 plan augmentation during iter-3 surfaced 4 user-supplied benchmarking reference figures; ported to v2 Phase 4 (Fig-A8 with TRITON-vs-SWMM split user-locked addition) and v2 Phase 7 (HW-config table, BLUPS scaling, DEM-resolution × device-count grid) with embedded reference webps. | toolkit `0613e5d` (committed + pushed 2026-05-01 19:23 EDT, 10 files, 1072 insertions, 21 deletions); agentic-workspace `c69bf51af` (phase doc refresh + skill smell-fix) + `c02de8edb` (move-phase + planning tables) — both pushed 2026-05-01 19:25 EDT |

## Follow-up Ideas






- Worktree-slug derivation ambiguity: for workspace names containing underscores (e.g., `TRITON-SWMM_toolkit`), `_derive_worktree_slug` in `scripts/generate/setup_pwi_worktree.py` strips at the first underscore (mechanical), while a semantic read of `worktree isolation.md` § Step 2 / Naming convention strips the full workspace name. The two produce different slugs. Options: (a) clarify the protocol wording to explicitly say "strip up to and including the first underscore," (b) add a shared helper that both the script and the `EnterWorktree` invocation docs in the protocol reference, (c) add a protocol-side worked example for workspaces with underscored names. Encountered during `/proceed-with-implementation` 2026-04-23 1737 visualization-and-reporting session — lost ~5 tool calls recovering.
- Baseline audit scalability for code-only plans: `pwi_audit_snapshot.py` runs `scripts.audit` over the full prompt-doc tree, which took >3 min without completing on this TRITON-SWMM_toolkit session. For plans with `modifies_prompts: false` or narrow prompt-doc surface (e.g., this plan modifies only 2 prompt docs in Phase 6), a targeted `--files` baseline would produce a meaningful delta in seconds. Consider adding `--files` or `--scope <workspace>` support to `pwi_audit_snapshot.py` so the baseline gate scales with surface size, not tree size.
- Dedicated data-viz subagent (wrapup `/plan-implementation` target): stand up a data-vis-review specialist agent with its own prompt-doc-backed rubric and its own scratch-doc composition, replacing the generic-subagent + verbatim-rubric pattern from the `# Data Viz Review Algorithm` section. Triggered at the end of this `/record-process` session once the algorithm, rubric, subagent-invocation template, round checklist, and iteration-record template have stabilized through real iteration rounds. The prompt doc + `/create-skill`-built skill produced during process-recording become the input substrate for the agent spec.

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

- **Planning-Instruction Improvement Proposal 1**: Add a "Substrate attribute/method verification" atomic to /plan-implementation that, for every phase doc referencing `scen_paths.<attr>`, `analysis.<method>`, or `sensitivity.<method>`, runs `grep -n "<attr>|<method>"` against the current source and fails the phase-doc writing step if the reference does not match. Would have caught Flags 1, 2, 3, 5 at plan-authorship time in this session.
- **Planning-Instruction Improvement Proposal 2**: Expand the master-plan "File-by-File Change Plan" atomic to require a "Facade threading audit" row whenever a new runtime parameter is added to `analysis.run()` — explicitly enumerating the 4 facade layers from architecture.md § submit_workflow() Facade Layers and asserting a change (or an explicit "no change needed because {reason}") at each layer. Would have caught Flag 4.
- **Planning-Instruction Improvement Proposal 3**: Add a "classmethod surface verification" atomic that, when `_cli.py` or any runner script references `ClassName.from_*()` classmethods, grep-validates those classmethods exist. Would have caught Flag 5.
- **Planning-Instruction Improvement Proposal 4**: Expand /plan-implementation's Validation Plan atomic to require one idempotence test for any new Python facade method advertised as idempotent. Would have caught Flag 13 / R11 case.

## Worktree Status

*(section not present in main scratch)*

