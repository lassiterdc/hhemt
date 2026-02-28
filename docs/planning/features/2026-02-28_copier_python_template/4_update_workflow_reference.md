# Phase 4: `copier update` Workflow Reference Guide

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — renumbered from Phase 3 to Phase 4; added copier-specialist agent startup reads update step

---

## Goal

Produce a concise written reference you can follow independently in future sessions when you want to propagate template changes to one or more downstream projects. This is a living reference — update it as you encounter edge cases.

This phase is completed **during** Phase 3 (Step 3.4 produces the hands-on experience; this phase formalizes it as a reusable document and then uses it to fully initialize the copier-specialist agent).

---

## The Reference Document

After completing Phase 3 Step 3.4, Claude will write a `docs/planning/reference/copier_update_workflow.md` file **in the `copier-python-template` repo** covering:

### Part 1: Making a Template Change

```
1. Edit the relevant file(s) in template/
2. Test locally: copier copy --defaults . /tmp/test-update-project
3. Verify the change looks correct in the generated output
4. Commit to the template repo
5. Tag the release: git tag vX.Y.Z && git push --tags
```

### Part 2: Propagating to a Downstream Project

```
1. cd into the downstream project
2. Run: copier update --skip-tasks
   - Copier fetches the latest template tag
   - Performs 3-way merge (BASE=last template version, OURS=your file, THEIRS=new template)
   - Auto-merges non-conflicting changes
   - Inserts conflict markers for conflicting sections
3. Review all changed files (git diff)
4. Resolve any conflict markers manually
5. Commit: git add . && git commit -m "chore: apply template update vX.Y.Z"
```

### Part 3: Conflict Scenarios and Resolutions

| Scenario | What you see | What to do |
|----------|-------------|------------|
| Template changed Part I; you only touched Part II | File updated automatically, no markers | Just commit |
| Template changed Part I; you also changed Part I | Conflict markers in the file | Keep the version you want; remove markers; commit |
| Template added a new file you don't have | New file appears in your project | Review and commit |
| Template changed a file you deleted | File reappears | Delete it again; commit |
| Template renamed a variable in `copier.yml` | `.copier-answers.yml` may have stale key | Run `copier update --defaults` and re-answer changed questions |

### Part 4: Keeping Track of Which Version Each Project Is On

```bash
# Check which template version a project is pinned to:
cat .copier-answers.yml | grep _commit

# Check all downstream projects at once (if organized under ~/dev/):
grep -r "_commit:" ~/dev/*/.copier-answers.yml 2>/dev/null
```

---

## Definition of Done

- [ ] `docs/planning/reference/copier_update_workflow.md` written in `copier-python-template`
- [ ] Document covers all four parts: making a change, propagating it, conflict scenarios, version tracking
- [ ] Document committed and pushed to the template repo
- [ ] `copier-specialist` agent startup reads updated to include the reference doc (remove the "until that file exists" placeholder in `2_build_copier_specialist.md` body; add the path as an active startup read)
- [ ] Agent confirmed via self-report: reads both `README.md` and `copier_update_workflow.md` on startup
- [ ] `@.prompts/qaqc_and_commit.md` completed and findings reported to developer
