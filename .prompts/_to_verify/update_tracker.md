# Update Tracker After Phase Completion

Update the appropriate tracker document after completing a phase of work.

## Common Trackers

- `docs/planning/cruft_cleanup_tracker.md` — for cruft cleanup phases
- `docs/planning/priorities.md` — top-level priority checklist
- Phase-specific tracker docs in `docs/planning/refactors/`

## What to Update

1. **Phase status** — change from "In Progress" to "Complete"
2. **Touched files list** — add all files created/modified/deleted
3. **Implementation details** — bulleted list of what was done
4. **Test status** — report test results (pass counts, duration)
5. **Net line changes** — if significant (e.g., "305 deleted, 13 inserted")

## Cross-References

- Ensure `priorities.md` checkboxes match tracker completion status
- Update "Last Updated" dates
- Check for stale status claims in related docs

## Expected Output

- Updated tracker with complete phase documentation
- Congruent status across all related planning docs
