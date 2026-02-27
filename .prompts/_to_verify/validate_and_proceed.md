# Validate Work & Proceed

Review recent work against planning documents, fix any inconsistencies, then continue with the next priority.

## Steps

1. **Validate congruence:**
   - Check that `docs/planning/priorities.md` accurately reflects work completed
   - Check that tracker files (e.g., `cruft_cleanup_tracker.md`) are up-to-date
   - Verify all "Last Updated" dates are current
   - Ensure checkboxes match actual completion state

2. **Fix inconsistencies:**
   - Update stale statuses
   - Mark completed items as `[x]`
   - Update touched file lists in trackers
   - Refresh test status sections

3. **Identify next priority:**
   - Based on `priorities.md` tier ordering
   - Check for blockers or dependencies
   - Prefer completing current tier before moving to next

4. **Execute next work:**
   - Run all 4 smoke tests (PC_01, PC_02, PC_04, PC_05) after significant changes
   - Commit with descriptive message including Co-Authored-By line
   - Update tracker/priorities as work completes

## Expected Output

- Summary of validation findings and fixes made
- Clear statement of next priority and why
- Execution of the next priority work
